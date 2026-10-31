"""LLMExtractor: Tier 3 — LLM-driven extraction.

Phase 2: working implementation. Three concrete adapters:
    - `MockLLMAdapter`     deterministic canned outputs (offline tests)
    - `NRPOpenAIAdapter`    NRP Nautilus hosted models (production)
    - `LocalVLLMAdapter`    locally-running vLLM/SGLang server

All adapters speak the OpenAI Chat Completions protocol. Authentication and
endpoint are taken from environment variables (never hardcoded):

    ERISML_LLM_BASE_URL    default: https://ellm.nrp-nautilus.io/v1
    ERISML_LLM_API_KEY     no default; required for NRP adapter
    ERISML_LLM_MODEL       default: qwen3
    ERISML_LLM_TIMEOUT_S   default: 60
    ERISML_LLM_MAX_RETRIES default: 2

The LLMExtractor issues one prompt per element class -- stakeholders,
relations, commitments, ethical facts, canonical form -- then up to
`max_completion_passes` completion passes, each showing the model the draft
encoding and asking only for what it misses. It validates every element
against the IR schemas and assembles an ExtractorResult. Two steps are code,
not prompts: a first-person text always has a narrator who is the agent, and
every duty-bearing relation of the agent's (DUTY_RELATIONS) becomes a
`role_duty` commitment the agent holds toward the other party.

Parsing strategy: each prompt asks for a JSON array. We extract the first
balanced JSON array from the model response (LLMs often add prose around
the JSON), validate each element against the relevant Pydantic schema, and
collect successes. Failures are logged but do not abort extraction; the
critic pass will catch missing entries.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from erisml_compiler.annotation.base import Extractor, ExtractorResult
from erisml_compiler.ir.schemas import (
    Commitment,
    EthicalFact,
    Relation,
    Stakeholder,
)

# =============================================================================
# Adapter interface
# =============================================================================


class ModelAdapter(ABC):
    """Abstract base for a concrete model adapter."""

    name: str = "abstract_adapter"

    @abstractmethod
    def call(self, system: str, user: str, **kwargs) -> str:
        """Issue one chat-completions call and return the raw assistant text."""
        raise NotImplementedError


# =============================================================================
# MockLLMAdapter — deterministic canned outputs (offline-testable)
# =============================================================================


class MockLLMAdapter(ModelAdapter):
    """Returns canned responses for known prompts. Used by the test suite to
    exercise LLMExtractor without network access.

    The mapping is keyed by a substring fingerprint of (system, user). The
    test suite registers fixtures via `register_response(prompt_fragment,
    response)`."""

    name = "mock_llm"

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses: dict[str, str] = dict(responses or {})

    def register_response(self, prompt_fragment: str, response: str) -> None:
        self._responses[prompt_fragment] = response

    def call(self, system: str, user: str, **kwargs) -> str:
        combined = f"{system}\n{user}"
        for fragment, response in self._responses.items():
            if fragment in combined:
                return response
        return "[]"  # empty list is a safe default


# =============================================================================
# NRPOpenAIAdapter — NRP-hosted OpenAI-compatible endpoint
# =============================================================================


class NRPOpenAIAdapter(ModelAdapter):
    """Uses the OpenAI Python SDK against the NRP base URL.

    Reads ERISML_LLM_BASE_URL, ERISML_LLM_API_KEY, ERISML_LLM_MODEL from the
    environment. Raises a clear error if the API key is missing.
    """

    name = "nrp_openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.base_url = base_url or os.environ.get(
            "ERISML_LLM_BASE_URL", "https://ellm.nrp-nautilus.io/v1"
        )
        self.api_key = api_key or os.environ.get("ERISML_LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "NRPOpenAIAdapter requires ERISML_LLM_API_KEY in the "
                "environment. Get a token from https://nrp.ai and set "
                "ERISML_LLM_API_KEY=<token>."
            )
        self.model = model or os.environ.get("ERISML_LLM_MODEL", "qwen3")
        self.timeout_s = timeout_s or float(os.environ.get("ERISML_LLM_TIMEOUT_S", "60"))
        # Lazy import so the optional dependency is only required at use time.
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def call(self, system: str, user: str, **kwargs) -> str:
        max_retries = int(os.environ.get("ERISML_LLM_MAX_RETRIES", "2"))
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 2048),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_exc = exc
                if attempt == max_retries:
                    raise
        raise RuntimeError(f"All retries failed: {last_exc}")


# =============================================================================
# LocalVLLMAdapter — locally-running vLLM/SGLang
# =============================================================================


class LocalVLLMAdapter(ModelAdapter):
    """Identical wire protocol to NRP; different default base URL."""

    name = "local_vllm"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ):
        self.base_url = base_url or os.environ.get(
            "ERISML_LLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = os.environ.get("ERISML_LLM_API_KEY", "EMPTY")
        self.model = model or os.environ.get("ERISML_LLM_MODEL", "local-model")
        self.timeout_s = timeout_s or float(os.environ.get("ERISML_LLM_TIMEOUT_S", "60"))
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def call(self, system: str, user: str, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return (resp.choices[0].message.content or "").strip()


# =============================================================================
# Prompt templates
# =============================================================================


_SYSTEM_PROMPT = """You are a moral-structure parser. You read a passage of natural-language text and extract its ethical structure as STRICT JSON only. You never produce prose, commentary, or markdown outside the JSON. If a value is unknown, omit the field; do not invent. If asked for a list, always return a JSON array, even if empty."""


def _stakeholders_prompt(text: str) -> str:
    return f"""Identify every stakeholder in the text below.

Include every person or group the text mentions, also one named only by a relation ("my sister",
"his boss"). If the text is in the first person, the narrator is a stakeholder with id "narrator"
and the role "agent".

Return ONLY a JSON array. Each element has these fields:
  id              snake_case slug, unique per stakeholder
  label           short human-readable name
  type            one of: individual, community, institution, system, group, abstract, future, collective
  roles           list, each from: agent, patient, beneficiary, victim, bystander, nonconsenting_third_party, authority, coercer, dependent, future_person, collective, vow_holder, protector
  vulnerability   one of: low, moderate, high, extreme  (omit if unclear)
  consent_status  one of: obtained, not_obtained, coerced, n/a  (omit if unclear)
  confidence      number in [0,1]

Example single-element response:
[{{"id":"villager","label":"Villager","type":"individual","roles":["agent","vow_holder","protector"],"vulnerability":"moderate","consent_status":"n/a","confidence":0.9}}]

TEXT:
{text}"""


# Relation types the extractor asks for. `source` stands in the relation to `target`.
RELATION_TYPES: dict[str, str] = {
    "parent_of": "source is a parent of target",
    "child_of": "source is a child of target",
    "sibling_of": "source and target are siblings",
    "kin_of": "other family: grandparent, grandchild, aunt, uncle, cousin, in-law, step-relative",
    "spouse_of": "married",
    "partner_of": "romantic partner, boyfriend, girlfriend, fiance",
    "ex_partner_of": "former spouse or romantic partner",
    "friend_of": "friends",
    "guardian_of": "source has legal or practical charge of target",
    "caregiver_of": "source looks after target (a nurse, carer, babysitter)",
    "employer_of": "source employs or manages target",
    "colleague_of": "coworkers or classmates",
    "roommate_of": "share a home without being family or partners",
    "neighbour_of": "neighbours",
    "service_provider_to": "source serves target as a customer or client",
    "acquaintance_of": "know each other slightly",
    "stranger_to": "no prior relation",
}

# Relations that carry special obligations between the two parties, in both directions: each
# becomes a role_duty the agent holds toward the other party. The others carry only the duties
# owed to anyone. This table is a normative choice, kept in one place so it can be argued with.
DUTY_RELATIONS: frozenset[str] = frozenset(
    {
        "parent_of",
        "child_of",
        "sibling_of",
        "kin_of",
        "spouse_of",
        "partner_of",
        "friend_of",
        "guardian_of",
        "caregiver_of",
        "employer_of",
    }
)


def _relations_prompt(text: str, stakeholder_ids: list[str]) -> str:
    ids = ", ".join(stakeholder_ids) or "(none)"
    types = "\n".join(f"  {k:<20} {v}" for k, v in RELATION_TYPES.items())
    return f"""Identify every relation between stakeholders that the text states or plainly implies.

Stakeholder ids available for `source` and `target`: {ids}

Relation types (source stands in the relation to target):
{types}

Return ONLY a JSON array. Each element has these fields:
  id              snake_case slug
  type            one of the relation types above
  source          stakeholder id
  target          stakeholder id
  description     a few words

Example: "my sister" in a first-person text gives
[{{"id":"r1","type":"sibling_of","source":"narrator","target":"sister","description":"narrator's sister"}}]

TEXT:
{text}"""


def _completion_prompt(text: str, draft: dict[str, Any]) -> str:
    types = ", ".join(RELATION_TYPES)
    return f"""Below are a text and a draft encoding of its ethical structure.
Review the draft encoding for what it is MISSING: stakeholders, relations, commitments or ethical
facts that the text states or plainly implies. In particular, an act that harms, neglects, deceives or coerces a
party should be an ethical fact naming that party in `subjects`.

Return ONLY a JSON object with four arrays, each holding new elements only, in the same fields as
the draft: {{"stakeholders": [], "relations": [], "commitments": [], "ethical_facts": []}}.
Leave an array empty when nothing of that kind is missing. Do not repeat or change draft elements.
Relation types: {types}.

TEXT:
{text}

DRAFT ENCODING:
{json.dumps(draft, indent=1)}"""


def _commitments_prompt(text: str, stakeholder_ids: list[str]) -> str:
    holders = ", ".join(stakeholder_ids) or "(any stakeholder id)"
    return f"""Identify every commitment, vow, promise, contract, oath, covenant, or role-duty in the text below.

Stakeholder ids available for `holder` and `beneficiary`: {holders}

Return ONLY a JSON array. Each element has these fields:
  id              snake_case slug
  type            one of: vow, promise, contract, oath, covenant, role_duty, policy
  holder          stakeholder id who is bound by the commitment
  beneficiary     stakeholder id who benefits (omit if none)
  content         brief description of the commitment
  voluntariness   one of: voluntary, coerced, incapacitated, uncertain
  status          one of: active, active_but_defeasible, defeated, fulfilled, violated, void, expired
  legitimacy      one of: prima_facie_valid, fully_legitimate, defeasible, void, fraudulent, tyrannical
  defeasibility_conditions  list of: coerced_vow, vow_to_commit_wrong, catastrophic_nonconsensual_externality, higher_duty_conflict, legitimacy_collapse, impossibility

TEXT:
{text}"""


def _facts_prompt(text: str, stakeholder_ids: list[str]) -> str:
    subjects = ", ".join(stakeholder_ids) or "(any stakeholder id)"
    return f"""Identify every ethical fact in the text below.

Stakeholder ids available for `subjects`: {subjects}

Return ONLY a JSON array. Each element has these fields:
  id            snake_case slug
  kind          one of: coercion, consent, legitimacy, harm, vulnerability, uncertainty, externality, justice, care, truth, role_duty, deception, reciprocity, non_maleficence, benefit, urgency
  subjects      list of stakeholder ids
  description   one sentence
  severity      one of: minor, moderate, grave, catastrophic
  confidence    number in [0,1]

Name in `subjects` the parties who bear each fact: an act that harms, neglects, deceives or coerces a party is a fact naming that party.

Be specific. A line like 'soldiers threaten village' produces two facts: one of kind `coercion` and one of kind `externality`, both subject to the village. For `benefit` (a good secured or a harm averted) and `urgency` (what follows if nothing is done in time), severity names the stakes.

TEXT:
{text}"""


def _canonical_form_prompt(text: str, facts_summary: str, known_forms: list[str]) -> str:
    forms = "\n".join(f"  - {f}" for f in known_forms) if known_forms else "  (no registry yet)"
    return f"""You are choosing a canonical tag for the moral situation described in the text.

KNOWN TAGS in the registry:
{forms}

INSTRUCTIONS:
  1. If the situation matches one of the known tags closely, return that tag.
  2. Otherwise, propose a new snake_case tag describing the situation type.
  3. The tag should be invariant under semantically equivalent redescriptions.

Return ONLY a JSON object: {{"tag": "...", "confidence": 0.0_to_1.0, "matched_known_tag": true_or_false}}

TEXT:
{text}

EXTRACTED FACTS:
{facts_summary}"""


# =============================================================================
# JSON-from-LLM parsing
# =============================================================================


_ARRAY_PATTERN = re.compile(r"\[[\s\S]*?\](?=\s*$|\s*[A-Za-z])|\[[\s\S]*\]", re.MULTILINE)
_OBJECT_PATTERN = re.compile(r"\{[\s\S]*?\}(?=\s*$|\s*[A-Za-z])|\{[\s\S]*\}", re.MULTILINE)


def _extract_first_json(text: str, expect_array: bool = True) -> Any:
    """Extract the first balanced JSON array or object from a string.

    LLMs often wrap their JSON in markdown fences or explanatory prose. We
    strip fences first, then try to parse the whole string; if that fails,
    we walk the brackets to find a balanced span.
    """
    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    open_ch, close_ch = ("[", "]") if expect_array else ("{", "}")
    start = cleaned.find(open_ch)
    if start < 0:
        raise ValueError(f"No {open_ch!r} found in model response.")
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                span = cleaned[start : i + 1]
                return json.loads(span)
    raise ValueError("Unbalanced JSON in model response.")


def _as_array(data: Any) -> Any:
    """A model asked for an array sometimes wraps it in an object with one key
    (``{"stakeholders": [...]}``); unwrap that."""
    if isinstance(data, dict) and len(data) == 1:
        (value,) = data.values()
        if isinstance(value, list):
            return value
    return data


# =============================================================================
# Steps done in code: the narrator, duties from relations, merging additions
# =============================================================================

_FIRST_PERSON = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")
_NARRATOR_IDS = ("narrator", "i", "me", "speaker", "self", "author", "writer")


def ensure_narrator(text: str, stakeholders: list[Stakeholder]) -> list[Stakeholder]:
    """A first-person text has a narrator, and when no stakeholder is the agent the narrator is.

    A model given a one-line first-person act ("not letting my sister move in") often returns no
    stakeholders, or none with the agent role, and then no duty, maxim or breach can be the
    agent's. A stakeholder the model already gave as the narrator gains the agent role; otherwise
    one is added. A text not in the first person, or one that already has an agent, is left as
    it is."""
    if any("agent" in s.roles for s in stakeholders) or not _FIRST_PERSON.search(text):
        return stakeholders
    for i, s in enumerate(stakeholders):
        if s.id.lower() in _NARRATOR_IDS:
            out = list(stakeholders)
            out[i] = s.model_copy(update={"roles": [*s.roles, "agent"]})
            return out
    narrator = Stakeholder(id="narrator", label="Narrator", type="individual", roles=["agent"])
    return [narrator, *stakeholders]


def duties_from_relations(
    relations: list[Relation], stakeholders: list[Stakeholder], commitments: list[Commitment]
) -> list[Commitment]:
    """A `role_duty` commitment held by the agent toward each party it stands in a
    DUTY_RELATIONS relation with, in either direction, unless the agent already holds a
    role duty toward that party. The duty is `active`: whether the act breaches it is read
    from the facts (erisml_backend.v3_bridge.breached_duties)."""
    agents = {s.id for s in stakeholders if "agent" in s.roles}
    have = {(c.holder, c.beneficiary) for c in commitments if c.type == "role_duty"}
    taken = {c.id for c in commitments}
    new: list[Commitment] = []
    for r in relations:
        if r.type not in DUTY_RELATIONS:
            continue
        for holder, other in ((r.source, r.target), (r.target, r.source)):
            if holder in agents and other not in agents and (holder, other) not in have:
                cid = f"role_duty_{holder}_{other}"
                if cid in taken:
                    continue
                new.append(
                    Commitment(
                        id=cid,
                        type="role_duty",
                        holder=holder,
                        beneficiary=other,
                        content=f"the duties of {r.type.removesuffix('_of').replace('_', ' ')}",
                        status="active",
                        source_spans=list(r.source_spans),
                    )
                )
                have.add((holder, other))
                taken.add(cid)
    return new


def _validated(model: type, items: Any, known: set[str] | None = None) -> list:
    """The elements of `items` that validate as `model`; with `known`, also those whose
    stakeholder references (source/target, holder/beneficiary, subjects) all name a known id."""
    out = []
    for raw in items if isinstance(items, list) else []:
        try:
            el = model(**raw)
        except (ValidationError, TypeError):
            continue
        if known is not None:
            refs = [getattr(el, f, None) for f in ("source", "target", "holder", "beneficiary")]
            refs += list(getattr(el, "subjects", []) or [])
            if any(r is not None and r not in known for r in refs):
                continue
        out.append(el)
    return out


def _merge(existing: list, additions: list, key) -> list:
    """`additions` not already present by id or by `key`; an addition reusing a taken id with
    different content gets a suffixed id."""
    ids = {e.id for e in existing}
    keys = {key(e) for e in existing}
    out = []
    for el in additions:
        if key(el) in keys:
            continue
        if el.id in ids:
            n = 2
            while f"{el.id}_{n}" in ids:
                n += 1
            el = el.model_copy(update={"id": f"{el.id}_{n}"})
        ids.add(el.id)
        keys.add(key(el))
        out.append(el)
    return out


def _stakeholder_key(s):
    return s.id


def _relation_key(r):
    return (r.type, r.source, r.target)


def _commitment_key(c):
    return (c.type, c.holder, c.beneficiary, c.content)


def _fact_key(f):
    return (f.kind, tuple(sorted(f.subjects)))


# =============================================================================
# LLMExtractor
# =============================================================================


class LLMExtractor(Extractor):
    """LLM-driven extractor. Requires a ModelAdapter."""

    name = "llm"

    def __init__(
        self,
        adapter: ModelAdapter,
        known_canonical_forms: list[str] | None = None,
        max_completion_passes: int = 1,
    ):
        self.adapter = adapter
        self.max_completion_passes = max_completion_passes
        self.known_canonical_forms = known_canonical_forms or [
            "coercive_murderous_interrogation_with_collective_reprisal",
            "professional_privilege_versus_duty_to_warn",
            "institutional_loyalty_versus_public_truth_telling",
        ]

    # ------------------------------------------------------ private helpers

    def _call_and_parse_array(self, prompt: str, attempts: int = 2) -> list[dict]:
        """One JSON array from the model. An array wrapped in a one-key object is unwrapped,
        and a response that holds no array is asked for again, once by default."""
        for attempt in range(attempts):
            raw = self.adapter.call(_SYSTEM_PROMPT, prompt)
            try:
                data = _as_array(_extract_first_json(raw, expect_array=True))
            except (ValueError, json.JSONDecodeError) as exc:
                if attempt + 1 < attempts:
                    continue
                raise RuntimeError(
                    f"LLM response did not contain a JSON array: {exc}\nRaw response: {raw[:500]}"
                )
            if isinstance(data, list):
                return data
            if attempt + 1 == attempts:
                raise RuntimeError(f"Expected a JSON array, got {type(data).__name__}.")
        raise RuntimeError("unreachable")

    def _call_and_parse_object(self, prompt: str) -> dict:
        raw = self.adapter.call(_SYSTEM_PROMPT, prompt)
        try:
            data = _extract_first_json(raw, expect_array=False)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"LLM response did not contain a JSON object: {exc}\nRaw response: {raw[:500]}"
            )
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected a JSON object, got {type(data).__name__}.")
        return data

    # ------------------------------------------------------ extract

    def extract(self, document, segments) -> ExtractorResult:
        text = document.raw_text

        # Pass 1: stakeholders (malformed entries are skipped), and the narrator.
        stakeholders = _validated(
            Stakeholder, self._call_and_parse_array(_stakeholders_prompt(text))
        )
        stakeholders = ensure_narrator(text, stakeholders)
        stakeholder_ids = [s.id for s in stakeholders]

        # Pass 2: relations between known stakeholders.
        relations = _merge(
            [],
            _validated(
                Relation,
                self._call_and_parse_array(_relations_prompt(text, stakeholder_ids)),
                known=set(stakeholder_ids),
            ),
            _relation_key,
        )

        # Pass 3: commitments, and the role duties the agent's relations carry.
        commitments = _validated(
            Commitment, self._call_and_parse_array(_commitments_prompt(text, stakeholder_ids))
        )
        commitments += duties_from_relations(relations, stakeholders, commitments)

        # Pass 4: ethical facts.
        ethical_facts = _validated(
            EthicalFact, self._call_and_parse_array(_facts_prompt(text, stakeholder_ids))
        )

        # Completion passes: the model sees the draft and gives only what it misses. Additions
        # must reference known stakeholders; the passes stop early once nothing is added.
        added_per_pass: list[int] = []
        for _ in range(self.max_completion_passes):
            draft = {
                "stakeholders": [s.model_dump(mode="json") for s in stakeholders],
                "relations": [r.model_dump(mode="json") for r in relations],
                "commitments": [c.model_dump(mode="json") for c in commitments],
                "ethical_facts": [f.model_dump(mode="json") for f in ethical_facts],
            }
            try:
                more = self._call_and_parse_object(_completion_prompt(text, draft))
            except RuntimeError:
                more = {}
            new_s = _merge(
                stakeholders, _validated(Stakeholder, more.get("stakeholders")), _stakeholder_key
            )
            stakeholders = ensure_narrator(text, stakeholders + new_s)
            known = {s.id for s in stakeholders}
            new_r = _merge(
                relations, _validated(Relation, more.get("relations"), known), _relation_key
            )
            relations += new_r
            new_c = _merge(
                commitments,
                _validated(Commitment, more.get("commitments"), known),
                _commitment_key,
            )
            commitments += new_c
            new_d = duties_from_relations(relations, stakeholders, commitments)
            commitments += new_d
            new_f = _merge(
                ethical_facts, _validated(EthicalFact, more.get("ethical_facts"), known), _fact_key
            )
            ethical_facts += new_f
            added = len(new_s) + len(new_r) + len(new_c) + len(new_d) + len(new_f)
            added_per_pass.append(added)
            if not added:
                break

        # Canonical form.
        facts_summary = "; ".join(
            f"{f.kind}({','.join(f.subjects)}): {f.description}" for f in ethical_facts[:10]
        )
        try:
            canonical_obj = self._call_and_parse_object(
                _canonical_form_prompt(text, facts_summary, self.known_canonical_forms)
            )
            canonical_form = canonical_obj.get("tag")
        except (RuntimeError, ValueError):
            canonical_form = None

        return ExtractorResult(
            stakeholders=stakeholders,
            relations=relations,
            commitments=commitments,
            ethical_facts=ethical_facts,
            canonical_form=canonical_form,
            extractor_metadata={
                "extractor": "llm",
                "adapter": self.adapter.name,
                "completion_passes_added": added_per_pass,
            },
        )
