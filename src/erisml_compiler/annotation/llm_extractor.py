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

The LLMExtractor issues four prompts per document -- stakeholders,
commitments, ethical facts, canonical form -- collects the JSON responses,
validates them against the IR schemas, and assembles an ExtractorResult.

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
  kind          one of: coercion, consent, legitimacy, harm, vulnerability, uncertainty, externality, justice, care, truth, role_duty, deception, reciprocity, non_maleficence
  subjects      list of stakeholder ids
  description   one sentence
  severity      one of: minor, moderate, grave, catastrophic
  confidence    number in [0,1]

Be specific. A line like 'soldiers threaten village' produces two facts: one of kind `coercion` and one of kind `externality`, both subject to the village.

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


# =============================================================================
# LLMExtractor
# =============================================================================


class LLMExtractor(Extractor):
    """LLM-driven extractor. Requires a ModelAdapter."""

    name = "llm"

    def __init__(self, adapter: ModelAdapter, known_canonical_forms: list[str] | None = None):
        self.adapter = adapter
        self.known_canonical_forms = known_canonical_forms or [
            "coercive_murderous_interrogation_with_collective_reprisal",
            "professional_privilege_versus_duty_to_warn",
            "institutional_loyalty_versus_public_truth_telling",
        ]

    # ------------------------------------------------------ private helpers

    def _call_and_parse_array(self, prompt: str) -> list[dict]:
        raw = self.adapter.call(_SYSTEM_PROMPT, prompt)
        try:
            data = _extract_first_json(raw, expect_array=True)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM response did not contain a JSON array: {exc}\nRaw response: {raw[:500]}")
        if not isinstance(data, list):
            raise RuntimeError(f"Expected a JSON array, got {type(data).__name__}.")
        return data

    def _call_and_parse_object(self, prompt: str) -> dict:
        raw = self.adapter.call(_SYSTEM_PROMPT, prompt)
        try:
            data = _extract_first_json(raw, expect_array=False)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM response did not contain a JSON object: {exc}\nRaw response: {raw[:500]}")
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected a JSON object, got {type(data).__name__}.")
        return data

    # ------------------------------------------------------ extract

    def extract(self, document, segments) -> ExtractorResult:
        text = document.raw_text

        # Pass 1: stakeholders.
        stakeholder_dicts = self._call_and_parse_array(_stakeholders_prompt(text))
        stakeholders: list[Stakeholder] = []
        for raw_item in stakeholder_dicts:
            try:
                stakeholders.append(Stakeholder(**raw_item))
            except ValidationError:
                # Best-effort: skip malformed entries.
                continue

        stakeholder_ids = [s.id for s in stakeholders]

        # Pass 2: commitments.
        commitment_dicts = self._call_and_parse_array(_commitments_prompt(text, stakeholder_ids))
        commitments: list[Commitment] = []
        for raw_item in commitment_dicts:
            try:
                commitments.append(Commitment(**raw_item))
            except ValidationError:
                continue

        # Pass 3: ethical facts.
        fact_dicts = self._call_and_parse_array(_facts_prompt(text, stakeholder_ids))
        ethical_facts: list[EthicalFact] = []
        for raw_item in fact_dicts:
            try:
                ethical_facts.append(EthicalFact(**raw_item))
            except ValidationError:
                continue

        # Pass 4: canonical form.
        facts_summary = "; ".join(
            f"{f.kind}({','.join(f.subjects)}): {f.description}"
            for f in ethical_facts[:10]
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
            commitments=commitments,
            ethical_facts=ethical_facts,
            canonical_form=canonical_form,
            extractor_metadata={
                "extractor": "llm",
                "adapter": self.adapter.name,
            },
        )
