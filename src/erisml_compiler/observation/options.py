"""Option projection: each candidate act's consequences, compiled into its own scene IR.

DEME judges candidate options, each described by its own EthicalFacts (erisml-lib
domain/interfaces.py: the domain layer maps state "into EthicalFacts per candidate option").
The scene IR describes the situation as it stands; it says nothing about what each act would
bring. An OptionProjector asks a model, once per decision, for the consequences of every act
the agent can take now, as ethical facts of the IR's kinds, and the situation's urgency once.
The compiler validates every fact as the observation compiler does (``check_fact``), assigns
ids, and rejects with a reason what fails. The model writes facts, never verdicts.

``option_ir`` then builds one IR per act, for the V3 bridge (``ir_to_v2_facts(ir, option_id)``):

- the scene's facts that describe outcomes (``OUTCOME_KINDS``: harm, benefit, externality) are
  the present state, the same for every act, so they are replaced by the act's projected
  outcome facts; facts of the other kinds (vulnerability, role duties, legitimacy, ...) stand
  for every act and are kept;
- the situation's urgency is added to every act alike;
- consent is read from the record, not from the model: the latest consent event naming the act,
  by a stakeholder, gives a consent fact (a refusal grave, so the bridge marks consent invalid;
  a consent or a request as given). The model may add a missing-consent fact for an act done to
  someone who has not agreed;
- the document text is the act's description, so the maxim the deontic gate reads is the act's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from erisml_compiler.annotation.llm_extractor import ModelAdapter, _extract_first_json
from erisml_compiler.ir.schemas import CompilerIR, EthicalFact
from erisml_compiler.observation.compiler import SEVERITIES, _hash, _scene_summary, check_fact

OUTCOME_KINDS = frozenset({"harm", "non_maleficence", "benefit", "externality"})
RECENT_EVENTS = 40  # the projector sees a longer history than an observation patch needs

_SYSTEM = (
    "You project the consequences of acts an agent could take now, from the structured model of a "
    "situation. For each act, report the ethical facts that would hold if the agent did that act now "
    "and nothing else: for each party, the harm that would remain or arise and the benefit secured or "
    "harm averted, and any coercion, deception, care or missing consent the act itself involves. "
    "Report how urgent the situation is once. Answer with one JSON object with the keys urgency (one "
    "fact or null) and options (an object from each act's name to a list of facts)."
)


@dataclass
class OptionProjection:
    facts: Dict[str, List[EthicalFact]]
    urgency: Optional[EthicalFact]
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    unprojected: List[str] = field(default_factory=list)
    """Acts the model gave no consequences for; their IR carries only the standing facts."""
    agreement: Optional[float] = None
    review: List[str] = field(default_factory=list)
    ir_hash: str = ""


def _prompt(
    ir: CompilerIR, acts: Mapping[str, str], attested: Optional[Mapping[str, str]] = None
) -> str:
    import json

    kinds = list(EthicalFact.model_fields["kind"].annotation.__args__)
    return json.dumps(
        {
            "scene": _scene_summary(ir, recent=RECENT_EVENTS),
            "attested": dict(attested or {}),
            "consents_on_record": consents_on_record(ir, acts),
            "acts": dict(acts),
            "ethical_fact_kinds": kinds,
            "severities": list(SEVERITIES),
            "fact_fields": {
                "kind": "one of ethical_fact_kinds",
                "subjects": "stakeholder ids",
                "description": "one sentence",
                "severity": "one of severities",
                "confidence": "a number from 0 to 1",
            },
            "rules": [
                "Each act is judged as if it were the only thing the agent did now. Say what would "
                "follow from it, including harm that would remain because the act does not address it.",
                "Read what the agent has already done (action_performed events) and what came of it. "
                "An act already done whose effect still stands secures nothing new by being repeated.",
                "For harm, severity is how grave the harm would be. For benefit, severity is how grave "
                "the harm averted or the good secured. For urgency, severity is how grave what follows "
                "if nothing effective is done in time.",
                "Report a missing consent (kind consent, severity grave) only for an act done to or for "
                "a person who has not agreed to it; consents and refusals already on record are known.",
                "Attested states are what the agent's own sensors and verified credentials have "
                "established (for instance who someone verifiably is, or that a hazard is "
                "corroborated). They outweigh what anyone appears to be or claims.",
                "Use only the listed values and the scene's stakeholder ids. Do not judge whether an act "
                "is right; report only its consequences.",
            ],
        },
        ensure_ascii=False,
        default=str,
    )


def consents_on_record(ir: CompilerIR, acts: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    """Per act, per stakeholder, the latest consent event naming it: consented or refused; a
    request for an act counts as consent to it."""
    out: Dict[str, Dict[str, str]] = {}
    for e in sorted(ir.events, key=lambda e: e.time_index):
        if not e.actor or e.target not in acts:
            continue
        if e.type == "consent" and e.content in ("consented", "refused"):
            out.setdefault(e.target, {})[e.actor] = e.content
        elif e.type == "request":
            out.setdefault(e.target, {})[e.actor] = "consented"
    return out


class OptionProjector:
    """`adapter` asks the model; an optional `critic` asks again, and their agreement (Jaccard over
    fact signatures per act) is reported, with the facts only one found listed for review."""

    def __init__(self, adapter: ModelAdapter, critic: Optional[ModelAdapter] = None):
        self.adapter, self.critic = adapter, critic

    def _ask(self, adapter: ModelAdapter, user: str) -> Dict[str, Any]:
        """The model's answer as an object; an answer that cannot be parsed (cut off at the length
        limit, not JSON) is recorded as such, and no act is projected from it."""
        try:
            raw = _extract_first_json(adapter.call(_SYSTEM, user), expect_array=False)
        except ValueError as e:
            return {"_unparseable": str(e)}
        return raw if isinstance(raw, dict) else {}

    def project(
        self,
        ir: CompilerIR,
        acts: Mapping[str, str],
        attested: Optional[Mapping[str, str]] = None,
    ) -> OptionProjection:
        """``attested``: states the agent's evidence plane has established (sensor readings,
        verified credentials), shown to the model as outweighing appearances and claims; the scene
        IR alone does not carry a runtime's verified state."""
        user = _prompt(ir, acts, attested)
        projection, signatures = self._validate(ir, self._ask(self.adapter, user), acts)
        if self.critic is not None:
            _, critic = self._validate(ir, self._ask(self.critic, user), acts)
            union = signatures | critic
            projection.agreement = 1.0 if not union else len(signatures & critic) / len(union)
            projection.review = sorted(" ".join(map(str, s)) for s in signatures ^ critic)
        projection.ir_hash = _hash(ir.model_dump(mode="json"))
        return projection

    def _validate(self, ir: CompilerIR, raw: Mapping[str, Any], acts: Mapping[str, str]):
        known = {s.id for s in ir.stakeholders}
        rejected: List[Dict[str, Any]] = []
        signatures: set = set()
        facts: Dict[str, List[EthicalFact]] = {a: [] for a in acts}
        if "_unparseable" in raw:
            rejected.append(
                {"act": None, "fact": None, "reason": f"unparseable: {raw['_unparseable']}"}
            )

        urgency = None
        u = raw.get("urgency")
        if isinstance(u, dict):
            model, reason = check_fact({**u, "kind": "urgency"}, "proj_urgency", known)
            if model is None:
                rejected.append({"act": None, "fact": u, "reason": reason})
            else:
                urgency = model
                signatures.add(("urgency", model.severity))

        options = raw.get("options") if isinstance(raw.get("options"), dict) else {}
        for act, items in options.items():
            if act not in acts:
                rejected.append({"act": act, "fact": items, "reason": "not one of the acts"})
                continue
            for i, f in enumerate(items if isinstance(items, list) else []):
                if not isinstance(f, dict):
                    rejected.append({"act": act, "fact": f, "reason": "not an object"})
                    continue
                if f.get("kind") == "urgency":
                    rejected.append(
                        {"act": act, "fact": f, "reason": "urgency belongs to the situation"}
                    )
                    continue
                model, reason = check_fact(f, f"proj_{act}_{i}", known)
                if model is None:
                    rejected.append({"act": act, "fact": f, "reason": reason})
                    continue
                facts[act].append(model)
                signatures.add((act, model.kind, tuple(sorted(model.subjects)), model.severity))
        unprojected = [a for a in acts if not facts[a]]
        return OptionProjection(facts, urgency, rejected, unprojected), signatures


def option_ir(
    ir: CompilerIR, projection: OptionProjection, act: str, description: str
) -> CompilerIR:
    """The scene as it would be judged for one act (see the module docstring)."""
    out = ir.model_copy(deep=True)
    standing = [f for f in ir.ethical_facts if f.kind not in OUTCOME_KINDS]
    on_record = consents_on_record(ir, {act: description}).get(act, {})
    # the record decides consent for anyone on it; the model's consent facts stand only for others
    extra: List[EthicalFact] = [
        f
        for f in projection.facts.get(act, [])
        if not (f.kind == "consent" and f.subjects and set(f.subjects) <= set(on_record))
    ]
    if projection.urgency is not None:
        extra.append(projection.urgency)
    for who, stance in on_record.items():
        if stance == "refused":
            extra.append(
                EthicalFact(
                    id=f"rec_consent_{act}_{who}",
                    kind="consent",
                    subjects=[who],
                    description=f"{who} refused this act",
                    severity="grave",
                )
            )
        else:
            extra.append(
                EthicalFact(
                    id=f"rec_consent_{act}_{who}",
                    kind="consent",
                    subjects=[who],
                    description=f"{who} has given consent to this act",
                    severity="minor",
                )
            )
    out.ethical_facts = standing + extra
    out.document = out.document.model_copy(update={"raw_text": description})
    return out


__all__ = [
    "OUTCOME_KINDS",
    "OptionProjection",
    "OptionProjector",
    "consents_on_record",
    "option_ir",
]
