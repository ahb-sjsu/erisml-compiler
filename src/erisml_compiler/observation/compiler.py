"""Observation compilation: the scene IR updated from what is perceived.

The bond extractor of the Bond Invariance Principle, run continuously. Given the current scene IR,
a batch of observations (speech, text on a screen, messages, structured perception, video-witness
observables) and the names of the acts the agent can perform, an ObservationCompiler asks a model
for typed elements and returns an ObservationPatch: ops in the correction format
(erisml_compiler.correction), restricted to an allow-list, every element validated.

The model writes elements, never ops. The compiler validates each element against the IR schema
and the observation vocabulary below, assigns ids and time order itself, rejects what fails with a
reason, and builds the ops. `apply_observation_patch` checks the allow-list again before applying,
through `apply_corrections`, so each batch is recorded in the IR's correction history with the
hashes before and after.

The observation vocabulary is general, a small set of speech acts and states, never a list of
cases: an utterance; a consent or refusal of a named act; a request for a named act; a claim a
person makes about themselves; a presence (arrived or left); a hazard (how soon severe harm is
expected for whom). Acts are named only from the agent's own capability vocabulary. Ethical facts
use the IR's fourteen kinds.

What an observation can never change: norms, commitments, the document, `extra`. Those belong to
the scene's authors (the service agreement, the constitutional set, the capability manifest).
Events are append-only, so a scene rebuilt from the patched IR replays exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from erisml_compiler.annotation.llm_extractor import ModelAdapter, _extract_first_json
from erisml_compiler.ir.schemas import CompilerIR, EthicalFact, Event, Relation, Stakeholder
from erisml_compiler.observation import clock as vc

OBSERVATION_EVENTS: Dict[str, Dict[str, Any]] = {
    "utterance": {
        "description": "a person said something; actor is the speaker, content the words"
    },
    "consent": {
        "description": "a person consents to or refuses, in their own words, an act done for or to them; "
        "target is the act, content consented or refused",
        "content": ["consented", "refused"],
        "target_is_act": True,
    },
    "request": {
        "description": "a person asks for an act to be done; target is the act, content what they said",
        "target_is_act": True,
    },
    "claim": {
        "description": "a person asserts something about themselves (who they are, why they are here); "
        "content is the claim; a claim is evidence only that it was made"
    },
    "presence": {
        "description": "a person or animal arrives or leaves; content arrived or left",
        "content": ["arrived", "left"],
    },
    "hazard": {
        "description": "a danger to someone; target is who is endangered, content how soon severe harm is "
        "expected without intervention",
        "content": ["none", "latent", "emerging", "imminent"],
    },
}
SEVERITIES = ("minor", "moderate", "grave", "catastrophic")
VULNERABILITY = ("low", "moderate", "high", "extreme")


def _choices(model: Any, name: str) -> List[str]:
    """The allowed values of a Literal field (or of a list of Literals), read from the schema, so the
    prompt can never drift from what validation accepts."""
    import typing

    ann = model.model_fields[name].annotation
    while typing.get_origin(ann) in (list, List):
        ann = typing.get_args(ann)[0]
    return list(typing.get_args(ann))


STAKEHOLDER_FIELDS_SETTABLE = ("label", "vulnerability", "agency")

_SYSTEM = (
    "You update the structured model of a situation from new observations. You report only what the "
    "observations show, never what might follow, and only what is new. Answer with one JSON object with "
    "the keys stakeholders, events, ethical_facts and relations, each a list (empty when nothing is new)."
)


@dataclass(frozen=True)
class Observation:
    """One observation. `kind` is speech, screen, message, perception or video; `source` who or what
    produced it; `content` the words, or a mapping for structured perception."""

    kind: str
    source: str
    content: Any


@dataclass
class ObservationPatch:
    ops: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    review: List[str]
    agreement: Optional[float]
    ir_hash_before: str
    observations_hash: str
    base_clock: Dict[str, int] = field(
        default_factory=dict
    )  # the snapshot's clock: what the observer had seen


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _scene_summary(ir: CompilerIR, recent: int = 12) -> Dict[str, Any]:
    return {
        "stakeholders": [{"id": s.id, "label": s.label, "type": s.type} for s in ir.stakeholders],
        "recent_events": [
            {"type": e.type, "actor": e.actor, "target": e.target, "content": e.content}
            for e in sorted(ir.events, key=lambda e: e.time_index)[-recent:]
        ],
        "ethical_facts": [
            {
                "id": f.id,
                "kind": f.kind,
                "subjects": f.subjects,
                "description": f.description,
                "severity": f.severity,
            }
            for f in ir.ethical_facts
        ],
    }


def _prompt(ir: CompilerIR, observations: Sequence[Observation], acts: Mapping[str, str]) -> str:
    return json.dumps(
        {
            "scene": _scene_summary(ir),
            "acts": dict(acts),
            "event_types": {
                k: {kk: vv for kk, vv in v.items() if kk != "target_is_act"}
                for k, v in OBSERVATION_EVENTS.items()
            },
            "ethical_fact_kinds": list(EthicalFact.model_fields["kind"].annotation.__args__),
            "severities": list(SEVERITIES),
            "element_fields": {
                "stakeholders": {
                    "id": "snake_case, new only (reuse the scene's ids for anyone already in it)",
                    "label": "short name",
                    "type": _choices(Stakeholder, "type"),
                    "roles": _choices(Stakeholder, "roles"),
                    "vulnerability": list(VULNERABILITY),
                },
                "events": {
                    "type": "one of event_types",
                    "actor": "a stakeholder id",
                    "target": "a stakeholder id, or for consent and request an act from acts",
                    "content": "as the event type says",
                },
                "ethical_facts": {
                    "kind": "one of ethical_fact_kinds",
                    "subjects": "stakeholder ids",
                    "description": "one sentence",
                    "severity": "one of severities",
                    "confidence": "a number from 0 to 1",
                },
                "relations": {
                    "type": "snake_case",
                    "source": "a stakeholder id",
                    "target": "a stakeholder id",
                    "description": "one sentence",
                },
            },
            "rules": [
                "A consent to, or refusal of, something the agent can do is an event of type consent "
                "whose target is that act's name from acts; never an ethical fact or a relation.",
                "A standing agreement, a person's stated wishes, or a request each give consent or "
                "request events naming acts.",
                "Use only the listed values; omit a field you cannot fill from the observations.",
            ],
            "observations": [
                {"kind": o.kind, "source": o.source, "content": o.content} for o in observations
            ],
        },
        ensure_ascii=False,
        default=str,
    )


class ObservationCompiler:
    """`adapter` asks the model; an optional `critic` adapter asks a second time, and the agreement
    between the two (Jaccard over element signatures) is reported, with every element only one of
    them found listed for review."""

    def __init__(self, adapter: ModelAdapter, critic: Optional[ModelAdapter] = None):
        self.adapter, self.critic = adapter, critic

    def _ask(self, adapter: ModelAdapter, user: str) -> Dict[str, Any]:
        raw = _extract_first_json(adapter.call(_SYSTEM, user), expect_array=False)
        return raw if isinstance(raw, dict) else {}

    def compile(
        self, ir: CompilerIR, observations: Sequence[Observation], acts: Mapping[str, str]
    ) -> ObservationPatch:
        user = _prompt(ir, observations, acts)
        raw = self._ask(self.adapter, user)
        ops, rejected, signatures = self._validate(ir, raw, acts)
        agreement, review = None, []
        if self.critic is not None:
            _, _, critic_signatures = self._validate(ir, self._ask(self.critic, user), acts)
            union = signatures | critic_signatures
            agreement = 1.0 if not union else len(signatures & critic_signatures) / len(union)
            review = sorted(" ".join(map(str, s)) for s in signatures ^ critic_signatures)
        return ObservationPatch(
            ops,
            rejected,
            review,
            agreement,
            _hash(ir.model_dump(mode="json")),
            _hash([o.__dict__ for o in observations]),
            base_clock=history_clock(ir),
        )

    def _validate(self, ir: CompilerIR, raw: Mapping[str, Any], acts: Mapping[str, str]):
        ops: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        signatures: set = set()
        known = {s.id for s in ir.stakeholders}
        taken = (
            known
            | {e.id for e in ir.events}
            | {f.id for f in ir.ethical_facts}
            | {r.id for r in ir.relations}
        )
        t = max((e.time_index for e in ir.events), default=-1) + 1
        n = len(ir.events)

        def reject(kind: str, element: Any, reason: str) -> None:
            rejected.append({"kind": kind, "element": element, "reason": reason})

        for s in raw.get("stakeholders") or []:
            try:
                model = Stakeholder.model_validate({**s, "requires_review": True})
            except (ValidationError, TypeError) as e:
                reject("stakeholder", s, f"schema: {str(e).splitlines()[0]}")
                continue
            if model.id in taken:
                reject("stakeholder", s, f"id {model.id!r} exists")
                continue
            if model.vulnerability is not None and model.vulnerability not in VULNERABILITY:
                reject(
                    "stakeholder",
                    s,
                    f"vulnerability {model.vulnerability!r} is not one of {VULNERABILITY}",
                )
                continue
            taken.add(model.id)
            known.add(model.id)
            ops.append(
                {
                    "op": "add",
                    "path": f"stakeholders.{model.id}",
                    "value": model.model_dump(mode="json"),
                }
            )
            signatures.add(("stakeholder", model.id))

        for i, e in enumerate(raw.get("events") or []):
            if not isinstance(e, dict) or e.get("type") not in OBSERVATION_EVENTS:
                reject(
                    "event",
                    e,
                    f"type {e.get('type') if isinstance(e, dict) else e!r} is not an observation type",
                )
                continue
            spec = OBSERVATION_EVENTS[e["type"]]
            actor, target, content = e.get("actor"), e.get("target"), e.get("content")
            if actor is not None and actor not in known:
                reject("event", e, f"actor {actor!r} is not a stakeholder")
                continue
            if spec.get("target_is_act"):
                if target not in acts:
                    reject("event", e, f"target {target!r} is not one of the acts")
                    continue
            elif target is not None and target not in known:
                reject("event", e, f"target {target!r} is not a stakeholder")
                continue
            if "content" in spec and content not in spec["content"]:
                reject("event", e, f"content {content!r} is not one of {spec['content']}")
                continue
            if content is not None and not isinstance(content, str):
                reject("event", e, "content is not a string")
                continue
            eid = f"obs{n}_{i}"
            while eid in taken:
                eid += "_"
            taken.add(eid)
            model = Event(
                id=eid, time_index=t, type=e["type"], actor=actor, target=target, content=content
            )
            ops.append(
                {"op": "add", "path": f"events.{eid}", "value": model.model_dump(mode="json")}
            )
            signatures.add(
                ("event", e["type"], actor, target, content if spec.get("content") else None)
            )

        for i, f in enumerate(raw.get("ethical_facts") or []):
            if not isinstance(f, dict):
                reject("ethical_fact", f, "not an object")
                continue
            fid = f"fact{n}_{i}"
            while fid in taken:
                fid += "_"
            try:
                model = EthicalFact.model_validate({**f, "id": fid})
            except (ValidationError, TypeError) as err:
                reject("ethical_fact", f, f"schema: {str(err).splitlines()[0]}")
                continue
            if model.severity is not None and model.severity not in SEVERITIES:
                reject("ethical_fact", f, f"severity {model.severity!r} is not one of {SEVERITIES}")
                continue
            unknown = [x for x in model.subjects if x not in known]
            if unknown:
                reject("ethical_fact", f, f"subjects {unknown} are not stakeholders")
                continue
            taken.add(fid)
            ops.append(
                {
                    "op": "add",
                    "path": f"ethical_facts.{fid}",
                    "value": model.model_dump(mode="json"),
                }
            )
            signatures.add(("fact", model.kind, tuple(sorted(model.subjects)), model.severity))

        for i, r in enumerate(raw.get("relations") or []):
            rid = f"rel{n}_{i}"
            try:
                model = Relation.model_validate({**r, "id": rid})
            except (ValidationError, TypeError) as err:
                reject("relation", r, f"schema: {str(err).splitlines()[0]}")
                continue
            if model.source not in known or model.target not in known:
                reject("relation", r, "source or target is not a stakeholder")
                continue
            taken.add(rid)
            ops.append(
                {"op": "add", "path": f"relations.{rid}", "value": model.model_dump(mode="json")}
            )
            signatures.add(("relation", model.type, model.source, model.target))
        return ops, rejected, signatures


ALLOWED_OP_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("add", "stakeholders."),
    ("add", "events."),
    ("add", "ethical_facts."),
    ("add", "relations."),
)


class PatchNotAllowed(ValueError):
    """An op outside the observation allow-list."""


def check_allowed(ops: Sequence[Mapping[str, Any]]) -> None:
    for op in ops:
        kind, path = op.get("op"), str(op.get("path", ""))
        parts = path.split(".")
        if (
            not any(kind == k and path.startswith(p) for k, p in ALLOWED_OP_PREFIXES)
            or len(parts) != 2
        ):
            raise PatchNotAllowed(f"{kind} {path} is outside the observation allow-list")


def history_clock(ir: CompilerIR) -> Dict[str, int]:
    """The clock that has seen every event of the IR."""
    return vc.frontier(e.clock for e in ir.events)


def _rebased(
    ir: CompilerIR, ops: Sequence[Mapping[str, Any]], base_clock: Mapping[str, int]
) -> List[Dict[str, Any]]:
    """The patch's events stamped against the IR it is applied to, not the snapshot it was compiled
    from. A patch compiled while other events were appended (a caller compiles outside its lock)
    would otherwise sort its events before the newer ones and rewrite the order a running scene had
    already stepped. Observed events take the next time index of this IR; ids that this IR already
    uses are renamed. Each takes the observer's next tick on the snapshot's clock, so it comes after
    what the observer saw and after the observer's own earlier events, and is concurrent with what
    was appended since the snapshot (erisml_compiler.observation.clock)."""
    t = max((e.time_index for e in ir.events), default=-1) + 1
    taken = (
        {e.id for e in ir.events}
        | {f.id for f in ir.ethical_facts}
        | {r.id for r in ir.relations}
        | {s.id for s in ir.stakeholders}
    )
    if not vc.leq(base_clock, history_clock(ir)):
        raise PatchNotAllowed("the patch was compiled against a history this IR has not seen")
    seen = vc.merge(base_clock, {"observer": history_clock(ir).get("observer", 0)})
    out = []
    for op in ops:
        op = dict(op)
        if op.get("op") == "add" and str(op.get("path", "")).startswith("events."):
            seen = vc.tick(seen, "observer")
            value = dict(op["value"], time_index=t, clock=dict(seen))
            eid = value["id"]
            while eid in taken:
                eid += "_"
            value["id"] = eid
            taken.add(eid)
            op.update(path=f"events.{eid}", value=value)
        out.append(op)
    return out


def apply_observation_patch(ir: CompilerIR, patch: ObservationPatch) -> CompilerIR:
    """Apply a patch after checking the allow-list again; every op must apply, or none does."""
    from erisml_compiler.correction.corrector import apply_corrections

    check_allowed(patch.ops)
    ops = _rebased(ir, patch.ops, patch.base_clock)
    new_ir, record, summaries = apply_corrections(
        ir,
        {
            "corrector_id": "observation_compiler",
            "rationale": f"observations {patch.observations_hash}",
            "patches": ops,
        },
    )
    if record.n_patches_failed:
        raise ValueError(
            "observation patch did not apply cleanly: "
            + "; ".join(s for s in summaries if "FAILED" in s)
        )
    return new_ir
