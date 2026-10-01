"""Scene runtime: step a compiled scene's moral state machines on a live event stream.

A Tier-1 scene (ErisML source or structured JSON, loaded into a CompilerIR) declares stakeholders,
commitments, norms and events. This runtime turns it into a running moral state:

- one CommitmentFSM per commitment, one ConsentFSM per stakeholder that has a consent status, and
  one LegitimacyFSM per stakeholder in an authority or protector role;
- after every event, the set of norms in force, and from it the agent's prohibited, obliged and
  allowed actions.

Everything scene-specific lives in the scene, not here:

- `Norm.conditions` says when a norm is in force (see the field's comment in ir.schemas).
- A defeasible norm whose `source` is `commitment:<id>` is lifted while that commitment is not in
  force (any state other than `active`), e.g. a privacy promise overridden in an emergency.
- `extra["conditions"]` names conditions as lists of tokens, so a commitment's
  `defeasibility_conditions` and a norm's `cond:<name>` share one definition.
- `extra["capabilities"]` lists the agent's actions; each is `{"action": name, "elevated": bool}`.
  A norm whose action is `elevated_action` applies to every elevated capability.
- `extra["oversight"]` lists event types that resolve defeasibility in favour of a commitment
  (human oversight restoring it); nothing else restores an overridden commitment.

Event types that drive the machines directly (all optional, generic):

- `consent_given`, `consent_withdrawn`, `coerced_assent` (actor: the consenting stakeholder)
- `procedural_violation`, `coercion_detected`, `legitimacy_restored`, `escalates`,
  `catastrophic_intent`, `evidence_revealed` (target: the authority)
- `commitment_fulfilled`, `commitment_violated`, `commitment_expired` (content: commitment id)
- `action_performed` (content: an action); it discharges obligations for that action

Conditions are re-evaluated after every event, so a commitment becomes overridable as soon as one
of its defeasibility conditions holds, and is restored only by an oversight event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from erisml_compiler.fsm import CommitmentFSM, ConsentFSM, LegitimacyFSM
from erisml_compiler.ir.schemas import CompilerIR, Event, Norm

_CONSENT_INITIAL = {"obtained": "obtained", "coerced": "coerced", "withdrawn": "withdrawn"}
_CONSENT_EVENTS = {
    "consent_given": "consent_given",
    "consent_withdrawn": "withdrawn",
    "coerced_assent": "coerced_assent",
}
_LEGITIMACY_EVENTS = {
    "procedural_violation": "procedural_violation",
    "coercion_detected": "coercion_detected",
    "legitimacy_restored": "restored",
    "escalates": "escalates",
    "catastrophic_intent": "catastrophic_intent",
    "evidence_revealed": "evidence_revealed",
}
_COMMITMENT_EVENTS = {
    "commitment_fulfilled": "fulfilling_event",
    "commitment_violated": "violating_event",
    "commitment_expired": "expiration_event",
}
ELEVATED = "elevated_action"


@dataclass
class Snapshot:
    """The scene's moral state after one event."""

    time_index: int
    event: dict[str, Any] | None
    machines: dict[str, str]
    norms_in_force: list[str]
    prohibited: list[str]
    obliged: list[str]
    allowed: list[str]
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_index": self.time_index,
            "event": self.event,
            "machines": self.machines,
            "norms_in_force": self.norms_in_force,
            "prohibited": self.prohibited,
            "obliged": self.obliged,
            "allowed": self.allowed,
            "reasons": self.reasons,
        }


class SceneRuntime:
    """Running moral state of one compiled scene."""

    def __init__(self, ir: CompilerIR, agent: str | None = None):
        self.ir = ir
        extra = ir.extra or {}
        self.capabilities: list[dict[str, Any]] = list(extra.get("capabilities", []))
        self.conditions: dict[str, list[str]] = {
            k: list(v) for k, v in (extra.get("conditions") or {}).items()
        }
        self.oversight: set[str] = set(extra.get("oversight", []))
        self.agent = agent or extra.get("agent")
        self.commitments = {c.id: (c, CommitmentFSM(c)) for c in ir.commitments}
        self.consent = {
            s.id: ConsentFSM(s.id, _CONSENT_INITIAL.get(s.consent_status or "", "not_obtained"))
            for s in ir.stakeholders
            if s.consent_status and s.consent_status != "n/a"
        }
        self.legitimacy = {
            s.id: LegitimacyFSM(s.id)
            for s in ir.stakeholders
            if {"authority", "protector"} & set(s.roles)
        }
        self.events: list[Event] = []
        self.discharged_at: dict[str, int] = {}  # norm id -> event count when last discharged
        self.restored_at: dict[str, int] = {}  # commitment id -> event count when last restored
        self.time_index = 0
        for e in sorted(ir.events, key=lambda e: e.time_index):
            self.step(e)

    # ------------------------------------------------------------------ conditions
    def machine_states(self) -> dict[str, str]:
        out = {f"commitment:{k}": f.state for k, (_, f) in self.commitments.items()}
        out.update({f"consent:{k}": f.state for k, f in self.consent.items()})
        out.update({f"legitimacy:{k}": f.state for k, f in self.legitimacy.items()})
        return out

    def _event_seen(self, typ: str, content: str | None, since: int) -> bool:
        for e in self.events[since:]:
            if e.type == typ and (content is None or (e.content or "") == content):
                return True
        return False

    def holds(self, token: str, since: int = 0, _depth: int = 0) -> bool:
        """Whether one condition token holds now, counting events from index `since`."""
        if _depth > 16:
            raise ValueError(f"condition definitions recurse too deeply at {token!r}")
        if token.startswith("not:"):
            return not self.holds(token[4:], since, _depth + 1)
        kind, _, rest = token.partition(":")
        if kind == "event":
            typ, _, content = rest.partition("=")
            return self._event_seen(typ, content or None, since)
        if kind == "latest":
            typ, _, content = rest.partition("=")
            for e in reversed(self.events[since:]):
                if e.type == typ:
                    return (e.content or "") == content
            return False
        if kind == "cond":
            if rest not in self.conditions:
                raise KeyError(f"condition {rest!r} is not defined in extra['conditions']")
            return all(self.holds(t, since, _depth + 1) for t in self.conditions[rest])
        if kind == "state":
            mid, _, want = rest.partition("=")
            return self.machine_states().get(mid) == want
        raise ValueError(f"unknown condition token {token!r}")

    # ------------------------------------------------------------------ stepping
    def step(self, event: Event | dict[str, Any]) -> Snapshot:
        if isinstance(event, dict):
            event = Event(
                **{"id": f"live{len(self.events)}", "time_index": self.time_index + 1, **event}
            )
        self.time_index = max(self.time_index, event.time_index)
        self.events.append(event)
        t = event.time_index

        if event.type in _CONSENT_EVENTS and event.actor in self.consent:
            self.consent[event.actor].step(_CONSENT_EVENTS[event.type], t)
        if event.type in _LEGITIMACY_EVENTS and event.target in self.legitimacy:
            self.legitimacy[event.target].step(_LEGITIMACY_EVENTS[event.type], t)
        if event.type in _COMMITMENT_EVENTS and event.content in self.commitments:
            self.commitments[event.content][1].step(_COMMITMENT_EVENTS[event.type], t)

        # defeasibility: overridden when a condition holds; restored only by oversight, and
        # after a restoration only events since then can override it again
        for cid, (c, fsm) in self.commitments.items():
            since = self.restored_at.get(cid, 0)
            if fsm.state == "active" and any(
                self.holds(f"cond:{name}", since)
                for name in c.defeasibility_conditions
                if name in self.conditions
            ):
                fsm.step("defeasibility_condition_triggered", t)
            elif (
                fsm.state == "active_but_defeasible"
                and event.type in self.oversight
                and (event.content in (None, "", cid) or event.target == cid)
            ):
                fsm.step("defeasibility_resolved_in_favor", t)
                self.restored_at[cid] = len(self.events)

        if event.type == "action_performed":
            for n in self.ir.norms:
                if n.modality == "obligation" and n.action == event.content:
                    self.discharged_at[n.id] = len(self.events)
        return self.snapshot(event)

    # ------------------------------------------------------------------ norms and actions
    def in_force(self, n: Norm) -> bool:
        if n.source.startswith("commitment:"):
            cid = n.source.split(":", 1)[1]
            if (
                n.defeasible
                and cid in self.commitments
                and self.commitments[cid][1].state != "active"
            ):
                return False
        since = self.discharged_at.get(n.id, 0) if n.modality == "obligation" else 0
        return all(self.holds(tok, since) for tok in n.conditions)

    def _applies(self, n: Norm, action: str) -> bool:
        if self.agent and n.actor != self.agent:
            return False
        if n.action == action:
            return True
        return n.action == ELEVATED and any(
            c["action"] == action and c.get("elevated") for c in self.capabilities
        )

    def snapshot(self, event: Event | None = None) -> Snapshot:
        active = [n for n in self.ir.norms if self.in_force(n)]
        names = [c["action"] for c in self.capabilities]
        reasons: dict[str, list[str]] = {}
        prohibited, obliged = [], []
        for a in names:
            pro = [n for n in active if n.modality == "prohibition" and self._applies(n, a)]
            perm = [
                n
                for n in active
                if n.modality in ("permission", "exception")
                and self._applies(n, a)
                and n.action != ELEVATED
            ]
            # a permission lifts a prohibition only from a strictly higher priority (lower tier)
            if pro and (
                not perm or min(n.priority_tier for n in pro) <= min(n.priority_tier for n in perm)
            ):
                prohibited.append(a)
                reasons[a] = [n.id for n in pro]
        for n in active:
            if n.modality == "obligation" and (not self.agent or n.actor == self.agent):
                obliged.append(n.action)
                reasons.setdefault(n.action, []).append(n.id)
        allowed = [a for a in names if a not in prohibited]
        return Snapshot(
            time_index=self.time_index,
            event=event.model_dump(exclude_none=True) if event is not None else None,
            machines=self.machine_states(),
            norms_in_force=[n.id for n in active],
            prohibited=prohibited,
            obliged=sorted(set(obliged)),
            allowed=allowed,
            reasons=reasons,
        )
