"""Scene agent: observe, classify into scene events, step the moral state, choose an action.

The two learned steps are deliberately narrow and checked:

- ObservationClassifier turns perception facts (any JSON) into events of the scene's declared
  vocabulary, `extra["event_types"]` (`{type: {"description": ..., "content": [...] optional,
  "source": "system" optional}}`). Types marked `source: system` (a ruling, an action the agent
  performed, an authenticated oversight message) are never offered to it and are rejected if it
  proposes one. A proposed event of an undeclared type, or with a content outside a declared list,
  is rejected and reported, never stepped.
- ActionChooser picks one action from the runtime's *allowed* set for the current snapshot, with
  the obligations in force listed first. A choice outside the allowed set is rejected; the agent
  then falls back to the first allowed obligation, or to `extra["default_action"]`.

Both speak to a model through `annotation.llm_extractor.ModelAdapter`, so the mock adapter tests
them offline and the NRP adapter runs them for real. Neither can change the norms, the machines
or the allowed set; they only read them. Authority for elevated actions stays with whatever gate
the caller applies (in the home-care twin, the governor).

The agent's role (`extra["role"]`, e.g. "a monitoring-centre operator") names who is deciding in
both prompts, so one runtime serves every agent of a multi-agent scene; it defaults to the care
robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from erisml_compiler.annotation.llm_extractor import ModelAdapter, _extract_first_json
from erisml_compiler.runtime.scene import SceneRuntime, Snapshot

_CLASSIFY_SYSTEM = (
    "You turn the observations of {role} into events of a declared vocabulary. Use only the "
    "declared event types and, where a type lists contents, only those contents. Report what the "
    "facts show, not what might follow, and only what is new since the recent events (do not "
    "repeat an event that is already there and unchanged). Answer with a JSON array of objects "
    "with keys type, "
    "actor, target, content, conditions (list of strings). An empty array is a valid answer."
)
_CHOOSE_SYSTEM = (
    "You choose the next action of {role}. You may choose only from the allowed actions. "
    "Obligations in force come first unless an allowed action better protects the person. Without "
    "an obligation in force or a sign in the facts that someone needs something, choose the "
    "default action and do not intrude. Answer "
    "with one JSON object with keys action, args (object; for speak, args.text), reason (one "
    "sentence)."
)
# who the agent is, from the scene's `extra["role"]`; a scene without one is the care robot the
# runtime was first written for
_DEFAULT_ROLE = "a care robot"


def _role(runtime: SceneRuntime) -> str:
    return str((runtime.ir.extra or {}).get("role") or _DEFAULT_ROLE)


@dataclass
class Decision:
    events: list[dict[str, Any]]
    rejected_events: list[dict[str, Any]]
    snapshot: dict[str, Any]
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    chooser_rejected: dict[str, Any] | None = None
    fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "rejected_events": self.rejected_events,
            "snapshot": self.snapshot,
            "action": self.action,
            "args": self.args,
            "reason": self.reason,
            "chooser_rejected": self.chooser_rejected,
            "fallback": self.fallback,
        }


class ObservationClassifier:
    """The canonicalizer: perception facts in, events of the scene's vocabulary out.

    Isolated (``free_text=False``), it is also the boundary of mandatory canonicalization: an
    event's actor must be one the scene declares (``extra["actors"]``, id to description), snapped
    there by ``canonicalizer`` (erisml_compiler.canonicalizer) or else ``"unknown"``; and free-text
    content (an event type with no declared content list) is never stepped, only reported in
    ``last_quarantined``. What is stepped, and so what any later reader of the moral state sees, is
    then canonical: declared types, declared contents, declared actors.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        runtime: SceneRuntime,
        canonicalizer: Any = None,
        free_text: bool = True,
    ):
        self.adapter, self.rt = adapter, runtime
        extra = runtime.ir.extra or {}
        declared = dict(extra.get("event_types", {}))
        self.system = {
            k for k, v in declared.items() if isinstance(v, dict) and v.get("source") == "system"
        }
        self.vocab: dict[str, dict[str, Any]] = {
            k: v for k, v in declared.items() if k not in self.system
        }
        self.actors: dict[str, str] = dict(extra.get("actors") or {})
        self.canonicalizer, self.free_text = canonicalizer, free_text
        self.last_snapped: list[dict[str, Any]] = []
        self.last_quarantined: list[dict[str, Any]] = []

    def _check(self, e: Any) -> str | None:
        if not isinstance(e, dict) or "type" not in e:
            return "not an event object"
        if e["type"] in self.system:
            return f"{e['type']!r} comes from the system, not from perception"
        spec = self.vocab.get(e["type"])
        if spec is None:
            return f"undeclared event type {e['type']!r}"
        allowed = spec.get("content")
        if allowed and (e.get("content") or "") not in allowed:
            return f"content {e.get('content')!r} not among {allowed}"
        return None

    def _canonical_actor(self, actor: Any) -> str:
        a = str(actor or "")
        if not self.actors or a in self.actors:
            return a
        tag = None
        if self.canonicalizer is not None and a:
            tag = self.canonicalizer.canonicalize(a, self.actors).tag
        tag = tag if tag in self.actors else "unknown"
        self.last_snapped.append({"from": a, "to": tag})
        return tag

    def classify(self, facts: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.last_snapped, self.last_quarantined = [], []
        recent = [e.model_dump(exclude_none=True) for e in self.rt.events[-8:]]
        user = json.dumps(
            {
                "scene": self.rt.ir.document.raw_text,
                "event_types": self.vocab,
                "recent_events": recent,
                "facts": facts,
            },
            ensure_ascii=False,
        )
        raw = (
            _extract_first_json(
                self.adapter.call(_CLASSIFY_SYSTEM.format(role=_role(self.rt)), user),
                expect_array=True,
            )
            or []
        )
        ok, bad = [], []
        for e in raw:
            why = self._check(e)
            if why:
                bad.append({"event": e, "why": why})
                continue
            ev = {
                k: e[k]
                for k in ("type", "actor", "target", "content", "conditions")
                if e.get(k) not in (None, "")
            }
            if not self.free_text:
                if "actor" in ev:
                    ev["actor"] = self._canonical_actor(ev["actor"])
                if "target" in ev:
                    ev["target"] = self._canonical_actor(ev["target"])
                ev.pop("conditions", None)
                if "content" in ev and not self.vocab[ev["type"]].get("content"):
                    self.last_quarantined.append({"type": ev["type"], "content": ev.pop("content")})
            ok.append(ev)
        return ok, bad


# what an isolated chooser may still read from the caller's context, besides the moral state
_CANONICAL_CONTEXT = ("governor_ruling", "requested")


class ActionChooser:
    """Chooses one action from the allowed set.

    With ``view="canonical"`` it never sees the perception facts: only the canonical event history
    the runtime has stepped, the moral state, and the allowed, obliged and prohibited actions (the
    No Escape theorem's mandatory canonicalization; erisml-lib
    docs/papers/foundations/no_escape.tex). Raw text in the world (a television, a message, a
    stranger's words) then cannot reach the model that picks the action.
    """

    def __init__(self, adapter: ModelAdapter, runtime: SceneRuntime, view: str = "facts"):
        if view not in ("facts", "canonical"):
            raise ValueError(f"view must be 'facts' or 'canonical', not {view!r}")
        self.adapter, self.rt, self.view = adapter, runtime, view
        self.caps = {c["action"]: c for c in (runtime.ir.extra or {}).get("capabilities", [])}

    def _situation(self, facts: Any) -> dict[str, Any]:
        if self.view == "facts":
            return {"facts": facts}
        recent = [
            {
                k: v
                for k, v in e.model_dump(exclude_none=True).items()
                if k in ("type", "actor", "target", "content")
            }
            for e in self.rt.events[-16:]
        ]
        context = {
            k: facts[k] for k in _CANONICAL_CONTEXT if isinstance(facts, dict) and k in facts
        }
        return {"recent_events": recent, **({"context": context} if context else {})}

    def choose(
        self, snap: Snapshot, facts: Any
    ) -> tuple[str, dict[str, Any], str, dict[str, Any] | None, bool]:
        allowed = snap.allowed
        obliged = [a for a in snap.obliged if a in allowed]
        user = json.dumps(
            {
                "scene": self.rt.ir.document.raw_text,
                **self._situation(facts),
                "obligations_in_force": obliged,
                "default_action": (self.rt.ir.extra or {}).get("default_action"),
                "allowed_actions": {a: self.caps.get(a, {}) for a in allowed},
                "prohibited_actions": snap.prohibited,
                "moral_state": snap.machines,
            },
            ensure_ascii=False,
        )
        out = (
            _extract_first_json(
                self.adapter.call(_CHOOSE_SYSTEM.format(role=_role(self.rt)), user),
                expect_array=False,
            )
            or {}
        )
        act = out.get("action") if isinstance(out, dict) else None
        if act in allowed:
            return act, dict(out.get("args") or {}), str(out.get("reason", ""))[:400], None, False
        default = (self.rt.ir.extra or {}).get("default_action")
        fb = (
            obliged[0]
            if obliged
            else (default if default in allowed else (allowed[0] if allowed else ""))
        )
        return fb, {}, "the proposed action was not allowed; fell back", {"proposed": out}, True


class SceneAgent:
    """One decision cycle: facts in, a checked action out, the moral state stepped in between.

    ``isolated=True`` separates the two models: the classifier (the canonicalizer) reads the facts
    and writes canonical events only; the chooser reads only the canonical state.
    """

    def __init__(
        self,
        runtime: SceneRuntime,
        adapter: ModelAdapter,
        isolated: bool = False,
        canonicalizer: Any = None,
    ):
        self.rt = runtime
        self.isolated = isolated
        self.classifier = ObservationClassifier(
            adapter, runtime, canonicalizer=canonicalizer, free_text=not isolated
        )
        self.chooser = ActionChooser(adapter, runtime, view="canonical" if isolated else "facts")

    def decide(self, facts: Any) -> Decision:
        events, rejected = self.classifier.classify(facts)
        snap = self.rt.snapshot()
        for e in events:
            snap = self.rt.step(e)
        action, args, reason, rej, fb = self.chooser.choose(snap, facts)
        return Decision(events, rejected, snap.as_dict(), action, args, reason, rej, fb)

    def record(self, event: dict[str, Any]) -> Snapshot:
        """Step an event that did not come from perception (a ruling, a performed action)."""
        return self.rt.step(event)
