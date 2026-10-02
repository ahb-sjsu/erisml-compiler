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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from erisml_compiler.annotation.llm_extractor import ModelAdapter, _extract_first_json
from erisml_compiler.runtime.scene import SceneRuntime, Snapshot

_CLASSIFY_SYSTEM = (
    "You turn a robot's perception facts into events of a declared vocabulary. Use only the "
    "declared event types and, where a type lists contents, only those contents. Report what the "
    "facts show, not what might follow, and only what is new since the recent events (do not "
    "repeat an event that is already there and unchanged). Answer with a JSON array of objects "
    "with keys type, "
    "actor, target, content, conditions (list of strings). An empty array is a valid answer."
)
_CHOOSE_SYSTEM = (
    "You choose the next action of a care robot. You may choose only from the allowed actions. "
    "Obligations in force come first unless an allowed action better protects the person. Without "
    "an obligation in force or a sign in the facts that someone needs something, choose the "
    "default action and do not intrude. Answer "
    "with one JSON object with keys action, args (object; for speak, args.text), reason (one "
    "sentence)."
)


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
    def __init__(self, adapter: ModelAdapter, runtime: SceneRuntime):
        self.adapter, self.rt = adapter, runtime
        declared = dict((runtime.ir.extra or {}).get("event_types", {}))
        self.system = {
            k for k, v in declared.items() if isinstance(v, dict) and v.get("source") == "system"
        }
        self.vocab: dict[str, dict[str, Any]] = {
            k: v for k, v in declared.items() if k not in self.system
        }

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

    def classify(self, facts: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            _extract_first_json(self.adapter.call(_CLASSIFY_SYSTEM, user), expect_array=True) or []
        )
        ok, bad = [], []
        for e in raw:
            why = self._check(e)
            if why:
                bad.append({"event": e, "why": why})
            else:
                ok.append(
                    {
                        k: e[k]
                        for k in ("type", "actor", "target", "content", "conditions")
                        if e.get(k) not in (None, "")
                    }
                )
        return ok, bad


class ActionChooser:
    def __init__(self, adapter: ModelAdapter, runtime: SceneRuntime):
        self.adapter, self.rt = adapter, runtime
        self.caps = {c["action"]: c for c in (runtime.ir.extra or {}).get("capabilities", [])}

    def choose(
        self, snap: Snapshot, facts: Any
    ) -> tuple[str, dict[str, Any], str, dict[str, Any] | None, bool]:
        allowed = snap.allowed
        obliged = [a for a in snap.obliged if a in allowed]
        user = json.dumps(
            {
                "scene": self.rt.ir.document.raw_text,
                "facts": facts,
                "obligations_in_force": obliged,
                "default_action": (self.rt.ir.extra or {}).get("default_action"),
                "allowed_actions": {a: self.caps.get(a, {}) for a in allowed},
                "prohibited_actions": snap.prohibited,
                "moral_state": snap.machines,
            },
            ensure_ascii=False,
        )
        out = _extract_first_json(self.adapter.call(_CHOOSE_SYSTEM, user), expect_array=False) or {}
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
    """One decision cycle: facts in, a checked action out, the moral state stepped in between."""

    def __init__(self, runtime: SceneRuntime, adapter: ModelAdapter):
        self.rt = runtime
        self.classifier = ObservationClassifier(adapter, runtime)
        self.chooser = ActionChooser(adapter, runtime)

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
