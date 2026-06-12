"""Authority legitimacy FSM.

States (per spec 8.5):

    fully_legitimate ──(procedural_violation)──> defeasible
    fully_legitimate ──(coercion_detected)──> coercive
    defeasible       ──(restored)──> fully_legitimate
    defeasible       ──(coercion_detected)──> coercive
    coercive         ──(escalates)──> tyrannical
    tyrannical       ──(catastrophic_intent)──> void
    fraudulent  ──(evidence_revealed)──> void

`void` is terminal/absorbing.
"""
from __future__ import annotations

from typing import Literal

LegitimacyState = Literal[
    "fully_legitimate", "defeasible", "coercive", "tyrannical", "fraudulent", "void"
]

LegitimacyEvent = Literal[
    "procedural_violation",
    "coercion_detected",
    "restored",
    "escalates",
    "catastrophic_intent",
    "evidence_revealed",
]

_TRANSITIONS: dict[LegitimacyState, dict[LegitimacyEvent, LegitimacyState]] = {
    "fully_legitimate": {
        "procedural_violation": "defeasible",
        "coercion_detected": "coercive",
        "evidence_revealed": "fraudulent",
    },
    "defeasible": {
        "restored": "fully_legitimate",
        "coercion_detected": "coercive",
        "evidence_revealed": "fraudulent",
    },
    "coercive": {
        "escalates": "tyrannical",
        "catastrophic_intent": "void",
    },
    "tyrannical": {
        "catastrophic_intent": "void",
    },
    "fraudulent": {
        "evidence_revealed": "void",
    },
}


class LegitimacyFSM:
    """One FSM instance per authority. State 'void' is absorbing."""

    def __init__(self, authority_id: str, initial: LegitimacyState = "fully_legitimate"):
        self.authority_id = authority_id
        self.state: LegitimacyState = initial
        self.history: list[tuple[int, LegitimacyState, LegitimacyEvent | None]] = [
            (0, self.state, None)
        ]

    def step(self, event_tag: LegitimacyEvent, time_index: int) -> LegitimacyState:
        if self.state == "void":
            self.history.append((time_index, self.state, event_tag))
            return self.state
        transitions = _TRANSITIONS.get(self.state, {})
        next_state = transitions.get(event_tag, self.state)
        self.state = next_state
        self.history.append((time_index, next_state, event_tag))
        return next_state
