"""Consent FSM. Tracks consent state for an interaction or affected party.

States:
    not_obtained ──(consent_given)──> obtained
    not_obtained ──(coerced_assent)──> coerced
    obtained     ──(withdrawn)──> withdrawn
    obtained     ──(coerced_revisit)──> coerced
    coerced, withdrawn  --> terminal absorbing
"""
from __future__ import annotations

from typing import Literal

ConsentState = Literal["not_obtained", "obtained", "coerced", "withdrawn"]
ConsentEvent = Literal["consent_given", "coerced_assent", "withdrawn", "coerced_revisit"]

_TRANSITIONS: dict[ConsentState, dict[ConsentEvent, ConsentState]] = {
    "not_obtained": {"consent_given": "obtained", "coerced_assent": "coerced"},
    "obtained": {"withdrawn": "withdrawn", "coerced_revisit": "coerced"},
}


class ConsentFSM:
    def __init__(self, subject_id: str, initial: ConsentState = "not_obtained"):
        self.subject_id = subject_id
        self.state: ConsentState = initial
        self.history: list[tuple[int, ConsentState, ConsentEvent | None]] = [
            (0, self.state, None)
        ]

    def step(self, event_tag: ConsentEvent, time_index: int) -> ConsentState:
        if self.state in ("coerced", "withdrawn"):
            self.history.append((time_index, self.state, event_tag))
            return self.state
        transitions = _TRANSITIONS.get(self.state, {})
        next_state = transitions.get(event_tag, self.state)
        self.state = next_state
        self.history.append((time_index, next_state, event_tag))
        return next_state
