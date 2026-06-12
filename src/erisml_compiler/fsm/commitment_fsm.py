"""Commitment lifecycle FSM.

States (per spec 13.3 and 8.6):

    active
        |
        | --(defeasibility_condition_triggered)--> active_but_defeasible
        | --(fulfilling_event)--> fulfilled
        | --(violating_event)--> violated
        | --(legitimacy_collapses)--> void
        | --(expiration_event)--> expired
    active_but_defeasible
        | --(defeasibility_resolved_in_favor)--> active
        | --(defeasibility_overrides)--> defeated
        | --(violating_event)--> violated
    defeated, fulfilled, violated, void, expired  --> terminal

Defeasibility conditions (spec 13.3) include:
    coerced_vow, vow_to_commit_wrong, catastrophic_nonconsensual_externality,
    higher_duty_conflict.

All transitions are deterministic given the event tag. This makes the FSM
amenable to silicon casting: state is a 3-bit register, transition function
is a small lookup table.
"""
from __future__ import annotations

from typing import Literal

from erisml_compiler.ir.schemas import Commitment

CommitmentState = Literal[
    "active", "active_but_defeasible", "defeated", "fulfilled", "violated", "void", "expired"
]

TERMINAL_STATES: frozenset[CommitmentState] = frozenset(
    {"defeated", "fulfilled", "violated", "void", "expired"}
)


# Event tags drive transitions. Tier 1 input supplies them directly; Tier 2/3
# extractors classify events into these tags.
EventTag = Literal[
    "fulfilling_event",
    "violating_event",
    "expiration_event",
    "defeasibility_condition_triggered",
    "defeasibility_resolved_in_favor",
    "defeasibility_overrides",
    "legitimacy_collapses",
]


_TRANSITIONS: dict[CommitmentState, dict[EventTag, CommitmentState]] = {
    "active": {
        "defeasibility_condition_triggered": "active_but_defeasible",
        "fulfilling_event": "fulfilled",
        "violating_event": "violated",
        "legitimacy_collapses": "void",
        "expiration_event": "expired",
    },
    "active_but_defeasible": {
        "defeasibility_resolved_in_favor": "active",
        "defeasibility_overrides": "defeated",
        "violating_event": "violated",
        "fulfilling_event": "fulfilled",
        "legitimacy_collapses": "void",
        "expiration_event": "expired",
    },
}


class CommitmentFSM:
    """One FSM instance per commitment. Deterministic transitions."""

    def __init__(self, commitment: Commitment):
        self.commitment_id = commitment.id
        self.state: CommitmentState = commitment.status
        self.history: list[tuple[int, CommitmentState, EventTag | None]] = [
            (0, self.state, None)
        ]

    def step(self, event_tag: EventTag, time_index: int) -> CommitmentState:
        if self.state in TERMINAL_STATES:
            # Terminal states are absorbing.
            self.history.append((time_index, self.state, event_tag))
            return self.state
        transitions = _TRANSITIONS.get(self.state, {})
        next_state = transitions.get(event_tag, self.state)
        self.state = next_state
        self.history.append((time_index, next_state, event_tag))
        return next_state

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
