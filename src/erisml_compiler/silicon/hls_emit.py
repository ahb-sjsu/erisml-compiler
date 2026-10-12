"""Emit Vitis HLS C++ from the Python FSM definitions.

Until 0.10 this module also emitted the EM-DAG pipeline, a top-level evaluator kernel wrapping
it, and a v++ Makefile for that kernel. The EM-DAG is retired (erisml-lib's DEME is the
evaluator), and with it the evaluator target; retargeting a silicon evaluator to DEME's modules
is future work. The FSM emission below is unchanged.

The FSMs in `erisml_compiler/fsm/` are declarative: each one is a state
enum + a transition table (dict-of-dicts). We translate each FSM to a
synthesizable C++ function whose interface is an `ap_uint<3>` state
register plus an `ap_uint<4>` event-tag enum, and whose body is a
single `switch` over the transition table.

Format note: the generated C++ targets **Vitis HLS** (Xilinx). Pragmas
follow `#pragma HLS` style. The output is a `.cpp` file ready to drop
into a Vitis HLS project on NRP Coder.
"""

from __future__ import annotations

from collections.abc import Mapping
from textwrap import dedent
from typing import TypeVar

from erisml_compiler.fsm.commitment_fsm import _TRANSITIONS as COMMITMENT_TRANSITIONS
from erisml_compiler.fsm.consent_fsm import _TRANSITIONS as CONSENT_TRANSITIONS
from erisml_compiler.fsm.legitimacy_fsm import _TRANSITIONS as LEGITIMACY_TRANSITIONS

# FSM transition tables are keyed by per-FSM Literal state/event types
# (e.g. dict[ConsentState, dict[ConsentEvent, ConsentState]]). These
# TypeVars let the emitters accept any such table without dict's
# invariance rejecting the Literal keys.
_State = TypeVar("_State", bound=str)
_Event = TypeVar("_Event", bound=str)

# ============================================================================
# FSM emission
# ============================================================================


def _state_enum(states: list[str], prefix: str) -> str:
    lines = [f"enum {prefix}_State : ap_uint<3> {{"]
    for i, s in enumerate(states):
        lines.append(f"    {prefix.upper()}_{s.upper()} = {i},")
    lines.append("};")
    return "\n".join(lines)


def _event_enum(events: list[str], prefix: str) -> str:
    lines = [f"enum {prefix}_Event : ap_uint<4> {{"]
    for i, e in enumerate(events):
        lines.append(f"    {prefix.upper()}_EVT_{e.upper()} = {i},")
    lines.append("};")
    return "\n".join(lines)


def _transition_switch(
    transitions: Mapping[_State, Mapping[_Event, _State]],
    states: list[str],
    events: list[str],
    prefix: str,
    terminal_states: list[str] | None = None,
) -> str:
    body = [
        "    // Terminal states are absorbing.",
    ]
    if terminal_states:
        terminal_or = " || ".join(
            f"current == {prefix.upper()}_{ts.upper()}" for ts in terminal_states
        )
        body.append(f"    if ({terminal_or}) return current;")
        body.append("")

    body.append("    switch (current) {")
    for state, trans in transitions.items():
        body.append(f"    case {prefix.upper()}_{state.upper()}:")
        body.append("        switch (event_tag) {")
        for evt, next_state in trans.items():
            body.append(
                f"        case {prefix.upper()}_EVT_{evt.upper()}: return {prefix.upper()}_{next_state.upper()};"
            )
        body.append("        default: return current;")
        body.append("        }")
    body.append("    default: return current;")
    body.append("    }")
    return "\n".join(body)


def _emit_one_fsm(
    name: str,
    transitions: Mapping[_State, Mapping[_Event, _State]],
    all_states: list[str],
    all_events: list[str],
    terminal_states: list[str] | None = None,
) -> str:
    prefix = name
    state_enum = _state_enum(all_states, prefix)
    event_enum = _event_enum(all_events, prefix)
    switch = _transition_switch(transitions, all_states, all_events, prefix, terminal_states)
    return dedent(f"""\
        // ---------------------------------------------------------------
        // {name} FSM
        // Generated from erisml_compiler.fsm.{name.lower()}_fsm by hls_emit.py.
        // Synthesizable for Vitis HLS targeting Xilinx Alveo U55C.
        // ---------------------------------------------------------------

        {state_enum}

        {event_enum}

        {prefix}_State {prefix.lower()}_step({prefix}_State current, {prefix}_Event event_tag) {{
        #pragma HLS INLINE
        #pragma HLS PIPELINE II=1
        {switch}
        }}
    """)


def emit_fsm_cpp() -> str:
    """Emit the three FSMs (Commitment, Legitimacy, Consent) as one .cpp file."""
    # Commitment
    commit_states = [
        "active",
        "active_but_defeasible",
        "defeated",
        "fulfilled",
        "violated",
        "void",
        "expired",
    ]
    commit_events = [
        "fulfilling_event",
        "violating_event",
        "expiration_event",
        "defeasibility_condition_triggered",
        "defeasibility_resolved_in_favor",
        "defeasibility_overrides",
        "legitimacy_collapses",
    ]
    commit_block = _emit_one_fsm(
        "Commitment",
        COMMITMENT_TRANSITIONS,
        commit_states,
        commit_events,
        terminal_states=["defeated", "fulfilled", "violated", "void", "expired"],
    )

    # Legitimacy
    legit_states = [
        "fully_legitimate",
        "defeasible",
        "coercive",
        "tyrannical",
        "fraudulent",
        "void",
    ]
    legit_events = [
        "procedural_violation",
        "coercion_detected",
        "restored",
        "escalates",
        "catastrophic_intent",
        "evidence_revealed",
    ]
    legit_block = _emit_one_fsm(
        "Legitimacy",
        LEGITIMACY_TRANSITIONS,
        legit_states,
        legit_events,
        terminal_states=["void"],
    )

    # Consent
    consent_states = ["not_obtained", "obtained", "coerced", "withdrawn"]
    consent_events = [
        "consent_given",
        "coerced_assent",
        "withdrawn",
        "coerced_revisit",
    ]
    consent_block = _emit_one_fsm(
        "Consent",
        CONSENT_TRANSITIONS,
        consent_states,
        consent_events,
        terminal_states=["coerced", "withdrawn"],
    )

    header = dedent("""\
        // ===============================================================
        // erisml_fsm.cpp
        //
        // Synthesizable C++ for Vitis HLS targeting Xilinx Alveo U55C.
        // Generated by erisml_compiler.silicon.hls_emit (Phase 3 / Track C).
        //
        // Do not edit by hand; regenerate via:
        //     eris-compile silicon-emit --out src/silicon/erisml_fsm.cpp
        // ===============================================================

        #include <ap_int.h>

        extern "C" {
        """)
    footer = '\n}  // extern "C"\n'
    return header + "\n".join([commit_block, legit_block, consent_block]) + footer
