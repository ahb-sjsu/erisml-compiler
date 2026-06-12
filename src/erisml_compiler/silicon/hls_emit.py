"""Emit Vitis HLS C++ from the Python FSM and EM-DAG definitions.

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

from textwrap import dedent
from typing import Any

from erisml_compiler.fsm.commitment_fsm import _TRANSITIONS as COMMITMENT_TRANSITIONS
from erisml_compiler.fsm.consent_fsm import _TRANSITIONS as CONSENT_TRANSITIONS
from erisml_compiler.fsm.legitimacy_fsm import _TRANSITIONS as LEGITIMACY_TRANSITIONS


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
    transitions: dict[str, dict[str, str]],
    states: list[str],
    events: list[str],
    prefix: str,
    terminal_states: list[str] | None = None,
) -> str:
    body = [
        f"    // Terminal states are absorbing.",
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
            body.append(f"        case {prefix.upper()}_EVT_{evt.upper()}: return {prefix.upper()}_{next_state.upper()};")
        body.append("        default: return current;")
        body.append("        }")
    body.append("    default: return current;")
    body.append("    }")
    return "\n".join(body)


def _emit_one_fsm(
    name: str,
    transitions: dict[str, dict[str, str]],
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
        "active", "active_but_defeasible", "defeated", "fulfilled", "violated", "void", "expired",
    ]
    commit_events = [
        "fulfilling_event", "violating_event", "expiration_event",
        "defeasibility_condition_triggered", "defeasibility_resolved_in_favor",
        "defeasibility_overrides", "legitimacy_collapses",
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
        "fully_legitimate", "defeasible", "coercive", "tyrannical", "fraudulent", "void",
    ]
    legit_events = [
        "procedural_violation", "coercion_detected", "restored",
        "escalates", "catastrophic_intent", "evidence_revealed",
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
        "consent_given", "coerced_assent", "withdrawn", "coerced_revisit",
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
    footer = "\n}  // extern \"C\"\n"
    return header + "\n".join([commit_block, legit_block, consent_block]) + footer


# ============================================================================
# EM-DAG pipeline emission
# ============================================================================


def emit_em_dag_pipeline(
    profile_path: str | None = None,
    fixed_point_total_bits: int = 16,
    fixed_point_int_bits: int = 4,
) -> str:
    """Emit a Vitis HLS skeleton for the EM-DAG pipeline.

    Each module becomes a small C++ function with a fixed-point input
    (the IR-derived feature vector) and a `dim_score_t` output. The
    pipeline composes them in topological order, with upstream outputs
    fed into downstream inputs.

    For the MVP this is a SKELETON: each module's body is a placeholder
    returning a constant. A Phase-5 deliverable will fill in the actual
    per-module fixed-point arithmetic.
    """
    from erisml_compiler.em_dag import load_profile
    from pathlib import Path
    if profile_path is None:
        profile_path = (
            Path(__file__).resolve().parent.parent
            / "em_dag" / "profiles" / "default.yaml"
        )
    dag = load_profile(profile_path)

    order = dag.topological_order
    modules = dag.modules

    fp_cfg_str = f"ap_fixed<{fixed_point_total_bits}, {fixed_point_int_bits}>"

    header = dedent(f"""\
        // ===============================================================
        // erisml_em_dag.cpp
        //
        // EM-DAG pipeline for Vitis HLS / Alveo U55C.
        // Topological order: {order}
        // Fixed-point type: {fp_cfg_str}
        //
        // SKELETON: each module returns a placeholder constant. The
        // Phase-5 silicon deliverable fills in per-module arithmetic
        // from the Python reference implementation.
        // ===============================================================

        #include <ap_int.h>
        #include <ap_fixed.h>

        typedef {fp_cfg_str} dim_score_t;
        typedef ap_fixed<{fixed_point_total_bits}, {fixed_point_int_bits}> feature_t;

        struct EMOutput {{
            dim_score_t score;
            ap_uint<8>  confidence_q8;  // 0..255
        }};

        struct IRFeatures {{
            // Placeholder. The real version will be a struct generated
            // from the Pydantic IR's hot fields.
            feature_t harm_signal;
            feature_t legitimacy_signal;
            feature_t consent_signal;
            feature_t externality_signal;
            feature_t fairness_signal;
            feature_t epistemic_signal;
            feature_t care_signal;
            feature_t fidelity_signal;
            feature_t rights_signal;
        }};

        extern "C" {{
        """)

    module_stubs = []
    for name in order:
        mod = modules[name]
        deps = mod.dependencies or ()
        dep_args = "".join(f", const EMOutput &{d}" for d in deps)
        module_stubs.append(dedent(f"""\
            // EM[{name}] dimension={mod.dimension}, depends_on={list(deps)}
            EMOutput em_{name}(const IRFeatures &features{dep_args}) {{
            #pragma HLS PIPELINE II=1
                EMOutput o;
                o.score = 0;          // Phase-5: replace with real arithmetic.
                o.confidence_q8 = 200;
                return o;
            }}
            """))

    top_calls = []
    cached_ref = {}
    for name in order:
        mod = modules[name]
        deps = mod.dependencies or ()
        args = "features"
        for d in deps:
            args += f", em_out_{d}"
        top_calls.append(f"    EMOutput em_out_{name} = em_{name}({args});")
        cached_ref[name] = f"em_out_{name}"

    aggregate = dedent("""\

        // Top-level pipeline: walk the DAG in topological order.
        void em_dag_pipeline(const IRFeatures &features, EMOutput out[10]) {
        #pragma HLS DATAFLOW
        """)
    aggregate += "\n".join(top_calls)
    out_assignments = []
    for i, name in enumerate(order):
        out_assignments.append(f"    out[{i}] = em_out_{name};")
    aggregate += "\n" + "\n".join(out_assignments)
    aggregate += "\n}\n"

    footer = "\n}  // extern \"C\"\n"
    return header + "\n".join(module_stubs) + aggregate + footer


# ============================================================================
# Top module + Makefile
# ============================================================================


def emit_top_module() -> str:
    """Emit a Vitis HLS top-level wrapper combining FSMs + EM-DAG."""
    return dedent("""\
        // ===============================================================
        // erisml_top.cpp
        //
        // Top-level Vitis HLS entry point for the Tier-1 evaluator
        // targeting Xilinx Alveo U55C via NRP Coder.
        //
        // The host program (NRP Coder workspace) feeds a stream of
        // (IR feature vector, FSM event tags) tuples; the FPGA returns
        // the DEME verdict + 10-dim EMOutput array.
        // ===============================================================

        #include <ap_int.h>
        #include <ap_fixed.h>
        #include <hls_stream.h>

        // Forward declarations from erisml_fsm.cpp and erisml_em_dag.cpp.
        struct IRFeatures;
        struct EMOutput;
        void em_dag_pipeline(const IRFeatures &features, EMOutput out[10]);

        extern "C" {

        // Top-level kernel: one IRFeatures in, 10 EMOutput out, plus a
        // packed verdict word.
        void erisml_evaluate(
            const IRFeatures &features,
            EMOutput em_outputs[10],
            ap_uint<32> &packed_verdict
        ) {
        #pragma HLS INTERFACE m_axi port=em_outputs offset=slave bundle=gmem
        #pragma HLS INTERFACE s_axilite port=features
        #pragma HLS INTERFACE s_axilite port=packed_verdict
        #pragma HLS INTERFACE s_axilite port=return

            em_dag_pipeline(features, em_outputs);

            // Phase-5: pack DEME verdict bits.
            // bit  0-2 : verdict_kind (8 verdict types)
            // bit  3-10: confidence_q8 (0..255)
            // bit 11    : escalation_required
            // bit 12-31: reserved
            packed_verdict = 0;
        }

        }  // extern "C"
    """)


def emit_makefile(top_cpp: str = "erisml_top.cpp") -> str:
    """Emit a Makefile that calls v++ (Vitis compiler) to build for U55C."""
    return dedent(f"""\
        # ===============================================================
        # Makefile for erisml-compiler silicon target
        # Target: Xilinx Alveo U55C
        # Toolchain: Vitis HLS / v++ (Vivado 2024.x recommended)
        # NRP Coder template: "U55C FPGA Vitis Workflow"
        # ===============================================================

        VPP        := v++
        PLATFORM   := xilinx_u55c_gen3x16_xdma_3_202210_1
        TARGET     := hw
        KERNEL     := erisml_evaluate
        SOURCES    := erisml_fsm.cpp erisml_em_dag.cpp {top_cpp}
        OBJ_DIR    := build_$(TARGET)

        XO         := $(OBJ_DIR)/erisml.xo
        XCLBIN     := $(OBJ_DIR)/erisml.xclbin

        .PHONY: all hls hw_emu hw clean

        all: $(XCLBIN)

        $(OBJ_DIR):
        \tmkdir -p $(OBJ_DIR)

        $(XO): $(SOURCES) | $(OBJ_DIR)
        \t$(VPP) -c -t $(TARGET) --platform $(PLATFORM) --kernel $(KERNEL) \\
        \t       -o $@ $(SOURCES)

        $(XCLBIN): $(XO)
        \t$(VPP) -l -t $(TARGET) --platform $(PLATFORM) \\
        \t       -o $@ $<

        hw_emu:
        \t$(MAKE) all TARGET=hw_emu

        clean:
        \trm -rf build_hw build_hw_emu build_sw_emu *.log *.jou
    """)
