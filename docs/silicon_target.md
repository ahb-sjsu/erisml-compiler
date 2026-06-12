# Silicon target

The Tier-1 evaluator core is designed to be cast into silicon (FPGA / ASIC).
This document describes the reference deployment path; see
`docs/nrp_coder_deployment.md` for the step-by-step build workflow.

## Reference target

**Hardware:** Xilinx Alveo U55C PCIe accelerator (datacenter-grade, HBM2,
~9000 DSP slices), available via the
[NRP Coder](https://nrp.ai/documentation/userdocs/coder/coder/) environment.

**Toolchain:** Vitis HLS / `v++` (Vivado 2024.x), pre-installed in the
"U55C FPGA Vitis Workflow" template on NRP Coder.

**Code path:** `eris-compile silicon-emit --out-dir build/silicon` produces
the Vitis HLS C++ sources + Makefile from the Python FSM and EM-DAG
definitions. See `src/erisml_compiler/silicon/examples/` for example
emitted output.

## What is silicon-castable today

- **FSM layer** (`fsm/`). Each FSM has a finite state space (≤ 8 states),
  deterministic transitions encoded in a Python dict, no floating-point
  arithmetic, no dynamic memory allocation. Direct mapping to a flat
  hardware state register + combinational transition logic.
- **EM-DAG evaluation** (`em_dag/`). Topology is fixed at compile time;
  modules are pure functions of the IR and upstream module outputs. Each
  module is a small block; the DAG topology is a fixed pipeline of
  combinational/registered stages.
- **DEME bridge** (`erisml_backend/deme_bridge.py`). Decision logic is
  a deterministic cascade of fixed-threshold comparisons over the
  MoralVector. Direct mapping to comparators + multiplexers.
- **Mahalanobis evaluator**. Currently performed in `numpy` floats; for
  silicon, switch to fixed-point arithmetic and pre-compute the inverse
  covariance matrix at compile time.

## What is NOT silicon-castable

- The natural-language extraction frontend (Tiers 2 and 3): regex engines,
  LLM inference. These stay in software and feed structured input to the
  silicon core.
- The Pydantic schema layer: this is a software construct for IR
  validation and serialisation. The silicon core sees only the structured
  values, not the JSON.
- The audit hashing: SHA-256 is silicon-implementable but is not the
  cycle-critical part of the evaluation. Move it off-chip if convenient.

## What would need to happen

The path I'd take:

1. **Define a strict subset language** for FSM and EM-DAG implementations.
   The subset disallows floating-point, dynamic dispatch, string
   manipulation, and unbounded loops. Existing FSMs already fit. Existing
   EMs need a fixed-point port of their score arithmetic.

2. **Choose an HLS toolchain**. Three plausible paths:
   - **PyMTL3** (Cornell): Python-based, designed for hardware DSLs.
   - **MyHDL**: Python → Verilog with a more imperative feel.
   - **Vitis HLS** (AMD/Xilinx) or **Catapult HLS** (Siemens): C++ HLS;
     port the silicon-target subset to C++ first.

3. **Port the evaluator core** to the chosen toolchain. Each FSM becomes
   a small `case` statement; each EM becomes a combinational block
   computing a fixed-point score from input registers; the EM-DAG becomes
   a pipeline of registered stages whose order matches the topological
   sort.

4. **Verify equivalence**. Run the same calibration test vectors
   (`tests/test_pipeline.py`) through the Python implementation and the
   RTL implementation; assert bit-exact match on the MoralVector and
   verdict. This is the silicon-vs-reference acceptance test.

5. **Synthesise** to FPGA (target: Xilinx Zynq Ultrascale+, AMD/Xilinx
   Alveo, or Intel Stratix) or to a small ASIC corner. Expected resource
   usage for the default 10-module DAG on a fixed-point port: < 5K LUTs,
   < 10 KB BRAM, ≤ 10 ns critical path. Well within commodity FPGA
   capacity.

## What the silicon target unlocks

- **Hard-real-time agent safety interlocks.** An autonomous vehicle's
  ethical-decision gate evaluates each candidate action in microseconds,
  guaranteed by hardware not by best-effort software latency.
- **Hardware kill-switches.** A model-level kill switch driven by a
  geometric ethical evaluator that the model cannot influence (read-only
  probes from activations to fixed-point evaluator to override gate).
- **Embedded agent ethics**. Drones, surgical robots, industrial
  manipulators, military targeting systems — any agent in safety-critical
  loops where software-level safety is not acceptable.
- **Tamper-resistant audit**. Hardware-rooted audit trails for legal-AI
  systems where verdict provenance must be unforgeable.

## What the silicon target is NOT

- It is not a replacement for the natural-language extraction frontend.
  Real moral content arrives in natural language; the LLM/Rule tier
  remains in software.
- It is not a complete ethical agent. The silicon evaluator gates
  proposed actions but does not generate them. Action generation stays
  in software.
- It is not a substitute for legal and constitutional review. The
  evaluator produces a single quantitative verdict with a confidence
  interval; whether deployment is lawful is a separate question.

## Phase 5+ deliverables

A Phase-5+ silicon-target work package would deliver:

1. A fixed-point port of the FSM + EM-DAG + DEME spine, verified
   bit-exact against the Python reference on the calibration test vectors.
2. An HLS-toolchain implementation producing synthesisable RTL.
3. FPGA validation on a commodity board (e.g., Xilinx Zynq Ultrascale+
   MPSoC ZCU104), with end-to-end timing and resource numbers.
4. A formal-equivalence proof between the Python reference and the RTL
   (via simulation-based equivalence or symbolic verification, depending
   on toolchain).
5. A reference design for integrating the silicon evaluator into an
   agent stack (gate on output of policy network, refuse-or-allow signal,
   audit trail to off-chip memory).
