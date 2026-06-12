# Deploying the Tier-1 Silicon Target on NRP Coder

This document describes the workflow for taking the
`eris-compile silicon-emit` output and turning it into a synthesised
bitstream running on a Xilinx Alveo U55C FPGA, using the NRP Coder
environment.

## Why NRP Coder

[NRP Coder](https://nrp.ai/documentation/userdocs/coder/coder/) is a
JupyterHub-style development environment on the NRP Nautilus cluster. It
provides:

- **Xilinx Alveo U55C FPGAs** as a requestable resource (datacenter-grade
  PCIe accelerator, HBM2 memory, ~9000 DSP slices).
- **Vivado / Vitis** pre-installed via the "U55C FPGA Vitis Workflow"
  template, including the license server.
- **5 GB persistent home** (more on request) and OIDC authentication via
  institutional credentials.
- 100 Gbps networking and P4 SmartNIC tooling (for designs that want to
  receive IR events directly from the network).

This is the reference deployment target for the ErisML compiler's Tier-1
spine. The FSM + EM-DAG cores were designed in Phase 1 to be silicon-
castable; the Phase-3 `silicon-emit` command produces the Vitis HLS C++
that targets this exact environment.

## End-to-end workflow

### 0. Prereqs

- An NRP account ([https://nrp.ai](https://nrp.ai)).
- Permission to request U55C resources (request quota via the Coder portal
  if your namespace lacks it).
- A local copy of `erisml-compiler` (this repo) on your laptop or in a
  Coder workspace.

### 1. Emit the Vitis HLS sources

On your laptop or on a CPU-only Coder workspace:

```bash
pip install -e .
eris-compile silicon-emit --out-dir build/silicon
```

This writes:

- `build/silicon/erisml_fsm.cpp`     — Commitment / Legitimacy / Consent FSMs
- `build/silicon/erisml_em_dag.cpp`  — EM-DAG pipeline skeleton
- `build/silicon/erisml_top.cpp`     — top-level kernel
- `build/silicon/Makefile`           — `v++` build recipe for the U55C

### 2. Bring up an NRP Coder workspace with the U55C template

1. Visit `https://coder.nrp-nautilus.io`.
2. Create a new workspace from the **"U55C FPGA Vitis Workflow"**
   template.
3. Request: 1 FPGA, ≥8 CPU cores, ≥32 GB RAM, ≥5 GB additional storage
   for build artefacts.
4. Wait for the workspace to start (the FPGA-equipped pods take a few
   minutes to allocate).
5. SSH or open the JupyterLab interface and open a terminal.

### 3. Copy the emitted sources into the workspace

```bash
# From your laptop:
scp build/silicon/* coder.nrp-nautilus.io:~/erisml_silicon/

# Or use git from inside the workspace:
git clone https://github.com/<your-fork>/erisml-compiler ~/erisml_silicon
cd ~/erisml_silicon
eris-compile silicon-emit --out-dir build/silicon
cd build/silicon
```

### 4. Build the bitstream

Inside the workspace, with the Vitis environment sourced:

```bash
# Source the Xilinx tools (typically auto-loaded by the template):
source /opt/Xilinx/Vitis/2024.1/settings64.sh    # version may vary

# Hardware emulation (fast, for debug):
make hw_emu

# Real hardware (takes 1-3 hours):
make hw
```

The `make hw` step invokes `v++ -c` (compile) and `v++ -l` (link) to
produce `build_hw/erisml.xclbin`, the synthesised bitstream.

### 5. Run the bitstream

A host program loads the bitstream and feeds IR feature vectors. A
minimal host example (Phase 5+ deliverable; not yet written):

```cpp
// host.cpp -- skeleton
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
// ... open device, load erisml.xclbin, get the erisml_evaluate kernel ...
// ... feed IRFeatures struct, read back EMOutput[10] + packed_verdict ...
```

Compile with `g++ host.cpp $(pkg-config --libs xrt)` and run.

### 6. Iterate

When the EM-DAG profile changes (e.g., you add a module, change a
dependency edge), re-run `eris-compile silicon-emit --out-dir build/silicon`
on the host, rsync the new sources into the Coder workspace, and rebuild.

## Resource budget (estimated)

For the default 10-module EM-DAG + 3 FSMs at `ap_fixed<16,4>`:

| Resource     | Estimated usage on U55C | U55C capacity |
| ------------ | ----------------------- | ------------- |
| LUTs         | < 5K                    | 1,304K        |
| FFs          | < 8K                    | 2,608K        |
| DSP slices   | < 50                    | 9,024         |
| BRAM         | < 100 KB                | 43 MB         |
| HBM channels | 1 (for IRFeatures in)   | 32            |
| Critical path | < 10 ns               | 100 ns budget at 100 MHz |

These are estimates from the architecture, not measurements. Phase 5+ will
report actual numbers after synthesis.

## What this gives you

- **Hard real-time agent gates.** An autonomous agent's ethical decision
  evaluator runs in **microseconds** per IR, guaranteed by hardware
  timing rather than best-effort software latency.
- **Tamper resistance.** A read-only probe on a kernel that the model
  cannot influence — the kernel sees only the IR features and emits only
  EMOutputs + a verdict.
- **A reproducible audit story.** The host program logs every
  (input, output) pair; the bitstream itself has a hash; the Python
  reference has a hash; equivalence between them is verifiable.

## What's still TODO at silicon level

(Phase 5+ deliverables — outside the Phase 3 scope.)

1. **Per-EM fixed-point arithmetic** — the emitted EM stubs return
   placeholder constants. Phase 5 replaces them with the actual
   Mahalanobis + threshold logic from the Python reference, ported to
   `ap_fixed`.
2. **Host program** for streaming IRs into the kernel and reading
   verdicts back.
3. **Formal equivalence** between the Python reference and the RTL —
   simulation-based (Verilator co-simulation) or symbolic.
4. **Throughput tuning** — pipeline depth, dataflow partitioning, HBM
   bank assignment.
5. **Multi-IR batching** — the U55C has plenty of room to batch many IRs
   per FPGA call.
