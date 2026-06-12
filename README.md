# ErisML Compiler

[![CI](https://github.com/ahb-sjsu/erisml-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/ahb-sjsu/erisml-compiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![Schema](https://img.shields.io/badge/IR%20schema-erisml__compiler__ir__v0.1-orange.svg)](SCOPE.md)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-red.svg)](SCOPE.md)

A structure-preserving compiler from natural-language moral material into a
canonical **ErisML Intermediate Representation** (IR) that can be evaluated by
DEME, exported for RLEF training, and audited as a structured trace.

The compiler operationalises the thesis that **moral reasoning requires
structure-preserving representation before decision contraction**. A scalar
"good / bad / safe / unsafe" label discards the dimensions that justify or
defeat a candidate action: who the stakeholders are, what commitments bind
them, which authorities are legitimate, who bears imposed risk. The compiler
preserves this tensorial structure as a first-class object.

See `ErisML-Compiler.md` for the full design spec (31 sections). See
`SCOPE.md` for what each phase actually delivers versus what is deferred.
This release (v0.3.0) bundles Phases 1–3 (IR + DEME + calibration + silicon
emitters); Phase 4 (I-EIP Monitor: Internal / Activation / Delta lenses)
is in-flight on the `main` branch.

## Quick start

```bash
# Install (editable; choose extras as needed)
pip install -e ".[test,notebook]"

# Compile one of the bundled examples
eris-compile compile examples/nazi_attic.txt --out out/nazi_attic.json

# Validate an IR file
eris-compile validate out/nazi_attic.json

# Export as an RLEF training record
eris-compile rlef out/nazi_attic.json --out out/nazi_attic.rlef.json

# Run the test suite
pytest

# Open the quickstart notebook
jupyter notebook notebooks/quickstart.ipynb
```

## Architecture

The compiler implements the 12-pass pipeline from spec §12 with a tiered
extractor stack and silicon-castable evaluation kernel:

```
text  ──► ingest ──► segment ──► extract ──► canonicalize ──► tensorize
                          │            │
                          │            └── Mock | Rule | LLM (NRP / local vLLM)
                          │                + Critic + ProbeExtractor
                          │
                          └──► EM-DAG (10 modules) ──► FSMs ──► DEME ──► audit
                                                    │
                                                    └──► silicon emit (Vitis HLS)
```

Three extractor tiers cover the latency / faithfulness frontier:

- **Mock / Rule** — deterministic, real-time, silicon-castable.
- **LLM** — NRP OpenAI-compatible (`gpt-oss`, `qwen3`, etc.) or local vLLM,
  with a critic pass that flags off-canon outputs for `requires_human_review`.
- **Probe** — calibrated LaBSE-backed classifier head (Phase 3) using
  sqnd-probe v10.16.9 methods: spectral decoupling, VIB, multi-head GRL
  adversarial, confusion loss.

### Layered architecture

| Layer | Purpose |
|---|---|
| `ingestion/` | Load text from files or strings, attach metadata |
| `segmentation/` | Split text into morally-coherent segments |
| `annotation/` | Mock / Rule / LLM / Probe extractors + critic |
| `canonicalizer/` | Registry (Jaccard) + LaBSE cosine canonical-form snap |
| `ontology/` | YAML registries: dimensions, roles, commitments, canonical forms |
| `ir/` | Pydantic v2 IR schemas and validators |
| `em_dag/` | 10 ethical modules + topological DAG evaluator |
| `fsm/` | Commitment / Legitimacy / Consent finite-state machines |
| `evaluation/` | MoralVector / MoralTensor construction; conflict detection |
| `calibration/` | Probe training: losses, adversarial heads, VIB, bond index |
| `correction/` | IR diff + apply-corrections (RLEF feedback loop) |
| `erisml_backend/` | ErisML codegen and DEME bridge |
| `silicon/` | Fixed-point conversion + Vitis HLS C++ emitters (FSM + DAG) |
| `audit/` | SHA-256 hash chain and per-pass provenance |
| `export/` | JSON, ErisML source, RLEF training records |
| `viz/` | HTML report + timeline plot |
| `streaming/` | Real-time captioner of pipeline events |
| `monitor/` | I-EIP Monitor activation lens: ActivationSource + ActivationProbe + IEIPMonitor |
| `delta/` | Delta lens: compare_morals, BIP equivariance check, 5-mode failure detector |
| `cli.py` | `eris-compile {bundle,calibrate,compile,correct,delta,diff,monitor,report,rlef,silicon-emit,validate,version}` |

### What is NOT yet in v0.3.0

See `SCOPE.md` for the full list. Headline in-flight items: the production
web app, NRP runtime deployment, and silicon hardware verification on the
U55C target. The I-EIP Monitor (Internal / Activation / Delta lenses) is
implemented as of Phase 4 — see `docs/i_eip_monitor.md`.

## Project layout

```
erisml-compiler/
  ErisML-Compiler.docx        # Original design spec
  ErisML-Compiler.md          # Same, converted to Markdown for reading
  SCOPE.md                    # What is built vs stubbed vs deferred
  README.md                   # This file
  LICENSE                     # MIT
  pyproject.toml
  src/erisml_compiler/
    cli.py
    config.py
    ingestion/
    segmentation/
    annotation/
    ontology/
    ir/
    evaluation/
    erisml_backend/
    audit/
    export/
  examples/
    nazi_attic.txt
    medical_confidentiality.txt
    whistleblower.txt
  tests/
  notebooks/
    quickstart.ipynb
  docs/
    architecture.md
```

## Status

v0.3.0 — alpha. 82 tests passing across IR, EM-DAG, FSMs, canonicalizer,
critic, correction, calibration, export, and silicon emit. CLI exposes 10
subcommands. NRP LLM integration verified end-to-end on the bundled
`nazi_attic` example. Silicon emitters produce Vitis HLS C++ for FSMs and the
EM-DAG; the web app, storage layer, and I-EIP Monitor (Phase 4) are in
flight. See `SCOPE.md`.

## License

MIT. See `LICENSE`.
