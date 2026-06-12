# ErisML Compiler

[![CI](https://github.com/ahb-sjsu/erisml-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/ahb-sjsu/erisml-compiler/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![Schema](https://img.shields.io/badge/IR%20schema-erisml__compiler__ir__v0.1-orange.svg)](SCOPE.md)
[![Tests](https://img.shields.io/badge/tests-194%20passing-brightgreen.svg)](#status)
[![Ruff](https://img.shields.io/badge/lint-ruff-blueviolet)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-red.svg)](SCOPE.md)
[![PyPI](https://img.shields.io/pypi/v/erisml-compiler.svg)](https://pypi.org/project/erisml-compiler/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20659432.svg)](https://doi.org/10.5281/zenodo.20659432)

A structure-preserving compiler from natural-language moral material into a
canonical **ErisML Intermediate Representation** (IR) that can be evaluated by
DEME, exported for RLEF training, audited as a structured trace, and
introspected by the I-EIP Monitor's three lenses.

The compiler operationalises the thesis that **moral reasoning requires
structure-preserving representation before decision contraction**. A scalar
"good / bad / safe / unsafe" label discards the dimensions that justify or
defeat a candidate action: who the stakeholders are, what commitments bind
them, which authorities are legitimate, who bears imposed risk. The compiler
preserves this tensorial structure as a first-class object, then closes the
loop by inspecting whether the text output and the model's internal state
actually agree about it.

See `ErisML-Compiler.md` for the full design spec (31 sections) and `SCOPE.md`
for what each phase actually delivers versus what is deferred. Current `main`
covers the original Phases 1–4 (IR + DEME + calibration + silicon emitters +
I-EIP Monitor) plus the **DEME V3 alignment** rolling in over an additional
six-phase migration (`docs/migration/deme_v3_alignment.md`). Phases 1–4 of
that alignment have landed: 9-dimension moral state, rank-1 through rank-6
tensors with `(k, n, τ, a, c, s)` axes, per-party verdicts, fairness
metrics (Gini + worst-off), and a bridge invoking DEME V3 modules
(`GenevaEMV3`, `TriageEMV3`) directly. The production web app and silicon
hardware verification remain deferred.

## Quick start

```bash
# Install from PyPI
pip install erisml-compiler                        # core
pip install 'erisml-compiler[llm,calibration,monitor]'  # full stack

# Or, install from source (editable; choose extras as needed)
pip install -e ".[test,calibration,monitor,notebook]"

# Compile one of the bundled examples — emits both V2 moral_vectors and a
# DEME V3 MoralTensorV3 with the requested rank (default 2 = per-stakeholder).
eris-compile compile examples/nazi_attic.txt --rank 2 --out out/nazi_attic.ir.json

# Validate the IR
eris-compile validate out/nazi_attic.ir.json

# Export as an RLEF training record
eris-compile rlef out/nazi_attic.ir.json --out out/nazi_attic.rlef.json

# Run the activation lens (mock source for offline use)
eris-compile monitor "Soldiers at the door asking about hidden refugees." \
    --source mock --hidden-dim 64 --n-layers 8 \
    --out out/nazi_attic.trace.json

# Compare the two lenses — fires requires_human_review when they disagree
eris-compile delta out/nazi_attic.ir.json out/nazi_attic.trace.json \
    --out out/nazi_attic.delta.json

# Emit synthesizable Vitis HLS C++ for the silicon target
eris-compile silicon-emit --out-dir out/silicon

# Run the full test suite (194 tests including V3 alignment;
# ~30s for the V2 core + extras when LaBSE is cached)
pytest

# Run the linters / formatters that CI uses
ruff check src tests
black --check src tests

# Quickstart notebook
jupyter notebook notebooks/quickstart.ipynb
```

## Architecture

The compiler implements the 12-pass pipeline from spec §12 with a tiered
extractor stack, a silicon-castable evaluation kernel, and the I-EIP Monitor
on top.

```
text  ──► ingest ──► segment ──► extract ──► canonicalize ──► tensorize
                          │            │
                          │            └── Mock | Rule | LLM (NRP / local vLLM)
                          │                + Critic + ProbeExtractor
                          │
                          └──► EM-DAG (10 modules) ──► FSMs ──► DEME ──► audit
                                                    │
                                                    └──► silicon emit (Vitis HLS)

                          (out-of-band, sampled audit)
                          model ──► hooks ──► IEIPMonitor ──► Delta lens
                                                                │
                                                                └─► requires_human_review
                                                                    + failure-mode report
```

Three extractor tiers cover the latency / faithfulness frontier:

- **Mock / Rule** — deterministic, real-time, silicon-castable.
- **LLM** — NRP OpenAI-compatible (`gpt-oss`, `qwen3`, etc.) or local vLLM,
  with a critic pass that flags off-canon outputs for `requires_human_review`.
- **Probe** — calibrated LaBSE-backed classifier head using
  sqnd-probe v10.16.9 methods: spectral decoupling, VIB, multi-head GRL
  adversarial, confusion loss.

Three lenses cover the alignment frontier:

- **Text lens** (Phases 1–3) — what the model *says*.
- **Activation lens** (Phase 4) — what the model *internally exhibits*
  at chosen transformer layers (forward hooks on Qwen2.5-7B-Instruct,
  LLaMA, Mistral, GPT-2, or BERT-family models).
- **Delta lens** (Phase 4) — where they disagree, structured by moral
  dimension, with five named failure modes
  (`text_internal_mismatch`, `layerwise_drift`, `group_symmetry_break`,
  `probe_uncertainty_spike`, `audit_chain_break`). Any firing sets
  `requires_human_review`; the Monitor never overrules DEME.

See `docs/i_eip_monitor.md` for the threat model, trust-boundary
diagram, and the precise semantics of each failure mode.

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
| `cli.py` | 12 subcommands: `bundle calibrate compile correct delta diff monitor report rlef silicon-emit validate version` |

### What is NOT yet in `main`

See `SCOPE.md` for the full list. Headline in-flight items:

- **Production web app** (deferred from the Phase 4 redirect to the I-EIP Monitor)
- **NRP runtime deployment** (orchestrator + pod templates)
- **Silicon hardware verification** on the Xilinx U55C target — Vitis HLS C++
  is emitted and builds; on-FPGA bring-up is gated by the NRP Coder bitstream
  pipeline (see `project_epu_phase3_hw_blocked` in the user's notes).

## Project layout

```
erisml-compiler/
  ErisML-Compiler.docx        # Original design spec (31 sections)
  ErisML-Compiler.md          # Same, converted to Markdown
  SCOPE.md                    # What is built / stubbed / deferred
  README.md                   # This file
  LICENSE                     # MIT
  pyproject.toml              # Extras: [llm] [calibration] [monitor] [test] [dev] [notebook]
  src/erisml_compiler/
    cli.py
    ingestion/  segmentation/  annotation/  ontology/  ir/  evaluation/
    em_dag/     fsm/           canonicalizer/          correction/
    calibration/  monitor/  delta/  silicon/  erisml_backend/
    audit/        export/   viz/    streaming/
  examples/
    nazi_attic.txt
    medical_confidentiality.txt
    whistleblower.txt
  tests/                      # 142 tests
  notebooks/quickstart.ipynb
  docs/
    architecture.md
    silicon_target.md
    nrp_coder_deployment.md
    i_eip_monitor.md          # I-EIP Monitor threat model & trust boundaries
  scripts/atlas/
    probe_models.py           # Recon: enumerate HF + GGUF models on Atlas
```

## DEME V3 alignment

The original V2 IR carries 10 moral dimensions and a rank-2 per-stakeholder
`MoralTensor`. **DEME V3** (`erisml-lib`) speaks a different shape:
9 dimensions derived from the *Nine Dimensions of Ethical Assessment* 3×3
matrix, tensors at ranks 1–6 over axes `(k, n, τ, a, c, s)` (dimension /
stakeholder / time / action / coalition / uncertainty sample), per-party
verdicts, distributional veto locations, Gini + worst-off fairness
metrics, and a sprint-tiered module hierarchy (Constitutional,
Core Safety, Rights/Fairness, Soft Values, Meta-Governance).

The compiler is rolling onto V3 over a documented six-phase migration
(`docs/migration/deme_v3_alignment.md`). What's landed today:

| Phase | Deliverable |
|---|---|
| 1 | `MoralTensorV3` Pydantic schema with rank/shape/axes/values + V2→V3 migration helpers |
| 2 | Orchestrator produces `ir.moral_tensor_v3` at the requested rank; `--rank N` CLI flag |
| 3 | Bridge wires the IR through `EthicalFactsV3` and invokes registered V3 modules (Geneva, Triage) |
| 4 | Per-party facts built directly from `EthicalFact.subjects`; per-party verdicts and Gini surfaced on the IR; `requires_human_review` is now per-stakeholder |
| 5 (planned) | Coalition + temporal + uncertainty axes (rank 3–6) |
| 6 (planned) | Strategic layer (Shapley + Nash) + `DecisionProof` audit chain |

The V2 surface remains alive — `moral_vectors`, `moral_tensors`, the V2
EM-DAG — so existing IRs still parse and the legacy `MoralVector` API still
works. Phase 5+ may deprecate the V2 fields after the silicon and Monitor
paths migrate.

## Status

**Phases 1–4 on `main`, plus DEME V3 alignment Phases 1–4** — alpha.
**194 tests passing** across IR (V2 + V3), EM-DAG, FSMs, canonicalizer,
critic, correction, calibration, export, silicon emit, activation lens,
delta lens, equivariance, failure-mode detectors, V3 schema, V3 pipeline,
V3 bridge, V3 direct-facts builder. CI green on Ubuntu × Python
3.10/3.11/3.12, ruff lint + black format checks both clean.

End-to-end verified:

- NRP LLM integration on the bundled `nazi_attic` example (the LLM picks the
  wrong canonical form, the canonicalizer corrects it, the critic pass
  triggers `requires_human_review`).
- I-EIP Monitor on the same example: divergence 0.70, 6 direction breaks,
  two failure modes fire, `requires_human_review=True`.
- **DEME V3 bridge** on the same example: rank-2 `(9, 4)` tensor over
  speaker / hidden_refugees / nazis / village. Per-party harm splits cleanly:
  speaker 0.76 (forbid), village 0.83 (forbid), nazis 0.18 (neutral),
  refugees 0.0 (prefer). Gini over harm = 0.43.
- Vitis HLS C++ emit for FSMs + EM-DAG (NRP Coder bitstream blocked
  separately — see SCOPE.md).

## Citing

If you use this work academically, please cite via the Zenodo DOI. The
**concept DOI** always resolves to the latest release; the version DOI
pins a specific release.

```bibtex
@software{bond2026erisml,
  author    = {Bond, Andrew H.},
  title     = {ErisML Compiler: A Structure-Preserving Compiler from
               Natural Language to a Moral Intermediate Representation},
  year      = {2026},
  version   = {0.4.0},
  doi       = {10.5281/zenodo.20659432},
  url       = {https://github.com/ahb-sjsu/erisml-compiler}
}
```

- **Concept DOI** (latest): https://doi.org/10.5281/zenodo.20659432
- **v0.4.0 DOI**:           https://doi.org/10.5281/zenodo.20659433

## License

MIT. See `LICENSE`.
