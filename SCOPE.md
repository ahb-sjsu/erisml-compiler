# ErisML Compiler — Scope

This file states **what is built**, **what is stubbed**, and **what is deferred**,
so a maintainer can compare the running code against the 30-section design spec
(`ErisML-Compiler.md`) without confusion.

## Component truth table (as of v0.7.0)

| Component                        | Current status              | Production readiness         |
|----------------------------------|-----------------------------|------------------------------|
| Text compiler (12-pass pipeline) | shipped                     | alpha                        |
| DEME V2 (10-dim MoralVector)     | shipped                     | stable-ish (backward-compat) |
| DEME V3 bridge (Phases 1–4)      | shipped                     | alpha                        |
| Rank 1–6 tensors (Phase 5)       | shipped                     | alpha                        |
| Action / coalition `a` axis      | stub (length only)          | research                     |
| Coalition `c` axis               | real (4 enumeration modes)  | alpha                        |
| Strategic layer (Phase 6)        | shipped (Shapley + welfare) | alpha                        |
| DecisionProof (Phase 6)          | shipped (hash-chained)      | alpha                        |
| I-EIP Monitor (Phase 4)          | shipped                     | sampled-audit only           |
| Probe extractor (Tier 2.5)       | shipped                     | needs calibrated checkpoints |
| Calibration provenance on traces | not shipped                 | future (v0.8.0)              |
| Full ρ-estimation equivariance   | not shipped (identity only) | future (v0.8.0)              |
| MoralTensor-Bench                | not shipped                 | future (v0.8.0)              |
| `--strict-v3` enforcement        | shipped (v0.7.0)            | alpha                        |
| Runtime gating                   | not shipped                 | deferred                     |
| Web app / FastAPI surface        | not shipped                 | deferred                     |
| Silicon emit (Vitis HLS C++)     | shipped                     | no FPGA bring-up yet         |

## Phase 4 additions (in-flight on `main`)

Phase 4 ships the I-EIP Monitor (spec §31 follow-on) — Internal /
Activation / Delta lenses on top of v0.3.0. Four parallel tracks, all
green.

### Track A — Activation lens (`monitor/`)

- `ActivationSource` ABC + `LayerActivation` / `ActivationCapture`
  dataclasses defining the per-layer activation contract.
- `MockActivationSource` — deterministic, GPU-free; text-hash-anchored
  hidden states with a layer-dependent positional drift so probes have
  something structural to bite into. Used in CI.
- `HuggingFaceActivationSource` — loads any HF causal LM (default
  Qwen2.5-7B-Instruct), registers forward hooks on the resolved layer
  list (LLaMA/Qwen2/3/Mistral, GPT-2, BERT/RoBERTa), pools to per-layer
  `(D,)` vectors. Supports a `layers=` subset for memory bounding.
- `RemoteAtlasActivationSource` — paramiko-driven Atlas inference. The
  harness is a literal Python string baked into the source (no remote
  `git pull` at runtime; trust boundary is the SSH user account).
- `ActivationProbe` — per-layer head reusing the Phase-3 `ProbeHead`
  shape; can load a `ProbeBackbone.state_dict_for_checkpoint()` payload
  directly.
- `IEIPMonitor` — orchestrator: capture → per-layer probe → aggregated
  `MonitorTrace` with `trace_hash()` for audit-chain anchoring.
- 18 tests in `tests/test_monitor.py`, all CPU/MockActivationSource.

### Track B — Delta lens (`delta/`)

- `compare_morals(text_mv, activation_mv, **thresholds)` →
  `DeltaResult`. Per-dimension `value_delta`, `direction_match`,
  `confidence_gap`, `joint_uncertainty`. Overall `divergence` in [0, 1]
  with direction-break inflation. `flag_for_review` set when any of
  three triggers fires.
- `delta/equivariance.py` — BIP criterion `h_ℓ(g·x) ≈ ρ_ℓ(g)·h_ℓ(x)`
  with `ρ_ℓ(g) = identity` (invariance test). Default rewrites are
  surface-form only (whitespace, case, trailing period); semantic
  rewrites are caller-supplied and validated.
- `delta/failure_modes.py` — five named detectors:
  `TEXT_INTERNAL_MISMATCH`, `LAYERWISE_DRIFT`, `GROUP_SYMMETRY_BREAK`,
  `PROBE_UNCERTAINTY_SPIKE`, `AUDIT_CHAIN_BREAK`. Any firing sets
  `requires_human_review = True`. The Monitor does NOT produce verdicts.
- 20 tests in `tests/test_delta.py`.

### Track C — CLI

- `eris-compile monitor <text>` — runs the activation lens, emits a
  JSON trace with per-layer MoralVectors and `trace_hash`.
- `eris-compile delta <ir.json> <trace.json>` — runs the delta lens
  and the five failure-mode detectors, emits a structured report.
- CLI smoke tests in `tests/test_cli_monitor.py` (4 tests).

### Track D — Trust-boundary docs

- `docs/i_eip_monitor.md` — authoritative doc on the three lenses,
  threat model (T1 probe poisoning, T2 activation spoofing, T3
  group-action ambiguity), the trust-boundary diagram, the five
  failure modes, and what the Monitor is NOT.

### Phase 4 acceptance

- 42 new tests (18+20+4); 142 total tests passing across Phases 1–4.
- New CLI subcommands: `monitor`, `delta`.
- New package extras: `[monitor]` pulls `torch`, `transformers`,
  `paramiko`.
- Atlas-side inventory captured in `scripts/atlas/probe_models.py` and
  the user's memory note `reference_atlas_phase4_models.md`.

## Phase 3 additions (v0.3.0)

Phase 3 ships three parallel tracks on top of v0.2.0:

### Track A — Human-correction loop (spec §25 Phase 3 explicit)

- **`correction/` package**: `diff_irs`, `IRDiff`, `apply_corrections`,
  `CorrectionRecord`, `Corrector`. Structured diff between two IR JSONs;
  patch ops (`set`, `add`, `remove`) over dotted paths into keyed
  collections; corrections recorded in the IR's `extra["corrections"]`
  history with pre-/post-hash and rationale.
- **CLI subcommands**: `eris-compile diff <a> <b>` (with optional JSON
  output) and `eris-compile correct <ir> <corrections.json> --out <new.json>
  [--reevaluate]`. The `--reevaluate` flag re-runs the EM-DAG and DEME
  after corrections so the verdict reflects the corrected state.
- 12 tests covering diff, patch ops, audit-trail recording, hash drift,
  and round-trip.

### Track B — Probe-based calibration adopting sqnd-probe (v10.16.9 methods)

- **`ProbeExtractor`** as a new Tier-2.5 extractor: frozen LaBSE backbone
  + trainable probe heads. Sits between rule (Tier 2) and LLM (Tier 3):
  deterministic, multilingual, no API cost at inference. Falls back
  gracefully when checkpoints aren't available.
- **`calibration/` package**, adopting sqnd-probe's invariance machinery:
  - `losses.py`: `spectral_decoupling_loss` (cross-covariance minimisation),
    `vib_kl_loss` (variational information bottleneck), `confusion_loss`
    (negative entropy of the nuisance posterior), `GradientReversalFn` (GRL).
  - `adversarial_heads.py`: `MultiHeadAdversarial` with 4 varied
    architectures, per sqnd-probe v10.16.7+.
  - `probe_head.py`: `VIBLayer` + `ProbeHead` + `ProbeBackbone` (LaBSE
    frozen + VIB + head).
  - `train.py`: full training loop with `CalibrationConfig`,
    `save_checkpoint`, `load_checkpoint`.
  - `bond_index.py`: aggregate cross-cultural / cross-lingual
    generalisation score.
  - `dataset.py`: `ProbeBatch`, `ProbeTrainingDataset`, `synthetic_dataset`.
- **CLI**: `eris-compile calibrate --task synthetic --out probe.pt` runs
  the full training loop on a deterministic toy dataset (verifies the
  loop end-to-end). For real corpora, extend `calibration/dataset.py`
  with a corrected-IR loader.
- 16 tests covering each loss, GRL gradient flow, adversarial heads,
  Bond Index aggregation, dataset determinism. (1 skipped, LaBSE
  download-gated.)
- **Phase 3 does NOT ship trained weights.** Real weights need (a) a
  corpus of corrected-IR + source-text pairs from real annotation work,
  and (b) GPU compute (the LaBSE backbone is frozen but the heads still
  need many epochs). The drop-in path: produce a `.pt` checkpoint via
  the training loop, then point `ProbeExtractorConfig.role_checkpoint`
  (or `.fact_kind_checkpoint`) at it.
- **The learning loop closes**: extract -> human correction (Track A)
  -> calibrate (Track B) -> re-extract with the new probe.

### Track C — Silicon target concretisation (Alveo U55C / NRP Coder)

- **`silicon/` package**:
  - `fixed_point.py`: `FixedPointConfig`, scalar/array quantisation
    with saturation, `ap_fixed<W,I>` typedef emission for Vitis HLS.
  - `hls_emit.py`: `emit_fsm_cpp` (Commitment / Legitimacy / Consent FSMs
    -> synthesizable C++ with HLS pragmas), `emit_em_dag_pipeline` (EM-DAG
    skeleton in topological order with `#pragma HLS DATAFLOW`),
    `emit_top_module` (top-level kernel with AXI interfaces),
    `emit_makefile` (`v++` build for Xilinx U55C).
- **CLI**: `eris-compile silicon-emit --out-dir build/silicon` writes the
  full toolchain output ready to drop into an NRP Coder workspace with
  the "U55C FPGA Vitis Workflow" template.
- **`docs/nrp_coder_deployment.md`**: full step-by-step workflow from
  CLI emission through `make hw` to bitstream + host program.
- **`docs/silicon_target.md`**: updated with U55C-specific reference
  hardware and the toolchain.
- **`src/erisml_compiler/silicon/examples/`**: example emitted output
  for inspection (3 .cpp files + Makefile).
- 13 tests covering fixed-point quantisation, brace balance in emitted
  C++, topological order of EM-DAG pipeline, HLS pragma presence,
  U55C-targeted Makefile.
- **Phase 3 does NOT ship a synthesised bitstream.** The EM modules'
  bodies are placeholders returning constants. Phase 5+ fills in
  per-module fixed-point arithmetic and runs `v++` on real NRP hardware.

## Phase 2 additions (v0.2.0)

Phase 2 added the following on top of Phase 1:

- **`LLMExtractor` is no longer stubbed.** Full implementation with prompt
  templates, JSON-from-LLM parsing (handles markdown fences and prose),
  retry logic, and three concrete adapters: `MockLLMAdapter` for testing,
  `NRPOpenAIAdapter` for the NRP Nautilus hosted endpoint, `LocalVLLMAdapter`
  for self-hosted vLLM/SGLang.
- **`CriticExtractor`** runs two extractors and fuses their results with
  per-class agreement scoring (stakeholders, commitments, fact kinds,
  canonical form). Disagreements flag stakeholders for human review and
  embed a `critic_report` in the IR's extractor metadata. Usable from
  the CLI: `eris-compile compile <file> --extractor llm --critic rule`.
- **`Canonicalizer` layer** with two backends:
  - `RegistryCanonicalizer` (default, no ML deps): Jaccard-similarity
    matching against `ontology/canonical_forms.yaml`.
  - `LaBSECanonicalizer` (optional, `[ml]` extra): frozen LaBSE encoder
    (sentence-transformers/LaBSE) → cosine similarity in 768-dim
    multilingual space. Mirrors the BIP probe architecture from the
    `sqnd-probe` repo.
  - `auto_canonicalizer()` selects LaBSE when available, falls back to
    registry otherwise.
- **The pipeline orchestrator** runs the canonicalizer as Pass 7, augmenting
  or replacing the extractor's `canonical_form` based on situation-summary
  similarity to known forms.
- **Verified live against NRP**: smoke test `scripts/nrp_smoke_test.py`
  exercises NRPOpenAIAdapter against `gpt-oss` (qwen3 timed out / returned
  empty content). Token configuration via `ERISML_LLM_API_KEY` env var
  only; no hardcoded credentials anywhere.

A real Phase-2 LLM run on the Nazi-attic example exposed exactly the
safety property the architecture was designed for: the LLM proposed the
wrong canonical form (`institutional_loyalty_versus_public_truth_telling`),
but the registry canonicalizer overrode with the right tag
(`coercive_murderous_interrogation_with_collective_reprisal`), and the
critic pass detected disagreement between LLM and rule extractor (0.10
overall agreement), triggering `requires_human_review` rather than
auto-permitting. The verdict surfaces the uncertainty rather than hiding it.

### Live-tested NRP models

- `gpt-oss` — works cleanly, returns clean JSON, recommended default.
- `qwen3` — returned empty completions in our test; likely needs different
  prompt format (probable "thinking" mode). Investigation deferred.
- Untested in this session: `kimi`, `glm-5`, `minimax-m2`, `gemma-small`.

## What is built (Phase 1 MVP)

The Phase 1 MVP, per spec §25, satisfies the eleven acceptance criteria of §31.

### Functional
- **Text ingestion** (plain text, .txt, .md): `ingestion/text_loader.py`.
- **Paragraph-based segmentation**: `segmentation/segmenter.py`. Each paragraph
  becomes a segment with a `[start, end]` source span.
- **Stakeholder, commitment, ethical-fact, event, conflict extraction** via a
  `MockExtractor` that returns hand-curated IR for the three example texts.
  See "Mock vs LLM" below.
- **Canonical Intermediate Representation** as Pydantic v2 models matching §13.
- **MoralVector timeline** with the 10 dimensions of §14.
- **DEME-stub evaluation** returning a structured verdict, matching the
  structural-fuzzing target interface so Phase 2 can drop in real DEME.
- **Audit trail** with SHA-256 hash chain over the canonical JSON IR and
  per-pass provenance.
- **RLEF export** producing one training record per compiled document.
- **CLI** with subcommands: `compile`, `validate`, `rlef`, `version`.
- **Test suite** covering schema validation, pipeline end-to-end on the three
  examples, audit determinism, and RLEF export round-trip.
- **Quickstart notebook** at `notebooks/quickstart.ipynb`.

### Three example texts
- `examples/nazi_attic.txt` — coercive murderous interrogation with collective reprisal.
- `examples/medical_confidentiality.txt` — professional privilege vs.\ duty to warn.
- `examples/whistleblower.txt` — loyalty vs.\ truth-telling about institutional wrong.

Each comes with a hand-curated `MockExtractor` output exercising different
moral structures.

## What is stubbed

These components have the **right interface** but a placeholder implementation.
Phase 2+ will replace them.

| Component | Phase 1 stub | Replacement |
|---|---|---|
| `LLMExtractor` | Raises `NotImplementedError` with prompt-template scaffolding in docstrings | LLM adapter wiring (Phase 2) |
| `NRPOpenAIAdapter`, `LocalVLLMAdapter` | Skeleton only | Phase 2 |
| `DEMEBridge` | Returns canned `DEMEVerdict` shaped like the structural-fuzzing target | Real DEME execution (Phase 2) |
| Critic pass | Not present | Phase 3 |
| Human correction UI | JSON round-trip via CLI only | Web app (Phase 4) |
| Batch corpus mode | Not present | Phase 5 |
| FastAPI HTTP layer | Not present | Phase 4 |

## What is deferred entirely

These appear in the spec but are out of scope for the MVP build.

- **Internal lens / Activation lens / Delta lens** (§31 follow-on, three-lens architecture).
- **Web app frontend** (§16, React/Next.js + three-pane editor).
- **NRP / Kubernetes deployment** (§27, container + vLLM serving).
- **Storage layer** (§21, PostgreSQL + Celery/RQ queue).
- **Visualization** (§16.4, Plotly/Vega timeline rendering).
- **PDF / transcript ingestion** (§9 FR-1 later versions).
- **Sophisticated canonicalization** (§12 Pass 7, requires a learned semantic
  mapping; MVP uses free-text canonical-form tags with a small registry).

## Mock vs LLM extraction

The Phase 1 MVP ships with **two extractor implementations**:

1. **`MockExtractor`** (default): Returns hand-curated IR for the three example
   texts. Raises `UnknownDocumentError` for any other input. **This is not a
   general extractor.** It exists so the rest of the pipeline can be tested,
   exercised, and demonstrated without LLM API costs and without making
   capability claims the underlying model cannot back up.

2. **`LLMExtractor`** (skeleton): Defines the abstract interface and includes
   prompt-template *scaffolding* with detailed docstrings describing the
   expected output structure for each pipeline pass. Concrete model adapters
   (`NRPOpenAIAdapter`, `LocalVLLMAdapter`) are skeletons that raise
   `NotImplementedError` and document the integration points.

When wiring a real LLM adapter, the only changes required are:
- Implement the `_call_model(prompt: str) -> str` method on a concrete adapter
- Replace the prompt template scaffolding with model-tuned prompts
- Configure the CLI default extractor via `eris-compile compile --extractor llm`

The IR schema, pipeline, audit, and export layers are model-agnostic and need
no changes.

## Acceptance criteria status (§31)

| # | Criterion | Status |
|---|---|---|
| 1 | Take short scenario as text | ✓ CLI accepts text files |
| 2 | Extract stakeholders with source spans | ✓ MockExtractor on 3 examples |
| 3 | Detect at least one commitment/vow | ✓ |
| 4 | Detect at least one coercion or consent issue | ✓ |
| 5 | Detect at least one third-party externality | ✓ |
| 6 | Emit valid ErisML Compiler IR | ✓ Pydantic v2 schemas |
| 7 | Construct MoralVector timeline | ✓ 10-dim per §14 |
| 8 | Run or mock DEME evaluation | ✓ Stub with target interface |
| 9 | Produce audit record | ✓ SHA-256 hash chain |
| 10 | Allow human correction | ✓ JSON round-trip via CLI |
| 11 | Export one RLEF training record | ✓ `rlef` subcommand |

## What this MVP is honest about

- The MockExtractor does not generalize. It is fixtures.
- The DEME stub returns a deterministic verdict shape; it does not actually
  evaluate against geometric constraints.
- Canonicalization is a free-text tag; semantically equivalent texts will not
  reliably map to the same canonical form until Phase 2 lands a learned
  canonicalizer.
- Audit determinism is preserved within a single Python version + dependency
  set. Cross-version determinism requires pinning Pydantic and PyYAML versions,
  which `pyproject.toml` does at minor-version granularity.
