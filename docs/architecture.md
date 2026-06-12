# Architecture

This document describes the runtime architecture of the ErisML Compiler as
of Phase 4 on `main`. For the original 31-section design spec see
`ErisML-Compiler.md`. For per-phase delivery status see `SCOPE.md`. For the
I-EIP Monitor (Phase 4) threat model see `docs/i_eip_monitor.md`.

## Three tiers, one spine

The compiler offers three tiers of operation; the **evaluator core** is
shared across all three. Tiers differ in how structured input arrives at
the evaluator.

| Tier | Input | Path |
|---|---|---|
| 1 Geometric | Pre-parsed JSON event stream | structured loader → FSM + EM-DAG + DEME |
| 2 Rules | Natural-language text | text loader → segmenter → RuleExtractor → FSM + EM-DAG + DEME |
| 3 LLM | Natural-language text | text loader → segmenter → LLMExtractor → critic → FSM + EM-DAG + DEME |
| 2.5 Probe | Natural-language text | text loader → segmenter → ProbeExtractor (calibrated LaBSE+VIB head) → FSM + EM-DAG + DEME |

The evaluator core is FSM + EM-DAG + DEME bridge. It is deterministic,
bounded-memory, fixed-point-friendly, and silicon-castable. The four
extractor tiers differ only in their frontend; their outputs are
homogeneous IR fragments validated against `ir/schemas.py`.

## The 12-pass pipeline

Per spec section 12, in tier-agnostic form:

| Pass | Stage | Implementation |
|---|---|---|
| 0 | Ingestion | `ingestion/text_loader.py` or `ingestion/structured_loader.py` |
| 1 | Segmentation | `segmentation/segmenter.py` (text tiers only) |
| 2–6 | Extraction (entity, stakeholder, event, norm, ethical fact) | `annotation/{mock,rule,llm,probe}_extractor.py` |
| 7 | Canonical-form resolution | `canonicalizer/` (Jaccard registry → LaBSE cosine snap) + `annotation/critic.py` |
| 8 | Tensorisation | `evaluation/tensor_builder.py` (walks the EM-DAG at each time step) |
| 9 | ErisML IR | `erisml_backend/codegen.py` (renders the in-memory IR) |
| 10 | DEME evaluation | `erisml_backend/deme_bridge.py` |
| 11 | Contraction & residue | Embedded in `DEMEVerdict` |
| 12 | Audit artifact | `audit/hash_chain.py`, `audit/artifact.py` |

## EM-DAG

The DEME evaluator is a directed acyclic graph of **Ethical Modules**
(see `em_dag/`). Each module owns one MoralVector dimension (per spec §14)
and declares its dependencies on other modules. The DAG is topologically
sorted at construction time and evaluated in that order.

The default DAG (`em_dag/profiles/default.yaml`) has 10 modules:

- **Roots** (no deps): harm, rights, fairness, legitimacy, epistemic
- **Second tier**: autonomy (← legitimacy), fidelity (← legitimacy),
  externality (← harm), care (← harm)
- **Composition tier**: repair (← harm, externality, fidelity)

A different deployment can swap in a different DAG by writing a new YAML
profile and pointing the CLI at it via `--em-profile`. The topology is
the DEME profile.

## FSM layer

For Tier 1 (silicon target), moral state is tracked by deterministic
finite-state machines (`fsm/`):

- `CommitmentFSM`: vow lifecycle (active → defeasible → defeated/violated/fulfilled/void/expired)
- `LegitimacyFSM`: authority legitimacy (fully_legitimate → defeasible/coercive/fraudulent → tyrannical/void)
- `ConsentFSM`: consent (not_obtained → obtained/coerced → withdrawn)

Each FSM is a small state register (3 bits is enough for any of them)
with a deterministic transition table. Terminal states are absorbing. In
silicon, the entire moral-state vector for a scenario is bounded by
`O(n_commitments + n_authorities + n_consenters)` state registers.

## Canonicalizer and critic

Tier 3 LLM output is canonicalized in two stages before tensorisation:

1. **Registry pass** (`canonicalizer/registry.py`): Jaccard similarity
   over the LLM-produced canonical_form against a hand-curated registry
   of known canonical forms. Snap to the highest-similarity entry above
   threshold.
2. **LaBSE pass** (`canonicalizer/labse.py`): if no registry hit, use
   the frozen LaBSE encoder + cosine similarity to pick a canonical form.

Independently, the **critic** (`annotation/critic.py`) compares the
LLM's choice against a deterministic second-opinion extractor. When the
two disagree, the IR is flagged with `requires_human_review=True` —
the canonicalizer's correction does not silence the disagreement, it
records it.

## Probe extractor (Tier 2.5)

The probe extractor (`annotation/probe_extractor.py`,
`calibration/probe_head.py`) replaces text-pattern extraction with a
calibrated classifier head over a frozen multilingual encoder
(LaBSE). The training stack adopts sqnd-probe v10.16.9 methods:

- Spectral decoupling: cross-covariance minimisation against nuisance
  channels (language, topic).
- VIB: variational information bottleneck on the LaBSE → probe
  representation.
- Multi-head GRL adversarial: four parallel adversarial heads (varied
  architectures) with gradient reversal, per sqnd-probe v10.16.7+.
- Confusion loss: negative entropy of the nuisance posterior.

Probes are trained via `eris-compile calibrate`, checkpointed with
`save_checkpoint`, and consumed at inference by the ProbeExtractor.

## DEME V3 alignment

The compiler is fully aligned with DEME V3's tensor + module model
(see `docs/migration/deme_v3_alignment.md` for the migration history).
**All six phases of the alignment have shipped.** Six V3 surface fields
on `CompilerIR`:

- `ir.moral_tensor_v3: MoralTensorV3` — ranks 1–6 over axes `(k, n, τ, a, c, s)`
- `ir.per_party_verdicts: dict[stakeholder_id, str]` — conservative aggregate
- `ir.fairness_metrics: dict[str, float]` — `gini_harm`, `worst_off_harm_value`
- `ir.strategic_analysis: dict` — Shapley values per stakeholder + welfare metrics
- `ir.decision_proof: dict` — hash-chained tamper-evident decision provenance
- `ir.schema_version = "erisml_compiler_ir_v0.2"`

Modules:

- `ir/v3/` — `MoralTensorV3` Pydantic schema (ranks 1–6, validators for
  shape / first-axis-length / single-axis veto convention) + V2→V3
  migration helpers. JSON-serialisable mirror of
  `erisml.ethics.moral_tensor.MoralTensor`.
- `evaluation/tensor_builder_v3.py` — Phase-2-style fallback (rank 1
  or 2) when erisml-lib is absent; the bridge takes over otherwise.
- `erisml_backend/v3_facts_direct.py` — builds `EthicalFactsV3`
  directly from compiler IR with per-party attribution from
  `EthicalFact.subjects`; cross-dimensional propagation rules
  (coercion → harm + rights, externality → harm + societal, …) make
  signals from one fact-kind register on multiple V3 dimensions.
- `erisml_backend/v3_bridge.py` — invokes registered V3 modules
  (`GenevaEMV3`, `TriageEMV3`), aggregates by `default_weight`,
  produces rank-1 / rank-2 tensors with per-party verdicts +
  fairness metrics.
- `erisml_backend/v3_higher_rank.py` — stacks rank-2 slices into
  ranks 3-6. Real axes: τ (event-timeline filtering), s (Monte Carlo).
- `erisml_backend/v3_phase6.py` — Shapley attribution + welfare
  metrics + hash-chained `DecisionProof`. Also injects real
  coalition semantics on the c axis: per-coalition rank-2 slices
  via `grand_only` / `all_subsets` / `singletons_only` / `pairwise`
  enumeration, with non-coalition members zeroed.

The V2 surface (10-dim `MoralVector`, per-stakeholder `MoralTensor`,
V2 EM-DAG) remains alive for backward compatibility. The V2
`moral_vectors` field and `timeline` list still populate on every
compile; consumers gradually migrating to V3 can keep both.

CLI knobs:

```
eris-compile compile <input> \
  --rank {1..6} \
  --n-actions N --n-coalitions N --n-samples N \
  --sample-noise STD --sample-seed N
```

The `--coalition-mode` for strategic-side Shapley enumeration is fixed
at `all_subsets` today; expose as a CLI flag in a future iteration.

## Audit

Every compilation produces an `AuditRecord` (`audit/hash_chain.py`)
containing:

- SHA-256 of the canonical-JSON IR (excluding the `audit` field, and
  excluding `document.timestamp` and `document.source` so the hash is
  reproducible across runs).
- SHA-256 of the source text.
- Compiler version, schema version.
- Tier and extractor names.
- EM-DAG profile name.
- Timestamp (UTC).
- Per-pass duration records.

The `bundle` subcommand bundles the IR, the source text, the ErisML
rendering, and a human-readable Markdown report into a single
directory. The `report` subcommand produces a self-contained HTML
artifact with embedded CSS and (if matplotlib is available) an
embedded base64 timeline PNG.

## Correction loop

`correction/` implements the human-in-the-loop feedback path:

- `diff_irs(a, b)` produces an `IRDiff` of patch ops (`set`, `add`,
  `remove`) over dotted paths into keyed collections.
- `apply_corrections(ir, corrections)` applies a patch and records a
  `CorrectionRecord` in `extra["corrections"]` with pre-/post-hash and
  rationale.
- The CLI `eris-compile correct ... --reevaluate` re-runs the EM-DAG
  and DEME after corrections so the verdict reflects the corrected
  state.

This is what closes the RLEF loop: corrected IRs become training
records via `eris-compile rlef`, and the (original IR, correction,
post-IR) triple feeds future probe calibration.

## Streaming layer

`streaming/` exposes a `MoralStreamer` that yields `StreamEvent` objects
as the pipeline progresses. The `TerminalCaptioner` renders the stream
as live text on stdout (`compile --stream`). The same stream feeds the
HTML report's timeline section.

## Silicon backend

`silicon/` and the `eris-compile silicon-emit` subcommand produce
Vitis HLS C++ for the FSM and EM-DAG cores. The fixed-point conversion
(`silicon/fixed_point.py`) takes the float MoralVector arithmetic and
re-expresses it in `ap_fixed<TOTAL, INT>` types parameterised on the
command line. The emitted files target the Xilinx Alveo U55C in the
NRP Coder "U55C FPGA Vitis Workflow" template — see
`docs/nrp_coder_deployment.md` for the build workflow and
`docs/silicon_target.md` for what is and is not silicon-castable.

## I-EIP Monitor (Phase 4)

The I-EIP Monitor (`monitor/`, `delta/`) is the activation-side
complement to the text-side pipeline above. It runs out-of-band on a
sampled subset of inputs and emits `requires_human_review` plus a
structured failure-mode report when the three lenses disagree.

```
text  ──► (pipeline above)            ──► CompilerIR + DEME verdict
           │
           └── same text ──► ActivationSource (Mock / HF / Atlas)
                                ──► IEIPMonitor (per-layer ActivationProbe)
                                ──► MonitorTrace

       CompilerIR.global moral state ──┐
                                       ├─► Delta lens (compare_morals,
       MonitorTrace.aggregated  ───────┘            equivariance,
                                                    5 failure modes)
                                                    │
                                                    └─► requires_human_review
```

Three lenses:

- **Text lens** — Phases 1–3, the IR extracted from model output.
- **Activation lens** — Phase 4 Track A. `ActivationSource` (Mock, HF
  transformers, or paramiko-driven Atlas), forward hooks on
  `model.model.layers` (LLaMA/Qwen/Mistral) or equivalent, per-layer
  `ActivationProbe` reusing the Phase-3 head shape.
- **Delta lens** — Phase 4 Track B. `compare_morals` for per-dimension
  deltas + BIP equivariance test `h_ℓ(g·x) ≈ ρ_ℓ(g)·h_ℓ(x)` with
  `ρ_ℓ = identity` (invariance under semantics-preserving rewrites).

The Monitor is silicon-incompatible by design. Activation hooks need a
runtime that can introspect transformer hidden states; the silicon
target ships only the deterministic Tier-1 path. In production, the
silicon path handles the real-time loop and the Monitor runs out-of-band
audit on sampled inputs.

## Phase boundaries

| Phase | Status | Adds |
|---|---|---|
| 1 | shipped v0.1.0 | Schema, pipeline, mock + rule extractor, EM-DAG, FSMs, DEME stub, audit, streaming, HTML, RLEF, CLI, tests |
| 2 | shipped v0.2.0 | LLM adapters (NRP + vLLM), critic pass, canonicaliser, real DEME, IR diff + correction |
| 3 | shipped v0.3.0 | Probe extractor (Tier 2.5) + calibration stack + sqnd-probe losses; silicon-emit (Vitis HLS C++) |
| 4 | shipped v0.4.0 | I-EIP Monitor: Internal / Activation / Delta lenses + 5 failure-mode detectors + trust-boundary docs |
| 5 (DEME V3) | shipped v0.5.0–v0.7.0 | Full DEME V3 alignment in six sub-phases: ranks-1..6 `MoralTensorV3`, per-party verdicts, Gini + worst-off fairness, V3 bridge invoking GenevaEMV3 + TriageEMV3 + per-party direct facts, temporal (τ) + Monte Carlo (s) axes, real coalitions on c, Shapley attribution, hash-chained DecisionProof |
| 5+ | deferred | Production web app: React frontend + FastAPI HTTP layer + three-pane editor |
| 5+ | deferred | Batch corpus mode, PostgreSQL + Celery, RLEF dataset generation |
| 5+ | partially blocked | Silicon hardware bring-up on the U55C (Vitis HLS emit is shipped; on-FPGA bitstream gated by NRP Coder pipeline) |

See `SCOPE.md` for what is built versus stubbed versus deferred at the
file level.
