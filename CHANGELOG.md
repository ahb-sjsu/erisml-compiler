# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.8.0 — 2026-06-13 — Framework-pluralist DAG-native architecture

The largest architectural release since the original DEME V3 alignment.
Driven by a r/Compiler review thread that correctly identified the IR
as encoding metaethical commitments below the profile layer. The
response is the full Option-C refactor from
`release-planning-06-framework-pluralist-architecture.md`: a two-layer
IR with a DAG-native descriptive substrate and N framework
projections.

### Added

- **`MoralGraph` substrate** (`src/erisml_compiler/ir/graph/`) —
  typed DAG with nodes ∈ {stakeholder, act, maxim, commitment, fact,
  norm} and edges ∈ {performs, imposes_on, consents_to,
  holds_commitment, commitment_binds, treats_as, under_maxim,
  coerces, surfaces_fact, fact_subject,
  would_violate_if_universalised}. Carries a canonical SHA-256
  `graph_hash` reproducible across compiles, included in every
  audit record.
- **`graph_from_flat` promotion** + **`flat_from_graph` back-
  derivation**. Round-trip is bit-stable (hash-equality verified on
  bundled examples).
- **Two-layer IR**: `MoralSubstrate` (descriptive view over the
  graph) + `Projection` (framework-bound analyser). Each projection
  emits a `ProjectionResult` with a normalised `polarity` field
  (`permit` / `forbid` / `escalate` / `neutral`) for cross-framework
  comparison.
- **Four framework projections**:
  - `ConsequentialistProjection` — the per-stakeholder tensor + Gini
    + DEME verdict path; records `graph_aware` + `graph_summary` in
    its metadata.
  - `DeonticProjection` — Kantian categorical *gates*
    (`universalizability`, `mere_means`, `valid_consent`,
    `legitimate_authority`). Emits `GateFinding(passed, severity,
    reason)` objects, not channel contributions. Pattern-matches
    `treats_as` and `imposes_on`/`consents_to` edges directly.
  - `VirtueProjection` — Aristotelian character / habit / power-
    asymmetry reading. Verdicts: `virtuous` /
    `requires_practical_wisdom` / `vicious`.
  - `CareEthicsProjection` — Gilligan/Noddings/Tronto relational
    primitives (attentiveness, asymmetric responsibility, dependency
    response). Verdicts: `caring` / `requires_caring_attention` /
    `uncaring`.
- **Cross-projection disagreement**: when ≥2 projections disagree
  by polarity, `ir.cross_projection_disagreement` surfaces all
  verdicts. The compiler refuses to aggregate; choosing across
  frameworks is left to the caller as an explicit metaethical move.
- **EM-DAG graph-native helpers** (`em_dag/modules/_helpers.py`).
  `facts_of_kind`, `active_commitments`, `vulnerable_stakeholders`,
  `nonconsenting_third_party_ids`, and `stakeholders_with_role`
  now read graph nodes/edges when `ir.graph` is set, falling back
  to flat fields otherwise. The `nonconsenting_third_party`
  detection now uses typed-edge semantics (`IMPOSES_ON` without
  paired `CONSENTS_TO`) rather than role-label string matching.
  Verdicts byte-identical to the v0.7.x flat baseline (golden test
  covers 10 modules × 3 scenarios × {value, confidence,
  direction}).
- **Extractor graph emission**: `ExtractorResult.graph: MoralGraph |
  None`. The `RuleExtractor` emits the graph alongside flat lists;
  the orchestrator skips its own promotion step when the extractor
  provided one.
- **CLI**:
  - `--projection consequentialist_distributive,deontic_kantian,virtue_aristotelian,care_ethics_relational`
    (default: all four)
  - `--ethos-profile <yaml>` — applies fitted per-EM-module weights
    at projection time. Two profiles bundled:
    `dear_abby_socialchem_v0.1`, `aita_socialchem_v0.1` (fit from
    Social Chem 101, Forbes et al. EMNLP 2020).
  - `eris-compile bench run` — runs MoralTensor-Bench against the
    bundled seed corpus (3 scenarios).
- **`AuditRecord` gains `graph_hash`, `ethos_profile`,
  `ethos_profile_sha256`** so the audit chain captures the full
  provenance of the substrate + the ethos applied.
- **RLEF export schema v0.2**: every record now includes
  `moral_graph` (nodes + edges + canonical_json + graph_hash) plus
  the per-framework `projections` block plus
  `cross_projection_disagreement`. Trainers can learn against the
  typed graph, per-framework verdicts, or the flat tensor timeline
  — whichever fits their objective.
- **`ρ-estimation` core** (`delta/transforms.py`,
  `delta/rho_estimation.py`) — closed-form Procrustes + LSTSQ
  estimators of the per-layer linear map ρ_ℓ(g) over activation
  pairs, plus `RhoEstimate` + per-pair residual computation. Extends
  `check_equivariance` to accept an optional `rho_map`. CLI
  `fit-rho` subcommand deferred.
- **MoralTensor-Bench harness** with 3 seed scenarios (re-cast from
  the bundled examples) and 7 per-metric scorers. Baseline score on
  the seed corpus with `--extractor rule`: 0.136 (honest finding
  about extractor coverage — semantic stakeholder IDs don't fuzzy-
  match the extractor's generic `self`/`collective_*` IDs).
- **Eigenvalue spectral scalar** + higher-rank mode-n unfolding
  attached to every `MoralTensorV3` (release-planning-04).
- **Named ethos fitter** (`social_chem/`): reads the AI2 Social
  Chem 101 corpus, projects MFT foundations onto EM-DAG modules,
  fits per-module weights normalised so unmapped modules implicitly
  default to 1.0 at the moral_vector projector.

### Changed

- **IR schema version**: `erisml_compiler_ir_v0.2` →
  `erisml_compiler_ir_v0.3` (additive — old IRs still parse).
- **CompilerIR** gains optional fields: `graph: MoralGraph | None`,
  `projections: dict[str, Any]`,
  `cross_projection_disagreement: dict | None`.
- **Pass 8 (tensorisation)** now runs the projection pass with all
  enabled frameworks; the consequentialist projection back-fills
  the legacy `moral_tensor_v3` / `deme_verdict` /
  `per_party_verdicts` / `fairness_metrics` fields for backward
  compat. Pass 10 (DEME evaluation) becomes a no-op since the
  consequentialist projection already does it.
- **Pass 7.5 (graph identity)** new pass — promotes the flat
  extractor output to a `MoralGraph` and records the `graph_hash`
  in the audit chain.
- **README** explicitly notes the IR substrate still encodes
  choices (extraction categories) even after the refactor — the
  metaethical commitment shrank, but did not vanish. Honest about
  what's still load-bearing.

### Architectural decisions documented

- `docs/plans/release-planning-06-framework-pluralist-architecture.md`
  — the architectural argument + Kantian-gate limitations + the
  r/Compiler review thread response.
- `docs/plans/release-planning-07-graph-consumer-migration.md` —
  migration matrix for graph-consumer subsystems + explicit
  deferral rationale for monitor (operates on activations not
  facts), silicon (emits evaluator not data), bench-per-projection
  (multiplies gold-curation cost by N).

### Backward compatibility

- All existing flat-field IRs continue to load; new fields default
  to None / empty.
- Legacy reads via `ir.stakeholders`, `ir.moral_tensor_v3`,
  `ir.deme_verdict`, etc. remain valid — populated by graph
  back-derivation + the consequentialist projection.
- The legacy `--em-profile` flag still loads a DAG profile (which
  modules + their deps). The new `--ethos-profile` flag is the
  separate fitted-weights path.

## v0.7.0 — 2026-06-12 — DEME V3 alignment + ethos profiles

(Pre-existing release; see commit history.)
