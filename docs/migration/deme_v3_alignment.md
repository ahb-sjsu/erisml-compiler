# DEME V3 Alignment — Migration Plan (Scope C)

**Status:** in progress. Live document — update as each phase completes.

## Goal

Bring `erisml-compiler` from its current V2-shape IR (10 dimensions, rank-1
MoralVector + rank-2 MoralTensor with per-stakeholder dict, custom 10-module
EM-DAG) to full DEME V3 alignment as defined in `erisml-lib`:

- 9 moral dimensions derived from the 3×3 "Nine Dimensions of Ethical
  Assessment" matrix (axis k);
- MoralTensor ranks 1–6 with axes `(k, n, τ, a, c, s)`;
- EthicalFactsV3 + EthicalJudgementV3 as the canonical IR↔DEME boundary;
- Tier 0–4 module hierarchy with the V3 module variants
  (`geneva_em_v3`, `triage_em_v3`, `autonomy_consent_em`,
  `greek_tragedy_tragic_conflict_em`);
- Coalition, Game Theory (Shapley), Strategic Layer integration;
- DecisionProof audit artifacts alongside the existing SHA-256 chain.

## Why now

The current 10-dimension layout has held since the spec draft. DEME V3's
9-dimension layout aligns with the published "Nine Dimensions" framework
and is structurally cleaner (3×3 derivation, no orphan dimensions like
`repair_residue` that are really tensor operations rather than axes).
The longer we keep two layouts in two repos, the more bridging code
accumulates. The JOSS submission is in pre-review window — clean state
landing during review reads as an in-flight improvement, not as
instability.

## Phase plan

Each phase ships standalone (tests green, CI green, repo usable). The
ordering minimises simultaneous breakage in the test suite.

### Phase 1 — IR schema for ranks 1-6 (foundation)

Replace `MoralVector` and `MoralTensor` in `src/erisml_compiler/ir/schemas.py`
with `MoralTensorV3`: serialisable Pydantic model that mirrors
`erisml.ethics.moral_tensor.MoralTensor`'s shape but stays JSON-friendly.

- `MORAL_DIMENSIONS` becomes the V3 9-tuple.
- Add `MoralTensorV3` with `rank`, `shape`, `axis_names`, `axis_labels`,
  `values` (nested list for JSON portability), `veto_flags`,
  `veto_locations`, `metadata`.
- Add `MoralVectorV3` as a thin convenience accessor for rank-1 tensors
  (back-compat surface; deprecated for new code).
- Add `migrate_v2_to_v3` helper for the old 10-dim → new 9-dim mapping.
- Update `__schema_version__` → `erisml_compiler_ir_v0.2`.

**Test surface:** new `tests/test_v3_schema.py` (round-trip JSON, rank
validation, axis-name validation). Old `test_ir_schemas.py` updated to
expect V3; old V2 path expected to raise a clear deprecation.

### Phase 2 — Tensor builder produces rank-2 by default

`src/erisml_compiler/evaluation/tensor_builder.py` rewritten to produce
rank-2 `(k=9, n=stakeholder_count)` tensors as the new default. Examples
re-curated (`nazi_attic`, `medical_confidentiality`, `whistleblower`)
with per-stakeholder rank-2 reference output.

- Canonicalizer + critic updated for the 9-dim ontology
  (`ontology/moral_dimensions.yaml`).
- Bundled `--rank N` CLI flag with default 2; rank-1 still available
  for I-EIP Monitor and silicon paths.

### Phase 3 — EM-DAG migration to V3 module tiers

Replace the custom 10-module DAG with imports of `erisml-lib`'s
`modules.tier0.geneva_em_v3`, `modules.triage_em_v3`,
`modules.tier2.autonomy_consent_em`,
`modules.greek_tragedy_tragic_conflict_em`. Compiler-side becomes a
*routing layer* that picks which V3 modules to invoke based on the
detected scenario shape (presence of consent issues → autonomy_consent_em;
tragic-conflict shape → greek_tragedy module; etc.).

- Add `erisml-lib` as a dependency in `pyproject.toml`.
- Remove `src/erisml_compiler/em_dag/modules/*` files (autonomy.py,
  care.py, etc.) — their semantics live in V3 modules now.
- Keep `em_dag/dag.py` topological-sort logic; rewire to V3 modules.

### Phase 4 — V3 facts + V3 judgement at the bridge

`src/erisml_compiler/erisml_backend/deme_bridge.py` rewritten to:

- assemble compiler IR fragments (stakeholders, commitments,
  ethical_facts) into `EthicalFactsV3`;
- invoke the V3 module tier through the DAG;
- collect outputs into `EthicalJudgementV3` with per-party verdicts and
  veto locations.

Surface Gini coefficients + worst-off identification in the IR
(`CompilerIR.fairness_metrics`).

### Phase 5 — Coalition + temporal axes (ranks 3-6)

Pipeline produces rank-3+ when scenarios have:

- multiple time steps → rank-3 `(k, n, τ)`;
- explicit action choices → rank-4 `(k, n, a, c)` via
  `erisml.ethics.coalition.CoalitionContext`;
- uncertainty Monte Carlo samples → adds the `s` axis.

Add a rank-6 demo to `examples/` showing the full `(k, n, τ, a, c, s)`
analysis on a synthetic scenario.

### Phase 6 — Strategic layer + decision proofs

Compiler emits a `DecisionProof` artifact alongside the existing
`AuditRecord`. Strategic-layer recommendations (Nash equilibrium,
policy hints, coalition stability) attached when the input scenario
has more than one decision agent.

Hash chains merged: `AuditRecord.ir_hash` becomes
`DecisionProof.layer_outputs[i].input_hash`.

### Phase 7 — Silicon emit migration

`silicon/hls_emit.py` rewritten for the V3 tensor representation
(rank-2 minimum). Decide whether rank-3+ goes off-chip or gets its own
hardware shape. Fixed-point conversion of the rank-2 tensor as the
baseline.

### Phase 8 — I-EIP Monitor migration

`monitor/activation_probe.py` outputs rank-1 V3 tensors (not 10-dim
V2 MoralVectors). `delta/compare.py` updated for rank semantics.
Possibly add a new failure mode `rank_mismatch` for when the text
lens and activation lens disagree on tensor rank itself.

### Phase 9 — Docs + JOSS paper

Update README, `docs/architecture.md`, `paper/paper.md`,
`SCOPE.md`, `CITATION.cff` for the 9-dimension, ranks 1–6 layout.
Add `docs/deme_v3.md` explaining tensor semantics and rank promotion.

The JOSS reviewer experience: bot picks up changes from `main`. Phase
9 lands as one cohesive update; reviewers see a clean state, not a
piecewise migration. JOSS bot regenerates the paper PDF automatically.

### Phase 10 — DEME V3 enhancements in erisml-lib (as needed)

Things the compiler may need that erisml-lib doesn't ship today:

- **Canonical JSON serialisation for MoralTensor.** The current numpy
  storage isn't disk-stable. Add `MoralTensor.to_json_dict()` and
  `from_json_dict()` in erisml-lib.
- **Compiler bridge module** — `erisml.ethics.bridges.compiler` that
  takes compiler IR fragments and emits `EthicalFactsV3`. Belongs in
  erisml-lib so the contract is enforced symmetrically.
- **Empty module tiers** — `tier1`, `tier3`, `tier4` packages exist
  but are empty; if the compiler's scenarios require them, add
  skeleton modules.

These enhancements go to erisml-lib via PR. **Blocked locally** by the
existing unresolved merge conflict in
`docs/geometric-communication/appendix_a_math.html`. Work will happen
on a feature branch from `origin/main` not touching that file.

## Versioning

- Schema: `erisml_compiler_ir_v0.1` → `erisml_compiler_ir_v0.2`.
- Package: `0.4.0` → `0.5.0` lands at Phase 5 / 6 boundary (rank-3+
  available); `0.6.0` at Phase 8 (Monitor migrated); `1.0.0` after
  Phase 10 once the cross-repo contracts are firm.
- The JOSS paper's `version: 0.4.0` field updates with each release.

## Risk register

| Risk | Mitigation |
|---|---|
| Phase 3 destroys Phase 1–3 test suite | Rewrite tests phase-by-phase; do NOT defer the V3 test suite to the end |
| Mid-migration JOSS reviewer arrives | Pause migration on `main` between phases; finish current phase before any push |
| erisml-lib merge conflict | Work only on feature branches off `origin/main`; don't touch user's unresolved files |
| Coalition / Strategic layer pulls in dependencies the JOSS reviewers won't tolerate | Phase 5 + 6 land in `[strategic]` extras, not core |
| Silicon emit becomes a rank-2 only artifact | Document this explicitly in `docs/silicon_target.md` — the hardware path is rank-bounded by design |

## Status tracker

- [x] Phase 1: IR schema for ranks 1-6 — done 2026-06-12 (22 tests, V2 untouched)
- [x] Phase 2: Tensor builder produces rank-2 by default — done 2026-06-12 (12 new tests, V2 untouched, --rank CLI flag, end-to-end nazi_attic rank-2 verified)
- [x] Phase 3: V3 bridge invokes DEME V3 modules — done 2026-06-12 (erisml_backend/v3_bridge.py wires IR -> V2 EthicalFacts -> EthicalFactsV3.from_v2 -> GenevaEMV3 + TriageEMV3 -> weighted-mean MoralTensorV3. Orchestrator dispatches to bridge when erisml-lib available, falls back to Phase 2 fanout otherwise. 7 new tests, 16 pre-existing V2 tests still green. Per-party uniformity remains until Phase 4 builds per-party facts from EthicalFact.subjects.)
- [x] Phase 4: V3 facts + V3 judgement at the bridge — done 2026-06-12 (erisml_backend/v3_facts_direct.py constructs EthicalFactsV3 directly from compiler IR using EthicalFact.subjects for per-party attribution; bridge now uses it preferentially with V2-aggregation fallback. ir.per_party_verdicts + ir.fairness_metrics surfaced on CompilerIR. Per-stakeholder divergence confirmed on nazi_attic: speaker/village→harm 0.76/0.83/forbid, refugees→0.0/prefer, nazis→0.18/neutral, Gini=0.43. veto_location validator relaxed to accept DEME V3's (party_idx,) single-axis convention. 10 new tests, 51 V3 tests total green, 16 V2 still green.)
- [x] Phase 5: Higher-rank tensors (ranks 3-6) — done 2026-06-12 (erisml_backend/v3_higher_rank.py builds rank-3..6 by stacking rank-2 V3-bridge slices over time / action / coalition / uncertainty axes. CLI: `--rank 3..6 --n-actions --n-coalitions --n-samples --sample-noise --sample-seed`. Real axes: τ (event-timeline filtering of ethical_facts by source-span overlap with events), s (Monte Carlo over fact.confidence with deterministic per-sample gaussian). Stub axes: a, c (length parametric but values replicated; Phase 6 will inject CoalitionContext semantics). Verified end-to-end on nazi_attic: temporal evolution shows speaker harm 0.18→0.76→0.76, village 0.0→0.828→0.828 monotonically as events unfold. 15 new tests covering shapes, real/stub axis assertions, config validation, and JSON roundtrip at rank-6.)
- [ ] Phase 6: Strategic layer + decision proofs
- [ ] Phase 7: Silicon emit migration
- [ ] Phase 8: I-EIP Monitor migration
- [ ] Phase 9: Docs + JOSS paper
- [ ] Phase 10: erisml-lib enhancements (as discovered)
