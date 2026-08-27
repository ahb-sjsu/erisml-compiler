# Spec — DEME verdict function v1 (replacing the v0 bridge heuristic)

**Status:** proposal, 2026-08-27. Not implemented. Written against the
DEME architecture of record (`erisml-lib/docs/moralvector_v2_architecture.md`,
`erisml-lib/docs/moralvector_reference.md`), not invented fresh: most
of what follows is *conformance* to a design already specified there
but not implemented in `erisml_backend/deme_bridge.py`.

**Provenance and a disclosure.** This spec was prompted by two
findings from an external campaign (LBI-3, a criminal-justice
governance study in `erisml-lib/docs/papers/foundations/lbi3-pluralist-governance/`)
whose measurement is *blocked* on the defects below. To keep that
dependency from bending the design, the spec is written to the
architecture's requirements and to properties testable without
reference to LBI-3's data, and **no LBI-3 statistic may be computed
until this is implemented and frozen.** Where a choice is genuinely
ours rather than the architecture's, it is marked **[choice]**.

---

## 1. What v0 does, and the three defects

`DEMEBridge` (v0.1) computes a verdict from six values
(`deme_bridge.py:43–48`) through five conjunctive branches, else
`indeterminate`.

**D1 — reads 6 of 10 channels.** `fairness_equity`,
`autonomy_respect`, and `rights_respect` appear in **no branch**. A
maximal fairness violation (−0.85) changes no verdict. This
contradicts the architecture, which classifies `fairness` and
`autonomy` as **substantive** policy blocks that "map to `k`
dimension(s) and [are] realized as an EM contraction."

**D2 — gates substantive verdicts on a procedural attestation.** Both
permit branches require `fidelity_v > 0.3` / `> 0.5`. Fidelity is
commitment-keeping — the architecture's `rule_following` block, which
it explicitly classifies as **procedural**: an attestation node,
"NOT tensor contractions… properties of the *system/process*, not
scores of a scenario." Gating a substantive verdict on it crosses the
substantive/procedural boundary the standards crosswalk depends on,
and makes the verdict unreachable for any input lacking commitments.

**D3 — silent under-determination.** When no branch fires the bridge
returns `indeterminate`, `confidence=0.5`, rationale "Insufficient
signal to resolve" — without naming a single missing input, and while
non-zero channel values sit unread in the vector it just computed. A
governance layer failing this way looks like it is working.

A boundary bug follows from D2's style of construction and is worth
fixing with it: `prohibited` requires `harm_v < -0.85` while
`severity_score("grave") == 0.85` exactly, so **no `grave` fact can
ever prohibit** — only `catastrophic` can. Strict inequalities against
exact severity weights are latent off-by-one bugs throughout.

## 2. Scope

Replace the verdict function only. Out of scope: the EM-DAG modules,
the tensor construction, the projections' interfaces, and the
cross-framework no-aggregation rule (which stays exactly as is — this
is a *within*-framework contraction).

## 3. Inputs

### 3.1 The canonical `k` axis (9 dimensions)

Per `moralvector_reference.md`; the verdict function is defined over
these and no others:

`0 physical_harm · 1 rights_respect · 2 fairness_equity ·
3 autonomy_respect · 4 privacy_protection · 5 societal_environmental ·
6 virtue_care · 7 legitimacy_trust · 8 epistemic_quality`

**Vocabulary reconciliation required.** The compiler's ten EM modules
do not align with this axis:

| compiler EM | canonical k | note |
|---|---|---|
| harm | 0 physical_harm | |
| rights | 1 rights_respect | |
| fairness | 2 fairness_equity | |
| autonomy | 3 autonomy_respect | |
| — | 4 privacy_protection | **no compiler module** |
| externality | 5 societal_environmental | name mismatch to reconcile |
| care | 6 virtue_care | |
| legitimacy | 7 legitimacy_trust | |
| epistemic | 8 epistemic_quality | |
| fidelity | — | **procedural** (`rule_following`) → attestation |
| repair | — | DEME *moral residue* output, not a `k` dimension |

**[choice]** `fidelity` and `repair` move out of the substantive
contraction: fidelity becomes a procedural attestation, repair becomes
residue reporting. Both remain in the result; neither gates a
substantive verdict.

### 3.2 Three-state availability (the D3 fix)

Every dimension arrives as one of:

- **`available`** — a reader produced a score; carries `value ∈
  [−1,+1]`, `confidence`, `uncertainty`, `reliability_weight`;
- **`neutral`** — a reader ran and found nothing relevant (a genuine
  0.0);
- **`unavailable`** — no reader, or a reader that failed validation.

`unavailable ≠ neutral` is the single most important requirement in
this document. v0 cannot express the distinction, which is why a dark
channel is indistinguishable from a clean one.

### 3.3 Reliability weighting

`reliability_weight = max(0, 2·AUROC − 1)`, the formula already
registered by `moral-spectrum-analyzer`; adopted rather than invented
so the two projects weight evidence identically.

- Encoder-sourced (`xbse`) channels take the weight from their
  registered cross-dataset AUROC. A channel whose encoder failed its
  bar (e.g. `rights_respect`, AUROC 0.509 → 0.018) is reported
  **`unavailable`**, not down-weighted to near-zero, honouring xbse's
  rule that an unvalidated encoder may not be used downstream.
- Structured-reader channels (kind + severity, no learned component)
  take `reliability_weight = 1.0`, with the recorded caveat that this
  asserts confidence in the *fact-extraction* layer, not a measured
  property of a model. **[choice]**

## 4. The contraction

### 4.1 Per-block verdicts, then an explicit rule

Per the architecture, EMs contract `k` from an interest perspective
and "no single EM sees the full tensor." So the function computes a
verdict **per substantive policy block** (safety, autonomy, fairness,
privacy, environment, vulnerable_priority) using that block's
`dimension_weights`, and then contracts block verdicts to one
projection verdict by a **declared, inspectable rule**.

**[choice]** The rule is *severity-dominant with disagreement
surfacing*: the projection verdict is the most restrictive block
verdict, and every block verdict is carried in `framework_specific`.
Rationale: within a consequentialist projection some contraction must
happen, so it should be the one that cannot hide a serious violation
behind an average — and the block detail is preserved so the
contraction is auditable rather than lossy in fact.

### 4.2 Verdict vocabulary

`permitted · permitted_with_residue · requires_human_review ·
prohibited · insufficient_evidence`

`indeterminate` is **retired**. Its replacement,
`insufficient_evidence`, is only reachable *with* a populated
`missing` list (§5.3) — the D3 fix in the type system rather than in a
convention.

### 4.3 Severity and risk tier

The dominant `k`-eigenvalue (the empirically grounded general factor,
eig₁ ≈ 66 % of variance) sets a severity scale driving the EU-AI-Act
risk tier, per architecture §7. Residual spectrum reports structured
content. Severity informs `requires_human_review` escalation
thresholds; it never overrides a constraint (§4.4).

### 4.4 Constraints and overrides

`constraints` from the policy block are hard: a violated constraint
yields `prohibited` regardless of other channels. `override_policy`
governs whether any block verdict may be relaxed, and every applied
override is recorded in the result with its authorizing policy.

## 5. Required properties (the acceptance gates)

Each is a test the implementation must pass; none references LBI-3.

**P1 — no unread substantive channel.** For every substantive
dimension *d*, there exist two inputs differing only in *d* that
produce different verdicts. *Directly falsifies D1;* v0 fails this for
fairness, autonomy, and rights.

**P2 — monotonicity.** Making any dimension strictly worse (more
negative), holding all else equal, never yields a strictly more
permissive verdict. Catches boundary bugs of the `harm < −0.85` kind.

**P3 — boundary reachability.** Every verdict in the vocabulary is
reachable from at least one valid input, and every branch boundary is
tested at the exact severity weights (`0.2 / 0.5 / 0.85 / 1.0`), not
near them. *Directly falsifies the `grave`-can-never-prohibit bug.*

**P4 — unavailable ≠ neutral.** An input with dimension *d*
`unavailable` and one with *d* `neutral` must not produce the same
result; the former must either escalate or report *d* in `missing`.

**P5 — procedural non-gating.** No substantive verdict changes as a
function of a procedural attestation alone. *Directly falsifies D2.*

**P6 — evidence-naming.** `insufficient_evidence` is unreachable with
an empty `missing` list (enforced by the model validator, not by
convention).

**P7 — determinism and trace.** Same input, same verdict, and a
contraction trace sufficient to reconstruct the result — the
architecture's standards crosswalk discharges "explainable &
interpretable" with the trace plus spectral summary plus
`DimensionScore.explanation`.

**P8 — no silent aggregation across frameworks.** Unchanged
cross-framework behaviour: this function contracts *within* the
consequentialist projection only.

## 6. Backward compatibility

v0's branch semantics are preserved where they were reachable, so
existing narrative fixtures should not change verdict except where a
property gate says they must. Any fixture whose verdict *does* change
is listed in the migration note with the property that forced it —
`nazi_attic` and the other committed examples are the regression set.
The bridge keeps its `profile_name` mechanism; v1 ships as
`default_em_dag_v1.0` alongside v0.1, and callers opt in.

## 7. Out of scope, explicitly

Tuning any threshold to make a downstream study's statistic come out a
particular way. The property gates in §5 are all satisfiable without
reference to any dataset, and that is deliberate: the acceptance
criteria for this function must be checkable before anyone knows what
it will say about a real corpus.
