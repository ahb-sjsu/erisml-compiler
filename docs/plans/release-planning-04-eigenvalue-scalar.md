# Release planning 04 — Eigenvalue-based moral-magnitude scalar

**Status:** design + implementation in this commit.
**Estimated effort:** ~1 working day.
**Predecessor:** none formal; closes a gap noted in conversation
after [release-planning-03](./release-planning-03-moraltensor-bench.md).

## The problem

The compiler today emits a deliberately structure-preserving IR. When a
downstream consumer asks "give me one number for how morally loaded
this scenario is", the existing surface is silent on the question:

- `deme_verdict` is categorical, not a scalar magnitude
- `fairness_metrics.gini_harm` is a *distributional* scalar (inequality
  of one row), not a magnitude
- `fairness_metrics.worst_off_harm_value` is a point estimate of one
  extreme, not a magnitude
- `strategic_analysis.shapley_values` is per-stakeholder, not a single
  number
- `delta.divergence` is about text-vs-activation disagreement, not
  about the scenario's intrinsic moral content

The closest thing to a "moral magnitude" is `||values||` over the
whole tensor, but raw Frobenius norm has no geometric meaning — it's
basis-dependent and treats all dimensions equally, which is the very
thing the whole framework rejects.

## The principled answer

The Geometric Series has been consistent on this point: when a
tensorial structure exists and you want to extract a magnitude, the
right scalar is an **eigenvalue**.

- `geometric-aesthetics` v3 calls them perceptual eigenvalues —
  "the importance weights of perception" indexing the perceiver's
  covariance.
- `geometric-economics` uses the eigenspectrum of the Mahalanobis
  metric to characterise decision manifolds.
- `sqnd-probe` measures D4 gauge structure via eigendecomposition
  of activation pair operators.

Applying the same machinery here is straightforward. For the rank-2
moral tensor `T ∈ ℝ^{9 × n}` over (dimension, stakeholder), two
natural eigenproblems exist, with the same non-zero eigenvalues but
eigenvectors in different spaces (SVD identity):

  - **`T · Tᵀ ∈ ℝ^{9×9}`** — covariance across dimensions.
    Eigenvalues = principal moral-stress magnitudes. Eigenvector for
    λ₁ = which combination of moral dimensions dominates the
    scenario.
  - **`Tᵀ · T ∈ ℝ^{n×n}`** — covariance across stakeholders.
    Eigenvalues = stakeholder-loading magnitudes. Eigenvector for
    λ₁ = which combination of stakeholders is most affected.

We compute **two** eigendecompositions — uncentered and centered:

  - **Uncentered** (`T · Tᵀ` directly): "moral magnitude". λ₁ large
    means the scenario is morally loaded; eigenvalue spread tells
    you whether it's loaded along one axis (clean case) or many
    (tragic conflict).
  - **Centered** (subtract mean across stakeholders first): "moral
    disagreement". λ₁ large means stakeholders are in conflict;
    eigenvector tells you the direction of maximum stakeholder
    disagreement.

A scenario can be high-magnitude/low-conflict (everyone affected
similarly, no conflict — pure tragedy) or high-magnitude/high-conflict
(nazi-attic style — large total weight, plus stakeholders pushed in
opposite directions).

## What `MoralTensorV3.spectral` will carry

For the full rank-N shape, see the "Higher-rank tensors" section
below. The headline structure:

```python
class AxisSpectrum(BaseModel):
    axis_name: str
    eigenvalues: list[float]
    principal_axis: list[float]
    effective_rank: float
    axis_labels: list[str] | None


class SpectralSummary(BaseModel):
    # Global magnitude (rank-independent)
    total_stress: float                # ||T||_F
    total_stress_squared: float        # = sum of any mode's eigenvalues

    # Headline scalars — always from the dimension (k) axis
    principal_stress: float            # λ₁ of mode-k unfolding
    principal_concentration: float     # λ₁ / total_stress²
    stress_spread: float               # (λ₁ − λ₂) / λ₁ when λ₁ > 0
    effective_moral_rank: float        # participation ratio
    principal_axis: list[float]        # length-9 unit eigenvector

    # Centered counterparts (subtract mean across non-k axes)
    principal_conflict: float
    principal_conflict_axis: list[float]   # length-9

    # Per-axis spectra (one entry per tensor axis)
    per_axis: list[AxisSpectrum]
```

For rank-2 tensors `per_axis` has 2 entries (k, n). For rank-6 it
has 6. The headline scalars are always defined (always from the
dimension axis) so consumers can compare across rank.

All these are derived deterministically from `MoralTensorV3.values`
at tensor-build time. They're not new information — they're the
*right* projection of information that's already there. Storing
them on the IR saves consumers from re-computing per query.

## Higher-rank tensors (ranks 3–6)

For rank-2 the picture is clean — two eigendecompositions, dimension-
and stakeholder-side. For rank ≥ 3 (temporal, coalition, MC axes),
the right generalisation is **one mode-unfolding eigendecomposition
per axis**, plus one global magnitude scalar that doesn't depend on
the axis.

### Mode-n unfolding

For tensor `T ∈ ℝ^{d_1 × d_2 × ... × d_R}` and axis `m`, the
mode-m unfolding `T_(m) ∈ ℝ^{d_m × (∏_{i≠m} d_i)}` flattens every
other axis into the column space. Then `T_(m) · T_(m)ᵀ ∈ ℝ^{d_m × d_m}`
is the mode-m second-moment matrix.

For a rank-6 moral tensor `T ∈ ℝ^{9 × n × τ × a × c × s}` we get
six such eigendecompositions:

| Mode | Shape of `T_(m)·T_(m)ᵀ` | Eigenvalues mean | Eigenvector e₁ tells you |
|---|---|---|---|
| k (dims)         | 9×9             | principal moral-dimension stresses | which combo of moral dimensions dominates (this is **the** canonical scalar's source) |
| n (stakeholders) | n×n             | principal stakeholder loadings     | which combo of stakeholders is most affected |
| τ (time)         | τ×τ             | principal moments                   | which time step carries most moral weight |
| a (actions)      | a×a             | principal action-spread             | which action under consideration is most loaded |
| c (coalitions)   | c×c             | principal coalition stresses        | which coalition shows the strongest moral signature |
| s (MC samples)   | s×s             | sample variance                     | typically near-uniform; large λ₁ means MC noise dominates the signal — a calibration warning |

### Frobenius identity — one global magnitude

A clean property keeps the framework grounded: for any tensor,

```
sum of mode-m eigenvalues  =  ||T||_F²
```

regardless of which axis `m` you unfolded along. So:

- **`total_stress = ||T||_F`** is the intrinsic magnitude, basis-
  independent, axis-independent. One number per tensor.
- **`principal_stress`** is still the largest eigenvalue from the
  dimension-axis unfolding — but its meaning is now precise:
  `principal_stress / total_stress²` is the *fraction of the
  scenario's total moral magnitude that lives on the principal
  moral-dimension axis*.

### The headline scalars at rank ≥ 3

`SpectralSummary` becomes:

```python
class AxisSpectrum(BaseModel):
    axis_name: str                     # "k" | "n" | "tau" | "a" | "c" | "s"
    eigenvalues: list[float]           # descending, length = axis dimension
    principal_axis: list[float]        # unit eigenvector for λ₁
    effective_rank: float              # participation ratio
    axis_labels: list[str] | None      # labels for interpretation


class SpectralSummary(BaseModel):
    # Global magnitude (axis-independent)
    total_stress: float                # ||T||_F
    total_stress_squared: float        # ||T||_F²; equal to sum of any mode's eigenvalues

    # Headline scalars — always derived from the dimension axis (mode-k)
    principal_stress: float            # largest λ from mode-k unfolding
    principal_concentration: float     # λ₁ / total_stress²; in [0, 1]
    stress_spread: float               # (λ₁ − λ₂) / λ₁ when λ₁ > 0 else 0
    effective_moral_rank: float        # participation ratio of mode-k eigenvalues

    # Centered counterparts (subtract mean across all non-k axes)
    principal_conflict: float
    principal_conflict_axis: list[float]   # length 9

    # Per-axis spectrum — one entry per axis in the tensor
    per_axis: list[AxisSpectrum]
```

For rank-2 tensors `per_axis` has 2 entries (k and n); for rank-6 it
has 6. The headline scalars (`principal_stress`, `total_stress`,
`stress_spread`, `effective_moral_rank`, `principal_conflict`) are
*always defined* — they always come from the dimension axis — so
consumers can compare across rank.

### Interpretive examples

A scenario where:

- `principal_concentration ≈ 1.0` → moral content lives essentially
  along one dimension. Clean case. The eigenvector tells you which.
- `principal_concentration ≈ 1/9` → moral content is spread evenly
  across all 9 dimensions. Maximally entangled.
- `per_axis[k].effective_rank` is high → many dimensions matter.
- `per_axis[τ].principal_axis` peaked at the last time index → the
  decision point is the moment of moral concentration.
- `per_axis[c].principal_axis` peaked on one coalition → that
  coalition's particular choice carries the most weight.
- `per_axis[s].effective_rank` near 1 → MC samples are concentrated
  near the unperturbed result (good); near `n_samples` → MC noise is
  dominating (calibration warning).

### What we don't do

- **No Tucker decomposition / HOSVD**. The per-axis spectra are
  the *marginals* of HOSVD's mode-m singular values, computed
  directly without forming the core tensor. Full HOSVD would be the
  natural Phase-2 extension if cross-axis interaction matters.
- **No CP / PARAFAC decomposition**. That would let us factor `T`
  into rank-r outer-product components; useful for compression
  questions but not for the moral-magnitude scalar.
- **No spectral comparison between scenarios**. The bench-side work
  (release-planning-03) will likely want this — measure the cosine
  similarity of principal moral-dimension axes between paraphrased
  scenarios, etc. — but it's distinct from constructing the per-IR
  spectrum, which is what this milestone delivers.

### Cost

Mode-k unfolding of the rank-6 tensor is `9 × (n·τ·a·c·s)`. For the
defaults the bench currently uses (n≤8, τ≤4, a≤4, c≤8, s≤16), that
caps at `9 × 16384`, eigendecomposed at sub-millisecond on CPU. All
six modes together are still sub-millisecond. Stored size: each
`AxisSpectrum` is `O(d² + d)` floats per axis; rank-6 worst-case
adds ~10 KB to the IR. Acceptable.

## Signed values

`MoralTensorV3.values` are in `[-1, 1]` (signed) — a stakeholder can
*benefit* from a moral dimension (positive) or *bear cost* (negative).
For the uncentered eigendecomp we use the second moment `T·Tᵀ`
directly without centering, so positive and negative contributions
both add to the magnitude. That matches the reading "how
morally-loaded is this scenario regardless of direction" — a heavy
benefit is just as morally significant as a heavy harm.

The centered eigendecomp does subtract the per-dimension mean
across stakeholders, so it surfaces directional disagreement only.
This is the right pair: magnitude (uncentered) + disagreement
(centered).

## Tests

- **Zero tensor** → all eigenvalues 0, `principal_axis = e₁` by
  convention, `effective_rank = 0`.
- **Single-dimension tensor** (only physical_harm row nonzero) →
  `λ₁ = sum-of-squares of that row`, all other λ = 0,
  `principal_axis ≈ unit vector along physical_harm`,
  `effective_rank ≈ 1`.
- **Uniform tensor** (all cells equal to 1.0) → `effective_rank` is
  high (multi-axis loading), `stress_spread` is low (no one direction
  dominates).
- **nazi_attic golden numbers** → recorded once empirically and
  asserted; regression-protect them.
- **JSON roundtrip** of the spectral block.
- **Rank-3 unfolding consistency**: spectral summary of a rank-3
  tensor must equal that of the matching rank-2 slice when the τ
  axis has length 1.

## CLI

No new flags. The spectral summary is always computed when a tensor
is produced. Surfaced in the existing `compile` output:

```
[+] DEME V3 tensor: rank=2 shape=(9, 4) axes=('k', 'n')
[+] Principal moral stress: 0.834  (axis: virtue_care=0.51 legitimacy_trust=0.43 ...)
[+] Principal conflict:     0.418
[+] Effective rank:         3.21
[+] IR hash: ...
```

## Non-goals

- Not Tucker decomposition / HOSVD (multi-rank generalised SVD).
- Not training a parametric "moral scalar" that mixes eigenvalues
  with other features.
- Not a replacement for the verdict or for the per-party verdicts.
  The eigenvalue is *additional* information, not a contracted
  decision.

## Risks

- **Eigendecomposition of a 9×9 matrix is cheap** but rank-6 mode-1
  unfolding can be `9 × 7680` (4 parties × 3 times × 2 actions × 4
  coalitions × 4 samples × 9 dims). Still trivially fast (sub-ms).
- **Sign convention on eigenvectors** is arbitrary; we choose the
  sign that makes the eigenvector's largest-magnitude entry positive.
  Deterministic and matches the geometric-aesthetics convention.
- **Numerical noise on near-zero eigenvalues** could make
  `stress_spread` jittery. We define it as 0 when `λ₁ < 1e-9`.
