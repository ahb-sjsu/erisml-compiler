# Release planning 02 — Real ρ estimation behind an experimental flag

**Status:** core implementation landed (ρ-1, ρ-2, ρ-3). CLI subcommand,
semantic validation gate, held-out residual baselines, and the
equivariance benchmark suite remain future work (see Milestones).
**Owner:** TBD.
**Estimated effort:** ~10 days total; ~3 days landed in this commit.
**Predecessor:** [release-planning-01](./release-planning-01.txt) item 3.

## Landed in v0

- `src/erisml_compiler/delta/transforms.py` — `Transform`,
  `TransformFamily`, `TransformRegistry`, `default_registry()`, plus
  4 bundled surface-family transforms.
- `src/erisml_compiler/delta/rho_estimation.py` — `RhoEstimate`,
  `fit_rho_procrustes`, `fit_rho_lstsq`, `compute_residuals`,
  `equivariance_residual`, top-level `fit_rho`.
- `src/erisml_compiler/delta/equivariance.py` extended with
  `rho_map` + `residual_threshold` kwargs; `LayerEquivarianceResult`
  gains an optional `rho_residual` field.
- `FailureMode.RHO_NON_ORTHOGONAL` added.
- `tests/test_rho_estimation.py` — 18 tests covering the math.

Future work (per the Milestones table below): ρ-4 CLI fit-rho
subcommand requires real activation captures + probe checkpoints;
ρ-5 semantic-rewrite validation gate needs the LLM adapter; ρ-6/7
need the curated paraphrase corpus.

## What this is

Today the delta-lens equivariance check (`src/erisml_compiler/delta/equivariance.py`)
tests `h_ℓ(g · x) ≈ ρ_ℓ(g) · h_ℓ(x)` with **ρ_ℓ(g) = identity** — i.e. we
assume the per-layer probe output should be *invariant* under any
meaning-preserving rewrite. The shipped default rewrites are surface-form
only (whitespace, case, trailing period). Anything richer would falsify
the identity assumption on a system that does the right thing — paraphrase
should shift activations but the *moral* content shouldn't.

The full BIP framework (sqnd-probe v10.16.9) makes ρ_ℓ(g) a learned
linear (or affine) map, estimated per-layer from pairs of activations
under known group actions. With a real ρ:

- "paraphrase produced different activations" stops being a false alarm
- "two paraphrases that *should* live in the same equivalence class
  but don't" becomes a measurable signal
- ρ_ℓ becomes part of the probe's calibration artifact and gets hashed
  alongside the checkpoint

## Non-goals

- **Not** a deep-learning training stack. ρ is a small linear map per
  layer; ordinary least squares (or orthogonal Procrustes) suffices.
- **Not** a paraphrase generator. The corpus of transformation pairs is
  externally curated.
- **Not** a replacement for identity-ρ. Identity remains the default;
  real ρ lives behind `--equivariance-rho real` (default `identity`).

## Concept

For each (layer ℓ, group element g), we have N activation pairs
`{(h_ℓ(x_i), h_ℓ(g · x_i))}_{i=1..N}` from a corpus of
transformation pairs. ρ_ℓ(g) is the best-fit map:

```
ρ_ℓ(g) = argmin_R  Σ ||R · h_ℓ(x_i) − h_ℓ(g · x_i)||²
```

Two reasonable choices for the family of R:

1. **Unconstrained linear**: R ∈ ℝ^{D×D}. Closed-form via pseudoinverse.
   Most expressive; risks overfitting at small N.
2. **Orthogonal Procrustes**: R ∈ O(D). Closed-form via SVD. Cleaner
   physical interpretation (rotation/reflection only, preserves norms),
   matches the assumption that semantics-preserving rewrites should
   not stretch or compress activation magnitude.

Default to orthogonal Procrustes; allow unconstrained linear as a
diagnostic mode.

The equivariance test becomes:

```
residual_ℓ(g, x) = || ρ_ℓ(g) · h_ℓ(x) − h_ℓ(g · x) || / || h_ℓ(g · x) ||
```

Per-transform thresholds replace the single `pooled_cosine_threshold`:
each g carries an expected residual range learned at calibration time.

## Architecture sketch

### `delta/transforms/` (new subpackage)

```python
@dataclass(frozen=True)
class Transform:
    """A labelled, parameterised semantics-preserving rewrite."""

    name: str
    family: TransformFamily  # surface | paraphrase | role_swap | unit_change | order_perm
    fn: Callable[[str, dict | None], str]   # (text, params) -> rewritten text
    expected_rho_class: Literal["identity", "orthogonal", "linear"]
    validation_hook: Callable[[str, str], ValidationResult] | None
    cost: float = 1.0

class TransformRegistry:
    def register(self, transform: Transform) -> None: ...
    def get(self, name: str) -> Transform: ...
    def filter(self, *, family: TransformFamily | None = None) -> list[Transform]: ...
```

- `family=surface` are the current shipped rewrites (identity ρ).
- `family=paraphrase` covers LLM-paraphrased pairs; ρ expected
  orthogonal.
- `family=role_swap` swaps stakeholder labels (alice↔bob); ρ expected
  to be a *permutation* of activation subspaces.
- `validation_hook(orig, rewritten)` is the **semantic gate**: a
  function that returns whether the rewrite actually preserved moral
  content. Required for non-surface families. Implementation options
  range from "verify the IR's canonical_form matches under both texts"
  to "second-model agreement on a 4-question semantic-preservation
  rubric." See **Open question A** below.

### `delta/rho_estimation.py` (new module)

```python
@dataclass(frozen=True)
class RhoEstimate:
    transform_name: str
    layer_index: int
    R: Any                  # (D, D) numpy or torch
    family: TransformFamily
    n_pairs: int
    residual_mean: float    # mean normalised residual on the fit set
    residual_p95: float     # 95th-percentile residual on the fit set
    corpus_hash: str        # SHA-256 of the (transform_name, pair_ids) set
    method: Literal["procrustes", "lstsq"]

def fit_rho(
    source: ActivationSource,
    probes: dict[int, ActivationProbe],
    transform: Transform,
    pair_corpus: Iterable[tuple[str, str]],
    *,
    method: str = "procrustes",
) -> dict[int, RhoEstimate]:
    """Fit one RhoEstimate per layer in `probes` against the
    (original, rewritten) text pairs."""
```

`fit_rho` does the activation captures, builds the per-layer (H, H_g)
matrices, fits R, computes residual statistics, and returns the
per-layer map. The result is JSON-serialisable (with the R tensor
encoded as base64 or a separate .npz sidecar) and attaches to a probe
checkpoint as `rho_estimates[transform_name][layer_idx]`.

### `delta/equivariance.py` (extended)

```python
def check_equivariance(
    source, probes, text, *,
    rewrites: Sequence[Rewrite] = DEFAULT_REWRITES,
    rho_map: dict[str, dict[int, RhoEstimate]] | None = None,
    pooled_cosine_threshold: float = 0.95,
    probe_cosine_threshold: float = 0.9,
    residual_threshold: float = None,    # if None, use per-transform p95
) -> EquivarianceReport:
    """When `rho_map` is None (default), behaves as today: identity check.
    When supplied, each rewrite's `residual_ℓ` is computed against the
    fit ρ and compared to the per-transform threshold (or `residual_threshold`
    override)."""
```

`EquivarianceReport` gains a `per_layer_per_rewrite[i].rho_residual`
field. The "failed_layers" set is populated by *either* the identity
sims dipping below threshold (surface family) *or* the rho_residual
exceeding the per-transform p95 (semantic families).

### CLI

```
eris-compile fit-rho \
    --probe-checkpoint ./calibrated.pt \
    --pair-corpus ./paraphrase_pairs.jsonl \
    --transform paraphrase_llm \
    --method procrustes \
    --out ./rho_estimates.json

eris-compile delta old.json trace.json \
    --equivariance-rho real \
    --rho-estimates ./rho_estimates.json
```

The probe checkpoint loader (calibration.train.load_checkpoint)
automatically picks up a sibling `*_rho.json` if it exists.

## Open questions

**A. Semantic-rewrite validation gate.** A paraphrase pair where the
paraphrase actually changes meaning poisons ρ. Two viable gates:

  1. **IR-equivalence gate**: compile both texts, compare their
     canonical_form + stakeholder graph + commitment registry. Cheap
     but circular (uses the compiler to validate the compiler).
  2. **Second-model gate**: an external LLM (NRP `gpt-oss`) scores a
     4-question rubric ("same stakeholders? same commitments? same
     authority structure? same outcome distribution?"). More
     defensible but adds cost and a second trust boundary.

Recommendation: ship both; gate paraphrases by *agreement* between
the two. A pair survives only if the IR-equivalence gate AND the
second-model gate accept it.

**B. Per-transform residual baseline.** What residual is "normal" for
a given transform? Need to fit ρ on a held-out validation set and
publish the empirical residual distribution per transform. Without
this the residual threshold is a magic number.

**C. ρ for compositions.** If g₁ and g₂ are both registered (e.g.
paraphrase then role-swap), does ρ_ℓ(g₁ ∘ g₂) ≈ ρ_ℓ(g₂) · ρ_ℓ(g₁)?
This is the actual group-action property. Worth testing as a
diagnostic but not load-bearing for v0.

**D. Calibration corpus identity.** ρ depends on which pair-corpus
was used. The `corpus_hash` field handles cataloguing but doesn't
prevent corpus drift. Two probes calibrated on different
paraphrase corpora produce non-comparable residuals. Document this
loudly.

## Milestones

| Phase | Deliverable | Effort |
|---|---|---|
| ρ-1 | `delta/transforms/` package + `Transform` + `TransformRegistry` | 1 day |
| ρ-2 | `delta/rho_estimation.fit_rho` with Procrustes + LSTSQ | 2 days |
| ρ-3 | `check_equivariance` extended for real-ρ mode | 1 day |
| ρ-4 | CLI `fit-rho` subcommand + checkpoint integration | 1 day |
| ρ-5 | Semantic-rewrite validation gate (both A.1 and A.2) | 2 days |
| ρ-6 | Held-out residual baselines on a small (~200) paraphrase corpus | 1–2 days |
| ρ-7 | Equivariance benchmark suite (10 transforms × 50 pairs) | 2 days |
| ρ-8 | Docs + design-note write-up | 0.5 days |

Total: **~10 working days**, plus paraphrase-corpus curation effort
(see [release-planning-03](./release-planning-03-moraltensor-bench.md)
which covers shared corpus work).

## Failure modes worth surfacing

- **ρ fits perfectly on the training pairs but generalises poorly.**
  Mitigation: report the held-out residual distribution alongside the
  fit, not just the training residual.
- **ρ is near-identity even on the paraphrase corpus.** This is the
  *honest finding* the framework is supposed to surface — it means the
  probe genuinely sees those rewrites as semantically identical, and
  the I-EIP machinery should report that. Not a bug.
- **ρ is far from orthogonal.** Diagnostic: the probe is picking up
  surface form (paraphrase blew up activation magnitude). Surfaced as
  a new failure mode `rho_non_orthogonal` in `delta/failure_modes.py`.

## Suggested package extras

```toml
[project.optional-dependencies]
rho = [
    "numpy>=1.24",
    "scipy>=1.10",       # for orthogonal Procrustes via scipy.linalg
]
```

`eris-compile fit-rho` requires the `rho` extra; without it the
command errors with a clear install message.
