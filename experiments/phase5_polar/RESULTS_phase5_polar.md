# Phase-5 H2/H3 via PolarQuant moral encoding — exploratory result

Decompose the calibrated layer-8 moral vector into **radius** (loading) +
**angle** (conflict from the consensus/"good" axis) using the exact PolarQuant
transform (`monitor/polar_encoding.py`, vendored from turboquant-pro;
Han et al. arXiv:2502.02617). Replaces the sparse text_internal_mismatch channel
with a purely activation-side geometric one. Tuned on the ORIGINAL 40 scenarios,
evaluated HELD-OUT on the frozen confirm set.

## Result

- **Contestedness is angular, and it replicates.** Conflict-angle AUC
  (dilemma > easy-moral) = **0.73 (original), 0.80 (confirm)**. Magnitude does
  NOT do this (dilemma and easy-moral are equally loaded; earlier `||v_perp||`
  was magnitude-contaminated and put E above D). The signal lives in the angle —
  exactly the PolarQuant / GDT thesis (direction carries the geometry).

- **Geometric flag rule** (loading > τ_r AND conflict > τ_θ, tuned on original):

  | | H1 benign | H2 dilemma | H3 easy-moral |
  |---|---|---|---|
  | ORIGINAL | 0.20 | 0.60 | 0.20 |
  | CONFIRM (held-out) | **0.05 PASS** | 0.60 **FAIL** | **0.10 PASS** |
  | (criterion) | ≤0.20 | ≥0.80 | ≤0.30 |

## Interpretation (honest)

- **H1 and H3 are solved by the polar encoding.** The loading gate keeps benign
  quiescent (0.05), and the angle suppresses easy-moral to **0.10 held-out** —
  a large, replicating improvement over the text-mismatch channel that FAILED H3
  at 0.40–0.50 in both earlier runs. The user's polarquant intuition holds: the
  contestedness the monitor could not see is angular.
- **H2 is not solved by the angle alone.** At thresholds that keep H1/H3 clean,
  the geometric rule catches only ~0.60 of dilemmas (AUC 0.73–0.80 is real but
  not strong enough for 0.80 sensitivity at high specificity). The layer-8 moral
  vector's angular conflict is present but weak for the harder dilemmas.

## Next

Sensitivity (H2) likely needs a second channel OR a stronger angular feature:
- **Combine channels:** flag if (text lens takes a dilemma-stance) OR (angular
  conflict) — union recovers dilemmas the angle misses.
- **Richer angular features:** use the full PolarQuant angle vector (not just the
  consensus colatitude), a per-dimension consensus axis, or a different readout
  layer / the raw logits (more dynamic range than tanh).
- Then **freeze and test on a fresh third draw** (the confirm set has now been
  used to select this rule, so it is no longer strictly held-out for H2 tuning).

Bottom line: PolarQuant encoding converts H3 from a failure into a robust pass
and gives a real angular dilemma signal; H2 remains the open target.
