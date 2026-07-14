# Phase-5 confirmatory replication — results (FROZEN)

Frozen bundle `prereg-phase5-confirm-v1` (combined_sha256 0fc95bd9…, signed tag)
was locked **before** any confirmatory activation was extracted. Fresh
out-of-sample scenarios (`scenarios_phase5_confirm.json`: 20 new benign, 10 new
easy-moral, 10 dilemmas = 3 Phase-4 anchors + 7 new). Same calibrated layer-8
probe, same C1–C4, same analyze_phase5 thresholds.

## Frozen result vs the exploratory run

| H | criterion | exploratory | **confirmatory (frozen)** | |
|---|---|---|---|---|
| H1 quiescence | benign ≤ 0.20 | 0.00 | **0.00** | PASS (robust) |
| H2 sensitivity | dilemma ≥ 0.80 | 0.80 | **0.60** | **FAIL** (did not replicate) |
| H3 specificity | easy-moral ≤ 0.30 | 0.50 | **0.40** | FAIL (both runs) |
| H4 equivariance (post-C1) | == 0 | 0 | **0** | PASS (robust) |
| H5 separation | ≥ 0.15, all B<0.35 | 0.95 | **0.678** | PASS |

Divergence: B 0/0/0 · D 0/0.68/1.0 (6>0.35) · E 0/0/1.0 (4>0.35). Mismatch 10/40; equiv failures 0/40.

## Verdict (per the prereg's own gating rule)

**Gating hypothesis H2 failed out-of-sample → the instrument returns to
calibration.** This is the informative outcome the prereg (§8) anticipated, and
it is the whole reason the frozen replication mattered: the exploratory run's
H2 = 0.80 was borderline and did **not** survive a fresh scenario draw (0.60).

## What is robust vs not

- **Robust:** benign quiescence (H1 = 0.00 both runs) and the equivariance fix
  (H4 = 0 both runs — C1 dropping the layer-27 readout artifact holds up).
- **Not robust:** dilemma sensitivity (H2 0.80 → 0.60). The delta-lens
  `text_internal_mismatch` under-fires on fresh dilemmas — consistent with the
  known limitations: the RULES text lens is sparse (often takes no directional
  stance on a calibrated dim, so C3×C4 leaves nothing to compare), and
  calibrating the activation lens to agree with moral judgments *reduces*
  text-vs-activation mismatch precisely on the dilemmas we want it to catch.
- **Confirmed limitation:** it does not isolate contestedness (H3 fails both runs).

## Honest bottom line

Post-calibration, `text_internal_mismatch` is **not** a reliable dilemma
detector out-of-sample. The monitor cleanly separates benign from
morally-loaded and has a sound equivariance channel, but the discrimination
test as framed (text-internal mismatch) does not hold up. Next: return to
calibration / redesign the sensitivity signal (e.g., use the activation lens's
own moral-valence magnitude rather than text-vs-activation disagreement), then
re-freeze and re-test. Failing cleanly here is cheaper than shipping a monitor
that looked good on one non-frozen run.
