# Phase-5 confirm2 — frozen PolarQuant contestedness rule on a fresh draw

Frozen bundle `prereg-phase5-confirm2-v1` (combined_sha256 bf321818…, signed tag)
locked BEFORE confirm2 activations were extracted. Rule: full PolarQuant angle
vector + loading, LDA (dilemma vs rest) fit on original+confirm dev data, UNION
with the text-conflict channel; flag = (score > τ) OR text_conflict.

## Result (clean, never-seen fresh draw)

| H | criterion | confirm2 | verdict |
|---|---|---|---|
| H1 quiescence | benign ≤ 0.20 | **0.20** (4/20) | PASS (boundary) |
| H2 sensitivity | dilemma ≥ 0.80 | **0.90** (9/10) | PASS |
| H3 specificity | easy-moral ≤ 0.30 | **0.00** (0/10) | PASS |

Benign false alarms: GB01, GB07, GB09, GB18 (mundane texts). Missed dilemma:
GD10 (AI overriding a pilot). Easy-moral false alarms: none.

All three pass — a large improvement over the text_internal_mismatch monitor,
which FAILED H2 (0.60) and H3 (0.40) on confirm-v1. The PolarQuant angular
encoding is what turned H3 and H2 around: contestedness is angular, and the full
angle vector (AUC 0.93/0.94 dilemma-vs-easy-moral) plus the loading gate and the
text-conflict union clears all three on out-of-sample data.

## Honest caveats (do NOT overclaim)

1. **Scenario-authoring bias (the main one).** The confirm2 dilemmas (GD01–GD10)
   were written *by the same author who had just characterized the
   conflict-based mechanism*, with explicit tension framing ("violates his
   autonomy; refusing lets a child die"). That may make them unusually easy for a
   conflict detector, inflating H2. The statistical protocol is clean (rule
   frozen before confirm2 was seen), but the *stimuli* are not
   mechanism-blind. A decisive confirmation needs dilemmas authored blind to the
   detector (independent annotators, or a held-out slice of a standard moral-
   dilemma corpus).
2. **H1 is at the boundary (0.20).** Four mundane benign texts flag — the rule is
   right at the pass/fail line on quiescence, not comfortably inside it.
3. **Small n** (20/10/10). Wide confidence intervals; single missed/extra
   scenarios move a rate by 0.10.

## Bottom line

The PolarQuant contestedness encoding, frozen and tested on a fresh draw, passes
H1/H2/H3 (0.20/0.90/0.00) — strong methodological progress and a real vindication
that contestedness lives in the *angle*. The remaining threat to validity is that
the fresh dilemmas were author-written with mechanism knowledge; the next clean
step is a mechanism-blind dilemma set (and tightening H1's benign margin).
