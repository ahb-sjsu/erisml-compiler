# Phase-5 mechanism-blind replication — the PolarQuant contestedness result does NOT survive

**Status: NULL (decisive).** The confirm2 pass (H1/H2/H3 = 0.20/0.90/0.00) does **not** replicate on
stimuli authored *blind* to the layer-8 conflict mechanism. This confirms the #1 caveat the
confirm2 RESULTS.md itself flagged: *"the fresh dilemmas were author-written with mechanism
knowledge."*

## Why this test
`phase5_confirm2/RESULTS.md` was explicit that the decisive next step is **mechanism-blind
dilemmas** — stimuli written by people with no knowledge of the conflict detector. We ran two,
through the identical pipeline (Qwen2.5-7B activations → calibrated layer probes → PolarQuant
angle-from-consensus), with the classifier/threshold **frozen on the original+confirm dev data**.

## Test 1 — Scruples (real AITA, n=20)
Contested (D) = balanced AITA vote split; clear (E) = lopsided; benign (B) = mundane. Every config
(incl. the frozen `layer8/cal7/angles+r` rule) gives **D ≈ E** — no separation of contested from
clear moral scenarios once stimuli aren't author-framed:

| config | B | D | E |
|---|---|---|---|
| FROZEN L8/cal7/angles+r | 0.30 | 0.40 | 0.50 |
| L4/all10/angles+r | 0.15 | 0.90 | 0.90 |  ← flags *all* moral content, not contest |

## Test 2 — Moral Machine (n=628, continuous graded label)
Contestedness = entropy of choices across a 15-agent panel on shared benchmark scenarios
(mechanism-blind; agents have no knowledge of layer-8 geometry), stratified across the full range.
**Spearman(conflict-angle, contestedness) ≈ 0 at every layer** (|ρ| < 0.09; layer 8: −0.086).

## Caveat closed — it is a real null, not low power
| | layer 8 |
|---|---|
| conflict-angle variance on MM | std 0.105, range [0.42, 1.07] — **has variance** |
| contestedness variance | std 0.254, range [0.00, 1.00] — **full spread** |
| DEV angle AUC(dilemma > easy-moral) | **0.760** — the effect, on author-framed stimuli |
| MM angle AUC(high > low contestedness) | **0.463** — mechanism-blind: chance |

The angle varies, contestedness varies, the test has power — and the *same feature* that separates
dilemmas at 0.76–0.84 on author-framed dev collapses to chance on mechanism-blind stimuli.

## Conclusion
**"Contestedness is angular" is an author-framing artifact.** A conflict detector latched onto the
explicit tension framing of the hand-written dilemmas ("violates his autonomy; refusing lets a child
die"), not onto contestedness in the activation geometry. The methodology (frozen rule, signed tag,
fresh draw) was sound; the *stimuli* were not mechanism-blind, exactly as flagged. The confirm2
numbers should not stand unqualified.

**Scope (do not over-refute):** this refutes the *generality* of the angular-contestedness claim for
moral monitoring. It says nothing about PolarQuant's other, independently-validated uses (vector
quantization; angle-only geodesic preservation). Those stand.

## Artifacts
`experiments/eval_scruples.py`, `experiments/eval_moralmachine.py`, `experiments/angle_variance.py`,
`experiments/fuzz_polarquant.py`; caps under `experiments/{scruples,mm}_caps` (gitignored).
