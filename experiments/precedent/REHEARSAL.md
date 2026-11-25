# Rehearsal record: the precedent experiment on planted truths

**All results here are simulated.** Truths were planted on real LaBSE embeddings (27,982 Scruples
texts, snapshot `836121a0`, `/archive/experiments/learned-coords/emb/labse.npz` on Atlas) with the
labels overwritten. No human judgement was read. Run on Atlas GPU 1, 2026-11-25, in a CUDA-capable
Python environment (the system Python's torch build does not see the GPU).

Code: `precedent_arms.py` (arms and metric), `grade_precedent.py` (bars), `rehearse.py` (worlds and
self-tests). Outputs: `rehearsal/round1/`, `rehearsal/round2/`.

## What was asked of the chain

Six predictors ("arms") are fitted on 3,000 training cases and scored on held-out cases:

| Arm | What it uses |
|---|---|
| G | the mean judgement of the query's stratum (gates, no retrieval) |
| A | the 10 nearest cases by cosine (retrieval, no gates) |
| M | the 10 nearest under a learned metric (no gates) |
| B | the 10 nearest by cosine inside the query's stratum |
| Cp | as B, under one learned metric shared by all strata |
| Cl | as B, under a per-stratum metric |

Three hypotheses are graded, each as a paired difference in squared error beyond 2.5 standard errors:
H1 (B beats A on gate flips), H2 (Cp beats B on ordinary held-out cases), H3 (Cl beats Cp).
The triple of verdicts is read as one of five conclusions. The rehearsal checks that the reading
matches what was planted.

## Round 2 (final for today): 45 runs

| Planted truth | Runs | Read back correctly | H1 passes | H2 passes | H3 passes |
|---|---|---|---|---|---|
| noise only | 20 | 20 | 0 | 0 | 0 |
| one smooth trade-off, gates irrelevant | 5 | 5 | 0 | 5 | 0 |
| a constant per stratum | 5 | 5 | 5 | 0 | 0 |
| constant per stratum + one shared trade-off | 5 | 5 | 5 | 5 | 0 |
| constant per stratum + per-stratum trade-offs, weak (3 coordinates each from a shared 16) | 5 | 0 | 5 | 5 | 0 |
| constant per stratum + per-stratum trade-offs, strong (5 disjoint coordinates each) | 5 | 2 | 5 | 5 | 2 |

**What this establishes.**

- **H1 and H2 are gradeable.** Each passed in every run where it was planted (20 of 20 each) and in
  no run where it was not (0 of 25 each).
- **No false pass on noise** in 20 runs, for any hypothesis.
- **H3 is not gradeable as built.** It never passed falsely (0 of 35), but it detected a strong planted
  per-stratum metric in only 2 of 5 runs and a weak one in 0 of 5. A failed H3 on real data would say
  nothing about whether trade-offs differ by stratum.

**Self-tests, all passed.**

| Test | Result |
|---|---|
| S1 a sampling unit on both sides of the split is refused, and the refusal names the split | fired |
| S2 the grader refuses an undeclared data source, and the refusal names the source | fired |
| S3 renaming the strata changes no arm's error | relative change 0.0 |
| S4 negating the judgements changes no arm's squared error | relative change 0.0 |
| S5 no hypothesis passes on noise more than twice in 20 | 0, 0, 0, 0 |

## Round 1, and what changed

Round 1 fitted each stratum's metric on that stratum alone. H3 passed in 0 of 5 runs of the weak
per-stratum world (z from -0.7 to +1.5). The errors of the next run showed why: the shared metric
already reached 0.361 against a floor near 0.275, and the per-stratum fit reached 0.355.

Change for round 2: the per-stratum metric is the shared metric plus a penalised per-stratum part,
started at the shared one (Parameswaran and Weinberger 2010), penalty weight 0.01. A stronger planted
world was added to find where detection begins. Result above: better, still short.

The round 1 full run was interrupted by me (I moved its output directory while it was still writing),
so round 1 has per-run logs and no summary file.

## Three observations that matter for the design

1. **Similarity alone fails on gate flips, by a wide margin.** With gates planted, cosine retrieval's
   squared error on flips was 1.9 to 2.5, against 0.26 to 0.51 for retrieval inside the stratum.
   This is the planted effect, so it shows only that the instrument sees it.
2. **A learned metric with no gates (arm M) does not recover the gates.** Its error stayed at 0.78 to
   1.09 where gate-aware arms reached 0.27 to 0.37.
3. **The learned low-rank metric made the retrieved set much less stable under a simulated rewording.**
   Overlap of the 10 retrieved cases before and after the rewording was 0.62 for cosine and 0.14 to
   0.30 for the learned metric, in every world. The simulated rewording has an arbitrary size, so the
   numbers do not transfer. The direction is a warning: projecting to 8 dimensions packs the cases
   closely, and small moves reshuffle the nearest ten. Retrieval stability has to be measured on real
   paraphrases, and paraphrase pairs may need to enter the metric's loss as must-link constraints.

## Not done

- No real judgements, no gate extraction, no real paraphrases.
- H3 was not repaired. Candidates: more training cases per stratum, a longer fit, or a direct test of
  a stratum-by-metric interaction in place of an arm comparison.
- The bar of 2.5 was not tuned. Its measured false-pass count is 0 in 20 for each hypothesis, which
  bounds the rate only loosely (below about 14% at 95% confidence).
- BGE-M3 embeddings were not rehearsed.

## Round 3: the code as sealed

After round 2 the grader was changed in two ways: a family with no data is reported as not run, and
void condition V5 (an arm must beat the grand-mean predictor) is checked. The arms gained the
grand-mean predictor. Round 3 re-ran all 45 runs and the self-tests on this code
(`rehearsal/round3/`). Every count in the round 2 table was reproduced. V5 found signal in 25 of 25
structured runs and in 0 of 20 noise runs.
