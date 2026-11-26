# Results: prereg-precedent-v1

**Outcome: VOID under condition V5.** No arm predicted the held-out judgements better than
the grand mean of the training judgements by 2.5 standard errors. By the registration
(Section 5), a miss on a void condition voids the run and refutes nothing. H2 and H4 are
therefore not graded, and neither row of the registration's reading table applies.

The run was made on 2026-11-25 from a clone of this repository at the signed tag
`prereg-precedent-v1` (commit fc25fca). The seal was checked after the gate reading and
before any held-out judgement was read, and was intact (`run/seal_check.txt`). Written by
Claude from the run's output files. Until the owner's spot-check of 200 gate readings is
done, everything here is provisional (registration, V2).

## What ran

| Step | Result |
|---|---|
| Cases | 5,500 Scruples Anecdotes posts, the sealed file (sha256 `0844a1f4...ed07b1b`): 4,000 training, 1,500 from the test split |
| Gate reading, primary (`openai/gpt-oss-120b`) | 5,500 of 5,500 read. Unread share 0 in both roles (V8 not met) |
| Gate reading, second (`google/gemma-4-31B-it-qat-w4a16-ct`) | 5,500 of 5,500 read. Five posts failed on a first pass and were read on a later one, so the file has 5,505 lines |
| Judgements | 2,949 training posts and 1,165 held-out posts have at least 5 votes. 1,051 and 335 were left out |
| Training judgement scale | 362 distinct values (V6 not met) |
| Strata, from the two kept gates | neither 2,744, abuse only 302, commitment only 892, both 62 training posts. No stratum merged (V3 not met) |

The gate reading took 3 hours 45 minutes. The graded run took about 2 minutes on one GPU.

## Gate agreement on all 5,500 posts (V2)

| Gate | Found by the primary reader | Kappa between the readers | In the strata |
|---|---|---|---|
| abuse | 9.4% | 0.677 | yes |
| coercion | 8.0% | 0.638 | no (not sealed) |
| commitment | 23.7% | 0.632 | yes |
| no consent | 27.4% | 0.552 | no |
| emergency | 4.2% | 0.526 | no |
| impossibility | 2.2% | 0.347 | no |

Both sealed gates stayed at or above 0.6, so V2 did not void the run. Coercion, which missed
the bar in the pilot at 0.59, is above it on the full data. It was not in the sealed list
and was not used. Consent stayed below the bar, as in both pilots.

## The arms on the 1,165 held-out posts

Mean squared error of the predicted share of votes finding the author in the wrong. The
last column is the paired difference from the grand-mean predictor in standard errors
(resampling posts). Negative is better than the mean.

| Arm | What it predicts with | Mean squared error | Against the grand mean (z) |
|---|---|---|---|
| Mean | the grand mean of the training judgements | 0.09751 | |
| G | the mean of the post's stratum | 0.09701 | −1.30 |
| A | 10 nearest posts by cosine | 0.10167 | +2.14 |
| B | 10 nearest by cosine inside the stratum | 0.10242 | +2.64 |
| M | 10 nearest under the learned metric, no gates | 0.11016 | +4.94 |
| Cp | 10 nearest under the learned metric inside the stratum | 0.11036 | +5.36 |
| Cl | the same with a per-stratum metric | 0.11075 | +5.68 |

V5 asks for one arm at or below −2.5. The best is the stratum mean at −1.30. **The instrument
read no signal by its own registered test, so the run is void.**

## The registered comparisons, reported and not graded

| | Compared | Mean difference in squared error | z | Sign |
|---|---|---|---|---|
| H2 | Cp against B | +0.00794 | +2.98 | the learned metric was worse than cosine |
| H4 | B against G | +0.00540 | +2.97 | precedents were worse than the gate rule alone |
| H3 (reported only) | Cl against Cp | +0.00039 | +0.51 | no difference |
| H1 | B against A on flips | not run | | no flip data exists |

A pass needed a difference below −2.5. Both graded comparisons came out on the wrong side
by about three standard errors. Because the run is void these are observations, not
failures of H2 and H4.

## What can and cannot be said

- **Cannot be said:** that the learned metric is refuted, or that precedents add nothing.
  The registration fixed in advance that a run in which nothing beats the mean tests
  neither.
- **Cannot be said:** that the metric has support. It has none from this run.
- **Can be said:** on this corpus, with this encoder and these two gates, every
  nearest-neighbour arm predicted worse than a constant, and the learned metric made the
  prediction worse again. The only arm on the right side of the constant was the gate rule
  with no retrieval, and its margin (1.3 standard errors) is not distinguishable from zero.
- **Can be said:** two of six gates can be read by two models at a kappa of 0.6 or more on
  5,500 full posts, a third reaches it without having been sealed, and consent cannot.

## Why the instrument may have read nothing (my reading, not a result)

These are conjectures for the design of a second registration. None was tested.

1. **The label is noisy.** The median post has 8 votes. The variance of a share over 8 votes
   is a large part of the 0.095 variance of the judgement, and no predictor can remove it.
2. **Ten neighbours is a high-variance estimator.** Averaging 10 noisy labels has more
   variance than the grand mean of 2,949, and the bias it removes has to exceed that. A
   predictor that shrinks toward the mean (many more neighbours, or a kernel) was not among
   the arms. The rehearsal did not expose this because its planted labels had far less noise.
3. **The learned metric fitted noise.** It was worse than cosine in both the gated and the
   ungated arm, by about the same amount. A rank-8 metric fitted to labels this noisy would
   do that.
4. **The embedding may carry little of the judgement.** The encoder read the start of each
   post up to its length limit, and the posts run to about 1,500 characters at the median.

## What a second registration needs first

A void run says the instrument must be shown to read signal before it is used to compare
arms. Before sealing again I recommend, on training posts only:

- an arm that is allowed to shrink toward the mean, and a check on a held-out part of the
  training split that some arm beats the constant by the registered margin,
- a noise ceiling: how much of the variance of the judgement a perfect predictor could
  explain given the vote counts,
- posts with more votes, at the cost of fewer posts, to raise that ceiling,
- a rehearsal whose planted labels have the noise of the real ones.

The flip data for H1 (single-gate changes rated by people) is still the stronger test of
the gates and is still missing.

## Departures from the registration

None in the analysis. The script that chained the stages (gate reading by both readers, the seal check, then `run_precedent.py`; not committed, since it holds machine paths) made up to four passes of the gate reader so that
failed calls were retried. The registration does not mention retries. They are visible in
the second reader's file as five records with no reading followed by five with one.

## Files

| File | Content |
|---|---|
| `run/results_sealed.json` | the graded output (sha256 `19c8dfed...db2750b7`) |
| `run/run_precedent_report.txt` | the run's printed report (`run_precedent.log` on the run machine; renamed because this repository ignores `*.log`) |
| `run/seal_check.txt`, `run/status.txt` | the seal check before the run, and the two completion times |
| `run/gates_gpt-oss.jsonl`, `run/gates_gemma.jsonl` | every gate reading, with the model's raw answer |
| `run/emb_labse.meta.json` | the encoder snapshot record |

The cases file (post texts) and the embeddings are not committed. The cases file is fixed
by its digest in `stage0/cases.sha256.json` and is rebuilt by `build_cases.py` from the
dataset.

## Addendum, 2026-11-26: why nothing beat the constant (exploratory, training posts only)

After the result, two checks were run on the **training split only**. No dev or test
judgement was read. Nothing here was registered, so the class is exploratory. Scripts:
`after/noise_ceiling.py`, `after/signal_check.py`, `after/signal_check_bge.py`,
`after/embed_bge.py`.

**The label noise is not the cause.** This corrects item 1 of my reading above. For the
27,766 training posts, the sampling variance of a share of votes was estimated from each
post's own vote count and compared with the variance of the judgement.

| Minimum votes | Posts | Variance of the judgement | Of which vote noise | Share that is not noise |
|---|---|---|---|---|
| 5 | 20,484 | 0.0962 | 0.0132 | 0.86 |
| 10 | 10,323 | 0.1031 | 0.0076 | 0.93 |
| 20 | 3,896 | 0.1088 | 0.0035 | 0.97 |
| 50 | 1,386 | 0.1082 | 0.0012 | 0.99 |

At the registered threshold of 5 votes, a perfect predictor could explain 86% of the
variance. The run's arms explained none.

**The embedding carries almost none of the judgement.** Five-fold cross-validation on the
2,949 registered training posts, squared error against the mean of the training folds:

| Predictor | LaBSE (the registered encoder): variance explained, z against the mean | BGE-M3, 1,024 tokens: variance explained, z |
|---|---|---|
| Ridge regression on the embedding, best of three penalties | 2.5%, −4.8 | 4.0%, −6.5 |
| 10 nearest by cosine (the registered arm A) | −3.5%, +2.7 | −3.5%, +2.8 |
| 100 nearest by cosine | 1.9%, −3.6 | 3.2%, −5.6 |
| 300 nearest by cosine | 1.8%, −5.2 | 3.0%, −7.8 |

Three things follow.

1. Ten neighbours is worse than the constant in cross-validation too, by the same margin as
   in the sealed run. That part of the void is the estimator: with 100 or more neighbours the
   same embedding beats the constant by more than the registered margin.
2. Even the best simple predictor explains 2.5% to 4% of a variance that is 86% signal.
   Whether a forum's verdict falls on the author is, on this evidence, barely a function of
   what a sentence encoder keeps of the post. A metric learned on such an embedding has
   almost nothing to work with, whatever its form.
3. The second encoder is the stronger one, with a longer context, and gains about 1.5 points.
   The limit is not the truncation of the text.

**What this means for a second registration.** More votes or more posts of this kind would
not help. The test needs material in which the judgement depends on factors stated in the
text, with many raters per item and, ideally, pairs that differ in one factor. A survey of
such datasets is under way. On this corpus, a fair test of "does a metric add to gates"
would need a representation that reads the judgement first, for example features read from
the text by a language model, with that reading registered as part of the instrument.
