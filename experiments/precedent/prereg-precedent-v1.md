# prereg-precedent-v1: gates, precedents and a learned metric as predictors of moral judgement

**Status: SEALED 2026-11-25** (`prereg-precedent-v1.sha256`, tag `prereg-precedent-v1`). No held-out
judgement had been read when it was sealed. Drafted by Claude. The owner said on 2026-11-25: "seal as
drafted and proceed as you recommend. wait for test results to update book". Section 0 lists what
changed between the draft and the seal. Each change was made before any held-out judgement was read.

Conventions follow `geometric-methods` Chapter 21 and the sealed registrations in
`geometric-economics/experiments/papers/` (bundle, seal file, signed tag, bars in the grader,
rehearsal on planted truths).

## 0. Changes between the draft and the seal

The draft is in the repository history (commit 53501a9, `PREREG-precedent-v1-DRAFT.md`). The Stage 0
record is `STAGE0.md`.

1. **H1 (gates beat similarity on gate flips) is not run in v1.** A check of seven published datasets
   found none with human-rated single-gate flips for consent, promise, coercion or emergency. The
   draft provided for this case. H1 goes to a second registration. The grader reports it as not run,
   never as a miss.
2. **The corpus is the Scruples Anecdotes, not Social-Chem titles.** The first gate pilot, on one-line
   AITA titles, failed: one gate met the rule below. Titles do not state the gates. Scruples carries
   the full post and community votes on the full post.
3. **Gates are read by one structured prompt** (`read_gates.py`, registry `gates.json` version
   gates-v2) put to two hosted models, and not through the compiler's full extractor. The primary
   reader is `openai/gpt-oss-120b`, the second `google/gemma-4-31B-it-qat-w4a16-ct`. The registry was
   narrowed once, after the first pilot.
4. **The strata use two gates, commitment and abuse,** the two that met the rule in the second pilot.
5. **The other-domain family is dropped.** Scruples has one domain.
6. **Void condition V5 is in the grader** (an arm must beat the grand-mean predictor), and **V8** was
   added (more than 5% of a role's cases with no gate reading).
7. **Reported-only items were cut to what v1 can compute** (Section 7).

## 1. The question

The stratified model of Geometric Ethics Chapter 8 has two parts. Gates switch which rules apply.
Inside a stratum a metric sets the trade-offs. The analysis of 2026-11-25
(`geometric-ethics/notes/2026-11-25-hnsw-and-stratified-geometry.md`) found that the gates are the
part with support and that the metric's contribution is untested.

**Does a learned metric predict held-out moral judgements better than gates and precedents alone?**

The companion question, whether explicit gates predict better than similarity alone where a single
gate changes, needs flip data that does not exist yet (Section 0, item 1).

Every claim involved is posited today.

## 2. Data

Scruples Anecdotes v1.0 (Lourie, Le Bras, Choi, arXiv 2008.09094): posts from a forum where the
author asks whether they were in the wrong, with community votes. The dataset's own splits are used.

| Role | Source | Unit | Count |
|---|---|---|---|
| Case base (training) | training split, 27,766 posts | post id | the 4,000 lowest by `sha256("train|" + id)` |
| Ordinary held-out cases | test split, 2,500 posts | post id | the 1,500 lowest by `sha256("ordinary|" + id)` |

The dev split is not used. Cases were chosen from ids, titles and texts only (`build_cases.py`).

**The judgement** of a post is the share of its binarized votes that find the author in the wrong,
from 0 to 1. A post with fewer than 5 votes is left out when judgements are read, and counted.

Counted on the chosen training cases only (`stage0/cases.sha256.json`): 2,949 of 4,000 have at least
5 votes, the median post has 8 votes, 362 distinct judgement values occur, the mean is 0.296 and the
variance 0.095. About 24% of posts have a majority finding the author in the wrong. No held-out vote
count has been read, so the number of held-out cases that survive the 5-vote rule is not known. It
should be near 1,100 if the test split resembles the training split.

With a median of 8 votes the judgement is a noisy reading. No power analysis was done. The
rehearsal's effect sizes are planted and do not transfer.

## 3. Instrument

**Text.** Title and body joined. The encoder reads the start of the text up to its own length limit.
The gate reader reads the first 4,000 characters.

**Coordinates.** LaBSE at snapshot `836121a0533e5664b21c7aacc5d22951f2b8b25b`. A label-free projection
(48 principal components of the training embeddings, standardised). No hand-made moral axis is read.

**Gates.** Six gates are asked about (`gates.json`): no consent, commitment, coercion, emergency,
abuse, impossibility. A gate enters the strata when, in a pilot of 300 training posts, the two readers
agreed at kappa of at least 0.6 and the primary reader found it in at least 2% of texts. The rule was
written before the first pilot's agreement figures were seen. Second pilot
(`stage0/pilot2_agreement.json`):

| Gate | Found by gpt-oss | Found by gemma | Kappa | Kept |
|---|---|---|---|---|
| commitment | 27.3% | 24.7% | 0.64 | yes |
| abuse | 5.0% | 3.3% | 0.63 | yes |
| coercion | 7.0% | 6.0% | 0.59 | no |
| emergency | 3.0% | 1.7% | 0.56 | no |
| no consent | 23.3% | 16.7% | 0.55 | no |
| impossibility | 5.0% | 3.0% | 0.48 | no |

Both kept gates passed narrowly, on 57 and 8 jointly positive posts. Their agreement is measured again
on all 5,500 cases at the run (V2), and a gate that falls below 0.6 there is dropped.

A case's stratum is its pattern over the kept gates as the primary reader read them. A pattern with
fewer than 40 training cases, or one unseen in training, goes to a stratum called "other". A case the
primary reader could not read is left out of every arm and counted.

**Metric.** A low-rank factor U (rank 8), cost ||U(x - x')||, fitted on training cases only by
kernel-regression metric learning on the judgement with the stratum mean removed. This differs from
`fit_learned.py`, whose loss is for paired choices. The per-stratum metric is the shared one plus a
penalised per-stratum part (weight 0.01).

**Arms.** Mean (the grand mean of the training judgements), G (stratum mean), A (10 nearest by
cosine), M (10 nearest under the metric, no gates), B (A inside the query's stratum), Cp (B under the
shared metric), Cl (B under the per-stratum metric). Fixed constants are at the top of
`precedent_arms.py`.

## 4. Predictions and bars

Each is a paired difference in per-item squared error on the held-out cases. It passes when the mean
difference is below -2.5 standard errors, with the standard error from resampling whole units. The
bars are in `grade_precedent.py` and nowhere else.

| | Prediction | Compared | Status |
|---|---|---|---|
| H2 | A learned metric adds prediction beyond gates and precedents | Cp vs B | **Graded. The reading rests on it** |
| H4 | Precedents add prediction beyond the gate rule | B vs G | Graded, beside the reading |
| H1 | An explicit gate check beats similarity where one gate changes | B vs A on flips | Not run in v1. No flip data |
| H3 | A per-stratum metric beats a shared one | Cl vs Cp | Reported only. The rehearsal detected a strong planted effect in 2 of 5 runs, so a miss cannot be read |

**Reading.**

| H2 | Reading | Consequence |
|---|---|---|
| pass | A learned metric adds prediction beyond gates and precedents | Plan phase P4. First measured support for the manifold component. The book may say so |
| fail | The metric is unsupported on this data | Retrieval uses cosine inside strata. The metric stays out of the product. The book states that the metric is a modeling choice without evidence so far |

H4 says whether precedents add anything to the gate rule. If H4 fails as well, retrieval itself is
unsupported on this data and plan phase P3 is reconsidered.

A bar in noise units is an effect-size bar. More data sharpens the step and does not move it. H2 can
fail on a real but small metric effect. A failure here is a failure for these two gates, this encoder,
rank 8 and a judgement averaged over a median of 8 votes, and should be read no more widely.

## 5. Void conditions (instrument gates)

A miss here voids the run and refutes nothing.

| | Condition | Why |
|---|---|---|
| V1 | A unit appears on both sides of the split | Leakage. The run refuses it |
| V2 | On the full data the two readers agree on a kept gate at kappa below 0.6, or a human spot-check of 200 cases agrees with the primary reader below 0.8 | A gate that cannot be read cannot be tested. The gate is dropped, and if fewer than two remain the run is void. The human spot-check is the owner's to do. Until it is done the result is provisional |
| V3 | A stratum has fewer than 40 training cases | Merged into "other", stated in the results |
| V4 | Encoder snapshot differs from the sealed one | Different instrument |
| V5 | No arm beats the grand-mean predictor on held-out cases by 2.5 standard errors | The instrument reads no signal |
| V6 | The training judgements take fewer than 5 distinct values | Attenuation |
| V7 | The data source is not declared `held_out` | The grader refuses, and simulated inputs get a rehearsal reading only |
| V8 | More than 5% of a role's cases have no gate reading from the primary reader | The strata would be decided by which cases the reader failed on |

V2 is a live risk. Both kept gates passed the pilot by less than 0.04.

## 6. Admissibility (geometric-methods 21.4)

| Check | Discharged by |
|---|---|
| A1 Nonvacuity | 20 noise runs, 0 passes for each hypothesis. Planted worlds in which H2 is false: 0 passes in 25 runs. V5 fired in 0 of 20 noise runs and in 25 of 25 structured runs (`rehearsal/round3`) |
| A2 Identification | The metric is fitted on training units only. The split is the dataset's own, and the run refuses an id on both sides. Rank is fixed at 8, not selected |
| A3 Intervention | Not applicable in v1. No manipulation is graded |
| A4 Inference | Estimand and unit fixed above. One hypothesis in the reading (H2) and one beside it (H4), no correction applied, stated |

One of the four checks is declared not applicable.

## 7. Reported without a bar

- H3 (per-stratum metric against shared metric) and arm M (learned metric, no gates).
- Agreement and prevalence of all six gates on the full data.

Not computed in v1 and left to later work: retrieval stability on real paraphrases, a doubly robust
estimate of the jump across each gate, recall of an HNSW index against exact search, a BGE-M3 run, and
a second domain.

## 8. Rehearsal and smoke test

See `REHEARSAL.md`. Round 1 failed on H3. Round 2 changed the per-stratum fit and added a stronger
planted world. Round 3 re-ran the arms and the grader as sealed and reproduced round 2. H2 and the
noise world read back correctly in every run. H3 did not, and is reported only.

`run_precedent.py --smoke` was run once on the 300 pilot posts, with both roles taking their
judgements from the training split. It showed that the code runs end to end. It is not a result.

## 9. Stages

- **Stage 0, no held-out judgement read.** Done before the seal: cases chosen by hash, training-side
  counts, two gate pilots, the gate list fixed. To do after the seal, reading texts only: the
  embedding and the gate reading of all 5,500 cases.
- **Seal.** `seal_precedent.py` hashes the bundle. A signed commit and the signed tag
  `prereg-precedent-v1` are pushed before `run_precedent.py` is run without `--smoke`.
- **Run.** `run_precedent.py` refuses a broken seal, a different cases file or a different encoder
  snapshot. Results go to `RESULTS-prereg-precedent-v1.md`.
- **After the run.** The owner's spot-check of 200 gate readings (V2). The book is updated only after
  the result, as the owner directed.
