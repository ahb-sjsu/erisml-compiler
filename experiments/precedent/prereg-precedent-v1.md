# prereg-precedent-v1: gates, precedents and a learned metric as predictors of moral judgement

**Status: NOT SEALED, 2026-11-25.** The owner said on 2026-11-25: "seal as drafted and proceed as you
recommend. wait for test results to update book". Stage 0 then failed: in the pilot only one gate met
the rule in Section 3, and Section 5 voids a run with fewer than two. See `STAGE0.md`. This file is the
text that was ready for sealing, kept as written. Sections 2 and 3 will be rewritten for the corpus
that passes a pilot. No held-out judgement has been read. Drafted by Claude.

Conventions follow `geometric-methods` Chapter 21 and the sealed registrations in
`geometric-economics/experiments/papers/` (bundle, seal file, signed tag, bars in the grader,
rehearsal on planted truths).

## 0. Changes between the draft and the seal

The draft is in the repository history (commit 53501a9, `PREREG-precedent-v1-DRAFT.md`).

1. **H1 (gates beat similarity on gate flips) is not run in v1.** A check of seven published datasets
   found none with human-rated single-gate flips for consent, promise, coercion or emergency. The
   nearest, delta-SOCIAL, has human-written updates with a direction label only and unnamed factors.
   The draft provided for this case ("an authored set with new human ratings is needed"). H1 goes to
   a second registration. The grader reports it as not run, never as a miss.
2. **Gates are read by one structured prompt** (`read_gates.py`, registry `gates.json`) put to two
   hosted models, and not through the compiler's full extractor. The full extractor makes four calls
   per case and has no node for emergency or the nullifiers. The primary reader is
   `openai/gpt-oss-120b`, the second `google/gemma-4-31B-it-qat-w4a16-ct`.
3. **The split is fixed by a hash** of the situation id (`stage0_counts.py`), and the cases are the
   lowest hashes on each side (`build_cases.py`): 6,000 training, 1,500 ordinary held-out, 1,500 from
   another domain.
4. **Void condition V5 is now in the grader** (an arm must beat the grand-mean predictor), and **V8**
   was added (more than 5% of a role's cases with no gate reading).
5. **The gate list was fixed from the Stage 0 pilot** (Section 3).
6. **Reported-only items were cut to what v1 can compute** (Section 7).

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

Counted at Stage 0 with no held-out judgement read (`stage0/counts_aita.json`,
`stage0/counts_dearabby.json`).

| Role | Source | Unit | Count |
|---|---|---|---|
| Case base (training) | Social-Chem-101, AITA slice, hash bucket below 80 | situation id | 6,000 of 23,385 judged situations |
| Ordinary held-out cases | AITA slice, hash bucket 80 and above | situation id | 1,500 of 5,821 |
| Another domain | Social-Chem-101, Dear Abby slice, hash bucket 80 and above | situation id | 1,500 of 2,321. Never in the case base. Reported, not graded |

The judgement of a situation is the mean `action-moral-judgment` over its rows, on a scale from -2 to
2. On the case-base side the median situation has 2 rows and 14% have one. 261 distinct values occur.
About two thirds of case-base situations round to 0 and the variance is 0.36, so the room for any arm
to improve on a constant is limited. No power analysis was done. The rehearsal's effect sizes are
planted and do not transfer.

The texts are situation titles, with a median length of 48 characters in the pilot.

## 3. Instrument

**Coordinates.** LaBSE at snapshot `836121a0533e5664b21c7aacc5d22951f2b8b25b`. A label-free projection
(48 principal components of the training embeddings, standardised). No hand-made moral axis is read.

**Gates.** Six gates are asked about (`gates.json`): no consent, commitment, coercion, emergency,
abuse, impossibility. A gate enters the strata when, in the Stage 0 pilot of 300 case-base texts, the
two readers agreed at kappa of at least 0.6 and the primary reader found it in at least 2% of texts.

The pilot result is in `STAGE0.md`. Only commitment met the rule (kappa 0.68, found in 3.3% of
titles). No consent and coercion fell below 0.6 and emergency was too rare.

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

Each is a paired difference in per-item squared error on a held-out family. It passes when the mean
difference is below -2.5 standard errors, with the standard error from resampling whole units. The
bars are in `grade_precedent.py` and nowhere else.

| | Prediction | Compared | Family | Status |
|---|---|---|---|---|
| H2 | A learned metric adds prediction beyond gates and precedents | Cp vs B | ordinary | **Graded. The reading rests on it** |
| H4 | Precedents add prediction beyond the gate rule | B vs G | ordinary | Graded, beside the reading |
| H1 | An explicit gate check beats similarity where one gate changes | B vs A | flips | Not run in v1. No flip data |
| H3 | A per-stratum metric beats a shared one | Cl vs Cp | ordinary | Reported only. The rehearsal detected a strong planted effect in 2 of 5 runs, so a miss cannot be read |

**Reading.**

| H2 | Reading | Consequence |
|---|---|---|
| pass | A learned metric adds prediction beyond gates and precedents | Plan phase P4. First measured support for the manifold component. The book may say so |
| fail | The metric is unsupported on this data | Retrieval uses cosine inside strata. The metric stays out of the product. The book states that the metric is a modeling choice without evidence so far |

H4 says whether precedents add anything to the gate rule. If H4 fails as well, retrieval itself is
unsupported on this data and plan phase P3 is reconsidered.

A bar in noise units is an effect-size bar. More data sharpens the step and does not move it. H2 can
fail on a real but small metric effect, and the registration says so here. A failed H2 on one-line
situation titles also says little about longer texts.

## 5. Void conditions (instrument gates)

A miss here voids the run and refutes nothing.

| | Condition | Why |
|---|---|---|
| V1 | A unit appears on both sides of a split | Leakage. The run refuses it |
| V2 | On the full data the two readers agree on a kept gate at kappa below 0.6, or a human spot-check of 200 cases agrees with the primary reader below 0.8 | A gate that cannot be read cannot be tested. The gate is dropped, and if fewer than two remain the run is void. The human spot-check is the owner's to do. Until it is done the result is provisional |
| V3 | A stratum has fewer than 40 training cases | Merged into "other", stated in the results |
| V4 | Encoder snapshot differs from the sealed one | Different instrument |
| V5 | No arm beats the grand-mean predictor on ordinary cases by 2.5 standard errors | The instrument reads no signal |
| V6 | The training judgements take fewer than 5 distinct values | Attenuation |
| V7 | The data source is not declared `held_out` | The grader refuses, and simulated inputs get a rehearsal reading only |
| V8 | More than 5% of a role's cases have no gate reading from the primary reader | The strata would be decided by which cases the reader failed on |

## 6. Admissibility (geometric-methods 21.4)

| Check | Discharged by |
|---|---|
| A1 Nonvacuity | 20 noise runs, 0 passes for each hypothesis. Planted worlds in which H2 is false: 0 passes in 25 runs. V5 fired in 0 of 20 noise runs and in 25 of 25 structured runs (`rehearsal/round3`) |
| A2 Identification | The metric is fitted on training units only. The split guard is tested (S1). Rank is fixed at 8, not selected |
| A3 Intervention | Not applicable in v1. No manipulation is graded |
| A4 Inference | Estimand and unit fixed above. One hypothesis in the reading (H2) and one beside it (H4), no correction applied, stated |

One of the four checks is declared not applicable.

## 7. Reported without a bar

- The other-domain family (Dear Abby): every arm's error.
- H3 (per-stratum metric against shared metric) and arm M (learned metric, no gates).
- Gate prevalence and reader agreement on the full data.

Not computed in v1 and left to later work: retrieval stability on real paraphrases (none exist for
these texts), a doubly robust estimate of the jump across each gate, recall of an HNSW index against
exact search, and a BGE-M3 run.

## 8. Rehearsal

See `REHEARSAL.md`. Round 1 failed on H3. Round 2 changed the per-stratum fit and added a stronger
planted world. Round 3 re-ran the code as sealed, after the grader learned to report a missing family
and to check V5, and reproduced round 2. H2 and the noise world read back correctly in every run.
H3 did not, and is reported only.

## 9. Stages

- **Stage 0, no held-out judgement read.** Done before the seal: units counted, cases chosen by hash,
  texts embedded, the gate reader piloted and the gate list fixed. Started before the seal and still
  running at the seal: the gate reading of all 9,000 cases, which reads texts only.
- **Seal.** `seal_precedent.py` hashes the bundle. A signed commit and the signed tag
  `prereg-precedent-v1` are pushed before `run_precedent.py` is run.
- **Run.** `run_precedent.py` refuses a broken seal, a different cases file or a different encoder
  snapshot. Results go to `RESULTS-prereg-precedent-v1.md`.
- **After the run.** The owner's spot-check of 200 gate readings (V2). The book is updated only after
  the result, as the owner directed.
