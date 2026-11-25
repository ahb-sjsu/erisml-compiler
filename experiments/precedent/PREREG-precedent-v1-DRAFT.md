# prereg-precedent-v1: gates, precedents and a learned metric as predictors of moral judgement

**Status: DRAFT, 2026-11-25. Not sealed. No judgement file opened. Drafted by Claude at the owner's
request; every choice below is Claude's and is open to the owner. Sealing needs the owner's OK.**

Conventions follow `geometric-methods` Chapter 21 and the sealed registrations in
`geometric-economics/experiments/papers/` (bundle, seal file, signed tag, bars in the grader,
rehearsal on planted truths).

## 1. The question

The stratified model of Geometric Ethics Chapter 8 has two parts. Gates switch which rules apply.
Inside a stratum a metric sets the trade-offs. The analysis of 2026-11-25
(`geometric-ethics/notes/2026-11-25-hnsw-and-stratified-geometry.md`) found that the gates are the
part with support and that the metric's contribution is untested.

**Does a learned metric predict held-out moral judgements better than gates and precedents alone?**
And before that: do explicit gates predict better than similarity alone where a single gate changes?

Every claim involved is posited today. The gate rates printed in the book trace, in the files
examined, to constants typed into a generator, so this would also be the first measured gate effect.

## 2. Data

Nothing here has been opened beyond file names, column names and sizes.

| Role | Proposed source | Unit for clustering | Note |
|---|---|---|---|
| Case base and development | Social-Chem-101, AITA slice (`data/social-chem-101/`, 96,083 rows) | situation id | Judgement = mean `action-moral-judgment` over the situation's rows, scale -2 to 2 |
| Family 1, ordinary held-out cases | AITA situations held out by situation id | situation id | |
| Family 2, single-gate flips | δ-Rules-of-Thumb (arXiv 2310.15431) or MoCa (arXiv 2310.19677), which carry human labels for a context that changes one factor | base situation | Contents not yet verified. If neither maps to named gates, an authored set with new human ratings is needed, which needs ethics approval |
| Family 3, another domain | Social-Chem-101, Dear Abby slice (50,301 rows) | situation id | Never in the case base. Reported, not graded |
| **Held back unseen** | a second flip source, named at sealing | | Spent when touched |

A consent-specific flip set does not exist in published form as far as two searches found. It is
left to a second registration.

**Size is counted in units, not rows.** The number of distinct situations and of distinct flip bases
is to be counted at Stage 0 and written into the seal. No power analysis has been done. The rehearsal's
effect sizes are planted and do not transfer.

## 3. Instrument

**Coordinates.** LaBSE at snapshot `836121a0533e5664b21c7aacc5d22951f2b8b25b`. A label-free projection
(48 principal components of the training embeddings, standardised). No hand-made moral axis is read.
BGE-M3 is run as a second encoder and reported, not graded.

**Gates.** Read from the compiler's `MoralGraph` by the LLM extractor, through a gate registry
(consent, commitment, coercion, emergency, and the nullifiers abuse, danger, impossibility). A case's
stratum is its gate pattern. An unknown gate value is its own value.

**Metric.** A low-rank factor U (rank 8), cost ||U(x - x')||, fitted on training cases only by
kernel-regression metric learning on the judgement with the stratum mean removed. This differs from
`fit_learned.py`, whose loss is for paired choices. The per-stratum metric is the shared one plus a
penalised per-stratum part (weight 0.01).

**Arms.** G (stratum mean), A (10 nearest by cosine), M (10 nearest under the metric, no gates),
B (A inside the query's stratum), Cp (B under the shared metric), Cl (B under the per-stratum metric).
Fixed constants are at the top of `precedent_arms.py`.

## 4. Predictions and bars

Each is a paired difference in per-item squared error on a held-out family. It passes when the mean
difference is below -2.5 standard errors, with the standard error from resampling whole units. The
bars are in `grade_precedent.py` and nowhere else.

| | Prediction | Compared | Family | Status |
|---|---|---|---|---|
| H1 | An explicit gate check beats similarity where one gate changes | B vs A | flips | Graded |
| H2 | A learned metric adds prediction beyond gates and precedents | Cp vs B | ordinary | Graded |
| H4 | Precedents add prediction beyond the gate rule | B vs G | ordinary | Graded, not part of the reading |
| H3 | A per-stratum metric beats a shared one | Cl vs Cp | ordinary | **Reported only.** The rehearsal detected a strong planted effect in 2 of 5 runs, so a miss cannot be read |

**Reading of H1 and H2.**

| H1 | H2 | Reading | Consequence |
|---|---|---|---|
| pass | fail | Gates carry the prediction. The metric is unsupported on this data | Build precedent retrieval with gate checks (plan phases P2, P3). Keep the metric out of the product. The book states that the metric is a modeling choice without evidence |
| pass | pass | Gates and a learned metric both carry prediction | Phases P3 and P4. First measured support for the manifold component |
| fail | pass | Smooth structure only. Gates as extracted do not help | Look first at gate extraction (Section 5, V2) before concluding against strata |
| fail | fail | Nothing detected | See void condition V5 |

A bar in noise units is an effect-size bar. More data sharpens the step and does not move it. H2 can
fail on a real but small metric effect, and the registration says so here.

## 5. Void conditions (instrument gates)

A miss here voids the run and refutes nothing.

| | Condition | Why |
|---|---|---|
| V1 | A unit appears on both sides of a split | Leakage. The harness refuses it |
| V2 | Two extractors agree on a gate at κ below 0.6, or a human spot-check of 200 cases agrees below 0.8 | A gate that cannot be read cannot be tested. The gate is dropped, and if fewer than two remain the run is void |
| V3 | A stratum has fewer than 40 training cases | Merged into "other", stated in the results |
| V4 | Encoder snapshot differs from the sealed one | Different instrument |
| V5 | No arm beats the grand-mean predictor on ordinary cases by 2.5 standard errors | The instrument reads no signal |
| V6 | The judgement scale resolves fewer than 5 distinct values across the case base | Attenuation. The mean over few rows per situation may be too coarse |
| V7 | The data source is not declared `held_out` | The grader refuses, and simulated inputs get a rehearsal reading only |

## 6. Admissibility (geometric-methods 21.4)

| Check | Discharged by |
|---|---|
| A1 Nonvacuity | 20 noise runs, 0 passes for each hypothesis. Planted worlds in which each graded hypothesis is false: H1 0 of 25, H2 0 of 25 (`REHEARSAL.md`) |
| A2 Identification | The metric is fitted on training units only. The split guard is tested (S1). Rank is fixed at 8, not selected |
| A3 Intervention | For flips: the two members of a pair differ in one named gate, checked by the extractor on both members. Pairs failing this are excluded and counted |
| A4 Inference | Estimand and unit fixed above. Two graded hypotheses in the reading, no correction applied, stated |

## 7. Reported without a bar

- Family 3 (another domain): every arm's error.
- Retrieval stability: Jaccard at 10 and rank-biased overlap of the retrieved set across real
  paraphrases, for cosine and for the learned metric. The rehearsal found the learned metric far less
  stable under a simulated rewording, so this is watched closely.
- The jump across each gate, by a cross-fitted doubly robust estimator, with contrast pairs shown as
  its explanation.
- Recall of an HNSW index against exact search on the case base, and time.
- H3.

## 8. Rehearsal

See `REHEARSAL.md`. Round 1 failed on H3. Round 2 changed the per-stratum fit and added a stronger
planted world. H1, H2 and the noise world read back correctly in 35 of 35 runs. H3 did not, and is
demoted to "reported only" in Section 4.

## 9. Stages

- **Stage 0, no judgement scored.** Count units. Compile the cases. Measure gate agreement (V2).
  Open and verify the flip sources. Fix the gate list. Write the seal.
- **Seal.** Bundle: this file, `precedent_arms.py`, `grade_precedent.py`, `rehearse.py`, the Stage 0
  run script, the gate registry, the rehearsal summary. `seal_precedent.py` (to be written on the
  pattern of `seal_learned.py`) hashes them, a signed commit and signed tag `prereg-precedent-v1` are
  pushed before any held-out judgement is scored.
- **Run** from a clone at the tag, on Atlas. Results to `RESULTS-prereg-precedent-v1.md`.

## 10. Open before sealing

1. Owner's OK on the question, the arms and the reading table.
2. Which flip source, and which is held back.
3. Whether to spend another rehearsal round on H3 or leave it reported only.
4. Whether to add gate erasure (LEACE) before the metric fit. It would make arm M gate-blind by
   construction. It needs its own rehearsal.
5. The Stage 0 extraction needs an LLM slot (NRP or Atlas). Atlas was running the Lean build on
   2026-11-25.
6. `seal_precedent.py` and the Stage 0 script are not written.
