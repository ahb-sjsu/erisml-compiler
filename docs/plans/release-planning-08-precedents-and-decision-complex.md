# Release planning 08 — precedents, contrast retrieval and the decision complex, on discovered coordinates

**Status: DRAFT, 2026-11-25. Drafted by Claude at the owner's request. Nothing below is
implemented except the experiment harness in `experiments/precedent/`.**
**Predecessors:** release-planning-05 (Social Chem grounding), -06 (framework-pluralist architecture),
-07 (graph consumer migration).
**Companion files:** `experiments/precedent/PREREG-precedent-v1-DRAFT.md` (the test plan),
`experiments/precedent/REHEARSAL.md` (what was run on 2026-11-25),
`geometric-ethics/notes/2026-11-25-hnsw-and-stratified-geometry.md` (the survey and analysis this follows).

## 1. What this release is for

The owner asked whether a Hierarchical Navigable Small World (HNSW) index should replace or join the
stratified geometry. The analysis concluded that HNSW is an index and not a model, and that three
things are missing from the stack and worth building:

1. a **case base** of decided cases and retrieval of **precedents** from it;
2. **contrast retrieval**, the nearest case on the other side of a named gate, which gives the first
   estimator of the jump across a boundary (Geometric Ethics §8.5, Definition 8.11);
3. the **decision complex** of Geometric Ethics §8.10a and geometric-methods Definition 6.1 as code.
   It exists today only as text.

The owner then set two directions on 2026-11-25:

- coordinate the upgrade across `moral-spectrum-analyzer` and the related repositories;
- move past the nine hand-made dimensions to a **discovered** coordinate set, such as the 768
  dimensions of LaBSE.

So every new component here reads coordinates through one interface, and that interface is an encoder
embedding with a learned metric. The nine axes stay as named read-outs for audit and for the standards
crosswalk. They are not the space the geometry is computed in.

## 2. What exists today (scan of 2026-11-25)

| Piece | Where | State |
|---|---|---|
| Executable strata and gates | `runtime/strata.py` | Shipped. States are labels, edges are gates. No weights, no metric. Containment and absorption are checked at scene load. |
| Per-act consequences | `observation/options.py` (`OptionProjector`, `option_ir`) | Shipped. One IR per candidate act, judged by DEME. One step ahead only. |
| Typed situation graph | `ir/graph` (`MoralGraph`) | Shipped. Edges include `consents_to{duress}`, `coerces`, `holds_commitment`. These are the gate features. |
| Decided-act history | `history/sqlite_store.py`, erisml-lib `strategic._decision_history` | Append-only. No lookup by similarity. |
| Learned metric | `geometric-economics/experiments/papers/learned/fit_learned.py` | Sealed test run 2026-10-30. A low-rank factor `U` over LaBSE beat the nine-axis codebook on Scruples. The selected rank was 32, the top of the grid. |
| Discovery of coordinates | `moral-spectrum-analyzer/docs/DISCOVERY_ROADMAP.md` | Registered 2026-07-12, not run. Residual directions screened by stability, replication and non-redundancy. |
| HNSW index | `turboquant-pro` `CompressedHNSW` | Shipped. No `search` takes a filter. |
| "Tier 1 Mahalanobis evaluator" | named in `tiers.py` | No function implementing a Mahalanobis cost was found in `src/`. |
| Decision complex, A* search | none | Not found as code in any scanned repository. |
| Case or precedent retrieval in DEME | none | Not found. |

## 3. Design rules

Each rule comes from the analysis note or from a result of 2026-11-25.

1. **Two distances on one set of cases.** The *retrieval distance* is the learned metric with the gate
   features left out. The *action distance* is the same metric plus a penalty for each gate crossed,
   infinite for forbidden crossings. A contrast pair is close in the first and far in the second.
2. **One gate-blind index, gates checked afterwards.** A hard filter inside a graph index can
   disconnect it. Contrast retrieval also needs to cross gates on purpose.
3. **Retrieval is advisory.** A retrieved precedent is a learned reading. By Theorem 8.6 it may never
   fire a gate into an authority stratum and may never lift a veto. This is enforced when a scene
   loads, as containment is today.
4. **A precedent transfers only when its set of relevant differences on gate features is empty**
   (Prakken 2021). An unknown gate value is a difference.
5. **Exact search first.** At the sizes on disk (about 150 thousand cases) an exact scan is fast.
   HNSW enters only above a stated size, behind the same interface, and is accepted on recall against
   the exact result.
6. **Index shortcuts are never action edges.** Action edges are local. The connection radius must grow
   relative to the sample spacing: a grid with fixed connectivity keeps an 8.2% path-length error at
   every resolution, measured on 2026-11-25 (`geometric-ethics/notes/checks-2026-11-25/results.json`).
7. **Every learned artifact carries its origin and the consumer refuses foreign origins.** A coordinate
   system or a case base built from simulated data lives at a different path and is refused in strict
   mode.
8. **Fail closed.** No precedent close enough means "no precedent", which is reported, and the
   decision proceeds as it does today.
9. **What is built next depends on what the test finds.** Section 6 ties each phase to a hypothesis
   of the registered test.

## 4. Components

### 4.1 erisml-compiler

**C1. `coords/`, the coordinate system.**
- `CoordinateSystem`: encoder name and snapshot, the label-free projection (mean, components, scale),
  one metric factor `U` (pooled) and optionally one per stratum, the data it was fitted on, a digest.
- Stored as `.npz` plus `.meta.json`, the format `fit_learned.py` already writes (`U`, `r`), so the
  sealed 2026-10-30 artifact can be loaded as the first one.
- `distance(x, y, stratum=None)` and `project(e)`. This is the one place a Mahalanobis cost is
  computed, and it gives `tiers.py` a function behind its claim.
- Strict mode refuses an artifact whose meta does not declare a sealed registration or that declares
  `simulation: true`.

**C2. `casebase/`, the case store.**
- A case: id, `graph_hash`, `ir_hash`, embedding reference (by coordinate digest), gate features
  (name → value, with `unknown` a value of its own), strata snapshot, the act, the verdict, the
  decision-proof hash, source (`corpus`, `own_decision`, `human_reviewed`, `simulated`), sampling unit.
- SQLite in WAL mode, as `history/sqlite_store.py`, with vectors in a sidecar array.
- A digest over the sorted records. Appending changes the digest and nothing else does.
- Simulated cases go to a separate store file. The production store refuses them.

**C3. `ontology/gates.yaml`, the gate registry.**
- One entry per named gate: where it is read in the `MoralGraph` or the scene strata, its boundary
  type (threshold, phase, absorbing, constraint), whether it is an authority gate, and its penalty
  (a number, `inf`, or `unmeasured`).
- First entries: consent (given, refused, under duress, unknown), promise or commitment (active,
  released, violated), coercion, emergency, the three nullifiers (abuse, danger, impossibility).
- The registry is a normative choice kept in one place, as `DUTY_RELATIONS` is.

**C4. `retrieval/`.**
- `Retriever` protocol with `ExactRetriever` (NumPy) and, behind the `[ann]` extra, `HNSWRetriever`
  on turboquant-pro's `CompressedHNSW`.
- `precedents(query, k)`: nearest cases under the retrieval distance, each with its distance and its
  relevant differences. Those with an empty difference set are marked transferable.
- `contrast(query, gate)`: the nearest cases that differ from the query on that gate and on no other.
- `abstain` when the nearest transferable case is beyond a radius calibrated on held-out data.
- A retrieval-stability figure (overlap of the retrieved set across paraphrases of the query) is
  computed by the same module and reported with every evaluation.

**C5. IR and audit.**
- `ir.precedents`: retriever kind, coordinate digest, case-base digest, the retrieved ids with
  distances and differences, the contrast pairs, the abstention flag. IR schema v0.5.
- The audit bundle records both digests, so a decision can be replayed against the same case base.

**C6. Runtime.**
- `SceneAgent` may show transferable precedents to the chooser as structured fields. No retrieved
  prose enters a prompt.
- A new event source class `retrieval`. The load-time check that today rejects a non-system event
  leading into an authority stratum rejects this class too. Tests drive the guard to fire.

**C7. `complex/`, the decision complex (research phase).**
- Vertices are states in the coordinate system. Edges are acts with a projected consequence, from the
  option projector. Edge weight is the square root of the metric form plus the gate penalties, as in
  geometric-methods Definition 6.1.
- Dijkstra first. A* with landmark lower bounds second (Goldberg and Harrelson 2005), which are
  admissible only when the stored costs are true shortest-path costs in this same graph.
- A separate bound on the number of crossings, if wanted, is a different and harder problem
  (resource-constrained shortest path). The additive-penalty form is ordinary shortest path, checked
  on 2026-11-25.
- This is the route to real semantics for the `a` axis, which is a stub today.

**C8. Bench.** MoralTensor-Bench gains the gate-flip categories its manifest already plans
(`coerced_consent`, `vow_paraphrases`, `authority_legitimacy_swaps`), sharing items with the
registered test's flip set once that test is graded.

### 4.2 erisml-lib

- **`PrecedentEM`**, an `EthicsModuleV2`, advisory (`vetoes=False`), tag `precedent`. It reports the
  transferable precedents' verdicts as one moral reading with its uncertainty. It fits the existing
  composition stages with no pipeline change.
- **`DecisionProof`**: a provenance record for the case base and the coordinate system, modelled on
  `feeder_provenance`.
- **`MoralVector.distance`** gains an explicit metric argument. Today it is unweighted.
- **`defaults/ground_state_loader.py`**: the constant `GROUND_STATE_VERSION` is `"1.0.0-synthetic"` and
  the JSON it mirrors (`sqnd-probe/ground_state/dear_abby_ground_state_v1.json`) says "Synthetic
  derivation based on moral philosophy patterns ... Replace with empirical data", while the module
  docstring says the values were derived from 20,030 letters. The docstring should say what the
  version string says, and a caller asking for empirical defaults should be refused until measured
  values exist. The registered test's flip family supplies the first measured gate effects.

### 4.3 moral-spectrum-analyzer

- **Same coordinate system.** The discovery band and the retrieval layer load one `CoordinateSystem`
  artifact. A direction that passes the roadmap's screens becomes a *named read-out* of those
  coordinates, with its gate record. It does not become a new axis of a hand-made vector.
- **The anomaly module is the abstention radius.** The roadmap calls the Mahalanobis anomaly module
  "the region-selector in production form". It and C4's abstention use the same metric and the same
  calibrated radius. Build it once.
- **Escalations feed the case base.** A decision the analyzer escalates, once a person has ruled on
  it, becomes a `human_reviewed` case. This turns the escalation stream into precedent.
- **Paraphrase classes feed the stability figure.** The invariance suites already hold
  equivalence classes of rewordings. C4 scores retrieval overlap on them.
- **Audit proof.** The analyzer's re-verifiable proof records the coordinate and case-base digests.
- The dependency pin `erisml-compiler>=0.9.0` moves to the release that ships C1 to C5.

### 4.4 xbse

- The feeders stay what they are: validated, named read-outs with reliability weights. They are
  labels on the discovered coordinates and are not the coordinate system.
- One addition: a pinned-snapshot embedding export, so the coordinates, the feeders and the case base
  are computed from the same encoder revision. The digest records it.
- Reliability weights do not enter the metric. They weight read-outs.

### 4.5 turboquant-pro (only if the case base outgrows exact search)

- A single-area query on `StratifiedIndex`, or an allow-list argument to `CompressedHNSW.search`, so a
  per-stratum search does not need a second index type.
- Acceptance is recall at k against the exact result on the case base, never reconstruction cosine.

### 4.6 geometric-economics

- `fit_learned.py` is the reference metric fitter. C1 reads its output. The registered test here uses
  a different loss (kernel regression on judgements) because its data are single judgements, not
  paired choices, and the draft says so.
- The `paths/` line and C7 share Definition 6.1. They should share the edge-weight code once C7
  exists.

### 4.7 geometric-ethics (the book)

Four corrections, three now backed by a run (Section 8 of the analysis note, and
`notes/checks-2026-11-25/results.json`):

1. MORAL-PATHFIND as defined is ordinary shortest path. 300 of 300 random instances agreed with
   Dijkstra. NP-hardness needs a separate budget.
2. Proposition 8.5(i): fixed-connectivity grid paths do not converge to geodesics. The error stayed
   at 8.24% from n = 50 to n = 400 and fell to 0.19% only when the connection radius grew with n.
3. Proposition 8.1, Step 2: condition (A) fails for the Whitney umbrella with the z-axis as one
   stratum. The theorem gives a refinement.
4. The gate rates of §8.4 and §17: in the files examined, the rates are literals. See 4.2.

geometric-methods Chapter 6 makes none of claims 1 to 3, so it needs no change.

### 4.8 Techniques taken from the literature (survey of 2026-11-25)

The survey read abstracts and proceedings pages, not full texts. Effect sizes and theorem conditions
must be checked before any of these is cited in a registration. "[V]" means the citation was confirmed
this session.

**Adopted.**

| # | Technique | Where it goes | Why | Source |
|---|---|---|---|---|
| 1 | A calibrated escalation threshold on the distance to the k-th transferable precedent, chosen by Learn then Test so that the error rate among non-escalated cases exceeds α with probability at most δ | C4 abstention, and the analyzer's anomaly module | Gives "no precedent is close enough" a stated error rate. The proof records τ, α, δ and the calibration-set hash. The guarantee is marginal and assumes new cases resemble the calibration cases. | Angelopoulos et al., arXiv 2110.01052 [V] |
| 2 | Local scaling of distances at re-rank time | C4, most of all contrast retrieval | A hub case reused as the contrast for many queries shrinks the effective sample. No training. | Schnitzer et al., JMLR 13, 2012 [V] |
| 3 | LEACE erasure of each gate label from the embedding before the metric is fitted | C1 | Makes the retrieval distance linearly gate-blind by construction. Rule 1 of Section 3 only asked for this. | Belrose et al., arXiv 2306.03819 [V] |
| 4 | A shared metric plus a penalised per-stratum part, U_s = U_0 + ΔU_s | C1, and the per-stratum arm of the test | Thin strata fall back to the shared metric smoothly. The penalty weight is fixed in the registration. | Parameswaran and Weinberger, NIPS 2010 [V] |
| 5 | A cross-fitted doubly robust estimate of the jump across a gate, with contrast retrieval as its explanation | C4, and the book's ΔS | Nearest-neighbour matching in a learned multi-dimensional metric has no supporting theory, and the bootstrap is invalid for matching estimators. | Chernozhukov et al., arXiv 1608.00060 [V]; Abadie and Imbens 2008 [V]; Gui and Veitch, arXiv 2210.00079 [V] |
| 6 | Contrast sets and CheckList tests, scored by contrast consistency, with rank-biased overlap and Jaccard at k for retrieval stability | The test's flip family, C8 | One-gate minimal edits, and a top-weighted overlap measure. | Gardner et al., arXiv 2004.02709 [V]; Ribeiro et al., arXiv 2005.04118 [V]; Webber et al., TOIS 2010 [V] |
| 7 | Search over the product of case and stratum state, returning the Pareto set over (metric cost, crossings) | C7 | Removes hand-set penalties from the search. A separate budget on crossings is handled by weight-constrained A*. | Toro Icarte et al., arXiv 2010.03950 [V]; Hernández Ulloa et al., ICAPS 2020 [V]; Ahmadi et al., SoCS 2022 [V] |
| 8 | Explicit Estimation Search: a learned estimate orders the search and the landmark bound certifies the result | C7 | Lets an inadmissible heuristic guide without losing the bound. | Thayer and Ruml, IJCAI 2011 [V] |
| 9 | Weisfeiler-Lehman kernel as a structural second stage, and a graph edit path as the list of relevant differences | C4 | The situation graphs are small. The edit path is the difference set rule 4 needs. | Shervashidze et al., JMLR 12, 2011 [V]; Riesen and Bunke 2009 [V] |
| 10 | Extend the rank grid past 32 before fixing a rank | C1 | The sealed fit chose the top of its grid, so "rank 32" is a statement about the grid. | Jain, Mason, Nowak, arXiv 1709.06171 [V] |

**For later, as research.** Defeasible deontic logic for gate priority and contrary-to-duty rules
(Governatori et al. 2013 [V]; the normative-supervisor architecture of Neufeld et al. 2021 [V] is the
closest published design to strata plus DEME). Containment would have to be proved again in it.

**Not adopted.**

- Sparse-autoencoder latents as coordinates or as gate features. Reported overlap across seeds was
  about 30% in one large study (Paulo and Belrose, arXiv 2501.16615 [V]) and parent features can fail
  to fire (Chanin et al., arXiv 2409.14507 [V]). They remain useful for naming directions.
- Nearest-neighbour matching with bootstrap intervals as the reported jump estimator. See row 5.
- Neural A*, GraphRAG summarisation, whitening on top of a learned metric, control barrier functions.
- Regression discontinuity for binary gates. It needs a continuous running variable, so it applies
  only to graded-ladder data.

**Data found for the test's families** (all citations confirmed, contents not opened): NormBank
(arXiv 2305.17008) and δ-Rules-of-Thumb (arXiv 2310.15431) for one-feature flips, MoCa
(arXiv 2310.19677) and OffTheRails (arXiv 2404.10975) for factor-wise minimal pairs with human
judgements, MultiTP (arXiv 2407.02273) for graded ladders across languages, ARN (arXiv 2310.00996) and
StoryAnalogy (arXiv 2310.12874) for cross-domain analogy. No published consent-manipulation set was
found.

## 5. Security and trust, designed in

| Threat | Control |
|---|---|
| A poisoned case steers later decisions | Cases carry source and proof hash. Only `corpus` (with a registered digest), `human_reviewed` and proof-chained `own_decision` cases are retrievable. The store refuses a case with no source. |
| A retrieved text injects instructions into the chooser | Only structured fields of a case reach a prompt, never its prose. |
| A precedent grants authority | Event source class `retrieval` can never lead into an authority stratum. Checked at load. |
| A rehearsal artifact is used for real | Simulated artifacts live at other paths and declare themselves. Strict mode refuses them. |
| A decision cannot be replayed | Coordinate and case-base digests are in the IR and the proof. |
| Cases from real users leak | The store holds graph features and hashes by default. Raw text is optional and off by default. |
| The index silently returns worse neighbours | The approximate backend is accepted on measured recall, and the retriever kind is in the IR. |

## 6. Phases, and what each waits for

| Phase | Work | Waits for | Repositories |
|---|---|---|---|
| P0 | Book checks. Rehearsal of the registered test on planted truths. | Done 2026-11-25. | geometric-ethics, erisml-compiler |
| P1 | Stage 0 of the test: compile the cases, measure gate-extraction agreement, write the flip set, seal, run. | Owner's OK to seal. An NRP or Atlas slot for extraction. | erisml-compiler, geometric-ethics ledger |
| P2 | C2, C3, C4 (exact), C5, C6. Advisory only. | Nothing. Useful whatever the test finds. | erisml-compiler, erisml-lib (`DecisionProof`) |
| P3 | `PrecedentEM` with the gate check. Bench flip categories. | H1 passes (gates beat similarity on flips). | erisml-lib, erisml-compiler |
| P4 | C1 with the learned metric as the retrieval distance. Analyzer and xbse wiring. | H2 passes (a learned metric adds prediction beyond gates). If H2 fails, retrieval uses whitened cosine and the metric stays out of the product. | erisml-compiler, moral-spectrum-analyzer, xbse |
| P5 | One metric per stratum. | H3 passes. | erisml-compiler |
| P6 | C7, the decision complex and search. | P4, and act projections that chain. | erisml-compiler, geometric-economics |
| P7 | HNSW backend. | A case base above about one million. | turboquant-pro, erisml-compiler |

Versions: P2 is erisml-compiler 0.11 with IR schema v0.5 and erisml-lib 3.3. P4 is 0.12.

## 7. Tests for the code

- **Unit**: distance symmetry and non-negativity, digest stability, difference sets.
- **Metamorphic**: renaming gates or strata changes no retrieved set. Re-ordering the case base
  changes no result.
- **Containment**: a scene with a `retrieval`-sourced gate into an authority stratum fails to load,
  and the error names containment.
- **Provenance**: the strict loader refuses a simulated artifact, and the refusal names provenance.
- **Recall**: the approximate retriever against the exact one on a fixed case base.
- **Determinism**: same case base and coordinates give the same `ir.precedents`.
- **Self-test of the gate**: each guard above is run once against a known-bad input and must fail.

## 8. Decisions that are the owner's

1. Whether to seal the test (`PREREG-precedent-v1-DRAFT.md`) as drafted.
2. Which corpus supplies the confirmatory cases and which is held back unseen.
3. Who writes the consent flip set. It must be someone who does not know the arms.
4. Whether the first coordinate artifact is the sealed LaBSE metric of 2026-10-30 or a new fit.
5. Whether the book corrections go into v1.26 now or wait for the test.
6. Whether `erisml-compiler` work lands on `main` while `feat/extractor-relations-and-duties` is open
   (it is two commits ahead of `main` and unmerged as of 2026-11-25).
