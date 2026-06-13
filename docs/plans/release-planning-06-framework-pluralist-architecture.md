# Release planning 06 — Framework-pluralist architecture

**Status:** Option C (two-layer IR) landed in v0.x. The user chose to
take the full architectural fix rather than the Option-A patch.
**Owner:** TBD.
**Estimated effort:** landed in one session (substrate + projections
package + Consequentialist + Deontic with universalizability +
mere-means + consent + legitimacy gates + cross-projection
disagreement surface).
**Predecessor:** r/Compiler review thread, 2026-06-12.

## Update — DAG-native substrate landed (commit follow-up)

The two-layer (substrate + projections) refactor was the first step.
The deeper move — making the substrate a typed graph instead of a
flat record — landed as a follow-up. The compiler now compiles
facts-about-the-world into a `MoralGraph` (typed DAG: nodes ∈
{stakeholder, act, maxim, commitment, fact, norm}, edges ∈
{performs, imposes_on, consents_to, holds_commitment,
commitment_binds, treats_as, under_maxim, coerces, surfaces_fact,
fact_subject, would_violate_if_universalised}). The graph has a
canonical SHA-256 hash (`audit.graph_hash`) — same input → bit-
identical hash regardless of insertion order.

Projections continue to read a `MoralSubstrate`, but the substrate
is now a *view over the graph*: `Maxim`, `ConsentState`, and
`AuthorityLegitimacy` are derived from graph queries
(`imposes_on` without paired `consents_to`, `treats_as[role=mere_means]`,
authority-labeled stakeholder nodes with legitimacy fact context)
rather than from flat-list heuristics. `DeonticProjection` accepts
the graph directly and pattern-matches `treats_as` edges when
present.

Side effect: the gates fire more *precisely*. Earlier, every
stakeholder labelled `nonconsenting_third_party` triggered a
mere_means failure even when the act wasn't actually imposing harm
on them. Now the gate requires an `imposes_on` edge as evidence.
`medical_confidentiality` and `whistleblower` correctly come out
`permissible` under the deontic projection (the act protects /
discloses, not imposes), while `nazi_attic` correctly stays
`forbidden`. The earlier "all 3 forbidden" result was the previous
implementation over-firing on a label proxy.

`tests/test_moral_graph.py` — 12 tests covering graph schema, query
helpers, canonical hash determinism, promotion from flat IR, and
hash reproducibility across compiles.

What still lives in the design queue (not yet landed):

- Extractors emit graph nodes/edges directly (today, rule extractor
  emits flat, orchestrator promotes at Pass 7.5)
- `flat_from_graph()` back-derivation so mutating the graph updates
  the flat fields automatically
- `ConsequentialistProjection` reading the graph (today still goes
  through the EM-DAG flat-field interface)
- Bench/monitor/RLEF/silicon emitters using graph queries

## What landed in the two-layer commit (recap)

- `src/erisml_compiler/projections/` subpackage:
  - `substrate.py` — `MoralSubstrate` Pydantic model + first-class
    `Maxim`, `ConsentState`, `AuthorityLegitimacy` + heuristic
    `substrate_from_ir(ir)` that derives these from the existing
    rule extractor's output.
  - `base.py` — `Projection` ABC + `ProjectionResult` + `GateFinding`
    (categorical pass/fail with severity).
  - `consequentialist.py` — `ConsequentialistProjection` wraps the
    existing tensor/DEME/Gini machinery; back-fills
    `ir.moral_tensor_v3`, `ir.deme_verdict`, `ir.per_party_verdicts`,
    `ir.fairness_metrics` for backward compat.
  - `deontic.py` — `DeonticProjection` runs four gates over the
    substrate: `universalizability`, `mere_means`, `valid_consent`,
    `legitimate_authority`. Verdicts: `permissible`, `requires_review`,
    `forbidden`.
- `CompilerIR` gains `projections: dict[str, Any]` and
  `cross_projection_disagreement: dict | None` fields. When ≥2
  projections disagree, the orchestrator surfaces both verdicts
  without aggregating.
- CLI: `--projection consequentialist_distributive,deontic_kantian`
  (default: both).
- `CompileOptions.projections` tuple controls which projections run.
- `tests/test_projections.py` — 12 tests covering substrate
  derivation, deontic gates, cross-projection surfacing, backward
  compat for the legacy IR fields.

## Initial findings on the bundled examples

Running with both projections enabled (`rule` extractor):

| Scenario | Consequentialist | Deontic |
|---|---|---|
| `nazi_attic` | `tragic_conflict_escalate` | `forbidden` (universalizability + valid_consent fail) |
| `medical_confidentiality` | `permitted` | `forbidden` (mere_means + valid_consent fail) |
| `whistleblower` | `permitted` | `forbidden` (mere_means + valid_consent fail) |

The two frameworks meaningfully disagree on every bundled case. The
compiler surfaces both verdicts via `cross_projection_disagreement`
and refuses to choose between them — choosing is itself a
metaethical move and is now explicitly deferred to the caller.

## Known limitations (carries from the original design)

The Option-C landing is the *architecture*, not a finished Kantian
analyser. Honest limitations of the v0 implementation:

1. **Maxim extraction is heuristic.** v0 derives `Maxim.action_kind`
   from the dominant `ethical_fact.kind`. Real Kantian analysis
   needs maxim extraction directly from prose (likely an LLM
   step). The current approach catches obvious deception/coercion
   maxims but misses subtler ones.
2. **Universalizability check is rule-list-based.** Two action
   kinds (`deceive`, `impose_externality`) trigger the gate. A
   proper universalizability test builds the universalized-world
   model and runs a contradiction test against it. Out of scope
   for v0.
3. **`mere_means` defaults to "any non-consenting third party is
   being treated as means."** Strong reading, possibly over-firing.
   The proper test depends on whether the agent's maxim genuinely
   ignores the affected party's status as a self-determining end.
4. **Bench harness doesn't yet score per-projection.**
   MoralTensor-Bench (release-planning-03) scores against the
   legacy single-verdict path. Should be extended to score each
   projection separately + the cross-projection-disagreement rate.
5. **Monitor / RLEF / silicon emit haven't been ported yet.** They
   still read the consequentialist projection's output via the
   back-fill fields. Future work to make them per-projection.
6. **Virtue ethics projection not yet implemented.** Three would be
   the natural triple; v0 ships with two.

## Out of scope after this landing

The original Option-A / B / C taxonomy is resolved (we took C).
What remains is fill-in work, not architectural choices:

- Richer Kantian gate implementations (true universalizability test,
  better mere-means test, maxim extraction from text)
- `VirtueProjection` (character-trait/habit consistency analysis)
- Per-projection bench scoring + bench gold for non-consequentialist
  verdicts
- Per-projection monitor / RLEF / silicon paths
- A *care-ethics* projection (Held / Noddings tradition) — different
  primitives again (relational webs, particularised attentiveness)

## The challenge

A reviewer raised a substantive objection:

> "Saying the system is neutral because it lets the user swap in
> `--em-profile kantian-deontological` vs `--em-profile dear-abby-baseline`
> doesn't make the architecture neutral, it relocates the substantive
> commitment one level up from 'which verdict' to 'which axes exist to
> be weighted in the first place'. The choice of different ethical
> dimensions, the decision that per-stakeholder harm is the correct
> unit of account, the decision that a Gini coefficient over the harm
> distribution is a meaningful summary of 'how unequally the cost lands'
> is not neutral. These are first-order ethical commitments (broadly
> consequentialist/distributive ones) packed into the IR itself, before
> any profile is chosen."
>
> "Matter of fact, plugging a 'kantian-deontological' profile into this
> is mechanically incoherent, because a Kantian framework doesn't operate
> on harm magnitudes at all. It asks a categorically different question
> such as 'can the maxim behind this action be universalized' or 'does
> this action treat any person merely as a means'. A Kantian profile
> isn't just underweighted by this IR, it's inexpressible in it."

The reviewer is correct on both points.

## Concession

The IR commits, **before any profile is chosen**, to:

- per-stakeholder accounting as the basic unit (the `n` axis of the
  rank-2 tensor and beyond)
- a fixed set of 9–10 dimensions as the channels morality cashes out in
- distributional aggregates (Gini, Shapley, worst-off) as meaningful
  summaries of "how the cost lands"

That is a pluralist-consequentialist-with-deontic-side-constraints
stance, not framework neutrality. The earlier `--ethos-profile` framing
("named ethos, never implicit default") overclaimed. The README now
documents this honestly in **Current limitations**.

The Kantian-specific objection has two halves that deserve separate
answers:

**Half 1: "Kantian ethics doesn't operate on harm magnitudes."** True at
the level of justification. But the current IR isn't as purely
magnitudinal as it looks. `autonomy_consent`, `legitimacy_trust`, and
`vow_fidelity` are non-magnitudinal dimensions doing dignitarian/deontic
work: `autonomy_consent` *is* the consent-violation check;
`legitimacy_trust` *is* the procedural-justification check;
`vow_fidelity` *is* commitment-tracking. These channels saturate at ±1
and can trip categorical responses in DEME (a void-consent finding can
override harm aggregates via the `veto_location` mechanism). So the IR
already has deontic side-constraints — just not labelled as such.

**Half 2: "A Kantian profile is inexpressible."** Partly right, partly
fixable.

  - A profile that zeros the magnitudinal channels and weights only the
    deontic ones would mechanically run today and would behave more
    Kantianly than the current default.
  - But the *structural tests* Kant cares about — universalizability of
    the maxim, treating persons as mere means — aren't EM modules.
    Those aren't profile-weight tunings; they're missing modules with
    different output semantics (gate-style, categorical pass/fail)
    than the dimension-score modules currently in the EM-DAG.

## Three architectural options

### Option A — Status-quo plus deontic modules (incremental)

Add `UniversalizabilityEM` and `MeansEndsTestEM` as EM-DAG modules.
Their outputs are *gate-shaped* — categorical findings that can fire
`veto_location` in DEME — rather than continuous channel
contributions.

- **Pros:** Fits the existing EM-DAG/DEME architecture cleanly. ~3 days
  per module. Closes the most visible gap.
- **Cons:** Doesn't actually solve the deeper problem. Adding more
  channels still treats them as channels in the same IR shape. A
  Kantian could fairly say: "you're treating universalizability as one
  module among ten that get aggregated, when for me it's *the entire
  test*." The aggregation step itself is the consequentialist move.

### Option B — Multiple coexisting IRs per framework

Have a `ConsequentialistIR`, a `DeonticIR`, a `VirtueIR`, each with its
own primitives. Compile produces N IRs, one per framework selected.

- **Pros:** Honest. Each framework gets a representation faithful to
  its primitives.
- **Cons:** Massive engineering cost. N evaluators, N test suites,
  N audit chains. The comparison/aggregation across IRs is itself a
  metaethical choice. Worse: doubles or triples the surface area
  reviewers and users have to reason about.

### Option C — Two-layer IR: descriptive substrate + framework projections

The compiler bifurcates into:

  1. **`MoralSubstrate`** — descriptive layer. Stakeholders, acts,
     commitments-already-standing, consent states, who-imposed-what,
     authorities and their procedural standing, **maxims** (the action
     under its description, as a first-class object), counterfactuals,
     repair states. Framework-neutral *up to which categories of fact
     you choose to extract*. Still encodes choices ("we extract maxims
     because they're a real thing") but a much smaller commitment than
     "we aggregate per-stakeholder harm via Gini."

  2. **`Projection`** — multiple, framework-bound, pluggable. The
     current `MoralTensorV3` becomes one projection
     (`ConsequentialistDistributiveProjection`). A `DeonticProjection`
     reads the same substrate and outputs maxim-universalizability
     findings, mere-means findings, consent/legitimacy gates as
     categorical pass/fail. A `VirtueProjection` reads the substrate
     and outputs character-trait assessments and habit-consistency
     analysis. Comparison across projections is itself a metaethical
     move, surfaced explicitly in the report rather than hidden in an
     aggregation.

In compiler terms: today we have one parse tree that's also the
type-checker output. We should have a parse tree and N type checkers,
each honest about its framework.

- **Pros:** The correct architecture for a project that genuinely
  wants to support framework pluralism. Makes the IR's commitments
  smaller (substrate-level) and labels what's framework-bound as such.
  Pluralism becomes a real feature instead of marketing.
- **Cons:** Big refactor. Touches the IR schema, the pipeline
  orchestrator, all the V2/V3 builders, the audit chain (each
  projection has its own audit), the CLI, the bench harness, the
  monitor. Probably a v1.0 milestone, possibly v2.0.

## Recommendation

**Long-term: Option C.** It's the only architecture that can honestly
support the framework pluralism the project claims.

**Short-term (v0.x): Option A as a partial fix**, with explicit
acknowledgment in the README that this is a patch, not a solution.
Specifically, add two gate-style EM modules:

```python
class UniversalizabilityEM(EthicalModule):
    """Reads `ir.maxim` (an action-description object the extractor
    surfaces) and produces a categorical finding: is the maxim
    universalizable? Fires `veto_location=("universalizability",)` in
    DEME when not."""

class MeansEndsTestEM(EthicalModule):
    """Reads `ir.stakeholders` and the act-target relations and
    produces: does this action treat any rational agent merely as a
    means? Fires `veto_location=("mere_means", party_idx)` when so."""
```

These modules have a different output contract than the existing
dimension-score modules. They emit a `GateFinding` (a Pydantic model
with `passed: bool`, `reason: str`, `severity: Literal[...]`)
alongside the standard `EMOutput`, and DEME treats their failure as a
categorical input — not a number to be averaged.

Adding them does not invalidate the critic's point. The architecture
still has metaethical commitments at the substrate level. But it
moves us from "Kantian is inexpressible" to "Kantian's central tests
are now expressible as gates, even though the surrounding accounting
is still framework-bound." Honest partial credit.

## What this isn't

- **Not a v0.x deliverable**. Option C is a long-term refactor.
- **Not a one-evening fix even at Option A**. The gate-module output
  contract has to be wired through DEME's veto-handling, the audit
  chain, the bench scoring (gate failures need their own metric), and
  the I-EIP Monitor (gate findings need their own delta-comparison
  shape).
- **Not a defense against the critic's substantive point**. The
  architecture remains metaethically committed. Adding gate modules
  closes the most visible specific gap; it doesn't change the
  underlying commitment.

## Headline answer to the critic

> You're right. The IR has metaethical commitments below the profile
> layer. The honest claim isn't framework neutrality — it's
> structure-preservation against premature scalar contraction, within
> a pluralist-consequentialist-with-deontic-side-constraints frame.
> The Kantian central tests (universalizability, mere-means) are
> *partially* expressible today via autonomy_consent / legitimacy /
> fidelity doing dignitarian work, but the named structural tests
> aren't modules yet. Adding them as gate-style EMs is the v0.x patch.
> The actual fix is a two-layer IR — descriptive substrate plus
> pluggable framework projections — and that's a v1.0+ refactor.

## Open questions

**A. Maxim extraction.** Option C presupposes that "the maxim of the
action" is extractable from natural language. That's an open empirical
question. Some actions wear their maxim on their sleeve ("I lied to
protect them"); many don't.

**B. How many projections? Which ones?** Consequentialist, deontic,
virtue is the obvious triple. Care ethics, contractualism, divine
command, natural-law — at what point does "framework pluralism"
become "framework menu paralysis"?

**C. Comparing across projections.** When the consequentialist
projection says "permitted" and the deontic projection says "forbidden,"
what does the compiler output? Refuse to aggregate? Surface both?
Defer to a meta-policy? This is itself a metaethical question.

**D. Bench across projections.** MoralTensor-Bench (release-planning-03)
currently scores only the consequentialist projection. A truly
framework-pluralist bench would need framework-specific gold per
scenario — orders of magnitude more curation work.

## Milestones (Option A, v0.x patch)

| Phase | Deliverable | Effort |
|---|---|---|
| 06-1 | `GateFinding` Pydantic model + EM-DAG output contract extension | 1 day |
| 06-2 | `UniversalizabilityEM` — requires `ir.maxim` extraction (currently absent) | 2-3 days |
| 06-3 | `MeansEndsTestEM` — reads existing `ir.stakeholders` + relations | 1 day |
| 06-4 | DEME wiring: gate findings produce `veto_location` entries | 1 day |
| 06-5 | Bench metric: gate-coverage rate (scenarios where Kantian gates fire) | 0.5 days |
| 06-6 | README + design-note updates documenting the partial-fix nature | 0.5 days |

Total: **~6-7 working days** for Option A. Option C is its own
multi-month roadmap.

## Suggested follow-up note

`release-planning-07-substrate-projection-refactor.md` should sketch
Option C in implementation detail. Out of scope for this note.
