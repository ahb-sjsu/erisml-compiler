# Release planning 05 — Dear Abby corpus as a named ethos profile

**Status:** design only.
**Estimated effort:** 3–4 weeks once licence + corpus access decided.
**Predecessor:** none formal; raised in conversation after
[release-planning-04](./release-planning-04-eigenvalue-scalar.md).

## The thesis

Every ethical evaluator has an implicit ethos somewhere. The
compiler's EM-DAG today uses `em_dag/profiles/default.yaml` with
hand-set weights — i.e. *"whatever the maintainer thought reasonable
in 2026"*. That's a position, but it's an invisible one. The honest
move is to make the ethos a **named, fitted, swappable artifact**,
so callers can pick (and audit) which moral baseline the compiler
applies.

**Dear Abby** (Pauline Phillips 1956–2002, Jeanne Phillips 2002–) is
a candidate first corpus because:

- ~22,000 columns; structured (reader letter → columnist advice)
- Single columnist's voice for decades → self-consistent ethos
- Everyday scenarios — family, work, romance, money, friendship —
  exactly the cases the moral mathematics is supposed to apply to
- Implicit per-party verdicts on most letters
- Documented mainstream-American mid-to-late-20C ethos: the bias is
  legible, which makes it *useful* as a named baseline rather than
  a problem

## Two artifacts, not one

The corpus gives us two distinct things:

### Artifact 1: A named ethos profile

`em_dag/profiles/dear_abby_baseline.yaml` — fitted EM module weights
+ priors that minimise disagreement with the columnist's verdicts
across the corpus. Selectable via:

```bash
eris-compile compile some_scenario.txt --em-profile dear-abby-baseline
```

The profile itself is small, MIT-licensable, and ships in the repo.
It contains:
- per-module aggregation weight
- per-module veto thresholds
- per-dimension prior over signed magnitudes
- corpus fingerprint hash (so reviewers can verify provenance)
- date range + sample count + an explicit `ethos_description:`
  block ("mid-to-late 20th-c mainstream American advice
  columnist; heteronormative; class-coded; child-centric...")

### Artifact 2: A calibration corpus for the Tier-2.5 Probe extractor

Per [release-planning-01 item 2](./release-planning-01.txt) and
the `monitor/provenance.py` work already shipped, the probe
extractor's `CalibrationProvenance.probe_training_corpus_hash` is
currently always null because we have no training corpus. Dear
Abby is one — `(text, gold per-party verdicts)` pairs, where the
gold verdicts come from a parsed and validated projection of the
columnist's advice.

The probe checkpoint would then hash to a known Dear Abby corpus
fingerprint; the `CalibrationProvenance.notes` records "Dear Abby
1996–2010, n=4837 letters, IRR=0.78".

## Three concerns, addressed honestly

### 1. Licensing

Columns are copyrighted (Andrews McMeel / Universal Uclick
syndication). The corpus itself cannot be redistributed in any form
that reproduces the column text. What we *can* do:

- Use the corpus for training. Fair use for transformative academic
  purposes is plausible; consult the university's licensing office.
- Ship fitted artifacts: weights, profiles, probe checkpoints. These
  contain no column text.
- Ship the corpus *fingerprint* — sample count, date range, hash of
  the canonical-form deduplicated set, schema version of the
  letter→verdict projection.
- Ship a small public-domain *substitute corpus* for reproducibility
  — perhaps the Library of Congress's Aesop's Fables ethical
  scenarios, or 19th-century etiquette manuals. Smaller, weaker
  signal, but unblocked.

The compiler ships *both* fitted profiles: `dear-abby-baseline.yaml`
(real corpus, license-restricted training) and
`public-domain-baseline.yaml` (smaller substitute, MIT-licensable
end-to-end).

### 2. Bias is the point, not the problem

Dear Abby's ethos is observably:
- middle-class American
- heteronormative
- child-centric (family stability heavily weighted)
- white-collar work norms (loyalty, propriety)
- mainstream-Christian-secular ("be a good person, don't make
  waves")
- mid-to-late 20th-century

Using it as **the default** would smuggle those assumptions into
every uninstrumented evaluation. The fix is structural: ship multiple
named profiles, never set one as the default, require `--em-profile`
to be explicit, and document each profile's biases in its YAML
header.

Companion profiles to fit alongside Dear Abby:

| Profile | Source | Ethos label |
|---|---|---|
| `dear-abby-baseline` | Dear Abby 1996–2010 | American advice-column mainstream |
| `kantian-deontological` | Kant 1785, Korsgaard | Deontological / duty-first |
| `utilitarian-mill` | Mill 1863, Singer 1979 | Welfare-maximisation |
| `confucian-relational` | Analects + neo-Confucian texts | Relational / role-based |
| `aaai-ethics-board` | AIES 2018–2025 proceedings | Contemporary AI-ethics |
| `public-domain-baseline` | Substitute corpus | Unblocked baseline |

When the user runs `eris-compile compile` without `--em-profile`,
the compiler errors with a clear message listing the available
named profiles. **No implicit default.**

### 3. Verdict labels are noisy

Abby's advice is rarely binary. The projection from advice paragraph
to per-party verdict needs to be:

- **explicit** — defined per-stakeholder, not a single scalar
- **validated** — inter-rater reliability on a held-out subset
- **two-pass** — first an LLM-driven projection (NRP `gpt-oss` or
  Atlas-hosted Qwen2.5), then a human-reviewed sample (~200 letters)

The projection schema:

```yaml
letter_id: 1998-03-15-001
raw_letter: |
  ...the reader's question...
raw_advice: |
  ...the columnist's response...
projection_method: gpt-oss-v1
projection_timestamp: ...
per_party_verdicts:
  reader: prefer_alternative_action       # rationale: "Abby explicitly
                                          # suggests calling the
                                          # daughter instead of
                                          # confronting in person"
  reader_husband: neutral
  daughter: prefer_being_called
  third_party_grandchild: prefer_stability
overall_advice_sentiment: prefer_alternative
confidence: 0.78
reviewed_by_human: false
```

A scenario survives into the fitting corpus only when
`confidence ≥ 0.7` and (for the held-out subset)
`reviewed_by_human=True`.

## Implementation outline

### Phase 1: Corpus acquisition + projection schema

| Step | Deliverable | Effort |
|---|---|---|
| 1.1 | License consultation (university IP office) | 2 days |
| 1.2 | Corpus acquisition (Universal Uclick API or archive) | 1–3 days |
| 1.3 | `dear_abby/schema.py` — Letter + ProjectedVerdicts Pydantic models | 1 day |
| 1.4 | LLM projection pipeline using Atlas-hosted gpt-oss + IR fragments | 2 days |
| 1.5 | Human-validated golden subset (200 letters) for IRR | 3–4 days |

### Phase 2: Profile fitting

| Step | Deliverable | Effort |
|---|---|---|
| 2.1 | `dear_abby/fitting.py` — fit EM module weights via per-letter compile + verdict-disagreement loss | 3 days |
| 2.2 | Held-out validation (~10% split) | 1 day |
| 2.3 | `em_dag/profiles/dear_abby_baseline.yaml` + ethos header | 1 day |

### Phase 3: Probe corpus + integration

| Step | Deliverable | Effort |
|---|---|---|
| 3.1 | Format Dear Abby corpus for the Phase-3 probe-extractor calibration pipeline | 1 day |
| 3.2 | Train calibrated probe; emit checkpoint + `CalibrationProvenance` block | 1–2 days |
| 3.3 | Bench-mark probe vs. rule-extractor on the Dear Abby held-out set | 1 day |

### Phase 4: Companion profile + CLI tightening

| Step | Deliverable | Effort |
|---|---|---|
| 4.1 | Public-domain substitute corpus + fitted profile | 3–4 days |
| 4.2 | Remove implicit default from `compile` (force `--em-profile`) | 1 day |
| 4.3 | Ship at least one philosophical-tradition profile (kantian or utilitarian) for contrast | 2–3 days |
| 4.4 | Docs: `docs/profiles/index.md` with ethos descriptions + bias notes | 1 day |

Total: **~3–4 working weeks** after licence + corpus access cleared.

## What this is NOT

- **Not** a "moral classifier" — fitting weights to match Abby's
  verdicts produces a fitted *baseline*, not a normative claim.
- **Not** an attempt to settle ethics. Multiple profiles ship
  simultaneously so the user picks which ethos they want to apply.
- **Not** a substitute for MoralTensor-Bench. The bench (rp-03) is
  about *compiler correctness*; this is about *fitted ethos baselines*.
- **Not** an excuse to ship without documented bias. Every profile
  ships with an explicit `ethos_description:` and a section in the
  README enumerating its known limitations.

## Why this matters for the broader thesis

The Geometric Series argues that moral reasoning admits structural
representation. Implementing that thesis at the compiler level means
producing per-IR tensors with rich structure. **But the *verdict*
contraction is still scalar-shaped** (`forbid` / `prefer` etc.) and
some choice of weights drives it. The current implicit choice ("the
maintainer's intuition") is the weakest part of the stack: it's the
one place where the compiler embodies an unargued moral position.

Naming the position — by fitting it from an identifiable corpus,
labelling the ethos, and offering alternatives — is the structural
fix. Dear Abby is a defensible first choice because it gives us
something to label clearly, not because that ethos is the right one.

## Open questions

**A. Default behaviour with no `--em-profile`.** Current options:

  1. **Hard-fail** — require explicit `--em-profile`. Cleanest;
     forces the user to make the choice legible.
  2. **Pick public-domain-baseline** — default to the smaller MIT-
     licensable profile. Lower friction but smuggles in a bias.
  3. **Pick a meta-profile** — average across all fitted profiles.
     Mathematically defensible but ethically incoherent.

  Recommendation: **(1)** hard-fail with a clear error listing
  available profiles. The friction is the *feature*.

**B. How often to re-fit.** Each Dear Abby decade has noticeably
different ethos (Pauline 1980s vs Jeanne 2010s on same-sex couples,
for instance). Should we ship one fitted profile per decade?

  Recommendation: ship the union ("Dear Abby 1996–2010") as v0.1,
  add per-decade slices as v0.2 once the infrastructure is stable.

**C. Verdict-projection model choice.** The projection from advice
paragraph to per-party verdicts is the load-bearing inference step.
If `gpt-oss` is biased in a way that *also* matches Abby's biases,
the fit looks better than it is.

  Mitigation: use two different projection models (gpt-oss and
  Qwen2.5-7B), keep only letters where both agree, document the
  disagreement rate.

**D. Citation / academic propriety.** If the Dear Abby corpus is
used for academic publication (the JOSS paper, the AIES paper),
we need a careful framing of fair use and a clear acknowledgement
to Andrews McMeel.

## Failure modes worth surfacing

- **The fit looks great but generalises poorly.** Dear Abby letters
  are self-selected (people write in *with problems*); they aren't
  representative of the population. A profile fitted on Abby may
  systematically over-weight conflict-resolution and under-weight
  ordinary good behaviour.
- **The projection model is the actual bias source.** If gpt-oss
  reads "the daughter was rude" and projects "verdict: forbid the
  daughter", the fit is to gpt-oss-mediated-Abby, not to Abby.
- **Multiple-ethos drift.** Once we have N named profiles, callers
  will start ensembling them ("average kantian + utilitarian"). That
  ensemble has no documented bias — it's an artifact. Don't ship the
  ensemble; only ship named singles.

## Concrete first step (under 1 day, no licence question)

Before any corpus acquisition, write the projection schema +
fitting harness against a *synthetic* test corpus of 20 hand-written
"Abby-like" letters with hand-set gold verdicts. Verify the fitting
loop reduces disagreement, the profile YAML serialises cleanly, and
the CLI `--em-profile` flag picks it up. This is the engineering
groundwork; the licence question and the real corpus acquisition
follow.
