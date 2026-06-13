# Release planning 05 — Social Chem 101 as named ethos profiles

**Status:** implemented.
**Implementation:** `src/erisml_compiler/social_chem/` + two profile YAMLs
in `src/erisml_compiler/em_dag/profiles/`.
**Predecessor:** none formal; raised in conversation after
[release-planning-04](./release-planning-04-eigenvalue-scalar.md).
**Supersedes:** the earlier draft of this note that proposed a Kaggle
Dear-Abby-questions-only fit with title-polarity weak supervision —
that path was abandoned once we confirmed Social Chem 101 already
contains AI2-worker-extracted, MFT-tagged RoTs over the same Dear Abby
source.

## The thesis

Every ethical evaluator has an implicit ethos somewhere. The
compiler's EM-DAG today uses `em_dag/profiles/default.yaml` with
hand-set module weights — i.e. *"whatever the maintainer thought
reasonable in 2026"*. That's a position, but it's an invisible one.
The honest move is to make the ethos a **named, fitted, swappable
artifact**, so callers can pick (and audit) which moral baseline the
compiler applies.

## Source: Social Chem 101 (Forbes et al., EMNLP 2020)

Social Chem 101 (CC-BY 4.0, AI2 + UW) is a 355k-row corpus of
worker-curated rules-of-thumb (RoTs) extracted from four sources:

    dearabby       50,300 rows   /  11,729 unique situations
    amitheasshole  96,082 rows   /  29,510 unique situations
    rocstories    101,791 rows
    confessions   107,749 rows

Each row carries an MFT-foundation set, a signed action-moral-judgment
in [-2, +2], and an ordinal agreement scale (rot-agree 1–5).

This data **already solves** the verdict-supervision problem that the
Kaggle Dear-Abby-questions-only dataset created — AI2 workers read the
original posts/letters and tagged them with explicit moral foundations
and verdict directions. We don't need title-keyword heuristics; we use
the labels directly.

## Architecture

`src/erisml_compiler/social_chem/`:

  - `schema.py` — Situation, SituationRoT, SituationAggregate,
    CorpusFingerprint, ProfileFitResult
  - `loader.py` — TSV reader filtered by `area`, grouped by
    `situation-short-id`
  - `projection.py` — MFT-foundation → EM-DAG module mapping
    (`DEFAULT_MFT_TO_EM_DAG`); per-situation aggregation
    weighted by `rot-agree`
  - `fitting.py` — weight = mean_confidence × salience; sum-to-N
    normalised; floor modules dropped from `weights:` so the
    moral-vector loader's default of 1.0 takes over
  - `profile_writer.py` — YAML emit

`scripts/fit_socialchem_profiles.py` runs both fits end-to-end.

## MFT → EM-DAG mapping

Hand-curated default in `projection.DEFAULT_MFT_TO_EM_DAG`:

| MFT foundation         | EM modules (with contribution coefficient) |
|------------------------|--------------------------------------------|
| care-harm              | HarmEM (1.0), CareEM (0.8)                 |
| fairness-cheating      | FairnessEM (1.0)                            |
| loyalty-betrayal       | FidelityEM (1.0), LegitimacyEM (0.4)        |
| authority-subversion   | LegitimacyEM (1.0)                          |
| sanctity-degradation   | RightsEM (0.5)                              |

The mapping is recorded in every emitted profile YAML's
`mft_to_em_mapping:` block, so the projection step is auditable.
Modules with **no** MFT channel (EpistemicEM, AutonomyEM,
ExternalityEM, RepairEM) receive zero contribution from this
projection and therefore take the loader's implicit baseline weight
of 1.0 — that's a real coverage gap in the source corpus, explicitly
recorded in the profile's `coverage:` field.

## Emitted profiles

`em_dag/profiles/dear_abby_socialchem_v0.1.yaml` (n=11,729 situations):

    HarmEM         1.748   prior=-0.05   coverage=81.3%
    CareEM         1.748   prior=-0.05   coverage=81.3%
    LegitimacyEM   0.970   prior=-0.09   coverage=62.6%
    FidelityEM     0.724   prior=-0.06   coverage=46.8%
    FairnessEM     0.594   prior=-0.18   coverage=42.2%
    RightsEM       0.217   prior=-0.17   coverage=19.1%
    (epistemic, autonomy, externality, repair: floor → loader default 1.0)

`em_dag/profiles/aita_socialchem_v0.1.yaml` (n=29,510 situations):

    HarmEM         1.572   prior=-0.12   coverage=73.7%
    CareEM         1.572   prior=-0.12   coverage=73.7%
    LegitimacyEM   1.108   prior=-0.14   coverage=64.2%
    FidelityEM     0.855   prior=-0.11   coverage=50.8%
    FairnessEM     0.739   prior=-0.17   coverage=45.5%
    RightsEM       0.155   prior=-0.24   coverage=12.1%
    (epistemic, autonomy, externality, repair: floor → loader default 1.0)

The contrast between the two reflects what one would predict on
inspection: Dear Abby's letters surface care/harm slightly more
heavily; AITA arguments lean more on legitimacy/standing and fairness.

## Bias disclosure (in every profile YAML)

Every profile YAML carries a `bias_notes:` array enumerating known
limitations. Shared across both profiles:

- AI2 MTurk worker demographics (US, 2020-skewed).
- MFT's 5 channels don't partition the EM-DAG's 10 modules.
- Hand-curated MFT → EM mapping with explicit recorded weights.

Plus per-source notes (Dear Abby letter-writer self-selection,
1985–2017 cultural slice; AITA Reddit community norms, etc.).

## Data caching

The Social Chem 101 zip + Scruples anecdotes + Scruples dilemmas are
cached on Atlas at:

    /archive/ethics-corpora/social-chem-101/
    /archive/ethics-corpora/scruples/

with per-area pre-extracted slices at:

    /archive/ethics-corpora/social-chem-101/slices/
        social-chem-101.dearabby.tsv      24 MB
        social-chem-101.amitheasshole.tsv 38 MB
        social-chem-101.rocstories.tsv    44 MB
        social-chem-101.confessions.tsv   41 MB

Local copies of the dearabby + AITA slices live in
`data/social-chem-101/` (gitignored). To rehydrate from scratch, run
`scripts/atlas_cache_aic.py` (Atlas), then
`scripts/atlas_extract_slices.py` (pulls slices local).

## Why not pursue more profiles right now

Two profiles ship in v0.x. Further candidates worth fitting later:

- `rocstories_socialchem` and `confessions_socialchem` — same shape,
  different audience.
- `scruples_anecdotes` — verdict-supervised by Reddit community
  voting (YTA/NTA/ESH/NAH), denser per-case label than Social Chem's
  per-RoT supervision.
- One-shot fits from religious/classical corpora cached at
  `/archive/ethics-corpora/{islamic, sefaria, pali_canon,
  perseus-greek, perseus-latin, sanskrit, chinese_classics}` — these
  are full-text canons rather than per-case datasets, so the fit shape
  has to differ (a single ethos profile per canon, not per-situation
  aggregation). Out of scope for v0.x.

## What this is not

- Not a moral authority. Two profiles, both named, both biased by
  construction. Choosing one is itself an ethical act and the compiler
  surfaces it as such.
- Not foundation-complete. The MFT channels don't cover all moral
  reasoning the EM-DAG models (no epistemic, no autonomy, no
  externality, no repair). The coverage gap is explicit.
- Not the columnist's verdicts. For Dear Abby specifically, the
  verdict directions come from AI2 workers reading Abby's published
  letters — not from Abby herself. The Kaggle questions-only corpus
  that lacked Abby's replies has been retired from the fit path.
