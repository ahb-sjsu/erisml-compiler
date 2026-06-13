"""Pydantic models for Social Chem 101 rows + fitting artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SituationRoT(BaseModel):
    """One Social Chem 101 row: a worker's rule-of-thumb about a situation.

    Mirrors the TSV columns we actually use for fitting; less-used
    metadata (worker IDs, char-targeting, split) is dropped at the
    loader to keep memory small for 100k+ row slices.
    """

    model_config = ConfigDict(frozen=True)

    area: str
    """One of {dearabby, amitheasshole, rocstories, confessions}."""

    situation_short_id: str
    """Stable per-situation identifier (e.g.
    'dearabby/title/dearabby_1996_1_19_0_...').
    Multiple RoT rows share this id."""

    situation: str
    """Free-text situation, e.g. the Dear Abby letter title."""

    rot: str
    """The full rule-of-thumb sentence the worker wrote."""

    rot_categorization: str
    """One of {morality-ethics, social-norms, advice, description}
    (or pipe-separated mixture). Empty for some rows."""

    rot_moral_foundations: tuple[str, ...] = ()
    """Subset of {care-harm, fairness-cheating, loyalty-betrayal,
    authority-subversion, sanctity-degradation}. Empty for ~20% of rows."""

    rot_agree: int | None = None
    """1-5 ordinal: how widely the worker thinks the RoT is endorsed.
    None when missing."""

    action: str = ""
    """The action being judged (only set for action-bearing rows)."""

    action_moral_judgment: int | None = None
    """Signed -2..+2 verdict on the action. None when missing."""

    action_agree: int | None = None


class Situation(BaseModel):
    """All RoT rows that share a situation-short-id, grouped."""

    model_config = ConfigDict(frozen=True)

    situation_short_id: str
    situation: str
    area: str
    rots: tuple[SituationRoT, ...] = ()


class SituationAggregate(BaseModel):
    """Per-situation aggregated signal across its RoT rows.

    Each EM module gets a value in [-1, 1] (signed by the corpus's
    average action-moral-judgment across rows that touch a foundation
    associated with that module) and a confidence in [0, 1]
    (fraction of the situation's RoT rows that tagged a relevant
    foundation, weighted by rot-agree)."""

    model_config = ConfigDict(frozen=True)

    situation_short_id: str
    per_module_value: dict[str, float] = Field(default_factory=dict)
    per_module_confidence: dict[str, float] = Field(default_factory=dict)
    n_rots: int = 0


class CorpusFingerprint(BaseModel):
    """SHA-256 + summary of a Social Chem 101 slice used for a fit."""

    model_config = ConfigDict(frozen=True)

    source: str
    """E.g. 'social-chem-101.v1.0.tsv :: area=dearabby'."""

    license: str
    """The corpus's license — CC-BY 4.0 per Forbes et al. 2020."""

    citation: str
    """The bibliographic citation to surface in any UI using this profile."""

    schema_tsv_columns: list[str]
    n_rows: int
    n_situations: int
    canonical_sha256: str
    """Hash over the sorted situation_short_ids + their RoT counts.
    Deterministic per corpus slice."""

    foundation_distribution: dict[str, int] = Field(default_factory=dict)
    judgment_distribution: dict[str, int] = Field(default_factory=dict)


class ProfileFitResult(BaseModel):
    """What `fit_profile` returns and what the YAML profile records."""

    model_config = ConfigDict(frozen=False)

    name: str
    description: str
    ethos_description: str
    bias_notes: list[str] = Field(default_factory=list)

    corpus: CorpusFingerprint

    fit_method: str
    """E.g. 'mft_to_em_via_agreement_weighted_means'."""

    mft_to_em_mapping: dict[str, dict[str, float]] = Field(default_factory=dict)
    """The hand-curated MFT-foundation -> EM-module mapping used.
    Recorded in the profile so callers can audit the projection."""

    weights: dict[str, float] = Field(default_factory=dict)
    """Per-EM-module weight, normalised so sum == len(modules) (i.e.
    equal-weight baseline is all 1.0)."""

    priors: dict[str, float] = Field(default_factory=dict)
    """Per-EM-module signed mean value in [-1, 1] across the corpus."""

    coverage: dict[str, float] = Field(default_factory=dict)
    """Per-EM-module 'fraction of situations where this module fires
    at all' — audit signal. Modules with no MFT mapping show 0.0
    here even when their normalised weight is 1.0."""

    fitted_date: str
    metadata: dict[str, Any] = Field(default_factory=dict)
