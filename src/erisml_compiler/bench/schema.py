"""Pydantic schema for one MoralTensor-Bench scenario.

Minimal v0.1 shape — the full design (release-planning-03) calls for
paraphrase pairs, invariance tolerances, and per-scenario diagnoses;
those land as the bench corpus grows. The v0.1 schema covers the
must-haves: expected stakeholders, commitments, ethical-fact kinds,
per-party verdicts, and a canonical form.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExpectedStakeholder(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    type: str = "individual"
    """One of {individual, group, community, organisation, ...} — keep
    flexible at v0.1; the extractor's typing is still evolving."""
    roles: list[str] = Field(default_factory=list)
    """Roles this stakeholder occupies (agent, patient, beneficiary,
    coercer, authority, bystander, vow_holder, ...)."""


class ExpectedCommitment(BaseModel):
    model_config = ConfigDict(frozen=True)

    holder: str
    """Stakeholder id of the party making the commitment."""
    beneficiary: str | None = None
    type: str = "vow"
    """One of {vow, promise, oath, contract, professional_duty}."""


class ExpectedScenario(BaseModel):
    """The 'gold' answer the compiler is expected to produce."""

    model_config = ConfigDict(frozen=True)

    canonical_form: str | None = None
    """Canonical form tag the compiler should emit. None means
    'unspecified — don't score canonical_form for this scenario'."""

    stakeholders: list[ExpectedStakeholder] = Field(default_factory=list)
    commitments: list[ExpectedCommitment] = Field(default_factory=list)
    ethical_fact_kinds: list[str] = Field(default_factory=list)
    """Kinds the rule/llm extractor should surface for this scenario,
    e.g. ['coercion', 'externality', 'care', 'deception']."""

    per_party_verdicts: dict[str, str] = Field(default_factory=dict)
    """stakeholder_id -> {permit, prefer, forbid, neutral}. The
    aggregate verdict is computed by DEME; this records what each
    party would *separately* think of the act."""

    overall_verdict: str | None = None
    """The DEME-level verdict expected (tragic_conflict_escalate,
    requires_human_review, permitted, ...). None to skip scoring."""

    expect_premature_contraction: bool = False
    """True if a clean verdict here would be wrong — the scenario
    should escalate to requires_human_review."""


class ScenarioGold(BaseModel):
    """One scenario file."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    """Stable identifier matching the filename (without .yaml)."""
    category: str
    """One of the 10 categories listed in the design note."""
    source: str = ""
    """Citation / origin note — literature, case, etc."""
    license: str = "MIT"

    raw_text: str
    """The natural-language scenario the compiler will read."""

    expected: ExpectedScenario


class ScenarioScore(BaseModel):
    """Per-scenario score across all metrics."""

    model_config = ConfigDict(frozen=False)

    scenario_id: str
    category: str

    stakeholder_recall: float
    """Fraction of expected stakeholders matched (by id or fuzzy label)."""

    stakeholder_role_f1: float
    """F1 across (stakeholder_id, role) pairs."""

    commitment_f1: float

    canonical_form_match: float
    """1.0 if expected.canonical_form == ir.canonical_form,
    0.0 otherwise. NaN-as-zero when no expected canonical form."""

    ethical_fact_kind_recall: float

    per_party_verdict_accuracy: float
    """Fraction of expected per-party verdicts matched exactly."""

    overall_verdict_match: float
    """1.0 if expected.overall_verdict matches ir.deme_verdict.verdict;
    0.0 otherwise. NaN-as-zero when no expected overall_verdict."""

    premature_contraction: float
    """1.0 if the scenario expected requires_human_review and the
    compiler emitted a clean verdict (this is a *failure* — higher
    is worse); 0.0 otherwise. Aggregated as 'premature_contraction_rate'."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Raw IR snippets for debugging / per-scenario inspection."""


class BenchAggregate(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_scenarios: int
    n_failed_compile: int
    mean_stakeholder_recall: float
    mean_stakeholder_role_f1: float
    mean_commitment_f1: float
    mean_canonical_form_match: float
    mean_ethical_fact_kind_recall: float
    mean_per_party_verdict_accuracy: float
    mean_overall_verdict_match: float
    premature_contraction_rate: float
    moral_tensor_bench_score: float
    """Weighted aggregate. See bench/v0.1/weights.yaml for the weights."""
