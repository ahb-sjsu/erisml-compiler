"""Pydantic v2 schemas for the ErisML Compiler Intermediate Representation.

Mirrors spec section 13 (Internal representation) and section 14 (MoralVector).
Adds EM-DAG outputs in `CompilerIR.em_outputs` per the architectural addition.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from erisml_compiler import __schema_version__


# =============================================================================
# Primitives
# =============================================================================


class SourceSpan(BaseModel):
    """A pointer into the source text. Format: `seg_NNN:start-end` where
    `start` and `end` are character offsets within the named segment."""

    model_config = ConfigDict(frozen=True)

    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "start" in info.data and v < info.data["start"]:
            raise ValueError(f"end ({v}) must be >= start ({info.data['start']})")
        return v

    def as_string(self) -> str:
        return f"{self.segment_id}:{self.start}-{self.end}"

    @classmethod
    def parse(cls, s: str) -> "SourceSpan":
        seg, rng = s.split(":")
        start_s, end_s = rng.split("-")
        return cls(segment_id=seg, start=int(start_s), end=int(end_s))


# =============================================================================
# Document and segmentation
# =============================================================================


class Document(BaseModel):
    """Metadata about the source document."""

    doc_id: str
    title: str | None = None
    source: str | None = None
    domain: str | None = None
    language: str = "en"
    timestamp: str | None = None
    sha256: str | None = None
    raw_text: str = ""


SegmentType = Literal[
    "background", "decision_point", "outcome", "narration", "dialogue", "norm_statement"
]


class Segment(BaseModel):
    segment_id: str
    type: SegmentType
    text: str
    span: tuple[int, int]


# =============================================================================
# Stakeholders, relations, events
# =============================================================================


StakeholderRole = Literal[
    "agent",
    "patient",
    "beneficiary",
    "victim",
    "bystander",
    "nonconsenting_third_party",
    "authority",
    "coercer",
    "dependent",
    "future_person",
    "collective",
    "vow_holder",
    "protector",
]

StakeholderType = Literal[
    "individual", "community", "institution", "system", "group", "abstract", "future", "collective"
]


class Stakeholder(BaseModel):
    id: str
    label: str
    type: StakeholderType
    roles: list[StakeholderRole] = Field(default_factory=list)
    agency: str | None = None  # e.g., "full", "collective_limited", "incapacitated"
    vulnerability: str | None = None  # "low" | "moderate" | "high" | "extreme"
    consent_status: str | None = None  # "obtained" | "not_obtained" | "coerced" | "n/a"
    source_spans: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    requires_review: bool = False


class Relation(BaseModel):
    id: str
    type: str  # e.g., "guardian_of", "subordinate_to", "kin_of"
    source: str  # stakeholder id
    target: str  # stakeholder id
    description: str | None = None
    source_spans: list[str] = Field(default_factory=list)


class Event(BaseModel):
    id: str
    time_index: int = Field(ge=0)
    type: str  # e.g., "vow_made", "demand_made", "harm_inflicted", "threat_uttered"
    actor: str | None = None
    target: str | None = None
    content: str | None = None
    conditions: list[str] = Field(default_factory=list)
    source_spans: list[str] = Field(default_factory=list)


# =============================================================================
# Commitments and norms
# =============================================================================


CommitmentType = Literal["vow", "promise", "contract", "oath", "covenant", "role_duty", "policy"]
CommitmentStatus = Literal[
    "active", "active_but_defeasible", "defeated", "fulfilled", "violated", "void", "expired"
]
Voluntariness = Literal["voluntary", "coerced", "incapacitated", "uncertain"]
Legitimacy = Literal[
    "prima_facie_valid", "fully_legitimate", "defeasible", "void", "fraudulent", "tyrannical"
]


class Commitment(BaseModel):
    id: str
    type: CommitmentType
    holder: str  # stakeholder id
    beneficiary: str | None = None  # stakeholder id
    content: str
    voluntariness: Voluntariness = "voluntary"
    created_at_event: str | None = None  # event id
    status: CommitmentStatus = "active"
    legitimacy: Legitimacy = "prima_facie_valid"
    defeasibility_conditions: list[str] = Field(default_factory=list)
    source_spans: list[str] = Field(default_factory=list)


NormModality = Literal["prohibition", "obligation", "permission", "exception"]


class Norm(BaseModel):
    id: str
    modality: NormModality
    actor: str  # who the norm applies to
    action: str
    target: str | None = None
    priority_tier: int = Field(ge=0, default=1)
    defeasible: bool = True
    source: str  # "statutory", "constitutional", "customary", "inferred_human_rights_baseline", etc.
    source_spans: list[str] = Field(default_factory=list)


# =============================================================================
# Ethical facts
# =============================================================================


EthicalFactKind = Literal[
    "coercion",
    "consent",
    "legitimacy",
    "harm",
    "vulnerability",
    "uncertainty",
    "externality",
    "justice",
    "care",
    "truth",
    "role_duty",
    "deception",
    "reciprocity",
    "non_maleficence",
]


class EthicalFact(BaseModel):
    id: str
    kind: EthicalFactKind
    subjects: list[str] = Field(default_factory=list)  # stakeholder ids
    description: str
    severity: str | None = None  # "minor" | "moderate" | "grave" | "catastrophic"
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source_spans: list[str] = Field(default_factory=list)


# =============================================================================
# Conflicts
# =============================================================================


ConflictSeverity = Literal["minor", "moderate", "grave", "catastrophic"]
ConflictStatus = Literal["unresolved", "tragic", "resolved_by_priority", "resolved_by_consent"]


class Conflict(BaseModel):
    id: str
    name: str
    stakeholders: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    severity: ConflictSeverity = "moderate"
    resolution_status: ConflictStatus = "unresolved"
    requires_escalation: bool = False
    description: str | None = None


# =============================================================================
# MoralVector and timeline (spec section 14)
# =============================================================================


MORAL_DIMENSIONS: tuple[str, ...] = (
    "physical_harm",
    "rights_respect",
    "fairness_equity",
    "autonomy_consent",
    "legitimacy_trust",
    "epistemic_quality",
    "care_protection",
    "vow_fidelity",
    "third_party_externality",
    "repair_residue",
)


class DimensionScore(BaseModel):
    """One axis of a MoralVector."""

    value: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0, default=0.0)
    direction: Literal["positive", "negative", "neutral"] = "neutral"
    source_spans: list[str] = Field(default_factory=list)
    explanation: str | None = None


class MoralVector(BaseModel):
    """A 10-dimensional moral state at a single time index, indexed by
    dimension name. Field names match `MORAL_DIMENSIONS`."""

    physical_harm: DimensionScore
    rights_respect: DimensionScore
    fairness_equity: DimensionScore
    autonomy_consent: DimensionScore
    legitimacy_trust: DimensionScore
    epistemic_quality: DimensionScore
    care_protection: DimensionScore
    vow_fidelity: DimensionScore
    third_party_externality: DimensionScore
    repair_residue: DimensionScore


class MoralTensor(BaseModel):
    """A rank-2 moral state indexed by stakeholder and dimension. Used when
    moral state must be tracked per stakeholder rather than aggregated."""

    stakeholder_id: str
    time_index: int = Field(ge=0)
    by_dimension: dict[str, DimensionScore]


class TimelineEntry(BaseModel):
    time_index: int = Field(ge=0)
    event_id: str | None = None
    event_label: str
    vector: MoralVector


# =============================================================================
# EM-DAG evaluation outputs
# =============================================================================


class EMOutput(BaseModel):
    """The output of a single Ethical Module evaluation."""

    module_name: str
    score: DimensionScore
    contributing_facts: list[str] = Field(default_factory=list)  # ethical-fact ids
    upstream_dependencies: list[str] = Field(default_factory=list)  # module names
    notes: str | None = None


# =============================================================================
# DEME verdict
# =============================================================================


VerdictKind = Literal[
    "permitted",
    "permitted_with_residue",
    "tragic_conflict_escalate",
    "prohibited",
    "requires_human_review",
    "indeterminate",
]


class DEMEVerdict(BaseModel):
    """The DEME-stub verdict shaped like the structural-fuzzing target so
    Phase 2 can drop in real DEME without changing this contract."""

    verdict: VerdictKind
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str
    allowed_actions: list[str] = Field(default_factory=list)
    prohibited_actions: list[str] = Field(default_factory=list)
    moral_residue: list[str] = Field(default_factory=list)
    escalation_required: bool = False


# =============================================================================
# Audit
# =============================================================================


class PassRecord(BaseModel):
    """One pass of the 12-pass pipeline."""

    pass_index: int
    pass_name: str
    tier: str  # CompilerTier enum value
    duration_ms: float | None = None
    notes: str | None = None


class AuditRecord(BaseModel):
    ir_hash: str  # sha256 of the canonical-JSON IR (excluding audit itself)
    source_text_hash: str
    schema_version: str = __schema_version__
    compiler_version: str
    tier: str
    extractor: str
    model_version: str | None = None
    em_profile: str | None = None
    timestamp_utc: str
    passes: list[PassRecord] = Field(default_factory=list)


# =============================================================================
# Top-level IR
# =============================================================================


class CompilerIR(BaseModel):
    """Top-level ErisML Compiler IR (spec section 13.1).

    Adds `em_outputs` and `timeline` beyond the section 13.1 listing to capture
    EM-DAG evaluation results and MoralVector trajectory.
    """

    model_config = ConfigDict(validate_assignment=True)

    schema_version: str = __schema_version__
    document: Document
    segments: list[Segment] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    commitments: list[Commitment] = Field(default_factory=list)
    norms: list[Norm] = Field(default_factory=list)
    ethical_facts: list[EthicalFact] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    moral_vectors: list[MoralVector] = Field(default_factory=list)
    moral_tensors: list[MoralTensor] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    em_outputs: dict[str, EMOutput] = Field(default_factory=dict)
    deme_verdict: DEMEVerdict | None = None
    canonical_form: str | None = None
    audit: AuditRecord | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
