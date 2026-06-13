"""`Projection` ABC + shared `ProjectionResult` schema.

Each `Projection` reads a `MoralSubstrate` and returns a
`ProjectionResult`. Result *shapes* differ across frameworks:

  - ConsequentialistProjection: tensor + Gini + per-party verdicts
  - DeonticProjection: GateFinding list (categorical pass/fail)
  - VirtueProjection (future): character-trait vector

But every result carries the same minimum interface — `framework`,
`verdict`, `findings`, `metadata` — so cross-projection comparison +
audit + reporting can iterate uniformly.

Verdicts are framework-relative: the consequentialist projection's
`permitted` is not the same claim as the deontic projection's
`permissible_under_categorical_imperative`. When projections
disagree, the orchestrator does NOT aggregate — it surfaces both.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from erisml_compiler.projections.substrate import MoralSubstrate


GateSeverity = Literal["minor", "moderate", "grave", "catastrophic"]


class GateFinding(BaseModel):
    """A categorical pass/fail finding from a gate-style projection.

    Unlike `DimensionScore` (continuous value + confidence), a
    GateFinding represents a yes/no test that either fires (`passed=False`)
    or doesn't. Deontic frameworks naturally produce gates; the
    `veto_location` mechanism in DEME V3 accepts them as input.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Stable identifier for the gate, e.g. `universalizability`,
    `mere_means`, `valid_consent`, `legitimate_authority`."""
    passed: bool
    """True iff the gate does NOT fire (the action survives this
    test). False means the gate fired — the test was failed."""
    reason: str
    """Brief explanation tied to substrate elements (which stakeholder,
    which fact, which authority) — auditable, not just a verdict."""
    severity: GateSeverity = "moderate"
    """How seriously a failure of this gate should be taken.
    Categorical-imperative failures are typically `grave`."""
    subjects: list[str] = Field(default_factory=list)
    """Stakeholder ids implicated in the finding."""
    detail: dict[str, Any] = Field(default_factory=dict)


class ProjectionResult(BaseModel):
    """Uniform interface across framework projections."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    framework: str
    """E.g. `consequentialist_distributive`, `deontic_kantian`, `virtue`."""
    verdict: str
    """Framework-relative verdict. Consequentialist verdicts are
    {permitted, requires_human_review, tragic_conflict_escalate, ...};
    deontic verdicts are {permissible, forbidden, requires_review}.
    Cross-projection comparison treats these as opaque strings."""
    confidence: float = 1.0
    findings: list[GateFinding] = Field(default_factory=list)
    """Framework-specific findings. For deontic this carries the
    universalizability + mere-means gates. For consequentialist this
    is typically empty (the per-stakeholder tensor and DEME verdict
    carry the analysis; see `framework_specific`)."""
    framework_specific: dict[str, Any] = Field(default_factory=dict)
    """Bag for framework-specific output: tensors, Gini, virtue traits,
    etc. Each projection documents what it puts here."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Projection(ABC):
    """Read a substrate, emit a framework-bound result."""

    framework: str

    @abstractmethod
    def project(self, substrate: MoralSubstrate, **kwargs: Any) -> ProjectionResult:
        """Run this projection over the substrate."""

    @property
    def name(self) -> str:
        return self.framework
