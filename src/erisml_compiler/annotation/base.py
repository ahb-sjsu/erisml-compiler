"""Extractor abstract base and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from erisml_compiler.ir.schemas import (
    Commitment,
    Conflict,
    EthicalFact,
    Event,
    Norm,
    Relation,
    Stakeholder,
)

if TYPE_CHECKING:
    from erisml_compiler.ir.graph import MoralGraph


class UnknownDocumentError(RuntimeError):
    """Raised by MockExtractor when given a document outside its fixture set."""


@dataclass
class ExtractorResult:
    """The structured output of any extractor.

    Returned by Tiers 2 and 3 (text -> structure). Tier 1 reads the same
    structure directly from a JSON input and does not invoke an extractor.

    Extractors may *optionally* emit a `graph` alongside the flat lists.
    When present, the orchestrator skips Pass 7.5 (graph promotion) and
    uses the extractor's graph directly. When absent, the orchestrator
    promotes the flat lists into a graph automatically. Extractors that
    want to be graph-native (e.g. an LLM extractor that builds typed
    edges as part of its reasoning) populate this field; legacy
    extractors leave it None and pay the promotion cost at Pass 7.5.
    """

    stakeholders: list[Stakeholder] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    norms: list[Norm] = field(default_factory=list)
    ethical_facts: list[EthicalFact] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    canonical_form: str | None = None
    extractor_metadata: dict = field(default_factory=dict)
    graph: "MoralGraph | None" = None


class Extractor(ABC):
    """Abstract base for all extractors (Mock, Rule, LLM).

    All extractors take a fully-loaded `Document` plus the post-segmentation
    segments and return an `ExtractorResult`. The pipeline orchestrator
    composes the result into a `CompilerIR` and hands it to the EM-DAG.
    """

    name: str  # short identifier used in audit records

    @abstractmethod
    def extract(self, document, segments) -> ExtractorResult:  # noqa: D401
        """Extract moral structure from document + segments."""
        raise NotImplementedError
