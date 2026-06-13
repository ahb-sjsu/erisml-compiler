"""Typed node and edge schema for the MoralGraph."""
from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeKind(str, enum.Enum):
    STAKEHOLDER = "stakeholder"
    ACT = "act"
    MAXIM = "maxim"
    COMMITMENT = "commitment"
    FACT = "fact"
    NORM = "norm"


class EdgeKind(str, enum.Enum):
    PERFORMS = "performs"
    """stakeholder -> act. The agent who is performing the act."""

    IMPOSES_ON = "imposes_on"
    """act -> stakeholder. The act produces an effect on this party.
    Payload: {severity: str | None, confidence: float, fact_id: str}."""

    CONSENTS_TO = "consents_to"
    """stakeholder -> act. The party has affirmatively consented to
    bearing the act's effects. Payload: {given: bool, under_duress:
    bool, informed: bool}. Absence of this edge from an
    `imposes_on` target = no consent on record (assume not given)."""

    HOLDS_COMMITMENT = "holds_commitment"
    """stakeholder -> commitment. The party who owns the commitment."""

    COMMITMENT_BINDS = "commitment_binds"
    """commitment -> stakeholder. The beneficiary the commitment is
    owed to. Empty when the commitment beneficiary is unspecified."""

    TREATS_AS = "treats_as"
    """act -> stakeholder. Categorical relation: how the act treats
    this party. Payload: {role: 'end' | 'means' | 'mere_means'}.
    `mere_means` is the Kantian failure mode."""

    UNDER_MAXIM = "under_maxim"
    """act -> maxim. The action under a particular description.
    Universalizability tests operate on the maxim, not the act."""

    COERCES = "coerces"
    """stakeholder -> stakeholder. The first party applies coercive
    pressure on the second. Payload: {severity: str | None}."""

    SURFACES_FACT = "surfaces_fact"
    """act -> fact OR stakeholder -> fact. The act/stakeholder gave
    rise to or is mentioned in this ethical fact."""

    FACT_SUBJECT = "fact_subject"
    """fact -> stakeholder. The party who is the subject of this
    fact (e.g. who bears the harm, whose autonomy is impacted)."""

    WOULD_VIOLATE_IF_UNIVERSALISED = "would_violate_if_universalised"
    """act -> norm. If the maxim of this act were universalised, this
    norm would be violated. Populated by the deontic projection /
    universalizability gate, not by the extractor."""


class MoralNode(BaseModel):
    """One node in the moral graph.

    `id` is the stable identifier — typically `<kind>:<local_id>` for
    nodes created during graph promotion (e.g. `stakeholder:self`,
    `act:event_001`, `maxim:m_001`).

    `payload` is the node's type-specific data, stored as a dict so
    the graph stays serialisable without a discriminated union. The
    extractor's typed Pydantic models (`Stakeholder`, `Event`, etc.)
    become payload dicts here.
    """

    model_config = ConfigDict(frozen=False)

    id: str
    kind: NodeKind
    payload: dict[str, Any] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    """Free-form labels for filtering / grouping (e.g. 'agent',
    'authority', 'vulnerable'). Independent of `kind`."""


class MoralEdge(BaseModel):
    """One directed edge in the moral graph.

    Multi-edges are permitted (two edges with the same src+dst+kind
    distinguished by payload). The canonical hash treats edges as a
    sorted multiset by `(src, dst, kind, sorted_payload)`.
    """

    model_config = ConfigDict(frozen=False)

    src: str
    dst: str
    kind: EdgeKind
    payload: dict[str, Any] = Field(default_factory=dict)
