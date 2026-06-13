"""`flat_from_graph(graph)` — back-derive the flat-list IR shape.

Counterpart to `promote.py`. When the graph is the primary
representation, downstream consumers that read flat fields
(`ir.stakeholders`, `ir.events`, etc.) need a deterministic way to
materialise those lists from graph payloads.

Semantics:

    promote.graph_from_flat(ir).then_lower() ≈ flat round-trip

Equal-modulo-list-order: the lower step sorts entities by id (or the
canonical id form `<kind>:<local>`) so two compiles of the same input
produce identical flat lists.

What's preserved (for nodes that came from a flat-list field
originally):
  - Stakeholder, Event, Commitment, Norm, EthicalFact payloads
  - all id strings
  - subjects on EthicalFact
  - holder / beneficiary on Commitment
  - actor / target on Event
  - source_spans, severity, confidence, all enum-typed fields

What's NOT round-tripped (graph-only state):
  - Derived edges from heuristic interpretation in promote.py
    (imposes_on, treats_as[role=mere_means]). These re-derive on
    promote, so a lower→promote→lower yields the same graph.
  - The Maxim node (synthesised by promote; not in any flat field).
"""

from __future__ import annotations

from typing import Any

from erisml_compiler.ir.graph.container import MoralGraph
from erisml_compiler.ir.graph.schema import NodeKind


def _strip_id_prefix(node_id: str, prefix: str) -> str:
    """Reverse the `<kind>:<local>` -> `<local>` mapping."""
    return node_id.removeprefix(f"{prefix}:")


def flat_from_graph(graph: MoralGraph) -> dict[str, list[Any]]:
    """Back-derive flat IR fields from `graph`.

    Returns a dict with keys: `stakeholders`, `events`, `commitments`,
    `norms`, `ethical_facts`. Each value is a list of the
    corresponding Pydantic model. Lists are sorted by id for
    determinism.
    """
    from erisml_compiler.ir.schemas import (
        Commitment,
        EthicalFact,
        Event,
        Norm,
        Stakeholder,
    )

    stakeholders: list[Stakeholder] = []
    events: list[Event] = []
    commitments: list[Commitment] = []
    norms: list[Norm] = []
    ethical_facts: list[EthicalFact] = []

    for n in graph.nodes:
        if n.kind == NodeKind.STAKEHOLDER:
            if n.payload:
                stakeholders.append(Stakeholder.model_validate(n.payload))
        elif n.kind == NodeKind.ACT:
            if n.payload:
                events.append(Event.model_validate(n.payload))
        elif n.kind == NodeKind.COMMITMENT:
            if n.payload:
                commitments.append(Commitment.model_validate(n.payload))
        elif n.kind == NodeKind.NORM:
            if n.payload:
                norms.append(Norm.model_validate(n.payload))
        elif n.kind == NodeKind.FACT:
            if n.payload:
                ethical_facts.append(EthicalFact.model_validate(n.payload))

    stakeholders.sort(key=lambda s: s.id)
    events.sort(key=lambda e: e.id)
    commitments.sort(key=lambda c: c.id)
    norms.sort(key=lambda n: n.id)
    ethical_facts.sort(key=lambda f: f.id)

    return {
        "stakeholders": stakeholders,
        "events": events,
        "commitments": commitments,
        "norms": norms,
        "ethical_facts": ethical_facts,
    }


def apply_flat_to_ir(ir, flat: dict[str, list[Any]]) -> None:
    """Mutate `ir` in place to set its flat fields from `flat`.

    Convenience wrapper for graph-primary code paths that have just
    edited the graph and want the flat fields refreshed."""
    if "stakeholders" in flat:
        ir.stakeholders = list(flat["stakeholders"])
    if "events" in flat:
        ir.events = list(flat["events"])
    if "commitments" in flat:
        ir.commitments = list(flat["commitments"])
    if "norms" in flat:
        ir.norms = list(flat["norms"])
    if "ethical_facts" in flat:
        ir.ethical_facts = list(flat["ethical_facts"])
