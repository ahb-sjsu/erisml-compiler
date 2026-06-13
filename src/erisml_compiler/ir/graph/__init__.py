"""DAG-native IR: the descriptive substrate as a typed directed graph.

The compiler now compiles facts-about-the-world into a `MoralGraph` —
a typed DAG where nodes are entities (stakeholder, act, maxim,
commitment, fact, norm) and edges are typed moral relations
(performs, imposes_on, consents_to, treats_as, holds_commitment,
under_maxim, coerces, surfaces_fact, fact_subject,
would_violate_if_universalised).

This replaces the flat-list MoralSubstrate as the *primary*
representation. The flat lists on `CompilerIR` (`stakeholders`,
`events`, `commitments`, `ethical_facts`, `relations`) are now
**derived views** populated by the extractor and promoted to a graph
at Pass 7.5. Projections read the graph directly via typed queries;
flat-field consumers continue to work because the orchestrator
maintains both representations.

See `docs/plans/release-planning-06-framework-pluralist-architecture.md`
for the architectural argument. The DAG-native refactor is the
follow-up to the two-layer (substrate + projections) split shipped
earlier — it makes the substrate first-class graph-shaped instead of
a flat record.

The graph carries a canonical SHA-256 hash (`graph_hash`) included in
the audit record. Same node+edge content → same hash regardless of
insertion order, so two compiles of the same input produce
bit-identical audit chains.
"""
from erisml_compiler.ir.graph.canonical import (
    canonical_graph_json,
    graph_hash,
)
from erisml_compiler.ir.graph.container import MoralGraph
from erisml_compiler.ir.graph.promote import graph_from_flat
from erisml_compiler.ir.graph.schema import (
    EdgeKind,
    MoralEdge,
    MoralNode,
    NodeKind,
)

__all__ = [
    "EdgeKind",
    "MoralEdge",
    "MoralGraph",
    "MoralNode",
    "NodeKind",
    "canonical_graph_json",
    "graph_from_flat",
    "graph_hash",
]
