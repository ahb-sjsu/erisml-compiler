"""`MoralGraph` container with typed query API."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from erisml_compiler.ir.graph.schema import EdgeKind, MoralEdge, MoralNode, NodeKind


class MoralGraph(BaseModel):
    """A typed directed graph of moral entities and relations.

    Stored as flat lists for JSON round-trip; indexed lazily for
    efficient queries. Mutations rebuild the indices.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    nodes: list[MoralNode] = Field(default_factory=list)
    edges: list[MoralEdge] = Field(default_factory=list)

    # ----------------------- mutation -----------------------

    def add_node(self, node: MoralNode) -> None:
        for existing in self.nodes:
            if existing.id == node.id:
                # Idempotent: same id + same kind is a no-op merge.
                if existing.kind != node.kind:
                    raise ValueError(
                        f"Node id {node.id!r} already exists with kind "
                        f"{existing.kind}, cannot re-add as {node.kind}"
                    )
                existing.payload.update(node.payload)
                for lbl in node.labels:
                    if lbl not in existing.labels:
                        existing.labels.append(lbl)
                return
        self.nodes.append(node)

    def add_edge(self, edge: MoralEdge) -> None:
        self.edges.append(edge)

    # ----------------------- query --------------------------

    def get_node(self, node_id: str) -> MoralNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def nodes_of_kind(self, kind: NodeKind) -> list[MoralNode]:
        return [n for n in self.nodes if n.kind == kind]

    def edges_of_kind(self, kind: EdgeKind) -> list[MoralEdge]:
        return [e for e in self.edges if e.kind == kind]

    def out_edges(self, node_id: str, *, kind: EdgeKind | None = None) -> list[MoralEdge]:
        return [e for e in self.edges if e.src == node_id and (kind is None or e.kind == kind)]

    def in_edges(self, node_id: str, *, kind: EdgeKind | None = None) -> list[MoralEdge]:
        return [e for e in self.edges if e.dst == node_id and (kind is None or e.kind == kind)]

    def neighbors_out(self, node_id: str, *, kind: EdgeKind | None = None) -> list[str]:
        return [e.dst for e in self.out_edges(node_id, kind=kind)]

    def neighbors_in(self, node_id: str, *, kind: EdgeKind | None = None) -> list[str]:
        return [e.src for e in self.in_edges(node_id, kind=kind)]

    def has_edge(self, src: str, dst: str, *, kind: EdgeKind | None = None) -> bool:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                if kind is None or e.kind == kind:
                    return True
        return False

    # ----------------------- summary ------------------------

    def node_count_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for n in self.nodes:
            counts[n.kind.value] += 1
        return dict(counts)

    def edge_count_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in self.edges:
            counts[e.kind.value] += 1
        return dict(counts)

    # ----------------------- iteration ----------------------

    def __iter__(self) -> Any:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)
