"""Tests for the DAG-native IR: schema, container, canonical hash,
flat-list promotion, and end-to-end integration via the orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.ir.graph import (
    EdgeKind,
    MoralEdge,
    MoralGraph,
    MoralNode,
    NodeKind,
    canonical_graph_json,
    graph_from_flat,
    graph_hash,
)
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------- schema + container


def test_add_node_idempotent_on_same_id_same_kind() -> None:
    g = MoralGraph()
    g.add_node(MoralNode(id="x:1", kind=NodeKind.STAKEHOLDER, payload={"a": 1}))
    g.add_node(MoralNode(id="x:1", kind=NodeKind.STAKEHOLDER, payload={"b": 2}))
    assert len(g.nodes) == 1
    assert g.get_node("x:1").payload == {"a": 1, "b": 2}


def test_add_node_rejects_id_with_different_kind() -> None:
    g = MoralGraph()
    g.add_node(MoralNode(id="x:1", kind=NodeKind.STAKEHOLDER))
    with pytest.raises(ValueError, match="already exists with kind"):
        g.add_node(MoralNode(id="x:1", kind=NodeKind.ACT))


def test_query_helpers() -> None:
    g = MoralGraph(
        nodes=[
            MoralNode(id="s:alice", kind=NodeKind.STAKEHOLDER, labels=["agent"]),
            MoralNode(id="s:bob", kind=NodeKind.STAKEHOLDER),
            MoralNode(id="a:1", kind=NodeKind.ACT),
        ],
        edges=[
            MoralEdge(src="s:alice", dst="a:1", kind=EdgeKind.PERFORMS),
            MoralEdge(src="a:1", dst="s:bob", kind=EdgeKind.IMPOSES_ON,
                      payload={"severity": "grave"}),
        ],
    )
    assert len(g.nodes_of_kind(NodeKind.STAKEHOLDER)) == 2
    assert len(g.edges_of_kind(EdgeKind.PERFORMS)) == 1
    assert g.has_edge("a:1", "s:bob", kind=EdgeKind.IMPOSES_ON)
    assert not g.has_edge("s:bob", "a:1", kind=EdgeKind.CONSENTS_TO)
    assert g.neighbors_out("a:1") == ["s:bob"]
    assert g.neighbors_in("a:1") == ["s:alice"]
    assert g.node_count_by_kind() == {"stakeholder": 2, "act": 1}


# ---------------------------------------------------- canonical hash


def test_graph_hash_deterministic_under_insertion_order() -> None:
    g1 = MoralGraph()
    g1.add_node(MoralNode(id="s:a", kind=NodeKind.STAKEHOLDER, labels=["agent"]))
    g1.add_node(MoralNode(id="s:b", kind=NodeKind.STAKEHOLDER))
    g1.add_edge(MoralEdge(src="s:a", dst="s:b", kind=EdgeKind.COERCES, payload={"severity": "grave"}))

    g2 = MoralGraph()
    g2.add_node(MoralNode(id="s:b", kind=NodeKind.STAKEHOLDER))
    g2.add_node(MoralNode(id="s:a", kind=NodeKind.STAKEHOLDER, labels=["agent"]))
    g2.add_edge(MoralEdge(src="s:a", dst="s:b", kind=EdgeKind.COERCES, payload={"severity": "grave"}))

    assert graph_hash(g1) == graph_hash(g2)


def test_graph_hash_differs_under_payload_change() -> None:
    g1 = MoralGraph(edges=[MoralEdge(src="a", dst="b", kind=EdgeKind.IMPOSES_ON, payload={"severity": "minor"})])
    g2 = MoralGraph(edges=[MoralEdge(src="a", dst="b", kind=EdgeKind.IMPOSES_ON, payload={"severity": "grave"})])
    assert graph_hash(g1) != graph_hash(g2)


def test_canonical_graph_json_sorts_consistently() -> None:
    g = MoralGraph()
    g.add_node(MoralNode(id="z", kind=NodeKind.NORM, labels=["c", "a", "b"]))
    g.add_node(MoralNode(id="a", kind=NodeKind.MAXIM))
    js = canonical_graph_json(g)
    # Nodes should appear in id-sorted order, labels in sorted order.
    a_pos = js.index('"id":"a"')
    z_pos = js.index('"id":"z"')
    assert a_pos < z_pos
    assert '"labels":["a","b","c"]' in js


# ---------------------------------------------------- promote


def _compile(name: str):
    return compile_document(
        REPO_ROOT / "examples" / f"{name}.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="rule",
            canonicalizer=RegistryCanonicalizer(), tensor_rank=2,
        ),
    )


def test_graph_promoted_from_nazi_attic() -> None:
    ir = _compile("nazi_attic")
    g = ir.graph
    assert g is not None
    counts = g.node_count_by_kind()
    # nazi_attic produces: 2 stakeholders (self, village), 1 act,
    # 1 commitment, 4 facts, 1 maxim.
    assert counts.get("stakeholder", 0) >= 2
    assert counts.get("act", 0) >= 1
    assert counts.get("commitment", 0) >= 1
    assert counts.get("fact", 0) >= 1
    assert counts.get("maxim", 0) >= 1


def test_graph_has_imposes_on_edge_for_village_externality() -> None:
    ir = _compile("nazi_attic")
    g = ir.graph
    # The externality fact maps the act -> village via imposes_on.
    imposes = g.edges_of_kind(EdgeKind.IMPOSES_ON)
    assert any("village" in e.dst for e in imposes), (
        f"Expected an imposes_on edge targeting the village; "
        f"got: {[(e.src, e.dst) for e in imposes]}"
    )


def test_graph_has_under_maxim_edge() -> None:
    ir = _compile("nazi_attic")
    g = ir.graph
    assert g.edges_of_kind(EdgeKind.UNDER_MAXIM)


def test_graph_hash_recorded_in_audit() -> None:
    ir = _compile("nazi_attic")
    assert ir.audit is not None
    assert ir.audit.graph_hash is not None
    assert len(ir.audit.graph_hash) == 64


def test_graph_hash_reproducible_across_compiles() -> None:
    ir1 = _compile("nazi_attic")
    ir2 = _compile("nazi_attic")
    assert ir1.audit.graph_hash == ir2.audit.graph_hash


def test_graph_hash_differs_across_scenarios() -> None:
    g_nazi = _compile("nazi_attic").audit.graph_hash
    g_med = _compile("medical_confidentiality").audit.graph_hash
    g_whistle = _compile("whistleblower").audit.graph_hash
    assert len({g_nazi, g_med, g_whistle}) == 3


# ---------------------------------------------------- promote — empty / edge cases


def test_graph_from_flat_handles_empty_ir() -> None:
    """An IR with no extracted facts still yields a (mostly empty) graph."""
    from erisml_compiler.ir.schemas import CompilerIR, Document

    doc = Document(doc_id="empty", title="empty", raw_text="")
    ir = CompilerIR(document=doc)
    g = graph_from_flat(ir)
    assert isinstance(g, MoralGraph)
    assert g.node_count_by_kind() == {}
