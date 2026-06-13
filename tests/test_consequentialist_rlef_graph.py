"""Tests for ConsequentialistProjection graph-awareness + RLEF graph export."""

from __future__ import annotations

import json
from pathlib import Path

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.export.rlef import to_rlef_record
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

REPO_ROOT = Path(__file__).resolve().parent.parent


def _compile(name: str):
    return compile_document(
        REPO_ROOT / "examples" / f"{name}.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
        ),
    )


# -------------------------------------- consequentialist graph-awareness


def test_consequentialist_records_graph_aware_metadata() -> None:
    ir = _compile("nazi_attic")
    cres = ir.projections["consequentialist_distributive"]
    assert cres["metadata"]["graph_aware"] is True
    gs = cres["metadata"]["graph_summary"]
    assert gs["n_stakeholders"] >= 1
    assert gs["n_imposes_on_edges"] >= 1  # village externality
    assert gs["n_coerces_edges"] >= 0
    assert gs["n_facts"] >= 1


def test_consequentialist_graph_summary_distinguishes_scenarios() -> None:
    nazi = _compile("nazi_attic").projections["consequentialist_distributive"]["metadata"][
        "graph_summary"
    ]
    med = _compile("medical_confidentiality").projections["consequentialist_distributive"][
        "metadata"
    ]["graph_summary"]
    # The two scenarios have meaningfully different graph topologies.
    assert nazi != med


# -------------------------------------- RLEF graph export


def test_rlef_record_schema_bumped_to_v02() -> None:
    ir = _compile("nazi_attic")
    rec = to_rlef_record(ir)
    assert rec["schema"] == "rlef_v0.2"


def test_rlef_record_includes_moral_graph_block() -> None:
    ir = _compile("nazi_attic")
    rec = to_rlef_record(ir)
    assert "moral_graph" in rec
    g = rec["moral_graph"]
    assert g is not None
    assert "graph_hash" in g
    assert "nodes" in g and len(g["nodes"]) > 0
    assert "edges" in g and len(g["edges"]) > 0
    assert "node_counts" in g and "edge_counts" in g
    # Canonical JSON is included for trainer downstream
    assert "canonical_json" in g
    parsed = json.loads(g["canonical_json"])
    assert "nodes" in parsed and "edges" in parsed


def test_rlef_includes_per_projection_verdicts() -> None:
    ir = _compile("nazi_attic")
    rec = to_rlef_record(ir)
    assert "projections" in rec
    assert "deontic_kantian" in rec["projections"]
    assert "virtue_aristotelian" in rec["projections"]
    assert "care_ethics_relational" in rec["projections"]


def test_rlef_record_serialisable_to_json() -> None:
    ir = _compile("nazi_attic")
    rec = to_rlef_record(ir)
    s = json.dumps(rec, default=str)
    assert "rlef_v0.2" in s


def test_rlef_graph_hash_matches_audit_graph_hash() -> None:
    ir = _compile("nazi_attic")
    rec = to_rlef_record(ir)
    assert rec["moral_graph"]["graph_hash"] == ir.audit.graph_hash
