"""Phase 2 — orchestrator produces ir.moral_tensor_v3 at the requested rank."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.ir.v3 import MORAL_DIMENSIONS_V3
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def nazi_attic_ir() -> CompilerIR:
    return compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )


@pytest.fixture(scope="module")
def medical_confidentiality_ir() -> CompilerIR:
    return compile_document(
        EXAMPLES_DIR / "medical_confidentiality.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )


# ---------- tensor shape ----------


def test_rank2_tensor_attached_to_ir(nazi_attic_ir):
    t = nazi_attic_ir.moral_tensor_v3
    assert t is not None
    assert t.rank == 2
    assert t.shape[0] == 9
    assert t.axis_names == ("k", "n")


def test_n_axis_labels_are_stakeholder_ids(nazi_attic_ir):
    t = nazi_attic_ir.moral_tensor_v3
    n_labels = t.axis_labels.get("n", [])
    assert n_labels  # at least one stakeholder
    ir_ids = {s.id for s in nazi_attic_ir.stakeholders}
    assert set(n_labels) == ir_ids


def test_k_axis_labels_match_canonical_dimensions(nazi_attic_ir):
    t = nazi_attic_ir.moral_tensor_v3
    assert tuple(t.axis_labels["k"]) == MORAL_DIMENSIONS_V3


# ---------- rank 1 ----------


def test_rank1_produces_global_vector():
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=1,
        ),
    )
    assert ir.moral_tensor_v3 is not None
    assert ir.moral_tensor_v3.rank == 1
    assert ir.moral_tensor_v3.shape == (9,)


# ---------- migration metadata preserved ----------


def test_build_strategy_metadata_recorded(nazi_attic_ir):
    """Every supported producer tags itself in metadata.build_strategy
    so downstream tooling can route on it. Accept any of the currently
    supported producers; the rest of the suite makes per-phase
    assertions about which one was selected."""
    md = nazi_attic_ir.moral_tensor_v3.metadata
    assert md.get("build_strategy") in {
        "phase2_fanout_from_rank1",   # erisml-lib unavailable
        "phase3_v3_bridge",            # bridge active, V2 facts aggregation
        "phase4_v3_bridge",            # bridge active, direct V3 facts
    }


def test_phase2_fallback_carries_migration_metadata(monkeypatch):
    """When the V3 bridge isn't available, the orchestrator falls back
    to Phase 2's V2-migration tensor builder which embeds a `migration`
    metadata block plus `repair_residue`. Force that path via the
    bridge's ImportError check."""
    from erisml_compiler.erisml_backend import v3_bridge as bridge_mod

    def fake_import_error(*args, **kwargs):
        raise ImportError("simulated missing erisml-lib")

    monkeypatch.setattr(bridge_mod, "compile_to_v3_tensor", fake_import_error)
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )
    md = ir.moral_tensor_v3.metadata
    assert md["build_strategy"] == "phase2_fanout_from_rank1"
    assert md.get("migration", {}).get("source") == "v2_moral_vector"
    assert "privacy_protection" in md["migration"]["synthesised_dims"]
    assert "repair_residue" in md


# ---------- V2 surface still produced ----------


def test_v2_moral_vectors_still_populated(nazi_attic_ir):
    assert nazi_attic_ir.moral_vectors  # at least one entry
    v2 = nazi_attic_ir.moral_vectors[0]
    # All 10 V2 dims still present.
    for dim in [
        "physical_harm", "rights_respect", "fairness_equity",
        "autonomy_consent", "legitimacy_trust", "epistemic_quality",
        "care_protection", "vow_fidelity",
        "third_party_externality", "repair_residue",
    ]:
        assert hasattr(v2, dim)


def test_v2_timeline_still_populated(nazi_attic_ir):
    assert nazi_attic_ir.timeline


# ---------- JSON roundtrip ----------


def test_compile_output_v3_tensor_roundtrips_json(tmp_path, nazi_attic_ir):
    out = tmp_path / "ir.json"
    out.write_text(nazi_attic_ir.model_dump_json(), encoding="utf-8")
    loaded = CompilerIR.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded.moral_tensor_v3 is not None
    assert loaded.moral_tensor_v3.rank == nazi_attic_ir.moral_tensor_v3.rank
    assert loaded.moral_tensor_v3.shape == nazi_attic_ir.moral_tensor_v3.shape


# ---------- rank fanout invariant ----------


def test_rank2_columns_diverge_on_at_least_one_dimension(nazi_attic_ir):
    """Phase 4 inverts the Phase 2/3 invariant: per-party values must
    diverge on at least one dimension when the underlying facts
    distinguish between stakeholders. nazi_attic does — coercion
    targets [speaker, village]; care targets [hidden_refugees]."""
    t = nazi_attic_ir.moral_tensor_v3
    n = t.shape[1]
    divergent_dims = []
    for k in range(9):
        col = [round(t.get_cell(k, j), 6) for j in range(n)]
        if len(set(col)) > 1:
            divergent_dims.append(k)
    # Phase 4 should produce divergence; Phase 2 fallback wouldn't.
    # We only enforce the assertion when the bridge path actually ran.
    md = t.metadata.get("build_strategy", "")
    if "v3_bridge" in md and "phase4" in md:
        assert divergent_dims, (
            "Phase 4 v3 bridge should produce ≥1 divergent dimension; got none"
        )


# ---------- multi-scenario sanity ----------


def test_medical_confidentiality_also_produces_v3(medical_confidentiality_ir):
    t = medical_confidentiality_ir.moral_tensor_v3
    assert t is not None
    assert t.rank == 2
    assert t.shape[0] == 9
    assert t.shape[1] >= 2  # at least two stakeholders


# ---------- rank cap ----------


def test_rank_above_2_raises_helpful_error():
    with pytest.raises(NotImplementedError, match="Phase 2 supports rank 1 and 2"):
        compile_document(
            EXAMPLES_DIR / "nazi_attic.txt",
            CompileOptions(
                tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
                tensor_rank=3,
            ),
        )
