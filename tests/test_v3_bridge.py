"""Phase 3 — V3 bridge wires compiler IR to DEME V3 modules."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("erisml.ethics.modules.tier0.geneva_em_v3")

from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def bridge_ir() -> CompilerIR:
    return compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )


def test_bridge_invoked_when_erisml_lib_available(bridge_ir):
    """The bridge path tags itself with one of the supported strategy
    names. As migration progresses, accept any phaseN_v3_bridge marker;
    fail loudly if the orchestrator silently fell back to phase 2."""
    md = bridge_ir.moral_tensor_v3.metadata
    strategy = md.get("build_strategy")
    assert strategy in {"phase3_v3_bridge", "phase4_v3_bridge"}, (
        f"expected v3 bridge path, got {strategy}"
    )


def test_modules_recorded_in_metadata(bridge_ir):
    md = bridge_ir.moral_tensor_v3.metadata
    modules = md.get("modules_invoked", [])
    # The bridge ships with at least Geneva + Triage as the default set.
    assert "geneva_v3" in modules
    assert "triage_v3" in modules


def test_bridge_produces_non_trivial_values(bridge_ir):
    """Confirms the V3 modules actually evaluate. After the V3 modules
    run, at least one dimension should have a non-zero value."""
    t = bridge_ir.moral_tensor_v3
    flat = [v for row in t.values for v in row]
    assert any(v != 0.0 for v in flat), "V3 bridge produced an all-zero tensor"


def test_bridge_shape_matches_stakeholders(bridge_ir):
    t = bridge_ir.moral_tensor_v3
    assert t.shape[0] == 9
    assert t.shape[1] == len(bridge_ir.stakeholders)


def test_bridge_rank1_collapse():
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=1,
        ),
    )
    assert ir.moral_tensor_v3.rank == 1
    assert ir.moral_tensor_v3.shape == (9,)
    assert ir.moral_tensor_v3.metadata.get("collapsed_from_rank2") is True


def test_bridge_v2_facts_aggregator_directly():
    """Spot-check the IR -> V2 facts heuristic without invoking modules."""
    from erisml_compiler.erisml_backend.v3_bridge import ir_to_v2_facts

    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )
    v2 = ir_to_v2_facts(ir)
    # nazi_attic has harm + coercion facts; we should see Consequences
    # affected_count > 0 and either harm > 0 or coercion flagged.
    assert v2.consequences.affected_count == len(ir.stakeholders)
    has_harm_or_coercion = (
        v2.consequences.expected_harm > 0.0
        or v2.autonomy_and_agency.coercion_or_undue_influence
    )
    assert has_harm_or_coercion, "nazi_attic should produce harm or coercion signal"


def test_bridge_falls_back_to_phase2_on_module_exception(monkeypatch):
    """If the V3 bridge raises, the orchestrator should silently fall
    back to the Phase 2 fanout. Verifies the dispatch logic."""
    from erisml_compiler.erisml_backend import v3_bridge as bridge_mod

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(bridge_mod, "compile_to_v3_tensor", boom)
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES, extractor="mock", canonicalizer=None,
            tensor_rank=2,
        ),
    )
    # Bridge failed -> Phase 2 fanout path was used.
    assert ir.moral_tensor_v3.metadata.get("build_strategy") == "phase2_fanout_from_rank1"
