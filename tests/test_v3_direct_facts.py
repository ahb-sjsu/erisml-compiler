"""Phase 4 — direct EthicalFactsV3 builder + per-party divergence + IR metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("erisml.ethics.facts_v3")

from erisml_compiler.erisml_backend.v3_facts_direct import ir_to_v3_facts
from erisml_compiler.ir.schemas import CompilerIR
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


# ---------- direct builder produces V3 facts ----------


def test_direct_builder_produces_v3_facts():
    """ir_to_v3_facts returns EthicalFactsV3 with non-empty per_party tuples."""
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock", tensor_rank=2),
    )
    facts = ir_to_v3_facts(ir)
    assert len(facts.consequences.per_party) == len(ir.stakeholders)
    assert len(facts.rights_and_duties.per_party) == len(ir.stakeholders)
    assert all(p.party_id for p in facts.consequences.per_party)


def test_direct_builder_attributes_facts_to_subjects():
    """A fact with subjects=['speaker'] should bump speaker's harm but
    not refugees'."""
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock", tensor_rank=2),
    )
    facts = ir_to_v3_facts(ir)
    by_id = {p.party_id: p for p in facts.consequences.per_party}
    # nazi_attic mock emits coercion subjects=['speaker', 'village']
    # at catastrophic severity -> expected_harm should be > 0 for those
    # and 0 for refugees (who receive `care`, not coercion).
    assert by_id["speaker"].expected_harm > 0.0
    assert by_id["village"].expected_harm > 0.0
    assert by_id["hidden_refugees"].expected_harm == 0.0


# ---------- bridge picks the direct path ----------


def test_phase4_build_strategy_marker(nazi_attic_ir):
    md = nazi_attic_ir.moral_tensor_v3.metadata
    assert md["build_strategy"] == "phase4_v3_bridge"
    assert md["facts_source"] == "direct"


# ---------- per-stakeholder columns diverge ----------


def test_rank2_columns_diverge_on_harm_dimension(nazi_attic_ir):
    """The whole point of Phase 4: per-stakeholder values are no longer
    uniform. At least one dimension's row should have >1 unique value."""
    t = nazi_attic_ir.moral_tensor_v3
    n = t.shape[1]
    divergent_dims = []
    for k in range(9):
        col = [round(t.get_cell(k, j), 6) for j in range(n)]
        if len(set(col)) > 1:
            divergent_dims.append(k)
    assert divergent_dims, "Phase 4 should produce ≥1 divergent dimension"


def test_physical_harm_specifically_diverges(nazi_attic_ir):
    """nazi_attic explicitly assigns coercion to ['speaker', 'village']
    and care to ['hidden_refugees']; harm must differ."""
    t = nazi_attic_ir.moral_tensor_v3
    harm_row = t.values[0]
    assert len(set(round(v, 4) for v in harm_row)) > 1


# ---------- per-party verdicts surfaced on IR ----------


def test_per_party_verdicts_populated(nazi_attic_ir):
    verdicts = nazi_attic_ir.per_party_verdicts
    assert verdicts is not None
    expected_parties = {s.id for s in nazi_attic_ir.stakeholders}
    assert set(verdicts.keys()) == expected_parties


def test_per_party_verdicts_have_meaningful_distribution(nazi_attic_ir):
    """In nazi_attic, at least one party should get a non-neutral
    verdict (forbid for the harmed parties)."""
    verdicts = nazi_attic_ir.per_party_verdicts
    assert any(v != "neutral" for v in verdicts.values()), (
        f"All verdicts neutral suggests modules didn't engage: {verdicts}"
    )


# ---------- fairness metrics surfaced on IR ----------


def test_fairness_metrics_populated(nazi_attic_ir):
    fm = nazi_attic_ir.fairness_metrics
    assert fm is not None
    assert "gini_harm" in fm
    assert "worst_off_harm_value" in fm
    assert 0.0 <= fm["gini_harm"] <= 1.0


def test_gini_nonzero_when_harm_diverges(nazi_attic_ir):
    """Per-stakeholder harm divergence implies non-zero Gini."""
    fm = nazi_attic_ir.fairness_metrics
    assert fm["gini_harm"] > 0.0


# ---------- veto location semantics ----------


def test_veto_locations_use_single_axis_convention(nazi_attic_ir):
    """V3 modules emit veto_locations as (party_idx,) rank-2 vetoes.
    Validator should accept length 1 even though tensor rank is 2."""
    t = nazi_attic_ir.moral_tensor_v3
    if t.veto_locations:
        for loc in t.veto_locations:
            assert len(loc) in (0, 1, t.rank), (
                f"veto_location {loc} has unexpected length"
            )
