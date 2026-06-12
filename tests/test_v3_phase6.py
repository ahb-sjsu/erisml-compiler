"""Phase 6 — strategic analysis + decision proof + real coalition semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("erisml.ethics.game_theory")

from erisml_compiler.erisml_backend.v3_phase6 import (
    build_coalition_c_axis_slices,
    build_decision_proof,
    compute_strategic_analysis,
)
from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _compile(rank: int = 2, **kw) -> CompilerIR:
    return compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="mock",
            tensor_rank=rank,
            **kw,
        ),
    )


@pytest.fixture(scope="module")
def ir_with_phase6() -> CompilerIR:
    return _compile(2)


# ---------- strategic analysis attached ----------


def test_strategic_analysis_is_populated(ir_with_phase6):
    sa = ir_with_phase6.strategic_analysis
    assert sa is not None
    assert sa["n_agents"] == len(ir_with_phase6.stakeholders)
    assert "shapley_values" in sa
    assert "shapley_method" in sa
    assert "welfare_metrics" in sa


def test_shapley_values_one_per_stakeholder(ir_with_phase6):
    sa = ir_with_phase6.strategic_analysis
    expected = {s.id for s in ir_with_phase6.stakeholders}
    assert set(sa["shapley_values"].keys()) == expected
    for v in sa["shapley_values"].values():
        assert isinstance(v, (int, float))


def test_welfare_metrics_have_harm_summary(ir_with_phase6):
    metrics = ir_with_phase6.strategic_analysis["welfare_metrics"]
    for key in ("mean_harm", "max_harm", "min_harm", "harm_range"):
        assert key in metrics
    # max ≥ mean ≥ min by construction
    assert metrics["max_harm"] >= metrics["mean_harm"] >= metrics["min_harm"]


def test_strategic_analysis_skipped_when_disabled():
    """When emit_strategic_analysis=False, IR should not carry it."""
    ir = _compile(2, emit_strategic_analysis=False, emit_decision_proof=False)
    assert ir.strategic_analysis is None


# ---------- decision proof attached + hash chain ----------


def test_decision_proof_is_populated(ir_with_phase6):
    dp = ir_with_phase6.decision_proof
    assert dp is not None
    for key in (
        "decision_id",
        "timestamp",
        "input_facts_hash",
        "profile_hash",
        "proof_hash",
        "layer_outputs",
        "ranked_options",
        "forbidden_options",
        "previous_proof_hash",
        "moral_vector_summary",
    ):
        assert key in dp, f"DecisionProof missing key {key}"


def test_decision_proof_hash_chain_to_audit(ir_with_phase6):
    """proof.previous_proof_hash should chain to the IR's audit hash."""
    dp = ir_with_phase6.decision_proof
    if ir_with_phase6.audit and ir_with_phase6.audit.ir_hash:
        assert dp["previous_proof_hash"] == ir_with_phase6.audit.ir_hash


def test_decision_proof_proof_hash_is_deterministic(ir_with_phase6):
    """Same IR + tensor should produce the same proof_hash (modulo
    decision_id and timestamp). Verify by rebuilding with same inputs
    and comparing the hashes of the canonical parts."""
    import hashlib
    import json

    dp1 = ir_with_phase6.decision_proof
    dp2 = build_decision_proof(
        ir_with_phase6,
        ir_with_phase6.moral_tensor_v3,
        strategic_analysis=ir_with_phase6.strategic_analysis,
    )

    def _canon(p):
        stable = {k: v for k, v in p.items() if k not in ("decision_id", "timestamp", "proof_hash")}
        # layer_outputs have per-layer timestamps too
        if "layer_outputs" in stable:
            stable["layer_outputs"] = [
                {k: v for k, v in L.items() if k != "timestamp"} for L in stable["layer_outputs"]
            ]
        return hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()

    assert _canon(dp1) == _canon(dp2)


def test_decision_proof_forbidden_matches_per_party_verdicts(ir_with_phase6):
    """Stakeholders with verdict='forbid' should be in proof.forbidden_options."""
    ppv = ir_with_phase6.per_party_verdicts
    dp = ir_with_phase6.decision_proof
    if ppv:
        expected_forbidden = {pid for pid, v in ppv.items() if v == "forbid"}
        assert set(dp["forbidden_options"]) == expected_forbidden


def test_decision_proof_skipped_when_disabled():
    ir = _compile(2, emit_strategic_analysis=False, emit_decision_proof=False)
    assert ir.decision_proof is None


# ---------- real coalition semantics on c axis ----------


def test_rank4_with_real_coalitions_varies_on_c():
    """coalition_mode='all_subsets' should produce real per-coalition
    variation on the c axis, not the Phase-5 stub replication."""
    ir = _compile(4, tensor_n_coalitions=4, coalition_mode="all_subsets")
    t = ir.moral_tensor_v3
    arr = np.array(t.values)
    c_std = arr.std(axis=3)
    assert c_std.max() > 0.0, "coalition c axis should have non-zero std under real semantics"
    assert "c" not in t.metadata.get("stub_axes", [])
    assert "real" in t.metadata["axis_semantics"]["c"]


def test_grand_only_coalition_still_stub():
    """coalition_mode='grand_only' produces only one coalition (the
    whole stakeholder set); replicating that across n_coalitions=4 is
    Phase 5's stub behaviour."""
    ir = _compile(4, tensor_n_coalitions=4, coalition_mode="grand_only")
    t = ir.moral_tensor_v3
    arr = np.array(t.values)
    c_std = arr.std(axis=3)
    assert c_std.max() < 1e-9
    assert "c" in t.metadata.get("stub_axes", [])


def test_coalition_slice_builder_directly():
    ir = _compile(2)
    slices, labels = build_coalition_c_axis_slices(
        ir,
        coalition_mode="singletons_only",
        n_coalitions_requested=4,
    )
    assert len(slices) == 4
    assert len(labels) == 4
    for s in slices:
        assert s.shape == (9, len(ir.stakeholders))


def test_singleton_coalitions_zero_out_other_parties():
    """When coalition_mode='singletons_only', each slice has only one
    party with non-zero entries — the rest are zeroed."""
    ir = _compile(2)
    slices, _ = build_coalition_c_axis_slices(
        ir,
        coalition_mode="singletons_only",
        n_coalitions_requested=len(ir.stakeholders),
    )
    for i, s in enumerate(slices):
        # Count non-zero columns; should be exactly 1.
        col_norms = np.abs(s).sum(axis=0)  # (n,)
        non_zero_cols = int((col_norms > 1e-9).sum())
        assert non_zero_cols == 1, (
            f"coalition slice {i} should have exactly 1 non-zero stakeholder column; "
            f"got {non_zero_cols}"
        )


# ---------- direct helpers + handling of missing erisml-lib path ----------


def test_compute_strategic_analysis_returns_none_for_single_stakeholder():
    """One-stakeholder scenarios have no Shapley to compute."""
    ir = _compile(2)
    # Force single-stakeholder IR.
    ir.stakeholders = ir.stakeholders[:1]
    sa = compute_strategic_analysis(ir, ir.moral_tensor_v3)
    assert sa is None


def test_strategic_analysis_includes_recommendations_block(ir_with_phase6):
    """welfare_metrics + Shapley should at least give us a coherent
    summary block. The DecisionProof captures it as the 'strategic'
    layer output."""
    dp = ir_with_phase6.decision_proof
    layer_names = [L["layer_name"] for L in dp["layer_outputs"]]
    assert "strategic" in layer_names
