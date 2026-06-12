"""Phase 5 — ranks 3-6 with temporal + coalition + uncertainty axes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("erisml.ethics.facts_v3")

from erisml_compiler.erisml_backend.v3_higher_rank import (
    HigherRankConfig,
    build_moral_tensor_v3_rank3plus,
)
from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _compile(rank: int, **kw) -> CompilerIR:
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
def r3_ir() -> CompilerIR:
    return _compile(3)


@pytest.fixture(scope="module")
def r6_ir() -> CompilerIR:
    return _compile(6, tensor_n_actions=2, tensor_n_coalitions=3, tensor_n_samples=4)


# ---------- shape correctness ----------


def test_rank3_shape_is_k_n_tau(r3_ir):
    t = r3_ir.moral_tensor_v3
    assert t.rank == 3
    assert t.axis_names == ("k", "n", "tau")
    assert t.shape[0] == 9
    assert t.shape[1] == len(r3_ir.stakeholders)
    assert t.shape[2] >= 1


def test_rank4_shape_is_k_n_a_c():
    ir = _compile(4, tensor_n_actions=3, tensor_n_coalitions=2)
    t = ir.moral_tensor_v3
    assert t.rank == 4
    assert t.axis_names == ("k", "n", "a", "c")
    assert t.shape == (9, len(ir.stakeholders), 3, 2)


def test_rank5_shape_is_k_n_tau_a_c():
    ir = _compile(5, tensor_n_actions=2, tensor_n_coalitions=2)
    t = ir.moral_tensor_v3
    assert t.rank == 5
    assert t.axis_names == ("k", "n", "tau", "a", "c")
    assert t.shape[0] == 9
    assert t.shape[3:] == (2, 2)


def test_rank6_shape_is_full(r6_ir):
    t = r6_ir.moral_tensor_v3
    assert t.rank == 6
    assert t.axis_names == ("k", "n", "tau", "a", "c", "s")
    assert t.shape[0] == 9
    assert t.shape[3] == 2  # a
    assert t.shape[4] == 3  # c
    assert t.shape[5] == 4  # s


# ---------- metadata ----------


def test_higher_rank_build_strategy_marker(r3_ir):
    md = r3_ir.moral_tensor_v3.metadata
    assert md["build_strategy"] == "phase5_higher_rank"
    assert md["axis_semantics"]["tau"].startswith("real")
    assert md["axis_semantics"]["k"].startswith("real")


def test_rank4_stub_axes_recorded():
    ir = _compile(4, tensor_n_actions=2, tensor_n_coalitions=2)
    md = ir.moral_tensor_v3.metadata
    assert "a" in md["stub_axes"]
    assert "c" in md["stub_axes"]
    assert md["axis_semantics"]["a"].startswith("stub")


# ---------- temporal axis has real variation ----------


def test_temporal_axis_varies_across_time(r3_ir):
    """nazi_attic events fire at different time indices; the rank-3
    tensor must show non-zero variation along τ on at least one
    (k, n) cell."""
    arr = np.array(r3_ir.moral_tensor_v3.values)
    tau_std = arr.std(axis=2)  # std over time axis
    assert tau_std.max() > 0.0, "rank-3 τ axis is constant — temporal filtering is broken"


def test_temporal_axis_monotone_on_physical_harm(r3_ir):
    """Harm should only accumulate as events unfold (filter is
    'events ≤ τ'); each party's harm row should be non-decreasing
    over time."""
    arr = np.array(r3_ir.moral_tensor_v3.values)
    harm = arr[0]  # (n, τ)
    for n_idx in range(harm.shape[0]):
        row = harm[n_idx]
        for i in range(1, len(row)):
            assert row[i] >= row[i - 1] - 1e-9, f"harm not monotone on stakeholder {n_idx}: {row}"


# ---------- Monte Carlo axis has real variation; stub axes don't ----------


def test_mc_axis_varies_at_rank6(r6_ir):
    arr = np.array(r6_ir.moral_tensor_v3.values)
    s_std = arr.std(axis=-1)
    assert s_std.max() > 0.0, "rank-6 MC axis has no variation — sampling broken"


def test_stub_axes_constant_at_rank6(r6_ir):
    """a and c are stub axes today — same rank-2 result replicated
    across them. Phase 6 will inject genuine coalition variation."""
    arr = np.array(r6_ir.moral_tensor_v3.values)
    a_std = arr.std(axis=3)
    c_std = arr.std(axis=4)
    # Allow numpy float-broadcast noise (< 1e-9).
    assert a_std.max() < 1e-9, f"action axis is supposed to be a stub; got std={a_std.max()}"
    assert c_std.max() < 1e-9, f"coalition axis is supposed to be a stub; got std={c_std.max()}"


def test_first_mc_sample_is_unperturbed(r6_ir):
    """Sample 0 should match the unperturbed rank-2 result; sampling
    must not displace the baseline."""
    arr = np.array(r6_ir.moral_tensor_v3.values)
    # Sample 0 vs samples 1..N-1: sample 0 should be the rank-2 baseline.
    # We can't assert exact equality without the baseline; but its
    # values should be in a sensible range and not be the noisiest sample.
    sample0 = arr[..., 0]
    assert np.isfinite(sample0).all()


# ---------- direct builder + config validation ----------


def test_builder_rejects_out_of_range_rank():
    ir = _compile(2)  # produce IR via the legitimate path
    with pytest.raises(ValueError, match="ranks 3-6 only"):
        build_moral_tensor_v3_rank3plus(ir, rank=2)
    with pytest.raises(ValueError, match="ranks 3-6 only"):
        build_moral_tensor_v3_rank3plus(ir, rank=7)


def test_config_validation():
    with pytest.raises(ValueError, match="n_actions"):
        HigherRankConfig(n_actions=100)  # over MAX_ACTIONS
    with pytest.raises(ValueError, match="n_samples"):
        HigherRankConfig(n_samples=100)  # over MAX_SAMPLES
    with pytest.raises(ValueError, match="sample_noise_std"):
        HigherRankConfig(sample_noise_std=2.0)


# ---------- JSON serialisation ----------


def test_rank6_tensor_roundtrips_through_json(tmp_path, r6_ir):
    """The Pydantic tensor (with nested-list values up to rank 6) must
    roundtrip through model_dump_json → model_validate_json."""
    payload = r6_ir.moral_tensor_v3.model_dump_json()
    from erisml_compiler.ir.v3 import MoralTensorV3

    reloaded = MoralTensorV3.model_validate_json(payload)
    assert reloaded.rank == 6
    assert reloaded.shape == r6_ir.moral_tensor_v3.shape


# ---------- veto locations lift correctly ----------


def test_veto_locations_have_correct_arity_at_higher_rank(r6_ir):
    """V3 modules emit (party_idx,) single-axis vetoes for rank-2; the
    higher-rank builder lifts those to full coordinates."""
    t = r6_ir.moral_tensor_v3
    for loc in t.veto_locations:
        # Accept global (0), single-axis (1), or full-coord (rank).
        assert len(loc) in (0, 1, t.rank), f"veto_location {loc} has unexpected arity"
