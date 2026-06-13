"""Tests for the eigenvalue-based spectral summary.

See docs/plans/release-planning-04-eigenvalue-scalar.md.

Covers:
  - zero tensor edge case
  - single-dimension tensor (one non-zero row -> one non-zero eigenvalue)
  - uniform tensor (high effective rank, low spread)
  - Frobenius identity (sum of mode-m eigenvalues == ||T||_F^2 for any m)
  - sign-fix convention
  - rank-3+ per-axis spectrum coverage
  - JSON roundtrip through MoralTensorV3.metadata
  - end-to-end attachment via the orchestrator
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from erisml_compiler.evaluation.spectral import (
    SpectralSummary,
    attach_spectral_summary,
    compute_spectral_summary,
    principal_dimension_label,
)
from erisml_compiler.ir.v3 import MORAL_DIMENSIONS_V3, MoralTensorV3

# ---------- edge cases ---------------------------------------------------


def _rank2_tensor_with_values(values: np.ndarray, n_labels: list[str]) -> MoralTensorV3:
    return MoralTensorV3(
        rank=2,
        shape=(9, len(n_labels)),
        axis_names=("k", "n"),
        axis_labels={"k": list(MORAL_DIMENSIONS_V3), "n": n_labels},
        values=values.tolist(),
    )


def test_zero_tensor_has_zero_total_stress():
    t = _rank2_tensor_with_values(np.zeros((9, 4)), ["a", "b", "c", "d"])
    s = compute_spectral_summary(t)
    assert s.total_stress == 0.0
    assert s.total_stress_squared == 0.0
    assert s.principal_stress == 0.0
    assert s.stress_spread == 0.0
    assert s.effective_moral_rank == 0.0
    assert s.principal_concentration == 0.0


def test_single_dimension_tensor_concentrates_on_that_axis():
    """Only the physical_harm row (k=0) is non-zero -> the principal
    axis must point at physical_harm."""
    vals = np.zeros((9, 4))
    vals[0, :] = [0.5, 0.7, 0.3, 0.2]
    t = _rank2_tensor_with_values(vals, ["a", "b", "c", "d"])
    s = compute_spectral_summary(t)
    # Only one non-zero eigenvalue.
    assert s.principal_stress > 0
    assert s.effective_moral_rank == pytest.approx(1.0, abs=1e-9)
    # Principal axis is the physical_harm one-hot (sign-fixed positive).
    assert s.principal_axis[0] == pytest.approx(1.0, abs=1e-9)
    for i in range(1, 9):
        assert abs(s.principal_axis[i]) < 1e-9


def test_uniform_tensor_spreads_across_all_dimensions():
    """All cells equal to 1 -> moral magnitude is high, principal-
    axis concentration is the rank-1-of-9 case (effective rank 1
    actually, because the row is constant across n -> rank-1
    matrix). What we want is: every dimension equally loaded."""
    vals = np.ones((9, 4))
    t = _rank2_tensor_with_values(vals, ["a", "b", "c", "d"])
    s = compute_spectral_summary(t)
    # T*T^T is 9x9 with all entries equal to 4 -> single non-zero
    # eigenvalue 36, eigenvector = ones/sqrt(9). So principal axis
    # is the uniform direction.
    expected_axis_value = 1.0 / np.sqrt(9.0)
    for v in s.principal_axis:
        assert abs(v - expected_axis_value) < 1e-6


def test_diagonal_tensor_distributes_eigenvalues():
    """A tensor where the k-th row has only the k-th column non-zero
    yields nine equal non-zero eigenvalues (effective rank == 9)."""
    vals = np.eye(9)
    # Need square for an exact 9-equal eigenvalue spectrum.
    t = MoralTensorV3(
        rank=2,
        shape=(9, 9),
        axis_names=("k", "n"),
        axis_labels={"k": list(MORAL_DIMENSIONS_V3), "n": [f"p{i}" for i in range(9)]},
        values=vals.tolist(),
    )
    s = compute_spectral_summary(t)
    # All 9 eigenvalues equal -> effective_moral_rank == 9.
    assert s.effective_moral_rank == pytest.approx(9.0, abs=1e-6)
    # stress_spread is 0 because lambda_1 == lambda_2.
    assert s.stress_spread == pytest.approx(0.0, abs=1e-9)


# ---------- Frobenius identity ------------------------------------------


def test_frobenius_identity_across_modes():
    """sum of mode-m eigenvalues == ||T||_F^2 for every mode m."""
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(9, 5)) * 0.4
    t = _rank2_tensor_with_values(vals, ["p0", "p1", "p2", "p3", "p4"])
    s = compute_spectral_summary(t)
    expected = float(np.sum(vals * vals))
    assert s.total_stress_squared == pytest.approx(expected, rel=1e-9)
    for axis in s.per_axis:
        assert sum(axis.eigenvalues) == pytest.approx(
            expected, rel=1e-6
        ), f"axis {axis.axis_name} sum != ||T||_F^2"


def test_total_stress_equals_frobenius_norm():
    vals = np.array([[0.1, 0.2], [0.3, 0.4], [0.0, 0.5], [0.0, 0.0]] + [[0.0, 0.0]] * 5)
    t = _rank2_tensor_with_values(vals, ["a", "b"])
    s = compute_spectral_summary(t)
    expected = float(np.linalg.norm(vals))
    assert s.total_stress == pytest.approx(expected, rel=1e-9)


# ---------- sign-fix convention -----------------------------------------


def test_principal_axis_sign_fixed_to_largest_positive():
    """When the eigenvector has its largest |component| at a negative
    coordinate, the sign-fix should flip the vector."""
    vals = np.zeros((9, 2))
    # Force the second moral dimension (rights_respect) to dominate
    # with mixed signs that yield a negative-leading eigenvector.
    vals[1, :] = [-0.9, -0.8]
    t = _rank2_tensor_with_values(vals, ["a", "b"])
    s = compute_spectral_summary(t)
    # rights_respect index = 1; should be the largest |component|
    # AND should be positive after sign-fix.
    largest = int(np.argmax(np.abs(s.principal_axis)))
    assert largest == 1
    assert s.principal_axis[1] > 0


# ---------- rank 3+ per-axis spectrum -----------------------------------


def test_rank3_tensor_has_three_axis_spectra():
    """A rank-3 tensor (k, n, tau) should produce three AxisSpectrum
    entries, one for each axis."""
    vals = np.random.default_rng(7).normal(size=(9, 3, 4)) * 0.2
    t = MoralTensorV3(
        rank=3,
        shape=(9, 3, 4),
        axis_names=("k", "n", "tau"),
        axis_labels={
            "k": list(MORAL_DIMENSIONS_V3),
            "n": ["alice", "bob", "carol"],
            "tau": ["0", "1", "2", "3"],
        },
        values=vals.tolist(),
    )
    s = compute_spectral_summary(t)
    assert len(s.per_axis) == 3
    names = [a.axis_name for a in s.per_axis]
    assert names == ["k", "n", "tau"]
    # Each axis's spectrum has length matching that axis dimension.
    assert len(s.per_axis[0].eigenvalues) == 9
    assert len(s.per_axis[1].eigenvalues) == 3
    assert len(s.per_axis[2].eigenvalues) == 4


def test_rank6_tensor_has_six_axis_spectra():
    vals = np.random.default_rng(11).normal(size=(9, 2, 2, 2, 2, 2)) * 0.1
    t = MoralTensorV3(
        rank=6,
        shape=(9, 2, 2, 2, 2, 2),
        axis_names=("k", "n", "tau", "a", "c", "s"),
        axis_labels={
            "k": list(MORAL_DIMENSIONS_V3),
            "n": ["p0", "p1"],
            "tau": ["t0", "t1"],
            "a": ["a0", "a1"],
            "c": ["c0", "c1"],
            "s": ["s0", "s1"],
        },
        values=vals.tolist(),
    )
    s = compute_spectral_summary(t)
    assert len(s.per_axis) == 6
    assert [a.axis_name for a in s.per_axis] == ["k", "n", "tau", "a", "c", "s"]


def test_principal_stress_invariant_under_tau_replication():
    """A rank-3 tensor with τ=1 should have the same mode-k spectrum
    as the equivalent rank-2 slice — adding a singleton time axis
    must not change the principal magnitude."""
    vals2 = np.random.default_rng(3).normal(size=(9, 4)) * 0.3
    t2 = _rank2_tensor_with_values(vals2, ["a", "b", "c", "d"])
    s2 = compute_spectral_summary(t2)

    vals3 = vals2.reshape(9, 4, 1)
    t3 = MoralTensorV3(
        rank=3,
        shape=(9, 4, 1),
        axis_names=("k", "n", "tau"),
        axis_labels={
            "k": list(MORAL_DIMENSIONS_V3),
            "n": ["a", "b", "c", "d"],
            "tau": ["0"],
        },
        values=vals3.tolist(),
    )
    s3 = compute_spectral_summary(t3)
    assert s3.principal_stress == pytest.approx(s2.principal_stress, rel=1e-9)
    assert s3.total_stress == pytest.approx(s2.total_stress, rel=1e-9)


# ---------- attachment + roundtrip --------------------------------------


def test_attach_spectral_summary_writes_metadata():
    t = _rank2_tensor_with_values(np.ones((9, 2)), ["a", "b"])
    attach_spectral_summary(t)
    assert "spectral" in t.metadata
    assert "principal_stress" in t.metadata["spectral"]


def test_attach_is_idempotent():
    t = _rank2_tensor_with_values(np.ones((9, 2)), ["a", "b"])
    attach_spectral_summary(t)
    first = json.dumps(t.metadata["spectral"], sort_keys=True)
    attach_spectral_summary(t)
    second = json.dumps(t.metadata["spectral"], sort_keys=True)
    assert first == second


def test_spectral_block_survives_json_roundtrip():
    vals = np.random.default_rng(2).normal(size=(9, 3)) * 0.5
    t = _rank2_tensor_with_values(vals, ["a", "b", "c"])
    attach_spectral_summary(t)
    payload = t.model_dump_json()
    reloaded = MoralTensorV3.model_validate_json(payload)
    assert "spectral" in reloaded.metadata
    assert reloaded.metadata["spectral"]["principal_stress"] == pytest.approx(
        t.metadata["spectral"]["principal_stress"], rel=1e-9
    )


# ---------- end-to-end via the orchestrator ----------------------------


def test_compile_attaches_spectral_to_moral_tensor_v3():
    """The whole nazi_attic pipeline should leave a spectral block on
    ir.moral_tensor_v3.metadata."""
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier

    examples = Path(__file__).parent.parent / "examples"
    ir = compile_document(
        examples / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="mock",
            canonicalizer=None,
            tensor_rank=2,
        ),
    )
    assert ir.moral_tensor_v3 is not None
    spectral = ir.moral_tensor_v3.metadata.get("spectral")
    assert spectral is not None
    assert "principal_stress" in spectral
    # The nazi_attic per-stakeholder harm shows clear differentiation,
    # so principal_stress should be strictly positive.
    assert spectral["principal_stress"] > 0
    # And concentration is the fraction of total magnitude on the
    # leading axis — must be in [0, 1].
    assert 0.0 <= spectral["principal_concentration"] <= 1.0


# ---------- helper ------------------------------------------------------


def test_principal_dimension_label_returns_dim_name():
    """For a tensor where physical_harm dominates, the helper should
    return 'physical_harm'."""
    vals = np.zeros((9, 2))
    vals[0, :] = 0.9
    t = _rank2_tensor_with_values(vals, ["a", "b"])
    attach_spectral_summary(t)
    assert principal_dimension_label(t) == "physical_harm"
