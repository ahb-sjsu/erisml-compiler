"""Phase 1 — MoralTensorV3 schema + V2 migration helpers."""
from __future__ import annotations

import pytest

from erisml_compiler.ir.schemas import DimensionScore, MoralTensor, MoralVector
from erisml_compiler.ir.v3 import (
    DIMENSION_MATRIX_3X3,
    DimensionMetadata,
    MORAL_DIMENSIONS_V3,
    MoralTensorV3,
    V2_TO_V3_DIMENSION_MAP,
    migrate_v2_tensor_to_v3,
    migrate_v2_vector_to_v3,
)
from erisml_compiler.ir.v3.tensor import K_DIM


# ---------- dimension layout ----------


def test_canonical_dimensions_have_nine_entries():
    assert len(MORAL_DIMENSIONS_V3) == 9
    assert len(set(MORAL_DIMENSIONS_V3)) == 9


def test_dimension_matrix_covers_all_nine():
    assert set(DIMENSION_MATRIX_3X3.keys()) == set(MORAL_DIMENSIONS_V3)
    rows = {pos[0] for pos in DIMENSION_MATRIX_3X3.values()}
    cols = {pos[1] for pos in DIMENSION_MATRIX_3X3.values()}
    assert rows == {"Individual", "Relational", "Collective"}
    assert cols == {"What Matters", "Who Decides", "What We Know"}


def test_v2_to_v3_map_keeps_seven_targets_in_v3():
    for v2, targets in V2_TO_V3_DIMENSION_MAP.items():
        for t in targets:
            assert t in MORAL_DIMENSIONS_V3, (v2, t)


# ---------- MoralTensorV3 construction + validation ----------


def test_rank1_tensor_constructs():
    t = MoralTensorV3.zeros(shape=(K_DIM,))
    assert t.rank == 1
    assert t.shape == (K_DIM,)
    assert t.axis_names == ("k",)
    assert len(t.values) == K_DIM


def test_rank2_tensor_with_stakeholders():
    t = MoralTensorV3.zeros(
        shape=(K_DIM, 3),
        axis_labels={"k": list(MORAL_DIMENSIONS_V3), "n": ["alice", "bob", "carol"]},
    )
    assert t.rank == 2
    assert t.axis_names == ("k", "n")
    assert t.axis_labels["n"] == ["alice", "bob", "carol"]


def test_rank_six_full_axes():
    # (k=9, n=2, τ=2, a=2, c=2, s=2) — smallest legitimate rank-6 shape.
    t = MoralTensorV3.zeros(shape=(K_DIM, 2, 2, 2, 2, 2))
    assert t.rank == 6
    assert t.axis_names == ("k", "n", "tau", "a", "c", "s")


def test_first_axis_must_be_nine():
    with pytest.raises(ValueError, match="moral dimensions"):
        MoralTensorV3.zeros(shape=(10,))


def test_rank_must_match_shape_length():
    with pytest.raises(ValueError):
        MoralTensorV3(rank=2, shape=(K_DIM,), axis_names=("k",))


def test_axis_names_must_start_with_k():
    with pytest.raises(ValueError, match="axis_names"):
        MoralTensorV3(rank=1, shape=(K_DIM,), axis_names=("n",))


def test_rank_over_six_rejected():
    with pytest.raises(ValueError):
        MoralTensorV3.zeros(shape=(K_DIM, 2, 2, 2, 2, 2, 2))


def test_values_shape_validation():
    bad = [[0.0] * (K_DIM - 1)]
    with pytest.raises(ValueError):
        MoralTensorV3(
            rank=2, shape=(K_DIM, 1), axis_names=("k", "n"), values=bad
        )


def test_set_and_get_cell_roundtrip():
    t = MoralTensorV3.zeros(shape=(K_DIM, 2))
    t.set_cell(3, 1, 0.42)
    assert t.get_cell(3, 1) == pytest.approx(0.42)


def test_metadata_keyed_by_indices():
    t = MoralTensorV3.zeros(shape=(K_DIM, 2))
    md = DimensionMetadata(
        confidence=0.8, uncertainty=0.2, direction="positive",
        source_spans=["seg_001:0-10"], explanation="test",
    )
    t.set_metadata((2, 1), md)
    got = t.get_metadata(2, 1)
    assert got is not None
    assert got.confidence == 0.8
    assert got.source_spans == ["seg_001:0-10"]


def test_veto_locations_are_arity_checked():
    with pytest.raises(ValueError, match="veto_location"):
        MoralTensorV3(
            rank=2, shape=(K_DIM, 2), axis_names=("k", "n"),
            values=[[0.0, 0.0]] * K_DIM,
            veto_locations=[(0, 0, 0)],   # too long for rank=2
        )


def test_global_veto_allowed():
    t = MoralTensorV3(
        rank=2, shape=(K_DIM, 2), axis_names=("k", "n"),
        values=[[0.0, 0.0]] * K_DIM,
        veto_flags=["hard_constraint"],
        veto_locations=[()],   # global
    )
    assert t.veto_flags == ["hard_constraint"]


# ---------- JSON serialisation roundtrip ----------


def test_json_roundtrip():
    t1 = MoralTensorV3.zeros(shape=(K_DIM, 2))
    t1.set_cell(0, 0, 0.5)
    t1.set_metadata((0, 0), DimensionMetadata(confidence=0.9, direction="positive"))
    payload = t1.model_dump()
    t2 = MoralTensorV3.model_validate(payload)
    assert t2.shape == t1.shape
    assert t2.get_cell(0, 0) == pytest.approx(0.5)
    md = t2.get_metadata(0, 0)
    assert md is not None and md.confidence == 0.9


# ---------- V2 → V3 vector migration ----------


def _full_v2_vector(value: float = 0.5) -> MoralVector:
    score = DimensionScore(value=value, confidence=0.8, uncertainty=0.1, direction="positive")
    return MoralVector(
        physical_harm=score, rights_respect=score, fairness_equity=score,
        autonomy_consent=score, legitimacy_trust=score, epistemic_quality=score,
        care_protection=score, vow_fidelity=score,
        third_party_externality=score, repair_residue=score,
    )


def test_migrate_v2_vector_produces_rank1_tensor():
    t = migrate_v2_vector_to_v3(_full_v2_vector(0.5))
    assert t.rank == 1
    assert t.shape == (K_DIM,)
    assert t.axis_names == ("k",)


def test_migrate_v2_vector_renames_dims():
    t = migrate_v2_vector_to_v3(_full_v2_vector(0.5))
    # autonomy_consent in V2 -> autonomy_respect in V3 at k=3.
    auto_k = MORAL_DIMENSIONS_V3.index("autonomy_respect")
    # Should be non-zero (we set every V2 dim to 0.5).
    assert t.get_cell(auto_k) != 0.0


def test_migrate_v2_vector_carries_repair_residue_in_metadata():
    t = migrate_v2_vector_to_v3(_full_v2_vector(0.7))
    assert t.metadata["repair_residue"] == pytest.approx(0.7)
    assert t.metadata["repair_residue_metadata"]["confidence"] == 0.8


def test_migrate_v2_vector_synthesises_privacy_protection():
    """V2 has no privacy dim; V3 needs one. Migration should mark it
    as synthesised in metadata."""
    t = migrate_v2_vector_to_v3(_full_v2_vector(0.5))
    assert "privacy_protection" in t.metadata["migration"]["synthesised_dims"]
    privacy_k = MORAL_DIMENSIONS_V3.index("privacy_protection")
    assert t.get_cell(privacy_k) == 0.0


def test_migrate_v2_vector_splits_vow_fidelity():
    # vow_fidelity is split between legitimacy_trust and virtue_care.
    # Construct a V2 vector with vow_fidelity at the maximum and other
    # dims at zero; check the two targets ended up with non-zero values.
    zero = DimensionScore(value=0.0)
    vow = DimensionScore(value=1.0, confidence=1.0, direction="positive")
    v2 = MoralVector(
        physical_harm=zero, rights_respect=zero, fairness_equity=zero,
        autonomy_consent=zero, legitimacy_trust=zero, epistemic_quality=zero,
        care_protection=zero, vow_fidelity=vow,
        third_party_externality=zero, repair_residue=zero,
    )
    t = migrate_v2_vector_to_v3(v2)
    leg_k = MORAL_DIMENSIONS_V3.index("legitimacy_trust")
    care_k = MORAL_DIMENSIONS_V3.index("virtue_care")
    assert t.get_cell(leg_k) > 0
    assert t.get_cell(care_k) > 0
    # Sum approximately preserved (each got half).
    assert t.get_cell(leg_k) + t.get_cell(care_k) == pytest.approx(1.0)


# ---------- V2 → V3 tensor migration ----------


def test_migrate_v2_tensor_produces_rank2():
    score = DimensionScore(value=0.5, confidence=0.7, direction="positive")
    v2_alice = MoralTensor(
        stakeholder_id="alice", time_index=0,
        by_dimension={
            "physical_harm": score, "rights_respect": score,
            "fairness_equity": score, "autonomy_consent": score,
            "legitimacy_trust": score, "epistemic_quality": score,
            "care_protection": score, "vow_fidelity": score,
            "third_party_externality": score, "repair_residue": score,
        },
    )
    t = migrate_v2_tensor_to_v3([v2_alice], stakeholder_ids=["alice"])
    assert t.rank == 2
    assert t.shape == (K_DIM, 1)
    assert t.axis_labels["n"] == ["alice"]
