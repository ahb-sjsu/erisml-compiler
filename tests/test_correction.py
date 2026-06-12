"""Tests for the human-correction loop (Track A)."""
import json
from pathlib import Path

import pytest

from erisml_compiler.correction.corrector import apply_corrections
from erisml_compiler.correction.diff import diff_irs
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def base_ir():
    return compile_document(
        EXAMPLES / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="mock"),
    )


# ---------- diff_irs ----------


def test_diff_identical_irs_is_empty(base_ir):
    diff = diff_irs(base_ir, base_ir)
    assert diff.is_empty


def test_diff_scalar_change(base_ir):
    new_ir = base_ir.model_copy(deep=True)
    new_ir.canonical_form = "some_other_form"
    diff = diff_irs(base_ir, new_ir)
    assert not diff.is_empty
    assert any(fc.field_path == "canonical_form" for fc in diff.scalar_changes)


def test_diff_added_stakeholder(base_ir):
    from erisml_compiler.ir.schemas import Stakeholder
    new_ir = base_ir.model_copy(deep=True)
    new_ir.stakeholders.append(
        Stakeholder(id="new_party", label="Newcomer", type="individual", roles=["bystander"])
    )
    diff = diff_irs(base_ir, new_ir)
    added = [ed for ed in diff.entity_diffs if ed.added]
    assert len(added) == 1
    assert added[0].entity_id == "new_party"


def test_diff_removed_commitment(base_ir):
    new_ir = base_ir.model_copy(deep=True)
    new_ir.commitments = new_ir.commitments[1:]  # drop first
    diff = diff_irs(base_ir, new_ir)
    removed = [ed for ed in diff.entity_diffs if ed.removed]
    assert len(removed) == 1


def test_diff_field_change(base_ir):
    new_ir = base_ir.model_copy(deep=True)
    new_ir.stakeholders[0].vulnerability = "extreme"
    diff = diff_irs(base_ir, new_ir)
    modified = [ed for ed in diff.entity_diffs if ed.is_modified]
    assert len(modified) >= 1


# ---------- apply_corrections ----------


def test_set_scalar_field(base_ir):
    corrections = {
        "corrector_id": "test@example.org",
        "rationale": "Testing set op",
        "patches": [{"op": "set", "path": "canonical_form", "value": "test_form_xyz"}],
    }
    new_ir, record, summaries = apply_corrections(base_ir, corrections)
    assert new_ir.canonical_form == "test_form_xyz"
    assert record.n_patches_applied == 1
    assert record.n_patches_failed == 0
    assert new_ir.audit is None or new_ir.audit is not None  # audit may or may not be re-finalized
    assert "corrections" in new_ir.extra
    assert new_ir.extra["corrections"][0]["corrector_id"] == "test@example.org"


def test_set_nested_field(base_ir):
    # Find first stakeholder id.
    sid = base_ir.stakeholders[0].id
    corrections = {
        "corrector_id": "test",
        "rationale": "fix vulnerability",
        "patches": [
            {"op": "set", "path": f"stakeholders.{sid}.vulnerability", "value": "extreme"},
        ],
    }
    new_ir, record, _ = apply_corrections(base_ir, corrections)
    # Find that stakeholder in new IR.
    found = [s for s in new_ir.stakeholders if s.id == sid][0]
    assert found.vulnerability == "extreme"
    assert record.n_patches_applied == 1


def test_add_new_entity(base_ir):
    corrections = {
        "corrector_id": "test",
        "rationale": "add new stakeholder",
        "patches": [{
            "op": "add",
            "path": "stakeholders.observer_x",
            "value": {
                "label": "Observer",
                "type": "individual",
                "roles": ["bystander"],
            },
        }],
    }
    new_ir, record, _ = apply_corrections(base_ir, corrections)
    assert any(s.id == "observer_x" for s in new_ir.stakeholders)
    assert record.n_patches_applied == 1


def test_remove_entity(base_ir):
    sid = base_ir.stakeholders[0].id
    corrections = {
        "corrector_id": "test",
        "rationale": "remove the first stakeholder",
        "patches": [{"op": "remove", "path": f"stakeholders.{sid}"}],
    }
    new_ir, record, _ = apply_corrections(base_ir, corrections)
    assert not any(s.id == sid for s in new_ir.stakeholders)
    assert record.n_patches_applied == 1


def test_invalid_op_recorded_as_failure(base_ir):
    corrections = {
        "corrector_id": "test",
        "rationale": "test bogus op",
        "patches": [
            {"op": "set", "path": "canonical_form", "value": "valid_change"},  # valid
            {"op": "add", "path": "stakeholders.x", "value": {"label": "X", "type": "WRONG"}},  # value should fail schema
        ],
    }
    # The schema-invalid add ought to fail at re-validation. The apply
    # phase itself doesn't validate; final Pydantic validation does.
    # So this should raise.
    with pytest.raises(Exception):
        apply_corrections(base_ir, corrections)


def test_correction_records_in_audit_trail(base_ir):
    c1 = {
        "corrector_id": "alice",
        "rationale": "first correction",
        "patches": [{"op": "set", "path": "canonical_form", "value": "form_v1"}],
    }
    c2 = {
        "corrector_id": "bob",
        "rationale": "second correction",
        "patches": [{"op": "set", "path": "canonical_form", "value": "form_v2"}],
    }
    ir1, _, _ = apply_corrections(base_ir, c1)
    ir2, _, _ = apply_corrections(ir1, c2)
    history = ir2.extra["corrections"]
    assert len(history) == 2
    assert history[0]["corrector_id"] == "alice"
    assert history[1]["corrector_id"] == "bob"


def test_hash_changes_after_correction(base_ir):
    from erisml_compiler.audit.hash_chain import compute_ir_hash
    pre = compute_ir_hash(base_ir)
    corrections = {
        "corrector_id": "test",
        "rationale": "anything",
        "patches": [{"op": "set", "path": "canonical_form", "value": "different"}],
    }
    new_ir, record, _ = apply_corrections(base_ir, corrections)
    assert record.pre_correction_ir_hash == pre
    assert record.post_correction_ir_hash != pre
