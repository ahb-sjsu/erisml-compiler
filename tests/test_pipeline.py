"""End-to-end pipeline tests against the three example fixtures."""

from pathlib import Path

import pytest

from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def options_mock():
    return CompileOptions(tier=CompilerTier.RULES, extractor="mock")


@pytest.fixture
def options_rule():
    return CompileOptions(tier=CompilerTier.RULES, extractor="rule")


# ----------------------------------------------------------------- nazi attic


def test_nazi_attic_mock(options_mock):
    ir = compile_document(EXAMPLES_DIR / "nazi_attic.txt", options_mock)
    assert ir.deme_verdict is not None
    assert ir.deme_verdict.verdict == "tragic_conflict_escalate"
    assert ir.deme_verdict.escalation_required is True
    assert ir.canonical_form == "coercive_murderous_interrogation_with_collective_reprisal"
    # Spec §31 acceptance criteria:
    assert len(ir.stakeholders) >= 1  # #2
    assert any(c.type == "vow" for c in ir.commitments)  # #3
    assert any(f.kind == "coercion" for f in ir.ethical_facts)  # #4
    assert any(f.kind == "externality" for f in ir.ethical_facts)  # #5
    assert ir.audit is not None  # #9
    assert ir.timeline  # #7


def test_nazi_attic_rule(options_rule):
    """The rule extractor should still produce a coherent (if shallower) IR
    and the same general verdict on nazi attic, because the text contains
    a vow + a threat + a collective harm target."""
    ir = compile_document(EXAMPLES_DIR / "nazi_attic.txt", options_rule)
    assert ir.deme_verdict is not None
    # rule extractor should at minimum detect coercion and externality
    assert any(f.kind == "coercion" for f in ir.ethical_facts)
    assert any(f.kind == "externality" for f in ir.ethical_facts)
    # Verdict on rule-extracted nazi attic should still reflect tragic conflict
    assert ir.deme_verdict.verdict in (
        "tragic_conflict_escalate",
        "prohibited",
        "requires_human_review",
    )


# ----------------------------------------------------------------- medical


def test_medical_confidentiality_mock(options_mock):
    ir = compile_document(EXAMPLES_DIR / "medical_confidentiality.txt", options_mock)
    assert ir.deme_verdict is not None
    assert ir.canonical_form == "professional_privilege_versus_duty_to_warn"
    assert any(c.type == "role_duty" for c in ir.commitments)
    assert any(f.kind == "role_duty" for f in ir.ethical_facts)
    assert any(f.kind == "externality" for f in ir.ethical_facts)


# ----------------------------------------------------------------- whistleblower


def test_whistleblower_mock(options_mock):
    ir = compile_document(EXAMPLES_DIR / "whistleblower.txt", options_mock)
    assert ir.deme_verdict is not None
    assert ir.canonical_form == "institutional_loyalty_versus_public_truth_telling"
    assert any(c.type == "role_duty" for c in ir.commitments)
    assert any(f.kind == "externality" for f in ir.ethical_facts)
    assert any(f.kind == "truth" for f in ir.ethical_facts)


# ----------------------------------------------------------------- audit determinism


def test_audit_hash_deterministic(options_mock):
    """Same input + same options produces the same IR hash."""
    ir1 = compile_document(EXAMPLES_DIR / "nazi_attic.txt", options_mock)
    ir2 = compile_document(EXAMPLES_DIR / "nazi_attic.txt", options_mock)
    assert ir1.audit.ir_hash == ir2.audit.ir_hash
    assert ir1.audit.source_text_hash == ir2.audit.source_text_hash
