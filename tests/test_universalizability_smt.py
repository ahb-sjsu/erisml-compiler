"""Tests for the Z3-based SMT universalizability solver."""

from __future__ import annotations

import pytest

from erisml_compiler.delta.universalizability_smt import (
    is_smt_available,
    test_universalizability_smt as run_smt,
)

pytestmark = pytest.mark.skipif(not is_smt_available(), reason="z3-solver not installed")


# ----------------------------------------------------- CIC


def test_smt_deceive_is_cic() -> None:
    r = run_smt("deceive")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_conception"
    assert r.base.passes is False
    # UNSAT means no satisfying model — model_facts should be empty.
    assert not r.model_facts


def test_smt_break_commitment_is_cic() -> None:
    r = run_smt("break_commitment")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_conception"


def test_smt_cheat_is_cic() -> None:
    r = run_smt("cheat")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_conception"


# ----------------------------------------------------- CIW


def test_smt_refuse_is_ciw_classic_kant() -> None:
    """Kant's Groundwork example: universal indifference is internally
    coherent but destroys help_available_when_needed, an agent end."""
    r = run_smt("refuse")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_will"
    assert r.base.passes is False


def test_smt_coerce_is_ciw() -> None:
    r = run_smt("coerce")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_will"


def test_smt_inflict_harm_is_ciw() -> None:
    r = run_smt("inflict_harm")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_will"


def test_smt_impose_externality_is_ciw() -> None:
    r = run_smt("impose_externality")
    assert r.used_z3 is True
    assert r.base.contradiction_type == "contradiction_in_will"


# ----------------------------------------------------- no contradiction


def test_smt_protect_passes() -> None:
    r = run_smt("protect")
    assert r.used_z3 is True
    assert r.base.passes is True
    assert r.base.contradiction_type == "no_contradiction"
    # SAT → there should be a satisfying assignment recorded.
    assert r.model_facts
    # Agent ends should be SAT-True in the model.
    assert r.model_facts.get("bodily_integrity_respected") is True
    assert r.model_facts.get("help_available_when_needed") is True


def test_smt_help_passes() -> None:
    r = run_smt("help")
    assert r.used_z3 is True
    assert r.base.passes is True


def test_smt_make_or_keep_commitment_passes() -> None:
    r = run_smt("make_or_keep_commitment")
    assert r.used_z3 is True
    assert r.base.passes is True


def test_smt_disclose_passes_with_contested_reading() -> None:
    r = run_smt("disclose")
    assert r.used_z3 is True
    assert r.base.passes is True
    assert r.base.contested_reading is not None


# ----------------------------------------------------- audit


def test_smt_satisfying_model_includes_relevant_facts() -> None:
    """SAT cases should expose all 9 institutional facts in the model
    so reviewers can see the world-state the solver accepted."""
    r = run_smt("protect")
    assert r.used_z3 is True
    expected_facts = {
        "truth_telling_default",
        "promises_create_trust",
        "property_rules_followed",
        "bodily_integrity_respected",
        "autonomy_respected",
        "help_available_when_needed",
    }
    assert expected_facts.issubset(set(r.model_facts.keys()))


# ----------------------------------------------------- fallback


def test_smt_unknown_action_kind_falls_back_to_kb() -> None:
    r = run_smt("unknown_action_kind_xyz")
    assert r.used_z3 is False
    assert r.fallback_reason == "action_kind not in Z3 model"


# ----------------------------------------------------- integration


def test_deontic_projection_records_solver_z3_in_detail() -> None:
    """When SMT is available, the gate's detail block records
    `solver_used: z3`."""
    from pathlib import Path

    from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
    from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
    from erisml_compiler.tiers import CompilerTier

    REPO_ROOT = Path(__file__).resolve().parent.parent
    ir = compile_document(
        REPO_ROOT / "examples" / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
        ),
    )
    univ = next(
        f
        for f in ir.projections["deontic_kantian"]["findings"]
        if f["name"] == "universalizability"
    )
    assert univ["detail"].get("solver_used") == "z3"
