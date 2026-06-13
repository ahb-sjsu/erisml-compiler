"""Tests for the two-layer IR: substrate + framework projections."""

from __future__ import annotations

from pathlib import Path

import pytest

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.projections import (
    AuthorityLegitimacy,
    ConsentState,
    ConsequentialistProjection,
    DeonticProjection,
    GateFinding,
    Maxim,
    MoralSubstrate,
    Projection,
    ProjectionResult,
    substrate_from_ir,
)
from erisml_compiler.tiers import CompilerTier

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


# ----------------------------------------------------------- schema basics


def test_projection_abc_requires_project_method() -> None:
    with pytest.raises(TypeError):
        Projection()  # type: ignore[abstract]


def test_gate_finding_severity_constrained() -> None:
    f = GateFinding(name="t", passed=False, reason="r", severity="grave")
    assert f.severity == "grave"


def test_maxim_treats_persons_as_dict() -> None:
    m = Maxim(
        description="deceive to save life",
        action_kind="deceive",
        treats_persons_as={"alice": "mere_means", "bob": "end"},
    )
    assert m.treats_persons_as["alice"] == "mere_means"


# ----------------------------------------------------------- substrate derivation


def _compile_example(name: str):
    return compile_document(
        EXAMPLES_DIR / f"{name}.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
        ),
    )


def test_substrate_from_ir_derives_maxim_on_nazi_attic() -> None:
    ir = _compile_example("nazi_attic")
    sub = substrate_from_ir(ir)
    assert sub.maxim is not None
    # The rule extractor surfaces a deception fact, so the maxim's
    # action_kind should reflect deception.
    assert sub.maxim.action_kind == "deceive"


def test_substrate_from_ir_emits_nonconsenting_states() -> None:
    ir = _compile_example("nazi_attic")
    sub = substrate_from_ir(ir)
    # The village is a non-consenting third party in the rule-extracted IR.
    assert any(not c.given for c in sub.consent_states)


def test_substrate_carries_descriptive_fields() -> None:
    ir = _compile_example("medical_confidentiality")
    sub = substrate_from_ir(ir)
    assert sub.stakeholders
    assert sub.ethical_facts
    assert sub.document is ir.document


# ----------------------------------------------------------- deontic projection


def test_deontic_projection_forbidden_on_deceive_maxim() -> None:
    ir = _compile_example("nazi_attic")
    sub = substrate_from_ir(ir)
    res = DeonticProjection().project(sub)
    assert res.framework == "deontic_kantian"
    assert res.verdict == "forbidden"
    by_name = {f.name: f for f in res.findings}
    # Deceive is not universalizable.
    assert by_name["universalizability"].passed is False
    # Village/self are under duress / nonconsenting.
    assert by_name["valid_consent"].passed is False


def test_deontic_projection_with_clean_substrate_permits() -> None:
    """A substrate with no failure-inducing content yields `permissible`."""
    from erisml_compiler.ir.schemas import Document

    sub = MoralSubstrate(
        document=Document(doc_id="t", title="t", raw_text="t"),
        maxim=Maxim(description="help neighbour", action_kind="protect"),
        consent_states=[ConsentState(stakeholder_id="alice", given=True)],
        authority_legitimacies=[AuthorityLegitimacy(authority_id="court", legitimate=True)],
    )
    res = DeonticProjection().project(sub)
    assert res.verdict == "permissible"
    assert all(f.passed for f in res.findings)


def test_deontic_projection_requires_review_on_only_moderate_failures() -> None:
    """Only an illegitimate authority (moderate) and nothing else — verdict
    should escalate to requires_review, not forbidden."""
    from erisml_compiler.ir.schemas import Document

    sub = MoralSubstrate(
        document=Document(doc_id="t", title="t", raw_text="t"),
        maxim=Maxim(description="comply with order", action_kind="comply"),
        consent_states=[ConsentState(stakeholder_id="alice", given=True)],
        authority_legitimacies=[AuthorityLegitimacy(authority_id="usurper", legitimate=False)],
    )
    res = DeonticProjection().project(sub)
    assert res.verdict == "requires_review"


# ----------------------------------------------------------- consequentialist


def test_consequentialist_projection_back_fills_legacy_fields() -> None:
    ir = _compile_example("nazi_attic")
    # The orchestrator runs the consequentialist projection by default.
    assert ir.moral_tensor_v3 is not None
    assert ir.deme_verdict is not None
    assert ir.moral_vectors
    assert "consequentialist_distributive" in ir.projections


# ----------------------------------------------------------- cross-projection


def test_cross_projection_disagreement_surfaced_when_verdicts_differ() -> None:
    ir = _compile_example("nazi_attic")
    assert ir.cross_projection_disagreement is not None
    assert "consequentialist_distributive" in ir.cross_projection_disagreement["verdicts"]
    assert "deontic_kantian" in ir.cross_projection_disagreement["verdicts"]
    # Crucial property: the compiler does NOT pick a winner.
    assert "metaethical" in ir.cross_projection_disagreement["note"].lower()


def test_disabling_deontic_projection_yields_no_disagreement_field() -> None:
    ir = compile_document(
        EXAMPLES_DIR / "nazi_attic.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
            projections=("consequentialist_distributive",),
        ),
    )
    assert ir.cross_projection_disagreement is None
    assert "deontic_kantian" not in ir.projections


# ----------------------------------------------------------- ProjectionResult shape


def test_projection_result_serialises_findings_to_dict() -> None:
    ir = _compile_example("nazi_attic")
    dres = ir.projections["deontic_kantian"]
    assert isinstance(dres, dict)
    assert dres["framework"] == "deontic_kantian"
    assert dres["verdict"] == "forbidden"
    assert isinstance(dres["findings"], list)
    assert all("passed" in f for f in dres["findings"])
