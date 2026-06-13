"""Tests for VirtueProjection + CareEthicsProjection (release-planning-06 follow-up)."""

from __future__ import annotations

from pathlib import Path

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.ir.schemas import Document
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.projections import (
    AuthorityLegitimacy,
    CareEthicsProjection,
    ConsentState,
    Maxim,
    MoralSubstrate,
    VirtueProjection,
)
from erisml_compiler.projections.base import polarity_for_verdict
from erisml_compiler.tiers import CompilerTier


REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------- polarity map


def test_polarity_for_verdict_normalises_across_frameworks() -> None:
    # Each framework's "OK" maps to permit
    assert polarity_for_verdict("permitted") == "permit"
    assert polarity_for_verdict("permissible") == "permit"
    assert polarity_for_verdict("virtuous") == "permit"
    assert polarity_for_verdict("caring") == "permit"

    # Each framework's "not OK" maps to forbid
    assert polarity_for_verdict("forbidden") == "forbid"
    assert polarity_for_verdict("vicious") == "forbid"
    assert polarity_for_verdict("uncaring") == "forbid"

    # Escalations
    assert polarity_for_verdict("tragic_conflict_escalate") == "escalate"
    assert polarity_for_verdict("requires_practical_wisdom") == "escalate"
    assert polarity_for_verdict("requires_caring_attention") == "escalate"

    # Unknown verdicts default to neutral
    assert polarity_for_verdict("inscrutable_widget") == "neutral"


# ----------------------------------------------------- VirtueProjection


def _empty_substrate(maxim: Maxim | None = None) -> MoralSubstrate:
    return MoralSubstrate(
        document=Document(doc_id="t", title="t", raw_text="t"),
        maxim=maxim,
    )


def test_virtue_neutral_substrate_yields_virtuous() -> None:
    sub = _empty_substrate(maxim=Maxim(description="protect", action_kind="protect"))
    res = VirtueProjection().project(sub)
    assert res.framework == "virtue_aristotelian"
    assert res.verdict == "virtuous"


def test_virtue_deceit_flags_character_concern() -> None:
    sub = _empty_substrate(maxim=Maxim(description="deceive", action_kind="deceive"))
    res = VirtueProjection().project(sub)
    by_name = {f.name: f for f in res.findings}
    assert by_name["character_consistency"].passed is False
    assert by_name["character_consistency"].detail.get("virtue") == "honesty"


def test_virtue_runs_on_compiled_nazi_attic_and_disagrees() -> None:
    ir = compile_document(
        REPO_ROOT / "examples" / "nazi_attic.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="rule",
                       canonicalizer=RegistryCanonicalizer(), tensor_rank=2),
    )
    assert "virtue_aristotelian" in ir.projections
    v = ir.projections["virtue_aristotelian"]
    # nazi_attic should not be cleanly virtuous — there's deceit + power asymmetry
    assert v["verdict"] in {"requires_practical_wisdom", "vicious"}


# ----------------------------------------------------- CareEthicsProjection


def test_care_caring_when_relations_present_and_no_imposition() -> None:
    from erisml_compiler.ir.schemas import Relation

    rel = Relation(id="r1", type="kin_of", source="self", target="alice")
    sub = MoralSubstrate(
        document=Document(doc_id="t", title="t", raw_text="t"),
        maxim=Maxim(description="visit", action_kind="protect"),
        relations=[rel],
    )
    res = CareEthicsProjection().project(sub)
    assert res.framework == "care_ethics_relational"
    assert res.verdict == "caring"


def test_care_flags_missing_relational_attentiveness() -> None:
    sub = _empty_substrate(maxim=Maxim(description="x", action_kind="act_under_norm"))
    res = CareEthicsProjection().project(sub)
    # No relations -> relational_attentiveness fails -> requires_caring_attention
    by_name = {f.name: f for f in res.findings}
    assert by_name["relational_attentiveness"].passed is False
    assert res.verdict == "requires_caring_attention"


def test_care_runs_on_compiled_examples() -> None:
    for fname in ("nazi_attic", "medical_confidentiality", "whistleblower"):
        ir = compile_document(
            REPO_ROOT / "examples" / f"{fname}.txt",
            CompileOptions(tier=CompilerTier.RULES, extractor="rule",
                           canonicalizer=RegistryCanonicalizer(), tensor_rank=2),
        )
        assert "care_ethics_relational" in ir.projections


# ----------------------------------------------------- cross-projection


def test_cross_projection_disagreement_uses_polarity_not_verdict_string() -> None:
    ir = compile_document(
        REPO_ROOT / "examples" / "medical_confidentiality.txt",
        CompileOptions(tier=CompilerTier.RULES, extractor="rule",
                       canonicalizer=RegistryCanonicalizer(), tensor_rank=2),
    )
    # On medical: consequentialist + deontic + virtue all polarity=permit,
    # care ethics polarity=escalate. Disagreement should be FLAGGED (escalate
    # vs permit), but the verdict strings differ across all 4 frameworks
    # ("permitted" vs "permissible" vs "virtuous" vs "requires_caring_attention").
    # Old (string-based) comparison would always flag; new (polarity-based)
    # only flags real polarity disagreement.
    assert ir.cross_projection_disagreement is not None
    polarities = set(ir.cross_projection_disagreement["polarities"].values())
    assert "permit" in polarities
    assert "escalate" in polarities


def test_no_disagreement_when_all_polarities_agree() -> None:
    """Synthetic: if every projection returned `permit`, no disagreement."""
    from erisml_compiler.projections.base import ProjectionResult

    r1 = ProjectionResult(framework="a", verdict="permitted")
    r2 = ProjectionResult(framework="b", verdict="permissible")
    r3 = ProjectionResult(framework="c", verdict="virtuous")
    assert r1.polarity == r2.polarity == r3.polarity == "permit"
