"""Tests for the principled universalizability test
(release-planning-06 v1 — replaces the rule-list gate)."""

from __future__ import annotations

from pathlib import Path

from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer
from erisml_compiler.delta.universalizability import (
    DEFAULT_INSTITUTION_DEPENDENCIES,
    InstitutionDependency,
    test_universalizability as run_universalizability,
)
from erisml_compiler.pipeline.orchestrator import CompileOptions, compile_document
from erisml_compiler.tiers import CompilerTier


REPO_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------- knowledge base coverage


def test_kb_has_all_action_kinds_from_maxim_extractor() -> None:
    """The KB must cover every action_kind the maxim extractor can emit."""
    from erisml_compiler.annotation.maxim_extractor import _VERB_PATTERNS

    emitted_kinds = {kind for _, kind in _VERB_PATTERNS}
    kb_kinds = set(DEFAULT_INSTITUTION_DEPENDENCIES.keys())
    missing = emitted_kinds - kb_kinds
    assert not missing, f"Maxim extractor emits {missing} but the KB has no entries"


def test_kb_entry_well_formed_for_every_action_kind() -> None:
    """Every entry must have a justification and a contradiction_type."""
    for kind, dep in DEFAULT_INSTITUTION_DEPENDENCIES.items():
        assert dep.justification, f"{kind} has no justification"
        assert dep.contradiction_type in (
            "contradiction_in_conception",
            "contradiction_in_will",
            "no_contradiction",
            "undetermined",
        ), f"{kind} has invalid contradiction_type"
        if dep.contradiction_type in ("contradiction_in_conception", "contradiction_in_will"):
            assert dep.passes is False, f"{kind} marked as contradiction but passes=True"
        if dep.contradiction_type == "no_contradiction":
            assert dep.passes is True, f"{kind} marked as no_contradiction but passes=False"


# ----------------------------------------------------- specific Kantian readings


def test_deceive_fails_cic() -> None:
    dep = run_universalizability("deceive")
    assert dep.passes is False
    assert dep.contradiction_type == "contradiction_in_conception"
    assert "truth" in " ".join(dep.presupposes).lower()


def test_break_commitment_fails_cic() -> None:
    dep = run_universalizability("break_commitment")
    assert dep.passes is False
    assert dep.contradiction_type == "contradiction_in_conception"


def test_cheat_fails_cic() -> None:
    dep = run_universalizability("cheat")
    assert dep.passes is False
    assert dep.contradiction_type == "contradiction_in_conception"


def test_refuse_to_help_fails_ciw_classic_kant() -> None:
    """Kant's famous Groundwork example: universal indifference to
    others' suffering is coherent but cannot be willed without
    contradicting the agent's own ends."""
    dep = run_universalizability("refuse")
    assert dep.passes is False
    assert dep.contradiction_type == "contradiction_in_will"


def test_use_as_means_fails_ciw_humanity_formula() -> None:
    dep = run_universalizability("use_as_means")
    assert dep.passes is False
    assert "humanity" in dep.justification.lower() or "mere" in dep.justification.lower()


def test_coerce_fails_ciw() -> None:
    dep = run_universalizability("coerce")
    assert dep.passes is False
    assert dep.contradiction_type == "contradiction_in_will"


def test_protect_passes() -> None:
    dep = run_universalizability("protect")
    assert dep.passes is True
    assert dep.contradiction_type == "no_contradiction"


def test_help_passes() -> None:
    dep = run_universalizability("help")
    assert dep.passes is True


def test_make_or_keep_commitment_passes() -> None:
    dep = run_universalizability("make_or_keep_commitment")
    assert dep.passes is True


def test_disclose_default_reading_passes_but_flags_contested() -> None:
    """Disclose passes by default (whistleblower reading), with a
    flagged contested reading for confidentiality cases."""
    dep = run_universalizability("disclose")
    assert dep.passes is True
    assert dep.contested_reading is not None
    assert "confidentiality" in dep.contested_reading.lower()


def test_unknown_action_kind_returns_undetermined() -> None:
    dep = run_universalizability("invent_warp_drive")
    assert dep.contradiction_type == "undetermined"
    # Benefit-of-doubt pass — gate decides what to do with undetermined.
    assert dep.passes is True


def test_none_action_kind_returns_undetermined() -> None:
    dep = run_universalizability(None)
    assert dep.contradiction_type == "undetermined"


def test_custom_mapping_overrides_default() -> None:
    """Callers can pass a custom mapping (e.g. for confidentiality
    cases that should treat 'disclose' as CIC-failing)."""
    custom: dict[str, InstitutionDependency] = {
        "disclose": InstitutionDependency(
            action_kind="disclose",
            presupposes=("confidentiality",),
            contradiction_type="contradiction_in_conception",
            passes=False,
            justification="Universal disclosure destroys confidentiality.",
        ),
    }
    dep = run_universalizability("disclose", mapping=custom)
    assert dep.passes is False


# ----------------------------------------------------- gate integration


def _compile(name: str):
    return compile_document(
        REPO_ROOT / "examples" / f"{name}.txt",
        CompileOptions(
            tier=CompilerTier.RULES,
            extractor="rule",
            canonicalizer=RegistryCanonicalizer(),
            tensor_rank=2,
        ),
    )


def test_gate_records_justification_in_finding_detail() -> None:
    ir = _compile("nazi_attic")
    univ = next(
        f for f in ir.projections["deontic_kantian"]["findings"]
        if f["name"] == "universalizability"
    )
    detail = univ["detail"]
    assert detail["action_kind"] == "deceive"
    assert detail["contradiction_type"] == "contradiction_in_conception"
    assert detail["presupposes"]
    assert detail["justification"]


def test_gate_records_contested_reading_when_present() -> None:
    ir = _compile("whistleblower")
    univ = next(
        f for f in ir.projections["deontic_kantian"]["findings"]
        if f["name"] == "universalizability"
    )
    # whistleblower's action_kind is 'disclose' → has contested reading
    assert "contested_reading" in univ["detail"]


def test_regression_nazi_attic_still_forbidden() -> None:
    ir = _compile("nazi_attic")
    assert ir.projections["deontic_kantian"]["verdict"] == "forbidden"
