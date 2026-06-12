"""Tests for the canonicalizer backends."""

from pathlib import Path

import pytest
import yaml

from erisml_compiler.canonicalizer.base import auto_canonicalizer
from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer


@pytest.fixture
def known_forms():
    path = (
        Path(__file__).parent.parent
        / "src"
        / "erisml_compiler"
        / "ontology"
        / "canonical_forms.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {tag: entry.get("description", "") for tag, entry in data["canonical_forms"].items()}


# ---------- RegistryCanonicalizer ----------


def test_registry_canonicalizer_matches_nazi_attic(known_forms):
    # Low threshold reflects the honest performance of the no-ML
    # token-based backend; the LaBSE canonicalizer matches at much
    # higher similarity, see test_labse_canonicalizer_if_available.
    canon = RegistryCanonicalizer(threshold=0.02)
    summary = (
        "An agent under coercive interrogation by illegitimate authority "
        "where truth would lead to murder of innocents and authority "
        "threatens collective reprisal for deception. Murderous soldiers, "
        "hidden refugees, threatened village."
    )
    result = canon.canonicalize(summary, known_forms)
    assert result.tag == "coercive_murderous_interrogation_with_collective_reprisal"
    assert result.matched_known_form is True
    assert result.confidence >= 0.02


def test_registry_canonicalizer_no_match_returns_none(known_forms):
    canon = RegistryCanonicalizer(threshold=0.5)  # high threshold
    summary = "the cat sat on the mat"
    result = canon.canonicalize(summary, known_forms)
    assert result.tag is None
    assert result.matched_known_form is False


def test_registry_canonicalizer_empty_inputs():
    canon = RegistryCanonicalizer()
    result = canon.canonicalize("", {})
    assert result.tag is None


def test_registry_canonicalizer_matches_medical(known_forms):
    canon = RegistryCanonicalizer(threshold=0.02)
    summary = (
        "A professional bound by confidentiality, doctor lawyer or clergy, "
        "learns information signalling imminent serious harm to a third "
        "party. The role-duty of confidentiality conflicts with the "
        "protective duty owed to identifiable potential victims. Duty to warn."
    )
    result = canon.canonicalize(summary, known_forms)
    assert result.tag == "professional_privilege_versus_duty_to_warn"


# ---------- auto_canonicalizer ----------


def test_auto_canonicalizer_falls_back_to_registry():
    """If LaBSE isn't installed, auto falls back to registry without error."""
    canon = auto_canonicalizer(prefer_labse=False)
    assert canon.name == "registry"


# ---------- LaBSECanonicalizer (only if installed) ----------


def test_labse_canonicalizer_if_available(known_forms):
    pytest.importorskip("sentence_transformers")
    from erisml_compiler.canonicalizer.labse import LaBSECanonicalizer

    # Skip the network-dependent model load by default; this test is
    # marked optional. To opt in, set ERISML_TEST_LABSE=1.
    import os

    if not os.environ.get("ERISML_TEST_LABSE"):
        pytest.skip("Set ERISML_TEST_LABSE=1 to run LaBSE load tests.")

    canon = LaBSECanonicalizer()
    summary = "A villager vowed to hide refugees from murderous soldiers."
    result = canon.canonicalize(summary, known_forms)
    assert result.backend == "labse"
    assert isinstance(result.confidence, float)
