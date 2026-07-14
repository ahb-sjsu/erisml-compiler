"""Cross-package guard: the MoralVector ontology has ONE source of truth.

`erisml_compiler.ir.v3.dimensions` is the single source of truth (SSOT) for the
moral vector's channels. Other packages reference it:
  - erisml.ethics.moral_tensor.MORAL_DIMENSION_NAMES       (erisml-lib) — IMPORTS the
    SSOT, with a literal fallback (guarded here) for standalone installs.
  - agi.safety.erisml.moral_tensor.MORAL_DIMENSION_NAMES   (agi-hpc safety gateway) —
    a separate, safety-critical system carrying its own guarded copy.

The rank-1..6 MoralTensor's `k` axis is indexed positionally, so any reorder/rename
silently misaligns the whole stack. This test freezes the canonical tuples and checks
every importable sibling against them, so drift fails CI instead of corrupting a tensor
at runtime.
"""

from __future__ import annotations

import importlib

import pytest

from erisml_compiler.ir.v3.dimensions import (
    DIMENSION_MATRIX_3X3,
    FRAMINGS,
    LEVELS,
    MORAL_DIMENSIONS_V3,
    MORAL_EXTENSION_CHANNELS,
    MORAL_VECTOR_CHANNELS,
)

# The frozen canonical order — DEME 3.0 "Nine Dimensions of Ethical Assessment" (3x3).
# Editing this tuple is a deliberate, breaking act: update the SSOT + any guarded copies.
CANONICAL_9: tuple[str, ...] = (
    "physical_harm",  # k0  Relational / Consequences-and-Welfare
    "rights_respect",  # k1  Individual  / Rights-and-Duties
    "fairness_equity",  # k2  Collective  / Justice-and-Fairness
    "autonomy_respect",  # k3  Individual  / Autonomy-and-Agency
    "privacy_protection",  # k4  Individual  / Privacy-and-Data
    "societal_environmental",  # k5  Collective  / Societal-and-Environmental
    "virtue_care",  # k6  Relational  / Virtue-and-Care
    "legitimacy_trust",  # k7  Collective  / Procedural-Legitimacy
    "epistemic_quality",  # k8  Relational  / Epistemic-Status
)

# The frozen extension channels — validated moral foundations OUTSIDE the k-axis.
# Adding one is a deliberate, reviewed act (needs a validated feeder; see
# EXTENSION_CHANNEL_PROVENANCE).
CANONICAL_EXTENSIONS: tuple[str, ...] = ("purity", "loyalty")

# The frozen scope×mode (level×framing) assignment. This is the SEMANTICS the
# drift-guard protects, not just the names: the keystone Table 1, the MoralVector
# paper's k-table, and the reference diagram must all match THIS assignment.
# Editing it is a major-version change to the ontology, never an editorial one.
CANONICAL_MATRIX_3X3: dict[str, tuple[str, str]] = {
    "physical_harm": ("Relational", "Who Decides"),
    "rights_respect": ("Individual", "Who Decides"),
    "fairness_equity": ("Collective", "What Matters"),
    "autonomy_respect": ("Individual", "What Matters"),
    "privacy_protection": ("Individual", "What We Know"),
    "societal_environmental": ("Collective", "What We Know"),
    "virtue_care": ("Relational", "What Matters"),
    "legitimacy_trust": ("Collective", "Who Decides"),
    "epistemic_quality": ("Relational", "What We Know"),
}


def test_compiler_matches_canonical():
    assert tuple(MORAL_DIMENSIONS_V3) == CANONICAL_9


def test_matrix_matches_canonical_assignment():
    """The scope×mode assignment is frozen; papers + diagram generate from it."""
    assert DIMENSION_MATRIX_3X3 == CANONICAL_MATRIX_3X3


def test_matrix_is_a_bijection_onto_the_nine_cells():
    """Each of the nine dimensions occupies exactly one distinct level×framing cell."""
    assert set(DIMENSION_MATRIX_3X3) == set(CANONICAL_9)
    cells = list(DIMENSION_MATRIX_3X3.values())
    assert len(cells) == len(set(cells)) == 9
    assert set(cells) == {(lv, fr) for lv in LEVELS for fr in FRAMINGS}


def test_extension_channels_frozen():
    assert tuple(MORAL_EXTENSION_CHANNELS) == CANONICAL_EXTENSIONS
    # extension channels must not collide with the k-axis
    assert not (set(MORAL_EXTENSION_CHANNELS) & set(MORAL_DIMENSIONS_V3))


def test_full_vocabulary_is_axis_plus_extensions():
    assert tuple(MORAL_VECTOR_CHANNELS) == CANONICAL_9 + CANONICAL_EXTENSIONS


@pytest.mark.parametrize(
    "modname,attr",
    [
        ("erisml.ethics.moral_tensor", "MORAL_DIMENSION_NAMES"),  # erisml-lib
        ("agi.safety.erisml.moral_tensor", "MORAL_DIMENSION_NAMES"),  # agi-hpc
    ],
)
def test_sibling_packages_match_canonical(modname: str, attr: str):
    """Skip if the sibling package isn't installed in this environment; otherwise
    assert its k-axis is byte-for-byte the canonical DEME-9."""
    try:
        mod = importlib.import_module(modname)
    except Exception:  # ImportError or transitive dep failure
        pytest.skip(f"{modname} not importable in this environment")
    got = tuple(getattr(mod, attr))
    assert got == CANONICAL_9, f"{modname}.{attr} drifted from the canonical DEME-9:\n{got}"


def test_erisml_lib_references_ssot():
    """erisml-lib should re-export the SSOT extension channels (proves the import wiring,
    not just an equal literal). Skips if erisml-lib isn't installed."""
    try:
        mod = importlib.import_module("erisml.ethics.moral_tensor")
    except Exception:
        pytest.skip("erisml-lib not importable in this environment")
    assert tuple(getattr(mod, "MORAL_EXTENSION_CHANNELS")) == CANONICAL_EXTENSIONS
    assert tuple(getattr(mod, "MORAL_VECTOR_CHANNELS")) == CANONICAL_9 + CANONICAL_EXTENSIONS
