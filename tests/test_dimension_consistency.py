"""Cross-package guard: the DEME-9 k-axis must be identical everywhere.

The canonical nine moral dimensions are, by design, duplicated in three packages:
  - erisml_compiler.ir.v3.dimensions.MORAL_DIMENSIONS_V3   (this repo)
  - erisml.ethics.moral_tensor.MORAL_DIMENSION_NAMES       (erisml-lib / DEME engine)
  - agi.safety.erisml.moral_tensor.MORAL_DIMENSION_NAMES   (agi-hpc safety gateway)

They MUST stay in sync (same names, same order) — the rank-1..6 MoralTensor's `k` axis is
indexed positionally, so any reorder/rename in one package silently misaligns the whole stack.
This test freezes the canonical tuple and checks every importable sibling against it, so a drift
fails CI instead of corrupting a tensor at runtime.
"""

from __future__ import annotations

import importlib

import pytest

from erisml_compiler.ir.v3.dimensions import MORAL_DIMENSIONS_V3

# The frozen canonical order — DEME 3.0 "Nine Dimensions of Ethical Assessment" (3x3).
# Editing this tuple is a deliberate, breaking act: update ALL THREE packages together.
CANONICAL_9: tuple[str, ...] = (
    "physical_harm",           # k0  Relational / Consequences-and-Welfare
    "rights_respect",          # k1  Individual  / Rights-and-Duties
    "fairness_equity",         # k2  Collective  / Justice-and-Fairness
    "autonomy_respect",        # k3  Individual  / Autonomy-and-Agency
    "privacy_protection",      # k4  Individual  / Privacy-and-Data
    "societal_environmental",  # k5  Collective  / Societal-and-Environmental
    "virtue_care",             # k6  Relational  / Virtue-and-Care
    "legitimacy_trust",        # k7  Collective  / Procedural-Legitimacy
    "epistemic_quality",       # k8  Relational  / Epistemic-Status
)


def test_compiler_matches_canonical():
    assert tuple(MORAL_DIMENSIONS_V3) == CANONICAL_9


@pytest.mark.parametrize(
    "modname,attr",
    [
        ("erisml.ethics.moral_tensor", "MORAL_DIMENSION_NAMES"),          # erisml-lib
        ("agi.safety.erisml.moral_tensor", "MORAL_DIMENSION_NAMES"),      # agi-hpc
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
