"""End-to-end test that --ethos-profile actually scales per-module values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from erisml_compiler.em_dag import load_profile
from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
from erisml_compiler.ir.schemas import DimensionScore, EMOutput
from erisml_compiler.pipeline.orchestrator import _resolve_ethos_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = REPO_ROOT / "src" / "erisml_compiler" / "em_dag" / "profiles"


def _make_em_output(module_name: str, value: float, confidence: float = 0.8) -> EMOutput:
    return EMOutput(
        module_name=module_name,
        score=DimensionScore(
            value=value,
            confidence=confidence,
            uncertainty=0.0,
            direction="negative" if value < 0 else ("positive" if value > 0 else "neutral"),
            source_spans=[],
            explanation="test",
        ),
        contributing_facts=[],
    )


def _stub_em_outputs(dag) -> dict[str, EMOutput]:
    """Give every module in the default DAG a non-zero output so weighting
    has something to bite on."""
    return {name: _make_em_output(name, 0.6) for name in dag.modules}


def test_resolve_ethos_profile_returns_weights_name_sha(tmp_path: Path) -> None:
    payload = {
        "name": "test_profile",
        "weights": {"harm": 1.5, "care": 0.8},
        "priors": {"harm": -0.1, "care": -0.1},
        "coverage": {"harm": 0.7, "care": 0.7},
        "fit_method": "manual",
    }
    p = tmp_path / "ethos.yaml"
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")

    weights, name, sha = _resolve_ethos_profile(p)
    assert name == "test_profile"
    assert weights == {"harm": 1.5, "care": 0.8}
    assert len(sha) == 64


def test_resolve_ethos_profile_none_path() -> None:
    weights, name, sha = _resolve_ethos_profile(None)
    assert weights is None and name is None and sha is None


def test_dear_abby_profile_uses_canonical_module_names() -> None:
    """The shipped fitted profile must use lowercase module names so it
    actually matches the EM-DAG modules at projection time. Class-name
    keys (HarmEM/CareEM) would silently no-op."""
    path = PROFILES_DIR / "dear_abby_socialchem_v0.1.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    weights = data["weights"]
    canonical = {
        "harm",
        "rights",
        "fairness",
        "legitimacy",
        "epistemic",
        "autonomy",
        "fidelity",
        "externality",
        "care",
        "repair",
    }
    for k in weights:
        assert k in canonical, f"Profile weight key {k!r} is not a canonical EM module name"


def test_ethos_weights_scale_moral_vector_values() -> None:
    """Loading the shipped dearabby profile and applying it via
    build_moral_vector_from_em_outputs scales the relevant module
    values, leaves unmapped modules at the original value."""
    dag = load_profile(PROFILES_DIR / "default.yaml")
    em_outputs = _stub_em_outputs(dag)

    profile_path = PROFILES_DIR / "dear_abby_socialchem_v0.1.yaml"
    weights, _, _ = _resolve_ethos_profile(profile_path)
    assert weights is not None

    unweighted = build_moral_vector_from_em_outputs(em_outputs, dag)
    weighted = build_moral_vector_from_em_outputs(em_outputs, dag, ethos_weights=weights)

    # 'autonomy_consent' has no mapping in the profile (autonomy is not
    # in weights dict, defaults to 1.0) -> unchanged.
    assert weighted.autonomy_consent.value == pytest.approx(
        unweighted.autonomy_consent.value, abs=1e-9
    )
    # 'vow_fidelity' (fidelity module) has weight 0.7237 in the shipped profile.
    assert abs(weighted.vow_fidelity.value) < abs(unweighted.vow_fidelity.value)
    expected = unweighted.vow_fidelity.value * weights["fidelity"]
    assert weighted.vow_fidelity.value == pytest.approx(expected, abs=1e-5)
    # 'physical_harm' (harm module) has weight 1.748 — clamps to ±1.0 since
    # the stub value 0.6 * 1.748 = 1.049 > 1.0.
    assert weighted.physical_harm.value == pytest.approx(1.0, abs=1e-5)
