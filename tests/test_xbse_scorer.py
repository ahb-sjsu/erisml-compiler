"""Integration tests for the xbse dimension-scoring backend (offline — fake encoder, no BGE-M3)."""

import numpy as np
import pytest

from erisml_compiler.ir.schemas import DimensionScore
from erisml_compiler.ir.v3.dimensions import MORAL_DIMENSIONS_V3

xbse_scorer = pytest.importorskip("xbse.scorer")  # skip if the `scorers` extra isn't installed
from erisml_compiler.scoring import (  # noqa: E402
    DEME9_REGISTRY,
    XBSEDimensionScorer,
    valence_to_dimension_score,
)


class FakeEncoder:
    """text → embedding by a 'good'/'bad' keyword rule; stands in for BGE-M3."""

    def encode(self, texts):
        return np.asarray(
            [[float(t.count("good") - t.count("bad")), 1.0, 0.0] for t in texts], dtype="float32"
        )


def _feeder(name="care"):
    enc = FakeEncoder()
    return xbse_scorer.DimensionScorer.fit(
        enc, ["good", "very good", "good good"], ["bad", "very bad", "bad bad"], name=name
    )


def test_registry_covers_all_nine_dimensions():
    assert set(DEME9_REGISTRY) == set(MORAL_DIMENSIONS_V3)


def test_validation_records_match_audit_schema():
    """The provenance dicts must carry exactly the FeederValidationRecord fields, so DEMEv3's
    DecisionProof can bind every xbse-scored dimension to its validated feeder + bar."""
    from xbse.report import Report

    report = Report(
        instance="care_joint",
        checkpoint_hash="deadbeef",
        thresholds={"auroc>": 0.79, "fuzz>": 1.0},
        metrics={"structure_auroc": 0.811, "bow_auroc": 0.527, "lexical_margin": 0.284},
        passed=True,
        bar_source="noise-ceiling(dual-judge)*0.9",
        bar_derivation="perfect-scorer AUROC vs dual-judge-flipped labels, *0.9, pre-training",
        bar_registered="2026-07-10",
    )
    backend = XBSEDimensionScorer({"virtue_care": _feeder()}, {"virtue_care": report})
    recs = backend.validation_records()
    assert len(recs) == 1
    r = recs[0]
    expected = {
        "dimension", "feeder_name", "checkpoint_hash", "bar_auroc_min", "bar_source",
        "bar_derivation", "bar_registered", "structure_auroc", "bow_auroc", "lexical_margin",
        "validated",
    }
    assert set(r) == expected
    assert r["dimension"] == "virtue_care" and r["feeder_name"] == "care_joint"
    assert r["bar_auroc_min"] == 0.79 and r["structure_auroc"] == 0.811 and r["validated"] is True


def test_wraps_valence_into_dimension_score():
    backend = XBSEDimensionScorer({"virtue_care": _feeder()})
    ds = backend.score("good good good", "virtue_care")
    assert isinstance(ds, DimensionScore)
    assert ds.value > 0 and ds.direction == "positive"
    assert -1.0 <= ds.value <= 1.0 and 0.0 <= ds.confidence <= 1.0


def test_missing_feeder_scores_neutral():
    backend = XBSEDimensionScorer({"virtue_care": _feeder()})
    ds = backend.score("anything", "physical_harm")
    assert ds.value == 0.0 and ds.direction == "neutral"


def test_score_all_covers_deme9():
    backend = XBSEDimensionScorer({"virtue_care": _feeder()})
    scored = backend.score_all("bad bad")
    assert set(scored) == set(MORAL_DIMENSIONS_V3)
    assert all(isinstance(v, DimensionScore) for v in scored.values())
    assert scored["virtue_care"].direction == "negative"


def test_valence_conversion_clamps_and_sets_uncertainty():
    v = xbse_scorer.Valence(value=1.7, confidence=0.9, direction="positive")
    ds = valence_to_dimension_score(v, "physical_harm")
    assert ds.value == 1.0
    assert abs(ds.uncertainty - 0.1) < 1e-6
