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
        "dimension",
        "feeder_name",
        "checkpoint_hash",
        "bar_auroc_min",
        "bar_source",
        "bar_derivation",
        "bar_registered",
        "structure_auroc",
        "bow_auroc",
        "lexical_margin",
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


def _calibrated_report(instance="care_joint", weight=0.625):
    from xbse.report import Report

    return Report(
        instance=instance,
        checkpoint_hash="deadbeef",
        thresholds={"auroc>": 0.79},
        metrics={"structure_auroc": 0.813, "bow_auroc": 0.527, "lexical_margin": 0.284},
        passed=True,
        bar_registered="2026-07-10",
        calibration={
            "calibration_method": "isotonic",
            "calibration_ece": 0.057,
            "raw_ece": 0.168,
            "reliability_weight": weight,
            "n_calibration_pairs": 1200,
        },
    )


def test_reliability_weight_scales_confidence_and_uncertainty():
    """R2 closure: the report's reliability_weight must reach MoralVector uncertainty — a weighted
    feeder's confidence is its raw confidence times the weight, never equal authority."""
    backend = XBSEDimensionScorer(
        {"virtue_care": _feeder()}, {"virtue_care": _calibrated_report(weight=0.5)}
    )
    unweighted = XBSEDimensionScorer({"virtue_care": _feeder()})
    ds = backend.score("good good good", "virtue_care")
    ds0 = unweighted.score("good good good", "virtue_care")
    assert abs(ds.confidence - 0.5 * ds0.confidence) < 1e-9
    assert abs(ds.uncertainty - (1.0 - ds.confidence)) < 1e-9
    assert "reliability_weight=0.500" in ds.explanation


def test_uncalibrated_report_scores_unweighted():
    """Pre-calibration reports (no block) must behave exactly as before — no silent zero-weight."""
    from xbse.report import Report

    legacy = Report(
        instance="care_joint", checkpoint_hash="deadbeef", thresholds={}, metrics={}, passed=True
    )
    backend = XBSEDimensionScorer({"virtue_care": _feeder()}, {"virtue_care": legacy})
    ds = backend.score("good good good", "virtue_care")
    ds0 = XBSEDimensionScorer({"virtue_care": _feeder()}).score("good good good", "virtue_care")
    assert abs(ds.confidence - ds0.confidence) < 1e-9
    assert "reliability_weight" not in ds.explanation


def test_demoted_dimension_carries_disposition():
    """A DEMOTE-to-G feeder's score must say it reads general valence, not its named axis."""
    from erisml_compiler.scoring import SPECIFICITY_DISPOSITIONS

    assert SPECIFICITY_DISPOSITIONS["virtue_care"] == "DEMOTE-to-G"
    assert SPECIFICITY_DISPOSITIONS["privacy_protection"] == "own-axis"
    backend = XBSEDimensionScorer({"virtue_care": _feeder()})
    ds = backend.score("good good good", "virtue_care")
    assert "DEMOTE-to-G" in ds.explanation
    # an own-axis dimension is never flagged
    backend2 = XBSEDimensionScorer({"privacy_protection": _feeder("privacy")})
    assert "DEMOTE-to-G" not in backend2.score("good", "privacy_protection").explanation


def test_reliability_records_expose_calibration_block():
    backend = XBSEDimensionScorer({"virtue_care": _feeder()}, {"virtue_care": _calibrated_report()})
    recs = backend.reliability_records()
    assert len(recs) == 1
    r = recs[0]
    assert r["dimension"] == "virtue_care"
    assert r["reliability_weight"] == 0.625
    assert r["calibration_ece"] == 0.057 and r["raw_ece"] == 0.168
    assert r["calibration_method"] == "isotonic"
    assert r["specificity_disposition"] == "DEMOTE-to-G"


def test_validation_records_schema_is_unchanged_by_calibration():
    """validation_records is pinned to FeederValidationRecord (erisml-lib) — the calibration wiring
    must not leak new fields into it."""
    backend = XBSEDimensionScorer({"virtue_care": _feeder()}, {"virtue_care": _calibrated_report()})
    r = backend.validation_records()[0]
    assert "reliability_weight" not in r and "calibration_ece" not in r
