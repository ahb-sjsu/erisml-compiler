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
