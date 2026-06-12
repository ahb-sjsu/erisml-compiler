"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from erisml_compiler.ir.schemas import (
    Commitment,
    Conflict,
    DimensionScore,
    EthicalFact,
    SourceSpan,
    Stakeholder,
)


def test_dimension_score_bounds():
    DimensionScore(value=0.5, confidence=0.8, uncertainty=0.2, direction="positive")
    with pytest.raises(ValidationError):
        DimensionScore(value=1.5, confidence=0.8, uncertainty=0.2, direction="positive")
    with pytest.raises(ValidationError):
        DimensionScore(value=0.5, confidence=1.5, uncertainty=0.2, direction="positive")


def test_source_span_round_trip():
    sp = SourceSpan(segment_id="seg_001", start=10, end=50)
    assert SourceSpan.parse(sp.as_string()) == sp


def test_source_span_end_before_start_rejected():
    with pytest.raises(ValidationError):
        SourceSpan(segment_id="seg_001", start=50, end=10)


def test_stakeholder_minimum():
    s = Stakeholder(id="x", label="X", type="individual")
    assert s.confidence == 1.0
    assert s.requires_review is False


def test_commitment_status_validation():
    c = Commitment(id="c1", type="vow", holder="self", content="x")
    assert c.status == "active"
    with pytest.raises(ValidationError):
        Commitment(id="c1", type="vow", holder="self", content="x", status="bogus_status")


def test_conflict_severity_validation():
    cf = Conflict(id="c1", name="x", severity="grave")
    assert cf.severity == "grave"
    with pytest.raises(ValidationError):
        Conflict(id="c1", name="x", severity="apocalyptic")


def test_ethical_fact_kinds():
    EthicalFact(id="f1", kind="coercion", description="x")
    with pytest.raises(ValidationError):
        EthicalFact(id="f1", kind="unrecognised_kind", description="x")
