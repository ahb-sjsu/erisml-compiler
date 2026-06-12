"""Shared helpers for EM evaluators.

These helpers count and aggregate ethical facts of given kinds from the IR.
They keep individual EMs short and consistent. All arithmetic here is fixed
sign convention: positive scores indicate the dimension is being respected
(harm avoided, autonomy honored); negative scores indicate violation.
"""
from __future__ import annotations

from erisml_compiler.ir.schemas import (
    Commitment,
    DimensionScore,
    EthicalFact,
    EthicalFactKind,
    CompilerIR,
)

SEVERITY_WEIGHT = {
    "minor": 0.2,
    "moderate": 0.5,
    "grave": 0.85,
    "catastrophic": 1.0,
}


def facts_of_kind(ir: CompilerIR, kind: EthicalFactKind) -> list[EthicalFact]:
    return [f for f in ir.ethical_facts if f.kind == kind]


def severity_score(fact: EthicalFact) -> float:
    """Map a fact's severity to a magnitude in [0, 1]."""
    if not fact.severity:
        return 0.5
    return SEVERITY_WEIGHT.get(fact.severity, 0.5)


def aggregate_negative(
    facts: list[EthicalFact],
    explanation_prefix: str = "",
) -> DimensionScore:
    """Aggregate facts into a negative-valued DimensionScore (violation
    detected, score moves toward -1).

    The aggregation is: max severity over contributing facts, with confidence
    averaged. If no facts, returns a neutral 0.0 score.
    """
    if not facts:
        return DimensionScore(
            value=0.0,
            confidence=1.0,
            uncertainty=0.0,
            direction="neutral",
            source_spans=[],
            explanation=f"{explanation_prefix}No relevant facts detected.",
        )
    magnitudes = [severity_score(f) for f in facts]
    max_mag = max(magnitudes)
    mean_conf = sum(f.confidence for f in facts) / len(facts)
    spans: list[str] = []
    for f in facts:
        spans.extend(f.source_spans)
    descriptions = "; ".join(f.description for f in facts[:3])
    return DimensionScore(
        value=-max_mag,
        confidence=mean_conf,
        uncertainty=1.0 - mean_conf,
        direction="negative",
        source_spans=spans,
        explanation=f"{explanation_prefix}{descriptions}",
    )


def aggregate_positive(
    facts: list[EthicalFact],
    explanation_prefix: str = "",
) -> DimensionScore:
    """Aggregate facts into a positive-valued DimensionScore (good behaviour
    detected, score moves toward +1)."""
    if not facts:
        return DimensionScore(
            value=0.0,
            confidence=1.0,
            uncertainty=0.0,
            direction="neutral",
            source_spans=[],
            explanation=f"{explanation_prefix}No relevant facts detected.",
        )
    magnitudes = [severity_score(f) for f in facts]
    max_mag = max(magnitudes)
    mean_conf = sum(f.confidence for f in facts) / len(facts)
    spans: list[str] = []
    for f in facts:
        spans.extend(f.source_spans)
    descriptions = "; ".join(f.description for f in facts[:3])
    return DimensionScore(
        value=+max_mag,
        confidence=mean_conf,
        uncertainty=1.0 - mean_conf,
        direction="positive",
        source_spans=spans,
        explanation=f"{explanation_prefix}{descriptions}",
    )


def active_commitments(ir: CompilerIR) -> list[Commitment]:
    return [c for c in ir.commitments if c.status in ("active", "active_but_defeasible", "fulfilled")]


def violated_commitments(ir: CompilerIR) -> list[Commitment]:
    return [c for c in ir.commitments if c.status == "violated"]
