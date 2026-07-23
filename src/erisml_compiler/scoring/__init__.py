"""Pluggable dimension-scoring backends for the MoralVector.

The default compiler path scores dimensions with a rubric over extracted EthicalFacts (`em_dag`).
This package adds an alternative, validated backend: the cross-dataset `xbse` feeders, one trained
encoder per DEME-9 dimension, each gated by `require_pass`. Kept swappable — the rubric remains the
default; the xbse backend is opt-in via the `scorers` extra.
"""

from .xbse_scorer import (
    DEME9_REGISTRY,
    SPECIFICITY_DISPOSITIONS,
    DimensionScoringBackend,
    XBSEDimensionScorer,
    valence_to_dimension_score,
)

__all__ = [
    "DimensionScoringBackend",
    "XBSEDimensionScorer",
    "DEME9_REGISTRY",
    "SPECIFICITY_DISPOSITIONS",
    "valence_to_dimension_score",
]
