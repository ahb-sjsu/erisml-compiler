"""Pluggable dimension-scoring backends for the MoralVector.

The compiler's evaluation is erisml-lib's DEME (since 0.10; the rubric EM-DAG is retired).
This package adds an alternative, validated backend: the cross-dataset `xbse` feeders, one trained
encoder per DEME-9 dimension, each gated by `require_pass`. Opt-in via the `scorers` extra.
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
