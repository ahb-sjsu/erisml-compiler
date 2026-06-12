"""Project EM-DAG outputs onto the 10-dimensional MoralVector.

Each EM module declares which `dimension` of the MoralVector it owns; the
projection is a simple lookup.
"""
from __future__ import annotations

from erisml_compiler.em_dag import EMDAG
from erisml_compiler.ir.schemas import (
    DimensionScore,
    EMOutput,
    MORAL_DIMENSIONS,
    MoralVector,
)


def build_moral_vector_from_em_outputs(
    em_outputs: dict[str, EMOutput],
    dag: EMDAG,
) -> MoralVector:
    """Project per-module outputs to the 10-dimensional vector.

    For each MoralVector dimension, find the EM whose `dimension` attribute
    matches and copy its DimensionScore. If no EM owns the dimension, emit a
    neutral zero score.
    """
    by_dimension: dict[str, DimensionScore] = {}
    for module in dag.modules.values():
        if module.dimension in MORAL_DIMENSIONS:
            output = em_outputs.get(module.name)
            if output is not None:
                by_dimension[module.dimension] = output.score

    # Fill any missing dimensions with neutral zero.
    for dim in MORAL_DIMENSIONS:
        if dim not in by_dimension:
            by_dimension[dim] = DimensionScore(
                value=0.0, confidence=1.0, uncertainty=0.0, direction="neutral",
                source_spans=[],
                explanation="No EM module owns this dimension in the current DAG.",
            )

    return MoralVector(**by_dimension)
