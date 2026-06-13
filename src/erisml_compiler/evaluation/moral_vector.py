"""Project EM-DAG outputs onto the 10-dimensional MoralVector.

Each EM module declares which `dimension` of the MoralVector it owns; the
projection is a simple lookup. An optional ethos profile (per-module
weights) scales the per-dimension `value` at projection time.
"""

from __future__ import annotations

from erisml_compiler.em_dag import EMDAG
from erisml_compiler.ir.schemas import (
    DimensionScore,
    EMOutput,
    MORAL_DIMENSIONS,
    MoralVector,
)


def _scale_dimension_score(score: DimensionScore, weight: float) -> DimensionScore:
    """Multiply the score's value by `weight`, clamped to [-1, 1].

    Confidence and direction are unchanged — weighting expresses how
    much *salience* the named ethos assigns to a channel, not how
    confident the module was in its reading.
    """
    if weight == 1.0:
        return score
    scaled = max(-1.0, min(1.0, score.value * weight))
    return score.model_copy(update={"value": scaled})


def build_moral_vector_from_em_outputs(
    em_outputs: dict[str, EMOutput],
    dag: EMDAG,
    *,
    ethos_weights: dict[str, float] | None = None,
) -> MoralVector:
    """Project per-module outputs to the 10-dimensional vector.

    For each MoralVector dimension, find the EM whose `dimension` attribute
    matches and copy its DimensionScore. If no EM owns the dimension, emit a
    neutral zero score.

    If `ethos_weights` is provided (a per-module-name -> weight mapping
    from a fitted ethos profile, e.g. dear_abby_audience_baseline),
    each contributing module's value is multiplied by its weight.
    Modules absent from the mapping get weight 1.0 (no scaling).
    """
    by_dimension: dict[str, DimensionScore] = {}
    for module in dag.modules.values():
        if module.dimension in MORAL_DIMENSIONS:
            output = em_outputs.get(module.name)
            if output is not None:
                w = 1.0 if ethos_weights is None else float(ethos_weights.get(module.name, 1.0))
                by_dimension[module.dimension] = _scale_dimension_score(output.score, w)

    # Fill any missing dimensions with neutral zero.
    for dim in MORAL_DIMENSIONS:
        if dim not in by_dimension:
            by_dimension[dim] = DimensionScore(
                value=0.0,
                confidence=1.0,
                uncertainty=0.0,
                direction="neutral",
                source_spans=[],
                explanation="No EM module owns this dimension in the current DAG.",
            )

    return MoralVector(**by_dimension)
