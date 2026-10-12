"""The I-EIP comparison in DEME's nine dimensions.

Since 0.10 the text lens is DEME's moral vector (erisml-lib, nine dimensions, values in [0, 1],
higher better except physical harm). The activation lens stays in the probes' ten channels
(``ir.schemas.MoralVector``), because their calibration (Phase-5 prereg C3) is defined there.
To compare them, both are put in DEME's nine dimensions, with the same shipped table the
compiler uses to migrate ten channels to nine (``ir.v3.dimensions.V2_TO_V3_DIMENSION_MAP``):
a channel's score goes to the dimension it maps to, vow fidelity split evenly between legitimacy
and virtue/care, repair dropped. Values are on the signed scale the probes use, +1 fully
respected, -1 fully violated. Phase 5 was run in the ten-channel space against the EM-DAG text
lens; it is reproducible at the tag ``phase5-results``.
"""

from __future__ import annotations

from erisml.ethics.moral_vector import MoralVector as DemeMoralVector
from pydantic import BaseModel

from erisml_compiler.ir.schemas import DimensionScore, MoralVector
from erisml_compiler.ir.v3.dimensions import V2_TO_V3_DIMENSION_MAP

DEME_DIMENSIONS: tuple[str, ...] = tuple(DemeMoralVector.core_dimension_names())
NEUTRAL_BAND = 0.1  # |signed value| below this has no direction


class DemeLens(BaseModel):
    """One lens's reading in DEME's nine dimensions, scored as the probes score."""

    physical_harm: DimensionScore
    rights_respect: DimensionScore
    fairness_equity: DimensionScore
    autonomy_respect: DimensionScore
    privacy_protection: DimensionScore
    societal_environmental: DimensionScore
    virtue_care: DimensionScore
    legitimacy_trust: DimensionScore
    epistemic_quality: DimensionScore


def _direction(value: float) -> str:
    if value > NEUTRAL_BAND:
        return "positive"
    if value < -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def text_lens(vector: DemeMoralVector) -> DemeLens:
    """DEME's vector on the signed scale (physical harm inverted first)."""
    scores = {}
    for dim in DEME_DIMENSIONS:
        v = getattr(vector, dim)
        signed = 2.0 * ((1.0 - v) if dim == "physical_harm" else v) - 1.0
        scores[dim] = DimensionScore(value=signed, direction=_direction(signed))
    return DemeLens(**scores)


def activation_lens(probes: MoralVector) -> DemeLens:
    """The probes' ten channels in DEME's nine dimensions (see the module docstring). A DEME
    dimension no channel reaches is neutral with full uncertainty."""
    parts: dict[str, list[tuple[float, DimensionScore]]] = {d: [] for d in DEME_DIMENSIONS}
    for channel, targets in V2_TO_V3_DIMENSION_MAP.items():
        score = getattr(probes, channel)
        for dim in targets:
            parts[dim].append((score.value / len(targets), score))
    out = {}
    for dim, contributions in parts.items():
        if not contributions:
            out[dim] = DimensionScore(value=0.0, uncertainty=1.0, confidence=0.0)
            continue
        value = max(-1.0, min(1.0, sum(v for v, _ in contributions)))
        out[dim] = DimensionScore(
            value=value,
            direction=_direction(value),
            confidence=min(s.confidence for _, s in contributions),
            uncertainty=max(s.uncertainty for _, s in contributions),
        )
    return DemeLens(**out)


__all__ = ["DEME_DIMENSIONS", "DemeLens", "activation_lens", "text_lens"]
