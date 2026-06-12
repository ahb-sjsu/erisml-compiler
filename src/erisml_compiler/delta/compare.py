"""Compare a text-lens MoralVector with an activation-lens MoralVector.

`compare_morals(text_mv, activation_mv)` returns:

  - per-dimension delta: `value_delta`, `direction_match`, `confidence_gap`,
    `joint_uncertainty`
  - overall `divergence` in [0, 1] — a weighted average of |value_delta|
    inflated when directions disagree
  - `flag_for_review` — boolean; true when divergence exceeds a configured
    threshold, when more than `direction_break_count` dimensions disagree
    in direction, or when joint uncertainty is high *and* values differ

Threshold defaults are conservative; the I-EIP Monitor's intent is to
err on the side of escalation (`requires_human_review`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, MoralVector

# Direction encoding: -1 negative, 0 neutral, +1 positive. Lets the
# delta arithmetic stay numerical.
_DIRECTION_SIGN = {"positive": 1, "neutral": 0, "negative": -1}


@dataclass(frozen=True)
class DimensionDelta:
    dimension: str
    text_value: float
    activation_value: float
    value_delta: float  # activation - text
    text_direction: str
    activation_direction: str
    direction_match: bool
    confidence_gap: float  # |text.confidence - activation.confidence|
    joint_uncertainty: float  # max(text.uncertainty, activation.uncertainty)


@dataclass(frozen=True)
class DeltaResult:
    per_dimension: list[DimensionDelta]
    divergence: float
    direction_break_count: int
    high_uncertainty_dimensions: list[str]
    flag_for_review: bool
    config: dict = field(default_factory=dict)

    def by_dimension(self, name: str) -> DimensionDelta:
        for d in self.per_dimension:
            if d.dimension == name:
                return d
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "per_dimension": [
                {
                    "dimension": d.dimension,
                    "text_value": d.text_value,
                    "activation_value": d.activation_value,
                    "value_delta": d.value_delta,
                    "text_direction": d.text_direction,
                    "activation_direction": d.activation_direction,
                    "direction_match": d.direction_match,
                    "confidence_gap": d.confidence_gap,
                    "joint_uncertainty": d.joint_uncertainty,
                }
                for d in self.per_dimension
            ],
            "divergence": self.divergence,
            "direction_break_count": self.direction_break_count,
            "high_uncertainty_dimensions": self.high_uncertainty_dimensions,
            "flag_for_review": self.flag_for_review,
            "config": self.config,
        }


def compare_morals(
    text_mv: MoralVector,
    activation_mv: MoralVector,
    *,
    weights: dict[str, float] | None = None,
    divergence_threshold: float = 0.35,
    direction_break_max: int = 2,
    high_uncertainty_threshold: float = 0.7,
    uncertain_value_delta_max: float = 0.25,
) -> DeltaResult:
    """Compute the delta between text and activation moral vectors.

    Args:
        text_mv: the MoralVector from the text lens (Phases 1-3).
        activation_mv: the MoralVector from the activation lens (Track A).
        weights: optional per-dimension weights for the divergence
            average. Unspecified dimensions default to 1.0.
        divergence_threshold: if `divergence` exceeds this, raise
            flag_for_review. Default 0.35.
        direction_break_max: if more than this many dimensions disagree
            in direction, raise flag_for_review. Default 2 (i.e., 3+
            mismatches triggers review).
        high_uncertainty_threshold: a dimension is "uncertain" when both
            sides' uncertainty exceeds this. Default 0.7.
        uncertain_value_delta_max: when a dimension is uncertain *and*
            |value_delta| exceeds this, raise flag_for_review.
    """
    weights = weights or {}
    per_dim: list[DimensionDelta] = []
    weighted_sum = 0.0
    weight_total = 0.0
    direction_breaks = 0
    uncertain_dims: list[str] = []

    for dim in MORAL_DIMENSIONS:
        t = getattr(text_mv, dim)
        a = getattr(activation_mv, dim)

        direction_match = t.direction == a.direction
        if not direction_match:
            direction_breaks += 1

        value_delta = a.value - t.value
        confidence_gap = abs(t.confidence - a.confidence)
        joint_uncertainty = max(t.uncertainty, a.uncertainty)

        per_dim.append(
            DimensionDelta(
                dimension=dim,
                text_value=float(t.value),
                activation_value=float(a.value),
                value_delta=float(value_delta),
                text_direction=t.direction,
                activation_direction=a.direction,
                direction_match=direction_match,
                confidence_gap=float(confidence_gap),
                joint_uncertainty=float(joint_uncertainty),
            )
        )

        w = weights.get(dim, 1.0)
        # Direction disagreement inflates the per-dim contribution by
        # 50% (capped at 1.0). This makes sign-flips dominate magnitude
        # noise in the divergence summary.
        contribution = abs(value_delta) + (0.5 if not direction_match else 0.0)
        contribution = min(contribution, 1.0)
        weighted_sum += w * contribution
        weight_total += w

        if (
            t.uncertainty >= high_uncertainty_threshold
            and a.uncertainty >= high_uncertainty_threshold
        ):
            uncertain_dims.append(dim)

    divergence = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    flag = (
        divergence >= divergence_threshold
        or direction_breaks > direction_break_max
        or any(
            d.dimension in uncertain_dims and abs(d.value_delta) >= uncertain_value_delta_max
            for d in per_dim
        )
    )

    return DeltaResult(
        per_dimension=per_dim,
        divergence=float(divergence),
        direction_break_count=direction_breaks,
        high_uncertainty_dimensions=uncertain_dims,
        flag_for_review=bool(flag),
        config={
            "divergence_threshold": divergence_threshold,
            "direction_break_max": direction_break_max,
            "high_uncertainty_threshold": high_uncertainty_threshold,
            "uncertain_value_delta_max": uncertain_value_delta_max,
        },
    )
