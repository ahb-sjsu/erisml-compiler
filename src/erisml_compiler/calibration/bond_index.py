"""Bond Index: aggregate cross-cultural / cross-lingual generalisation score.

Per sqnd-probe, the Bond Index aggregates probe-head accuracy across
multiple held-out evaluation axes:

    - cross-lingual transfer (train on lang A, test on lang B)
    - cross-tradition (train on tradition X, test on tradition Y)
    - cross-script (Hebrew-Arabic, etc.)
    - temporal (ancient -> modern)
    - cultural-invariance (combined held-out splits)

The compiler's implementation takes per-axis accuracy values and returns
a weighted mean (the BI itself) plus per-axis breakdown.

A high BI indicates the probe extracts culture-invariant moral structure;
a low BI indicates the probe is using language/period/tradition shortcuts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BondIndexResult:
    """Aggregate Bond Index plus per-axis breakdown."""

    bond_index: float
    per_axis: dict[str, float]
    weights: dict[str, float]

    def summary_lines(self) -> list[str]:
        lines = [f"Bond Index: {self.bond_index:.4f}"]
        for axis, acc in self.per_axis.items():
            w = self.weights.get(axis, 1.0)
            lines.append(f"  {axis:30s} acc={acc:.4f}  (weight={w:.2f})")
        return lines


def compute_bond_index(
    per_axis_accuracy: dict[str, float],
    weights: dict[str, float] | None = None,
) -> BondIndexResult:
    """Weighted-mean Bond Index.

    Args:
        per_axis_accuracy: e.g. {"cross_lingual": 0.81, "temporal": 0.75, ...}
        weights: optional per-axis weights. Defaults to equal weights.

    Returns:
        BondIndexResult.
    """
    if not per_axis_accuracy:
        return BondIndexResult(bond_index=0.0, per_axis={}, weights={})
    if weights is None:
        weights = {k: 1.0 for k in per_axis_accuracy}

    total_w = sum(weights.get(k, 0.0) for k in per_axis_accuracy)
    if total_w == 0.0:
        return BondIndexResult(bond_index=0.0, per_axis=per_axis_accuracy, weights=weights)

    bi = sum(per_axis_accuracy[k] * weights.get(k, 0.0) for k in per_axis_accuracy) / total_w
    return BondIndexResult(bond_index=bi, per_axis=per_axis_accuracy, weights=weights)
