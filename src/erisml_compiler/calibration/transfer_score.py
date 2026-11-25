"""Transfer score: how well a probe head generalises across held-out axes.

Per sqnd-probe, a probe is evaluated on several held-out axes:

    - cross-lingual transfer (train on lang A, test on lang B)
    - cross-tradition (train on tradition X, test on tradition Y)
    - cross-script (Hebrew-Arabic, etc.)
    - temporal (ancient -> modern)
    - cultural-invariance (combined held-out splits)

This module takes the accuracy on each axis and returns their weighted mean with the
per-axis breakdown. A high score indicates the probe extracts culture-invariant moral
structure; a low score indicates it is using language, period or tradition shortcuts.

This quantity was called the Bond Index here until November 2026. It is not the Bond
Index: that is the mean deviation of an evaluation under declared re-descriptions, over a
tolerance, where 0 is best (erisml-lib, ``erisml.ethics.bond_index`` and
``docs/BOND_INDEX.md``). A mean of held-out accuracies, where 1 is best, measures something
else, so it has its own name. ``compute_bond_index`` and ``BondIndexResult`` remain as
deprecated aliases.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass
class TransferScore:
    """The weighted mean of held-out accuracies, with the per-axis breakdown."""

    score: float
    per_axis: dict[str, float]
    weights: dict[str, float]

    @property
    def bond_index(self) -> float:
        """Deprecated name of ``score``."""
        warnings.warn(
            "bond_index is the old name of the transfer score; use .score. The Bond Index is "
            "erisml.ethics.bond_index in erisml-lib.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.score

    def summary_lines(self) -> list[str]:
        lines = [f"Transfer score: {self.score:.4f}"]
        for axis, acc in self.per_axis.items():
            w = self.weights.get(axis, 1.0)
            lines.append(f"  {axis:30s} acc={acc:.4f}  (weight={w:.2f})")
        return lines


def compute_transfer_score(
    per_axis_accuracy: dict[str, float],
    weights: dict[str, float] | None = None,
) -> TransferScore:
    """Weighted mean of held-out accuracies.

    Args:
        per_axis_accuracy: e.g. {"cross_lingual": 0.81, "temporal": 0.75, ...}
        weights: optional per-axis weights. Defaults to equal weights.

    Returns:
        TransferScore.
    """
    if not per_axis_accuracy:
        return TransferScore(score=0.0, per_axis={}, weights={})
    if weights is None:
        weights = {k: 1.0 for k in per_axis_accuracy}

    total_w = sum(weights.get(k, 0.0) for k in per_axis_accuracy)
    if total_w == 0.0:
        return TransferScore(score=0.0, per_axis=per_axis_accuracy, weights=weights)

    score = sum(per_axis_accuracy[k] * weights.get(k, 0.0) for k in per_axis_accuracy) / total_w
    return TransferScore(score=score, per_axis=per_axis_accuracy, weights=weights)


BondIndexResult = TransferScore


def compute_bond_index(
    per_axis_accuracy: dict[str, float],
    weights: dict[str, float] | None = None,
) -> TransferScore:
    """Deprecated name of ``compute_transfer_score``."""
    warnings.warn(
        "compute_bond_index is the old name of compute_transfer_score. The Bond Index is "
        "erisml.ethics.bond_index in erisml-lib.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_transfer_score(per_axis_accuracy, weights)
