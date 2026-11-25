"""Deprecated: this module's quantity is the transfer score (``transfer_score.py``).

The Bond Index is the mean deviation of an evaluation under declared re-descriptions, over a
tolerance; it lives in erisml-lib (``erisml.ethics.bond_index``).
"""

from erisml_compiler.calibration.transfer_score import (
    BondIndexResult,
    TransferScore,
    compute_bond_index,
    compute_transfer_score,
)

__all__ = ["BondIndexResult", "TransferScore", "compute_bond_index", "compute_transfer_score"]
