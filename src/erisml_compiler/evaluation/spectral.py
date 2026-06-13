"""Spectral summary of a MoralTensorV3.

Computes eigenvalue-based scalars from the tensor:

  - principal_stress, principal_concentration, stress_spread,
    effective_moral_rank: derived from the dimension-axis (mode-k)
    unfolding. These are the canonical "magnitude" scalars and are
    always well-defined regardless of tensor rank.
  - principal_conflict + axis: derived from the centered mode-k
    unfolding (subtract per-dimension mean across all non-k cells).
  - per_axis: one AxisSpectrum per tensor axis (so a rank-6 tensor
    gets six entries, one each for k, n, tau, a, c, s).
  - total_stress: ||T||_F. Axis-independent (Frobenius identity).

The work is small numpy. No torch, no deep learning. For the
default rank-6 ceilings (n=8, tau=4, a=4, c=8, s=16) this runs in
sub-millisecond.

See `docs/plans/release-planning-04-eigenvalue-scalar.md` for the
design rationale and the higher-rank treatment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from erisml_compiler.ir.v3 import MoralTensorV3

# Numerical floor below which we treat an eigenvalue as zero.
_EIG_FLOOR = 1e-12


class AxisSpectrum(BaseModel):
    """Per-axis eigendecomposition of the mode-m unfolding's
    second-moment matrix `T_(m) · T_(m)^T`.
    """

    model_config = ConfigDict(frozen=False)

    axis_name: str
    """One of 'k', 'n', 'tau', 'a', 'c', 's'."""

    eigenvalues: list[float]
    """Descending eigenvalues; length equals the axis dimension."""

    principal_axis: list[float]
    """Unit eigenvector for the largest eigenvalue. Sign-fixed so the
    component with the largest |value| is positive."""

    effective_rank: float
    """Participation ratio: (sum(lambda))^2 / sum(lambda^2). High
    means many modes contribute; low means concentration on one
    direction. Equals 1 for a single non-zero eigenvalue."""

    axis_labels: list[str] | None = None
    """Labels for the axis positions (e.g. dimension names, party
    ids). Copied from MoralTensorV3.axis_labels for interpretation."""


class SpectralSummary(BaseModel):
    """Spectral scalars + per-axis spectra of a MoralTensorV3."""

    model_config = ConfigDict(frozen=False)

    # ----- global magnitude (axis-independent) -----
    total_stress: float
    """||T||_F. The intrinsic Frobenius magnitude."""

    total_stress_squared: float
    """||T||_F**2. Equal to sum of any mode's eigenvalues."""

    # ----- headline scalars (always from the dimension axis) -----
    principal_stress: float
    """Largest eigenvalue of the mode-k unfolding."""

    principal_concentration: float
    """principal_stress / total_stress_squared, in [0, 1]. The
    fraction of total moral magnitude that lives on the principal
    moral-dimension axis."""

    stress_spread: float
    """(lambda_1 - lambda_2) / lambda_1 when lambda_1 > floor else 0.
    Tells you how dominant the principal axis is over the rest."""

    effective_moral_rank: float
    """Participation ratio of the mode-k eigenvalues. Equals 1 when a
    single dimension dominates; equals 9 when all dimensions are
    equally loaded."""

    principal_axis: list[float] = Field(default_factory=list)
    """Length-9 unit eigenvector of the principal moral-dimension axis."""

    # ----- centered counterparts -----
    principal_conflict: float
    """Largest eigenvalue of the centered mode-k unfolding (per-row
    mean subtracted). Measures stakeholder/time/coalition
    disagreement; high means parties pushed in opposite directions
    across the moral dimensions."""

    principal_conflict_axis: list[float] = Field(default_factory=list)
    """Length-9 unit eigenvector of the principal conflict direction."""

    # ----- per-axis spectra -----
    per_axis: list[AxisSpectrum] = Field(default_factory=list)


# ---------- low-level helpers ---------------------------------------------


def _participation_ratio(eigenvalues: np.ndarray) -> float:
    """Effective rank via the inverse participation ratio.

    Standard formula: (sum lambda)^2 / sum(lambda^2). For a single
    non-zero eigenvalue this evaluates to 1; for k equally-large
    eigenvalues it evaluates to k. Returns 0 for an all-zero spectrum.
    """
    s1 = float(np.sum(eigenvalues))
    s2 = float(np.sum(eigenvalues * eigenvalues))
    if s2 <= _EIG_FLOOR:
        return 0.0
    return (s1 * s1) / s2


def _sign_fix(eigvec: np.ndarray) -> np.ndarray:
    """Choose the sign so the component with the largest absolute
    value is positive. Deterministic; matches the geometric-aesthetics
    convention."""
    if eigvec.size == 0:
        return eigvec
    idx = int(np.argmax(np.abs(eigvec)))
    if eigvec[idx] < 0:
        return -eigvec
    return eigvec


def _mode_unfold(arr: np.ndarray, mode: int) -> np.ndarray:
    """Mode-m unfolding: move axis `mode` to the front, then flatten
    all other axes into the column space.

    For `arr.shape == (d_0, d_1, ..., d_{R-1})`, returns
    `arr.shape == (d_mode, prod of the others)`.
    """
    return np.moveaxis(arr, mode, 0).reshape(arr.shape[mode], -1)


def _eig_descending(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a symmetric matrix, descending order.

    Returns (eigenvalues, eigenvector_matrix) where columns of the
    matrix are unit eigenvectors aligned with the sorted eigenvalues.
    """
    # `eigh` is the symmetric/Hermitian eigensolver; deterministic and
    # numerically stable for the second-moment matrices we feed it.
    vals, vecs = np.linalg.eigh(matrix)
    # eigh returns ascending; reverse.
    vals = vals[::-1].copy()
    vecs = vecs[:, ::-1].copy()
    # Clip near-zero negative eigenvalues (numerical noise on a
    # positive-semidefinite matrix).
    vals = np.where(vals < _EIG_FLOOR, 0.0, vals)
    return vals, vecs


# ---------- public API ----------------------------------------------------


def compute_spectral_summary(tensor: MoralTensorV3) -> SpectralSummary:
    """Compute the full spectral summary of a MoralTensorV3.

    Cheap (sub-millisecond at the rank-6 ceiling). Pure numpy.
    """
    arr = np.array(tensor.values, dtype=float)
    if arr.shape != tuple(tensor.shape):
        # Should never happen for a valid tensor; surface clearly if it does.
        raise ValueError(
            f"MoralTensorV3.values shape {arr.shape} != declared shape {tuple(tensor.shape)}"
        )

    rank = tensor.rank
    axes = list(tensor.axis_names)

    # Global magnitude — Frobenius.
    total_stress_squared = float(np.sum(arr * arr))
    total_stress = float(np.sqrt(total_stress_squared))

    # Per-axis mode-m eigendecompositions.
    per_axis: list[AxisSpectrum] = []
    for m in range(rank):
        unfolded = _mode_unfold(arr, m)  # shape (d_m, N)
        gram = unfolded @ unfolded.T  # (d_m, d_m), symmetric PSD
        eigvals, eigvecs = _eig_descending(gram)
        e1 = _sign_fix(eigvecs[:, 0]) if eigvecs.shape[1] > 0 else np.zeros(0)
        axis_name = axes[m]
        labels = tensor.axis_labels.get(axis_name)
        per_axis.append(
            AxisSpectrum(
                axis_name=axis_name,
                eigenvalues=eigvals.tolist(),
                principal_axis=e1.tolist(),
                effective_rank=_participation_ratio(eigvals),
                axis_labels=list(labels) if labels else None,
            )
        )

    # Headline scalars from the dimension axis (mode 0 by construction;
    # MoralTensorV3 validates that axis 0 is always 'k').
    k_spectrum = per_axis[0]
    principal_stress = k_spectrum.eigenvalues[0] if k_spectrum.eigenvalues else 0.0
    second = k_spectrum.eigenvalues[1] if len(k_spectrum.eigenvalues) > 1 else 0.0
    if principal_stress > _EIG_FLOOR:
        stress_spread = (principal_stress - second) / principal_stress
    else:
        stress_spread = 0.0
    if total_stress_squared > _EIG_FLOOR:
        principal_concentration = principal_stress / total_stress_squared
    else:
        principal_concentration = 0.0

    # Centered eigendecomposition: subtract the per-row mean across
    # the flattened non-k cells. This gives the principal direction
    # of stakeholder/time/coalition disagreement.
    k_unfolded = _mode_unfold(arr, 0)
    if k_unfolded.shape[1] > 0:
        row_means = k_unfolded.mean(axis=1, keepdims=True)
        centered = k_unfolded - row_means
        gram_centered = centered @ centered.T
        eig_c_vals, eig_c_vecs = _eig_descending(gram_centered)
        principal_conflict = float(eig_c_vals[0]) if eig_c_vals.size > 0 else 0.0
        if eig_c_vecs.shape[1] > 0:
            principal_conflict_axis = _sign_fix(eig_c_vecs[:, 0]).tolist()
        else:
            principal_conflict_axis = []
    else:
        principal_conflict = 0.0
        principal_conflict_axis = []

    return SpectralSummary(
        total_stress=total_stress,
        total_stress_squared=total_stress_squared,
        principal_stress=float(principal_stress),
        principal_concentration=float(principal_concentration),
        stress_spread=float(stress_spread),
        effective_moral_rank=k_spectrum.effective_rank,
        principal_axis=k_spectrum.principal_axis,
        principal_conflict=principal_conflict,
        principal_conflict_axis=principal_conflict_axis,
        per_axis=per_axis,
    )


def attach_spectral_summary(tensor: MoralTensorV3) -> MoralTensorV3:
    """Compute the spectral summary and attach it to the tensor's
    `metadata["spectral"]` block. Returns the same tensor object for
    chaining. Idempotent — overwrites any existing spectral entry."""
    summary = compute_spectral_summary(tensor)
    tensor.metadata["spectral"] = summary.model_dump()
    return tensor


# ---------- convenience helpers for downstream consumers -----------------


def principal_dimension_label(tensor_or_summary: Any) -> str | None:
    """Return the moral-dimension name with the largest |component|
    in the principal axis. Useful for human-readable CLI output."""
    if isinstance(tensor_or_summary, MoralTensorV3):
        summary_dict = tensor_or_summary.metadata.get("spectral")
        if not summary_dict:
            summary_dict = compute_spectral_summary(tensor_or_summary).model_dump()
    elif isinstance(tensor_or_summary, SpectralSummary):
        summary_dict = tensor_or_summary.model_dump()
    else:
        summary_dict = tensor_or_summary
    axis = summary_dict.get("principal_axis") or []
    if not axis:
        return None
    idx = int(np.argmax(np.abs(np.array(axis))))
    # The dimension labels are on the k-axis spectrum.
    for ax in summary_dict.get("per_axis", []):
        if ax.get("axis_name") == "k" and ax.get("axis_labels"):
            return ax["axis_labels"][idx]
    return None
