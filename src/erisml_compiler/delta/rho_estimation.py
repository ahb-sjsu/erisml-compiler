"""Estimate the per-layer linear map ρ_ℓ(g) from activation pairs.

The BIP equivariance criterion is `h_ℓ(g · x) ≈ ρ_ℓ(g) · h_ℓ(x)`. For
surface-form transforms we hold ρ = I (identity-ρ mode, the v0 shipped
behaviour). For semantic transforms (paraphrase, role-swap, ...) the
identity assumption fails: a paraphrase *should* shift activations
even when moral content is preserved. ρ_ℓ(g) is the best-fit linear
map relating pre- and post-transform activations on a calibration
corpus of pair-wise activation captures.

Two methods supported:

  1. **Orthogonal Procrustes** (default) — R is constrained to be
     orthogonal (R^T R = I). Closed-form via SVD. Best when we
     expect the transform to be a rotation/reflection of activation
     space with no magnitude change (paraphrase, role-swap).

  2. **Unconstrained least squares** — R is any (D, D) real matrix.
     Closed-form via pseudoinverse. More expressive but risks
     overfitting on small N; also can absorb non-meaning-preserving
     drift into magnitude, which we usually don't want.

Inputs: two (N, D) numpy arrays, one per layer, where row i is the
pooled activation for the original (resp. rewritten) text. Output: a
`RhoEstimate` carrying the fitted matrix + residual statistics.

This module is pure numpy — no torch, no scipy dependency required.
Activation captures happen elsewhere (in `monitor/`); ρ estimation
operates on already-captured pooled activations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Method = Literal["procrustes", "lstsq"]


@dataclass(frozen=True)
class RhoEstimate:
    """Per-layer fitted ρ for one transform.

    `R` has shape (D, D). `residual_mean` / `residual_p95` are computed
    on the *fit* set; for honest generalisation numbers, also fit on a
    held-out set and report both (see release-planning-02 milestone ρ-6).

    `R_orthogonal_score` is the Frobenius norm of `R^T R - I`. For an
    orthogonal R this is ~0; for a wildly non-orthogonal fit it is
    large — diagnostic for the `rho_non_orthogonal` failure mode.
    """

    transform_name: str
    layer_index: int
    R: np.ndarray
    family: str
    n_pairs: int
    residual_mean: float
    residual_p95: float
    R_orthogonal_score: float
    method: Method
    corpus_hash: str
    """SHA-256 of the JSON-canonicalised list of (orig, rewritten) pair
    text hashes, so two ρ estimates can be compared for corpus
    identity."""

    metadata: dict = field(default_factory=dict)

    def apply(self, h: np.ndarray) -> np.ndarray:
        """Apply ρ to activations. h has shape (D,) or (N, D)."""
        return h @ self.R.T if h.ndim == 2 else self.R @ h


def fit_rho_procrustes(H: np.ndarray, H_g: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthogonal Procrustes: min_{R: R^T R = I} ||R H^T - H_g^T||_F.

    Returns (R, orthogonality_score). R is (D, D). The scaling factor
    that some Procrustes variants emit is NOT applied here — we want
    pure rotation/reflection.

    Implementation: cross-covariance M = H_g^T @ H, SVD M = U S V^T,
    then R = U V^T.
    """
    if H.ndim != 2 or H_g.ndim != 2 or H.shape != H_g.shape:
        raise ValueError(f"H and H_g must be same-shape 2D arrays, got {H.shape} vs {H_g.shape}")
    M = H_g.T @ H  # (D, D)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    eye = np.eye(R.shape[0])
    ortho_score = float(np.linalg.norm(R.T @ R - eye, ord="fro"))
    return R, ortho_score


def fit_rho_lstsq(H: np.ndarray, H_g: np.ndarray) -> tuple[np.ndarray, float]:
    """Unconstrained linear fit via pseudoinverse: R = H_g^T (H^T)^+.

    Returns (R, orthogonality_score). Score is informational —
    unconstrained R can be very non-orthogonal.
    """
    if H.ndim != 2 or H_g.ndim != 2 or H.shape != H_g.shape:
        raise ValueError(f"H and H_g must be same-shape 2D arrays, got {H.shape} vs {H_g.shape}")
    # We want R such that R H[i] ≈ H_g[i] for each row.
    # Equivalently H R^T ≈ H_g, so R^T = pinv(H) @ H_g.
    R_T = np.linalg.pinv(H) @ H_g  # (D, D)
    R = R_T.T
    eye = np.eye(R.shape[0])
    ortho_score = float(np.linalg.norm(R.T @ R - eye, ord="fro"))
    return R, ortho_score


def compute_residuals(R: np.ndarray, H: np.ndarray, H_g: np.ndarray) -> np.ndarray:
    """Per-pair normalised residual = ||R h - h_g|| / ||h_g||.

    Returns a (N,) array of residuals. Pairs where ||h_g|| ≈ 0 are
    reported as 0 (degenerate; no signal to compare to).
    """
    predicted = H @ R.T
    diff = predicted - H_g
    diff_norms = np.linalg.norm(diff, axis=1)
    target_norms = np.linalg.norm(H_g, axis=1)
    # Avoid division by zero.
    safe = np.where(target_norms > 1e-12, target_norms, 1.0)
    res = diff_norms / safe
    res = np.where(target_norms > 1e-12, res, 0.0)
    return res


def _hash_corpus(pair_ids: list[str]) -> str:
    canonical = json.dumps(sorted(pair_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fit_rho(
    H: np.ndarray,
    H_g: np.ndarray,
    *,
    transform_name: str,
    layer_index: int,
    family: str = "paraphrase",
    method: Method = "procrustes",
    pair_ids: list[str] | None = None,
    metadata: dict | None = None,
) -> RhoEstimate:
    """Top-level entry point: fit ρ + compute residual stats + package.

    H : (N, D) original-text pooled activations at this layer.
    H_g: (N, D) rewritten-text pooled activations at the same layer.

    `pair_ids` is an optional list of stable per-pair identifiers used
    to compute `corpus_hash`. When None, the index 0..N-1 is hashed —
    callers should pass real ids if they intend to compare ρ estimates
    across runs.
    """
    if method == "procrustes":
        R, ortho = fit_rho_procrustes(H, H_g)
    elif method == "lstsq":
        R, ortho = fit_rho_lstsq(H, H_g)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'procrustes' or 'lstsq'")

    res = compute_residuals(R, H, H_g)
    return RhoEstimate(
        transform_name=transform_name,
        layer_index=layer_index,
        R=R,
        family=family,
        n_pairs=int(H.shape[0]),
        residual_mean=float(res.mean()),
        residual_p95=float(np.quantile(res, 0.95)) if res.size else 0.0,
        R_orthogonal_score=ortho,
        method=method,
        corpus_hash=_hash_corpus(
            pair_ids if pair_ids is not None else [str(i) for i in range(H.shape[0])]
        ),
        metadata=metadata or {},
    )


def equivariance_residual(
    rho: RhoEstimate,
    h: np.ndarray,
    h_g: np.ndarray,
) -> float:
    """Single-instance ρ-corrected residual for a new (h, h_g) pair.

    This is what `delta.equivariance.check_equivariance` will compare
    against `rho.residual_p95` (or a caller-supplied threshold) when
    operating in real-ρ mode.
    """
    if h.shape != h_g.shape or h.ndim != 1:
        raise ValueError(f"h and h_g must be 1D same-shape; got {h.shape}, {h_g.shape}")
    predicted = rho.R @ h
    diff = float(np.linalg.norm(predicted - h_g))
    target = float(np.linalg.norm(h_g))
    return diff / target if target > 1e-12 else 0.0
