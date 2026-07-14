"""Polar (PolarQuant) encoding of the moral state, for Phase-5 H2/H3.

Motivation. The calibrated activation moral vector separates benign from
morally-loaded by *magnitude* (radius) — that carries H1/H5 — but magnitude does
NOT separate contested dilemmas from clear-cut easy-moral cases (they are equally
loaded). The discriminating structure is *angular*: an easy-moral case points
along the consensus ("all dimensions agree — clearly good or clearly bad") axis,
while a dilemma points into the conflicted region of the moral sphere.

This is the PolarQuant decomposition (Han et al., arXiv:2502.02617; TurboQuant
`benchmarks/benchmark_polar.py`): split a vector into radius (magnitude) + angles
(direction) and treat them separately, because the *direction* carries the
faithful geometry (cf. the GDT thesis: angle carries the geometry, radius is
geometry-free). `polar_encode`/`polar_decode`/`polar_quantize` below are copied
verbatim from turboquant-pro for provenance and bit-exact agreement.

Moral features:
  - loading  = radius = ||v||                      (H1: benign vs loaded)
  - conflict = polar angle from the consensus pole  (H3/H2: aligned vs contested)
             = arccos(|<v, u>| / (||v|| ||u||)),  u = consensus/"good" axis.
    All 7 calibrated dims share the convention +value == morally good, so the
    consensus axis is 1/sqrt(7). conflict small => aligned (easy-moral),
    conflict large => orthogonal to consensus (dilemma).
"""

from __future__ import annotations

import numpy as np

# ---- exact PolarQuant transform (vendored from turboquant-pro benchmark_polar.py;
#      Han et al. arXiv:2502.02617, recursive polar transform Def. 1) ----


def polar_encode(X):
    """Cartesian -> (radius, angles). X: (N, D), D a power of two."""
    angles = []
    v = X
    while v.shape[1] > 1:
        a, b = v[:, 0::2], v[:, 1::2]
        angles.append(np.arctan2(b, a))
        v = np.sqrt(a * a + b * b)
    return v[:, 0], angles


def polar_decode(radius, angles):
    """(radius, angles) -> Cartesian (N, D)."""
    v = radius[:, None]
    for th in reversed(angles):
        a, b = v * np.cos(th), v * np.sin(th)
        nv = np.empty((v.shape[0], 2 * v.shape[1]), dtype=np.float32)
        nv[:, 0::2], nv[:, 1::2] = a, b
        v = nv
    return v


def _quant_uniform(x, lo, hi, bits):
    levels = 2**bits
    step = (hi - lo) / levels
    q = np.clip(np.floor((x - lo) / step), 0, levels - 1)
    return (lo + (q + 0.5) * step).astype(np.float32)


def polar_quantize(Xr, abits, rbits=8, adaptive=True):
    """Quantize rotated vectors via the polar transform (returns reconstruction).
    Verbatim from turboquant-pro; used to show the moral signal survives
    low-bit angle quantization (the 'quant' in PolarQuant)."""
    radius, angles = polar_encode(Xr)
    lr = np.log(np.maximum(radius, 1e-20))
    radius_q = np.exp(_quant_uniform(lr, lr.min(), lr.max(), rbits))
    angles_q = []
    for th in angles:
        if adaptive:
            lv = np.quantile(th, (np.arange(2**abits) + 0.5) / 2**abits).astype(np.float32)
            bnd = ((lv[:-1] + lv[1:]) / 2).astype(np.float32)
            angles_q.append(lv[np.searchsorted(bnd, th)].astype(np.float32))
        else:
            angles_q.append(_quant_uniform(th, -np.pi, np.pi, abits))
    return polar_decode(radius_q.astype(np.float32), angles_q)


# ---- moral polar features ----


def _pad_pow2(V):
    """Right-pad the feature dim to the next power of two with zeros
    (arctan2(0, r)=0, so zero-padding is angle-neutral)."""
    n, d = V.shape
    p = 1
    while p < d:
        p *= 2
    if p == d:
        return V
    out = np.zeros((n, p), dtype=np.float32)
    out[:, :d] = V
    return out


def moral_polar_features(V: np.ndarray, consensus: np.ndarray | None = None) -> dict:
    """V: (N, D) calibrated moral vectors (columns share +==good). Returns
    loading (radius), conflict (polar angle from consensus pole, radians),
    alignment (|cos| to consensus), and the PolarQuant (radius, angles) of the
    padded vector for downstream quantization/analysis."""
    V = np.asarray(V, dtype=np.float32)
    n, d = V.shape
    u = np.ones(d, dtype=np.float32) if consensus is None else np.asarray(consensus, np.float32)
    u = u / np.maximum(np.linalg.norm(u), 1e-12)

    loading = np.linalg.norm(V, axis=1)
    proj = np.abs(V @ u)
    alignment = proj / np.maximum(loading, 1e-9)
    conflict = np.arccos(np.clip(alignment, 0.0, 1.0))  # 0=aligned, pi/2=orthogonal

    radius, angles = polar_encode(_pad_pow2(V))
    return {
        "loading": loading,
        "conflict": conflict,
        "alignment": alignment,
        "radius": radius,
        "angles": angles,
    }
