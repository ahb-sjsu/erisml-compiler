"""Tests for ρ estimation and the transforms registry."""

from __future__ import annotations

import numpy as np
import pytest

from erisml_compiler.delta.failure_modes import FailureMode
from erisml_compiler.delta.rho_estimation import (
    RhoEstimate,
    compute_residuals,
    equivariance_residual,
    fit_rho,
    fit_rho_lstsq,
    fit_rho_procrustes,
)
from erisml_compiler.delta.transforms import (
    SURFACE_TRANSFORMS,
    Transform,
    TransformFamily,
    TransformRegistry,
    default_registry,
)

# ----------------------------------------------------------- registry


def test_default_registry_has_surface_transforms() -> None:
    reg = default_registry()
    for t in SURFACE_TRANSFORMS:
        assert t.name in reg
        assert reg.get(t.name).family == TransformFamily.SURFACE


def test_default_registry_is_singleton() -> None:
    assert default_registry() is default_registry()


def test_registry_filter_by_family() -> None:
    reg = TransformRegistry()
    for t in SURFACE_TRANSFORMS:
        reg.register(t)
    paraphrase_t = Transform(
        name="lexical_synonym_swap",
        family=TransformFamily.PARAPHRASE,
        fn=lambda s: s.replace("good", "fine"),
        expected_rho_class="orthogonal",
    )
    reg.register(paraphrase_t)

    surfs = reg.filter(family=TransformFamily.SURFACE)
    paras = reg.filter(family=TransformFamily.PARAPHRASE)
    assert len(surfs) == len(SURFACE_TRANSFORMS)
    assert len(paras) == 1
    assert paras[0].name == "lexical_synonym_swap"


def test_registry_double_register_raises() -> None:
    reg = TransformRegistry()
    reg.register(SURFACE_TRANSFORMS[0])
    with pytest.raises(ValueError, match="already registered"):
        reg.register(SURFACE_TRANSFORMS[0])


def test_registry_get_missing_raises() -> None:
    with pytest.raises(KeyError):
        TransformRegistry().get("nonexistent")


def test_transform_apply_with_and_without_params() -> None:
    def fn(text: str, params: dict | None = None) -> str:
        prefix = (params or {}).get("prefix", "")
        return prefix + text

    t = Transform(name="prefix", family=TransformFamily.SURFACE, fn=fn)
    assert t.apply("hello") == "hello"
    assert t.apply("hello", {"prefix": ">> "}) == ">> hello"


# ----------------------------------------------------------- procrustes math


def _random_orthogonal(D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    return Q


def test_procrustes_recovers_known_rotation() -> None:
    D = 8
    N = 200
    rng = np.random.default_rng(42)
    R_true = _random_orthogonal(D, seed=7)
    H = rng.standard_normal((N, D))
    H_g = H @ R_true.T

    R_hat, ortho = fit_rho_procrustes(H, H_g)
    assert ortho == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(R_hat, R_true, atol=1e-6)


def test_procrustes_residuals_zero_on_perfect_data() -> None:
    D = 6
    N = 100
    rng = np.random.default_rng(1)
    R_true = _random_orthogonal(D, seed=2)
    H = rng.standard_normal((N, D))
    H_g = H @ R_true.T

    R_hat, _ = fit_rho_procrustes(H, H_g)
    res = compute_residuals(R_hat, H, H_g)
    assert res.max() < 1e-6


def test_procrustes_residuals_track_noise_level() -> None:
    D = 6
    N = 200
    rng = np.random.default_rng(3)
    R_true = _random_orthogonal(D, seed=11)
    H = rng.standard_normal((N, D))
    H_g_clean = H @ R_true.T

    for noise in (0.01, 0.05, 0.2):
        H_g = H_g_clean + rng.standard_normal((N, D)) * noise
        R_hat, _ = fit_rho_procrustes(H, H_g)
        res = compute_residuals(R_hat, H, H_g)
        # Monotonic in noise level
        assert res.mean() > 0
        if noise > 0.01:
            assert res.mean() > 0.005


def test_procrustes_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same-shape 2D"):
        fit_rho_procrustes(np.zeros((10, 5)), np.zeros((10, 6)))


# ----------------------------------------------------------- lstsq


def test_lstsq_recovers_general_linear_map() -> None:
    D = 5
    N = 100
    rng = np.random.default_rng(8)
    # A general (non-orthogonal) linear map
    R_true = rng.standard_normal((D, D)) * 0.5 + np.eye(D)
    H = rng.standard_normal((N, D))
    H_g = H @ R_true.T

    R_hat, ortho = fit_rho_lstsq(H, H_g)
    assert np.allclose(R_hat, R_true, atol=1e-6)
    # Unconstrained -> not orthogonal in general
    assert ortho > 0.0


def test_lstsq_residuals_zero_on_perfect_data() -> None:
    D = 4
    N = 60
    rng = np.random.default_rng(9)
    R_true = rng.standard_normal((D, D)) * 0.5 + np.eye(D)
    H = rng.standard_normal((N, D))
    H_g = H @ R_true.T

    R_hat, _ = fit_rho_lstsq(H, H_g)
    res = compute_residuals(R_hat, H, H_g)
    assert res.max() < 1e-6


# ----------------------------------------------------------- top-level fit_rho


def test_fit_rho_packages_estimate() -> None:
    D, N = 6, 80
    rng = np.random.default_rng(4)
    R_true = _random_orthogonal(D, seed=5)
    H = rng.standard_normal((N, D))
    H_g = H @ R_true.T

    est = fit_rho(
        H,
        H_g,
        transform_name="paraphrase_v0",
        layer_index=12,
        family="paraphrase",
        method="procrustes",
        pair_ids=[f"p{i}" for i in range(N)],
    )
    assert isinstance(est, RhoEstimate)
    assert est.transform_name == "paraphrase_v0"
    assert est.layer_index == 12
    assert est.n_pairs == N
    assert est.method == "procrustes"
    assert est.residual_mean < 1e-6
    assert est.R_orthogonal_score < 1e-6
    assert len(est.corpus_hash) == 64


def test_fit_rho_corpus_hash_order_invariant() -> None:
    D, N = 4, 30
    rng = np.random.default_rng(6)
    H = rng.standard_normal((N, D))
    H_g = H @ _random_orthogonal(D).T
    ids = [f"p{i}" for i in range(N)]
    est1 = fit_rho(H, H_g, transform_name="t", layer_index=0, pair_ids=ids)
    est2 = fit_rho(H, H_g, transform_name="t", layer_index=0, pair_ids=list(reversed(ids)))
    assert est1.corpus_hash == est2.corpus_hash


def test_fit_rho_unknown_method_raises() -> None:
    H = np.zeros((5, 3))
    H_g = np.zeros((5, 3))
    with pytest.raises(ValueError, match="Unknown method"):
        fit_rho(H, H_g, transform_name="t", layer_index=0, method="schmocrustes")  # type: ignore[arg-type]


# ----------------------------------------------------------- equivariance_residual


def test_equivariance_residual_zero_on_perfect_pair() -> None:
    D = 5
    rng = np.random.default_rng(10)
    R_true = _random_orthogonal(D, seed=12)
    H = rng.standard_normal((30, D))
    H_g = H @ R_true.T
    est = fit_rho(H, H_g, transform_name="t", layer_index=0)

    h_new = rng.standard_normal(D)
    h_g_new = R_true @ h_new
    r = equivariance_residual(est, h_new, h_g_new)
    assert r < 1e-6


def test_equivariance_residual_nonzero_under_drift() -> None:
    D = 5
    rng = np.random.default_rng(11)
    R_true = _random_orthogonal(D, seed=13)
    H = rng.standard_normal((30, D))
    H_g = H @ R_true.T
    est = fit_rho(H, H_g, transform_name="t", layer_index=0)

    h_new = rng.standard_normal(D)
    h_g_drifted = R_true @ h_new + rng.standard_normal(D) * 0.5
    r = equivariance_residual(est, h_new, h_g_drifted)
    assert r > 0.05


# ----------------------------------------------------------- failure_modes hookup


def test_rho_non_orthogonal_enum_value_exists() -> None:
    assert FailureMode.RHO_NON_ORTHOGONAL.value == "rho_non_orthogonal"
