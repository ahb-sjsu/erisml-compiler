"""Activation-probe calibration must (a) recover moral dimensions that are
linearly present in the activations, (b) correctly EXCLUDE dimensions that are
pure noise (so the monitor never treats random probes as calibrated), and
(c) round-trip into a calibrated ActivationProbe whose provenance flips the
Phase-5 C3 gate. Run: python -m pytest tests/test_activation_calibration.py -q
"""
from __future__ import annotations

import torch

from erisml_compiler.calibration.activation_calibration import (
    ActivationCalibConfig,
    calibration_table_rows,
    load_calibrated_probe,
    save_layer_checkpoint,
    train_layer_probe,
)
from erisml_compiler.ir.schemas import MORAL_DIMENSIONS


def _make_data(n=800, d=64, n_signal=6, seed=0):
    """First `n_signal` moral dims are a linear (tanh) function of X; the rest
    are independent noise. A faithful calibrator recovers the signal dims and
    flags the noise dims as uncalibrated."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    W = torch.randn(d, n_signal, generator=g)
    Y = torch.zeros(n, len(MORAL_DIMENSIONS))
    Y[:, :n_signal] = torch.tanh(1.5 * (X @ W) / d**0.5)
    Y[:, n_signal:] = torch.tanh(torch.randn(n, len(MORAL_DIMENSIONS) - n_signal, generator=g))
    return X, Y, n_signal


def test_recovers_signal_excludes_noise():
    X, Y, n_signal = _make_data()
    cfg = ActivationCalibConfig(epochs=60, seed=1, log_every=0)
    _, res = train_layer_probe(X, Y, cfg, layer_index=12)

    dims = list(MORAL_DIMENSIONS)
    signal_acc = [res.per_dim_signacc[dims[i]] for i in range(n_signal)]
    noise_acc = [res.per_dim_signacc[dims[i]] for i in range(n_signal, len(dims))]

    # Signal dims are strongly recovered and marked included.
    assert min(signal_acc) >= 0.80, f"signal dims under-recovered: {signal_acc}"
    assert all(res.included[dims[i]] for i in range(n_signal)), "signal dims not included"
    # Noise dims sit near chance and are excluded — the whole point of the C3 gate.
    assert max(noise_acc) < 0.70, f"a noise dim was (wrongly) calibratable: {noise_acc}"
    assert not any(res.included[dims[i]] for i in range(n_signal, len(dims))), \
        "a noise dim was wrongly included"


def test_checkpoint_roundtrip_flips_calibration_gate(tmp_path):
    X, Y, _ = _make_data()
    cfg = ActivationCalibConfig(epochs=40, seed=2, log_every=0)
    head, res = train_layer_probe(X, Y, cfg, layer_index=12)

    ckpt = save_layer_checkpoint(head, tmp_path / "layer12.pt", res,
                                 corpus_fingerprint={"n_samples": X.shape[0], "d": X.shape[1]},
                                 model_id="Qwen/Qwen2.5-7B-Instruct", teacher="unit-synth")
    probe = load_calibrated_probe(ckpt, hidden_dim=X.shape[1])

    # The gate the monitor branches on must now read True, with the per-dim
    # held-out accuracy recorded for the auditor.
    assert probe.provenance.is_calibrated is True
    assert probe.provenance.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert probe.provenance.probe_checkpoint_hash
    assert any(k.startswith("signacc.") for k in probe.provenance.calibration_metrics)

    # Loaded weights reproduce the trained head's predictions (tanh of logits).
    from erisml_compiler.monitor.base import LayerActivation
    x0 = X[0]
    la = LayerActivation(layer_index=12, layer_name="t", hidden=x0.unsqueeze(0), pooled=x0)
    got = probe.probe_layer(la).logits
    with torch.no_grad():
        want = head(x0.unsqueeze(0)).squeeze(0)
    assert torch.allclose(got, want, atol=1e-5)


def test_section6_table_marks_included():
    X, Y, n_signal = _make_data()
    cfg = ActivationCalibConfig(epochs=60, seed=3, log_every=0)
    _, res = train_layer_probe(X, Y, cfg, layer_index=8)
    rows = calibration_table_rows([res], teacher="unit-synth")
    dims = list(MORAL_DIMENSIONS)
    included = {r["dimension"]: r["included"] for r in rows}
    assert all(included[dims[i]] for i in range(n_signal))
    assert not any(included[dims[i]] for i in range(n_signal, len(dims)))
    # Every dimension is represented exactly once.
    assert {r["dimension"] for r in rows} == set(MORAL_DIMENSIONS)


if __name__ == "__main__":
    test_recovers_signal_excludes_noise()
    print("PASS recovers_signal_excludes_noise")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as t:
        test_checkpoint_roundtrip_flips_calibration_gate(Path(t))
    print("PASS checkpoint_roundtrip_flips_calibration_gate")
    test_section6_table_marks_included()
    print("PASS section6_table_marks_included")
    print("all activation-calibration tests passed")
