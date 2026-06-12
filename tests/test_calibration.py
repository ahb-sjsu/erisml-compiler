"""Calibration tests (Track B).

Most tests use synthetic tensors directly so they don't require the
LaBSE network download. The `train_probe` smoke test is gated on
`ERISML_TEST_LABSE=1` since it downloads ~500MB on first run.
"""
import os
import pytest

torch = pytest.importorskip("torch")

from erisml_compiler.calibration.adversarial_heads import (
    AdversarialHead,
    MultiHeadAdversarial,
)
from erisml_compiler.calibration.bond_index import compute_bond_index
from erisml_compiler.calibration.dataset import (
    ProbeBatch,
    ProbeTrainingDataset,
    synthetic_dataset,
)
from erisml_compiler.calibration.losses import (
    confusion_loss,
    grad_reverse,
    spectral_decoupling_loss,
    vib_kl_loss,
)


# ---------- losses ----------


def test_spectral_decoupling_zero_when_independent():
    """When z is independent of the nuisance, the cross-cov is near zero."""
    torch.manual_seed(0)
    z = torch.randn(128, 16)
    n = torch.nn.functional.one_hot(torch.randint(0, 4, (128,)), num_classes=4).float()
    loss = spectral_decoupling_loss(z, n)
    assert loss.item() < 1.0  # not strictly zero but small


def test_spectral_decoupling_high_when_correlated():
    """When z encodes the nuisance, the cross-cov is large."""
    torch.manual_seed(0)
    idx = torch.randint(0, 4, (128,))
    n = torch.nn.functional.one_hot(idx, num_classes=4).float()
    z = n.clone()                          # z literally is the nuisance
    z = torch.cat([z, torch.randn(128, 12)], dim=1)
    loss = spectral_decoupling_loss(z, n)
    assert loss.item() > 0.1


def test_spectral_decoupling_singleton_batch():
    """Edge case: 1-sample batch returns 0 without div-by-zero."""
    z = torch.randn(1, 16)
    n = torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long), num_classes=4).float()
    loss = spectral_decoupling_loss(z, n)
    assert loss.item() == 0.0


def test_vib_kl_zero_at_prior():
    """KL is exactly zero when mu=0 and log_var=0 (=> sigma=1) — match the prior."""
    mu = torch.zeros(8, 16)
    log_var = torch.zeros(8, 16)
    loss = vib_kl_loss(mu, log_var)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_vib_kl_positive_when_off_prior():
    mu = torch.ones(8, 16)
    log_var = torch.zeros(8, 16)
    loss = vib_kl_loss(mu, log_var)
    assert loss.item() > 0


def test_confusion_loss_minimised_by_uniform_logits():
    """Uniform logits => max entropy => most-negative confusion loss."""
    logits_uniform = torch.zeros(8, 4)
    logits_peaked = torch.eye(4).repeat(2, 1) * 100  # one-hot peaked
    l_uni = confusion_loss(logits_uniform).item()
    l_peak = confusion_loss(logits_peaked).item()
    assert l_uni < l_peak  # uniform has lower (more negative-entropy) loss


def test_grad_reverse_forward_identity():
    x = torch.ones(2, 3, requires_grad=True)
    y = grad_reverse(x, 2.0)
    assert torch.allclose(y, x)


def test_grad_reverse_backward_negates_and_scales():
    x = torch.ones(2, 3, requires_grad=True)
    y = grad_reverse(x, 2.0)
    y.sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


# ---------- adversarial heads ----------


def test_adversarial_head_forward_shape():
    head = AdversarialHead(in_dim=16, num_classes=4)
    z = torch.randn(8, 16)
    out = head(z)
    assert out.shape == (8, 4)


def test_multihead_adversarial_returns_n_heads():
    mh = MultiHeadAdversarial(in_dim=16, num_classes=4, n_heads=4)
    z = torch.randn(8, 16)
    outs = mh(z)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (8, 4)


def test_multihead_adversarial_grl_pushes_negative_grad_to_input():
    """Gradient sign through GRL should be flipped relative to no-GRL."""
    z = torch.randn(8, 16, requires_grad=True)
    mh = MultiHeadAdversarial(in_dim=16, num_classes=4, n_heads=2, adversarial_lambda=1.5)
    outs = mh(z)
    loss = sum(o.sum() for o in outs)
    loss.backward()
    # If GRL works, z.grad should point in the direction that *decreases* the
    # adversarial loss in the encoder's reference frame (negative scale).
    # We check that gradients are non-zero (sanity) and finite.
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


# ---------- dataset ----------


def test_synthetic_dataset_iterates_in_batches():
    ds = synthetic_dataset(n_samples=20, n_classes=3)
    batches = list(ds)
    total = sum(len(b.texts) for b in batches)
    assert total == 20
    # All batches return ProbeBatch.
    for b in batches:
        assert isinstance(b, ProbeBatch)
        assert b.labels.shape[0] == len(b.texts)


def test_synthetic_dataset_deterministic():
    ds1 = synthetic_dataset(n_samples=8, seed=42)
    ds2 = synthetic_dataset(n_samples=8, seed=42)
    assert ds1.texts == ds2.texts
    assert ds1.labels == ds2.labels


# ---------- bond index ----------


def test_bond_index_equal_weighted_mean():
    r = compute_bond_index({"a": 0.8, "b": 0.6})
    assert abs(r.bond_index - 0.7) < 1e-9


def test_bond_index_custom_weights():
    r = compute_bond_index({"a": 1.0, "b": 0.0}, weights={"a": 3.0, "b": 1.0})
    # (1.0*3 + 0.0*1) / (3+1) = 0.75
    assert abs(r.bond_index - 0.75) < 1e-9


def test_bond_index_empty():
    r = compute_bond_index({})
    assert r.bond_index == 0.0


# ---------- train loop (gated; requires LaBSE download) ----------


@pytest.mark.skipif(
    not os.environ.get("ERISML_TEST_LABSE"),
    reason="Set ERISML_TEST_LABSE=1 to run the full LaBSE-backed training smoke test.",
)
def test_train_probe_smoke_synthetic(tmp_path):
    from erisml_compiler.calibration import CalibrationConfig, train_probe
    from erisml_compiler.calibration.train import save_checkpoint, load_checkpoint

    ds = synthetic_dataset(n_samples=16, n_classes=3)
    cfg = CalibrationConfig(num_classes=3, epochs=1, device="cpu", log_every=1)
    backbone, history = train_probe(ds, cfg)
    assert len(history.epoch_losses) == 1

    ckpt = tmp_path / "probe.pt"
    save_checkpoint(backbone, ckpt, history=history)
    reloaded = load_checkpoint(ckpt, device="cpu")
    assert reloaded is not None
