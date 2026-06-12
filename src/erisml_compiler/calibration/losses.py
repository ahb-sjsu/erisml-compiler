"""Loss functions adopted from sqnd-probe v10.16.9.

These are the invariance-enforcing losses that make the probe's bond
extraction independent of language and period. The compiler reuses them
for its probe-extractor training.

Mathematical references:
  - Spectral decoupling: Cross-covariance between embedding z and a
    nuisance one-hot vector y_n. Minimising ||Cov(z, y_n)||_F^2 forces
    the embedding to be linearly uninformative about the nuisance.
  - VIB: KL(q(z|x) || prior(z)). Forces compressed stochastic
    representations.
  - Confusion loss: -H(p(y_n | z)). Drives the nuisance posterior
    toward uniform, i.e., no classifier can predict y_n from z.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def spectral_decoupling_loss(
    z: torch.Tensor,
    nuisance_onehot: torch.Tensor,
) -> torch.Tensor:
    """Frobenius-norm cross-covariance between z and the nuisance one-hot.

    Args:
        z: (B, D) embedding batch.
        nuisance_onehot: (B, K) one-hot encoded nuisance labels.

    Returns:
        scalar loss = sum over (d, k) of Cov(z_d, nuisance_k)^2.

    Minimising this loss removes any linear signal about the nuisance from
    z, in expectation.
    """
    z_c = z - z.mean(dim=0, keepdim=True)
    n_c = nuisance_onehot - nuisance_onehot.mean(dim=0, keepdim=True)
    n_samples = z.shape[0]
    if n_samples <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (z_c.t() @ n_c) / (n_samples - 1)
    return (cov ** 2).sum()


def vib_kl_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """KL divergence from N(mu, diag(exp(log_var))) to N(0, I).

    Args:
        mu: (B, D) variational mean.
        log_var: (B, D) variational log-variance.

    Returns:
        mean-per-sample KL.
    """
    kl = 0.5 * (mu.pow(2) + log_var.exp() - 1.0 - log_var).sum(dim=1)
    return kl.mean()


def confusion_loss(
    nuisance_logits: torch.Tensor,
) -> torch.Tensor:
    """Negative entropy of the nuisance posterior.

    Args:
        nuisance_logits: (B, K) logits from an adversarial nuisance head.

    Returns:
        mean negative entropy. Minimising this drives the posterior toward
        uniform, i.e., the encoder makes the nuisance unpredictable from z.
    """
    log_probs = F.log_softmax(nuisance_logits, dim=-1)
    probs = log_probs.exp()
    neg_entropy = (probs * log_probs).sum(dim=-1)  # H = -sum p log p; neg = sum p log p
    # We MAXIMIZE entropy => minimise neg-entropy.
    return neg_entropy.mean()


def gradient_reversal_factor(adversarial_lambda: float) -> torch.Tensor:
    """Helper for the GRL trick: the adversarial head's gradients flow back
    into the encoder with sign flipped and scaled by lambda. PyTorch's
    standard pattern is a custom autograd Function; we expose the scalar
    here so callers can multiply or use the explicit GRL module.
    """
    return torch.tensor(adversarial_lambda)


class GradientReversalFn(torch.autograd.Function):
    """Standard gradient reversal layer (Ganin & Lempitsky 2015).

    Forward pass is identity; backward pass scales by -lambda.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalFn.apply(x, lam)
