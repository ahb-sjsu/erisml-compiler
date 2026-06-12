"""Training loop for the probe extractor.

Composes:
    - frozen LaBSE backbone
    - optional VIB layer
    - main probe head (classification)
    - multi-head adversarial(s) for language and period
    - spectral decoupling loss
    - confusion loss
    - VIB KL loss

Total loss:
    L = L_main
      + lambda_vib * L_VIB
      + lambda_spec * (L_spec_lang + L_spec_period)
      + lambda_conf * (L_conf_lang + L_conf_period)
      + (adversarial heads contribute through reversed gradients)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from erisml_compiler.calibration.adversarial_heads import MultiHeadAdversarial
from erisml_compiler.calibration.dataset import ProbeBatch, ProbeTrainingDataset
from erisml_compiler.calibration.losses import (
    confusion_loss,
    spectral_decoupling_loss,
    vib_kl_loss,
)
from erisml_compiler.calibration.probe_head import ProbeBackbone


@dataclass
class CalibrationConfig:
    num_classes: int = 3
    num_languages: int = 4
    num_periods: int = 3
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    use_vib: bool = True
    vib_bottleneck: int = 256
    lambda_vib: float = 0.01
    lambda_spec: float = 0.5
    lambda_conf: float = 0.5
    lambda_adv: float = 1.5
    n_adv_heads: int = 4
    labse_model: str = "sentence-transformers/LaBSE"
    device: str = "cpu"
    log_every: int = 1


@dataclass
class TrainingHistory:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_main_accs: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _onehot(idx: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(idx, num_classes=n).float()


def train_probe(
    dataset: ProbeTrainingDataset,
    config: CalibrationConfig,
) -> tuple[ProbeBackbone, TrainingHistory]:
    """Train a probe with the full sqnd-probe-style loss composition.

    Returns the trained backbone and a TrainingHistory.
    """
    backbone = ProbeBackbone(
        num_classes=config.num_classes,
        labse_model=config.labse_model,
        use_vib=config.use_vib,
        vib_bottleneck=config.vib_bottleneck,
        device=config.device,
    )

    # Adversarial heads operate on the same representation the main head sees.
    if config.use_vib:
        adv_in_dim = config.vib_bottleneck
    else:
        adv_in_dim = backbone.backbone_dim

    adv_lang = MultiHeadAdversarial(
        in_dim=adv_in_dim,
        num_classes=config.num_languages,
        n_heads=config.n_adv_heads,
        adversarial_lambda=config.lambda_adv,
    ).to(config.device)
    adv_period = MultiHeadAdversarial(
        in_dim=adv_in_dim,
        num_classes=config.num_periods,
        n_heads=config.n_adv_heads,
        adversarial_lambda=config.lambda_adv,
    ).to(config.device)

    backbone.to(config.device)

    params = list(backbone.head.parameters())
    if config.use_vib:
        params += list(backbone.vib.parameters())
    params += list(adv_lang.parameters())
    params += list(adv_period.parameters())

    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    history = TrainingHistory()
    backbone.train()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        for batch in dataset:
            batch: ProbeBatch = batch
            # 1. Encode (no gradient through backbone).
            embeddings = backbone.encode_texts(batch.texts).to(config.device)
            labels = batch.labels.to(config.device)
            lang = batch.language_idx.to(config.device)
            period = batch.period_idx.to(config.device)

            # 2. Forward through VIB + head.
            logits, mu, log_var, z = backbone(embeddings)
            main_loss = F.cross_entropy(logits, labels)

            # 3. VIB KL.
            if config.use_vib and mu is not None and log_var is not None:
                vib_l = vib_kl_loss(mu, log_var)
            else:
                vib_l = torch.tensor(0.0, device=config.device)

            # 4. Spectral decoupling.
            spec_lang = spectral_decoupling_loss(z, _onehot(lang, config.num_languages))
            spec_period = spectral_decoupling_loss(z, _onehot(period, config.num_periods))

            # 5. Adversarial logits + cross-entropy (heads try to predict).
            adv_lang_logits_list = adv_lang(z)
            adv_period_logits_list = adv_period(z)
            adv_lang_ce = sum(F.cross_entropy(logit, lang) for logit in adv_lang_logits_list) / len(
                adv_lang_logits_list
            )
            adv_period_ce = sum(
                F.cross_entropy(logit, period) for logit in adv_period_logits_list
            ) / len(adv_period_logits_list)

            # 6. Confusion loss (encoder tries to make nuisance posterior uniform).
            conf_lang = sum(confusion_loss(logit) for logit in adv_lang_logits_list) / len(
                adv_lang_logits_list
            )
            conf_period = sum(confusion_loss(logit) for logit in adv_period_logits_list) / len(
                adv_period_logits_list
            )

            total = (
                main_loss
                + config.lambda_vib * vib_l
                + config.lambda_spec * (spec_lang + spec_period)
                + config.lambda_conf * (conf_lang + conf_period)
                + adv_lang_ce
                + adv_period_ce  # GRL on z handles encoder-side direction
            )

            opt.zero_grad()
            total.backward()
            opt.step()

            epoch_loss += float(total.item())
            preds = logits.argmax(dim=-1)
            n_correct += int((preds == labels).sum())
            n_total += len(labels)

        avg = epoch_loss / max(1, len(dataset))
        acc = n_correct / max(1, n_total)
        history.epoch_losses.append(avg)
        history.epoch_main_accs.append(acc)
        if (epoch + 1) % config.log_every == 0:
            history.notes.append(f"epoch {epoch + 1}/{config.epochs}: loss={avg:.4f} acc={acc:.4f}")

    backbone.eval()
    return backbone, history


def save_checkpoint(
    backbone: ProbeBackbone,
    path: str | Path,
    history: TrainingHistory | None = None,
    *,
    corpus_fingerprint: dict | None = None,
    schema_version: str | None = None,
    model_id: str | None = None,
    model_weight_fingerprint: str | None = None,
) -> Path:
    """Save the trainable parts of the backbone to a .pt file with
    embedded `CalibrationProvenance`.

    Extra kwargs become provenance fields:
      `corpus_fingerprint` — dict of size + class counts + lang
            distribution (NOT raw texts); hashed for the
            probe_training_corpus_hash.
      `schema_version` — the IR schema this training ran against.
      `model_id`, `model_weight_fingerprint` — about the upstream LM
            the activations were drawn from.
    """
    from erisml_compiler.monitor.provenance import build_provenance_for_training

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state_dict = backbone.state_dict_for_checkpoint()
    provenance = build_provenance_for_training(
        state_dict=state_dict,
        history=history,
        corpus_fingerprint=corpus_fingerprint,
        schema_version=schema_version,
        model_id=model_id,
        model_weight_fingerprint=model_weight_fingerprint,
    )
    payload = {
        "state_dict": state_dict,
        "history": {
            "epoch_losses": history.epoch_losses if history else [],
            "epoch_main_accs": history.epoch_main_accs if history else [],
            "notes": history.notes if history else [],
        },
        "provenance": provenance.model_dump(),
    }
    torch.save(payload, p)
    return p


def load_checkpoint(
    path: str | Path,
    labse_model: str = "sentence-transformers/LaBSE",
    device: str = "cpu",
    *,
    return_provenance: bool = False,
):
    """Load a checkpoint and reconstruct the ProbeBackbone.

    By default returns the bare `ProbeBackbone` (back-compat). Pass
    `return_provenance=True` to also receive the embedded
    `CalibrationProvenance`; the call then returns
    `(backbone, provenance)`.
    """
    from erisml_compiler.monitor.provenance import CalibrationProvenance

    payload = torch.load(path, map_location=device, weights_only=False)
    sd = payload["state_dict"]
    cfg = sd["config"]
    backbone = ProbeBackbone(
        num_classes=cfg["num_classes"],
        labse_model=labse_model,
        use_vib=cfg.get("use_vib", True),
        device=device,
    )
    backbone.load_state_dict_from_checkpoint(sd)
    backbone.eval()
    if not return_provenance:
        return backbone
    provenance = CalibrationProvenance.from_checkpoint_payload(payload, path=path)
    return backbone, provenance
