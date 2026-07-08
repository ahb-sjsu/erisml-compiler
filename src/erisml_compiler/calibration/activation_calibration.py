"""Calibrate per-layer activation probes for the I-EIP monitor.

The existing `calibration.train` path trains the *text* probe (frozen LaBSE ->
3-class role head, cross-entropy). This module trains the *activation* probe:
one `ProbeHead(hidden_dim -> 10 moral dims)` per model layer, supervised by a
teacher's 10-dim MoralVector labels, from pooled hidden-state features. It is
multi-output regression over [-1, 1], not classification.

Separation of concerns:
  - Feature extraction (Qwen activations) runs on GPU, offline, and is dumped
    to an .npz per layer: X (N, D) pooled hidden states, Y (N, 10) teacher
    moral values in [-1, 1]. See `scripts/experiments/atlas_calibration_extract.py`.
  - Training here is a tiny MLP fit on CPU — no LM load required.

The headline output is, per (layer, dimension), a held-out **sign-agreement
accuracy** — exactly the number Phase-5 prereg C3 gates on (>= 0.70 to include
a dimension in the delta lens). Dimensions below threshold are reported as
uncalibrated so the monitor excludes them rather than emitting noise.

Why sign-agreement (not R^2 / MSE)? The delta lens's `text_internal_mismatch`
counts *direction breaks* between the text and activation moral vectors, with a
0.05 dead band. The calibration metric that matches what the monitor consumes is
therefore whether the probe recovers the teacher's sign on dimensions the teacher
took a stance on — not how tightly it fits the magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from erisml_compiler.ir.schemas import MORAL_DIMENSIONS

DEAD_BAND = 0.05          # matches _logits_to_moral_vector's neutral band
INCLUDE_THRESHOLD = 0.70  # Phase-5 prereg C3: held-out acc >= 0.70 to include a dim


@dataclass
class ActivationCalibConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    head_hidden: int = 256
    head_layers: int = 2
    dropout: float = 0.2
    val_frac: float = 0.25
    seed: int = 0
    dead_band: float = DEAD_BAND
    include_threshold: float = INCLUDE_THRESHOLD
    device: str = "cpu"
    log_every: int = 10


@dataclass
class LayerCalibResult:
    layer_index: int
    per_dim_signacc: dict[str, float]     # held-out sign-agreement, per dimension
    per_dim_mae: dict[str, float]         # held-out mean abs error, per dimension
    per_dim_n_eval: dict[str, int]        # held-out items with |target| > dead_band
    included: dict[str, bool]             # signacc >= include_threshold
    n_train: int
    n_val: int
    final_train_loss: float
    notes: list[str] = field(default_factory=list)

    @property
    def n_included(self) -> int:
        return sum(self.included.values())

    def summary(self) -> str:
        inc = ",".join(d for d, ok in self.included.items() if ok) or "(none)"
        return (f"layer {self.layer_index}: {self.n_included}/{len(MORAL_DIMENSIONS)} dims "
                f"calibrated (>= {INCLUDE_THRESHOLD:.2f} sign-acc)  included=[{inc}]")


def _split(n: int, val_frac: float, seed: int):
    import torch

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]  # (train_idx, val_idx)


def _sign_agreement(pred, targ, dead_band: float):
    """Per-dimension sign agreement on items where the teacher took a stance
    (|target| > dead_band). Returns (signacc[10], mae[10], n_eval[10])."""
    import torch

    stance = targ.abs() > dead_band                     # (N, 10) bool
    agree = (torch.sign(pred) == torch.sign(targ)) & stance
    n_eval = stance.sum(dim=0).clamp(min=1)
    signacc = agree.sum(dim=0).float() / n_eval.float()
    signacc = torch.where(stance.sum(dim=0) > 0, signacc, torch.zeros_like(signacc))
    mae = (pred - targ).abs().mean(dim=0)
    return signacc, mae, stance.sum(dim=0)


def train_layer_probe(X, Y, config: ActivationCalibConfig, layer_index: int = -1):
    """Fit one ProbeHead(D -> 10) on (X, Y) and score held-out sign agreement.

    X: (N, D) float tensor/array of pooled activations.
    Y: (N, 10) float tensor/array of teacher moral values in [-1, 1], column
       order == MORAL_DIMENSIONS.
    Returns (head, LayerCalibResult). The head maps activations to *logits*;
    the monitor reads tanh(logits) as the moral value, so we train tanh(logits)
    toward Y with MSE (stable, and exactly what inference computes).
    """
    import torch
    import torch.nn.functional as F

    from erisml_compiler.calibration.probe_head import ProbeHead

    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be (N, D), got {tuple(X.shape)}")
    if Y.shape != (X.shape[0], len(MORAL_DIMENSIONS)):
        raise ValueError(f"Y must be (N, {len(MORAL_DIMENSIONS)}), got {tuple(Y.shape)}")

    n, d = X.shape
    tr_idx, va_idx = _split(n, config.val_frac, config.seed)
    Xtr, Ytr = X[tr_idx].to(config.device), Y[tr_idx].to(config.device)
    Xva, Yva = X[va_idx].to(config.device), Y[va_idx].to(config.device)

    torch.manual_seed(config.seed)
    head = ProbeHead(in_dim=d, num_classes=len(MORAL_DIMENSIONS),
                     hidden_dim=config.head_hidden, dropout=config.dropout,
                     n_layers=config.head_layers).to(config.device)
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    head.train()
    g = torch.Generator().manual_seed(config.seed)
    final_loss = float("nan")
    notes: list[str] = []
    for epoch in range(config.epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        epoch_loss, nb = 0.0, 0
        for s in range(0, len(Xtr), config.batch_size):
            bi = perm[s:s + config.batch_size]
            pred = torch.tanh(head(Xtr[bi]))
            loss = F.mse_loss(pred, Ytr[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            nb += 1
        final_loss = epoch_loss / max(1, nb)
        if config.log_every and (epoch + 1) % config.log_every == 0:
            notes.append(f"epoch {epoch + 1}/{config.epochs}: train_mse={final_loss:.4f}")

    head.eval()
    with torch.no_grad():
        pva = torch.tanh(head(Xva))
    signacc, mae, n_eval = _sign_agreement(pva.cpu(), Yva.cpu(), config.dead_band)

    dims = list(MORAL_DIMENSIONS)
    per_dim_signacc = {dims[i]: float(signacc[i]) for i in range(len(dims))}
    per_dim_mae = {dims[i]: float(mae[i]) for i in range(len(dims))}
    per_dim_n = {dims[i]: int(n_eval[i]) for i in range(len(dims))}
    included = {dims[i]: bool(signacc[i] >= config.include_threshold and n_eval[i] > 0)
                for i in range(len(dims))}

    result = LayerCalibResult(
        layer_index=layer_index,
        per_dim_signacc=per_dim_signacc,
        per_dim_mae=per_dim_mae,
        per_dim_n_eval=per_dim_n,
        included=included,
        n_train=len(Xtr),
        n_val=len(Xva),
        final_train_loss=final_loss,
        notes=notes,
    )
    return head, result


def save_layer_checkpoint(head, path: str | Path, result: LayerCalibResult, *,
                          corpus_fingerprint: dict | None = None,
                          model_id: str | None = None,
                          schema_version: str | None = None,
                          teacher: str | None = None) -> Path:
    """Serialise a calibrated activation-probe head in the shape
    `ActivationProbe.load_head_state` consumes, with embedded provenance
    whose `calibration_metrics` carry the per-dim held-out sign accuracy."""
    import torch

    from erisml_compiler.monitor.provenance import build_provenance_for_training

    class _Hist:  # duck-types the fields build_provenance_for_training reads
        epoch_losses = [result.final_train_loss]
        epoch_main_accs = [
            (sum(result.per_dim_signacc.values()) / len(result.per_dim_signacc))
            if result.per_dim_signacc else 0.0
        ]

    prov = build_provenance_for_training(
        state_dict={"head": head.state_dict()},
        history=_Hist(),
        corpus_fingerprint=corpus_fingerprint,
        schema_version=schema_version,
        model_id=model_id,
    )
    # Fold the per-dim held-out accuracy and the included set into the metrics
    # so an auditor sees exactly which dimensions this checkpoint calibrated.
    metrics = dict(prov.calibration_metrics)
    for dim, acc in result.per_dim_signacc.items():
        metrics[f"signacc.{dim}"] = round(acc, 4)
    metrics["n_dims_included"] = float(result.n_included)
    notes = list(prov.notes)
    if teacher:
        notes.append(f"teacher={teacher}")
    notes.append("included=" + ",".join(d for d, ok in result.included.items() if ok))
    prov = prov.model_copy(update={"calibration_metrics": metrics, "notes": notes})

    payload = {
        "state_dict": {"head": head.state_dict()},
        "history": {"epoch_losses": [result.final_train_loss], "notes": result.notes},
        "provenance": prov.model_dump(),
        "layer_index": result.layer_index,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return p


def load_calibrated_probe(path: str | Path, hidden_dim: int, device: str = "cpu"):
    """Reconstruct a calibrated `ActivationProbe` from a checkpoint written by
    `save_layer_checkpoint`, carrying its `CalibrationProvenance`."""
    import torch

    from erisml_compiler.monitor.activation_probe import ActivationProbe
    from erisml_compiler.monitor.provenance import CalibrationProvenance

    payload = torch.load(path, map_location=device, weights_only=False)
    prov = CalibrationProvenance.from_checkpoint_payload(payload, path=path)
    probe = ActivationProbe(hidden_dim=hidden_dim, device=device)
    probe.load_head_state(payload["state_dict"], provenance=prov)
    return probe


def calibration_table_rows(results: Sequence[LayerCalibResult], teacher: str) -> list[dict]:
    """Rows for the Phase-5 prereg §6 probe-calibration table: for each
    dimension, the best (max over layers) held-out sign accuracy, the layer
    that achieved it, the eval-n there, and the include decision."""
    dims = list(MORAL_DIMENSIONS)
    rows = []
    for dim in dims:
        best = max(results, key=lambda r: r.per_dim_signacc.get(dim, 0.0), default=None)
        acc = best.per_dim_signacc.get(dim, 0.0) if best else 0.0
        rows.append({
            "dimension": dim,
            "teacher": teacher,
            "best_layer": best.layer_index if best else None,
            "held_out_signacc": round(acc, 3),
            "n_eval": best.per_dim_n_eval.get(dim, 0) if best else 0,
            "included": bool(best.included.get(dim, False)) if best else False,
        })
    return rows


def render_calibration_table_md(rows: list[dict]) -> str:
    """Render §6's table as markdown (drop-in for phase5_prereg.md)."""
    out = ["| dimension | teacher | best layer | held-out sign-acc | n | included? |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['dimension']} | {r['teacher']} | {r['best_layer']} | "
                    f"{r['held_out_signacc']:.3f} | {r['n_eval']} | "
                    f"{'yes' if r['included'] else 'NO (excluded)'} |")
    return "\n".join(out)
