"""CalibrationProvenance — first-class checkpoint identity for activation probes.

Per the I-EIP Monitor threat model (`docs/i_eip_monitor.md`), a trained
`ActivationProbe`'s weights can be corrupted at training time (calibrate
on poisoned data), at serialisation time (tampered .pt blob), or at
load time (substituted path). The docs already flag this as T1 (probe
poisoning) and tell callers to hash the checkpoint themselves.

This module makes that hash a first-class object that flows through to
every `MonitorTrace`, so a downstream auditor can verify:

  - which checkpoint was loaded (`probe_checkpoint_hash`)
  - which corpus that checkpoint was trained on
    (`probe_training_corpus_hash`)
  - which Pydantic IR schema the calibration ran against
    (`probe_schema_version`)
  - the calibration's headline metrics
    (`calibration_metrics`)
  - when calibration happened (`calibration_date`)
  - which model id the probe targets (`model_id`)
  - a content-hash of the model's frozen weights
    (`model_weight_fingerprint`)

Random / uncalibrated probes get `is_calibrated=False` with all
hash fields explicitly None so a missing hash never looks like a
silently-attached anonymous probe.

The provenance is stored on `ActivationProbe.provenance` and copied
into `LayerProbeResult.provenance` when the probe runs. The
`MonitorTrace.to_dict()` serialisation emits it per layer.

Reserved metadata keys:
  - `is_calibrated: bool` — True iff loaded from a .pt checkpoint
  - `notes: list[str]` — free-form annotation (e.g. "fresh seed=42")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _sha256_dict(obj: Any) -> str:
    """SHA-256 over a canonical-JSON of a dict-like."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_state_dict(state_dict: dict) -> str:
    """SHA-256 over a torch state_dict.

    We can't json the raw tensors, so we hash their shape + bytes per
    key, in sorted key order. The result is deterministic for a given
    set of weights regardless of dict insertion order.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return _sha256_dict({k: str(v) for k, v in state_dict.items()})

    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        if isinstance(v, torch.Tensor):
            h.update(str(tuple(v.shape)).encode("utf-8"))
            h.update(v.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            h.update(_sha256_dict(v).encode("utf-8"))
        else:
            h.update(str(v).encode("utf-8"))
    return h.hexdigest()


class CalibrationProvenance(BaseModel):
    """Per-probe checkpoint identity carried on `ActivationProbe.provenance`.

    All hash fields are optional because random / uncalibrated probes
    legitimately have nothing to hash. `is_calibrated` is the only
    field a consumer should branch on — it disambiguates "no
    provenance because never calibrated" from "provenance lost".
    """

    model_config = ConfigDict(frozen=True)

    is_calibrated: bool = False
    """True iff this probe's weights came from a saved checkpoint."""

    probe_checkpoint_hash: str | None = None
    """SHA-256 of the probe's state_dict (deterministic per weights)."""

    probe_training_corpus_hash: str | None = None
    """SHA-256 of the calibration corpus fingerprint (size + dim + per-class
    counts + lang/period distributions, NOT the raw texts)."""

    probe_schema_version: str | None = None
    """Schema version of the IR the calibration trained against
    (e.g. erisml_compiler_ir_v0.2)."""

    calibration_metrics: dict[str, float] = Field(default_factory=dict)
    """Headline metrics from the training run — final loss, final acc,
    plus any task-specific numbers."""

    calibration_date: str | None = None
    """ISO-8601 timestamp; None for uncalibrated probes."""

    model_id: str | None = None
    """The activation-source model id (e.g. Qwen/Qwen2.5-7B-Instruct).
    Distinguishes a probe trained against Qwen activations from one
    trained against LLaMA's."""

    model_weight_fingerprint: str | None = None
    """SHA-256 of the underlying model's frozen weights. Optional
    because for very large models this is expensive to compute; we
    record it when the source can supply it cheaply (HF model_id ->
    safetensors hash, etc.)."""

    notes: list[str] = Field(default_factory=list)
    """Free-form annotation."""

    @classmethod
    def uncalibrated(
        cls, *, source: str = "random", seed: int | None = None
    ) -> "CalibrationProvenance":
        """Construct an honest 'no calibration' provenance.

        Use this for fresh random-init probes. It's not None / not
        missing; the trace will explicitly say is_calibrated=False so
        an auditor knows the activation lens output is uncalibrated noise.
        """
        notes = [f"uncalibrated source={source}"]
        if seed is not None:
            notes.append(f"seed={seed}")
        return cls(is_calibrated=False, notes=notes)

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: dict,
        *,
        path: str | Path | None = None,
    ) -> "CalibrationProvenance":
        """Reconstruct from a `torch.save(payload)` blob.

        Expects the payload shape produced by
        `erisml_compiler.calibration.train.save_checkpoint`:

            {"state_dict": ..., "history": ..., "provenance": ?}

        If the payload already has a `provenance` block (probes saved
        with this module), it's returned verbatim. Otherwise we
        synthesise minimal provenance from whatever fields are
        present — hashing the state_dict, lifting metrics from
        history, dating from path mtime, and flagging
        is_calibrated=True since a checkpoint was loaded.
        """
        existing = payload.get("provenance")
        if isinstance(existing, dict):
            return cls.model_validate(existing)

        state_dict = payload.get("state_dict") or {}
        history = payload.get("history") or {}
        ckpt_hash = _sha256_state_dict(state_dict) if state_dict else None

        metrics: dict[str, float] = {}
        if history.get("epoch_losses"):
            metrics["final_loss"] = float(history["epoch_losses"][-1])
        if history.get("epoch_main_accs"):
            metrics["final_acc"] = float(history["epoch_main_accs"][-1])

        date = None
        if path:
            try:
                import datetime as _dt

                ts = Path(path).stat().st_mtime
                date = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()
            except Exception:
                date = None

        return cls(
            is_calibrated=True,
            probe_checkpoint_hash=ckpt_hash,
            calibration_metrics=metrics,
            calibration_date=date,
            notes=["synthesised from legacy checkpoint without provenance block"],
        )


def build_provenance_for_training(
    *,
    state_dict: dict,
    history: Any,
    corpus_fingerprint: dict | None = None,
    schema_version: str | None = None,
    model_id: str | None = None,
    model_weight_fingerprint: str | None = None,
) -> CalibrationProvenance:
    """Build provenance at `save_checkpoint` time.

    `corpus_fingerprint` should be a dict like
    `{"n_samples": ..., "n_classes": ..., "languages": [...]}` — NOT
    the raw texts. We hash it canonically.
    """
    import datetime as _dt

    metrics: dict[str, float] = {}
    if history is not None:
        if getattr(history, "epoch_losses", None):
            metrics["final_loss"] = float(history.epoch_losses[-1])
        if getattr(history, "epoch_main_accs", None):
            metrics["final_acc"] = float(history.epoch_main_accs[-1])

    corpus_hash = _sha256_dict(corpus_fingerprint) if corpus_fingerprint else None
    return CalibrationProvenance(
        is_calibrated=True,
        probe_checkpoint_hash=_sha256_state_dict(state_dict),
        probe_training_corpus_hash=corpus_hash,
        probe_schema_version=schema_version,
        calibration_metrics=metrics,
        calibration_date=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        model_id=model_id,
        model_weight_fingerprint=model_weight_fingerprint,
    )
