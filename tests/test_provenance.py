"""Tests for CalibrationProvenance — item 2 of release-planning-01.

Confirms that every monitor trace carries provenance, that random
probes are explicitly marked uncalibrated rather than emitting null
fields, and that checkpoint save/load round-trips the hash and
calibration metadata.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from erisml_compiler.monitor import MockActivationSource
from erisml_compiler.monitor.activation_probe import ActivationProbe
from erisml_compiler.monitor.ieip_monitor import IEIPMonitor
from erisml_compiler.monitor.provenance import (
    CalibrationProvenance,
    _sha256_state_dict,
    build_provenance_for_training,
)

# ---------- uncalibrated default ----------


def test_uncalibrated_constructor_marks_is_calibrated_false():
    p = CalibrationProvenance.uncalibrated(seed=7)
    assert p.is_calibrated is False
    assert p.probe_checkpoint_hash is None
    assert "seed=7" in " ".join(p.notes)


def test_fresh_activationprobe_has_uncalibrated_provenance():
    """The whole point: no probe is ever silently missing provenance.
    Random probes carry an explicit is_calibrated=False marker."""
    probe = ActivationProbe(hidden_dim=8, seed=0)
    assert probe.provenance is not None
    assert probe.provenance.is_calibrated is False


def test_layer_probe_result_carries_provenance():
    src = MockActivationSource(hidden_dim=8, n_layers=2)
    cap = src.capture("hello")
    probe = ActivationProbe(hidden_dim=8, seed=0)
    result = probe.probe_layer(cap.layers[0])
    assert result.provenance is not None
    assert result.provenance.is_calibrated is False


# ---------- MonitorTrace serialisation ----------


def test_monitor_trace_dict_emits_per_layer_provenance():
    src = MockActivationSource(hidden_dim=12, n_layers=3)
    mon = IEIPMonitor(src, seed=42)
    trace = mon.monitor("text")
    d = trace.to_dict()
    for layer in d["per_layer"]:
        assert "provenance" in layer
        prov = layer["provenance"]
        assert prov is not None
        assert prov["is_calibrated"] is False


def test_monitor_trace_provenance_visible_in_json():
    """Roundtrip via JSON — auditor consuming the on-disk trace must
    see the provenance block."""
    import json

    src = MockActivationSource(hidden_dim=8, n_layers=2)
    mon = IEIPMonitor(src, seed=1)
    trace = mon.monitor("x")
    payload = json.dumps(trace.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["per_layer"][0]["provenance"]["is_calibrated"] is False


# ---------- checkpoint round-trip ----------


def test_save_and_load_checkpoint_includes_provenance(tmp_path):
    """Saving a checkpoint must embed a CalibrationProvenance; loading
    with return_provenance=True must surface it."""
    from erisml_compiler.calibration.probe_head import ProbeHead

    # Build a minimal "backbone-shaped" object that save_checkpoint
    # accepts. We bypass full ProbeBackbone construction (which downloads
    # LaBSE) by stubbing the methods save_checkpoint actually calls.
    class FakeBackbone:
        head = ProbeHead(in_dim=8, num_classes=10)

        def state_dict_for_checkpoint(self):
            sd = {"head": self.head.state_dict(), "config": {"num_classes": 10, "use_vib": False}}
            return sd

    fake = FakeBackbone()

    # Save with provenance kwargs
    from erisml_compiler.calibration.train import save_checkpoint

    ckpt = tmp_path / "probe.pt"
    save_checkpoint(
        fake,
        ckpt,
        history=None,
        corpus_fingerprint={"n_samples": 64, "n_classes": 3, "languages": ["en"]},
        schema_version="erisml_compiler_ir_v0.2",
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )
    assert ckpt.exists()

    # Reload + verify provenance round-trip
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "provenance" in payload
    prov = CalibrationProvenance.model_validate(payload["provenance"])
    assert prov.is_calibrated is True
    assert prov.probe_checkpoint_hash is not None
    assert len(prov.probe_checkpoint_hash) == 64  # SHA-256 hex
    assert prov.probe_training_corpus_hash is not None
    assert prov.probe_schema_version == "erisml_compiler_ir_v0.2"
    assert prov.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert prov.calibration_date is not None


def test_checkpoint_hash_deterministic_per_weights():
    """Same weights -> same hash, regardless of dict insertion order."""
    sd = {
        "head.net.0.weight": torch.ones(4, 8),
        "head.net.0.bias": torch.zeros(4),
        "config": {"num_classes": 10, "use_vib": True},
    }
    sd_reordered = {
        "config": {"num_classes": 10, "use_vib": True},
        "head.net.0.bias": torch.zeros(4),
        "head.net.0.weight": torch.ones(4, 8),
    }
    assert _sha256_state_dict(sd) == _sha256_state_dict(sd_reordered)


def test_checkpoint_hash_differs_when_weights_differ():
    sd_a = {"head.0.weight": torch.ones(4, 8)}
    sd_b = {"head.0.weight": torch.ones(4, 8) * 2}
    assert _sha256_state_dict(sd_a) != _sha256_state_dict(sd_b)


# ---------- legacy checkpoint synthesis ----------


def test_from_legacy_checkpoint_payload_synthesises_provenance():
    """Old checkpoints (no 'provenance' key) must still load with a
    synthesised provenance block flagged as legacy."""
    legacy_payload = {
        "state_dict": {"head.weight": torch.ones(4, 8)},
        "history": {"epoch_losses": [0.5, 0.3], "epoch_main_accs": [0.6, 0.8]},
    }
    prov = CalibrationProvenance.from_checkpoint_payload(legacy_payload)
    assert prov.is_calibrated is True  # weights exist, so we trust the load
    assert prov.probe_checkpoint_hash is not None
    assert prov.calibration_metrics.get("final_loss") == pytest.approx(0.3)
    assert prov.calibration_metrics.get("final_acc") == pytest.approx(0.8)
    assert any("legacy" in n.lower() for n in prov.notes)


# ---------- load_head_state propagation ----------


def test_load_head_state_replaces_provenance():
    probe = ActivationProbe(hidden_dim=8, seed=0)
    head_sd = probe.head.state_dict()
    new_prov = CalibrationProvenance(
        is_calibrated=True,
        probe_checkpoint_hash="a" * 64,
        model_id="Qwen/Qwen2.5-7B-Instruct",
    )
    probe.load_head_state({"head": head_sd, "config": {"num_classes": 10}}, provenance=new_prov)
    assert probe.provenance.is_calibrated is True
    assert probe.provenance.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_build_provenance_for_training_helper():
    sd = {"head.weight": torch.ones(2, 4)}

    class FakeHistory:
        epoch_losses = [1.0, 0.5]
        epoch_main_accs = [0.5, 0.9]

    prov = build_provenance_for_training(
        state_dict=sd,
        history=FakeHistory(),
        corpus_fingerprint={"n": 100},
        schema_version="v0.2",
        model_id="m",
    )
    assert prov.is_calibrated is True
    assert prov.calibration_metrics["final_loss"] == 0.5
    assert prov.calibration_metrics["final_acc"] == 0.9
    assert prov.probe_schema_version == "v0.2"
    assert prov.model_id == "m"
