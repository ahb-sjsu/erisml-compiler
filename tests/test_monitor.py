"""Track A tests — activation sources, probes, IEIPMonitor.

All tests use MockActivationSource so they run without a GPU or any
downloaded HF weights. They are CPU-cheap and run in the
test-llm-calibration CI job (which already has torch installed).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, MoralVector
from erisml_compiler.monitor import MockActivationSource
from erisml_compiler.monitor.activation_probe import (
    ActivationProbe,
    LayerProbeResult,
    _logits_to_moral_vector,
)
from erisml_compiler.monitor.ieip_monitor import IEIPMonitor, MonitorTrace

# ---------- MockActivationSource ----------


def test_mock_source_layer_shapes():
    src = MockActivationSource(hidden_dim=32, n_layers=4, n_tokens=8)
    cap = src.capture("hello")
    assert cap.hidden_dim == 32
    assert len(cap.layers) == 4
    for la in cap.layers:
        assert la.hidden.shape == (8, 32)
        assert la.pooled.shape == (32,)


def test_mock_source_deterministic_same_input():
    src = MockActivationSource(hidden_dim=16, n_layers=3, n_tokens=4)
    c1 = src.capture("medical confidentiality")
    c2 = src.capture("medical confidentiality")
    for a, b in zip(c1.layers, c2.layers):
        assert torch.allclose(a.hidden, b.hidden)
        assert torch.allclose(a.pooled, b.pooled)


def test_mock_source_text_changes_response():
    src = MockActivationSource(hidden_dim=16, n_layers=2)
    c1 = src.capture("a")
    c2 = src.capture("b")
    # Different texts must produce different activations at every layer.
    for a, b in zip(c1.layers, c2.layers):
        assert not torch.allclose(a.pooled, b.pooled)


def test_mock_source_layer_drift_is_monotonic_in_first_dim():
    """Layer-dependent drift was injected on dim 0 for probe gradient sanity."""
    src = MockActivationSource(hidden_dim=8, n_layers=5)
    cap = src.capture("x")
    means = [la.pooled[0].item() for la in cap.layers]
    # Should be non-decreasing modulo small noise.
    diffs = [b - a for a, b in zip(means, means[1:])]
    assert sum(d > 0 for d in diffs) >= len(diffs) - 1


def test_mock_source_subset_of_layers():
    src = MockActivationSource(hidden_dim=8, n_layers=8)
    cap = src.capture("x", layers=[0, 3, 7])
    assert cap.layer_indices() == [0, 3, 7]


def test_layer_path_resolver_handles_both_qwen_loaders():
    """AutoModel returns base; AutoModelForCausalLM wraps under .model.
    The resolver must accept both."""
    import torch.nn as nn

    from erisml_compiler.monitor.huggingface_source import _resolve_layers

    class FakeBase:
        def __init__(self):
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    class FakeWrapper:
        def __init__(self):
            self.model = FakeBase()

    # AutoModel-style (layers directly under the returned model).
    base = FakeBase()
    layers_a = _resolve_layers(base, "qwen2")
    assert len(layers_a) == 3

    # AutoModelForCausalLM-style (layers under .model).
    wrap = FakeWrapper()
    layers_b = _resolve_layers(wrap, "qwen2")
    assert len(layers_b) == 3


def test_mock_source_rejects_out_of_range_layer():
    src = MockActivationSource(n_layers=4)
    with pytest.raises(ValueError):
        src.capture("x", layers=[10])


# ---------- _logits_to_moral_vector ----------


def test_logits_to_moral_vector_shape_check():
    bad = torch.zeros(3)
    with pytest.raises(ValueError):
        _logits_to_moral_vector(bad)


def test_logits_to_moral_vector_neutral_at_zero():
    logits = torch.zeros(len(MORAL_DIMENSIONS))
    mv = _logits_to_moral_vector(logits)
    for dim in MORAL_DIMENSIONS:
        ds = getattr(mv, dim)
        assert ds.direction == "neutral"
        assert abs(ds.value) < 1e-9


def test_logits_to_moral_vector_signs():
    logits = torch.tensor([2.0, -2.0, 0.0, 1.5, -1.5, 0.5, -0.5, 0.01, -0.01, 3.0])
    mv = _logits_to_moral_vector(logits)
    # First and last large-positive -> positive direction.
    assert mv.physical_harm.direction == "positive"
    assert mv.repair_residue.direction == "positive"
    # Second strong-negative -> negative.
    assert mv.rights_respect.direction == "negative"
    # Small magnitudes near zero -> neutral.
    assert mv.vow_fidelity.direction == "neutral"
    assert mv.third_party_externality.direction == "neutral"


# ---------- ActivationProbe ----------


def test_activation_probe_outputs_moral_vector():
    src = MockActivationSource(hidden_dim=24, n_layers=3)
    cap = src.capture("test")
    probe = ActivationProbe(hidden_dim=24, seed=42)
    result = probe.probe_layer(cap.layers[0])
    assert isinstance(result, LayerProbeResult)
    assert isinstance(result.moral_vector, MoralVector)
    assert result.layer_index == 0


def test_activation_probe_dim_mismatch_raises():
    src = MockActivationSource(hidden_dim=24, n_layers=2)
    cap = src.capture("x")
    probe = ActivationProbe(hidden_dim=16, seed=0)  # mismatched
    with pytest.raises(ValueError):
        probe.probe_layer(cap.layers[0])


def test_activation_probe_load_wrapped_checkpoint_format():
    """ProbeBackbone.state_dict_for_checkpoint wraps under 'head' — accept that."""
    probe = ActivationProbe(hidden_dim=16, seed=0)
    sd = probe.head.state_dict()
    probe2 = ActivationProbe(hidden_dim=16, seed=99)
    probe2.load_head_state({"head": sd, "config": {"num_classes": 10}})
    # After loading, the heads should produce identical outputs.
    x = torch.randn(1, 16)
    with torch.no_grad():
        a = probe.head(x)
        b = probe2.head(x)
    assert torch.allclose(a, b)


# ---------- IEIPMonitor ----------


def test_monitor_end_to_end_with_fresh_probes():
    src = MockActivationSource(hidden_dim=16, n_layers=4)
    mon = IEIPMonitor(src, seed=0)
    trace = mon.monitor("nazi at the door")
    assert isinstance(trace, MonitorTrace)
    assert len(trace.per_layer) == 4
    assert trace.layer_indices == [0, 1, 2, 3]
    assert len(trace.activation_norms) == 4
    assert isinstance(trace.aggregated, MoralVector)


def test_monitor_trace_hash_is_deterministic():
    src = MockActivationSource(hidden_dim=12, n_layers=3)
    mon1 = IEIPMonitor(src, seed=7)
    mon2 = IEIPMonitor(src, seed=7)
    t1 = mon1.monitor("x")
    t2 = mon2.monitor("x")
    assert t1.trace_hash() == t2.trace_hash()


def test_monitor_trace_hash_differs_on_text_change():
    src = MockActivationSource(hidden_dim=12, n_layers=3)
    mon = IEIPMonitor(src, seed=7)
    t1 = mon.monitor("a")
    t2 = mon.monitor("b")
    assert t1.trace_hash() != t2.trace_hash()


def test_monitor_to_dict_is_json_serializable():
    import json

    src = MockActivationSource(hidden_dim=8, n_layers=2)
    mon = IEIPMonitor(src, seed=1)
    trace = mon.monitor("x")
    s = json.dumps(trace.to_dict())
    assert "aggregated" in s
    assert "per_layer" in s


def test_monitor_reuses_provided_probes():
    src = MockActivationSource(hidden_dim=10, n_layers=3)
    probes = {i: ActivationProbe(hidden_dim=10, seed=i) for i in range(3)}
    mon = IEIPMonitor(src, probes=probes)
    trace = mon.monitor("x")
    # The monitor should have stored *the same* probe instances.
    for i in range(3):
        assert mon._probes[i] is probes[i]
    # And aggregated must be a valid MoralVector.
    assert isinstance(trace.aggregated, MoralVector)


def test_monitor_rejects_probe_with_wrong_hidden_dim():
    src = MockActivationSource(hidden_dim=10, n_layers=2)
    bad_probes = {0: ActivationProbe(hidden_dim=8, seed=0)}  # mismatched dim
    mon = IEIPMonitor(src, probes=bad_probes)
    with pytest.raises(ValueError):
        mon.monitor("x", layers=[0])
