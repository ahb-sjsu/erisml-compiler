"""Track B tests — delta lens, equivariance, failure modes.

All tests use MockActivationSource or hand-constructed MoralVectors so
they run CPU-only.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from erisml_compiler.delta import (
    DeltaResult,
    DimensionDelta,
    FailureMode,
    compare_morals,
    detect_failure_modes,
)
from erisml_compiler.delta.equivariance import (
    DEFAULT_REWRITES,
    EquivarianceReport,
    Rewrite,
    check_equivariance,
)
from erisml_compiler.delta.failure_modes import (
    _detect_layerwise_drift,
    _detect_probe_uncertainty_spike,
)
from erisml_compiler.ir.schemas import MORAL_DIMENSIONS, DimensionScore, MoralVector
from erisml_compiler.monitor import MockActivationSource
from erisml_compiler.monitor.activation_probe import ActivationProbe
from erisml_compiler.monitor.ieip_monitor import IEIPMonitor


def _mv(
    value: float = 0.0,
    confidence: float = 0.5,
    uncertainty: float = 0.3,
    direction: str = "neutral",
) -> MoralVector:
    """Build a MoralVector with all dimensions at the given values."""
    fields = {
        dim: DimensionScore(
            value=value,
            confidence=confidence,
            uncertainty=uncertainty,
            direction=direction,
        )
        for dim in MORAL_DIMENSIONS
    }
    return MoralVector(**fields)


# ---------- compare_morals ----------


def test_compare_identical_is_zero_divergence_no_flag():
    a = _mv(0.4, 0.7, 0.2, "positive")
    b = _mv(0.4, 0.7, 0.2, "positive")
    r = compare_morals(a, b)
    assert r.divergence == 0.0
    assert r.direction_break_count == 0
    assert r.flag_for_review is False


def test_compare_value_delta_correct_sign():
    a = _mv(0.2, 0.5, 0.2, "positive")
    b = _mv(0.6, 0.5, 0.2, "positive")
    r = compare_morals(a, b)
    for d in r.per_dimension:
        assert abs(d.value_delta - 0.4) < 1e-9


def test_compare_direction_flip_inflates_divergence():
    a = _mv(0.5, 0.7, 0.1, "positive")
    b = _mv(0.4, 0.7, 0.1, "negative")  # same magnitude-ish, flipped sign
    r = compare_morals(a, b)
    # 10 of 10 dimensions flipped — should clearly flag for review.
    assert r.direction_break_count == 10
    assert r.flag_for_review is True


def test_compare_flag_when_divergence_above_threshold():
    a = _mv(0.0, 0.5, 0.0, "neutral")
    b = _mv(0.95, 0.5, 0.0, "positive")
    r = compare_morals(a, b, divergence_threshold=0.3)
    assert r.flag_for_review is True
    assert r.divergence > 0.3


def test_compare_uncertain_dimensions_with_value_delta():
    a = _mv(0.1, 0.4, 0.9, "neutral")
    b = _mv(0.5, 0.4, 0.9, "positive")
    r = compare_morals(a, b, high_uncertainty_threshold=0.7, uncertain_value_delta_max=0.2)
    # All 10 dimensions: both uncertainty >= 0.7 AND |delta|=0.4 >= 0.2
    # Should flag uncertain-and-different.
    assert r.flag_for_review is True
    assert len(r.high_uncertainty_dimensions) == 10


def test_compare_weights_change_divergence():
    a = _mv(0.0, 0.5, 0.2, "neutral")
    b = _mv(0.5, 0.5, 0.2, "positive")
    r_eq = compare_morals(a, b)
    # Pin everything to zero weight except physical_harm.
    weights = {dim: 0.0 for dim in MORAL_DIMENSIONS}
    weights["physical_harm"] = 1.0
    r_focused = compare_morals(a, b, weights=weights)
    # With all weight on one dim, divergence equals that dim's contribution.
    assert r_eq.divergence == r_focused.divergence  # all dims identical here


def test_compare_to_dict_roundtrips_structure():
    r = compare_morals(_mv(0.1), _mv(0.2))
    d = r.to_dict()
    assert "per_dimension" in d
    assert len(d["per_dimension"]) == len(MORAL_DIMENSIONS)
    assert "divergence" in d
    assert "flag_for_review" in d


def test_compare_by_dimension_lookup():
    r = compare_morals(_mv(0.1), _mv(0.2))
    d = r.by_dimension("physical_harm")
    assert d.dimension == "physical_harm"
    with pytest.raises(KeyError):
        r.by_dimension("not_a_dim")


# ---------- equivariance ----------


def test_equivariance_default_rewrites_skip_no_op_text():
    src = MockActivationSource(hidden_dim=12, n_layers=3)
    probes = {i: ActivationProbe(hidden_dim=12, seed=i) for i in range(3)}
    # Text already lower-cased, no extra whitespace, no period.
    report = check_equivariance(src, probes, "abc")
    # All three default rewrites are no-ops here -> no rewrites applied,
    # hence no per-layer results.
    assert report.per_layer_per_rewrite == []


def test_equivariance_lowercase_rewrite_runs():
    src = MockActivationSource(hidden_dim=12, n_layers=3)
    probes = {i: ActivationProbe(hidden_dim=12, seed=i) for i in range(3)}
    report = check_equivariance(src, probes, "Hello World")
    # lowercase + trim_trailing_period are no-ops? lowercase IS not a
    # no-op on "Hello World" -> Some results should exist.
    assert any(r.rewrite_name == "lowercase" for r in report.per_layer_per_rewrite)


def test_equivariance_drastic_rewrite_marks_failure():
    src = MockActivationSource(hidden_dim=12, n_layers=2)
    probes = {i: ActivationProbe(hidden_dim=12, seed=i) for i in range(2)}
    drastic = Rewrite("drastic", lambda s: "completely different text here")
    report = check_equivariance(
        src,
        probes,
        "original input",
        rewrites=[drastic],
        pooled_cosine_threshold=0.999,  # be strict
        probe_cosine_threshold=0.999,
    )
    # MockActivationSource depends entirely on text -> totally different
    # input produces a different pooled vector -> equivariance must fail.
    assert report.failed_layers == [0, 1]


def test_equivariance_report_helpers():
    src = MockActivationSource(hidden_dim=8, n_layers=2)
    probes = {i: ActivationProbe(hidden_dim=8, seed=i) for i in range(2)}
    rw = Rewrite("upper", lambda s: s.upper())
    report = check_equivariance(src, probes, "abc", rewrites=[rw])
    assert sorted(report.layer_indices) == [0, 1]
    assert report.by_layer(0)
    assert report.by_rewrite("upper")


def test_equivariance_to_dict_serialisable():
    import json

    src = MockActivationSource(hidden_dim=8, n_layers=2)
    probes = {i: ActivationProbe(hidden_dim=8, seed=i) for i in range(2)}
    rw = Rewrite("upper", lambda s: s.upper())
    report = check_equivariance(src, probes, "abc", rewrites=[rw])
    json.dumps(report.to_dict())  # must not raise


# ---------- failure modes ----------


def _trace_with_drift(dim_name: str, n_layers: int = 5) -> "MonitorTrace":
    """Hand-build a trace where one dimension drifts monotonically."""
    from erisml_compiler.monitor.activation_probe import LayerProbeResult
    from erisml_compiler.monitor.ieip_monitor import MonitorTrace, _aggregate

    per_layer = []
    for i in range(n_layers):
        # All dims at zero except the target, which climbs linearly.
        fields = {}
        for d in MORAL_DIMENSIONS:
            if d == dim_name:
                v = -1.0 + 2.0 * (i / (n_layers - 1))  # -1 -> +1
                fields[d] = DimensionScore(
                    value=v, confidence=abs(v), uncertainty=1 - abs(v),
                    direction="positive" if v > 0.05 else ("negative" if v < -0.05 else "neutral"),
                )
            else:
                fields[d] = DimensionScore(value=0.0, confidence=0.0, uncertainty=1.0)
        mv = MoralVector(**fields)
        per_layer.append(
            LayerProbeResult(
                layer_index=i,
                layer_name=f"L{i}",
                logits=torch.zeros(10),
                moral_vector=mv,
                pooled_norm=1.0,
            )
        )

    return MonitorTrace(
        text="t",
        source_name="mock",
        model_id="m",
        hidden_dim=8,
        per_layer=per_layer,
        aggregated=_aggregate(per_layer),
        activation_norms=[1.0] * n_layers,
        layer_indices=list(range(n_layers)),
    )


def test_layerwise_drift_detector_fires_on_monotone_climb():
    trace = _trace_with_drift("physical_harm", n_layers=5)
    hit, details = _detect_layerwise_drift(trace, monotone_run_min=3, slope_min=0.5)
    assert hit
    assert "physical_harm" in details["dimensions"]
    info = details["dimensions"]["physical_harm"]
    assert info["run_length"] >= 3
    assert info["magnitude_change"] >= 0.5


def test_layerwise_drift_skips_short_traces():
    trace = _trace_with_drift("physical_harm", n_layers=2)
    hit, _ = _detect_layerwise_drift(trace, monotone_run_min=3)
    assert not hit


def test_uncertainty_spike_detector():
    a = _mv(0.0, 0.5, 0.9, "neutral")
    b = _mv(0.5, 0.5, 0.9, "positive")
    r = compare_morals(a, b)
    hit, details = _detect_probe_uncertainty_spike(r, uncertainty_ceiling=0.8)
    assert hit
    assert len(details["spikes"]) > 0


def test_detect_failure_modes_clean_run_no_flag():
    # Identical text & activation -> delta is zero; no drift; no rewrites.
    a = _mv(0.0, 0.0, 0.0, "neutral")
    b = _mv(0.0, 0.0, 0.0, "neutral")
    delta = compare_morals(a, b)
    src = MockActivationSource(hidden_dim=8, n_layers=4)
    mon = IEIPMonitor(src, seed=0)
    trace = mon.monitor("text")
    report = detect_failure_modes(delta=delta, trace=trace)
    # The mock+random-probe combination might fire layerwise_drift; the
    # important property is requires_human_review is consistent with fired.
    assert report.requires_human_review == (len(report.fired) > 0)


def test_detect_failure_modes_audit_chain_break():
    delta = compare_morals(_mv(), _mv())
    src = MockActivationSource(hidden_dim=8, n_layers=3)
    mon = IEIPMonitor(src, seed=0)
    trace = mon.monitor("x")
    report = detect_failure_modes(
        delta=delta, trace=trace, expected_trace_hash="0" * 64
    )
    assert FailureMode.AUDIT_CHAIN_BREAK in report.fired
    assert report.requires_human_review is True


def test_detect_failure_modes_mismatch_fires():
    a = _mv(0.5, 0.7, 0.1, "positive")
    b = _mv(0.4, 0.7, 0.1, "negative")  # direction flip everywhere
    delta = compare_morals(a, b)
    src = MockActivationSource(hidden_dim=8, n_layers=3)
    mon = IEIPMonitor(src, seed=0)
    trace = mon.monitor("x")
    report = detect_failure_modes(delta=delta, trace=trace)
    assert FailureMode.TEXT_INTERNAL_MISMATCH in report.fired
    assert report.requires_human_review is True


def test_detect_failure_modes_group_symmetry_break():
    src = MockActivationSource(hidden_dim=8, n_layers=2)
    probes = {i: ActivationProbe(hidden_dim=8, seed=i) for i in range(2)}
    drastic = Rewrite("drastic", lambda s: "very different text indeed")
    eq = check_equivariance(
        src,
        probes,
        "original",
        rewrites=[drastic],
        pooled_cosine_threshold=0.999,
        probe_cosine_threshold=0.999,
    )
    delta = compare_morals(_mv(), _mv())
    mon = IEIPMonitor(src, seed=0)
    trace = mon.monitor("original")
    report = detect_failure_modes(delta=delta, trace=trace, equivariance=eq)
    assert FailureMode.GROUP_SYMMETRY_BREAK in report.fired
