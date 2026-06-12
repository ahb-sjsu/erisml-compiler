"""Five named failure-mode detectors for the I-EIP Monitor.

These are the *behavioural signatures* the monitor watches for. Each is
a function `detector(...) -> FailureMode | None`. `detect_failure_modes`
runs all of them and returns a `FailureModeReport`.

The five modes (spec §31.5):

  1. text_internal_mismatch    — text-lens vs activation-lens disagree
                                 in direction on enough dimensions, or
                                 divergence above threshold.
  2. layerwise_drift           — per-layer probe outputs drift
                                 monotonically across layers in a way
                                 that suggests a representation the
                                 final-layer head is suppressing.
  3. group_symmetry_break      — equivariance test failed at layers
                                 the probe is supposed to be invariant.
  4. probe_uncertainty_spike   — joint uncertainty of any single
                                 dimension exceeds a hard ceiling.
  5. audit_chain_break         — the trace's audit hash does not match
                                 the expected chain (provenance
                                 violation, replay attack, or simple
                                 storage corruption).

A failure mode does NOT automatically force a verdict — by design, the
monitor's only authorised output is `requires_human_review` plus the
report. Verdicts remain DEME's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from erisml_compiler.delta.compare import DeltaResult
from erisml_compiler.delta.equivariance import EquivarianceReport
from erisml_compiler.ir.schemas import MORAL_DIMENSIONS
from erisml_compiler.monitor.ieip_monitor import MonitorTrace


class FailureMode(str, Enum):
    TEXT_INTERNAL_MISMATCH = "text_internal_mismatch"
    LAYERWISE_DRIFT = "layerwise_drift"
    GROUP_SYMMETRY_BREAK = "group_symmetry_break"
    PROBE_UNCERTAINTY_SPIKE = "probe_uncertainty_spike"
    AUDIT_CHAIN_BREAK = "audit_chain_break"


@dataclass(frozen=True)
class FailureModeReport:
    fired: list[FailureMode]
    details: dict[str, dict] = field(default_factory=dict)
    requires_human_review: bool = False

    def to_dict(self) -> dict:
        return {
            "fired": [m.value for m in self.fired],
            "details": self.details,
            "requires_human_review": self.requires_human_review,
        }


# ---------- individual detectors ----------


def _detect_text_internal_mismatch(delta: DeltaResult) -> tuple[bool, dict]:
    if not delta.flag_for_review:
        return False, {}
    return True, {
        "divergence": delta.divergence,
        "direction_break_count": delta.direction_break_count,
        "config": delta.config,
    }


def _detect_layerwise_drift(
    trace: MonitorTrace, *, monotone_run_min: int = 3, slope_min: float = 0.4
) -> tuple[bool, dict]:
    """Flag any dimension whose per-layer value drifts monotonically in
    the same sign across at least `monotone_run_min` consecutive layers
    and whose endpoint-to-endpoint magnitude change exceeds `slope_min`.

    Rationale: a monotone late-layer climb in (say) physical_harm
    that the final-layer aggregate then dampens looks exactly like a
    head suppressing a representation that is *present* deeper in the
    stack. That is precisely what we want the monitor to surface.
    """
    if len(trace.per_layer) < monotone_run_min:
        return False, {}

    layer_values: dict[str, list[float]] = {dim: [] for dim in MORAL_DIMENSIONS}
    for r in trace.per_layer:
        for dim in MORAL_DIMENSIONS:
            layer_values[dim].append(getattr(r.moral_vector, dim).value)

    flagged: dict[str, dict] = {}
    for dim, series in layer_values.items():
        # Look for a monotone run of length >= monotone_run_min.
        run = 1
        max_run = 1
        run_sign = 0
        max_run_start = 0
        max_run_end = 0
        for i in range(1, len(series)):
            diff = series[i] - series[i - 1]
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if sign != 0 and sign == run_sign:
                run += 1
            else:
                run = 2 if sign != 0 else 1
                run_sign = sign
            if run > max_run:
                max_run = run
                max_run_end = i
                max_run_start = i - run + 1
        if max_run >= monotone_run_min:
            magnitude = abs(series[max_run_end] - series[max_run_start])
            if magnitude >= slope_min:
                flagged[dim] = {
                    "run_length": max_run,
                    "start_layer_position": max_run_start,
                    "end_layer_position": max_run_end,
                    "magnitude_change": magnitude,
                    "endpoint_values": [series[max_run_start], series[max_run_end]],
                }

    return (len(flagged) > 0), {"dimensions": flagged} if flagged else {}


def _detect_group_symmetry_break(
    equivariance: EquivarianceReport | None,
) -> tuple[bool, dict]:
    if equivariance is None:
        return False, {}
    if not equivariance.failed_layers:
        return False, {}
    return True, {
        "failed_layers": equivariance.failed_layers,
        "n_results": len(equivariance.per_layer_per_rewrite),
        "config": equivariance.config,
    }


def _detect_probe_uncertainty_spike(
    delta: DeltaResult, *, uncertainty_ceiling: float = 0.85
) -> tuple[bool, dict]:
    spikes: list[dict] = []
    for d in delta.per_dimension:
        if d.joint_uncertainty >= uncertainty_ceiling:
            spikes.append(
                {
                    "dimension": d.dimension,
                    "joint_uncertainty": d.joint_uncertainty,
                    "value_delta": d.value_delta,
                }
            )
    return (len(spikes) > 0), {"spikes": spikes} if spikes else {}


def _detect_audit_chain_break(
    trace: MonitorTrace, *, expected_hash: str | None = None
) -> tuple[bool, dict]:
    if expected_hash is None:
        return False, {}
    actual = trace.trace_hash()
    if actual == expected_hash:
        return False, {}
    return True, {"expected_hash": expected_hash, "actual_hash": actual}


# ---------- orchestrator ----------


def detect_failure_modes(
    *,
    delta: DeltaResult,
    trace: MonitorTrace,
    equivariance: EquivarianceReport | None = None,
    expected_trace_hash: str | None = None,
    uncertainty_ceiling: float = 0.85,
    monotone_run_min: int = 3,
    slope_min: float = 0.4,
) -> FailureModeReport:
    """Run all five detectors and assemble a report.

    `expected_trace_hash` is used by the audit-chain check; pass the
    previously-recorded hash if you have one (provenance verification).
    """
    fired: list[FailureMode] = []
    details: dict[str, dict] = {}

    hit, d = _detect_text_internal_mismatch(delta)
    if hit:
        fired.append(FailureMode.TEXT_INTERNAL_MISMATCH)
        details[FailureMode.TEXT_INTERNAL_MISMATCH.value] = d

    hit, d = _detect_layerwise_drift(trace, monotone_run_min=monotone_run_min, slope_min=slope_min)
    if hit:
        fired.append(FailureMode.LAYERWISE_DRIFT)
        details[FailureMode.LAYERWISE_DRIFT.value] = d

    hit, d = _detect_group_symmetry_break(equivariance)
    if hit:
        fired.append(FailureMode.GROUP_SYMMETRY_BREAK)
        details[FailureMode.GROUP_SYMMETRY_BREAK.value] = d

    hit, d = _detect_probe_uncertainty_spike(delta, uncertainty_ceiling=uncertainty_ceiling)
    if hit:
        fired.append(FailureMode.PROBE_UNCERTAINTY_SPIKE)
        details[FailureMode.PROBE_UNCERTAINTY_SPIKE.value] = d

    hit, d = _detect_audit_chain_break(trace, expected_hash=expected_trace_hash)
    if hit:
        fired.append(FailureMode.AUDIT_CHAIN_BREAK)
        details[FailureMode.AUDIT_CHAIN_BREAK.value] = d

    return FailureModeReport(
        fired=fired,
        details=details,
        requires_human_review=len(fired) > 0,
    )
