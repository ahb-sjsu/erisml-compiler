"""Phase 5: rank-3..6 tensors via repeated rank-2 evaluation.

Strategy: build rank-3+ as **stacks** of rank-2 slices, where each slice
comes from running the V3 bridge against a perturbed IR. The axis a
slice corresponds to is determined by which perturbation we apply:

  rank 3 (k, n, τ)            — per-time-step IR snapshots (events ≤ τ).
                                Temporal evolution. Real axis.
  rank 4 (k, n, a, c)         — (action × coalition) cross. a and c
                                are stub axes (length 1 default, ≤ 8
                                parametrically); the slice is the
                                same rank-2 result replicated. Phase 6
                                will inject genuine coalition logic
                                via erisml.ethics.coalition.CoalitionContext.
  rank 5 (k, n, τ, a, c)      — coalition decisions evolving over time.
                                τ real; a and c stubs.
  rank 6 (k, n, τ, a, c, s)   — adds Monte Carlo over fact confidence.
                                τ + s real; a + c stubs.

(DEME V3 requires `rank == len(shape)`, so each new rank adds exactly
one new axis. The s axis is reserved for rank 6; users who want
samples only at rank ≤ 5 should request rank 6 with n_actions=1,
n_coalitions=1, and the desired n_samples.)

This module is intentionally a thin wrapper over `compile_to_v3_tensor`.
Higher ranks cost K × N × … times the rank-2 cost, so production users
should request a rank they actually need (don't ask for rank-6 if a
rank-2 will do).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.ir.v3 import MORAL_DIMENSIONS_V3, MoralTensorV3

log = logging.getLogger(__name__)


# Maximum axis lengths we accept on the CLI / programmatic path.
# Coalition factorial growth is brutal; keep these tight.
MAX_ACTIONS = 8
MAX_COALITIONS = 16
MAX_SAMPLES = 32


@dataclass(frozen=True)
class HigherRankConfig:
    """Configuration for ranks 3-6.

    `times` is set automatically from the IR's event timeline; the
    others are caller-controlled. Validates on construction.

    `coalition_mode` selects which subsets of stakeholders the c axis
    enumerates (Phase 6 — was stub in Phase 5):
      grand_only: one coalition, all stakeholders together
      singletons_only: each stakeholder alone
      pairwise: singletons + pairs
      all_subsets: every non-empty subset (capped at n_coalitions)
    """

    n_actions: int = 1
    n_coalitions: int = 1
    n_samples: int = 1
    sample_noise_std: float = 0.05  # gaussian width on fact.confidence
    sample_seed: int = 0
    coalition_mode: str = "grand_only"  # Phase 6: real semantics on c axis

    def __post_init__(self) -> None:
        if not (1 <= self.n_actions <= MAX_ACTIONS):
            raise ValueError(f"n_actions must be in [1, {MAX_ACTIONS}], got {self.n_actions}")
        if not (1 <= self.n_coalitions <= MAX_COALITIONS):
            raise ValueError(
                f"n_coalitions must be in [1, {MAX_COALITIONS}], got {self.n_coalitions}"
            )
        if not (1 <= self.n_samples <= MAX_SAMPLES):
            raise ValueError(f"n_samples must be in [1, {MAX_SAMPLES}], got {self.n_samples}")
        if not (0.0 <= self.sample_noise_std <= 0.5):
            raise ValueError(f"sample_noise_std must be in [0, 0.5], got {self.sample_noise_std}")


def build_moral_tensor_v3_rank3plus(
    ir: CompilerIR,
    *,
    rank: int,
    config: HigherRankConfig | None = None,
) -> MoralTensorV3:
    """Produce a rank-3..6 tensor by stacking rank-2 evaluations.

    Each higher-rank axis is constructed by calling the rank-2 V3
    bridge over a perturbation of the input IR (or, for `a`/`c` stub
    axes, replicating the same rank-2 result).
    """
    if rank < 3 or rank > 6:
        raise ValueError(f"This builder handles ranks 3-6 only; got {rank}")

    # HigherRankConfig validates on construction (__post_init__).
    config = config or HigherRankConfig()

    # Lazy import to keep the orchestrator's fallback path clean.
    from erisml_compiler.erisml_backend.v3_bridge import compile_to_v3_tensor

    # --- Axis 1 (always present beyond k): parties ----
    party_ids = [s.id for s in ir.stakeholders] if ir.stakeholders else ["aggregate"]
    n_parties = len(party_ids)

    # --- Axis 2 (τ for rank>=3): event time snapshots ----
    time_indices = _resolve_time_indices(ir)

    # Decide which axes the requested rank includes, in canonical order:
    #   3: (k, n, τ)
    #   4: (k, n, a, c)         — drops τ
    #   5: (k, n, τ, s)
    #   6: (k, n, τ, a, c, s)
    if rank == 3:
        axis_layout = ("k", "n", "tau")
        shape = (9, n_parties, len(time_indices))
        labels = {
            "k": list(MORAL_DIMENSIONS_V3),
            "n": list(party_ids),
            "tau": [str(t) for t in time_indices],
        }
    elif rank == 4:
        axis_layout = ("k", "n", "a", "c")
        shape = (9, n_parties, config.n_actions, config.n_coalitions)
        labels = {
            "k": list(MORAL_DIMENSIONS_V3),
            "n": list(party_ids),
            "a": [f"action_{i}" for i in range(config.n_actions)],
            "c": [f"coalition_{i}" for i in range(config.n_coalitions)],
        }
    elif rank == 5:
        # (k, n, τ, a, c) — coalition decisions evolving over time.
        # n_samples is ignored at rank 5; use rank 6 for MC.
        axis_layout = ("k", "n", "tau", "a", "c")
        shape = (
            9,
            n_parties,
            len(time_indices),
            config.n_actions,
            config.n_coalitions,
        )
        labels = {
            "k": list(MORAL_DIMENSIONS_V3),
            "n": list(party_ids),
            "tau": [str(t) for t in time_indices],
            "a": [f"action_{i}" for i in range(config.n_actions)],
            "c": [f"coalition_{i}" for i in range(config.n_coalitions)],
        }
    else:  # rank == 6
        axis_layout = ("k", "n", "tau", "a", "c", "s")
        shape = (
            9,
            n_parties,
            len(time_indices),
            config.n_actions,
            config.n_coalitions,
            config.n_samples,
        )
        labels = {
            "k": list(MORAL_DIMENSIONS_V3),
            "n": list(party_ids),
            "tau": [str(t) for t in time_indices],
            "a": [f"action_{i}" for i in range(config.n_actions)],
            "c": [f"coalition_{i}" for i in range(config.n_coalitions)],
            "s": [f"sample_{i}" for i in range(config.n_samples)],
        }

    # ---- evaluate ----------------------------------------------------
    data = np.zeros(shape, dtype=float)
    veto_locs: list[tuple[int, ...]] = []
    veto_flags: list[str] = []
    reason_codes: list[str] = []
    stub_axes: list[str] = []

    # Inner-loop: for each (τ, s, c), build the corresponding rank-2
    # tensor and broadcast across the (a) stub axis. Phase 6 makes the
    # c axis real when coalition_mode != "grand_only"; the a axis
    # remains a stub.
    has_tau = "tau" in axis_layout
    has_s = "s" in axis_layout
    has_action = "a" in axis_layout
    has_coalition = "c" in axis_layout
    coalition_is_real = has_coalition and config.coalition_mode != "grand_only"

    for tau_idx, tau_val in enumerate(time_indices) if has_tau else [(0, None)]:
        snapshot = _snapshot_at_time(ir, tau_val) if has_tau else ir

        for s_idx in range(config.n_samples) if has_s else [0]:
            perturbed = _perturb_for_sample(snapshot, s_idx, config) if has_s else snapshot

            if coalition_is_real:
                # Phase 6: real per-coalition slices.
                from erisml_compiler.erisml_backend.v3_phase6 import (
                    build_coalition_c_axis_slices,
                )

                c_slices, _c_labels = build_coalition_c_axis_slices(
                    perturbed,
                    coalition_mode=config.coalition_mode,
                    n_coalitions_requested=config.n_coalitions,
                )
                if len(c_slices) != config.n_coalitions:
                    log.warning(
                        "coalition builder produced %d slices, expected %d; truncating",
                        len(c_slices),
                        config.n_coalitions,
                    )
                    c_slices = c_slices[: config.n_coalitions]
                    while len(c_slices) < config.n_coalitions:
                        c_slices.append(c_slices[-1] if c_slices else np.zeros((9, n_parties)))
                for c_idx, slice_arr in enumerate(c_slices):
                    _write_coalition_slice(
                        data,
                        slice_arr,
                        axis_layout=axis_layout,
                        tau_idx=tau_idx if has_tau else None,
                        s_idx=s_idx if has_s else None,
                        c_idx=c_idx,
                    )
            else:
                # Phase 5 path: single rank-2 slice replicated across (a, c).
                slice_tensor = compile_to_v3_tensor(perturbed, rank=2, populate_ir_metrics=False)
                slice_arr = np.array(slice_tensor.values, dtype=float)
                if slice_arr.shape != (9, n_parties):
                    log.warning(
                        "rank-2 slice has shape %s, expected (9, %d); skipping",
                        slice_arr.shape,
                        n_parties,
                    )
                    continue
                _write_slice(
                    data,
                    slice_arr,
                    axis_layout=axis_layout,
                    tau_idx=tau_idx if has_tau else None,
                    s_idx=s_idx if has_s else None,
                    n_actions=config.n_actions if has_action else 1,
                    n_coalitions=config.n_coalitions if has_coalition else 1,
                )

                for loc in slice_tensor.veto_locations:
                    if len(loc) == 1:
                        veto_locs.append(
                            _lift_party_veto_to_full(
                                loc[0],
                                axis_layout,
                                tau_idx if has_tau else None,
                                s_idx if has_s else None,
                            )
                        )
                    elif len(loc) == 0:
                        veto_locs.append(())
                veto_flags.extend(slice_tensor.veto_flags)
                reason_codes.extend(slice_tensor.reason_codes)

    if has_action and config.n_actions > 1:
        stub_axes.append("a")
    if has_coalition and config.n_coalitions > 1 and not coalition_is_real:
        stub_axes.append("c")

    tensor = MoralTensorV3(
        rank=rank,
        shape=shape,
        axis_names=axis_layout,
        axis_labels=labels,
        values=data.tolist(),
        veto_flags=list(dict.fromkeys(veto_flags)),  # dedup, preserve order
        veto_locations=list(dict.fromkeys(veto_locs)),
        reason_codes=list(dict.fromkeys(reason_codes)),
        metadata={
            "build_strategy": "phase5_higher_rank",
            "n_time_steps": len(time_indices),
            "n_actions": config.n_actions,
            "n_coalitions": config.n_coalitions,
            "n_samples": config.n_samples,
            "sample_noise_std": config.sample_noise_std,
            "stub_axes": stub_axes,  # axes whose semantics are stub today
            "axis_semantics": {
                "k": "real (canonical 9-dim ordering)",
                "n": "real (per-stakeholder)",
                "tau": "real (event timeline)" if has_tau else "(absent)",
                "a": "stub (axis length only)" if has_action else "(absent)",
                "c": (
                    f"real (coalition_mode={config.coalition_mode})"
                    if coalition_is_real
                    else ("stub (axis length only)" if has_coalition else "(absent)")
                ),
                "s": "real (MC over fact confidence)" if has_s else "(absent)",
            },
            "coalition_mode": config.coalition_mode,
        },
    )
    # Attach spectral summary (eigenvalue scalars + per-axis spectra).
    # See docs/plans/release-planning-04-eigenvalue-scalar.md.
    from erisml_compiler.evaluation.spectral import attach_spectral_summary

    attach_spectral_summary(tensor)
    return tensor


# ---------- helpers -----------------------------------------------------


def _resolve_time_indices(ir: CompilerIR) -> list[int]:
    """Return the sorted unique event time_indices; defaults to [0] when
    no events exist (so the τ axis is always length ≥ 1)."""
    if not ir.events:
        return [0]
    times = sorted({int(e.time_index) for e in ir.events})
    return times if times else [0]


def _snapshot_at_time(ir: CompilerIR, time_index: int) -> CompilerIR:
    """Return an IR copy that keeps only the events and ethical_facts
    active at or before `time_index`.

    Events filter directly on `time_index`. EthicalFacts have no time
    field of their own, so we attribute them to events by **source-span
    overlap**: a fact survives at time τ iff at least one of its
    source_spans matches an event's source_spans (or segment_id prefix)
    where the event's time_index ≤ τ. Facts with no source_spans
    survive unconditionally (the conservative choice — assumes the
    fact is a precondition, not an event-driven consequence).

    Heuristic: when no fact survives the filter, fall back to the
    full fact list. This protects scenarios whose extractors don't
    populate source_spans cleanly.
    """
    events = [e for e in ir.events if e.time_index <= time_index]

    # Collect segment_ids touched by surviving events.
    active_segments: set[str] = set()
    active_spans: set[str] = set()
    for e in events:
        for span in e.source_spans:
            active_spans.add(span)
            seg = span.split(":", 1)[0]
            active_segments.add(seg)

    def _keep(fact) -> bool:
        if not fact.source_spans:
            return True  # untimed precondition
        for span in fact.source_spans:
            if span in active_spans:
                return True
            seg = span.split(":", 1)[0]
            if seg in active_segments:
                return True
        return False

    filtered_facts = [f for f in ir.ethical_facts if _keep(f)]
    if not filtered_facts:
        # Heuristic failed (no overlap or no source_spans on events);
        # keep all facts so the slice still has something to compute.
        filtered_facts = ir.ethical_facts

    return ir.model_copy(update={"events": events, "ethical_facts": filtered_facts})


def _perturb_for_sample(ir: CompilerIR, sample_idx: int, cfg: HigherRankConfig) -> CompilerIR:
    """Monte Carlo perturbation: jitter every EthicalFact's confidence by
    a per-sample deterministic gaussian. Sample 0 is always the
    unperturbed IR (so downstream tools can use it as a reference)."""
    if sample_idx == 0 or cfg.sample_noise_std == 0.0:
        return ir

    rng = np.random.default_rng(cfg.sample_seed + sample_idx)
    perturbed_facts = []
    for fact in ir.ethical_facts:
        noise = float(rng.normal(0.0, cfg.sample_noise_std))
        new_conf = max(0.0, min(1.0, fact.confidence + noise))
        perturbed_facts.append(fact.model_copy(update={"confidence": new_conf}))
    return ir.model_copy(update={"ethical_facts": perturbed_facts})


def _write_coalition_slice(
    data,
    slice_arr,
    *,
    axis_layout: tuple[str, ...],
    tau_idx: int | None,
    s_idx: int | None,
    c_idx: int,
) -> None:
    """Phase 6: place a (9, n) rank-2 slice at the c=c_idx position.

    For rank 4 (k, n, a, c): replicates across the `a` stub axis.
    For rank 5 (k, n, τ, a, c): pins τ and c, replicates across a.
    For rank 6 (k, n, τ, a, c, s): pins τ, c, s, replicates across a.
    """
    idx: list = [slice(None), slice(None)]  # k, n full
    for ax in axis_layout[2:]:
        if ax == "tau":
            idx.append(tau_idx)
        elif ax == "s":
            idx.append(s_idx)
        elif ax == "a":
            idx.append(slice(None))  # replicate across actions (still stub)
        elif ax == "c":
            idx.append(c_idx)
        else:
            raise ValueError(f"unknown axis name {ax!r}")
    target = data[tuple(idx)]
    if target.ndim == slice_arr.ndim:
        target[:] = slice_arr
    else:
        expanded = slice_arr.reshape(slice_arr.shape + (1,) * (target.ndim - slice_arr.ndim))
        import numpy as np

        target[:] = np.broadcast_to(expanded, target.shape)


def _write_slice(
    data,
    slice_arr,
    *,
    axis_layout: tuple[str, ...],
    tau_idx: int | None,
    s_idx: int | None,
    n_actions: int,
    n_coalitions: int,
) -> None:
    """Place a (9, n_parties) rank-2 slice into the higher-rank tensor.

    Higher-rank tensor's `(a, c)` stub axes get the slice replicated;
    `tau`/`s` axes (when present) get the slice at the given index.
    """
    # We index into `data` per axis_layout. data's shape matches axis_layout.
    # Build a slice index across all axes:
    idx: list = [slice(None), slice(None)]  # k, n always full
    for ax in axis_layout[2:]:
        if ax == "tau":
            idx.append(tau_idx)
        elif ax == "s":
            idx.append(s_idx)
        elif ax == "a":
            idx.append(slice(None))  # replicate across all actions
        elif ax == "c":
            idx.append(slice(None))  # replicate across all coalitions
        else:
            raise ValueError(f"unknown axis name {ax!r}")
    # Broadcast: `slice_arr` is (9, n_parties); the target is the
    # higher-rank tensor's sub-tensor whose remaining stub axes (a, c)
    # need replication. We promote slice_arr to the target's rank by
    # appending trivial axes, then numpy-broadcast.
    target = data[tuple(idx)]
    if target.ndim == slice_arr.ndim:
        target[:] = slice_arr
    else:
        expanded = slice_arr.reshape(slice_arr.shape + (1,) * (target.ndim - slice_arr.ndim))
        target[:] = np.broadcast_to(expanded, target.shape)


def _lift_party_veto_to_full(
    party_idx: int,
    axis_layout: tuple[str, ...],
    tau_idx: int | None,
    s_idx: int | None,
) -> tuple[int, ...]:
    """Lift a `(party_idx,)` rank-2 veto into a full coordinate matching
    the higher-rank tensor. Stub axes (a, c) collapse to 0."""
    coord: list[int] = [0, party_idx]  # k=0 placeholder, n=party_idx
    for ax in axis_layout[2:]:
        if ax == "tau":
            coord.append(int(tau_idx or 0))
        elif ax == "s":
            coord.append(int(s_idx or 0))
        else:  # a, c — stub axis, default to 0
            coord.append(0)
    return tuple(coord)
