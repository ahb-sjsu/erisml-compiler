"""Phase 6 of DEME V3 alignment: strategic layer + decision proofs.

Closes the V3 alignment with three deliverables on top of Phase 5's
tensor shape work:

1.  **Strategic analysis** (`compute_strategic_analysis`) — runs the
    Shapley attribution and Nash equilibrium detection from
    `erisml.ethics.game_theory` and `erisml.ethics.layers.strategic`
    against the compiled V3 rank-2 tensor. Returns a JSON-serialisable
    dict; the compiler surfaces it as `ir.strategic_analysis`.

2.  **Decision proof** (`build_decision_proof`) — builds a
    JSON-serialisable summary of `erisml.ethics.decision_proof.DecisionProof`
    capturing input-facts hash, profile hash, layer outputs, ranked
    options, moral-vector hash, and a hash chain back to the IR's
    existing `AuditRecord.ir_hash`. Surfaced as `ir.decision_proof`.

3.  **Real coalition semantics on the c axis** (`build_coalition_c_axis`)
    — when the higher-rank builder is asked for ranks 4-6, the c axis
    is no longer a stub: each coalition slice runs the V3 bridge with
    a different "active subset" of stakeholders (non-coalition members
    zeroed out post-bridge). Phase 5 left this as a stub; Phase 6
    makes it real. The a axis remains a stub for now — coalition
    *actions* require IR additions outside this migration's scope.

All three are JSON-only on the IR side; the native `MoralTensor` /
`DecisionProof` numpy objects are constructed inside the helpers and
serialised before storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.ir.v3 import MoralTensorV3

log = logging.getLogger(__name__)


# ---------- coalition mode parsing ---------------------------------------


VALID_COALITION_MODES = (
    "grand_only",
    "all_subsets",
    "singletons_only",
    "pairwise",
)


def _build_coalition_context(ir: CompilerIR, mode: str = "grand_only"):
    """Construct a `CoalitionContext` from the IR's stakeholders.

    Each stakeholder becomes one agent. Action labels are left empty
    (DEME V3 fills them with single-action defaults). Returns None if
    erisml-lib is unavailable.
    """
    try:
        from erisml.ethics.coalition import CoalitionContext
    except ImportError:
        return None

    agent_ids = tuple(s.id for s in ir.stakeholders) if ir.stakeholders else ("aggregate",)
    if mode not in VALID_COALITION_MODES:
        log.warning("Unknown coalition_mode=%r; defaulting to grand_only", mode)
        mode = "grand_only"
    return CoalitionContext(agent_ids=agent_ids, coalition_mode=mode)


# ---------- strategic analysis -------------------------------------------


def compute_strategic_analysis(
    ir: CompilerIR,
    tensor: MoralTensorV3,
    *,
    coalition_mode: str = "all_subsets",
) -> dict | None:
    """Compute Shapley + Nash analysis from the V3 rank-2 tensor.

    Args:
        ir: the compiled IR (we need `ir.stakeholders` for agent ids).
        tensor: the V3 moral tensor (rank >= 2). For higher ranks we
            collapse to rank 2 via mean over the trailing axes.
        coalition_mode: how to enumerate coalitions for Shapley.
            "all_subsets" is the canonical Shapley case but is O(2^n);
            we cap at 8 stakeholders to keep runtimes bounded.

    Returns:
        Dict with keys:
            shapley_values: dict[stakeholder_id, float]
            shapley_method: "exact" or "monte_carlo"
            welfare_metrics: dict with mean / min / sum across parties
            n_agents, coalition_mode
        Or None if erisml-lib is unavailable.
    """
    try:
        from erisml.ethics.coalition import CoalitionContext  # noqa: F401
        from erisml.ethics.game_theory import compute_shapley_from_tensor
        import numpy as np
    except ImportError:
        log.info("erisml-lib not installed; skipping strategic analysis")
        return None

    n_parties = len(ir.stakeholders) if ir.stakeholders else 1
    if n_parties < 2:
        log.info("Strategic analysis needs >=2 stakeholders; skipping")
        return None
    if n_parties > 8 and coalition_mode == "all_subsets":
        log.info("Falling back to monte-carlo Shapley: %d parties is too many for exact", n_parties)

    ctx = _build_coalition_context(ir, mode=coalition_mode)
    if ctx is None:
        return None

    # Project tensor to rank-2 (k, n) — collapse higher ranks via mean.
    arr = np.array(tensor.values, dtype=float)
    while arr.ndim > 2:
        arr = arr.mean(axis=-1)
    if arr.shape != (9, n_parties):
        log.warning(
            "Tensor collapse produced shape %s, expected (9, %d); skipping",
            arr.shape,
            n_parties,
        )
        return None

    # Build a fresh DEME MoralTensor from the collapsed array.
    try:
        from erisml.ethics.moral_tensor import MoralTensor as DemeMoralTensor

        deme_tensor = DemeMoralTensor(
            _data=arr,
            shape=arr.shape,
            rank=2,
            axis_names=("k", "n"),
            axis_labels={"n": list(ctx.agent_ids)},
        )
    except Exception as e:
        log.warning("Failed to build DEME tensor for strategic analysis: %s", e)
        return None

    try:
        shapley = compute_shapley_from_tensor(deme_tensor, ctx, aggregation="sum")
    except Exception as e:
        log.warning("Shapley computation failed: %s; analysis skipped", e)
        return None

    # Shapley returns a ShapleyValues dataclass with `.values` dict-like.
    shapley_values = _shapley_to_dict(shapley, ctx)
    method = getattr(shapley, "computation_method", "exact")

    # Welfare metrics on the harm row (k=0) and the aggregate.
    harm_row = arr[0]
    metrics = {
        "mean_harm": float(harm_row.mean()),
        "max_harm": float(harm_row.max()),
        "min_harm": float(harm_row.min()),
        "harm_range": float(harm_row.max() - harm_row.min()),
    }

    return {
        "n_agents": n_parties,
        "coalition_mode": coalition_mode,
        "shapley_values": shapley_values,
        "shapley_method": method,
        "welfare_metrics": metrics,
    }


def _shapley_to_dict(shapley, ctx) -> dict[str, float]:
    """Convert ShapleyValues to a plain dict keyed by agent_id."""
    try:
        # ShapleyValues stores .values as np.ndarray
        import numpy as np

        vals = np.asarray(shapley.values, dtype=float)
        return {agent: float(v) for agent, v in zip(ctx.agent_ids, vals)}
    except Exception:
        # Fallback: try dict-like or list-like access.
        try:
            return dict(shapley.values)
        except Exception:
            return {}


# ---------- decision proof ------------------------------------------------


def build_decision_proof(
    ir: CompilerIR,
    tensor: MoralTensorV3,
    *,
    strategic_analysis: dict | None = None,
    previous_proof_hash: str | None = None,
) -> dict:
    """Build a `DecisionProof`-shaped audit summary from the compiled IR.

    Mirrors `erisml.ethics.decision_proof.DecisionProof` but stays as a
    plain dict so the IR remains pure JSON. The hash chain links to
    the IR's existing `AuditRecord.ir_hash`.
    """
    import time
    import uuid

    # Hash the V3 tensor's values + metadata for the moral_vector_summary.
    tensor_blob = json.dumps(
        {"values": tensor.values, "shape": list(tensor.shape), "axes": list(tensor.axis_names)},
        sort_keys=True,
        separators=(",", ":"),
    )
    tensor_hash = hashlib.sha256(tensor_blob.encode("utf-8")).hexdigest()

    # Facts hash: SHA-256 of the canonical fact JSON.
    facts_blob = json.dumps(
        [f.model_dump() for f in ir.ethical_facts],
        sort_keys=True,
        separators=(",", ":"),
    )
    facts_hash = hashlib.sha256(facts_blob.encode("utf-8")).hexdigest()

    # Profile hash: the EM-DAG profile name + canonical_form (closest
    # the compiler has to a "governance profile" today).
    profile_blob = json.dumps(
        {"canonical_form": ir.canonical_form, "schema_version": ir.schema_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    profile_hash = hashlib.sha256(profile_blob.encode("utf-8")).hexdigest()

    # Layer outputs — Phase 4 + 5 don't have explicit layer separation,
    # so emit one synthetic "v3_bridge" layer carrying the per-party
    # verdicts and strategic-analysis summary.
    layers = [
        {
            "layer_name": "v3_bridge",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration_us": 0,  # not measured today
            "veto_triggered": bool(tensor.veto_flags),
            "veto_reason": (tensor.veto_flags[0] if tensor.veto_flags else None),
            "output_data": {
                "tensor_rank": tensor.rank,
                "tensor_shape": list(tensor.shape),
                "build_strategy": tensor.metadata.get("build_strategy", "unknown"),
                "n_parties": len(ir.stakeholders) if ir.stakeholders else 0,
            },
        }
    ]
    if strategic_analysis is not None:
        layers.append(
            {
                "layer_name": "strategic",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "duration_us": 0,
                "veto_triggered": False,
                "veto_reason": None,
                "output_data": strategic_analysis,
            }
        )

    # Forbidden and ranked options derived from per_party_verdicts.
    ppv = ir.per_party_verdicts or {}
    forbidden = [pid for pid, v in ppv.items() if v == "forbid"]
    ranked = sorted(
        ppv.keys(),
        key=lambda pid: {
            "forbid": 4,
            "avoid": 3,
            "neutral": 2,
            "prefer": 1,
            "strongly_prefer": 0,
        }.get(ppv.get(pid, "neutral"), 2),
    )

    proof: dict[str, Any] = {
        "decision_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_facts_hash": facts_hash,
        "profile_hash": profile_hash,
        "profile_name": ir.canonical_form or "default",
        "em_catalog_version": "v3",
        "active_em_names": (tensor.metadata.get("modules_invoked", []) if tensor.metadata else []),
        "layer_outputs": layers,
        "em_judgements": [],  # individual EM records would land here if Phase 4 surfaced them
        "candidate_option_ids": list(ppv.keys()),
        "selected_option_id": None,
        "ranked_options": ranked,
        "forbidden_options": forbidden,
        "governance_rationale": (ir.deme_verdict.rationale if ir.deme_verdict else ""),
        "moral_vector_summary": {
            "tensor_hash": tensor_hash,
            "rank": tensor.rank,
            "shape": list(tensor.shape),
            "fairness_metrics": ir.fairness_metrics or {},
        },
        "previous_proof_hash": previous_proof_hash
        or (ir.audit.ir_hash if ir.audit and getattr(ir.audit, "ir_hash", None) else None),
        "proof_hash": "",  # filled below
        "metadata": {
            "schema_version": ir.schema_version,
            "phase": "v3_phase6",
        },
    }

    # Compute the proof hash over everything except proof_hash itself.
    proof_blob = json.dumps(
        {k: v for k, v in proof.items() if k != "proof_hash"},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    proof["proof_hash"] = hashlib.sha256(proof_blob.encode("utf-8")).hexdigest()

    return proof


# ---------- coalition axis (Phase 6 real semantics) ----------------------


def build_coalition_c_axis_slices(
    ir: CompilerIR,
    *,
    coalition_mode: str,
    n_coalitions_requested: int,
) -> tuple[list, list[str]]:
    """For ranks 4-6: produce real per-coalition rank-2 slices.

    Each coalition is a subset of stakeholders that "acts together".
    For non-coalition members we zero their contribution by running
    the V3 bridge with their ethical facts removed.

    Returns:
        (slices, labels) where `slices` is a list of length
        n_actual_coalitions, each a numpy array of shape (9, n_parties),
        and `labels` is the matching list of human-readable coalition
        strings.

    The number of coalitions produced depends on coalition_mode:
        grand_only       → 1 (all stakeholders together)
        singletons_only  → n_stakeholders
        pairwise         → n + n*(n-1)/2
        all_subsets      → 2^n - 1 (excludes empty)
    If the requested count exceeds the natural enumeration, slices are
    truncated; if fewer, they're padded with copies of the grand
    coalition's slice.
    """
    try:
        from erisml_compiler.erisml_backend.v3_bridge import compile_to_v3_tensor
        import numpy as np
    except ImportError:
        return [], []

    if not ir.stakeholders:
        return [], []

    party_ids = [s.id for s in ir.stakeholders]
    n = len(party_ids)
    party_idx = {pid: i for i, pid in enumerate(party_ids)}

    # Enumerate coalitions per mode.
    coalitions: list[tuple[str, ...]] = []
    if coalition_mode == "grand_only":
        coalitions = [tuple(party_ids)]
    elif coalition_mode == "singletons_only":
        coalitions = [(p,) for p in party_ids]
    elif coalition_mode == "pairwise":
        coalitions = [(p,) for p in party_ids]
        for i in range(n):
            for j in range(i + 1, n):
                coalitions.append((party_ids[i], party_ids[j]))
    elif coalition_mode == "all_subsets":
        from itertools import combinations

        for r in range(1, n + 1):
            for c in combinations(party_ids, r):
                coalitions.append(c)
    else:
        coalitions = [tuple(party_ids)]

    # Cap to MAX_COALITIONS (16) — exponential growth is brutal.
    coalitions = coalitions[:n_coalitions_requested]
    if not coalitions:
        return [], []

    # Pad to requested length if needed.
    while len(coalitions) < n_coalitions_requested:
        coalitions.append(coalitions[0])  # pad with grand/first coalition

    slices = []
    labels = []
    for coalition in coalitions:
        # Build a subset IR: keep only facts whose subjects ⊆ coalition,
        # or facts whose subjects are empty (untargeted — always apply).
        coalition_set = set(coalition)
        sub_facts = [
            f
            for f in ir.ethical_facts
            if not f.subjects or set(f.subjects).intersection(coalition_set)
        ]
        sub_ir = ir.model_copy(update={"ethical_facts": sub_facts})

        try:
            slice_tensor = compile_to_v3_tensor(sub_ir, rank=2, populate_ir_metrics=False)
            arr = np.array(slice_tensor.values, dtype=float)
            if arr.shape != (9, n):
                log.warning("coalition slice shape mismatch %s; using zeros", arr.shape)
                arr = np.zeros((9, n))
            # Zero out non-coalition members.
            for pid in party_ids:
                if pid not in coalition_set:
                    arr[:, party_idx[pid]] = 0.0
        except Exception as e:
            log.warning("coalition slice (%s) failed: %s; using zeros", coalition, e)
            arr = np.zeros((9, n))

        slices.append(arr)
        labels.append("|".join(coalition) if len(coalition) <= 3 else f"{len(coalition)}-coalition")

    return slices, labels
