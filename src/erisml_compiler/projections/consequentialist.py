"""Consequentialist-distributive projection.

Reads the substrate and produces what the current `MoralTensorV3`
pipeline produces: per-stakeholder tensor + DEME verdict + Gini /
worst-off / Shapley aggregates. This is *one* projection among many,
not the IR's privileged output.

For backward compatibility, the orchestrator continues to populate
`ir.moral_tensor_v3`, `ir.deme_verdict`, `ir.per_party_verdicts`, and
`ir.fairness_metrics` from this projection's output. New code should
read from `ir.projections["consequentialist_distributive"]` instead.
"""

from __future__ import annotations

from typing import Any

from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
from erisml_compiler.evaluation.deme_evaluation import build_timeline
from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.projections.base import Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate


class ConsequentialistProjection(Projection):
    """Per-stakeholder harm/care/etc tensor + Gini-style aggregates.

    This is intentionally a thin orchestrator over erisml-lib's DEME:
    the verdict, the moral vector and the timeline come from DEME under the
    bridge's profile, the per-party tensor from the V3 bridge (erisml-lib's
    V3 modules). (Until 0.10 the compiler's own EM-DAG scored a ten-channel
    vector here; it is retired.) It exists to make the
    framework commitment *explicit* in the architecture: this
    projection *is* the consequentialist-distributive reading, not
    the privileged-and-only reading.
    """

    framework = "consequentialist_distributive"

    def __init__(
        self,
        *,
        bridge: DEMEBridge | None = None,
        tensor_rank: int = 2,
    ) -> None:
        """``bridge`` judges with its DEME profile and, when a fitted ethos profile applies, its
        dimension weights (evaluation.ethos)."""
        self.bridge = bridge or DEMEBridge()
        self.tensor_rank = tensor_rank

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        ir: CompilerIR | None = None,
        tensor_override: Any = None,
        tensor_given: bool = False,
        graph: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        """Run DEME over the substrate (via a CompilerIR view).

        For v0 we still require the caller to pass the full CompilerIR
        because DEME's bridge and the V3 bridge read from
        CompilerIR-shaped objects. The graph IS available (passed as
        `graph` and also reachable via `ir.graph`); we record graph-
        derived summary statistics in `metadata` to demonstrate the
        graph is the primary substrate even though the bridges
        have not been ported to read graph queries yet.

        `tensor_override` lets the orchestrator inject a tensor built
        via its V3-dispatch logic (strict_v3, higher-rank bridge,
        etc.) instead of having this projection always use the Phase 2
        V3 bridge at this projection's rank. When None, the bridge runs here.
        """
        if ir is None:
            raise ValueError(
                "ConsequentialistProjection.project requires the full CompilerIR "
                "via the `ir=` kwarg for v0. (DEME's bridge and the V3 bridge still "
                "read from CompilerIR; this constraint will be relaxed when "
                "the bridges are ported to read graph queries — "
                "release-planning-06 follow-up item.)"
            )

        # Graph-aware bookkeeping: even though the bridges still read
        # flat fields, record graph-derived statistics in metadata so
        # the projection result documents its graph awareness.
        graph_obj = graph if graph is not None else getattr(ir, "graph", None)
        graph_summary: dict[str, Any] = {}
        if graph_obj is not None:
            from erisml_compiler.ir.graph import EdgeKind, NodeKind

            graph_summary = {
                "n_stakeholders": len(graph_obj.nodes_of_kind(NodeKind.STAKEHOLDER)),
                "n_acts": len(graph_obj.nodes_of_kind(NodeKind.ACT)),
                "n_commitments": len(graph_obj.nodes_of_kind(NodeKind.COMMITMENT)),
                "n_facts": len(graph_obj.nodes_of_kind(NodeKind.FACT)),
                "n_imposes_on_edges": len(graph_obj.edges_of_kind(EdgeKind.IMPOSES_ON)),
                "n_coerces_edges": len(graph_obj.edges_of_kind(EdgeKind.COERCES)),
                "n_treats_as_edges": len(graph_obj.edges_of_kind(EdgeKind.TREATS_AS)),
            }

        timeline = build_timeline(ir, self.bridge)
        verdict_obj, final_vector = self.bridge.judge(ir)
        if tensor_override is not None or tensor_given:
            # the orchestrator built it (or tried and logged why there is none)
            tensor = tensor_override
        else:
            from erisml_compiler.erisml_backend.v3_bridge import compile_to_v3_tensor

            tensor = compile_to_v3_tensor(ir, rank=self.tensor_rank)

        per_party: dict[str, str] = {}
        pp = getattr(verdict_obj, "per_party", None) or {}
        for k, v in pp.items():
            per_party[k] = getattr(v, "verdict", None) or str(v)

        fairness: dict[str, float] = {}
        f = getattr(verdict_obj, "fairness", None)
        if f is not None:
            for attr in ("gini", "gini_harm", "worst_off", "worst_off_harm"):
                val = getattr(f, attr, None)
                if val is not None:
                    fairness[attr] = float(val)

        return ProjectionResult(
            framework=self.framework,
            verdict=verdict_obj.verdict,
            confidence=float(getattr(verdict_obj, "confidence", 1.0)),
            findings=[],
            framework_specific={
                "moral_tensor_v3": tensor,
                "moral_vector": final_vector,
                "timeline": timeline,
                "deme_verdict": verdict_obj,
                "per_party_verdicts": per_party,
                "fairness_metrics": fairness,
                "tensor_rank": self.tensor_rank,
            },
            metadata={
                "deme_profile": self.bridge.profile_name,
                "ethos_applied": self.bridge.ethos_applied,
                "graph_aware": graph_obj is not None,
                "graph_summary": graph_summary,
            },
        )
