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

from erisml_compiler.em_dag import EMDAG
from erisml_compiler.erisml_backend.deme_bridge import DEMEBridge
from erisml_compiler.evaluation.moral_vector import (
    build_moral_vector_from_em_outputs,
)
from erisml_compiler.evaluation.tensor_builder import build_timeline
from erisml_compiler.evaluation.tensor_builder_v3 import build_moral_tensor_v3
from erisml_compiler.ir.schemas import CompilerIR
from erisml_compiler.projections.base import Projection, ProjectionResult
from erisml_compiler.projections.substrate import MoralSubstrate


class ConsequentialistProjection(Projection):
    """Per-stakeholder harm/care/etc tensor + Gini-style aggregates.

    This is intentionally a thin orchestrator over the existing
    machinery (EM-DAG -> tensor + DEME). It exists to make the
    framework commitment *explicit* in the architecture: this
    projection *is* the consequentialist-distributive reading, not
    the privileged-and-only reading.
    """

    framework = "consequentialist_distributive"

    def __init__(
        self,
        *,
        dag: EMDAG,
        ethos_weights: dict[str, float] | None = None,
        tensor_rank: int = 2,
    ) -> None:
        self.dag = dag
        self.ethos_weights = ethos_weights
        self.tensor_rank = tensor_rank

    def project(
        self,
        substrate: MoralSubstrate,
        *,
        ir: CompilerIR | None = None,
        tensor_override: Any = None,
        **kwargs: Any,
    ) -> ProjectionResult:
        """Run the EM-DAG over the substrate (via a CompilerIR view).

        For v0 we require the caller to pass the full CompilerIR because
        the EM-DAG modules and the V3 bridge expect a CompilerIR
        (which the substrate is a view of). Future versions will let
        the projection take only a substrate once the EM-DAG is
        refactored to read from the substrate directly.

        `tensor_override` lets the orchestrator inject a tensor built
        via its V3-dispatch logic (strict_v3, higher-rank bridge,
        etc.) instead of having this projection always use the Phase 2
        fallback `build_moral_tensor_v3`. When None, falls back.
        """
        if ir is None:
            raise ValueError(
                "ConsequentialistProjection.project requires the full CompilerIR "
                "via the `ir=` kwarg for v0. (The EM-DAG and V3 bridge still "
                "read from CompilerIR; this constraint will be relaxed in a "
                "future refactor.)"
            )

        timeline = build_timeline(ir, self.dag, ethos_weights=self.ethos_weights)
        em_outputs = self.dag.evaluate(ir)
        final_vector = build_moral_vector_from_em_outputs(
            em_outputs, self.dag, ethos_weights=self.ethos_weights
        )
        if tensor_override is not None:
            tensor = tensor_override
        else:
            tensor = build_moral_tensor_v3(
                ir, em_outputs, self.dag, rank=self.tensor_rank,
                ethos_weights=self.ethos_weights,
            )
        verdict_obj = DEMEBridge(profile_name=self.dag.name).evaluate(ir, final_vector)

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
                "em_outputs": em_outputs,
                "deme_verdict": verdict_obj,
                "per_party_verdicts": per_party,
                "fairness_metrics": fairness,
                "tensor_rank": self.tensor_rank,
            },
            metadata={
                "dag_name": self.dag.name,
                "ethos_applied": bool(self.ethos_weights),
            },
        )
