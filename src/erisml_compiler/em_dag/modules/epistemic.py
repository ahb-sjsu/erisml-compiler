"""EpistemicEM: evaluates epistemic-quality dimension. No upstream deps."""

from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class EpistemicEM(EthicalModule):
    name = "epistemic"
    dimension = "epistemic_quality"
    dependencies: tuple[str, ...] = ()

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        # Epistemic violations: deception, manipulation, withholding evidence.
        # In the geometric framework, deception is permitted toward illegitimate
        # authority (see FidelityEM and the canonical-form mapping); the
        # epistemic module records the deception itself, while the verdict
        # layer decides whether it is justified.
        truth_facts = facts_of_kind(ir, "truth")
        deception_facts = facts_of_kind(ir, "deception")
        all_facts = truth_facts + deception_facts
        bad = [
            f
            for f in all_facts
            if any(
                kw in (f.description or "").lower() for kw in ("decept", "lie", "withh", "manipul")
            )
        ]
        score = aggregate_negative(bad, explanation_prefix="Epistemic assessment: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in bad],
            upstream_dependencies=[],
            notes=None,
        )
