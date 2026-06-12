"""ExternalityEM: evaluates third-party-externality dimension.

Depends on `harm`. Externality = harm imposed on a non-consenting third party.
The harm assessment from `HarmEM` provides the magnitude; this module
filters for the subset of harm where the subject is a non-consenting third
party (stakeholder with role `nonconsenting_third_party` or `bystander`
without explicit consent).
"""
from __future__ import annotations

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.modules._helpers import aggregate_negative, facts_of_kind
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class ExternalityEM(EthicalModule):
    name = "externality"
    dimension = "third_party_externality"
    dependencies: tuple[str, ...] = ("harm",)

    def evaluate(self, ir: CompilerIR, upstream: dict[str, EMOutput]) -> EMOutput:
        externality_facts = facts_of_kind(ir, "externality")
        # Identify non-consenting third parties.
        third_parties = {
            s.id for s in ir.stakeholders
            if "nonconsenting_third_party" in s.roles
            or (s.consent_status in ("not_obtained", "coerced") and "bystander" in s.roles)
        }
        # Filter externality facts to those affecting third parties.
        relevant = [
            f for f in externality_facts
            if any(sid in third_parties for sid in f.subjects)
        ]
        if not relevant:
            # Fall back to all externality facts.
            relevant = externality_facts
        score = aggregate_negative(relevant, explanation_prefix="Third-party externality: ")
        return EMOutput(
            module_name=self.name,
            score=score,
            contributing_facts=[f.id for f in relevant],
            upstream_dependencies=["harm"],
            notes=f"third_parties={sorted(third_parties)}" if third_parties else None,
        )
