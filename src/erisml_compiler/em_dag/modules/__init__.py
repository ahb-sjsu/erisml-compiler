"""Concrete Ethical Modules.

Each module evaluates one MoralVector dimension. Modules with no dependencies
(harm, rights, fairness, epistemic, legitimacy) evaluate raw facts;
dependent modules (autonomy depends on legitimacy; fidelity depends on
legitimacy; externality depends on harm; care depends on harm; repair depends
on most others) compose upstream outputs.
"""

from erisml_compiler.em_dag.modules.autonomy import AutonomyEM
from erisml_compiler.em_dag.modules.care import CareEM
from erisml_compiler.em_dag.modules.epistemic import EpistemicEM
from erisml_compiler.em_dag.modules.externality import ExternalityEM
from erisml_compiler.em_dag.modules.fairness import FairnessEM
from erisml_compiler.em_dag.modules.fidelity import FidelityEM
from erisml_compiler.em_dag.modules.harm import HarmEM
from erisml_compiler.em_dag.modules.legitimacy import LegitimacyEM
from erisml_compiler.em_dag.modules.repair import RepairEM
from erisml_compiler.em_dag.modules.rights import RightsEM

__all__ = [
    "AutonomyEM",
    "CareEM",
    "EpistemicEM",
    "ExternalityEM",
    "FairnessEM",
    "FidelityEM",
    "HarmEM",
    "LegitimacyEM",
    "RepairEM",
    "RightsEM",
]
