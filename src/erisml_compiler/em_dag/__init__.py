"""Ethical Module DAG (EM-DAG).

Per ErisML book Chapter 19, the DEME evaluator is a directed acyclic graph of
Ethical Modules, each evaluating a focused ethical concern, with dependency
edges that determine the topological evaluation order. The DAG topology IS the
DEME profile -- different deployments use different DAGs.

This package provides:
    - `EthicalModule` abstract base
    - `EMOutput` (re-exported from `ir.schemas`)
    - `EMDAG` graph class with topological evaluation
    - Concrete EMs in `em_dag.modules`
    - Profile loader for YAML-defined DAGs
"""

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.em_dag.dag import EMDAG, load_profile

__all__ = ["EthicalModule", "EMDAG", "load_profile"]
