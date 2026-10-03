"""Processes: a scene's protocols as BPMN 2.0, checked against the scene, exported, and traced.

    from erisml_compiler.process import load_all, to_bpmn, trace

``model`` loads and checks a scene's ``extra["processes"]``; ``bpmn`` exports one as BPMN 2.0 XML
with its diagram; ``importer`` reads a BPMN 2.0 process back (from any modeler) under the same
checks; ``trace`` maps an agent's records onto the process for a live view. ``erisml.json`` is the
moddle descriptor that lets bpmn-js and Camunda Modeler keep ``erisml:binding`` when they save.
"""

from erisml_compiler.process.bpmn import to_bpmn
from erisml_compiler.process.importer import from_bpmn, import_bpmn
from erisml_compiler.process.model import Flow, Node, Process, TokenChecker, load, load_all
from erisml_compiler.process.trace import Step, trace

__all__ = [
    "Flow",
    "Node",
    "Process",
    "Step",
    "TokenChecker",
    "from_bpmn",
    "import_bpmn",
    "load",
    "load_all",
    "to_bpmn",
    "trace",
]
