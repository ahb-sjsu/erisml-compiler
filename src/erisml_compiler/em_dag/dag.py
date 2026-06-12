"""EMDAG: graph container and topological evaluator."""

from __future__ import annotations

import importlib
from collections import deque
from pathlib import Path
from typing import Iterable

import yaml

from erisml_compiler.em_dag.base import EthicalModule
from erisml_compiler.ir.schemas import CompilerIR, EMOutput


class EMDAG:
    """A directed acyclic graph of Ethical Modules.

    Validates: every dependency points to a module in the DAG; no cycles.
    Evaluates: topological order; each module receives its upstream outputs.
    """

    def __init__(self, modules: Iterable[EthicalModule], name: str = "default"):
        self.name = name
        self._modules: dict[str, EthicalModule] = {m.name: m for m in modules}
        self._validate()
        self._topo_order: list[str] = self._topological_sort()

    # ------------------------------------------------------------ structure

    def _validate(self) -> None:
        """Check that all dependencies are present in the DAG."""
        for module in self._modules.values():
            for dep in module.dependencies:
                if dep not in self._modules:
                    raise ValueError(
                        f"Module {module.name!r} declares dependency on "
                        f"{dep!r}, which is not in the DAG"
                    )

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm. Raises if a cycle exists."""
        in_degree: dict[str, int] = {name: 0 for name in self._modules}
        adj: dict[str, list[str]] = {name: [] for name in self._modules}
        for name, m in self._modules.items():
            for dep in m.dependencies:
                adj[dep].append(name)
                in_degree[name] += 1

        queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in adj[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self._modules):
            cycle_nodes = [n for n, d in in_degree.items() if d > 0]
            raise ValueError(f"EM-DAG contains a cycle involving: {cycle_nodes}")
        return order

    # ------------------------------------------------------------ accessors

    @property
    def topological_order(self) -> list[str]:
        return list(self._topo_order)

    @property
    def modules(self) -> dict[str, EthicalModule]:
        return dict(self._modules)

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    # ------------------------------------------------------------ evaluation

    def evaluate(self, ir: CompilerIR) -> dict[str, EMOutput]:
        """Evaluate every module in topological order against the IR.

        Returns a dict keyed by module name with each module's EMOutput.
        Each module sees the outputs of its declared upstream dependencies.
        """
        outputs: dict[str, EMOutput] = {}
        for name in self._topo_order:
            module = self._modules[name]
            upstream = {dep: outputs[dep] for dep in module.dependencies}
            outputs[name] = module.evaluate(ir, upstream)
        return outputs


# ============================================================================
# Profile loading: YAML -> EMDAG
# ============================================================================


def load_profile(path: str | Path) -> EMDAG:
    """Load an EM-DAG profile from a YAML file.

    The YAML schema:

        name: "default"
        modules:
          - module: "erisml_compiler.em_dag.modules.legitimacy:LegitimacyEM"
          - module: "erisml_compiler.em_dag.modules.harm:HarmEM"
          - ...

    Each `module` entry is a `module.path:ClassName` reference that will be
    imported and instantiated with no arguments.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    name = config.get("name", path.stem)
    module_specs = config.get("modules", [])
    instances: list[EthicalModule] = []
    for entry in module_specs:
        spec = entry["module"]
        module_path, class_name = spec.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instances.append(cls())
    return EMDAG(modules=instances, name=name)
