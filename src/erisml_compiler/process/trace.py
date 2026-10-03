"""Which BPMN elements a run activated, for a live view of a process (process.bpmn).

A run is the agent's record stream: events stepped into the scene and actions performed. Each record
activates the nodes it matches: a start or message event whose trigger is the event's type (and
content, when the trigger names one), or a task whose action was performed. Between two activated
nodes, the path the process took is recovered through the gateways and timers the run does not
record (which are then activated too), shortest first.

The trace is a view of what happened; it decides nothing. A task appears in it only if the agent
performed the action, and the agent performed it only if the scene allowed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from erisml_compiler.process.model import Node, Process


@dataclass(frozen=True)
class Step:
    element: str  # a node or flow id
    kind: str  # node kind, or "flow"
    record: int  # index of the record that activated it


def _matches(n: Node, rec: dict[str, Any]) -> bool:
    if n.kind in ("start", "message") and rec.get("type"):
        typ, _, content = n.trigger.partition("=")
        return rec["type"] == typ and (not content or str(rec.get("content") or "") == content)
    if n.kind == "task" and not n.external:
        return rec.get("type") == "action_performed" and rec.get("content") == n.action
    return False


def _path(proc: Process, src: str, dst: str, depth: int = 4) -> list[str] | None:
    """The flows from src to dst, passing only through gateways and timers (which the run does
    not record), shortest first; None if there is no such path."""
    frontier: list[tuple[str, list[str]]] = [(src, [])]
    seen = {src}
    for _ in range(depth):
        nxt = []
        for node, path in frontier:
            for f in proc.outgoing(node):
                if f.target == dst:
                    return path + [f.id]
                mid = proc.nodes[f.target]
                if mid.kind in ("xor", "timer") and mid.id not in seen:
                    seen.add(mid.id)
                    nxt.append((mid.id, path + [f.id, mid.id]))
        frontier = nxt
    return None


def trace(proc: Process, records: Iterable[dict[str, Any]]) -> list[Step]:
    """The process elements the records activated, in order."""
    steps: list[Step] = []
    last: str | None = None
    for i, rec in enumerate(records):
        for n in proc.nodes.values():
            if not _matches(n, rec):
                continue
            if last is not None:
                for el in _path(proc, last, n.id) or []:
                    kind = proc.nodes[el].kind if el in proc.nodes else "flow"
                    steps.append(Step(el, kind, i))
            steps.append(Step(n.id, n.kind, i))
            last = n.id
            out = proc.outgoing(n.id)
            if len(out) == 1 and proc.nodes[out[0].target].kind == "end":  # its only way on ends it
                steps += [Step(out[0].id, "flow", i), Step(out[0].target, "end", i)]
            break
    return steps
