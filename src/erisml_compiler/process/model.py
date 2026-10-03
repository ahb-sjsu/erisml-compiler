"""Processes: a scene's protocols as BPMN-shaped graphs, checked against the scene.

A scene may declare its protocols (an escalation ladder, a care routine) in ``extra["processes"]``:

    escalation_ladder:
      name: The escalation ladder
      lanes: [robot, margaret, monitoring_center]
      nodes:
        - {id: s_fall, kind: start, lane: robot, trigger: fall, name: She falls}
        - {id: t_ask, kind: task, lane: robot, action: check_in, name: Ask her}
        - {id: x_ans, kind: xor, lane: robot, name: Did she answer?}
        - {id: m_ans, kind: message, lane: margaret, trigger: check_in_answered}
        - {id: w_wait, kind: timer, lane: robot, after: inactivity_limits_minutes.on_floor, unit: minutes}
        - {id: t_ext, kind: task, lane: monitoring_center, external: true, name: The operator calls her}
        - {id: e_ok, kind: end, lane: robot}
      flows:
        - {from: s_fall, to: t_ask}
        - {from: x_ans, to: e_ok, when: "event:check_in_answered"}

A process is a view of the scene, never a second source of authority. Every check below is made
when the process loads, and a process that fails one is refused:

- a task in the agent's lane is one of the scene's capabilities, and still runs only if the scene
  allows it and the output gate passes it; tasks in other lanes are `external` (another party's);
- a start or message event's trigger is a declared event type; a message is an authenticated
  channel, so its trigger must be a system event;
- a timer waits for an interval the scene declares (`after`: a dotted path into ``extra``);
- a flow's `when` is a condition token the scene can evaluate (event:, latest:, cond:, state:,
  stratum:, with not:);
- containment: a flow into a task whose action is governed (elevated or governed) rests only on
  structural tokens, those no model's reading can make true;
- every node is reachable from a start event, and every node but an end event has a way on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

KINDS = ("start", "end", "task", "xor", "message", "timer")
_NCNAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    lane: str
    name: str = ""
    action: str = ""  # task
    external: bool = False  # task: another party's
    trigger: str = ""  # start, message: "type" or "type=content"
    after: str = ""  # timer: a dotted path into extra
    seconds: float = 0.0  # timer: the resolved interval
    system: bool = False  # start, message: the trigger is a system event (an authenticated channel)


@dataclass(frozen=True)
class Flow:
    id: str
    source: str
    target: str
    when: str = ""


@dataclass
class Process:
    id: str
    name: str
    lanes: list[str]
    nodes: dict[str, Node]
    flows: list[Flow] = field(default_factory=list)

    def outgoing(self, node_id: str) -> list[Flow]:
        return [f for f in self.flows if f.source == node_id]

    def incoming(self, node_id: str) -> list[Flow]:
        return [f for f in self.flows if f.target == node_id]


def _resolve(extra: dict[str, Any], path: str) -> Any:
    cur: Any = extra
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


class TokenChecker:
    """Whether a condition token can be evaluated in this scene, and whether it is structural:
    true or false only by system events (an authenticated channel, a measurement, a ruling)."""

    def __init__(self, extra: dict[str, Any]):
        self.types = extra.get("event_types") or {}
        self.conditions = extra.get("conditions") or {}
        self.strata = extra.get("strata") or {}

    def _system(self, typ: str) -> bool:
        return (self.types.get(typ) or {}).get("source") == "system"

    def check(self, token: str, depth: int = 0) -> bool:
        """Raises ValueError if the token is not evaluable; returns whether it is structural."""
        if depth > 16:
            raise ValueError(f"condition {token!r} recurses too deeply")
        t = token[4:] if token.startswith("not:") else token
        kind, _, rest = t.partition(":")
        if kind in ("event", "latest"):
            typ = rest.partition("=")[0]
            if typ not in self.types:
                raise ValueError(f"token {token!r} names the undeclared event type {typ!r}")
            return self._system(typ)
        if kind == "cond":
            if rest not in self.conditions:
                raise ValueError(f"token {token!r} names the undefined condition {rest!r}")
            return all(self.check(x, depth + 1) for x in self.conditions[rest])
        if kind == "state":
            return True  # the runtime's machines, stepped only by declared rules
        if kind == "stratum":
            name, _, state = rest.partition("=")
            st = self.strata.get(name)
            if not st or state not in (st.get("states") or []):
                raise ValueError(f"token {token!r} names an undeclared stratum or state")
            # membership changes only through gates into or out of the state: structural iff all
            # of those fire on system events
            touching = [
                g
                for g in st.get("gates") or []
                if g.get("to") == state or state in (g.get("from") or st["states"])
            ]
            return all(self._system(str(g.get("trigger", "")).partition("=")[0]) for g in touching)
        raise ValueError(f"unknown condition token {token!r}")


def load(pid: str, decl: dict[str, Any], extra: dict[str, Any]) -> Process:
    """One declared process, checked against the scene (see the module docstring)."""
    agent = extra.get("agent")
    caps = {c["action"]: c for c in extra.get("capabilities") or []}
    types = extra.get("event_types") or {}
    tokens = TokenChecker(extra)
    lanes = list(decl.get("lanes") or [])
    if not lanes:
        raise ValueError(f"process {pid!r}: no lanes")
    nodes: dict[str, Node] = {}
    for n in decl.get("nodes") or []:
        nid, kind, lane = str(n.get("id", "")), n.get("kind"), n.get("lane", lanes[0])
        where = f"process {pid!r}, node {nid!r}"
        if not nid or nid in nodes:
            raise ValueError(f"{where}: missing or repeated id")
        if kind not in KINDS:
            raise ValueError(f"{where}: kind {kind!r} is not one of {KINDS}")
        if lane not in lanes:
            raise ValueError(f"{where}: lane {lane!r} is not declared")
        seconds = 0.0
        if not _NCNAME.fullmatch(nid):
            raise ValueError(
                f"{where}: an id must be an XML name (a letter or _, then letters, digits, _ . -)"
            )
        if kind == "task" and not n.get("external"):
            if lane != (agent or lanes[0]):
                raise ValueError(f"{where}: a task in another party's lane must be external")
            if n.get("action") not in caps:
                raise ValueError(
                    f"{where}: action {n.get('action')!r} is not a capability of the scene"
                )
        if kind in ("start", "message"):
            typ = str(n.get("trigger", "")).partition("=")[0]
            if typ not in types:
                raise ValueError(f"{where}: trigger {typ!r} is not a declared event type")
            if kind == "message" and (types[typ] or {}).get("source") != "system":
                raise ValueError(
                    f"{where}: a message is an authenticated channel; {typ!r} is not a system event"
                )
        if kind == "timer":
            try:
                value = float(_resolve(extra, str(n.get("after", ""))))
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"{where}: after {n.get('after')!r} is not an interval the scene declares"
                ) from None
            seconds = value * {"seconds": 1, "minutes": 60, "hours": 3600}[n.get("unit", "seconds")]
        nodes[nid] = Node(
            id=nid,
            kind=kind,
            lane=lane,
            name=str(n.get("name", "")),
            action=str(n.get("action", "")),
            external=bool(n.get("external")),
            trigger=str(n.get("trigger", "")),
            after=str(n.get("after", "")),
            seconds=seconds,
            system=kind in ("start", "message")
            and (types.get(str(n.get("trigger", "")).partition("=")[0]) or {}).get("source")
            == "system",
        )
    flows = []
    for i, f in enumerate(decl.get("flows") or []):
        src, dst, when = f.get("from"), f.get("to"), str(f.get("when", ""))
        where = f"process {pid!r}, flow {src!r} -> {dst!r}"
        if src not in nodes or dst not in nodes:
            raise ValueError(f"{where}: names an undeclared node")
        structural = all(tokens.check(t) for t in when.split() if t) if when else False
        tgt = nodes[dst]
        cap = caps.get(tgt.action, {})
        if tgt.kind == "task" and not tgt.external and (cap.get("elevated") or cap.get("governed")):
            if not structural:
                raise ValueError(
                    f"{where}: {tgt.action!r} is governed, so the flow into it must rest only on system "
                    f"events (when: {when!r})"
                )
        flows.append(Flow(id=f"{pid}_f{i}", source=src, target=dst, when=when))
    proc = Process(id=pid, name=str(decl.get("name", pid)), lanes=lanes, nodes=nodes, flows=flows)
    for f in flows:
        if nodes[f.target].kind == "start":
            raise ValueError(f"process {pid!r}: a flow into the start event {f.target!r}")
        if nodes[f.source].kind == "end":
            raise ValueError(f"process {pid!r}: a flow out of the end event {f.source!r}")
    starts = [n for n in nodes.values() if n.kind == "start"]
    if not starts:
        raise ValueError(f"process {pid!r}: no start event")
    seen, todo = set(), [s.id for s in starts]
    while todo:
        x = todo.pop()
        if x not in seen:
            seen.add(x)
            todo += [f.target for f in proc.outgoing(x)]
    if set(nodes) - seen:
        raise ValueError(
            f"process {pid!r}: unreachable from any start: {sorted(set(nodes) - seen)}"
        )
    stuck = [n.id for n in nodes.values() if n.kind != "end" and not proc.outgoing(n.id)]
    if stuck:
        raise ValueError(f"process {pid!r}: no way on from {stuck}")
    return proc


def load_all(extra: dict[str, Any]) -> dict[str, Process]:
    """Every process a scene declares, checked."""
    return {pid: load(pid, d, extra) for pid, d in (extra.get("processes") or {}).items()}
