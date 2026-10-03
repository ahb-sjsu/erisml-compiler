"""Import a BPMN 2.0 process (drawn in bpmn.io, Camunda Modeler, or exported by ``process.bpmn``)
as an ErisML process declaration, checked against the scene exactly as a declared one is.

A BPMN document is a proposal; the scene decides. The importer reads ErisML's meaning from the same
standard places the exporter writes it, and refuses what the runtime cannot honour:

- the pool's lanes are the parties; a node in no lane falls to the first lane (or ``default_lane``
  when the process has none);
- a task's capability is its ``erisml:binding action``; a task in another party's lane with no
  binding is that party's (``external``); a script, business-rule or receive task is refused;
- a start event names its trigger by ``erisml:binding trigger``, else its message (a system event,
  an authenticated channel), else its conditional ``event:`` token;
- an intermediate message catch is a message (its trigger as for a start); an intermediate timer
  catch must name the scene's interval in ``erisml:binding after`` (the scene owns the numbers),
  and a ``timeDuration`` beside it must agree with that interval, so a duration edited in a
  modeler is refused rather than silently ignored;
- a sequence flow's ``conditionExpression`` is an ErisML condition token string; an expression in
  another language (Camunda's ``${...}``, FEEL) is refused;
- parallel, inclusive, complex and event-based gateways, sub-processes, call activities and boundary
  events are refused: the runtime gives them no meaning, and a process that looks as if it waits
  for two branches but does not would mislead its readers.

Then ``model.load`` runs every check (capabilities, declared event types, containment of governed
tasks, reachability). The document is parsed with DTDs refused, so no entity can expand.
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree as ET

from erisml_compiler.process.bpmn import NS, TOKEN_LANGUAGE
from erisml_compiler.process.model import Process, load

MAX_BYTES = 5_000_000
_TASKS = {"task", "userTask", "manualTask", "serviceTask", "sendTask"}
_REFUSED = {
    "parallelGateway": "a parallel gateway (the runtime takes one branch, never all)",
    "inclusiveGateway": "an inclusive gateway",
    "complexGateway": "a complex gateway",
    "eventBasedGateway": "an event-based gateway",
    "subProcess": "a sub-process",
    "adHocSubProcess": "an ad-hoc sub-process",
    "transaction": "a transaction",
    "callActivity": "a call activity",
    "boundaryEvent": "a boundary event",
    "intermediateThrowEvent": "an intermediate throw event",
    "scriptTask": "a script task (the robot runs no scripts from a diagram)",
    "businessRuleTask": "a business-rule task (the scene's norms are the rules)",
    "receiveTask": "a receive task (use an intermediate message catch event)",
}
# elements a modeler writes that carry no behaviour
_IGNORED = {
    "laneSet",
    "sequenceFlow",
    "textAnnotation",
    "association",
    "documentation",
    "extensionElements",
    "dataObject",
    "dataObjectReference",
    "dataStoreReference",
    "group",
    "ioSpecification",
    "property",
}
_DURATION = re.compile(
    r"P(?:(?P<d>\d+(?:\.\d+)?)D)?(?:T(?:(?P<h>\d+(?:\.\d+)?)H)?(?:(?P<m>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)S)?)?"
)


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def _seconds(duration: str) -> float:
    m = _DURATION.fullmatch(duration.strip())
    if not m or duration.strip() in ("P", "PT"):
        raise ValueError(f"timeDuration {duration!r} is not an ISO 8601 duration")
    d, h, mi, s = (float(m.group(k) or 0) for k in "dhms")
    return ((d * 24 + h) * 60 + mi) * 60 + s


def _parse(xml: str | bytes) -> ET.Element:
    raw = xml.encode("utf-8") if isinstance(xml, str) else xml
    if len(raw) > MAX_BYTES:
        raise ValueError(f"the document is {len(raw)} bytes; the limit is {MAX_BYTES}")
    if re.search(rb"<!DOCTYPE|<!ENTITY", raw, re.I):
        raise ValueError("the document declares a DTD or entity; BPMN needs neither")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"not well-formed XML: {exc}") from None


def _binding(el: ET.Element) -> dict[str, str]:
    ext = el.find(_q("bpmn", "extensionElements"))
    b = ext.find(_q("erisml", "binding")) if ext is not None else None
    return dict(b.attrib) if b is not None else {}


def _choose(root: ET.Element, process_id: str | None) -> ET.Element:
    if _local(root.tag) != "definitions" or not root.tag.startswith(f"{{{NS['bpmn']}}}"):
        raise ValueError("not a BPMN 2.0 document (no bpmn:definitions root)")
    procs = root.findall(_q("bpmn", "process"))
    if process_id:
        for p in procs:
            if p.get("id") == process_id:
                return p
        raise ValueError(
            f"no process {process_id!r}; the document has {[p.get('id') for p in procs]}"
        )
    with_nodes = [p for p in procs if any(_local(c.tag) not in _IGNORED for c in p)]
    if len(with_nodes) != 1:
        raise ValueError(
            f"the document has {len(with_nodes)} processes with nodes "
            f"{[p.get('id') for p in with_nodes]}; name one"
        )
    return with_nodes[0]


def from_bpmn(
    xml: str | bytes,
    extra: dict[str, Any],
    process_id: str | None = None,
    default_lane: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """The process's id and its ErisML declaration (the shape ``extra["processes"][id]`` takes).

    The declaration is not yet checked against the scene; ``import_bpmn`` does both."""
    root = _parse(xml)
    proc = _choose(root, process_id)
    pid = str(proc.get("id", ""))
    messages = {
        m.get("id"): str(m.get("name") or m.get("id")) for m in root.iter(_q("bpmn", "message"))
    }
    where = f"process {pid!r}"

    lanes: list[str] = []
    lane_of: dict[str, str] = {}
    for lane in proc.iter(_q("bpmn", "lane")):
        name = str(lane.get("name") or lane.get("id"))
        if name in lanes:
            raise ValueError(f"{where}: two lanes are named {name!r}")
        lanes.append(name)
        for ref in lane.findall(_q("bpmn", "flowNodeRef")):
            lane_of[str(ref.text).strip()] = name
    if not lanes:
        lanes = [default_lane or extra.get("agent") or "agent"]
    agent = extra.get("agent") or lanes[0]

    def trigger_of(el: ET.Element, nid: str, bind: dict[str, str]) -> str:
        msg = el.find(_q("bpmn", "messageEventDefinition"))
        cond = el.find(_q("bpmn", "conditionalEventDefinition"))
        named = ""
        if msg is not None:
            ref = msg.get("messageRef")
            if ref not in messages:
                raise ValueError(f"{where}, node {nid!r}: messageRef {ref!r} names no message")
            named = messages[ref]
        elif cond is not None:
            c = cond.find(_q("bpmn", "condition"))
            text = (c.text or "").strip() if c is not None else ""
            if not text.startswith("event:") or " " in text:
                raise ValueError(
                    f"{where}, node {nid!r}: a conditional start must be one event: token, not {text!r}"
                )
            named = text[len("event:") :]
        trigger = bind.get("trigger", named)
        if named and trigger.partition("=")[0] != named.partition("=")[0]:
            raise ValueError(
                f"{where}, node {nid!r}: the binding's trigger {trigger!r} and the event's "
                f"definition {named!r} disagree"
            )
        if not trigger:
            raise ValueError(
                f"{where}, node {nid!r}: names no trigger (an erisml:binding trigger, a message, "
                "or an event: condition)"
            )
        return trigger

    nodes: list[dict[str, Any]] = []
    for el in proc:
        tag, nid = _local(el.tag), str(el.get("id", ""))
        if tag in _IGNORED:
            continue
        if tag in _REFUSED:
            raise ValueError(f"{where}, node {nid!r}: {_REFUSED[tag]} is not supported")
        bind = _binding(el)
        lane = lane_of.get(nid, lanes[0])
        n: dict[str, Any] = {"id": nid, "lane": lane}
        defs = {_local(c.tag) for c in el if _local(c.tag).endswith("EventDefinition")}
        if tag == "startEvent":
            n["kind"] = "start"
            n["trigger"] = trigger_of(el, nid, bind)
        elif tag == "endEvent":
            if defs:
                raise ValueError(
                    f"{where}, node {nid!r}: an end event that throws ({sorted(defs)})"
                )
            n["kind"] = "end"
        elif tag in _TASKS:
            n["kind"] = "task"
            if bind.get("action"):
                n["action"] = bind["action"]
            if bind.get("external") == "true" or ("action" not in n and lane != agent):
                n["external"] = True
            elif "action" not in n:
                raise ValueError(
                    f"{where}, node {nid!r}: a task in the agent's lane needs an erisml:binding "
                    "action naming one of the scene's capabilities"
                )
        elif tag == "exclusiveGateway":
            n["kind"] = "xor"
        elif tag == "intermediateCatchEvent":
            if defs == {"messageEventDefinition"}:
                n["kind"] = "message"
                n["trigger"] = trigger_of(el, nid, bind)
            elif defs == {"timerEventDefinition"}:
                n["kind"] = "timer"
                if not bind.get("after"):
                    raise ValueError(
                        f"{where}, node {nid!r}: a timer must name the scene's interval "
                        "(erisml:binding after); the scene owns the numbers"
                    )
                n["after"] = bind["after"]
                n["unit"] = bind.get("unit", "seconds")
                dur = el.find(f"{_q('bpmn', 'timerEventDefinition')}/{_q('bpmn', 'timeDuration')}")
                if dur is not None and (dur.text or "").strip():
                    n["_duration"] = _seconds(dur.text)
            else:
                raise ValueError(
                    f"{where}, node {nid!r}: an intermediate catch must be one message or one timer, "
                    f"not {sorted(defs) or 'none'}"
                )
        else:
            raise ValueError(f"{where}, node {nid!r}: {tag} is not supported")
        name = str(el.get("name", "")).strip()
        # the exporter labels an unnamed node by its binding or kind; that is not a name
        if name and name not in (n.get("action"), n.get("trigger"), n["kind"]):
            n["name"] = name
        nodes.append(n)

    ids = {n["id"] for n in nodes}
    flows: list[dict[str, Any]] = []
    for el in proc.findall(_q("bpmn", "sequenceFlow")):
        f: dict[str, Any] = {"from": el.get("sourceRef"), "to": el.get("targetRef")}
        if f["from"] not in ids or f["to"] not in ids:
            raise ValueError(f"{where}, flow {el.get('id')!r}: names an undeclared node")
        cond = el.find(_q("bpmn", "conditionExpression"))
        if cond is not None and (cond.text or "").strip():
            lang = cond.get("language")
            if lang not in (None, TOKEN_LANGUAGE):
                raise ValueError(
                    f"{where}, flow {el.get('id')!r}: the condition is in {lang!r}; only ErisML "
                    f"condition tokens ({TOKEN_LANGUAGE}) can be checked against the scene"
                )
            f["when"] = " ".join(cond.text.split())
        name = str(el.get("name", "")).strip()
        if name and name != f.get("when"):
            f["name"] = name
        flows.append(f)

    decl: dict[str, Any] = {"name": str(proc.get("name") or pid), "lanes": lanes, "nodes": nodes}
    decl["flows"] = flows
    return pid, decl


def import_bpmn(
    xml: str | bytes,
    extra: dict[str, Any],
    process_id: str | None = None,
    default_lane: str | None = None,
) -> tuple[Process, dict[str, Any]]:
    """The checked process and its declaration, ready for ``extra["processes"][process.id]``.

    Raises ValueError on anything the importer refuses or ``model.load`` rejects."""
    pid, decl = from_bpmn(xml, extra, process_id, default_lane)
    durations = {n["id"]: n.pop("_duration") for n in decl["nodes"] if "_duration" in n}
    proc = load(pid, decl, extra)
    for nid, secs in durations.items():
        if abs(proc.nodes[nid].seconds - secs) > 1e-6:
            raise ValueError(
                f"process {pid!r}, node {nid!r}: the diagram's timeDuration is {secs:g} s but the "
                f"scene's {proc.nodes[nid].after} is {proc.nodes[nid].seconds:g} s; change the "
                "interval in the scene, not in the diagram"
            )
    return proc, decl
