"""Export a checked process (process.model) as standard BPMN 2.0 XML, with diagram interchange.

The output is plain OMG BPMN 2.0 (validated in the tests against the OMG's own schemas), so it
opens in any BPMN tool (bpmn.io / bpmn-js, Camunda Modeler). ErisML's meaning travels in the
standard places:

- the process is one pool (the household) with a lane per party;
- a task carries its capability in an ``erisml:binding`` extension element;
- a start event triggered by a system event is a message start; one triggered by a perceived event
  (the classifier's reading) is a conditional start whose condition is that event's token;
- a message is an intermediate message catch event: an authenticated channel;
- a timer is an intermediate timer catch event with an ISO 8601 duration resolved from the scene;
- an exclusive gateway's outgoing flows carry the ErisML condition tokens as formal expressions
  (language ``https://erisml.org/condition-token``).

Layout is layered: a node's column is its breadth-first distance from a start event, its row is its
lane; edges into an earlier column (loops) are routed over the top.
"""

from __future__ import annotations

from collections import deque
from xml.etree import ElementTree as ET

from erisml_compiler.process.model import Node, Process

NS = {
    "bpmn": "http://www.omg.org/spec/BPMN/20100524/MODEL",
    "bpmndi": "http://www.omg.org/spec/BPMN/20100524/DI",
    "dc": "http://www.omg.org/spec/DD/20100524/DC",
    "di": "http://www.omg.org/spec/DD/20100524/DI",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "erisml": "https://erisml.org/schema/bpmn/1.0",
}
TOKEN_LANGUAGE = "https://erisml.org/condition-token"
for _p, _u in NS.items():
    ET.register_namespace(_p, _u)

SIZE = {
    "task": (120, 80),
    "xor": (50, 50),
    "start": (36, 36),
    "end": (36, 36),
    "message": (36, 36),
    "timer": (36, 36),
}
COL_W, ROW_H, POOL_X, POOL_Y, LABEL_W, PAD = 170, 100, 40, 40, 30, 40


def _q(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def _duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return (
        "PT"
        + (f"{h}H" if h else "")
        + (f"{m}M" if m else "")
        + (f"{sec}S" if sec or not (h or m) else "")
    )


def _layers(proc: Process) -> dict[str, int]:
    layer: dict[str, int] = {}
    q = deque()
    for n in proc.nodes.values():
        if n.kind == "start":
            layer[n.id] = 0
            q.append(n.id)
    while q:
        x = q.popleft()
        for f in proc.outgoing(x):
            if f.target not in layer:
                layer[f.target] = layer[x] + 1
                q.append(f.target)
    return layer


def _layout(
    proc: Process,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, tuple[float, float]], tuple]:
    """Node bounds, lane (y, height), and the pool's bounds."""
    layer = _layers(proc)
    stack: dict[tuple[str, int], int] = {}
    slot: dict[str, int] = {}
    for nid in sorted(proc.nodes, key=lambda i: (layer[i], list(proc.nodes).index(i))):
        key = (proc.nodes[nid].lane, layer[nid])
        slot[nid] = stack.get(key, 0)
        stack[key] = slot[nid] + 1
    lanes: dict[str, tuple[float, float]] = {}
    y = POOL_Y
    for lane in proc.lanes:
        rows = max([stack[k] for k in stack if k[0] == lane] or [1])
        lanes[lane] = (y, rows * ROW_H + PAD)
        y += lanes[lane][1]
    bounds = {}
    for nid, n in proc.nodes.items():
        w, h = SIZE[n.kind]
        cx = POOL_X + LABEL_W + PAD + layer[nid] * COL_W + 60
        cy = lanes[n.lane][0] + PAD / 2 + slot[nid] * ROW_H + ROW_H / 2
        bounds[nid] = (cx - w / 2, cy - h / 2, w, h)
    width = LABEL_W + PAD + (max(layer.values()) + 1) * COL_W + PAD
    return bounds, lanes, (POOL_X, POOL_Y, width, y - POOL_Y)


def _waypoints(b: dict, f, back: bool) -> list[tuple[float, float]]:
    sx, sy, sw, sh = b[f.source]
    tx, ty, tw, th = b[f.target]
    if back:
        top = min(sy, ty) - 25
        return [(sx + sw / 2, sy), (sx + sw / 2, top), (tx + tw / 2, top), (tx + tw / 2, ty)]
    s = (sx + sw, sy + sh / 2)
    t = (tx, ty + th / 2)
    if abs(s[1] - t[1]) < 0.5:
        return [s, t]
    mid = (s[0] + t[0]) / 2
    return [s, (mid, s[1]), (mid, t[1]), t]


def to_bpmn(proc: Process, scene_name: str = "") -> str:
    """The process as a BPMN 2.0 XML document (one pool, a lane per party, with its diagram)."""
    b = ET.Element(
        _q("bpmn", "definitions"),
        {
            "id": f"{proc.id}_definitions",
            "targetNamespace": "https://erisml.org/processes",
            "exporter": "erisml-compiler",
            "exporterVersion": "1",
        },
    )
    collab = ET.SubElement(b, _q("bpmn", "collaboration"), {"id": f"{proc.id}_collaboration"})
    ET.SubElement(
        collab,
        _q("bpmn", "participant"),
        {"id": f"{proc.id}_pool", "name": scene_name or proc.name, "processRef": proc.id},
    )
    messages = sorted({n.trigger.partition("=")[0] for n in proc.nodes.values() if n.system})
    for m in messages:
        ET.SubElement(b, _q("bpmn", "message"), {"id": f"msg_{m}", "name": m})
    p = ET.SubElement(
        b, _q("bpmn", "process"), {"id": proc.id, "name": proc.name, "isExecutable": "false"}
    )
    lane_set = ET.SubElement(p, _q("bpmn", "laneSet"), {"id": f"{proc.id}_lanes"})
    for lane in proc.lanes:
        el = ET.SubElement(
            lane_set, _q("bpmn", "lane"), {"id": f"{proc.id}_lane_{lane}", "name": lane}
        )
        for n in proc.nodes.values():
            if n.lane == lane:
                ET.SubElement(el, _q("bpmn", "flowNodeRef")).text = n.id

    def node_el(n: Node) -> None:
        tag = {
            "start": "startEvent",
            "end": "endEvent",
            "task": "task",
            "xor": "exclusiveGateway",
            "message": "intermediateCatchEvent",
            "timer": "intermediateCatchEvent",
        }[n.kind]
        e = ET.SubElement(
            p, _q("bpmn", tag), {"id": n.id, "name": n.name or n.action or n.trigger or n.kind}
        )
        binding = {
            k: v for k, v in (("action", n.action), ("trigger", n.trigger), ("after", n.after)) if v
        }
        if n.external:
            binding["external"] = "true"
        if binding:
            ext = ET.SubElement(e, _q("bpmn", "extensionElements"))
            ET.SubElement(ext, _q("erisml", "binding"), binding)
        for f in proc.incoming(n.id):
            ET.SubElement(e, _q("bpmn", "incoming")).text = f.id
        for f in proc.outgoing(n.id):
            ET.SubElement(e, _q("bpmn", "outgoing")).text = f.id
        if n.system:
            ET.SubElement(
                e,
                _q("bpmn", "messageEventDefinition"),
                {"id": f"{n.id}_def", "messageRef": f"msg_{n.trigger.partition('=')[0]}"},
            )
        elif n.kind == "start":
            cd = ET.SubElement(e, _q("bpmn", "conditionalEventDefinition"), {"id": f"{n.id}_def"})
            ET.SubElement(
                cd,
                _q("bpmn", "condition"),
                {_q("xsi", "type"): "bpmn:tFormalExpression", "language": TOKEN_LANGUAGE},
            ).text = f"event:{n.trigger}"
        elif n.kind == "timer":
            td = ET.SubElement(e, _q("bpmn", "timerEventDefinition"), {"id": f"{n.id}_def"})
            ET.SubElement(
                td, _q("bpmn", "timeDuration"), {_q("xsi", "type"): "bpmn:tFormalExpression"}
            ).text = _duration(n.seconds)

    for n in proc.nodes.values():
        node_el(n)
    for f in proc.flows:
        e = ET.SubElement(
            p,
            _q("bpmn", "sequenceFlow"),
            {
                "id": f.id,
                "sourceRef": f.source,
                "targetRef": f.target,
                **({"name": f.when} if f.when else {}),
            },
        )
        if f.when:
            ET.SubElement(
                e,
                _q("bpmn", "conditionExpression"),
                {_q("xsi", "type"): "bpmn:tFormalExpression", "language": TOKEN_LANGUAGE},
            ).text = f.when

    bounds, lanes, pool = _layout(proc)
    layer = _layers(proc)
    diagram = ET.SubElement(b, _q("bpmndi", "BPMNDiagram"), {"id": f"{proc.id}_diagram"})
    plane = ET.SubElement(
        diagram,
        _q("bpmndi", "BPMNPlane"),
        {"id": f"{proc.id}_plane", "bpmnElement": f"{proc.id}_collaboration"},
    )

    def shape(
        elem_id: str, x: float, y: float, w: float, h: float, horizontal: bool = False
    ) -> None:
        attrs = {"id": f"{elem_id}_di", "bpmnElement": elem_id}
        if horizontal:
            attrs["isHorizontal"] = "true"
        s = ET.SubElement(plane, _q("bpmndi", "BPMNShape"), attrs)
        ET.SubElement(
            s,
            _q("dc", "Bounds"),
            {"x": f"{x:g}", "y": f"{y:g}", "width": f"{w:g}", "height": f"{h:g}"},
        )

    shape(f"{proc.id}_pool", *pool, horizontal=True)
    for lane in proc.lanes:
        y, h = lanes[lane]
        shape(f"{proc.id}_lane_{lane}", pool[0] + LABEL_W, y, pool[2] - LABEL_W, h, horizontal=True)
    for nid, bb in bounds.items():
        shape(nid, *bb)
    for f in proc.flows:
        edge = ET.SubElement(
            plane, _q("bpmndi", "BPMNEdge"), {"id": f"{f.id}_di", "bpmnElement": f.id}
        )
        for x, y in _waypoints(bounds, f, layer[f.target] <= layer[f.source]):
            ET.SubElement(edge, _q("di", "waypoint"), {"x": f"{x:g}", "y": f"{y:g}"})
    ET.indent(b)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(b, encoding="unicode")
