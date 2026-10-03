"""Processes as BPMN 2.0: checked against the scene, exported as valid BPMN, traced from a run.

The export is validated against the OMG's own BPMN 2.0 schemas (tests/resources/bpmn20), so what
the compiler writes opens in any BPMN tool.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from erisml_compiler.process import load, load_all, to_bpmn, trace
from tests.test_scene_runtime import home

XSD = Path(__file__).parent / "resources" / "bpmn20" / "BPMN20.xsd"

TYPES = {
    "fall": {"description": "she falls"},
    "check_in_answered": {"description": "she answered", "source": "system"},
    "check_in_unanswered": {"description": "she did not answer", "source": "system"},
    "monitoring_center_reply": {
        "description": "the centre's answer",
        "source": "system",
        "content": ["on_the_line", "ems_sent", "unavailable"],
    },
    "governor_ruling": {"description": "the governor's ruling", "source": "system"},
    "help_requested": {"description": "she asks for help"},
}
CAPS = [
    {"action": "chores", "elevated": False},
    {"action": "check_in", "elevated": False},
    {"action": "contact_monitoring_center", "elevated": False},
    {"action": "call_emergency_services", "elevated": True},
]
LADDER = {
    "name": "The escalation ladder",
    "lanes": ["robot", "margaret", "monitoring_center"],
    "nodes": [
        {"id": "s_fall", "kind": "start", "lane": "robot", "trigger": "fall", "name": "She falls"},
        {"id": "t_ask", "kind": "task", "lane": "robot", "action": "check_in", "name": "Ask her"},
        {
            "id": "w_wait",
            "kind": "timer",
            "lane": "robot",
            "after": "inactivity_limits_minutes.on_floor",
            "unit": "minutes",
        },
        {"id": "x_ans", "kind": "xor", "lane": "robot", "name": "Did she answer?"},
        {"id": "m_her", "kind": "message", "lane": "margaret", "trigger": "check_in_answered"},
        {"id": "t_centre", "kind": "task", "lane": "robot", "action": "contact_monitoring_center"},
        {
            "id": "t_op",
            "kind": "task",
            "lane": "monitoring_center",
            "external": True,
            "name": "The operator calls her",
        },
        {
            "id": "m_centre",
            "kind": "message",
            "lane": "monitoring_center",
            "trigger": "monitoring_center_reply",
        },
        {"id": "x_centre", "kind": "xor", "lane": "robot", "name": "Centre reachable?"},
        {"id": "t_ems", "kind": "task", "lane": "robot", "action": "call_emergency_services"},
        {"id": "e_ok", "kind": "end", "lane": "robot", "name": "She is fine"},
        {"id": "e_help", "kind": "end", "lane": "robot", "name": "Help is on its way"},
    ],
    "flows": [
        {"from": "s_fall", "to": "t_ask"},
        {"from": "t_ask", "to": "w_wait"},
        {"from": "w_wait", "to": "x_ans"},
        {"from": "x_ans", "to": "m_her", "when": "event:check_in_answered"},
        {"from": "m_her", "to": "e_ok"},
        {"from": "x_ans", "to": "t_centre", "when": "event:check_in_unanswered"},
        {"from": "t_centre", "to": "t_op"},
        {"from": "t_op", "to": "m_centre"},
        {"from": "m_centre", "to": "x_centre"},
        {"from": "x_centre", "to": "e_help", "when": "latest:monitoring_center_reply=ems_sent"},
        {
            "from": "x_centre",
            "to": "t_ems",
            "when": "latest:monitoring_center_reply=unavailable latest:governor_ruling=elevate",
        },
        {"from": "t_ems", "to": "e_help"},
    ],
}


def extra(process=None, **over):
    ir = home(
        event_types=TYPES,
        capabilities=CAPS,
        inactivity_limits_minutes={"on_floor": 10},
        processes={"ladder": process or LADDER},
        **over,
    )
    return ir.extra


def test_the_ladder_loads():
    proc = load_all(extra())["ladder"]
    assert proc.nodes["w_wait"].seconds == 600 and proc.nodes["m_her"].system
    assert not proc.nodes["s_fall"].system  # a fall is perceived, not an authenticated channel


def test_the_export_is_valid_bpmn_2_with_a_shape_for_every_element():
    xmlschema = pytest.importorskip("xmlschema")
    proc = load_all(extra())["ladder"]
    xml = to_bpmn(proc, scene_name="home")
    xmlschema.XMLSchema(str(XSD)).validate(xml)
    root = ET.fromstring(xml)
    di = {e.get("bpmnElement") for e in root.iter() if e.tag.endswith(("BPMNShape", "BPMNEdge"))}
    assert set(proc.nodes) <= di and {f.id for f in proc.flows} <= di
    assert {"ladder_pool", "ladder_lane_robot", "ladder_lane_margaret"} <= di
    assert "PT10M" in xml and "latest:governor_ruling=elevate" in xml


def _break(mutate):
    p = copy.deepcopy(LADDER)
    mutate(p)
    return p


@pytest.mark.parametrize(
    "mutate,match",
    [
        # containment: a governed task behind a flow a model could open
        (
            lambda p: p["flows"].__setitem__(
                10, {"from": "x_centre", "to": "t_ems", "when": "event:help_requested"}
            ),
            "governed",
        ),
        (lambda p: p["flows"].__setitem__(10, {"from": "x_centre", "to": "t_ems"}), "governed"),
        # a message is an authenticated channel
        (lambda p: p["nodes"][4].__setitem__("trigger", "help_requested"), "not a system event"),
        # tasks
        (lambda p: p["nodes"][1].__setitem__("action", "sing"), "not a capability"),
        (lambda p: p["nodes"][6].pop("external"), "must be external"),
        # timers, tokens, references
        (lambda p: p["nodes"][2].__setitem__("after", "nap_minutes"), "interval"),
        (lambda p: p["flows"][3].__setitem__("when", "event:teleported"), "undeclared event type"),
        (lambda p: p["flows"][3].__setitem__("when", "vibes:good"), "unknown condition token"),
        (lambda p: p["flows"].append({"from": "t_ask", "to": "nowhere"}), "undeclared node"),
        # shape
        (
            lambda p: p["nodes"].append(
                {"id": "t_lost", "kind": "task", "lane": "robot", "action": "chores"}
            ),
            "unreachable",
        ),
        (lambda p: p["flows"].append({"from": "t_ask", "to": "s_fall"}), "into the start"),
        (lambda p: p["flows"].pop(11), "no way on"),
        (lambda p: p["nodes"][1].__setitem__("id", "1st"), "XML name"),
    ],
)
def test_a_process_that_breaks_a_rule_is_refused(mutate, match):
    with pytest.raises(ValueError, match=match):
        load("ladder", _break(mutate), extra())


def test_the_trace_follows_the_run_through_gateways_and_timers():
    proc = load_all(extra())["ladder"]
    run = [
        {"type": "fall"},
        {"type": "action_performed", "content": "check_in"},
        {"type": "check_in_answered"},
    ]
    steps = [s.element for s in trace(proc, run)]
    assert steps == [
        "s_fall",
        "ladder_f0",
        "t_ask",
        "ladder_f1",
        "w_wait",
        "ladder_f2",
        "x_ans",
        "ladder_f3",
        "m_her",
        "ladder_f4",
        "e_ok",
    ]
    assert "m_her" not in [s.element for s in trace(proc, run[:2])]


def test_the_trace_shows_only_what_was_performed():
    proc = load_all(extra())["ladder"]
    steps = [s.element for s in trace(proc, [{"type": "fall"}, {"type": "help_requested"}])]
    assert steps == ["s_fall"] and "t_ems" not in steps
