"""BPMN 2.0 import: a process drawn in any modeler becomes an ErisML process only if it passes every
check a declared one does. Export then import is the identity on checked processes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from erisml_compiler.process import import_bpmn, load_all, to_bpmn
from tests.test_process_bpmn import extra

BPMN = "http://www.omg.org/spec/BPMN/20100524/MODEL"
TOKENS = "https://erisml.org/condition-token"


def exported() -> str:
    return to_bpmn(load_all(extra())["ladder"], scene_name="home")


def test_export_then_import_is_the_identity():
    original = load_all(extra())["ladder"]
    proc, decl = import_bpmn(to_bpmn(original), extra())
    assert proc == original
    # and the declaration is what a scene would hold: loading it again changes nothing
    assert load_all(extra(process=decl))["ladder"] == original


def test_the_declaration_survives_yaml():
    _, decl = import_bpmn(exported(), extra())
    again = yaml.safe_load(yaml.safe_dump(decl, sort_keys=False))
    assert load_all(extra(process=again))["ladder"] == load_all(extra())["ladder"]


MODELER = f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="{BPMN}" xmlns:erisml="https://erisml.org/schema/bpmn/1.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" id="d" targetNamespace="x">
  <bpmn:message id="Message_1" name="check_in_answered"/>
  <bpmn:process id="drawn" name="Drawn in a modeler" isExecutable="false">
    <bpmn:laneSet id="ls">
      <bpmn:lane id="L1" name="robot"><bpmn:flowNodeRef>S</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>T</bpmn:flowNodeRef><bpmn:flowNodeRef>X</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>E1</bpmn:flowNodeRef></bpmn:lane>
      <bpmn:lane id="L2" name="margaret"><bpmn:flowNodeRef>M</bpmn:flowNodeRef>
        <bpmn:flowNodeRef>E2</bpmn:flowNodeRef></bpmn:lane>
    </bpmn:laneSet>
    <bpmn:startEvent id="S" name="She falls">
      <bpmn:conditionalEventDefinition id="c1"><bpmn:condition xsi:type="bpmn:tFormalExpression">event:fall</bpmn:condition></bpmn:conditionalEventDefinition>
    </bpmn:startEvent>
    <bpmn:userTask id="T" name="Ask her">
      <bpmn:extensionElements><erisml:binding action="check_in"/></bpmn:extensionElements>
    </bpmn:userTask>
    <bpmn:exclusiveGateway id="X" name="Answered?"/>
    <bpmn:intermediateCatchEvent id="M" name="She answers">
      <bpmn:messageEventDefinition id="md" messageRef="Message_1"/>
    </bpmn:intermediateCatchEvent>
    <bpmn:endEvent id="E1" name="Keep watching"/>
    <bpmn:endEvent id="E2"/>
    <bpmn:textAnnotation id="note"><bpmn:text>drawn by hand</bpmn:text></bpmn:textAnnotation>
    <bpmn:sequenceFlow id="f1" sourceRef="S" targetRef="T"/>
    <bpmn:sequenceFlow id="f2" sourceRef="T" targetRef="X"/>
    <bpmn:sequenceFlow id="f3" name="she answered" sourceRef="X" targetRef="M">
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression" language="{TOKENS}">event:check_in_answered</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    <bpmn:sequenceFlow id="f4" sourceRef="X" targetRef="E1">
      <bpmn:conditionExpression xsi:type="bpmn:tFormalExpression">event:check_in_unanswered</bpmn:conditionExpression>
    </bpmn:sequenceFlow>
    <bpmn:sequenceFlow id="f5" sourceRef="M" targetRef="E2"/>
  </bpmn:process>
</bpmn:definitions>"""


def test_a_process_drawn_in_a_modeler_imports():
    proc, decl = import_bpmn(MODELER, extra())
    assert proc.id == "drawn" and proc.lanes == ["robot", "margaret"]
    assert proc.nodes["T"].action == "check_in" and proc.nodes["M"].system
    assert proc.nodes["S"].trigger == "fall" and not proc.nodes["S"].system
    assert [f.name for f in proc.flows if f.when][0] == "she answered"
    assert "note" not in proc.nodes


def _refused(xml: str, match: str, **kw) -> None:
    with pytest.raises(ValueError, match=match):
        import_bpmn(xml, extra(), **kw)


@pytest.mark.parametrize(
    "old,new,match",
    [
        # the runtime gives these no meaning
        ('<bpmn:exclusiveGateway id="X"', '<bpmn:parallelGateway id="X"', "parallel gateway"),
        ("userTask", "scriptTask", "script task"),
        # a condition the scene cannot evaluate
        (f'language="{TOKENS}"', 'language="http://camunda.org/juel"', "only ErisML"),
        # a task the robot would run must be one of its capabilities
        ('<erisml:binding action="check_in"/>', "", "needs an erisml:binding action"),
        ('action="check_in"', 'action="sing"', "not a capability"),
        # a start must say what starts it
        ("event:fall</bpmn:condition>", "she:fell</bpmn:condition>", "one event: token"),
        # a message is an authenticated channel: a perceived event cannot be one
        ('name="check_in_answered"/>', 'name="help_requested"/>', "not a system event"),
        ('messageRef="Message_1"', 'messageRef="Message_9"', "names no message"),
        # the shape
        ('targetRef="E2"/>', 'targetRef="nowhere"/>', "undeclared node"),
    ],
)
def test_a_drawing_the_runtime_cannot_honour_is_refused(old, new, match):
    xml = MODELER.replace(old, new)
    assert xml != MODELER
    _refused(xml, match)


def test_containment_holds_for_a_drawn_process():
    """Someone draws emergency services behind the robot's own reading of a cry for help: refused,
    exactly as it would be in the scene."""
    xml = exported().replace(
        "latest:monitoring_center_reply=unavailable latest:governor_ruling=elevate",
        "event:help_requested",
    )
    _refused(xml, "governed")


def test_a_duration_edited_in_the_modeler_is_refused():
    xml = exported()
    assert "PT10M" in xml
    _refused(xml.replace("PT10M", "PT2M"), "change the interval in the scene")


def test_a_timer_without_the_scenes_interval_is_refused():
    xml = exported().replace('after="inactivity_limits_minutes.on_floor"', "")
    _refused(xml, "scene owns the numbers")


def test_a_dtd_is_refused_before_parsing():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>' + MODELER.split("?>", 1)[1]
    _refused(bomb, "DTD or entity")


def test_not_bpmn_is_refused():
    _refused("<html/>", "not a BPMN 2.0 document")
    _refused("<bpmn:definitions", "not well-formed")


def test_a_document_with_two_processes_must_name_one():
    two = MODELER.replace(
        "</bpmn:definitions>",
        '<bpmn:process id="other"><bpmn:startEvent id="S2"/></bpmn:process></bpmn:definitions>',
    )
    _refused(two, "name one")
    proc, _ = import_bpmn(two, extra(), process_id="drawn")
    assert proc.id == "drawn"


def test_the_moddle_descriptor_declares_the_binding():
    path = Path(__file__).parent.parent / "src" / "erisml_compiler" / "process" / "erisml.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["uri"] == "https://erisml.org/schema/bpmn/1.0" and d["prefix"] == "erisml"
    props = {p["name"] for p in d["types"][0]["properties"]}
    assert props == {"action", "trigger", "after", "unit", "external"}


def test_the_cli_imports_and_checks(tmp_path):
    from erisml_compiler.cli import cli

    scene = tmp_path / "home.json"
    scene.write_text(json.dumps({"extra": extra()}), encoding="utf-8")
    src = tmp_path / "drawn.bpmn"
    src.write_text(MODELER, encoding="utf-8")
    r = CliRunner().invoke(cli, ["bpmn-import", str(src), "--scene", str(scene)])
    assert r.exit_code == 0, r.output
    assert yaml.safe_load(r.output)["drawn"]["nodes"][1]["action"] == "check_in"
    src.write_text(MODELER.replace("exclusiveGateway", "parallelGateway"), encoding="utf-8")
    r = CliRunner().invoke(cli, ["bpmn-import", str(src), "--scene", str(scene)])
    assert r.exit_code == 1 and "parallel gateway" in r.output
