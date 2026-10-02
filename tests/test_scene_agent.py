"""Scene agent: classified events are checked against the vocabulary, choices against the allowed set."""

import json

from erisml_compiler.annotation.llm_extractor import MockLLMAdapter
from erisml_compiler.runtime import SceneRuntime
from erisml_compiler.runtime.agent import SceneAgent
from tests.test_scene_runtime import home

VOCAB = {
    "inactivity_exceeded": {
        "description": "a person has been in one pose too long",
        "content": ["seated", "in_bed"],
    },
    "governor_ruling": {
        "description": "the authority gate's ruling",
        "content": ["elevate", "refuse"],
        "source": "system",
    },
}


def agent(classify: list, choose: dict) -> SceneAgent:
    rt = SceneRuntime(home(event_types=VOCAB, default_action="chores"))
    mock = MockLLMAdapter(
        {
            "You turn a robot's perception facts": json.dumps(classify),
            "You choose the next action": json.dumps(choose),
        }
    )
    return SceneAgent(rt, mock)


def test_undeclared_or_out_of_vocabulary_events_are_rejected_not_stepped():
    a = agent(
        [
            {"type": "dog_attack", "actor": "dog"},
            {"type": "inactivity_exceeded", "content": "standing"},
            {"type": "inactivity_exceeded", "content": "seated"},
        ],
        {"action": "check_in", "reason": "she has been seated too long"},
    )
    d = a.decide({"margaret": {"pose": "seated", "minutes_in_pose": 95}})
    assert [e["content"] for e in d.events] == ["seated"]
    assert len(d.rejected_events) == 2
    assert d.snapshot["obliged"] == ["check_in"] and d.action == "check_in" and not d.fallback


def test_a_choice_outside_the_allowed_set_falls_back_to_the_obligation():
    a = agent(
        [{"type": "inactivity_exceeded", "content": "seated"}],
        {"action": "call_emergency_services"},
    )
    d = a.decide({"margaret": {"pose": "seated"}})
    assert "call_emergency_services" in d.snapshot["prohibited"]
    assert (
        d.action == "check_in"
        and d.fallback
        and d.chooser_rejected == {"proposed": {"action": "call_emergency_services"}}
    )


def test_with_nothing_obliged_the_fallback_is_the_declared_default():
    a = agent([], {"action": "record"})
    d = a.decide({})
    assert d.action == "chores" and d.fallback


def test_an_allowed_choice_is_taken_with_its_args():
    a = agent(
        [], {"action": "chores", "args": {"task": "tidy"}, "reason": "nothing needs attention"}
    )
    d = a.decide({})
    assert (d.action, d.args, d.fallback) == ("chores", {"task": "tidy"}, False)


def test_system_events_are_not_offered_to_the_classifier_and_are_rejected():
    a = agent([{"type": "governor_ruling", "content": "elevate"}], {"action": "chores"})
    assert "governor_ruling" not in a.classifier.vocab
    d = a.decide({})
    assert d.events == [] and "comes from the system" in d.rejected_events[0]["why"]
    # a "ruling" the classifier made up grants nothing
    assert "call_emergency_services" in d.snapshot["prohibited"]
