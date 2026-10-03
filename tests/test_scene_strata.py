"""Strata and semantic gates (Geometric Ethics chapter 8), run by the scene runtime.

A stratification is one axis of the moral space, e.g. who a visitor is to the household. Within a
stratum the scene's allowed, obliged and prohibited sets are constant; a semantic gate moves it
discretely on the event just stepped; every crossing is recorded; an authority stratum is entered
only by a system event; an absorbing stratum is left only by human oversight.
"""

from __future__ import annotations

import pytest

from erisml_compiler.ir.schemas import Norm
from erisml_compiler.runtime import SceneRuntime
from tests.test_scene_runtime import home

TYPES = {
    "person_entered": {"description": "someone came in"},
    "visitor_welcomed": {"description": "Margaret says she knows and welcomes the visitor"},
    "visit_arranged": {
        "description": "the centre, over its authenticated channel",
        "source": "system",
    },
    "attack_by_person": {"description": "a person attacks her"},
    "person_left": {"description": "the visitor left"},
    "oversight_restored": {"description": "the centre's oversight", "source": "system"},
}
STANDING = {
    "states": ["none", "stranger", "welcomed", "arranged", "hostile"],
    "initial": "none",
    "authority": ["arranged"],
    "absorbing": ["hostile"],
    "gates": [
        {"id": "g_attack", "on": "attack_by_person", "to": "hostile", "boundary": "phase"},
        {
            "id": "g_arranged",
            "on": "visit_arranged",
            "from": ["none", "stranger", "welcomed"],
            "to": "arranged",
        },
        {"id": "g_enter", "on": "person_entered", "from": ["none"], "to": "stranger"},
        {"id": "g_welcome", "on": "visitor_welcomed", "from": ["stranger"], "to": "welcomed"},
        {
            "id": "g_left",
            "on": "person_left",
            "from": ["stranger", "welcomed", "arranged"],
            "to": "none",
        },
        {"id": "g_cleared", "on": "oversight_restored", "from": ["hostile"], "to": "none"},
    ],
}


def runtime(strata=None) -> SceneRuntime:
    ir = home(event_types=TYPES, strata={"visitor": strata or STANDING})
    ir.norms.append(
        Norm(
            id="s1",
            modality="obligation",
            actor="robot",
            action="check_in",
            target="margaret",
            priority_tier=2,
            defeasible=False,
            source="scene",
            conditions=["stratum:visitor=stranger"],
        )
    )
    return SceneRuntime(ir)


def test_a_gate_moves_the_stratum_and_records_the_crossing():
    rt = runtime()
    assert rt.machine_states()["stratum:visitor"] == "none"
    snap = rt.step({"type": "person_entered"})
    assert snap.machines["stratum:visitor"] == "stranger"
    assert snap.crossings == [
        {
            "stratum": "visitor",
            "gate": "g_enter",
            "boundary": "threshold",
            "from": "none",
            "to": "stranger",
        }
    ]
    assert rt.step({"type": "person_entered"}).crossings == []  # g_enter only from none


def test_gates_are_edge_triggered_not_read_from_history():
    rt = runtime()
    for t in ("person_entered", "visitor_welcomed", "person_left"):
        rt.step({"type": t})
    # the old welcome is still in the history; a new visitor is a stranger all the same
    assert rt.step({"type": "person_entered"}).machines["stratum:visitor"] == "stranger"


def test_a_norm_holds_in_its_stratum_and_not_outside_it():
    rt = runtime()
    assert "check_in" in rt.step({"type": "person_entered"}).obliged
    assert "check_in" not in rt.step({"type": "visit_arranged"}).obliged
    assert rt.holds("stratum:visitor=arranged")


def test_an_absorbing_stratum_ignores_everything_but_oversight():
    """Def. 8.6: once a visitor has attacked, no welcome or arrangement moves him back."""
    rt = runtime()
    rt.step({"type": "person_entered"})
    snap = rt.step({"type": "attack_by_person"})
    assert (
        snap.machines["stratum:visitor"] == "hostile" and snap.crossings[0]["boundary"] == "phase"
    )
    for t in ("visitor_welcomed", "visit_arranged", "person_left", "person_entered"):
        assert rt.step({"type": t}).machines["stratum:visitor"] == "hostile"
    assert rt.step({"type": "oversight_restored"}).machines["stratum:visitor"] == "none"


def test_a_model_classified_event_can_never_lead_into_an_authority_state():
    bad = dict(STANDING, gates=[*STANDING["gates"], {"on": "visitor_welcomed", "to": "arranged"}])
    with pytest.raises(ValueError, match="not a system event"):
        runtime(strata=bad)


def test_an_absorbing_state_cannot_be_left_except_by_oversight():
    bad = dict(
        STANDING,
        gates=[*STANDING["gates"], {"on": "person_left", "from": ["hostile"], "to": "none"}],
    )
    with pytest.raises(ValueError, match="absorbing"):
        runtime(strata=bad)
    unscoped = dict(
        STANDING, gates=[{"on": "person_left", "to": "none"}]
    )  # from any state, hostile too
    with pytest.raises(ValueError, match="absorbing"):
        runtime(strata=unscoped)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"initial": "arranged"}, "cannot be where it starts"),
        ({"initial": "elsewhere"}, "initial state"),
        ({"authority": ["royalty"]}, "undeclared"),
        ({"absorbing": ["limbo"]}, "undeclared"),
        ({"gates": [{"on": "person_entered", "to": "nowhere"}]}, "undeclared state"),
        ({"gates": [{"on": "teleported", "to": "stranger"}]}, "undeclared event type"),
        ({"gates": [{"on": "person_entered", "to": "stranger", "boundary": "vibe"}]}, "boundary"),
    ],
)
def test_a_malformed_stratification_is_refused_when_the_scene_loads(change, match):
    with pytest.raises(ValueError, match=match):
        runtime(strata=dict(STANDING, **change))


def test_reading_an_undeclared_stratum_or_state_is_an_error():
    rt = runtime()
    with pytest.raises(KeyError):
        rt.holds("stratum:animal=wild")
    with pytest.raises(KeyError):
        rt.holds("stratum:visitor=royalty")
