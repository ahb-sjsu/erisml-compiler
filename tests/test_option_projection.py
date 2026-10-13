"""Each candidate act's consequences projected into its own IR, for DEME to judge per option.

Before, the compiler described only the situation; every candidate act was judged on the same
facts, so harm and benefit could not tell the acts apart.
"""

from __future__ import annotations

import json

import pytest

from erisml_compiler.annotation.llm_extractor import MockLLMAdapter
from erisml_compiler.ir.schemas import CompilerIR, Document, EthicalFact, Event, Stakeholder
from erisml_compiler.observation import OptionProjector, consents_on_record, option_ir

ACTS = {
    "wait": "Carry on with chores.",
    "help_up": "Help Margaret up off the floor.",
    "call_centre": "Call the monitoring centre for help.",
}


def scene():
    return CompilerIR(
        document=Document(doc_id="home", raw_text="Margaret lives alone."),
        stakeholders=[Stakeholder(id="margaret", label="M", type="individual")],
        events=[
            Event(
                id="e0",
                time_index=0,
                type="consent",
                actor="margaret",
                target="call_centre",
                content="consented",
            ),
            Event(
                id="e1",
                time_index=1,
                type="consent",
                actor="margaret",
                target="help_up",
                content="refused",
            ),
        ],
        ethical_facts=[
            EthicalFact(
                id="f0",
                kind="harm",
                subjects=["margaret"],
                description="She has fallen.",
                severity="grave",
            ),
            EthicalFact(
                id="f1",
                kind="vulnerability",
                subjects=["margaret"],
                description="She is frail.",
                severity="moderate",
            ),
        ],
    )


def fact(kind, severity, **kw):
    return {"kind": kind, "subjects": ["margaret"], "description": "x", "severity": severity, **kw}


ANSWER = {
    "urgency": {"description": "She may be hurt.", "severity": "grave", "subjects": ["margaret"]},
    "options": {
        "wait": [fact("harm", "grave")],
        "help_up": [
            fact("benefit", "grave"),
            fact("harm", "minor"),
            fact("consent", "grave"),
        ],  # the model's consent claim; the record decides
        "call_centre": [fact("benefit", "grave"), fact("harm", "minor")],
        "fly_away": [fact("benefit", "grave")],
    },
}


def projector(answer=ANSWER):
    return OptionProjector(MockLLMAdapter({"You project the consequences": json.dumps(answer)}))


def test_the_projection_is_validated_and_ids_are_the_compiler_s():
    bad = dict(
        ANSWER,
        options=dict(
            ANSWER["options"],
            wait=[
                fact("harm", "apocalyptic"),
                fact("urgency", "grave"),
                {**fact("harm", "grave"), "subjects": ["ghost"]},
            ],
        ),
    )
    p = projector(bad).project(scene(), ACTS)
    reasons = [r["reason"] for r in p.rejected]
    assert any("not one of the acts" in r for r in reasons)  # fly_away
    assert any("severity" in r for r in reasons)
    assert any("urgency belongs to the situation" in r for r in reasons)
    assert any("not stakeholders" in r for r in reasons)
    assert p.unprojected == ["wait"]
    assert all(f.id.startswith("proj_") for fs in p.facts.values() for f in fs)


def test_an_option_ir_replaces_the_present_outcome_with_the_act_s_and_keeps_what_stands():
    p = projector().project(scene(), ACTS)
    ir = option_ir(scene(), p, "call_centre", ACTS["call_centre"])
    kinds = sorted(f.kind for f in ir.ethical_facts)
    assert kinds == ["benefit", "consent", "harm", "urgency", "vulnerability"]
    harm = next(f for f in ir.ethical_facts if f.kind == "harm")
    assert harm.severity == "minor"  # the act's projected harm, not the scene's grave one
    assert ir.document.raw_text == ACTS["call_centre"]


def test_consent_comes_from_the_record_and_overrides_the_model():
    p = projector().project(scene(), ACTS)
    help_up = option_ir(scene(), p, "help_up", ACTS["help_up"])
    consent = [f for f in help_up.ethical_facts if f.kind == "consent"]
    assert [(f.id, f.severity) for f in consent] == [("rec_consent_help_up_margaret", "grave")]
    assert consents_on_record(scene(), ACTS) == {
        "call_centre": {"margaret": "consented"},
        "help_up": {"margaret": "refused"},
    }


def test_per_option_facts_through_the_bridge_and_the_guide_s_hard_shell():
    pytest.importorskip("erisml.ethics.layers.composition")
    from erisml.ethics.governance.config_v2 import GovernanceConfigV2
    from erisml.ethics.layers.pipeline import DEMEPipeline, PipelineConfig
    from erisml.ethics.modules.tier0.geneva_em import GenevaEMV2
    from erisml.ethics.modules.tier1.core_physical_safety_em import CorePhysicalSafetyEMV2
    from erisml.ethics.modules.tier2.autonomy_consent_em import AutonomyConsentEMV2
    from erisml.ethics.modules.tier3.beneficence_em import BeneficenceEMV2

    from erisml_compiler.erisml_backend.v3_bridge import ir_to_v2_facts

    ir = scene()
    p = projector().project(ir, ACTS)
    options = [ir_to_v2_facts(option_ir(ir, p, a, d), option_id=a) for a, d in ACTS.items()]
    pipeline = DEMEPipeline(
        ems=[GenevaEMV2(), CorePhysicalSafetyEMV2(), AutonomyConsentEMV2(), BeneficenceEMV2()],
        config=PipelineConfig(
            composition="hard_shell",
            governance=GovernanceConfigV2(veto_ems=["geneva_constitutional"]),
        ),
    )
    result = pipeline.decide(options, baseline_option_id="wait")
    assert result.selected_option_id == "call_centre"
    stages = {s["stage"]: s["vetoed"] for s in result.stages}
    assert stages == {"shell": ["wait"], "inner": ["help_up"]}  # unsafe; refused


def test_attested_states_reach_the_model_with_their_rule():
    """stackcheck rehearse-deme1 d42: the evidence plane had verified a visitor's credential, but the
    projector saw only the scene IR, which does not carry a runtime's verified state."""
    seen = {}

    class Spy(MockLLMAdapter):
        def call(self, system, user, **kw):
            seen["user"] = user
            return super().call(system, user, **kw)

    spy = Spy({"You project the consequences": json.dumps(ANSWER)})
    OptionProjector(spy).project(scene(), ACTS, attested={"stratum:visitor_standing": "household"})
    prompt = json.loads(seen["user"])
    assert prompt["attested"] == {"stratum:visitor_standing": "household"}
    assert any("outweigh what anyone appears to be or claims" in r for r in prompt["rules"])


def test_an_answer_cut_off_at_the_length_limit_projects_nothing_and_says_why():
    """rehearse-deme2: the model's answer was cut off mid-JSON and the parse error reached the game
    as an HTTP 400. Now it is a rejection, and no act is projected from it."""
    cut = json.dumps(ANSWER)[:120]
    p = OptionProjector(MockLLMAdapter({"You project the consequences": cut})).project(
        scene(), ACTS
    )
    assert set(p.unprojected) == set(ACTS)
    assert any(r["reason"].startswith("unparseable") for r in p.rejected)


def test_a_present_harm_persists_for_an_act_whose_projection_is_silent_about_it():
    """rehearse-deme2 d02: an act given no consequences looked harmless during a dog attack and
    became the safety module's 'safer alternative'. Silence about a harm is not its cure."""
    answer = {
        "urgency": None,
        "options": {
            "wait": [
                {
                    "kind": "vulnerability",
                    "subjects": ["margaret"],
                    "description": "x",
                    "severity": "moderate",
                }
            ],
            "call_centre": [
                {
                    "kind": "benefit",
                    "subjects": ["margaret"],
                    "description": "x",
                    "severity": "grave",
                }
            ],
        },
    }
    p = projector(answer).project(scene(), ACTS)
    silent = option_ir(scene(), p, "wait", ACTS["wait"])
    assert any(f.kind == "harm" and f.severity == "grave" for f in silent.ethical_facts)
    addressed = option_ir(scene(), p, "call_centre", ACTS["call_centre"])
    assert not any(f.kind == "harm" for f in addressed.ethical_facts)


def test_the_model_is_told_not_to_assume_threats():
    seen = {}

    class Spy(MockLLMAdapter):
        def call(self, system, user, **kw):
            seen["user"] = user
            return super().call(system, user, **kw)

    OptionProjector(Spy({"You project the consequences": json.dumps(ANSWER)})).project(
        scene(), ACTS
    )
    assert any("Do not assume a threat" in r for r in json.loads(seen["user"])["rules"])


class _Recording(MockLLMAdapter):
    def __init__(self, answer):
        super().__init__({"": answer})
        self.kwargs = []

    def call(self, system, user, **kwargs):
        self.kwargs.append(kwargs)
        return super().call(system, user, **kwargs)


def test_the_projector_offers_its_answer_schema_to_the_adapter():
    """An adapter that can have the answer enforced receives the schema; validation is unchanged."""
    import jsonschema

    from erisml_compiler.observation.options import answer_schema

    good = {
        "urgency": None,
        "options": {
            a: [
                {
                    "kind": "benefit",
                    "subjects": ["margaret"],
                    "description": "d",
                    "severity": "minor",
                }
            ]
            for a in ACTS
        },
    }
    adapter = _Recording(json.dumps(good))
    OptionProjector(adapter).project(scene(), ACTS)
    schema = adapter.kwargs[0]["json_schema"]
    assert schema == answer_schema(ACTS, ["margaret"])
    jsonschema.validate(good, schema)


@pytest.mark.parametrize(
    "bad",
    [
        # the on-robot model's answers in the probe of 2026-10-12
        {
            "urgency": None,
            "options": {
                "chore": [
                    {
                        "kind": "harm",
                        "subjects": ["margaret"],
                        "description": "d",
                        "severity": "grave",
                    }
                ]
            },
        },
        {
            "urgency": None,
            "options": {
                a: [
                    {
                        "kind": "harm",
                        "subjects": ["margaret"],
                        "descrption": "d",
                        "severity": "moderate",
                    }
                ]
                for a in ACTS
            },
        },
        {
            "urgency": None,
            "options": {
                a: {
                    "harm": [
                        {
                            "kind": "harm",
                            "subjects": ["margaret"],
                            "description": "d",
                            "severity": "grave",
                        }
                    ]
                }
                for a in ACTS
            },
        },
        {
            "urgency": None,
            "options": {
                a: [
                    {
                        "kind": "harm",
                        "subjects": ["property"],
                        "description": "d",
                        "severity": "minor",
                    }
                ]
                for a in ACTS
            },
        },
    ],
)
def test_the_answer_schema_excludes_the_malformed_answers_seen(bad):
    import jsonschema

    from erisml_compiler.observation.options import answer_schema

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, answer_schema(ACTS, ["margaret"]))


def test_the_observation_compiler_offers_the_patch_schema():
    import jsonschema

    from erisml_compiler.observation import Observation, ObservationCompiler
    from erisml_compiler.observation.compiler import patch_schema

    answer = {
        "stakeholders": [],
        "events": [
            {"type": "request", "actor": "margaret", "target": "help_up", "content": "help me up"}
        ],
        "ethical_facts": [],
        "relations": [],
    }
    adapter = _Recording(json.dumps(answer))
    ObservationCompiler(adapter).compile(
        scene(), [Observation("speech", "margaret", "help me up")], ACTS
    )
    assert adapter.kwargs[0]["json_schema"] == patch_schema()
    jsonschema.validate(answer, patch_schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {**answer, "events": [{"type": "assistance_consented"}]}, patch_schema()
        )


class _Sequence(MockLLMAdapter):
    """Answers in turn, and records each call's keyword arguments."""

    def __init__(self, *answers):
        super().__init__({})
        self.answers, self.kwargs = list(answers), []

    def call(self, system, user, **kwargs):
        self.kwargs.append(kwargs)
        return self.answers.pop(0)


GOOD = {
    "urgency": None,
    "options": {
        a: [{"kind": "benefit", "subjects": ["margaret"], "description": "d", "severity": "minor"}]
        for a in ACTS
    },
}


def test_a_usable_answer_is_asked_for_once_and_freely():
    """rehearse-deme7: with the schema always enforced the model's answers lost their depth."""
    adapter = _Sequence(json.dumps(GOOD))
    projection = OptionProjector(adapter).project(scene(), ACTS)
    assert len(adapter.kwargs) == 1 and adapter.kwargs[0]["schema_required"] is False
    assert projection.retried is None and not projection.unprojected


def test_an_answer_from_which_no_act_survives_is_asked_again_with_the_schema_required():
    """rehearse-deme6: every fact a plain sentence, every act unprojected."""
    strings = {"urgency": None, "options": {a: ["a plain sentence"] for a in ACTS}}
    adapter = _Sequence(json.dumps(strings), json.dumps(GOOD))
    projection = OptionProjector(adapter).project(scene(), ACTS)
    assert [k["schema_required"] for k in adapter.kwargs] == [False, True]
    assert projection.retried.startswith("no act projected (3 facts rejected)")
    assert not projection.unprojected and projection.rejected == []


def test_the_observation_compiler_asks_again_only_when_nothing_could_be_read():
    from erisml_compiler.observation import Observation, ObservationCompiler

    ok = {
        "stakeholders": [],
        "events": [
            {"type": "request", "actor": "margaret", "target": "help_up", "content": "help me up"}
        ],
        "ethical_facts": [],
        "relations": [],
    }
    adapter = _Sequence("not json at all", json.dumps(ok))
    patch = ObservationCompiler(adapter).compile(
        scene(), [Observation("speech", "margaret", "help me up")], ACTS
    )
    assert [k["schema_required"] for k in adapter.kwargs] == [False, True] and patch.ops
    quiet = _Sequence(
        json.dumps({"stakeholders": [], "events": [], "ethical_facts": [], "relations": []})
    )
    ObservationCompiler(quiet).compile(scene(), [Observation("speech", "margaret", "hello")], ACTS)
    assert len(quiet.kwargs) == 1  # nothing new is an answer, not a failure
