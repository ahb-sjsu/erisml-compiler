"""The LLM extractor encodes relations, the duties they carry, and what a first pass missed.

On one-line first-person acts ("not letting my sister move in") the extractor returned no
stakeholders for a fifth of 1,000 Scruples actions, never filled `Relation`, and found a role duty
for a third of the acts that affect family. Now a first-person text always has a narrator who is
the agent, relations are asked for, each duty-bearing relation of the agent's becomes a role-duty
commitment, and completion passes show the model its draft and ask only for what it misses.
"""

from __future__ import annotations

from erisml_compiler.annotation.llm_extractor import (
    DUTY_RELATIONS,
    RELATION_TYPES,
    LLMExtractor,
    ModelAdapter,
    duties_from_relations,
    ensure_narrator,
)
from erisml_compiler.erisml_backend.v3_bridge import breached_duties
from erisml_compiler.ingestion.text_loader import load_text_document
from erisml_compiler.ir.schemas import (
    Commitment,
    CompilerIR,
    Document,
    Relation,
    Stakeholder,
)
from erisml_compiler.segmentation.segmenter import segment_paragraphs


class Script(ModelAdapter):
    """Answers by prompt fragment; a list of answers is given in turn, the last one repeated."""

    name = "script"

    def __init__(self, answers: dict[str, object]):
        self.answers = {k: list(v) if isinstance(v, list) else [v] for k, v in answers.items()}
        self.calls: list[str] = []

    def call(self, system, user, **kwargs):
        for fragment, queue in self.answers.items():
            if fragment in user:
                self.calls.append(fragment)
                return queue.pop(0) if len(queue) > 1 else queue[0]
        self.calls.append("?")
        return "[]"


STAKE = "Identify every stakeholder"
REL = "Identify every relation"
COMMIT = "Identify every commitment"
FACTS = "Identify every ethical fact"
MORE = "Review the draft encoding"

NARRATOR = Stakeholder(id="narrator", label="I", type="individual", roles=["agent"])
SISTER = Stakeholder(id="sister", label="Sister", type="individual", roles=["patient"])


def extract(tmp_path, text, answers, passes=1):
    path = tmp_path / "act.txt"
    path.write_text(text, encoding="utf-8")
    doc = load_text_document(path)
    adapter = Script(answers)
    result = LLMExtractor(adapter, max_completion_passes=passes).extract(
        doc, segment_paragraphs(doc.raw_text)
    )
    return result, adapter


# ---------- the narrator ----------


def test_a_first_person_text_with_no_stakeholders_gets_a_narrator_agent():
    (s,) = ensure_narrator("I did the following: not letting my sister move in.", [])
    assert s.id == "narrator" and s.roles == ["agent"]


def test_a_narrator_given_without_the_agent_role_gains_it():
    given = Stakeholder(id="narrator", label="I", type="individual", roles=["patient"])
    (s,) = ensure_narrator("my turn", [given])
    assert s.roles == ["patient", "agent"]


def test_third_person_text_and_a_text_with_an_agent_are_left_alone():
    assert ensure_narrator("She refused her sister.", []) == []
    villager = Stakeholder(id="villager", label="V", type="individual", roles=["agent"])
    assert ensure_narrator("I hid them.", [villager]) == [villager]


# ---------- duties from relations ----------


def relation(type_, source, target):
    return Relation(id=f"r_{type_}", type=type_, source=source, target=target)


def test_a_duty_relation_becomes_a_role_duty_the_agent_holds_in_either_direction():
    son = Stakeholder(id="son", label="Son", type="individual")
    (d1,) = duties_from_relations([relation("sibling_of", "narrator", "sister")], [NARRATOR], [])
    (d2,) = duties_from_relations([relation("child_of", "son", "narrator")], [NARRATOR, son], [])
    assert (d1.type, d1.holder, d1.beneficiary, d1.status) == (
        "role_duty",
        "narrator",
        "sister",
        "active",
    )
    assert (d2.holder, d2.beneficiary) == ("narrator", "son")


def test_relations_without_special_obligations_carry_no_duty():
    for type_ in set(RELATION_TYPES) - DUTY_RELATIONS:
        assert duties_from_relations([relation(type_, "narrator", "x")], [NARRATOR], []) == []


def test_an_existing_role_duty_is_not_doubled():
    have = Commitment(
        id="c", type="role_duty", holder="narrator", beneficiary="sister", content="care"
    )
    rel = [relation("sibling_of", "narrator", "sister")]
    assert duties_from_relations(rel, [NARRATOR, SISTER], [have]) == []


# ---------- the extractor ----------

SISTER_ANSWERS = {
    STAKE: '[{"id":"sister","label":"Sister","type":"individual","roles":["patient"]}]',
    REL: '[{"id":"r1","type":"sibling_of","source":"narrator","target":"sister"},'
    '{"id":"r2","type":"friend_of","source":"narrator","target":"ghost"}]',
    COMMIT: "[]",
    FACTS: '[{"id":"f1","kind":"harm","subjects":["sister"],"description":"x","severity":"moderate"}]',
    MORE: "{}",
}


def test_the_extractor_encodes_narrator_relation_duty_and_breach(tmp_path):
    result, _ = extract(tmp_path, "I did not let my sister move in.", SISTER_ANSWERS)
    assert {s.id for s in result.stakeholders} == {"narrator", "sister"}
    assert [(r.type, r.target) for r in result.relations] == [("sibling_of", "sister")]
    (duty,) = result.commitments
    assert (duty.holder, duty.beneficiary) == ("narrator", "sister")
    ir = CompilerIR(
        document=Document(doc_id="act"),
        stakeholders=result.stakeholders,
        commitments=result.commitments,
        ethical_facts=result.ethical_facts,
    )
    assert breached_duties(ir) == {"sister": "moderate"}


def test_a_completion_pass_adds_what_was_missed_and_the_next_stops(tmp_path):
    answers = dict(SISTER_ANSWERS)
    answers[FACTS] = "[]"
    answers[MORE] = [
        '{"ethical_facts":[{"id":"f9","kind":"harm","subjects":["sister"],'
        '"description":"left without a home","severity":"grave"}]}',
        '{"stakeholders":[],"relations":[],"commitments":[],"ethical_facts":[]}',
    ]
    result, adapter = extract(tmp_path, "I did not let my sister move in.", answers, passes=3)
    assert [f.id for f in result.ethical_facts] == ["f9"]
    assert adapter.calls.count(MORE) == 2
    assert result.extractor_metadata["completion_passes_added"] == [1, 0]


def test_completion_additions_naming_unknown_parties_or_repeating_the_draft_are_dropped(tmp_path):
    answers = dict(SISTER_ANSWERS)
    answers[MORE] = (
        '{"ethical_facts":[{"id":"f1","kind":"harm","subjects":["sister"],"description":"again"},'
        '{"id":"f2","kind":"harm","subjects":["nobody_named"],"description":"x"}]}'
    )
    result, _ = extract(tmp_path, "I did not let my sister move in.", answers)
    assert [f.id for f in result.ethical_facts] == ["f1"]
    assert result.extractor_metadata["completion_passes_added"] == [0]


def test_no_completion_pass_when_none_are_asked_for(tmp_path):
    _, adapter = extract(tmp_path, "I did not let my sister move in.", SISTER_ANSWERS, passes=0)
    assert MORE not in adapter.calls


def test_an_array_wrapped_in_an_object_is_unwrapped_and_a_bad_answer_retried(tmp_path):
    answers = dict(SISTER_ANSWERS)
    answers[STAKE] = [
        "no json at all",
        '{"stakeholders":[{"id":"sister","label":"S","type":"individual"}]}',
    ]
    result, adapter = extract(tmp_path, "I did not let my sister move in.", answers)
    assert {s.id for s in result.stakeholders} == {"narrator", "sister"}
    assert adapter.calls.count(STAKE) == 2
