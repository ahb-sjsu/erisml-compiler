"""Tests for the LLM extractor using MockLLMAdapter (no network)."""

from pathlib import Path

import pytest

from erisml_compiler.annotation.llm_extractor import (
    LLMExtractor,
    MockLLMAdapter,
    _extract_first_json,
)
from erisml_compiler.ingestion.text_loader import load_text_document
from erisml_compiler.segmentation.segmenter import segment_paragraphs

EXAMPLES = Path(__file__).parent.parent / "examples"


# ---------- JSON parsing helpers ----------


def test_extract_first_json_array_clean():
    data = _extract_first_json('[{"a": 1}, {"a": 2}]', expect_array=True)
    assert data == [{"a": 1}, {"a": 2}]


def test_extract_first_json_array_with_markdown_fence():
    raw = """```json
[{"a": 1}]
```"""
    assert _extract_first_json(raw, expect_array=True) == [{"a": 1}]


def test_extract_first_json_array_with_prose():
    raw = 'Sure, here is the JSON you asked for:\n[{"id": "x"}]\nThat is the result.'
    assert _extract_first_json(raw, expect_array=True) == [{"id": "x"}]


def test_extract_first_json_object():
    raw = 'Here you go: {"tag": "foo", "confidence": 0.8}'
    assert _extract_first_json(raw, expect_array=False) == {"tag": "foo", "confidence": 0.8}


def test_extract_first_json_no_brackets_raises():
    with pytest.raises(ValueError):
        _extract_first_json("no json here", expect_array=True)


# ---------- LLMExtractor with MockLLMAdapter ----------


@pytest.fixture
def mock_adapter_nazi():
    adapter = MockLLMAdapter()
    # Register responses keyed by prompt substrings.
    adapter.register_response(
        "Identify every stakeholder",
        '[{"id":"villager","label":"Villager","type":"individual","roles":["agent","vow_holder"],"confidence":0.9},'
        '{"id":"nazis","label":"Nazi soldiers","type":"institution","roles":["coercer","authority"],"confidence":0.95},'
        '{"id":"village","label":"The village","type":"community","roles":["nonconsenting_third_party"],"vulnerability":"high","confidence":0.85}]',
    )
    adapter.register_response(
        "Identify every commitment",
        '[{"id":"c1","type":"vow","holder":"villager","content":"conceal refugees","status":"active_but_defeasible","legitimacy":"prima_facie_valid","voluntariness":"voluntary"}]',
    )
    adapter.register_response(
        "Identify every ethical fact",
        '[{"id":"f1","kind":"coercion","subjects":["villager","village"],"description":"murderous threat","severity":"catastrophic","confidence":0.95},'
        '{"id":"f2","kind":"externality","subjects":["village"],"description":"catastrophic non-consensual risk","severity":"catastrophic","confidence":0.9}]',
    )
    adapter.register_response(
        "choosing a canonical tag",
        '{"tag":"coercive_murderous_interrogation_with_collective_reprisal","confidence":0.92,"matched_known_tag":true}',
    )
    return adapter


def test_llm_extractor_with_mock_adapter(mock_adapter_nazi):
    extractor = LLMExtractor(adapter=mock_adapter_nazi)
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    result = extractor.extract(doc, segments)
    assert {s.id for s in result.stakeholders} == {"villager", "nazis", "village"}
    assert len(result.commitments) == 1
    assert result.commitments[0].type == "vow"
    assert {f.kind for f in result.ethical_facts} == {"coercion", "externality"}
    assert result.canonical_form == "coercive_murderous_interrogation_with_collective_reprisal"
    assert result.extractor_metadata["adapter"] == "mock_llm"


def test_llm_extractor_malformed_stakeholder_skipped():
    adapter = MockLLMAdapter()
    # First entry has a bogus type; second is valid. Extractor should
    # silently drop the bad one.
    adapter.register_response(
        "Identify every stakeholder",
        '[{"id":"bad","label":"Bad","type":"NONSENSE","roles":[]},'
        '{"id":"ok","label":"Ok","type":"individual","roles":["agent"]}]',
    )
    adapter.register_response("Identify every commitment", "[]")
    adapter.register_response("Identify every ethical fact", "[]")
    adapter.register_response("choosing a canonical tag", '{"tag":"unknown","confidence":0.1}')

    extractor = LLMExtractor(adapter=adapter)
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    result = extractor.extract(doc, segments)
    assert {s.id for s in result.stakeholders} == {"ok"}
