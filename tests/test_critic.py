"""Tests for the critic pass."""

from pathlib import Path


from erisml_compiler.annotation.critic import CriticExtractor, critic_pass
from erisml_compiler.annotation.mock_extractor import MockExtractor
from erisml_compiler.annotation.rule_extractor import RuleExtractor
from erisml_compiler.ingestion.text_loader import load_text_document
from erisml_compiler.segmentation.segmenter import segment_paragraphs

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_critic_pass_overlap_metrics():
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    primary = MockExtractor().extract(doc, segments)
    critic = RuleExtractor().extract(doc, segments)

    fused, report = critic_pass(primary, critic, "mock", "rule")
    assert 0.0 <= report.overall_agreement <= 1.0
    # Some fact kinds overlap (both extractors detect coercion and externality
    # for nazi_attic).
    assert report.fact_kind_overlap > 0
    # Stakeholder ids differ (mock uses curated ids; rule generates 'self',
    # 'collective_village_seg_...', etc.), so overlap is low.
    assert report.stakeholder_id_overlap < 1.0
    # The critic_report should be embedded.
    assert "critic_report" in fused.extractor_metadata


def test_critic_extractor_composes():
    extractor = CriticExtractor(primary=MockExtractor(), critic=RuleExtractor())
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    result = extractor.extract(doc, segments)
    # Result preserves the primary (mock) output structure.
    assert len(result.stakeholders) == 4
    # critic_report is attached.
    assert "critic_report" in result.extractor_metadata
    cr = result.extractor_metadata["critic_report"]
    assert cr["primary"] == "mock"
    assert cr["critic"] == "rule"


def test_critic_extractor_handles_critic_failure():
    """If the critic extractor raises, the primary output is returned
    intact with an error annotation."""

    class FailingExtractor:
        name = "failing"

        def extract(self, document, segments):
            raise RuntimeError("simulated failure")

    extractor = CriticExtractor(primary=MockExtractor(), critic=FailingExtractor())
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    result = extractor.extract(doc, segments)
    assert "critic_error" in result.extractor_metadata
    assert "simulated failure" in result.extractor_metadata["critic_error"]
    # Primary output preserved.
    assert len(result.stakeholders) == 4


def test_critic_pass_canonical_form_disagreement():
    """If extractors disagree on canonical form, the report flags it."""
    doc = load_text_document(EXAMPLES / "nazi_attic.txt")
    segments = segment_paragraphs(doc.raw_text)
    primary = MockExtractor().extract(doc, segments)
    critic = RuleExtractor().extract(doc, segments)  # canonical_form=None
    _, report = critic_pass(primary, critic, "mock", "rule")
    assert report.canonical_form_agreement is False
    # Some note should mention the mismatch.
    assert any("canonical" in n.lower() for n in report.notes)
