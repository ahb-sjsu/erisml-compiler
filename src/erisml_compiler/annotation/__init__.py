"""Annotation: extract stakeholders, commitments, events, ethical facts."""
from erisml_compiler.annotation.base import (
    Extractor,
    ExtractorResult,
    UnknownDocumentError,
)
from erisml_compiler.annotation.critic import CriticExtractor, CriticReport, critic_pass
from erisml_compiler.annotation.llm_extractor import (
    LLMExtractor,
    LocalVLLMAdapter,
    MockLLMAdapter,
    ModelAdapter,
    NRPOpenAIAdapter,
)
from erisml_compiler.annotation.mock_extractor import MockExtractor
from erisml_compiler.annotation.rule_extractor import RuleExtractor

__all__ = [
    "CriticExtractor",
    "CriticReport",
    "Extractor",
    "ExtractorResult",
    "LLMExtractor",
    "LocalVLLMAdapter",
    "MockExtractor",
    "MockLLMAdapter",
    "ModelAdapter",
    "NRPOpenAIAdapter",
    "RuleExtractor",
    "UnknownDocumentError",
    "critic_pass",
]
