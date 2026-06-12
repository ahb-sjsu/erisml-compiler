"""Ingestion: load text or structured input into the pipeline."""

from erisml_compiler.ingestion.structured_loader import load_structured_input
from erisml_compiler.ingestion.text_loader import load_text_document

__all__ = ["load_structured_input", "load_text_document"]
