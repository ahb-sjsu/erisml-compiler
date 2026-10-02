"""Ingestion: load text, structured, or sensor (video) input into the pipeline."""

from erisml_compiler.ingestion.structured_loader import load_structured_input
from erisml_compiler.ingestion.text_loader import load_text_document
from erisml_compiler.ingestion.video_witness import VideoWitnessStream, encode_video

__all__ = [
    "load_structured_input",
    "load_text_document",
    "encode_video",
    "VideoWitnessStream",
]
