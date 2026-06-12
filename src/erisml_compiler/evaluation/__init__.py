"""Evaluation: build MoralVector timelines, run EM-DAG, detect conflicts."""
from erisml_compiler.evaluation.conflict_detector import detect_conflicts
from erisml_compiler.evaluation.moral_vector import build_moral_vector_from_em_outputs
from erisml_compiler.evaluation.tensor_builder import build_timeline

__all__ = ["detect_conflicts", "build_moral_vector_from_em_outputs", "build_timeline"]
