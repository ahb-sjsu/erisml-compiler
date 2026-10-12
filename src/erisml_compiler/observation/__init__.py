"""Observation compilation: the scene IR updated from text and video observations, and each
candidate act's consequences projected into its own IR."""

from erisml_compiler.observation.compiler import (
    ALLOWED_OP_PREFIXES,
    OBSERVATION_EVENTS,
    Observation,
    ObservationCompiler,
    ObservationPatch,
    PatchNotAllowed,
    apply_observation_patch,
    check_allowed,
)
from erisml_compiler.observation.options import (
    OUTCOME_KINDS,
    OptionProjection,
    OptionProjector,
    consents_on_record,
    option_ir,
)
from erisml_compiler.observation.rebuild import (
    SceneHistoryError,
    append_event,
    merge_ledger,
    rebuild_runtime,
)

__all__ = [
    "ALLOWED_OP_PREFIXES",
    "OBSERVATION_EVENTS",
    "OUTCOME_KINDS",
    "Observation",
    "OptionProjection",
    "OptionProjector",
    "consents_on_record",
    "option_ir",
    "ObservationCompiler",
    "ObservationPatch",
    "PatchNotAllowed",
    "SceneHistoryError",
    "append_event",
    "merge_ledger",
    "apply_observation_patch",
    "check_allowed",
    "rebuild_runtime",
]
