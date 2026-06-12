"""Canonicalization: map an extracted situation to a stable canonical tag.

Two backends share the `Canonicalizer` interface:

  - `RegistryCanonicalizer` — substring/fragment matching against
    `ontology/canonical_forms.yaml`. Deterministic, offline, no ML deps.

  - `LaBSECanonicalizer` — language-agnostic semantic matching using
    `sentence-transformers/LaBSE`. Optional, behind the `[ml]` extra.
    Falls back to `RegistryCanonicalizer` automatically if the model
    can't be loaded.

`auto_canonicalizer()` picks LaBSE if available, registry otherwise.
"""
from erisml_compiler.canonicalizer.base import (
    CanonicalizationResult,
    Canonicalizer,
    auto_canonicalizer,
)
from erisml_compiler.canonicalizer.labse import LaBSECanonicalizer
from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer

__all__ = [
    "CanonicalizationResult",
    "Canonicalizer",
    "LaBSECanonicalizer",
    "RegistryCanonicalizer",
    "auto_canonicalizer",
]
