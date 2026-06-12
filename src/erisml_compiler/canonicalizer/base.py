"""Canonicalizer abstract base + auto-selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CanonicalizationResult:
    """One canonicalization attempt's outcome.

    `tag` is the matched canonical-form slug, or None if no match cleared
    threshold.

    `confidence` is in [0, 1]; for LaBSE this is the cosine similarity, for
    the registry backend this is a fragment-match overlap ratio.

    `matched_known_form` flags whether the tag is in the bundled registry
    (`ontology/canonical_forms.yaml`) versus a free-text proposal.

    `evidence` is a list of strings describing why the match was chosen —
    useful for the audit trail.

    `backend` records which canonicalizer produced the result.
    """

    tag: str | None
    confidence: float
    matched_known_form: bool
    evidence: list[str]
    backend: str


class Canonicalizer(ABC):
    name: str

    @abstractmethod
    def canonicalize(self, summary: str, known_forms: dict[str, str]) -> CanonicalizationResult:
        """Map a situation summary to a canonical-form tag.

        Args:
            summary: a short natural-language description of the moral
                situation (typically built from concatenated ethical-fact
                descriptions plus the document title).
            known_forms: dict from canonical-form tag (key) to its
                description (value), loaded from
                `ontology/canonical_forms.yaml`.

        Returns:
            CanonicalizationResult.
        """
        raise NotImplementedError


def auto_canonicalizer(prefer_labse: bool = True) -> Canonicalizer:
    """Pick the strongest available canonicalizer at startup.

    If `prefer_labse` and `sentence-transformers` is importable, return a
    LaBSECanonicalizer. Otherwise return a RegistryCanonicalizer.
    """
    if prefer_labse:
        try:
            from erisml_compiler.canonicalizer.labse import LaBSECanonicalizer

            return LaBSECanonicalizer()
        except Exception:
            pass
    from erisml_compiler.canonicalizer.registry import RegistryCanonicalizer

    return RegistryCanonicalizer()
