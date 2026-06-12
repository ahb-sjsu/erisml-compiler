"""Registry-based canonicalizer.

Substring/fragment match against the canonical-form descriptions in
`ontology/canonical_forms.yaml`. Deterministic, offline, zero ML deps.
Used as the default when sentence-transformers is not installed, and as
the fallback when LaBSE fails to load.
"""
from __future__ import annotations

import re

from erisml_compiler.canonicalizer.base import CanonicalizationResult, Canonicalizer


STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "by", "on", "for", "with",
    "that", "this", "is", "are", "was", "were", "be", "been", "being",
    "as", "at", "from", "not", "no", "but", "if", "then", "else", "than",
    "into", "over", "under", "through", "between", "against",
})


def _tokenize(s: str) -> set[str]:
    """Cheap tokenizer: lowercase alphanumeric words minus stopwords."""
    return {
        t for t in re.findall(r"[a-z0-9_]+", s.lower())
        if t not in STOPWORDS and len(t) > 2
    }


class RegistryCanonicalizer(Canonicalizer):
    """Match by Jaccard similarity of meaningful tokens.

    For each known canonical form, compute the Jaccard similarity between
    the form's description tokens and the situation-summary tokens. Return
    the highest-scoring match if it exceeds threshold (default 0.10), else
    None.
    """

    name = "registry"

    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold

    def canonicalize(self, summary: str, known_forms: dict[str, str]) -> CanonicalizationResult:
        summary_tokens = _tokenize(summary)
        if not summary_tokens or not known_forms:
            return CanonicalizationResult(
                tag=None, confidence=0.0, matched_known_form=False,
                evidence=["empty summary or empty registry"], backend=self.name,
            )

        scores: list[tuple[str, float, set[str]]] = []
        for tag, description in known_forms.items():
            form_tokens = _tokenize(description) | _tokenize(tag)
            if not form_tokens:
                continue
            inter = summary_tokens & form_tokens
            union = summary_tokens | form_tokens
            jaccard = len(inter) / len(union) if union else 0.0
            scores.append((tag, jaccard, inter))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_tag, best_score, best_overlap = scores[0]
        if best_score >= self.threshold:
            return CanonicalizationResult(
                tag=best_tag,
                confidence=best_score,
                matched_known_form=True,
                evidence=[
                    f"Jaccard similarity={best_score:.3f} over {len(known_forms)} known forms.",
                    f"Overlap tokens: {sorted(best_overlap)[:10]}",
                ],
                backend=self.name,
            )
        return CanonicalizationResult(
            tag=None,
            confidence=best_score,
            matched_known_form=False,
            evidence=[
                f"Best match was {best_tag!r} at Jaccard={best_score:.3f} "
                f"(below threshold {self.threshold})."
            ],
            backend=self.name,
        )
