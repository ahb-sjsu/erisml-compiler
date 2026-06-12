"""LaBSE-based semantic canonicalizer.

Uses a frozen `sentence-transformers/LaBSE` encoder to embed both the
situation summary and each known canonical-form description, then matches
by cosine similarity. This mirrors the architecture used by the BIP probe
in https://github.com/ahb-sjsu/sqnd-probe; we use vanilla LaBSE for the
canonicalizer interface and document the upgrade path to BIP-fine-tuned
encoder weights.

Optional dependency: `sentence-transformers`. Install via the `[ml]`
extra. If the model fails to load (missing dep, no network, etc.) the
auto-canonicalizer falls back to RegistryCanonicalizer.
"""
from __future__ import annotations

import os
from typing import Any

from erisml_compiler.canonicalizer.base import CanonicalizationResult, Canonicalizer


_DEFAULT_MODEL = "sentence-transformers/LaBSE"


class LaBSECanonicalizer(Canonicalizer):
    """Cosine-similarity canonicalizer in LaBSE embedding space."""

    name = "labse"

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.55,
        device: str | None = None,
    ):
        # Lazy import; raises ImportError if sentence-transformers is missing.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "LaBSECanonicalizer requires the `sentence-transformers` "
                "package. Install via `pip install erisml-compiler[ml]`."
            ) from exc

        self.model_name = model_name or os.environ.get(
            "ERISML_LABSE_MODEL", _DEFAULT_MODEL
        )
        self.threshold = float(os.environ.get("ERISML_LABSE_THRESHOLD", str(threshold)))
        self.device = device or os.environ.get("ERISML_LABSE_DEVICE", "cpu")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_cache: dict[str, Any] = {}

    def _embed(self, text: str) -> Any:
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        emb = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._embedding_cache[text] = emb
        return emb

    def canonicalize(self, summary: str, known_forms: dict[str, str]) -> CanonicalizationResult:
        if not summary.strip() or not known_forms:
            return CanonicalizationResult(
                tag=None, confidence=0.0, matched_known_form=False,
                evidence=["empty summary or empty registry"], backend=self.name,
            )

        # Embed the summary.
        summary_emb = self._embed(summary)

        # Embed each known form's description + tag, and compute cosine
        # similarity. Embeddings are normalized so cosine = dot product.
        import numpy as np
        scores: list[tuple[str, float]] = []
        for tag, description in known_forms.items():
            corpus_text = f"{tag.replace('_', ' ')}. {description}"
            corpus_emb = self._embed(corpus_text)
            sim = float(np.dot(summary_emb, corpus_emb))
            scores.append((tag, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_tag, best_score = scores[0]
        if best_score >= self.threshold:
            return CanonicalizationResult(
                tag=best_tag,
                confidence=best_score,
                matched_known_form=True,
                evidence=[
                    f"LaBSE cosine similarity={best_score:.3f} over "
                    f"{len(known_forms)} known forms.",
                    f"Runner-up: {scores[1][0]} ({scores[1][1]:.3f})"
                    if len(scores) > 1 else "",
                ],
                backend=self.name,
            )
        return CanonicalizationResult(
            tag=None,
            confidence=best_score,
            matched_known_form=False,
            evidence=[
                f"Best LaBSE match was {best_tag!r} at cosine={best_score:.3f} "
                f"(below threshold {self.threshold}).",
            ],
            backend=self.name,
        )
