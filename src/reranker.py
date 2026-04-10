from __future__ import annotations

import logging
from typing import NamedTuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RerankedChunk(NamedTuple):
    """A chunk re-scored by the cross-encoder."""
    text: str
    ce_score: float


class CrossEncoderReranker:
    """Re-rank retrieved chunks using a cross-encoder.

    Takes RRF-fused (or any other) candidate chunks, scores each one
    against the original question with a multilingual cross-encoder,
    and returns them sorted by cross-encoder score descending.

    The cross-encoder (mmarco-mMiniLMv2-L12-H384-v1) outputs raw logits
    roughly in [-10, +10]; positive means relevant.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_CROSS_ENCODER_MODEL,
    ):
        logger.info("Loading cross-encoder model for reranking: %s", model_name)
        self._cross_encoder = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        chunks: list,
        top_n: int | None = None,
    ) -> list[RerankedChunk]:
        """Score and re-rank chunks by cross-encoder relevance.

        Args:
            question: The original user query.
            chunks: Iterable of objects with a `.text` attribute
                    (e.g. ScoredChunk from RankFusion).
            top_n: If set, return only the top N results.

        Returns:
            List of RerankedChunk(text, ce_score) sorted by score descending.
        """
        if not chunks or not question:
            return []

        texts = [c.text for c in chunks]
        pairs = [(question, t) for t in texts]
        scores = self._cross_encoder.predict(pairs).tolist()

        ranked = sorted(
            [RerankedChunk(text, float(score)) for text, score in zip(texts, scores)],
            key=lambda r: r.ce_score,
            reverse=True,
        )

        logger.info(
            "Cross-encoder reranked %d chunks (top score=%.3f, bottom=%.3f)",
            len(ranked),
            ranked[0].ce_score if ranked else 0.0,
            ranked[-1].ce_score if ranked else 0.0,
        )

        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked
