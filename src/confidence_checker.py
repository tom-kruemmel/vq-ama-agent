from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from .rank_fusion import ScoredChunk

logger = logging.getLogger(__name__)

_DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class ConfidenceChecker:
    def __init__(
        self,
        top_k: int = 5,
        min_chunks: int = 3,
        min_unique_chunks: int = 2,
        min_total_chars: int = 450,
        min_top1_score: float = 0.015,
        min_avg_top3_score: float = 0.011,
        min_relevance_score: float = -2.0,
        cross_encoder_model: str = _DEFAULT_CROSS_ENCODER_MODEL,
    ):
        """
        Confidence gate for RRF-style retrieval outputs: List[Tuple[text, fused_score]].

        Defaults are tuned for the current RRF setup (k=60 in rank fusion), where
        useful top fused scores are typically around 0.01-0.05.

        A multilingual cross-encoder (mmarco-mMiniLMv2-L12-H384-v1, 100+
        languages) scores the top retrieved chunks against the original
        question for semantic relevance.  It outputs raw logits (roughly
        -10 to +10); positive means relevant, negative means irrelevant.
        The default threshold of -2.0 is intentionally permissive.
        Tune upward once you have evaluation data.
        """
        self.top_k = top_k
        self.min_chunks = min_chunks
        self.min_unique_chunks = min_unique_chunks
        self.min_total_chars = min_total_chars
        self.min_top1_score = min_top1_score
        self.min_avg_top3_score = min_avg_top3_score
        self.min_relevance_score = min_relevance_score

        logger.info("Loading cross-encoder model: %s", cross_encoder_model)
        self._cross_encoder = CrossEncoder(cross_encoder_model)

    def _score_relevance(
        self, question: str, docs: list[ScoredChunk], n: int = 3,
    ) -> tuple[float, float]:
        """Return (top1_relevance, avg_top3_relevance) using the cross-encoder."""
        if not docs or not question:
            return 0.0, 0.0

        pairs = [(question, d.text) for d in docs[:n]]
        scores = self._cross_encoder.predict(pairs).tolist()
        logger.info("Cross-encoder relevance scores: %s", scores)

        top1 = scores[0] if scores else 0.0
        avg_topn = sum(scores) / len(scores) if scores else 0.0
        return float(top1), float(avg_topn)

    def evaluate(self, docs: list[ScoredChunk], question: str = "") -> tuple[bool, dict]:
        if not docs:
            return False, {
                "reason": "no_retrieval",
                "num_chunks": 0,
                "num_unique_chunks": 0,
                "total_chars": 0,
                "top1_score": 0.0,
                "avg_top3_score": 0.0,
                "top1_relevance": 0.0,
                "avg_top3_relevance": 0.0,
            }

        top_docs = docs[: self.top_k]
        top3 = docs[:3]

        texts = [d.text for d in top_docs]
        scores = [d.score for d in top_docs]
        top3_scores = [d.score for d in top3]

        num_chunks = len(top_docs)
        num_unique_chunks = len(set(texts))
        total_chars = sum(len(text) for text in texts)
        top1_score = scores[0] if scores else 0.0
        avg_top3_score = sum(top3_scores) / max(1, len(top3_scores))

        # Semantic relevance via cross-encoder
        top1_relevance, avg_top3_relevance = self._score_relevance(question, top_docs)

        checks = {
            "enough_chunks": num_chunks >= self.min_chunks,
            "enough_unique_chunks": num_unique_chunks >= self.min_unique_chunks,
            "enough_text": total_chars >= self.min_total_chars,
            "top1_strong_enough": top1_score >= self.min_top1_score,
            "top3_avg_strong_enough": avg_top3_score >= self.min_avg_top3_score,
            "top1_semantically_relevant": top1_relevance >= self.min_relevance_score,
        }

        confident = all(checks.values())
        failed = [name for name, ok in checks.items() if not ok]

        return confident, {
            "reason": "ok" if confident else "sparse_or_weak_context",
            "failed_checks": failed,
            "num_chunks": num_chunks,
            "num_unique_chunks": num_unique_chunks,
            "total_chars": total_chars,
            "top1_score": top1_score,
            "avg_top3_score": avg_top3_score,
            "top1_relevance": top1_relevance,
            "avg_top3_relevance": avg_top3_relevance,
        }

    def is_confident(self, docs: list[ScoredChunk], question: str = "") -> bool:
        confident, _ = self.evaluate(docs, question)
        return confident
