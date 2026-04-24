from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .reranker import RerankedChunk

logger = logging.getLogger(__name__)


class ConfidenceChecker:
    def __init__(
        self,
        min_chunks: int = 3,
        min_unique_chunks: int = 1,
        min_total_chars: int = 200,
        min_top1_relevance: float = -2.0,
        min_avg_top3_relevance: float = -2.0,
    ):
        """
        Confidence gate that operates on *already re-ranked* chunks.

        Expects a list of RerankedChunk(text, ce_score) produced by
        CrossEncoderReranker — the cross-encoder scores are reused
        directly, so no additional model inference is performed here.

        Checks:
          - enough_chunks:  retrieved ≥ min_chunks
          - enough_unique_chunks: ≥ min_unique_chunks distinct texts
          - enough_text: total chars ≥ min_total_chars
          - top1_semantically_relevant: best ce_score ≥ threshold
          - top3_avg_semantically_relevant: mean of top-3 ce_scores ≥ threshold

        Default thresholds are intentionally permissive.
        Tune upward once you have evaluation data.
        """
        self.min_chunks = min_chunks
        self.min_unique_chunks = min_unique_chunks
        self.min_total_chars = min_total_chars
        self.min_top1_relevance = min_top1_relevance
        self.min_avg_top3_relevance = min_avg_top3_relevance

    def evaluate(self, docs: list[RerankedChunk]) -> tuple[bool, dict]:
        """Evaluate confidence on re-ranked chunks.

        Args:
            docs: Re-ranked chunks with cross-encoder scores (sorted by
                  ce_score descending from CrossEncoderReranker).

        Returns:
            (confident, details_dict)
        """
        if not docs:
            return False, {
                "reason": "no_retrieval",
                "num_chunks": 0,
                "num_unique_chunks": 0,
                "total_chars": 0,
                "top1_relevance": 0.0,
                "avg_top3_relevance": 0.0,
            }

        texts = [d.text for d in docs]
        ce_scores = [d.ce_score for d in docs]
        top3_scores = ce_scores[:3]

        num_chunks = len(docs)
        num_unique_chunks = len(set(texts))
        total_chars = sum(len(t) for t in texts)
        top1_relevance = ce_scores[0]
        avg_top3_relevance = sum(top3_scores) / len(top3_scores)

        checks = {
            "enough_chunks": num_chunks >= self.min_chunks,
            "enough_unique_chunks": num_unique_chunks >= self.min_unique_chunks,
            "enough_text": total_chars >= self.min_total_chars,
            "top1_semantically_relevant": top1_relevance >= self.min_top1_relevance,
            "top3_avg_semantically_relevant": avg_top3_relevance >= self.min_avg_top3_relevance,
        }

        confident = all(checks.values())
        failed = [name for name, ok in checks.items() if not ok]

        return confident, {
            "reason": "ok" if confident else "sparse_or_weak_context",
            "failed_checks": failed,
            "num_chunks": num_chunks,
            "num_unique_chunks": num_unique_chunks,
            "total_chars": total_chars,
            "top1_relevance": top1_relevance,
            "avg_top3_relevance": avg_top3_relevance,
        }

    def is_confident(self, docs: list[RerankedChunk]) -> bool:
        confident, _ = self.evaluate(docs)
        return confident
