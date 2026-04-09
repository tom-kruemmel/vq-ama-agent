from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rank_fusion import ScoredChunk


class ConfidenceChecker:
    def __init__(
        self,
        top_k: int = 5,
        min_chunks: int = 3,
        min_unique_chunks: int = 2,
        min_total_chars: int = 450,
        min_top1_score: float = 0.015,
        min_avg_top3_score: float = 0.011,
    ):
        """
        Confidence gate for RRF-style retrieval outputs: List[Tuple[text, fused_score]].

        Defaults are tuned for the current RRF setup (k=60 in rank fusion), where
        useful top fused scores are typically around 0.01-0.05.
        """
        self.top_k = top_k
        self.min_chunks = min_chunks
        self.min_unique_chunks = min_unique_chunks
        self.min_total_chars = min_total_chars
        self.min_top1_score = min_top1_score
        self.min_avg_top3_score = min_avg_top3_score

    def evaluate(self, docs: list[ScoredChunk]) -> tuple[bool, dict]:
        if not docs:
            return False, {
                "reason": "no_retrieval",
                "num_chunks": 0,
                "num_unique_chunks": 0,
                "total_chars": 0,
                "top1_score": 0.0,
                "avg_top3_score": 0.0,
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

        checks = {
            "enough_chunks": num_chunks >= self.min_chunks,
            "enough_unique_chunks": num_unique_chunks >= self.min_unique_chunks,
            "enough_text": total_chars >= self.min_total_chars,
            "top1_strong_enough": top1_score >= self.min_top1_score,
            "top3_avg_strong_enough": avg_top3_score >= self.min_avg_top3_score,
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
        }

    def is_confident(self, docs: list[ScoredChunk]) -> bool:
        confident, _ = self.evaluate(docs)
        return confident
