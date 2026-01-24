class ConfidenceChecker:
    def __init__(self, min_score: float = 0.35, top_k: int = 3):
        """
        :param min_score: Minimum acceptable similarity score for retrieved docs
        :param top_k: How many top documents to consider
        """
        self.min_score = min_score
        self.top_k = top_k

    def score_docs(self, docs) -> float:
        """Compute a confidence score (0.0–1.0) for RRF-style docs: List[Tuple[str, float]]."""
        if not docs:
            return 0.0

        # Take top-k tuples (text, score)
        topk = docs[: self.top_k]

        # Pull scores from the tuples
        raw_scores = [s for _, s in topk]

        # Normalize RRF scores within top-k to [0,1] so the scale is comparable
        s_max = max(raw_scores)
        s_min = min(raw_scores)
        span = max(s_max - s_min, 1e-12)
        norm_scores = [(s - s_min) / span for s in raw_scores]

        # Confidence from worst of top-k (set should be consistently strong)
        min_doc_score = min(norm_scores)

        # Source diversity: use the text (or its hash) as an identifier
        # (If you later have real sources, swap this to that field.)
        unique_sources = {hash(text) for text, _ in topk}
        coverage = len(unique_sources) / max(1, len(topk))
        # Blend quality and coverage
        return 0.7 * min_doc_score + 0.3 * coverage

    def is_confident(self, docs) -> bool:
        # With normalization, 0.75 is a practical default threshold.
        return self.score_docs(docs) >= 0.75
