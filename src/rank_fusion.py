from collections import defaultdict
from langchain.schema import Document


class RankFusion:

    def reciprocal_rank_fusion(
        self,
        results_lists: list[list[Document]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists into one using Reciprocal Rank Fusion.
        
        Args:
            results_lists: List of lists, where each inner list contains ranked documents
                          from a single query.
            k: RRF constant that dampens the influence of high ranks (default 60,
               per the original Cormack et al. paper).
        
        Returns:
            List of (page_content, score) tuples sorted by fused score descending.
        """
        scores: dict[str, float] = defaultdict(float)
        for result_list in results_lists:
            for rank, doc in enumerate(result_list):
                scores[doc.page_content] += 1 / (rank + k + 1)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
