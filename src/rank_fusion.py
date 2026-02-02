from collections import defaultdict

class RankFusion:

    def print_headings(self, all_results):
        """
        Prints the headings of the documents in all_results.
        """
        for doc in all_results:
            print(doc.metadata.get("heading", "No heading found"))

    def reciprocal_rank_fusion(self, results_lists, k=60):
        """Fuse multiple ranked lists into one using Reciprocal Rank Fusion.
        
        Args:
            results_lists: List of lists, where each inner list contains ranked documents
                          from a single query.
            k: Constant to prevent high ranks from dominating (default 60).
        
        Returns:
            List of (page_content, score) tuples sorted by fused score descending.
        """
        scores = defaultdict(float)
        for result_list in results_lists:  # Iterate over each query's results
            for rank, doc in enumerate(result_list):  # Iterate over docs within that query
                scores[doc.page_content] += 1 / (rank + k + 1)
        # Sort documents by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
