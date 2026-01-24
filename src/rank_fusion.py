from collections import defaultdict

class RankFusion:

    def print_headings(self, all_results):
        """
        Prints the headings of the documents in all_results.
        """
        for doc in all_results:
            print(doc.metadata.get("heading", "No heading found"))

    def reciprocal_rank_fusion(self, results_list, k=60):
        scores = defaultdict(float)
        for rank, doc in enumerate(results_list):
            #for rank, doc in enumerate(result):
                #breakpoint()
            scores[doc.page_content] += 1 / (rank + 1 + k)
        # Sort documents by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs
