from collections import defaultdict

class RankFusion:
    def reciprocal_rank_fusion(self, results_list, k=60):
        scores = defaultdict(float)
        for rank, doc in enumerate(results_list):
            #for rank, doc in enumerate(result):
                #breakpoint()
            scores[doc.page_content] += 1 / (rank + 1 + k)
        # Sort documents by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]