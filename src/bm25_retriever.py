from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.schema import Document
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer with lowercasing.

    Sufficient for mixed English/German enterprise docs.
    Replace with nltk.word_tokenize for better multilingual handling.
    """
    return text.lower().split()


class BM25Retriever:
    """In-memory BM25 sparse retriever built from existing Chroma documents.

    Initialised once at startup by pulling all stored chunks from Chroma,
    then serves keyword-based retrieval alongside the dense vector search.
    """

    def __init__(self, documents: list[Document]):
        """Build BM25 index from a list of LangChain Documents.

        Args:
            documents: Corpus documents (same chunks stored in Chroma).
        """
        self._documents = documents
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(documents))

    @classmethod
    def from_chroma(
        cls,
        chroma_store,
        headings: list[str] | None = None,
    ) -> "BM25Retriever":
        """Build a BM25Retriever by extracting documents from a Chroma vector store.

        Args:
            chroma_store: A langchain_chroma.Chroma instance.
            headings: If given, only index documents whose heading metadata
                      is in this list.  Pass None to index everything.
        """
        collection = chroma_store._collection.get(
            include=["documents", "metadatas"],
        )
        documents = []
        for text, meta in zip(
            collection["documents"], collection["metadatas"]
        ):
            if headings is not None:
                if meta.get("heading") not in headings:
                    continue
            documents.append(Document(page_content=text, metadata=meta or {}))
        logger.info(
            "Loaded %d documents from Chroma for BM25 index (headings=%s)",
            len(documents),
            headings,
        )
        return cls(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        headings: list[str] | None = None,
    ) -> list[Document]:
        """Retrieve top-k documents by BM25 score.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            headings: Optional heading filter applied post-retrieval.

        Returns:
            List of Document objects sorted by BM25 score descending.
        """
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = scores.argsort()[-top_k * 3:][::-1]  # over-fetch for filtering

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self._documents[idx]
            if headings is not None and doc.metadata.get("heading") not in headings:
                continue
            results.append(doc)
            if len(results) >= top_k:
                break

        return results

    def retrieve_multi(
        self,
        queries: list[str],
        top_k: int = 10,
        headings: list[str] | None = None,
    ) -> list[list[Document]]:
        """Retrieve for multiple queries (one ranked list per query).

        Returns the same shape as Embeddings.retrieve_documents() so both
        can be concatenated before RRF.
        """
        return [self.retrieve(q, top_k=top_k, headings=headings) for q in queries]
