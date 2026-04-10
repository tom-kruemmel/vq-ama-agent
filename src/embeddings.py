import logging
import os

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

from .bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class Embeddings:

    def __init__(
        self,
        persist_directory: str = "./chroma_store",
        collection_name: str = "pdf_with_roles",
        model_id: str = "amazon.titan-embed-text-v2:0",
        enable_bm25: bool = True,
    ):
        """
        Initializes the embeddings class once and reuses the embedding model
        and vector store across retrieve calls.

        When enable_bm25 is True (default), a BM25 sparse index is built
        in-memory from the Chroma collection at startup for hybrid retrieval.
        """
        bedrock_client = boto3.client("bedrock-runtime")
        self._embedding_model = BedrockEmbeddings(
            client=bedrock_client,
            model_id=model_id,
        )
        self._vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self._embedding_model,
        )

        # Build BM25 index from Chroma for hybrid retrieval
        if enable_bm25:
            self._bm25 = BM25Retriever.from_chroma(self._vector_store)
        else:
            self._bm25 = None
    def split_string_by_newlines(self, input_string):
        return input_string.split('\n')

    def print_headings(self, all_results):
        """
        Prints the headings of the documents in all_results.
        """
        for doc in all_results:
            print(doc.metadata.get("heading", "No heading found"))

    def retrieve_documents(
        self,
        queries: list[str] | str,
        headings: list[str] | None = None,
        top_k: int = 10
    ) -> list:
        """
        Retrieve up to top_k chunks per query via dense search (and BM25 if enabled),
        returning one ranked list per retriever per query for RRF fusion.

        queries: list of search strings (or a newline-separated string for backwards compat).
        headings: optional list of headings to restrict to (e.g., ["PUBLIC", "CONFIDENTIAL"]).
        top_k: number of results per query per retriever.
        Returns: list of ranked lists (one per query per retriever).
        """
        if isinstance(queries, str):
            query_list = [q.strip() for q in queries.splitlines() if q.strip()]
        else:
            query_list = queries

        # Build filter for headings only
        combined_filter = None
        if headings is not None:
            combined_filter = {"heading": {"$in": headings}}

        all_results: list[list] = []
        for q in query_list:
            # Dense retrieval
            docs = self._vector_store.similarity_search(
                q,
                k=top_k,
                filter=combined_filter
            )
            all_results.append(docs)

            # Sparse (BM25) retrieval
            if self._bm25 is not None:
                bm25_docs = self._bm25.retrieve(q, top_k=top_k, headings=headings)
                all_results.append(bm25_docs)

        logger.info(
            "Hybrid retrieval: %d queries × %s retrievers = %d ranked lists",
            len(query_list),
            "2 (dense+BM25)" if self._bm25 else "1 (dense only)",
            len(all_results),
        )

        return all_results