import os

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma


class Embeddings:

    def __init__(
        self,
        persist_directory: str = "./chroma_store",
        collection_name: str = "pdf_with_roles",
        model_id: str = "amazon.titan-embed-text-v2:0",
    ):
        """
        Initializes the embeddings class once and reuses the embedding model
        and vector store across retrieve calls.
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
        user_roles: list[str],
        headings: list[str] | None = None,
        top_k: int = 10
    ) -> list:
        """
        Retrieve up to top_k chunks per query, filtered by user_roles and optional headings.

        queries: list of search strings (or a newline-separated string for backwards compat).
        user_roles: roles of the current user.
        headings: optional list of headings to restrict to (e.g., ["Section 1", "default"]).
        top_k: number of results per query.
        Returns: flat list of matching Document chunks.
        """
        if isinstance(queries, str):
            query_list = [q.strip() for q in queries.splitlines() if q.strip()]
        else:
            query_list = queries

        # Build a list of your individual filters
        filters = [
            {"allowed_roles": {"$in": user_roles}}
        ]
        if headings is not None:
            filters.append({"heading": {"$in": headings}})

        # Wrap them in a single '$and'
        combined_filter = {"$and": filters}


        all_results: list[list] = []
        for q in query_list:
            docs = self._vector_store.similarity_search(
                q,
                k=top_k,
                filter=combined_filter
            )
            all_results.append(docs)  # Keep as list of lists for proper RRF

        return all_results