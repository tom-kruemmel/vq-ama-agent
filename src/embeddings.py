from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
import boto3

class Embeddings:

    def __init__(self):
        """
        Initializes the embeddings class.
        This class is responsible for generating embeddings and retrieving documents based on those embeddings.
        """
        pass

    # Initialize the embedding mode
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
        queries: str,
        user_roles: list[str],
        headings: list[str] | None = None,
        top_k: int = 5
    ) -> list:
        """
        Retrieve up to top_k chunks per query, filtered by user_roles and optional headings.

        queries: newline-separated search strings.
        user_roles: roles of the current user.
        headings: optional list of headings to restrict to (e.g., ["Section 1", "default"]).
        top_k: number of results per query.
        Returns: flat list of matching Document chunks.
        """
        query_list = [q.strip() for q in queries.splitlines() if q.strip()]

        # Initialize Bedrock embedding model
        bedrock_client = boto3.client("bedrock-runtime")
        embedding_model = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v2:0",
        )

        # Load the vector store
        vector_store = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_with_roles",
            embedding_function=embedding_model
        )

        # Build a list of your individual filters
        filters = [
            {"allowed_roles": {"$in": user_roles}}
        ]
        if headings is not None:
            filters.append({"heading": {"$in": headings}})

        # Wrap them in a single '$and'
        combined_filter = {"$and": filters}


        all_results: list = []
        for q in query_list:
            docs = vector_store.similarity_search(
                q,
                k=top_k,
                filter=combined_filter
            )
            all_results.extend(docs)

       # self.print_headings(all_results)
        return all_results