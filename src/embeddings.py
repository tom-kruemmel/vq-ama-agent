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

    def retrieve_documents(self, queries: str, user_roles: list[str], top_k: int = 5):
        query_list = [q.strip() for q in queries.splitlines() if q.strip()]

        # Build Bedrock embedding model
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

        # Use the built-in filter argument to only match allowed_roles
        all_results = []
        # for q in query_list:
        #     docs = vector_store.similarity_search(
        #         q,
        #         k=top_k,
        #         filter={ 
        #             "allowed_roles": { "$in": user_roles }
        #         }
        #     )
        #     all_results.append(docs)
        for q in query_list:
            docs = vector_store.similarity_search(
                q,
                k=top_k,
                filter={ 
                    "allowed_roles": {"$in": user_roles}  # pass the whole list
                }
            )
            all_results.extend(docs)
        return all_results