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

    def retrieve_documents(self, queries, top_k=5):
        query_list = self.split_string_by_newlines(queries)
        bedrock_client = boto3.client(service_name='bedrock-runtime')
        embedding_model = BedrockEmbeddings(
            client=bedrock_client,
            model_id="amazon.titan-embed-text-v2:0"
        )

        # Initialize the vector store
        vector_store = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_docs",
            embedding_function=embedding_model
        )
        all_results = []
        for q in query_list:
            if len(q) == 0:
                continue 
            docs = vector_store.similarity_search(q, k=top_k)
            all_results.append(docs)
        return all_results
