from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from hashlib import md5

class PdfPersister:

    def __init__(self, directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the PdfPersister with a directory containing PDF files.

        Args:
            directory (str): Path to the directory containing PDF files.
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between chunks.
        """
        self.loader = PyPDFDirectoryLoader(directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        

    def generate_doc_id(self, doc) -> str:
        return md5(doc.page_content.encode("utf-8")).hexdigest()

    def load_and_split_pdfs(self, pdf_folder: str):
        docs = self.loader.load()
        return self.text_splitter.split_documents(docs)
    
    def persist_pdfs(self):
        # Initialize Bedrock embeddings
        bedrock = boto3.client("bedrock-runtime")
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=bedrock
        )

        # Load and chunk documents
        docs = self.load_and_split_pdfs("data/")

        for doc in docs:
            doc.metadata["doc_id"] = self.generate_doc_id(doc)

        # Prepare lists for Chroma
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [doc.metadata["doc_id"] for doc in docs] 

        texts = [doc.page_content for doc in docs]
        embeddings_list = embeddings.embed_documents(texts)

        collection = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_docs",
            embedding_function=embeddings,
        )

        collection._collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=[doc.metadata for doc in docs],
        )

        print(f"Document count: {collection._collection.count()}")

        # Persist the vector store
        #vector_store.persist()
