from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from hashlib import md5
import os
import json
import copy

class PdfPersister:
    def __init__(
        self,
        directory: str,
        role_map: dict[str, list[str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            directory: folder containing your PDFs.
            role_map: mapping from PDF-filename (or dirname) → list of roles allowed.
                      e.g. { "confidential.pdf": ["admin"], "public.pdf": ["admin","user"] }
        """
        self.loader = PyPDFDirectoryLoader(directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.role_map = role_map

    def generate_doc_id(self, doc, role) -> str:
        return md5(doc.page_content.encode("utf-8") + role.encode("utf-8")).hexdigest()

    def load_and_split_pdfs(self):
        docs = self.loader.load()
        split = self.text_splitter.split_documents(docs)
        all_docs = []

        for doc in split:
            fname = os.path.basename(doc.metadata["source"])
            allowed_roles = self.role_map.get(fname, [])  # default: nobody
            allowed_roles.sort()

            for role in allowed_roles:
                doc_copy = copy.deepcopy(doc)
                doc_copy.metadata["allowed_roles"] = role
                doc_copy.metadata["doc_id"] = self.generate_doc_id(doc_copy, role)
                # Optionally save doc_copy to disk or DB here
                all_docs.append(doc_copy)

        return all_docs

    def persist_pdfs(self):
        # Initialize Bedrock embeddings
        bedrock = boto3.client("bedrock-runtime")
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=bedrock
        )

        # 2) load & split + tag
        docs = self.load_and_split_pdfs()
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        ids   = [d.metadata["doc_id"]       for d in docs]

        # 3) embed
        embeddings_list = embeddings.embed_documents(texts)

        # 4) upsert into a single Chroma collection
        collection = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_with_roles",
            embedding_function=embeddings,
        )
        collection._collection.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metas,
        )
        print("Total docs:", collection._collection.count())
