from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from hashlib import md5
import os
import copy

class PdfPersister:
    def __init__(
        self,
        directory: str,
        role_map: dict[str, list[str]],
        heading_role_map: dict[str, list[str]] | None = None,
        default_heading: str = "default",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            directory: folder containing your PDFs.
            role_map: mapping from PDF-filename → list of roles allowed.
            heading_role_map: optional mapping from PDF-filename → list of headings to track.
            default_heading: label to use when no heading context applies.
        """
        self.loader = PyPDFDirectoryLoader(directory)
        self.role_map = role_map
        self.heading_role_map = heading_role_map or {}
        self.default_heading = default_heading
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def generate_doc_id(self, text: str, role: str) -> str:
        # unique per text + role
        return md5(text.encode("utf-8") + role.encode("utf-8")).hexdigest()

    def load_and_split_pdfs(self):
        raw_docs = self.loader.load()
        all_chunks = []

        for doc in raw_docs:
            fname = os.path.basename(doc.metadata["source"])
            roles = sorted(self.role_map.get(fname, []))
            headings = self.heading_role_map.get(fname, [])

            # if headings configured but none appear, emit whole page per role
            if headings and not any(h in doc.page_content for h in headings):
                for role in roles:
                    chunk = copy.deepcopy(doc)
                    chunk.metadata.update({
                        "allowed_roles": role,
                        "heading": self.default_heading,
                        "doc_id": self.generate_doc_id(chunk.page_content, role)
                    })
                    all_chunks.append(chunk)
                continue

            # split into paragraphs and track current heading
            paragraphs = doc.page_content.split("\n\n")
            current_heading = self.default_heading

            for para in paragraphs:
                text = para.strip()
                if not text:
                    continue

                # heading marker
                if text in headings:
                    current_heading = text
                    continue

                # chunk content
                if len(text) > self.chunk_size:
                    tmp = copy.deepcopy(doc)
                    tmp.page_content = text
                    subchunks = self.fallback_splitter.split_documents([tmp])
                else:
                    subchunks = [copy.deepcopy(doc)]
                    subchunks[0].page_content = text

                # save one chunk per role
                for chunk in subchunks:
                    for role in roles:
                        c = copy.deepcopy(chunk)
                        c.metadata.update({
                            "allowed_roles": role,
                            "heading": current_heading,
                            "doc_id": self.generate_doc_id(c.page_content, role)
                        })
                        all_chunks.append(c)

        return all_chunks

    def persist_pdfs(self):
        bedrock = boto3.client("bedrock-runtime")
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=bedrock
        )

        docs = self.load_and_split_pdfs()
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        ids = [m["doc_id"] for m in metadatas]

        embs = embeddings.embed_documents(texts)

        collection = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_with_roles",
            embedding_function=embeddings,
        )
        collection._collection.upsert(
            ids=ids,
            embeddings=embs,
            documents=texts,
            metadatas=metadatas,
        )
        print("Total docs:", collection._collection.count())
