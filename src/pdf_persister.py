from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
#from langchain_community.document_loaders import UnstructuredPDFLoader
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
        heading_list: list[str] | None = None,
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
        self.loader = PyPDFDirectoryLoader(directory) # UnstructuredPDFLoader(directory, mode="single") 
        self.role_map = role_map
        self.heading_list= heading_list or []
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

    def merge_docs(self, raw_docs: list) -> list:
        docs_by_file = {}
        for doc in raw_docs:
            fname = os.path.basename(doc.metadata["source"])
            docs_by_file.setdefault(fname, []).append(doc.page_content)

        merged_docs = []
        for fname, pages in docs_by_file.items():
            merged_text = "\n\n".join(pages)
            merged_docs.append(
                Document(
                page_content=merged_text,
                metadata={"source": fname}
                )
            )
        return merged_docs

    
    def check_for_heading(self, text, headings, current_heading) -> str:
        for heading in headings:
            true_heading = "\n" + heading + "\n"
            if true_heading in text:
                return heading
        return current_heading

    def load_and_split_pdfs(self):
        raw_docs = self.loader.load()
        all_chunks = []

        merged_docs = self.merge_docs(raw_docs)
    
        for doc in merged_docs:
            fname = os.path.basename(doc.metadata["source"])
            roles = sorted(self.role_map.get(fname, []))
            headings = self.heading_list

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
                current_heading = self.check_for_heading(text, headings, current_heading)

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
                        # if current_heading != self.default_heading:
                        #     print(self.generate_doc_id(c.page_content, role))
                        #     breakpoint()
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
