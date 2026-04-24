from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from hashlib import md5
import json
import logging
import os
import copy
from itertools import islice
from pathlib import Path

logger = logging.getLogger(__name__)

class PdfPersister:
    def __init__(
        self,
        directory: str,
        heading_list: list[str] | None = None,
        default_heading: str = "PUBLIC",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            directory: folder containing your PDFs.
            heading_list: optional list of headings to track (e.g. ["PUBLIC", "CONFIDENTIAL"]).
            default_heading: label to use when no heading context applies.
        """
        self.loader = PyPDFDirectoryLoader(directory) # UnstructuredPDFLoader(directory, mode="single") 
        self.heading_list= heading_list or []
        self.default_heading = default_heading
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self._manifest_path = os.path.join(
            os.path.dirname(directory) or ".", ".pdf_manifest.json"
        )

    # ------------------------------------------------------------------
    # Change detection: hash all PDFs so we can skip re-indexing
    # ------------------------------------------------------------------

    def _compute_directory_hash(self) -> str:
        """Return a single MD5 digest covering every PDF file's name, size and content."""
        h = md5()
        pdf_dir = Path(self.loader.path)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        for p in pdf_files:
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
            h.update(p.read_bytes())
        return h.hexdigest()

    def _load_manifest(self) -> dict:
        if os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self, directory_hash: str) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump({"directory_hash": directory_hash}, f)

    def has_changes(self) -> bool:
        """Return True if the PDFs have changed since the last persist."""
        current_hash = self._compute_directory_hash()
        manifest = self._load_manifest()
        return current_hash != manifest.get("directory_hash")

    def generate_doc_id(self, text: str, heading: str) -> str:
        # unique per text + heading
        return md5(
            text.encode("utf-8") + heading.encode("utf-8")
        ).hexdigest()

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
            if text.startswith(heading) or ("\n" + heading + "\n") in text:
                return heading
        return current_heading

    def load_and_split_pdfs(self):
        raw_docs = self.loader.load()
        all_chunks = []

        merged_docs = self.merge_docs(raw_docs)

        for doc in merged_docs:
            fname = os.path.basename(doc.metadata["source"])
            headings = self.heading_list

            # if headings configured but none appear, emit whole page with default heading
            if headings and not any(h in doc.page_content for h in headings):
                chunk = copy.deepcopy(doc)
                chunk.metadata.update({
                    "heading": self.default_heading,
                    "doc_id": self.generate_doc_id(chunk.page_content, self.default_heading)
                })
                all_chunks.append(chunk)
                continue

            paragraphs = doc.page_content.split("\n\n")
            current_heading = self.default_heading

            def _emit_text_as_chunks(text, heading):
                """Split large text if needed, then emit one chunk with given heading."""
                # Skip junk chunks (page numbers, TOC stubs, etc.)
                if len(text.strip()) < 50:
                    return

                # Prepend section heading for better retrieval context
                if heading and heading != self.default_heading:
                    prefixed_text = f"{heading}\n{text}"
                else:
                    prefixed_text = text

                if len(prefixed_text) > self.chunk_size:
                    tmp = copy.deepcopy(doc)
                    tmp.page_content = prefixed_text
                    subchunks = self.fallback_splitter.split_documents([tmp])
                else:
                    subchunks = [copy.deepcopy(doc)]
                    subchunks[0].page_content = prefixed_text

                for chunk in subchunks:
                    chunk.metadata.update({
                        "heading": heading,
                        "doc_id": self.generate_doc_id(chunk.page_content, heading)
                    })
                    all_chunks.append(chunk)

            def _find_next_heading(text):
                """Return (idx, heading) for earliest heading occurrence in text, else (None, None)."""
                if not headings:
                    return None, None
                best = None
                best_h = None
                for h in headings:
                    i = text.find(h)
                    if i != -1 and (best is None or i < best):
                        best = i
                        best_h = h
                return best, best_h

            for para in paragraphs:
                text = para.strip()
                if not text:
                    continue

                # Split within paragraph if a heading appears mid-paragraph.
                # We walk left-to-right so multiple headings in one paragraph are handled.
                remaining = text
                while remaining:
                    idx, h = _find_next_heading(remaining)

                    if idx is None:
                        # no heading at all in the remaining text
                        _emit_text_as_chunks(remaining, current_heading)
                        break

                    if idx == 0:
                        # heading at start: update heading, then emit this part (which starts with heading)
                        current_heading = self.check_for_heading(remaining, headings, current_heading)
                        _emit_text_as_chunks(remaining, current_heading)
                        break

                    # heading is mid-chunk => split into two chunks
                    before = remaining[:idx].strip()
                    after = remaining[idx:].strip()

                    if before:
                        _emit_text_as_chunks(before, current_heading)

                    # update heading for the part starting at the heading
                    current_heading = self.check_for_heading(after, headings, current_heading)

                    # continue scanning the remainder from the heading onward
                    remaining = after

        return all_chunks
    

    def persist_pdfs(self, batch_size: int = 1000):
        current_hash = self._compute_directory_hash()
        manifest = self._load_manifest()
        if current_hash == manifest.get("directory_hash"):
            logger.info("PDFs unchanged since last persist — skipping re-indexing.")
            return

        def _batched(iterable, n):
            it = iter(iterable)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        bedrock = boto3.client("bedrock-runtime")
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=bedrock
        )

        # Open/create the collection first so we can query for existing IDs
        collection = Chroma(
            persist_directory="./chroma_store",
            collection_name="pdf_with_roles",
            embedding_function=embeddings,
        )

        # Prepare all candidate docs/chunks
        docs = self.load_and_split_pdfs()
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        ids = [m["doc_id"] for m in metadatas]

        if not texts:
            print("No documents found to persist.")
            return

        # Embed all texts (Chroma upsert handles deduplication, so we
        # skip the previous two-pass "check then insert" pattern).
        embs = embeddings.embed_documents(texts)

        # Upsert in batches to stay within Chroma limits
        for batch in _batched(list(zip(ids, embs, texts, metadatas)), batch_size):
            b_ids, b_embs, b_texts, b_metas = zip(*batch)
            collection._collection.upsert(
                ids=list(b_ids),
                embeddings=list(b_embs),
                documents=list(b_texts),
                metadatas=list(b_metas),
            )

        print(
            f"Upserted {len(ids)} chunks. "
            f"Total docs now: {collection._collection.count()}"
        )
        self._save_manifest(current_hash)
        logger.info("PDF manifest updated.")

