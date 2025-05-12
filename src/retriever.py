import os
import pickle
from typing import List

import faiss
from langchain.embeddings import OpenAIEmbeddings


class VectorRetriever:
    """
    Wraps a FAISS vector index for retrieving text chunks based on semantic similarity.
    """

    def __init__(
        self,
        index_path: str,
        embeddings_path: str,
        metadata_path: str,
        embed_dim: int = 1536,
    ):
        """
        Args:
            index_path: Path to the FAISS index file (.faiss).
            embeddings_path: Path to the pickled text chunks list.
            metadata_path: Path to JSON or pickle containing metadata per chunk.
            embed_dim: Dimension of embedding vectors.
        """
        # Load FAISS index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(index_path)

        # Load text chunks and metadata
        with open(embeddings_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        # Embedding generator
        self.embedder = OpenAIEmbeddings()
        self.embed_dim = embed_dim

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """
        Returns the top-k most relevant text chunks for the given query.
        """
        # 1. Compute query embedding
        query_vector = self.embedder.embed_query(query)
        if len(query_vector) != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, got {len(query_vector)}"
            )

        # 2. Search FAISS index
        distances, indices = self.index.search(
            faiss.numpy_to_vector(np.array([query_vector], dtype="float32")), k
        )

        # 3. Retrieve chunks
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results