import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    Semantic retriever using Sentence-Transformers + FAISS
    """

    def __init__(
        self,
        faiss_index_path: str,
        chunks_path: str,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)

        # Load stored chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Embed the query and retrieve top-k semantically similar chunks
        """
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, top_k)

        return [self.chunks[i] for i in indices[0]]
