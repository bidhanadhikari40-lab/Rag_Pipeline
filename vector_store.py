"""
Embedding generation (Sentence-Transformers, fully local) and
vector storage / similarity search (FAISS, fully local).
"""

import os
import re
import contextlib
import hashlib

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback for broken environments
    SentenceTransformer = None

from chunking import Chunk

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality, free


class VectorStore:
    """
    Wraps a FAISS index plus the chunk metadata needed to map
    vector search results back to their original text and source file.
    """

    def __init__(self):
        self._model = None
        self.index = None
        self.chunks: list[Chunk] = []
        self.dimension = None

    @property
    def model(self):
        if self._model is None:
            if SentenceTransformer is not None:
                try:
                    with open(os.devnull, "w", encoding="utf-8") as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
                except Exception:
                    self._model = _FallbackEmbeddingModel()
            else:
                self._model = _FallbackEmbeddingModel()
        return self._model

    def build(self, chunks: list[Chunk]):
        """Embeds all chunks and builds a fresh FAISS index from scratch."""
        self.chunks = chunks
        if not chunks:
            self.index = None
            return

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts).astype("float32")

        self.dimension = embeddings.shape[1]
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Returns the top_k most relevant chunks for the query, each as a dict
        with chunk text, source file, chunk index, and similarity score.
        """
        if not self.is_ready():
            return []

        query_embedding = self.model.encode([query]).astype("float32")

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "score": float(score),
                }
            )
        return results


class _FallbackEmbeddingModel:
    """Fully local embedding fallback that does not depend on model downloads."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", text.lower())
        vector = np.zeros(self.dimensions, dtype=np.float32)
        if not tokens:
            return vector

        counts = {}
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            counts[index] = counts.get(index, 0) + 1

        for index, count in counts.items():
            vector[index] = float(count)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = [self._embed_one(text) for text in texts]
        return np.stack(vectors, axis=0)
