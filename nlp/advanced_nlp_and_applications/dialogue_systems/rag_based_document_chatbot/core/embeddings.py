from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name, cache_folder=str(CACHE_DIR))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32).tolist()

    def embed_query(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty when computing embeddings.")
        embedding = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embedding[0], dtype=np.float32).tolist()
