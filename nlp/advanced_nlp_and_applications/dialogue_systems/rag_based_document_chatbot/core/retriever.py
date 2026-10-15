from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.embeddings import EmbeddingService
from core.vector_store import BaseVectorStore, VectorSearchResult
from utils.validators import validate_question


@dataclass
class RetrievalBundle:
    query: str
    results: list[VectorSearchResult]


class Retriever:
    def __init__(self, vector_store: BaseVectorStore, embedding_service: EmbeddingService, top_k: int = 4, threshold: float | None = None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.threshold = threshold

    def search(self, question: str, metadata_filter: dict[str, Any] | None = None) -> RetrievalBundle:
        clean_question = validate_question(question)
        query_embedding = self.embedding_service.embed_query(clean_question)
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            filter_by=metadata_filter,
            threshold=self.threshold,
        )
        return RetrievalBundle(query=clean_question, results=results)
