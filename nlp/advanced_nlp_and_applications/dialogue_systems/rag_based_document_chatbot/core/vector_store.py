from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - used when FAISS is unavailable
    faiss = None

try:
    import chromadb
except ImportError:  # pragma: no cover - optional backend
    chromadb = None

from config.settings import settings


@dataclass
class VectorSearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float | None


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def similarity_search(self, query_embedding: list[float], top_k: int, filter_by: dict[str, Any] | None = None, threshold: float | None = None) -> list[VectorSearchResult]:  # pragma: no cover
        ...

    @abstractmethod
    def count(self) -> int:  # pragma: no cover
        ...

    @abstractmethod
    def delete_document(self, document_name: str) -> None:  # pragma: no cover
        ...


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "rag_document_chatbot"):
        if chromadb is None:
            raise ModuleNotFoundError("ChromaDB is not installed. Install chromadb to use the Chroma backend.")
        self.persist_dir = Path(settings.chroma_persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk.get("metadata", {}) for chunk in chunks]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def similarity_search(self, query_embedding: list[float], top_k: int, filter_by: dict[str, Any] | None = None, threshold: float | None = None) -> list[VectorSearchResult]:
        query = np.asarray([query_embedding], dtype=np.float32)
        response = self.collection.query(
            query_embeddings=query.tolist(),
            n_results=top_k,
            where=filter_by,
            include=["documents", "metadatas", "distances", "ids"],
        )
        results: list[VectorSearchResult] = []
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            score = None
            if i < len(distances) and distances[i] is not None:
                score = max(0.0, 1.0 - float(distances[i]))
            if threshold is not None and score is not None and score < threshold:
                continue
            results.append(VectorSearchResult(chunk_id=str(ids[i]), text=document, metadata=metadata, score=score))
        return results

    def count(self) -> int:
        return self.collection.count()

    def delete_document(self, document_name: str) -> None:
        ids = self.collection.get(where={"source_file": document_name}, include=[]).get("ids", [])
        if ids:
            self.collection.delete(ids=ids)


class FAISSVectorStore(BaseVectorStore):
    def __init__(self, dimension: int = 384, collection_name: str = "rag_document_chatbot"):
        self.index = faiss.IndexFlatIP(dimension) if faiss is not None else None
        self.dimension = dimension
        self.collection_name = collection_name
        self.persist_dir = Path(settings.faiss_index_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_map: dict[str, dict[str, Any]] = {}
        self.text_map: dict[str, str] = {}
        self.chunk_order: list[str] = []
        self._vector_map: dict[str, np.ndarray] = {}

    def _normalize(self, embeddings: list[list[float]]) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.size == 0:
            return array.reshape(0, self.dimension)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def add_documents(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        normalized = self._normalize(embeddings)
        start_index = len(self.chunk_order)
        for chunk, vector in zip(chunks, normalized):
            chunk_id = str(chunk["chunk_id"])
            self.metadata_map[chunk_id] = chunk.get("metadata", {})
            self.text_map[chunk_id] = chunk["text"]
            self._vector_map[chunk_id] = vector
            self.chunk_order.append(chunk_id)
        if not self._vector_map:
            return

        vectors = np.vstack([self._vector_map[chunk_id] for chunk_id in self.chunk_order[start_index:]])
        if vectors.size and self.index is not None:
            self.index.add(vectors)

    def similarity_search(self, query_embedding: list[float], top_k: int, filter_by: dict[str, Any] | None = None, threshold: float | None = None) -> list[VectorSearchResult]:
        if not self.chunk_order:
            return []
        normalized_query = np.asarray([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(normalized_query, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        normalized_query = normalized_query / norm
        if self.index is not None:
            scores, indices = self.index.search(normalized_query, min(top_k, self.index.ntotal))
            ranked = zip(scores[0], indices[0])
        else:
            candidates = [
                (float(np.dot(normalized_query[0], self._vector_map[chunk_id])), index)
                for index, chunk_id in enumerate(self.chunk_order)
                if chunk_id in self._vector_map
            ]
            ranked = sorted(candidates, key=lambda item: item[0], reverse=True)

        results: list[VectorSearchResult] = []
        for score, index in ranked:
            if index < 0:
                continue
            chunk_id = self.chunk_order[int(index)] if 0 <= int(index) < len(self.chunk_order) else ""
            if not chunk_id or chunk_id not in self.metadata_map:
                continue
            metadata = self.metadata_map[chunk_id]
            text = self.text_map.get(chunk_id, "")
            relevance = float(score)
            if threshold is not None and relevance < threshold:
                continue
            if filter_by is not None and any(metadata.get(key) != value for key, value in filter_by.items()):
                continue
            results.append(VectorSearchResult(chunk_id=chunk_id, text=text, metadata=metadata, score=relevance))
            if len(results) == top_k:
                break
        return results

    def count(self) -> int:
        return len(self.chunk_order)

    def delete_document(self, document_name: str) -> None:
        to_delete = [chunk_id for chunk_id, metadata in self.metadata_map.items() if metadata.get("source_file") == document_name]
        if not to_delete:
            return
        for chunk_id in to_delete:
            self.metadata_map.pop(chunk_id, None)
            self.text_map.pop(chunk_id, None)
            self._vector_map.pop(chunk_id, None)
            if chunk_id in self.chunk_order:
                self.chunk_order.remove(chunk_id)
        self.index = faiss.IndexFlatIP(self.dimension) if faiss is not None else None
        if self._vector_map and self.index is not None:
            vectors = np.vstack([self._vector_map[chunk_id] for chunk_id in self.chunk_order])
            self.index.add(vectors)


def create_vector_store(backend: str | None = None):
    backend_name = (backend or settings.vector_store).strip().lower()
    if backend_name == "chroma":
        if chromadb is None:
            return FAISSVectorStore(dimension=384)
        return ChromaVectorStore()
    if backend_name == "faiss":
        return FAISSVectorStore(dimension=384)
    raise ValueError(f"Unsupported vector backend '{backend_name}'.")
