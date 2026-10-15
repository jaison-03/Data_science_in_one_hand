from __future__ import annotations

from typing import Any

from config.settings import settings
from core.embeddings import EmbeddingService
from core.llm_service import LLMService
from core.prompts import build_rag_prompt
from core.retriever import Retriever
from core.vector_store import BaseVectorStore
from utils.validators import validate_question


class RAGPipeline:
    def __init__(self, vector_store: BaseVectorStore, embedding_service: EmbeddingService, llm_service: LLMService, retriever: Retriever):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.retriever = retriever

    def answer_question(self, question: str, chat_history: list[str] | None = None) -> dict[str, Any]:
        cleaned = validate_question(question)
        bundle = self.retriever.search(cleaned)
        results = bundle.results

        if not results:
            answer = "I could not find enough relevant information in the uploaded documents to answer that question confidently. Please try a question that is directly supported by the indexed content."
            return {
                "answer": answer,
                "sources": [],
                "grounded": False,
                "retrieval_results": [],
            }

        context_chunks = []
        for result in results:
            metadata = result.metadata or {}
            page_label = metadata.get("page_number")
            if page_label is not None:
                source_label = f"{metadata.get('source_file')} (Page {page_label})"
            else:
                source_label = metadata.get("source_file", "unknown source")
            context_chunks.append(f"[Source: {source_label}]\n{result.text}")

        context = "\n\n---\n\n".join(context_chunks)
        history = [] if not chat_history else chat_history[-4:]
        prompt = build_rag_prompt(cleaned, context, history)
        generated = self.llm_service.generate(prompt)

        answer_record = {
            "answer": generated,
            "sources": [
                {
                    "chunk_id": result.chunk_id,
                    "filename": (result.metadata or {}).get("source_file", "unknown"),
                    "page_number": (result.metadata or {}).get("page_number"),
                    "text": result.text,
                    "score": result.score,
                }
                for result in results
            ],
            "grounded": True,
            "retrieval_results": [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "metadata": result.metadata,
                    "score": result.score,
                }
                for result in results
            ],
        }
        return answer_record
