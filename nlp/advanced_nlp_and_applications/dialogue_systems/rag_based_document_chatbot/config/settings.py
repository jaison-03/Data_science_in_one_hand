from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    vector_store: str = Field(default="chroma", alias="VECTOR_STORE")
    chroma_persist_directory: str = Field(default=str(PROJECT_ROOT / "database" / "chroma_db"), alias="CHROMA_PERSIST_DIRECTORY")
    faiss_index_directory: str = Field(default=str(PROJECT_ROOT / "database" / "faiss_index"), alias="FAISS_INDEX_DIRECTORY")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=4, alias="RETRIEVAL_TOP_K")
    similarity_threshold: float | None = Field(default=None, alias="SIMILARITY_THRESHOLD")
    max_upload_size_mb: int = Field(default=15, alias="MAX_UPLOAD_SIZE_MB")
    supported_extensions: str = Field(default=".pdf,.txt,.md,.markdown", alias="SUPPORTED_EXTENSIONS")

    @field_validator("vector_store")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"chroma", "faiss"}:
            raise ValueError("VECTOR_STORE must be either 'chroma' or 'faiss'.")
        return normalized

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        if value < 100 or value > 4000:
            raise ValueError("CHUNK_SIZE must be between 100 and 4000 characters.")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int) -> int:
        if value < 0 or value >= 2000:
            raise ValueError("CHUNK_OVERLAP must be between 0 and 2000.")
        return value

    @field_validator("retrieval_top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20.")
        return value

    @property
    def supported_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.supported_extensions.split(",") if ext.strip()]


settings = Settings(**os.environ)
