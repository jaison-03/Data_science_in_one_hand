from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from config.settings import settings
from core.text_cleaner import clean_text
from utils.logging_config import get_logger
from utils.validators import validate_file_type

logger = get_logger(__name__)


@dataclass
class LoadedDocument:
    filename: str
    source_path: str
    content: str
    document_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)
    page_number: int | None = None
    is_scanned_pdf: bool = False
    error: str | None = None


class DocumentLoader:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]

    def load(self, file_path: str | Path) -> LoadedDocument:
        path = Path(file_path)
        ext = validate_file_type(path.name)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if ext == ".pdf":
            return self._load_pdf(path)
        return self._load_text_document(path)

    def _load_text_document(self, file_path: Path) -> LoadedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError(f"Document '{file_path.name}' is empty or contains no usable text.")

        metadata = self._build_metadata(file_path, file_path.name, file_path.suffix.lower(), source_type="text")
        return LoadedDocument(
            filename=file_path.name,
            source_path=str(file_path),
            content=cleaned,
            document_type=file_path.suffix.lower().lstrip("."),
            metadata=metadata,
            pages=[cleaned],
            page_number=1,
        )

    def _load_pdf(self, file_path: Path) -> LoadedDocument:
        doc = fitz.open(file_path)
        pages: list[str] = []
        extracted_text = []

        try:
            for page_number in range(doc.page_count):
                page = doc[page_number]
                page_text = clean_text(page.get_text())
                pages.append(page_text)
                extracted_text.append(page_text)
            if not any(page_text.strip() for page_text in pages):
                raise ValueError("PDF appears to be scanned or blank. OCR is required to extract text from this document.")
            content = "\n\n".join(page for page in pages if page.strip())
            metadata = self._build_metadata(file_path, file_path.name, ".pdf", source_type="pdf", page_count=doc.page_count)
            return LoadedDocument(
                filename=file_path.name,
                source_path=str(file_path),
                content=content,
                document_type="pdf",
                metadata=metadata,
                pages=pages,
                page_number=1,
                is_scanned_pdf=not any(page_text.strip() for page_text in extracted_text),
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Failed to read PDF '{file_path.name}': {exc}") from exc
        finally:
            doc.close()

    def _build_metadata(
        self,
        file_path: Path,
        file_name: str,
        extension: str,
        source_type: str,
        page_count: int | None = None,
    ) -> dict[str, Any]:
        stat = file_path.stat()
        return {
            "filename": file_name,
            "extension": extension,
            "size_bytes": stat.st_size,
            "source_type": source_type,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "page_count": page_count,
            "project_root": str(self.project_root),
        }


def load_document(file_path: str | Path) -> LoadedDocument:
    return DocumentLoader().load(file_path)
