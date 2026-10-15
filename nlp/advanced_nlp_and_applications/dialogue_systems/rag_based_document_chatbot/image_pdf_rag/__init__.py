"""OCR-enabled RAG pipeline for scanned and image-based PDFs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .ocr import OCREngine, OCRResult, TesseractOCR
from .pipeline import ImagePdfRAGPipeline

__all__ = ["ImagePdfRAGPipeline", "OCREngine", "OCRResult", "TesseractOCR"]
