from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class HandwritingOCRResult:
    text: str
    confidence: float | None
    text_type: str = "unknown"


class HandwritingOCREngine(Protocol):
    def extract(self, image: Image.Image) -> HandwritingOCRResult:
        """Extract text and optional confidence from a rendered page."""


def preprocess_page_image(image: Image.Image) -> Image.Image:
    """Enhance contrast and deskew without binarizing handwriting or punctuation."""
    grayscale = ImageOps.autocontrast(ImageOps.exif_transpose(image).convert("L"))
    pixels = np.asarray(grayscale)
    binary = cv2.threshold(
        pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    coordinates = cv2.findNonZero(binary)
    if coordinates is not None and len(coordinates) > 20:
        angle = cv2.minAreaRect(coordinates)[-1]
        correction = -(90 + angle) if angle < -45 else -angle
        if 0.5 <= abs(correction) <= 7:
            grayscale = grayscale.rotate(
                correction,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=255,
            )
    return ImageEnhance.Contrast(grayscale).enhance(1.15)


class TextractOCR:
    """AWS Textract OCR engine, which supports printed and handwritten text."""

    def __init__(self, *, region_name: str | None = None, client: Any | None = None):
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "AWS Textract needs boto3. Install it with `python -m pip install boto3`."
                ) from exc
            client = boto3.client("textract", region_name=region_name)
        self.client = client

    def extract(self, image: Image.Image) -> HandwritingOCRResult:
        processed = preprocess_page_image(image)
        payload = BytesIO()
        processed.save(payload, format="PNG", optimize=True)
        image_bytes = payload.getvalue()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError(
                "The rendered page exceeds Textract's 10 MB synchronous image limit. "
                "Try a lower render resolution or use local OCR."
            )

        try:
            response = self.client.detect_document_text(
                Document={"Bytes": image_bytes}
            )
        except Exception as exc:
            raise RuntimeError(
                "AWS Textract could not process this page. Check AWS credentials, "
                f"region, permissions, and service availability: {exc}"
            ) from exc

        lines = []
        confidences = []
        text_types: set[str] = set()
        for block in response.get("Blocks", []):
            if block.get("BlockType") == "WORD":
                text_type = str(block.get("TextType", "")).lower()
                if text_type in {"handwriting", "printed"}:
                    text_types.add(text_type)
            if block.get("BlockType") != "LINE" or not block.get("Text"):
                continue
            lines.append(str(block["Text"]))
            try:
                confidence = float(block["Confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            confidences.append(confidence)
        confidence = sum(confidences) / len(confidences) if confidences else None
        if len(text_types) > 1:
            text_type = "mixed"
        elif text_types:
            text_type = next(iter(text_types))
        else:
            text_type = "unknown"
        return HandwritingOCRResult(
            text="\n".join(lines),
            confidence=confidence,
            text_type=text_type,
        )


class LocalTesseractOCR:
    """Local fallback. Tesseract is not handwriting-specialized."""

    def __init__(self):
        try:
            from image_pdf_rag.ocr import TesseractOCR
        except ImportError:
            from ..image_pdf_rag.ocr import TesseractOCR
        self.engine = TesseractOCR(config="--oem 3 --psm 6", threshold=None)

    def extract(self, image: Image.Image) -> HandwritingOCRResult:
        result = self.engine.extract(preprocess_page_image(image))
        return HandwritingOCRResult(
            text=result.text,
            confidence=result.confidence,
        )
