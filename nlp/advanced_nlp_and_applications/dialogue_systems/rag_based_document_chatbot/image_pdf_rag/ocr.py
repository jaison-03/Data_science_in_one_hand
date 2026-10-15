from __future__ import annotations

import re
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None


class OCREngine(Protocol):
    def extract(self, image: Image.Image) -> OCRResult:
        """Extract text and optional word confidence from a page image."""


def clean_ocr_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned and re.search(r"\w", cleaned, flags=re.UNICODE):
            lines.append(cleaned)
    return "\n".join(lines)


def preprocess_image(image: Image.Image, threshold: int | None = 180) -> Image.Image:
    grayscale = ImageOps.autocontrast(image.convert("L"))
    if threshold is None:
        return grayscale
    return grayscale.point(lambda pixel: 255 if pixel >= threshold else 0, mode="1")


def _find_tesseract(pytesseract) -> str | None:
    configured = str(pytesseract.pytesseract.tesseract_cmd)
    if Path(configured).is_file():
        return configured
    discovered = shutil.which(configured) or shutil.which("tesseract")
    if discovered:
        return discovered

    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Tesseract-OCR"
        / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Tesseract-OCR"
        / "tesseract.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def get_tesseract_status() -> tuple[bool, str]:
    try:
        import pytesseract
    except ImportError:
        return (
            False,
            "The pytesseract Python package is missing. From the chatbot project "
            "folder, run: python -m pip install -r requirements.txt",
        )

    executable = _find_tesseract(pytesseract)
    if executable is None:
        return (
            False,
            "The Tesseract OCR application is not installed or is not on PATH. "
            "Install Tesseract, then restart Streamlit. In an Administrator "
            "PowerShell, run: choco install tesseract -y",
        )

    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        pytesseract.get_tesseract_version()
    except (OSError, subprocess.CalledProcessError, pytesseract.TesseractNotFoundError) as exc:
        return False, f"Tesseract was found at '{executable}' but could not run: {exc}"
    return True, f"Tesseract OCR is ready ({executable})."


class TesseractOCR:
    def __init__(
        self,
        language: str = "eng",
        config: str = "--oem 3 --psm 3",
        threshold: int | None = 180,
    ):
        try:
            import pytesseract
        except ImportError as exc:
            raise ModuleNotFoundError(
                "The pytesseract Python package is missing. From the chatbot project "
                "folder, run: python -m pip install -r requirements.txt"
            ) from exc

        self._pytesseract = pytesseract
        executable = _find_tesseract(pytesseract)
        if executable is not None:
            pytesseract.pytesseract.tesseract_cmd = executable
        self.language = language
        self.config = config
        self.threshold = threshold

    def extract(self, image: Image.Image) -> OCRResult:
        processed = preprocess_image(image, self.threshold)
        try:
            data = self._pytesseract.image_to_data(
                processed,
                lang=self.language,
                config=self.config,
                output_type=self._pytesseract.Output.DICT,
            )
        except self._pytesseract.pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "The Tesseract OCR application is not installed or is not on PATH. "
                "Install Tesseract, then restart Streamlit. In an Administrator "
                "PowerShell, run: choco install tesseract -y"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc

        grouped: dict[tuple[str, str, str], list[str]] = {}
        confidences: list[float] = []
        for index, raw_word in enumerate(data.get("text", [])):
            word = str(raw_word).strip()
            if not word:
                continue
            key = tuple(
                str(data[field][index])
                for field in ("block_num", "par_num", "line_num")
            )
            grouped.setdefault(key, []).append(word)
            try:
                confidence = float(data["conf"][index])
                if confidence >= 0:
                    confidences.append(confidence)
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        text = clean_ocr_text("\n".join(" ".join(words) for words in grouped.values()))
        confidence = sum(confidences) / len(confidences) if confidences else None
        return OCRResult(text=text, confidence=confidence)
