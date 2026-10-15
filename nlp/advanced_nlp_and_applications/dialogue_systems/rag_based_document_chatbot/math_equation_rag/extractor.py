from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class ExtractedPage:
    text: str
    page_number: int | None


_PAGE_MARKER = re.compile(r"(?im)^\s*<!--\s*page(?:\s+(\d+))?\s*-->\s*$")


def extract_pdf(file_path: str | Path, engine: str = "native") -> list[ExtractedPage]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received '{path.name}'.")

    if engine == "native":
        return _extract_native(path)
    if engine == "marker":
        markdown = _run_marker(path)
    elif engine == "nougat":
        markdown = _run_nougat(path)
    else:
        raise ValueError(f"Unsupported math PDF extraction engine '{engine}'.")
    return _split_markdown_pages(markdown)


def _extract_native(path: Path) -> list[ExtractedPage]:
    try:
        with fitz.open(path) as pdf:
            if pdf.is_encrypted:
                raise ValueError(f"PDF '{path.name}' is encrypted and cannot be processed.")
            pages = [
                ExtractedPage(text=pdf[index].get_text("text").strip(), page_number=index + 1)
                for index in range(pdf.page_count)
            ]
    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(f"Could not read PDF '{path.name}': {exc}") from exc

    readable_pages = [page for page in pages if page.text]
    if not readable_pages:
        raise ValueError(
            f"PDF '{path.name}' contains no selectable text. Choose Marker or Nougat "
            "to recognize equations from page images."
        )
    return readable_pages


def _run_marker(path: Path) -> str:
    command = shutil.which("marker_single")
    if command is None:
        raise RuntimeError(
            "marker_single is not installed. Install the optional 'marker-pdf' package "
            "to use this extraction engine."
        )

    with tempfile.TemporaryDirectory(prefix="math-pdf-") as temp_dir:
        output_dir = Path(temp_dir)
        result = subprocess.run(
            [
                command,
                str(path),
                "--output_dir",
                str(output_dir),
                "--output_format",
                "markdown",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise RuntimeError(
                f"Marker could not extract '{path.name}' (exit code {result.returncode}): "
                f"{detail or 'no error details were reported'}"
            )
        markdown_files = sorted(output_dir.rglob("*.md"))
        if not markdown_files:
            raise RuntimeError(f"Marker produced no Markdown output for '{path.name}'.")
        return "\n\n".join(file.read_text(encoding="utf-8") for file in markdown_files)


def _run_nougat(path: Path) -> str:
    command = shutil.which("nougat")
    if command is None:
        raise RuntimeError(
            "nougat is not installed. Install the optional 'nougat-ocr' package "
            "to use this extraction engine."
        )

    with tempfile.TemporaryDirectory(prefix="math-pdf-") as temp_dir:
        output_dir = Path(temp_dir)
        result = subprocess.run(
            [command, str(path), "-o", str(output_dir)],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise RuntimeError(
                f"Nougat could not extract '{path.name}' (exit code {result.returncode}): "
                f"{detail or 'no error details were reported'}"
            )
        markdown_files = sorted(output_dir.rglob("*.mmd"))
        if not markdown_files:
            raise RuntimeError(f"Nougat produced no Markdown output for '{path.name}'.")
        return "\n\n".join(file.read_text(encoding="utf-8") for file in markdown_files)


def _split_markdown_pages(markdown: str) -> list[ExtractedPage]:
    markers = list(_PAGE_MARKER.finditer(markdown))
    if not markers:
        text = markdown.strip()
        return [ExtractedPage(text=text, page_number=None)] if text else []

    pages: list[ExtractedPage] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(markdown)
        text = markdown[marker.end():end].strip()
        if text:
            page_number = int(marker.group(1)) if marker.group(1) else index + 1
            pages.append(ExtractedPage(text=text, page_number=page_number))
    return pages
