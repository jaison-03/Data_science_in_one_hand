# Image / Screenshot PDF RAG Pipeline

This package is a complete OCR chatbot for scanned pages and PDFs containing
large embedded screenshots or images. From the main app, click **RAG-Based
Document Chatbot**, then select **Scanned / Image-based PDFs**. It lives alongside the chatbot's
existing `core` modules and reuses its Sentence Transformers embeddings,
ChromaDB/FAISS vector store, retriever, and dialogue LLM interface.

## Install

From the `rag_based_document_chatbot` directory, install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the **Tesseract OCR application** separately; `pytesseract` is only
its Python wrapper. On Windows, from an Administrator PowerShell, install it
with Chocolatey:

```powershell
choco install tesseract -y
```

Restart Streamlit after installation. The app detects the usual Windows
installation locations automatically. If Tesseract is installed elsewhere,
configure the executable before creating the pipeline:

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

PyMuPDF renders PDF pages directly, so Poppler/pdf2image is not required.

## Index and ask

```python
from image_pdf_rag import ImagePdfRAGPipeline

pipeline = ImagePdfRAGPipeline()
indexed = pipeline.index_pdf("uploads/scanned-report.pdf", routing="auto")
print(indexed["chunk_count"], "chunks;", indexed["ocr_page_count"], "OCR pages")

result = pipeline.answer_question("What are the key findings?")
print(result["answer"])
for source in result["sources"]:
    print(source["filename"], source["page_number"], source.get("page_image"))
```

The first call downloads/loads the configured embedding model. Answer
generation uses the chatbot's configured `GROQ_API_KEY`; indexing does not
require an LLM key. To run this interface by itself from the
`rag_based_document_chatbot` directory:

```powershell
streamlit run image_pdf_rag\app.py
```

`routing="auto"` selects OCR per page when native text is sparse or embedded
images cover at least 20% of the page. Use `routing="ocr"` to force OCR on
every page, or `routing="native"` for digitally generated PDFs. The interface
offers all three modes. The threshold, render DPI, OCR engine, embeddings,
vector store, and LLM can be configured or injected through
`ImagePdfRAGPipeline` for integration and tests.

If the UI is configured for ChromaDB but the `chromadb` package is not
installed, it automatically uses FAISS for the current Streamlit session and
shows a notice. Install the listed dependencies to enable persistent ChromaDB
storage.

Chunks retain `source_file`, `page_number`, `source_page_image`,
`extraction_method`, and available OCR confidence metadata. Answer citations
include the source filename and page number; each source also includes a
`page_image` identifier such as `scanned-report.pdf#page=3`.

## Tests

From the `rag_based_document_chatbot` directory:

```powershell
python -m pytest image_pdf_rag\tests
```

Tests use injected OCR and fake model services; running them does not require
a Tesseract executable, downloaded embedding model, vector database, or API key.
