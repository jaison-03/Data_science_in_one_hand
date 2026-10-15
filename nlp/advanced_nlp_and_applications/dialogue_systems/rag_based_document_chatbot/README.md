# Intelligent Document Intelligence — RAG Chatbot

This project is a tutorial-friendly but production-style Retrieval-Augmented Generation (RAG) application for document-based question answering. It is designed for learners exploring NLP, embeddings, vector databases, and LLM-powered document search.

## NLP roadmap placement

- NLP
  - Advanced NLP & Applications
    - Dialogue Systems
      - RAG-Based Document Chatbot

## Project goals

- Upload PDF, TXT, and Markdown documents.
- Clean and normalize extracted text.
- Chunk documents into manageable sections while preserving metadata.
- Generate embeddings with Sentence Transformers.
- Store vectors in ChromaDB or FAISS.
- Retrieve the most relevant passages for a user question.
- Use Groq-hosted LLMs to answer grounded questions.
- Show source evidence for each answer and explain when context is insufficient.

## Technology stack

- Python 3.11
- Streamlit for the user interface
- LangChain for orchestration and prompts
- Groq for LLM generation
- Sentence Transformers for embeddings
- ChromaDB and FAISS for vector search, with an exact NumPy fallback where
  those native vector-store packages are unavailable
- PyMuPDF for PDF extraction
- python-dotenv for environment configuration
- Pydantic for settings validation
- pytest for automated tests

## Project structure

```text
rag_based_document_chatbot/
├── app.py
├── math_equation_rag/
│   ├── app.py
│   ├── extractor.py
│   ├── pipeline.py
│   └── tests/
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_cleaner.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── llm_service.py
│   ├── rag_pipeline.py
│   └── evaluation.py
├── utils/
│   ├── __init__.py
│   ├── logging_config.py
│   ├── validators.py
│   └── helpers.py
├── database/
│   └── .gitkeep
├── uploads/
│   └── .gitkeep
├── tests/
│   ├── test_chunking.py
│   ├── test_document_loader.py
│   ├── test_retriever.py
│   └── test_validators.py
├── docs/
│   └── architecture.md
└── .env
```

## RAG workflow

1. The user uploads document files.
2. The system validates extension and size.
3. Text extraction cleans PDF, TXT, or Markdown content.
4. The text is split into chunks with overlap.
5. Each chunk is embedded using a local Sentence Transformer.
6. The embedding vectors are stored in the selected vector backend.
7. The question is embedded and compared with stored chunks.
8. The strongest matches are passed to the LLM with a grounded prompt.
9. The final answer includes source references and exposes any missing-context cases.

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`, then replace `your_groq_api_key_here` with your Groq API key. Keep `.env` private and do not commit it.
4. Start the app:

```bash
streamlit run app.py
```

## Configuration

Example environment variables:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE=chroma
CHROMA_PERSIST_DIRECTORY=database/chroma_db
FAISS_INDEX_DIRECTORY=database/faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4
SIMILARITY_THRESHOLD=0.25
```

After starting the app, upload your PDF, TXT, or Markdown files in **Document management**, choose **Index documents**, and then ask questions in the chat panel. Answers use retrieved document passages and include source references. The embedding model runs locally; Groq is used only to generate the answer.

On Windows, the root requirements skip ChromaDB and FAISS because their native
index packages are not available for every Python/Windows combination. The RAG
app remains usable: it transparently uses exact NumPy inner-product search
instead of failing at import time. On supported platforms, install the optional
vector-store dependencies from the root `requirements.txt` to use ChromaDB and
FAISS.

## Notes

- The app does not require Groq to upload or process documents.
- Groq is used only when generating answers.
- Uploaded document content is treated as untrusted data and is not executed.
- For scanned PDFs, OCR is required before textual retrieval is possible.
- The optional OCR-backed scanned/image PDF pipeline is documented in [`image_pdf_rag/README.md`](./image_pdf_rag/README.md). It uses Tesseract OCR for scanned pages while reusing this project's embeddings, vector stores, retriever, and LLM service.
- In the NLP mind map, clicking **RAG-Based Document Chatbot** opens **Standard Digital PDFs**, **Scanned / Image-based PDFs**, and **Math & Equations**.
- The math/equations pipeline uses PyMuPDF for selectable PDF text, and optionally Marker or Nougat to convert equation-heavy PDFs into Markdown with LaTeX. To install the tested Marker release without upgrading the chatbot's Transformers stack, use `python -m pip install "marker-pdf==1.10.2" "transformers>=4.41,<5" "packaging>=23.2,<26"`. These large ML dependencies are intentionally excluded from the default requirements. Nougat's current PyPI dependencies include legacy PyArrow 3, which cannot be built in this project's Python 3.13 environment; install Nougat in a compatible separate environment if required.
- In **Math & Equations**, choose **Marker** for equation-heavy PDFs or image-based formulas, **Nougat** for academic papers, or **Native PDF text** for selectable text. Native extraction reports when a PDF has no selectable text rather than silently indexing an empty document. Converter outputs without page markers are cited with the page unavailable.

## Learning focus

This project is intentionally readable and modular so learners can trace how modern RAG systems work in practice without hiding the pipeline behind a black box.
