# RAG Document Chatbot Architecture

This project follows a modular RAG pipeline: document ingestion, preprocessing, chunking, embedding generation, vector search, retrieval, and LLM answer generation.

## Flow

1. Upload PDF, TXT, or Markdown files.
2. Validate extensions and file size.
3. Extract and clean text while preserving readability.
4. Split large documents into text chunks with overlap.
5. Convert chunks into embeddings with Sentence Transformers.
6. Store vectors in ChromaDB or FAISS.
7. Embed user questions, retrieve the most relevant chunks, and apply a relevance threshold if configured.
8. Build a grounded prompt and send it to Groq with the selected model.
9. Return the answer with citations to the retrieved passages.

## Math and equation PDF pipeline

The chatbot's **Math & Equations** option is a separate ingestion path that shares the
existing embeddings, vector-store implementation, and grounded question-answering pattern:

1. Native mode extracts selectable PDF text page by page with PyMuPDF.
2. Optional Marker (`marker-pdf`) or Nougat (`nougat-ocr`) converts formula-heavy papers to
   Markdown with LaTeX. These model-heavy packages are installed separately from the main
   requirements. The tested Marker setup pins `marker-pdf==1.10.2` and Transformers below
   version 5 to preserve compatibility with the chatbot's Sentence Transformers dependency.
3. Extracted passages are chunked and tagged with their extraction engine and source page
   when the converter provides page markers. Display equations are kept intact while chunking.
4. The dedicated math collection is embedded, searched, and used to answer questions about
   equations, symbols, and derivations.

Native mode cannot recognize formulas that exist only as page images. Select Marker or Nougat
for those documents; failures from missing optional tools or unsuccessful conversions are
shown explicitly in the interface.

## Key modules

- `config/settings.py`: environment-driven configuration.
- `core/document_loader.py`: imports and validates source files.
- `core/text_cleaner.py`: normalizes text while preserving useful punctuation and structure.
- `core/chunking.py`: recursively splits text into manageable chunks.
- `core/embeddings.py`: loads the embedding model and creates vectors.
- `core/vector_store.py`: stores and queries vectors for both Chroma and FAISS backends.
- `core/retriever.py`: wraps embedding and search logic.
- `core/prompts.py`: controls the grounded answer template.
- `core/llm_service.py`: sends prompts to Groq.
- `core/rag_pipeline.py`: orchestrates retrieval and generation.

## Why RAG helps

RAG reduces hallucinations by grounding answers in retrieved source text instead of requiring the model to rely only on training memory. It also allows users to inspect the evidence behind each answer.
