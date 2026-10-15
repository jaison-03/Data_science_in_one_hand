# NLP project collection

The apps are grouped by the target project hierarchy under this directory. Run
the master launcher from the repository root with:

```powershell
streamlit run nlp\app.py
```

Each project also has an `app.py` entry point. For example:

```powershell
streamlit run nlp\core_nlp_tasks\classification\news_categorization\app.py
```

Groq-backed apps load `GROQ_API_KEY` and `GROQ_MODEL` from the repository-root
`.env` file. The project-root configuration selects `openai/gpt-oss-20b` for
text tasks. The text normalizer, tokenizer, POS tagger, extractive summarizer,
n-gram generator, TF-IDF classifier, word-vector explorer, and information
retrieval app call Groq and also show local analysis where useful for comparison
or grounding. The retrieval app grounds its generated answers in locally ranked
passages. News categorization also uses the configured text model. Audio
transcription uses Groq's Whisper endpoint; emotion classification then uses
the configured text model and labels the result as transcript-based rather than
acoustic analysis. The word-vector
explorer uses local context-count vectors rather than a hosted or pretrained
Word2Vec embedding model, and asks Groq to interpret those local comparisons.

The multilingual question-answering system, RAG-based document chatbot, and
email spam filter remain separate applications with their existing code and
dependencies. The legacy interactive mind map in the root dashboard routes
project selections to these same Groq-backed applications rather than rendering
the former local-only teaching screens.
