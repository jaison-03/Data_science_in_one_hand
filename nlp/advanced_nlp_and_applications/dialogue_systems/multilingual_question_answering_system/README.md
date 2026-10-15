# Multilingual Question Answering System

A local Streamlit application for multilingual Q&A, context-only answering,
translation, summarization, and persistent conversation history. Groq calls use
the official Python SDK. User questions are sent to Groq when an LLM feature is
used; generated content may be incorrect and is not independently verified.

## Run on Windows

From this project directory, install the local app requirements and launch the
application:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

To create and activate a virtual environment from this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`. Stop the
server with `Ctrl+C`.

## Configure Groq securely

Create or edit `.env` in this project folder:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
DATABASE_PATH=database/multilingual_qa.db
```

Replace the key placeholder locally with a key from
[Groq Console](https://console.groq.com/keys). Do not paste the key into Python
or Streamlit, and do not commit `.env`. The app only displays whether a key is
configured; it does not show or log any part of the secret. The model is read
from `GROQ_MODEL`; `openai/gpt-oss-20b` is used when the variable is unset.
Availability for a particular account is determined by Groq when a request is
made. If a model is unavailable, Settings shows a generic model/request error.

## Features

- Automatic language detection and selectable question/answer languages.
- General-knowledge and context-only question answering.
- Bounded conversation context and SQLite-backed history with feedback.
- Groq-powered translation and short, detailed, or bullet summaries.
- Configurable answer detail, temperature, and output-token limit.
- Safe user-facing handling for missing credentials, authentication failures,
  rate limits, timeouts, connection failures, and invalid model requests.

The database is created automatically. Use **Settings → Clear all conversation
history** to delete stored conversations. To remove local records manually,
stop Streamlit and delete `database\multilingual_qa.db`.

## Tests

From the `my_ai_project` root, run:

```powershell
python -m pytest nlp\advanced_nlp_and_applications\dialogue_systems\multilingual_question_answering_system\tests nlp\core_nlp_tasks\medical_ner\tests nlp\core_nlp_tasks\email_spam_filter\tests
```

Tests mock the Groq SDK and do not make external API calls.
