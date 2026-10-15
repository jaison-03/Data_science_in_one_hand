# AI-Powered Email Spam Filter & Security Intelligence

This application lives under `nlp/core_nlp_tasks/email_spam_filter` and is linked
from the **Email Spam Filter** leaf in the NLP mind map. It combines an offline
TF-IDF classifier, deterministic link and social-engineering checks, SQLite
history, and an optional Groq explanation. Groq is not required to run local
analysis.

## Run it

From the `my_ai_project` directory:

```powershell
python -m streamlit run nlp/core_nlp_tasks/email_spam_filter/app.py
```

Alternatively, launch the existing project dashboard and click **NLP** followed
by **Email Spam Filter**.

Install the pinned project dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The first analysis trains a built-in synthetic demonstration model if no saved
model exists. The selected model and its evaluation report are stored in
`models/spam_classifier.joblib` and `models/model_metrics.json`. Set the dataset
source to custom in Settings to validate and save a labeled CSV; retrain on the
Model Performance page to use it. Training CSVs need a `spam`/`ham` label column
and a `text`, `body`, `message`, `email`, or `subject` column (subject and body
columns can also be combined).

## Optional Groq explanation

Set `GROQ_API_KEY` in the project-root `.env` file. The optional explanation
uses `GROQ_MODEL`, defaulting to `openai/gpt-oss-20b`; subject and body are sent
only when an explanation or connection test is explicitly requested. The key
is never shown in the UI. The instructions identify email content as untrusted
data and prohibit following embedded instructions, links, or tool requests.
Without the key, local ML and heuristic checks continue to work.

## Data handling

Analysis history and application settings are kept in
`database/email_spam_filter.db`. `.eml` parsing extracts text without rendering
HTML or loading remote images and enforces a 10 MiB input limit. The bulk scanner
accepts up to 10,000 CSV rows at a time. Model predictions and heuristic scores
are risk indicators, not guarantees that an email is safe.

## Tests

From the `my_ai_project` directory, run:

```powershell
python -m pytest nlp/core_nlp_tasks/email_spam_filter/tests
```

The Groq tests mock the SDK; tests never make external API calls.
