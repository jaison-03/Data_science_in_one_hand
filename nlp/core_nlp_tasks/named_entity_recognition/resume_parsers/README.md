# Resume Parser and Candidate Fit

The Streamlit app parses PDF, DOCX, and TXT resumes, returns structured candidate
information as JSON, and explains the evidence for a target-job fit assessment.
It also suggests other roles based on experience and skills in the resume.

## Run

From the repository root, set `GROQ_API_KEY` in the root `.env` file and launch:

```powershell
streamlit run nlp\core_nlp_tasks\named_entity_recognition\resume_parsers\app.py
```

The app sends resume text and any supplied job description to the configured
Groq API. It does not save uploaded files. For scanned PDFs, install the
Tesseract OCR executable; the Python `pytesseract` package is already listed in
the project requirements.

Assessments are job-related decision support, not automated hiring decisions.
Review the resume and verify important claims before making decisions.
