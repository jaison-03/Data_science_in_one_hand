"""Shared Streamlit interfaces and Groq-backed NLP tasks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import APIError, Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
MAX_TEXT_LENGTH = 50_000
AUDIO_MODEL = "whisper-large-v3-turbo"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

PROFILES: dict[str, dict[str, str]] = {
    "news_categorization": {
        "title": "News Categorization",
        "description": "Classify a news story into one of eight standard categories.",
        "prompt": "You are an expert news classifier. Categorize the sentence strictly as exactly one of: Politics, Technology, Business & Finance, Sports, Entertainment, Health & Science, World News, or Crime & Law. Output only the category name with no explanation or punctuation. If multiple categories apply, choose the closest primary category.",
        "label": "News article",
    },
    "resume_parser": {
        "title": "Resume Parser",
        "description": "Parse PDF, DOCX, or TXT resumes and explain candidate fit for a target role.",
        "prompt": "Parse a resume and assess evidence-based job fit.",
        "label": "Resume text",
    },
    "contract_extractor": {
        "title": "Contract Entity Extractor",
        "description": "Identify parties, dates, obligations, payment terms, and risks in contract text.",
        "prompt": "Extract entities and key terms from the contract. Return valid JSON with parties, effective_date, end_date, obligations, payment_terms, termination_terms, governing_law, and potential_risks. Quote or briefly reference supporting text and mark absent information as unknown. This is not legal advice.",
        "label": "Contract text",
    },
    "medical_ner": {
        "title": "Medical Named Entity Recognition",
        "description": "Extract clinical entities from supplied text. This is an information-extraction demo, not medical advice.",
        "prompt": "Extract only explicitly mentioned clinical entities. Return valid JSON with entities as an array of objects containing text, type, and context. Use types such as CONDITION, MEDICATION, DOSAGE, PROCEDURE, TEST, ANATOMY, and TEMPORAL. Preserve negation and uncertainty; do not diagnose or infer.",
        "label": "Clinical text",
    },
    "abstractive_summarizer": {
        "title": "Abstractive Text Summarizer",
        "description": "Generate a concise, readable summary while retaining the source's key points.",
        "prompt": "Write a faithful abstractive summary of the text. Preserve important qualifications and do not add facts absent from the source.",
        "label": "Text to summarize",
    },
    "faq_chatbot": {
        "title": "FAQ Chatbot",
        "description": "Answer a question using only the FAQ or reference material provided.",
        "prompt": "Answer the user's question using only the supplied reference material. If the answer is not supported by the material, say that the reference does not contain the answer.",
        "label": "FAQ or reference material",
    },
    "domain_chatbot": {
        "title": "Domain-Specific Chatbot",
        "description": "Ask questions of a domain-focused assistant with an explicit scope and guardrails.",
        "prompt": "Answer helpfully within the user's specified domain. Be clear about uncertainty, do not claim credentials, and distinguish general information from professional advice when appropriate.",
        "label": "Question",
    },
    "english_to_spanish_translator": {
        "title": "English to Spanish Translator",
        "description": "Translate English text into natural Spanish while retaining meaning and tone.",
        "prompt": "Translate the following English text into natural Spanish. Preserve meaning, names, formatting, and tone. Return only the translation.",
        "label": "English text",
    },
    "grammar_corrector": {
        "title": "Grammar Corrector",
        "description": "Correct grammar, spelling, and punctuation without changing the author's intent.",
        "prompt": "Correct grammar, spelling, and punctuation while preserving the author's meaning, voice, and formatting. Return the corrected text, then a short list of notable edits.",
        "label": "Text to correct",
    },
    "keyword_extractor": {
        "title": "Keyword Extractor",
        "description": "Extract the most informative concepts and phrases from text.",
        "prompt": "Extract up to 15 informative keywords and keyphrases from the text. Prefer specific multi-word concepts, remove duplicates, and return a concise numbered list with no unsupported terms.",
        "label": "Text",
    },
    "code_generator": {
        "title": "Simple Python Code Generator",
        "description": "Generate a small, readable Python solution for a described task.",
        "prompt": "Write a simple, readable Python implementation for the request. State assumptions briefly, include a small usage example, and avoid unsafe operations. Return code in a fenced Python block.",
        "label": "Describe the Python task",
    },
    "blog_post_idea_generator": {
        "title": "Blog Post Idea Generator",
        "description": "Create focused, audience-aware blog post concepts.",
        "prompt": "Generate 8 distinct blog post ideas for the topic and audience. For each give a compelling title, a one-sentence angle, and the intended reader. Avoid clickbait and unsupported claims.",
        "label": "Topic and target audience",
    },
    "poetry_writing_assistant": {
        "title": "Poetry Writing Assistant",
        "description": "Create an original poem from a theme, style, and optional constraints.",
        "prompt": "Write an original poem from the user's theme and constraints. Use fresh imagery, respect requested form and tone, and do not imitate a named living writer.",
        "label": "Theme and optional constraints",
    },
    "sentiment_with_word_vectors": {
        "title": "Sentiment with Word Vectors",
        "description": "Classify sentiment with Groq and inspect a local context-vector representation of the text.",
        "prompt": "Classify the sentiment as positive, negative, mixed, or neutral. Give a brief evidence-based explanation and identify any uncertainty. Do not overstate sentiment from sarcasm or missing context.",
        "label": "Text to analyze",
    },
    "context_aware_chatbot": {
        "title": "Context-Aware Chatbot",
        "description": "Have a conversation with an assistant that remembers recent turns in this browser session.",
        "prompt": "You are a helpful assistant. Use the conversation history to resolve references and maintain context. Be transparent about uncertainty.",
        "label": "Conversation",
    },
}

LOCAL_PROFILES: dict[str, tuple[str, str]] = {
    "extractive_text_summarizer": ("Extractive Text Summarizer", "Ask Groq to select source sentences and compare with a local frequency-based extract."),
    "text_normalizer": ("Text Normalizer", "Use Groq to normalize text, with deterministic local normalization for comparison."),
    "custom_tokenizer": ("Custom Tokenizer", "Ask Groq to explain token boundaries and compare them with a local tokenizer."),
    "pos_tagger": ("POS Tagger", "Get contextual part-of-speech tags from Groq and compare them with local heuristics."),
    "tf_idf_classifier": ("TF-IDF Classifier", "Compare local TF-IDF predictions with a Groq classification grounded in your labeled examples."),
    "word2vec_explorer": ("Word Vector Explorer", "Explore local context vectors and ask Groq to interpret the nearest contextual terms."),
    "n_gram_generator": ("N-gram Generator", "Ask Groq to interpret n-gram patterns and compare them with local frequency counts."),
    "information_retrieval_engine": ("Information Retrieval Engine", "Retrieve documents locally, then ask Groq to answer using the ranked source passages."),
}


def _groq_client() -> tuple[Groq, str]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in the project-root .env file.")
    return Groq(api_key=api_key), model


def _complete(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> str:
    if len(user_prompt) > MAX_TEXT_LENGTH:
        raise ValueError(f"Input is too long; limit it to {MAX_TEXT_LENGTH:,} characters.")
    client, model = _groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content


def _run_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> None:
    try:
        with st.spinner("Processing with Groq..."):
            result = _complete(system_prompt, user_prompt, temperature)
        st.subheader("Result")
        st.markdown(result)
    except (APIError, RuntimeError, ValueError) as error:
        st.error(str(error))


def _render_groq(profile_id: str) -> None:
    profile = PROFILES[profile_id]
    st.title(profile["title"])
    st.caption(profile["description"])
    if profile_id == "resume_parser":
        _render_resume_parser()
        return
    if profile_id == "context_aware_chatbot":
        _render_context_chat(profile["prompt"])
        return
    temperature = 0.2

    if profile_id in {"faq_chatbot", "domain_chatbot"}:
        if profile_id == "faq_chatbot":
            context = st.text_area(profile["label"], height=220, max_chars=MAX_TEXT_LENGTH)
            question = st.text_input("Question")
            if st.button("Get answer", type="primary"):
                if not context.strip() or not question.strip():
                    st.warning("Provide both reference material and a question.")
                else:
                    _run_groq(profile["prompt"], f"Reference material:\n{context}\n\nQuestion:\n{question}")
        else:
            domain = st.text_input("Domain or scope", placeholder="For example: introductory astronomy")
            question = st.text_area(profile["label"], height=150)
            if st.button("Ask", type="primary"):
                if not domain.strip() or not question.strip():
                    st.warning("Provide a domain and a question.")
                else:
                    _run_groq(profile["prompt"], f"Domain: {domain}\n\nQuestion: {question}")
        return

    if profile_id == "sentiment_with_word_vectors":
        text = st.text_area(profile["label"], height=180, max_chars=MAX_TEXT_LENGTH)
        if st.button("Analyze sentiment", type="primary"):
            if not text.strip():
                st.warning("Enter text to analyze.")
                return
            _run_groq(profile["prompt"], text)
            vectors = _context_vectors(text)
            if vectors:
                st.subheader("Local context-vector terms")
                st.dataframe(
                    [{"term": word, "context features": len(vector)} for word, vector in vectors.items()],
                    use_container_width=True,
                    hide_index=True,
                )
        return

    text = st.text_area(profile["label"], height=240, max_chars=MAX_TEXT_LENGTH)
    if profile_id == "poetry_writing_assistant":
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
    if st.button("Generate", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            _run_groq(profile["prompt"], text, temperature)


def _render_resume_parser() -> None:
    from nlp.core_nlp_tasks.named_entity_recognition.resume_parsers.parser import (
        MAX_RESUME_CHARS,
        analyze_resume,
        extract_resume_text,
    )

    st.caption(
        "Upload a resume to extract its details, assess fit for an optional job description, "
        "and discover other evidence-aligned roles."
    )
    st.info(
        "This is decision-support, not an automated hiring decision. Assessments use job-related "
        "evidence only; review the source resume and verify important claims. Resume text is sent "
        "to the configured Groq API for analysis."
    )
    uploaded = st.file_uploader("Upload a resume", type=["pdf", "docx", "txt"])
    resume_text = st.text_area(
        "Or paste resume text",
        height=220,
        max_chars=MAX_RESUME_CHARS,
        help="If a file is uploaded, it will be used instead of pasted text.",
    )
    target_role = st.text_input(
        "Target role (optional)",
        max_chars=200,
        placeholder="For example: Senior Data Analyst",
    )
    job_description = st.text_area(
        "Job description (optional)",
        height=160,
        max_chars=8_000,
        help="Add the job description for a more specific fit assessment.",
    )
    if uploaded is not None:
        upload_bytes = uploaded.getvalue()
        source_identity = uploaded.name.encode("utf-8") + b"\0" + upload_bytes
    else:
        upload_bytes = None
        source_identity = resume_text.strip().encode("utf-8")
    input_signature = hashlib.sha256(
        source_identity
        + b"\0"
        + target_role.strip().encode("utf-8")
        + b"\0"
        + job_description.strip().encode("utf-8")
    ).hexdigest()

    if st.button("Analyze resume", type="primary"):
        st.session_state.pop("resume_parser_result", None)
        if uploaded is None and not resume_text.strip():
            st.warning("Upload a resume or paste resume text first.")
            return
        try:
            with st.spinner("Extracting resume and evaluating job fit..."):
                if uploaded is not None and upload_bytes is not None:
                    source_text = extract_resume_text(uploaded.name, upload_bytes)
                else:
                    source_text = resume_text.strip()
                result = analyze_resume(source_text, target_role, job_description, _groq_client)
            st.session_state["resume_parser_result"] = {
                "analysis": result,
                "target_role": target_role.strip(),
                "input_signature": input_signature,
            }
        except (APIError, RuntimeError, ValueError) as error:
            st.error(str(error))
            return

    saved_result = st.session_state.get("resume_parser_result")
    if not saved_result:
        return
    if saved_result["input_signature"] != input_signature:
        st.info("Resume or job details changed. Analyze again to update this result.")
        return
    result = saved_result["analysis"]

    parsed_data = result["parsed_data"]
    fit = result["job_fit_analysis"]
    quality = result["resume_quality_assessment"]
    name = parsed_data["contact_info"]["full_name"]
    st.subheader(f"Candidate analysis{f': {name}' if name else ''}")
    st.metric("Fit assessment", fit["qualification_status"])
    if saved_result["target_role"]:
        st.caption(f"Assessment for: {saved_result['target_role']}")
    st.write(fit["suitability_reasoning"])

    st.markdown("**Other suitable roles**")
    for role in fit["suitable_job_roles"]:
        st.markdown(f"- {role}")

    with st.expander("Resume quality and career profile", expanded=True):
        st.write(f"**Career level:** {quality['career_level']}")
        st.write(f"**Presentation rating:** {quality['presentation_rating_1_to_10']}/10")
        st.markdown("**Strengths**")
        for strength in quality["key_strengths"]:
            st.markdown(f"- {strength}")
        st.markdown("**Gaps to verify or address**")
        for gap in quality["weaknesses_or_gaps"]:
            st.markdown(f"- {gap}")

    with st.expander("Parsed resume and evaluation JSON"):
        st.json(result)
    st.download_button(
        "Download analysis as JSON",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="resume_analysis.json",
        mime="application/json",
    )


def _render_context_chat(system_prompt: str) -> None:
    history_key = "nlp_context_chat_history"
    history = st.session_state.setdefault(history_key, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Send a message")
    if not prompt:
        return
    if len(prompt) > MAX_TEXT_LENGTH:
        st.error(f"Message is too long; limit it to {MAX_TEXT_LENGTH:,} characters.")
        return
    history.append({"role": "user", "content": prompt})
    try:
        client, model = _groq_client()
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, *history[-20:]],
                temperature=0.4,
            )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("Groq returned an empty response.")
        history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    except (APIError, RuntimeError, ValueError) as error:
        history.pop()
        st.error(str(error))


def _context_vectors(text: str, window: int = 2) -> dict[str, dict[str, float]]:
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, token in enumerate(tokens):
        for offset in range(max(0, index - window), min(len(tokens), index + window + 1)):
            if offset != index:
                counts[token][tokens[offset]] += 1
    return {word: dict(vector) for word, vector in counts.items()}


def _render_audio(profile_id: str) -> None:
    title, description = (
        ("Speech-to-Text Transcriber", "Transcribe an audio clip with Groq's Whisper transcription endpoint."),
        ("Emotion Detection from Voice", "Transcribe a voice clip, then classify emotion from the transcript with GROQ_MODEL."),
    )[profile_id == "emotion_detection_from_voice"]
    st.title(title)
    st.caption(description)
    uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "mpga", "mpeg", "webm", "ogg", "flac"])
    if st.button("Transcribe" if profile_id == "speech_to_text_transcriber" else "Transcribe and analyze", type="primary"):
        if uploaded is None:
            st.warning("Upload an audio file first.")
            return
        if uploaded.size > 25 * 1024 * 1024:
            st.error("Audio file exceeds the 25 MB transcription limit.")
            return
        try:
            client, _ = _groq_client()
            with st.spinner("Transcribing audio..."):
                response = client.audio.transcriptions.create(
                    file=(uploaded.name, uploaded.getvalue()),
                    model=AUDIO_MODEL,
                    response_format="json",
                )
            transcript = response.text.strip()
            if not transcript:
                st.error("The transcription endpoint returned no text.")
                return
            st.subheader("Transcript")
            st.write(transcript)
            st.audio(uploaded)
            if profile_id == "emotion_detection_from_voice":
                _run_groq(
                    "Classify the emotional cues conveyed by the words in this transcript. Return a likely emotion label, concise evidence, and a confidence level (low, medium, or high). State that vocal tone itself was not analyzed.",
                    transcript,
                )
        except (APIError, RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_local(profile_id: str) -> None:
    title, description = LOCAL_PROFILES[profile_id]
    st.title(title)
    st.caption(description)

    if profile_id == "tf_idf_classifier":
        examples = st.text_area(
            "Labeled training examples (one per line: label<TAB>text)",
            value="sports\tThe team won the championship match.\nsports\tThe striker scored a late goal.\ntech\tThe new processor improves computer performance.\ntech\tDevelopers released a faster programming tool.\nfinance\tThe company reported quarterly revenue growth.\nfinance\tInvestors reviewed the latest market results.",
            height=180,
            max_chars=MAX_TEXT_LENGTH,
        )
        query = st.text_area("Text to classify", max_chars=MAX_TEXT_LENGTH)
        if st.button("Classify with Groq and TF-IDF", type="primary"):
            try:
                pairs = []
                for line_number, line in enumerate(examples.splitlines(), start=1):
                    if not line.strip():
                        continue
                    label, separator, sample = line.partition("\t")
                    if not separator or not label.strip() or not sample.strip():
                        raise ValueError(f"Training line {line_number} must contain a label, a tab, and text.")
                    pairs.append((label.strip(), sample.strip()))
                if not pairs:
                    raise ValueError("Add labeled training examples before fitting the classifier.")
                labels, texts = zip(*pairs)
                if len(set(labels)) < 2:
                    raise ValueError("Provide training examples from at least two labels.")
                if any(Counter(labels)[label] < 2 for label in set(labels)):
                    raise ValueError("Provide at least two training examples for each label.")
                if not query.strip():
                    raise ValueError("Enter text to classify.")
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
                matrix = vectorizer.fit_transform(texts)
                classifier = LogisticRegression(max_iter=1000, random_state=0).fit(matrix, labels)
                prediction = classifier.predict(vectorizer.transform([query]))[0]
                st.subheader("Local TF-IDF prediction")
                st.success(f"Predicted category: {prediction}")
                st.caption(
                    "The local model is trained on the examples above; its quality depends on "
                    "the representativeness and size of those examples."
                )
                examples_for_model = "\n".join(
                    f"- {label}: {sample}" for label, sample in pairs
                )
                _run_groq(
                    "Classify the text into exactly one of the supplied labels. Use the labeled examples as guidance, not unquestionable truth. Return the predicted label, a brief evidence-based explanation, and a confidence level. Do not invent a label that is not in the list.",
                    f"Examples:\n{examples_for_model}\n\nAllowed labels: {', '.join(sorted(set(labels)))}\n\nText to classify:\n{query}",
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                st.error(str(error))
        return

    if profile_id == "information_retrieval_engine":
        documents = st.text_area(
            "Documents (separate documents with a blank line)",
            height=240,
            max_chars=MAX_TEXT_LENGTH - 1_000,
            help="A title on the first line is used as the document label.",
        )
        query = st.text_input("Search query")
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)
        if st.button("Search", type="primary"):
            blocks = [block.strip() for block in re.split(r"\n\s*\n", documents) if block.strip()]
            if not blocks or not query.strip():
                st.warning("Provide at least one document and a search query.")
                return
            if len(blocks) == 1:
                st.warning("Add a blank line between documents to search a collection.")
                return
            labels = [block.splitlines()[0][:100] for block in blocks]
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                matrix = vectorizer.fit_transform(blocks)
                scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
                ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
                if not ranked or ranked[0][1] <= 0:
                    st.info("No documents matched the query terms.")
                for index, score in ranked:
                    st.markdown(f"**{labels[index]}** — relevance {score:.3f}")
                    st.write(blocks[index])
                retrieved = "\n\n".join(
                    f"[{rank + 1}] {labels[index]}\n{blocks[index]}"
                    for rank, (index, _) in enumerate(ranked)
                )
                _run_groq(
                    "Answer the question using only the provided retrieved documents. Cite supporting documents by their bracketed number. If the documents do not support an answer, say so explicitly. Never fill gaps with outside knowledge.",
                    f"Question: {query}\n\nRetrieved documents:\n{retrieved}",
                )
            except ValueError as error:
                st.error(f"Could not index documents: {error}")
        return

    text = st.text_area("Input text", height=220, max_chars=MAX_TEXT_LENGTH)
    if profile_id == "text_normalizer":
        lowercase = st.checkbox("Convert to lowercase", value=True)
        remove_punctuation = st.checkbox("Remove punctuation", value=False)
        strip_diacritics = st.checkbox("Remove diacritics", value=False)
        normalizer_instructions = (
            f"Normalize the input text. lowercase={lowercase}; "
            f"remove_punctuation={remove_punctuation}; strip_diacritics={strip_diacritics}. "
            "Also normalize repeated whitespace. Return only the normalized text and do not alter its meaning."
        )
    elif profile_id == "custom_tokenizer":
        include_punctuation = st.checkbox("Include punctuation tokens", value=True)
        tokenizer_instructions = (
            "Tokenize the text into an ordered list, preserving contractions as one token and "
            f"{'including' if include_punctuation else 'omitting'} standalone punctuation tokens. "
            "Return a numbered token list, then briefly explain ambiguous token boundaries."
        )
    elif profile_id == "extractive_text_summarizer":
        sentence_count = st.slider("Summary sentences", 1, 10, 3)
    elif profile_id == "n_gram_generator":
        n = st.slider("N-gram size", 1, 6, 2)
    if profile_id == "word2vec_explorer":
        if not text.strip():
            st.info("Enter text containing repeated contextual relationships to explore word vectors.")
            return
        vectors = _context_vectors(text)
        if not vectors:
            st.info("Add at least three words to build context vectors.")
            return
        vocabulary = sorted(vectors)
        feature_names = sorted({feature for vector in vectors.values() for feature in vector})
        if len(vocabulary) < 2 or not feature_names:
            st.info("Add at least two different words with neighboring context to explore vectors.")
            return
        matrix = np.array([[vectors[word].get(feature, 0.0) for feature in feature_names] for word in vocabulary])
        query_word = st.selectbox("Word", vocabulary)
        similarities = cosine_similarity(matrix, matrix[vocabulary.index(query_word)].reshape(1, -1)).ravel()
        nearest = sorted(
            ((word, float(score)) for word, score in zip(vocabulary, similarities) if word != query_word),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        st.dataframe(
            [{"word": word, "cosine similarity": round(score, 3)} for word, score in nearest],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Local context-count vectors provide the neighbors; Groq can explain their likely relationship.")
        if st.button("Explain context with Groq", type="primary"):
            neighbor_text = ", ".join(f"{word} ({score:.3f})" for word, score in nearest)
            _run_groq(
                "Explain the likely contextual relationship between a selected word and its nearest neighbors based on the supplied text and count-vector similarities. Treat the values as co-occurrence evidence, not semantic embeddings. Do not claim that these are pretrained Word2Vec vectors.",
                f"Selected word: {query_word}\nNearest local context-vector neighbors: {neighbor_text}\n\nSource text:\n{text}",
            )
        return
    if st.button("Process text", type="primary"):
        if not text.strip():
            st.warning("Enter text first.")
            return
        if profile_id == "text_normalizer":
            result = " ".join(text.split())
            if strip_diacritics:
                import unicodedata

                result = "".join(
                    char for char in unicodedata.normalize("NFKD", result) if not unicodedata.combining(char)
                )
            if lowercase:
                result = result.lower()
            if remove_punctuation:
                result = re.sub(r"[^\w\s]", "", result)
            st.subheader("Local normalized text")
            st.code(result)
            _run_groq(normalizer_instructions, text)
        elif profile_id == "custom_tokenizer":
            tokens = re.findall(r"\b[\w]+(?:['’][\w]+)*\b|[^\w\s]", text)
            if not include_punctuation:
                tokens = [token for token in tokens if re.search(r"\w", token)]
            st.subheader(f"Local tokens ({len(tokens)})")
            st.code(tokens)
            _run_groq(tokenizer_instructions, text)
        elif profile_id == "pos_tagger":
            tagged = _heuristic_pos_tags(text)
            st.subheader("Local heuristic tags")
            st.dataframe([{"token": token, "tag": tag} for token, tag in tagged], use_container_width=True, hide_index=True)
            _run_groq(
                "Perform contextual English part-of-speech tagging. Return a markdown table with the exact token, a standard Penn Treebank POS tag, and a short contextual rationale. Keep token order and punctuation. Explain ambiguous tags briefly.",
                text,
            )
        elif profile_id == "extractive_text_summarizer":
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
            if len(sentences) <= sentence_count:
                local_summary = " ".join(sentences)
            else:
                sentence_tokens = [re.findall(r"\b[\w'-]+\b", sentence.lower()) for sentence in sentences]
                frequencies = Counter(token for row in sentence_tokens for token in row if len(token) > 2)
                scores = [
                    sum(frequencies[token] / max(1, len(row)) for token in row) / math.sqrt(max(1, len(row)))
                    for row in sentence_tokens
                ]
                selected = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:sentence_count])
                local_summary = " ".join(sentences[index] for index in selected)
            st.subheader("Local frequency-based extract")
            st.write(local_summary)
            _run_groq(
                f"Create an extractive summary of the text using at most {sentence_count} complete source sentences. Copy chosen sentences verbatim and preserve their original order. Do not paraphrase, combine, or add information. Return the selected sentences only.",
                text,
            )
        elif profile_id == "n_gram_generator":
            tokens = re.findall(r"\b[\w'-]+\b", text.lower())
            ngrams = [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]
            frequencies = Counter(ngrams)
            st.subheader(f"Local {n}-gram counts")
            st.dataframe(
                [
                    {"n-gram": " ".join(ngram), "count": count}
                    for ngram, count in frequencies.most_common(100)
                ],
                use_container_width=True,
                hide_index=True,
            )
            _run_groq(
                f"Analyze the text's {n}-gram patterns. Identify recurring phrases, notable collocations, and what they suggest about the text. Distinguish observations from interpretation and do not invent n-grams absent from the text.",
                text,
            )


def _heuristic_pos_tags(text: str) -> list[tuple[str, str]]:
    tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", text)
    determiners = {"a", "an", "the", "this", "that", "these", "those"}
    pronouns = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
    conjunctions = {"and", "or", "but", "so", "because"}
    prepositions = {"in", "on", "at", "by", "for", "from", "with", "to", "of", "under", "over"}
    tags = []
    for token in tokens:
        word = token.lower()
        if not re.search(r"\w", token):
            tag = "PUNCT"
        elif word in determiners:
            tag = "DET"
        elif word in pronouns:
            tag = "PRON"
        elif word in conjunctions:
            tag = "CONJ"
        elif word in prepositions:
            tag = "ADP"
        elif word.endswith("ly"):
            tag = "ADV"
        elif word.endswith(("ing", "ed", "ize", "ise")):
            tag = "VERB"
        elif word.endswith(("ous", "ful", "able", "ible", "ive", "al")):
            tag = "ADJ"
        elif word.isdigit():
            tag = "NUM"
        elif word.endswith("s"):
            tag = "NOUN"
        else:
            tag = "OTHER"
        tags.append((token, tag))
    return tags


def render_app(profile_id: str) -> None:
    if profile_id in PROFILES:
        _render_groq(profile_id)
    elif profile_id in {"speech_to_text_transcriber", "emotion_detection_from_voice"}:
        _render_audio(profile_id)
    elif profile_id in LOCAL_PROFILES:
        _render_local(profile_id)
    else:
        raise ValueError(f"Unknown NLP app profile: {profile_id}")
