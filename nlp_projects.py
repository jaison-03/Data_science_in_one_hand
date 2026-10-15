"""Interactive NLP mind map and Groq-powered project demos."""

from collections import Counter
import re

import streamlit as st
import streamlit.components.v1 as components


BRANCHES = [
    {
        "name": "Text Processing & Representation",
        "color": "#19c7d4",
        "side": "left",
        "position": "top",
        "categories": [
            ("Basic Processing", ["Text Normalizer", "Custom Tokenizer", "POS Tagger", "Keyword Extractor"]),
            ("Vectorization", ["Tf-Idf Classifier", "Word2Vec Explorer"]),
            ("Embeddings", ["N-gram Generator", "Sentiment with Word Vectors"]),
        ],
    },
    {
        "name": "Core NLP Tasks",
        "color": "#ed4da5",
        "side": "right",
        "position": "top",
        "categories": [
            ("Classification", ["Email Spam Filter", "News Categorization"]),
            ("Named Entity Recognition (NER)", ["Resume Parsers", "Entity Extractor from Contracts", "Medical NER"]),
            ("Summarization", ["Abstractive Summarizer", "Extractive Text Summarizer"]),
        ],
    },
    {
        "name": "Language Modeling & Generation",
        "color": "#9770ed",
        "side": "left",
        "position": "bottom",
        "categories": [
            ("Sequence-to-Sequence", ["English to Spanish Translator", "Grammar Corrector"]),
            ("Transformer Models", ["Context-Aware Chatbot", "Code Generator [Simple Python]", "Blog Post Idea Generator"]),
            ("Creative Writing", ["Poetry Writing Assistant"]),
        ],
    },
    {
        "name": "Advanced NLP & Applications",
        "color": "#e8ad36",
        "side": "right",
        "position": "bottom",
        "categories": [
            ("Speech Processing", ["Speech-to-Text Transcriber", "Emotion Detection from Voice"]),
            ("Dialogue Systems", ["FAQ Chatbot", "Multilingual Question Answering System", "RAG-Based Document Chatbot"]),
            ("Specialized Tasks", ["Information Retrieval Engine", "Domain-Specific Chatbot"]),
        ],
    },
]

PALETTE = {
    "Text Processing & Representation": ("#116b78", "#18d2da"),
    "Core NLP Tasks": ("#71225f", "#f05cae"),
    "Language Modeling & Generation": ("#3f347d", "#a178f3"),
    "Advanced NLP & Applications": ("#765318", "#f3c14f"),
}

PROJECT_PROFILES = {
    "News Categorization": "news_categorization",
    "Resume Parsers": "resume_parser",
    "Entity Extractor from Contracts": "contract_extractor",
    "Medical NER": "medical_ner",
    "Abstractive Summarizer": "abstractive_summarizer",
    "Extractive Text Summarizer": "extractive_text_summarizer",
    "Speech-to-Text Transcriber": "speech_to_text_transcriber",
    "Emotion Detection from Voice": "emotion_detection_from_voice",
    "FAQ Chatbot": "faq_chatbot",
    "Information Retrieval Engine": "information_retrieval_engine",
    "Domain-Specific Chatbot": "domain_chatbot",
    "English to Spanish Translator": "english_to_spanish_translator",
    "Grammar Corrector": "grammar_corrector",
    "Context-Aware Chatbot": "context_aware_chatbot",
    "Code Generator [Simple Python]": "code_generator",
    "Blog Post Idea Generator": "blog_post_idea_generator",
    "Poetry Writing Assistant": "poetry_writing_assistant",
    "Text Normalizer": "text_normalizer",
    "Custom Tokenizer": "custom_tokenizer",
    "POS Tagger": "pos_tagger",
    "Keyword Extractor": "keyword_extractor",
    "Tf-Idf Classifier": "tf_idf_classifier",
    "Word2Vec Explorer": "word2vec_explorer",
    "N-gram Generator": "n_gram_generator",
    "Sentiment with Word Vectors": "sentiment_with_word_vectors",
}


def _svg_text(label, x, y, width, color, node_type, view, lines=None, height=38, radius=12):
    lines = lines or [label]
    first_y = y - ((len(lines) - 1) * 7)
    text = "".join(
        f'<tspan x="{x + width / 2:.0f}" dy="{0 if i == 0 else 14}">{line}</tspan>'
        for i, line in enumerate(lines)
    )
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<g class="node {node_type}" role="button" tabindex="0" '
        f'aria-label="{safe_label}" data-view="{view}" data-node="{safe_label}">'
        f'<rect x="{x}" y="{y - height / 2:.0f}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="url(#gradient-{color[1:]})" stroke="{color}" stroke-width="1.5"/>'
        f'<text x="{x + width / 2:.0f}" y="{first_y:.0f}" text-anchor="middle">{text}</text></g>'
    )


def _mind_map_svg():
    width, height, center_x, center_y = 1400, 980, 700, 490
    markup = [
        f'<svg id="nlp-map" viewBox="0 0 {width} {height}" '
        'style="display:block;width:100%;min-width:900px;height:auto" role="img" '
        'aria-labelledby="nlp-map-title nlp-map-desc" xmlns="http://www.w3.org/2000/svg">',
        '<title id="nlp-map-title">Natural Language Processing mind map</title>',
        '<desc id="nlp-map-desc">Four clickable NLP branches, their topic groups, and interactive project demos.</desc>',
        '<defs>',
        '<filter id="glow" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="10" result="blur"/><feMerge>'
        '<feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
    ]
    for branch in BRANCHES:
        start, end = PALETTE[branch["name"]]
        markup.append(
            f'<linearGradient id="gradient-{branch["color"][1:]}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{start}"/><stop offset="100%" stop-color="{end}"/></linearGradient>'
        )
    markup.extend([
        '<linearGradient id="gradient-center" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#124e4e"/><stop offset="100%" stop-color="#062d35"/></linearGradient>',
        '</defs>',
        '<style>'
        'text{font-family:Inter,"Segoe UI",sans-serif;fill:#f2f5fa;font-size:15px;pointer-events:none}'
        '.node{cursor:pointer;outline:none}.node rect{transition:filter .16s,stroke-width .16s}'
        '.node:hover rect,.node:focus rect{filter:brightness(1.24) drop-shadow(0 0 7px currentColor);stroke-width:3}'
        '.leaf text{font-size:13px}.branch text{font-size:16px;font-weight:650}'
        '.center text{font-size:21px;font-weight:750}.connector{fill:none;stroke-width:3;stroke-linecap:round;opacity:.72}'
        '</style>',
    ])

    left_branch_x, right_branch_x = 440, 750
    left_sub_x, right_sub_x = 240, 980
    left_leaf_x, right_leaf_x = 20, 1190
    branch_ys = {"top": 280, "bottom": 700}
    sub_ys = {
        ("top", 0): 95, ("top", 1): 280, ("top", 2): 445,
        ("bottom", 0): 580, ("bottom", 1): 735, ("bottom", 2): 890,
    }
    for branch in BRANCHES:
        left = branch["side"] == "left"
        branch_x = left_branch_x if left else right_branch_x
        sub_x = left_sub_x if left else right_sub_x
        leaf_x = left_leaf_x if left else right_leaf_x
        branch_y = branch_ys[branch["position"]]
        color = branch["color"]
        direction = -1 if left else 1
        start_x = center_x + direction * 82
        end_x = branch_x + (210 if left else 0)
        markup.append(
            f'<path class="connector" stroke="{color}" d="M{start_x} {center_y} '
            f'C{(start_x + end_x) / 2:.0f} {center_y}, {(start_x + end_x) / 2:.0f} {branch_y}, {end_x} {branch_y}"/>'
        )
        branch_lines = {
            "Text Processing & Representation": ["Text Processing", "& Representation"],
            "Core NLP Tasks": ["Core NLP", "Tasks"],
            "Language Modeling & Generation": ["Language Modeling", "& Generation"],
            "Advanced NLP & Applications": ["Advanced NLP", "& Applications"],
        }[branch["name"]]
        markup.append(_svg_text(
            branch["name"], branch_x, branch_y, 210, color, "branch", "category",
            lines=branch_lines, height=76, radius=22,
        ))
        for category_index, (category, leaves) in enumerate(branch["categories"]):
            sub_y = sub_ys[(branch["position"], category_index)]
            sub_center_x = sub_x + 90
            branch_edge = branch_x if left else branch_x + 210
            sub_edge = sub_x + 180 if left else sub_x
            midpoint_x = (branch_edge + sub_edge) / 2
            markup.append(
                f'<path class="connector" stroke="{color}" d="M{branch_edge} {branch_y} '
                f'C{midpoint_x} {branch_y}, {midpoint_x} {sub_y}, {sub_edge} {sub_y}"/>'
            )
            cat_lines = [category] if len(category) <= 22 else category.replace(" (", "\n(").split("\n")
            markup.append(_svg_text(
                category, sub_x, sub_y, 180, color, "subcategory", "subcategory",
                lines=cat_lines, height=max(40, len(cat_lines) * 17 + 12), radius=15,
            ))
            gap = 44
            first_leaf_y = sub_y - (len(leaves) - 1) * gap / 2
            for leaf_index, leaf in enumerate(leaves):
                leaf_y = first_leaf_y + leaf_index * gap
                leaf_edge = leaf_x + 190 if left else leaf_x
                markup.append(
                    f'<path class="connector" stroke="{color}" d="M{sub_edge} {sub_y} '
                    f'C{(sub_edge + leaf_edge) / 2:.0f} {sub_y}, '
                    f'{(sub_edge + leaf_edge) / 2:.0f} {leaf_y:.0f}, {leaf_edge} {leaf_y:.0f}"/>'
                )
                leaf_lines = [leaf] if len(leaf) <= 24 else leaf.replace(" [", "\n[").split("\n")
                markup.append(_svg_text(
                    leaf, leaf_x, leaf_y, 190, color, "leaf", "project",
                    lines=leaf_lines, height=max(34, len(leaf_lines) * 15 + 8), radius=11,
                ))

    markup.append(
        f'<g class="node center" role="button" tabindex="0" aria-label="NLP, Natural Language Processing" '
        f'data-view="map" data-node=""><circle cx="{center_x}" cy="{center_y}" r="88" '
        'fill="url(#gradient-center)" stroke="#dfc66b" stroke-width="3" filter="url(#glow)"/>'
        f'<circle cx="{center_x}" cy="{center_y}" r="76" fill="none" stroke="#35bec0" stroke-opacity=".65"/>'
        f'<text x="{center_x}" y="{center_y - 12}" text-anchor="middle">NLP</text>'
        f'<text x="{center_x}" y="{center_y + 14}" text-anchor="middle" font-size="13">Natural Language</text>'
        f'<text x="{center_x}" y="{center_y + 34}" text-anchor="middle" font-size="13">Processing</text></g>'
    )
    markup.extend([
        '</svg>',
        '<script>',
        'function openNode(node){const url=new URL(window.parent.location);'
        'url.searchParams.set("nlp_view",node.dataset.view);'
        'if(node.dataset.node)url.searchParams.set("nlp_node",node.dataset.node);'
        'else url.searchParams.delete("nlp_node");'
        'window.parent.history.pushState({}, "", url);window.parent.location.reload();}'
        'document.querySelectorAll("#nlp-map .node").forEach(node=>{'
        'node.addEventListener("click",()=>openNode(node));'
        'node.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){'
        'event.preventDefault();openNode(node);}});});',
        '</script>',
    ])
    return "".join(markup)


def _word_tokens(text):
    return re.findall(r"\b[\w'-]+\b", text.lower())


def _basic_text_demo(name, text):
    if name == "Text Normalizer":
        options = st.columns(3)
        lowercase = options[0].checkbox("Lowercase", True)
        remove_punctuation = options[1].checkbox("Remove punctuation", True)
        normalize_space = options[2].checkbox("Normalize whitespace", True)
        result = text.lower() if lowercase else text
        if remove_punctuation:
            result = re.sub(r"[^\w\s]", "", result)
        if normalize_space:
            result = " ".join(result.split())
        st.code(result or "Enter text above to see the normalized result.")
    elif name == "Custom Tokenizer":
        mode = st.selectbox("Tokenization mode", ["Words", "Sentences", "Words and punctuation"])
        if mode == "Sentences":
            tokens = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        elif mode == "Words and punctuation":
            tokens = re.findall(r"\w+|[^\w\s]", text)
        else:
            tokens = re.findall(r"\b[\w'-]+\b", text)
        st.write(f"**{len(tokens)} tokens**")
        st.code(" | ".join(tokens))
    elif name == "POS Tagger":
        tokens = re.findall(r"\b[\w'-]+\b", text)
        tags = []
        for word in tokens:
            lower = word.lower()
            tag = "VERB" if lower.endswith(("ing", "ed")) else (
                "ADVERB" if lower.endswith("ly") else
                "ADJECTIVE" if lower.endswith(("ous", "ful", "ive", "able")) else
                "NOUN" if lower.endswith(("tion", "ness", "ment")) else
                "PROPN" if word[:1].isupper() else "WORD"
            )
            tags.append(f"{word}/{tag}")
        st.code("  ".join(tags) or "No words found.")
        st.caption("Rule-based teaching heuristic; this is not a trained linguistic tagger.")
    elif name == "Keyword Extractor":
        stop_words = set("a an and are as at be by for from in is it of on or that the this to was with".split())
        counts = Counter(token for token in _word_tokens(text) if token not in stop_words and len(token) > 2)
        count = st.slider("Keywords to show", 3, 12, 6)
        st.dataframe(
            [{"Keyword": word, "Occurrences": amount} for word, amount in counts.most_common(count)],
            hide_index=True,
            use_container_width=True,
        )
    elif name == "Tf-Idf Classifier":
        docs = st.text_area(
            "Small labeled corpus (one document per line; label and text separated by a colon)",
            "sports: team won the football match\ntechnology: new software and AI technology\nsports: player scored a goal",
            height=120,
        )
        query = st.text_input("Document to classify", text)
        documents = [line.split(":", 1) for line in docs.splitlines() if ":" in line]
        query_terms = set(_word_tokens(query))
        scores = Counter()
        for label, document in documents:
            terms = set(_word_tokens(document))
            for term in query_terms & terms:
                scores[label.strip()] += 1 + (1 / (1 + sum(term in _word_tokens(d) for _, d in documents)))
        if scores:
            st.metric("Best matching label", scores.most_common(1)[0][0])
            st.dataframe([{"Label": key, "TF-IDF-style score": round(value, 3)} for key, value in scores.most_common()], hide_index=True)
        else:
            st.info("Add labeled documents using `label: text` format.")
        st.caption("A compact, transparent TF-IDF-style baseline over the editable corpus.")
    elif name == "Word2Vec Explorer":
        corpus = st.text_area("Tiny context corpus", "natural language models learn from context\nlanguage models predict the next word\nwords appear in similar contexts", height=100)
        target = st.text_input("Explore context for word", ( _word_tokens(text) or ["language"])[0])
        window = st.slider("Context window", 1, 5, 2)
        found = []
        for sentence in re.split(r"[.!?\n]+", corpus.lower()):
            words = _word_tokens(sentence)
            for index, word in enumerate(words):
                if word == target.lower():
                    found.extend(words[max(0, index - window):index] + words[index + 1:index + window + 1])
        cooccurrence = Counter(found)
        st.write("**Nearest context words in this toy corpus**")
        st.write(", ".join(f"{word} ({count})" for word, count in cooccurrence.most_common(8)) or "No occurrences found.")
        st.caption("Context-count illustration inspired by distributional embeddings, not a trained Word2Vec model.")
    elif name == "N-gram Generator":
        tokens = _word_tokens(text)
        size = st.slider("N-gram size", 1, 5, 2)
        grams = [" ".join(tokens[index:index + size]) for index in range(max(0, len(tokens) - size + 1))]
        st.code("\n".join(grams) or "Enter enough words to generate an n-gram.")
    elif name == "Sentiment with Word Vectors":
        positive = {"good", "great", "love", "excellent", "happy", "wonderful", "best", "enjoy"}
        negative = {"bad", "awful", "hate", "poor", "sad", "terrible", "worst", "angry"}
        tokens = _word_tokens(text)
        score = sum(word in positive for word in tokens) - sum(word in negative for word in tokens)
        sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        st.metric("Lexicon similarity demo", sentiment, delta=f"{score} matched sentiment words")
        st.caption("Uses a tiny transparent vocabulary as a stand-in for vector similarity.")


def _project_demo(name):
    if name == "Email Spam Filter":
        from nlp.core_nlp_tasks.email_spam_filter.ui import render_app

        render_app(embed=True)
        return
    if name == "Multilingual Question Answering System":
        from nlp.advanced_nlp_and_applications.dialogue_systems.multilingual_question_answering_system.ui import render_app

        render_app(embed=True)
        return
    if name == "RAG-Based Document Chatbot":
        st.markdown("Choose the PDF processing pipeline to open.")
        digital, scanned, math = st.columns(3)
        if digital.button(
            "Standard Digital PDFs",
            key="rag-standard-pdf",
            use_container_width=True,
        ):
            st.query_params["nlp_view"] = "rag_standard"
            st.rerun()
        if scanned.button(
            "Scanned / Image-based PDFs",
            key="rag-scanned-pdf",
            use_container_width=True,
        ):
            st.query_params["nlp_view"] = "rag_scanned"
            st.rerun()
        if math.button(
            "Math & Equations",
            key="rag-math-pdf",
            use_container_width=True,
        ):
            st.query_params["nlp_view"] = "rag_math"
            st.rerun()
        return
    if name in PROJECT_PROFILES:
        from nlp.common.apps import render_app

        render_app(PROJECT_PROFILES[name])
        return
    st.subheader(name)
    st.info("This project does not yet have a dedicated app entry.")
    examples = {
        "Email Spam Filter": "Congratulations! You are a winner. Click here to claim your free prize!",
        "News Categorization": "The football team won the match after a player scored a late goal.",
        "Resume Parsers": "Jordan Lee | jordan.lee@example.com | Python, SQL, NLP",
        "Entity Extractor from Contracts": "Agreement between Acme Corp and Jordan Lee, dated March 12, 2025. Payment of $12,500 USD is due.",
        "Medical NER": "Patient reports fever and cough and takes aspirin.",
        "English to Spanish Translator": "thank you",
        "Grammar Corrector": "their is a teh typo",
        "Context-Aware Chatbot": "How can I summarize this text?",
        "Code Generator [Simple Python]": "Write a Python word count function",
        "FAQ Chatbot": "What are your opening hours?",
        "Multilingual Question Answering System": "What is NLP?",
        "RAG-Based Document Chatbot": "Summarize the main ideas in my uploaded document.",
        "Domain-Specific Chatbot": "How do I get started?",
        "Sentiment with Word Vectors": "I love this excellent and wonderful language demo!",
        "Tf-Idf Classifier": "new software and AI technology",
    }
    text = st.text_area(
        "Input text",
        examples.get(
            name,
            "Natural language processing helps computers understand text. A small interactive demo makes language tools easier to explore.",
        ),
        height=130,
        key=f"nlp-input-{name}",
    )

    if name in {
        "Text Normalizer", "Custom Tokenizer", "POS Tagger", "Keyword Extractor",
        "Tf-Idf Classifier", "Word2Vec Explorer", "N-gram Generator", "Sentiment with Word Vectors",
    }:
        _basic_text_demo(name, text)
    elif name == "Email Spam Filter":
        threshold = st.slider("Flag as spam at score", 1, 8, 3)
        signals = {
            "urgent": 2, "free": 2, "winner": 3, "claim": 2, "click": 2,
            "unsubscribe": 1, "prize": 3, "limited time": 2, "verify your account": 3,
        }
        lowered = text.lower()
        matches = [(term, score) for term, score in signals.items() if term in lowered]
        score = min(sum(value for _, value in matches), 10)
        st.metric("Rule-based spam score", f"{score} / 10")
        st.warning("Likely spam — review carefully." if score >= threshold else "No strong spam signals found.")
        st.write("Matched signals:", ", ".join(f"{term} (+{value})" for term, value in matches) or "none")
    elif name == "News Categorization":
        categories = {
            "Technology": {"software", "technology", "computer", "AI", "internet", "device"},
            "Sports": {"team", "match", "player", "score", "league", "game"},
            "Business": {"market", "company", "stock", "business", "sales", "economy"},
            "Health": {"health", "doctor", "medical", "patient", "hospital", "research"},
        }
        tokens = set(_word_tokens(text))
        scores = {label: len(tokens & {word.lower() for word in terms}) for label, terms in categories.items()}
        best = max(scores, key=scores.get)
        st.metric("Predicted section", best if scores[best] else "Unclear")
        st.dataframe([{"Section": key, "Matched terms": value} for key, value in scores.items()], hide_index=True)
    elif name == "Resume Parsers":
        emails = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        phones = re.findall(r"(?:\+?\d[\d ().-]{7,}\d)", text)
        skill_list = ["Python", "SQL", "Java", "machine learning", "NLP", "Excel", "leadership", "project management"]
        skills = [skill for skill in skill_list if skill.lower() in text.lower()]
        st.write({"Email addresses": emails, "Phone numbers": phones, "Matched skills": skills})
        st.caption("Extracts visible contact patterns and a small editable-in-code demo skill vocabulary.")
    elif name == "Entity Extractor from Contracts":
        parties = re.findall(r"\b(?:between|by and between)\s+([A-Z][\w &.,'-]+?)(?=\s+and\s+|\s*,|\s+dated)", text, re.I)
        dates = re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", text, re.I)
        amounts = re.findall(r"(?:[$€£]\s?\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*(?:\.\d{2})?\s?(?:USD|EUR|GBP)\b)", text)
        st.write({"Possible parties": parties, "Dates": dates, "Amounts": amounts})
        st.caption("Pattern-based extraction for exploration only; have a qualified professional review contracts.")
    elif name == "Medical NER":
        terms = {
            "Conditions": ["diabetes", "asthma", "hypertension", "migraine"],
            "Medications": ["insulin", "aspirin", "metformin", "ibuprofen"],
            "Body parts": ["heart", "lung", "kidney", "brain"],
            "Symptoms": ["fever", "cough", "pain", "nausea"],
        }
        result = {label: [term for term in values if re.search(rf"\b{term}\b", text, re.I)] for label, values in terms.items()}
        st.write(result)
        st.warning("Educational named-entity highlighting only — not a diagnostic or clinical tool.")
    elif name in ("Abstractive Summarizer", "Extractive Text Summarizer"):
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        count = st.slider("Summary sentences", 1, max(1, min(5, len(sentences))), min(2, max(1, len(sentences))))
        if name == "Extractive Text Summarizer":
            frequencies = Counter(token for token in _word_tokens(text) if len(token) > 3)
            ranked = sorted(
                enumerate(sentences),
                key=lambda item: sum(frequencies[token] for token in _word_tokens(item[1])),
                reverse=True,
            )
            chosen = sorted(index for index, _ in ranked[:count])
            summary = " ".join(sentences[index] for index in chosen)
        else:
            summary = " ".join(sentences[:count])
            if summary:
                summary = "In brief: " + summary
        st.write("**Draft summary**")
        st.info(summary or "Enter text to generate a summary.")
        st.caption("The abstractive example is a simple local heuristic, not a generative language model.")
    elif name == "English to Spanish Translator":
        phrase_map = {
            "hello": "hola", "good morning": "buenos días", "thank you": "gracias",
            "how are you": "¿cómo estás?", "goodbye": "adiós", "i love language": "me encanta el idioma",
            "the cat is happy": "el gato está feliz", "where is the station": "¿dónde está la estación?",
        }
        source = st.text_input("English phrase", text)
        translated = phrase_map.get(source.strip().lower(), " ".join(phrase_map.get(word.lower(), f"[{word}]") for word in source.split()))
        st.write("**Spanish demo output**")
        st.info(translated)
        st.caption("A small phrase dictionary demonstrates sequence-to-sequence translation; unknown words are shown in brackets.")
    elif name == "Grammar Corrector":
        corrections = {
            r"\btheir is\b": "there is", r"\bi has\b": "I have", r"\bhe don't\b": "he doesn't",
            r"\bshe don't\b": "she doesn't", r"\bteh\b": "the", r"\brecieve\b": "receive",
            r"\b(\w+)\s+\1\b": r"\1",
        }
        result = text
        for pattern, replacement in corrections.items():
            result = re.sub(pattern, replacement, result, flags=re.I)
        st.write("**Suggested correction**")
        st.info(result)
        st.caption("Rule-based suggestions only; grammar and style are context-dependent.")
    elif name == "Context-Aware Chatbot":
        prompt = st.text_input("Ask the demo assistant", text)
        lowered = prompt.lower()
        if any(word in lowered for word in ("summar", "shorten", "brief")):
            answer = "I can help summarize: select the Extractive Text Summarizer in the mind map and paste the passage there."
        elif any(word in lowered for word in ("translate", "spanish")):
            answer = "Try the English to Spanish Translator demo with one of its supported sample phrases."
        elif any(word in lowered for word in ("hello", "hi", "hey")):
            answer = "Hello! I can point you to text processing, summarization, translation, or classification demos."
        else:
            answer = "I understood your message. This compact context demo keeps the current turn and suggests a related NLP tool."
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(answer)
    elif name == "Code Generator [Simple Python]":
        request = st.text_input("Describe a tiny Python task", text)
        if "average" in request.lower():
            code = "def average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0"
        elif "factorial" in request.lower():
            code = "def factorial(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result"
        else:
            code = "def word_count(text):\n    return len(text.split())"
        st.code(code, language="python")
        st.caption("Template-based coding example. Review and test generated code before using it.")
    elif name == "Blog Post Idea Generator":
        topic = st.text_input("Blog topic", "natural language processing")
        audience = st.selectbox("Audience", ["Beginners", "Practitioners", "Researchers"])
        st.write("**Idea starters**")
        for idea in (
            f"A beginner's guide to {topic}",
            f"Five practical ways {topic} helps {audience.lower()}",
            f"What I learned building a small {topic} project",
        ):
            st.markdown(f"- {idea}")
    elif name == "Poetry Writing Assistant":
        subject = st.text_input("Poem subject", "language")
        style = st.selectbox("Mood", ["Hopeful", "Reflective", "Playful"])
        last_line = {"Hopeful": "new meaning blooms in every phrase.", "Reflective": "old echoes soften into light.", "Playful": "words dance, tumble, and take flight."}[style]
        st.write("**A short writing prompt**")
        st.code(f"Small {subject} wakes,\nA quiet thought takes shape,\n{last_line}")
    elif name == "Speech-to-Text Transcriber":
        components.html(
            """<div style="font:15px sans-serif;color:#f2f5fa">
            <button id="start" style="padding:9px 14px;border-radius:8px;cursor:pointer">🎙 Start browser transcription</button>
            <p id="status">Browser support varies. Your browser may send audio to its configured speech service; do not speak sensitive information.</p>
            <textarea id="transcript" rows="4" style="width:100%;box-sizing:border-box" placeholder="Recognized words will appear here"></textarea>
            <script>
            const button=document.getElementById('start'),status=document.getElementById('status'),output=document.getElementById('transcript');
            const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
            button.onclick=()=>{if(!SpeechRecognition){status.textContent='Speech recognition is not supported in this browser. You can use the text box to inspect/edit a transcript.';return;}
            const recognition=new SpeechRecognition();recognition.lang='en-US';recognition.interimResults=true;
            status.textContent='Listening…';recognition.onresult=e=>{output.value=Array.from(e.results).map(r=>r[0].transcript).join(' ');};
            recognition.onerror=e=>{status.textContent='Recognition error: '+e.error;};recognition.onend=()=>{status.textContent='Transcription complete. Edit the transcript above if needed.';};
            recognition.start();};
            </script></div>""",
            height=210,
        )
    elif name == "Emotion Detection from Voice":
        st.caption("Use the controls to model a voice feature snapshot. The demo maps hand-entered features to a broad illustrative label.")
        energy = st.slider("Energy", 0, 100, 55)
        pitch = st.slider("Pitch", 0, 100, 50)
        pace = st.slider("Speaking pace", 0, 100, 50)
        label = "Excited" if energy > 72 and pace > 60 else "Calm" if energy < 35 and pace < 45 else "Expressive / mixed"
        st.metric("Illustrative voice-tone label", label)
        st.caption("This does not listen to or analyze a voice recording.")
    elif name == "FAQ Chatbot":
        question = st.text_input("Ask a sample FAQ", text)
        answers = [
            (("hours", "open"), "We are open Monday to Friday, 9:00 a.m. to 5:00 p.m."),
            (("contact", "email"), "You can contact the team at help@example.test."),
            (("return", "refund"), "Items can be returned within 30 days with proof of purchase."),
            (("shipping", "delivery"), "Standard delivery usually takes 3–5 business days."),
        ]
        response = next((answer for terms, answer in answers if any(term in question.lower() for term in terms)), "I couldn't match that FAQ. Try asking about hours, contact, returns, or shipping.")
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(response)
    elif name == "Multilingual Question Answering System":
        question = st.text_input("Ask about NLP", text)
        language = st.selectbox("Answer language", ["English", "Español", "Français"])
        answers = {
            "English": "NLP is the study of methods that help computers work with human language.",
            "Español": "El PLN estudia métodos que ayudan a las computadoras a trabajar con el lenguaje humano.",
            "Français": "Le TAL étudie des méthodes qui aident les ordinateurs à traiter le langage humain.",
        }
        st.write("**Answer**")
        st.info(answers[language])
        st.caption(f"Question received: {question}")
    elif name == "Information Retrieval Engine":
        corpus = st.text_area(
            "Searchable mini-corpus (one passage per line)",
            "Natural language processing studies language and computers.\nText normalization cleans and standardizes written text.\nNamed entity recognition finds names, places, and dates.\nSummarization creates a shorter version of a document.",
            height=120,
        )
        query = st.text_input("Search query", "language text")
        query_terms = set(_word_tokens(query))
        documents = [line.strip() for line in corpus.splitlines() if line.strip()]
        ranked = []
        for document in documents:
            terms = _word_tokens(document)
            counts = Counter(terms)
            score = sum(counts[term] / len(terms) for term in query_terms) if terms else 0
            ranked.append((score, document))
        ranked.sort(reverse=True)
        st.write("**Ranked results**")
        for score, document in ranked[:5]:
            st.write(f"**{score:.3f}** — {document}")
    elif name == "Domain-Specific Chatbot":
        domain = st.selectbox("Knowledge domain", ["Travel", "Cooking", "Programming", "Study skills"])
        question = st.text_input("Ask the domain demo", text)
        responses = {
            "Travel": "For a trip, check entry requirements, local transport, and weather before you go.",
            "Cooking": "Start with the recipe ingredients and method; adapt seasoning to your taste.",
            "Programming": "Break the task into small functions, test edge cases, and inspect errors.",
            "Study skills": "Try short focused sessions, active recall, and spaced review.",
        }
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(responses[domain])


def render_nlp_workspace(in_dashboard=True):
    """Render the mind map, a topic hub, or the selected interactive leaf demo."""
    view = st.query_params.get("nlp_view", "map")
    node = st.query_params.get("nlp_node", "")

    if in_dashboard:
        if st.button("← Back to main concentric layers view", key="nlp-main-back"):
            st.query_params.clear()
            st.session_state["selected_module"] = "Main Dashboard"
            st.rerun()
    else:
        if st.button("← Back to main concentric layers view", key="nlp-page-main-back"):
            st.query_params.clear()
            st.session_state["selected_module"] = "Main Dashboard"
            st.switch_page("app.py")

    if view == "project" and node == "RAG-Based Document Chatbot":
        if st.button("← Back to NLP mind map", key="nlp-rag-choice-back"):
            st.query_params.pop("nlp_view", None)
            st.query_params.pop("nlp_node", None)
            st.rerun()
        st.title("RAG-Based Document Chatbot")
        _project_demo(node)
        return

    if view == "rag_standard":
        if st.button("← Back to PDF pipeline choices", key="nlp-rag-standard-back"):
            st.query_params["nlp_view"] = "project"
            st.query_params["nlp_node"] = "RAG-Based Document Chatbot"
            st.rerun()
        from nlp.advanced_nlp_and_applications.dialogue_systems.rag_based_document_chatbot.app import render_app

        render_app(embed=True)
        return

    if view == "rag_scanned":
        if st.button("← Back to PDF pipeline choices", key="nlp-rag-scanned-back"):
            st.query_params["nlp_view"] = "project"
            st.query_params["nlp_node"] = "RAG-Based Document Chatbot"
            st.rerun()
        from nlp.advanced_nlp_and_applications.dialogue_systems.rag_based_document_chatbot.image_pdf_rag.app import render_app

        render_app(embed=True)
        return

    if view == "rag_math":
        if st.button("← Back to PDF pipeline choices", key="nlp-rag-math-back"):
            st.query_params["nlp_view"] = "project"
            st.query_params["nlp_node"] = "RAG-Based Document Chatbot"
            st.rerun()
        from nlp.advanced_nlp_and_applications.dialogue_systems.rag_based_document_chatbot.math_equation_rag.app import render_app

        render_app(embed=True)
        return

    if view == "project":
        if st.button("← Back to NLP mind map", key="nlp-project-back"):
            st.query_params.pop("nlp_view", None)
            st.query_params.pop("nlp_node", None)
            st.rerun()
        st.title(node or "NLP project")
        if node not in {leaf for branch in BRANCHES for _, leaves in branch["categories"] for leaf in leaves}:
            st.error("That NLP demo could not be found. Return to the mind map and choose a listed project.")
            return
        _project_demo(node)
        return

    if view in ("category", "subcategory"):
        if st.button("← Back to NLP mind map", key="nlp-category-back"):
            st.query_params.pop("nlp_view", None)
            st.query_params.pop("nlp_node", None)
            st.rerun()
        if view == "category":
            branch = next((item for item in BRANCHES if item["name"] == node), None)
            if branch is None:
                st.error("That NLP topic could not be found. Return to the mind map and choose a listed topic.")
                return
            st.title(node)
            groups = branch["categories"]
        else:
            found = next(
                ((branch, category, leaves) for branch in BRANCHES
                 for category, leaves in branch["categories"] if category == node),
                None,
            )
            if found is None:
                st.error("That NLP topic could not be found. Return to the mind map and choose a listed topic.")
                return
            _, category, leaves = found
            st.title(category)
            groups = [(category, leaves)]
        for category, leaves in groups:
            if view == "category":
                st.subheader(category)
            columns = st.columns(2)
            for index, leaf in enumerate(leaves):
                if columns[index % 2].button(leaf, key=f"nlp-topic-{category}-{leaf}", use_container_width=True):
                    st.query_params["nlp_view"] = "project"
                    st.query_params["nlp_node"] = leaf
                    st.rerun()
        return

    st.title("NLP (Natural Language Processing)")
    st.caption("Choose a colored branch, topic, or leaf to explore its interactive demo.")
    st.markdown(
        "<style>div[data-testid='stAppViewContainer']{background:#0b1527}"
        "div[data-testid='stMarkdownContainer']{color:#e7edf6}</style>",
        unsafe_allow_html=True,
    )
    components.html(
        "<style>html,body{margin:0;background:#0b1527}</style>" + _mind_map_svg(),
        height=840,
        scrolling=True,
    )
