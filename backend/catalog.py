"""Catalog the existing Python demos without importing their UI or model code."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_AREA_NAMES = {
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "nlp": "Natural Language Processing",
}

_DIRECT_DASHBOARDS = (
    (
        "ml/supervised/regression/estimating_life_expectancy/life_expectancy.py",
        "Estimating Life Expectancy",
        "Regression Dashboards",
        "Life Expectancy Prediction",
    ),
    (
        "ml/supervised/regression/market_forecasting/market_forecasting.py",
        "Market Forecasting",
        "Regression Dashboards",
        "Market Forecasting",
    ),
    (
        "ml/supervised/regression/population_growth_prediction/population_growth.py",
        "Population Growth Prediction",
        "Regression Dashboards",
        "Population Growth Prediction",
    ),
    (
        "ml/supervised/regression/weather_forecasting/weather_forecasting.py",
        "Weather Forecasting",
        "Regression Dashboards",
        "Weather Forecasting",
    ),
)

_TEACHING_PROJECTS = (
    "Advertising Popularity Prediction",
    "Diagnostics",
    "Customer Retention",
    "Image Classification",
    "Identity Fraud Detection",
    "Feature Elicitation",
    "Feature Selection",
    "Structure Discovery",
    "Meaningful Compression",
    "Big Data Visualisation",
    "Big Data Visualization",
    "Customer Segmentation",
    "Targeted Marketing",
    "Recommender Systems",
    "Recommender System",
    "Game AI",
    "Skill Acquisition",
    "Learning Tasks",
    "Robot Navigation",
    "Real-time decisions",
    "Real-Time Decisions",
)

_TITLE_OVERRIDES = {
    "a_star_pathfinding": "A* Pathfinding",
    "advanced_nlp_and_applications": "Advanced NLP & Applications",
    "core_nlp_tasks": "Core NLP Tasks",
    "code_generator_simple_python": "Code Generator [Simple Python]",
    "context_aware_chatbot": "Context-Aware Chatbot",
    "domain_specific_chatbot": "Domain-Specific Chatbot",
    "emerging_domains": "Emerging Domains",
    "finance_agentic_chatbot": "Finance Agentic AI Chatbot",
    "entity_extractor_from_contracts": "Entity Extractor from Contracts",
    "email_spam_filter": "Email Spam Filter",
    "explainable_ai_xai": "Explainable AI (XAI)",
    "faq_chatbot": "FAQ Chatbot",
    "extractive_text_summarizer": "Extractive Text Summarizer",
    "gans_for_image_generation": "GANs for Image Generation",
    "handwritten_pdf_rag": "Handwritten PDF Intelligence",
    "image_pdf_rag": "Scanned / Image-based PDFs",
    "language_modeling_and_generation": "Language Modeling & Generation",
    "large_language_models_llms": "Large Language Models (LLMs)",
    "logic_and_reasoning": "Logic & Reasoning",
    "medical_ner": "Medical NER",
    "machine_learning": "Machine Learning",
    "min_max_for_games": "Min-Max for Games",
    "math_equation_rag": "Math & Equations",
    "n_gram_generator": "N-gram Generator",
    "named_entity_recognition": "Named Entity Recognition",
    "pos_tagger": "POS Tagger",
    "rag_based_document_chatbot": "RAG-Based Document Chatbot",
    "robotics_and_control": "Robotics & Control",
    "rule_based_explanation": "Rule-based Explanation",
    "resume_parsers": "Resume Parsers",
    "sequence_to_sequence": "Sequence-to-Sequence",
    "sentiment_with_word_vectors": "Sentiment with Word Vectors",
    "speech_to_text_transcriber": "Speech-to-Text Transcriber",
    "supervised_learning": "Supervised Learning",
    "text_processing_and_representation": "Text Processing & Representation",
    "tf_idf_classifier": "TF-IDF Classifier",
    "unsupervised_learning": "Unsupervised Learning",
    "word2vec_explorer": "Word2Vec Explorer",
}


class Project(TypedDict):
    id: str
    title: str
    area: str
    category: str
    description: str
    source: str
    legacy_module: str
    legacy_node: str | None
    legacy_view: str | None


def _title(value: str) -> str:
    if value in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[value]
    return " ".join(part.capitalize() for part in value.split("_"))


def _identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _legacy_destination(
    area: str,
    title: str,
    feature: str,
) -> tuple[str, str | None, str | None]:
    if area == "Artificial Intelligence":
        return "AI", title, None
    if area == "Natural Language Processing":
        nlp_views = {
            "handwritten_pdf_rag": "rag_handwritten",
            "image_pdf_rag": "rag_scanned",
            "math_equation_rag": "rag_math",
            "rag_based_document_chatbot": "rag_standard",
        }
        if feature in nlp_views:
            return "NLP", None, nlp_views[feature]
        return "NLP", title, "project"
    if title in {entry[1] for entry in _DIRECT_DASHBOARDS}:
        return title, None, None
    return "Mini Project", title, None


def build_catalog(project_root: Path = PROJECT_ROOT) -> list[Project]:
    """Return the existing AI, ML, and NLP features with safe relative source paths."""
    projects: list[Project] = []

    for area_key, area_name in _AREA_NAMES.items():
        root = project_root / area_key
        if not root.is_dir():
            continue

        for source_path in sorted(root.rglob("app.py")):
            relative = source_path.relative_to(project_root)
            if area_key == "nlp" and len(relative.parts) == 2:
                continue

            feature = source_path.parent.name
            title = _title(feature)
            category = " › ".join(_title(part) for part in relative.parts[1:-1]) or area_name
            legacy_module, legacy_node, legacy_view = _legacy_destination(
                area_name,
                title,
                feature,
            )
            description = (
                f"Existing {area_name} project. Its Python implementation is preserved."
            )
            projects.append(
                Project(
                    id=_identifier(relative.with_suffix("").as_posix()),
                    title=title,
                    area=area_name,
                    category=category,
                    description=description,
                    source=str(relative),
                    legacy_module=legacy_module,
                    legacy_node=legacy_node,
                    legacy_view=legacy_view,
                )
            )

    for relative_source, title, category, legacy_module in _DIRECT_DASHBOARDS:
        source_path = project_root / Path(relative_source)
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(project_root)
        projects.append(
            Project(
                id=_identifier(relative.with_suffix("").as_posix()),
                title=title,
                area="Machine Learning",
                category=category,
                description=f"Existing interactive Python {title.lower()} dashboard.",
                source=str(relative),
                legacy_module=legacy_module,
                legacy_node=None,
                legacy_view=None,
            )
        )

    for title in _TEACHING_PROJECTS:
        projects.append(
            Project(
                id=f"mini-project-{_identifier(title)}",
                title=title,
                area="Machine Learning",
                category="Interactive Teaching Demos",
                description="Existing interactive machine-learning teaching demo.",
                source="mini_projects.py",
                legacy_module="Mini Project",
                legacy_node=title,
                legacy_view=None,
            )
        )

    seen_ids: set[str] = set()
    for project in projects:
        base_id = project["id"]
        suffix = 2
        while project["id"] in seen_ids:
            project["id"] = f"{base_id}-{suffix}"
            suffix += 1
        seen_ids.add(project["id"])

    return sorted(projects, key=lambda item: (item["area"], item["title"].casefold(), item["id"]))


def filter_catalog(
    projects: list[Project],
    *,
    query: str = "",
    area: str | None = None,
    sort: Literal["title", "area", "category"] = "title",
) -> list[Project]:
    """Search and sort catalog entries without loading feature modules."""
    normalized_query = query.strip().casefold()
    matches = [
        project
        for project in projects
        if (area is None or project["area"] == area)
        and (
            not normalized_query
            or normalized_query in project["title"].casefold()
            or normalized_query in project["category"].casefold()
            or normalized_query in project["description"].casefold()
        )
    ]
    return sorted(matches, key=lambda item: (item[sort].casefold(), item["title"].casefold(), item["id"]))
