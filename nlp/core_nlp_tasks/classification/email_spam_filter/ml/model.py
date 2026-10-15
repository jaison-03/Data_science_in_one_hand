"""TF-IDF spam model training, evaluation, persistence, and inference."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from nlp.core_nlp_tasks.email_spam_filter.ml.preprocessing import (
    load_synthetic_dataset,
    normalize_text,
    validate_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "spam_classifier.joblib"
METRICS_PATH = PROJECT_DIR / "models" / "model_metrics.json"
CUSTOM_DATASET_PATH = PROJECT_DIR / "data" / "custom_dataset.csv"


def load_training_data(dataset_path=None):
    """Load the selected CSV or use the built-in dataset when none is selected."""
    if dataset_path is None:
        return validate_dataset(load_synthetic_dataset())
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return validate_dataset(pd.read_csv(path))


def _make_pipeline(classifier):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))),
            ("classifier", classifier),
        ]
    )


def _evaluate(model, x_test, y_test):
    predicted = model.predict(x_test)
    class_names = list(model.classes_)
    spam_index = class_names.index("spam")
    probabilities = model.predict_proba(x_test)[:, spam_index]
    matrix = confusion_matrix(y_test, predicted, labels=["ham", "spam"])
    fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label="spam")
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, pos_label="spam", zero_division=0)),
        "recall": float(recall_score(y_test, predicted, pos_label="spam", zero_division=0)),
        "f1_score": float(f1_score(y_test, predicted, pos_label="spam", zero_division=0)),
        "roc_auc": float(auc(fpr, tpr)),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": ["ham", "spam"],
        "roc_curve": {"false_positive_rate": fpr.tolist(), "true_positive_rate": tpr.tolist()},
        "test_samples": int(len(y_test)),
    }


def train_and_evaluate(dataset_path=None, random_state=42):
    """Compare Naive Bayes and logistic regression; persist the stronger F1 model."""
    data = load_training_data(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.25,
        random_state=random_state,
        stratify=data["label"],
    )
    candidates = {
        "Multinomial Naive Bayes": _make_pipeline(MultinomialNB(alpha=0.5)),
        "Logistic Regression": _make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
    }
    comparisons = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        comparisons[name] = _evaluate(candidate, x_test, y_test)

    best_name = max(comparisons, key=lambda name: comparisons[name]["f1_score"])
    best_model = _make_pipeline(
        MultinomialNB(alpha=0.5)
        if best_name == "Multinomial Naive Bayes"
        else LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    best_model.fit(data["text"], data["label"])
    best_metrics = comparisons[best_name]
    metrics = {
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "model_comparison": comparisons,
        "training_samples": int(len(data)),
        "dataset": str(dataset_path) if dataset_path is not None else "Built-in synthetic demonstration data",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _load_model.cache_clear()
    return best_model, metrics


@lru_cache(maxsize=1)
def _load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained spam model not found: {path}")
    return joblib.load(path)


def get_model(model_path=None):
    """Load a saved model, training the demonstration model on first use."""
    path = Path(model_path) if model_path is not None else MODEL_PATH
    if not path.is_file():
        if model_path is not None:
            raise FileNotFoundError(f"Trained spam model not found: {path}")
        train_and_evaluate()
    return _load_model(str(path))


def predict_spam_probability(text, model=None):
    """Return the model's spam probability in the range [0, 1]."""
    active_model = model if model is not None else get_model()
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Email text must not be empty.")
    probabilities = active_model.predict_proba([normalized])[0]
    spam_index = list(active_model.classes_).index("spam")
    return float(np.clip(probabilities[spam_index], 0.0, 1.0))


def predict_email(text, threshold=0.5, model=None):
    """Classify one email and return its label and calibrated model score."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    probability = predict_spam_probability(text, model=model)
    return {
        "label": "spam" if probability >= threshold else "ham",
        "spam_probability": probability,
        "ham_probability": 1.0 - probability,
        "threshold": float(threshold),
    }


def load_saved_metrics():
    """Read the metrics from the most recent successful training run."""
    if not METRICS_PATH.is_file():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
