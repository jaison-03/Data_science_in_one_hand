"""Tests for normalized datasets, training, and spam predictions."""

import pandas as pd
import pytest

from nlp.core_nlp_tasks.email_spam_filter.ml.model import predict_email, train_and_evaluate
from nlp.core_nlp_tasks.email_spam_filter.ml.preprocessing import (
    load_synthetic_dataset,
    normalize_text,
    validate_dataset,
)


def test_normalize_text_handles_missing_and_whitespace():
    assert normalize_text(None) == ""
    assert normalize_text("  Hello\n\tworld  ") == "Hello world"


def test_synthetic_dataset_is_labeled_and_balanced():
    data = validate_dataset(load_synthetic_dataset())
    assert set(data["label"]) == {"spam", "ham"}
    assert data["label"].value_counts().nunique() == 1


def test_validate_dataset_combines_subject_and_body():
    data = validate_dataset(
        pd.DataFrame(
            {
                "Subject": ["Good morning", "Claim your prize"] * 4,
                "Body": ["See you tomorrow", "Click now"] * 4,
                "Label": ["ham", "spam"] * 4,
            }
        )
    )
    assert data.loc[0, "text"] == "Good morning See you tomorrow"
    assert data.loc[1, "label"] == "spam"


def test_validate_dataset_rejects_unrecognized_labels():
    frame = pd.DataFrame({"text": ["a", "b"] * 4, "label": ["ham", "junk"] * 4})
    with pytest.raises(ValueError, match="Labels must be"):
        validate_dataset(frame)


def test_training_compares_models_and_predicts_spam(monkeypatch, tmp_path):
    model_module = __import__(
        "nlp.core_nlp_tasks.email_spam_filter.ml.model",
        fromlist=["MODEL_PATH", "METRICS_PATH"],
    )
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "spam_classifier.joblib")
    monkeypatch.setattr(model_module, "METRICS_PATH", tmp_path / "model_metrics.json")
    model_module._load_model.cache_clear()

    model, metrics = train_and_evaluate(random_state=3)

    assert model_module.MODEL_PATH.is_file()
    assert set(metrics["model_comparison"]) == {"Multinomial Naive Bayes", "Logistic Regression"}
    for values in metrics["model_comparison"].values():
        assert {"accuracy", "precision", "recall", "f1_score", "confusion_matrix", "roc_auc"} <= values.keys()
    result = predict_email(
        "URGENT claim your free prize now by clicking this link",
        threshold=0.5,
        model=model,
    )
    assert result["label"] == "spam"
    assert 0 <= result["spam_probability"] <= 1


def test_prediction_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="Threshold"):
        predict_email("A message", threshold=1.2, model=object())
