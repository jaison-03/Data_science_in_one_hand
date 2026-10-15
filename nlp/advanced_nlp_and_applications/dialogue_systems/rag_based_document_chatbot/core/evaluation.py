from __future__ import annotations

from typing import Iterable


def precision_at_k(relevant_items: Iterable[str], retrieved_items: Iterable[str], k: int | None = None) -> float:
    relevant = set(relevant_items)
    retrieved = list(retrieved_items)[:k] if k is not None else list(retrieved_items)
    if not retrieved:
        return 0.0
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / len(retrieved)


def recall_at_k(relevant_items: Iterable[str], retrieved_items: Iterable[str], k: int | None = None) -> float:
    relevant = set(relevant_items)
    retrieved = list(retrieved_items)[:k] if k is not None else list(retrieved_items)
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved if item in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(relevant_items: Iterable[str], retrieved_items: Iterable[str]) -> float:
    relevant = set(relevant_items)
    for index, item in enumerate(retrieved_items, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def summarize_eval_metrics(relevant_items: Iterable[str], retrieved_items: Iterable[str], k: int | None = None) -> dict[str, float]:
    return {
        "precision@k": precision_at_k(relevant_items, retrieved_items, k),
        "recall@k": recall_at_k(relevant_items, retrieved_items, k),
        "mrr": mean_reciprocal_rank(relevant_items, retrieved_items),
    }
