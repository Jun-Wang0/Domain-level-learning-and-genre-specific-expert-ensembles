"""Shared configuration and validation for genre-specific expert ensembles."""

import ast
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


GENRES = {
    "Book": ["Biography", "Fantasy", "Historical Fiction", "Horror", "Mystery",
             "Romance", "Science Fiction", "Self-Help", "Thriller", "Young Adult"],
    "Game": ["Adventure", "Battle Royale", "Fighting", "Puzzle", "Racing", "RPG",
             "Shooter", "Simulation", "Sports", "Strategy"],
    "Movie": ["Action", "Adventure", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Horror", "Romance", "Thriller"],
}
ALLOCATIONS = {"4+0": (4, 0), "0+4": (0, 4), "1+3": (1, 3),
               "2+2": (2, 2), "3+1": (3, 1)}
MODEL_NAME = "roberta-large"
MAX_LENGTH = 128


def project_root():
    return Path(os.environ.get("ENSEMBLE_ROOT", "/content/drive/MyDrive/PhD"))


def output_root():
    # A new directory protects the original experiment checkpoints.
    return Path(os.environ.get("ENSEMBLE_OUTPUT", str(project_root() / "genre_ensemble_runs")))


def domain_csv(domain):
    override = os.environ.get("ENSEMBLE_" + domain.upper() + "_CSV")
    # Historical filenames are retained for compatibility. Only text and sentiment
    # columns are read by the sentiment pipeline.
    return Path(override) if override else (
        project_root() / "Sentiment_model" / "Dataset" / domain / "Labeled"
        / (domain.lower() + "_metaphor.csv"))


def genre_csv(domain):
    override = os.environ.get("ENSEMBLE_" + domain.upper() + "_GENRE_CSV")
    return Path(override) if override else (
        project_root() / "DataSet" / domain / "merged_balanced_reviews_multi_genre.csv")


def genre_subset(domain, genre):
    return project_root() / "DataSet" / domain / genre / (genre + "_balanced_reviews.xlsx")


def sentiment_root():
    return Path(os.environ.get("ENSEMBLE_SENTIMENT_MODELS", str(output_root() / "sentiment")))


def router_root():
    return Path(os.environ.get("ENSEMBLE_ROUTER_MODELS", str(output_root() / "genre")))


def sentiment_path(domain, allocation, genre=None):
    if allocation == "4+0":
        return sentiment_root() / domain / "All_Combined" / "nometaphor"
    if genre not in GENRES[domain]:
        raise ValueError("An expert checkpoint requires a valid source genre.")
    # The original genre-only runs used no allocation suffix.
    suffix = "" if allocation == "0+4" else "_" + allocation
    return sentiment_root() / domain / genre / (genre + "_nometaphor" + suffix)


def router_path(domain):
    return router_root() / (domain + "_Genre_Classification")


def binary_label(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            raise ValueError("Invalid binary sentiment label: %r" % value)
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected a scalar or a singleton label.")
        value = value[0]
    if isinstance(value, str):
        value = float(value)
    if value not in (0, 1):
        raise ValueError("Sentiment labels must be 0 or 1, received %r" % value)
    return int(value)


def load_reviews(path):
    path = Path(path)
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    text_col = next((c for c in ("review", "sentence", "text") if c in frame), None)
    label_col = next((c for c in ("sentiment", "label") if c in frame), None)
    if text_col is None or label_col is None:
        raise ValueError("Expected review/sentence/text and sentiment/label columns in %s" % path)
    if frame.empty:
        raise ValueError("Empty evaluation/training corpus: %s" % path)
    # Preserve the original notebook's astype(str) behavior and row order.
    return frame[text_col].astype(str).tolist(), [binary_label(x) for x in frame[label_col]]


def sentiment_metrics(labels, predictions):
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("Labels and predictions must have the same nonzero length.")
    p, r, f, _ = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(labels, predictions) * 100,
            "Precision": p * 100, "Recall": r * 100, "F1": f * 100}


def resolve_genre_order(config, expected, explicit=None):
    mapping = config.id2label or {}
    order = explicit if explicit is not None else [
        mapping.get(i, mapping.get(str(i))) for i in range(len(expected))]
    if (config.num_labels != len(expected) or len(order) != len(expected)
            or len(set(order)) != len(order) or set(order) != set(expected)):
        raise ValueError("Router label order is not verified. Supply its original CSV "
                         "column order in --router-labels; do not assume alphabetical order.")
    return list(order)


def combine_logits(logits, weights=None):
    """Combine [reviews, experts, 2] raw logits, never expert probabilities."""
    if logits.ndim != 3 or logits.shape[1] == 0 or logits.shape[2] != 2:
        raise ValueError("Expected nonempty [N, K, 2] sentiment logits.")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite expert logits.")
    if weights is None:
        weights = torch.full(logits.shape[:2], 1 / logits.shape[1],
                             dtype=logits.dtype, device=logits.device)
    if (weights.shape != logits.shape[:2] or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5)):
        raise ValueError("Expected finite nonnegative expert weights summing to one.")
    return (weights.unsqueeze(-1) * logits).sum(1)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
