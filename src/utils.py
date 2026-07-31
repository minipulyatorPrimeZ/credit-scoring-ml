"""Вспомогательные функции."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


RANDOM_STATE = 42


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Считает основные метрики для бинарной классификации."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def get_feature_names_from_preprocessor(preprocessor, num_features, low_card) -> list[str]:
    """Достаёт имена признаков после ColumnTransformer (num + OHE)."""
    ohe = preprocessor.named_transformers_["low"].named_steps["ohe"]
    ohe_names = list(ohe.get_feature_names_out(low_card))
    return list(num_features) + ohe_names
