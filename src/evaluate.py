"""Метрики, графики, SHAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from .utils import evaluate, RANDOM_STATE


def print_metrics(name: str, metrics: dict) -> None:
    print(f"{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def plot_confusion(y_true, y_prob, threshold: float = 0.5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No default", "Default"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (thr={threshold})")
    return ax


def plot_roc_pr(y_true, y_prob):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=axes[0])
    axes[0].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=axes[1])
    axes[1].set_title("Precision-Recall")
    plt.tight_layout()
    return fig


def plot_calibration(y_true, y_prob):
    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
    ax.set_title("Calibration Curve")
    # TODO: после Isotonic/Platt калибровка будет ближе к диагонали —
    # для бизнес-решений это важнее чистого AUC
    plt.tight_layout()
    return fig


def plot_shap_summary(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 1500,
    max_display: int = 12,
):
    """SHAP summary для tree-моделей."""
    explainer = shap.TreeExplainer(model)
    n = min(max_samples, X.shape[0])
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    shap_values = explainer.shap_values(X[idx])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X[idx],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary (top features)")
    plt.tight_layout()
    return plt.gcf()


def plot_permutation_importance(
    model: Any,
    X: np.ndarray,
    y,
    feature_names: list[str],
    top_n: int = 12,
):
    r = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
    ax = imp.head(top_n).plot.barh(figsize=(8, 5), color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Permutation Importance (top-{top_n})")
    plt.tight_layout()
    return ax.figure


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).T.round(4)
    return df.sort_values("ROC-AUC", ascending=False)


def save_model_bundle(
    path: str | Path,
    model: Any,
    preprocessor: Any,
    meta: dict | None = None,
) -> Path:
    """Сохраняет модель + preprocessor + метаданные одним joblib-файлом."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "meta": meta or {},
    }
    joblib.dump(bundle, path)
    return path
