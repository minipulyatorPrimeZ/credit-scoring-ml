"""Preprocessing: split, encoding, scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import DROP_COLS
from .utils import RANDOM_STATE


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    """70 / 15 / 15 stratified split."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size от всего ≈ 0.15 → доля от temp
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=["TARGET"] + drop)
    y = df["TARGET"]
    return X, y


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Делит признаки на numeric / low-card cat / high-card cat."""
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    low_card = [c for c in cat_features if X[c].nunique(dropna=False) <= 10]
    high_card = [c for c in cat_features if X[c].nunique(dropna=False) > 10]
    return num_features, low_card, high_card


def build_preprocessors(
    num_features: list[str], low_card: list[str]
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    Два препроцессора:
    - linear: медиана + StandardScaler + OHE
    - tree: медиана + OHE (без масштабирования)
    """
    numeric_linear = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_tree = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    low_card_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor_linear = ColumnTransformer(
        [
            ("num", numeric_linear, num_features),
            ("low", low_card_tf, low_card),
        ],
        remainder="drop",
    )
    preprocessor_tree = ColumnTransformer(
        [
            ("num", numeric_tree, num_features),
            ("low", low_card_tf, low_card),
        ],
        remainder="drop",
    )
    return preprocessor_linear, preprocessor_tree


def apply_target_encoding(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    high_card: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Target Encoding для high-cardinality категорий.
    Fit только на train, чтобы не ликнуть target.
    """
    if not high_card:
        return X_train.copy(), X_val.copy(), X_test.copy(), []

    te = TargetEncoder(cols=high_card, smoothing=10, min_samples_leaf=20)
    X_train_te = te.fit_transform(X_train[high_card], y_train)
    X_val_te = te.transform(X_val[high_card])
    X_test_te = te.transform(X_test[high_card])

    # FIXME: в проде для unseen категорий TE вернёт global mean —
    # лучше заранее завести bucket "Other" по частоте на train

    X_train_enc = X_train.drop(columns=high_card).copy()
    X_val_enc = X_val.drop(columns=high_card).copy()
    X_test_enc = X_test.drop(columns=high_card).copy()

    te_cols = []
    for c in high_card:
        col_name = c + "_TE"
        X_train_enc[col_name] = X_train_te[c].values
        X_val_enc[col_name] = X_val_te[c].values
        X_test_enc[col_name] = X_test_te[c].values
        te_cols.append(col_name)

    return X_train_enc, X_val_enc, X_test_enc, te_cols


def transform_all(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_tr = preprocessor.fit_transform(X_train)
    X_va = preprocessor.transform(X_val)
    X_te = preprocessor.transform(X_test)
    return X_tr, X_va, X_te
