"""Обучение моделей: baseline + RF + boosting с Optuna."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from .utils import RANDOM_STATE, evaluate


def train_logreg(X_train, y_train, X_val, y_val) -> tuple[Any, dict]:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    return model, evaluate(y_val, prob)


def train_random_forest(X_train, y_train, X_val, y_val) -> tuple[Any, dict]:
    base = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12, None],
        "min_samples_leaf": [5, 15],
    }
    gs = GridSearchCV(base, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    prob = gs.predict_proba(X_val)[:, 1]
    return gs.best_estimator_, evaluate(y_val, prob)


def _pos_weight(y_train) -> float:
    return float((y_train == 0).sum() / max((y_train == 1).sum(), 1))


def train_xgboost(
    X_train, y_train, X_val, y_val, n_trials: int = 40
) -> tuple[Any, dict, optuna.Study]:
    pos_w = _pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": pos_w,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=RANDOM_STATE)
    )
    # show_progress_bar=False — меньше шума в логах
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "scale_pos_weight": pos_w,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    prob = model.predict_proba(X_val)[:, 1]
    return model, evaluate(y_val, prob), study


def train_lightgbm(
    X_train, y_train, X_val, y_val, n_trials: int = 40
) -> tuple[Any, dict, optuna.Study]:
    pos_w = _pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": pos_w,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        prob = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "scale_pos_weight": pos_w,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    prob = model.predict_proba(X_val)[:, 1]
    return model, evaluate(y_val, prob), study
