"""
Простой скрипт инференса.

Пример:
    python predict.py --model models/best_model.joblib --input sample.csv --output preds.csv

Если --input не указан, генерируется небольшой синтетический набор для smoke-test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.features import add_features, DROP_COLS


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if "model" not in bundle or "preprocessor" not in bundle:
        raise ValueError("Bundle must contain 'model' and 'preprocessor'")
    return bundle


def prepare_for_inference(df: pd.DataFrame, high_card_te_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Feature engineering + дроп сырых колонок.
    Target Encoding для high-card в проде уже «зашит» в preprocessor pipeline
    либо должен приходить готовыми *_TE колонками.
    Здесь упрощённый путь: ожидаем, что preprocessor умеет работать с сырыми
    low-card + numeric после add_features.
    """
    df = add_features(df)
    drop = [c for c in DROP_COLS if c in df.columns]
    if "TARGET" in df.columns:
        drop = drop + ["TARGET"]
    return df.drop(columns=drop, errors="ignore")


def main():
    parser = argparse.ArgumentParser(description="Credit scoring inference")
    parser.add_argument("--model", type=str, default="models/best_model.joblib")
    parser.add_argument("--input", type=str, default=None, help="CSV with features")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Сначала обучите модель в ноутбуке и сохраните bundle."
        )

    bundle = load_bundle(model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    meta = bundle.get("meta", {})

    if args.input:
        raw = pd.read_csv(args.input)
    else:
        # smoke-test на синтетике
        raw = load_data(n_samples=200)
        print("No --input provided, using synthetic sample (n=200)")

    X = prepare_for_inference(raw)

    # если в meta сохранены high_card TE-колонки — они должны уже быть в X
    # (в текущем упрощённом пайплайне TE делается до preprocessor)
    # для полноценного прода лучше упаковать TE внутрь sklearn Pipeline

    try:
        X_t = preprocessor.transform(X)
    except Exception as e:
        # fallback: если набор колонок не совпал (частая проблема при смене схемы)
        raise RuntimeError(
            f"Preprocessor transform failed: {e}\n"
            "Проверьте, что входные данные содержат те же признаки, "
            "что и при обучении (см. meta['feature_columns'])."
        ) from e

    proba = model.predict_proba(X_t)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = pd.DataFrame(
        {
            "SK_ID_CURR": raw["SK_ID_CURR"] if "SK_ID_CURR" in raw.columns else np.arange(len(raw)),
            "default_proba": proba,
            "default_pred": pred,
        }
    )
    out_path = Path(args.output)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} predictions → {out_path}")
    print(f"Positive rate @ thr={args.threshold}: {pred.mean():.2%}")
    if meta:
        print(f"Model meta: {meta}")


if __name__ == "__main__":
    main()
