"""Feature engineering для кредитного скоринга."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные признаки.

    Логика:
    - CREDIT_INCOME_RATIO / ANNUITY_INCOME_RATIO — долговая нагрузка
    - AGE_YEARS + AGE_GROUP — возраст и нелинейность по группам
    - IS_UNEMPLOYED + EMPLOYED_YEARS — аккуратная обработка DAYS_EMPLOYED=365243
    - LATE_RATIO / HAS_LATE — нормализованная история просрочек
    - EXT_SOURCE_MEAN / MIN — агрегат внешних скоров (частично закрывает пропуски)
    - LOG_* — для линейных моделей
    """
    df = data.copy()

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["AGE_GROUP"] = pd.cut(
        df["AGE_YEARS"],
        bins=[0, 25, 35, 45, 55, 70],
        labels=["<25", "25-35", "35-45", "45-55", "55+"],
    )

    df["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED_CLEAN"] / 365.25).clip(0, 50)

    df["LATE_RATIO"] = df["CNT_LATE_PAYMENTS"] / (df["CNT_PREV_CREDITS"] + 1)
    df["HAS_LATE"] = (df["CNT_LATE_PAYMENTS"] > 0).astype(int)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)

    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)

    # интересно, а если взять логарифм — часто помогает линейным моделям
    df["LOG_INCOME"] = np.log1p(df["AMT_INCOME_TOTAL"])
    df["LOG_CREDIT"] = np.log1p(df["AMT_CREDIT"])

    return df


# колонки, которые дропаем перед обучением (сырые дни уже преобразованы)
DROP_COLS = [
    "SK_ID_CURR",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_SINCE_LAST_LATE",
    "DAYS_EMPLOYED_CLEAN",
]
