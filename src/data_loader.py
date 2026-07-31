"""Загрузка / генерация данных для кредитного скоринга."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RANDOM_STATE


def generate_home_credit_like(n_samples: int = 28000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Синтетический датасет, близкий по структуре к application_train Home Credit.

    Сохраняет:
    - дисбаланс TARGET ~8–9%
    - пропуски в EXT_SOURCE_*, OCCUPATION_TYPE, OWN_CAR_AGE
    - артефакт DAYS_EMPLOYED = 365243
    """
    rng = np.random.RandomState(random_state)

    age_days = rng.randint(-25000, -8000, n_samples)
    days_employed = rng.randint(-15000, 0, n_samples)
    days_employed[rng.rand(n_samples) < 0.18] = 365243

    income = np.exp(rng.normal(11.2, 0.55, n_samples)).clip(20000, 1_000_000)
    credit = income * rng.uniform(1.5, 6.5, n_samples) + rng.normal(0, 20000, n_samples)
    credit = credit.clip(20000, 2_000_000)
    annuity = credit / rng.uniform(10, 25, n_samples)

    children = rng.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.55, 0.25, 0.12, 0.05, 0.02, 0.01])
    family = children + rng.choice([1, 2], n_samples, p=[0.3, 0.7])

    contract = rng.choice(["Cash loans", "Revolving loans"], n_samples, p=[0.9, 0.1])
    gender = rng.choice(["M", "F"], n_samples, p=[0.34, 0.66])
    education = rng.choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        n_samples,
        p=[0.71, 0.24, 0.03, 0.015, 0.005],
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        n_samples,
        p=[0.64, 0.18, 0.1, 0.05, 0.03],
    )
    housing = rng.choice(
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment",
        ],
        n_samples,
        p=[0.88, 0.05, 0.035, 0.02, 0.01, 0.005],
    )
    occupation = rng.choice(
        [
            "Laborers",
            "Sales staff",
            "Core staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Security staff",
            "Cooking staff",
            "Cleaning staff",
            "Private service staff",
            "Low-skill Laborers",
            "Waiters/barmen staff",
            "Secretaries",
            "Realty agents",
            "HR staff",
            "IT staff",
        ],
        n_samples,
    ).astype(object)
    occupation[rng.rand(n_samples) < 0.31] = np.nan

    org_type = rng.choice(
        [
            "Business Entity Type 3",
            "XNA",
            "Self-employed",
            "Other",
            "Medicine",
            "Government",
            "School",
            "Trade: type 7",
            "Kindergarten",
            "Construction",
            "Business Entity Type 2",
            "Trade: type 3",
            "Military",
            "Industry: type 9",
            "Industry: type 3",
            "Security Ministries",
            "Transport: type 4",
            "Industry: type 11",
            "Bank",
            "Police",
        ],
        n_samples,
    )

    ext1 = rng.beta(2, 5, n_samples)
    ext2 = rng.beta(3, 4, n_samples)
    ext3 = rng.beta(2.5, 4.5, n_samples)
    ext1[rng.rand(n_samples) < 0.56] = np.nan
    ext2[rng.rand(n_samples) < 0.002] = np.nan
    ext3[rng.rand(n_samples) < 0.20] = np.nan

    cnt_prev = rng.poisson(2.5, n_samples).clip(0, 20)
    cnt_late = rng.binomial(cnt_prev, 0.12)
    days_last_late = rng.randint(-2000, 0, n_samples).astype(float)
    days_last_late[cnt_late == 0] = np.nan

    # risk score → TARGET с долей дефолтов ~8.5%
    risk = (
        -1.8 * np.nan_to_num(ext2, nan=0.5)
        - 1.2 * np.nan_to_num(ext3, nan=0.5)
        - 0.6 * np.nan_to_num(ext1, nan=0.4)
        + 0.000015 * credit
        - 0.000008 * income
        + 0.4 * (cnt_late / (cnt_prev + 1))
        + 0.00004 * (-age_days)
        + 0.3 * (days_employed > 0).astype(float)
        + rng.normal(0, 0.7, n_samples)
    )
    threshold = np.percentile(risk, 91.5)
    target = (risk > threshold).astype(int)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n_samples),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": contract,
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n_samples, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n_samples, p=[0.69, 0.31]),
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income.round(0),
            "AMT_CREDIT": credit.round(0),
            "AMT_ANNUITY": annuity.round(0),
            "AMT_GOODS_PRICE": (credit * rng.uniform(0.85, 1.05, n_samples)).round(0),
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": housing,
            "REGION_POPULATION_RELATIVE": rng.uniform(0.0005, 0.07, n_samples).round(6),
            "DAYS_BIRTH": age_days,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": rng.randint(-20000, 0, n_samples),
            "DAYS_ID_PUBLISH": rng.randint(-7000, 0, n_samples),
            "OWN_CAR_AGE": np.where(
                rng.rand(n_samples) < 0.66, np.nan, rng.randint(0, 40, n_samples)
            ),
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": rng.binomial(1, 0.82, n_samples),
            "FLAG_WORK_PHONE": rng.binomial(1, 0.28, n_samples),
            "FLAG_CONT_MOBILE": rng.binomial(1, 0.998, n_samples),
            "FLAG_PHONE": rng.binomial(1, 0.28, n_samples),
            "FLAG_EMAIL": rng.binomial(1, 0.06, n_samples),
            "OCCUPATION_TYPE": occupation,
            "CNT_FAM_MEMBERS": family,
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n_samples, p=[0.1, 0.75, 0.15]),
            "REGION_RATING_CLIENT_W_CITY": rng.choice(
                [1, 2, 3], n_samples, p=[0.11, 0.74, 0.15]
            ),
            "WEEKDAY_APPR_PROCESS_START": rng.choice(
                ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
                n_samples,
            ),
            "HOUR_APPR_PROCESS_START": rng.randint(0, 24, n_samples),
            "REG_REGION_NOT_LIVE_REGION": rng.binomial(1, 0.015, n_samples),
            "REG_REGION_NOT_WORK_REGION": rng.binomial(1, 0.05, n_samples),
            "LIVE_REGION_NOT_WORK_REGION": rng.binomial(1, 0.04, n_samples),
            "REG_CITY_NOT_LIVE_CITY": rng.binomial(1, 0.08, n_samples),
            "REG_CITY_NOT_WORK_CITY": rng.binomial(1, 0.23, n_samples),
            "LIVE_CITY_NOT_WORK_CITY": rng.binomial(1, 0.18, n_samples),
            "ORGANIZATION_TYPE": org_type,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(1.4, n_samples).clip(0, 30),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.poisson(0.15, n_samples).clip(0, 10),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.poisson(1.4, n_samples).clip(0, 30),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.poisson(0.1, n_samples).clip(0, 8),
            "DAYS_LAST_PHONE_CHANGE": rng.randint(-4000, 0, n_samples),
            "CNT_PREV_CREDITS": cnt_prev,
            "CNT_LATE_PAYMENTS": cnt_late,
            "DAYS_SINCE_LAST_LATE": days_last_late,
        }
    )
    return df


def load_data(
    path: str | Path | None = None,
    n_samples: int = 28000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Если path указан и файл существует — читаем CSV.
    Иначе генерируем синтетику (удобно для демо и CI).
    """
    if path is not None:
        path = Path(path)
        if path.exists():
            df = pd.read_csv(path)
            # TODO: базовая валидация схемы (наличие TARGET, ключевых колонок)
            return df

    return generate_home_credit_like(n_samples=n_samples, random_state=random_state)
