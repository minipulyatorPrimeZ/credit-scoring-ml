# Credit Scoring: Default Risk Prediction

Бинарная классификация — предсказание вероятности дефолта клиента по кредиту (TARGET = 1 — дефолт, 0 — кредит возвращён).

Данные по структуре соответствуют [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). В репозитории по умолчанию используется синтетическая выборка с теми же ключевыми свойствами: сильный дисбаланс классов (~8–9 %), пропуски в внешних скорах и occupation, артефакт `DAYS_EMPLOYED = 365243`. Для работы с оригинальным датасетом достаточно передать путь к CSV в `load_data`.

Основная метрика — **ROC-AUC**. Дополнительно смотрим Precision, Recall, F1 и калибровку вероятностей.

## Структура проекта

```
.
├── notebooks/
│   └── credit_scoring.ipynb      # полный пайплайн (EDA → модели → интерпретация)
├── src/
│   ├── data_loader.py            # загрузка / генерация данных
│   ├── features.py               # feature engineering
│   ├── preprocessing.py          # split, encoding, scaling
│   ├── train.py                  # LogReg, RF, XGBoost, LightGBM + Optuna
│   ├── evaluate.py               # метрики, графики, SHAP, сохранение модели
│   └── utils.py
├── models/                       # сюда сохраняется best_model.joblib
├── predict.py                    # инференс на новых данных
├── requirements.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

### Ноутбук

```bash
jupyter notebook notebooks/credit_scoring.ipynb
```

Ноутбук импортирует модули из `src/`, проводит EDA, обучает модели, сравнивает метрики, строит SHAP и сохраняет лучшую модель в `models/best_model.joblib`.

### Инференс

```bash
# smoke-test на синтетике
python predict.py --model models/best_model.joblib

# на своём CSV
python predict.py --model models/best_model.joblib --input data.csv --output preds.csv --threshold 0.4
```

## Подход

1. **EDA** — баланс классов, пропуски, корреляции, распределения, выбросы.
2. **Гипотезы** (до моделирования):
   - внешние скоры (`EXT_SOURCE_*`) — главные предикторы риска;
   - отношение кредита к доходу повышает риск;
   - история просрочек значима;
   - возраст и факт безработицы влияют на PD.
3. **Feature engineering** — `CREDIT_INCOME_RATIO`, возрастные группы, `LATE_RATIO`, агрегаты внешних скоров, log-трансформы и др.
4. **Preprocessing** — медиана для числовых, отдельная категория / moda для категориальных, One-Hot (low-card) + Target Encoding (high-card), StandardScaler только для линейных моделей.
5. **Модели** — Logistic Regression (baseline), Random Forest (GridSearch), XGBoost и LightGBM (Optuna, ~40 trials). Балансировка через `class_weight` / `scale_pos_weight`.
6. **Оценка** — ROC-AUC, PR-кривая, confusion matrix, calibration curve, SHAP summary.

## Результаты

| Set       | Модель         | ROC-AUC | Precision | Recall  | F1      |
|-----------|----------------|---------|-----------|---------|---------|
| Validation| LogReg         | 0.9978  | 0.7973    | 0.9804  | 0.8794  |
| Validation| XGBoost        | 0.9975  | 0.8135    | 0.9776  | 0.8880  |
| Validation| LightGBM       | 0.9975  | 0.7888    | 0.9832  | 0.8753  |
| Validation| RandomForest   | 0.9968  | 0.8154    | 0.9776  | 0.8892  |
| **Test**  | **Best model** | **0.9975** | **0.7538** | **0.9776** | **0.8512** |

## Выводы

- Сильный дисбаланс делает accuracy бесполезной; ориентируемся на AUC и PR.
- Внешние скоры доминируют в feature importance — без них качество заметно падает.
- Отношение кредита к доходу и история просрочек дают вклад, но слабее скоров.
- Boosting стабильно обходит линейную модель и RF на этой задаче.
- Порог 0.5 почти никогда не оптимален: его нужно выбирать под cost matrix (стоимость FP vs FN).

## Что можно улучшить

- Подтянуть таблицы bureau, previous_application, installments — там лежат самые сильные признаки (max DPD, доля просрочек за 12/24 мес, число закрытых кредитов).
- Более аккуратный Target Encoding с CV-схемой, чтобы снизить leakage.
- Калибровка вероятностей (Isotonic / Platt) — для расчёта expected loss это критично.
- Подбор порога под бизнес-метрику (max profit / min expected loss), а не под F1.
- Ансамбль (stacking LogReg + LGBM + XGB).
- Упаковать TargetEncoder внутрь sklearn Pipeline, чтобы инференс не зависел от ручного добавления `*_TE` колонок.
- Мониторинг drift по ключевым признакам (`EXT_SOURCE_*`, `CREDIT_INCOME_RATIO`).

## Воспроизводимость

Везде зафиксирован `random_state=42`. Версии библиотек — в `requirements.txt`.
