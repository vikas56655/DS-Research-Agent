"""Deterministic, reproducible data profiling and baseline modelling."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_csv(source: bytes | io.BytesIO, max_bytes: int = 15_000_000) -> pd.DataFrame:
    """Read an uploaded CSV; reject oversized, empty and unusable inputs."""
    raw = source.getvalue() if hasattr(source, "getvalue") else source
    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise ValueError("Upload a non-empty CSV smaller than 15 MB.")
    try:
        frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Could not parse this CSV. Please export a UTF-8 CSV.") from exc
    if frame.empty or not len(frame.columns):
        raise ValueError("CSV must have at least one data row and column.")
    frame.columns = [str(c) for c in frame.columns]
    return frame


def sample_data(n: int = 450, seed: int = 42) -> pd.DataFrame:
    """Synthetic customer-retention dataset; never presented as real observations."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    spend = np.maximum(10, rng.normal(75, 23, n) + tenure * 0.22).round(2)
    visits = rng.poisson(4, n)
    channel = rng.choice(["Organic", "Paid", "Referral"], n, p=[0.45, 0.35, 0.2])
    logit = -0.7 - 0.025 * tenure + 0.015 * (spend - 75) - 0.13 * visits + (channel == "Paid") * 0.65
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({"tenure_months": tenure, "monthly_spend": spend,
                         "monthly_visits": visits, "acquisition_channel": channel,
                         "churn": churn})


def profile(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": list(numeric.columns),
        "categorical_columns": list(df.select_dtypes(exclude="number").columns),
        "missing_by_column": df.isna().sum().sort_values(ascending=False),
    }


def baseline(df: pd.DataFrame, target: str, task: str) -> dict:
    """Train/test baseline. Exploratory only: not a production evaluation."""
    if target not in df.columns or task not in ("classification", "regression"):
        raise ValueError("Select a valid target and task.")
    clean = df.dropna(subset=[target]).copy()
    if len(clean) < 30:
        raise ValueError("At least 30 non-missing target rows are required.")
    X = clean.drop(columns=[target]).copy()
    X = X.drop(columns=[c for c in X if X[c].isna().all() or X[c].nunique(dropna=True) <= 1])
    if X.empty:
        raise ValueError("No informative predictors are available.")
    # Prevent pathological feature expansion from unique IDs and free-text fields.
    cat = list(X.select_dtypes(exclude="number").columns)
    high_cardinality = [c for c in cat if X[c].nunique(dropna=True) > 30]
    X = X.drop(columns=high_cardinality)
    if X.empty:
        raise ValueError("All predictors were high-cardinality or constant.")
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = list(X.select_dtypes(exclude="number").columns)
    if task == "regression":
        y = pd.to_numeric(clean[target], errors="coerce")
        valid = y.notna() & np.isfinite(y)
        X, y = X.loc[valid], y.loc[valid]
        if len(y) < 30:
            raise ValueError("Regression requires 30 numeric target values.")
        model = Ridge(alpha=1.0)
        stratify = None
    else:
        y = clean[target].astype(str)
        if y.nunique() < 2 or y.nunique() > 20:
            raise ValueError("Classification requires between 2 and 20 target classes.")
        counts = y.value_counts()
        if counts.min() < 2:
            raise ValueError("Each class needs at least two rows.")
        model = LogisticRegression(max_iter=1500, class_weight="balanced")
        stratify = y if counts.min() >= 3 else None
    if len(X) < 30:
        raise ValueError("Not enough usable observations.")
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                               ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    try:
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Model fitting failed: {exc}") from exc
    if task == "classification":
        metrics = {"Accuracy": float(accuracy_score(y_test, predictions)),
                   "Macro F1": float(f1_score(y_test, predictions, average="macro", zero_division=0))}
    else:
        metrics = {"MAE": float(mean_absolute_error(y_test, predictions)),
                   "R²": float(r2_score(y_test, predictions))}
    return {"metrics": metrics, "actual": list(y_test), "predicted": list(predictions),
            "train_rows": len(X_train), "test_rows": len(X_test),
            "dropped_high_cardinality": high_cardinality}
