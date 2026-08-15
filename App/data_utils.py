"""
data_utils.py
--------------
Shared load + clean + encode logic for the Bank Customer Churn project.
Mirrors Project_Notebook.ipynb step-for-step (same order, same imputation
rules, same de-dup, same one-hot encoding) so the deployed app and the
notebook never disagree on what "cleaned data" means.

Steps (in the exact order used in the notebook):
  1. Drop CustomerId, Surname (identifiers, not predictive)
  2. Impute: Geography/Gender -> mode, Age -> median,
     HasCrCard/IsActiveMember -> mode (then cast to int)
  3. Drop duplicate rows
  4. One-hot encode Geography & Gender (drop_first=True)
  5. Scale ONLY the 6 numeric columns with the saved StandardScaler
     (HasCrCard, IsActiveMember, and the one-hot columns are left as 0/1 —
     this matches how the model was trained, see feature_columns in the .pkl)

NOTE: an earlier version of the notebook's imputation used
`df[col].fillna(x, inplace=True)`, which is a no-op under pandas'
copy-on-write behaviour (pandas >= 2.x) — it silently fails to update the
original DataFrame. This module uses `df[col] = df[col].fillna(x)` instead,
which is the safe, version-proof way to do the same thing.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../App
REPO_ROOT = os.path.dirname(BASE_DIR)                           # repo root

NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CAT_COLS = ["Geography", "Gender"]
DROP_COLS = ["CustomerId", "Surname"]
TARGET = "Exited"

_CSV_CANDIDATE_NAMES = ["Data.csv", "Churn_Modelling.csv", "churn_data.csv"]
_CANDIDATE_DIRS = [
    BASE_DIR,
    REPO_ROOT,
    os.path.join(REPO_ROOT, "Data"),
    os.path.join(BASE_DIR, "Data"),
    os.getcwd(),
    os.path.join(os.getcwd(), "Data"),
]


def _find_csv() -> str:
    checked = []
    for d in _CANDIDATE_DIRS:
        for name in _CSV_CANDIDATE_NAMES:
            candidate = os.path.join(d, name)
            checked.append(candidate)
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(
        "Could not find the churn dataset CSV. Looked for "
        f"{_CSV_CANDIDATE_NAMES} in:\n  - " + "\n  - ".join(checked)
    )


def load_raw() -> pd.DataFrame:
    """Load the raw churn CSV exactly as-is."""
    df = pd.read_csv(_find_csv())
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's cleaning steps, in order."""
    df = df.copy()

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df["Geography"] = df["Geography"].fillna(df["Geography"].mode()[0])
    df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["HasCrCard"] = df["HasCrCard"].fillna(df["HasCrCard"].mode()[0]).astype(int)
    df["IsActiveMember"] = df["IsActiveMember"].fillna(df["IsActiveMember"].mode()[0]).astype(int)

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def encode_and_scale(df_clean: pd.DataFrame, scaler, feature_columns: list) -> pd.DataFrame:
    """One-hot encode categoricals + scale numeric columns with the *already-fitted*
    scaler (transform only — never re-fit at inference/EDA time)."""
    df_encoded = pd.get_dummies(df_clean, columns=CAT_COLS, drop_first=True)

    # Make sure every column the model expects exists (e.g. if a category is
    # entirely absent from a filtered slice of data).
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded[NUM_COLS] = scaler.transform(df_encoded[NUM_COLS])

    ordered_cols = feature_columns + ([TARGET] if TARGET in df_encoded.columns else [])
    return df_encoded[ordered_cols]


def load_clean_encoded(scaler, feature_columns: list):
    """Convenience one-call entry point used by app.py.
    Returns (df_clean, df_encoded) — df_clean is human-readable (for EDA),
    df_encoded is model-ready (for prediction / clustering / metrics)."""
    df_raw = load_raw()
    df_clean = clean(df_raw)
    df_encoded = encode_and_scale(df_clean, scaler, feature_columns)
    return df_clean, df_encoded
