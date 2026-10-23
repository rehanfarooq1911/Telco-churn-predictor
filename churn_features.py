"""Shared feature definitions so training and the app can never drift apart."""
import pandas as pd

# Lower edge is -1 so that tenure == 0 lands in the first group instead of becoming NaN.
TENURE_BINS = [-1, 12, 24, 48, 60, 100]
TENURE_LABELS = ["0-1 yr", "1-2 yrs", "2-4 yrs", "4-5 yrs", "5+ yrs"]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "TenureGroup",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a TenureGroup column (as plain strings)."""
    out = df.copy()
    out["TenureGroup"] = pd.cut(
        out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)
    return out


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the Telco CSV, fix TotalCharges, drop unusable rows, add features."""
    df = pd.read_csv(path)
    # TotalCharges contains blank strings for brand-new customers (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return add_tenure_group(df)
