"""
Feature engineering module for Customer Churn Intelligence.

Transforms raw/cleaned customer records into the 27 engineered features
expected by the trained ML pipeline.
"""

from typing import Union, Dict, Any
import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

SUPPORT_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["New", "Early", "Established", "Loyal"]


def categorize_monthly_charge(charge: float) -> str:
    """Categorizes monthly charges into Low, Medium, or High tier."""
    if charge < 35.0:
        return "Low"
    elif charge <= 70.0:
        return "Medium"
    else:
        return "High"


def engineer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all behavioral and structural features from raw customer records.

    Args:
        input_df: DataFrame with raw Telco attributes.

    Returns:
        DataFrame with 27 model-ready features.
    """
    df = input_df.copy()

    # Ensure required raw columns exist, filling defaults if missing
    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            if col in ["SeniorCitizen", "tenure"]:
                df[col] = 0
            elif col in ["MonthlyCharges", "TotalCharges"]:
                df[col] = 0.0
            else:
                df[col] = "No"

    # Type coercion
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # If TotalCharges is 0 and tenure > 0, provide sensible fallback
    zero_tc_mask = (df["TotalCharges"] == 0) & (df["tenure"] > 0)
    df.loc[zero_tc_mask, "TotalCharges"] = df.loc[zero_tc_mask, "MonthlyCharges"] * df.loc[zero_tc_mask, "tenure"]

    # 1. tenure_group
    # Clip tenure to avoid out-of-bin errors
    safe_tenure = df["tenure"].clip(lower=0, upper=72)
    df["tenure_group"] = pd.cut(safe_tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    # 2. service_count
    df["service_count"] = 0
    for col in SERVICE_COLUMNS:
        df["service_count"] += (df[col] == "Yes").astype(int)

    # 3. avg_monthly_revenue (safely handle tenure == 0)
    tenure_nonzero = np.where(df["tenure"] == 0, 1, df["tenure"])
    df["avg_monthly_revenue"] = np.where(
        df["tenure"] == 0,
        df["MonthlyCharges"],
        df["TotalCharges"] / tenure_nonzero,
    ).astype(float)

    # 4. monthly_charge_diff
    df["monthly_charge_diff"] = (df["MonthlyCharges"] - df["avg_monthly_revenue"]).round(2).astype(float)

    # 5. streaming_count
    df["streaming_count"] = (
        (df["StreamingTV"] == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )

    # 6. support_security_count
    df["support_security_count"] = 0
    for col in SUPPORT_COLUMNS:
        df["support_security_count"] += (df[col] == "Yes").astype(int)

    # 7. contract_tenure_group
    df["contract_tenure_group"] = df["Contract"].astype(str) + "_" + df["tenure_group"].astype(str)

    # 8. monthly_charges_tier
    df["monthly_charges_tier"] = df["MonthlyCharges"].apply(categorize_monthly_charge).astype(str)

    # Feature ordering expected by model
    model_feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "tenure_group", "service_count",
        "avg_monthly_revenue", "monthly_charge_diff", "streaming_count",
        "support_security_count", "contract_tenure_group", "monthly_charges_tier",
    ]

    return df[model_feature_cols]


def raw_dict_to_model_features(raw_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a single customer dictionary into a 1-row model-ready DataFrame.
    """
    df_single = pd.DataFrame([raw_dict])
    return engineer_features(df_single)
