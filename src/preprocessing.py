"""
Preprocessing module for Customer Churn Intelligence.

Provides reusable scikit-learn transformers, feature classification,
and leak-free train/test splitting routines.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


TARGET_COLUMN = "Churn"


def validate_dataset(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict:
    """
    Validates dataset integrity before modeling.

    Returns a summary dictionary with shape, missing values, duplicates,
    and target class distribution.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_dtype": str(df[target_col].dtype),
        "target_unique_values": df[target_col].unique().tolist(),
        "target_counts": df[target_col].value_counts().to_dict(),
        "target_proportions": df[target_col].value_counts(normalize=True).to_dict(),
    }
    return summary


def get_feature_lists(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[List[str], List[str]]:
    """
    Extracts numerical and categorical feature names, strictly excluding the target.

    Args:
        df: DataFrame containing features and target.
        target_col: Name of target column to exclude.

    Returns:
        (numerical_features, categorical_features)
    """
    features = [c for c in df.columns if c != target_col]
    
    # Identify numerical columns (exclude target)
    num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    # Identify categorical / string columns
    cat_cols = df[features].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    return num_cols, cat_cols


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits DataFrame into stratified train and test sets to maintain
    class balance and prevent data leakage.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles numerical scaling and
    categorical one-hot encoding with imputer fallbacks to ensure robust inference.

    Args:
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.

    Returns:
        Fitted or unfitted ColumnTransformer instance.
    """
    # Numerical pipeline: impute median if any missing, then standardize
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute mode if any missing, then one-hot encode
    # drop='first' avoids multicollinearity; handle_unknown='ignore' prevents errors on unseen test categories
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Enable pandas output so feature names are preserved for LightGBM/XGBoost
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

    return preprocessor


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str],
) -> List[str]:
    """
    Extracts feature names produced by the preprocessor after fitting.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out is unavailable
        cat_ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cat_encoded_names = list(cat_ohe.get_feature_names_out(cat_cols))
        return list(num_cols) + cat_encoded_names
