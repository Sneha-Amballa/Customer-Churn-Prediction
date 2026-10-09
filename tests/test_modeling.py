"""
Unit tests for Customer Churn Intelligence Phase 5 Modeling & Evaluation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    validate_dataset,
    get_feature_lists,
    create_train_test_split,
    build_preprocessor,
)
from src.evaluate import compute_classification_metrics


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib")


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


def test_validate_dataset(df):
    summary = validate_dataset(df)
    assert summary["num_rows"] == 7043
    assert summary["missing_values"] == 0
    assert 0 in summary["target_counts"]
    assert 1 in summary["target_counts"]
    # Verify churn rate is ~26.5%
    assert 0.25 < summary["target_proportions"][1] < 0.28


def test_train_test_split_stratification(df):
    X_train, X_test, y_train, y_test = create_train_test_split(df, test_size=0.20, random_state=42)
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_test) == 1409
    assert len(X_train) == 5634
    
    # Check stratification
    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.005
    assert abs(train_churn_rate - 0.2654) < 0.005


def test_preprocessing_pipeline_fit_transform(df):
    num_cols, cat_cols = get_feature_lists(df)
    X_train, X_test, y_train, y_test = create_train_test_split(df, test_size=0.20, random_state=42)
    
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Converted shape checks
    assert X_train_trans.shape[0] == len(X_train)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    # No NaNs generated
    assert not np.isnan(np.array(X_train_trans)).any()
    assert not np.isnan(np.array(X_test_trans)).any()


def test_metrics_calculation():
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.6, 0.8, 0.9])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["true_positives"] == 3
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 3


def test_saved_model_artifact_inference(df):
    assert os.path.exists(MODEL_PATH), f"Model artifact not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    
    X = df.drop(columns=["Churn"]).head(10)
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)
