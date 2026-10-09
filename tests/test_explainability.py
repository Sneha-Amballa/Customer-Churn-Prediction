"""
Unit tests for Customer Churn Intelligence Phase 7: SHAP Explainability.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.explainability import ChurnExplainer


DATA_PATH = "data/processed/telco_churn_features.csv"
MODEL_PATH = "models/best_model_pipeline.joblib"


@pytest.fixture
def explainer():
    return ChurnExplainer(MODEL_PATH)


@pytest.fixture
def sample_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["Churn"]).head(15)


def test_explainer_initialization(explainer):
    assert explainer.pipeline is not None
    assert explainer.preprocessor is not None
    assert explainer.classifier is not None
    assert explainer.explainer is not None

    feature_names = explainer.get_feature_names()
    assert len(feature_names) == 51
    assert "tenure" in feature_names
    assert "MonthlyCharges" in feature_names


def test_shap_output_dimensions_and_class_alignment(explainer, sample_data):
    exp = explainer.compute_shap_values(sample_data)
    
    # 2D explanation targeting positive class Churn = 1
    assert len(exp.shape) == 2
    assert exp.shape[0] == len(sample_data)
    assert exp.shape[1] == 51
    # Base value is scalar ~ 0.50 (RF balanced baseline)
    assert 0.0 <= exp.base_values[0] <= 1.0


def test_shap_exact_additivity(explainer, sample_data):
    # Verify: base_value + sum(shap_values) == model predict_proba (class 1)
    probs = explainer.pipeline.predict_proba(sample_data)[:, 1]
    exp = explainer.compute_shap_values(sample_data)

    shap_sums = exp.values.sum(axis=1) + exp.base_values
    max_discrepancy = np.max(np.abs(probs - shap_sums))

    # Should match to floating point precision
    assert max_discrepancy < 1e-4


def test_explain_customer_formats(explainer, sample_data):
    # 1. Test 1-row DataFrame
    cust_df = sample_data.iloc[[0]]
    res_df = explainer.explain_customer(cust_df)
    assert "predicted_churn_probability" in res_df
    assert "top_risk_drivers" in res_df
    assert "top_retention_drivers" in res_df
    assert res_df["additivity_verified"] is True

    # 2. Test pd.Series
    cust_series = sample_data.iloc[0]
    res_series = explainer.explain_customer(cust_series)
    assert res_series["predicted_churn_probability"] == res_df["predicted_churn_probability"]

    # 3. Test dictionary
    cust_dict = cust_series.to_dict()
    res_dict = explainer.explain_customer(cust_dict)
    assert res_dict["predicted_churn_probability"] == res_df["predicted_churn_probability"]


def test_robustness_unknown_category_and_missing_values(explainer, sample_data):
    # Create record with unknown category and missing numerical value
    corrupt_record = sample_data.iloc[0].to_dict()
    corrupt_record["PaymentMethod"] = "CompletelyUnknownMethod123"
    corrupt_record["MonthlyCharges"] = np.nan

    # Preprocessing pipeline should handle this via handle_unknown='ignore' and SimpleImputer
    res = explainer.explain_customer(corrupt_record)
    assert 0.0 <= res["predicted_churn_probability"] <= 1.0
    assert res["additivity_verified"] is True


def test_generated_artifacts_exist():
    expected_artifacts = [
        "models/shap_summary.png",
        "models/shap_feature_importance.png",
        "models/shap_local_example.png",
    ]
    for art in expected_artifacts:
        assert os.path.exists(art), f"Artifact missing: {art}"
        assert os.path.getsize(art) > 5000, f"Artifact empty: {art}"
