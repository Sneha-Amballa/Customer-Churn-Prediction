"""
Unit tests for Customer Churn Intelligence Phase 8: Streamlit Dashboard & Integration.
"""

import pytest
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import engineer_features, raw_dict_to_model_features
from src.dashboard_utils import (
    load_dataset,
    load_pipeline,
    load_explainer,
    filter_dataset,
    get_risk_tier,
    predict_and_explain,
    plot_churn_by_feature,
    plot_tenure_distribution,
    plot_service_adoption_churn,
    plot_interactive_waterfall,
    plot_interactive_profit_curve,
)


@pytest.fixture
def pipeline():
    return load_pipeline()


@pytest.fixture
def explainer(pipeline):
    return load_explainer(pipeline)


@pytest.fixture
def sample_raw_record():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.50,
        "TotalCharges": 159.00,
    }


def test_feature_engineering_schema_compatibility(sample_raw_record, pipeline):
    df_feat = raw_dict_to_model_features(sample_raw_record)
    assert df_feat.shape == (1, 27)

    # Test pipeline prediction
    prob = pipeline.predict_proba(df_feat)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_zero_tenure_edge_case(sample_raw_record, pipeline):
    # Customer who just signed up (tenure = 0)
    zero_tenure_record = sample_raw_record.copy()
    zero_tenure_record["tenure"] = 0
    zero_tenure_record["TotalCharges"] = 0.0

    df_feat = raw_dict_to_model_features(zero_tenure_record)
    assert df_feat["tenure"].values[0] == 0
    # avg_monthly_revenue should fallback to MonthlyCharges without division by zero
    assert df_feat["avg_monthly_revenue"].values[0] == zero_tenure_record["MonthlyCharges"]
    assert not np.isnan(df_feat["avg_monthly_revenue"].values[0])
    assert not np.isinf(df_feat["avg_monthly_revenue"].values[0])

    prob = pipeline.predict_proba(df_feat)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_missing_raw_fields_fallback(pipeline):
    # Sparse dictionary with missing non-essential fields
    sparse_record = {
        "Contract": "Month-to-month",
        "tenure": 5,
        "MonthlyCharges": 70.0,
    }
    df_feat = raw_dict_to_model_features(sparse_record)
    assert df_feat.shape == (1, 27)
    prob = pipeline.predict_proba(df_feat)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_predict_and_explain_integration(sample_raw_record, pipeline, explainer):
    diag = predict_and_explain(sample_raw_record, pipeline, explainer, threshold=0.65)

    assert "churn_probability" in diag
    assert "risk_tier" in diag
    assert "top_risk_drivers" in diag
    assert "top_retention_drivers" in diag
    assert diag["additivity_verified"] is True
    assert len(diag["top_risk_drivers"]) > 0


def test_risk_tier_classification():
    assert get_risk_tier(0.70, threshold=0.65)[0] == "High Risk"
    assert get_risk_tier(0.50, threshold=0.65)[0] == "Medium Risk"
    assert get_risk_tier(0.20, threshold=0.65)[0] == "Low Risk"


def test_filter_dataset():
    df = load_dataset()
    filtered = filter_dataset(
        df,
        contracts=["Month-to-month"],
        internet_services=["DSL", "Fiber optic", "No"],
        payment_methods=df["PaymentMethod"].unique().tolist(),
        tenure_range=(0, 72),
    )
    assert len(filtered) > 0
    assert set(filtered["Contract"].unique()) == {"Month-to-month"}


def test_plotly_chart_builders(sample_raw_record, pipeline, explainer):
    df = load_dataset().head(50)

    # 1. Churn bar
    fig_bar = plot_churn_by_feature(df, "Contract", "Test Contract")
    assert fig_bar is not None

    # 2. Tenure histogram
    fig_hist = plot_tenure_distribution(df)
    assert fig_hist is not None

    # 3. Service adoption
    fig_srv = plot_service_adoption_churn(df)
    assert fig_srv is not None

    # 4. Interactive waterfall
    diag = predict_and_explain(sample_raw_record, pipeline, explainer)
    fig_waterfall = plot_interactive_waterfall(diag)
    assert fig_waterfall is not None

    # 5. Interactive profit curve
    threshs = np.linspace(0.1, 0.9, 10)
    profits = np.linspace(1000, 5000, 10)
    targets = np.linspace(100, 10, 10)
    fig_profit = plot_interactive_profit_curve(threshs, profits, targets, optimal_thresh=0.65)
    assert fig_profit is not None
