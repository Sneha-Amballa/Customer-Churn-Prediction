"""
Unit tests for Customer Churn Intelligence Phase 6: Business Impact & Retention Simulation.
"""

import pytest
import numpy as np
import pandas as pd
import joblib

from src.business_impact import (
    BusinessAssumptions,
    RetentionSimulator,
)


def test_business_assumptions_validation():
    # Valid assumptions
    assumptions = BusinessAssumptions(offer_cost=50.0, customer_value=500.0, retention_rate=0.20)
    assert assumptions.offer_cost == 50.0
    assert assumptions.customer_value == 500.0
    assert assumptions.retention_rate == 0.20

    # Invalid offer_cost
    with pytest.raises(ValueError):
        BusinessAssumptions(offer_cost=-10.0)

    # Invalid customer_value
    with pytest.raises(ValueError):
        BusinessAssumptions(customer_value=0.0)

    # Invalid retention_rate
    with pytest.raises(ValueError):
        BusinessAssumptions(retention_rate=0.0)
    with pytest.raises(ValueError):
        BusinessAssumptions(retention_rate=1.5)


def test_empty_targeted_edge_case():
    simulator = RetentionSimulator()
    n = 100
    is_targeted = np.zeros(n, dtype=bool)
    probabilities = np.random.uniform(0, 1, n)
    y_true = np.random.binomial(1, 0.3, n)
    customer_values = np.full(n, 500.0)

    metrics = simulator.calculate_campaign_metrics(
        is_targeted=is_targeted,
        y_true=y_true,
        probabilities=probabilities,
        customer_values=customer_values,
    )

    assert metrics["customers_targeted"] == 0
    assert metrics["campaign_cost"] == 0.0
    assert metrics["expected_retained_customers"] == 0.0
    assert metrics["expected_value_preserved"] == 0.0
    assert metrics["net_financial_impact"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["roi_percentage"] == 0.0


def test_manual_financial_calculation():
    # Hand-crafted test case
    # 4 customers:
    # Customer 0: y=1, targeted=True
    # Customer 1: y=1, targeted=False
    # Customer 2: y=0, targeted=True
    # Customer 3: y=0, targeted=False
    assumptions = BusinessAssumptions(
        offer_cost=50.0,
        customer_value=500.0,
        retention_rate=0.20,
    )
    simulator = RetentionSimulator(assumptions=assumptions)

    is_targeted = np.array([True, False, True, False])
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.7, 0.4, 0.1])
    customer_values = np.full(4, 500.0)

    metrics = simulator.calculate_campaign_metrics(
        is_targeted=is_targeted,
        y_true=y_true,
        probabilities=probabilities,
        customer_values=customer_values,
        assumptions=assumptions,
    )

    # Targeted: 2 customers -> cost = 2 * 50 = $100
    assert metrics["customers_targeted"] == 2
    assert metrics["campaign_cost"] == 100.0

    # Actual churners targeted: 1 (Customer 0)
    assert metrics["targeted_actual_churners"] == 1
    assert metrics["targeted_non_churners"] == 1
    assert metrics["precision"] == 0.5  # 1 / 2
    assert metrics["recall"] == 0.5     # 1 / 2 total churners

    # Expected retained: 1 * 0.20 = 0.20 customers
    assert metrics["expected_retained_customers"] == 0.20

    # Expected value preserved: 0.20 * $500 = $100.0
    assert metrics["expected_value_preserved"] == 100.0

    # Net impact: $100.0 preserved - $100.0 cost = $0.0
    assert metrics["net_financial_impact"] == 0.0
    assert metrics["roi_percentage"] == 0.0


def test_dynamic_customer_value():
    assumptions = BusinessAssumptions(
        offer_cost=50.0,
        use_dynamic_value=True,
        annual_multiplier=12.0,
        contribution_margin=0.70,
    )
    df = pd.DataFrame({"MonthlyCharges": [100.0, 50.0]})
    values = assumptions.get_customer_values(df)

    # 100 * 12 * 0.7 = 840
    # 50 * 12 * 0.7 = 420
    assert np.isclose(values[0], 840.0)
    assert np.isclose(values[1], 420.0)


def test_ranking_and_strategies():
    assumptions = BusinessAssumptions(offer_cost=50.0, customer_value=500.0, retention_rate=0.20)
    simulator = RetentionSimulator(assumptions=assumptions)

    df_test = pd.DataFrame({
        "MonthlyCharges": [70.0, 80.0, 30.0, 90.0, 40.0],
    })
    probs = np.array([0.90, 0.70, 0.50, 0.30, 0.10])
    y_true = np.array([1, 1, 0, 0, 0])

    # Test top_k_percent: top 40% (2 customers)
    res_top_k = simulator.simulate_strategy(
        strategy_name="top_k_percent",
        df=df_test,
        probabilities=probs,
        y_true=y_true,
        top_k_percent=0.40,
    )
    assert res_top_k["customers_targeted"] == 2
    assert res_top_k["targeted_actual_churners"] == 2
    assert res_top_k["precision"] == 1.0

    # Test threshold: p >= 0.60 (2 customers: 0.90, 0.70)
    res_thresh = simulator.simulate_strategy(
        strategy_name="threshold",
        df=df_test,
        probabilities=probs,
        y_true=y_true,
        threshold=0.60,
    )
    assert res_thresh["customers_targeted"] == 2
    assert res_thresh["targeted_actual_churners"] == 2


def test_threshold_optimization_strictly_positive():
    assumptions = BusinessAssumptions(offer_cost=50.0, customer_value=500.0, retention_rate=0.20)
    simulator = RetentionSimulator(assumptions=assumptions)

    # Synthetic training set with predictable probabilities
    np.random.seed(42)
    n = 200
    y_train = np.random.binomial(1, 0.3, n)
    # Give higher probabilities to true churners
    probs_train = np.where(y_train == 1, np.random.uniform(0.5, 0.95, n), np.random.uniform(0.05, 0.5, n))
    df_train = pd.DataFrame({"dummy": np.zeros(n)})

    opt_res = simulator.optimize_threshold(
        df_train=df_train,
        y_train=y_train,
        probabilities_train=probs_train,
        assumptions=assumptions,
    )

    opt_thresh = opt_res["optimal_threshold"]
    max_impact = opt_res["max_net_financial_impact"]

    assert 0.05 <= opt_thresh <= 0.95
    assert max_impact > 0


def test_pipeline_integration_with_saved_artifact():
    pipeline = joblib.load("models/best_model_pipeline.joblib")
    simulator = RetentionSimulator(pipeline=pipeline)

    df = pd.read_csv("data/processed/telco_churn_features.csv").head(20)
    probs = simulator.predict_churn_probabilities(df.drop(columns=["Churn"]))

    assert len(probs) == 20
    assert np.all((probs >= 0.0) & (probs <= 1.0))
