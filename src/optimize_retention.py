"""
Execution script for Phase 6: Business Impact Analysis & Retention Optimization.

Loads the saved best model pipeline, optimizes the retention targeting threshold
strictly on training data, evaluates on the holdout test set, and exports artifacts.
"""

import sys
import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import create_train_test_split
from src.business_impact import (
    BusinessAssumptions,
    RetentionSimulator,
    plot_profit_curve,
    plot_cumulative_gains,
)


def run_business_impact_analysis(
    data_path: str = "data/processed/telco_churn_features.csv",
    model_path: str = "models/best_model_pipeline.joblib",
    output_dir: str = "models",
    offer_cost: float = 50.0,
    customer_value: float = 500.0,
    retention_rate: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Executes the full business impact simulation and threshold optimization.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65, flush=True)
    print("CUSTOMER CHURN INTELLIGENCE -- PHASE 6: BUSINESS IMPACT & ROI", flush=True)
    print("=" * 65, flush=True)

    # 1. Load Data and Best Pipeline
    print(f"\n[1/5] Loading Data and Pipeline...", flush=True)
    df = pd.read_csv(data_path)
    pipeline = joblib.load(model_path)
    print(f"  * Dataset loaded: {len(df):,} rows", flush=True)
    print(f"  * Pipeline loaded: {type(pipeline.named_steps['classifier']).__name__}", flush=True)

    # 2. Stratified 80/20 Train/Test Split (Identical to Phase 5)
    print(f"\n[2/5] Creating Stratified 80/20 Split (Seed={random_state})...", flush=True)
    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        target_col="Churn",
        test_size=0.20,
        random_state=random_state,
    )
    print(f"  * Train rows: {len(X_train):,} (Churners: {int(y_train.sum())})", flush=True)
    print(f"  * Test rows:  {len(X_test):,} (Churners: {int(y_test.sum())})", flush=True)

    # 3. Setup Business Assumptions & Predict Probabilities
    assumptions = BusinessAssumptions(
        offer_cost=offer_cost,
        customer_value=customer_value,
        retention_rate=retention_rate,
        use_dynamic_value=False,
    )
    simulator = RetentionSimulator(assumptions=assumptions, pipeline=pipeline)

    print(f"\n[3/5] Business Economics Parameters:", flush=True)
    print(f"  * Offer Cost per Customer (c):     ${assumptions.offer_cost:.2f}", flush=True)
    print(f"  * Customer Value Preserved (V):    ${assumptions.customer_value:.2f}", flush=True)
    print(f"  * Retention Offer Success Rate (r): {assumptions.retention_rate:.1%}", flush=True)
    breakeven_p = assumptions.offer_cost / (assumptions.retention_rate * assumptions.customer_value)
    print(f"  * Theoretical Breakeven Threshold:  p* = c / (r * V) = {breakeven_p:.4f}", flush=True)

    # Predict probabilities for train and test
    print("\n  Generating probabilities...", flush=True)
    probs_train = simulator.predict_churn_probabilities(X_train)
    probs_test = simulator.predict_churn_probabilities(X_test)

    # 4. Threshold Optimization on Training Data Only (Zero Test Leakage)
    print(f"\n[4/5] Optimizing Decision Threshold on Training Set Only...", flush=True)
    opt_result = simulator.optimize_threshold(
        df_train=X_train,
        y_train=y_train.values,
        probabilities_train=probs_train,
        assumptions=assumptions,
    )
    optimal_threshold = opt_result["optimal_threshold"]
    max_train_profit = opt_result["max_net_financial_impact"]
    print(f"  [*] Optimal Threshold (Train): p* = {optimal_threshold:.2f}", flush=True)
    print(f"  [*] Max Net Financial Impact (Train): ${max_train_profit:,.2f}", flush=True)

    # 5. Evaluate Targeting Strategies on Holdout Test Set
    print(f"\n[5/5] Evaluating Targeting Strategies on Holdout Test Set (N={len(X_test):,})...", flush=True)
    df_comparison = simulator.compare_strategies(
        df=X_test,
        y_true=y_test.values,
        probabilities=probs_test,
        optimal_threshold=optimal_threshold,
        assumptions=assumptions,
    )
    print("\n--- TEST SET STRATEGY BENCHMARK ---", flush=True)
    print(df_comparison.to_string(index=False), flush=True)

    # 6. Save Artifacts & Visualizations
    print(f"\nSaving Visualizations and Results to '{output_dir}/'...", flush=True)
    profit_curve_path = os.path.join(output_dir, "profit_curve.png")
    plot_profit_curve(
        optimization_curve_df=opt_result["curve_data"],
        optimal_threshold=optimal_threshold,
        save_path=profit_curve_path,
    )
    print(f"  * Saved: {profit_curve_path}", flush=True)

    gains_curve_path = os.path.join(output_dir, "cumulative_gains_curve.png")
    plot_cumulative_gains(
        y_true=y_test.values,
        probabilities=probs_test,
        save_path=gains_curve_path,
    )
    print(f"  * Saved: {gains_curve_path}", flush=True)

    csv_path = os.path.join(output_dir, "targeting_strategies_comparison.csv")
    df_comparison.to_csv(csv_path, index=False)
    print(f"  * Saved: {csv_path}", flush=True)

    # Save detailed JSON summary
    json_summary = {
        "assumptions": {
            "offer_cost": assumptions.offer_cost,
            "customer_value": assumptions.customer_value,
            "retention_rate": assumptions.retention_rate,
            "theoretical_breakeven_threshold": round(breakeven_p, 4),
        },
        "train_optimization": {
            "optimal_threshold": optimal_threshold,
            "max_net_financial_impact_train": max_train_profit,
        },
        "test_evaluation": df_comparison.to_dict(orient="records"),
    }
    json_path = os.path.join(output_dir, "business_simulation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4)
    print(f"  * Saved: {json_path}", flush=True)

    print("\n[OK] Phase 6 Business Impact Analysis Completed Successfully!", flush=True)
    return df_comparison


if __name__ == "__main__":
    run_business_impact_analysis()
