"""
Dashboard helper utilities, caching, and visualization functions
for Customer Churn Intelligence Streamlit application.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.feature_engineering import raw_dict_to_model_features
from src.business_impact import BusinessAssumptions, RetentionSimulator
from src.explainability import ChurnExplainer


DATA_FEATURES_PATH = "data/processed/telco_churn_features.csv"
DATA_RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/best_model_pipeline.joblib"
SIM_RESULTS_PATH = "models/business_simulation_results.json"
STRATEGIES_PATH = "models/targeting_strategies_comparison.csv"


def load_dataset() -> pd.DataFrame:
    """Loads features dataset and joins customerID from raw dataset if available."""
    df = pd.read_csv(DATA_FEATURES_PATH)
    if os.path.exists(DATA_RAW_PATH):
        try:
            df_raw = pd.read_csv(DATA_RAW_PATH, usecols=["customerID"])
            if len(df_raw) == len(df):
                df["customerID"] = df_raw["customerID"]
        except Exception:
            df["customerID"] = [f"CUST-{i:05d}" for i in range(len(df))]
    else:
        df["customerID"] = [f"CUST-{i:05d}" for i in range(len(df))]
    return df


def load_pipeline() -> Any:
    """Loads the fitted model pipeline."""
    return joblib.load(MODEL_PATH)


def load_explainer(pipeline: Optional[Any] = None) -> ChurnExplainer:
    """Loads the ChurnExplainer instance."""
    pipe = pipeline or load_pipeline()
    return ChurnExplainer(pipe)


def load_simulation_data() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Loads pre-computed Phase 6 business simulation results and benchmark table."""
    with open(SIM_RESULTS_PATH, "r", encoding="utf-8") as f:
        sim_json = json.load(f)
    strategies_df = pd.read_csv(STRATEGIES_PATH)
    return sim_json, strategies_df


def filter_dataset(
    df: pd.DataFrame,
    contracts: List[str],
    internet_services: List[str],
    payment_methods: List[str],
    tenure_range: Tuple[int, int],
) -> pd.DataFrame:
    """Filters dataset by categorical attributes and tenure range."""
    filtered = df[
        (df["Contract"].isin(contracts)) &
        (df["InternetService"].isin(internet_services)) &
        (df["PaymentMethod"].isin(payment_methods)) &
        (df["tenure"] >= tenure_range[0]) &
        (df["tenure"] <= tenure_range[1])
    ]
    return filtered


def get_risk_tier(prob: float, threshold: float = 0.65) -> Tuple[str, str, str]:
    """
    Returns risk category, CSS color, and badge label.
    """
    if prob >= threshold:
        return "High Risk", "#dc3545", "HIGH CHURN RISK"
    elif prob >= 0.35:
        return "Medium Risk", "#fd7e14", "MODERATE RISK"
    else:
        return "Low Risk", "#198754", "LOW RISK / STABLE"


def predict_and_explain(
    raw_customer_dict: Dict[str, Any],
    pipeline: Any,
    explainer: ChurnExplainer,
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Runs end-to-end inference on a single customer dictionary:
    1. Feature engineering
    2. Model prediction
    3. SHAP local attribution
    """
    df_features = raw_dict_to_model_features(raw_customer_dict)
    churn_prob = float(pipeline.predict_proba(df_features)[0, 1])
    risk_tier, risk_color, risk_badge = get_risk_tier(churn_prob, threshold=threshold)

    local_exp = explainer.explain_customer(df_features, top_n=5)

    return {
        "churn_probability": churn_prob,
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "risk_badge": risk_badge,
        "base_value": local_exp["base_value"],
        "top_risk_drivers": local_exp["top_risk_drivers"],
        "top_retention_drivers": local_exp["top_retention_drivers"],
        "additivity_verified": local_exp["additivity_verified"],
        "engineered_features": df_features.to_dict(orient="records")[0],
        "explanation_object": local_exp["explanation_object"],
    }


# =====================================================================
# PLOTLY CHART BUILDERS
# =====================================================================

def plot_churn_by_feature(df: pd.DataFrame, feature_col: str, title: str) -> go.Figure:
    """Generates an interactive dual-bar / churn-rate chart for a categorical feature."""
    grouped = (
        df.groupby(feature_col)["Churn"]
        .agg(Total="count", Churners="sum", ChurnRate="mean")
        .reset_index()
    )
    grouped["NonChurners"] = grouped["Total"] - grouped["Churners"]
    grouped["ChurnRatePct"] = (grouped["ChurnRate"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped[feature_col],
            y=grouped["NonChurners"],
            name="Retained (0)",
            marker_color="#2b5c8f",
        )
    )
    fig.add_trace(
        go.Bar(
            x=grouped[feature_col],
            y=grouped["Churners"],
            name="Churned (1)",
            marker_color="#d9534f",
            text=grouped["ChurnRatePct"].apply(lambda v: f"{v}% churn"),
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        barmode="stack",
        xaxis_title=feature_col,
        yaxis_title="Customer Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    """Interactive histogram of customer tenure broken down by Churn."""
    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        barmode="overlay",
        nbins=36,
        color_discrete_map={0: "#2b5c8f", 1: "#d9534f"},
        labels={"tenure": "Tenure (Months)", "count": "Customer Count", "Churn": "Status"},
        title="<b>Tenure Distribution & Churn Concentration</b>",
    )
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_service_adoption_churn(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing churn rate by total number of active services."""
    grouped = (
        df.groupby("service_count")["Churn"]
        .agg(Customers="count", ChurnRate="mean")
        .reset_index()
    )
    grouped["ChurnRatePct"] = (grouped["ChurnRate"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grouped["service_count"],
            y=grouped["ChurnRatePct"],
            marker_color="#17a2b8",
            text=grouped["ChurnRatePct"].apply(lambda v: f"{v}%"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="<b>Churn Rate vs. Active Services Subscribed (0 to 8)</b>",
        xaxis_title="Active Service Count",
        yaxis_title="Observed Churn Rate (%)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    return fig


def plot_interactive_waterfall(local_result: Dict[str, Any], max_display: int = 7) -> go.Figure:
    """
    Constructs an interactive Plotly waterfall chart visualizing
    the transition from baseline probability to predicted churn probability.
    """
    base = local_result["base_value"]
    top_pos = local_result["top_risk_drivers"][:max_display]
    top_neg = local_result["top_retention_drivers"][:max_display]

    labels = ["Base Rate E[p]"]
    values = [base]
    measures = ["absolute"]

    # Add positive contributors
    for d in top_pos:
        feat_name = d["feature"].replace("contract_tenure_group_", "CT: ").replace("InternetService_", "Net: ")
        labels.append(f"+ {feat_name}")
        values.append(d["impact"])
        measures.append("relative")

    # Add negative contributors
    for d in top_neg:
        feat_name = d["feature"].replace("contract_tenure_group_", "CT: ").replace("InternetService_", "Net: ")
        labels.append(f"- {feat_name}")
        values.append(d["impact"])
        measures.append("relative")

    labels.append("Predicted Churn Risk")
    values.append(local_result["churn_probability"])
    measures.append("total")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            textposition="outside",
            text=[f"{v:+.3f}" if m == "relative" else f"{v:.1%}" for v, m in zip(values, measures)],
            connector={"line": {"color": "#6c757d"}},
            decreasing={"marker": {"color": "#28a745"}},
            increasing={"marker": {"color": "#dc3545"}},
            totals={"marker": {"color": "#1b6ca8"}},
        )
    )

    fig.update_layout(
        title="<b>Customer SHAP Probability Waterfall</b> (Additivity: Base + Σ SHAP = Pred)",
        showlegend=False,
        template="plotly_white",
        yaxis=dict(title="Churn Probability", tickformat=".0%"),
        margin=dict(l=40, r=40, t=60, b=80),
        height=450,
    )
    return fig


def plot_interactive_profit_curve(
    thresholds: np.ndarray,
    net_profits: np.ndarray,
    targeted_counts: np.ndarray,
    optimal_thresh: float,
) -> go.Figure:
    """Plots interactive dual-axis profit vs threshold curve with optimal marker."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=net_profits,
            mode="lines",
            name="Net Financial Impact ($)",
            line=dict(color="#1b6ca8", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=targeted_counts,
            mode="lines",
            name="Targeted Customers",
            yaxis="y2",
            line=dict(color="#e65c00", dash="dot", width=2),
        )
    )

    # Highlight optimal threshold
    fig.add_vline(
        x=optimal_thresh,
        line_width=2,
        line_dash="dash",
        line_color="#28a745",
        annotation_text=f"Optimal p* = {optimal_thresh:.2f}",
        annotation_position="top left",
    )

    fig.update_layout(
        title="<b>Net Financial Impact & Campaign Reach vs. Decision Threshold</b>",
        xaxis=dict(title="Decision Threshold (p)"),
        yaxis=dict(title="Net Financial Impact ($)", title_font=dict(color="#1b6ca8")),
        yaxis2=dict(
            title="Customers Targeted (Count)",
            title_font=dict(color="#e65c00"),
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=50, r=50, t=60, b=40),
        height=420,
    )
    return fig
