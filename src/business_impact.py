"""
Business Impact Analysis & Retention Campaign Simulator.

Translates machine learning churn probability predictions into financial
and operational metrics for customer retention decision-making.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BusinessAssumptions:
    """
    Configurable parameters defining retention unit economics and intervention efficacy.

    Attributes:
        offer_cost: Direct cost of extending a retention offer to one customer (e.g. discount, credit, gift).
        customer_value: Estimated lifetime value or annual contribution margin preserved per saved customer.
        retention_rate: Probability that a targeted customer who was going to churn accepts the offer and is retained.
        use_dynamic_value: Whether to compute customer-specific value from MonthlyCharges.
        annual_multiplier: Number of months of revenue preserved if using dynamic value (default 12 months).
        contribution_margin: Gross profit margin applied to revenue (default 70%).
    """
    offer_cost: float = 50.0
    customer_value: float = 500.0
    retention_rate: float = 0.20
    use_dynamic_value: bool = False
    annual_multiplier: float = 12.0
    contribution_margin: float = 0.70

    def __post_init__(self):
        if self.offer_cost < 0:
            raise ValueError(f"offer_cost must be non-negative, got {self.offer_cost}")
        if self.customer_value <= 0:
            raise ValueError(f"customer_value must be positive, got {self.customer_value}")
        if not (0.0 < self.retention_rate <= 1.0):
            raise ValueError(f"retention_rate must be in (0, 1], got {self.retention_rate}")
        if self.annual_multiplier <= 0:
            raise ValueError(f"annual_multiplier must be positive, got {self.annual_multiplier}")
        if not (0.0 < self.contribution_margin <= 1.0):
            raise ValueError(f"contribution_margin must be in (0, 1], got {self.contribution_margin}")

    def get_customer_values(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns an array of customer values (either constant or derived from MonthlyCharges).
        """
        if self.use_dynamic_value:
            if "MonthlyCharges" not in df.columns:
                raise ValueError("Column 'MonthlyCharges' required for dynamic customer value calculation.")
            return (
                df["MonthlyCharges"].values
                * self.annual_multiplier
                * self.contribution_margin
            )
        else:
            return np.full(len(df), fill_value=self.customer_value, dtype=float)


class RetentionSimulator:
    """
    Simulates customer retention campaigns under various targeting strategies
    and computes business P&L metrics.
    """

    def __init__(
        self,
        assumptions: Optional[BusinessAssumptions] = None,
        pipeline: Optional[Any] = None,
    ):
        self.assumptions = assumptions or BusinessAssumptions()
        self.pipeline = pipeline

    def predict_churn_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """
        Extracts churn probabilities using the fitted pipeline.
        """
        if self.pipeline is None:
            raise ValueError("A fitted model pipeline is required to predict probabilities.")
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)[:, 1]
        elif hasattr(self.pipeline, "decision_function"):
            scores = self.pipeline.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        else:
            raise AttributeError("Pipeline does not support probability estimation.")

    def calculate_campaign_metrics(
        self,
        is_targeted: np.ndarray,
        y_true: Optional[np.ndarray],
        probabilities: np.ndarray,
        customer_values: np.ndarray,
        assumptions: Optional[BusinessAssumptions] = None,
    ) -> Dict[str, Any]:
        """
        Calculates campaign cost, expected retained customers, and net financial impact.

        Args:
            is_targeted: Boolean array indicating whether each customer is targeted.
            y_true: Actual binary churn labels (optional; if None, uses expected probabilities).
            probabilities: Model predicted churn probabilities.
            customer_values: Array of dollar values per customer.
            assumptions: Business assumptions override.

        Returns:
            Dictionary with financial and operational metrics.
        """
        params = assumptions or self.assumptions
        total_customers = len(is_targeted)
        num_targeted = int(np.sum(is_targeted))
        targeting_rate = float(num_targeted / total_customers) if total_customers > 0 else 0.0

        # Direct campaign cost
        campaign_cost = float(num_targeted * params.offer_cost)

        if num_targeted == 0:
            return {
                "customers_targeted": 0,
                "targeting_rate": 0.0,
                "campaign_cost": 0.0,
                "targeted_actual_churners": 0,
                "targeted_non_churners": 0,
                "precision": 0.0,
                "recall": 0.0,
                "expected_retained_customers": 0.0,
                "expected_value_preserved": 0.0,
                "net_financial_impact": 0.0,
                "roi_percentage": 0.0,
            }

        # Calculate retained customers & preserved value
        if y_true is not None:
            # Historical / Validation evaluation using ground truth churners
            y_true = np.asarray(y_true)
            actual_churners_targeted = int(np.sum(y_true[is_targeted] == 1))
            non_churners_targeted = int(num_targeted - actual_churners_targeted)
            total_actual_churners = int(np.sum(y_true == 1))

            precision = float(actual_churners_targeted / num_targeted)
            recall = float(actual_churners_targeted / total_actual_churners) if total_actual_churners > 0 else 0.0

            # Only actual churners can be saved by a retention offer
            targeted_churner_indices = np.where(is_targeted & (y_true == 1))[0]
            expected_retained = float(actual_churners_targeted * params.retention_rate)
            expected_value_preserved = float(
                np.sum(customer_values[targeted_churner_indices]) * params.retention_rate
            )
        else:
            # Prospective scoring: use expected probabilities
            actual_churners_targeted = float(np.sum(probabilities[is_targeted]))
            non_churners_targeted = float(num_targeted - actual_churners_targeted)
            precision = float(actual_churners_targeted / num_targeted)
            recall = float(actual_churners_targeted / np.sum(probabilities)) if np.sum(probabilities) > 0 else 0.0

            expected_retained = float(np.sum(probabilities[is_targeted]) * params.retention_rate)
            expected_value_preserved = float(
                np.sum(probabilities[is_targeted] * customer_values[is_targeted]) * params.retention_rate
            )

        net_impact = float(expected_value_preserved - campaign_cost)
        roi = float((net_impact / campaign_cost) * 100) if campaign_cost > 0 else 0.0

        return {
            "customers_targeted": num_targeted,
            "targeting_rate": round(targeting_rate, 4),
            "campaign_cost": round(campaign_cost, 2),
            "targeted_actual_churners": actual_churners_targeted,
            "targeted_non_churners": non_churners_targeted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "expected_retained_customers": round(expected_retained, 2),
            "expected_value_preserved": round(expected_value_preserved, 2),
            "net_financial_impact": round(net_impact, 2),
            "roi_percentage": round(roi, 2),
        }

    def simulate_strategy(
        self,
        strategy_name: str,
        df: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        top_k_percent: Optional[float] = None,
        top_n_count: Optional[int] = None,
        assumptions: Optional[BusinessAssumptions] = None,
    ) -> Dict[str, Any]:
        """
        Executes a targeting policy on a dataset.

        Supported strategies:
            - 'none': Target nobody (baseline)
            - 'all': Target 100% of customers
            - 'threshold': Target where probability >= threshold
            - 'top_k_percent': Target top K% highest churn probabilities
            - 'top_n_count': Target top N highest churn probabilities
            - 'value_weighted': Target where expected profit per customer > 0
        """
        params = assumptions or self.assumptions
        customer_values = params.get_customer_values(df)

        if probabilities is None:
            feature_cols = [c for c in df.columns if c != "Churn"]
            probabilities = self.predict_churn_probabilities(df[feature_cols])

        probabilities = np.asarray(probabilities)
        n = len(probabilities)

        if strategy_name == "none":
            is_targeted = np.zeros(n, dtype=bool)
        elif strategy_name == "all":
            is_targeted = np.ones(n, dtype=bool)
        elif strategy_name == "threshold":
            t = 0.50 if threshold is None else threshold
            is_targeted = probabilities >= t
        elif strategy_name == "top_k_percent":
            k = 0.20 if top_k_percent is None else top_k_percent
            if not (0.0 <= k <= 1.0):
                raise ValueError("top_k_percent must be between 0.0 and 1.0")
            n_target = int(np.round(n * k))
            if n_target == 0:
                is_targeted = np.zeros(n, dtype=bool)
            else:
                top_indices = np.argsort(probabilities)[::-1][:n_target]
                is_targeted = np.zeros(n, dtype=bool)
                is_targeted[top_indices] = True
        elif strategy_name == "top_n_count":
            count = 100 if top_n_count is None else min(top_n_count, n)
            top_indices = np.argsort(probabilities)[::-1][:count]
            is_targeted = np.zeros(n, dtype=bool)
            is_targeted[top_indices] = True
        elif strategy_name == "value_weighted":
            # Individual expected profit = p_i * r * V_i - c
            expected_ind_profit = (
                probabilities * params.retention_rate * customer_values
            ) - params.offer_cost
            is_targeted = expected_ind_profit > 0
        else:
            raise ValueError(f"Unknown strategy: '{strategy_name}'")

        metrics = self.calculate_campaign_metrics(
            is_targeted=is_targeted,
            y_true=y_true,
            probabilities=probabilities,
            customer_values=customer_values,
            assumptions=params,
        )
        metrics["strategy"] = strategy_name
        return metrics

    def optimize_threshold(
        self,
        df_train: pd.DataFrame,
        y_train: np.ndarray,
        probabilities_train: Optional[np.ndarray] = None,
        threshold_grid: Optional[np.ndarray] = None,
        assumptions: Optional[BusinessAssumptions] = None,
    ) -> Dict[str, Any]:
        """
        Finds the decision threshold that maximizes Net Financial Impact
        strictly on training / validation data.

        Returns:
            Dictionary with optimal threshold, maximum net impact, and full evaluation curve.
        """
        params = assumptions or self.assumptions
        customer_values = params.get_customer_values(df_train)

        if probabilities_train is None:
            feature_cols = [c for c in df_train.columns if c != "Churn"]
            probabilities_train = self.predict_churn_probabilities(df_train[feature_cols])

        if threshold_grid is None:
            threshold_grid = np.linspace(0.05, 0.95, 91)

        records = []
        best_threshold = 0.50
        max_net_impact = -float("inf")

        for t in threshold_grid:
            is_targeted = probabilities_train >= t
            metrics = self.calculate_campaign_metrics(
                is_targeted=is_targeted,
                y_true=y_train,
                probabilities=probabilities_train,
                customer_values=customer_values,
                assumptions=params,
            )
            metrics["threshold"] = round(float(t), 4)
            records.append(metrics)

            if metrics["net_financial_impact"] > max_net_impact:
                max_net_impact = metrics["net_financial_impact"]
                best_threshold = float(t)

        df_curve = pd.DataFrame(records)
        return {
            "optimal_threshold": round(best_threshold, 4),
            "max_net_financial_impact": round(max_net_impact, 2),
            "curve_data": df_curve,
        }

    def compare_strategies(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        optimal_threshold: float = 0.50,
        assumptions: Optional[BusinessAssumptions] = None,
    ) -> pd.DataFrame:
        """
        Evaluates and compiles a comparative benchmark table across multiple strategies.
        """
        params = assumptions or self.assumptions
        if probabilities is None:
            feature_cols = [c for c in df.columns if c != "Churn"]
            probabilities = self.predict_churn_probabilities(df[feature_cols])

        strategies = [
            ("Do Nothing (Baseline)", "none", {}),
            ("Mass Campaign (Target 100%)", "all", {}),
            ("Standard Threshold (p >= 0.50)", "threshold", {"threshold": 0.50}),
            ("Risk-Ranked Top 10%", "top_k_percent", {"top_k_percent": 0.10}),
            ("Risk-Ranked Top 20%", "top_k_percent", {"top_k_percent": 0.20}),
            ("Risk-Ranked Top 30%", "top_k_percent", {"top_k_percent": 0.30}),
            (f"Cost-Benefit Optimized (p >= {optimal_threshold:.2f})", "threshold", {"threshold": optimal_threshold}),
            ("Value-Weighted ROI Ranking", "value_weighted", {}),
        ]

        results = []
        for label, st_name, kwargs in strategies:
            res = self.simulate_strategy(
                strategy_name=st_name,
                df=df,
                probabilities=probabilities,
                y_true=y_true,
                assumptions=params,
                **kwargs,
            )
            results.append(
                {
                    "Strategy": label,
                    "Targeted Count": res["customers_targeted"],
                    "Targeting Rate": f"{res['targeting_rate']:.1%}",
                    "Campaign Cost ($)": f"${res['campaign_cost']:,.2f}",
                    "Retained Churners (Exp)": res["expected_retained_customers"],
                    "Value Preserved ($)": f"${res['expected_value_preserved']:,.2f}",
                    "Net Financial Impact ($)": f"${res['net_financial_impact']:,.2f}",
                    "ROI (%)": f"{res['roi_percentage']:.1f}%",
                    "Precision": f"{res['precision']:.3f}",
                    "Recall": f"{res['recall']:.3f}",
                    "net_impact_numeric": res["net_financial_impact"],
                }
            )

        df_comp = (
            pd.DataFrame(results)
            .sort_values("net_impact_numeric", ascending=False)
            .drop(columns=["net_impact_numeric"])
            .reset_index(drop=True)
        )
        return df_comp


def plot_profit_curve(
    optimization_curve_df: pd.DataFrame,
    optimal_threshold: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots Net Financial Impact as a function of the probability threshold.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color_profit = "#1b6ca8"
    color_cost = "#d9534f"
    color_retained = "#28a745"

    ax1.plot(
        optimization_curve_df["threshold"],
        optimization_curve_df["net_financial_impact"],
        color=color_profit,
        linewidth=2.5,
        label="Net Financial Impact ($)",
    )
    ax1.axvline(
        x=optimal_threshold,
        color="#e65c00",
        linestyle="--",
        linewidth=2,
        label=f"Optimal Threshold (p* = {optimal_threshold:.2f})",
    )
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.7)

    ax1.set_xlabel("Churn Probability Decision Threshold (p)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Net Financial Impact ($)", color=color_profit, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_profit)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Secondary axis: campaign cost and customers targeted
    ax2 = ax1.twinx()
    ax2.plot(
        optimization_curve_df["threshold"],
        optimization_curve_df["customers_targeted"],
        color=color_cost,
        linestyle="-.",
        linewidth=1.8,
        label="Customers Targeted",
    )
    ax2.set_ylabel("Customers Targeted (Count)", color=color_cost, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_cost)
    ax2.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True, facecolor="#f8f9fa")

    plt.title("Net Financial Impact vs. Churn Decision Threshold", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_cumulative_gains(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots cumulative gains and lift curves illustrating targeting efficiency.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    order = np.argsort(probabilities)[::-1]
    y_sorted = y_true[order]

    total_churners = np.sum(y_true == 1)
    n = len(y_true)

    percent_targeted = np.linspace(0, 100, n)
    cumulative_churners = np.cumsum(y_sorted == 1) / total_churners * 100

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(percent_targeted, cumulative_churners, color="#2b5c8f", linewidth=2.5, label="Model-Ranked Targeting")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.7, label="Random Untargeted Baseline")

    # Mark 20% and 30% points
    idx_20 = int(n * 0.20)
    idx_30 = int(n * 0.30)
    ax.scatter([20], [cumulative_churners[idx_20]], color="#e65c00", s=60, zorder=5)
    ax.annotate(
        f"Top 20% -> {cumulative_churners[idx_20]:.1f}% Churners",
        (20, cumulative_churners[idx_20]),
        textcoords="offset points",
        xytext=(10, -10),
        fontweight="bold",
        color="#e65c00",
    )

    ax.set_xlabel("Percentage of Customers Targeted (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Percentage of Churners Captured (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cumulative Gains Curve (Retention Targeting Efficiency)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
