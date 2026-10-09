"""
SHAP Explainability Module for Customer Churn Intelligence.

Provides global and local model interpretability using SHAP TreeExplainer
tailored for the trained Random Forest classifier.
"""

from typing import Dict, Any, List, Optional, Union
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class ChurnExplainer:
    """
    Orchestrates global and local SHAP explanations for the Customer Churn model.

    Ensures explanations specifically isolate the positive class (Churn = 1)
    in probability space with exact mathematical additivity.
    """

    def __init__(self, pipeline_or_path: Union[str, Any] = "models/best_model_pipeline.joblib"):
        if isinstance(pipeline_or_path, str):
            if not os.path.exists(pipeline_or_path):
                raise FileNotFoundError(f"Pipeline artifact not found at: {pipeline_or_path}")
            self.pipeline = joblib.load(pipeline_or_path)
        else:
            self.pipeline = pipeline_or_path

        if not hasattr(self.pipeline, "named_steps"):
            raise ValueError("Provided object is not a valid scikit-learn Pipeline with named_steps.")

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # Initialize TreeExplainer on tree-based classifier
        self.explainer = shap.TreeExplainer(self.classifier)

    def get_feature_names(self) -> List[str]:
        """
        Retrieves feature names produced by the ColumnTransformer.
        """
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            cat_step = self.preprocessor.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(cat_step.get_feature_names_out())
            num_cols = self.preprocessor.transformers_[0][2]
            return list(num_cols) + cat_names

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw feature inputs through the fitted preprocessor
        preserving column names.
        """
        X_trans = self.preprocessor.transform(X)
        if isinstance(X_trans, pd.DataFrame):
            return X_trans
        feature_names = self.get_feature_names()
        return pd.DataFrame(X_trans, columns=feature_names, index=X.index)

    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Computes SHAP values for the positive churn class (Churn = 1).

        Returns:
            shap.Explanation object targeting class 1.
        """
        X_trans = self.transform_features(X)
        explanation_3d = self.explainer(X_trans)

        # For binary Random Forest, explanation_3d has shape (N, features, 2)
        # We specifically extract index 1 (positive class: Churn = 1)
        if len(explanation_3d.shape) == 3 and explanation_3d.shape[2] == 2:
            churn_explanation = explanation_3d[:, :, 1]
        else:
            churn_explanation = explanation_3d

        return churn_explanation

    def explain_customer(
        self,
        customer_data: Union[pd.DataFrame, pd.Series, dict],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a single customer record.

        Args:
            customer_data: Dict, Series, or 1-row DataFrame containing raw customer features.
            top_n: Number of positive and negative drivers to highlight.

        Returns:
            Dictionary with churn probability, baseline, top risk factors, and protective factors.
        """
        if isinstance(customer_data, dict):
            df_cust = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.Series):
            df_cust = pd.DataFrame([customer_data.to_dict()])
        elif isinstance(customer_data, pd.DataFrame):
            if len(customer_data) != 1:
                raise ValueError(f"Expected exactly 1 customer record, got {len(customer_data)}")
            df_cust = customer_data.copy()
        else:
            raise TypeError("customer_data must be a dict, pd.Series, or 1-row pd.DataFrame")

        # Exclude target column if present
        if "Churn" in df_cust.columns:
            df_cust = df_cust.drop(columns=["Churn"])

        # Predict probability
        churn_prob = float(self.pipeline.predict_proba(df_cust)[0, 1])
        predicted_class = int(churn_prob >= 0.50)

        # Compute SHAP explanation
        exp = self.compute_shap_values(df_cust)
        single_exp = exp[0]

        values = single_exp.values
        feature_names = single_exp.feature_names
        data_values = single_exp.data
        base_value = float(single_exp.base_values)

        shap_sum = float(np.sum(values))
        additivity_check = bool(np.isclose(base_value + shap_sum, churn_prob, atol=1e-5))

        df_contributions = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": values,
                "feature_value": data_values,
            }
        )

        # Factors increasing churn risk (positive SHAP contributions)
        top_positive = (
            df_contributions[df_contributions["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
        )

        # Factors decreasing churn risk (negative SHAP contributions)
        top_negative = (
            df_contributions[df_contributions["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(top_n)
            .to_dict(orient="records")
        )

        return {
            "predicted_churn_probability": round(churn_prob, 4),
            "predicted_class": predicted_class,
            "base_value": round(base_value, 4),
            "sum_shap_contributions": round(shap_sum, 4),
            "reconstructed_probability": round(base_value + shap_sum, 4),
            "additivity_verified": additivity_check,
            "top_risk_drivers": [
                {
                    "feature": r["feature"],
                    "impact": round(float(r["shap_value"]), 4),
                    "encoded_value": r["feature_value"],
                }
                for r in top_positive
            ],
            "top_retention_drivers": [
                {
                    "feature": r["feature"],
                    "impact": round(float(r["shap_value"]), 4),
                    "encoded_value": r["feature_value"],
                }
                for r in top_negative
            ],
            "explanation_object": single_exp,
        }

    def plot_summary(
        self,
        explanation: shap.Explanation,
        X_sample: pd.DataFrame,
        max_display: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates beeswarm summary plot showing feature impact distribution.
        """
        fig = plt.figure(figsize=(10, 7))
        X_trans = self.transform_features(X_sample)
        shap.summary_plot(
            explanation.values,
            X_trans,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Beeswarm Summary Plot (Impact on Churn = 1)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_feature_importance(
        self,
        explanation: shap.Explanation,
        max_display: int = 15,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots mean absolute SHAP feature importance bar chart.
        """
        fig = plt.figure(figsize=(9, 6))
        shap.plots.bar(
            explanation,
            max_display=max_display,
            show=False,
        )
        plt.title(f"Top {max_display} Features by Mean |SHAP Value| (Churn Class)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_dependence(
        self,
        explanation: shap.Explanation,
        feature_name: str,
        X_sample: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots dependence scatter plot for a key continuous feature.
        """
        fig = plt.figure(figsize=(8, 5.5))
        X_trans = self.transform_features(X_sample)
        shap.dependence_plot(
            feature_name,
            explanation.values,
            X_trans,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot for '{feature_name}'", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_local_waterfall(
        self,
        single_explanation: shap.Explanation,
        max_display: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generates waterfall plot explaining an individual customer's prediction.
        """
        fig = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(
            single_explanation,
            max_display=max_display,
            show=False,
        )
        plt.title("Local SHAP Waterfall Explanation (Churn Risk)", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


def run_explainability_pipeline(
    data_path: str = "data/processed/telco_churn_features.csv",
    model_path: str = "models/best_model_pipeline.joblib",
    output_dir: str = "models",
    sample_size: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes global explainability and generates visual artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65, flush=True)
    print("CUSTOMER CHURN INTELLIGENCE -- PHASE 7: SHAP EXPLAINABILITY", flush=True)
    print("=" * 65, flush=True)

    print(f"\n[1/4] Loading model and data...", flush=True)
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    explainer = ChurnExplainer(model_path)

    # Sample representative subset for fast, reproducible SHAP calculation
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    print(f"  * Sample size for global SHAP: {len(X_sample)} customers", flush=True)

    print("\n[2/4] Computing SHAP values for class 'Churn = 1'...", flush=True)
    explanation = explainer.compute_shap_values(X_sample)
    print(f"  * SHAP Explanation shape: {explanation.shape}", flush=True)
    print(f"  * Base churn value E[p]: {float(explanation.base_values[0]):.4f}", flush=True)

    print("\n[3/4] Exporting Global SHAP Visualizations...", flush=True)
    summary_path = os.path.join(output_dir, "shap_summary.png")
    explainer.plot_summary(explanation, X_sample, max_display=15, save_path=summary_path)
    plt.close("all")
    print(f"  * Saved: {summary_path}", flush=True)

    bar_path = os.path.join(output_dir, "shap_feature_importance.png")
    explainer.plot_feature_importance(explanation, max_display=15, save_path=bar_path)
    plt.close("all")
    print(f"  * Saved: {bar_path}", flush=True)

    # Optional dependence plot for tenure
    dep_path = os.path.join(output_dir, "shap_dependence_tenure.png")
    explainer.plot_dependence(explanation, "tenure", X_sample, save_path=dep_path)
    plt.close("all")
    print(f"  * Saved: {dep_path}", flush=True)

    print("\n[4/4] Generating Local Explanation for Sample High-Risk Customer...", flush=True)
    # Pick a high-risk customer from sample
    probs = explainer.pipeline.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argsort(probs)[-1]
    sample_customer = X_sample.iloc[[high_risk_idx]]

    local_result = explainer.explain_customer(sample_customer, top_n=5)
    print(f"  * High-Risk Customer Predicted Churn Probability: {local_result['predicted_churn_probability']:.1%}", flush=True)
    print(f"  * Baseline Churn Rate: {local_result['base_value']:.1%}", flush=True)
    print("  * Top Risk Factors (Increasing Churn):", flush=True)
    for driver in local_result["top_risk_drivers"][:3]:
        print(f"      + {driver['feature']}: +{driver['impact']:.4f}", flush=True)
    print("  * Top Protective Factors (Decreasing Churn):", flush=True)
    for driver in local_result["top_retention_drivers"][:3]:
        print(f"      - {driver['feature']}: {driver['impact']:.4f}", flush=True)

    local_path = os.path.join(output_dir, "shap_local_example.png")
    explainer.plot_local_waterfall(local_result["explanation_object"], max_display=10, save_path=local_path)
    plt.close("all")
    print(f"  * Saved: {local_path}", flush=True)

    print("\n[OK] Phase 7 SHAP Explainability Completed Successfully!", flush=True)
    return {
        "global_explanation": explanation,
        "sample_customer_explanation": local_result,
    }


if __name__ == "__main__":
    run_explainability_pipeline()
