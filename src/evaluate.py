"""
Evaluation module for Customer Churn Intelligence.

Provides comprehensive metric calculation, model comparison,
confusion matrix analysis, ROC/PR curves, and visualization export.
"""

from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Computes a comprehensive set of classification metrics tailored for imbalanced data.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def evaluate_pipeline(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates a fitted scikit-learn Pipeline on a test set.
    """
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)

    metrics = compute_classification_metrics(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def create_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts a dictionary of model results into a structured comparison DataFrame.
    """
    rows = []
    for model_name, res in results_dict.items():
        m = res["metrics"]
        rows.append(
            {
                "Model": model_name,
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "Recall": round(m["recall"], 4),
                "Precision": round(m["precision"], 4),
                "F1-Score": round(m["f1_score"], 4),
                "Specificity": round(m["specificity"], 4),
                "Accuracy": round(m["accuracy"], 4),
            }
        )
    df_comp = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_comp


def plot_roc_curves(
    results_dict: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots overlayed ROC curves for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    palette = ["#2b5c8f", "#e65c00", "#2ca02c", "#d62728", "#9467bd"]
    for i, (model_name, res) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc_score = res["metrics"]["roc_auc"]
        color = palette[i % len(palette)]
        ax.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {auc_score:.3f})",
            linewidth=2,
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves Comparison — Churn Prediction", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_precision_recall_curves(
    results_dict: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots overlayed Precision-Recall curves for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()

    palette = ["#2b5c8f", "#e65c00", "#2ca02c", "#d62728", "#9467bd"]
    for i, (model_name, res) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        pr_auc = res["metrics"]["pr_auc"]
        color = palette[i % len(palette)]
        ax.plot(
            recall,
            precision,
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})",
            linewidth=2,
            color=color,
        )

    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"Baseline Churn Rate ({baseline:.1%})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Coverage of Churners)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Accuracy of Churn Alerts)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_confusion_matrices(
    results_dict: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots 2x2 grid of confusion matrices for the 4 evaluated models.
    """
    models = list(results_dict.keys())
    n = len(models)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, model_name in enumerate(models):
        ax = axes[i]
        res = results_dict[model_name]
        cm = confusion_matrix(y_test, res["y_pred"])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        m = res["metrics"]
        ax.set_title(
            f"{model_name}\nRecall: {m['recall']:.2f} | Precision: {m['precision']:.2f} | F1: {m['f1_score']:.2f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Actual Label", fontsize=10)
        ax.set_xticklabels(["Non-Churn (0)", "Churn (1)"])
        ax.set_yticklabels(["Non-Churn (0)", "Churn (1)"])

    # Hide extra axes if odd number
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots horizontal bar chart of top N features.
    """
    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_imp["feature"], df_imp["importance"], color="#2b5c8f", edgecolor="#1a3b5c")
    ax.set_xlabel("Relative Importance / Magnitude", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Predictive Features — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def save_evaluation_summary(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: str = "models",
) -> None:
    """
    Saves model comparison table to CSV and full metrics dictionary to JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV comparison
    df_comp = create_comparison_table(results_dict)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)

    # Save JSON metrics
    json_metrics = {
        model_name: res["metrics"] for model_name, res in results_dict.items()
    }
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=4)
