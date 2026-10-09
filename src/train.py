"""
Model training and pipeline orchestration module for Customer Churn Intelligence.

Trains, cross-validates, and evaluates four candidate models:
1. Logistic Regression (Baseline)
2. Random Forest
3. XGBoost
4. LightGBM

Handles class imbalance, guarantees leak-free evaluation, and serializes
the best end-to-end pipeline ready for production / Streamlit serving.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb
import lightgbm as lgb

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    validate_dataset,
    get_feature_lists,
    create_train_test_split,
    build_preprocessor,
    get_transformed_feature_names,
)
from src.evaluate import (
    evaluate_pipeline,
    create_comparison_table,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importances,
    save_evaluation_summary,
)


def get_model_candidates(
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Instantiates candidate classifiers configured to handle class imbalance.

    Args:
        scale_pos_weight: Ratio of negative to positive samples in training data.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary of model instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
    }
    return models


def cross_validate_pipelines(
    models: Dict[str, Any],
    preprocessor_builder,
    num_cols: list,
    cat_cols: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Executes leak-free Stratified K-Fold cross validation on the training set.
    The preprocessor is re-fit inside every fold.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_records = []

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
    }

    for name, clf in models.items():
        print(f"    Evaluating {name} with {cv_folds}-fold CV...", flush=True)
        preprocessor = preprocessor_builder(num_cols, cat_cols)
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

        cv_records.append(
            {
                "Model": name,
                "CV ROC-AUC (Mean)": np.mean(scores["test_roc_auc"]),
                "CV ROC-AUC (Std)": np.std(scores["test_roc_auc"]),
                "CV PR-AUC (Mean)": np.mean(scores["test_pr_auc"]),
                "CV F1 (Mean)": np.mean(scores["test_f1"]),
                "CV Recall (Mean)": np.mean(scores["test_recall"]),
            }
        )

    return pd.DataFrame(cv_records)


def train_and_evaluate_all(
    data_path: str = "data/processed/telco_churn_features.csv",
    output_dir: str = "models",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Full training, evaluation, comparison, and serialization pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CUSTOMER CHURN INTELLIGENCE — PHASE 5 MODELING PIPELINE")
    print("=" * 60)

    # 1. Dataset Validation
    print(f"\n[1/6] Loading & Validating Dataset from: {data_path}")
    df = pd.read_csv(data_path)
    val_summary = validate_dataset(df)
    print(f"  * Rows: {val_summary['num_rows']:,} | Columns: {val_summary['num_columns']}")
    print(f"  * Missing values: {val_summary['missing_values']}")
    print(f"  * Target counts: {val_summary['target_counts']}")
    print(f"  * Target proportion: {val_summary['target_proportions']}")

    # 2. Extract feature sets & Split
    print("\n[2/6] Feature Separation & Stratified 80/20 Split")
    num_cols, cat_cols = get_feature_lists(df)
    print(f"  * Numerical features ({len(num_cols)}): {num_cols}")
    print(f"  * Categorical features ({len(cat_cols)}): {cat_cols}")

    X_train, X_test, y_train, y_test = create_train_test_split(
        df,
        target_col="Churn",
        test_size=0.20,
        random_state=random_state,
    )
    print(f"  * Training set: {X_train.shape[0]} rows (Churn rate: {y_train.mean():.2%})")
    print(f"  * Test set: {X_test.shape[0]} rows (Churn rate: {y_test.mean():.2%})")

    # Class weight calculation
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    print(f"  * Class weight ratio (Negative/Positive): {scale_pos_weight:.3f}")

    # 3. Stratified 5-Fold Cross Validation
    print("\n[3/6] Running 5-Fold Stratified Cross-Validation on Training Set...")
    models = get_model_candidates(scale_pos_weight=scale_pos_weight, random_state=random_state)
    df_cv = cross_validate_pipelines(
        models=models,
        preprocessor_builder=build_preprocessor,
        num_cols=num_cols,
        cat_cols=cat_cols,
        X_train=X_train,
        y_train=y_train,
        cv_folds=5,
        random_state=random_state,
    )
    print(df_cv.to_string(index=False))

    # 4. Fit Full Pipelines on Training Set & Evaluate on Test Set
    print("\n[4/6] Fitting Full Pipelines on Training Set & Evaluating on Holdout Test Set...")
    fitted_pipelines = {}
    test_results = {}

    for name, clf in models.items():
        print(f"  -> Fitting {name} pipeline...")
        preprocessor = build_preprocessor(num_cols, cat_cols)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on test set
        res = evaluate_pipeline(pipeline, X_test, y_test)
        test_results[name] = res

    # 5. Model Comparison
    print("\n[5/6] Model Comparison on Test Set:")
    df_comparison = create_comparison_table(test_results)
    print(df_comparison.to_string(index=False))

    # Identify best model by ROC-AUC (and PR-AUC)
    best_model_name = df_comparison.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n  [*] BEST MODEL SELECTED: {best_model_name}")

    # 6. Generate Visualizations & Save Artifacts
    print(f"\n[6/6] Saving Visualizations and Artifacts to '{output_dir}/'...")
    plot_roc_curves(
        results_dict=test_results,
        y_test=y_test,
        save_path=os.path.join(output_dir, "roc_curves.png"),
    )
    plot_precision_recall_curves(
        results_dict=test_results,
        y_test=y_test,
        save_path=os.path.join(output_dir, "precision_recall_curves.png"),
    )
    plot_confusion_matrices(
        results_dict=test_results,
        y_test=y_test,
        save_path=os.path.join(output_dir, "confusion_matrices.png"),
    )

    # Feature importances for best model
    best_clf = best_pipeline.named_steps["classifier"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)

    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        plot_feature_importances(
            feature_names=feature_names,
            importances=importances,
            model_name=best_model_name,
            top_n=15,
            save_path=os.path.join(output_dir, "feature_importance.png"),
        )
    elif hasattr(best_clf, "coef_"):
        importances = np.abs(best_clf.coef_[0])
        plot_feature_importances(
            feature_names=feature_names,
            importances=importances,
            model_name=best_model_name,
            top_n=15,
            save_path=os.path.join(output_dir, "feature_importance.png"),
        )

    # Save summary tables & metrics
    save_evaluation_summary(test_results, output_dir=output_dir)

    # Save best end-to-end pipeline
    best_model_path = os.path.join(output_dir, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"  * Saved Best Model Pipeline: {best_model_path}")

    # Save preprocessor independently for inference flexibility
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  * Saved Preprocessor: {preprocessor_path}")

    # Verify pipeline can directly predict on raw DataFrame sample
    sample_test = X_test.head(3)
    sample_preds = best_pipeline.predict(sample_test)
    sample_probs = best_pipeline.predict_proba(sample_test)[:, 1]
    print(f"\n[Verification] Pipeline inference test on raw input (3 samples):")
    for i, (pred, prob) in enumerate(zip(sample_preds, sample_probs)):
        print(f"  Sample {i+1}: Churn Pred = {pred}, Churn Prob = {prob:.4f}")

    print("\n[OK] Phase 5 Model Training and Evaluation Pipeline Completed Successfully!")
    return df_comparison, test_results, best_model_name


if __name__ == "__main__":
    train_and_evaluate_all()
