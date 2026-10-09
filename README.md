# Customer Churn Intelligence

An end-to-end enterprise machine learning system analyzing customer retention and predicting churn risk using the standard IBM Telco Customer Churn dataset.

## Project Structure

```
customer-churn-intelligence/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Unmodified raw IBM dataset (7,043 rows)
│   └── processed/
│       ├── telco_churn_clean.csv                  # Phase 2: Leak-free cleaned dataset
│       └── telco_churn_features.csv               # Phase 4: Engineered feature dataset
│
├── notebooks/
│   ├── 01_data_understanding.ipynb                # Phase 1: Exploration, inspection & data profiling
│   ├── 02_data_cleaning.ipynb                      # Phase 2: Missing values, type fixing, target mapping
│   ├── 03_eda.ipynb                                # Phase 3: Business-driven exploratory data analysis
│   ├── 04_feature_engineering.ipynb                # Phase 4: Behavioral & structural feature generation
│   ├── 05_model_training.ipynb                     # Phase 5: Benchmark modeling & evaluation
│   ├── 06_business_impact_analysis.ipynb           # Phase 6: Financial impact simulation & ROI optimization
│   └── 07_shap_explainability.ipynb                # Phase 7: Global & local SHAP interpretability
│
├── src/                                           # Reusable modular code
│   ├── __init__.py
│   ├── preprocessing.py                           # Leak-free ColumnTransformer & data splitters
│   ├── feature_engineering.py                     # Raw attribute transformer (27 features)
│   ├── evaluate.py                                # Metric computation, curve plotting & summaries
│   ├── train.py                                   # Training orchestration & pipeline export
│   ├── business_impact.py                         # Retention economics simulator & policies
│   ├── optimize_retention.py                      # Threshold optimization & policy benchmarking
│   ├── explainability.py                          # Global/local SHAP TreeExplainer & waterfall charts
│   └── dashboard_utils.py                         # App caching, Plotly chart builders & predictors
│
├── models/                                        # Trained model binaries & evaluation artifacts
│   ├── best_model_pipeline.joblib                 # End-to-end trained inference pipeline
│   ├── preprocessor.joblib                        # Standalone preprocessor artifact
│   ├── model_comparison.csv                       # Experimental benchmark comparison
│   ├── evaluation_metrics.json                    # Full test evaluation metrics
│   ├── targeting_strategies_comparison.csv        # Phase 6: Retention strategy financial P&L
│   ├── business_simulation_results.json           # Phase 6: Simulation parameters & ROI metrics
│   ├── profit_curve.png                           # Phase 6: Net financial impact vs decision threshold
│   ├── cumulative_gains_curve.png                 # Phase 6: Cumulative churner capture vs targeted %
│   ├── shap_summary.png                           # Phase 7: Global beeswarm summary plot
│   ├── shap_feature_importance.png                # Phase 7: Mean |SHAP| feature ranking bar chart
│   ├── shap_dependence_tenure.png                 # Phase 7: Non-linear tenure dependence curve
│   ├── shap_local_example.png                     # Phase 7: Individual customer waterfall explanation
│   ├── roc_curves.png                             # Multi-model ROC curves
│   ├── precision_recall_curves.png                # Multi-model PR curves
│   ├── confusion_matrices.png                     # Multi-model confusion matrices
│   └── feature_importance.png                     # Top predictive features plot
│
├── tests/                                         # Unit and integration test suite
│   ├── test_modeling.py                           # Tests for leakage, preprocessing, and inference
│   ├── test_business_impact.py                    # Tests for unit economics, ranking, and edge cases
│   ├── test_explainability.py                     # Tests for SHAP additivity, dimensions, and customer explanations
│   └── test_dashboard.py                          # Tests for input schemas, zero tenure, and charts
│
├── app/                                           # Interactive Streamlit application
│   └── main.py                                    # Dashboard entry point (3 views)
├── requirements.txt                               # Project dependencies
├── README.md                                      # Documentation
└── .gitignore                                     # Ignored files
```

## Dataset Overview

- **Source:** IBM Sample Data Sets (Telco Customer Churn)
- **Rows:** 7,043 customers
- **Features:** 21 attributes spanning demographics, account data, services, and billing
- **Target:** `Churn` (Yes: ~26.5%, No: ~73.5%)

## Phase 5 Model Benchmark Results (Holdout Test Set)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.8452** | 0.6525 | 0.7834 | **0.5337** | **0.6349** | **0.7527** | **0.7608** |
| **XGBoost** | 0.8444 | **0.6597** | 0.7968 | 0.5147 | 0.6254 | 0.7285 | 0.7466 |
| **Logistic Regression** | 0.8432 | 0.6372 | 0.8021 | 0.5093 | 0.6231 | 0.7208 | 0.7424 |
| **LightGBM** | 0.8429 | **0.6597** | **0.8102** | 0.5188 | 0.6326 | 0.7285 | 0.7502 |

*Note: Models evaluated with class weighting / scale_pos_weight to prioritize churner recall.*

## Phase 6 Business Impact & Retention Simulation (Holdout Test Set, N=1,409)

*Baseline Assumptions: Offer Cost $c = \$50$, Preserved CLV $V = \$500$, Offer Success Rate $r = 20\%$.*

| Strategy | Targeted | Campaign Cost | Retained (Exp) | Preserved Value | Net Financial Impact | ROI | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| **Cost-Benefit Optimized ($p \ge 0.65$)** | **359 (25.5%)** | **$17,950** | **45.0** | **$22,500** | **+$4,550** | **25.4%** | **0.627** | **0.602** |
| **Risk-Ranked Top 20%** | 282 (20.0%) | $14,100 | 37.0 | $18,500 | +$4,400 | 31.2% | 0.656 | 0.495 |
| **Risk-Ranked Top 30%** | 423 (30.0%) | $21,150 | 49.8 | $24,900 | +$3,750 | 17.7% | 0.589 | 0.666 |
| **Risk-Ranked Top 10%** | 141 (10.0%) | $7,050 | 21.2 | $10,600 | +$3,550 | **50.4%** | 0.752 | 0.283 |
| **Standard Threshold ($p \ge 0.50$)** | 549 (39.0%) | $27,450 | 58.6 | $29,300 | +$1,850 | 6.7% | 0.534 | 0.783 |
| **Do Nothing (Baseline)** | 0 (0.0%) | $0 | 0.0 | $0 | $0 | 0.0% | 0.000 | 0.000 |
| **Mass Campaign (Target 100%)** | 1,409 (100%) | $70,450 | 74.8 | $37,400 | **-$33,050** | -46.9% | 0.265 | 1.000 |

*Key finding: Optimizing threshold on training data ($p^* = 0.65$) increases net profit by +146% over the naive 0.50 threshold while saving $9,500 in campaign budget. Untargeted mass outreach produces a massive -$33,050 loss.*

## Phase 7 SHAP Explainability & Risk Attribution

- **TreeExplainer Formulation:** Explanations specifically target the positive class `Churn = 1` in probability space with exact mathematical additivity ($\hat{p} = \mathbb{E}[p] + \sum \phi_j$, verified to $< 10^{-14}$ error).
- **Dominant Churn Drivers:**
  1. `tenure` (low tenure strongly accelerates churn risk; drops sharply after 24 months)
  2. `Contract_Month-to-month` & `contract_tenure_group_Month-to-month_New` (lack of contract lock-in)
  3. `TotalCharges` & `MonthlyCharges` (monetary strain and revenue intensity)
  4. `InternetService_Fiber optic` (higher churn segment when unbundled with security/backup)
  5. `PaymentMethod_Electronic check` (consistently associated with higher churn likelihood)
- **Local Waterfall Attributions:** Each customer prediction decomposes into baseline probability plus distinct positive risk drivers and negative protective retention factors.

## Phase 8 Interactive Streamlit Dashboard

The project includes an enterprise-grade analytics and decision-support web application built with Streamlit and Plotly.

### Launching the Dashboard Locally
```powershell
python -m streamlit run app/main.py
```
*Access via browser at:* `http://localhost:8501`

### Dashboard Sections
1. **📊 Executive Insights & Cohort Analysis:**
   - Real-time KPI cards: Total Customers (7,043), Observed Churn Rate (26.5%), Average Monthly Bill ($64.76), Monthly Revenue at Risk ($139,131).
   - Interactive Plotly visualizations: Churn by Contract, Churn by Internet Technology, Tenure Distribution, and Multi-Service Adoption curve.
   - Global SHAP feature ranking and beeswarm summary plots.
   - Cohort filters (Contract, Internet Service, Payment Method, Tenure Range).

2. **🔮 Customer Churn Predictor & Local SHAP Diagnosis:**
   - Validated customer input form supporting all 19 raw Telco attributes with real-time automated feature engineering (27 model-ready features).
   - Instant churn risk probability, risk category (Low, Moderate, High), and baseline comparison.
   - Interactive Plotly SHAP Waterfall chart decomposing the exact contribution of each feature in probability space.
   - Highlighted Top 3 Churn Risk Drivers and Top 3 Protective Retention Strengths.
   - Presets for high-risk, low-risk, and moderate-risk customer personas.

3. **💰 Retention ROI Simulator & Shortlist Export:**
   - Interactive controls for Offer Cost ($c$), Preserved Customer Lifetime Value ($V$), and Offer Success Lift ($r$).
   - Real-time breakeven threshold indicator: $p^* = \frac{c}{r \times V}$.
   - P&L projections: Campaign Cost, Saved Customers, Value Preserved, Net Financial Impact, and ROI.
   - Interactive Profit Curve comparing threshold policies against targeting reach.
   - Multi-strategy benchmark table comparing test-set policies.
   - One-click CSV export of prioritized high-risk customer targeting shortlist.

## Project Roadmap

- **Phase 1: Project Setup & Dataset Understanding** (Completed)
- **Phase 2: Data Cleaning & Integrity** (Completed)
- **Phase 3: Business-Focused EDA** (Completed)
- **Phase 4: Feature Engineering** (Completed)
- **Phase 5: ML Modeling & Evaluation** (Completed)
- **Phase 6: Business Impact Analysis & Retention Optimization** (Completed)
- **Phase 7: SHAP Explainability & Interpretability** (Completed)
- **Phase 8: Interactive Streamlit Dashboard** (Completed)
