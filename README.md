# Customer Churn Intelligence & Retention Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-EB5424.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-02569B.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Tests-25%20Passed-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade machine learning system that transforms customer churn prediction into actionable retention economics. Built on the IBM Telco Customer Churn dataset, this system pairs leak-free predictive modeling and SHAP explainability with a cost-benefit optimization framework and an interactive Streamlit decision-support application.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture](#system-architecture)
- [Key Business Findings](#key-business-findings)
- [Dataset & Feature Engineering](#dataset--feature-engineering)
- [Model Benchmark & Evaluation](#model-benchmark--evaluation)
- [Retention ROI & Business Simulation](#retention-roi--business-simulation)
- [SHAP Interpretability & Risk Attribution](#shap-interpretability--risk-attribution)
- [Interactive Streamlit Application](#interactive-streamlit-application)
- [Test Suite & Quality Assurance](#test-suite--quality-assurance)
- [Project Directory Structure](#project-directory-structure)
- [Quickstart & Installation](#quickstart--installation)
- [License](#license)

---

## Executive Summary

Customer attrition is one of the highest cost drivers in subscription telecom businesses. Predicting churn alone is insufficient—intervention programs incur marketing costs and offer discounts that risk negative return on investment if misallocated.

This project delivers an end-to-end intelligence engine that:
1. **Accurately Identifies Churn Risk:** Benchmarks 4 candidate models with class imbalance mitigation, selecting an ensemble pipeline achieving **0.8452 ROC-AUC** and **0.81 Recall**.
2. **Decodes Behavioral Churn Drivers:** Employs SHAP TreeExplainer with strict positive-class probability additivity ($\hat{p} = \mathbb{E}[p] + \sum \phi_j$) to isolate actionable risk factors per customer.
3. **Maximizes Campaign ROI:** Formulates expected net financial impact as $E[\Delta \Pi] = r \cdot V \cdot p - c$, proving that optimizing the intervention threshold ($p^* = 0.65$) yields **+$4,550 net profit (+146% profit boost)** over a standard 0.50 threshold while eliminating $9,500 in wasted outreach.
4. **Empowers Retention Teams:** Provides an interactive Streamlit dashboard featuring cohort analytics, individual customer risk diagnosis with live SHAP waterfall plots, dynamic P&L simulation, and downloadable targeting shortlists.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Data Pipeline
        RAW[Raw IBM Telco Data\n7,043 rows, 21 attributes] --> CLEAN[Phase 2: Cleaning\nMissing charge imputation, type fixing]
        CLEAN --> FE[Phase 4: Leak-Free Feature Engineering\n27 behavioral & interaction features]
    end

    subgraph Modeling & Explainability
        FE --> SPLIT[Stratified 80/20 Train/Test Split]
        SPLIT --> PIPE[ColumnTransformer Pipeline\nStandardScaler + OneHotEncoder]
        PIPE --> BENCH[Phase 5: Benchmark Modeling\nRF, XGBoost, LightGBM, Logistic Regression]
        BENCH --> BEST[Best Model: Random Forest\nROC-AUC: 0.8452 | Recall: 0.7834]
        BEST --> SHAP[Phase 7: SHAP TreeExplainer\nExact probability additivity]
    end

    subgraph Business Optimization & Serving
        BEST --> OPT[Phase 6: Retention Economics\nE[Net Profit] = r·V·p - c]
        OPT --> THRESH[Optimal Threshold p* = 0.65\nNet Profit: +$4,550 | Campaign Cost: $17,950]
        SHAP --> APP[Phase 8: Streamlit Application\nExecutive KPIs, Churn Predictor, ROI Simulator]
        THRESH --> APP
        APP --> EXPORT[Targeting Shortlist CSV Export]
    end
```

---

## Key Business Findings

- **Untargeted Outreach is Value-Destructive:** An untargeted mass campaign (reaching 100% of customers) costs \$70,450 and yields a **-$33,050 net financial loss** (-46.9% ROI).
- **Threshold Optimization Unlocks +146% Profit Uplift:** Adjusting the intervention threshold from the conventional 0.50 cutoff to the mathematically optimal **$p^* = 0.65$** increases net campaign profit from \$1,850 to **\$4,550** on the test cohort ($N=1,409$), while saving **\$9,500** in campaign overhead.
- **Top 10% Risk Tier Delivers Highest ROI:** For budget-constrained scenarios, targeting solely the top 10% highest-risk customers captures 28.3% of all churners with **75.2% campaign precision** and a peak **50.4% ROI**.
- **Tenure Cliff (The First 24 Months):** Customers in months 1–12 exhibit a 47% churn rate, dropping to under 14% after month 24. Contract lock-in and bundled tech support are the single strongest retention levers during this critical window.

---

## Dataset & Feature Engineering

- **Source:** IBM Telco Customer Churn Dataset (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Scale:** 7,043 customer accounts, 21 original attributes
- **Target Distribution:** 1,869 Churned (26.54%), 5,174 Retained (73.46%)

### Leak-Free Engineered Features (27 Features)
To maximize predictive signal without data leakage across cross-validation folds:
- **`tenure_group`:** Binned cohorts (`0-12m`, `12-24m`, `24-48m`, `48-72m`) capturing non-linear customer lifecycle attrition.
- **`monthly_to_total_ratio`:** Ratio of monthly bill to cumulative lifetime spend, signaling early monetary strain.
- **`has_internet_service` & `internet_type`:** Distinguishing Fiber Optic, DSL, and No Internet accounts.
- **`total_services_count`:** Cumulative count of active value-add services (Security, Backup, Protection, Tech Support, Streaming TV, Streaming Movies).
- **`streaming_services_count`:** Discrete count of entertainment add-ons.
- **`contract_tenure_group`:** High-leverage interaction feature pairing contract stability with customer lifecycle stage (e.g., `Month-to-month_New` vs `Two year_Veteran`).

---

## Model Benchmark & Evaluation

All candidate models were evaluated on an identical **Stratified 80/20 Holdout Test Set ($N=1,409$)** using strict scikit-learn `Pipeline` structures to prevent preprocessing leakage. Imbalance was addressed through `class_weight='balanced'` and `scale_pos_weight`.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Specificity | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** 🏆 | **0.8452** | 0.6525 | 0.7834 | **0.5337** | **0.6349** | **0.7527** | **0.7608** |
| **XGBoost** | 0.8444 | **0.6597** | 0.7968 | 0.5147 | 0.6254 | 0.7285 | 0.7466 |
| **LightGBM** | 0.8429 | **0.6597** | **0.8102** | 0.5188 | 0.6326 | 0.7285 | 0.7502 |
| **Logistic Regression** | 0.8432 | 0.6372 | 0.8021 | 0.5093 | 0.6231 | 0.7208 | 0.7424 |

*Selected Production Pipeline: Tuned Random Forest with calibrated probabilities and robust generalization across minority class recall.*

---

## Retention ROI & Business Simulation

### Unit Economics Formulation
Intervention economics are defined by three key parameters:
- **$c$ (Offer / Campaign Cost):** \$50 per targeted customer
- **$V$ (Preserved Customer Lifetime Value):** \$500 per successfully saved customer
- **$r$ (Intervention Success Rate):** 20% rescue lift among potential churners
- **Theoretical Breakeven Threshold:**
  $$p^* = \frac{c}{r \times V} = \frac{50}{0.20 \times 500} = 0.50$$

When optimizing empirically against the empirical training distribution to balance false-positive costs against false-negative customer loss, the profit-maximizing threshold rises to **$p^* = 0.65$**.

### Campaign Strategy Performance Comparison ($N=1,409$ Holdout Set)

| Strategy | Targeted Cohort | Campaign Cost | Expected Retained | Preserved Value | Net Financial Impact | Campaign ROI | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cost-Benefit Optimized ($p \ge 0.65$)** | **359 (25.5%)** | **\$17,950** | **45.0** | **\$22,500** | **+\$4,550** | **25.4%** | **0.627** | **0.602** |
| **Risk-Ranked Top 20%** | 282 (20.0%) | \$14,100 | 37.0 | \$18,500 | +\$4,400 | 31.2% | 0.656 | 0.495 |
| **Risk-Ranked Top 30%** | 423 (30.0%) | \$21,150 | 49.8 | \$24,900 | +\$3,750 | 17.7% | 0.589 | 0.666 |
| **Risk-Ranked Top 10%** | 141 (10.0%) | \$7,050 | 21.2 | \$10,600 | +\$3,550 | **50.4%** | 0.752 | 0.283 |
| **Standard Threshold ($p \ge 0.50$)** | 549 (39.0%) | \$27,450 | 58.6 | \$29,300 | +\$1,850 | 6.7% | 0.534 | 0.783 |
| **Do Nothing (Baseline)** | 0 (0.0%) | \$0 | 0.0 | \$0 | \$0 | 0.0% | 0.000 | 0.000 |
| **Mass Campaign (Target 100%)** | 1,409 (100%) | \$70,450 | 74.8 | \$37,400 | **-\$33,050** | -46.9% | 0.265 | 1.000 |

---

## SHAP Interpretability & Risk Attribution

Model predictions are fully explainable using **SHAP TreeExplainer**. Explanations operate strictly in probability space for the positive class (`Churn = 1`), satisfying exact mathematical additivity:
$$\hat{p}(\mathbf{x}) = \mathbb{E}[p] + \sum_{j=1}^{M} \phi_j(\mathbf{x})$$

*(Verified across all test observations to machine precision $\Delta < 10^{-14}$)*

### Primary Churn Risk Drivers
1. **Tenure Duration (`tenure`):** Strongest protective factor. Accounts past 24 months exhibit sharply negative SHAP attribution values.
2. **Contract Structure (`Contract_Month-to-month`):** Primary risk accelerator. Month-to-month customers lack switching barriers and incur immediate positive churn attribution.
3. **Monetary Intensity (`TotalCharges` & `MonthlyCharges`):** High bills without bundled utility amplify price sensitivity.
4. **Internet Technology (`InternetService_Fiber optic`):** High churn risk unless paired with Online Security and Tech Support add-ons.
5. **Billing Mode (`PaymentMethod_Electronic check`):** Associated with significantly higher churn compared to credit card or automated bank transfer.

---

## Interactive Streamlit Application

The interactive web application (`app/main.py`) serves business leaders, retention managers, and customer success agents through three tailored views:

```
┌────────────────────────────────────────────────────────────────────────┐
│                     CUSTOMER CHURN INTELLIGENCE                        │
├─────────────────────┬───────────────────────────┬──────────────────────┤
│ 📊 Executive        │ 🔮 Individual Customer    │ 💰 Retention ROI     │
│    Insights         │    Churn Predictor        │    Simulator         │
│                     │                           │                      │
│ • Real-time KPIs    │ • 19 raw Telco attributes │ • Interactive c,V,r  │
│ • Cohort filters    │ • Real-time FE (27 feats) │ • Real-time breakeven│
│ • Service adoption  │ • Interactive Waterfall   │ • Profit curve plot  │
│ • Global SHAP plots │ • Persona presets         │ • Shortlist CSV DL   │
└─────────────────────┴───────────────────────────┴──────────────────────┘
```

### 1. 📊 Executive Insights & Cohort Analysis
- Real-time KPI summaries: Total Customer Base (7,043), Observed Churn (26.54%), Average Bill (\$64.76), Revenue at Risk (\$139,131/mo).
- Interactive Plotly visualizations: Churn rate by contract, internet type, tenure distribution, and multi-service adoption curve.
- Global SHAP beeswarm summary and mean absolute importance ranking plots.
- Dynamic filtering by contract type, payment method, internet technology, and tenure range.

### 2. 🔮 Customer Churn Predictor & Local SHAP Diagnosis
- Form interface supporting all 19 raw customer attributes with automated feature engineering to 27 model-ready inputs.
- One-click persona presets: **High-Risk Newcomer**, **Loyal Long-Term Customer**, and **Moderate-Risk Fiber User**.
- Real-time churn probability gauge, risk classification (Low, Moderate, High), and baseline comparison.
- **Interactive Plotly Waterfall Chart:** Decomposes the exact contribution of each feature in probability space.
- Distinct breakdown of **Top 3 Risk Drivers** (what to fix) and **Top 3 Protective Strengths** (what to reinforce).

### 3. 💰 Retention ROI Simulator & Targeting Shortlist
- Dynamic financial inputs: Campaign Offer Cost ($c$), Preserved Customer Value ($V$), and Offer Success Rate ($r$).
- Real-time breakeven calculation ($p^* = \frac{c}{r \times V}$) and recommended profit-maximizing threshold.
- Financial P&L summary: Campaign Cost, Saved Customers, Value Preserved, Net Profit, and ROI.
- Interactive Profit Curve comparing threshold policies against targeting reach.
- Multi-strategy comparison table benchmarking test-set policies.
- **One-click CSV Export:** Generates prioritized retention candidate shortlists filtered by predicted probability and expected net value.

---

## Test Suite & Quality Assurance

The codebase includes **25 automated tests** covering data integrity, leakage prevention, mathematical invariants, and UI contracts:

```powershell
python -m pytest tests/ -v
```

```
tests/test_business_impact.py ........ [ 28%]
tests/test_dashboard.py .......        [ 56%]
tests/test_explainability.py ......    [ 80%]
tests/test_modeling.py .....           [100%]
======================= 25 passed in 18.56s =======================
```

- **`tests/test_modeling.py`:** Validates leak-free data splitting, `ColumnTransformer` integrity, shape alignment, and inference pipeline determinism.
- **`tests/test_business_impact.py`:** Tests unit economics calculations, breakeven formulas, risk ranking order, threshold boundary conditions, and P&L edge cases.
- **`tests/test_explainability.py`:** Enforces mathematical additivity of TreeExplainer probability attributions, checks feature alignment, and verifies local waterfall attributions.
- **`tests/test_dashboard.py`:** Tests dynamic raw-to-engineered feature transformation, missing field fallbacks, zero-tenure boundary conditions, and Plotly chart generation.

---

## Project Directory Structure

```
customer-churn-intelligence/
├── app/
│   └── main.py                                    # Streamlit application entry point (3 views)
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Unmodified raw IBM dataset (7,043 rows)
│   └── processed/
│       ├── telco_churn_clean.csv                  # Cleaned dataset (type-fixed, leak-free)
│       └── telco_churn_features.csv               # Engineered feature dataset (27 features)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb                # Data exploration, inspection & profiling
│   ├── 02_data_cleaning.ipynb                      # Type conversion & whitespace imputation
│   ├── 03_eda.ipynb                                # Business-focused exploratory analysis
│   ├── 04_feature_engineering.ipynb                # Feature extraction & interaction creation
│   ├── 05_model_training.ipynb                     # Cross-validation & benchmark modeling
│   ├── 06_business_impact_analysis.ipynb           # Financial simulation & threshold optimization
│   └── 07_shap_explainability.ipynb                # Global & local SHAP tree explainability
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                           # Preprocessing pipelines & data splitters
│   ├── feature_engineering.py                     # Raw feature transformer (27 features)
│   ├── evaluate.py                                # Evaluation metrics & comparison summaries
│   ├── train.py                                   # Training orchestration & pipeline export
│   ├── business_impact.py                         # Retention economics simulator & policies
│   ├── optimize_retention.py                      # Threshold optimization & policy benchmarking
│   ├── explainability.py                          # TreeExplainer wrapper & waterfall generators
│   └── dashboard_utils.py                         # App caching, Plotly chart builders & predictors
│
├── models/
│   ├── best_model_pipeline.joblib                 # Serialized production pipeline
│   ├── preprocessor.joblib                        # Standalone preprocessor artifact
│   ├── model_comparison.csv                       # Multi-model test benchmark metrics
│   ├── evaluation_metrics.json                    # Detailed evaluation metric record
│   ├── targeting_strategies_comparison.csv        # Retention strategy financial P&L comparison
│   ├── business_simulation_results.json           # Retention simulation parameters & ROI metrics
│   ├── profit_curve.png                           # Financial net profit vs decision threshold
│   ├── cumulative_gains_curve.png                 # Cumulative churner capture vs targeted reach
│   ├── shap_summary.png                           # Global SHAP beeswarm summary plot
│   ├── shap_feature_importance.png                # Mean |SHAP| feature ranking bar chart
│   ├── shap_dependence_tenure.png                 # Non-linear tenure dependence curve
│   ├── shap_local_example.png                     # Individual customer waterfall explanation
│   ├── roc_curves.png                             # Candidate ROC curves comparison
│   ├── precision_recall_curves.png                # Precision-recall curves comparison
│   ├── confusion_matrices.png                     # Confusion matrices across candidate models
│   └── feature_importance.png                     # Gini feature importances plot
│
├── tests/
│   ├── test_modeling.py                           # Modeling & leakage prevention tests
│   ├── test_business_impact.py                    # Unit economics & financial simulation tests
│   ├── test_explainability.py                     # SHAP additivity & explanation tests
│   └── test_dashboard.py                          # Dashboard input schema & chart tests
│
├── requirements.txt                               # Pinned project dependencies
├── README.md                                      # Project documentation
└── .gitignore                                     # Git ignore rules
```

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sneha-Amballa/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app/main.py
```
*Open your browser and navigate to:* `http://localhost:8501`

### 5. Reproduce Model Training & Retention Analysis (Optional)
```bash
# Retrain candidate models and export best pipeline:
python src/train.py

# Run retention ROI optimization and export financial metrics:
python src/optimize_retention.py
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
