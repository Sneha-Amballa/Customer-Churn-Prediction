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
│   └── 04_feature_engineering.ipynb                # Phase 4: Behavioral & structural feature generation
│
├── src/                                           # Reusable modular code
├── models/                                        # Trained model binaries & evaluation artifacts
├── app/                                           # Interactive dashboard / deployment app
├── tests/                                         # Unit and integration test suite
├── requirements.txt                               # Project dependencies
├── README.md                                      # Documentation
└── .gitignore                                     # Ignored files
```

## Dataset Overview

- **Source:** IBM Sample Data Sets (Telco Customer Churn)
- **Rows:** 7,043 customers
- **Features:** 21 attributes spanning demographics, account data, services, and billing
- **Target:** `Churn` (Yes: ~26.5%, No: ~73.5%)

## Project Roadmap

- **Phase 1: Project Setup & Dataset Understanding** (Completed)
- **Phase 2: Data Cleaning & Integrity** (Completed)
- **Phase 3: Business-Focused EDA** (Completed)
- **Phase 4: Feature Engineering** (Completed)
- **Phase 5: ML Modeling & Evaluation** (Upcoming)
