# Customer Churn Prediction & Explainability Pipeline 

## Project Overview
This project implements an end-to-end, leakage-safe customer churn prediction system using ride-history data. It covers the full machine learning lifecycle — from feature engineering and churn labeling to model training, evaluation, explainability, and deployment via an interactive Streamlit app.

The goal is to identify high-risk customers early and provide interpretable insights that can support data-driven retention strategies. 

## Key Features
- Leakage-safe churn labeling using temporal windows
- Robust feature engineering from historical ride data
- Modeling on highly imbalanced data
- Interpretable machine learning with SHAP
- Deployment via Streamlit for interactive scoring and analysis

## Project Structure
```
churn_prediction_uber/
│
├── data/
│   ├── ncr_ride_bookings.csv          # Raw ride-level dataset
│   ├── churn_dataset.pkl              # Processed modeling dataset
│   └── churn_dataset.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and data understanding
│   ├── 02_feature_building.ipynb       # Leakage-safe labeling + features
│   └── 03_modeling_explainability_FIXED.ipynb
│
├── models/
│   ├── logit_pipeline.pkl              # Trained Logistic Regression model
│   └── xgb_pipeline.pkl                # Trained XGBoost model
│
├── app/
│   └── streamlit_app.py                # Interactive Streamlit dashboard
│
├── src/
│   ├── predict.py                      # Scoring & inference utilities
│   └── utils.py                        # Shared helper functions
│
├── notebooks/retention_list_top_decile.csv
│
├── requirements.txt
└── README.md

```

## Data Description
Source: Ride booking data containing timestamps, customer IDs, trip outcomes, distances, ratings, and booking values. 

Key fields used:

- Customer ID
- Ride date & time
- Booking status (completed, cancelled, incomplete)
- Ride distance
- Booking value
- Customer & driver ratings
- Vehicle type 

### Churn Definition (Leakage-Safe) 
A customer is labeled as churned if:
- They had no completed rides during a defined future window (HORIZON_D)
- Given sufficient activity during a historical lookback window (LOOKBACK_D)

This ensures:
- No future information leaks into training
- Labels reflect realistic business usage

### Feature Engineering 
Features are aggregated per customer from historical data, including:
- Ride frequency and recency
- Time-of-day ride behavior
- Cancellation and completion patterns
- Average distance and spend
- Rating statistics
- Vehicle usage patterns
Only past data is used for feature creation.

### Models
Two complementary models are trained:

Logistic Regression
- Baseline, interpretable linear model
- Handles class imbalance via class weights
- Useful for calibration and explainability

XGBoost
- Non-linear tree-based model
- Captures complex behavioral patterns
- Tuned using cross-validation and threshold optimization
Both models are evaluated using ROC-AUC, PR-AUC, F1, and business-aligned lift metrics.

### Model Explainability
SHAP (SHapley Additive exPlanations) is used to:
- Identify key drivers of churn
- Explain individual predictions
- Provide transparency for downstream decision-making
This makes the models suitable for real-world deployment and stakeholder trust.

### Streamlit Application
The Streamlit app allows users to:
- Upload customer data
- Score churn risk using trained models
- View key feature drivers
- Export top-risk customer lists for retention action

### To run the app locally

source .venv/bin/activate
streamlit run app/streamlit_app.py 

### Reproducibility
To regenerate all artifacts end-to-end:

source .venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/02_feature_building.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling_explainability_FIXED.ipynb 

This will regenerate:
- Trained model pipelines
- Retention lists
- Evaluation outputs

### Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- Jupyter

### Key Takeaways
- Demonstrates production-ready ML workflow
- Emphasizes data leakage prevention
- Balances predictive performance with interpretability
- Designed with real retention use-cases in mind

### Author 
Jenish Simkhada
Individual project — designed, implemented, and deployed end-to-end. 