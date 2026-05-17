# Walmart Demand Forecasting & Decision Support System

End-to-end machine learning project to forecast weekly Walmart sales at Store-Department level and support inventory/staffing decisions through an interactive Streamlit dashboard.

## Problem Statement
Retail demand is highly sensitive to promotions, holidays, store profile, and macro factors. This project predicts weekly sales and turns those forecasts into decision-ready insights for operations planning.

## What This Project Does
- Preprocesses and merges Walmart historical datasets (`train`, `test`, `features`, `stores`).
- Trains a LightGBM regression model for weekly sales prediction.
- Benchmarks model performance against a 4-week moving average baseline.
- Computes forecast uncertainty intervals and model metrics.
- Generates business insights and feature-importance outputs (including SHAP).
- Provides a Streamlit app for KPI tracking and what-if scenario simulation.

## Tech Stack
- Python, Pandas, NumPy
- LightGBM, scikit-learn
- Matplotlib
- Streamlit

## Project Structure
```text
.
├── app/
│   └── app.py
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── real_time_simulation.py
├── data/
│   ├── train.csv, test.csv, features.csv, stores.csv
│   └── generated artifacts (models, metrics, predictions, insights)
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Run Pipeline
```bash
python src/preprocessing.py
python src/train_model.py
python src/real_time_simulation.py
streamlit run app/app.py
```

## Key Outputs
- `data/lgb_model.pkl` (trained model)
- `data/model_metrics.json` (RMSE/WMAPE, baseline comparison, interval stats)
- `data/test_predictions.csv` (forecast horizon predictions)
- `data/validation_predictions.csv` (validation outputs)
- `data/business_insights.md` and `data/business_insights.json` (auto insights)
- `data/feature_importance_shap.png` and `.csv` (feature drivers)

## Resume Line
Built an end-to-end LightGBM + Streamlit solution to forecast weekly store-department sales with scenario analysis and dashboard-driven inventory/staffing decisions.
