# Personal Finance Analytics & Intelligent Expense Prediction Dashboard

## Project overview
End-to-end personal finance analytics: data cleaning and feature engineering (Pandas, NumPy), SQL storage and queries (SQLite),
anomaly detection (Isolation Forest), expense forecasting (Linear, Ridge, Random Forest, Gradient Boosting), static and interactive
charts (Matplotlib, Seaborn, Plotly) and a Streamlit dashboard that non-technical users can use without code.

## Dataset
Kaggle: *Daily Household Transactions* (`heeraldedhia/daily-household-transactions`). Data source of this run: **SYNTHETIC fallback (not real data)**.
Rows after cleaning: 1,380 | period: 2022-01 to 2024-06.

## Setup
```
pip install -r requirements.txt
```

## How to run
1. Keep `app.py`, `finance.db`, `model.pkl`, `anomaly_model.pkl` and `metrics.json` in the same folder.
2. Start the dashboard: `streamlit run app.py`
3. To rebuild everything from scratch, run the Colab notebook top to bottom (only `kaggle.json` may need uploading).

## Folder structure
```
app.py                       Streamlit dashboard
finance.db                   SQLite database (transactions, monthly_summary, budget, anomalies)
model.pkl                    Best expense-prediction pipeline (joblib)
anomaly_model.pkl            Isolation Forest + scaler (joblib)
metrics.json                 Held-out metrics, comparison table, forecast, anomaly validation
processed_transactions.csv   Cleaned data with engineered features
monthly_summary.csv          Monthly aggregates
forecast.csv                 Next 3 months forecast
charts/                      EDA PNG charts and interactive Plotly HTML files
requirements.txt  README.md
```

## Results (this run)
- Chosen regression model: **Random Forest (tuned)** (selected by TimeSeriesSplit CV, evaluated on 5 held-out months)
- Held-out MAE 5,618 | RMSE 5,834 | R2 -1.78 | MAPE 26.4%
- Anomalies flagged: 41 of 1,350; 49% are genuinely unusual for their category (unflagged baseline 4%)
- Next-month forecast: 21,984 (2024-07)

## Limitations
Small monthly sample, no external drivers (income changes, holidays), recursive multi-step forecast, unsupervised anomaly labels.
