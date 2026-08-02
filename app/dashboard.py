"""
dashboard.py

Minimal Streamlit dashboard to visualize:
    - Recent price action for a chosen ticker
    - The model's predicted signal over the test period
    - Backtest performance vs. buy-and-hold
    - Feature importances

Run from the project root with:
    streamlit run app/dashboard.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# allow importing from models/ when run via `streamlit run app/dashboard.py`
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.train_model import FEATURE_COLUMNS, time_based_split, backtest, FEATURES_DIR, MODEL_OUT_DIR

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")
st.title("Stock Price Direction Predictor")
st.caption(
    "Predicts short-term price direction from technical indicators. "
    "Built as a learning project — not investment advice, and not a "
    "strategy anyone should actually trade on."
)

# --- Ticker selection ---
available = [
    f.replace("_features.csv", "")
    for f in os.listdir(FEATURES_DIR)
    if f.endswith("_features.csv") and f != "all_features.csv"
]

if not available:
    st.error("No processed feature files found. Run fetch_data.py and build_features.py first.")
    st.stop()

ticker = st.selectbox("Ticker", available)

# --- Load data + model ---
df = pd.read_csv(os.path.join(FEATURES_DIR, f"{ticker}_features.csv"), index_col="Date", parse_dates=True)
df = df.sort_index()

model_path = os.path.join(MODEL_OUT_DIR, f"model_{ticker}.joblib")
combined_model_path = os.path.join(MODEL_OUT_DIR, "model_combined.joblib")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    model_label = f"per-ticker model ({ticker})"
elif os.path.exists(combined_model_path):
    model = joblib.load(combined_model_path)
    model_label = "combined multi-ticker model"
else:
    st.error("No trained model found. Run models/train_model.py first.")
    st.stop()

st.caption(f"Using {model_label}")

# --- Predictions on test split ---
_, test_df = time_based_split(df)
X_test = test_df[FEATURE_COLUMNS]
preds = model.predict(X_test)
test_df = test_df.copy()
test_df["predicted_direction"] = preds

# --- Price chart with predicted signal ---
st.subheader(f"{ticker} — Price with Predicted Signal")
chart_df = test_df[["Close"]].copy()
chart_df["Signal (up=1)"] = test_df["predicted_direction"]
st.line_chart(chart_df["Close"])
st.bar_chart(chart_df["Signal (up=1)"])

# --- Backtest ---
st.subheader("Backtest: Strategy vs. Buy-and-Hold")
bt = backtest(test_df, preds)

strategy_returns = test_df["daily_return"].values * preds
buy_hold_returns = test_df["daily_return"].values

strategy_cum_series = (1 + pd.Series(strategy_returns, index=test_df.index)).cumprod() - 1
buy_hold_cum_series = (1 + pd.Series(buy_hold_returns, index=test_df.index)).cumprod() - 1

perf_df = pd.DataFrame({
    "Strategy": strategy_cum_series,
    "Buy & Hold": buy_hold_cum_series,
})
st.line_chart(perf_df)

col1, col2 = st.columns(2)
col1.metric("Strategy cumulative return", f"{bt['strategy_cumulative_return']:.2%}")
col2.metric("Buy-and-hold cumulative return", f"{bt['buy_hold_cumulative_return']:.2%}")

# --- Feature importances ---
st.subheader("Feature Importances")
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
st.bar_chart(importances)

st.caption(
    "Note: backtested performance on historical data is not a reliable "
    "predictor of future returns, and it's easy to accidentally overfit "
    "to the specific test window shown here."
)
