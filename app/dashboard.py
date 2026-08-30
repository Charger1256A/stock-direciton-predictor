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

from models.train_model import (
    FEATURE_COLUMNS, MODEL_TYPES, MODEL_LABELS, time_based_split, backtest,
    get_feature_importances, FEATURES_DIR, MODEL_OUT_DIR,
)

MODEL_DESCRIPTIONS = {
    "rf": (
        "**Random Forest** builds many decision trees on random subsets of the data "
        "and features, then averages their votes. Robust to noisy data, captures "
        "non-linear patterns, and doesn't need feature scaling."
    ),
    "hist_gb": (
        "**Histogram Gradient Boosting** builds decision trees one at a time, each "
        "one correcting the errors of the previous, using histogram-binned features "
        "for speed. Often more accurate than a single Random Forest, but more prone "
        "to overfitting if not tuned carefully."
    ),
    "xgboost": (
        "**XGBoost** is an optimized gradient boosting implementation that builds "
        "trees sequentially with built-in regularization to reduce overfitting. "
        "A popular choice for tabular data like this."
    ),
}

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

col_ticker, col_model = st.columns(2)
ticker = col_ticker.selectbox("Ticker", available)
model_type = col_model.selectbox(
    "Model", MODEL_TYPES, format_func=lambda m: MODEL_LABELS[m]
)
st.caption(MODEL_DESCRIPTIONS[model_type])

# --- Load data + model ---
df = pd.read_csv(os.path.join(FEATURES_DIR, f"{ticker}_features.csv"), index_col="Date", parse_dates=True)
df = df.sort_index()

model_path = os.path.join(MODEL_OUT_DIR, f"model_{ticker}_{model_type}.joblib")
combined_model_path = os.path.join(MODEL_OUT_DIR, f"model_combined_{model_type}.joblib")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    model_label = f"per-ticker {MODEL_LABELS[model_type]} model ({ticker})"
elif os.path.exists(combined_model_path):
    model = joblib.load(combined_model_path)
    model_label = f"combined multi-ticker {MODEL_LABELS[model_type]} model"
else:
    status = st.empty()
    status.info(f"No saved {MODEL_LABELS[model_type]} model found — training one now. This may take a moment...")
    with st.spinner(f"Training {MODEL_LABELS[model_type]} on the combined multi-ticker dataset..."):
        from models.train_model import train
        train(ticker=None, model_type=model_type)
    status.success(f"{MODEL_LABELS[model_type]} model trained and saved.")
    model = joblib.load(combined_model_path)
    model_label = f"combined multi-ticker {MODEL_LABELS[model_type]} model"

st.caption(f"Using {model_label}")

# --- Predictions on test split ---
_, test_df = time_based_split(df)
X_test = test_df[FEATURE_COLUMNS]
preds = model.predict(X_test)
test_df = test_df.copy()
test_df["predicted_direction"] = preds

# --- Price chart with predicted signal ---
st.subheader(
    f"{ticker} — Price with Predicted Signal",
    help=(
        "Top chart: closing price over the test period. "
        "Bottom chart: the model's predicted direction for each day "
        "(1 = predicted price goes up, 0 = predicted flat/down)."
    ),
)
chart_df = test_df[["Close"]].copy()
chart_df["Signal (up=1)"] = test_df["predicted_direction"]
st.line_chart(chart_df["Close"])
st.bar_chart(chart_df["Signal (up=1)"])

# --- Backtest ---
st.subheader(
    "Backtest: Strategy vs. Buy-and-Hold",
    help=(
        "**Strategy return**: cumulative return from only holding the stock on "
        "days the model predicted 'up', staying in cash the rest of the time.\n\n"
        "**Buy & Hold return**: cumulative return from holding the stock every "
        "day of the test period, ignoring the model entirely.\n\n"
        "Comparing the two shows whether the model's predictions would have "
        "added value over doing nothing extra."
    ),
)
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
col1.metric(
    "Strategy cumulative return",
    f"{bt['strategy_cumulative_return']:.2%}",
    help="Return from holding the stock only on days the model predicted 'up', flat (cash) otherwise.",
)
col2.metric(
    "Buy-and-hold cumulative return",
    f"{bt['buy_hold_cumulative_return']:.2%}",
    help="Return from holding the stock every day of the test period, regardless of the model's predictions.",
)

# --- Feature importances ---
st.subheader(
    "Feature Importances",
    help=(
        "How much each technical indicator contributed to the model's "
        "predictions. Higher bars mean the model relied on that feature more "
        "when deciding whether the price would go up."
    ),
)
with st.spinner("Computing feature importances..."):
    importances = get_feature_importances(model, X_test, test_df["direction"])
st.bar_chart(importances)

st.caption(
    "Note: backtested performance on historical data is not a reliable "
    "predictor of future returns, and it's easy to accidentally overfit "
    "to the specific test window shown here."
)
