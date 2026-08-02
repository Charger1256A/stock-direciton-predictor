# Stock Direction Predictor

A small ML pipeline that predicts short-term stock price direction (up/down
over the next N trading days) using technical indicators, with a backtest
comparing the resulting signal against buy-and-hold, and a Streamlit
dashboard to visualize it all.

This is a learning/portfolio project, not a trading strategy. Backtested
performance on historical data does not reliably predict future returns,
and short lookback windows are easy to overfit to by accident.

## Project structure

```
stock-predictor/
  data/
    fetch_data.py       # pulls OHLCV data via yfinance -> data/raw/
  features/
    build_features.py   # computes technical indicators -> features/processed/
  models/
    train_model.py       # trains + evaluates + backtests -> models/saved/
  app/
    dashboard.py          # Streamlit UI
  requirements.txt
```

## Setup

```bash
cd stock-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
# 1. Pull historical price data (edit TICKERS in fetch_data.py to change tickers)
python data/fetch_data.py

# 2. Build technical indicator features + labels
python features/build_features.py

# 3. Train a model and see backtest results in your terminal
python models/train_model.py --all
# or, for a single ticker:
python models/train_model.py --ticker AAPL

# 4. Launch the dashboard
streamlit run app/dashboard.py
```

## What's actually going on here

- **Labels**: `direction` = 1 if the close price N days ahead (default 5)
  is higher than today's close, else 0. This is a classification problem,
  not a price-forecasting one — deliberately, since predicting exact
  future prices from price history alone is not a solvable problem, and
  claiming otherwise is a common tell in weaker versions of this project.
- **Features**: SMA/EMA, RSI, MACD, rolling volatility, volume change,
  price relative to its 50-day average. All computed from OHLCV only, no
  external data sources needed to get started.
- **Split**: time-based, not random. The test set is always the most
  recent slice of the series — shuffling would leak future data into
  training.
- **Backtest**: converts predictions into a simple long/flat signal and
  compares cumulative returns to buy-and-hold over the same window. This
  is the part that turns a bare accuracy number into something you can
  actually reason about and explain.

## Ideas to extend it (pick 1-2, don't do all of them)

- Swap Random Forest for XGBoost/LightGBM and compare.
- Add a walk-forward validation loop instead of a single train/test split,
  to see how stable performance is across different time windows.
- Add sentiment features from news headlines or earnings call transcripts.
- Try predicting magnitude (regression) instead of just direction, and see
  how much harder/noisier that is.
- Add position sizing to the backtest (e.g., scale position by prediction
  confidence) instead of a flat long/flat signal.
- Deploy the dashboard (Streamlit Community Cloud is free and easy) so you
  have a live link to put on your resume/LinkedIn.

## Suggested resume bullet (fill in your actual numbers once you've run it)

> Built an end-to-end ML pipeline to predict short-term stock price
> direction from technical indicators (RSI, MACD, moving averages),
> achieving X% backtested cumulative return vs. Y% buy-and-hold across
> Z tickers; included a Streamlit dashboard for interactive backtesting.
