"""
build_features.py

Reads raw OHLCV CSVs from data/raw/, computes technical indicators,
and writes labeled feature tables to features/processed/.

Technical indicators included (kept intentionally simple/interpretable):
    - Simple moving averages (SMA 10, 50)
    - Exponential moving average (EMA 12, 26)
    - RSI (14-day)
    - MACD + signal line
    - Rolling volatility (std dev of returns)
    - Volume change

Label:
    - direction: 1 if Close price N days ahead > Close today, else 0
      (N controlled by LABEL_HORIZON)

Usage:
    python features/build_features.py
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

LABEL_HORIZON = 5  # predict direction N trading days ahead


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def build_features_for_ticker(ticker: str) -> pd.DataFrame:
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    df = df.sort_index()

    close = df["Close"]

    # --- Trend indicators ---
    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- Momentum indicators ---
    df["rsi_14"] = compute_rsi(close, 14)
    macd_line, signal_line = compute_macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line

    # --- Volatility / volume ---
    df["daily_return"] = close.pct_change()
    df["volatility_10"] = df["daily_return"].rolling(10).std()
    df["volume_change"] = df["Volume"].pct_change()

    # --- Relative price position ---
    df["price_vs_sma50"] = close / df["sma_50"] - 1

    # --- Label: direction N days ahead ---
    future_close = close.shift(-LABEL_HORIZON)
    df["direction"] = (future_close > close).astype(int)

    df["ticker"] = ticker

    # Drop rows with NaNs from rolling windows / shifting
    df = df.dropna()

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]

    if not raw_files:
        print(f"No raw data found in {RAW_DIR}. Run data/fetch_data.py first.")
        return

    all_frames = []
    for f in raw_files:
        ticker = f.replace(".csv", "")
        print(f"Building features for {ticker}...")
        feat_df = build_features_for_ticker(ticker)
        out_path = os.path.join(OUT_DIR, f"{ticker}_features.csv")
        feat_df.to_csv(out_path)
        print(f"  Saved {len(feat_df)} rows to {out_path}")
        all_frames.append(feat_df)

    # Also save a combined file across all tickers, useful for training
    combined = pd.concat(all_frames)
    combined.to_csv(os.path.join(OUT_DIR, "all_features.csv"))
    print(f"Saved combined feature table with {len(combined)} rows.")


if __name__ == "__main__":
    main()
