"""
train_model.py

Trains a classifier to predict short-term stock price direction using the
technical-indicator features built in features/build_features.py.

Key design choices (worth understanding, not just running):
    - Time-respecting train/test split. Shuffling would leak future
      information into training, which is a common and easy-to-miss mistake
      with time series data.
    - Random Forest as the default model: handles non-linear feature
      interactions well, gives feature importances for free, and doesn't
      need feature scaling. A reasonable starting point before reaching for
      anything more complex.
    - Also selectable: HistGradientBoostingClassifier (sklearn's built-in
      gradient boosting, no extra dependency) and XGBoost, both of which
      tend to outperform Random Forest on tabular data like this.
    - Backtest: converts the classifier's predictions into a simple trading
      signal (long when predicted up, flat otherwise) and compares
      cumulative returns against a buy-and-hold baseline. This is what
      turns "accuracy: 55%" into something interpretable.

Usage:
    python models/train_model.py --ticker AAPL
    python models/train_model.py --all   # train on combined multi-ticker data
    python models/train_model.py --all --model xgboost
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from xgboost import XGBClassifier

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features", "processed")
MODEL_OUT_DIR = os.path.join(os.path.dirname(__file__), "saved")

MODEL_TYPES = ["rf", "hist_gb", "xgboost"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "hist_gb": "Histogram Gradient Boosting",
    "xgboost": "XGBoost",
}


def build_model(model_type: str):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,   # guards against overfitting on noisy price data
            random_state=42,
        )
    elif model_type == "hist_gb":
        return HistGradientBoostingClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
        )
    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {MODEL_TYPES}.")


def get_feature_importances(model, X_test, y_test) -> pd.Series:
    """
    RandomForest and XGBoost expose feature_importances_ directly.
    HistGradientBoostingClassifier doesn't, so fall back to permutation
    importance (computed on the held-out test set) in that case.
    """
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    else:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = pd.Series(result.importances_mean, index=FEATURE_COLUMNS)
    return importances.sort_values(ascending=False)

FEATURE_COLUMNS = [
    "sma_10", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "daily_return", "volatility_10", "volume_change",
    "price_vs_sma50",
]

TEST_FRACTION = 0.2  # last 20% of the time series held out for testing


def load_data(ticker: str = None) -> pd.DataFrame:
    if ticker:
        path = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")
    else:
        path = os.path.join(FEATURES_DIR, "all_features.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run features/build_features.py first."
        )

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def time_based_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def backtest(test_df: pd.DataFrame, predictions: np.ndarray) -> dict:
    """
    Simple long/flat backtest:
    - If model predicts 1 (up), hold the stock for that day's return.
    - If model predicts 0 (down), stay in cash (0 return) that day.
    Compared against buy-and-hold over the same period.
    """
    strategy_returns = test_df["daily_return"].values * predictions
    buy_hold_returns = test_df["daily_return"].values

    strategy_cum = np.prod(1 + strategy_returns) - 1
    buy_hold_cum = np.prod(1 + buy_hold_returns) - 1

    return {
        "strategy_cumulative_return": strategy_cum,
        "buy_hold_cumulative_return": buy_hold_cum,
    }


def train(ticker: str = None, model_type: str = "rf"):
    df = load_data(ticker)
    train_df, test_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["direction"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["direction"]

    model = build_model(model_type)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)

    print("\n=== Classification performance (held-out test set) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    bt = backtest(test_df, preds)
    print("=== Backtest (long when predicted up, flat otherwise) ===")
    print(f"Strategy cumulative return:  {bt['strategy_cumulative_return']:.2%}")
    print(f"Buy-and-hold cumulative return: {bt['buy_hold_cumulative_return']:.2%}")

    importances = get_feature_importances(model, X_test, y_test)
    print("\n=== Feature importances ===")
    print(importances.to_string())

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    label = ticker if ticker else "combined"
    model_path = os.path.join(MODEL_OUT_DIR, f"model_{label}_{model_type}.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to {model_path}")

    return model, bt, importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None, help="Train on a single ticker's feature file")
    parser.add_argument("--all", action="store_true", help="Train on combined multi-ticker feature file")
    parser.add_argument(
        "--model", type=str, default="rf", choices=MODEL_TYPES,
        help="Which model to train: rf (Random Forest), hist_gb (Histogram Gradient Boosting), or xgboost",
    )
    args = parser.parse_args()

    if args.all or not args.ticker:
        train(ticker=None, model_type=args.model)
    else:
        train(ticker=args.ticker, model_type=args.model)


if __name__ == "__main__":
    main()
