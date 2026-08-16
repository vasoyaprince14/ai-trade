"""
Walk-Forward ML Trainer
========================
Trains the NiftyMLModel using a walk-forward (anchored or rolling)
validation scheme to avoid lookahead bias.

Walk-forward scheme:
  - Minimum 90 days of training data
  - Retrain every 30 days
  - Test on next 30-day window
  - Record metrics per fold

Also provides:
  - Full retrain on all available data (for live deployment)
  - Auto-saves the best model checkpoint
  - Training report as DataFrame

Inspired by Qlib's rolling refit mechanism.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ml.models import NiftyMLModel
from core.features.engineer import FeatureEngineer


class WalkForwardTrainer:
    """
    Trains and validates ML models using walk-forward methodology.
    Prevents future data leakage by never training on test period data.
    """

    def __init__(
        self,
        symbol: str = "NIFTY",
        model_type: str = "binary",
        train_days: int = 90,
        test_days: int = 30,
        rolling: bool = False,
    ):
        """
        Args:
            symbol     : NSE index symbol
            model_type : 'binary' | '5class' | 'regression'
            train_days : Minimum bars per training fold (5-min bars)
            test_days  : Bars per test fold
            rolling    : True = rolling window, False = expanding (anchored)
        """
        self.symbol     = symbol
        self.model_type = model_type
        self.train_days = train_days
        self.test_days  = test_days
        self.rolling    = rolling
        self.engineer   = FeatureEngineer(symbol)

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_features(
        self,
        ohlcv: pd.DataFrame,
        oi_df: Optional[pd.DataFrame] = None,
        sentiment_score: float = 0.0,
        vix: float = 0.0,
        pcr: float = 1.0,
        max_pain: float = 0.0,
    ) -> pd.DataFrame:
        """Build feature matrix from raw data."""
        return self.engineer.build_features(
            ohlcv=ohlcv,
            oi_df=oi_df,
            sentiment_score=sentiment_score,
            vix=vix,
            pcr=pcr,
            max_pain=max_pain,
        )

    # ── Walk-forward ──────────────────────────────────────────────────────────

    def walk_forward_validate(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run walk-forward cross-validation.

        Returns:
            DataFrame with per-fold metrics (train_acc, val_acc, etc.)
        """
        feature_cols = FeatureEngineer.get_feature_cols(feature_df)
        label_col    = {
            "binary":     "label_binary",
            "5class":     "label_5class",
            "regression": "target_return",
        }[self.model_type]

        data = feature_df[feature_cols + [label_col]].dropna()
        n    = len(data)

        if n < self.train_days + self.test_days:
            logger.error(f"Not enough data for walk-forward: {n} rows (need {self.train_days + self.test_days})")
            return pd.DataFrame()

        folds = []
        start = 0

        while start + self.train_days + self.test_days <= n:
            train_end   = start + self.train_days
            test_end    = train_end + self.test_days

            train_data  = data.iloc[start:train_end]
            test_data   = data.iloc[train_end:test_end]

            X_train     = train_data[feature_cols].values
            y_train     = train_data[label_col].values
            X_test      = test_data[feature_cols].values
            y_test      = test_data[label_col].values

            model = NiftyMLModel(self.symbol, self.model_type)
            fold_metrics = model.train(
                pd.concat([train_data]),
                feature_cols=feature_cols,
                val_frac=0.0,   # no internal val split during walk-forward
            )

            # Evaluate on test set
            fold_result = {
                "fold":      len(folds) + 1,
                "train_start": int(start),
                "train_end":   int(train_end),
                "test_start":  int(train_end),
                "test_end":    int(test_end),
                "train_size":  int(len(X_train)),
                "test_size":   int(len(X_test)),
            }

            if self.model_type == "regression":
                from sklearn.metrics import mean_absolute_error, r2_score
                pred = model.predict(X_test)
                fold_result["test_mae"] = round(float(mean_absolute_error(y_test, pred)), 6)
                fold_result["test_r2"]  = round(float(r2_score(y_test, pred)), 4)
            else:
                from sklearn.metrics import accuracy_score
                pred = model.predict(X_test)
                fold_result["test_acc"] = round(float(accuracy_score(y_test, pred)), 4)
                # Direction profit factor (if binary)
                if self.model_type == "binary" and "target_return" in test_data.columns:
                    correct_mask = (pred == y_test)
                    fwd_ret      = test_data["target_return"].values
                    profit       = fwd_ret[correct_mask].sum()
                    loss         = abs(fwd_ret[~correct_mask].sum())
                    fold_result["profit_factor"] = round(profit / (loss + 1e-9), 3)

            folds.append(fold_result)
            logger.info(f"Fold {len(folds)}: {fold_result}")

            # Advance window
            step = self.test_days
            if self.rolling:
                start += step
            else:
                start = 0   # anchored: always start from beginning
                self.train_days += step   # grow training set

        results = pd.DataFrame(folds)
        if len(results):
            logger.info(f"\nWalk-forward summary ({len(results)} folds):")
            if "test_acc" in results.columns:
                logger.info(f"  Mean test accuracy: {results['test_acc'].mean():.4f}")
            if "test_mae" in results.columns:
                logger.info(f"  Mean test MAE: {results['test_mae'].mean():.6f}")
        return results

    # ── Full retrain (for live use) ───────────────────────────────────────────

    def train_and_save(self, feature_df: pd.DataFrame) -> NiftyMLModel:
        """
        Train on ALL available data and save to disk.
        Use this for generating live predictions.
        """
        model = NiftyMLModel(self.symbol, self.model_type)
        metrics = model.train(feature_df, val_frac=0.1)
        path = model.save()
        logger.info(f"Full retrain complete. Saved to {path}")
        logger.info(f"Metrics: {metrics}")
        return model

    # ── Convenience: train from historical data ───────────────────────────────

    def train_from_historical(self, days: int = 365) -> NiftyMLModel:
        """
        Fetch historical OHLCV, build features, train, and save model.
        Convenience method for one-line training.
        """
        from core.data.historical import fetch_historical

        logger.info(f"Fetching {days} days of historical data for {self.symbol}...")
        ohlcv = fetch_historical(self.symbol, "5min", days=days)
        if ohlcv.empty:
            logger.error("No historical data available")
            return None

        logger.info(f"Building features for {len(ohlcv)} bars...")
        feature_df = self.prepare_features(ohlcv)
        if feature_df.empty:
            logger.error("Feature engineering failed")
            return None

        logger.info(f"Feature matrix: {feature_df.shape}")
        return self.train_and_save(feature_df)
