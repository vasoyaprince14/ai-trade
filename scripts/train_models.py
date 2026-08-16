"""
Train ML Models
================
Convenience script to fetch historical data, build features,
and train all three model types (binary, 5class, regression)
for each configured symbol.

Usage:
  python scripts/train_models.py
  python scripts/train_models.py --symbol BANKNIFTY --days 500
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger
from core.ml.trainer import WalkForwardTrainer


def train_all(symbol: str = "NIFTY", days: int = 365, validate: bool = True):
    logger.info(f"=== Training ML models for {symbol} ({days} days) ===")

    for model_type in ["binary", "5class", "regression"]:
        logger.info(f"\n--- {model_type.upper()} ---")
        trainer = WalkForwardTrainer(
            symbol=symbol,
            model_type=model_type,
            train_days=max(500, days * 2 // 3),   # 5-min bars
            test_days=200,
        )

        if validate:
            from core.data.historical import fetch_historical
            from core.features.engineer import FeatureEngineer

            ohlcv = fetch_historical(symbol, "5min", days=days)
            if ohlcv.empty:
                logger.error(f"No data for {symbol}")
                continue

            feat_df = trainer.prepare_features(ohlcv)
            if feat_df.empty:
                logger.error("Feature engineering failed")
                continue

            logger.info(f"Feature matrix: {feat_df.shape}")

            # Walk-forward validation
            wf_results = trainer.walk_forward_validate(feat_df)
            if not wf_results.empty:
                print(f"\n{model_type} Walk-Forward Results:")
                print(wf_results.to_string(index=False))

            # Final full retrain
            model = trainer.train_and_save(feat_df)
        else:
            model = trainer.train_from_historical(days=days)

        if model:
            logger.info(f"{model_type} model saved.")

    logger.info(f"\n=== Training complete for {symbol} ===")


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Nifty trading")
    parser.add_argument("--symbol", default="NIFTY", help="Index symbol (NIFTY/BANKNIFTY/FINNIFTY)")
    parser.add_argument("--days",   type=int, default=365, help="Days of historical data")
    parser.add_argument("--no-validate", action="store_true", help="Skip walk-forward validation")
    args = parser.parse_args()

    symbols = [args.symbol]
    if args.symbol == "all":
        symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

    for sym in symbols:
        train_all(sym, args.days, validate=not args.no_validate)


if __name__ == "__main__":
    main()
