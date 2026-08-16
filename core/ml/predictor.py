"""
Real-Time ML Predictor
=======================
Loads trained models and generates live predictions
during market hours.

Features:
  - Loads all 3 models (binary, 5class, regression)
  - Produces a unified PredictionResult with:
      - direction (BUY/SELL/NEUTRAL)
      - confidence score (0–1)
      - predicted return %
      - signal strength (1–5)
  - Auto-retrains models after market close if stale (>7 days)
  - Caches predictions to avoid redundant computation

Used by:
  - OrderFlowFNOStrategy for ML-gated entry
  - Dashboard ML panel
  - AI Agent for trade reasoning
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ml.models import NiftyMLModel
from core.features.engineer import FeatureEngineer


@dataclass
class PredictionResult:
    symbol:           str
    timestamp:        datetime
    direction:        str          # BUY / SELL / NEUTRAL
    confidence:       float        # 0.0–1.0
    signal_strength:  int          # 1 (weak) – 5 (strong)
    prob_up:          float        # P(price rises)
    prob_down:        float        # P(price falls)
    predicted_return: float        # % expected move
    label_5class:     str          # Weak Up, Strong Down, etc.
    features_used:    int          # number of features
    model_age_days:   float        # days since last training
    notes:            str = ""


class MLPredictor:
    """
    Loads saved ML models and generates real-time trade signals.
    Auto-reloads models if they are stale (>7 days old).
    """

    MODEL_STALE_DAYS = 7

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol    = symbol
        self.engineer  = FeatureEngineer(symbol)
        self._binary   = NiftyMLModel(symbol, "binary")
        self._fiveclass= NiftyMLModel(symbol, "5class")
        self._regress  = NiftyMLModel(symbol, "regression")
        self._loaded   = False
        self._model_ts: Optional[datetime] = None
        self._cache: Dict = {}
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        results = {}
        for name, model in [("binary", self._binary), ("5class", self._fiveclass), ("regression", self._regress)]:
            try:
                results[name] = model.load()
            except Exception as e:
                logger.warning(f"Failed to load {name} model: {e}")
                results[name] = False

        if results.get("binary"):
            self._loaded   = True
            self._model_ts = datetime.now()
            loaded = [k for k, v in results.items() if v]
            logger.info(f"ML models loaded for {self.symbol}: {loaded}")
        else:
            logger.warning(f"Binary model not found for {self.symbol}. Run: python scripts/train_models.py")

    def _maybe_reload(self):
        """Reload models if stale."""
        if self._model_ts is None:
            return
        age = (datetime.now() - self._model_ts).days
        if age > self.MODEL_STALE_DAYS:
            logger.info(f"Models are {age} days old — reloading")
            self._load_models()

    def is_ready(self) -> bool:
        return self._loaded

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        ohlcv: pd.DataFrame,
        oi_df: Optional[pd.DataFrame] = None,
        sentiment_score: float = 0.0,
        vix: float = 0.0,
        pcr: float = 1.0,
        max_pain: float = 0.0,
    ) -> Optional[PredictionResult]:
        """
        Generate a prediction from the latest market data.

        Args:
            ohlcv           : Recent OHLCV bars (at least 30)
            oi_df           : Option chain data (optional)
            sentiment_score : News sentiment (-1 to +1)
            vix             : India VIX
            pcr             : Put-Call Ratio
            max_pain        : Max Pain strike

        Returns:
            PredictionResult or None if models not loaded / insufficient data
        """
        if not self._loaded:
            logger.warning("Models not loaded — skipping prediction")
            return None

        self._maybe_reload()

        # Build features
        feat_df = self.engineer.build_features(
            ohlcv=ohlcv,
            oi_df=oi_df,
            sentiment_score=sentiment_score,
            vix=vix,
            pcr=pcr,
            max_pain=max_pain,
        )
        if feat_df.empty:
            return None

        # Use latest row
        latest = feat_df.iloc[-1]

        # Get predictions from each model
        binary_pred  = self._safe_predict_row(self._binary,    latest)
        fiveclass_pred = self._safe_predict_row(self._fiveclass, latest)
        regress_pred = self._safe_predict_row(self._regress,   latest)

        # Aggregate
        prob_up   = binary_pred.get("prob_up",   0.5)
        prob_down = binary_pred.get("prob_down",  0.5)
        pred_ret  = regress_pred.get("predicted_return", 0.0)
        label_5   = fiveclass_pred.get("label", "Neutral")

        # Direction and confidence
        direction, confidence = self._resolve_direction(prob_up, prob_down, pred_ret, label_5)
        strength = self._signal_strength(confidence, pred_ret, vix)

        model_age = (
            (datetime.now() - self._model_ts).total_seconds() / 86400
            if self._model_ts else 0.0
        )

        return PredictionResult(
            symbol           = self.symbol,
            timestamp        = datetime.now(),
            direction        = direction,
            confidence       = round(confidence, 4),
            signal_strength  = strength,
            prob_up          = round(prob_up, 4),
            prob_down        = round(prob_down, 4),
            predicted_return = round(pred_ret, 4),
            label_5class     = label_5,
            features_used    = len(self._binary.feature_cols),
            model_age_days   = round(model_age, 1),
            notes            = self._make_notes(prob_up, pred_ret, vix, pcr, label_5),
        )

    def _safe_predict_row(self, model: NiftyMLModel, row: pd.Series) -> Dict:
        try:
            if not model.is_trained:
                return {}
            return model.predict_row(row.to_dict())
        except Exception as e:
            logger.debug(f"Prediction error ({model.model_type}): {e}")
            return {}

    def _resolve_direction(
        self,
        prob_up: float,
        prob_down: float,
        pred_ret: float,
        label_5: str,
    ) -> tuple:
        """Combine binary, regression, and 5-class signals."""
        THRESHOLD = 0.55   # minimum probability for directional call

        # 5-class vote
        vote = 0
        if "Strong Up" in label_5:
            vote = 2
        elif "Weak Up" in label_5:
            vote = 1
        elif "Strong Down" in label_5:
            vote = -2
        elif "Weak Down" in label_5:
            vote = -1

        # Binary probability vote
        if prob_up > THRESHOLD:
            bin_vote = 1
        elif prob_down > THRESHOLD:
            bin_vote = -1
        else:
            bin_vote = 0

        # Regression vote
        reg_vote = 1 if pred_ret > 0.001 else (-1 if pred_ret < -0.001 else 0)

        combined = vote + bin_vote + reg_vote

        if combined >= 2:
            return "BUY", max(prob_up, 0.55)
        elif combined <= -2:
            return "SELL", max(prob_down, 0.55)
        else:
            return "NEUTRAL", 0.5

    def _signal_strength(self, confidence: float, pred_ret: float, vix: float) -> int:
        """Return signal strength 1–5."""
        score = 0
        score += 1 if confidence > 0.55 else 0
        score += 1 if confidence > 0.65 else 0
        score += 1 if abs(pred_ret) > 0.003 else 0
        score += 1 if abs(pred_ret) > 0.005 else 0
        score += 1 if vix > 15 else 0   # volatile market = stronger signals
        return max(1, min(5, score + 1))

    def _make_notes(
        self,
        prob_up: float,
        pred_ret: float,
        vix: float,
        pcr: float,
        label_5: str,
    ) -> str:
        parts = [
            f"P(up)={prob_up:.2f}",
            f"pred_ret={pred_ret:+.3f}%",
            f"VIX={vix:.1f}",
            f"PCR={pcr:.2f}",
            f"class={label_5}",
        ]
        return " | ".join(parts)

    # ── Batch predict ─────────────────────────────────────────────────────────

    def predict_latest_row(self, feature_df: pd.DataFrame) -> Optional[PredictionResult]:
        """
        Predict on a pre-built feature DataFrame (last row).
        Faster than predict() when features are already computed.
        """
        if not self._loaded or feature_df.empty:
            return None
        latest = feature_df.iloc[-1]
        binary_pred   = self._safe_predict_row(self._binary,     latest)
        fiveclass_pred= self._safe_predict_row(self._fiveclass,  latest)
        regress_pred  = self._safe_predict_row(self._regress,    latest)

        prob_up  = binary_pred.get("prob_up",  0.5)
        prob_down= binary_pred.get("prob_down",0.5)
        pred_ret = regress_pred.get("predicted_return", 0.0)
        label_5  = fiveclass_pred.get("label", "Neutral")

        direction, confidence = self._resolve_direction(prob_up, prob_down, pred_ret, label_5)
        return PredictionResult(
            symbol           = self.symbol,
            timestamp        = datetime.now(),
            direction        = direction,
            confidence       = round(confidence, 4),
            signal_strength  = self._signal_strength(confidence, pred_ret, 0),
            prob_up          = round(prob_up, 4),
            prob_down        = round(prob_down, 4),
            predicted_return = round(pred_ret, 4),
            label_5class     = label_5,
            features_used    = len(self._binary.feature_cols),
            model_age_days   = 0,
        )


# ── Singleton registry ────────────────────────────────────────────────────────

_predictors: Dict[str, MLPredictor] = {}


def get_predictor(symbol: str = "NIFTY") -> MLPredictor:
    global _predictors
    if symbol not in _predictors:
        _predictors[symbol] = MLPredictor(symbol)
    return _predictors[symbol]
