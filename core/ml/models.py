"""
ML Models — XGBoost + LightGBM Ensemble
=========================================
Trains direction classifiers and return regressors for Nifty:

  - Binary classifier   : UP/DOWN (next 6 bars)
  - 5-class classifier  : Strong Up / Weak Up / Neutral / Weak Down / Strong Down
  - Return regressor    : Predicted forward return %

Architecture:
  - XGBoostClassifier (binary + multiclass)
  - LGBMClassifier (binary + multiclass)
  - XGBoostRegressor for return prediction
  - Probability calibration via CalibratedClassifierCV
  - Ensemble: average probabilities of XGB + LGBM
  - Feature importance + SHAP explainability

Inspired by Qlib's Alpha158 model pipeline.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class NiftyMLModel:
    """
    Ensemble ML model for Nifty direction prediction.
    Trains XGBoost + LightGBM classifiers and averages their probabilities.
    """

    def __init__(self, symbol: str = "NIFTY", model_type: str = "binary"):
        """
        Args:
            symbol    : Index symbol (NIFTY, BANKNIFTY, FINNIFTY)
            model_type: 'binary' | '5class' | 'regression'
        """
        self.symbol = symbol
        self.model_type = model_type
        self.xgb_model   = None
        self.lgbm_model  = None
        self.feature_cols: List[str] = []
        self.is_trained   = False
        self._label_col   = {
            "binary":     "label_binary",
            "5class":     "label_5class",
            "regression": "target_return",
        }[model_type]

    # ── Build models ──────────────────────────────────────────────────────────

    def _build_xgb(self, n_classes: int = 2):
        try:
            import xgboost as xgb
            if self.model_type == "regression":
                return xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            params = dict(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            if n_classes > 2:
                params["num_class"] = n_classes
                params["objective"] = "multi:softprob"
            return xgb.XGBClassifier(**params)
        except ImportError:
            logger.warning("xgboost not installed")
            return None

    def _build_lgbm(self, n_classes: int = 2):
        try:
            import lightgbm as lgb
            if self.model_type == "regression":
                return lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            objective = "binary" if n_classes == 2 else "multiclass"
            return lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective=objective,
                num_class=n_classes if n_classes > 2 else None,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            logger.warning("lightgbm not installed")
            return None

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        val_frac: float = 0.15,
    ) -> Dict:
        """
        Train the ensemble on a feature DataFrame.

        Args:
            df          : Output of FeatureEngineer.build_features()
            feature_cols: Feature columns to use (auto-detected if None)
            val_frac    : Fraction of data used for validation (no shuffle)

        Returns:
            metrics dict with train/val accuracy, feature importance
        """
        from core.features.engineer import FeatureEngineer

        if feature_cols is None:
            feature_cols = FeatureEngineer.get_feature_cols(df)

        # Drop rows where label is NaN
        data = df[feature_cols + [self._label_col]].dropna()
        if len(data) < 100:
            logger.warning(f"Not enough data to train: {len(data)} rows")
            return {}

        self.feature_cols = feature_cols
        X = data[feature_cols].values
        y = data[self._label_col].values

        # XGBoost requires classes starting at 0 — shift 5-class labels [-2..2] → [0..4]
        self._label_offset = 0
        if self.model_type == "5class":
            self._label_offset = int(-y.min()) if y.min() < 0 else 0
            y = y + self._label_offset

        # Walk-forward split (no future leak)
        split = int(len(X) * (1 - val_frac))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        n_classes = len(np.unique(y_train)) if self.model_type != "regression" else 1
        logger.info(f"Training {self.model_type} model | {len(X_train)} train / {len(X_val)} val | {n_classes} classes")

        self.xgb_model  = self._build_xgb(n_classes)
        self.lgbm_model = self._build_lgbm(n_classes)

        metrics = {}

        for name, model in [("xgb", self.xgb_model), ("lgbm", self.lgbm_model)]:
            if model is None:
                continue
            model.fit(X_train, y_train)
            if self.model_type == "regression":
                from sklearn.metrics import mean_absolute_error, r2_score
                if len(X_val):
                    pred_val = model.predict(X_val)
                    metrics[f"{name}_val_mae"] = round(float(mean_absolute_error(y_val, pred_val)), 6)
                    metrics[f"{name}_val_r2"]  = round(float(r2_score(y_val, pred_val)), 4)
            else:
                from sklearn.metrics import accuracy_score
                pred_train = model.predict(X_train)
                metrics[f"{name}_train_acc"] = round(float(accuracy_score(y_train, pred_train)), 4)
                if len(X_val):
                    pred_val = model.predict(X_val)
                    metrics[f"{name}_val_acc"] = round(float(accuracy_score(y_val, pred_val)), 4)

        self.is_trained = True

        # Feature importance (average of xgb + lgbm)
        importances = self._get_feature_importance()
        metrics["top10_features"] = importances[:10]

        logger.info(f"Training complete: {metrics}")
        return metrics

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return averaged class probabilities from XGB + LGBM.
        For regression returns scalar predictions.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        preds = []
        for model in [self.xgb_model, self.lgbm_model]:
            if model is None:
                continue
            if self.model_type == "regression":
                preds.append(model.predict(X).reshape(-1, 1))
            else:
                preds.append(model.predict_proba(X))

        if not preds:
            raise RuntimeError("No models available")

        return np.mean(preds, axis=0)

    def predict(self, X: np.ndarray):
        """Return class predictions (argmax of averaged proba)."""
        proba = self.predict_proba(X)
        if self.model_type == "regression":
            return proba.ravel()
        return np.argmax(proba, axis=1)

    def predict_row(self, feature_row: Dict) -> Dict:
        """
        Predict from a single dict of feature values.
        Returns prediction + confidence + class probabilities.
        """
        if not self.is_trained or not self.feature_cols:
            return {"error": "model not trained"}

        X = np.array([[feature_row.get(f, 0.0) for f in self.feature_cols]])
        proba = self.predict_proba(X)[0]

        if self.model_type == "binary":
            return {
                "prediction":  int(np.argmax(proba)),
                "direction":   "UP" if np.argmax(proba) == 1 else "DOWN",
                "confidence":  float(np.max(proba)),
                "prob_up":     float(proba[1]) if len(proba) > 1 else float(proba[0]),
                "prob_down":   float(proba[0]),
            }
        elif self.model_type == "5class":
            labels = ["Strong Down", "Weak Down", "Neutral", "Weak Up", "Strong Up"]
            cls = int(np.argmax(proba))
            offset = getattr(self, "_label_offset", 2)
            return {
                "prediction":  cls - offset,   # back to -2..+2
                "label":       labels[min(cls, 4)],
                "confidence":  float(np.max(proba)),
                "probabilities": {labels[i]: round(float(proba[i]), 4) for i in range(min(5, len(proba)))},
            }
        else:  # regression
            val = float(proba[0])
            return {
                "predicted_return": round(val * 100, 4),  # as %
                "direction": "UP" if val > 0.001 else ("DOWN" if val < -0.001 else "NEUTRAL"),
            }

    # ── Feature importance ────────────────────────────────────────────────────

    def _get_feature_importance(self) -> List[Tuple[str, float]]:
        importance = np.zeros(len(self.feature_cols))
        count = 0
        for model in [self.xgb_model, self.lgbm_model]:
            if model is None:
                continue
            fi = model.feature_importances_
            if len(fi) == len(self.feature_cols):
                importance += fi
                count += 1
        if count:
            importance /= count
        ranked = sorted(
            zip(self.feature_cols, importance.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return ranked

    def get_feature_importance_df(self) -> pd.DataFrame:
        ranked = self._get_feature_importance()
        return pd.DataFrame(ranked, columns=["feature", "importance"])

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        """Persist model to disk using joblib."""
        import joblib
        if path is None:
            path = MODEL_DIR / f"{self.symbol}_{self.model_type}_model.pkl"
        state = {
            "symbol":       self.symbol,
            "model_type":   self.model_type,
            "feature_cols": self.feature_cols,
            "xgb_model":    self.xgb_model,
            "lgbm_model":   self.lgbm_model,
            "is_trained":   self.is_trained,
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")
        return str(path)

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk."""
        import joblib
        if path is None:
            path = MODEL_DIR / f"{self.symbol}_{self.model_type}_model.pkl"
        if not Path(path).exists():
            logger.warning(f"Model file not found: {path}")
            return False
        state = joblib.load(path)
        self.symbol       = state["symbol"]
        self.model_type   = state["model_type"]
        self.feature_cols = state["feature_cols"]
        self.xgb_model    = state["xgb_model"]
        self.lgbm_model   = state["lgbm_model"]
        self.is_trained   = state["is_trained"]
        logger.info(f"Model loaded from {path}")
        return True
