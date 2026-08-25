"""
SHAP Model Explainer
=====================
Explains WHY the ML model made each trading decision.

For each prediction, shows the top features that pushed the model
towards BUY_CE / BUY_PE / SELL_STRADDLE / WAIT and by how much.

Usage:
    from core.agent.explainer import ModelExplainer
    exp = ModelExplainer(xgb_model)
    explanation = exp.explain(features_dict, predicted_action)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from loguru import logger

from core.memory.vector_store import VECTOR_FEATURE_KEYS

FEATURE_KEYS = VECTOR_FEATURE_KEYS

FEATURE_LABELS = {
    "f_pcr_oi":             "PCR (OI)",
    "f_atm_iv":             "ATM IV",
    "f_bull_pct":           "Bull% (tape)",
    "f_vix":                "India VIX",
    "f_smart_money_bias":   "Smart Money Bias",
    "f_tape_bias_numeric":  "Tape Bias",
    "f_fii_bias_score":     "FII Bias Score",
    "f_fii_net_futures":    "FII Net Futures",
    "f_fii_bias_numeric":   "FII Direction",
    "f_minutes_since_open": "Time (mins open)",
    "f_net_oi_bias":        "Net OI Bias",
    "f_recent_bias":        "Recent OI Bias",
    "f_atm_net_build":      "ATM Net OI Build",
    "f_net_oi_velocity":    "OI Velocity",
    "f_intraday_progress":  "Intraday Progress",
    "f_total_events":       "Tape Events",
    "f_avg_vol_ratio":      "Volume Ratio",
    "f_atm_ce_iv":          "CE IV",
    "f_atm_pe_iv":          "PE IV",
    "f_is_first_hour":      "First Hour",
    "f_is_last_hour":       "Last Hour",
    "f_dii_bias_numeric":   "DII Direction",
    "f_pcr_vol":            "PCR (Volume)",
    "f_ce_oi_velocity":     "CE OI Velocity",
    "f_pe_oi_velocity":     "PE OI Velocity",
}


class ModelExplainer:
    """
    SHAP-based explainer for TapeXGBModel.
    Shows contribution of each feature to the final decision.
    """

    def __init__(self, xgb_model):
        self._model     = xgb_model
        self._explainer = None
        self._background = None

    def _build_explainer(self):
        if self._explainer is not None:
            return True
        try:
            import shap
            if self._model._xgb is None:
                return False
            # TreeExplainer is fastest for XGBoost
            self._explainer = shap.TreeExplainer(self._model._xgb)
            logger.info("SHAP TreeExplainer initialized")
            return True
        except Exception as e:
            logger.warning(f"SHAP explainer init failed: {e}")
            return False

    def explain(
        self,
        features: Dict,
        predicted_action: str,
        top_n: int = 10,
    ) -> Dict:
        """
        Explain a single prediction.

        Returns:
            {
              action: str,
              top_contributors: [(feature_label, shap_value, feature_value), ...],
              summary_text: str,
              all_shap: dict,
            }
        """
        if not self._build_explainer():
            return self._fallback_explanation(features, predicted_action, top_n)

        try:
            import shap
            from core.agent.ml_agent import ACTIONS, ACTION_IDX

            x = np.array([[features.get(k, 0.0) for k in FEATURE_KEYS]], dtype=np.float32)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            shap_values = self._explainer.shap_values(x)   # shape: [n_classes, 1, n_features]

            action_idx = ACTION_IDX.get(predicted_action, 3)

            # Get SHAP values for the predicted class
            if isinstance(shap_values, list):
                sv = shap_values[action_idx][0]
            else:
                sv = shap_values[0, :, action_idx] if shap_values.ndim == 3 else shap_values[0]

            # Pair with feature names and values
            contrib = []
            for i, (k, s) in enumerate(zip(FEATURE_KEYS, sv)):
                fval = float(x[0, i])
                contrib.append((k, float(s), fval))

            # Sort by abs SHAP value
            contrib.sort(key=lambda t: abs(t[1]), reverse=True)
            top = contrib[:top_n]

            # Human-readable summary
            positives = [(k, s, v) for k, s, v in top if s > 0][:3]
            negatives = [(k, s, v) for k, s, v in top if s < 0][:2]

            pos_text = ", ".join(
                f"{FEATURE_LABELS.get(k, k)} ({v:.2f})"
                for k, s, v in positives
            )
            neg_text = ", ".join(
                f"{FEATURE_LABELS.get(k, k)} ({v:.2f})"
                for k, s, v in negatives
            )
            summary = f"Key drivers: {pos_text}"
            if neg_text:
                summary += f" | Opposing: {neg_text}"

            top_contributors = [
                (FEATURE_LABELS.get(k, k), round(s, 4), round(v, 4))
                for k, s, v in top
            ]

            return {
                "action":           predicted_action,
                "top_contributors": top_contributors,
                "summary_text":     summary,
                "all_shap":         {k: round(float(s), 4) for k, s, _ in contrib},
            }

        except Exception as e:
            logger.warning(f"SHAP explain error: {e}")
            return self._fallback_explanation(features, predicted_action, top_n)

    def _fallback_explanation(self, features: Dict, action: str, top_n: int) -> Dict:
        """Rule-based explanation when SHAP is unavailable."""
        key_feats = [
            ("f_tape_bias_numeric",  features.get("f_tape_bias_numeric", 0)),
            ("f_fii_bias_numeric",   features.get("f_fii_bias_numeric", 0)),
            ("f_pcr_oi",             features.get("f_pcr_oi", 1.0)),
            ("f_atm_iv",             features.get("f_atm_iv", 15)),
            ("f_vix",                features.get("f_vix", 15)),
            ("f_smart_money_bias",   features.get("f_smart_money_bias", 0)),
            ("f_bull_pct",           features.get("f_bull_pct", 0.5)),
        ]
        contribs = [
            (FEATURE_LABELS.get(k, k), round(v, 4), round(v, 4))
            for k, v in key_feats
        ]
        summary = f"Action: {action} | " + " | ".join(
            f"{FEATURE_LABELS.get(k, k)}={v:.2f}" for k, v in key_feats[:4]
        )
        return {
            "action":           action,
            "top_contributors": contribs,
            "summary_text":     summary,
            "all_shap":         {},
        }


# Singleton per model
_explainers: Dict = {}

def get_explainer(xgb_model) -> ModelExplainer:
    key = id(xgb_model)
    if key not in _explainers:
        _explainers[key] = ModelExplainer(xgb_model)
    return _explainers[key]
