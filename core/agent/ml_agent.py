"""
Open-Source Trading Agent — No Anthropic Required
===================================================
Replaces Claude API with fully open-source stack:

  Layer 1 — Ollama LLM (optional, for reasoning text)
    Local LLM server: llama3.2, mistral, qwen2.5, phi3, etc.
    OpenAI-compatible API at http://localhost:11434
    Install: brew install ollama && ollama pull llama3.2:3b
    OR Docker: docker-compose up ollama

  Layer 2 — XGBoost Classifier (trade decision)
    Input : 78 tape/OI/FII features from TapeReader.extract_features()
    Output: BUY_CE | BUY_PE | SELL_STRADDLE | WAIT
    Train : synthetic bootstrap on day 1, improves with real outcomes

  Layer 3 — Rule-based fallback
    Always works, no dependencies

Decision pipeline:
  features → XGBoost → action + confidence
  features + action → Ollama prompt → reasoning text (if Ollama running)
  → TradeDecision (same interface as claude TradingAgent)

Model persistence:
  Saved to data/models/tape_agent_{symbol}.pkl
  Retrain: agent.train()  or  agent.train(force=True)

Usage (same interface as TradingAgent):
  from core.agent.ml_agent import get_ml_agent
  agent = get_ml_agent()
  decision = agent.decide(oi_tracker, symbol="NIFTY")
  print(decision.summary())
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from loguru import logger

from core.agent.trading_agent import TradeDecision    # reuse same dataclass
from core.memory.vector_store import get_vector_store, VECTOR_FEATURE_KEYS

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DIR   = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS     = ["BUY_CE", "BUY_PE", "SELL_STRADDLE", "WAIT"]
ACTION_IDX  = {a: i for i, a in enumerate(ACTIONS)}

# Feature vector keys (same as vector_store, guaranteed order)
FEATURE_KEYS = VECTOR_FEATURE_KEYS

N_SYNTHETIC  = 15_000    # synthetic training samples to bootstrap model
OLLAMA_URL   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


# ── Synthetic Data Generator ─────────────────────────────────────────────────

def _generate_synthetic_training_data(n: int = N_SYNTHETIC, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap training data using domain-knowledge rules.
    Generates realistic feature vectors with known correct labels.

    Rules encoded:
      BUY_CE   : tape_bias +2 AND fii bullish AND pcr < 0.9 AND low IV
      BUY_PE   : tape_bias -2 AND fii bearish AND pcr > 1.2 AND low IV
      STRADDLE : abs(tape_bias) <= 1 AND high IV (>20) AND pcr 0.9-1.2
      WAIT     : all other combinations (low signal / conflicting)
    """
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURE_KEYS)
    X = np.zeros((n, n_feat), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    feat_map = {k: i for i, k in enumerate(FEATURE_KEYS)}

    def f(name): return feat_map.get(name, -1)

    for i in range(n):
        # Randomize base features
        tape_bias   = rng.choice([-2, -1, 0, 1, 2], p=[0.15, 0.2, 0.3, 0.2, 0.15])
        fii_bias    = rng.choice([-2, -1, 0, 1, 2], p=[0.15, 0.2, 0.3, 0.2, 0.15])
        pcr         = rng.uniform(0.5, 1.8)
        atm_iv      = rng.uniform(8, 35)
        vix         = rng.uniform(10, 30)
        bull_pct    = rng.uniform(0.1, 0.9)
        total_ev    = rng.integers(0, 50)
        smart_money = rng.choice([-2, -1, 0, 1, 2])

        # Set features
        if f("f_tape_bias_numeric") >= 0:    X[i, f("f_tape_bias_numeric")]  = tape_bias
        if f("f_smart_money_bias") >= 0:     X[i, f("f_smart_money_bias")]   = smart_money
        if f("f_fii_bias_numeric") >= 0:     X[i, f("f_fii_bias_numeric")]   = fii_bias
        if f("f_pcr_oi") >= 0:               X[i, f("f_pcr_oi")]             = pcr
        if f("f_atm_iv") >= 0:               X[i, f("f_atm_iv")]             = atm_iv
        if f("f_vix") >= 0:                  X[i, f("f_vix")]                = vix
        if f("f_bull_pct") >= 0:             X[i, f("f_bull_pct")]           = bull_pct
        if f("f_total_events") >= 0:         X[i, f("f_total_events")]       = total_ev
        if f("f_fii_net_futures") >= 0:      X[i, f("f_fii_net_futures")]    = fii_bias * rng.uniform(10000, 100000)
        if f("f_fii_bias_score") >= 0:       X[i, f("f_fii_bias_score")]     = fii_bias * rng.uniform(5000, 80000)
        if f("f_recent_bias") >= 0:          X[i, f("f_recent_bias")]        = tape_bias * rng.uniform(0, 200000)
        if f("f_net_oi_bias") >= 0:          X[i, f("f_net_oi_bias")]        = tape_bias * rng.uniform(0, 500000)
        if f("f_atm_net_build") >= 0:        X[i, f("f_atm_net_build")]      = tape_bias * rng.uniform(0, 100000)

        # Time features
        hour = rng.integers(9, 16)
        minute = rng.integers(0, 60)
        mins_open = (hour - 9) * 60 + minute - 15
        if f("f_minutes_since_open") >= 0:   X[i, f("f_minutes_since_open")] = max(0, mins_open)
        if f("f_intraday_progress") >= 0:    X[i, f("f_intraday_progress")]  = max(0, mins_open) / 375.0
        if f("f_is_first_hour") >= 0:        X[i, f("f_is_first_hour")]      = int(mins_open < 60)
        if f("f_is_last_hour") >= 0:         X[i, f("f_is_last_hour")]       = int(mins_open > 315)

        # OI velocity / momentum
        oi_vel = tape_bias * rng.uniform(0, 50000)
        if f("f_net_oi_velocity") >= 0:      X[i, f("f_net_oi_velocity")]    = oi_vel

        # Add noise
        noise_mask = rng.random(n_feat) < 0.3
        X[i, noise_mask] += rng.normal(0, 0.05, noise_mask.sum())

        # Label assignment using relaxed but realistic rules
        # Goal: ~15% BUY_CE, ~15% BUY_PE, ~20% SELL_STRADDLE, ~50% WAIT
        combined    = tape_bias + fii_bias + smart_money
        high_iv     = atm_iv > 19
        neutral_pcr = 0.80 <= pcr <= 1.25

        if combined >= 3 and pcr < 1.05 and bull_pct > 0.55:
            label = ACTION_IDX["BUY_CE"]
        elif combined <= -3 and pcr > 1.05 and bull_pct < 0.45:
            label = ACTION_IDX["BUY_PE"]
        elif high_iv and neutral_pcr and abs(tape_bias) <= 1:
            label = ACTION_IDX["SELL_STRADDLE"]
        elif combined >= 2 and pcr < 0.95 and rng.random() < 0.4:
            label = ACTION_IDX["BUY_CE"]
        elif combined <= -2 and pcr > 1.15 and rng.random() < 0.4:
            label = ACTION_IDX["BUY_PE"]
        elif high_iv and abs(combined) <= 2 and rng.random() < 0.5:
            label = ACTION_IDX["SELL_STRADDLE"]
        else:
            label = ACTION_IDX["WAIT"]

        # Label noise (real-world uncertainty)
        if rng.random() < 0.06:
            label = rng.integers(0, len(ACTIONS))

        y[i] = label

    return X, y


# ── XGBoost Tape Agent ────────────────────────────────────────────────────────

class TapeXGBModel:
    """
    XGBoost classifier trained on tape/OI/FII features.
    Outputs: action probabilities across [BUY_CE, BUY_PE, SELL_STRADDLE, WAIT]
    """

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol
        self._model_path = MODEL_DIR / f"tape_agent_{symbol}.pkl"
        self._xgb = None
        self._lgbm = None
        self._trained = False
        self._train_samples = 0
        self._trained_at: Optional[datetime] = None
        # Auto-load saved model if it exists
        if self._model_path.exists():
            self._load()

    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Train on provided data or generate synthetic bootstrap data.
        Saves model to disk after training.
        """
        if X is None or y is None:
            logger.info(f"Generating {N_SYNTHETIC} synthetic training samples...")
            X, y = _generate_synthetic_training_data(N_SYNTHETIC)

        self._train_samples = len(X)
        n_classes = len(ACTIONS)

        # XGBoost
        try:
            import xgboost as xgb
            self._xgb = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                num_class=n_classes,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            self._xgb.fit(X, y)
            logger.info(f"XGBoost trained | {self._train_samples} samples | {n_classes} classes")
        except ImportError:
            logger.warning("xgboost not available")

        # LightGBM (optional, but boosts accuracy)
        try:
            import lightgbm as lgb
            self._lgbm = lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                num_class=n_classes,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            self._lgbm.fit(X, y)
            logger.info("LightGBM trained")
        except ImportError:
            logger.debug("lightgbm not available (optional)")

        self._trained = True
        self._trained_at = datetime.now()
        self._save()
        return self

    def predict(self, features: Dict) -> Dict:
        """
        Predict trade action from feature dict.
        Returns: {action, confidence, probabilities, method}
        """
        if not self._trained:
            if self._model_path.exists():
                self._load()
            else:
                logger.info("No trained model found — training on synthetic data")
                self.train()

        x = np.array([[features.get(k, 0.0) for k in FEATURE_KEYS]], dtype=np.float32)
        # Replace NaN/Inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        probas = []
        for model in [self._xgb, self._lgbm]:
            if model is not None:
                probas.append(model.predict_proba(x)[0])

        if not probas:
            return self._rule_fallback(features)

        proba = np.mean(probas, axis=0)
        action_idx = int(np.argmax(proba))
        action = ACTIONS[action_idx]
        confidence = float(proba[action_idx])

        return {
            "action":      action,
            "confidence":  round(confidence, 3),
            "probabilities": {a: round(float(p), 3) for a, p in zip(ACTIONS, proba)},
            "method":      "xgboost+lgbm" if self._lgbm else "xgboost",
        }

    def _rule_fallback(self, features: Dict) -> Dict:
        """Pure rule-based fallback when models are unavailable."""
        tape  = features.get("f_tape_bias_numeric", 0)
        smart = features.get("f_smart_money_bias",  0)
        pcr   = features.get("f_pcr_oi",            1.0)
        iv    = features.get("f_atm_iv",             15.0)
        combined = tape + smart
        if combined >= 3 and pcr < 1.0:
            return {"action": "BUY_CE",       "confidence": 0.6, "method": "rules"}
        if combined <= -3 and pcr > 1.1:
            return {"action": "BUY_PE",       "confidence": 0.6, "method": "rules"}
        if iv > 20 and abs(combined) <= 1:
            return {"action": "SELL_STRADDLE","confidence": 0.55,"method": "rules"}
        return {"action": "WAIT",             "confidence": 0.7, "method": "rules"}

    def get_feature_importance(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """Return top features ranked by importance."""
        if not self._trained:
            return []
        importances = np.zeros(len(FEATURE_KEYS))
        count = 0
        for model in [self._xgb, self._lgbm]:
            if model is None:
                continue
            try:
                fi = model.feature_importances_
                if len(fi) == len(FEATURE_KEYS):
                    importances += fi
                    count += 1
            except Exception:
                continue
        if count:
            importances /= count
        ranked = sorted(zip(FEATURE_KEYS, importances), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]

    def _save(self):
        state = {
            "xgb":          self._xgb,
            "lgbm":         self._lgbm,
            "trained_at":   self._trained_at,
            "train_samples": self._train_samples,
            "symbol":       self.symbol,
        }
        joblib.dump(state, self._model_path)
        logger.info(f"Tape agent model saved: {self._model_path}")

    def _load(self):
        state = joblib.load(self._model_path)
        self._xgb          = state.get("xgb")
        self._lgbm         = state.get("lgbm")
        self._trained_at   = state.get("trained_at")
        self._train_samples= state.get("train_samples", 0)
        self._trained      = True
        logger.info(f"Tape agent model loaded | trained_at={self._trained_at} | samples={self._train_samples}")

    @property
    def is_trained(self) -> bool:
        return self._trained or self._model_path.exists()

    @property
    def info(self) -> Dict:
        return {
            "trained":        self._trained,
            "trained_at":     str(self._trained_at) if self._trained_at else "not trained",
            "train_samples":  self._train_samples,
            "model_path":     str(self._model_path),
            "has_xgb":        self._xgb is not None,
            "has_lgbm":       self._lgbm is not None,
        }


# ── Ollama LLM Reasoning ──────────────────────────────────────────────────────

class OllamaReasoner:
    """
    Optional: use local Ollama LLM to generate trade reasoning text.
    Falls back gracefully if Ollama is not running.

    Supported models (install with `ollama pull <model>`):
      llama3.2:3b   (fast, ~2GB)
      llama3.2:1b   (very fast, ~1.3GB)
      mistral:7b    (good quality, ~4GB)
      qwen2.5:7b    (best quality, ~4.5GB)
      phi3:mini     (tiny, fast)
    """

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available: Optional[bool] = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                if not models:
                    logger.info("Ollama running but no models installed. Run: ollama pull llama3.2:3b")
                    self._available = False
                else:
                    # Pick best available model
                    preferred = ["llama3.2:3b", "llama3.2:1b", "mistral:7b", "qwen2.5:7b", "phi3:mini"]
                    for pref in preferred:
                        if any(pref in m for m in models):
                            self.model = pref
                            break
                    else:
                        self.model = models[0]  # use whatever is available
                    logger.info(f"Ollama available | model={self.model} | all: {models}")
                    self._available = True
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def generate_reasoning(
        self,
        action: str,
        features: Dict,
        tape_summary: Dict,
        participant: Dict,
        similar_states: List[Dict],
        confidence: float,
    ) -> str:
        """
        Ask Ollama to generate a concise reasoning explanation for the trade decision.
        Returns a plain-text reasoning string.
        """
        if not self._check_available():
            return self._template_reasoning(action, features, tape_summary, participant, confidence)

        # Build compact context (avoid huge prompts)
        tape_bias   = tape_summary.get("bias", "NEUTRAL")
        smart_money = participant.get("smart_money_bias", "NEUTRAL")
        fii_bias    = participant.get("fno", {}).get("fii_bias", "NEUTRAL")
        pcr         = features.get("f_pcr_oi", 1.0)
        iv          = features.get("f_atm_iv", 15)
        vix         = features.get("f_vix", 0)
        bull_pct    = features.get("f_bull_pct", 0.5)
        events      = features.get("f_total_events", 0)
        spot        = features.get("f_spot", 0)

        # Format top similar states
        hist_text = ""
        if similar_states:
            outcomes = [s for s in similar_states[:5] if s.get("outcome")]
            if outcomes:
                hist_text = "\nHistorical similar states:\n"
                for s in outcomes:
                    o = s["outcome"]
                    hist_text += (
                        f"  {s['timestamp'][:10]}: tape={s['tape_bias']} "
                        f"→ {o.get('direction','?')} move {o.get('move_pct',0):+.2%}\n"
                    )

        prompt = f"""You are a Nifty F&O trader. Based on this data, explain why the decision is {action}.
Be concise (3-4 sentences max). Focus on the key signals.

Current market:
- Spot: {spot:.0f}  |  PCR OI: {pcr:.2f}  |  ATM IV: {iv:.1f}%  |  VIX: {vix:.1f}
- Tape bias: {tape_bias} ({events} institutional events, {bull_pct:.0%} bullish OI)
- FII bias: {fii_bias}  |  Smart money: {smart_money}
- Model confidence: {confidence:.0%}
{hist_text}
Decision: {action}

Reasoning (3-4 sentences):"""

        try:
            import requests
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 150},
                },
                timeout=30,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except Exception as e:
            logger.debug(f"Ollama reasoning failed: {e}")

        return self._template_reasoning(action, features, tape_summary, participant, confidence)

    def _template_reasoning(
        self,
        action: str,
        features: Dict,
        tape_summary: Dict,
        participant: Dict,
        confidence: float,
    ) -> str:
        """Generate rule-based reasoning text (no LLM needed)."""
        tape_bias   = tape_summary.get("bias", "NEUTRAL")
        smart_money = participant.get("smart_money_bias", "NEUTRAL")
        fii_bias    = participant.get("fno", {}).get("fii_bias", "NEUTRAL")
        pcr         = features.get("f_pcr_oi",  1.0)
        iv          = features.get("f_atm_iv",  15.0)
        vix         = features.get("f_vix",     0.0)
        events      = int(features.get("f_total_events", 0))
        bull_pct    = features.get("f_bull_pct", 0.5)
        spot        = features.get("f_spot", 0)

        template = {
            "BUY_CE": (
                f"Tape is {tape_bias} with {events} institutional events ({bull_pct:.0%} bullish OI). "
                f"FII positioning is {fii_bias} and smart money is {smart_money}. "
                f"PCR at {pcr:.2f} confirms call side strength. "
                f"VIX at {vix:.1f} is moderate — buying ATM call near {spot:.0f}."
            ),
            "BUY_PE": (
                f"Tape is {tape_bias} with {events} institutional events ({1-bull_pct:.0%} bearish OI). "
                f"FII positioning is {fii_bias} and smart money is {smart_money}. "
                f"PCR at {pcr:.2f} shows put buying pressure. "
                f"VIX at {vix:.1f} — buying ATM put near {spot:.0f}."
            ),
            "SELL_STRADDLE": (
                f"ATM IV at {iv:.1f}% is elevated — premium selling is favored. "
                f"Tape bias is {tape_bias} (close to neutral) with {events} events. "
                f"PCR at {pcr:.2f} is balanced. "
                f"Market likely to stay rangebound near {spot:.0f} — selling ATM straddle."
            ),
            "WAIT": (
                f"Tape bias is {tape_bias} but not strong enough for a trade ({events} events, {bull_pct:.0%} bullish). "
                f"FII is {fii_bias}, smart money is {smart_money} — signals are mixed. "
                f"PCR {pcr:.2f}, IV {iv:.1f}%, VIX {vix:.1f}. "
                f"Waiting for clearer confluence before entering."
            ),
        }
        base = template.get(action, f"Decision: {action}. Confidence: {confidence:.0%}.")
        return f"[ML Agent | conf={confidence:.0%}] {base}"

    def list_available_models(self) -> List[str]:
        """List Ollama models installed locally."""
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []


# ── Open-Source Trading Agent ─────────────────────────────────────────────────

class OpenSourceTradingAgent:
    """
    Full trading agent using XGBoost + optional Ollama LLM.
    Same interface as TradingAgent (uses TradeDecision).

    Decision flow:
      1. Extract 78 features from OITracker
      2. XGBoost → action + confidence
      3. Query Qdrant for similar historical states
      4. Ollama (if available) → reasoning text
      5. Build TradeDecision with strike, SL, target inferred from option chain
    """

    def __init__(self, symbol: str = "NIFTY"):
        self._xgb_model  = TapeXGBModel(symbol)
        self._ollama     = OllamaReasoner()
        self._vector_store = get_vector_store()
        self._last_decision: Optional[TradeDecision] = None

        # Auto-train if no model file exists
        if not self._xgb_model.is_trained:
            logger.info("No trained model found — bootstrapping with synthetic data")
            self._xgb_model.train()

    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, force: bool = False):
        """
        Train/retrain the XGBoost model.
        - No args: use synthetic bootstrap data
        - Pass X, y: use real labeled data (collected from Qdrant outcomes)
        - force=True: retrain even if model already exists
        """
        if not force and self._xgb_model._model_path.exists():
            logger.info("Model already trained. Use force=True to retrain.")
            return
        self._xgb_model.train(X, y)

    def decide(
        self,
        oi_tracker,
        risk_manager=None,
        symbol: str = "NIFTY",
    ) -> TradeDecision:
        """
        Make a trade decision. Same interface as TradingAgent.decide().
        """
        # 1. Extract features
        try:
            features = oi_tracker.get_model_features()
            tape_summary  = oi_tracker.tape_reader.get_flow_summary()
            participant   = oi_tracker._participant.get_full_picture()
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return TradeDecision(
                action="WAIT", symbol=symbol,
                reasoning=f"Feature extraction failed: {e}", confidence=0.0,
            )

        # 2. Risk check
        if risk_manager:
            try:
                if not risk_manager.can_trade():
                    return TradeDecision(
                        action="WAIT", symbol=symbol,
                        reasoning="Risk limit reached — trading halted for today",
                        confidence=1.0,
                    )
            except Exception:
                pass

        # 3. XGBoost decision
        pred = self._xgb_model.predict(features)
        action     = pred["action"]
        confidence = pred["confidence"]
        method     = pred.get("method", "xgboost")

        # 4. Store state in vector memory
        tape_bias   = tape_summary.get("bias",       "NEUTRAL")
        fii_bias    = participant.get("fno",{}).get("fii_bias","NEUTRAL")
        smart_money = participant.get("smart_money_bias", "NEUTRAL")
        state_id = self._vector_store.store_state(
            features=features, symbol=symbol,
            tape_bias=tape_bias, fii_bias=fii_bias,
            smart_money=smart_money, signal=action,
        )

        # 5. Find similar historical states
        similar = self._vector_store.find_similar(features, top_k=10) if self._vector_store.is_available else []
        hist_stats = self._vector_store.get_outcome_stats(similar)

        # Adjust confidence based on historical outcomes
        if hist_stats.get("sample_size", 0) >= 5:
            hist_conf = hist_stats.get("confidence", 0)
            hist_dir  = hist_stats.get("direction", "UNCLEAR")
            # Boost confidence if history agrees
            if (action in ("BUY_CE",) and hist_dir == "UP") or \
               (action in ("BUY_PE",) and hist_dir == "DOWN"):
                confidence = min(confidence + 0.10, 0.95)
            elif hist_dir not in ("UNCLEAR",) and \
                 ((action in ("BUY_CE",) and hist_dir == "DOWN") or
                  (action in ("BUY_PE",) and hist_dir == "UP")):
                confidence = max(confidence - 0.15, 0.10)
                if confidence < 0.4:
                    action = "WAIT"

        # 6. Get option chain details (for strike / SL / target)
        strike, expiry, entry_low, entry_high, sl, target, sl_spot, tgt_spot = \
            self._get_trade_levels(oi_tracker, action, features)

        # 7. Generate reasoning (Ollama or template)
        reasoning = self._ollama.generate_reasoning(
            action=action,
            features=features,
            tape_summary=tape_summary,
            participant=participant,
            similar_states=similar,
            confidence=confidence,
        )

        # Add historical context to reasoning
        if hist_stats.get("sample_size", 0) >= 3:
            reasoning += (
                f"\n\nHistorical memory ({hist_stats['sample_size']} similar states): "
                f"{hist_stats['up_pct']:.0%} UP / {hist_stats['down_pct']:.0%} DOWN "
                f"(avg move: {hist_stats['avg_move']:+.2%})"
            )

        decision = TradeDecision(
            action=action,
            symbol=symbol,
            strike=int(strike) if strike else 0,
            expiry=expiry,
            qty_lots=1,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=sl,
            target=target,
            sl_spot=sl_spot,
            target_spot=tgt_spot,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            risk_reward=(
                (target - entry_high) / max(entry_high - sl, 0.01)
            ) if target and entry_high and sl else 0.0,
        )

        self._last_decision = decision
        logger.info(f"[{method}] {decision.summary()}")
        return decision

    def _get_trade_levels(
        self,
        oi_tracker,
        action: str,
        features: Dict,
    ) -> Tuple:
        """
        Derive strike, expiry, and price levels from the current option chain.
        Returns: (strike, expiry, entry_low, entry_high, sl, target, sl_spot, tgt_spot)
        """
        if action == "WAIT":
            return (0, "", 0, 0, 0, 0, 0, 0)

        try:
            spot = features.get("f_spot", 0)
            atm  = features.get("f_atm_strike", round(spot / 50) * 50 if spot else 0)

            # Get expiry
            expiries = oi_tracker.scraper.get_expiry_dates(oi_tracker.symbol)
            expiry   = expiries[0] if expiries else ""

            # Get option chain for LTP
            records = oi_tracker.scraper.parse_option_chain(
                oi_tracker.symbol, expiry, strikes_range=5
            )
            if not records:
                return (atm, expiry, 0, 0, 0, 0, 0, 0)

            df = pd.DataFrame(records)

            if action == "BUY_CE":
                opt_type = "CE"
                # Slightly OTM call (ATM+50 for directional play, ATM for safe)
                strike = atm
                opt_df = df[(df["option_type"] == "CE") & (df["strike"] == strike)]
            elif action == "BUY_PE":
                opt_type = "PE"
                strike = atm
                opt_df = df[(df["option_type"] == "PE") & (df["strike"] == strike)]
            elif action in ("SELL_CE", "SELL_STRADDLE"):
                opt_type = "CE"
                strike = atm
                opt_df = df[(df["option_type"] == "CE") & (df["strike"] == strike)]
            elif action == "SELL_PE":
                opt_type = "PE"
                strike = atm
                opt_df = df[(df["option_type"] == "PE") & (df["strike"] == strike)]
            else:
                opt_df = pd.DataFrame()
                opt_type = "CE"
                strike = atm

            if opt_df.empty:
                return (atm, expiry, 0, 0, 0, 0, 0, 0)

            ltp = float(opt_df["ltp"].iloc[0])
            if ltp <= 0:
                return (atm, expiry, 0, 0, 0, 0, 0, 0)

            # Entry range: ±5% of LTP
            entry_low  = round(ltp * 0.95, 1)
            entry_high = round(ltp * 1.05, 1)

            if "BUY" in action:
                sl      = round(ltp * 0.60, 1)     # 40% SL
                target  = round(ltp * 2.00, 1)     # 100% target (2x)
                sl_spot    = spot - 150 if opt_type == "CE" else spot + 150
                tgt_spot   = spot + 200 if opt_type == "CE" else spot - 200
            else:  # SELL
                sl      = round(ltp * 2.00, 1)     # option doubles = SL
                target  = round(ltp * 0.30, 1)     # 70% decay = target
                sl_spot    = spot + 200 if opt_type == "CE" else spot - 200
                tgt_spot   = spot        # want spot to stay here

            return (strike, expiry, entry_low, entry_high, sl, target,
                    round(sl_spot, 0), round(tgt_spot, 0))

        except Exception as e:
            logger.debug(f"_get_trade_levels error: {e}")
            return (0, "", 0, 0, 0, 0, 0, 0)

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        return self._xgb_model.get_feature_importance(top_n)

    @property
    def model_info(self) -> Dict:
        return {
            **self._xgb_model.info,
            "ollama_available": self._ollama._check_available(),
            "ollama_model":     self._ollama.model,
            "ollama_models":    self._ollama.list_available_models(),
            "vector_memory":    self._vector_store.is_available,
        }

    @property
    def last_decision(self) -> Optional[TradeDecision]:
        return self._last_decision


# ── Singleton ─────────────────────────────────────────────────────────────────

_ml_agents: Dict[str, OpenSourceTradingAgent] = {}


def get_ml_agent(symbol: str = "NIFTY") -> OpenSourceTradingAgent:
    """Get or create a cached ML agent for a symbol."""
    if symbol not in _ml_agents:
        _ml_agents[symbol] = OpenSourceTradingAgent(symbol)
    return _ml_agents[symbol]
