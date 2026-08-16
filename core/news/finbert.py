"""
FinBERT Sentiment Analyzer
===========================
Uses ProsusAI/finbert (from HuggingFace) as a higher-accuracy
alternative to VADER for Indian financial news.

FinBERT is fine-tuned on financial text and outputs:
  - positive / negative / neutral probabilities

Falls back to VADER automatically if transformers/GPU not available.

Inspired by: vendors/finbert/ (ProsusAI FinBERT)
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


class FinBERTAnalyzer:
    """
    Financial sentiment using FinBERT (transformer-based).
    Loads model lazily on first use to avoid slow startup.
    """

    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512

    def __init__(self):
        self._pipeline  = None
        self._available = False
        self._vader_fallback = None
        self._try_load()

    def _try_load(self):
        """Try loading FinBERT pipeline (lazy, no crash if unavailable)."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                tokenizer=self.MODEL_NAME,
                device=device,
                top_k=None,       # return all 3 class scores
                truncation=True,
                max_length=self.MAX_LENGTH,
            )
            self._available = True
            logger.info(f"FinBERT loaded on {'GPU' if device == 0 else 'CPU'}")
        except Exception as e:
            logger.info(f"FinBERT not available ({e}) — falling back to VADER")
            self._init_vader()

    def _init_vader(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_fallback = SentimentIntensityAnalyzer()
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._available

    # ── Analyze ───────────────────────────────────────────────────────────────

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of financial text.

        Returns dict with:
          sentiment       : POSITIVE / NEGATIVE / NEUTRAL
          sentiment_score : float in [-1, +1]  (positive - negative)
          pos / neg / neu : individual probabilities
        """
        if not text or not text.strip():
            return self._neutral_result()

        if self._available:
            return self._finbert_analyze(text)
        elif self._vader_fallback:
            return self._vader_analyze(text)
        return self._neutral_result()

    def _finbert_analyze(self, text: str) -> Dict:
        try:
            results = self._pipeline(text[:self.MAX_LENGTH])[0]
            scores  = {r["label"].lower(): r["score"] for r in results}
            pos = scores.get("positive", 0)
            neg = scores.get("negative", 0)
            neu = scores.get("neutral",  0)
            compound = pos - neg
            if compound >= 0.05:
                label = "POSITIVE"
            elif compound <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            return {
                "sentiment":       label,
                "sentiment_score": round(compound, 4),
                "pos":             round(pos, 4),
                "neg":             round(neg, 4),
                "neu":             round(neu, 4),
                "model":           "finbert",
            }
        except Exception as e:
            logger.debug(f"FinBERT inference error: {e}")
            return self._neutral_result()

    def _vader_analyze(self, text: str) -> Dict:
        scores   = self._vader_fallback.polarity_scores(text.lower())
        compound = scores["compound"]
        label    = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
        return {
            "sentiment":       label,
            "sentiment_score": round(compound, 4),
            "pos":             round(scores["pos"], 4),
            "neg":             round(scores["neg"], 4),
            "neu":             round(scores["neu"], 4),
            "model":           "vader",
        }

    def _neutral_result(self) -> Dict:
        return {
            "sentiment":       "NEUTRAL",
            "sentiment_score": 0.0,
            "pos":             0.0,
            "neg":             0.0,
            "neu":             1.0,
            "model":           "none",
        }

    # ── Batch analyze ─────────────────────────────────────────────────────────

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Batch analysis (more efficient for FinBERT)."""
        if self._available and texts:
            try:
                truncated = [t[:self.MAX_LENGTH] for t in texts]
                batch = self._pipeline(truncated)
                results = []
                for item in batch:
                    scores = {r["label"].lower(): r["score"] for r in item}
                    pos = scores.get("positive", 0)
                    neg = scores.get("negative", 0)
                    neu = scores.get("neutral",  0)
                    compound = pos - neg
                    label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
                    results.append({
                        "sentiment": label,
                        "sentiment_score": round(compound, 4),
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                        "neu": round(neu, 4),
                        "model": "finbert",
                    })
                return results
            except Exception as e:
                logger.debug(f"FinBERT batch error: {e}")

        return [self.analyze(t) for t in texts]

    def aggregate_scores(self, results: List[Dict]) -> Dict:
        """Aggregate multiple article sentiments into one market score."""
        if not results:
            return self._neutral_result()
        scores = [r["sentiment_score"] for r in results]
        avg    = sum(scores) / len(scores)
        label  = "POSITIVE" if avg >= 0.05 else ("NEGATIVE" if avg <= -0.05 else "NEUTRAL")
        return {
            "sentiment":       label,
            "sentiment_score": round(avg, 4),
            "count":           len(results),
            "model":           results[0].get("model", "unknown"),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_analyzer: Optional[FinBERTAnalyzer] = None


def get_finbert() -> FinBERTAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBERTAnalyzer()
    return _analyzer
