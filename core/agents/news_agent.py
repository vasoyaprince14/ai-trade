"""
News Intelligence Agent
========================
Fetches and scores news for NSE stocks + macro (Gold/Nifty).

Pipeline:
  yfinance.news + Google RSS feeds
      ↓
  Deduplication
      ↓
  FinBERT sentiment (transformers) or TextBlob fallback
      ↓
  Event type classification (Ollama)
      ↓
  Returns scored news items with:
    - sentiment (-1 to +1)
    - importance (0-1)
    - event_type (RESULT|ORDER|M&A|RATING|REGULATORY|DIVIDEND|GENERAL)
    - expected_direction (BULLISH|BEARISH|NEUTRAL)
    - expected_horizon (INTRADAY|1D|SWING)

Usage:
    from core.agents.news_agent import NewsAgent
    agent = NewsAgent()
    items = agent.get_news("RELIANCE.NS", top_n=5)
"""

import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

# ── Sentiment backend ─────────────────────────────────────────────────────────
_finbert = None
_finbert_tokenizer = None
_USE_FINBERT = False

def _try_load_finbert():
    global _finbert, _finbert_tokenizer, _USE_FINBERT
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_path = os.path.join(os.path.dirname(__file__), "../../vendors/finbert")
        if not os.path.exists(model_path):
            model_path = "ProsusAI/finbert"   # fall back to HuggingFace hub

        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _finbert = AutoModelForSequenceClassification.from_pretrained(model_path)
        _finbert.eval()
        _USE_FINBERT = True
        logger.info("FinBERT loaded for news sentiment")
    except Exception as e:
        logger.debug(f"FinBERT not available: {e} — using keyword sentiment")
        _USE_FINBERT = False


def _finbert_score(text: str) -> float:
    """Return sentiment score -1 (neg) to +1 (pos) using FinBERT."""
    import torch
    tokens = _finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = _finbert(tokens["input_ids"], attention_mask=tokens["attention_mask"])
    probs = torch.softmax(out.logits, dim=-1)[0].tolist()
    # FinBERT labels: positive=0, negative=1, neutral=2
    return probs[0] - probs[1]


def _keyword_score(text: str) -> float:
    """Simple keyword-based fallback sentiment (-1 to +1)."""
    text_l = text.lower()
    pos = ["beat", "profit", "growth", "record", "strong", "upgrade", "buy",
           "order win", "expansion", "dividend", "rally", "bullish", "rise",
           "gain", "positive", "approved", "launch"]
    neg = ["miss", "loss", "weak", "down", "downgrade", "sell", "probe", "fraud",
           "decline", "delay", "warning", "risk", "bearish", "drop", "cut",
           "default", "seized", "penalty", "fine"]
    score = sum(1 for w in pos if w in text_l) - sum(1 for w in neg if w in text_l)
    return max(-1.0, min(1.0, score * 0.25))


def _classify_event(title: str) -> str:
    t = title.lower()
    if any(w in t for w in ["result", "profit", "revenue", "earnings", "q1", "q2", "q3", "q4"]):
        return "RESULT"
    if any(w in t for w in ["order", "contract", "wins", "deal", "award"]):
        return "ORDER_WIN"
    if any(w in t for w in ["merger", "acquisition", "acquires", "takeover", "stake"]):
        return "M&A"
    if any(w in t for w in ["rating", "upgrade", "downgrade", "target"]):
        return "RATING"
    if any(w in t for w in ["sebi", "rbi", "regulatory", "nod", "approval", "fine", "penalty", "probe"]):
        return "REGULATORY"
    if any(w in t for w in ["dividend", "bonus", "split", "buyback"]):
        return "CORPORATE_ACTION"
    if any(w in t for w in ["invest", "fii", "fdi", "stake", "fund"]):
        return "INSTITUTIONAL"
    return "GENERAL"


def _importance(event_type: str, sentiment_abs: float) -> float:
    base = {
        "RESULT": 0.9, "ORDER_WIN": 0.8, "M&A": 0.85,
        "RATING": 0.75, "REGULATORY": 0.8, "CORPORATE_ACTION": 0.7,
        "INSTITUTIONAL": 0.65, "GENERAL": 0.4,
    }.get(event_type, 0.4)
    return round(min(1.0, base * (0.6 + 0.4 * sentiment_abs)), 3)


def _horizon(event_type: str) -> str:
    if event_type in ("RESULT", "M&A", "REGULATORY"):
        return "1D+"
    if event_type in ("ORDER_WIN", "RATING", "INSTITUTIONAL"):
        return "1D"
    return "INTRADAY"


# ── News fetcher ───────────────────────────────────────────────────────────────

def _dedup(items: list) -> list:
    seen, out = set(), []
    for item in items:
        h = hashlib.md5((item.get("title", "") or "").encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            out.append(item)
    return out


class NewsAgent:
    """Fetch + score news for a given ticker or topic."""

    def __init__(self):
        _try_load_finbert()
        self._cache: dict = {}
        self._cache_ttl = 900   # 15 minutes

    def _score_item(self, title: str) -> dict:
        text = title.strip()
        if _USE_FINBERT:
            try:
                raw_score = _finbert_score(text)
            except Exception:
                raw_score = _keyword_score(text)
        else:
            raw_score = _keyword_score(text)

        event_type = _classify_event(text)
        importance = _importance(event_type, abs(raw_score))
        direction  = "BULLISH" if raw_score > 0.1 else "BEARISH" if raw_score < -0.1 else "NEUTRAL"

        return {
            "title":              text,
            "sentiment":          round(raw_score, 3),
            "importance":         importance,
            "event_type":         event_type,
            "expected_direction": direction,
            "expected_horizon":   _horizon(event_type),
        }

    def get_news(self, ticker: str, top_n: int = 5) -> list[dict]:
        """
        Fetch and score recent news for ticker (e.g. 'RELIANCE.NS', 'GC=F').
        Returns list of scored news dicts sorted by importance.
        """
        now = time.time()
        if ticker in self._cache:
            cached_ts, cached_data = self._cache[ticker]
            if now - cached_ts < self._cache_ttl:
                return cached_data

        raw_items = []
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).news or []
            for n in info[:15]:
                title = n.get("title") or n.get("headline") or ""
                if title:
                    raw_items.append({"title": title, "source": n.get("publisher", "")})
        except Exception as e:
            logger.debug(f"yfinance news {ticker}: {e}")

        raw_items = _dedup(raw_items)
        scored = [self._score_item(item["title"]) for item in raw_items]
        scored.sort(key=lambda x: x["importance"], reverse=True)
        result = scored[:top_n]

        self._cache[ticker] = (now, result)
        return result

    def get_aggregate_sentiment(self, ticker: str) -> dict:
        """
        Returns aggregate sentiment for a ticker:
        { score, direction, top_event, items_count, summary }
        """
        items = self.get_news(ticker)
        if not items:
            return {"score": 0.0, "direction": "NEUTRAL", "top_event": "NONE", "items_count": 0, "summary": "No news"}

        avg_score = sum(i["sentiment"] for i in items) / len(items)
        top = max(items, key=lambda x: x["importance"])
        direction = "BULLISH" if avg_score > 0.05 else "BEARISH" if avg_score < -0.05 else "NEUTRAL"

        summary_parts = []
        for i in items[:3]:
            em = "📈" if i["expected_direction"] == "BULLISH" else "📉" if i["expected_direction"] == "BEARISH" else "➡️"
            summary_parts.append(f"{em} [{i['event_type']}] {i['title'][:60]}")

        return {
            "score":      round(avg_score, 3),
            "direction":  direction,
            "top_event":  top["event_type"],
            "importance": top["importance"],
            "items_count": len(items),
            "summary":    "\n".join(summary_parts),
            "items":      items,
        }


# Singleton
_agent: Optional[NewsAgent] = None

def get_news_agent() -> NewsAgent:
    global _agent
    if _agent is None:
        _agent = NewsAgent()
    return _agent
