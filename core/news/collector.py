"""
News & Sentiment Collector
===========================
Collects financial news from Indian sources:
  - Economic Times
  - Moneycontrol
  - Mint
  - BSE/NSE announcements
  - Livemint RSS
  - Business Standard

Applies:
  - VADER sentiment (fast, works offline)
  - Stock/symbol extraction
  - Event type classification
  - Importance scoring

Inspired by the Indian news sentiment project from Reddit (r/IndianStocks).
"""
import sys
import re
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.data.db import DBManager


# ── RSS Feeds ──────────────────────────────────────────────────────────────────

RSS_FEEDS = {
    "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "moneycontrol":   "https://www.moneycontrol.com/rss/latestnews.xml",
    "mint_markets":   "https://www.livemint.com/rss/markets",
    "business_std":   "https://www.business-standard.com/rss/markets-106.rss",
    "ndtv_profit":    "https://feeds.feedburner.com/ndtvprofit-latest",
    "hindu_business": "https://www.thehindubusinessline.com/markets/?service=rss",
}

# NSE F&O stocks for mention extraction
NSE_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
    "SBIN", "ITC", "LT", "HINDUNILVR", "KOTAKBANK", "AXISBANK",
    "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH",
    "ADANIENT", "TATAMOTORS", "TATASTEEL", "POWERGRID", "NTPC", "ONGC",
    "NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "RBI", "SEBI", "FII", "DII",
    "COAL INDIA", "JSWSTEEL", "DRREDDY", "BAJAJFINSV", "ASIANPAINT",
    "HDFC", "BAJAJ", "ADANI", "TATA", "INFOSYS", "WIPRO", "HCL",
]

# Event keywords → category
EVENT_KEYWORDS = {
    "results":       ["quarterly results", "q1", "q2", "q3", "q4", "earnings", "profit", "revenue", "PAT", "EBITDA"],
    "rbi_policy":    ["rbi", "repo rate", "monetary policy", "inflation", "rate cut", "rate hike"],
    "fii_dii":       ["fii", "dii", "foreign institutional", "domestic institutional", "bought", "sold"],
    "merger_acq":    ["merger", "acquisition", "takeover", "deal", "buy stake"],
    "ipo":           ["ipo", "listing", "subscribe", "allotment", "grey market"],
    "gst_budget":    ["gst", "budget", "tax", "government", "ministry", "policy"],
    "corporate":     ["dividend", "buyback", "bonus", "split", "rights issue", "order", "contract"],
    "global":        ["fed", "us market", "china", "crude oil", "dollar", "rupee", "global"],
}

# Boost sentiment for financial-specific positive/negative words
FINANCIAL_LEXICON = {
    "POSITIVE": [
        "outperform", "beat estimates", "strong buy", "upgrade", "rally",
        "all-time high", "record high", "order win", "strong results",
        "profit up", "revenue growth", "expansion", "rbi rate cut",
    ],
    "NEGATIVE": [
        "miss estimates", "downgrade", "sell off", "crash", "circuit",
        "loss", "write-off", "default", "npa", "fraud", "scam",
        "rbi rate hike", "inflation", "recession", "slowdown",
    ],
}


class NewsCollector:
    """Collects and analyzes Indian financial news."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Add financial lexicon to VADER
        self.vader.lexicon.update({w: 2.0 for w in FINANCIAL_LEXICON["POSITIVE"]})
        self.vader.lexicon.update({w: -2.0 for w in FINANCIAL_LEXICON["NEGATIVE"]})
        self._seen_ids = set()
        logger.info("NewsCollector initialized")

    def collect_all(self) -> List[Dict]:
        """Fetch news from all RSS feeds."""
        all_news = []
        for source, url in RSS_FEEDS.items():
            try:
                articles = self._fetch_rss(source, url)
                all_news.extend(articles)
                logger.debug(f"[News] {source}: {len(articles)} articles")
            except Exception as e:
                logger.warning(f"[News] {source} failed: {e}")
        return all_news

    def _fetch_rss(self, source: str, url: str) -> List[Dict]:
        """Parse RSS feed and process articles."""
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:20]:
            news_id = hashlib.md5(entry.get("link", entry.get("title", "")).encode()).hexdigest()[:12]
            if news_id in self._seen_ids:
                continue
            self._seen_ids.add(news_id)

            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title}. {summary}"
            pub_time = self._parse_time(entry)

            # Skip old news (>4 hours)
            if pub_time and datetime.utcnow() - pub_time > timedelta(hours=4):
                continue

            article = {
                "news_id":    news_id,
                "source":     source,
                "title":      title[:300],
                "summary":    summary[:500],
                "url":        entry.get("link", ""),
                "published":  pub_time.isoformat() if pub_time else datetime.utcnow().isoformat(),
                "symbols":    self._extract_symbols(text),
                "event_type": self._classify_event(text),
                **self._analyze_sentiment(text),
            }
            articles.append(article)
        return articles

    def _parse_time(self, entry) -> Optional[datetime]:
        try:
            import time as time_mod
            t = entry.get("published_parsed") or entry.get("updated_parsed")
            if t:
                return datetime(*t[:6])
        except Exception:
            pass
        return None

    def _analyze_sentiment(self, text: str) -> Dict:
        """VADER sentiment analysis with financial adjustments."""
        scores = self.vader.polarity_scores(text.lower())
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return {
            "sentiment":       label,
            "sentiment_score": round(compound, 4),
            "pos":             round(scores["pos"], 3),
            "neg":             round(scores["neg"], 3),
            "neu":             round(scores["neu"], 3),
        }

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract NSE symbols mentioned in article."""
        text_upper = text.upper()
        return [sym for sym in NSE_SYMBOLS if sym in text_upper]

    def _classify_event(self, text: str) -> str:
        """Classify news event type."""
        text_lower = text.lower()
        for event_type, keywords in EVENT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return event_type
        return "general"

    def get_symbol_sentiment(self, symbol: str, hours: int = 4) -> Dict:
        """Get aggregated sentiment for a specific symbol."""
        articles = self.collect_all()
        sym_articles = [a for a in articles if symbol.upper() in a.get("symbols", [])]
        if not sym_articles:
            return {"symbol": symbol, "sentiment": "NEUTRAL", "score": 0, "count": 0}
        scores = [a["sentiment_score"] for a in sym_articles]
        avg = sum(scores) / len(scores)
        return {
            "symbol":    symbol,
            "sentiment": "POSITIVE" if avg > 0.05 else ("NEGATIVE" if avg < -0.05 else "NEUTRAL"),
            "score":     round(avg, 4),
            "count":     len(sym_articles),
            "articles":  sym_articles[:5],
        }

    def get_market_sentiment(self) -> Dict:
        """Overall market sentiment from all news."""
        articles = self.collect_all()
        if not articles:
            return {"sentiment": "NEUTRAL", "score": 0, "count": 0}
        scores = [a["sentiment_score"] for a in articles]
        avg = sum(scores) / len(scores)
        events = {}
        for a in articles:
            et = a.get("event_type", "general")
            events[et] = events.get(et, 0) + 1
        return {
            "sentiment":    "POSITIVE" if avg > 0.05 else ("NEGATIVE" if avg < -0.05 else "NEUTRAL"),
            "score":        round(avg, 4),
            "count":        len(articles),
            "event_counts": events,
            "top_articles": articles[:5],
        }


# Singleton
_collector: Optional[NewsCollector] = None

def get_news_collector() -> NewsCollector:
    global _collector
    if _collector is None:
        _collector = NewsCollector()
    return _collector
