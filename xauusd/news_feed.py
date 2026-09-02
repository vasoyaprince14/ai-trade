"""
XAUUSD News Feed + Economic Calendar
======================================
Aggregates gold-relevant news headlines and high-impact USD economic events.

Sources:
  1. yfinance  - GC=F gold futures news (free, no API key)
  2. ForexFactory calendar JSON  - https://nfs.faireconomy.media/ff_calendar_thisweek.json
     (free public endpoint, USD high-impact events)

Output (get_news_context()):
  {
    "headlines":       list[dict],   # [{title, publisher, age_min, sentiment, url}]
    "sentiment":       str,          # BULLISH / BEARISH / NEUTRAL
    "sentiment_score": float,        # -1.0 to +1.0
    "calendar":        list[dict],   # upcoming USD high-impact events
    "news_filter":     bool,         # True = high-impact event within 30 min → skip trade
    "news_filter_reason": str,       # e.g. "NFP in 12 min"
    "fetched_at":      str,
  }

Cached for 5 minutes so the OF engine doesn't hammer the endpoints.
"""

from __future__ import annotations
import json
import time
import calendar as _cal
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import yfinance as yf
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────
FF_CALENDAR_URL    = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_TTL          = 300          # 5 min cache
CALENDAR_CACHE_TTL = 3600         # 1 hour — calendar rarely changes intra-hour
FILTER_WINDOW_MIN  = 30           # block trades if event within 30 min
NEWS_LOOKBACK_MIN  = 240          # show headlines from last 4 hours
CACHE_FILE         = Path("/tmp/xauusd_news_cache.json")
CALENDAR_CACHE_FILE= Path("/tmp/xauusd_calendar_cache.json")

# Gold-bullish keywords in headlines
_BULL_KEYWORDS = [
    "surge", "rally", "gain", "rise", "jump", "bullish", "safe haven",
    "haven demand", "buy gold", "gold up", "higher", "upside", "rate cut",
    "fed dovish", "inflation hedge", "dollar weak", "dollar falls",
    "geopolit", "tension", "conflict", "uncertainty", "recession fears",
    "lower yields", "bond yields fall", "central bank buy", "reserve",
]
_BEAR_KEYWORDS = [
    "fall", "drop", "decline", "sell", "bearish", "gold down", "lower",
    "dollar strong", "dollar rally", "rate hike", "hawkish", "fed hike",
    "risk on", "equities rally", "strong jobs", "nfp beat", "cpi hot",
    "yields rise", "bond yields surge", "profit taking", "outflow",
]

# High-impact USD events that move gold
_GOLD_MOVERS = [
    "non-farm", "nfp", "fomc", "federal reserve", "interest rate decision",
    "cpi", "inflation", "pce", "gdp", "ism manufacturing", "ism services",
    "retail sales", "unemployment", "average hourly earnings", "jackson hole",
    "powell", "dot plot", "qe", "tapering", "treasury",
]

_cache: dict = {}
_cache_ts: float = 0.0


def _score_headline(title: str) -> float:
    """Simple keyword sentiment score. Returns -1.0 to +1.0."""
    t = title.lower()
    bull = sum(1 for kw in _BULL_KEYWORDS if kw in t)
    bear = sum(1 for kw in _BEAR_KEYWORDS if kw in t)
    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 2)


def _fetch_gold_news() -> list[dict]:
    """Fetch GC=F news from yfinance."""
    try:
        ticker = yf.Ticker("GC=F")
        raw = ticker.get_news(count=20) or []
        results = []
        now = datetime.now(timezone.utc)
        for item in raw:
            content = item.get("content", item)
            title = content.get("title", "")
            if not title:
                continue
            pub_str = content.get("pubDate", "")
            pub_dt  = None
            if pub_str:
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except Exception:
                    pass
            age_min = int((now - pub_dt).total_seconds() / 60) if pub_dt else 9999
            if age_min > NEWS_LOOKBACK_MIN:
                continue
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
            url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
            results.append({
                "title":     title,
                "publisher": publisher,
                "age_min":   age_min,
                "sentiment": _score_headline(title),
                "url":       url,
            })
        return results
    except Exception as e:
        logger.warning(f"[News] yfinance news error: {e}")
        return []


def _fetch_calendar() -> list[dict]:
    """Fetch this week's ForexFactory USD high-impact calendar (disk-cached 1h)."""
    # Check disk cache first — ForexFactory rate-limits aggressive callers
    try:
        if CALENDAR_CACHE_FILE.exists():
            age = time.time() - CALENDAR_CACHE_FILE.stat().st_mtime
            if age < CALENDAR_CACHE_TTL:
                cached_raw = json.loads(CALENDAR_CACHE_FILE.read_text())
                events = cached_raw
                # Still filter by time window below
                logger.debug(f"[News] Calendar: using disk cache ({int(age)}s old)")
                return _parse_calendar_events(events)
    except Exception:
        pass

    try:
        r = requests.get(FF_CALENDAR_URL, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        events = r.json()
        # Save raw to disk
        try:
            CALENDAR_CACHE_FILE.write_text(json.dumps(events))
        except Exception:
            pass
        return _parse_calendar_events(events)
    except Exception as e:
        logger.warning(f"[News] ForexFactory calendar error: {e}")
        # Try stale disk cache as last resort
        try:
            if CALENDAR_CACHE_FILE.exists():
                cached_raw = json.loads(CALENDAR_CACHE_FILE.read_text())
                logger.info("[News] Calendar: using stale disk cache after error")
                return _parse_calendar_events(cached_raw)
        except Exception:
            pass
        # Final fallback: rule-based schedule approximation
        logger.info("[News] Calendar: using rule-based event approximation")
        return _approx_upcoming_events()


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """Return the nth weekday (0=Mon..6=Sun) of the given month."""
    first = datetime(year, month, 1, tzinfo=timezone.utc)
    offset = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=offset)
    return first_occurrence + timedelta(weeks=n - 1)


def _approx_upcoming_events() -> list[dict]:
    """
    Rule-based approximation of known USD high-impact event schedule.
    Used as fallback when ForexFactory is unavailable.
    Covers: NFP (1st Fri), FOMC (8 dates/year), CPI (~2nd Wed), PCE (~last Fri)
    """
    now = datetime.now(timezone.utc)
    events = []

    # ── NFP: First Friday of each month, 08:30 ET = 13:30 UTC ─────────────
    for m_off in range(0, 2):
        dt = now + timedelta(days=30 * m_off)
        year, month = dt.year, dt.month
        if month > 12:
            month -= 12; year += 1
        nfp = _nth_weekday(year, month, 4, 1).replace(hour=13, minute=30)  # 4=Friday
        events.append({
            "title": "Non-Farm Employment Change", "impact": "High",
            "date": nfp.strftime("%Y-%m-%d %H:%M UTC"),
            "minutes_away": int((nfp - now).total_seconds() / 60),
            "forecast": "", "previous": "", "is_gold_mover": True,
        })

    # ── CPI: ~2nd Wednesday of each month, 08:30 ET = 13:30 UTC ──────────
    for m_off in range(0, 2):
        dt = now + timedelta(days=30 * m_off)
        year, month = dt.year, dt.month
        cpi = _nth_weekday(year, month, 2, 2).replace(hour=13, minute=30)  # 2=Wednesday
        events.append({
            "title": "CPI m/m (Inflation)", "impact": "High",
            "date": cpi.strftime("%Y-%m-%d %H:%M UTC"),
            "minutes_away": int((cpi - now).total_seconds() / 60),
            "forecast": "", "previous": "", "is_gold_mover": True,
        })

    # Filter to upcoming 8h and sort
    events = [e for e in events if 0 <= e["minutes_away"] <= 480]
    events.sort(key=lambda x: x["minutes_away"])
    return events


def _parse_calendar_events(events: list) -> list[dict]:
    """Filter and format raw ForexFactory event list."""
    results = []
    now = datetime.now(timezone.utc)
    for ev in events:
        if ev.get("country") != "USD":
            continue
        if ev.get("impact") not in ("High", "Medium"):
            continue
        date_str = ev.get("date", "")
        if not date_str:
            continue
        try:
            ev_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if ev_dt.tzinfo is None:
                ev_dt = ev_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        minutes_away = int((ev_dt - now).total_seconds() / 60)
        if not (-30 <= minutes_away <= 480):
            continue
        title = ev.get("title", "")
        is_gold_mover = any(kw in title.lower() for kw in _GOLD_MOVERS)
        results.append({
            "title":         title,
            "impact":        ev.get("impact", ""),
            "date":          ev_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "minutes_away":  minutes_away,
            "forecast":      ev.get("forecast", ""),
            "previous":      ev.get("previous", ""),
            "is_gold_mover": is_gold_mover,
        })
    results.sort(key=lambda x: x["minutes_away"])
    return results


def get_news_context(force: bool = False) -> dict:
    """
    Main entry point. Returns aggregated news + calendar context for XAUUSD.
    Results are cached for CACHE_TTL seconds.
    """
    global _cache, _cache_ts

    now_ts = time.time()
    if not force and _cache and (now_ts - _cache_ts) < CACHE_TTL:
        return _cache

    headlines = _fetch_gold_news()
    calendar  = _fetch_calendar()

    # Aggregate sentiment
    if headlines:
        avg = sum(h["sentiment"] for h in headlines) / len(headlines)
        if avg >= 0.15:
            sentiment = "BULLISH"
        elif avg <= -0.15:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        sentiment_score = round(avg, 3)
    else:
        sentiment = "NEUTRAL"
        sentiment_score = 0.0

    # News filter: block if high-impact gold-mover within FILTER_WINDOW_MIN
    news_filter = False
    news_filter_reason = ""
    for ev in calendar:
        if ev["is_gold_mover"] and ev["impact"] == "High":
            ma = ev["minutes_away"]
            if -5 <= ma <= FILTER_WINDOW_MIN:
                news_filter = True
                label = "NOW" if ma <= 0 else f"in {ma} min"
                news_filter_reason = f"{ev['title']} {label}"
                break

    result = {
        "headlines":          headlines,
        "sentiment":          sentiment,
        "sentiment_score":    sentiment_score,
        "calendar":           calendar,
        "news_filter":        news_filter,
        "news_filter_reason": news_filter_reason,
        "fetched_at":         datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    }

    # Persist to disk for dashboard
    try:
        CACHE_FILE.write_text(json.dumps(result, indent=2, default=str))
    except Exception:
        pass

    _cache    = result
    _cache_ts = now_ts
    return result


def news_summary_line(ctx: dict) -> str:
    """One-line summary for console / engine output."""
    sent  = ctx["sentiment"]
    score = ctx["sentiment_score"]
    n     = len(ctx["headlines"])
    cal   = len(ctx["calendar"])
    filt  = " | ** NEWS FILTER ACTIVE **" if ctx["news_filter"] else ""
    return (f"News: {sent} ({score:+.2f}) | {n} headlines | "
            f"{cal} upcoming USD events{filt}")


def print_news_context(ctx: dict):
    """Print a human-readable news summary to console."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  GOLD NEWS FEED  [{ctx['fetched_at']}]")
    print(sep)

    # Sentiment
    sent = ctx["sentiment"]
    marker = "+" if sent == "BULLISH" else ("-" if sent == "BEARISH" else "=")
    print(f"  Sentiment: [{marker}] {sent}  (score={ctx['sentiment_score']:+.2f})")

    if ctx["news_filter"]:
        print(f"  !! NEWS FILTER: {ctx['news_filter_reason']} — avoid new trades !!")

    # Headlines
    if ctx["headlines"]:
        print(f"\n  Recent Headlines (last {NEWS_LOOKBACK_MIN//60}h):")
        for h in ctx["headlines"][:6]:
            s = "+" if h["sentiment"] > 0.1 else ("-" if h["sentiment"] < -0.1 else "=")
            print(f"  [{s}] {h['title'][:75]}  ({h['age_min']}m ago)")

    # Calendar
    if ctx["calendar"]:
        print(f"\n  Upcoming USD Events (next 8h):")
        for ev in ctx["calendar"][:6]:
            ma = ev["minutes_away"]
            when = "NOW" if ma <= 0 else f"in {ma}m"
            star = " *" if ev["is_gold_mover"] else ""
            impact_tag = f"[{ev['impact'][:1]}]"
            print(f"  {impact_tag} {ev['title'][:45]:<45} {when}{star}")

    print(sep)
