"""
Finology Ticker Scraper
========================
Scrapes ticker.finology.in for NSE stock fundamentals, holdings data,
and screening metrics. Used to enrich the Nifty strategy with institutional
holding context.

Usage:
    from core.data.finology import get_stock_data, get_nifty50_screen, get_nifty50_holdings_summary

Data available per stock:
  - Price: Market Cap, P/E, P/B, EPS, Div Yield, 52W High/Low
  - Fundamentals: ROE, ROCE, Sales Growth, Profit Growth, Debt/Equity
  - Holdings: Promoter %, Promoter Pledging %, quarterly history
  - Screening score: composite bullishness rating 0-10
"""

from __future__ import annotations
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://ticker.finology.in/company/{ticker}"
CACHE_DIR  = Path("/tmp/finology_cache")
CACHE_TTL  = 3600          # 1 hour per stock
RATE_LIMIT = 1.5           # seconds between requests

_last_req_ts = 0.0
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

# Nifty 50 tickers as used on Finology
NIFTY50_TICKERS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
    "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHREECEM", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _fetch_html(url: str) -> str:
    global _last_req_ts
    elapsed = time.time() - _last_req_ts
    if elapsed < RATE_LIMIT:
        time.sleep(RATE_LIMIT - elapsed)
    try:
        resp = _session.get(url, timeout=15)
        _last_req_ts = time.time()
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"[Finology] Fetch error {url}: {e}")
        raise


def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker.upper()}.json"


def _cache_load(ticker: str) -> Optional[dict]:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        fetched_at = datetime.fromisoformat(data.get("_fetched_at", "2000-01-01"))
        if datetime.utcnow() - fetched_at < timedelta(seconds=CACHE_TTL):
            return data
    except Exception:
        pass
    return None


def _cache_save(ticker: str, data: dict):
    try:
        data["_fetched_at"] = datetime.utcnow().isoformat()
        _cache_path(ticker).write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.warning(f"[Finology] Cache save error: {e}")


# ── HTML parsing helpers ───────────────────────────────────────────────────────
def _text(html: str, pattern: str) -> str:
    """Extract first match text, stripping tags and HTML entities."""
    m = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
    if not m:
        return ""
    raw = m.group(1) if m.lastindex else m.group(0)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = raw.replace("&#8377;", "").replace("&nbsp;", "").replace("&amp;", "&")
    raw = raw.replace("Cr.", "").replace("%", "").strip()
    return raw


def _float(s: str) -> Optional[float]:
    s = s.strip().lstrip("₹").replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_compess_blocks(html: str) -> dict:
    """Parse all <div class='col-* compess'> label→value blocks."""
    result = {}
    for m in re.finditer(
        r'<div class="col-\d+ col-md-\d+ compess">(.*?)</div>',
        html, re.DOTALL
    ):
        block = m.group(1)
        labels = re.findall(r"<small[^>]*>(.*?)</small>", block, re.DOTALL)
        vals   = re.findall(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
        label = re.sub(r"<[^>]+>", "", " ".join(labels)).strip()
        val   = re.sub(r"<[^>]+>", "", " ".join(vals)).strip()
        # Clean HTML entities
        val = val.replace("&#8377;", "").replace("&nbsp;", "").replace("Cr.", "").strip()
        if label and val:
            result[label] = val
    return result


def _parse_promoter_history(html: str) -> list[dict]:
    """
    Extract quarterly promoter + pledging history from the Pledge % table.
    Returns list of {quarter, promoter, pledging} dicts (oldest first).
    """
    history = []
    idx = html.find("Pledge %")
    if idx < 0:
        return history
    section = html[idx: idx + 3000]
    rows = re.findall(
        r"<tr[^>]*>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>",
        section,
    )
    for quarter, promo, pledge in rows:
        entry = {
            "quarter": quarter.strip(),
            "promoter": _float(promo.strip()),
            "pledging": _float(pledge.strip()),
        }
        history.append(entry)
    # Return oldest-first
    return list(reversed(history))


def _parse_institutional_holding(html: str) -> dict:
    """
    Extract latest FII, DII, Public, Mutual Fund holdings from the
    InstitutionalInvestors section or summary text in the page.
    """
    holding = {}

    # Try to find FII/DII/Public from summary narrative text
    # e.g. "FII holding of 23.45%"
    for pattern, key in [
        (r"FII.{0,50}?(\d+\.?\d*)\s*%", "fii_pct"),
        (r"DII.{0,50}?(\d+\.?\d*)\s*%", "dii_pct"),
        (r"[Pp]ublic.{0,60}?(\d+\.?\d*)\s*%", "public_pct"),
        (r"[Mm]utual\s+[Ff]und.{0,50}?(\d+\.?\d*)\s*%", "mf_pct"),
    ]:
        m = re.search(pattern, html)
        if m:
            holding[key] = _float(m.group(1))

    # Try table-based extraction
    for cat, key in [
        ("FII", "fii_pct"),
        ("DII", "dii_pct"),
        ("Mutual Fund", "mf_pct"),
    ]:
        m = re.search(
            rf'{cat}.{{0,200}}?(\d{{1,3}}\.?\d*)\s*%',
            html, re.DOTALL
        )
        if m and key not in holding:
            holding[key] = _float(m.group(1))

    return holding


# ── Main scraper ───────────────────────────────────────────────────────────────
def get_stock_data(ticker: str, force: bool = False) -> dict:
    """
    Scrape Finology company page for ticker and return structured data.

    Returns dict with keys:
      ticker, company_name, price_high, price_low, week52_high, week52_low,
      market_cap_cr, pe, pb, eps_ttm, div_yield, roe, roce, sales_growth,
      profit_growth, debt_cr, book_value, cash_cr, promoter_pct,
      promoter_pledging_pct, fii_pct, dii_pct, mf_pct, public_pct,
      promoter_history (list), screener_score (0-10),
      screener_signals (list[str]), _fetched_at
    """
    ticker = ticker.upper()

    if not force:
        cached = _cache_load(ticker)
        if cached:
            return cached

    url = BASE_URL.format(ticker=ticker)
    logger.info(f"[Finology] Fetching {url}")

    try:
        html = _fetch_html(url)
    except Exception as e:
        logger.error(f"[Finology] Failed to fetch {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

    data: dict = {"ticker": ticker}

    # Company name
    data["company_name"] = _text(html, r'id="mainContent_ltrlCompName"[^>]*>([^<]+)<')

    # Parse all compess metric blocks
    metrics = _parse_compess_blocks(html)
    mapping = {
        "Market Cap":      ("market_cap_cr", None),
        "P/E":             ("pe", None),
        "P/B":             ("pb", None),
        "EPS (TTM)":       ("eps_ttm", None),
        "Div. Yield":      ("div_yield", None),
        "52 Week High":    ("week52_high", None),
        "52 Week Low":     ("week52_low", None),
        "CASH":            ("cash_cr", None),
        "DEBT":            ("debt_cr", None),
        "Promoter Holding":("promoter_pct", None),
        "ROE":             ("roe", None),
        "ROCE":            ("roce", None),
        "Sales Growth":    ("sales_growth", None),
        "Profit Growth":   ("profit_growth", None),
        "Book Value (TTM)":("book_value", None),
    }
    for label, (key, _) in mapping.items():
        val = metrics.get(label, "")
        if val:
            # Remove trailing % and parse
            v = val.replace("%", "").strip()
            data[key] = _float(v)

    # Promoter quarterly history (also gives us pledging)
    data["promoter_history"] = _parse_promoter_history(html)

    # Most recent pledging from history (last entry = most recent)
    if data["promoter_history"]:
        data["promoter_pledging_pct"] = data["promoter_history"][-1].get("pledging")

    # Institutional holding
    inst = _parse_institutional_holding(html)
    data.update(inst)

    # Derive a simple screener score (0-10) for Nifty strategy integration
    data["screener_score"], data["screener_signals"] = _compute_screener_score(data)

    _cache_save(ticker, data)
    return data


def _compute_screener_score(d: dict) -> tuple[float, list[str]]:
    """
    Compute a 0-10 score and list of signals from stock fundamentals.
    Higher = more fundamentally bullish for Nifty analysis.
    """
    score = 0.0
    signals = []

    pe  = d.get("pe")
    pb  = d.get("pb")
    roe = d.get("roe")
    roce = d.get("roce")
    promo = d.get("promoter_pct")
    pledging = d.get("promoter_pledging_pct", 0) or 0
    sg  = d.get("sales_growth")
    pg  = d.get("profit_growth")
    de_ratio = None
    debt = d.get("debt_cr")
    mcap = d.get("market_cap_cr")
    div  = d.get("div_yield")
    fii  = d.get("fii_pct")

    # --- Valuation (2 pts) ---
    if pe and 0 < pe < 25:
        score += 1.0; signals.append(f"Cheap P/E {pe:.1f}")
    elif pe and 25 <= pe < 40:
        score += 0.5

    if pb and pb < 3:
        score += 1.0; signals.append(f"Low P/B {pb:.1f}")
    elif pb and pb < 5:
        score += 0.5

    # --- Profitability (2 pts) ---
    if roe and roe > 20:
        score += 1.0; signals.append(f"High ROE {roe:.1f}%")
    elif roe and roe > 12:
        score += 0.5

    if roce and roce > 20:
        score += 1.0; signals.append(f"High ROCE {roce:.1f}%")
    elif roce and roce > 12:
        score += 0.5

    # --- Growth (2 pts) ---
    if sg and sg > 10:
        score += 1.0; signals.append(f"Sales growth {sg:.1f}%")
    elif sg and sg > 0:
        score += 0.5

    if pg and pg > 15:
        score += 1.0; signals.append(f"Profit growth {pg:.1f}%")
    elif pg and pg > 0:
        score += 0.5

    # --- Promoter holding quality (2 pts) ---
    if promo and promo > 50:
        score += 1.0; signals.append(f"High promoter {promo:.1f}%")
    elif promo and promo > 30:
        score += 0.5

    if pledging < 5:
        score += 1.0; signals.append(f"Low pledging {pledging:.1f}%")
    elif pledging < 15:
        score += 0.5

    # Check if promoter holding increased in last 2 quarters
    hist = d.get("promoter_history", [])
    if len(hist) >= 3:
        recent = [h.get("promoter") for h in hist[-3:] if h.get("promoter") is not None]
        if len(recent) >= 2 and recent[-1] > recent[0]:
            score += 0.5; signals.append("Promoter increasing")

    # --- FII interest (1 pt) ---
    if fii and fii > 20:
        score += 0.5; signals.append(f"High FII {fii:.1f}%")

    # --- Dividend (0.5 pt) ---
    if div and div > 1.5:
        score += 0.5; signals.append(f"Good div yield {div:.1f}%")

    return round(min(score, 10.0), 2), signals


# ── Batch / screener ───────────────────────────────────────────────────────────
def get_nifty50_screen(tickers: list[str] | None = None,
                       max_stocks: int = 50) -> list[dict]:
    """
    Scrape fundamentals for all Nifty 50 stocks (or given list).
    Returns list of dicts sorted by screener_score descending.
    Rate-limited to ~1.5s per stock.
    """
    tickers = tickers or NIFTY50_TICKERS
    results = []
    for i, t in enumerate(tickers[:max_stocks]):
        logger.info(f"[Finology] Scraping {t} ({i+1}/{len(tickers)})")
        try:
            d = get_stock_data(t)
            if "error" not in d:
                results.append(d)
        except Exception as e:
            logger.warning(f"[Finology] Error for {t}: {e}")
    results.sort(key=lambda x: x.get("screener_score", 0), reverse=True)
    return results


def get_nifty50_holdings_summary(tickers: list[str] | None = None) -> dict:
    """
    Aggregate holdings summary across Nifty 50:
      - avg promoter holding
      - avg FII holding
      - stocks with rising promoter holding (bullish signal)
      - stocks with high pledging (risk)
      - overall institutional sentiment score
    """
    stocks = get_nifty50_screen(tickers)

    promos   = [s["promoter_pct"] for s in stocks if s.get("promoter_pct")]
    fiis     = [s["fii_pct"]      for s in stocks if s.get("fii_pct")]
    pledges  = [s.get("promoter_pledging_pct", 0) or 0 for s in stocks]
    scores   = [s["screener_score"] for s in stocks if s.get("screener_score") is not None]

    rising_promoter = [
        s["ticker"] for s in stocks
        if len(s.get("promoter_history", [])) >= 2
        and all(h.get("promoter") is not None for h in s["promoter_history"][-2:])
        and s["promoter_history"][-1]["promoter"] > s["promoter_history"][-2]["promoter"]
    ]

    high_pledge = [
        s["ticker"] for s in stocks
        if (s.get("promoter_pledging_pct") or 0) > 20
    ]

    avg_promo  = round(sum(promos) / len(promos), 2)  if promos  else None
    avg_fii    = round(sum(fiis) / len(fiis), 2)      if fiis    else None
    avg_pledge = round(sum(pledges) / len(pledges), 2) if pledges else None
    avg_score  = round(sum(scores) / len(scores), 2)   if scores  else None

    # Institutional sentiment: 0-10 based on FII + promoter trends
    inst_sentiment = 5.0
    if avg_fii and avg_fii > 25:
        inst_sentiment += 1.0
    if avg_promo and avg_promo > 50:
        inst_sentiment += 1.0
    if len(rising_promoter) > 15:
        inst_sentiment += 1.0
    if len(high_pledge) < 5:
        inst_sentiment += 0.5
    if avg_score and avg_score > 6:
        inst_sentiment += 1.0

    return {
        "stock_count":          len(stocks),
        "avg_promoter_pct":     avg_promo,
        "avg_fii_pct":          avg_fii,
        "avg_pledging_pct":     avg_pledge,
        "avg_screener_score":   avg_score,
        "rising_promoter":      rising_promoter,
        "high_pledge_stocks":   high_pledge,
        "inst_sentiment_score": round(min(inst_sentiment, 10.0), 1),
        "top_stocks":           [s["ticker"] for s in stocks[:10]],
        "stocks":               stocks,
        "_fetched_at":          datetime.utcnow().isoformat(),
    }


def print_stock_summary(d: dict):
    """Print a compact summary of a stock's Finology data."""
    sep = "-" * 50
    print(sep)
    print(f"  {d.get('ticker')} — {d.get('company_name', '')}")
    print(sep)
    print(f"  P/E: {d.get('pe')}  |  P/B: {d.get('pb')}  |  EPS: {d.get('eps_ttm')}")
    print(f"  ROE: {d.get('roe')}%  |  ROCE: {d.get('roce')}%")
    print(f"  Sales Growth: {d.get('sales_growth')}%  |  Profit Growth: {d.get('profit_growth')}%")
    print(f"  Promoter: {d.get('promoter_pct')}%  |  Pledging: {d.get('promoter_pledging_pct')}%")
    print(f"  FII: {d.get('fii_pct')}%  |  MF: {d.get('mf_pct')}%")
    print(f"  Market Cap: ₹{d.get('market_cap_cr')} Cr  |  Debt: ₹{d.get('debt_cr')} Cr")
    print(f"  Screener Score: {d.get('screener_score')}/10")
    if d.get("screener_signals"):
        print(f"  Signals: {' | '.join(d['screener_signals'])}")
    print(sep)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RELIANCE", "TCS", "INFY"]
    if tickers == ["--nifty50"]:
        summary = get_nifty50_holdings_summary()
        print(f"\nNifty 50 Holdings Summary ({summary['stock_count']} stocks)")
        print(f"  Avg Promoter: {summary['avg_promoter_pct']}%")
        print(f"  Avg FII:      {summary['avg_fii_pct']}%")
        print(f"  Avg Pledge:   {summary['avg_pledging_pct']}%")
        print(f"  Inst Score:   {summary['inst_sentiment_score']}/10")
        print(f"  Rising Promoters: {summary['rising_promoter']}")
        print(f"  High Pledge Risk: {summary['high_pledge_stocks']}")
        print(f"  Top Screened:     {summary['top_stocks']}")
    else:
        for t in tickers:
            d = get_stock_data(t, force=True)
            print_stock_summary(d)
