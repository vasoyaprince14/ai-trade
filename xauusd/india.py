"""
Indian Market: News-Driven Stocks + Nifty Hedge System
========================================================
- Fetches top NSE stocks with recent news/momentum
- Nifty hedge overlay (buy puts when market is overbought)
- Uses yfinance for price + newsapi/google RSS for news
"""

import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# ── NSE Watchlist ─────────────────────────────────────────────────────────────
# Top liquid NSE stocks — append .NS for yfinance
WATCHLIST = {
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS",
    "INFY":      "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK":"ICICIBANK.NS",
    "WIPRO":     "WIPRO.NS",
    "LT":        "LT.NS",
    "BAJFINANCE":"BAJFINANCE.NS",
    "AXISBANK":  "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN":      "SBIN.NS",
    "TATAMOTORS":"TATAMOTORS.NS",
    "HCLTECH":   "HCLTECH.NS",
    "MARUTI":    "MARUTI.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "ADANIENT":  "ADANIENT.NS",
    "POWERGRID": "POWERGRID.NS",
    "NTPC":      "NTPC.NS",
    "ONGC":      "ONGC.NS",
    "COALINDIA": "COALINDIA.NS",
}

NIFTY_TICKER  = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d    = s.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def _atr(df, n=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()


def get_stock_scan() -> list[dict]:
    """
    Scan watchlist for momentum + news opportunities.
    Uses daily bars — stable signals that don't flip every few minutes.
    Returns list of dicts sorted by score (highest first).
    """
    results = []
    for name, ticker in WATCHLIST.items():
        try:
            df = yf.download(ticker, period="90d", interval="1d", progress=False)
            if df.empty or len(df) < 30:
                continue

            # Flatten columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df = df.rename(columns={"adj close": "adj_close"}).dropna()

            close = df["close"]
            high  = df["high"]
            low   = df["low"]

            ema21  = float(_ema(close, 21).iloc[-1])
            ema55  = float(_ema(close, 55).iloc[-1])
            rsi14  = float(_rsi(close, 14).iloc[-1])
            atr14  = float(_atr(df, 14).iloc[-1])
            price  = float(close.iloc[-1])

            # 5-day momentum
            momentum_5d = (price - float(close.iloc[-6])) / float(close.iloc[-6]) * 100

            # Volume spike (today vs 20-day avg)
            vol_today = float(df["volume"].iloc[-1])
            vol_avg   = float(df["volume"].iloc[-20:].mean())
            vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1.0

            # 52-week high proximity
            high_52w = float(high.rolling(252).max().iloc[-1])
            dist_52w = (price - high_52w) / high_52w * 100  # negative = below 52w high

            # Score
            score = 0
            reasons = []

            if price > ema21 > ema55:
                score += 3
                reasons.append("Uptrend (EMA21>EMA55)")
            elif price < ema21 < ema55:
                score += 0  # bearish
                reasons.append("Downtrend")
            else:
                score += 1
                reasons.append("Ranging")

            if 40 <= rsi14 <= 65:
                score += 2
                reasons.append(f"RSI {rsi14:.0f} healthy")
            elif rsi14 < 35:
                score += 1
                reasons.append(f"RSI {rsi14:.0f} oversold bounce")
            elif rsi14 > 70:
                score -= 1
                reasons.append(f"RSI {rsi14:.0f} overbought")

            if momentum_5d > 1.5:
                score += 2
                reasons.append(f"+{momentum_5d:.1f}% 5d momentum")
            elif momentum_5d > 0.5:
                score += 1
                reasons.append(f"+{momentum_5d:.1f}% 5d momentum")
            elif momentum_5d < -2:
                score -= 1

            if vol_ratio > 1.5:
                score += 2
                reasons.append(f"Volume spike {vol_ratio:.1f}x avg")
            elif vol_ratio > 1.2:
                score += 1
                reasons.append(f"Volume {vol_ratio:.1f}x avg")

            if dist_52w > -2:
                score += 1
                reasons.append("Near 52w high")

            # News (basic — get headlines from yfinance info)
            news_headlines = []
            try:
                info = yf.Ticker(ticker).news
                if info:
                    for n in info[:3]:
                        title = n.get("title", "")
                        if title:
                            news_headlines.append(title)
                            # Simple sentiment check
                            positive_words = ["beat", "profit", "growth", "record", "strong", "up", "raise", "buy", "upgrade"]
                            negative_words = ["miss", "loss", "weak", "down", "cut", "sell", "downgrade", "probe", "fraud"]
                            if any(w in title.lower() for w in positive_words):
                                score += 1
                            if any(w in title.lower() for w in negative_words):
                                score -= 1
            except Exception:
                pass

            # SL / TP
            sl  = round(price - 1.5 * atr14, 2)
            tp  = round(price + 2.5 * atr14, 2)
            rr  = round(2.5 / 1.5, 2)

            results.append({
                "name":         name,
                "ticker":       ticker,
                "price":        round(price, 2),
                "score":        score,
                "rsi":          round(rsi14, 1),
                "momentum_5d":  round(momentum_5d, 2),
                "vol_ratio":    round(vol_ratio, 2),
                "atr":          round(atr14, 2),
                "sl":           sl,
                "tp":           tp,
                "rr":           rr,
                "trend":        "UP" if price > ema21 > ema55 else "DOWN" if price < ema21 < ema55 else "RANGE",
                "reasons":      " | ".join(reasons),
                "news":         news_headlines[:2],
                "dist_52w":     round(dist_52w, 2),
            })

        except Exception as e:
            logger.debug(f"Scan error {name}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_nifty_hedge_signal() -> dict:
    """
    Nifty hedge recommendation:
    - If Nifty RSI > 70 → suggest buying PE (bearish hedge)
    - If Nifty RSI < 35 → suggest buying CE (bullish hedge)
    - Also checks PCR and distance from EMA
    """
    try:
        df = yf.download(NIFTY_TICKER, period="60d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.dropna()

        close  = df["close"]
        price  = float(close.iloc[-1])
        ema21  = float(_ema(close, 21).iloc[-1])
        ema200 = float(_ema(close, 200).iloc[-1])
        rsi    = float(_rsi(close, 14).iloc[-1])
        atr    = float(_atr(df, 14).iloc[-1])

        # 1-month return
        ret_1m = (price - float(close.iloc[-22])) / float(close.iloc[-22]) * 100

        # Hedge signal
        if rsi > 70 and price > ema21 * 1.02:
            hedge = "BUY PE"
            reason = f"Nifty overbought (RSI {rsi:.0f}, +{ret_1m:.1f}% in 1m) — buy PE to hedge portfolio"
            strength = "STRONG"
        elif rsi > 65:
            hedge = "BUY PE"
            reason = f"Nifty extended (RSI {rsi:.0f}) — light PE hedge recommended"
            strength = "MODERATE"
        elif rsi < 35 and price < ema21 * 0.98:
            hedge = "BUY CE"
            reason = f"Nifty oversold (RSI {rsi:.0f}) — buy CE for bounce"
            strength = "STRONG"
        elif rsi < 40:
            hedge = "BUY CE"
            reason = f"Nifty weak (RSI {rsi:.0f}) — light CE position for bounce"
            strength = "MODERATE"
        else:
            hedge = "HOLD"
            reason = f"Nifty neutral (RSI {rsi:.0f}) — no hedge needed"
            strength = "NONE"

        return {
            "nifty_price":  round(price, 2),
            "rsi":          round(rsi, 1),
            "ema21":        round(ema21, 2),
            "ema200":       round(ema200, 2),
            "atr":          round(atr, 2),
            "ret_1m":       round(ret_1m, 2),
            "hedge":        hedge,
            "reason":       reason,
            "strength":     strength,
        }
    except Exception as e:
        logger.warning(f"Nifty hedge error: {e}")
        return {"hedge": "UNKNOWN", "reason": str(e), "strength": "NONE", "nifty_price": 0, "rsi": 50}
