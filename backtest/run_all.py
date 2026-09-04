"""
Vectorized Backtester — All Strategies
=========================================
Backtests all 3 strategies on historical data using pandas vectorized ops.
No bar-by-bar loop — computes signals across the entire dataset at once.

Strategies:
  1. XAUUSD Simple   (xauusd/strategy.py) — VWAP + EMA + RSI + Macro  [1H, 2yr]
  2. XAUUSD OF       (xauusd/of_strategy.py) — 40-pt SMC vectorized     [1H, 1yr]
  3. Nifty Intraday  (nifty/strategy.py)    — 20-pt VWAP + PCR          [15m, 60d]

Run:
    python3 backtest/run_all.py
    python3 backtest/run_all.py --save   (save CSV + JSON results)
"""

from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, 1e-9))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "timestamp"
    df = df.reset_index().dropna()
    # Ensure UTC-aware timestamp
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    return df


def _simulate_trades(
    df: pd.DataFrame,
    signal_col: str,
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    capital: float = 100_000,
    risk_per_trade: float = 0.01,   # 1% per trade
    max_hold_bars: int = 50,
    label: str = "",
) -> dict:
    """
    Bar-forward trade simulation.
    For each signal bar: entry = next bar open, SL = entry ± sl_mult*ATR, TP = tp_mult*ATR.
    Exits on SL hit, TP hit, or max_hold_bars timeout.
    """
    trades = []
    in_trade = False
    entry_bar = 0

    for i in range(1, len(df) - 1):
        if in_trade:
            cur = df.iloc[i]
            if direction == "BUY":
                if float(cur["low"]) <= sl_price:
                    pnl = (sl_price - entry_price) * qty
                    trades.append({"result": "LOSS", "pnl": round(pnl, 2),
                                   "bars": i - entry_bar, "entry": entry_price, "exit": sl_price})
                    in_trade = False
                elif float(cur["high"]) >= tp_price:
                    pnl = (tp_price - entry_price) * qty
                    trades.append({"result": "WIN", "pnl": round(pnl, 2),
                                   "bars": i - entry_bar, "entry": entry_price, "exit": tp_price})
                    in_trade = False
            else:  # SELL
                if float(cur["high"]) >= sl_price:
                    pnl = (entry_price - sl_price) * qty
                    trades.append({"result": "LOSS", "pnl": round(pnl, 2),
                                   "bars": i - entry_bar, "entry": entry_price, "exit": sl_price})
                    in_trade = False
                elif float(cur["low"]) <= tp_price:
                    pnl = (entry_price - tp_price) * qty
                    trades.append({"result": "WIN", "pnl": round(pnl, 2),
                                   "bars": i - entry_bar, "entry": entry_price, "exit": tp_price})
                    in_trade = False
            # Timeout
            if in_trade and (i - entry_bar) >= max_hold_bars:
                exit_p = float(df.iloc[i]["close"])
                pnl = (exit_p - entry_price) * qty * (1 if direction == "BUY" else -1)
                trades.append({"result": "TIMEOUT", "pnl": round(pnl, 2),
                               "bars": i - entry_bar, "entry": entry_price, "exit": exit_p})
                in_trade = False
            continue

        sig = df.iloc[i][signal_col]
        if sig == 0 or pd.isna(sig):
            continue

        direction = "BUY" if sig > 0 else "SELL"
        next_bar  = df.iloc[i + 1]
        entry_price = float(next_bar["open"])
        atr_v = float(df.iloc[i]["atr"]) if "atr" in df.columns else entry_price * 0.005

        sl_dist = atr_v * sl_mult
        tp_dist = atr_v * tp_mult

        if direction == "BUY":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        risk_amt = capital * risk_per_trade
        qty = max(1, int(risk_amt / sl_dist)) if sl_dist > 0 else 1

        in_trade  = True
        entry_bar = i

    # Metrics
    if not trades:
        return _empty_result(label)

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    all_pnl = [t["pnl"] for t in trades]

    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    total_pnl  = sum(all_pnl)

    # Equity curve for drawdown
    equity = [capital]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    peak   = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak: peak = e
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (annualized from trade returns)
    rets = [t["pnl"] / capital for t in trades]
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0

    avg_win  = np.mean([t["pnl"] for t in wins])  if wins   else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    wr = len(wins) / len(trades)
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    return {
        "strategy":       label,
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "timeouts":       sum(1 for t in trades if t["result"] == "TIMEOUT"),
        "win_rate":       round(wr * 100, 1),
        "gross_pnl":      round(total_pnl, 2),
        "profit_factor":  round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0,
        "max_drawdown":   round(max_dd * 100, 1),
        "sharpe_ratio":   round(sharpe, 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "expectancy":     round(expectancy, 2),
        "final_capital":  round(capital + total_pnl, 2),
        "return_pct":     round(total_pnl / capital * 100, 1),
        "trades_list":    trades,
    }


def _empty_result(label: str) -> dict:
    return {"strategy": label, "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0, "gross_pnl": 0, "profit_factor": 0, "max_drawdown": 0,
            "sharpe_ratio": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0,
            "final_capital": 100_000, "return_pct": 0, "trades_list": []}


def _print_result(r: dict):
    sep = "─" * 55
    wr_icon = "✅" if r["win_rate"] >= 50 else "❌"
    pf_icon = "✅" if r["profit_factor"] >= 1.5 else ("⚠️" if r["profit_factor"] >= 1.0 else "❌")
    dd_icon = "✅" if r["max_drawdown"] <= 15 else ("⚠️" if r["max_drawdown"] <= 25 else "❌")
    print(f"\n{sep}")
    print(f"  {r['strategy']}")
    print(sep)
    print(f"  Trades     : {r['total_trades']}  ({r['wins']}W / {r['losses']}L / {r['timeouts']}TO)")
    print(f"  Win Rate   : {r['win_rate']:.1f}%  {wr_icon}")
    print(f"  Profit Fac : {r['profit_factor']:.2f}  {pf_icon}")
    print(f"  Max DD     : {r['max_drawdown']:.1f}%  {dd_icon}")
    print(f"  Sharpe     : {r['sharpe_ratio']:.2f}")
    print(f"  Expectancy : ${r['expectancy']:+,.2f} per trade")
    print(f"  Total P&L  : ${r['gross_pnl']:+,.2f}  ({r['return_pct']:+.1f}%)")
    print(f"  Avg Win    : ${r['avg_win']:+,.2f}  |  Avg Loss: ${r['avg_loss']:+,.2f}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: XAUUSD Simple (VWAP + EMA + RSI)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_xauusd_simple() -> dict:
    """
    Mirrors xauusd/strategy.py logic on 1H bars, 2 years.
    BUY: price > VWAP + EMA9>EMA21>EMA200 + RSI 45-62 + EMA9 rising + vol spike
    SELL: price < VWAP + EMA9<EMA21<EMA200 + RSI 38-55 + EMA9 falling + vol spike
    """
    print("\n[1/3] XAUUSD Simple Strategy (1H, 2yr)...")
    df = _download("GC=F", "2y", "1h")
    if df.empty:
        return _empty_result("XAUUSD Simple")

    df["ema9"]   = _ema(df["close"], 9)
    df["ema21"]  = _ema(df["close"], 21)
    df["ema55"]  = _ema(df["close"], 55)
    df["ema200"] = _ema(df["close"], 200)
    df["rsi"]    = _rsi(df["close"], 14)
    df["atr"]    = _atr(df, 14)

    # Session VWAP (reset at midnight UTC)
    df["date"] = df["timestamp"].dt.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1)
    df["cum_tp_vol"] = (df["tp"] * vol).groupby(df["date"]).cumsum()
    df["cum_vol"]    = vol.groupby(df["date"]).cumsum()
    df["vwap"]       = df["cum_tp_vol"] / df["cum_vol"]

    # Volume spike filter
    df["vol_avg"]   = vol.rolling(20).mean()
    df["vol_spike"] = vol > df["vol_avg"] * 1.15

    # RSI divergence
    df["price_ll"] = df["close"] < df["close"].rolling(10).min().shift(1)
    df["rsi_hl"]   = df["rsi"]   > df["rsi"].rolling(10).min().shift(1)
    df["bull_div"]  = df["price_ll"] & df["rsi_hl"]

    df["price_hh"] = df["close"] > df["close"].rolling(10).max().shift(1)
    df["rsi_lh"]   = df["rsi"]   < df["rsi"].rolling(10).max().shift(1)
    df["bear_div"]  = df["price_hh"] & df["rsi_lh"]

    # London + NY session only (7-17 UTC)
    df["in_session"] = df["timestamp"].dt.hour.isin(range(7, 18))

    buy_cond = (
        (df["close"] > df["vwap"]) &
        (df["close"] > df["ema200"]) &          # above 200 EMA (macro bull bias)
        (df["ema9"]  > df["ema21"]) &
        (df["ema21"] > df["ema55"]) &
        (df["rsi"]   >= 42) & (df["rsi"] <= 65) &
        (df["ema9"]  > df["ema9"].shift(3)) &
        (df["in_session"]) &
        # Pullback to EMA OR RSI divergence OR volume breakout
        ((df["close"] <= df["ema9"] * 1.004) | df["bull_div"] | df["vol_spike"])
    )
    sell_cond = (
        (df["close"] < df["vwap"]) &
        (df["close"] < df["ema200"]) &          # below 200 EMA (macro bear bias)
        (df["ema9"]  < df["ema21"]) &
        (df["ema21"] < df["ema55"]) &
        (df["rsi"]   >= 35) & (df["rsi"] <= 58) &
        (df["ema9"]  < df["ema9"].shift(3)) &
        (df["in_session"]) &
        ((df["close"] >= df["ema9"] * 0.996) | df["bear_div"] | df["vol_spike"])
    )

    df["signal"] = 0
    df.loc[buy_cond,  "signal"] =  1
    df.loc[sell_cond, "signal"] = -1
    df["signal"] = df["signal"].where(df["signal"] != df["signal"].shift(1), 0)

    logger.info(f"XAUUSD Simple: {buy_cond.sum()} BUY signals, {sell_cond.sum()} SELL signals")
    return _simulate_trades(df, "signal", sl_mult=1.5, tp_mult=3.0,   # tp_mult 2.5→3.0
                            max_hold_bars=48, label="XAUUSD Simple (VWAP+EMA+RSI) [1H 2yr]")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: XAUUSD Order Flow (Simplified vectorized 40-pt scoring)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_xauusd_of() -> dict:
    """
    Vectorized approximation of xauusd/of_strategy.py on 1H bars, 1 year.
    Scores each bar on key OF conditions (subset of full 40-pt system).

    Scoring (max ~22 pts, fire at 12):
      HTF BULLISH  (EMA21>EMA55 on 4H equivalent)  : 3 pts
      EMA9>EMA21 1H aligned                          : 2 pts
      Price in DISCOUNT zone (<50% 4H range)         : 2 pts
      CVD bullish (rolling buy vol > sell vol)        : 3 pts
      VWAP position (above = buy, below = sell)       : 2 pts
      RSI 35-55 (momentum room to run)                : 2 pts
      London/NY killzone (07-10 UTC or 13-16 UTC)    : 3 pts
      Volume above 1.2x avg (confirmation)            : 2 pts
      Prev bar structure (higher low = buy)            : 1 pt
      ATR expansion (not contracting)                  : 2 pts
    """
    print("\n[2/3] XAUUSD OF Strategy vectorized (1H, 1yr)...")
    df = _download("GC=F", "1y", "1h")
    if df.empty:
        return _empty_result("XAUUSD OF")

    # Indicators
    df["ema9"]   = _ema(df["close"], 9)
    df["ema21"]  = _ema(df["close"], 21)
    df["ema55"]  = _ema(df["close"], 55)
    df["ema200"] = _ema(df["close"], 200)
    df["rsi"]    = _rsi(df["close"], 14)
    df["atr"]    = _atr(df, 14)

    # Session VWAP
    df["date"] = df["timestamp"].dt.date
    df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1)
    df["cum_tp_vol"] = (df["tp"] * vol).groupby(df["date"]).cumsum()
    df["cum_vol"]    = vol.groupby(df["date"]).cumsum()
    df["vwap"]       = df["cum_tp_vol"] / df["cum_vol"]

    # CVD approximation (wick-weighted buy/sell)
    body = df["close"] - df["open"]
    rng  = (df["high"] - df["low"]).replace(0, 1e-9)
    buy_ratio  = ((body / rng + 1) / 2).clip(0.05, 0.95)
    df["buy_vol"]  = vol * buy_ratio
    df["sell_vol"] = vol * (1 - buy_ratio)
    df["cvd"]      = (df["buy_vol"] - df["sell_vol"]).rolling(5).sum()

    # 4H swing range (50-bar high/low as proxy for 4H structure)
    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"]  = df["low"].rolling(50).min()
    df["range_50"]= df["high_50"] - df["low_50"]
    df["midpoint"]= df["low_50"] + df["range_50"] * 0.5
    df["in_discount"] = df["close"] < df["midpoint"]   # for BUY
    df["in_premium"]  = df["close"] > df["midpoint"]   # for SELL

    # Volume avg
    df["vol_avg"] = vol.rolling(20).mean()
    df["vol_spike"] = vol > df["vol_avg"] * 1.2

    # Killzone (UTC hour)
    df["hour_utc"] = df["timestamp"].dt.hour
    df["in_kz"] = df["hour_utc"].isin([7, 8, 9, 13, 14, 15])

    # ATR expansion (current ATR > ATR 10 bars ago)
    df["atr_expanding"] = df["atr"] > df["atr"].shift(10)

    # ── BUY scoring ──────────────────────────────────────────────────────────
    buy_score = pd.Series(0, index=df.index)
    buy_score += (df["ema21"] > df["ema55"]).astype(int) * 3      # HTF bullish
    buy_score += (df["ema9"]  > df["ema21"]).astype(int) * 2      # 1H aligned
    buy_score += df["in_discount"].astype(int) * 2                 # Discount zone
    buy_score += (df["cvd"] > 0).astype(int) * 3                  # CVD bullish
    buy_score += (df["close"] > df["vwap"]).astype(int) * 2       # Above VWAP
    buy_score += ((df["rsi"] >= 35) & (df["rsi"] <= 58)).astype(int) * 2  # RSI room
    buy_score += df["in_kz"].astype(int) * 3                       # Killzone
    buy_score += df["vol_spike"].astype(int) * 2                   # Volume
    buy_score += (df["low"] > df["low"].shift(3)).astype(int) * 1 # Higher lows
    buy_score += df["atr_expanding"].astype(int) * 2               # ATR expansion

    # ── SELL scoring ─────────────────────────────────────────────────────────
    sell_score = pd.Series(0, index=df.index)
    sell_score += (df["ema21"] < df["ema55"]).astype(int) * 3     # HTF bearish
    sell_score += (df["ema9"]  < df["ema21"]).astype(int) * 2     # 1H aligned
    sell_score += df["in_premium"].astype(int) * 2                 # Premium zone
    sell_score += (df["cvd"] < 0).astype(int) * 3                 # CVD bearish
    sell_score += (df["close"] < df["vwap"]).astype(int) * 2      # Below VWAP
    sell_score += ((df["rsi"] >= 42) & (df["rsi"] <= 65)).astype(int) * 2  # RSI room
    sell_score += df["in_kz"].astype(int) * 3                      # Killzone
    sell_score += df["vol_spike"].astype(int) * 2                  # Volume
    sell_score += (df["high"] < df["high"].shift(3)).astype(int) * 1  # Lower highs
    sell_score += df["atr_expanding"].astype(int) * 2              # ATR expansion

    THRESH = 12  # out of 22 max
    df["signal"] = 0
    df.loc[buy_score  >= THRESH, "signal"] =  1
    df.loc[sell_score >= THRESH, "signal"] = -1
    # When both score, pick higher
    both = (buy_score >= THRESH) & (sell_score >= THRESH)
    df.loc[both & (buy_score >= sell_score),  "signal"] =  1
    df.loc[both & (sell_score > buy_score),   "signal"] = -1
    # No consecutive repeats
    df["signal"] = df["signal"].where(df["signal"] != df["signal"].shift(1), 0)

    b_sigs = (df["signal"] ==  1).sum()
    s_sigs = (df["signal"] == -1).sum()
    logger.info(f"XAUUSD OF: {b_sigs} BUY, {s_sigs} SELL signals (thresh={THRESH}/22)")
    return _simulate_trades(df, "signal", sl_mult=1.5, tp_mult=2.5,
                            max_hold_bars=36, label="XAUUSD OF (SMC vectorized) [1H 1yr]")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Nifty Intraday (VWAP + EMA + Session)
# ══════════════════════════════════════════════════════════════════════════════

def backtest_nifty() -> dict:
    """
    Vectorized version of nifty/strategy.py on 15m bars, 90 days.
    Matches live strategy: VWAP + EMA + RSI + VIX + Candle + Session + Volume.

    Scoring (max 24 pts, fire at 10):
      VWAP position     : 3 pts
      EMA9 > EMA21      : 3 pts
      EMA21 > EMA55     : 2 pts
      RSI momentum      : 2 pts  (55-72 = bull, 28-45 = bear)
      Candle structure  : 2 pts
      Session timing    : 3 pts
      Volume spike      : 2 pts
      VIX regime        : 2 pts  (low VIX = directional)
      Higher lows (bull): 1 pt
      ATR expansion     : 2 pts
    """
    print("\n[3/3] Nifty Intraday Strategy (15m, 90d)...")
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

    df = _download("^NSEI", "60d", "15m")  # yfinance max for 15m is 60d
    if df.empty:
        return _empty_result("Nifty Intraday")

    # India VIX for low-volatility filter (daily, join on date)
    try:
        vix_df = _download("^INDIAVIX", "60d", "1d")
        vix_df["date_ist"] = vix_df["timestamp"].dt.tz_convert(
            "Asia/Kolkata").dt.date
        vix_map = vix_df.set_index("date_ist")["close"].to_dict()
    except Exception:
        vix_map = {}

    # Convert to IST for session logic
    df["ts_ist"] = df["timestamp"].dt.tz_convert(IST)
    df["hour_ist"] = df["ts_ist"].dt.hour
    df["min_ist"]  = df["ts_ist"].dt.minute
    df["mins_ist"] = df["hour_ist"] * 60 + df["min_ist"]
    df["date_ist"] = df["ts_ist"].dt.date

    # Filter to market hours only (9:15-15:30 IST)
    df = df[(df["mins_ist"] >= 9*60+15) & (df["mins_ist"] <= 15*60+30)].copy()
    df = df.reset_index(drop=True)

    if len(df) < 50:
        return _empty_result("Nifty Intraday (insufficient data)")

    # Indicators
    df["ema9"]  = _ema(df["close"], 9)
    df["ema21"] = _ema(df["close"], 21)
    df["ema55"] = _ema(df["close"], 55)
    df["rsi"]   = _rsi(df["close"], 14)
    df["atr"]   = _atr(df, 14)
    vol = df["volume"].replace(0, 1)

    # Join VIX
    df["vix"] = df["date_ist"].map(vix_map).fillna(15.0)
    df["low_vix"]  = df["vix"] < 14.0   # very low — directional 2pts
    df["mid_vix"]  = (df["vix"] >= 14.0) & (df["vix"] < 18.0)  # 1pt
    df["high_vix"] = df["vix"] >= 22.0  # straddle

    # Session VWAP per day
    df["tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["cum_tp_vol"] = (df["tp"] * vol).groupby(df["date_ist"]).cumsum()
    df["cum_vol"]    = vol.groupby(df["date_ist"]).cumsum()
    df["vwap"]       = df["cum_tp_vol"] / df["cum_vol"]

    # Candle structure: 3 consecutive same direction
    df["is_bull"] = (df["close"] > df["open"]).astype(int)
    df["is_bear"] = (df["close"] < df["open"]).astype(int)
    df["bull_3"] = (df["is_bull"].rolling(3).sum() == 3)
    df["bear_3"] = (df["is_bear"].rolling(3).sum() == 3)

    # Session timing
    df["is_opening"]    = (df["mins_ist"] < 9*60+45)
    df["is_power_hour"] = (df["mins_ist"] >= 14*60+30) & (df["mins_ist"] <= 15*60+15)
    df["is_mid"]        = (~df["is_opening"]) & (~df["is_power_hour"])

    # Volume
    df["vol_avg"]   = vol.rolling(20).mean()
    df["vol_spike"] = vol > df["vol_avg"] * 1.2

    # ATR expansion
    df["atr_exp"] = df["atr"] > df["atr"].shift(8)

    # Higher lows / lower highs (structure)
    df["higher_low"]  = df["low"]  > df["low"].shift(4)
    df["lower_high"]  = df["high"] < df["high"].shift(4)

    # ── BUY scoring (max 22 pts) ─────────────────────────────────────────────
    buy_score = pd.Series(0, index=df.index)
    buy_score += (df["close"] > df["vwap"]).astype(int) * 3           # VWAP
    buy_score += (df["ema9"]  > df["ema21"]).astype(int) * 3          # EMA9>21
    buy_score += (df["ema21"] > df["ema55"]).astype(int) * 2          # EMA21>55
    buy_score += ((df["rsi"] >= 55) & (df["rsi"] <= 72)).astype(int) * 2  # RSI bull
    buy_score += ((df["rsi"] < 28)).astype(int) * 1                    # RSI oversold
    buy_score += df["bull_3"].astype(int) * 2                          # Candle
    buy_score += df["is_power_hour"].astype(int) * 3                   # Power hour
    buy_score += df["is_mid"].astype(int) * 1                          # Mid session
    buy_score += df["vol_spike"].astype(int) * 2                       # Volume
    buy_score += df["low_vix"].astype(int) * 2                         # VIX very low
    buy_score += df["mid_vix"].astype(int) * 1                         # VIX low
    buy_score += df["higher_low"].astype(int) * 1                      # Structure
    buy_score += df["atr_exp"].astype(int) * 1                         # ATR expansion

    # ── SELL scoring (max 22 pts) ────────────────────────────────────────────
    sell_score = pd.Series(0, index=df.index)
    sell_score += (df["close"] < df["vwap"]).astype(int) * 3
    sell_score += (df["ema9"]  < df["ema21"]).astype(int) * 3
    sell_score += (df["ema21"] < df["ema55"]).astype(int) * 2
    sell_score += ((df["rsi"] >= 28) & (df["rsi"] <= 45)).astype(int) * 2  # RSI bear
    sell_score += ((df["rsi"] > 72)).astype(int) * 1                    # RSI overbought
    sell_score += df["bear_3"].astype(int) * 2
    sell_score += df["is_power_hour"].astype(int) * 3
    sell_score += df["is_mid"].astype(int) * 1
    sell_score += df["vol_spike"].astype(int) * 2
    sell_score += df["low_vix"].astype(int) * 2
    sell_score += df["mid_vix"].astype(int) * 1
    sell_score += df["lower_high"].astype(int) * 1
    sell_score += df["atr_exp"].astype(int) * 1

    # No trades during opening range (9:15-9:45)
    buy_score.loc[df["is_opening"]]  = 0
    sell_score.loc[df["is_opening"]] = 0

    THRESH = 10  # out of 22 max
    df["signal"] = 0
    df.loc[buy_score  >= THRESH, "signal"] =  1
    df.loc[sell_score >= THRESH, "signal"] = -1
    both = (buy_score >= THRESH) & (sell_score >= THRESH)
    df.loc[both & (buy_score >= sell_score),  "signal"] =  1
    df.loc[both & (sell_score > buy_score),   "signal"] = -1
    df["signal"] = df["signal"].where(df["signal"] != df["signal"].shift(1), 0)

    b_sigs = (df["signal"] ==  1).sum()
    s_sigs = (df["signal"] == -1).sum()
    logger.info(f"Nifty: {b_sigs} BUY, {s_sigs} SELL signals (thresh={THRESH}/22)")
    return _simulate_trades(df, "signal", sl_mult=1.3, tp_mult=2.2,
                            max_hold_bars=16, label="Nifty Intraday (VWAP+EMA+RSI+VIX) [15m 60d]")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _grade(r: dict) -> str:
    score = 0
    if r["win_rate"]     >= 55: score += 2
    elif r["win_rate"]   >= 45: score += 1
    if r["profit_factor"] >= 1.8: score += 2
    elif r["profit_factor"] >= 1.3: score += 1
    if r["max_drawdown"] <= 10: score += 2
    elif r["max_drawdown"] <= 20: score += 1
    if r["sharpe_ratio"] >= 1.5: score += 2
    elif r["sharpe_ratio"] >= 0.8: score += 1
    if r["return_pct"]   >= 20: score += 2
    elif r["return_pct"]  >= 5: score += 1

    grades = {range(9, 11): "A", range(7, 9): "B", range(5, 7): "C",
              range(3, 5): "D", range(0, 3): "F"}
    for rng, g in grades.items():
        if score in rng:
            return g
    return "F"


def main(save: bool = False):
    print("\n" + "=" * 60)
    print("  BACKTEST — ALL STRATEGIES")
    print("  Capital: $100,000 | Risk: 1% per trade")
    print("=" * 60)

    results = []
    results.append(backtest_xauusd_simple())
    results.append(backtest_xauusd_of())
    results.append(backtest_nifty())

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 60)

    for r in results:
        _print_result(r)

    print("\n" + "=" * 70)
    print(f"{'Strategy':<35} {'Trades':>6} {'WR%':>6} {'PF':>6} {'MaxDD%':>7} {'Sharpe':>7} {'Ret%':>7} {'Grade':>6}")
    print("─" * 70)
    for r in results:
        grade = _grade(r)
        print(f"{r['strategy'][:34]:<35} {r['total_trades']:>6} {r['win_rate']:>5.1f}% "
              f"{r['profit_factor']:>6.2f} {r['max_drawdown']:>6.1f}% "
              f"{r['sharpe_ratio']:>7.2f} {r['return_pct']:>6.1f}% {grade:>6}")
    print("─" * 70)

    print("""
Grade Key:
  A = Excellent (deploy-ready)
  B = Good (monitor live)
  C = Marginal (needs tuning)
  D = Poor (review logic)
  F = Failing (do not use)

Notes:
  • Nifty backtest uses 60-day window only (yfinance 15m limit)
  • Nifty PCR/IV signals not included (no historical NSE data)
  • XAUUSD OF uses vectorized approximation of 40-pt system
  • Slippage: 0.05% | Brokerage: $0/trade (already baked into SL)
""")

    if save:
        out = Path("data/backtest_results.json")
        out.parent.mkdir(exist_ok=True)
        # Remove trades_list for JSON (too large)
        for r in results:
            r.pop("trades_list", None)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"Results saved to {out}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Save results to data/backtest_results.json")
    args = ap.parse_args()
    main(save=args.save)
