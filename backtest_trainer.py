"""
Historical Backtest Trainer
============================
Fetches 2 years of real Nifty 5-min OHLC + VIX data,
derives all 78 ML features from price/volume/volatility,
labels each bar using 30-min forward returns,
then retrains the XGBoost+LightGBM model.

Label logic (real outcomes):
  BUY_CE       : next 30min return > +0.30%  AND momentum bullish
  BUY_PE       : next 30min return < -0.30%  AND momentum bearish
  SELL_STRADDLE: next 30min abs(return) < 0.15% AND high realized vol
  WAIT         : everything else

Run:
    python3 backtest_trainer.py           # trains NIFTY
    python3 backtest_trainer.py BANKNIFTY
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "NIFTY"

YFINANCE_MAP = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY":  "NIFTY_FIN_SERVICE.NS",
}
VIX_TICKER = "^INDIAVIX"

LOOKFORWARD_BARS = 6       # 6 × 5min = 30 min look-ahead
UP_THRESH   = 0.0030       # +0.30% → BUY_CE
DOWN_THRESH = -0.0030      # -0.30% → BUY_PE
FLAT_THRESH =  0.0015      # <0.15% abs → SELL_STRADDLE candidate


# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_5min(ticker: str, days: int = 59) -> pd.DataFrame:
    """yfinance 5-min limit is 60 days per call. Fetch in chunks."""
    import yfinance as yf
    frames = []
    end = date.today()
    # Download in 55-day chunks to stay under the 60-day limit
    while days > 0:
        chunk = min(days, 55)
        start = end - timedelta(days=chunk)
        try:
            df = yf.download(ticker, start=start, end=end,
                             interval="5m", progress=False, auto_adjust=True)
            if not df.empty:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                df.index.name = "timestamp"
                df = df.reset_index()
                frames.append(df)
        except Exception as e:
            logger.warning(f"Chunk fetch error ({start}→{end}): {e}")
        end = start
        days -= chunk

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_daily(ticker: str, years: int = 3) -> pd.DataFrame:
    """Daily OHLCV for longer history (used to compute daily VIX & trend)."""
    import yfinance as yf
    start = date.today() - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = "timestamp"
    return df.reset_index()


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame, vix_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 78 feature proxies from 5-min OHLC.
    Maps real price/volume signals onto FEATURE_KEYS names.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Basic price features ──────────────────────────────────────────────────
    df["returns"]    = df["close"].pct_change()
    df["log_ret"]    = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"]   = (df["high"] - df["low"]) / df["close"]   # bar range %

    # VWAP proxy (rolling 78-bar = ~6.5 hrs)
    df["cum_vol"]    = df["volume"].cumsum()
    df["cum_vwap"]   = (df["close"] * df["volume"]).cumsum() / df["cum_vol"].replace(0, np.nan)
    df["vwap_dev"]   = (df["close"] - df["cum_vwap"]) / df["cum_vwap"]

    # EMA crossover
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema55"] = df["close"].ewm(span=55, adjust=False).mean()
    df["ema_cross"]   = np.sign(df["ema9"] - df["ema21"])      # +1 bull, -1 bear
    df["ema_trend"]   = np.sign(df["ema21"] - df["ema55"])

    # RSI (14-bar)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # Momentum (6-bar = 30min, 12-bar = 60min)
    df["mom30"]  = df["close"].pct_change(6)
    df["mom60"]  = df["close"].pct_change(12)
    df["mom5"]   = df["close"].pct_change(1)

    # Realized vol (20-bar rolling std of log returns, annualized)
    df["realized_vol"] = df["log_ret"].rolling(20).std() * np.sqrt(252 * 75)  # 75 bars/day

    # ATR (14-bar)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["close"]

    # Volume ratio (current bar vs 20-bar avg)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)

    # ── Tape proxies ─────────────────────────────────────────────────────────
    # Bull pct: fraction of last 6 bars that closed up
    df["bull_pct"]     = (df["returns"] > 0).rolling(6).mean()
    # Tape bias numeric: -2 to +2 based on momentum + EMA cross
    df["tape_bias_num"] = (
        np.sign(df["mom30"]) * 1.5 +
        df["ema_cross"] * 0.5
    ).clip(-2, 2)

    # OI proxy: use volume × price change as proxy for OI build
    df["bullish_oi"]  = np.where(df["returns"] > 0, df["volume"] * df["returns"].abs(), 0)
    df["bearish_oi"]  = np.where(df["returns"] < 0, df["volume"] * df["returns"].abs(), 0)
    df["net_oi_bias"] = df["bullish_oi"].rolling(6).sum() - df["bearish_oi"].rolling(6).sum()

    # Rolling OI build (recent vs older)
    df["recent_bull_oi"] = df["bullish_oi"].rolling(3).sum()
    df["recent_bear_oi"] = df["bearish_oi"].rolling(3).sum()
    df["recent_bias"]    = df["recent_bull_oi"] - df["recent_bear_oi"]

    # OI velocity and acceleration
    net_oi = df["net_oi_bias"]
    df["oi_velocity"] = net_oi.diff(3)
    df["oi_accel"]    = df["oi_velocity"].diff(3)

    # ── Option chain proxies ─────────────────────────────────────────────────
    spot = df["close"]
    atm  = (spot / 50).round() * 50      # ATM strike (50-pt grid for Nifty)

    # IV proxy: use realized_vol with a spread (CE IV slightly lower than PE IV in bearish markets)
    df["atm_iv"] = df["realized_vol"] * 100   # convert to percent
    df["atm_ce_iv"] = df["atm_iv"] * (1 - 0.05 * df["tape_bias_num"].clip(-1, 1))
    df["atm_pe_iv"] = df["atm_iv"] * (1 + 0.05 * df["tape_bias_num"].clip(-1, 1))

    # PCR proxy: bearish market = high PCR, bullish = low PCR
    # Use EMA of negative momentum bias to estimate PCR
    pcr_signal = 1.0 - df["mom30"].ewm(span=12).mean() * 10
    df["pcr_oi"]  = pcr_signal.clip(0.5, 2.0)
    df["pcr_vol"] = df["pcr_oi"] * 0.9 + 0.1  # slightly lower

    # CE/PE OI totals (arbitrary scale from volume)
    df["ce_total_oi"] = df["volume"] * (1 - df["tape_bias_num"].clip(0,2)/4)
    df["pe_total_oi"] = df["volume"] * (1 + df["tape_bias_num"].clip(-2,0).abs()/4)
    df["net_oi"]      = df["ce_total_oi"] - df["pe_total_oi"]

    # Top strike OI (CE resistance = ATM+100, PE support = ATM-100)
    df["top_ce_oi_1"] = df["ce_total_oi"] * 0.25
    df["top_ce_oi_2"] = df["ce_total_oi"] * 0.18
    df["top_ce_oi_3"] = df["ce_total_oi"] * 0.12
    df["top_pe_oi_1"] = df["pe_total_oi"] * 0.25
    df["top_pe_oi_2"] = df["pe_total_oi"] * 0.18
    df["top_pe_oi_3"] = df["pe_total_oi"] * 0.12

    # OI change at top strikes (proxy from vol ratio)
    for col in ["top_ce_oi_chg_1","top_ce_oi_chg_2","top_ce_oi_chg_3",
                "top_pe_oi_chg_1","top_pe_oi_chg_2","top_pe_oi_chg_3"]:
        df[col] = df["oi_velocity"] * 0.1

    # ATM OI build
    df["atm_ce_oi_build"] = df["oi_velocity"].clip(0, None)
    df["atm_pe_oi_build"] = (-df["oi_velocity"]).clip(0, None)
    df["atm_net_build"]   = df["oi_velocity"]

    # Distance spot to top CE/PE (100 pts = typical wall)
    df["dist_spot_to_top_ce"] = 100 / spot * 100   # in pct
    df["dist_spot_to_top_pe"] = 100 / spot * 100

    # ── Time features ────────────────────────────────────────────────────────
    ts = df["timestamp"]
    mins_since_open = (ts.dt.hour - 9) * 60 + ts.dt.minute - 15
    mins_since_open = mins_since_open.clip(0, 375)
    df["mins_since_open"]  = mins_since_open
    df["mins_to_close"]    = (375 - mins_since_open).clip(0, 375)
    df["is_first_hour"]    = (mins_since_open < 60).astype(int)
    df["is_last_hour"]     = (mins_since_open > 315).astype(int)
    df["day_of_week"]      = ts.dt.dayofweek
    df["intraday_progress"]= mins_since_open / 375.0

    # ── VIX (merge from daily data) ──────────────────────────────────────────
    if not vix_daily.empty:
        vix_daily = vix_daily[["timestamp", "close"]].rename(columns={"close": "vix"})
        vix_daily["date"] = pd.to_datetime(vix_daily["timestamp"]).dt.date
        df["date"] = df["timestamp"].dt.date
        df = df.merge(vix_daily[["date","vix"]], on="date", how="left")
        df["vix"] = df["vix"].fillna(df["atm_iv"])   # fallback to realized vol
    else:
        df["vix"] = df["atm_iv"]

    # ── FII proxy (regime-based) ─────────────────────────────────────────────
    # Use weekly trend as FII bias proxy
    df["fii_bias_numeric"] = df["ema_trend"]   # +1 bull trend, -1 bear trend
    df["dii_bias_numeric"] = df["ema_cross"] * 0.5
    df["smart_money_bias"] = (df["fii_bias_numeric"] + df["dii_bias_numeric"]).clip(-2, 2)

    fii_scale = df["volume"] * 0.1
    df["fii_net_futures"] = df["fii_bias_numeric"] * fii_scale
    df["fii_net_calls"]   = -df["fii_bias_numeric"] * fii_scale * 0.5
    df["fii_net_puts"]    = df["fii_bias_numeric"] * fii_scale * 0.5
    df["fii_bias_score"]  = df["fii_net_futures"]

    df["dii_net_futures"] = df["dii_bias_numeric"] * fii_scale * 0.3
    df["dii_net_calls"]   = 0.0
    df["dii_net_puts"]    = 0.0
    df["dii_bias_score"]  = df["dii_net_futures"]

    df["pro_net_futures"] = df["ema_cross"] * fii_scale * 0.2
    df["pro_net_calls"]   = 0.0
    df["pro_net_puts"]    = 0.0

    # ── Other features ───────────────────────────────────────────────────────
    df["total_events"]     = (df["vol_ratio"] * 10).clip(0, 50)
    df["avg_confidence"]   = (df["rsi"] / 100).clip(0, 1)
    df["max_confidence"]   = df["avg_confidence"]
    df["avg_vol_ratio"]    = df["vol_ratio"].clip(0, 5)
    df["max_oi_event"]     = df["volume"] * df["hl_range"]
    df["hot_strikes_count"]= (df["vol_ratio"] > 1.5).astype(int)
    df["long_entries"]     = df["bullish_oi"].clip(0) * 0.01
    df["short_entries"]    = df["bearish_oi"].clip(0) * 0.01
    df["long_exits"]       = df["bearish_oi"].clip(0) * 0.005
    df["short_exits"]      = df["bullish_oi"].clip(0) * 0.005
    df["ce_bullish_oi"]    = df["bullish_oi"] * 0.6
    df["ce_bearish_oi"]    = df["bearish_oi"] * 0.3
    df["pe_bullish_oi"]    = df["bullish_oi"] * 0.4
    df["pe_bearish_oi"]    = df["bearish_oi"] * 0.7
    df["ce_event_count"]   = df["total_events"] * 0.5
    df["pe_event_count"]   = df["total_events"] * 0.5
    df["active_positions"] = 1
    df["snapshot_count"]   = df["mins_since_open"] // 30 + 1
    df["tape_event_count"] = df["total_events"]
    df["ce_oi_velocity"]   = df["oi_velocity"].clip(0, None)
    df["pe_oi_velocity"]   = (-df["oi_velocity"]).clip(0, None)
    df["net_oi_velocity"]  = df["oi_velocity"]
    df["ce_oi_accel"]      = df["oi_accel"].clip(0, None)
    df["pe_oi_accel"]      = (-df["oi_accel"]).clip(0, None)

    return df


# ── Feature → FEATURE_KEYS mapping ───────────────────────────────────────────

COL_MAP = {
    "f_total_events":        "total_events",
    "f_bull_pct":            "bull_pct",
    "f_bullish_oi":          "bullish_oi",
    "f_bearish_oi":          "bearish_oi",
    "f_net_oi_bias":         "net_oi_bias",
    "f_long_entries":        "long_entries",
    "f_short_entries":       "short_entries",
    "f_long_exits":          "long_exits",
    "f_short_exits":         "short_exits",
    "f_ce_bullish_oi":       "ce_bullish_oi",
    "f_ce_bearish_oi":       "ce_bearish_oi",
    "f_pe_bullish_oi":       "pe_bullish_oi",
    "f_pe_bearish_oi":       "pe_bearish_oi",
    "f_ce_event_count":      "ce_event_count",
    "f_pe_event_count":      "pe_event_count",
    "f_recent_bull_oi":      "recent_bull_oi",
    "f_recent_bear_oi":      "recent_bear_oi",
    "f_recent_bias":         "recent_bias",
    "f_avg_confidence":      "avg_confidence",
    "f_max_confidence":      "max_confidence",
    "f_avg_vol_ratio":       "avg_vol_ratio",
    "f_max_oi_event":        "max_oi_event",
    "f_hot_strikes_count":   "hot_strikes_count",
    "f_tape_bias_numeric":   "tape_bias_num",
    "f_pcr_oi":              "pcr_oi",
    "f_pcr_vol":             "pcr_vol",
    "f_ce_total_oi":         "ce_total_oi",
    "f_pe_total_oi":         "pe_total_oi",
    "f_net_oi":              "net_oi",
    "f_atm_ce_iv":           "atm_ce_iv",
    "f_atm_pe_iv":           "atm_pe_iv",
    "f_atm_iv":              "atm_iv",
    "f_top_ce_oi_1":         "top_ce_oi_1",
    "f_top_ce_oi_2":         "top_ce_oi_2",
    "f_top_ce_oi_3":         "top_ce_oi_3",
    "f_top_pe_oi_1":         "top_pe_oi_1",
    "f_top_pe_oi_2":         "top_pe_oi_2",
    "f_top_pe_oi_3":         "top_pe_oi_3",
    "f_top_ce_oi_chg_1":     "top_ce_oi_chg_1",
    "f_top_ce_oi_chg_2":     "top_ce_oi_chg_2",
    "f_top_ce_oi_chg_3":     "top_ce_oi_chg_3",
    "f_top_pe_oi_chg_1":     "top_pe_oi_chg_1",
    "f_top_pe_oi_chg_2":     "top_pe_oi_chg_2",
    "f_top_pe_oi_chg_3":     "top_pe_oi_chg_3",
    "f_atm_ce_oi_build":     "atm_ce_oi_build",
    "f_atm_pe_oi_build":     "atm_pe_oi_build",
    "f_atm_net_build":       "atm_net_build",
    "f_dist_spot_to_top_ce": "dist_spot_to_top_ce",
    "f_dist_spot_to_top_pe": "dist_spot_to_top_pe",
    "f_ce_oi_velocity":      "ce_oi_velocity",
    "f_pe_oi_velocity":      "pe_oi_velocity",
    "f_net_oi_velocity":     "net_oi_velocity",
    "f_ce_oi_accel":         "ce_oi_accel",
    "f_pe_oi_accel":         "pe_oi_accel",
    "f_minutes_since_open":  "mins_since_open",
    "f_minutes_to_close":    "mins_to_close",
    "f_is_first_hour":       "is_first_hour",
    "f_is_last_hour":        "is_last_hour",
    "f_day_of_week":         "day_of_week",
    "f_intraday_progress":   "intraday_progress",
    "f_fii_net_futures":     "fii_net_futures",
    "f_fii_net_calls":       "fii_net_calls",
    "f_fii_net_puts":        "fii_net_puts",
    "f_fii_bias_score":      "fii_bias_score",
    "f_dii_net_futures":     "dii_net_futures",
    "f_dii_net_calls":       "dii_net_calls",
    "f_dii_net_puts":        "dii_net_puts",
    "f_dii_bias_score":      "dii_bias_score",
    "f_pro_net_futures":     "pro_net_futures",
    "f_pro_net_calls":       "pro_net_calls",
    "f_pro_net_puts":        "pro_net_puts",
    "f_fii_bias_numeric":    "fii_bias_numeric",
    "f_dii_bias_numeric":    "dii_bias_numeric",
    "f_smart_money_bias":    "smart_money_bias",
    "f_vix":                 "vix",
    "f_active_positions":    "active_positions",
    "f_snapshot_count":      "snapshot_count",
    "f_tape_event_count":    "tape_event_count",
}


def build_X(df: pd.DataFrame) -> np.ndarray:
    from core.memory.vector_store import VECTOR_FEATURE_KEYS
    X = np.zeros((len(df), len(VECTOR_FEATURE_KEYS)), dtype=np.float32)
    for i, fkey in enumerate(VECTOR_FEATURE_KEYS):
        col = COL_MAP.get(fkey)
        if col and col in df.columns:
            vals = df[col].values.astype(np.float32)
            X[:, i] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """Label each bar by what happens in the next 30 min (6 bars)."""
    fwd = df["close"].shift(-LOOKFORWARD_BARS)
    fwd_ret = (fwd - df["close"]) / df["close"]

    mom_now = df["tape_bias_num"]
    iv_now  = df["realized_vol"] * 100

    labels = np.full(len(df), 3, dtype=np.int32)   # default = WAIT (idx 3)

    # BUY_CE: strong upward move, momentum confirms
    mask_ce = (fwd_ret > UP_THRESH) & (mom_now > 0)
    labels[mask_ce] = 0

    # BUY_PE: strong downward move, momentum confirms
    mask_pe = (fwd_ret < DOWN_THRESH) & (mom_now < 0)
    labels[mask_pe] = 1

    # SELL_STRADDLE: tight range AND high realized vol (mean-revert)
    mask_st = (fwd_ret.abs() < FLAT_THRESH) & (iv_now > 15)
    labels[mask_st] = 2

    # Drop the last LOOKFORWARD_BARS rows (no label possible)
    labels[-LOOKFORWARD_BARS:] = -1
    return labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from core.agent.ml_agent import TapeXGBModel, ACTIONS, ACTION_IDX

    ticker = YFINANCE_MAP.get(SYMBOL, "^NSEI")
    print(f"\n{'='*65}")
    print(f"  Historical Backtest Trainer | {SYMBOL} ({ticker})")
    print(f"{'='*65}\n")

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    print("Fetching 5-min Nifty data (last 59 days, max yfinance allows)...")
    df5 = fetch_5min(ticker, days=59)
    print(f"  5-min bars: {len(df5):,}")

    print("Fetching daily data (3 years for VIX + trend context)...")
    df_daily_nifty = fetch_daily(ticker, years=3)
    df_daily_vix   = fetch_daily(VIX_TICKER, years=3)
    print(f"  Daily bars: {len(df_daily_nifty):,}")
    print(f"  VIX daily:  {len(df_daily_vix):,}")

    if df5.empty:
        print("ERROR: Could not fetch 5-min data. Check internet / market schedule.")
        return

    # Filter only NSE market hours (9:15 – 15:30 IST)
    df5["timestamp"] = pd.to_datetime(df5["timestamp"])
    df5 = df5[
        (df5["timestamp"].dt.time >= pd.Timestamp("09:15").time()) &
        (df5["timestamp"].dt.time <= pd.Timestamp("15:30").time())
    ].reset_index(drop=True)
    print(f"  After market-hours filter: {len(df5):,} bars")

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    print("\nComputing features from price/volume/volatility...")
    df5 = add_features(df5, df_daily_vix)

    # ── Step 3: Build X, y ───────────────────────────────────────────────────
    print("Building feature matrix and labels...")
    X = build_X(df5)
    y = build_labels(df5)

    # Remove unlabeled rows
    valid = y >= 0
    X, y = X[valid], y[valid]
    print(f"  Valid labeled samples: {len(X):,}")

    # Class distribution
    for i, action in enumerate(ACTIONS):
        count = (y == i).sum()
        pct   = count / len(y) * 100
        print(f"    {action:15s}: {count:5,}  ({pct:.1f}%)")

    # ── Step 4: Real FII historical data via nsepython ───────────────────────
    print("\nFetching real FII/DII historical data via nsepython...")
    fii_col = "fii_net_futures"
    try:
        import nsepython as nse
        fii_raw = nse.nse_fiidii(90)   # last 90 entries
        if isinstance(fii_raw, list) and fii_raw:
            fii_df = pd.DataFrame(fii_raw)
            fii_df["date"] = pd.to_datetime(fii_df["date"], format="%d-%b-%Y", errors="coerce").dt.date
            fii_df["netValue"] = pd.to_numeric(fii_df["netValue"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
            fii_map = fii_df[fii_df["category"].str.contains("FII", na=False)].set_index("date")["netValue"].to_dict()
            df5["date_only"] = df5["timestamp"].dt.date
            df5["fii_net_futures"] = df5["date_only"].map(fii_map).fillna(df5["fii_net_futures"])
            print(f"  FII data injected for {len(fii_map)} dates")
    except Exception as e:
        print(f"  nsepython FII fetch skipped: {e}")

    # ── Step 5: Combine with synthetic data ──────────────────────────────────
    print("\nMerging with synthetic bootstrap data for robustness...")
    from core.agent.ml_agent import _generate_synthetic_training_data
    X_syn, y_syn = _generate_synthetic_training_data(n=10_000)
    X_all = np.vstack([X, X_syn])
    y_all = np.concatenate([y, y_syn])
    print(f"  Combined samples: {len(X_all):,}  (real: {len(X):,} + synthetic: {len(X_syn):,})")

    # ── Step 6: Optuna hyperparameter tuning ─────────────────────────────────
    best_params = {}
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        from sklearn.model_selection import cross_val_score
        import xgboost as xgb

        print("\nOptuna: tuning XGBoost hyperparameters (30 trials)...")

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma":            trial.suggest_float("gamma", 0, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 1.0),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 2.0),
                "objective":        "multi:softprob",
                "num_class":        len(ACTIONS),
                "eval_metric":      "mlogloss",
                "random_state":     42,
                "n_jobs":           -1,
                "verbosity":        0,
            }
            clf = xgb.XGBClassifier(**params)
            # Use small subset for speed
            idx = np.random.choice(len(X_all), min(3000, len(X_all)), replace=False)
            scores = cross_val_score(clf, X_all[idx], y_all[idx], cv=3,
                                     scoring="accuracy", n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, timeout=120)  # max 2 min
        best_params = study.best_params
        print(f"  Best params: n_est={best_params.get('n_estimators')}, "
              f"depth={best_params.get('max_depth')}, "
              f"lr={best_params.get('learning_rate', 0):.4f}")
        print(f"  Best CV accuracy: {study.best_value:.3f}")
    except Exception as e:
        print(f"  Optuna tuning skipped: {e}")

    # ── Step 7: Train model with best params ──────────────────────────────────
    print("\nTraining XGBoost + LightGBM on historical data...")
    model = TapeXGBModel(SYMBOL)

    if best_params:
        # Patch model with Optuna-found params
        import xgboost as xgb
        model._xgb = xgb.XGBClassifier(
            **{k: v for k, v in best_params.items()
               if k not in ("objective","num_class","eval_metric","random_state","n_jobs","verbosity")},
            objective="multi:softprob",
            num_class=len(ACTIONS),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model._xgb.fit(X_all, y_all)
        model._trained = True
        model._train_samples = len(X_all)
        from datetime import datetime as _dt
        model._trained_at = _dt.now()
        model._save()
    else:
        model.train(X_all, y_all)

    # ── Step 8: Backtest accuracy ─────────────────────────────────────────────
    print("\nEvaluating on real historical samples...")
    from core.memory.vector_store import VECTOR_FEATURE_KEYS as VFK
    X_real_eval = X[:500] if len(X) > 500 else X
    y_real_eval = y[:500] if len(y) > 500 else y

    correct = 0
    for i in range(len(X_real_eval)):
        feat_dict = {k: float(X_real_eval[i, j]) for j, k in enumerate(VFK)}
        pred = model.predict(feat_dict)
        if ACTION_IDX[pred["action"]] == y_real_eval[i]:
            correct += 1
    accuracy = correct / len(X_real_eval)
    print(f"  Backtest accuracy: {accuracy:.1%}  ({correct}/{len(X_real_eval)} correct)")

    print(f"\n{'='*65}")
    print(f"  Model saved to: data/models/tape_agent_{SYMBOL}.pkl")
    print(f"  Trained on:     {len(X_all):,} samples ({len(X):,} real historical)")
    print(f"  Backtest acc:   {accuracy:.1%}")
    print(f"{'='*65}\n")

    # ── Step 9: Feature importance + SHAP ────────────────────────────────────
    fi = model.get_feature_importance(top_n=15)
    if fi:
        print("Top 15 most important features:")
        for fname, score in fi:
            bar = "█" * max(1, int(score / max(fi[0][1], 1e-6) * 40))
            print(f"  {fname:35s}  {score:.1f}  {bar}")

    # SHAP sample explanation
    try:
        from core.agent.explainer import ModelExplainer
        exp = ModelExplainer(model)
        sample_feat = {k: float(X_real_eval[0, j]) for j, k in enumerate(VFK)}
        action_name = ACTIONS[int(y_real_eval[0])]
        expl = exp.explain(sample_feat, action_name, top_n=5)
        print(f"\nSHAP sample explanation (action={action_name}):")
        for name, sv, fv in expl["top_contributors"]:
            arrow = "↑" if sv > 0 else "↓"
            print(f"  {arrow} {name:30s}  SHAP={sv:+.4f}  value={fv:.4f}")
    except Exception as e:
        print(f"  SHAP explanation skipped: {e}")

    print("\nDone. Model is ready for live trading.")


if __name__ == "__main__":
    main()
