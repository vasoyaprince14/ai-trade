"""
Historical OHLCV data fetcher.
Sources (in priority order):
  1. jugaad-data (NSE official, most reliable)
  2. yfinance (fallback, good for daily/weekly)
  3. Local DB cache

jugaad-data docs: https://marketsetup.in/documentation/jugaad-data/
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
from loguru import logger

# Add vendor jugaad-data to path
VENDOR_DIR = Path(__file__).parent.parent.parent / "vendors"
sys.path.insert(0, str(VENDOR_DIR / "jugaad-data"))

# NSE index symbol mapping for yfinance
YFINANCE_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "SENSEX": "^BSESN",
}

TIMEFRAME_MAP = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


def fetch_historical(
    symbol: str,
    timeframe: str = "1d",
    days: int = 365,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol and timeframe.
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=days))

    # Try jugaad-data first for NSE data
    df = _fetch_jugaad(symbol, timeframe, start, end)
    if df is not None and not df.empty:
        return df

    # Fallback to yfinance
    df = _fetch_yfinance(symbol, timeframe, start, end)
    if df is not None and not df.empty:
        return df

    logger.error(f"Could not fetch historical data for {symbol}")
    return pd.DataFrame()


def _fetch_jugaad(symbol: str, timeframe: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Fetch from jugaad-data (NSE official source)."""
    try:
        if timeframe in ("1d", "1wk", "1mo"):
            from jugaad_data.nse import index_raw
            # jugaad-data uses index names like "NIFTY 50", "NIFTY BANK"
            jugaad_map = {
                "NIFTY": "NIFTY 50",
                "BANKNIFTY": "NIFTY BANK",
                "FINNIFTY": "NIFTY FIN SERVICE",
                "MIDCPNIFTY": "NIFTY MIDCAP SELECT",
            }
            jugaad_sym = jugaad_map.get(symbol.upper(), symbol)
            raw = index_raw(jugaad_sym, start, end)
            if raw:
                df = pd.DataFrame(raw)
                df = df.rename(columns={
                    "CHG_IN_INDEX_CLSG_VALUE": "close",
                    "OPEN_INDEX_VAL": "open",
                    "HIGH_INDEX_VAL": "high",
                    "LOW_INDEX_VAL": "low",
                    "INDEX_DATE": "timestamp",
                })
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["volume"] = 0
                df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
                return df
    except ImportError:
        logger.debug("jugaad-data not available, falling back to yfinance")
    except Exception as e:
        logger.debug(f"jugaad fetch failed: {e}")
    return None


def _fetch_yfinance(symbol: str, timeframe: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Fetch from yfinance."""
    try:
        import yfinance as yf
        ticker = YFINANCE_MAP.get(symbol.upper(), f"{symbol}.NS")
        tf = TIMEFRAME_MAP.get(timeframe, "1d")

        # yfinance limits intraday to 60 days
        if tf in ("1m", "5m", "15m", "30m", "1h"):
            start = max(start, date.today() - timedelta(days=59))

        data = yf.download(
            ticker,
            start=start,
            end=end + timedelta(days=1),
            interval=tf,
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None

        data.index = pd.to_datetime(data.index)
        # Handle MultiIndex columns from yfinance (ticker in second level)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]

        data = data.reset_index()
        # Rename whatever index column exists to 'timestamp'
        for col in ("Date", "Datetime", "date", "datetime", "index"):
            if col in data.columns:
                data = data.rename(columns={col: "timestamp"})
                break

        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in data.columns]
        data = data[keep].dropna()
        data["symbol"] = symbol
        logger.info(f"Fetched {len(data)} bars for {symbol} ({timeframe}) via yfinance")
        return data
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to OHLCV DataFrame."""
    if df.empty or len(df) < 20:
        return df

    try:
        import pandas_ta as ta
        df = df.copy()
        df.set_index("timestamp", inplace=True)

        # Trend
        df["ema9"] = ta.ema(df["close"], length=9)
        df["ema21"] = ta.ema(df["close"], length=21)
        df["ema50"] = ta.ema(df["close"], length=50)
        df["ema200"] = ta.ema(df["close"], length=200)

        # Momentum
        rsi = ta.rsi(df["close"], length=14)
        df["rsi"] = rsi

        macd = ta.macd(df["close"])
        if macd is not None:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

        # Volatility
        atr = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr"] = atr

        bb = ta.bbands(df["close"], length=20)
        if bb is not None:
            bb_cols = {c.split("_")[0]: c for c in bb.columns}
            if "BBU" in bb_cols: df["bb_upper"] = bb[bb_cols["BBU"]]
            if "BBM" in bb_cols: df["bb_mid"] = bb[bb_cols["BBM"]]
            if "BBL" in bb_cols: df["bb_lower"] = bb[bb_cols["BBL"]]

        # Volume
        df["volume_ma"] = ta.sma(df["volume"], length=20)
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1)

        # ADX
        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None:
            df["adx"] = adx["ADX_14"]

        # VWAP (intraday)
        if "volume" in df.columns and df["volume"].sum() > 0:
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        df = df.reset_index()
        return df

    except ImportError:
        logger.warning("pandas-ta not installed, skipping indicators")
        return df
    except Exception as e:
        logger.warning(f"Indicator calculation failed: {e}")
        return df


def get_nifty_data(timeframe: str = "1d", days: int = 365) -> pd.DataFrame:
    """Convenience function for Nifty data with indicators."""
    df = fetch_historical("NIFTY", timeframe, days)
    return add_indicators(df) if not df.empty else df


def get_banknifty_data(timeframe: str = "1d", days: int = 365) -> pd.DataFrame:
    """Convenience function for BankNifty data with indicators."""
    df = fetch_historical("BANKNIFTY", timeframe, days)
    return add_indicators(df) if not df.empty else df
