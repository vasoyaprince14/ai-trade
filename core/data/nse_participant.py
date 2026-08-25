"""
NSE Participant Data — FII / DII / Pro / Client Positioning
============================================================
Fetches who is doing what in F&O markets.

Data Sources:
  1. NSE Archives — Participant-wise OI CSV (EOD)
     https://archives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv
     Columns: Client Type | Future Index Long/Short | Option Index Call/Put Long/Short

  2. NSE API — FII/DII Cash Market Activity (intraday)
     /api/fiidiiTradeReact

  3. NSE API — Market Activity (index futures volume/OI)
     /api/market-data-pre-open?key=FO

Participant types:
  FII   — Foreign Institutional Investors (biggest movers, track these)
  DII   — Domestic Institutional Investors (mutual funds, insurance)
  Pro   — Proprietary / HFT desks
  Client— Retail traders
"""
import io
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import requests
import pandas as pd
from loguru import logger

from core.data.nse_scraper import get_scraper


NSE_ARCHIVE = "https://archives.nseindia.com"

PARTICIPANT_TYPES = ["FII", "DII", "PRO", "CLIENT"]

# Bias thresholds (in number of contracts)
STRONG_BIAS_THRESHOLD = 50_000
MILD_BIAS_THRESHOLD   = 10_000


class NSEParticipantData:
    """
    Fetches and interprets NSE participant-wise F&O OI data.

    Key insight: FII net position tells you big money direction.
    If FII is net long futures + net long calls → strongly bullish.
    If FII is net short futures + net long puts → strongly bearish.
    """

    def __init__(self):
        self._scraper = get_scraper()
        self._cache: Dict = {}
        self._cache_date: Optional[date] = None

    # ── Archive Download ───────────────────────────────────────────────────────

    def _get_archive_csv(self, url: str) -> Optional[pd.DataFrame]:
        """Download a CSV from NSE archives (requires referer header)."""
        try:
            r = requests.get(
                url,
                headers={
                    "user-agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0"
                    ),
                    "referer": "https://www.nseindia.com",
                    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=15,
            )
            if r.status_code == 200 and r.text.strip():
                # NSE participant OI CSV has a title in row 0; actual header is row 1
                try:
                    df = pd.read_csv(io.StringIO(r.text), header=1)
                except Exception:
                    df = pd.read_csv(io.StringIO(r.text))
                return df
            logger.warning(f"NSE archive {r.status_code}: {url}")
        except Exception as e:
            logger.warning(f"NSE archive fetch error: {e}")
        return None

    def _most_recent_trading_day(self, ref: Optional[date] = None) -> date:
        """Return most recent weekday (skip weekends). NSE data uploads after 6 PM."""
        d = ref or date.today()
        # If before 18:00 today, use yesterday's data
        if ref is None and datetime.now().hour < 18:
            d -= timedelta(days=1)
        while d.weekday() >= 5:  # 5=Sat, 6=Sun
            d -= timedelta(days=1)
        return d

    # ── Participant OI (F&O) ───────────────────────────────────────────────────

    def get_participant_oi(self, for_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Download participant-wise F&O OI from NSE archives.

        Returns DataFrame indexed by client_type with columns:
          fut_idx_long, fut_idx_short, net_futures,
          call_long, call_short, net_calls,
          put_long, put_short, net_puts,
          overall_bias  (positive = bullish, negative = bearish)
        """
        trading_day = self._most_recent_trading_day(for_date)

        # Cache hit
        if self._cache_date == trading_day and "participant_oi" in self._cache:
            return self._cache["participant_oi"]

        date_str = trading_day.strftime("%d%m%Y")
        url = f"{NSE_ARCHIVE}/content/nsccl/fao_participant_oi_{date_str}.csv"
        df = self._get_archive_csv(url)

        # Try one day earlier if today's not uploaded yet
        if df is None:
            prev = self._most_recent_trading_day(trading_day - timedelta(days=1))
            date_str2 = prev.strftime("%d%m%Y")
            url2 = f"{NSE_ARCHIVE}/content/nsccl/fao_participant_oi_{date_str2}.csv"
            df = self._get_archive_csv(url2)
            if df is not None:
                logger.info(f"Participant OI: using {prev} (latest available)")

        if df is None:
            logger.warning("Could not fetch participant OI from NSE archives")
            return None

        parsed = self._parse_participant_df(df)
        if parsed is not None:
            self._cache["participant_oi"] = parsed
            self._cache_date = trading_day

        return parsed

    def _parse_participant_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize and enrich the raw participant OI CSV."""
        try:
            # Normalize column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Try to find the client_type column
            type_col = next(
                (c for c in df.columns if "client" in c or "type" in c), None
            )
            if type_col is None:
                logger.warning("Participant OI: can't find client_type column")
                return None

            df = df.rename(columns={type_col: "client_type"})
            df["client_type"] = df["client_type"].astype(str).str.strip().str.upper()
            df = df[df["client_type"].isin(PARTICIPANT_TYPES)].copy()

            if df.empty:
                return None

            # Column mapping — normalize stripped column names
            _col_candidates = {
                "fut_idx_long":  ["future_index_long",  "fut_idx_long",  "fi_long", "future_index_long_"],
                "fut_idx_short": ["future_index_short", "fut_idx_short", "fi_short","future_index_short_"],
                "call_long":     ["option_index_call_long",  "call_long",  "oi_call_long"],
                "call_short":    ["option_index_call_short", "call_short", "oi_call_short"],
                "put_long":      ["option_index_put_long",   "put_long",   "oi_put_long"],
                "put_short":     ["option_index_put_short",  "put_short",  "oi_put_short"],
            }
            rename_map = {}
            for std, candidates in _col_candidates.items():
                for cand in candidates:
                    if cand in df.columns:
                        rename_map[cand] = std
                        break
            df = df.rename(columns=rename_map)

            # Convert numeric
            numeric_cols = ["fut_idx_long", "fut_idx_short", "call_long", "call_short", "put_long", "put_short"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = (
                        pd.to_numeric(
                            df[col].astype(str).str.replace(",", "").str.strip(),
                            errors="coerce",
                        ).fillna(0).astype(int)
                    )
                else:
                    df[col] = 0

            # Derived columns
            df["net_futures"] = df["fut_idx_long"] - df["fut_idx_short"]
            df["net_calls"]   = df["call_long"]    - df["call_short"]
            df["net_puts"]    = df["put_long"]      - df["put_short"]
            # Positive = bullish (long fut + long call + short put)
            df["overall_bias"] = df["net_futures"] + df["net_calls"] - df["net_puts"]

            df = df.set_index("client_type")[
                [
                    "fut_idx_long", "fut_idx_short", "net_futures",
                    "call_long", "call_short", "net_calls",
                    "put_long", "put_short", "net_puts",
                    "overall_bias",
                ]
            ]
            return df

        except Exception as e:
            logger.error(f"Participant OI parse error: {e}")
            return None

    def get_participant_summary(self) -> Dict:
        """
        Returns human-readable dict:
        {
          "FII": {"net_futures": +120000, "net_calls": +50000, "net_puts": -30000,
                  "bias": "STRONGLY_BULLISH", "bias_score": 200000},
          "DII": {...},
          ...
          "combined_bias": "BULLISH",
          "data_date": "2026-08-22",
        }
        """
        df = self.get_participant_oi()
        summary: Dict = {"combined_bias": "NEUTRAL", "participants": {}}

        if df is not None:
            for ptype in PARTICIPANT_TYPES:
                if ptype not in df.index:
                    continue
                row = df.loc[ptype]
                bias_val = int(row["overall_bias"])
                bias = self._bias_label(bias_val)
                summary["participants"][ptype] = {
                    "net_futures": int(row["net_futures"]),
                    "net_calls":   int(row["net_calls"]),
                    "net_puts":    int(row["net_puts"]),
                    "bias_score":  bias_val,
                    "bias":        bias,
                    "fut_long":    int(row["fut_idx_long"]),
                    "fut_short":   int(row["fut_idx_short"]),
                    "call_long":   int(row["call_long"]),
                    "call_short":  int(row["call_short"]),
                    "put_long":    int(row["put_long"]),
                    "put_short":   int(row["put_short"]),
                }

            # Combined FII + DII (ignore retail client noise)
            fii_score = summary["participants"].get("FII", {}).get("bias_score", 0)
            dii_score = summary["participants"].get("DII", {}).get("bias_score", 0)
            combined  = fii_score + dii_score
            summary["combined_bias"] = self._bias_label(combined)
            summary["fii_bias"]  = summary["participants"].get("FII", {}).get("bias", "NEUTRAL")
            summary["dii_bias"]  = summary["participants"].get("DII", {}).get("bias", "NEUTRAL")
            summary["data_date"] = str(self._cache_date or "unknown")

        return summary

    def _bias_label(self, score: int) -> str:
        if score > STRONG_BIAS_THRESHOLD:
            return "STRONGLY_BULLISH"
        elif score > MILD_BIAS_THRESHOLD:
            return "BULLISH"
        elif score < -STRONG_BIAS_THRESHOLD:
            return "STRONGLY_BEARISH"
        elif score < -MILD_BIAS_THRESHOLD:
            return "BEARISH"
        return "NEUTRAL"

    # ── FII/DII Cash Market (Intraday) ─────────────────────────────────────────

    def get_fii_dii_cash(self) -> Optional[Dict]:
        """
        Live FII/DII buy/sell in cash market (updates throughout the day).
        Returns: {"FII": {"buy_cr": X, "sell_cr": X, "net_cr": X}, "DII": {...}}
        """
        data = self._scraper._get("fiidiiTradeReact")
        if not data:
            return None

        # NSE returns either a bare list or {"data": [...]}
        items = data if isinstance(data, list) else data.get("data", [])
        result = {}
        for item in items:
            cat = str(item.get("category", "")).strip().upper()
            if cat in ("FII", "DII"):
                def to_float(v):
                    return float(str(v).replace(",", "").strip() or "0")

                result[cat] = {
                    "buy_cr":  to_float(item.get("buyValue",  0)),
                    "sell_cr": to_float(item.get("sellValue", 0)),
                    "net_cr":  to_float(item.get("netValue",  0)),
                    "date":    item.get("date", ""),
                }

        if not result:
            return None

        # Combined signal
        fii_net = result.get("FII", {}).get("net_cr", 0)
        dii_net = result.get("DII", {}).get("net_cr", 0)
        combined = fii_net + dii_net
        result["combined_net_cr"] = combined
        result["cash_bias"] = (
            "BULLISH" if combined > 500
            else "BEARISH" if combined < -500
            else "NEUTRAL"
        )
        return result

    # ── All-in-one ─────────────────────────────────────────────────────────────

    def get_full_picture(self) -> Dict:
        """
        Single call that returns everything about FII/DII positioning:
        - F&O participant OI (who holds what in futures + options)
        - Cash market flows (today's buy/sell)
        - Derived signals

        Use this to feed into TapeReader or model feature extraction.
        """
        fno = self.get_participant_summary()
        cash = self.get_fii_dii_cash()

        # Infer whether smart money is net long or short
        fii_fno_bias  = fno.get("fii_bias",  "NEUTRAL")
        cash_bias     = (cash or {}).get("cash_bias", "NEUTRAL")

        # If both F&O and cash agree → strong signal
        bias_agree = fii_fno_bias == cash_bias
        if bias_agree and "BULLISH" in fii_fno_bias:
            smart_money = "STRONGLY_BULLISH"
        elif bias_agree and "BEARISH" in fii_fno_bias:
            smart_money = "STRONGLY_BEARISH"
        elif "BULLISH" in fii_fno_bias or "BULLISH" in cash_bias:
            smart_money = "BULLISH"
        elif "BEARISH" in fii_fno_bias or "BEARISH" in cash_bias:
            smart_money = "BEARISH"
        else:
            smart_money = "NEUTRAL"

        return {
            "smart_money_bias": smart_money,
            "fno": fno,
            "cash": cash,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton
_participant: Optional[NSEParticipantData] = None


def get_participant_data() -> NSEParticipantData:
    global _participant
    if _participant is None:
        _participant = NSEParticipantData()
    return _participant
