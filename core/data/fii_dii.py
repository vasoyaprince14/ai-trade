"""
FII/DII Institutional Data Fetcher
====================================
Fetches participant-wise derivatives positioning from NSE public APIs:
  - Participant-wise OI  (/api/historical/fnoparticipants)
  - FII Derivatives Summary (/api/fiiderivsummary)
  - FII/DII Cash Equity flow (/api/fiidiidata)

All endpoints reuse the same session/cookies as NSEScraper.

Returned structure (per participant):
  {
    date, client_type,
    fut_index_long, fut_index_short,
    fut_stock_long, fut_stock_short,
    opt_idx_call_long, opt_idx_call_short,
    opt_idx_put_long,  opt_idx_put_short,
    opt_stk_call_long, opt_stk_call_short,
    opt_stk_put_long,  opt_stk_put_short,
    total_long, total_short,
    # derived
    fut_index_net, opt_idx_call_net, opt_idx_put_net,
    fut_bias, call_bias, put_bias, composite_bias
  }
"""
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.data.nse_scraper import get_scraper


# NSE may use different field name styles across API versions.
# We normalise whatever we get into snake_case keys.
_FIELD_ALIASES = {
    # camelCase → snake_case
    "futIndexLong":       "fut_index_long",
    "futIndexShort":      "fut_index_short",
    "futStockLong":       "fut_stock_long",
    "futStockShort":      "fut_stock_short",
    "optIdxCallLong":     "opt_idx_call_long",
    "optIdxCallShort":    "opt_idx_call_short",
    "optIdxPutLong":      "opt_idx_put_long",
    "optIdxPutShort":     "opt_idx_put_short",
    "optStockCallLong":   "opt_stk_call_long",
    "optStockCallShort":  "opt_stk_call_short",
    "optStockPutLong":    "opt_stk_put_long",
    "optStockPutShort":   "opt_stk_put_short",
    "totalLongContracts": "total_long",
    "totalShortContracts":"total_short",
    # alternate camelCase seen in some versions
    "future_index_long":  "fut_index_long",
    "future_index_short": "fut_index_short",
    "future_stock_long":  "fut_stock_long",
    "future_stock_short": "fut_stock_short",
    "option_index_call_long":  "opt_idx_call_long",
    "option_index_call_short": "opt_idx_call_short",
    "option_index_put_long":   "opt_idx_put_long",
    "option_index_put_short":  "opt_idx_put_short",
    "option_stock_call_long":  "opt_stk_call_long",
    "option_stock_call_short": "opt_stk_call_short",
    "option_stock_put_long":   "opt_stk_put_long",
    "option_stock_put_short":  "opt_stk_put_short",
    "clientType":         "client_type",
    "client_type":        "client_type",
}

_NUMERIC_FIELDS = [
    "fut_index_long", "fut_index_short",
    "fut_stock_long", "fut_stock_short",
    "opt_idx_call_long", "opt_idx_call_short",
    "opt_idx_put_long",  "opt_idx_put_short",
    "opt_stk_call_long", "opt_stk_call_short",
    "opt_stk_put_long",  "opt_stk_put_short",
    "total_long", "total_short",
]


def _normalise_row(raw: dict) -> dict:
    """Rename keys to snake_case and cast numerics."""
    out = {}
    for k, v in raw.items():
        key = _FIELD_ALIASES.get(k, k)
        out[key] = v
    for f in _NUMERIC_FIELDS:
        out[f] = _to_int(out.get(f, 0))
    return out


def _to_int(v) -> int:
    try:
        return int(str(v).replace(",", "").strip() or 0)
    except (ValueError, TypeError):
        return 0


def _net_bias(long: int, short: int) -> float:
    """Returns -1..+1 directional bias from long/short counts."""
    total = long + short
    if total == 0:
        return 0.0
    return (long - short) / total


class FIIDIIFetcher:
    """
    Fetches and parses NSE participant-wise positioning data.
    Reuses NSEScraper's authenticated session for cookies.
    """

    CLIENT_TYPES = {"FII", "DII", "PRO", "CLI"}

    def __init__(self):
        self._scraper = get_scraper()

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        return self._scraper._get(path, params=params)

    # ── Participant-wise OI + Volume ───────────────────────────────────────────

    def get_participant_data(self, from_date: Optional[date] = None,
                             to_date: Optional[date] = None) -> List[dict]:
        """
        Fetch participant-wise F&O OI + volume.
        Endpoint: /api/historical/fnoparticipants

        Returns list of normalised dicts, one per (date × client_type).
        Falls back to empty list on failure.
        """
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date

        params = {
            "from": from_date.strftime("%d-%m-%Y"),
            "to":   to_date.strftime("%d-%m-%Y"),
        }
        raw = self._get("historical/fnoparticipants", params=params)
        if not raw:
            logger.warning("FII participant data: no response from NSE")
            return []

        # Response may be list directly or nested under a key
        rows = raw if isinstance(raw, list) else raw.get("data", [])
        result = []
        for r in rows:
            norm = _normalise_row(r)
            norm["date"] = norm.get("date") or from_date.isoformat()
            self._add_derived(norm)
            result.append(norm)

        logger.info(f"Participant data: {len(result)} rows for {from_date}")
        return result

    # ── FII Derivatives Summary ────────────────────────────────────────────────

    def get_fii_derivatives_summary(self) -> Optional[dict]:
        """
        Fetch FII derivatives statistics summary.
        Endpoint: /api/fiiderivsummary
        """
        raw = self._get("fiiderivsummary")
        if not raw:
            logger.warning("FII derivatives summary: no response")
            return None
        # The response structure varies; return as-is with light cleanup
        return raw

    # ── FII/DII Cash Equity Flow ───────────────────────────────────────────────

    def get_cash_flow(self, from_date: Optional[date] = None,
                      to_date: Optional[date] = None) -> List[dict]:
        """
        Fetch FII/DII cash equity buy/sell data.
        Endpoint: /api/fiidiidata or /api/historical/fiidiidata
        """
        if not from_date:
            from_date = date.today()
        if not to_date:
            to_date = from_date

        params = {
            "from": from_date.strftime("%d-%m-%Y"),
            "to":   to_date.strftime("%d-%m-%Y"),
        }
        # Try both known endpoint patterns
        for path in ("fiidiidata", "historical/fiidiidata"):
            raw = self._get(path, params=params)
            if raw:
                rows = raw if isinstance(raw, list) else raw.get("data", [])
                result = []
                for r in rows:
                    result.append({
                        "date":          r.get("date") or from_date.isoformat(),
                        "fii_buy":       _to_int(r.get("fiiBuy") or r.get("fii_buy", 0)),
                        "fii_sell":      _to_int(r.get("fiiSell") or r.get("fii_sell", 0)),
                        "fii_net":       _to_int(r.get("fiiNet") or r.get("fii_net", 0)),
                        "dii_buy":       _to_int(r.get("diiBuy") or r.get("dii_buy", 0)),
                        "dii_sell":      _to_int(r.get("diiSell") or r.get("dii_sell", 0)),
                        "dii_net":       _to_int(r.get("diiNet") or r.get("dii_net", 0)),
                    })
                logger.info(f"Cash flow: {len(result)} rows via /{path}")
                return result

        logger.warning("FII/DII cash flow: no response from either endpoint")
        return []

    # ── High-level: today's institutional state ────────────────────────────────

    def get_today_data(self) -> Dict:
        """
        Aggregate all institutional data for today into one structured dict.

        Returns:
          {
            "date": ...,
            "participants": {"FII": {...}, "DII": {...}, "PRO": {...}, "CLI": {...}},
            "cash": {"fii_net": ..., "dii_net": ...},
            "fii_regime": "BEARISH" | "BULLISH" | ...,
            "fii_composite": float,
            "fii_futures_bias": float,
            "fii_call_bias": float,
            "fii_put_bias": float,
          }
        """
        today = date.today()

        # Try today; if weekend/holiday, try last 3 days
        rows = []
        for delta in range(4):
            d = today - timedelta(days=delta)
            if d.weekday() >= 5:
                continue
            rows = self.get_participant_data(d, d)
            if rows:
                break

        participants = {}
        for row in rows:
            ct = str(row.get("client_type", "")).upper().strip()
            if ct in self.CLIENT_TYPES:
                participants[ct] = row

        # Cash flow
        cash_rows = []
        for delta in range(4):
            d = today - timedelta(days=delta)
            if d.weekday() >= 5:
                continue
            cash_rows = self.get_cash_flow(d, d)
            if cash_rows:
                break

        cash = cash_rows[0] if cash_rows else {}

        # FII-specific biases
        fii = participants.get("FII", {})
        fii_fut_bias  = _net_bias(fii.get("fut_index_long", 0),  fii.get("fut_index_short", 0))
        # For calls: institutions selling calls = bearish → short > long is bearish
        fii_call_bias = -_net_bias(fii.get("opt_idx_call_long", 0), fii.get("opt_idx_call_short", 0))
        # For puts: institutions buying puts = bearish → long > short is bearish
        fii_put_bias  = -_net_bias(fii.get("opt_idx_put_long", 0),  fii.get("opt_idx_put_short", 0))

        # Composite: futures 40%, calls 30%, puts 30%
        composite = (
            0.40 * fii_fut_bias +
            0.30 * fii_call_bias +
            0.30 * fii_put_bias
        )

        regime = _classify_regime(composite)

        logger.info(
            f"FII regime={regime} composite={composite:+.3f} "
            f"fut={fii_fut_bias:+.3f} call={fii_call_bias:+.3f} put={fii_put_bias:+.3f}"
        )

        return {
            "date":              today.isoformat(),
            "participants":      participants,
            "cash":              cash,
            "fii_regime":        regime,
            "fii_composite":     round(composite, 4),
            "fii_futures_bias":  round(fii_fut_bias, 4),
            "fii_call_bias":     round(fii_call_bias, 4),
            "fii_put_bias":      round(fii_put_bias, 4),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _add_derived(row: dict):
        """Add net position and bias fields to a normalised row."""
        row["fut_index_net"]   = row["fut_index_long"]   - row["fut_index_short"]
        row["opt_idx_call_net"] = row["opt_idx_call_long"] - row["opt_idx_call_short"]
        row["opt_idx_put_net"]  = row["opt_idx_put_long"]  - row["opt_idx_put_short"]

        row["fut_bias"]  = _net_bias(row["fut_index_long"],   row["fut_index_short"])
        # call bias: positive = net long calls = bullish
        row["call_bias"] = _net_bias(row["opt_idx_call_long"], row["opt_idx_call_short"])
        # put bias: positive = net long puts = bearish (inverted for composite)
        row["put_bias"]  = _net_bias(row["opt_idx_put_long"],  row["opt_idx_put_short"])


def _classify_regime(composite: float) -> str:
    if composite >  0.60: return "AGGRESSIVE_BULLISH"
    if composite >  0.30: return "BULLISH"
    if composite >  0.10: return "MILD_BULLISH"
    if composite > -0.10: return "NEUTRAL"
    if composite > -0.30: return "MILD_BEARISH"
    if composite > -0.60: return "BEARISH"
    return "AGGRESSIVE_BEARISH"


# ── Singleton ──────────────────────────────────────────────────────────────────

_fetcher: Optional[FIIDIIFetcher] = None


def get_fii_fetcher() -> FIIDIIFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = FIIDIIFetcher()
    return _fetcher
