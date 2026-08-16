"""
NSE Data Scraper — Updated for NSE's v3 API (2025+)
=====================================================
NSE changed their option chain API to:
  - option-chain-contract-info  (get expiries/strikes)
  - option-chain-v3             (get option chain data)

Session flow (must follow exactly):
  1. GET https://www.nseindia.com/option-chain  → sets cookies
  2. GET /api/option-chain-contract-info?symbol=NIFTY  → get expiry dates
  3. GET /api/option-chain-v3?type=Indices&symbol=NIFTY&expiry=DD-Mon-YYYY
"""
import time
import random
from datetime import datetime
from typing import Optional, Dict, List

import requests
from loguru import logger


NSE_BASE = "https://www.nseindia.com"
NSE_API  = "https://www.nseindia.com/api"

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "accept-language": "en,gu;q=0.9,hi;q=0.8",
    "accept-encoding": "gzip, deflate",
}

INDEX_TYPE = {
    "NIFTY":      "Indices",
    "BANKNIFTY":  "Indices",
    "FINNIFTY":   "Indices",
    "MIDCPNIFTY": "Indices",
    "SENSEX":     "Indices",
}


class NSEScraper:
    """
    Robust NSE scraper using the v3 option-chain API.
    Handles cookie refresh automatically every 25 minutes.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._cookies: Dict = {}
        self._last_refresh: float = 0
        self._cookie_ttl: int = 1500          # refresh every 25 min
        self._refresh_session()

    # ── Session / Cookies ─────────────────────────────────────────────────────

    def _refresh_session(self):
        """Visit NSE option-chain page to get fresh cookies."""
        try:
            r = self.session.get(f"{NSE_BASE}/option-chain", timeout=12)
            self._cookies = dict(r.cookies)
            self._last_refresh = time.time()
            logger.debug(f"NSE session refreshed | cookies: {list(self._cookies.keys())}")
        except Exception as e:
            logger.warning(f"NSE session refresh failed: {e}")

    def _ensure_session(self):
        if time.time() - self._last_refresh > self._cookie_ttl:
            self._refresh_session()

    def _get(self, path: str, params: dict = None, retries: int = 2) -> Optional[dict]:
        """GET NSE API endpoint with retry."""
        self._ensure_session()
        url = f"{NSE_API}/{path}"
        for attempt in range(retries):
            try:
                r = self.session.get(
                    url, params=params,
                    cookies=self._cookies,
                    timeout=12,
                )
                if r.status_code in (401, 403):
                    logger.debug("NSE session expired — refreshing")
                    self._refresh_session()
                    time.sleep(1)
                    continue
                if r.status_code == 404:
                    logger.debug(f"NSE 404: {url}")
                    return None
                r.raise_for_status()
                return r.json()
            except requests.exceptions.JSONDecodeError:
                logger.debug(f"Bad JSON from {url}")
                return None
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"NSE connection error: {e}")
                return None
            except Exception as e:
                wait = (2 ** attempt) + random.uniform(0.3, 0.8)
                logger.warning(f"NSE request failed ({attempt+1}/{retries}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        return None

    # ── Option Chain ──────────────────────────────────────────────────────────

    def get_expiry_dates(self, symbol: str = "NIFTY") -> List[str]:
        """Get list of expiry dates. Format: '18-Aug-2026'"""
        data = self._get("option-chain-contract-info", params={"symbol": symbol})
        if data:
            return data.get("expiryDates", [])
        return []

    def get_option_chain_raw(self, symbol: str = "NIFTY",
                              expiry: Optional[str] = None) -> Optional[dict]:
        """
        Fetch raw option chain from /api/option-chain-v3.
        Returns the full JSON (records.data, records.underlyingValue, etc.)
        """
        if not expiry:
            expiries = self.get_expiry_dates(symbol)
            if not expiries:
                return None
            expiry = expiries[0]

        oc_type = INDEX_TYPE.get(symbol.upper(), "Indices")
        data = self._get(
            "option-chain-v3",
            params={"type": oc_type, "symbol": symbol, "expiry": expiry},
        )
        return data

    def parse_option_chain(
        self,
        symbol: str = "NIFTY",
        expiry: Optional[str] = None,
        strikes_range: int = 10,
    ) -> List[dict]:
        """
        Parse option chain → flat list of records.
        Filters to ATM ± strikes_range strikes.
        """
        raw = self.get_option_chain_raw(symbol, expiry)
        if not raw:
            logger.error(f"Failed to fetch option chain for {symbol}")
            return []

        records    = raw.get("records", {})
        all_data   = records.get("data", [])
        underlying = records.get("underlyingValue", 0)
        expiry_used = expiry or (all_data[0].get("CE", all_data[0].get("PE", {})).get("expiryDate") if all_data else "")

        atm_strike = round(underlying / 50) * 50
        ts = datetime.now()
        parsed = []

        for row in all_data:
            strike = row.get("strikePrice", 0)
            if abs(strike - atm_strike) > strikes_range * 50:
                continue

            for opt_type in ("CE", "PE"):
                d = row.get(opt_type)
                if not d:
                    continue
                parsed.append({
                    "timestamp":  ts,
                    "symbol":     symbol,
                    "expiry":     expiry_used,
                    "strike":     strike,
                    "option_type": opt_type,
                    "oi":         d.get("openInterest", 0),
                    "oi_change":  d.get("changeinOpenInterest", 0),
                    "volume":     d.get("totalTradedVolume", 0),
                    "ltp":        d.get("lastPrice", 0),
                    "iv":         d.get("impliedVolatility", 0),
                    "delta":      d.get("delta", 0),
                    "theta":      d.get("theta", 0),
                    "vega":       d.get("vega", 0),
                    "underlying": underlying,
                    "atm_strike": atm_strike,
                })

        logger.info(f"Parsed {len(parsed)} option records for {symbol} expiry={expiry_used}")
        return parsed

    def get_atm_strike(self, symbol: str = "NIFTY") -> float:
        raw = self.get_option_chain_raw(symbol)
        if raw:
            uv = raw.get("records", {}).get("underlyingValue", 0)
            return round(uv / 50) * 50
        return 0

    # ── PCR & Max Pain ────────────────────────────────────────────────────────

    def get_pcr_data(self, symbol: str = "NIFTY",
                     expiry: Optional[str] = None) -> dict:
        records = self.parse_option_chain(symbol, expiry, strikes_range=20)
        if not records:
            return {}

        ce_oi  = sum(r["oi"] for r in records if r["option_type"] == "CE")
        pe_oi  = sum(r["oi"] for r in records if r["option_type"] == "PE")
        ce_vol = sum(r["volume"] for r in records if r["option_type"] == "CE")
        pe_vol = sum(r["volume"] for r in records if r["option_type"] == "PE")

        return {
            "symbol":         symbol,
            "expiry":         expiry,
            "pcr_oi":         round(pe_oi / ce_oi, 3) if ce_oi else 0,
            "pcr_vol":        round(pe_vol / ce_vol, 3) if ce_vol else 0,
            "total_call_oi":  int(ce_oi),
            "total_put_oi":   int(pe_oi),
            "total_call_vol": int(ce_vol),
            "total_put_vol":  int(pe_vol),
            "timestamp":      datetime.now(),
        }

    def get_max_pain(self, symbol: str = "NIFTY",
                     expiry: Optional[str] = None) -> Optional[float]:
        records = self.parse_option_chain(symbol, expiry, strikes_range=30)
        if not records:
            return None

        strikes = sorted(set(r["strike"] for r in records))
        oi_map  = {(r["strike"], r["option_type"]): r["oi"] for r in records}

        pain = {}
        for s in strikes:
            total = 0
            for r_s in strikes:
                ce_oi = oi_map.get((r_s, "CE"), 0)
                pe_oi = oi_map.get((r_s, "PE"), 0)
                if s > r_s:
                    total += ce_oi * (s - r_s)
                if s < r_s:
                    total += pe_oi * (r_s - s)
            pain[s] = total

        return min(pain, key=pain.get)

    # ── Index Quotes ──────────────────────────────────────────────────────────

    def get_index_quote(self, symbol: str = "NIFTY 50") -> Optional[dict]:
        data = self._get("allIndices")
        if not data:
            return None
        for idx in data.get("data", []):
            if symbol.upper() in idx.get("index", "").upper():
                return {
                    "symbol":      idx.get("index"),
                    "ltp":         idx.get("last", 0),
                    "open":        idx.get("open", 0),
                    "high":        idx.get("high", 0),
                    "low":         idx.get("low", 0),
                    "prev_close":  idx.get("previousClose", 0),
                    "change":      idx.get("change", 0),
                    "change_pct":  idx.get("percentChange", 0),
                    "timestamp":   datetime.now(),
                }
        return None

    def get_vix(self) -> Optional[float]:
        data = self._get("allIndices")
        if not data:
            return None
        for idx in data.get("data", []):
            if "VIX" in idx.get("index", ""):
                return idx.get("last", 0)
        return None

    # ── Market Status ─────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        import pytz
        tz  = pytz.timezone("Asia/Kolkata")
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return False
        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close


# ── Singleton ─────────────────────────────────────────────────────────────────

_scraper: Optional[NSEScraper] = None


def get_scraper() -> NSEScraper:
    global _scraper
    if _scraper is None:
        _scraper = NSEScraper()
    return _scraper
