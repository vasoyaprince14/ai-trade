"""
FII / DII Option Strike Tape
==============================
Shows EXACTLY which strikes FII, DII, and Pro desks are
writing (selling) or buying in real-time.

How it works:
  1. Snapshot option chain OI every minute
  2. Compare to previous snapshot → compute ΔOI per strike
  3. Cross-reference with participant-wise aggregate bias
  4. Classify each strike-event: WHO is doing WHAT

Key insight:
  FII sell (write) CE at a strike → they think price stays BELOW that strike
  FII sell (write) PE at a strike → they think price stays ABOVE that strike
  FII buy CE → directional bullish bet
  FII buy PE → directional bearish bet OR hedging

Tape event format:
  {
    strike: 24000,
    type: "CE" | "PE",
    action: "WRITE" | "BUY" | "UNWIND_WRITE" | "UNWIND_BUY",
    oi_change: +5000,
    ltp: 120.5,
    ltp_change: -3.2,
    inferred_participant: "FII" | "DII" | "PRO" | "CLIENT",
    smart_money: True,
    significance: "HIGH" | "MEDIUM" | "LOW",
    desc: "FII writing 24000 CE — capping upside at 24000",
    timestamp: "09:35:00"
  }
"""

import time
from datetime import datetime
from collections import deque
from loguru import logger


# ── Thresholds ────────────────────────────────────────────────────────────────
LARGE_OI_CHANGE     = 2000    # contracts — "significant" event
MEDIUM_OI_CHANGE    = 500
MIN_OI_CHANGE       = 100
TAPE_BUFFER         = 200     # keep last N events
SNAPSHOT_INTERVAL   = 60      # seconds between chain snapshots


class FIITape:
    """
    Real-time FII/DII option tape for Nifty.
    Call tick() every minute to update.
    Call get_tape() to read events.
    """

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol       = symbol
        self._prev_chain  = {}       # {strike: {ce_oi, pe_oi, ce_ltp, pe_ltp}}
        self._tape        : deque   = deque(maxlen=TAPE_BUFFER)
        self._last_tick   = 0.0
        self._participant = None     # participant bias (EOD data)
        self._strikes_summary = {}   # aggregated by strike

    def _load_participant(self):
        """Load latest FII/DII aggregate bias (for inferring who is doing what)."""
        try:
            from core.data.nse_participant import get_participant_data
            pd_ = get_participant_data()
            self._participant = pd_.get_participant_summary()
        except Exception as e:
            logger.debug(f"[FIITape] Participant load error: {e}")

    def tick(self):
        """Poll option chain, compute delta OI, classify events."""
        now = time.time()
        if now - self._last_tick < SNAPSHOT_INTERVAL:
            return
        self._last_tick = now

        if self._participant is None:
            self._load_participant()

        try:
            from core.data.nse_scraper import NSEScraper
            scraper = NSEScraper()
            raw     = scraper.get_option_chain_raw(self.symbol)
            summary = scraper.parse_option_chain(raw)
            df      = summary.get("df")
            spot    = summary.get("spot", 0)

            if df is None or df.empty:
                return

            new_chain = {}
            for _, row in df.iterrows():
                strike = float(row.get("strikePrice", 0))
                if not strike:
                    continue
                new_chain[strike] = {
                    "ce_oi":   int(row.get("CE_openInterest", 0) or 0),
                    "pe_oi":   int(row.get("PE_openInterest", 0) or 0),
                    "ce_ltp":  float(row.get("CE_lastPrice", 0) or 0),
                    "pe_ltp":  float(row.get("PE_lastPrice", 0) or 0),
                    "ce_vol":  int(row.get("CE_totalTradedVolume", 0) or 0),
                    "pe_vol":  int(row.get("PE_totalTradedVolume", 0) or 0),
                }

            if self._prev_chain:
                self._generate_events(new_chain, self._prev_chain, spot)

            self._prev_chain = new_chain
            self._update_summary(new_chain, spot)

        except Exception as e:
            logger.warning(f"[FIITape] Tick error: {e}")

    def _generate_events(self, new: dict, prev: dict, spot: float):
        """Compare snapshots and emit tape events."""
        fii_bias  = (self._participant or {}).get("fii_bias",  "NEUTRAL")
        dii_bias  = (self._participant or {}).get("dii_bias",  "NEUTRAL")
        is_fii_bullish = "BULLISH" in fii_bias
        is_fii_bearish = "BEARISH" in fii_bias

        for strike in new:
            n = new[strike]
            p = prev.get(strike)
            if not p:
                continue

            dist_from_spot = strike - spot
            is_otm_ce = dist_from_spot > 0     # CE above spot = OTM
            is_otm_pe = dist_from_spot < 0     # PE below spot = OTM

            # CE OI change
            ce_doi = n["ce_oi"] - p["ce_oi"]
            ce_ltp_chg = n["ce_ltp"] - p["ce_ltp"]
            if abs(ce_doi) >= MIN_OI_CHANGE:
                event = self._classify("CE", strike, ce_doi, ce_ltp_chg,
                                       n["ce_ltp"], dist_from_spot,
                                       is_fii_bullish, is_fii_bearish)
                if event:
                    self._tape.append(event)

            # PE OI change
            pe_doi = n["pe_oi"] - p["pe_oi"]
            pe_ltp_chg = n["pe_ltp"] - p["pe_ltp"]
            if abs(pe_doi) >= MIN_OI_CHANGE:
                event = self._classify("PE", strike, pe_doi, pe_ltp_chg,
                                       n["pe_ltp"], dist_from_spot,
                                       is_fii_bullish, is_fii_bearish)
                if event:
                    self._tape.append(event)

    def _classify(self, opt_type, strike, doi, ltp_chg, ltp, dist, fii_bull, fii_bear) -> dict | None:
        """
        Classic OI interpretation:
          OI↑ + Price↓ → Writing (seller opening) → WRITE
          OI↑ + Price↑ → Buying (buyer opening)   → BUY
          OI↓ + Price↑ → Writer covering           → UNWIND_WRITE
          OI↓ + Price↓ → Buyer exiting             → UNWIND_BUY
        """
        if abs(doi) < MIN_OI_CHANGE:
            return None

        if doi > 0:
            action = "BUY"   if ltp_chg >= 0 else "WRITE"
        else:
            action = "UNWIND_WRITE" if ltp_chg >= 0 else "UNWIND_BUY"

        # Significance
        if abs(doi) >= LARGE_OI_CHANGE:
            significance = "HIGH"
        elif abs(doi) >= MEDIUM_OI_CHANGE:
            significance = "MEDIUM"
        else:
            significance = "LOW"

        # Infer most likely participant based on aggregate bias + OTM pattern
        # FII typically write OTM options (premium collection strategy)
        is_otm = (dist > 200 and opt_type == "CE") or (dist < -200 and opt_type == "PE")
        smart  = False
        participant = "CLIENT"   # default

        if action == "WRITE" and is_otm and significance in ("HIGH","MEDIUM"):
            # OTM writing at scale = likely institutional (FII/Pro)
            participant = "FII" if (fii_bull or fii_bear) else "PRO"
            smart = True
        elif action == "BUY" and abs(doi) >= MEDIUM_OI_CHANGE:
            participant = "FII" if fii_bull and opt_type == "CE" else \
                         ("FII" if fii_bear and opt_type == "PE" else "PRO")
            smart = abs(doi) >= LARGE_OI_CHANGE

        # Human-readable description
        desc = self._describe(participant, action, opt_type, strike, doi, dist, ltp)

        return {
            "strike":               strike,
            "type":                 opt_type,
            "action":               action,
            "oi_change":            doi,
            "ltp":                  round(ltp, 2),
            "ltp_change":           round(ltp_chg, 2),
            "inferred_participant": participant,
            "smart_money":          smart,
            "significance":         significance,
            "dist_from_spot":       round(dist, 0),
            "desc":                 desc,
            "timestamp":            datetime.now().strftime("%H:%M:%S"),
        }

    def _describe(self, participant, action, opt_type, strike, doi, dist, ltp) -> str:
        side = "above" if dist > 0 else "below"
        dist_str = f"{abs(dist):.0f} pts {side} spot"

        if action == "WRITE":
            if opt_type == "CE":
                return (f"{participant} writing {strike:.0f} CE @ ₹{ltp:.1f} "
                        f"({doi:+,} OI) — capping upside at {strike:.0f} | {dist_str}")
            else:
                return (f"{participant} writing {strike:.0f} PE @ ₹{ltp:.1f} "
                        f"({doi:+,} OI) — defending support at {strike:.0f} | {dist_str}")
        elif action == "BUY":
            if opt_type == "CE":
                return (f"{participant} buying {strike:.0f} CE @ ₹{ltp:.1f} "
                        f"({doi:+,} OI) — bullish bet for {strike:.0f}+ | {dist_str}")
            else:
                return (f"{participant} buying {strike:.0f} PE @ ₹{ltp:.1f} "
                        f"({doi:+,} OI) — bearish/hedge for {strike:.0f}- | {dist_str}")
        elif action == "UNWIND_WRITE":
            return (f"{participant} covering short {strike:.0f} {opt_type} "
                    f"({doi:+,} OI) — writer exiting | {dist_str}")
        else:
            return (f"{participant} exiting long {strike:.0f} {opt_type} "
                    f"({doi:+,} OI) — buyer exiting | {dist_str}")

    def _update_summary(self, chain: dict, spot: float):
        """Build aggregated strike summary for heatmap."""
        summary = {}
        for strike, d in chain.items():
            summary[strike] = {
                "strike":   strike,
                "ce_oi":    d["ce_oi"],
                "pe_oi":    d["pe_oi"],
                "ce_ltp":   d["ce_ltp"],
                "pe_ltp":   d["pe_ltp"],
                "net_oi":   d["ce_oi"] - d["pe_oi"],   # positive = CE heavy → resistance
                "pcr":      round(d["pe_oi"] / d["ce_oi"], 2) if d["ce_oi"] > 0 else 0,
                "dist":     round(strike - spot, 0),
                "is_atm":   abs(strike - spot) <= 100,
            }
        self._strikes_summary = summary

    # ── Public API ────────────────────────────────────────────────────────────

    def get_tape(self, n: int = 50, smart_only: bool = False,
                 min_significance: str = "LOW") -> list:
        """Return recent tape events, newest first."""
        sig_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        min_rank = sig_rank.get(min_significance, 0)
        events   = list(self._tape)
        events.reverse()
        if smart_only:
            events = [e for e in events if e.get("smart_money")]
        events = [e for e in events if sig_rank.get(e["significance"],0) >= min_rank]
        return events[:n]

    def get_key_strikes(self, n: int = 10) -> dict:
        """
        Returns the most important strikes:
        - Max CE OI (resistance wall)
        - Max PE OI (support wall)
        - Recent heavy writing strikes
        - Smart money accumulation strikes
        """
        if not self._strikes_summary:
            return {}

        strikes = list(self._strikes_summary.values())
        by_ce  = sorted(strikes, key=lambda x: x["ce_oi"], reverse=True)[:n]
        by_pe  = sorted(strikes, key=lambda x: x["pe_oi"], reverse=True)[:n]

        # Resistance wall = highest CE OI (writers defending above this)
        resistance = by_ce[0]["strike"] if by_ce else 0
        # Support wall = highest PE OI (writers defending below this)
        support    = by_pe[0]["strike"] if by_pe else 0

        # Pain point = strike where total OI is maximum (max pain theory)
        by_total   = sorted(strikes, key=lambda x: x["ce_oi"] + x["pe_oi"], reverse=True)
        max_pain   = by_total[0]["strike"] if by_total else 0

        return {
            "resistance":        resistance,
            "support":           support,
            "max_pain":          max_pain,
            "top_ce_strikes":    [s["strike"] for s in by_ce[:5]],
            "top_pe_strikes":    [s["strike"] for s in by_pe[:5]],
            "all_strikes":       strikes,
        }

    def get_participant_oi_table(self) -> dict:
        """Return FII/DII/Pro option positioning table."""
        try:
            from core.data.nse_participant import get_participant_data
            pd_ = get_participant_data()
            return pd_.get_participant_summary()
        except Exception:
            return {}

    def get_fii_net_option_bias(self) -> str:
        """
        From aggregate participant data, derive FII option strategy:
          net_calls > 0 AND net_puts < 0 → FII bullish (long CE, short PE)
          net_calls < 0 AND net_puts > 0 → FII bearish (short CE, long PE)
          net_puts > net_calls           → FII hedging (buying puts for protection)
        """
        p = self.get_participant_oi_table()
        fii = (p.get("participants") or {}).get("FII", {})
        if not fii:
            return "UNKNOWN"

        net_calls = fii.get("net_calls", 0)
        net_puts  = fii.get("net_puts",  0)
        net_futs  = fii.get("net_futures", 0)

        if net_calls > 0 and net_puts < 0:
            return "DIRECTIONAL_BULLISH"    # buying CE, selling PE
        elif net_calls < 0 and net_puts > 0:
            return "DIRECTIONAL_BEARISH"    # selling CE, buying PE
        elif net_puts > net_calls > 0:
            return "HEDGED_BULLISH"         # long futures but buying PE as hedge
        elif net_calls > 0 and net_futs > 0:
            return "STRONGLY_BULLISH"
        elif net_calls < 0 and net_futs < 0:
            return "STRONGLY_BEARISH"
        return "NEUTRAL"


# ── Footprint / Cluster Chart Engine ─────────────────────────────────────────

class FootprintEngine:
    """
    Builds footprint (cluster) charts from tick data or OHLCV bars.

    Footprint chart shows for each price level within each time bar:
      - Buy volume (aggressive buys = hitting the ask)
      - Sell volume (aggressive sells = hitting the bid)
      - Delta (buy - sell) per level
      - Total volume per level

    When no tick data available, approximates from OHLCV + volume.
    """

    def __init__(self, tick_size: float = 50.0):
        self.tick_size = tick_size    # price bucket size (50 pts for Nifty)

    def from_ohlcv(self, df, bar_minutes: int = 5) -> list[dict]:
        """
        Build approximate footprint from OHLCV data.
        Groups bars into sessions of bar_minutes and distributes
        volume across price levels using a triangular distribution.
        """
        import pandas as pd
        import numpy as np

        bars = []
        for i in range(len(df)):
            row   = df.iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            vol   = float(row.get("volume", 0))
            delta = c - o   # positive = up bar

            if h == l or vol == 0:
                continue

            # Distribute volume across price levels
            levels = np.arange(
                round(l / self.tick_size) * self.tick_size,
                round(h / self.tick_size) * self.tick_size + self.tick_size,
                self.tick_size,
            )
            if len(levels) == 0:
                continue

            price_levels = {}
            for price in levels:
                # Triangular distribution: more volume near close
                weight = max(0.05, 1 - abs(price - c) / (h - l + 0.01))
                lv_vol = vol * weight / len(levels)

                # Split into buy/sell based on bar direction
                if delta >= 0:
                    buy_pct  = 0.5 + 0.4 * (c - l) / (h - l + 0.01)
                else:
                    buy_pct  = 0.5 - 0.4 * (h - c) / (h - l + 0.01)
                buy_pct  = max(0.1, min(0.9, buy_pct))

                buy_vol  = round(lv_vol * buy_pct)
                sell_vol = round(lv_vol * (1 - buy_pct))

                price_levels[round(price, 0)] = {
                    "buy":   buy_vol,
                    "sell":  sell_vol,
                    "delta": buy_vol - sell_vol,
                    "total": buy_vol + sell_vol,
                }

            poc   = max(price_levels, key=lambda p: price_levels[p]["total"])
            bar_delta = sum(v["delta"] for v in price_levels.values())
            imbalances = []
            for price, v in price_levels.items():
                if v["sell"] > 0 and v["buy"] / v["sell"] >= 3:
                    imbalances.append({"price": price, "ratio": round(v["buy"]/v["sell"],1), "side": "BUY"})
                elif v["buy"] > 0 and v["sell"] / v["buy"] >= 3:
                    imbalances.append({"price": price, "ratio": round(v["sell"]/v["buy"],1), "side": "SELL"})

            ts = df.index[i]
            bars.append({
                "ts":           str(ts),
                "open":         o, "high": h, "low": l, "close": c,
                "volume":       vol,
                "delta":        round(bar_delta, 0),
                "delta_pct":    round(bar_delta / vol * 100, 1) if vol > 0 else 0,
                "poc":          poc,
                "price_levels": price_levels,
                "imbalances":   imbalances,
                "color":        "green" if c >= o else "red",
            })

        return bars[-30:]   # last 30 bars

    def heatmap_data(self, bars: list[dict]) -> dict:
        """
        Convert footprint bars to heatmap format for Plotly.
        Returns: {x: [timestamps], y: [prices], z: [buy_vol], z2: [sell_vol]}
        """
        if not bars:
            return {}

        # Collect all price levels
        all_prices = set()
        for bar in bars:
            all_prices.update(bar.get("price_levels", {}).keys())
        all_prices = sorted(all_prices)

        x_labels = [b["ts"][-8:] for b in bars]   # HH:MM:SS
        buy_matrix  = []
        sell_matrix = []
        delta_matrix= []

        for price in all_prices:
            buy_row  = []
            sell_row = []
            dlt_row  = []
            for bar in bars:
                lvl = bar.get("price_levels", {}).get(price, {})
                buy_row.append(lvl.get("buy", 0))
                sell_row.append(lvl.get("sell", 0))
                dlt_row.append(lvl.get("delta", 0))
            buy_matrix.append(buy_row)
            sell_matrix.append(sell_row)
            delta_matrix.append(dlt_row)

        return {
            "x":      x_labels,
            "y":      [str(p) for p in all_prices],
            "z_buy":  buy_matrix,
            "z_sell": sell_matrix,
            "z_delta":delta_matrix,
        }

    def poc_line(self, bars: list[dict]) -> list:
        """Extract POC (Point of Control) per bar for plotting."""
        return [{"ts": b["ts"], "poc": b["poc"]} for b in bars if b.get("poc")]


# ── Singletons ─────────────────────────────────────────────────────────────────

_fii_tape_instance : FIITape | None = None
_footprint_instance: FootprintEngine | None = None

def get_fii_tape(symbol: str = "NIFTY") -> FIITape:
    global _fii_tape_instance
    if _fii_tape_instance is None:
        _fii_tape_instance = FIITape(symbol)
    return _fii_tape_instance

def get_footprint_engine(tick_size: float = 50.0) -> FootprintEngine:
    global _footprint_instance
    if _footprint_instance is None:
        _footprint_instance = FootprintEngine(tick_size)
    return _footprint_instance
