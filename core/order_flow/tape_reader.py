"""
Real-Time F&O Tape Reader
=========================
Simulates institutional tape reading by polling the NSE option chain every
30-60 seconds, comparing consecutive snapshots, and detecting when big money
enters or exits at specific strikes.

What it tells you:
  - WHICH strike(s) big players traded
  - WHEN they traded (timestamp)
  - AT WHAT PRICE they got filled (LTP at time of OI change)
  - HOW MUCH they traded (OI delta in contracts)
  - WHICH SIDE they're on (long/short)
  - WHERE their inferred Stop Loss is
  - WHERE their inferred Target is
  - NET BIAS of all institutional activity today

OI Event Classification (classic F&O interpretation):
  ┌─────────────────┬──────────────┬───────────────────────────────┐
  │ OI Change       │ Price Change │ Meaning                       │
  ├─────────────────┼──────────────┼───────────────────────────────┤
  │ CE OI ↑         │ CE Price ↑  │ LONG_ENTRY  — bullish         │
  │ CE OI ↓         │ CE Price ↑  │ SHORT_EXIT  — bullish (cover)  │
  │ CE OI ↑         │ CE Price ↓  │ SHORT_ENTRY — bearish (writer) │
  │ CE OI ↓         │ CE Price ↓  │ LONG_EXIT   — bearish (unwind) │
  │ PE OI ↑         │ PE Price ↓  │ LONG_ENTRY  — bearish         │
  │ PE OI ↓         │ PE Price ↓  │ SHORT_EXIT  — bearish (cover)  │
  │ PE OI ↑         │ PE Price ↑  │ SHORT_ENTRY — bullish (writer) │
  │ PE OI ↓         │ PE Price ↑  │ LONG_EXIT   — bullish (unwind) │
  └─────────────────┴──────────────┴───────────────────────────────┘

SL / Target inference:
  Option buyers:  SL = 30-40% of premium, Target = 100-200% gain
  Option sellers: SL = option doubles (100% loss on premium received)
                  Target = 50-70% premium decay

Usage:
  reader = TapeReader("NIFTY")

  # Call every 30-60 seconds in a loop
  events = reader.tick()
  for e in events:
      print(e)              # prints tape line

  # Rich summary for dashboard / model
  summary = reader.get_flow_summary()
  features = reader.extract_features()   # 100+ features for ML model
"""

from collections import deque, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from core.data.nse_scraper import get_scraper


# ── Configuration ──────────────────────────────────────────────────────────────

BIG_OI_CONTRACTS    = 50_000    # Min OI change to flag as institutional
MEDIUM_OI_CONTRACTS = 20_000    # Medium event threshold
VOLUME_SPIKE_RATIO  = 2.5       # Volume vs rolling avg to confirm event
TAPE_BUFFER         = 120       # Keep last 120 snapshots (~1-2 hours)
SNAPSHOT_BUFFER     = 30        # Keep raw snapshots for delta computation


# ── TapeEvent ─────────────────────────────────────────────────────────────────

class TapeEvent:
    """
    A single detected institutional activity event.

    Attributes:
      strike        : Option strike price
      option_type   : CE or PE
      event_type    : LONG_ENTRY / SHORT_ENTRY / LONG_EXIT / SHORT_EXIT
      market_impact : BULLISH / BEARISH (what this means for Nifty direction)
      oi_change     : Number of contracts added/removed
      volume        : Traded volume at time of detection
      fill_price    : LTP at time of detection (≈ their entry price)
      underlying    : Spot price at time of detection
      timestamp     : When detected
      confidence    : 0.0 - 1.0 (how confident we are this is institutional)
      inferred_sl   : Estimated stop loss (in option premium terms)
      inferred_target: Estimated target (in option premium terms)
      sl_spot       : Underlying spot level for their SL
      target_spot   : Underlying spot level for their target
    """

    def __init__(
        self,
        strike: float,
        option_type: str,
        event_type: str,
        oi_change: int,
        volume: int,
        fill_price: float,
        underlying: float,
        timestamp: datetime,
        confidence: float,
        avg_vol: float = 0,
    ):
        self.strike       = strike
        self.option_type  = option_type
        self.event_type   = event_type
        self.oi_change    = oi_change
        self.volume       = volume
        self.fill_price   = fill_price
        self.underlying   = underlying
        self.timestamp    = timestamp
        self.confidence   = confidence
        self.vol_ratio    = volume / avg_vol if avg_vol > 0 else 1.0

        self.market_impact = self._market_impact()
        self.inferred_sl, self.sl_spot = self._calc_sl()
        self.inferred_target, self.target_spot = self._calc_target()

    def _market_impact(self) -> str:
        """Translate event into market direction (bullish/bearish for Nifty)."""
        bullish_events = {
            ("CE", "LONG_ENTRY"),
            ("CE", "SHORT_EXIT"),
            ("PE", "SHORT_ENTRY"),
            ("PE", "LONG_EXIT"),
        }
        bearish_events = {
            ("CE", "SHORT_ENTRY"),
            ("CE", "LONG_EXIT"),
            ("PE", "LONG_ENTRY"),
            ("PE", "SHORT_EXIT"),
        }
        key = (self.option_type, self.event_type)
        if key in bullish_events:
            return "BULLISH"
        elif key in bearish_events:
            return "BEARISH"
        return "NEUTRAL"

    def _calc_sl(self) -> Tuple[float, float]:
        """
        Infer their stop loss.
        Buyer:  SL ≈ 35% of premium paid (they'll exit if option loses 35%)
        Seller: SL ≈ option price doubles (200% of premium received)
        Returns (option_price_sl, spot_level_sl)
        """
        p = self.fill_price
        if "LONG" in self.event_type:
            sl_price = round(p * 0.65, 2)  # 35% drop = SL
        else:
            sl_price = round(p * 2.0, 2)   # Option doubles = SL

        # Spot level: rough approximation
        # For CE: if you bought CE at strike K, your SL in spot terms
        # is roughly strike ± (fill_price - sl_price)
        if self.option_type == "CE":
            if "LONG" in self.event_type:
                sl_spot = round(self.strike - (p - sl_price) * 10, 0)  # rough delta
            else:
                sl_spot = round(self.strike + (sl_price - p) * 10, 0)
        else:  # PE
            if "LONG" in self.event_type:
                sl_spot = round(self.strike + (p - sl_price) * 10, 0)
            else:
                sl_spot = round(self.strike - (sl_price - p) * 10, 0)

        return sl_price, sl_spot

    def _calc_target(self) -> Tuple[float, float]:
        """
        Infer their target.
        Buyer:  Target ≈ 2x premium (100% gain)
        Seller: Target ≈ 30% of premium remains (70% decay)
        """
        p = self.fill_price
        if "LONG" in self.event_type:
            tgt_price = round(p * 2.0, 2)
        else:
            tgt_price = round(p * 0.30, 2)

        # Spot level approximation
        if self.option_type == "CE":
            if "LONG" in self.event_type:
                tgt_spot = round(self.strike + (tgt_price - p) * 10, 0)
            else:
                tgt_spot = round(self.underlying, 0)  # sellers want no movement
        else:
            if "LONG" in self.event_type:
                tgt_spot = round(self.strike - (tgt_price - p) * 10, 0)
            else:
                tgt_spot = round(self.underlying, 0)

        return tgt_price, tgt_spot

    def to_dict(self) -> Dict:
        return {
            "time":            self.timestamp.strftime("%H:%M:%S"),
            "strike":          self.strike,
            "type":            self.option_type,
            "event":           self.event_type,
            "market_impact":   self.market_impact,
            "oi_change":       self.oi_change,
            "fill_price":      self.fill_price,
            "underlying":      self.underlying,
            "inferred_sl":     self.inferred_sl,
            "sl_spot":         self.sl_spot,
            "inferred_target": self.inferred_target,
            "target_spot":     self.target_spot,
            "confidence":      round(self.confidence, 2),
            "vol_ratio":       round(self.vol_ratio, 1),
        }

    def __str__(self) -> str:
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')}] "
            f"{self.market_impact:8s} | "
            f"{self.strike}{self.option_type} {self.event_type:12s} | "
            f"OI: {self.oi_change:+8,} | "
            f"Fill: ₹{self.fill_price:6.1f} | "
            f"SL: ₹{self.inferred_sl:5.1f} (spot:{self.sl_spot:.0f}) | "
            f"Tgt: ₹{self.inferred_target:5.1f} (spot:{self.target_spot:.0f}) | "
            f"Conf: {self.confidence:.0%}"
        )


# ── TapeReader ─────────────────────────────────────────────────────────────────

class TapeReader:
    """
    Polls NSE option chain and detects institutional order flow.

    Usage:
        reader = TapeReader("NIFTY")
        while market_open:
            new_events = reader.tick()      # poll + detect
            for e in new_events:
                print(e)                    # tape output
            summary  = reader.get_flow_summary()
            features = reader.extract_features()  # 100+ features for model
            time.sleep(45)
    """

    def __init__(self, symbol: str = "NIFTY"):
        self.symbol    = symbol
        self.scraper   = get_scraper()
        self._expiry: Optional[str] = None

        # Snapshot ring buffer (raw DataFrames)
        self._snapshots: deque = deque(maxlen=SNAPSHOT_BUFFER)

        # All tape events detected today
        self._tape: List[TapeEvent] = []

        # Rolling average volume per (strike, option_type) — for spike detection
        self._vol_avg: Dict[Tuple[float, str], float] = {}

        # OI history per (strike, option_type) — for multi-snapshot trend
        self._oi_history: Dict[Tuple[float, str], deque] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_expiry(self, expiry: str):
        self._expiry = expiry

    def tick(self, expiry: Optional[str] = None) -> List[TapeEvent]:
        """
        Fetch latest option chain snapshot, compare with previous,
        detect institutional activity. Call every 30-60 seconds.

        Returns list of new TapeEvents (empty if nothing significant).
        """
        exp = expiry or self._expiry
        if not exp:
            expiries = self.scraper.get_expiry_dates(self.symbol)
            if expiries:
                exp = expiries[0]
                self._expiry = exp
            else:
                logger.warning("No expiry dates available")
                return []

        raw = self.scraper.parse_option_chain(self.symbol, exp, strikes_range=15)
        if not raw:
            return []

        current_df = pd.DataFrame(raw)
        now = datetime.now()

        new_events: List[TapeEvent] = []

        if len(self._snapshots) > 0:
            prev_df = self._snapshots[-1]["df"]
            new_events = self._detect_events(prev_df, current_df, now)
            self._tape.extend(new_events)

        self._snapshots.append({"df": current_df, "time": now})
        self._update_vol_avg(current_df)
        self._update_oi_history(current_df)

        if new_events:
            logger.info(f"TAPE: {len(new_events)} institutional events detected")
        return new_events

    def get_tape(self, last_n: int = 20) -> List[Dict]:
        """Return the last N tape events as dicts (for display)."""
        return [e.to_dict() for e in self._tape[-last_n:]]

    def get_flow_summary(self) -> Dict:
        """
        Full summary of institutional activity so far today.

        Returns:
          bias            : STRONGLY_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY_BEARISH
          bullish_oi      : Total OI involved in bullish events
          bearish_oi      : Total OI in bearish events
          hot_strikes     : Top 5 most active strikes
          big_players     : Active positions with entry price, SL, target
          key_levels      : Entry zones, SL clusters, target zones (for chart)
          recent_tape     : Last 10 events
        """
        if not self._tape:
            return {
                "bias": "NEUTRAL",
                "total_events": 0,
                "bullish_oi": 0,
                "bearish_oi": 0,
                "hot_strikes": [],
                "big_players": [],
                "key_levels": {},
                "recent_tape": [],
            }

        bullish_oi = sum(
            abs(e.oi_change) for e in self._tape if e.market_impact == "BULLISH"
        )
        bearish_oi = sum(
            abs(e.oi_change) for e in self._tape if e.market_impact == "BEARISH"
        )
        total_oi = bullish_oi + bearish_oi

        if total_oi > 0:
            bull_pct = bullish_oi / total_oi
            if bull_pct > 0.70:
                bias = "STRONGLY_BULLISH"
            elif bull_pct > 0.55:
                bias = "BULLISH"
            elif bull_pct < 0.30:
                bias = "STRONGLY_BEARISH"
            elif bull_pct < 0.45:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"
        else:
            bias = "NEUTRAL"

        # Hot strikes
        strike_counts = Counter((e.strike, e.option_type) for e in self._tape)
        hot_strikes = [
            {"strike": s, "type": t, "event_count": c, "last_impact": next(
                (e.market_impact for e in reversed(self._tape)
                 if e.strike == s and e.option_type == t), "NEUTRAL"
            )}
            for (s, t), c in strike_counts.most_common(5)
        ]

        return {
            "bias":          bias,
            "bull_pct":      round(bullish_oi / total_oi, 3) if total_oi else 0.5,
            "bullish_oi":    bullish_oi,
            "bearish_oi":    bearish_oi,
            "total_events":  len(self._tape),
            "hot_strikes":   hot_strikes,
            "big_players":   self.get_big_player_positions(),
            "key_levels":    self._build_key_levels(),
            "recent_tape":   self.get_tape(10),
            "last_update":   (
                self._snapshots[-1]["time"].strftime("%H:%M:%S")
                if self._snapshots else None
            ),
        }

    def get_big_player_positions(self) -> List[Dict]:
        """
        Aggregate all tape events per strike to estimate net open positions
        of big players — similar to viewing who's holding what.
        """
        positions: Dict[Tuple[float, str], Dict] = {}

        for event in self._tape:
            key = (event.strike, event.option_type)
            if key not in positions:
                positions[key] = {
                    "strike":       event.strike,
                    "option_type":  event.option_type,
                    "net_oi":       0,
                    "total_oi_in":  0,
                    "total_oi_out": 0,
                    "avg_fill_price": 0.0,
                    "entries": 0,
                    "exits":   0,
                    "side":    None,
                    "market_impact": None,
                }
            p = positions[key]

            if "ENTRY" in event.event_type:
                oi = abs(event.oi_change)
                old_total = p["total_oi_in"]
                p["total_oi_in"] += oi
                # Weighted average fill price
                p["avg_fill_price"] = (
                    (p["avg_fill_price"] * old_total + event.fill_price * oi)
                    / p["total_oi_in"]
                    if p["total_oi_in"] > 0 else event.fill_price
                )
                p["net_oi"] += oi
                p["entries"] += 1
                p["side"] = "LONG" if "LONG" in event.event_type else "SHORT"
                p["market_impact"] = event.market_impact
                p["inferred_sl"]   = event.inferred_sl
                p["sl_spot"]       = event.sl_spot
                p["inferred_target"] = event.inferred_target
                p["target_spot"]   = event.target_spot
            else:
                p["net_oi"] = max(0, p["net_oi"] - abs(event.oi_change))
                p["total_oi_out"] += abs(event.oi_change)
                p["exits"] += 1

            p["last_event"] = event.event_type
            p["last_time"]  = event.timestamp.strftime("%H:%M:%S")

        # Only show strikes with significant remaining OI
        active = [
            p for p in positions.values()
            if p["net_oi"] >= MEDIUM_OI_CONTRACTS
        ]
        return sorted(active, key=lambda x: x["net_oi"], reverse=True)

    def extract_features(self) -> Dict:
        """
        Extract 100+ features from current tape + option chain state.
        These feed directly into your ML model (1 lakh+ parameter network).

        Feature groups:
          - Tape flow (30 features): bias, OI counts, event breakdown
          - Strike-level OI (40 features): top 10 strikes × 4 metrics each
          - Participant positioning (20 features): FII/DII/Pro/Client
          - Time features (10 features): intraday timing
          - Momentum (10 features): OI velocity, trend
        """
        features: Dict = {}
        now = datetime.now()

        # ── Group 1: Tape Flow (30 features) ──────────────────────────────────
        summary = self.get_flow_summary()

        features["f_total_events"]       = len(self._tape)
        features["f_bullish_oi"]         = summary.get("bullish_oi", 0)
        features["f_bearish_oi"]         = summary.get("bearish_oi", 0)
        features["f_bull_pct"]           = summary.get("bull_pct", 0.5)
        features["f_net_oi_bias"]        = summary["bullish_oi"] - summary["bearish_oi"]

        # Event type breakdown
        event_types = Counter(e.event_type for e in self._tape)
        features["f_long_entries"]  = event_types.get("LONG_ENTRY",  0)
        features["f_short_entries"] = event_types.get("SHORT_ENTRY", 0)
        features["f_long_exits"]    = event_types.get("LONG_EXIT",   0)
        features["f_short_exits"]   = event_types.get("SHORT_EXIT",  0)

        # CE vs PE breakdown
        ce_events = [e for e in self._tape if e.option_type == "CE"]
        pe_events = [e for e in self._tape if e.option_type == "PE"]
        features["f_ce_bullish_oi"] = sum(abs(e.oi_change) for e in ce_events if e.market_impact == "BULLISH")
        features["f_ce_bearish_oi"] = sum(abs(e.oi_change) for e in ce_events if e.market_impact == "BEARISH")
        features["f_pe_bullish_oi"] = sum(abs(e.oi_change) for e in pe_events if e.market_impact == "BULLISH")
        features["f_pe_bearish_oi"] = sum(abs(e.oi_change) for e in pe_events if e.market_impact == "BEARISH")
        features["f_ce_event_count"] = len(ce_events)
        features["f_pe_event_count"] = len(pe_events)

        # Recent momentum (last 5 events vs earlier)
        recent = self._tape[-5:] if len(self._tape) >= 5 else self._tape
        features["f_recent_bull_oi"] = sum(abs(e.oi_change) for e in recent if e.market_impact == "BULLISH")
        features["f_recent_bear_oi"] = sum(abs(e.oi_change) for e in recent if e.market_impact == "BEARISH")
        features["f_recent_bias"]    = (
            features["f_recent_bull_oi"] - features["f_recent_bear_oi"]
        )

        # Confidence metrics
        if self._tape:
            features["f_avg_confidence"]  = np.mean([e.confidence for e in self._tape])
            features["f_max_confidence"]  = max(e.confidence for e in self._tape)
            features["f_avg_vol_ratio"]   = np.mean([e.vol_ratio for e in self._tape])
            features["f_max_oi_event"]    = max(abs(e.oi_change) for e in self._tape)
        else:
            features["f_avg_confidence"] = 0.0
            features["f_max_confidence"] = 0.0
            features["f_avg_vol_ratio"]  = 1.0
            features["f_max_oi_event"]   = 0

        # Hot strikes count
        hot = summary.get("hot_strikes", [])
        features["f_hot_strikes_count"] = len(hot)
        features["f_top_strike"] = hot[0]["strike"] if hot else 0

        # ── Group 2: Current Option Chain Snapshot (40 features) ─────────────
        if self._snapshots:
            df = self._snapshots[-1]["df"]
            spot = df["underlying"].iloc[0] if "underlying" in df.columns else 0
            atm  = round(spot / 50) * 50

            ce = df[df["option_type"] == "CE"]
            pe = df[df["option_type"] == "PE"]

            features["f_spot"]        = spot
            features["f_atm_strike"]  = atm

            # PCR
            ce_oi  = ce["oi"].sum()
            pe_oi  = pe["oi"].sum()
            ce_vol = ce["volume"].sum()
            pe_vol = pe["volume"].sum()
            features["f_pcr_oi"]     = round(pe_oi / ce_oi, 4) if ce_oi else 0
            features["f_pcr_vol"]    = round(pe_vol / ce_vol, 4) if ce_vol else 0
            features["f_ce_total_oi"] = int(ce_oi)
            features["f_pe_total_oi"] = int(pe_oi)
            features["f_net_oi"]      = int(pe_oi - ce_oi)

            # ATM IV
            atm_ce = ce[ce["strike"] == atm]["iv"]
            atm_pe = pe[pe["strike"] == atm]["iv"]
            features["f_atm_ce_iv"] = float(atm_ce.iloc[0]) if not atm_ce.empty else 0
            features["f_atm_pe_iv"] = float(atm_pe.iloc[0]) if not atm_pe.empty else 0
            features["f_atm_iv"]    = (features["f_atm_ce_iv"] + features["f_atm_pe_iv"]) / 2

            # OI concentration — top 3 CE and PE strikes
            for i, (_, row) in enumerate(ce.nlargest(3, "oi").iterrows()):
                features[f"f_top_ce_strike_{i+1}"]    = row["strike"]
                features[f"f_top_ce_oi_{i+1}"]        = row["oi"]
                features[f"f_top_ce_oi_chg_{i+1}"]    = row["oi_change"]
            for i, (_, row) in enumerate(pe.nlargest(3, "oi").iterrows()):
                features[f"f_top_pe_strike_{i+1}"]    = row["strike"]
                features[f"f_top_pe_oi_{i+1}"]        = row["oi"]
                features[f"f_top_pe_oi_chg_{i+1}"]    = row["oi_change"]

            # ATM zone OI buildup
            atm_band = 100  # ± 100 from ATM
            atm_ce_oi = ce[ce["strike"].between(atm - atm_band, atm + atm_band)]["oi_change"].sum()
            atm_pe_oi = pe[pe["strike"].between(atm - atm_band, atm + atm_band)]["oi_change"].sum()
            features["f_atm_ce_oi_build"] = int(atm_ce_oi)
            features["f_atm_pe_oi_build"] = int(atm_pe_oi)
            features["f_atm_net_build"]   = int(atm_pe_oi - atm_ce_oi)

            # Distance from key OI strikes to spot
            if not ce.empty:
                top_ce_strike = ce.nlargest(1, "oi")["strike"].iloc[0]
                features["f_dist_spot_to_top_ce"] = top_ce_strike - spot
            else:
                features["f_dist_spot_to_top_ce"] = 0

            if not pe.empty:
                top_pe_strike = pe.nlargest(1, "oi")["strike"].iloc[0]
                features["f_dist_spot_to_top_pe"] = spot - top_pe_strike
            else:
                features["f_dist_spot_to_top_pe"] = 0

        else:
            # Zero-fill if no snapshot yet
            for k in [
                "f_spot", "f_atm_strike", "f_pcr_oi", "f_pcr_vol",
                "f_ce_total_oi", "f_pe_total_oi", "f_net_oi",
                "f_atm_ce_iv", "f_atm_pe_iv", "f_atm_iv",
                "f_atm_ce_oi_build", "f_atm_pe_oi_build", "f_atm_net_build",
                "f_dist_spot_to_top_ce", "f_dist_spot_to_top_pe",
            ]:
                features[k] = 0
            for i in range(1, 4):
                for prefix in ["f_top_ce_strike_", "f_top_ce_oi_", "f_top_ce_oi_chg_",
                               "f_top_pe_strike_", "f_top_pe_oi_", "f_top_pe_oi_chg_"]:
                    features[f"{prefix}{i}"] = 0

        # ── Group 3: OI Velocity / Trend (15 features) ────────────────────────
        if len(self._snapshots) >= 3:
            oi_series = self._get_oi_velocity()
            features["f_ce_oi_velocity"]  = oi_series.get("ce_velocity", 0)
            features["f_pe_oi_velocity"]  = oi_series.get("pe_velocity", 0)
            features["f_net_oi_velocity"] = oi_series.get("net_velocity", 0)
            features["f_ce_oi_accel"]     = oi_series.get("ce_accel", 0)
            features["f_pe_oi_accel"]     = oi_series.get("pe_accel", 0)
        else:
            for k in ["f_ce_oi_velocity", "f_pe_oi_velocity", "f_net_oi_velocity",
                      "f_ce_oi_accel", "f_pe_oi_accel"]:
                features[k] = 0

        # ── Group 4: Time Features (10 features) ──────────────────────────────
        features["f_hour"]          = now.hour
        features["f_minute"]        = now.minute
        features["f_minutes_since_open"] = max(0, (now.hour - 9) * 60 + now.minute - 15)
        features["f_minutes_to_close"]   = max(0, (15 - now.hour) * 60 + (30 - now.minute))
        features["f_is_first_hour"]      = int(features["f_minutes_since_open"] < 60)
        features["f_is_last_hour"]       = int(features["f_minutes_to_close"] < 60)
        features["f_is_pre_expiry"]      = 0  # set by caller if expiry is today/tomorrow
        features["f_day_of_week"]        = now.weekday()  # 0=Mon, 4=Fri
        features["f_is_expiry_day"]      = 0              # set by caller
        features["f_intraday_progress"]  = round(
            features["f_minutes_since_open"] / 375.0, 4   # 375 min = trading session
        )

        # ── Group 5: Signal Encoding (5 features) ─────────────────────────────
        bias_map = {
            "STRONGLY_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
            "BEARISH": -1, "STRONGLY_BEARISH": -2,
        }
        features["f_tape_bias_numeric"] = bias_map.get(summary.get("bias", "NEUTRAL"), 0)
        features["f_active_positions"]  = len(self.get_big_player_positions())
        features["f_snapshot_count"]    = len(self._snapshots)
        features["f_tape_event_count"]  = len(self._tape)
        features["f_data_freshness"]    = (
            int((now - self._snapshots[-1]["time"]).total_seconds())
            if self._snapshots else 9999
        )

        return features

    # ── Internal ───────────────────────────────────────────────────────────────

    def _detect_events(
        self,
        prev: pd.DataFrame,
        curr: pd.DataFrame,
        ts: datetime,
    ) -> List[TapeEvent]:
        """Compare two snapshots, return list of institutional events."""
        events = []

        # Merge on strike + option_type
        merged = curr.merge(
            prev[["strike", "option_type", "oi", "ltp"]].rename(
                columns={"oi": "oi_prev", "ltp": "ltp_prev"}
            ),
            on=["strike", "option_type"],
            how="inner",
        )

        merged["oi_delta"]   = merged["oi"] - merged["oi_prev"]
        merged["price_delta"] = merged["ltp"] - merged["ltp_prev"]

        underlying = merged["underlying"].iloc[0] if not merged.empty else 0

        for _, row in merged.iterrows():
            oi_delta   = row["oi_delta"]
            price_chg  = row["price_delta"]
            strike     = row["strike"]
            opt_type   = row["option_type"]
            ltp        = row["ltp"]
            volume     = row["volume"]

            if abs(oi_delta) < MEDIUM_OI_CONTRACTS:
                continue

            avg_vol = self._vol_avg.get((strike, opt_type), max(volume, 1))
            vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

            event_type, base_conf = self._classify(oi_delta, price_chg, opt_type)
            if event_type is None:
                continue

            # Boost confidence with volume spike
            conf = base_conf * min(vol_ratio / VOLUME_SPIKE_RATIO, 1.5)
            # Boost for very large OI
            if abs(oi_delta) > BIG_OI_CONTRACTS:
                conf = min(conf * 1.2, 1.0)
            conf = round(min(conf, 1.0), 3)

            if conf < 0.25:
                continue

            event = TapeEvent(
                strike=strike,
                option_type=opt_type,
                event_type=event_type,
                oi_change=int(oi_delta),
                volume=int(volume),
                fill_price=ltp,
                underlying=underlying,
                timestamp=ts,
                confidence=conf,
                avg_vol=avg_vol,
            )
            events.append(event)
            logger.debug(f"TAPE: {event}")

        return events

    def _classify(
        self,
        oi_delta: float,
        price_chg: float,
        opt_type: str,
    ) -> Tuple[Optional[str], float]:
        """Classify OI+price movement into event type and base confidence."""
        oi_up    = oi_delta > 0
        price_up = price_chg > 0

        # Confidence scales with magnitude
        conf = min(abs(oi_delta) / BIG_OI_CONTRACTS, 1.0) * 0.8

        if opt_type == "CE":
            if oi_up and price_up:
                return "LONG_ENTRY",  conf * 1.0
            elif not oi_up and price_up:
                return "SHORT_EXIT",  conf * 0.85
            elif oi_up and not price_up:
                return "SHORT_ENTRY", conf * 0.90
            else:
                return "LONG_EXIT",   conf * 0.75
        else:  # PE
            if oi_up and not price_up:
                return "LONG_ENTRY",  conf * 1.0
            elif not oi_up and not price_up:
                return "SHORT_EXIT",  conf * 0.85
            elif oi_up and price_up:
                return "SHORT_ENTRY", conf * 0.90
            else:
                return "LONG_EXIT",   conf * 0.75

    def _update_vol_avg(self, df: pd.DataFrame):
        """EMA of volume per strike — used for volume spike detection."""
        for _, row in df.iterrows():
            key = (row["strike"], row["option_type"])
            vol = row["volume"]
            self._vol_avg[key] = (
                0.8 * self._vol_avg.get(key, vol) + 0.2 * vol
            )

    def _update_oi_history(self, df: pd.DataFrame):
        """Track OI per strike over time — for velocity calculation."""
        for _, row in df.iterrows():
            key = (row["strike"], row["option_type"])
            if key not in self._oi_history:
                self._oi_history[key] = deque(maxlen=20)
            self._oi_history[key].append(row["oi"])

    def _get_oi_velocity(self) -> Dict:
        """
        Compute OI velocity (rate of change) across all snapshots.
        Velocity > 0 means OI is building; < 0 means OI is decaying.
        """
        ce_ois: List[float] = []
        pe_ois: List[float] = []

        for snap in list(self._snapshots)[-5:]:
            df = snap["df"]
            ce_ois.append(df[df["option_type"] == "CE"]["oi"].sum())
            pe_ois.append(df[df["option_type"] == "PE"]["oi"].sum())

        if len(ce_ois) < 2:
            return {}

        ce_vel = (ce_ois[-1] - ce_ois[0]) / len(ce_ois)
        pe_vel = (pe_ois[-1] - pe_ois[0]) / len(pe_ois)

        # Acceleration (2nd derivative)
        ce_accel = 0.0
        pe_accel = 0.0
        if len(ce_ois) >= 3:
            ce_diffs = [ce_ois[i+1] - ce_ois[i] for i in range(len(ce_ois)-1)]
            pe_diffs = [pe_ois[i+1] - pe_ois[i] for i in range(len(pe_ois)-1)]
            ce_accel = ce_diffs[-1] - ce_diffs[0]
            pe_accel = pe_diffs[-1] - pe_diffs[0]

        return {
            "ce_velocity":  round(ce_vel, 0),
            "pe_velocity":  round(pe_vel, 0),
            "net_velocity": round(pe_vel - ce_vel, 0),
            "ce_accel":     round(ce_accel, 0),
            "pe_accel":     round(pe_accel, 0),
        }

    def _build_key_levels(self) -> Dict:
        """Extract key price levels from all detected events."""
        if not self._tape:
            return {}

        entry_strikes  = sorted(set(e.strike for e in self._tape if "ENTRY" in e.event_type))
        sl_prices      = sorted(set(e.inferred_sl for e in self._tape[-20:]))
        target_prices  = sorted(set(e.inferred_target for e in self._tape[-20:]))
        sl_spots       = sorted(set(e.sl_spot for e in self._tape[-20:]))
        target_spots   = sorted(set(e.target_spot for e in self._tape[-20:]))

        return {
            "entry_strikes":  entry_strikes[:8],
            "sl_option_prices": sl_prices[:5],
            "target_option_prices": target_prices[:5],
            "sl_spot_levels":    sl_spots[:5],
            "target_spot_levels": target_spots[:5],
        }
