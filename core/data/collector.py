"""
Continuous Market Data Collector
==================================
Runs every 5 minutes during market hours (9:15–15:30 IST).

What it collects:
  - NSE option chain (OI, IV, Greeks, bid/ask for every strike)
  - Index quotes (NIFTY, BANKNIFTY, FINNIFTY)
  - India VIX
  - PCR (Put-Call Ratio)
  - FII/DII data (once per day)
  - OHLCV bars (via yfinance)

Inspired by: vendors/nse-options-collector/collectors/oi_collector.py

Data is stored in:
  - SQLite DB (OISnapshot, OHLCBar tables)
  - Parquet files (data/snapshots/) for ML training
"""
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.data.nse_scraper import get_scraper
from core.data.historical import fetch_historical, add_indicators
from core.data.db import init_db, DBManager
from core.data.fii_dii import get_fii_fetcher
from core.institutional.positioning import get_analyzer
from core.institutional.strike_attribution import get_strike_engine

IST = pytz.timezone("Asia/Kolkata")
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]


class MarketDataCollector:
    """
    Collects and stores market data snapshots every 5 minutes.
    Integrates ideas from nse-options-collector (BarathGB007).
    """

    def __init__(self):
        self.scraper = get_scraper()
        self.fii_fetcher = get_fii_fetcher()
        self.inst_analyzer = get_analyzer()
        self.strike_engine = get_strike_engine()
        init_db()
        self._daily_collected = set()   # track what we've done today
        # Cache last institutional analysis for intraday snapshot enrichment
        self._last_inst: dict = {}
        logger.info("MarketDataCollector initialized")

    # ── Main collection tick ───────────────────────────────────────────────────

    def collect_snapshot(self, symbol: str = "NIFTY"):
        """Collect one full snapshot: OC + quote + PCR."""
        ts = datetime.now(IST)
        logger.info(f"[Collector] Snapshot: {symbol} @ {ts.strftime('%H:%M:%S')}")

        with DBManager() as db:
            # 1. Option chain
            records = self.scraper.parse_option_chain(symbol, strikes_range=15)
            if records:
                db.save_oi_snapshot(records)
                self._save_parquet(symbol, "option_chain", records, ts)

            # 2. Index quote
            quote_sym = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}.get(symbol, symbol)
            quote = self.scraper.get_index_quote(quote_sym)

            # 3. PCR
            pcr = self.scraper.get_pcr_data(symbol)

            # 4. VIX
            vix = self.scraper.get_vix()

            # 5. Max Pain
            max_pain = self.scraper.get_max_pain(symbol)

            # 6. Strike attribution (uses last institutional data if available)
            strike_result = {}
            if records and quote:
                spot = quote.get("ltp", 0) if quote else 0
                inst = self._last_inst
                try:
                    strike_result = self.strike_engine.analyze(
                        option_chain=records,
                        spot=spot,
                        fii_call_bias=inst.get("fii_call_bias", 0),
                        fii_put_bias=inst.get("fii_put_bias", 0),
                        fii_composite=inst.get("fii_composite", 0),
                    )
                except Exception as e:
                    logger.debug(f"Strike attribution error: {e}")

            # 7. Save summary
            summary = {
                "timestamp":   ts.isoformat(),
                "symbol":      symbol,
                "ltp":         quote.get("ltp", 0) if quote else 0,
                "change_pct":  quote.get("change_pct", 0) if quote else 0,
                "vix":         vix or 0,
                "pcr_oi":      pcr.get("pcr_oi", 0) if pcr else 0,
                "pcr_vol":     pcr.get("pcr_vol", 0) if pcr else 0,
                "total_call_oi": pcr.get("total_call_oi", 0) if pcr else 0,
                "total_put_oi":  pcr.get("total_put_oi", 0) if pcr else 0,
                "max_pain":    max_pain or 0,
                "oc_records":  len(records),
                # Institutional enrichment
                "fii_regime":   self._last_inst.get("regime", ""),
                "fii_composite": self._last_inst.get("composite", 0),
                "fii_direction": strike_result.get("inferred_direction", ""),
                "gamma_pin":    strike_result.get("gamma_pin_strike", 0) or 0,
                "nearest_resistance": (
                    strike_result.get("institutional_zones", {}).get("nearest_resistance") or 0
                ),
                "nearest_support": (
                    strike_result.get("institutional_zones", {}).get("nearest_support") or 0
                ),
            }
            self._save_parquet(symbol, "summary", [summary], ts)
            logger.info(
                f"[{symbol}] LTP={summary['ltp']} VIX={vix} PCR={summary['pcr_oi']:.3f} "
                f"MaxPain={max_pain} FII={summary['fii_regime']}"
            )

    def collect_ohlcv(self, symbol: str = "NIFTY"):
        """Collect 5-min OHLCV bar."""
        df = fetch_historical(symbol, "5min", days=2)
        if df.empty:
            return
        df = add_indicators(df)
        with DBManager() as db:
            bars = df.tail(3).to_dict("records")
            db.save_ohlc(bars)
        logger.debug(f"[{symbol}] OHLCV saved: {len(df)} bars")

    def collect_all(self):
        """Run one full collection cycle for all symbols."""
        for sym in SYMBOLS:
            try:
                self.collect_snapshot(sym)
                self.collect_ohlcv(sym)
                time.sleep(2)
            except Exception as e:
                logger.error(f"Collection error for {sym}: {e}")

    def collect_institutional(self):
        """Fetch FII/DII participant data, analyze, and persist."""
        try:
            today_data = self.fii_fetcher.get_today_data()
            if not today_data.get("participants"):
                logger.warning("No institutional participant data available")
                return

            # Full analysis (regime, divergences, velocity, etc.)
            analysis = self.inst_analyzer.analyze(today_data)
            self._last_inst = {**today_data, **analysis}

            # Persist to DB
            rows = []
            for ct, part in today_data.get("participants", {}).items():
                row = {k: v for k, v in part.items() if not isinstance(v, dict)}
                row["client_type"] = ct
                row["date"] = today_data["date"]
                # Attach cash flow for FII/DII rows
                if ct == "FII":
                    row["cash_net"]  = today_data.get("cash", {}).get("fii_net", 0)
                    row["cash_buy"]  = today_data.get("cash", {}).get("fii_buy", 0)
                    row["cash_sell"] = today_data.get("cash", {}).get("fii_sell", 0)
                    row["regime"]    = analysis.get("regime", "")
                    row["composite"] = analysis.get("composite", 0)
                elif ct == "DII":
                    row["cash_net"]  = today_data.get("cash", {}).get("dii_net", 0)
                    row["cash_buy"]  = today_data.get("cash", {}).get("dii_buy", 0)
                    row["cash_sell"] = today_data.get("cash", {}).get("dii_sell", 0)
                rows.append(row)

            with DBManager() as db:
                db.save_institutional_snapshot(rows)

            logger.info(
                f"Institutional data saved: regime={analysis.get('regime')} "
                f"composite={analysis.get('composite'):+.3f}"
            )

            # Persist analysis summary to parquet for ML
            ts = datetime.now(IST)
            flat = {
                "timestamp":         ts.isoformat(),
                "fii_regime":        analysis.get("regime", ""),
                "fii_composite":     analysis.get("composite", 0),
                "fii_futures_bias":  analysis.get("futures_bias", 0),
                "fii_call_bias":     analysis.get("call_bias", 0),
                "fii_put_bias":      analysis.get("put_bias", 0),
                "fii_cash_net":      analysis.get("fii_cash_net", 0),
                "dii_cash_net":      analysis.get("dii_cash_net", 0),
                "dii_absorption":    analysis.get("dii_absorption", 0),
                "divergences":       "|".join(analysis.get("divergences", [])),
            }
            self._save_parquet("INSTITUTIONAL", "daily", [flat], ts)

        except Exception as e:
            logger.error(f"Institutional collection error: {e}")

    def collect_daily_once(self):
        """Things to collect once per day (at market open)."""
        today = datetime.now(IST).date().isoformat()
        if today in self._daily_collected:
            return
        self._daily_collected.add(today)

        # 1. Daily OHLCV (longer history)
        for sym in SYMBOLS:
            df = fetch_historical(sym, "1d", days=365)
            if not df.empty:
                df = add_indicators(df)
                with DBManager() as db:
                    db.save_ohlc(df.to_dict("records"))
            logger.info(f"Daily OHLCV collected for {sym}: {len(df)} bars")

        # 2. FII/DII institutional data
        self.collect_institutional()

    # ── Parquet storage (for ML training) ─────────────────────────────────────

    def _save_parquet(self, symbol: str, data_type: str, records: List[dict], ts: datetime):
        """Save snapshot as parquet file for ML feature engineering."""
        date_str = ts.strftime("%Y%m%d")
        hour_str = ts.strftime("%H%M")
        path = SNAPSHOT_DIR / symbol / data_type / date_str
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{hour_str}.parquet"
        df = pd.DataFrame(records)
        df.to_parquet(file, index=False)

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def run_scheduled(self):
        """Run continuous collection on 5-min schedule during market hours."""
        logger.info("Starting scheduled collection (5-min intervals, 9:15–15:30 IST)")
        scheduler = BlockingScheduler(timezone=IST)

        # Every 5 minutes during market hours
        scheduler.add_job(
            self.collect_all,
            CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*/5",
                timezone=IST,
            ),
            id="snapshot_collector",
            name="NSE Snapshot",
        )

        # Daily once at 9:16
        scheduler.add_job(
            self.collect_daily_once,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=16, timezone=IST),
            id="daily_collector",
            name="Daily OHLCV",
        )

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Collector stopped")

    # ── Run one shot ──────────────────────────────────────────────────────────

    def run_once(self):
        """Single snapshot collection (for testing)."""
        self.collect_all()


def start_collector():
    collector = MarketDataCollector()
    collector.run_scheduled()
