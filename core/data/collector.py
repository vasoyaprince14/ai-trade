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
        init_db()
        self._daily_collected = set()   # track what we've done today
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

            # 6. Save summary
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
            }
            self._save_parquet(symbol, "summary", [summary], ts)
            logger.info(f"[{symbol}] LTP={summary['ltp']} VIX={vix} PCR={summary['pcr_oi']:.3f} MaxPain={max_pain}")

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

    def collect_daily_once(self):
        """Things to collect once per day (at market open)."""
        today = datetime.now(IST).date().isoformat()
        if today in self._daily_collected:
            return
        self._daily_collected.add(today)

        # Collect daily OHLCV (longer history)
        for sym in SYMBOLS:
            df = fetch_historical(sym, "1d", days=365)
            if not df.empty:
                df = add_indicators(df)
                with DBManager() as db:
                    db.save_ohlc(df.to_dict("records"))
            logger.info(f"Daily OHLCV collected for {sym}: {len(df)} bars")

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
