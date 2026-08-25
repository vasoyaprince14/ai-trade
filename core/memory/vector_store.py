"""
Market State Vector Store — Qdrant
====================================
Stores every market snapshot as a vector in Qdrant so the trading agent
can ask: "What happened in the past when conditions were like this?"

How it works:
  1. Every N minutes → extract 100+ features from option chain + tape
  2. L2-normalize the feature vector → store in Qdrant with metadata
  3. At decision time → query Qdrant for top-K most similar historical states
  4. Look at what outcome those states had (direction, P&L)
  5. Agent uses this "market memory" to decide the current trade

Collection schema:
  - id          : UUID (auto)
  - vector      : float32[N_FEATURES]  (normalized feature vector)
  - payload     :
      timestamp       : ISO string
      symbol          : NIFTY / BANKNIFTY
      spot            : float
      pcr_oi          : float
      tape_bias       : str
      fii_bias        : str
      smart_money     : str
      atm_iv          : float
      vix             : float
      features_json   : json string of full feature dict
      outcome         : null initially; filled when we know the result
        direction  : UP / DOWN / FLAT
        move_pct   : float (how much spot moved in next 30 min)
        signal     : the trade signal that was generated
        pnl        : P&L of the trade (if taken)

Qdrant connection:
  - Local (dev): localhost:6333
  - Docker:      qdrant:6333   (via QDRANT_HOST env var)
"""

import os
import json
import uuid
import math
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("qdrant-client not installed. Vector store disabled. Run: pip install qdrant-client")


COLLECTION_NAME = "market_states"

# Feature keys that form the vector — must be consistent across all snapshots
# These are the numeric features from TapeReader.extract_features() + extras
VECTOR_FEATURE_KEYS = [
    # Tape flow
    "f_total_events", "f_bull_pct", "f_bullish_oi", "f_bearish_oi", "f_net_oi_bias",
    "f_long_entries", "f_short_entries", "f_long_exits", "f_short_exits",
    "f_ce_bullish_oi", "f_ce_bearish_oi", "f_pe_bullish_oi", "f_pe_bearish_oi",
    "f_ce_event_count", "f_pe_event_count",
    "f_recent_bull_oi", "f_recent_bear_oi", "f_recent_bias",
    "f_avg_confidence", "f_max_confidence", "f_avg_vol_ratio", "f_max_oi_event",
    "f_hot_strikes_count", "f_tape_bias_numeric",
    # Option chain
    "f_pcr_oi", "f_pcr_vol", "f_ce_total_oi", "f_pe_total_oi", "f_net_oi",
    "f_atm_ce_iv", "f_atm_pe_iv", "f_atm_iv",
    "f_top_ce_oi_1", "f_top_ce_oi_2", "f_top_ce_oi_3",
    "f_top_pe_oi_1", "f_top_pe_oi_2", "f_top_pe_oi_3",
    "f_top_ce_oi_chg_1", "f_top_ce_oi_chg_2", "f_top_ce_oi_chg_3",
    "f_top_pe_oi_chg_1", "f_top_pe_oi_chg_2", "f_top_pe_oi_chg_3",
    "f_atm_ce_oi_build", "f_atm_pe_oi_build", "f_atm_net_build",
    "f_dist_spot_to_top_ce", "f_dist_spot_to_top_pe",
    # OI velocity
    "f_ce_oi_velocity", "f_pe_oi_velocity", "f_net_oi_velocity",
    "f_ce_oi_accel", "f_pe_oi_accel",
    # Time
    "f_minutes_since_open", "f_minutes_to_close",
    "f_is_first_hour", "f_is_last_hour", "f_day_of_week", "f_intraday_progress",
    # FII/DII
    "f_fii_net_futures", "f_fii_net_calls", "f_fii_net_puts", "f_fii_bias_score",
    "f_dii_net_futures", "f_dii_net_calls", "f_dii_net_puts", "f_dii_bias_score",
    "f_pro_net_futures", "f_pro_net_calls", "f_pro_net_puts",
    "f_fii_bias_numeric", "f_dii_bias_numeric", "f_smart_money_bias",
    # VIX
    "f_vix",
    # Active positions
    "f_active_positions", "f_snapshot_count", "f_tape_event_count",
]

N_FEATURES = len(VECTOR_FEATURE_KEYS)


def _normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector for cosine similarity in Qdrant."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _features_to_vector(features: Dict) -> List[float]:
    """Extract and order the feature vector from features dict."""
    vec = [float(features.get(k, 0.0)) for k in VECTOR_FEATURE_KEYS]
    # Replace NaN/Inf
    vec = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in vec]
    return _normalize(vec)


class MarketVectorStore:
    """
    Qdrant-backed market state memory.

    Stores every market snapshot as a searchable vector.
    At decision time, retrieves the most similar historical states
    so the agent can reason: "Last time conditions were like this, X happened."
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 6333,
    ):
        self._enabled = QDRANT_AVAILABLE
        self._client: Optional[Any] = None

        if not self._enabled:
            logger.warning("Vector store disabled (qdrant-client not installed)")
            return

        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", str(port)))

        try:
            self._client = QdrantClient(host=host, port=port, timeout=10)
            self._ensure_collection()
            logger.info(f"Vector store connected: {host}:{port} | collection={COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"Qdrant not available ({e}). Vector store disabled.")
            self._enabled = False

    # ── Collection Setup ───────────────────────────────────────────────────────

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=N_FEATURES,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' ({N_FEATURES}d)")

    # ── Write ──────────────────────────────────────────────────────────────────

    def store_state(
        self,
        features: Dict,
        symbol: str = "NIFTY",
        tape_bias: str = "NEUTRAL",
        fii_bias: str = "NEUTRAL",
        smart_money: str = "NEUTRAL",
        signal: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a market state snapshot as a vector.

        Returns the point ID (UUID) so you can later record the outcome.
        """
        if not self._enabled:
            return None

        try:
            point_id = str(uuid.uuid4())
            vector = _features_to_vector(features)

            payload = {
                "timestamp":    datetime.now().isoformat(),
                "symbol":       symbol,
                "spot":         features.get("f_spot", 0),
                "pcr_oi":       features.get("f_pcr_oi", 0),
                "atm_iv":       features.get("f_atm_iv", 0),
                "vix":          features.get("f_vix", 0),
                "tape_bias":    tape_bias,
                "fii_bias":     fii_bias,
                "smart_money":  smart_money,
                "signal":       signal or "",
                "features_json": json.dumps(features),
                "outcome":      None,     # filled later via record_outcome()
            }

            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=point_id, vector=vector, payload=payload)],
            )
            logger.debug(f"VectorStore: stored state {point_id} bias={tape_bias}")
            return point_id

        except Exception as e:
            logger.error(f"VectorStore store_state error: {e}")
            return None

    def record_outcome(
        self,
        point_id: str,
        direction: str,     # "UP" / "DOWN" / "FLAT"
        move_pct: float,    # how much spot moved in next 30 min
        pnl: float = 0.0,  # P&L of trade taken (0 if no trade)
    ):
        """
        After 30 minutes, record what actually happened.
        This turns stored states into labeled training data.
        """
        if not self._enabled or not point_id:
            return
        try:
            self._client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "outcome": {
                        "direction": direction,
                        "move_pct":  round(move_pct, 4),
                        "pnl":       round(pnl, 2),
                        "recorded_at": datetime.now().isoformat(),
                    }
                },
                points=[point_id],
            )
        except Exception as e:
            logger.error(f"VectorStore record_outcome error: {e}")

    # ── Query ──────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        features: Dict,
        top_k: int = 10,
        symbol: Optional[str] = None,
        only_with_outcomes: bool = False,
    ) -> List[Dict]:
        """
        Find the K most similar historical market states to the current one.

        Returns list of dicts with:
          score       : cosine similarity (0-1, higher=more similar)
          timestamp   : when this state occurred
          tape_bias   : what the tape said then
          fii_bias    : what FII/DII said then
          signal      : what signal was generated
          outcome     : what actually happened (if recorded)
          spot        : spot price then
          pcr_oi      : PCR then
          atm_iv      : IV then
        """
        if not self._enabled:
            return []

        try:
            vector = _features_to_vector(features)

            # Build filter
            filter_conditions = []
            if symbol:
                filter_conditions.append(
                    FieldCondition(key="symbol", match=MatchValue(value=symbol))
                )

            search_filter = Filter(must=filter_conditions) if filter_conditions else None

            limit = top_k * 2 if only_with_outcomes else top_k
            # qdrant-client >=1.9 uses query_points; older used search
            try:
                qr = self._client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=limit,
                    query_filter=search_filter,
                    with_payload=True,
                )
                results = qr.points
            except AttributeError:
                results = self._client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vector,
                    limit=limit,
                    query_filter=search_filter,
                    with_payload=True,
                )

            similar = []
            for r in results:
                p = r.payload or {}
                outcome = p.get("outcome")

                if only_with_outcomes and not outcome:
                    continue

                similar.append({
                    "score":        round(r.score, 4),
                    "id":           r.id,
                    "timestamp":    p.get("timestamp", ""),
                    "spot":         p.get("spot", 0),
                    "pcr_oi":       p.get("pcr_oi", 0),
                    "atm_iv":       p.get("atm_iv", 0),
                    "vix":          p.get("vix", 0),
                    "tape_bias":    p.get("tape_bias", "NEUTRAL"),
                    "fii_bias":     p.get("fii_bias", "NEUTRAL"),
                    "smart_money":  p.get("smart_money", "NEUTRAL"),
                    "signal":       p.get("signal", ""),
                    "outcome":      outcome,
                })

            return similar[:top_k]

        except Exception as e:
            logger.error(f"VectorStore find_similar error: {e}")
            return []

    def get_outcome_stats(self, similar_states: List[Dict]) -> Dict:
        """
        Summarize outcomes from similar historical states.

        Returns:
          up_pct      : % of similar states where market went UP
          down_pct    : % where market went DOWN
          avg_move    : average % move
          avg_pnl     : average P&L when a trade was taken
          confidence  : weighted confidence based on similarity scores
          direction   : dominant direction (UP / DOWN / UNCLEAR)
        """
        with_outcomes = [s for s in similar_states if s.get("outcome")]
        if not with_outcomes:
            return {
                "up_pct": 0.5, "down_pct": 0.5,
                "avg_move": 0, "avg_pnl": 0,
                "confidence": 0, "direction": "UNCLEAR",
                "sample_size": 0,
            }

        total_score = sum(s["score"] for s in with_outcomes)
        up_score = sum(
            s["score"] for s in with_outcomes
            if s["outcome"].get("direction") == "UP"
        )
        down_score = sum(
            s["score"] for s in with_outcomes
            if s["outcome"].get("direction") == "DOWN"
        )
        avg_move = sum(
            s["score"] * s["outcome"].get("move_pct", 0)
            for s in with_outcomes
        ) / total_score if total_score else 0

        pnl_states = [s for s in with_outcomes if s["outcome"].get("pnl", 0) != 0]
        avg_pnl = (
            sum(s["outcome"]["pnl"] for s in pnl_states) / len(pnl_states)
            if pnl_states else 0
        )

        up_pct   = up_score / total_score if total_score else 0.5
        down_pct = down_score / total_score if total_score else 0.5
        conf     = min(len(with_outcomes) / 10, 1.0) * max(up_pct, down_pct)

        direction = (
            "UP"   if up_pct > 0.60
            else "DOWN" if down_pct > 0.60
            else "UNCLEAR"
        )

        return {
            "up_pct":     round(up_pct, 3),
            "down_pct":   round(down_pct, 3),
            "avg_move":   round(avg_move, 4),
            "avg_pnl":    round(avg_pnl, 2),
            "confidence": round(conf, 3),
            "direction":  direction,
            "sample_size": len(with_outcomes),
        }

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return collection stats."""
        if not self._enabled:
            return {"enabled": False}
        try:
            info = self._client.get_collection(COLLECTION_NAME)
            return {
                "enabled":        True,
                "total_states":   info.points_count,
                "vector_size":    N_FEATURES,
                "collection":     COLLECTION_NAME,
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    @property
    def is_available(self) -> bool:
        return self._enabled


# Singleton
_store: Optional[MarketVectorStore] = None


def get_vector_store() -> MarketVectorStore:
    global _store
    if _store is None:
        _store = MarketVectorStore()
    return _store
