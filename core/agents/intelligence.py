"""
Market Intelligence Hub
========================
Single entry point that orchestrates all agents and returns
a comprehensive market picture in one call.

Usage:
    from core.agents.intelligence import get_market_intelligence
    mi = get_market_intelligence(tracker, symbol="NIFTY")
"""

from loguru import logger
from datetime import datetime


def get_market_intelligence(tracker, symbol: str = "NIFTY") -> dict:
    """
    Runs all intelligence agents and returns a combined dict:
    {
      regime, breadth, sector, gex, iv_surface, event_risk,
      news_sentiment, timestamp
    }
    """
    result = {"symbol": symbol, "timestamp": datetime.now().strftime("%H:%M:%S")}

    # Market summary + features
    try:
        summary  = tracker.get_market_summary()
        features = tracker.get_model_features()
        df_chain = summary.get("df")
        spot     = summary.get("spot", 24000)
        result["spot"] = spot
    except Exception as e:
        logger.warning(f"Intelligence: tracker error: {e}")
        return result

    # ── GEX ──────────────────────────────────────────────────────────────────
    try:
        from core.options.gex import compute_gex
        expiry_str = summary.get("expiry", "")
        try:
            from datetime import datetime as dt
            exp_date = dt.strptime(expiry_str, "%d-%m-%Y").date()
            expiry_days = max(1, (exp_date - dt.now().date()).days)
        except Exception:
            expiry_days = 1
        result["gex"] = compute_gex(df_chain, spot, expiry_days)
    except Exception as e:
        logger.debug(f"GEX error: {e}")
        result["gex"] = {}

    # ── IV Surface ────────────────────────────────────────────────────────────
    try:
        from core.options.iv_surface import get_iv_surface
        ivs = get_iv_surface()
        result["iv_surface"] = ivs.full_summary(df_chain, spot)
    except Exception as e:
        logger.debug(f"IV surface error: {e}")
        result["iv_surface"] = {}

    # ── Regime ────────────────────────────────────────────────────────────────
    try:
        from core.agents.regime import detect_regime
        iv_rank = result.get("iv_surface", {}).get("iv_rank", {}).get("iv_rank", 50)
        vix     = features.get("f_vix", 15)
        result["regime"] = detect_regime(features, iv_rank=iv_rank, vix=vix)
    except Exception as e:
        logger.debug(f"Regime error: {e}")
        result["regime"] = {}

    # ── Event Risk ────────────────────────────────────────────────────────────
    try:
        from core.agents.calendar import get_event_risk
        result["event_risk"] = get_event_risk(symbol)
    except Exception as e:
        logger.debug(f"Event risk error: {e}")
        result["event_risk"] = {}

    # ── Breadth (slow — only if requested) ───────────────────────────────────
    # NOTE: breadth and sector take ~30s on first run; cached thereafter
    # These are fetched lazily in the dashboard

    logger.info(
        f"[Intelligence] {symbol} | "
        f"Regime={result.get('regime', {}).get('regime', 'N/A')} | "
        f"GEX={result.get('gex', {}).get('net_total_gex', 0):.2f}B | "
        f"EventRisk={result.get('event_risk', {}).get('score', 0)}"
    )
    return result
