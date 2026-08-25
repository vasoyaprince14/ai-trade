"""
Telegram Signal Bot
===================
Sends live trade signals to your Telegram chat.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → get BOT_TOKEN
  2. Message your bot once, then run:
       python3 -c "from core.alerts.telegram_bot import get_chat_id; get_chat_id()"
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=your_token
       TELEGRAM_CHAT_ID=your_chat_id

What it sends:
  - Every new BUY_CE / BUY_PE / SELL_STRADDLE signal
  - Greeks (Delta, Theta, IV)
  - FII positioning summary
  - EOD retrain completion
"""

import os
import asyncio
from datetime import datetime
from typing import Optional, Dict
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


def _is_configured() -> bool:
    return bool(BOT_TOKEN and CHAT_ID)


def get_chat_id():
    """Helper: print recent updates to find your chat_id."""
    if not BOT_TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env first")
        return
    import requests
    r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates", timeout=10)
    updates = r.json().get("result", [])
    if not updates:
        print("No updates. Send your bot a message first, then run this again.")
        return
    for u in updates[-5:]:
        msg = u.get("message", {})
        chat = msg.get("chat", {})
        print(f"Chat ID: {chat.get('id')}  |  From: {chat.get('first_name', '')} {chat.get('username','')}")


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message synchronously. Returns True on success."""
    if not _is_configured():
        logger.debug("Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")
        return False
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        if r.status_code == 200:
            logger.debug("Telegram message sent")
            return True
        logger.warning(f"Telegram API error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
    return False


def send_signal(decision, greeks: Optional[Dict] = None, fii_summary: Optional[Dict] = None) -> bool:
    """
    Send a formatted signal card to Telegram.
    Only sends actionable signals (not WAIT).
    """
    if not decision or not decision.is_trade():
        return False

    ts  = datetime.now().strftime("%d %b %Y  %H:%M")
    act = decision.action
    emoji = {
        "BUY_CE":       "🟢",
        "BUY_PE":       "🔴",
        "SELL_STRADDLE":"🟡",
        "SELL_CE":      "🟠",
        "SELL_PE":      "🟠",
    }.get(act, "⚪")

    lines = [
        f"{emoji} <b>{act}</b>  |  {decision.symbol} {decision.strike}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📅 <b>Expiry:</b> {decision.expiry or 'N/A'}",
        f"💰 <b>Entry:</b>  ₹{decision.entry_low:.0f} – {decision.entry_high:.0f}",
        f"🛑 <b>SL:</b>     ₹{decision.stop_loss:.0f}  (Nifty {'below' if 'CE' in act else 'above'} {decision.sl_spot:.0f})",
        f"🎯 <b>Target:</b> ₹{decision.target:.0f}  (Nifty at {decision.target_spot:.0f})",
        f"📊 <b>R:R:</b>    1:{decision.risk_reward:.1f}",
        f"🤖 <b>Conf:</b>   {decision.confidence:.0%}",
    ]

    # Greeks
    if greeks and greeks.get("iv", 0) > 0:
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"<b>Greeks:</b>  Δ={greeks['delta']:+.3f}  Θ={greeks['theta_per_day']:+.2f}/d  "
            f"IV={greeks['iv']:.1f}%  B/E={greeks['breakeven']:.0f}",
        ]

    # FII summary
    if fii_summary:
        fii_bias = fii_summary.get("fno", {}).get("fii_bias", "N/A")
        smart    = fii_summary.get("smart_money_bias", "N/A")
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"🏦 <b>FII F&O:</b>  {fii_bias}",
            f"💵 <b>Smart $:</b>  {smart}",
        ]

    lines += [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🧠 <i>{decision.reasoning[:250]}</i>",
        f"\n⏰ {ts}",
    ]

    return send_message("\n".join(lines))


def send_eod_summary(symbol: str, model_info: Dict, vector_stats: Dict) -> bool:
    """Send end-of-day model retrain summary."""
    msg = (
        f"📈 <b>EOD Summary — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔄 Model retrained: {model_info.get('train_samples', 0):,} samples\n"
        f"🧠 Trained at: {str(model_info.get('trained_at',''))[:19]}\n"
        f"💾 Vector memory: {vector_stats.get('total_states', 0)} states\n"
        f"⏰ {datetime.now().strftime('%d %b %Y  %H:%M')}"
    )
    return send_message(msg)


def send_wait_update(reason: str, symbol: str = "NIFTY") -> bool:
    """Periodic WAIT update (less frequent, just a brief note)."""
    msg = (
        f"⏳ <b>WAIT — {symbol}</b>\n"
        f"{reason[:200]}\n"
        f"⏰ {datetime.now().strftime('%H:%M')}"
    )
    return send_message(msg)
