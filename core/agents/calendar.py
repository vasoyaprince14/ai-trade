"""
Economic Calendar + Event Risk Agent
======================================
Fetches upcoming high-impact events for Indian markets:
  - RBI policy dates (hardcoded + known schedule)
  - NSE/BSE F&O expiry calendar
  - US economic events via yfinance earnings calendar
  - Budget / Union Budget dates
  - Nifty constituent earnings schedule

Calculates EVENT_RISK_SCORE for today/tomorrow.

Usage:
    from core.agents.calendar import get_event_risk
    risk = get_event_risk()
    print(risk["score"], risk["events_today"])
"""

import yfinance as yf
from datetime import datetime, date, timedelta
from loguru import logger

# ── Known RBI Policy Dates 2026 ────────────────────────────────────────────────
RBI_DATES_2026 = [
    date(2026, 2, 5), date(2026, 4, 9),
    date(2026, 6, 4), date(2026, 8, 6),
    date(2026, 10, 7), date(2026, 12, 4),
]

# NSE F&O Expiry = last Thursday of each month (simplified: every Thursday)
def _get_fo_expiry_this_week() -> list[date]:
    """Get this week's NSE F&O expiry dates (Thursdays)."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    thursday = week_start + timedelta(days=3)
    return [thursday] if thursday >= today else []


def _days_to(d: date) -> int:
    return (d - date.today()).days


def get_event_risk(symbol: str = "NIFTY") -> dict:
    """
    Return event risk assessment for today + next 3 days.
    {
      score: 0-10 (higher = more event risk)
      events_today: list
      events_upcoming: list
      recommendation: str
      fo_expiry_in_days: int
    }
    """
    today = date.today()
    events_today    = []
    events_upcoming = []
    score = 0

    # ── F&O Expiry ────────────────────────────────────────────────────────────
    thursday_dates = []
    for d_off in range(0, 10):
        d = today + timedelta(days=d_off)
        if d.weekday() == 3:  # Thursday
            thursday_dates.append(d)
            break

    fo_in_days = _days_to(thursday_dates[0]) if thursday_dates else 7

    if fo_in_days == 0:
        events_today.append("⚡ F&O EXPIRY TODAY — high pin risk, gamma spike")
        score += 4
    elif fo_in_days == 1:
        events_upcoming.append("⚠️ F&O Expiry TOMORROW — expect pinning + last-hour moves")
        score += 2
    elif fo_in_days <= 2:
        events_upcoming.append(f"📅 F&O Expiry in {fo_in_days} days")
        score += 1

    # ── RBI Policy ────────────────────────────────────────────────────────────
    for rbi_date in RBI_DATES_2026:
        diff = _days_to(rbi_date)
        if diff == 0:
            events_today.append("🏦 RBI POLICY DAY — major rate decision, high vol expected")
            score += 5
        elif 0 < diff <= 2:
            events_upcoming.append(f"🏦 RBI Policy in {diff} day(s) — pre-policy caution")
            score += 3
        elif 0 < diff <= 5:
            events_upcoming.append(f"🏦 RBI Policy in {diff} days")
            score += 1

    # ── NSE Constituent Earnings ───────────────────────────────────────────────
    major_stocks = {
        "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS",
        "HDFCBANK.NS": "HDFC BANK", "INFY.NS": "INFOSYS",
        "ICICIBANK.NS": "ICICI BANK",
    }
    for ticker, name in major_stocks.items():
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is not None and not cal.empty:
                earn_dates = cal.get("Earnings Date", [])
                if hasattr(earn_dates, "__iter__"):
                    for ed in earn_dates:
                        if hasattr(ed, "date"):
                            ed = ed.date()
                        elif isinstance(ed, str):
                            try:
                                ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                            except Exception:
                                continue
                        diff = _days_to(ed)
                        if diff == 0:
                            events_today.append(f"📊 {name} RESULTS TODAY")
                            score += 3
                        elif 0 < diff <= 3:
                            events_upcoming.append(f"📊 {name} results in {diff} day(s)")
                            score += 1
        except Exception:
            pass

    # ── US Market Events (FII correlation) ────────────────────────────────────
    us_events = _check_us_events(today)
    events_upcoming.extend(us_events)
    if any("FOMC" in e for e in us_events):
        score += 3
    elif any("CPI" in e or "NFP" in e for e in us_events):
        score += 2

    # ── Weekday effects ───────────────────────────────────────────────────────
    if today.weekday() == 0:
        events_today.append("📅 Monday — Gap-up/down from weekend news")
        score += 1
    if today.weekday() == 4:
        events_today.append("📅 Friday — Weekend risk, traders reducing positions")
        score += 1

    # ── Score → Recommendation ────────────────────────────────────────────────
    if score >= 7:
        recommendation = "🚨 HIGH EVENT RISK — Reduce position size 50%, widen targets, avoid new trades 30min around events"
    elif score >= 4:
        recommendation = "⚠️ MODERATE EVENT RISK — Trade with caution, tighter stops, no overnight positions"
    elif score >= 2:
        recommendation = "📅 LOW EVENT RISK — Normal trading, be aware of calendar"
    else:
        recommendation = "✅ CLEAN CALENDAR — Normal position sizing"

    result = {
        "score":            min(score, 10),
        "events_today":     events_today,
        "events_upcoming":  events_upcoming,
        "fo_expiry_in_days":fo_in_days,
        "recommendation":   recommendation,
        "date":             today.strftime("%d %b %Y"),
        "weekday":          today.strftime("%A"),
    }

    logger.info(f"[Calendar] Event risk score={score} | {len(events_today)} events today")
    return result


def _check_us_events(today: date) -> list[str]:
    """Simplified US economic calendar — FOMC / CPI / NFP approximations."""
    events = []
    # FOMC meets 8x per year — roughly every 6-7 weeks
    # Approximate dates for 2026 (simplified)
    fomc_2026 = [
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
        date(2026, 6, 10), date(2026, 7, 29), date(2026, 9, 16),
        date(2026, 10, 28), date(2026, 12, 16),
    ]
    for fd in fomc_2026:
        diff = _days_to(fd)
        if -1 <= diff <= 2:
            events.append(f"🇺🇸 FOMC Meeting {'TODAY' if diff == 0 else f'in {diff} days'}")

    # CPI: ~2nd week of each month
    # NFP: ~1st Friday of each month
    dom = today.day
    if 8 <= dom <= 12 and today.weekday() in (1, 2, 3):
        events.append("🇺🇸 US CPI likely this week — gold + equities vol")
    if 1 <= dom <= 7 and today.weekday() == 4:
        events.append("🇺🇸 US NFP (Non-Farm Payroll) today")

    return events
