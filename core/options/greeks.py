"""
Options Greeks Calculator
==========================
Computes Delta, Gamma, Theta, Vega, IV for NSE options.

Uses py_vollib (Black-Scholes-Merton) with fallback to mibian.
All inputs in NSE standard units.

Usage:
    from core.options.greeks import compute_greeks, greeks_summary
    g = compute_greeks(spot=24300, strike=24300, expiry_days=7,
                       option_type='c', premium=85, risk_free=0.065)
"""

from datetime import date, datetime
from typing import Optional, Dict
from loguru import logger

RISK_FREE_RATE = 0.065   # RBI repo rate proxy


def _days_to_expiry(expiry_str: str) -> int:
    """Parse NSE expiry string (e.g. '28-Aug-2026') → days remaining."""
    if not expiry_str:
        return 7
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            exp = datetime.strptime(expiry_str, fmt).date()
            return max(1, (exp - date.today()).days)
        except ValueError:
            continue
    return 7


def compute_greeks(
    spot:        float,
    strike:      float,
    expiry_days: int,
    option_type: str,        # 'c' or 'p'
    premium:     float,
    risk_free:   float = RISK_FREE_RATE,
    volatility:  Optional[float] = None,   # annualized decimal (0.15 = 15%). If None, back-solve IV.
) -> Dict:
    """
    Compute Delta, Gamma, Theta, Vega, IV for an NSE option.

    Returns dict:
        delta, gamma, theta, vega, iv, intrinsic, time_value
        theta_per_day, breakeven, moneyness
    """
    if spot <= 0 or strike <= 0 or premium <= 0:
        return _empty_greeks()

    t = max(expiry_days, 1) / 365.0
    opt = option_type.lower()[0]   # 'c' or 'p'

    # Try py_vollib first (most accurate)
    try:
        from py_vollib.black_scholes_merton import black_scholes_merton as bsm
        from py_vollib.black_scholes_merton.greeks import analytical
        from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

        # IV: back-solve from market premium
        if volatility is None:
            try:
                iv = implied_volatility(premium, spot, strike, t, risk_free, 0.0, opt)
            except Exception:
                iv = 0.15  # fallback
        else:
            iv = volatility

        iv = max(0.01, min(iv, 5.0))   # clamp to [1%, 500%]

        delta = analytical.delta(opt, spot, strike, t, risk_free, iv, 0.0)
        gamma = analytical.gamma(opt, spot, strike, t, risk_free, iv, 0.0)
        theta = analytical.theta(opt, spot, strike, t, risk_free, iv, 0.0)
        vega  = analytical.vega( opt, spot, strike, t, risk_free, iv, 0.0)

    except Exception as e:
        logger.debug(f"py_vollib failed ({e}), falling back to mibian")
        # Fallback: mibian (simpler)
        try:
            import mibian
            if volatility is None:
                c = mibian.BS([spot, strike, risk_free * 100, expiry_days], callPrice=premium if opt == 'c' else None,
                              putPrice=premium if opt == 'p' else None)
                iv = (c.impliedVolatility or 15) / 100
            else:
                iv = volatility
            c = mibian.BS([spot, strike, risk_free * 100, expiry_days], volatility=iv * 100)
            delta = c.callDelta if opt == 'c' else c.putDelta
            gamma = c.gamma
            theta = (c.callTheta if opt == 'c' else c.putTheta) / 365
            vega  = c.vega / 100
        except Exception as e2:
            logger.debug(f"mibian also failed: {e2}")
            return _empty_greeks()

    # Derived metrics
    intrinsic   = max(0, (spot - strike) if opt == 'c' else (strike - spot))
    time_value  = max(0, premium - intrinsic)
    breakeven   = (strike + premium) if opt == 'c' else (strike - premium)
    moneyness   = spot / strike

    return {
        "delta":          round(float(delta), 4),
        "gamma":          round(float(gamma), 6),
        "theta":          round(float(theta), 4),      # per year
        "theta_per_day":  round(float(theta) / 365, 4),
        "vega":           round(float(vega),  4),
        "iv":             round(float(iv) * 100, 2),   # in percent
        "intrinsic":      round(intrinsic, 2),
        "time_value":     round(time_value, 2),
        "breakeven":      round(breakeven, 2),
        "moneyness":      round(moneyness, 4),
        "expiry_days":    expiry_days,
    }


def _empty_greeks() -> Dict:
    return {
        "delta": 0, "gamma": 0, "theta": 0, "theta_per_day": 0,
        "vega": 0, "iv": 0, "intrinsic": 0, "time_value": 0,
        "breakeven": 0, "moneyness": 0, "expiry_days": 0,
    }


def greeks_for_decision(decision) -> Dict:
    """
    Compute Greeks for a TradeDecision object.
    Uses entry_high as the premium reference price.
    """
    if not decision or not decision.is_trade():
        return _empty_greeks()

    spot    = decision.sl_spot   # we don't store spot directly, use sl_spot as proxy
    strike  = decision.strike
    premium = decision.entry_high
    expiry  = decision.expiry
    opt     = 'c' if 'CE' in decision.action else 'p'

    # Reconstruct spot from sl_spot (SL is spot ± 150)
    if 'CE' in decision.action:
        spot = decision.sl_spot + 150
    elif 'PE' in decision.action:
        spot = decision.sl_spot - 150
    else:
        spot = strike

    if spot <= 0:
        spot = strike

    days = _days_to_expiry(expiry)
    return compute_greeks(spot, strike, days, opt, premium)


def greeks_summary(g: Dict) -> str:
    """One-line summary of Greeks for signal card."""
    if not g or g.get("iv", 0) == 0:
        return "Greeks: N/A"
    return (
        f"Δ={g['delta']:+.3f}  Γ={g['gamma']:.5f}  "
        f"Θ={g['theta_per_day']:+.2f}/day  Vega={g['vega']:.3f}  "
        f"IV={g['iv']:.1f}%  Break-even={g['breakeven']:.0f}"
    )
