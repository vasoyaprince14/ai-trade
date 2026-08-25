"""
AI-Trade Dashboard
===================
Streamlit-based real-time trading dashboard.

Features:
  - Live Nifty/BankNifty quotes with chart
  - Order Flow: PCR, OI heatmap, Max Pain
  - Active signals from all strategies
  - Open positions & P&L
  - Trade history
  - Backtest runner
  - Market regime indicator

Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ai_path = str(ROOT / "vendors" / "ai-trader")
if _ai_path not in sys.path:
    sys.path.append(_ai_path)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

from core.data.nse_scraper import get_scraper
from core.data.historical import fetch_historical, add_indicators
from core.data.db import init_db, DBManager
from config.settings import config

# Cache OITracker per symbol (expensive to re-create)
@st.cache_resource
def get_oi_tracker(sym: str):
    from core.order_flow.oi_tracker import OITracker
    return OITracker(sym)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Trade | Nifty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize DB
init_db()

# Auto-refresh every 60 seconds during market hours
refresh_count = st_autorefresh(
    interval=config.get("dashboard", {}).get("auto_refresh_seconds", 60) * 1000,
    limit=None,
    key="auto_refresh",
)

# ── Dark Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .signal-bullish { color: #00e676; font-weight: bold; }
    .signal-bearish { color: #ff5252; font-weight: bold; }
    .signal-neutral { color: #ffd740; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_quote(symbol: str) -> dict:
    scraper = get_scraper()
    sym_map = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE"}
    return scraper.get_index_quote(sym_map.get(symbol, symbol)) or {}


@st.cache_data(ttl=60)
def get_pcr(symbol: str) -> dict:
    scraper = get_scraper()
    return scraper.get_pcr_data(symbol) or {}


@st.cache_data(ttl=60)
def get_option_chain(symbol: str) -> pd.DataFrame:
    scraper = get_scraper()
    records = scraper.parse_option_chain(symbol, strikes_range=15)
    return pd.DataFrame(records) if records else pd.DataFrame()


@st.cache_data(ttl=300)
def get_historical_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    df = fetch_historical(symbol, timeframe, days)
    return add_indicators(df) if not df.empty else df


@st.cache_data(ttl=60)
def get_vix() -> float:
    return get_scraper().get_vix() or 0


@st.cache_data(ttl=60)
def get_max_pain(symbol: str) -> float:
    return get_scraper().get_max_pain(symbol) or 0


def _show_decision(decision: dict):
    """Render a TradeDecision dict as a dashboard card."""
    action = decision.get("action", "WAIT")
    conf   = decision.get("confidence", 0)
    action_color = {
        "BUY_CE": "#00e676", "BUY_PE": "#ff5252",
        "SELL_STRADDLE": "#b388ff", "SELL_CE": "#ffab40", "SELL_PE": "#ffab40",
        "WAIT": "#ffd740", "EXIT": "#90a4ae",
    }.get(action, "#ffd740")
    st.markdown(
        f"<div style='background:#1a1d2e;border-left:6px solid {action_color};"
        f"padding:20px;border-radius:8px;margin:12px 0'>"
        f"<h2 style='color:{action_color};margin:0'>{action}</h2>"
        f"<p style='color:#aaa;margin:4px 0'>Confidence: {conf:.0%} | "
        f"Symbol: {decision.get('symbol')} | "
        f"Strike: {decision.get('strike') or 'N/A'} | "
        f"Expiry: {decision.get('expiry') or 'N/A'}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if action not in ("WAIT", "EXIT", ""):
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Entry Range", f"₹{decision.get('entry_low',0):.0f}–{decision.get('entry_high',0):.0f}")
        mc2.metric("SL (Premium)",   f"₹{decision.get('stop_loss',0):.1f}")
        mc3.metric("Target (Premium)",f"₹{decision.get('target',0):.1f}")
        mc4.metric("SL (Spot)",       f"{decision.get('sl_spot',0):.0f}")
        mc5.metric("Target (Spot)",   f"{decision.get('target_spot',0):.0f}")
        rr  = decision.get("risk_reward", 0)
        qty = decision.get("qty_lots", 1)
        st.info(f"Risk:Reward = **{rr:.1f}x** | Qty: **{qty} lot(s)**")
    st.markdown("**Reasoning:**")
    st.markdown(
        f"<div style='background:#0e1117;padding:12px;border-radius:6px;"
        f"border:1px solid #2d3748;color:#ccc'>{decision.get('reasoning','')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Decision at: {decision.get('timestamp','')[:19]}")


def color_signal(signal: str) -> str:
    colors = {
        "BULLISH": "#00e676", "LONG_BUILD_UP": "#00e676",
        "SHORT_COVERING": "#69f0ae", "SLIGHTLY_BULLISH": "#b9f6ca",
        "BEARISH": "#ff5252", "SHORT_BUILD_UP": "#ff5252",
        "LONG_UNWINDING": "#ff867f", "SLIGHTLY_BEARISH": "#ffcdd2",
        "NEUTRAL": "#ffd740", "GAMMA_PINNING": "#b388ff",
        "STABLE": "#e0e0e0",
    }
    return colors.get(signal, "#e0e0e0")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI-Trade")
    st.markdown("---")

    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
    timeframe = st.selectbox("Chart Timeframe", ["5min", "15min", "1h", "1d"], index=3)
    days_chart = st.slider("Chart History (days)", 7, 365, 90)

    st.markdown("---")
    broker_mode = config.get("brokers", {}).get("default", "paper")
    st.info(f"Broker: **{broker_mode.upper()}**")
    market_open = get_scraper().is_market_open()
    if market_open:
        st.success("Market: OPEN")
    else:
        st.warning("Market: CLOSED")

    last_refresh = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Last refresh: {last_refresh}")

    # Telegram status
    st.markdown("---")
    from core.alerts.telegram_bot import _is_configured as _tg_ok
    if _tg_ok():
        st.success("Telegram: Connected")
    else:
        st.caption("Telegram: Not configured")
        with st.expander("Setup Telegram alerts"):
            st.markdown("""
1. Message [@BotFather](https://t.me/BotFather) → `/newbot`
2. Copy the token
3. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```
4. Get chat ID: `python3 -c "from core.alerts.telegram_bot import get_chat_id; get_chat_id()"`
""")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "OI Pulse", "Order Flow", "AI Agent", "ML Predictions", "Strategies & Signals", "Positions & P&L", "Backtest", "Option Chain"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title(f"Market Overview | {symbol}")

    # ── Live Signal Card (top of overview) ────────────────────────────
    try:
        from core.agent.ml_agent import get_ml_agent
        _tracker = get_oi_tracker(symbol)
        _agent   = get_ml_agent(symbol)
        _dec     = _agent.decide(_tracker, symbol=symbol)
        action   = _dec.action
        _acolor  = {"BUY_CE": "#00e676", "BUY_PE": "#ff5252",
                    "SELL_STRADDLE": "#b388ff", "SELL_CE": "#ffab40",
                    "SELL_PE": "#ffab40", "WAIT": "#ffd740"}.get(action, "#ffd740")
        _emoji   = {"BUY_CE": "🟢", "BUY_PE": "🔴", "SELL_STRADDLE": "🟡",
                    "SELL_CE": "🟠", "SELL_PE": "🟠", "WAIT": "⏳"}.get(action, "⚪")
        st.markdown(
            f"<div style='background:#1a1d2e;border-left:8px solid {_acolor};"
            f"padding:18px 24px;border-radius:8px;margin-bottom:16px'>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{_acolor}'>"
            f"{_emoji} {action}</span>"
            + (
                f"&nbsp;&nbsp;<span style='color:#aaa'>|&nbsp; Strike: <b style='color:#fff'>"
                f"{_dec.strike}</b> &nbsp;|&nbsp; Entry: <b style='color:#fff'>"
                f"₹{_dec.entry_low:.0f}–{_dec.entry_high:.0f}</b> &nbsp;|&nbsp; "
                f"SL: <b style='color:#ff5252'>₹{_dec.stop_loss:.0f}</b> &nbsp;|&nbsp; "
                f"Target: <b style='color:#00e676'>₹{_dec.target:.0f}</b> &nbsp;|&nbsp; "
                f"R:R: <b style='color:#fff'>1:{_dec.risk_reward:.1f}</b> &nbsp;|&nbsp; "
                f"Conf: <b style='color:#fff'>{_dec.confidence:.0%}</b></span>"
                if _dec.is_trade() else
                f"&nbsp;&nbsp;<span style='color:#aaa'>{_dec.reasoning[:160]}</span>"
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    except Exception as _e:
        st.warning(f"Signal unavailable: {_e}")

    # Quote row
    col1, col2, col3, col4, col5 = st.columns(5)

    quote = get_quote(symbol)
    vix = get_vix()
    pcr_data = get_pcr(symbol)
    max_pain = get_max_pain(symbol)

    with col1:
        ltp = quote.get("ltp", 0)
        chg = quote.get("change_pct", 0)
        delta_color = "normal" if abs(chg) < 0.1 else ("normal" if chg > 0 else "inverse")
        st.metric(f"{symbol} LTP", f"₹{ltp:,.2f}", f"{chg:+.2f}%",
                  delta_color="normal" if chg >= 0 else "inverse")

    with col2:
        st.metric("India VIX", f"{vix:.2f}", help="Fear gauge - higher = more volatile")

    with col3:
        pcr = pcr_data.get("pcr_oi", 0)
        st.metric("PCR (OI)", f"{pcr:.3f}",
                  help="Put-Call Ratio. >1.3 Bullish, <0.7 Bearish")

    with col4:
        st.metric("Max Pain", f"{max_pain:.0f}" if max_pain else "N/A",
                  help="Strike where option buyers lose most money at expiry")

    with col5:
        open_pos = 0
        try:
            with DBManager() as db:
                open_pos = len(db.get_open_trades())
        except Exception:
            pass
        st.metric("Open Trades", open_pos)

    # Main chart
    st.markdown("---")
    st.subheader(f"{symbol} Price Chart ({timeframe})")
    hist = get_historical_data(symbol, timeframe, days_chart)

    if not hist.empty:
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist["timestamp"],
            open=hist["open"],
            high=hist["high"],
            low=hist["low"],
            close=hist["close"],
            name=symbol,
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252",
        ))

        # EMAs
        for ema_col, color, name in [
            ("ema9", "#ffd740", "EMA9"),
            ("ema21", "#40c4ff", "EMA21"),
            ("ema50", "#e040fb", "EMA50"),
        ]:
            if ema_col in hist.columns:
                fig.add_trace(go.Scatter(
                    x=hist["timestamp"], y=hist[ema_col],
                    mode="lines", name=name,
                    line=dict(color=color, width=1),
                ))

        fig.update_layout(
            height=500,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d2e",
            font=dict(color="#e0e0e0"),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="#1a1d2e"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume
        if "volume" in hist.columns and hist["volume"].sum() > 0:
            fig_vol = go.Figure(go.Bar(
                x=hist["timestamp"], y=hist["volume"],
                marker_color=["#00e676" if c >= o else "#ff5252"
                               for c, o in zip(hist["close"], hist["open"])],
                name="Volume",
            ))
            fig_vol.update_layout(height=120, paper_bgcolor="#0e1117",
                                   plot_bgcolor="#1a1d2e", font=dict(color="#e0e0e0"))
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning(f"Could not load {symbol} price data")

    # Recent signals
    st.markdown("---")
    st.subheader("Recent Signals")
    try:
        with DBManager() as db:
            signals = db.get_recent_signals(20)
        if signals:
            sig_df = pd.DataFrame([{
                "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Strategy": s.strategy,
                "Symbol": s.symbol,
                "Direction": s.direction,
                "Type": s.signal_type,
                "Price": f"₹{s.price:.2f}" if s.price else "-",
                "Executed": "✓" if s.executed else "○",
                "Notes": (s.notes or "")[:60],
            } for s in signals])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signals yet. Start the trading engine: `python main.py live`")
    except Exception as e:
        st.error(f"DB error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: OI Pulse — Real-Time Institutional Tape Reader
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "OI Pulse":
    st.title(f"OI Pulse — Institutional Tape | {symbol}")
    st.caption("Tracks what FII / DII / Big Players are doing in real-time. Updates on every dashboard refresh.")

    tracker = get_oi_tracker(symbol)

    # Poll tape (detects new institutional events since last refresh)
    with st.spinner("Reading tape..."):
        try:
            new_events = tracker.tick_tape()
        except Exception as e:
            new_events = []
            st.warning(f"Tape poll error: {e}")

        try:
            tape_summary  = tracker.tape_reader.get_flow_summary()
            participant   = tracker._participant.get_full_picture()
        except Exception as e:
            tape_summary  = {}
            participant   = {}
            st.warning(f"Data fetch error: {e}")

    fno_data = participant.get("fno", {})
    cash_data = participant.get("cash", {}) or {}
    participants = fno_data.get("participants", {})
    fii = participants.get("FII", {})
    dii = participants.get("DII", {})
    pro = participants.get("PRO", {})

    # ── Top bar: combined bias ─────────────────────────────────────────────
    bias      = tape_summary.get("bias", "NEUTRAL")
    sm_bias   = participant.get("smart_money_bias", "NEUTRAL")
    bias_color = {"STRONGLY_BULLISH": "#00c853", "BULLISH": "#00e676",
                  "NEUTRAL": "#ffd740", "BEARISH": "#ff5252", "STRONGLY_BEARISH": "#b71c1c"}.get(bias, "#ffd740")
    sm_color  = {"STRONGLY_BULLISH": "#00c853", "BULLISH": "#00e676",
                 "NEUTRAL": "#ffd740", "BEARISH": "#ff5252", "STRONGLY_BEARISH": "#b71c1c"}.get(sm_bias, "#ffd740")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"**Tape Bias**")
        st.markdown(f"<h3 style='color:{bias_color}'>{bias}</h3>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Smart Money**")
        st.markdown(f"<h3 style='color:{sm_color}'>{sm_bias}</h3>", unsafe_allow_html=True)
    with c3:
        bull_oi = tape_summary.get("bullish_oi", 0)
        bear_oi = tape_summary.get("bearish_oi", 0)
        st.metric("Bullish OI", f"{bull_oi:,}", help="Total contracts in bullish tape events")
    with c4:
        st.metric("Bearish OI", f"{bear_oi:,}", help="Total contracts in bearish tape events")
    with c5:
        st.metric("Events Today", tape_summary.get("total_events", 0))

    st.markdown("---")

    # ── FII / DII Positioning ─────────────────────────────────────────────
    st.subheader("FII / DII F&O Positioning (EOD)")
    data_date = fno_data.get("data_date", "latest available")
    st.caption(f"Source: NSE participant OI | Date: {data_date}")

    pc1, pc2, pc3 = st.columns(3)
    for col, name, pdata in [(pc1, "FII", fii), (pc2, "DII", dii), (pc3, "PRO", pro)]:
        with col:
            p_bias = pdata.get("bias", "N/A")
            p_color = {"STRONGLY_BULLISH": "#00c853", "BULLISH": "#00e676",
                       "NEUTRAL": "#ffd740", "BEARISH": "#ff5252",
                       "STRONGLY_BEARISH": "#b71c1c"}.get(p_bias, "#aaa")
            st.markdown(
                f"<div style='background:#1a1d2e;padding:12px;border-radius:8px;"
                f"border-left:4px solid {p_color}'>"
                f"<b style='font-size:16px'>{name}</b> "
                f"<span style='color:{p_color};float:right'>{p_bias}</span><br>"
                f"<small>Net Futures: <b>{pdata.get('net_futures', 0):+,}</b></small><br>"
                f"<small>Net Calls:   <b>{pdata.get('net_calls',   0):+,}</b></small><br>"
                f"<small>Net Puts:    <b>{pdata.get('net_puts',    0):+,}</b></small><br>"
                f"<small>Call Long: {pdata.get('call_long',0):,} | Short: {pdata.get('call_short',0):,}</small><br>"
                f"<small>Put Long:  {pdata.get('put_long', 0):,} | Short: {pdata.get('put_short', 0):,}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Cash market flows
    if cash_data:
        st.markdown("#### Cash Market Flows (Today)")
        cc1, cc2, cc3 = st.columns(3)
        fii_cash = cash_data.get("FII", {})
        dii_cash = cash_data.get("DII", {})
        cc1.metric("FII Net (₹Cr)", f"{fii_cash.get('net_cr', 0):+,.0f}")
        cc2.metric("DII Net (₹Cr)", f"{dii_cash.get('net_cr', 0):+,.0f}")
        combined_net = cash_data.get("combined_net_cr", 0)
        cc3.metric("Combined Net (₹Cr)", f"{combined_net:+,.0f}",
                   delta_color="normal" if combined_net >= 0 else "inverse")

    st.markdown("---")

    # ── Big Player Positions ──────────────────────────────────────────────
    st.subheader("Big Player Positions (from Tape)")
    st.caption("Strikes where institutional OI > 50,000 contracts detected this session")
    big_players = tape_summary.get("big_players", [])

    if big_players:
        bp_rows = []
        for bp in big_players:
            impact = bp.get("market_impact", "?")
            imp_color = "#00e676" if impact == "BULLISH" else "#ff5252" if impact == "BEARISH" else "#ffd740"
            bp_rows.append({
                "Strike":         int(bp["strike"]),
                "Type":           bp["option_type"],
                "Side":           bp.get("side", "?"),
                "Impact":         impact,
                "Net OI":         f"{bp['net_oi']:,}",
                "Avg Fill":       f"₹{bp.get('avg_fill_price', 0):.1f}",
                "Inferred SL":    f"₹{bp.get('inferred_sl', 0):.1f}",
                "SL (Spot)":      f"{bp.get('sl_spot', 0):.0f}",
                "Inferred Tgt":   f"₹{bp.get('inferred_target', 0):.1f}",
                "Tgt (Spot)":     f"{bp.get('target_spot', 0):.0f}",
                "Last Event":     bp.get("last_event", ""),
                "Last Seen":      bp.get("last_time", ""),
            })
        st.dataframe(pd.DataFrame(bp_rows), use_container_width=True, hide_index=True)

        # Key levels chart
        spot = tracker._last_spot or 0
        if spot > 0:
            st.markdown("#### Key Levels (Big Player SL & Target Zones)")
            key = tape_summary.get("key_levels", {})
            sl_spots     = [s for s in key.get("sl_spot_levels", []) if s > 0]
            target_spots = [s for s in key.get("target_spot_levels", []) if s > 0]
            entry_strikes = key.get("entry_strikes", [])

            fig_levels = go.Figure()
            fig_levels.add_hline(y=spot, line_color="#ffd740", line_dash="dash",
                                  annotation_text=f"Spot {spot:.0f}", annotation_position="right")
            for sl in sl_spots[:3]:
                fig_levels.add_hline(y=sl, line_color="#ff5252", line_width=1,
                                      annotation_text=f"SL {sl:.0f}", annotation_position="right")
            for tgt in target_spots[:3]:
                fig_levels.add_hline(y=tgt, line_color="#00e676", line_width=1,
                                      annotation_text=f"Tgt {tgt:.0f}", annotation_position="right")
            if entry_strikes:
                for es in entry_strikes[:5]:
                    fig_levels.add_hline(y=es, line_color="#b388ff", line_width=1, line_dash="dot",
                                          annotation_text=f"Entry {es:.0f}", annotation_position="left")

            y_range = [spot - 300, spot + 300]
            fig_levels.update_layout(
                height=350, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                font=dict(color="#e0e0e0"),
                yaxis=dict(range=y_range),
                xaxis=dict(showticklabels=False),
                title="Key Spot Levels inferred from Big Player positions",
                showlegend=False,
            )
            st.plotly_chart(fig_levels, use_container_width=True)
    else:
        st.info("No big player events detected yet this session. Tape updates every refresh cycle.")

    st.markdown("---")

    # ── Hot Strikes ────────────────────────────────────────────────────────
    hot = tape_summary.get("hot_strikes", [])
    if hot:
        st.subheader("Hot Strikes (Most Active)")
        hcols = st.columns(min(len(hot), 5))
        for col, h in zip(hcols, hot):
            impact = h.get("last_impact", "NEUTRAL")
            hcolor = "#00e676" if impact == "BULLISH" else "#ff5252" if impact == "BEARISH" else "#ffd740"
            with col:
                st.markdown(
                    f"<div style='background:#1a1d2e;padding:10px;border-radius:6px;"
                    f"border-top:3px solid {hcolor};text-align:center'>"
                    f"<b style='font-size:18px'>{int(h['strike'])}{h['type']}</b><br>"
                    f"<small style='color:{hcolor}'>{impact}</small><br>"
                    f"<small>{h['event_count']} events</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Live Tape Feed ─────────────────────────────────────────────────────
    st.subheader("Live Tape Feed")
    st.caption("Each row = institutional event (OI > 50K contracts at one strike). Color = market impact.")
    tape_events = tracker.tape_reader.get_tape(30)

    if tape_events:
        tape_df = pd.DataFrame(tape_events)
        # Rename for display
        tape_df = tape_df.rename(columns={
            "time": "Time", "strike": "Strike", "type": "Type",
            "event": "Event", "market_impact": "Impact",
            "oi_change": "OI Change", "fill_price": "Fill ₹",
            "underlying": "Spot", "inferred_sl": "SL ₹",
            "sl_spot": "SL Spot", "inferred_target": "Tgt ₹",
            "target_spot": "Tgt Spot", "confidence": "Conf",
            "vol_ratio": "Vol×",
        })

        def _color_impact(val):
            if val == "BULLISH":
                return "color: #00e676"
            elif val == "BEARISH":
                return "color: #ff5252"
            return "color: #ffd740"

        styled = tape_df.style.applymap(_color_impact, subset=["Impact"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Tape is empty. Events appear when large OI changes (>50K contracts) "
            "are detected at a single strike between two consecutive snapshots.\n\n"
            "This requires at least 2 option chain polls. Refresh again in ~60 seconds."
        )

    # ── Bullish vs Bearish OI bar ──────────────────────────────────────────
    if tape_summary.get("total_events", 0) > 0:
        st.markdown("---")
        bull_pct = tape_summary.get("bull_pct", 0.5)
        bear_pct = 1 - bull_pct
        fig_bias = go.Figure()
        fig_bias.add_trace(go.Bar(
            name="Bullish OI", x=["OI Bias"],
            y=[bull_pct * 100], marker_color="#00e676",
        ))
        fig_bias.add_trace(go.Bar(
            name="Bearish OI", x=["OI Bias"],
            y=[bear_pct * 100], marker_color="#ff5252",
        ))
        fig_bias.update_layout(
            barmode="stack", height=160,
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
            font=dict(color="#e0e0e0"),
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            legend=dict(orientation="h"),
            title="Institutional OI Bias (Bullish vs Bearish events)",
        )
        st.plotly_chart(fig_bias, use_container_width=True)

    # ── Vector Memory Stats ────────────────────────────────────────────────
    with st.expander("Vector Memory Stats (Qdrant)"):
        try:
            from core.memory.vector_store import get_vector_store
            vs = get_vector_store()
            stats = vs.get_stats()
            if stats.get("enabled"):
                st.json(stats)
            else:
                st.warning(
                    "Qdrant not running. Start with: `docker-compose up qdrant`\n\n"
                    "Vector memory lets the AI agent find similar historical market states."
                )
        except Exception as e:
            st.error(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Order Flow
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Order Flow":
    st.title(f"Order Flow Analysis | {symbol}")

    try:
        from core.order_flow.oi_tracker import OITracker
        oi_tracker = OITracker(symbol)
        state = oi_tracker.snapshot()
    except Exception as e:
        st.error(f"Failed to fetch order flow data: {e}")
        state = {}

    if state:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            flow = state.get("flow_signal", "NEUTRAL")
            color = color_signal(flow)
            st.markdown(f"**Flow Signal**")
            st.markdown(f"<h2 style='color:{color}'>{flow}</h2>", unsafe_allow_html=True)

        with col2:
            pcr = state.get("pcr_oi", 0)
            pcr_sig = state.get("pcr_signal", "NEUTRAL")
            color = color_signal(pcr_sig)
            st.markdown(f"**PCR (OI)**: {pcr:.3f}")
            st.markdown(f"<span style='color:{color}'>{pcr_sig}</span>", unsafe_allow_html=True)

        with col3:
            st.metric("ATM IV", f"{state.get('atm_iv', 0):.1f}%")

        with col4:
            st.metric("IV Skew (Put-Call)", f"{state.get('iv_skew', 0):.1f}%")

        st.markdown("---")

        # Support/Resistance from OI
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Support Levels (Put OI)")
            sup = state.get("support_strikes", [])
            if sup:
                st.success(f"Key Supports: {', '.join(str(int(s)) for s in sup)}")

        with col_r:
            st.subheader("Resistance Levels (Call OI)")
            res = state.get("resistance_strikes", [])
            if res:
                st.error(f"Key Resistances: {', '.join(str(int(r)) for r in res)}")

        # OI Chart
        df = state.get("df", pd.DataFrame())
        if not df.empty:
            st.markdown("---")
            st.subheader("OI by Strike")

            ce_df = df[df["option_type"] == "CE"].sort_values("strike")
            pe_df = df[df["option_type"] == "PE"].sort_values("strike")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ce_df["strike"], y=ce_df["oi"],
                name="Call OI", marker_color="#ff5252", opacity=0.8,
            ))
            fig.add_trace(go.Bar(
                x=pe_df["strike"], y=pe_df["oi"],
                name="Put OI", marker_color="#00e676", opacity=0.8,
            ))

            spot = state.get("spot", 0)
            if spot:
                fig.add_vline(x=spot, line_color="#ffd740", line_dash="dash",
                              annotation_text=f"Spot {spot:.0f}")
            if state.get("max_pain"):
                fig.add_vline(x=state["max_pain"], line_color="#b388ff",
                              line_dash="dot", annotation_text=f"Pain {state['max_pain']:.0f}")

            fig.update_layout(
                barmode="group", height=400,
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                font=dict(color="#e0e0e0"),
                title="Open Interest Distribution (Call vs Put)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # OI Change chart
            st.subheader("OI Change by Strike")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=ce_df["strike"], y=ce_df["oi_change"],
                name="Call OI Change", marker_color="#ff7043",
            ))
            fig2.add_trace(go.Bar(
                x=pe_df["strike"], y=pe_df["oi_change"],
                name="Put OI Change", marker_color="#66bb6a",
            ))
            fig2.update_layout(
                barmode="group", height=350,
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                font=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # PCR data
        st.markdown("---")
        st.subheader("PCR Details")
        pcr_all = get_pcr(symbol)
        if pcr_all:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Call OI", f"{pcr_all.get('total_call_oi', 0):,}")
            c2.metric("Total Put OI", f"{pcr_all.get('total_put_oi', 0):,}")
            c3.metric("PCR OI", f"{pcr_all.get('pcr_oi', 0):.3f}")
            c4.metric("PCR Volume", f"{pcr_all.get('pcr_vol', 0):.3f}")
    else:
        st.warning("Order flow data not available. NSE may be closed or unreachable.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Strategies & Signals
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Strategies & Signals":
    st.title("Strategy Signals")

    # Strategy status
    strategies_cfg = config.get("strategies", {})
    cols = st.columns(5)
    strategy_names = ["scalp_momentum", "orb", "swing_trend", "long_term", "order_flow_fno"]
    icons = ["⚡", "📊", "🌊", "🏔", "🔄"]

    for col, name, icon in zip(cols, strategy_names, icons):
        cfg = strategies_cfg.get(name, {})
        enabled = cfg.get("enabled", False)
        with col:
            status = "✅ ON" if enabled else "⏸ OFF"
            st.markdown(f"**{icon} {name.replace('_', ' ').title()}**")
            st.markdown(status)

    st.markdown("---")

    # Manual signal scan
    if st.button("Run Manual Scan Now", type="primary"):
        with st.spinner("Scanning market..."):
            broker = None  # No broker for scan
            from core.strategies.swing.trend_swing import SwingTrendStrategy
            from core.strategies.intraday.scalp_momentum import ScalpMomentumStrategy

            all_signals = []
            for sym in [symbol]:
                data = fetch_historical(sym, "1d", 365)
                if data.empty:
                    continue
                data = add_indicators(data)

                for strat in [SwingTrendStrategy(), ScalpMomentumStrategy()]:
                    try:
                        sigs = strat.generate_signals(data, symbol=sym)
                        all_signals.extend(sigs)
                    except Exception as e:
                        st.error(f"{strat.name}: {e}")

            if all_signals:
                for sig in all_signals:
                    color = "#00e676" if sig.direction == "BUY" else "#ff5252"
                    st.markdown(f"""
                    <div style='background:#1a1d2e; padding:12px; border-radius:8px; border-left:4px solid {color}; margin:8px 0'>
                    <b style='color:{color}'>{sig.direction}</b> | {sig.strategy} | {sig.symbol}<br>
                    <small>{sig.notes}</small><br>
                    Price: ₹{sig.price:.2f} | SL: ₹{sig.stop_loss:.2f} | Target: ₹{sig.target:.2f} | Confidence: {sig.confidence:.0%}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No signals at this moment.")

    # Historical signals from DB
    st.markdown("---")
    st.subheader("Signal History")
    try:
        with DBManager() as db:
            signals = db.get_recent_signals(50)
        if signals:
            sig_df = pd.DataFrame([{
                "Time": s.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Strategy": s.strategy,
                "Symbol": s.symbol,
                "Direction": s.direction,
                "Type": s.signal_type,
                "Price": s.price,
                "Target": s.target,
                "SL": s.stop_loss,
                "Executed": s.executed,
                "Notes": s.notes or "",
            } for s in signals])
            st.dataframe(sig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No signal history yet.")
    except Exception as e:
        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Positions & P&L
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Positions & P&L":
    st.title("Positions & P&L")

    try:
        with DBManager() as db:
            trades = db.get_all_trades(200)
            daily_pnl = db.get_daily_pnl(30)

        if trades:
            trades_df = pd.DataFrame([{
                "Entry Time": t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "",
                "Symbol": t.symbol,
                "Strategy": t.strategy,
                "Side": t.trade_type,
                "Qty": t.quantity,
                "Entry": f"₹{t.entry_price:.2f}" if t.entry_price else "",
                "Exit": f"₹{t.exit_price:.2f}" if t.exit_price else "-",
                "P&L": f"₹{t.pnl:,.2f}" if t.pnl else "Open",
                "Status": t.status,
            } for t in trades])

            # Summary metrics
            closed = [t for t in trades if t.status != "OPEN"]
            open_t = [t for t in trades if t.status == "OPEN"]
            total_pnl = sum(t.pnl or 0 for t in closed)
            wins = sum(1 for t in closed if (t.pnl or 0) > 0)
            win_rate = wins / len(closed) if closed else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Realized P&L", f"₹{total_pnl:,.2f}")
            c2.metric("Open Positions", len(open_t))
            c3.metric("Total Trades", len(closed))
            c4.metric("Win Rate", f"{win_rate:.1%}")

            st.markdown("---")
            st.subheader("All Trades")
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

        else:
            st.info("No real trades yet — running in observation mode.")
            st.markdown("**To start paper trading:**")
            st.code("python3 loop_engine.py NIFTY")
            st.markdown("Trades will appear here once the loop engine executes signals during market hours.")

        # Daily P&L chart
        if daily_pnl:
            st.markdown("---")
            st.subheader("Daily P&L (Last 30 Days)")
            pnl_df = pd.DataFrame([{
                "Date": p.date,
                "Net P&L": p.net_pnl,
                "Trades": p.total_trades,
            } for p in daily_pnl])
            pnl_df = pnl_df.sort_values("Date")
            pnl_df["Cumulative"] = pnl_df["Net P&L"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pnl_df["Date"],
                y=pnl_df["Net P&L"],
                marker_color=["#00e676" if v > 0 else "#ff5252" for v in pnl_df["Net P&L"]],
                name="Daily P&L",
            ))
            fig.add_trace(go.Scatter(
                x=pnl_df["Date"], y=pnl_df["Cumulative"],
                mode="lines", name="Cumulative",
                line=dict(color="#ffd740", width=2),
                yaxis="y2",
            ))
            fig.update_layout(
                height=400, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                font=dict(color="#e0e0e0"),
                yaxis2=dict(overlaying="y", side="right"),
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading P&L data: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Backtest
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Backtest":
    st.title("Backtest Engine")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], key="bt_sym")
    with col2:
        bt_strategy = st.selectbox(
            "Strategy",
            ["ml_agent", "swing_trend", "long_term", "scalp_momentum", "orb"],
            key="bt_strat"
        )
    with col3:
        bt_days = st.slider("History (days)", 30, 59, 59, key="bt_days")

    # ── ML Agent Backtest ─────────────────────────────────────────────────────
    if bt_strategy == "ml_agent":
        st.info("Trains XGBoost on real 5-min historical Nifty data with 30-min forward labels.")
        if st.button("Run ML Agent Backtest + Retrain", type="primary"):
            with st.spinner("Fetching historical data and training model..."):
                try:
                    import sys, importlib
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    import backtest_trainer
                    importlib.reload(backtest_trainer)
                    backtest_trainer.SYMBOL = bt_symbol

                    import io, contextlib
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        backtest_trainer.main()
                    output = buf.getvalue()

                    # Parse accuracy from output
                    acc_line = [l for l in output.split("\n") if "Backtest acc" in l]
                    samples_line = [l for l in output.split("\n") if "Trained on" in l]
                    acc_str = acc_line[0].split(":")[-1].strip() if acc_line else "N/A"
                    samp_str = samples_line[0].split(":")[-1].strip().split("(")[0].strip() if samples_line else "N/A"

                    st.success("Model retrained on historical data!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Backtest Accuracy", acc_str)
                    c2.metric("Training Samples", samp_str)
                    c3.metric("Model", "XGBoost + LightGBM")

                    # Feature importance
                    from core.agent.ml_agent import TapeXGBModel
                    m = TapeXGBModel(bt_symbol)
                    fi = m.get_feature_importance(top_n=15)
                    if fi:
                        st.markdown("---")
                        st.subheader("Top 15 Most Important Features")
                        fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])
                        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                     color="Importance", color_continuous_scale="viridis")
                        fig.update_layout(height=450, paper_bgcolor="#0e1117",
                                          plot_bgcolor="#1a1d2e", font=dict(color="#e0e0e0"),
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Full training log"):
                        st.code(output)

                except Exception as e:
                    st.error(f"ML backtest error: {e}")
                    import traceback; st.code(traceback.format_exc())
        st.stop()

    initial_cap = st.number_input("Initial Capital (₹)", value=500000, step=50000)

    if st.button("Run Backtest", type="primary"):
        with st.spinner(f"Running {bt_strategy} backtest on {bt_symbol} ({bt_days}d)..."):
            try:
                from core.backtest.engine import Backtester

                if bt_strategy == "swing_trend":
                    from core.strategies.swing.trend_swing import SwingTrendStrategy
                    strat = SwingTrendStrategy()
                    tf = "1d"
                elif bt_strategy == "long_term":
                    from core.strategies.positional.long_term import LongTermStrategy
                    strat = LongTermStrategy()
                    tf = "1wk"
                elif bt_strategy == "scalp_momentum":
                    from core.strategies.intraday.scalp_momentum import ScalpMomentumStrategy
                    strat = ScalpMomentumStrategy()
                    tf = "5min"
                else:
                    from core.strategies.intraday.orb import ORBStrategy
                    strat = ORBStrategy()
                    tf = "5min"

                bt = Backtester(
                    strategy=strat,
                    symbol=bt_symbol,
                    timeframe=tf,
                    days=bt_days,
                    initial_capital=initial_cap,
                )
                result = bt.run()
                summary = result.summary()

                # Show metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Trades", summary["total_trades"])
                c2.metric("Win Rate", summary["win_rate"])
                c3.metric("Total Return", summary["return_pct"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Gross P&L", summary["gross_pnl"])
                c2.metric("Profit Factor", summary["profit_factor"])
                c3.metric("Max Drawdown", summary["max_drawdown"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe Ratio", summary["sharpe_ratio"])
                c2.metric("Expectancy", summary["expectancy"])
                c3.metric("Final Capital", summary["final_capital"])

                # Trade list
                st.markdown("---")
                st.subheader("Trade List")
                trades_df = result.to_dataframe()
                if not trades_df.empty:
                    st.dataframe(trades_df, use_container_width=True, hide_index=True)

                    # Equity curve
                    equity = [initial_cap]
                    for _, row in trades_df.iterrows():
                        equity.append(equity[-1] + row.get("pnl", 0))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=equity, mode="lines",
                        line=dict(color="#00e676", width=2),
                        fill="tonexty" if equity[-1] > equity[0] else None,
                        name="Equity Curve",
                    ))
                    fig.add_hline(y=initial_cap, line_color="#ffd740", line_dash="dash")
                    fig.update_layout(
                        height=350, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                        font=dict(color="#e0e0e0"),
                        title="Equity Curve",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest error: {e}")
                import traceback
                st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Option Chain
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Option Chain":
    st.title(f"Option Chain | {symbol}")

    df = get_option_chain(symbol)

    if not df.empty:
        spot = df["underlying"].iloc[0] if "underlying" in df.columns else 0
        atm = round(spot / 50) * 50 if spot else 0

        st.info(f"Spot: ₹{spot:,.2f} | ATM Strike: {atm}")

        # Split CE and PE
        ce_df = df[df["option_type"] == "CE"][
            ["strike", "oi", "oi_change", "volume", "ltp", "iv"]
        ].sort_values("strike").rename(columns={
            "oi": "CE OI", "oi_change": "CE Chg OI", "volume": "CE Vol",
            "ltp": "CE LTP", "iv": "CE IV"
        })

        pe_df = df[df["option_type"] == "PE"][
            ["strike", "oi", "oi_change", "volume", "ltp", "iv"]
        ].sort_values("strike").rename(columns={
            "oi": "PE OI", "oi_change": "PE Chg OI", "volume": "PE Vol",
            "ltp": "PE LTP", "iv": "PE IV"
        })

        combined = ce_df.merge(pe_df, on="strike", how="outer").sort_values("strike")

        def highlight_atm(row):
            if row["strike"] == atm:
                return ["background-color: #2d3748"] * len(row)
            return [""] * len(row)

        st.dataframe(
            combined.style.apply(highlight_atm, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # IV smile chart
        st.subheader("IV Smile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ce_df["strike"], y=ce_df["CE IV"],
            mode="lines+markers", name="Call IV",
            line=dict(color="#ff5252"),
        ))
        fig.add_trace(go.Scatter(
            x=pe_df["strike"], y=pe_df["PE IV"],
            mode="lines+markers", name="Put IV",
            line=dict(color="#00e676"),
        ))
        if atm:
            fig.add_vline(x=atm, line_color="#ffd740", line_dash="dash",
                          annotation_text="ATM")
        fig.update_layout(
            height=350, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
            font=dict(color="#e0e0e0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Option chain data not available. NSE may be closed or unreachable.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: ML Predictions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ML Predictions":
    st.title(f"ML Predictions | {symbol}")
    st.caption("XGBoost + LightGBM ensemble predictions. Train models first: `python -m core.ml.trainer`")

    # ── Load predictor ─────────────────────────────────────────────────────
    try:
        @st.cache_resource
        def _load_predictor(sym):
            from core.ml.predictor import MLPredictor
            return MLPredictor(sym)
        predictor = _load_predictor(symbol)
        models_ready = predictor.is_ready()
    except Exception as e:
        st.error(f"ML module error: {e}")
        models_ready = False

    if not models_ready:
        st.warning("Models not trained yet. Run the trainer to generate predictions:")
        st.code("python -c \"from core.ml.trainer import WalkForwardTrainer; WalkForwardTrainer('NIFTY').train_from_historical(365)\"")
        st.stop()

    # ── Generate prediction ────────────────────────────────────────────────
    with st.spinner("Generating ML prediction..."):
        try:
            ohlcv = get_historical_data(symbol, "5min", 30)
            vix_val   = get_vix()
            pcr_data  = get_pcr(symbol)
            pcr_val   = pcr_data.get("pcr_oi", 1.0)
            max_p     = get_max_pain(symbol)

            result = predictor.predict(
                ohlcv=ohlcv,
                vix=vix_val,
                pcr=pcr_val,
                max_pain=max_p,
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            result = None

    if result is None:
        st.error("Could not generate prediction. Insufficient data.")
    else:
        # ── Signal banner ──────────────────────────────────────────────────
        dir_color = {"BUY": "#00e676", "SELL": "#ff5252", "NEUTRAL": "#ffd740"}.get(result.direction, "#ffd740")
        st.markdown(
            f"<div style='background:#1a1d2e;border-left:6px solid {dir_color};"
            f"padding:20px;border-radius:8px;margin-bottom:16px'>"
            f"<h2 style='color:{dir_color};margin:0'>{result.direction}</h2>"
            f"<p style='color:#aaa;margin:4px 0'>{result.label_5class} &nbsp;|&nbsp; "
            f"Confidence: {result.confidence:.1%} &nbsp;|&nbsp; "
            f"Signal Strength: {'★' * result.signal_strength}{'☆' * (5 - result.signal_strength)}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Metric row ─────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Direction",        result.direction)
        c2.metric("P(Up)",            f"{result.prob_up:.1%}")
        c3.metric("P(Down)",          f"{result.prob_down:.1%}")
        c4.metric("Predicted Return", f"{result.predicted_return:+.3f}%")
        c5.metric("Model Age",        f"{result.model_age_days:.1f}d")

        # ── Probability gauge chart ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Probability Breakdown")

        fig = go.Figure(go.Bar(
            x=["P(Down)", "P(Neutral)", "P(Up)"],
            y=[result.prob_down,
               max(0, 1 - result.prob_up - result.prob_down),
               result.prob_up],
            marker_color=["#ff5252", "#ffd740", "#00e676"],
            text=[f"{result.prob_down:.1%}",
                  f"{max(0, 1 - result.prob_up - result.prob_down):.1%}",
                  f"{result.prob_up:.1%}"],
            textposition="outside",
        ))
        fig.update_layout(
            height=280, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
            font=dict(color="#e0e0e0"), showlegend=False,
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Feature importance ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Top Feature Importances")
        try:
            fi_df = predictor._binary.get_feature_importance_df().head(20)
            if not fi_df.empty:
                fig2 = go.Figure(go.Bar(
                    x=fi_df["importance"][::-1],
                    y=fi_df["feature"][::-1],
                    orientation="h",
                    marker_color="#40c4ff",
                ))
                fig2.update_layout(
                    height=420, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                    font=dict(color="#e0e0e0"), margin=dict(l=160),
                )
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.info("Feature importance not available.")

        # ── Notes ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.caption(f"Model details: {result.features_used} features used | {result.notes}")


# ═══════════════════════════════════════════════════════════════════════════════
# Page: AI Agent — Claude-powered trade decision engine
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Agent":
    st.title("AI Trading Agent")
    st.caption(
        "Claude claude-sonnet-4-6 reads the tape, checks FII/DII positioning, queries market memory "
        "(Qdrant), and outputs a structured trade decision."
    )

    import os as _os
    api_key = _os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.warning(
            "ANTHROPIC_API_KEY not set. Add it to your `.env` file.\n\n"
            "The agent will run in **fallback mode** (rule-based, no Claude)."
        )

    tracker = get_oi_tracker(symbol)

    # ── Agent selector ─────────────────────────────────────────────────────
    agent_tab1, agent_tab2 = st.tabs(["ML Agent (Open-Source)", "Claude Agent (Anthropic)"])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1: Open-Source ML Agent
    # ═══════════════════════════════════════════════════════════════════════
    with agent_tab1:
        st.markdown("**XGBoost + LightGBM decision engine. Optional Ollama LLM for reasoning. No API key required.**")

        # Load ML agent (cached)
        @st.cache_resource
        def _get_ml_agent(sym):
            from core.agent.ml_agent import get_ml_agent
            return get_ml_agent(sym)

        ml_agent = _get_ml_agent(symbol)
        ml_info  = ml_agent.model_info

        # Status row
        s1, s2, s3 = st.columns(3)
        s1.metric("Model", "XGBoost + LGBM" if ml_info.get("has_lgbm") else "XGBoost")
        s2.metric("Train Samples", f"{ml_info.get('train_samples', 0):,}")
        s3.metric("Ollama", ml_info.get("ollama_model") if ml_info.get("ollama_available") else "Not running")

        if not ml_info.get("ollama_available"):
            st.info(
                "Ollama not running (reasoning will use templates). "
                "To enable: `brew install ollama && ollama pull llama3.2:3b && ollama serve`"
            )

        col_ml1, col_ml2 = st.columns([2, 1])
        with col_ml1:
            run_ml = st.button("Run ML Agent — Decide Now", type="primary", use_container_width=True)
        with col_ml2:
            auto_ml = st.checkbox("Auto-run on refresh", key="auto_ml")

        should_run_ml = run_ml or (auto_ml and tracker.scraper.is_market_open())

        if "ml_decision" not in st.session_state:
            st.session_state["ml_decision"] = None
        if "ml_history" not in st.session_state:
            st.session_state["ml_history"] = []

        if should_run_ml:
            with st.spinner("ML Agent thinking... (XGBoost → Qdrant memory → decision)"):
                try:
                    tracker.tick_tape()
                    decision = ml_agent.decide(tracker, symbol=symbol)
                    st.session_state["ml_decision"]  = decision.to_dict()
                    st.session_state["ml_history"].insert(0, decision.to_dict())
                    st.session_state["ml_history"] = st.session_state["ml_history"][:20]
                except Exception as e:
                    st.error(f"ML Agent error: {e}")
                    import traceback; st.code(traceback.format_exc())

        # Display decision
        ml_dec = st.session_state.get("ml_decision")
        if ml_dec:
            _show_decision(ml_dec)

            # ── Greeks panel ──────────────────────────────────────────────
            if ml_dec.get("action", "WAIT") not in ("WAIT", "EXIT"):
                with st.expander("Options Greeks", expanded=True):
                    try:
                        from core.options.greeks import compute_greeks, _days_to_expiry
                        spot   = (ml_dec.get("sl_spot", 0) or 0)
                        act    = ml_dec.get("action","")
                        spot   = spot + 150 if "CE" in act else spot - 150 if "PE" in act else spot
                        strike = ml_dec.get("strike", 0) or 0
                        prem   = ml_dec.get("entry_high", 0) or 0
                        days   = _days_to_expiry(ml_dec.get("expiry",""))
                        opt    = "c" if "CE" in act else "p"
                        if spot > 0 and strike > 0 and prem > 0:
                            g = compute_greeks(spot, strike, days, opt, prem)
                            gc1,gc2,gc3,gc4,gc5 = st.columns(5)
                            gc1.metric("Delta",   f"{g['delta']:+.3f}")
                            gc2.metric("Gamma",   f"{g['gamma']:.5f}")
                            gc3.metric("Theta/day", f"{g['theta_per_day']:+.2f}")
                            gc4.metric("Vega",    f"{g['vega']:.3f}")
                            gc5.metric("IV",      f"{g['iv']:.1f}%")
                            gc1,gc2,gc3 = st.columns(3)
                            gc1.metric("Break-even", f"₹{g['breakeven']:.0f}")
                            gc2.metric("Time Value",  f"₹{g['time_value']:.1f}")
                            gc3.metric("Intrinsic",   f"₹{g['intrinsic']:.1f}")
                    except Exception as ge:
                        st.warning(f"Greeks unavailable: {ge}")

            # ── SHAP Explanation ──────────────────────────────────────────
            with st.expander("SHAP — Why did the model decide this?", expanded=True):
                try:
                    from core.agent.explainer import get_explainer
                    _tracker2 = get_oi_tracker(symbol)
                    feats = _tracker2.get_model_features()
                    exp = get_explainer(ml_agent._xgb_model)
                    expl = exp.explain(feats, ml_dec.get("action","WAIT"), top_n=12)
                    st.caption(expl["summary_text"])
                    shap_df = pd.DataFrame(expl["top_contributors"],
                                          columns=["Feature","SHAP Value","Feature Value"])
                    shap_df["Direction"] = shap_df["SHAP Value"].apply(
                        lambda x: "Bullish / FOR trade" if x > 0 else "Bearish / AGAINST trade")
                    shap_df["SHAP Value"] = shap_df["SHAP Value"].round(4)
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df["SHAP Value"],
                        y=shap_df["Feature"],
                        orientation="h",
                        marker_color=["#00e676" if v > 0 else "#ff5252"
                                      for v in shap_df["SHAP Value"]],
                    ))
                    fig_shap.update_layout(
                        height=380, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                        font=dict(color="#e0e0e0"), margin=dict(l=200),
                        title="Feature contributions to this decision (SHAP values)",
                        xaxis_title="SHAP value → positive = pushes towards this action",
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception as se:
                    st.info(f"SHAP: {se}")
        else:
            st.info("Click **Run ML Agent** to get a decision.")

        # Feature importance
        with st.expander("Feature Importance (what the model looks at)"):
            fi = ml_agent.get_feature_importance()
            if fi:
                fi_df = pd.DataFrame(fi[:20], columns=["Feature", "Importance"])
                fig_fi = go.Figure(go.Bar(
                    x=fi_df["Importance"][::-1],
                    y=fi_df["Feature"][::-1],
                    orientation="h",
                    marker_color="#40c4ff",
                ))
                fig_fi.update_layout(
                    height=400, paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
                    font=dict(color="#e0e0e0"), margin=dict(l=180),
                    title="Top 20 features driving trade decisions",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        # Retrain button
        st.markdown("---")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Retrain Model (synthetic data)", key="retrain_ml"):
                with st.spinner("Retraining on synthetic data..."):
                    ml_agent.train(force=True)
                st.success("Model retrained!")
                st.cache_resource.clear()
        with col_r2:
            if st.button("Retrain on Historical Data (Optuna)", key="retrain_hist"):
                with st.spinner("Fetching history + Optuna tuning (2 min)..."):
                    try:
                        import importlib, sys
                        sys.path.insert(0, str(Path(__file__).parent.parent))
                        import backtest_trainer; importlib.reload(backtest_trainer)
                        backtest_trainer.SYMBOL = symbol
                        import io, contextlib
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            backtest_trainer.main()
                        ml_agent._xgb_model._load()
                        st.success("Historical retrain complete!")
                        with st.expander("Training log"):
                            st.code(buf.getvalue()[-3000:])
                    except Exception as re:
                        st.error(f"Retrain error: {re}")

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2: Claude Agent
    # ═══════════════════════════════════════════════════════════════════════
    with agent_tab2:
        st.markdown("**Claude claude-sonnet-4-6 agent — uses tool_use loop (tape → memory → option chain → decide).**")

        if not api_key:
            st.error("Set ANTHROPIC_API_KEY in .env to use Claude agent.")
        else:
            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                run_agent = st.button("Run Claude Agent", type="primary", use_container_width=True)
            with col_btn2:
                auto_run = st.checkbox("Auto-run every refresh", value=False, key="auto_claude")

            should_run = run_agent or (auto_run and tracker.scraper.is_market_open())

            if "agent_decision" not in st.session_state:
                st.session_state["agent_decision"] = None
            if "agent_history" not in st.session_state:
                st.session_state["agent_history"] = []

            if should_run:
                with st.spinner("Claude analyzing market... (tape → FII/DII → memory → decides)"):
                    try:
                        tracker.tick_tape()
                        from core.agent.trading_agent import get_agent as get_trading_agent
                        agent = get_trading_agent(api_key)
                        decision = agent.decide(tracker, symbol=symbol)
                        st.session_state["agent_decision"] = decision.to_dict()
                        st.session_state["agent_history"].insert(0, decision.to_dict())
                        st.session_state["agent_history"] = st.session_state["agent_history"][:20]
                    except Exception as e:
                        st.error(f"Claude Agent error: {e}")
                        import traceback; st.code(traceback.format_exc())

            claude_dec = st.session_state.get("agent_decision")
            if claude_dec:
                _show_decision(claude_dec)
            else:
                st.info("Click **Run Claude Agent** to get a decision.")

    # ── History (shared between both tabs) ─────────────────────────────────
    history = st.session_state.get("ml_history", []) + st.session_state.get("agent_history", [])
    history = sorted(history, key=lambda x: x.get("timestamp",""), reverse=True)[:20]
    if len(history) > 1:
        st.markdown("---")
        st.subheader("Decision History (this session)")
        hist_rows = [{
            "Time":  d.get("timestamp","")[:19], "Action": d.get("action"),
            "Strike": d.get("strike") or "-", "Conf": f"{d.get('confidence',0):.0%}",
            "SL": f"₹{d.get('stop_loss',0):.1f}", "Tgt": f"₹{d.get('target',0):.1f}",
            "R:R": f"{d.get('risk_reward',0):.1f}",
        } for d in history]
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # ── Vector memory ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Vector Memory (Qdrant)"):
        try:
            from core.memory.vector_store import get_vector_store
            vs = get_vector_store()
            stats = vs.get_stats()
            if stats.get("enabled"):
                c1, c2, c3 = st.columns(3)
                c1.metric("States Stored", stats.get("total_states", 0))
                c2.metric("Vector Size",   stats.get("vector_size", 0))
                c3.metric("Collection",    stats.get("collection", ""))
                st.success("Qdrant connected — agent has historical market memory.")
            else:
                st.warning("Start Qdrant: `docker-compose up -d qdrant`")
        except Exception as e:
            st.error(str(e))

    # ── Display latest decision ────────────────────────────────────────────
    decision = st.session_state.get("agent_decision")
    if decision:
        action = decision.get("action", "WAIT")
        conf   = decision.get("confidence", 0)
        action_color = {
            "BUY_CE": "#00e676", "BUY_PE": "#ff5252",
            "SELL_STRADDLE": "#b388ff", "SELL_CE": "#ffab40", "SELL_PE": "#ffab40",
            "WAIT": "#ffd740", "EXIT": "#90a4ae",
        }.get(action, "#ffd740")

        st.markdown(
            f"<div style='background:#1a1d2e;border-left:6px solid {action_color};"
            f"padding:20px;border-radius:8px;margin:12px 0'>"
            f"<h2 style='color:{action_color};margin:0'>{action}</h2>"
            f"<p style='color:#aaa;margin:4px 0'>Confidence: {conf:.0%} | "
            f"Symbol: {decision.get('symbol')} | "
            f"Strike: {decision.get('strike') or 'N/A'} | "
            f"Expiry: {decision.get('expiry') or 'N/A'}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if action not in ("WAIT", "EXIT", ""):
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Entry Range",
                f"₹{decision.get('entry_low', 0):.0f}–{decision.get('entry_high', 0):.0f}")
            mc2.metric("Stop Loss (Premium)",  f"₹{decision.get('stop_loss', 0):.1f}")
            mc3.metric("Target (Premium)",     f"₹{decision.get('target', 0):.1f}")
            mc4.metric("SL (Spot)",            f"{decision.get('sl_spot', 0):.0f}")
            mc5.metric("Target (Spot)",        f"{decision.get('target_spot', 0):.0f}")

            rr = decision.get("risk_reward", 0)
            qty = decision.get("qty_lots", 1)
            st.info(f"Risk:Reward = **{rr:.1f}x** | Quantity: **{qty} lot(s)**")

        st.markdown("**Agent Reasoning:**")
        st.markdown(
            f"<div style='background:#0e1117;padding:12px;border-radius:6px;"
            f"border:1px solid #2d3748;color:#ccc'>{decision.get('reasoning', '')}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Decision at: {decision.get('timestamp', '')[:19]}")

    else:
        st.info("Click **Run Agent Now** to get a trade decision based on current market conditions.")

    # ── Decision History ───────────────────────────────────────────────────
    history = st.session_state.get("agent_history", [])
    if len(history) > 1:
        st.markdown("---")
        st.subheader("Decision History (this session)")
        hist_rows = []
        for d in history:
            hist_rows.append({
                "Time":       d.get("timestamp", "")[:19],
                "Action":     d.get("action"),
                "Strike":     d.get("strike") or "-",
                "Confidence": f"{d.get('confidence', 0):.0%}",
                "SL":         f"₹{d.get('stop_loss', 0):.1f}",
                "Target":     f"₹{d.get('target', 0):.1f}",
                "R:R":        f"{d.get('risk_reward', 0):.1f}",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

    # ── Vector Memory status ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Vector Memory (Qdrant)"):
        try:
            from core.memory.vector_store import get_vector_store
            vs = get_vector_store()
            stats = vs.get_stats()
            if stats.get("enabled"):
                c1, c2, c3 = st.columns(3)
                c1.metric("States Stored", stats.get("total_states", 0))
                c2.metric("Vector Size",   stats.get("vector_size", 0))
                c3.metric("Collection",    stats.get("collection", ""))
                st.success("Qdrant connected. Agent has market memory.")
            else:
                st.warning(
                    "Qdrant not running — agent has no historical memory.\n\n"
                    "Start Qdrant: `docker-compose up -d qdrant`\n\n"
                    "After a few hours of running, the agent will build pattern memory."
                )
        except Exception as e:
            st.error(str(e))
