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

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Order Flow", "ML Predictions", "AI Agent", "Strategies & Signals", "Positions & P&L", "Backtest", "Option Chain"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title(f"Market Overview | {symbol}")

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
            st.info("No trades yet. Start trading engine: `python main.py live`")

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
    st.info("Run backtests on historical data to evaluate strategy performance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], key="bt_sym")
    with col2:
        bt_strategy = st.selectbox(
            "Strategy",
            ["swing_trend", "long_term", "scalp_momentum", "orb"],
            key="bt_strat"
        )
    with col3:
        bt_days = st.slider("History (days)", 90, 1000, 365, key="bt_days")

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
# Page: AI Agent
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Agent":
    st.title("AI Trading Copilot")
    st.caption("Powered by Claude claude-sonnet-4-6. Asks questions about signals, risk, and market state.")

    # ── Initialize agent ───────────────────────────────────────────────────
    try:
        from core.ai.agent import get_agent
        agent = get_agent()
    except Exception as e:
        st.error(f"Agent init error: {e}")
        st.stop()

    # ── Build market context ───────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        quote   = get_quote(symbol)
        vix_v   = get_vix()
        pcr_d   = get_pcr(symbol)
        max_p   = get_max_pain(symbol)
        hist_d  = get_historical_data(symbol, "1d", 30)

        technicals = {}
        if not hist_d.empty:
            last = hist_d.iloc[-1]
            for k in ["ema21", "ema50", "rsi", "macd", "atr"]:
                v = last.get(k)
                if v is not None and str(v) != "nan":
                    technicals[k] = float(v)

        market_ctx = {
            "ltp":      quote.get("ltp", 0),
            "change":   quote.get("change_pct", 0),
            "vix":      vix_v,
            "pcr":      pcr_d.get("pcr_oi", 1.0),
            "max_pain": max_p,
        }

        # Try get news sentiment
        sentiment_ctx = {}
        try:
            from core.news.collector import get_news_collector
            nc = get_news_collector()
            sentiment_ctx = nc.get_market_sentiment()
        except Exception:
            pass

        # Try get ML prediction
        pred_ctx = {}
        try:
            from core.ml.predictor import get_predictor
            pr = get_predictor(symbol)
            if pr.is_ready():
                ohlcv = get_historical_data(symbol, "5min", 30)
                res = pr.predict(ohlcv, vix=vix_v, pcr=market_ctx["pcr"], max_pain=max_p)
                if res:
                    pred_ctx = {
                        "direction":       res.direction,
                        "confidence":      res.confidence,
                        "predicted_return": res.predicted_return,
                        "label_5class":    res.label_5class,
                    }
        except Exception:
            pass

    context = {
        "market":     market_ctx,
        "technicals": technicals,
    }
    if sentiment_ctx:
        context["sentiment"] = {"sentiment": sentiment_ctx.get("sentiment"), "score": sentiment_ctx.get("score"), "count": sentiment_ctx.get("count")}
    if pred_ctx:
        context["prediction"] = pred_ctx

    # ── Quick action buttons ───────────────────────────────────────────────
    st.subheader("Quick Analysis")
    qcol1, qcol2, qcol3 = st.columns(3)

    if qcol1.button("Explain Current Signal"):
        with st.spinner("Thinking..."):
            reply = agent.explain_signal(context)
        st.session_state.setdefault("chat", []).append({"role": "assistant", "content": reply})

    if qcol2.button("Daily Market Summary"):
        with st.spinner("Thinking..."):
            reply = agent.daily_summary(context)
        st.session_state.setdefault("chat", []).append({"role": "assistant", "content": reply})

    if qcol3.button("Clear Chat"):
        st.session_state["chat"] = []
        agent.clear_history()
        st.rerun()

    st.markdown("---")

    # ── Chat interface ─────────────────────────────────────────────────────
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about signals, risk, market analysis..."):
        st.session_state["chat"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                reply = agent.query(prompt, context)
            st.markdown(reply)
            st.session_state["chat"].append({"role": "assistant", "content": reply})

    # ── Context summary ────────────────────────────────────────────────────
    with st.expander("Market Context (sent to AI)"):
        import json as _json
        st.json(context)
