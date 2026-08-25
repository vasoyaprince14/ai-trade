"""
XAUUSD Live Dashboard
======================
Run: streamlit run xauusd/dashboard.py --server.port 8502
"""

import sys, json, time
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xauusd.data     import get_bars, get_macro, get_price
from xauusd.strategy import analyze, _ema, _rsi, _atr, _macd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD Trader",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
body { background:#0e1117; }
.metric-big { font-size:2.4rem; font-weight:700; }
.buy  { color:#00d4aa; }
.sell { color:#ff4b4b; }
.wait { color:#ffa500; }
.card { background:#1e2130; border-radius:10px; padding:1rem; margin-bottom:0.5rem; }
.divider { border-top:1px solid #2d3250; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_latest_signal() -> dict | None:
    try:
        with open("/tmp/xauusd_signal.json") as f:
            return json.load(f)
    except Exception:
        return None


def load_history() -> list:
    try:
        with open("/tmp/xauusd_history.json") as f:
            return json.load(f)
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_data():
    df15 = get_bars("15m", "5d")
    df1h = get_bars("1h",  "60d")
    macro = get_macro()
    return df15, df1h, macro

@st.cache_data(ttl=900)   # India scan cached 15 minutes — signals don't flip
def fetch_india():
    from xauusd.india import get_stock_scan, get_nifty_hedge_signal
    return get_stock_scan(), get_nifty_hedge_signal()


def run_analysis():
    df15, df1h, macro = fetch_data()
    sig = analyze(df15, df1h, macro)
    return sig, df15, df1h, macro


def color_action(action: str) -> str:
    return {"BUY": "buy", "SELL": "sell"}.get(action, "wait")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=True)
    show_1h      = st.checkbox("Show 1H chart", value=False)
    st.markdown("---")
    st.caption("**Data:** GC=F (Gold Futures) via yfinance (~15m delay)")
    st.caption("**Macro:** DXY · 10Y · VIX")
    st.caption("**Engine:** `python3 xauusd/engine.py`")
    st.markdown("---")
    if st.button("Send Telegram Test"):
        try:
            from core.alerts.telegram_bot import send_message
            ok = send_message("🧪 XAUUSD Dashboard test ping!")
            st.success("Sent!" if ok else "Failed — check .env")
        except Exception as e:
            st.error(str(e))


# ── Auto refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 60:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🥇 XAUUSD Live Trader")

# ── Load / analyse ────────────────────────────────────────────────────────────
with st.spinner("Fetching live data..."):
    try:
        sig, df15, df1h, macro = run_analysis()
        error = None
    except Exception as e:
        sig, df15, df1h, macro, error = None, None, None, {}, str(e)

if error:
    st.error(f"Data fetch error: {error}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Live Signal", "📈 Chart", "🛡️ Hedge System", "🇮🇳 India + News", "🤖 Bull vs Bear", "📋 History"])


# ══ TAB 1: LIVE SIGNAL ═══════════════════════════════════════════════════════
with tabs[0]:

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    price = sig.entry if sig else 0
    with col1:
        st.metric("Gold (XAU/USD)", f"${price:,.2f}")
    with col2:
        st.metric("DXY", f"{macro.get('dxy', 0):.2f}")
    with col3:
        st.metric("US 10Y Yield", f"{macro.get('us10y', 0):.2f}%")
    with col4:
        st.metric("VIX", f"{macro.get('vix', 0):.1f}")
    with col5:
        now_utc = datetime.now(timezone.utc)
        st.metric("Session", sig.session if sig else "-")

    st.markdown("---")

    # Signal card
    if sig:
        action_cls = color_action(sig.action)
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏳"}.get(sig.action, "⏳")

        lcol, rcol = st.columns([1, 1])

        with lcol:
            st.markdown(f"### {action_emoji} Signal: <span class='{action_cls}'>{sig.action}</span>",
                        unsafe_allow_html=True)

            if sig.is_trade():
                risk = abs(sig.entry - sig.stop_loss)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entry",  f"${sig.entry:.2f}")
                c2.metric("Stop Loss", f"${sig.stop_loss:.2f}", delta=f"-{risk:.1f} pts", delta_color="inverse")
                c3.metric("Target", f"${sig.target:.2f}", delta=f"+{abs(sig.target-sig.entry):.1f} pts")
                c4.metric("R:R", f"1:{sig.risk_reward:.1f}")

                st.progress(sig.confidence, text=f"Confidence: {sig.confidence:.0%}  |  Score: {sig.score}/10")

                # Breakeven info
                be_price = sig.entry  # after moving SL
                if sig.action == "BUY":
                    one_r_price = sig.entry + (sig.entry - sig.stop_loss)
                    st.info(f"🔒 Move SL to breakeven (${sig.entry:.2f}) when price hits **${one_r_price:.2f}** (+1R)\nWorst case after that = **$0 loss**")
                else:
                    one_r_price = sig.entry - (sig.stop_loss - sig.entry)
                    st.info(f"🔒 Move SL to breakeven (${sig.entry:.2f}) when price hits **${one_r_price:.2f}** (+1R)\nWorst case after that = **$0 loss**")
            else:
                st.markdown(f"**Trend (1H):** {sig.trend_1h}")
                st.markdown(f"**RSI (14):** {sig.rsi:.1f}")
                st.warning(f"Reason: {sig.reason[:300]}")

        with rcol:
            # Score breakdown gauge
            st.markdown("#### Signal Score Breakdown")
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = sig.score,
                title = {"text": "Score (need ≥6 to trade)"},
                gauge = {
                    "axis":  {"range": [0, 10]},
                    "bar":   {"color": "#00d4aa" if sig.action == "BUY" else "#ff4b4b" if sig.action == "SELL" else "#ffa500"},
                    "steps": [
                        {"range": [0, 5],  "color": "#1e1e2e"},
                        {"range": [5, 7],  "color": "#2a2a3e"},
                        {"range": [7, 10], "color": "#1a2a1a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 6},
                },
                number={"suffix": "/10"},
            ))
            fig_gauge.update_layout(height=240, paper_bgcolor="#0e1117", font_color="white", margin=dict(t=40,b=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Reason breakdown
        if sig.reason:
            st.markdown("**Analysis factors:**")
            for r in sig.reason.split(" | "):
                if r.strip():
                    st.markdown(f"- {r.strip()}")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  |  Auto-refreshes every 60s")
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()


# ══ TAB 2: CHART ═════════════════════════════════════════════════════════════
with tabs[1]:

    chart_df = df1h if show_1h else df15
    tf_label  = "1H" if show_1h else "15m"

    if chart_df is None or chart_df.empty:
        st.warning("No chart data available")
    else:
        # Compute indicators
        c = chart_df.copy().tail(150)
        c["ema21"] = _ema(c["close"], 21)
        c["ema55"] = _ema(c["close"], 55)
        c["rsi"]   = _rsi(c["close"], 14)
        c["atr"]   = _atr(c, 14)
        c["macd"], c["macd_sig"] = _macd(c["close"])
        c["macd_hist"] = c["macd"] - c["macd_sig"]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            vertical_spacing=0.03,
            subplot_titles=[f"XAUUSD {tf_label} — Candlestick + EMA21/55", "RSI (14)", "MACD"],
        )

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=c.index, open=c["open"], high=c["high"],
            low=c["low"], close=c["close"],
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b4b",
            name="Price",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=c.index, y=c["ema21"], line=dict(color="#4fa3e0", width=1.5), name="EMA21"), row=1, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=c["ema55"], line=dict(color="#f0a500", width=1.5), name="EMA55"), row=1, col=1)

        # Current signal markers
        if sig and sig.is_trade():
            marker_color = "#00d4aa" if sig.action == "BUY" else "#ff4b4b"
            marker_sym   = "triangle-up" if sig.action == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[c.index[-1]], y=[sig.entry],
                mode="markers", marker=dict(color=marker_color, size=14, symbol=marker_sym),
                name=f"{sig.action} Signal",
            ), row=1, col=1)
            # SL and Target lines
            fig.add_hline(y=sig.stop_loss, line_dash="dash", line_color="#ff4b4b", annotation_text=f"SL ${sig.stop_loss:.0f}", row=1, col=1)
            fig.add_hline(y=sig.target,    line_dash="dash", line_color="#00d4aa", annotation_text=f"TP ${sig.target:.0f}", row=1, col=1)
            fig.add_hline(y=sig.entry,     line_dash="dot",  line_color="#ffffff",  annotation_text=f"Entry ${sig.entry:.0f}", row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=c.index, y=c["rsi"], line=dict(color="#a78bfa", width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", line_width=1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot",  line_color="#555555", line_width=1, row=2, col=1)

        # MACD
        colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in c["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=c.index, y=c["macd_hist"], marker_color=colors, name="MACD Hist", opacity=0.6), row=3, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=c["macd"],     line=dict(color="#4fa3e0", width=1), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=c["macd_sig"], line=dict(color="#f0a500", width=1), name="Signal"), row=3, col=1)

        fig.update_layout(
            height=700,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_xaxes(gridcolor="#1e2130")
        fig.update_yaxes(gridcolor="#1e2130")

        st.plotly_chart(fig, use_container_width=True)

        # Key stats below chart
        if sig:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("ATR(14)", f"{sig.atr:.2f}")
            m2.metric("RSI(14)", f"{sig.rsi:.1f}")
            m3.metric("1H Trend", sig.trend_1h)
            m4.metric("EMA21", f"${float(c['ema21'].iloc[-1]):.1f}")
            m5.metric("EMA55", f"${float(c['ema55'].iloc[-1]):.1f}")


# ══ TAB 3: 0-LOSS HEDGE ══════════════════════════════════════════════════════
with tabs[2]:

    st.markdown("## 🛡️ Zero-Loss Hedging System")
    st.markdown("""
    **Concept:** You can never be certain where price goes — but you CAN guarantee you never lose money.
    Two strategies below work together:

    | Strategy | How it works | Result |
    |----------|-------------|--------|
    | **Breakeven Trail** | Move SL to entry after +1R profit | Worst case = **$0 loss** |
    | **Lock & Hedge** | Open counter-position if trade goes wrong | Cap loss at hedge cost (~$2-3/oz) |
    """)

    st.markdown("---")

    # ── Strategy 1: Breakeven Trail ──
    st.markdown("### 1️⃣ Breakeven Trail (Simplest)")

    col_be1, col_be2 = st.columns(2)
    with col_be1:
        be_entry  = st.number_input("Your entry price ($)", value=float(sig.entry if sig and sig.is_trade() else 4700.0), step=0.1, format="%.2f")
        be_action = st.selectbox("Direction", ["BUY", "SELL"])
        be_sl     = st.number_input("Your SL ($)", value=float(sig.stop_loss if sig and sig.is_trade() else 4680.0), step=0.1, format="%.2f")
        be_tp     = st.number_input("Your Target ($)", value=float(sig.target if sig and sig.is_trade() else 4730.0), step=0.1, format="%.2f")
        lots      = st.number_input("Lot size (oz)", value=1.0, step=0.1)

    with col_be2:
        risk_pts   = abs(be_entry - be_sl)
        reward_pts = abs(be_tp - be_entry)
        rr_ratio   = reward_pts / risk_pts if risk_pts > 0 else 0
        move_be_at = be_entry + risk_pts if be_action == "BUY" else be_entry - risk_pts

        max_loss   = risk_pts * lots
        max_profit = reward_pts * lots

        st.markdown("#### Trade Summary")
        st.metric("Risk (per oz)",   f"${risk_pts:.2f}")
        st.metric("Reward (per oz)", f"${reward_pts:.2f}")
        st.metric("R:R Ratio",       f"1:{rr_ratio:.2f}")
        st.metric("Max Loss (total)", f"${max_loss:.2f}", delta="Before breakeven")
        st.metric("Max Profit",       f"${max_profit:.2f}")

        st.markdown("---")
        st.markdown(f"**⚡ Move SL to entry (${be_entry:.2f}) when price hits:**")
        st.markdown(f"### {'🟢' if be_action=='BUY' else '🔴'} ${move_be_at:.2f}  (+1R)")
        st.success(f"After that: worst case = **$0 loss** (SL at breakeven)")

    # Live check if current price hit breakeven trigger
    if sig:
        current = sig.entry
        if be_action == "BUY" and current >= move_be_at:
            st.warning(f"⚠️ Current price ${current:.2f} is at/above breakeven trigger ${move_be_at:.2f} — **MOVE SL TO ${be_entry:.2f} NOW**")
        elif be_action == "SELL" and current <= move_be_at:
            st.warning(f"⚠️ Current price ${current:.2f} is at/below breakeven trigger ${move_be_at:.2f} — **MOVE SL TO ${be_entry:.2f} NOW**")

    st.markdown("---")

    # ── Strategy 2: Lock & Hedge ──
    st.markdown("### 2️⃣ Lock & Hedge (Advanced)")
    st.markdown("""
    When your original trade is **-0.5R** (halfway to SL), open a **counter-position** of the same size.

    - **Trade 1 (original):** BUY 1 oz at $4713
    - **Trade 2 (hedge):** SELL 1 oz at $4703 (when -0.5R)
    - Now: **Whatever price does, one position profits**
    - Close both when net P&L ≥ 0
    """)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        h_entry   = st.number_input("Original entry ($)", value=float(sig.entry if sig and sig.is_trade() else 4713.0), step=0.1, format="%.2f", key="hedge_entry")
        h_dir     = st.selectbox("Original direction", ["BUY", "SELL"], key="hedge_dir")
        h_sl      = st.number_input("Original SL ($)", value=float(sig.stop_loss if sig and sig.is_trade() else 4693.0), step=0.1, format="%.2f", key="hedge_sl")
        h_lots    = st.number_input("Lots (oz)", value=1.0, step=0.1, key="hedge_lots")

    with col_h2:
        h_risk_pts     = abs(h_entry - h_sl)
        hedge_trigger  = h_entry - 0.5 * h_risk_pts if h_dir == "BUY" else h_entry + 0.5 * h_risk_pts
        hedge_dir      = "SELL" if h_dir == "BUY" else "BUY"

        st.markdown("#### Hedge Plan")
        st.metric("Original risk",    f"${h_risk_pts:.2f}/oz")
        st.metric("Hedge trigger",    f"${hedge_trigger:.2f}")
        st.metric("Hedge direction",  hedge_dir)

        # Scenarios
        tp_est = h_entry + h_risk_pts * 1.67 if h_dir == "BUY" else h_entry - h_risk_pts * 1.67

        st.markdown("#### Scenarios after hedge is opened:")
        scenario_data = {
            "Scenario": ["Price hits TP", "Price stays flat", "Price drops further"],
            "Trade 1 P&L": [
                f"+${(abs(tp_est - h_entry) * h_lots):.0f}",
                f"-${(0.5 * h_risk_pts * h_lots):.0f}",
                f"-${(h_risk_pts * h_lots):.0f}",
            ],
            "Hedge P&L": [
                f"-${(abs(tp_est - hedge_trigger) * h_lots):.0f}",
                f"+${(0.5 * h_risk_pts * h_lots):.0f}",
                f"+${(h_risk_pts * h_lots):.0f}",
            ],
            "Net": [
                f"≈ ${((abs(tp_est - h_entry) - abs(tp_est - hedge_trigger)) * h_lots):.0f}",
                "≈ $0",
                "≈ $0",
            ],
        }
        st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)

        st.info("Net outcome: **Profit if TP hits, ~$0 otherwise.** Your downside is the bid-ask spread cost (~$2-5).")

    st.markdown("---")

    # ── Strategy 3: Trailing SL ──
    st.markdown("### 3️⃣ Trail Stop (Protect Running Profits)")
    st.markdown("Once in profit, trail SL by **0.5× ATR** below each new high.")

    if sig and sig.is_trade():
        trail_atr = sig.atr
        trail_step = 0.5 * trail_atr
        st.markdown(f"**Current ATR(14):** {trail_atr:.2f}  →  Trail step: **{trail_step:.2f} pts**")

        # Show trail levels
        if sig.action == "BUY":
            levels = []
            cur_sl = sig.stop_loss
            cur_px = sig.entry
            for i in range(5):
                cur_px += trail_step
                cur_sl  = cur_px - trail_step
                levels.append({"When price reaches": f"${cur_px:.2f}", "Trail SL to": f"${cur_sl:.2f}", "Locked profit": f"+${cur_sl - sig.entry:.2f}/oz"})
        else:
            levels = []
            cur_sl = sig.stop_loss
            cur_px = sig.entry
            for i in range(5):
                cur_px -= trail_step
                cur_sl  = cur_px + trail_step
                levels.append({"When price reaches": f"${cur_px:.2f}", "Trail SL to": f"${cur_sl:.2f}", "Locked profit": f"+${sig.entry - cur_sl:.2f}/oz"})

        st.dataframe(pd.DataFrame(levels), use_container_width=True, hide_index=True)
    else:
        st.info("A live BUY or SELL signal is needed to compute trail levels.")


# ══ TAB 4: INDIA + NEWS ══════════════════════════════════════════════════════
with tabs[3]:

    st.markdown("## 🇮🇳 India Market — News Stocks + Nifty Hedge")

    col_nifty, col_scan = st.columns([1, 2])

    with col_nifty:
        st.markdown("### Nifty Hedge Signal")
        with st.spinner("Loading Nifty data..."):
            try:
                _, nh = fetch_india()

                hedge_color = {"BUY CE": "#00d4aa", "BUY PE": "#ff4b4b", "HOLD": "#ffa500"}.get(nh["hedge"], "#888")
                st.markdown(f"<h2 style='color:{hedge_color}'>{nh['hedge']}</h2>", unsafe_allow_html=True)
                st.metric("Nifty 50", f"{nh.get('nifty_price', 0):,.2f}")
                st.metric("RSI (14)", f"{nh.get('rsi', 0):.1f}")
                st.metric("1M Return", f"{nh.get('ret_1m', 0):+.2f}%")
                st.metric("ATR (daily)", f"{nh.get('atr', 0):.0f} pts")
                st.markdown(f"**Strength:** {nh.get('strength', 'N/A')}")
                st.info(nh.get("reason", ""))

                # Hedge sizing guide
                st.markdown("#### Hedge Size Guide")
                portfolio = st.number_input("Portfolio value (₹)", value=500000, step=50000)
                hedge_pct  = {"STRONG": 0.05, "MODERATE": 0.02, "NONE": 0}.get(nh.get("strength","NONE"), 0)
                hedge_amt  = portfolio * hedge_pct
                nifty_lots = max(1, int(hedge_amt / (nh.get("nifty_price", 24000) * 25)))
                st.metric("Hedge allocation", f"₹{hedge_amt:,.0f}  ({hedge_pct*100:.0f}%)")
                if hedge_pct > 0:
                    st.metric("Suggested Nifty lots", f"{nifty_lots} lot(s)")
            except Exception as e:
                st.error(f"Nifty data error: {e}")

    with col_scan:
        st.markdown("### Top NSE Stocks (Momentum + News)")
        with st.spinner("Scanning NSE stocks (cached 15min)..."):
            try:
                stocks, _ = fetch_india()

                if stocks:
                    top = stocks[:10]
                    rows = []
                    for s in top:
                        news_text = " | ".join(s.get("news", []))[:60] + "..." if s.get("news") else "No recent news"
                        rows.append({
                            "Stock":       s["name"],
                            "Price (₹)":   f"₹{s['price']:,.2f}",
                            "Score":       f"{s['score']}/10",
                            "Trend":       s["trend"],
                            "RSI":         f"{s['rsi']:.0f}",
                            "5D Mom%":     f"{s['momentum_5d']:+.1f}%",
                            "Vol Spike":   f"{s['vol_ratio']:.1f}x",
                            "SL":          f"₹{s['sl']:,.2f}",
                            "Target":      f"₹{s['tp']:,.2f}",
                            "News":        news_text,
                        })

                    def style_trend(val):
                        if val == "UP":   return "color:#00d4aa;font-weight:bold"
                        if val == "DOWN": return "color:#ff4b4b;font-weight:bold"
                        return "color:#ffa500"

                    df_stocks = pd.DataFrame(rows)
                    st.dataframe(
                        df_stocks.style.applymap(style_trend, subset=["Trend"]),
                        use_container_width=True, hide_index=True,
                    )

                    # Detailed card for top pick
                    if top:
                        best = top[0]
                        st.markdown(f"---\n#### Top Pick: **{best['name']}**")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Price",  f"₹{best['price']:,.2f}")
                        c2.metric("SL",     f"₹{best['sl']:,.2f}")
                        c3.metric("Target", f"₹{best['tp']:,.2f}")
                        if best.get("news"):
                            st.markdown("**Recent News:**")
                            for n in best["news"]:
                                st.markdown(f"- {n}")
                        st.caption(f"Reasons: {best['reasons']}")
            except Exception as e:
                st.error(f"Stock scan error: {e}")


# ══ TAB 5: BULL VS BEAR DEBATE ═══════════════════════════════════════════════
with tabs[4]:

    st.markdown("## 🤖 Bull vs Bear Debate Engine")
    st.markdown("""
    Two AI agents (Bull + Bear) debate the current XAUUSD setup using live market data.
    A judge agent produces the final verdict.
    *(Powered by Ollama llama3.2:3b — takes ~30s)*
    """)

    debate_ctx = {
        "symbol":    "XAUUSD",
        "price":     f"${sig.entry:.2f}" if sig else "N/A",
        "trend_1h":  sig.trend_1h if sig else "UNKNOWN",
        "rsi_14":    f"{sig.rsi:.1f}" if sig else "N/A",
        "atr_14":    f"{sig.atr:.2f}" if sig else "N/A",
        "session":   sig.session if sig else "N/A",
        "ml_score":  f"{sig.score}/10  ({sig.action})" if sig else "N/A",
        "dxy":       f"{macro.get('dxy', 0):.2f}",
        "us_10y":    f"{macro.get('us10y', 0):.2f}%",
        "vix":       f"{macro.get('vix', 0):.1f}",
        "signal_reason": sig.reason[:200] if sig else "",
    }

    if st.button("⚡ Run Bull vs Bear Debate", key="debate_btn"):
        with st.spinner("Bull and Bear agents are debating... (~30s)"):
            try:
                from core.agents.debate import run_debate
                result = run_debate(debate_ctx, symbol="XAUUSD", rounds=2)
                st.session_state["debate_result"] = result
            except Exception as e:
                st.error(f"Debate error: {e}")

    if "debate_result" in st.session_state:
        dr = st.session_state["debate_result"]
        verdict = dr.get("verdict", {})
        direction = verdict.get("direction", "NEUTRAL")
        confidence = verdict.get("confidence", 50)
        vcolor = {"BULLISH": "#00d4aa", "BEARISH": "#ff4b4b"}.get(direction, "#ffa500")

        st.markdown(f"### Verdict: <span style='color:{vcolor}'>{direction}</span> ({confidence}%)",
                    unsafe_allow_html=True)
        st.info(f"**Reason:** {verdict.get('reason', '')}")
        st.caption(f"**Invalidated if:** {verdict.get('invalidated_if', 'N/A')} | Agreed with: **{verdict.get('agreed_with', 'N/A')}**")

        col_b, col_s = st.columns(2)
        with col_b:
            st.markdown("#### 🟢 Bull Case")
            st.write(dr.get("bull_case", ""))
            if dr.get("bull_rebuttal"):
                st.markdown("**Rebuttal:**")
                st.write(dr.get("bull_rebuttal", ""))

        with col_s:
            st.markdown("#### 🔴 Bear Case")
            st.write(dr.get("bear_case", ""))
            if dr.get("bear_rebuttal"):
                st.markdown("**Rebuttal:**")
                st.write(dr.get("bear_rebuttal", ""))

        st.markdown("#### 🔍 Evidence Check")
        st.write(dr.get("evidence", ""))
        st.caption(f"Debated at: {dr.get('timestamp', '')}")


# ══ TAB 6: HISTORY ═══════════════════════════════════════════════════════════
with tabs[5]:

    st.markdown("## 📋 Signal History")

    history = load_history()

    if not history:
        st.info("No signals logged yet. Start the engine: `python3 xauusd/engine.py`")
    else:
        rows = []
        for h in reversed(history):
            ts  = h.get("timestamp", "")[:19].replace("T", " ")
            act = h.get("action", "-")
            rows.append({
                "Time (UTC)":  ts,
                "Signal":      act,
                "Entry":       f"${h.get('entry', 0):.2f}",
                "SL":          f"${h.get('stop_loss', 0):.2f}" if h.get("stop_loss") else "-",
                "Target":      f"${h.get('target', 0):.2f}"    if h.get("target") else "-",
                "R:R":         f"1:{h.get('risk_reward', 0):.1f}" if h.get("risk_reward") else "-",
                "Score":       f"{h.get('score', 0)}/10",
                "Session":     h.get("session", "-"),
                "Trend":       h.get("trend_1h", "-"),
            })

        df_hist = pd.DataFrame(rows)

        def style_signal(val):
            if val == "BUY":  return "color: #00d4aa; font-weight: bold"
            if val == "SELL": return "color: #ff4b4b; font-weight: bold"
            return "color: #ffa500"

        st.dataframe(
            df_hist.style.applymap(style_signal, subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )

        # Win rate (if we have history)
        trades = [h for h in history if h.get("action") in ("BUY", "SELL")]
        st.metric("Total signals logged", len(history))
        st.metric("Actionable signals (BUY/SELL)", len(trades))

    st.markdown("---")
    st.caption("History stored at `/tmp/xauusd_history.json` — persists until system restart.")
