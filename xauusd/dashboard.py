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


def load_nifty_signal() -> dict | None:
    try:
        with open("/tmp/nifty_signal.json") as f:
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
tabs = st.tabs([
    "📊 Live Signal", "📈 Chart", "🌊 Wave Analysis",
    "🔬 Order Flow L2", "🛡️ Hedge System",
    "⚡ GEX + IV Surface", "🌐 Regime + Breadth + Sector",
    "🇮🇳 India + News", "🤖 Bull vs Bear", "📋 History",
    "🏦 FII Tape + DOM", "🧠 OF Strategy"
])


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

    # ── Nifty F&O Panel (clubbed into same tab) ───────────────────────────────
    st.markdown("### 🇮🇳 Nifty F&O Signal")
    nifty_sig = load_nifty_signal()

    if nifty_sig:
        n_action = nifty_sig.get("action", "WAIT")
        n_color  = {"BUY_CE": "#00d4aa", "BUY_PE": "#ff4b4b", "WAIT": "#ffa500"}.get(n_action, "#ffa500")
        n_emoji  = {"BUY_CE": "🟢", "BUY_PE": "🔴", "WAIT": "⏳"}.get(n_action, "⏳")

        nc1, nc2, nc3, nc4, nc5, nc6 = st.columns(6)
        nc1.metric("Nifty Spot", f"₹{nifty_sig.get('spot', 0):,.0f}")
        nc2.metric("PCR", f"{nifty_sig.get('pcr', 0):.2f}")
        nc3.metric("India VIX", f"{nifty_sig.get('vix', 0):.2f}")
        nc4.metric("Strike", f"₹{nifty_sig.get('strike', 0):,.0f}")
        nc5.metric("Tape Bias", f"{nifty_sig.get('tape_bias', 0):+.2f}")
        expiry_str = nifty_sig.get("expiry", "")
        nc6.metric("Expiry", expiry_str[:10] if expiry_str else "-")

        nl, nr = st.columns([1, 2])
        with nl:
            st.markdown(
                f"<h3 style='color:{n_color}'>{n_emoji} {n_action}</h3>",
                unsafe_allow_html=True,
            )
            conf = nifty_sig.get("confidence", 0)
            st.progress(min(float(conf), 1.0), text=f"Confidence: {conf:.0%}")
            if n_action not in ("WAIT",) and nifty_sig.get("entry"):
                st.metric("Entry",     f"₹{nifty_sig.get('entry', 0):,.2f}")
                st.metric("Stop Loss", f"₹{nifty_sig.get('stop_loss', 0):,.2f}")
                st.metric("Target",    f"₹{nifty_sig.get('target', 0):,.2f}")
        with nr:
            ts = nifty_sig.get("timestamp", "")[:19].replace("T", " ")
            sym = nifty_sig.get("symbol", "NIFTY")
            st.caption(f"Signal: **{sym}** | Updated: {ts} | Loop engine must be running")
            # Quick interpretation
            pcr = nifty_sig.get("pcr", 1.0)
            vix = nifty_sig.get("vix", 15)
            tape = nifty_sig.get("tape_bias", 0)
            notes = []
            if pcr > 1.3:   notes.append("PCR >1.3 → Put heavy → bullish lean")
            elif pcr < 0.7: notes.append("PCR <0.7 → Call heavy → bearish lean")
            if vix > 20:    notes.append(f"VIX {vix:.1f} → High vol → options expensive")
            elif vix < 12:  notes.append(f"VIX {vix:.1f} → Low vol → sell premium")
            if tape > 0.3:  notes.append(f"Tape bias +{tape:.2f} → Buy flow dominant")
            elif tape < -0.3: notes.append(f"Tape bias {tape:.2f} → Sell flow dominant")
            for note in notes:
                st.markdown(f"- {note}")
    else:
        st.info("No Nifty signal yet — start `python3 loop_engine.py` to generate signals")
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Nifty Spot", "—")
        nc2.metric("PCR", "—")
        nc3.metric("India VIX", "—")

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


# ══ TAB 3: WAVE ANALYSIS ═════════════════════════════════════════════════════
with tabs[2]:

    st.markdown("## 🌊 Motive Wave Analysis")
    st.caption("Elliott Waves · Fibonacci · Harmonics · Market Structure — powered by custom wave engine")

    # ── Controls ──────────────────────────────────────────────────────────────
    wc1, wc2, wc3, wc4 = st.columns(4)
    wave_tf      = wc1.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    wave_window  = wc2.slider("Swing sensitivity", 3, 15, 5,
                               help="Higher = fewer, larger swings")
    wave_min_move= wc3.slider("Min swing %", 0.001, 0.02, 0.003, step=0.001, format="%.3f",
                               help="Minimum price move to count as a swing")
    wave_period  = wc4.selectbox("Period", ["5d","15d","30d","60d","90d"], index=2)

    @st.cache_data(ttl=120)
    def fetch_wave_data(tf, period):
        from xauusd.data import get_bars
        return get_bars(tf, period)

    with st.spinner("Running wave analysis..."):
        try:
            wave_df = fetch_wave_data(wave_tf, wave_period)
            from xauusd.wave import analyze_waves
            wa = analyze_waves(wave_df, window=wave_window, min_move_pct=wave_min_move)
            wave_ok = True
        except Exception as we:
            st.error(f"Wave analysis error: {we}")
            wa = {}
            wave_ok = False

    if wave_ok and wa.get("pivots"):
        pivots    = wa["pivots"]
        fibs      = wa["fib_levels"]
        elliott   = wa["elliott"]
        harmonics = wa["harmonics"]
        structure = wa["structure"]
        summary   = wa["summary"]

        # ── Summary strip ────────────────────────────────────────────────────
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Structure",    summary.get("trend", "N/A"))
        s2.metric("Wave Position",summary.get("wave_position", "N/A"))
        s3.metric("Wave Conf",    summary.get("wave_conf", "N/A"))
        s4.metric("Last BOS/CHoCH", summary.get("last_bos", "None")[:30] if summary.get("last_bos") else "None")
        s5.metric("Harmonic",     f"{summary.get('harmonic','None')} {summary.get('harmonic_dir','')}".strip())

        st.markdown("---")

        # ── Main Wave Chart ───────────────────────────────────────────────────
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        chart_df = wave_df.copy().tail(300)
        chart_df.columns = [c.lower() for c in chart_df.columns]

        fig_wave = go.Figure()

        # Candlesticks
        fig_wave.add_trace(go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"],   close=chart_df["close"],
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4b4b",
            name="Price",
            showlegend=False,
        ))

        # Fibonacci levels
        if fibs:
            fib_colors = {
                "23.6%": "#9b59b6", "38.2%": "#3498db",
                "50.0%": "#f1c40f", "61.8%": "#e67e22",
                "78.6%": "#e74c3c", "100.0%": "#95a5a6",
                "161.8%": "#1abc9c", "261.8%": "#2ecc71",
            }
            for fib in fibs:
                col = fib_colors.get(fib.label, "#888888")
                dash = "dot" if fib.is_ext else "dash"
                fig_wave.add_hline(
                    y=fib.price,
                    line_dash=dash, line_color=col, line_width=1,
                    annotation_text=f"Fib {fib.label} ${fib.price:.2f}",
                    annotation_font_color=col,
                    annotation_font_size=10,
                )

        # Pivot markers
        chart_start = chart_df.index[0]
        for pv in pivots:
            if pv.ts >= chart_start:
                color  = "#00d4aa" if pv.kind == "H" else "#ff4b4b"
                symbol = "triangle-down" if pv.kind == "H" else "triangle-up"
                label  = pv.label or pv.kind
                fig_wave.add_trace(go.Scatter(
                    x=[pv.ts], y=[pv.price],
                    mode="markers+text",
                    marker=dict(color=color, size=10, symbol=symbol),
                    text=[label], textposition="top center" if pv.kind == "H" else "bottom center",
                    textfont=dict(color=color, size=10),
                    name=f"Pivot {pv.kind}",
                    showlegend=False,
                ))

        # Elliott wave lines connecting pivots
        if elliott and elliott.waves:
            for wave in elliott.waves:
                wcolor = {
                    "1": "#4fa3e0", "2": "#ffa500", "3": "#00d4aa",
                    "4": "#ff8c00", "5": "#a78bfa",
                    "A": "#ff4b4b", "B": "#ffa500", "C": "#e74c3c",
                }.get(wave.label, "#888")
                mid_price = (wave.start.price + wave.end.price) / 2
                fig_wave.add_trace(go.Scatter(
                    x=[wave.start.ts, wave.end.ts],
                    y=[wave.start.price, wave.end.price],
                    mode="lines+text",
                    line=dict(color=wcolor, width=2, dash="solid"),
                    text=["", f"W{wave.label}"],
                    textposition="top right",
                    textfont=dict(color=wcolor, size=12, family="Arial Black"),
                    name=f"Wave {wave.label}",
                    showlegend=True,
                ))

        # Harmonic pattern — draw XABCD lines
        if harmonics:
            h = harmonics[0]
            h_pts  = [h.X, h.A, h.B, h.C, h.D]
            h_ts   = [p.ts    for p in h_pts]
            h_pxs  = [p.price for p in h_pts]
            h_lbls = ["X", "A", "B", "C", "D"]
            hcol   = "#00d4aa" if h.direction == "BULLISH" else "#ff4b4b"
            fig_wave.add_trace(go.Scatter(
                x=h_ts, y=h_pxs,
                mode="lines+markers+text",
                line=dict(color=hcol, width=2, dash="dot"),
                marker=dict(color=hcol, size=8),
                text=h_lbls,
                textposition="top center",
                textfont=dict(color=hcol, size=11),
                name=f"{h.name} ({h.direction})",
            ))
            # PRZ zone
            fig_wave.add_hrect(
                y0=h.prz_low, y1=h.prz_high,
                fillcolor=hcol, opacity=0.08,
                annotation_text=f"PRZ {h.name}",
                annotation_font_color=hcol,
            )

        # Wave projections
        if elliott and elliott.projections:
            proj_colors = {
                "W3_proj_161": "#00d4aa", "W3_proj_200": "#1abc9c",
                "W5_proj_eq_W1": "#a78bfa", "W5_proj_61pct": "#9b59b6",
                "C_proj_100pct_A": "#ff4b4b", "C_proj_61pct_A": "#e74c3c",
            }
            for key, price in elliott.projections.items():
                col = proj_colors.get(key, "#ffa500")
                fig_wave.add_hline(
                    y=price,
                    line_dash="longdash", line_color=col, line_width=1.5,
                    annotation_text=f"→ {key.replace('_',' ')} ${price:.2f}",
                    annotation_font_color=col,
                )

        fig_wave.update_layout(
            height=600,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", borderwidth=1),
            margin=dict(l=10, r=120, t=30, b=10),
            title=f"XAUUSD {wave_tf} — Wave Analysis",
        )
        fig_wave.update_xaxes(gridcolor="#1e2130")
        fig_wave.update_yaxes(gridcolor="#1e2130")

        st.plotly_chart(fig_wave, use_container_width=True)

        # ── Elliott Wave Details ──────────────────────────────────────────────
        col_ew, col_harm = st.columns(2)

        with col_ew:
            st.markdown("### 🌊 Elliott Wave Count")
            if elliott:
                ew_type = "Impulse (1-2-3-4-5)" if elliott.is_impulse else "Correction (A-B-C)"
                ew_col  = "#00d4aa" if elliott.direction == "UP" else "#ff4b4b"
                st.markdown(f"**Type:** {ew_type}")
                st.markdown(f"**Direction:** <span style='color:{ew_col}'>{elliott.direction}</span>",
                            unsafe_allow_html=True)
                st.metric("Confidence", f"{elliott.confidence:.0%}")
                st.metric("Current Wave", f"Wave {elliott.current_wave}")

                if not elliott.rules_ok:
                    for v in elliott.violations:
                        st.warning(f"Rule violation: {v}")
                else:
                    st.success("All Elliott rules satisfied")

                if elliott.projections:
                    st.markdown("**Projections:**")
                    for k, v in elliott.projections.items():
                        st.markdown(f"- `{k.replace('_',' ')}`: **${v:,.2f}**")

                if elliott.waves:
                    rows = []
                    for w in elliott.waves:
                        rows.append({
                            "Wave": w.label,
                            "From":   f"${w.start.price:,.2f}",
                            "To":     f"${w.end.price:,.2f}",
                            "Move":   f"${abs(w.end.price - w.start.price):,.2f}",
                            "Retrace/Ext": f"{w.retrace:.1%}" if w.retrace else f"{w.extend:.2f}x",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough pivots to count Elliott waves — try reducing swing sensitivity")

        with col_harm:
            st.markdown("### 🔺 Harmonic Patterns")
            if harmonics:
                for h in harmonics[:3]:
                    h_col = "#00d4aa" if h.direction == "BULLISH" else "#ff4b4b"
                    st.markdown(f"**{h.name}** — <span style='color:{h_col}'>{h.direction}</span>  ({h.confidence:.0%})",
                                unsafe_allow_html=True)
                    hc1, hc2, hc3 = st.columns(3)
                    hc1.metric("PRZ Low",   f"${h.prz_low:,.2f}")
                    hc2.metric("PRZ High",  f"${h.prz_high:,.2f}")
                    hc3.metric("Stop Loss", f"${h.stop_loss:,.2f}")
                    hh1, hh2 = st.columns(2)
                    hh1.metric("Target 1 (38.2%)", f"${h.target_1:,.2f}")
                    hh2.metric("Target 2 (61.8%)", f"${h.target_2:,.2f}")
                    st.markdown("---")
            else:
                st.info("No harmonic patterns detected in recent pivots")

        # ── Market Structure ──────────────────────────────────────────────────
        st.markdown("### 🏗️ Market Structure — BOS / CHoCH")
        if structure:
            ms_col = {"UPTREND": "#00d4aa", "DOWNTREND": "#ff4b4b", "RANGING": "#ffa500"}.get(structure.trend, "#888")
            st.markdown(f"**Trend:** <h3 style='color:{ms_col};display:inline'>{structure.trend}</h3>",
                        unsafe_allow_html=True)

            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Recent BOS (Break of Structure):**")
                if structure.bos:
                    for b in reversed(structure.bos[-3:]):
                        bcol = "#00d4aa" if b["direction"] == "UP" else "#ff4b4b"
                        st.markdown(f"<span style='color:{bcol}'>→ {b['desc']}</span>",
                                    unsafe_allow_html=True)
                else:
                    st.caption("No BOS detected")

            with mc2:
                st.markdown("**Recent CHoCH (Change of Character):**")
                if structure.choch:
                    for c in reversed(structure.choch[-3:]):
                        ccol = "#00d4aa" if c["direction"] == "UP" else "#ff4b4b"
                        st.markdown(f"<span style='color:{ccol}'>⚡ {c['desc']}</span>",
                                    unsafe_allow_html=True)
                else:
                    st.caption("No CHoCH detected")

            # HH/HL/LH/LL table
            if structure.hh_hl or structure.lh_ll:
                all_ms = sorted(
                    structure.hh_hl[-6:] + structure.lh_ll[-6:],
                    key=lambda x: x["ts"],
                )
                rows = [{"Type": x["type"], "Price": f"${x['price']:,.2f}"} for x in all_ms[-10:]]
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Fibonacci Table ───────────────────────────────────────────────────
        if fibs:
            st.markdown("### 📐 Fibonacci Levels (last major swing)")
            current_price = float(chart_df["close"].iloc[-1]) if not chart_df.empty else 0
            fib_rows = []
            for f in fibs:
                dist = current_price - f.price
                fib_rows.append({
                    "Ratio":    f.label,
                    "Price":    f"${f.price:,.2f}",
                    "Distance": f"{dist:+.2f}",
                    "Type":     "Extension" if f.is_ext else "Retracement",
                    "Status":   "ABOVE" if dist > 0 else "BELOW",
                })
            st.dataframe(pd.DataFrame(fib_rows), use_container_width=True, hide_index=True)

    else:
        st.warning("Wave analysis needs at least 30 bars of data. Try a longer period or smaller swing window.")


# ══ TAB 4: ORDER FLOW L2 ═════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔬 Order Flow — Level 2 Analysis")
    st.caption("GlobalDataFeeds L2 (set GDF_USER/GDF_PASS in .env) or NSE simulated fallback")

    # GDF credentials status
    import os as _os
    gdf_user = _os.getenv("GDF_USER", "")
    if gdf_user:
        st.success(f"GlobalDataFeeds connected as: {gdf_user}")
    else:
        st.warning("GDF_USER / GDF_PASS not set — using NSE simulated L2. Add to `.env` for real L2 data.")
        st.code("GDF_USER=your_username\nGDF_PASS=your_password\n# Get credentials: globaldatafeeds.in", language="bash")

    # Symbol selector
    of_col1, of_col2 = st.columns([2, 1])
    of_symbol = of_col1.selectbox(
        "Symbol", ["NIFTY25AUGFUT", "BANKNIFTY25AUGFUT", "NIFTY", "BANKNIFTY"],
        help="Use exact GDF symbol names for real L2 data",
    )
    bar_secs = of_col2.selectbox("Bar size", [30, 60, 120, 300], index=1, format_func=lambda x: f"{x}s")

    @st.cache_data(ttl=15)
    def fetch_order_flow(symbol, bar_s):
        from core.order_flow.order_flow_engine import OrderFlowEngine
        engine = OrderFlowEngine(symbol, bar_seconds=bar_s)
        engine.update()
        return engine.summary_dict()

    with st.spinner("Fetching L2 order book..."):
        try:
            of = fetch_order_flow(of_symbol, bar_secs)
            of_ok = True
        except Exception as of_err:
            st.error(f"Order flow error: {of_err}")
            of_ok = False
            of = {}

    if of_ok:
        # ── Composite signal strip ────────────────────────────────────────────
        bias  = of.get("composite_bias", "NEUTRAL")
        score = of.get("aggression_score", 5.0)
        of_sig  = of.get("trade_signal", "WAIT")
        b_col = {"BULLISH": "#00d4aa", "BEARISH": "#ff4b4b"}.get(bias, "#ffa500")
        s_emoji = {"LONG": "🟢", "SHORT": "🔴", "WAIT": "⏳"}.get(of_sig, "⏳")

        of1, of2, of3, of4, of5, of6 = st.columns(6)
        of1.metric("LTP",            f"₹{of.get('ltp',0):,.1f}")
        of2.metric("OBI Signal",      of.get("obi_signal", "N/A"))
        of3.metric("Delta (total)",   f"{of.get('cumulative_delta',0):+,.0f}")
        of4.metric("Delta (1min)",    f"{of.get('delta_1min',0):+,.0f}")
        of5.metric("VWAP",            f"₹{of.get('vwap',0):,.1f}")
        of6.metric("POC",             f"₹{of.get('poc',0):,.1f}")

        # Big signal card
        st.markdown(
            f"<h2 style='color:{b_col}'>{s_emoji} {of_sig} — {bias}  |  Score: {score}/10</h2>",
            unsafe_allow_html=True,
        )
        st.progress(score / 10, text=f"Order Flow Aggression: {score}/10  (0=max sell, 10=max buy)")
        if of.get("signal_reason"):
            st.caption(f"Reasons: {of['signal_reason']}")

        st.markdown("---")

        # ── Order Book Visualisation ──────────────────────────────────────────
        col_book, col_delta = st.columns(2)

        with col_book:
            st.markdown("### 📖 Level 2 Order Book")
            bids = of.get("bids", [])
            asks = of.get("asks", [])
            imb  = of.get("imbalance", 0)
            imb_col = "#00d4aa" if imb > 0 else "#ff4b4b"

            st.markdown(f"**Imbalance:** <span style='color:{imb_col}'>{imb:+.3f}</span>  "
                        f"| Stacked Bids: {of.get('stacked_bids',0)} | Stacked Asks: {of.get('stacked_asks',0)}",
                        unsafe_allow_html=True)

            if bids or asks:
                fig_book = go.Figure()
                if asks:
                    ask_px  = [a[0] for a in reversed(asks[:5])]
                    ask_qty = [a[1] for a in reversed(asks[:5])]
                    fig_book.add_trace(go.Bar(
                        y=ask_px, x=ask_qty, orientation="h",
                        marker_color="#ff4b4b", name="Asks", opacity=0.8,
                    ))
                if bids:
                    bid_px  = [b[0] for b in bids[:5]]
                    bid_qty = [b[1] for b in bids[:5]]
                    fig_book.add_trace(go.Bar(
                        y=bid_px, x=bid_qty, orientation="h",
                        marker_color="#00d4aa", name="Bids", opacity=0.8,
                    ))
                fig_book.update_layout(
                    height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="white", barmode="overlay",
                    xaxis_title="Quantity", yaxis_title="Price",
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_book, use_container_width=True)

                # Table view
                rows = []
                for i in range(5):
                    b = bids[i] if i < len(bids) else ("—", 0)
                    a = asks[i] if i < len(asks) else ("—", 0)
                    rows.append({
                        "Bid Qty": f"{b[1]:,}" if isinstance(b[1], int) else "—",
                        "Bid Price": f"₹{b[0]:,.2f}" if b[0] != "—" else "—",
                        "Ask Price": f"₹{a[0]:,.2f}" if a[0] != "—" else "—",
                        "Ask Qty": f"{a[1]:,}" if isinstance(a[1], int) else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Waiting for L2 data... (loop engine must be running with GDF feed)")

        with col_delta:
            st.markdown("### 📊 Cumulative Delta")
            vwap_pos = of.get("vwap_position", "N/A")
            vwap_col = {"ABOVE_VWAP": "#00d4aa", "BELOW_VWAP": "#ff4b4b"}.get(vwap_pos, "#ffa500")

            dv1, dv2, dv3 = st.columns(3)
            dv1.metric("VWAP Position", vwap_pos)
            dv2.metric("VAH (Value Area High)", f"₹{of.get('vah',0):,.1f}")
            dv3.metric("VAL (Value Area Low)",  f"₹{of.get('val',0):,.1f}")

            d_sig = of.get("delta_signal", "NEUTRAL")
            d_col = {"BULLISH": "#00d4aa", "BEARISH": "#ff4b4b"}.get(d_sig, "#ffa500")
            st.markdown(f"**Delta Signal:** <span style='color:{d_col}'>{d_sig}</span>",
                        unsafe_allow_html=True)
            if of.get("delta_divergence"):
                st.warning("⚡ Delta Divergence: price and delta moving opposite — potential reversal")

            # Delta gauge
            cum_d = of.get("cumulative_delta", 0)
            max_d = max(abs(cum_d), 1000)
            fig_d = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=cum_d,
                title={"text": "Cumulative Delta"},
                delta={"reference": 0},
                gauge={
                    "axis": {"range": [-max_d, max_d]},
                    "bar":  {"color": "#00d4aa" if cum_d > 0 else "#ff4b4b"},
                    "steps": [
                        {"range": [-max_d, 0], "color": "#1a0a0a"},
                        {"range": [0, max_d],  "color": "#0a1a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": 0},
                },
            ))
            fig_d.update_layout(height=280, paper_bgcolor="#0e1117", font_color="white", margin=dict(t=40, b=0))
            st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("---")

        # ── Event Alerts ──────────────────────────────────────────────────────
        ev1, ev2, ev3 = st.columns(3)

        with ev1:
            st.markdown("### 🧊 Iceberg / Large Orders")
            icebergs = of.get("icebergs", [])
            if icebergs:
                for ice in icebergs:
                    st.markdown(f"- {ice}")
            else:
                st.caption("No iceberg orders detected")

        with ev2:
            st.markdown("### 🧲 Absorption")
            absorptions = of.get("absorptions", [])
            if absorptions:
                for ab in absorptions:
                    st.markdown(f"- {ab}")
            else:
                st.caption("No absorption events detected")

        with ev3:
            st.markdown("### 🎯 Stop Hunt Alerts")
            stop_hunts = of.get("stop_hunts", [])
            if stop_hunts:
                for sh in stop_hunts:
                    st.warning(sh)
            else:
                st.caption("No stop hunts detected")

        st.markdown("---")

        # ── Strategy summary ──────────────────────────────────────────────────
        st.markdown("### 📋 Order Flow Strategies — Reading Guide")
        st.markdown("""
| Signal | What it means | Trade Action |
|--------|--------------|--------------|
| **OBI BUY_PRESSURE** | More bid qty than ask qty stacked | Look for LONG entry |
| **OBI SELL_PRESSURE** | More ask qty than bid qty stacked | Look for SHORT entry |
| **Stacked Imbalances** | 3+ consecutive levels same side | Strong directional bias |
| **Delta BULLISH** | Buy trades overwhelming sells | Buyers in control |
| **Delta Divergence** | Price rising but delta negative | Weak rally, possible reversal |
| **Absorption (BUY)** | Large buy volume, price not rising | Sellers absorbing buyers |
| **Iceberg at price** | Repeated hidden fills at same level | Large player accumulating |
| **Stop Hunt HIGH** | Price swept above key high then reversed | Short opportunity |
| **Stop Hunt LOW** | Price swept below key low then reversed | Long opportunity |
| **Above VWAP** | Price above session VWAP | Bullish bias, buy dips to VWAP |
| **Below VWAP** | Price below session VWAP | Bearish bias, sell rallies to VWAP |
        """)

        st.caption(f"Updated: {of.get('timestamp','—')} | Refresh rate: 15s | Symbol: {of_symbol}")
        if st.button("🔄 Refresh Order Flow", key="of_refresh"):
            st.cache_data.clear()
            st.rerun()


# ══ TAB 5: 0-LOSS HEDGE ══════════════════════════════════════════════════════
with tabs[4]:

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


# ══ TAB 6: GEX + IV SURFACE ══════════════════════════════════════════════════
with tabs[5]:

    st.markdown("## ⚡ GEX — Gamma Exposure  +  📊 IV Surface")
    st.caption("Nifty F&O data from NSE | Refreshed every 60s")

    @st.cache_data(ttl=60)
    def fetch_nifty_chain():
        from core.order_flow.oi_tracker import OITracker
        t = OITracker("NIFTY")
        t.tick_tape()
        s = t.get_market_summary()
        return s.get("df"), s.get("spot", 24000), s.get("expiry", "")

    with st.spinner("Fetching Nifty option chain..."):
        try:
            chain_df, nifty_spot, nifty_expiry = fetch_nifty_chain()
            chain_ok = chain_df is not None and not chain_df.empty
        except Exception as e:
            st.error(f"Chain fetch error: {e}")
            chain_ok = False
            chain_df, nifty_spot, nifty_expiry = None, 24000, ""

    # ── GEX ──────────────────────────────────────────────────────────────────
    st.markdown("### ⚡ Gamma Exposure (GEX)")

    if chain_ok:
        try:
            from core.options.gex import compute_gex
            from datetime import datetime as dt
            try:
                exp_date    = dt.strptime(nifty_expiry, "%d-%m-%Y").date()
                expiry_days = max(1, (exp_date - dt.now().date()).days)
            except Exception:
                expiry_days = 1

            gex = compute_gex(chain_df, nifty_spot, expiry_days)
            gex_df = gex.get("by_strike")

            # GEX summary metrics
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Spot", f"₹{nifty_spot:,.0f}")
            g2.metric("Net GEX", f"{gex['net_total_gex']:+.2f}B")
            g3.metric("Gamma Flip", f"₹{gex['gamma_flip']:.0f}")
            g4.metric("Dealer Bias", gex["dealer_bias"])
            g5.metric("Expected Vol", gex["expected_volatility"])

            # GEX bar chart
            if gex_df is not None and not gex_df.empty:
                fig_gex = go.Figure()
                colors = ["#00d4aa" if v >= 0 else "#ff4b4b" for v in gex_df["net_gex"]]
                fig_gex.add_trace(go.Bar(
                    x=gex_df["strike"], y=gex_df["net_gex"],
                    marker_color=colors, name="Net GEX",
                ))
                fig_gex.add_vline(x=nifty_spot, line_dash="dash", line_color="white",
                                  annotation_text=f"Spot {nifty_spot:.0f}")
                fig_gex.add_hline(y=0, line_color="#888", line_width=1)
                if gex["gamma_flip"]:
                    fig_gex.add_vline(x=gex["gamma_flip"], line_dash="dot", line_color="#ffa500",
                                      annotation_text=f"Flip {gex['gamma_flip']:.0f}")
                fig_gex.update_layout(
                    title="Net GEX by Strike (₹ Billions)",
                    height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="white", xaxis_title="Strike", yaxis_title="Net GEX (B)",
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_gex, use_container_width=True)

                # Interpretation
                bias = gex["dealer_bias"]
                if bias == "LONG_GAMMA":
                    st.success("🔒 Dealers LONG gamma → Market likely to **mean-revert** and pin near gamma flip. Sell options / straddle favorable.")
                elif bias == "SHORT_GAMMA":
                    st.error("⚡ Dealers SHORT gamma → Market can **accelerate** in either direction. Directional trades preferred.")
                else:
                    st.info("⚖️ Gamma NEUTRAL — No strong dealer bias.")
        except Exception as e:
            st.error(f"GEX error: {e}")
    else:
        st.warning("Option chain data not available")

    st.markdown("---")

    # ── IV Surface ────────────────────────────────────────────────────────────
    st.markdown("### 📊 IV Surface — Smile, Skew, Rank")

    if chain_ok:
        try:
            from core.options.iv_surface import get_iv_surface
            ivs_obj = get_iv_surface()
            ivs = ivs_obj.full_summary(chain_df, nifty_spot)

            smile_df = ivs.get("smile")
            atm      = ivs.get("atm", {})
            iv_rank  = ivs.get("iv_rank", {})

            # IV Rank metrics
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("India VIX", f"{iv_rank.get('vix_current', 0):.2f}")
            r2.metric("IV Rank",   f"{iv_rank.get('iv_rank', 0):.0f}/100")
            r3.metric("IV %ile",   f"{iv_rank.get('iv_percentile', 0):.0f}%")
            r4.metric("ATM IV",    f"{atm.get('atm_iv', 0):.1f}%")
            r5.metric("IV Regime", iv_rank.get("regime", "N/A"))

            st.info(f"**Strategy:** {ivs.get('strategy_hint', '')}")

            if smile_df is not None and not smile_df.empty:
                col_smile, col_skew = st.columns(2)

                with col_smile:
                    # IV Smile chart
                    fig_smile = go.Figure()
                    pe_iv_clean = smile_df["pe_iv"].dropna()
                    ce_iv_clean = smile_df["ce_iv"].dropna()
                    fig_smile.add_trace(go.Scatter(
                        x=smile_df["strike"], y=smile_df["pe_iv"],
                        name="PE IV", line=dict(color="#ff4b4b", width=2),
                        mode="lines+markers",
                    ))
                    fig_smile.add_trace(go.Scatter(
                        x=smile_df["strike"], y=smile_df["ce_iv"],
                        name="CE IV", line=dict(color="#00d4aa", width=2),
                        mode="lines+markers",
                    ))
                    fig_smile.add_vline(x=nifty_spot, line_dash="dash", line_color="white",
                                        annotation_text="Spot")
                    fig_smile.update_layout(
                        title="IV Smile", height=320,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                        xaxis_title="Strike", yaxis_title="IV %", margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)

                with col_skew:
                    # Skew chart (PE - CE)
                    fig_skew = go.Figure()
                    skew_colors = ["#ff4b4b" if v > 0 else "#00d4aa" for v in smile_df["skew"].fillna(0)]
                    fig_skew.add_trace(go.Bar(
                        x=smile_df["strike"], y=smile_df["skew"],
                        marker_color=skew_colors, name="PE-CE Skew",
                    ))
                    fig_skew.add_hline(y=0, line_color="#888", line_width=1)
                    fig_skew.update_layout(
                        title="PE vs CE Skew (PE-CE IV)", height=320,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                        xaxis_title="Strike", yaxis_title="Skew (IV%)", margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_skew, use_container_width=True)

            # Historical vol term structure
            st.markdown("#### HV Term Structure (India VIX history)")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("HV 7D",  f"{iv_rank.get('hv_7d', 0):.1f}%")
            t2.metric("HV 14D", f"{iv_rank.get('hv_14d', 0):.1f}%")
            t3.metric("HV 30D", f"{iv_rank.get('hv_30d', 0):.1f}%")
            t4.metric("HV 90D", f"{iv_rank.get('hv_90d', 0):.1f}%")
            st.caption(f"Term slope: {iv_rank.get('term_slope', 'N/A')}  |  "
                       f"Smile shape: {atm.get('smile_shape', 'N/A')}  |  "
                       f"Risk Reversal: {atm.get('risk_reversal', 0):+.2f}")

        except Exception as e:
            st.error(f"IV surface error: {e}")


# ══ TAB 7: REGIME + BREADTH + SECTOR ═════════════════════════════════════════
with tabs[6]:

    st.markdown("## 🌐 Market Intelligence — Regime · Breadth · Sectors · Calendar")

    # ── Regime ────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Market Regime")

    @st.cache_data(ttl=120)
    def fetch_regime():
        from core.order_flow.oi_tracker import OITracker
        from core.agents.regime import detect_regime
        t = OITracker("NIFTY")
        t.tick_tape()
        f = t.get_model_features()
        return detect_regime(f, iv_rank=50, vix=f.get("f_vix", 15))

    try:
        regime = fetch_regime()
        r_color = {
            "TRENDING_UP": "#00d4aa", "TRENDING_DOWN": "#ff4b4b",
            "HIGH_VOLATILITY": "#ff8c00", "LOW_VOLATILITY": "#4fa3e0",
            "RANGING": "#ffa500", "EVENT_DRIVEN": "#a78bfa",
        }.get(regime["regime"], "#888")

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(f"<h2 style='color:{r_color}'>{regime['regime']}</h2>", unsafe_allow_html=True)
            st.metric("Confidence", f"{regime['confidence']:.0f}%")
            st.markdown(f"**Strategy:** `{regime['strategy']}`")
            st.caption(regime["description"])
        with rc2:
            inds = regime.get("indicators", {})
            i1, i2, i3 = st.columns(3)
            i1.metric("Tape Bias",  f"{inds.get('tape_bias', 0):+.2f}")
            i2.metric("FII Bias",   f"{inds.get('fii_bias', 0):+.2f}")
            i3.metric("PCR",        f"{inds.get('pcr', 1):.2f}")
            i4, i5, i6 = st.columns(3)
            i4.metric("IV Rank",    f"{inds.get('iv_rank', 50):.0f}")
            i5.metric("VIX",        f"{inds.get('vix', 15):.1f}")
            i6.metric("Bull %",     f"{inds.get('bull_pct', 0.5):.0%}")
            if regime.get("reasons"):
                st.caption("Signals: " + " | ".join(regime["reasons"]))
    except Exception as e:
        st.error(f"Regime error: {e}")

    st.markdown("---")

    # ── Event Calendar ────────────────────────────────────────────────────────
    st.markdown("### 📅 Event Risk Calendar")

    @st.cache_data(ttl=3600)
    def fetch_events():
        from core.agents.calendar import get_event_risk
        return get_event_risk("NIFTY")

    try:
        evr = fetch_events()
        ev_color = "#ff4b4b" if evr["score"] >= 7 else "#ffa500" if evr["score"] >= 4 else "#00d4aa"
        st.markdown(f"**{evr['date']} ({evr['weekday']})** — "
                    f"<span style='color:{ev_color}'>Event Risk Score: {evr['score']}/10</span>",
                    unsafe_allow_html=True)
        st.info(evr["recommendation"])
        if evr["events_today"]:
            st.markdown("**Today:**")
            for e in evr["events_today"]:
                st.markdown(f"- {e}")
        if evr["events_upcoming"]:
            st.markdown("**Upcoming:**")
            for e in evr["events_upcoming"]:
                st.markdown(f"- {e}")
        st.caption(f"F&O Expiry in {evr['fo_expiry_in_days']} day(s)")
    except Exception as e:
        st.error(f"Calendar error: {e}")

    st.markdown("---")

    # ── Market Breadth ────────────────────────────────────────────────────────
    st.markdown("### 📊 Market Breadth (Nifty 50)")
    st.caption("Cached 15min — takes ~30s to compute")

    @st.cache_data(ttl=900)
    def fetch_breadth():
        from core.agents.breadth import get_breadth
        return get_breadth()

    if st.button("📊 Load Breadth Data", key="breadth_btn"):
        with st.spinner("Scanning 50 stocks... (~30s)"):
            try:
                b = fetch_breadth()
                st.session_state["breadth"] = b
            except Exception as e:
                st.error(f"Breadth error: {e}")

    if "breadth" in st.session_state:
        b = st.session_state["breadth"]
        b_color = "#00d4aa" if b["signal"] == "BULLISH" else "#ff4b4b" if b["signal"] == "BEARISH" else "#ffa500"
        st.markdown(f"<h3 style='color:{b_color}'>Breadth: {b['signal']} ({b['breadth_score']:.0f}/100)</h3>",
                    unsafe_allow_html=True)
        st.caption(b["description"])

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Advances",        b.get("advances", 0))
        b2.metric("Declines",        b.get("declines", 0))
        b3.metric("Above EMA20",     f"{b.get('above_ema20_pct', 0):.0f}%")
        b4.metric("Above EMA50",     f"{b.get('above_ema50_pct', 0):.0f}%")
        b5.metric("RSI > 50",        f"{b.get('rsi_breadth_pct', 0):.0f}%")

        bc1, bc2 = st.columns(2)
        bc1.metric("New 52W Highs",  b.get("new_52w_high", 0))
        bc2.metric("New 52W Lows",   b.get("new_52w_low", 0))

        # Breadth bar chart
        comp = b.get("component_scores", {})
        if comp:
            fig_b = go.Figure(go.Bar(
                x=list(comp.keys()), y=list(comp.values()),
                marker_color=["#00d4aa" if v > 50 else "#ff4b4b" for v in comp.values()],
            ))
            fig_b.add_hline(y=50, line_dash="dash", line_color="#888")
            fig_b.update_layout(
                title="Breadth Components", height=250,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")

    # ── Sector Rotation ───────────────────────────────────────────────────────
    st.markdown("### 🏭 Sector Rotation")

    @st.cache_data(ttl=900)
    def fetch_sectors():
        from core.agents.sector import get_sector_rotation
        return get_sector_rotation()

    if st.button("🏭 Load Sector Data", key="sector_btn"):
        with st.spinner("Fetching sector performance..."):
            try:
                sr = fetch_sectors()
                st.session_state["sectors"] = sr
            except Exception as e:
                st.error(f"Sector error: {e}")

    if "sectors" in st.session_state:
        sr = st.session_state["sectors"]
        reg_color = "#00d4aa" if sr["rotation_regime"] == "RISK_ON" else "#ff4b4b" if sr["rotation_regime"] == "RISK_OFF" else "#ffa500"
        st.markdown(f"**Regime:** <span style='color:{reg_color}'>{sr['rotation_regime']}</span> — {sr['regime_desc']}",
                    unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**💰 Money IN:**")
            for s in sr.get("money_in", []):
                st.markdown(f"- 🟢 {s}")
        with sc2:
            st.markdown("**💸 Money OUT:**")
            for s in sr.get("money_out", []):
                st.markdown(f"- 🔴 {s}")

        if sr.get("sectors"):
            sdf = pd.DataFrame(sr["sectors"])
            if "sector" in sdf.columns:
                fig_sr = go.Figure()
                colors_1d  = ["#00d4aa" if v > 0 else "#ff4b4b" for v in sdf["ret_1d"]]
                colors_5d  = ["#00d4aa" if v > 0 else "#ff4b4b" for v in sdf["ret_5d"]]
                fig_sr.add_trace(go.Bar(x=sdf["sector"], y=sdf["ret_1d"],  name="1D %",  marker_color=colors_1d, opacity=0.7))
                fig_sr.add_trace(go.Bar(x=sdf["sector"], y=sdf["ret_5d"],  name="5D %",  marker_color=colors_5d, opacity=0.5))
                fig_sr.add_hline(y=0, line_color="#888", line_width=1)
                fig_sr.update_layout(
                    title="Sector Returns", height=320, barmode="group",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_sr, use_container_width=True)

                def color_mom(val):
                    return "color:#00d4aa;font-weight:bold" if val == "UP" else "color:#ff4b4b;font-weight:bold"

                display_cols = ["sector", "ret_1d", "ret_5d", "ret_20d", "rs_score", "momentum"]
                st.dataframe(
                    sdf[display_cols].style.map(color_mom, subset=["momentum"]),
                    use_container_width=True, hide_index=True,
                )


# ══ TAB 8: INDIA + NEWS ═══════════════════════════════════════════════════════
with tabs[7]:

    st.markdown("## 🇮🇳 India Market — News Stocks + Nifty Hedge")

    col_nifty, col_scan = st.columns([1, 2])

    with col_nifty:
        st.markdown("### Nifty F&O Signal")

        # Live signal from loop engine (real-time)
        nifty_live = load_nifty_signal()
        if nifty_live:
            n_action = nifty_live.get("action", "WAIT")
            n_color  = {"BUY_CE": "#00d4aa", "BUY_PE": "#ff4b4b", "WAIT": "#ffa500"}.get(n_action, "#ffa500")
            st.markdown(f"<h2 style='color:{n_color}'>{n_action}</h2>", unsafe_allow_html=True)
            st.metric("Nifty Spot", f"₹{nifty_live.get('spot', 0):,.0f}")
            st.metric("Strike",     f"₹{nifty_live.get('strike', 0):,.0f}")
            st.metric("PCR",        f"{nifty_live.get('pcr', 0):.2f}")
            st.metric("India VIX",  f"{nifty_live.get('vix', 0):.2f}")
            st.metric("Tape Bias",  f"{nifty_live.get('tape_bias', 0):+.2f}")
            if nifty_live.get("entry"):
                st.metric("Entry",     f"₹{nifty_live.get('entry', 0):,.2f}")
                st.metric("Stop Loss", f"₹{nifty_live.get('stop_loss', 0):,.2f}")
                st.metric("Target",    f"₹{nifty_live.get('target', 0):,.2f}")
            ts = nifty_live.get("timestamp", "")[:19].replace("T", " ")
            st.caption(f"Updated: {ts}")
            st.markdown("---")

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
                        df_stocks.style.map(style_trend, subset=["Trend"]),
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


# ══ TAB 9: BULL VS BEAR DEBATE ═══════════════════════════════════════════════
with tabs[8]:

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

    # Check if Ollama is available
    try:
        import ollama as _test_ollama
        _test_ollama.list()
        ollama_ok = True
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        st.warning("⚠️ Ollama not running. Start it with: `ollama serve` — then pull model: `ollama pull llama3.2:3b`")

    if st.button("⚡ Run Bull vs Bear Debate", key="debate_btn", disabled=not ollama_ok):
        with st.spinner("Bull and Bear agents are debating... (~30s)"):
            try:
                from core.agents.debate import run_debate
                result = run_debate(debate_ctx, symbol="XAUUSD", rounds=2)
                st.session_state["debate_result"] = result
            except Exception as e:
                st.error(f"Debate error: {e}")
                st.caption("Make sure Ollama is running (`ollama serve`) and model is pulled (`ollama pull llama3.2:3b`)")

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


# ══ TAB 10: HISTORY + P&L BOARD ══════════════════════════════════════════════
with tabs[9]:

    st.markdown("## 📋 Trade Journal + P&L Board")

    # ── Load trade log ──────────────────────────────────────────────────────
    def load_trade_log() -> list:
        try:
            p = Path(__file__).parent.parent / "data" / "xauusd_trades.json"
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return []

    trade_log = load_trade_log()

    if trade_log:
        closed = [t for t in trade_log if t.get("outcome")]
        wins   = [t for t in closed if t["outcome"] == "TP_HIT"]
        losses = [t for t in closed if t["outcome"] == "SL_HIT"]
        manuals= [t for t in closed if t["outcome"] == "MANUAL"]
        pnls   = [t["pnl_pts"] for t in closed if t.get("pnl_pts") is not None]
        cum_pnl= [sum(pnls[:i+1]) for i in range(len(pnls))]

        # P&L summary strip
        pb1, pb2, pb3, pb4, pb5, pb6 = st.columns(6)
        pb1.metric("Total Trades",  len(closed))
        pb2.metric("Wins",          len(wins),   delta=f"{len(wins)/max(len(closed),1)*100:.0f}% WR")
        pb3.metric("Losses",        len(losses))
        total_pnl = sum(pnls)
        pb4.metric("Total P&L",     f"{total_pnl:+.2f} pts",
                   delta="Profit" if total_pnl > 0 else "Loss",
                   delta_color="normal" if total_pnl > 0 else "inverse")
        pb5.metric("Best Trade",    f"{max(pnls):+.2f}" if pnls else "—")
        pb6.metric("Worst Trade",   f"{min(pnls):+.2f}" if pnls else "—")

        # Equity curve
        if cum_pnl:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                y=cum_pnl, mode="lines+markers",
                line=dict(color="#00d4aa" if cum_pnl[-1] >= 0 else "#ff4b4b", width=2),
                marker=dict(color=["#00d4aa" if p > 0 else "#ff4b4b" for p in pnls], size=8),
                fill="tozeroy",
                fillcolor="rgba(0,212,170,0.1)" if cum_pnl[-1] >= 0 else "rgba(255,75,75,0.1)",
                name="Cumulative P&L (pts)",
            ))
            fig_eq.add_hline(y=0, line_color="#555", line_dash="dash")
            fig_eq.update_layout(
                title="Equity Curve (pts)", height=250,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                margin=dict(t=40, b=20), yaxis_title="Cumulative pts",
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # Trade log table
        st.markdown("### All Trades")
        rows = []
        for t in reversed(closed):
            outcome_emoji = {"TP_HIT": "🎯", "SL_HIT": "🛑", "MANUAL": "✋"}.get(t["outcome"], "?")
            pnl = t.get("pnl_pts", 0) or 0
            rows.append({
                "Time":     (t.get("open_ts") or "")[:16].replace("T"," "),
                "Dir":      t.get("action",""),
                "Entry":    f"${t.get('entry',0):.2f}",
                "SL":       f"${t.get('stop_loss',0):.2f}",
                "Target":   f"${t.get('target',0):.2f}",
                "Close @":  f"${t.get('close_px',0):.2f}",
                "P&L":      f"{pnl:+.2f}",
                "Outcome":  f"{outcome_emoji} {t['outcome']}",
                "Session":  t.get("session",""),
                "Score":    t.get("score",""),
            })
        if rows:
            df_trades = pd.DataFrame(rows)
            def style_pnl(val):
                try:
                    v = float(val)
                    return "color:#00d4aa;font-weight:bold" if v > 0 else "color:#ff4b4b;font-weight:bold"
                except: return ""
            st.dataframe(
                df_trades.style.map(style_pnl, subset=["P&L"]),
                use_container_width=True, hide_index=True,
            )

        # Active trade
        if trade_log and not trade_log[-1].get("outcome"):
            active = trade_log[-1] if isinstance(trade_log[-1], dict) else None
            if active and not active.get("close_ts"):
                st.markdown("---")
                st.markdown("### ⚡ Active Trade")
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("Direction", active.get("action",""))
                ac2.metric("Entry",     f"${active.get('entry',0):.2f}")
                ac3.metric("SL",        f"${active.get('stop_loss',0):.2f}")
                ac4.metric("Target",    f"${active.get('target',0):.2f}")
    else:
        st.info("No completed trades yet — start the engine: `python3 xauusd/engine.py`\nTrades are logged when SL or Target is hit.")

    st.markdown("---")
    st.markdown("### 📊 Signal History")

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
            df_hist.style.map(style_signal, subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )

        # Win rate (if we have history)
        trades = [h for h in history if h.get("action") in ("BUY", "SELL")]
        st.metric("Total signals logged", len(history))
        st.metric("Actionable signals (BUY/SELL)", len(trades))

    st.markdown("---")
    st.caption("History stored at `/tmp/xauusd_history.json` — persists until system restart.")


# ══ TAB 11: FII TAPE + DOM ═══════════════════════════════════════════════════
with tabs[10]:

    st.markdown("## 🏦 FII / DII Option Tape + Order Flow")
    st.caption("Which strikes are institutions writing/buying — updated every 60s during market hours")

    @st.cache_data(ttl=60)
    def fetch_fii_tape_data():
        try:
            from core.data.nse_scraper import NSEScraper
            scraper = NSEScraper()
            raw     = scraper.get_option_chain_raw("NIFTY")
            if not raw:
                return {"error": "No data from NSE"}
            records = raw.get("records", {})
            spot    = float(records.get("underlyingValue", 0))
            rows    = []
            for item in records.get("data", []):
                strike = item.get("strikePrice", 0)
                ce = item.get("CE", {}) or {}
                pe = item.get("PE", {}) or {}
                rows.append({
                    "strikePrice":          strike,
                    "CE_openInterest":      ce.get("openInterest", 0),
                    "CE_changeinOI":        ce.get("changeinOpenInterest", 0),
                    "CE_totalTradedVolume": ce.get("totalTradedVolume", 0),
                    "CE_lastPrice":         ce.get("lastPrice", 0),
                    "CE_impliedVolatility": ce.get("impliedVolatility", 0),
                    "PE_openInterest":      pe.get("openInterest", 0),
                    "PE_changeinOI":        pe.get("changeinOpenInterest", 0),
                    "PE_totalTradedVolume": pe.get("totalTradedVolume", 0),
                    "PE_lastPrice":         pe.get("lastPrice", 0),
                    "PE_impliedVolatility": pe.get("impliedVolatility", 0),
                })
            df = pd.DataFrame(rows)
            return {"df": df, "spot": spot}
        except Exception as e:
            return {"error": str(e)}

    @st.cache_data(ttl=300)
    def fetch_participant_data():
        try:
            from core.data.nse_participant import get_participant_data
            pd_ = get_participant_data()
            return pd_.get_participant_summary()
        except Exception as e:
            return {"error": str(e)}

    # ── Participant Positioning Table ────────────────────────────────────────
    st.markdown("### 👥 Participant Positioning (EOD)")
    part_data = fetch_participant_data()
    if "error" in part_data:
        st.warning(f"Participant data unavailable: {part_data['error']}")
        st.info("NSE participant data is published EOD (~18:00 IST). Ensure market hours or cached data.")
    else:
        participants = part_data.get("participants", {})
        if participants:
            rows_part = []
            for name, vals in participants.items():
                rows_part.append({
                    "Participant":    name,
                    "Net Futures":    f"{vals.get('net_futures', 0):+,}",
                    "Net Calls (CE)": f"{vals.get('net_calls', 0):+,}",
                    "Net Puts (PE)":  f"{vals.get('net_puts', 0):+,}",
                    "Bias":           vals.get("bias", "—"),
                })
            df_part = pd.DataFrame(rows_part)

            def style_bias(val):
                if "BULLISH" in str(val):  return "color:#00d4aa;font-weight:bold"
                if "BEARISH" in str(val):  return "color:#ff4b4b;font-weight:bold"
                return "color:#ffa500"

            st.dataframe(
                df_part.style.map(style_bias, subset=["Bias"]),
                use_container_width=True, hide_index=True,
            )

            # FII bias callout
            fii = participants.get("FII", {})
            if fii:
                net_c = fii.get("net_calls", 0)
                net_p = fii.get("net_puts",  0)
                net_f = fii.get("net_futures", 0)
                if net_c > 0 and net_p < 0:
                    fii_msg = "🟢 FII DIRECTIONAL BULLISH — buying CE, writing PE"
                    fii_col = "#00d4aa"
                elif net_c < 0 and net_p > 0:
                    fii_msg = "🔴 FII DIRECTIONAL BEARISH — writing CE, buying PE"
                    fii_col = "#ff4b4b"
                elif net_f > 0:
                    fii_msg = "🟡 FII HEDGED BULLISH — long futures + put hedge"
                    fii_col = "#ffa500"
                else:
                    fii_msg = "⚪ FII NEUTRAL / MIXED"
                    fii_col = "#888"
                st.markdown(
                    f"<div style='background:#1e2130;border-left:4px solid {fii_col};"
                    f"padding:0.8rem 1rem;border-radius:6px;margin:0.5rem 0;font-size:1.1rem;'>"
                    f"<b>{fii_msg}</b><br>"
                    f"<small>Net CE: {net_c:+,} | Net PE: {net_p:+,} | Net Fut: {net_f:+,}</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No participant data available.")

    st.markdown("---")

    # ── Option Chain OI Wall ─────────────────────────────────────────────────
    st.markdown("### 🧱 OI Wall — Resistance & Support Strikes")

    chain_data = fetch_fii_tape_data()
    if "error" in chain_data:
        st.warning(f"Option chain unavailable: {chain_data['error']}")
        st.info("NSE option chain requires Indian IP or VPN. May be restricted outside market hours.")
    else:
        df_chain = chain_data.get("df")
        spot     = chain_data.get("spot", 0)

        if df_chain is not None and not df_chain.empty and spot:
            # Filter ±2000 pts from spot
            df_near = df_chain[
                (df_chain["strikePrice"] >= spot - 2000) &
                (df_chain["strikePrice"] <= spot + 2000)
            ].copy()

            # Key levels
            if not df_near.empty:
                max_ce_row  = df_near.loc[df_near["CE_openInterest"].fillna(0).idxmax()]
                max_pe_row  = df_near.loc[df_near["PE_openInterest"].fillna(0).idxmax()]
                df_near["total_oi"] = df_near["CE_openInterest"].fillna(0) + df_near["PE_openInterest"].fillna(0)
                max_pain_row = df_near.loc[df_near["total_oi"].idxmax()]

                kc1, kc2, kc3, kc4 = st.columns(4)
                kc1.metric("Spot",       f"₹{spot:,.0f}")
                kc2.metric("Resistance (Max CE OI)", f"₹{max_ce_row['strikePrice']:,.0f}",
                           delta=f"+{max_ce_row['strikePrice']-spot:.0f} pts")
                kc3.metric("Support (Max PE OI)",    f"₹{max_pe_row['strikePrice']:,.0f}",
                           delta=f"{max_pe_row['strikePrice']-spot:.0f} pts")
                kc4.metric("Max Pain",   f"₹{max_pain_row['strikePrice']:,.0f}",
                           delta=f"{max_pain_row['strikePrice']-spot:+.0f} pts")

            # OI bar chart — CE vs PE per strike
            strikes  = df_near["strikePrice"].tolist()
            ce_oi    = df_near["CE_openInterest"].fillna(0).tolist()
            pe_oi    = df_near["PE_openInterest"].fillna(0).tolist()

            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(
                x=strikes, y=ce_oi, name="CE OI (writers cap upside)",
                marker_color="#ff4b4b", opacity=0.8,
            ))
            fig_oi.add_trace(go.Bar(
                x=strikes, y=[-v for v in pe_oi], name="PE OI (writers defend downside)",
                marker_color="#00d4aa", opacity=0.8,
            ))
            fig_oi.add_vline(x=spot, line_color="#ffa500", line_dash="dash",
                             annotation_text=f"Spot {spot:.0f}", annotation_position="top")
            fig_oi.update_layout(
                title="CE vs PE Open Interest by Strike",
                barmode="overlay", height=350,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                margin=dict(t=40, b=20),
                xaxis_title="Strike Price", yaxis_title="OI (CE positive, PE negative)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_oi, use_container_width=True)

            # PCR by strike
            df_near["pcr"] = (
                df_near["PE_openInterest"].fillna(0) /
                df_near["CE_openInterest"].fillna(1).replace(0, 1)
            )
            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Bar(
                x=df_near["strikePrice"].tolist(),
                y=df_near["pcr"].tolist(),
                marker_color=[
                    "#00d4aa" if v > 1.2 else ("#ff4b4b" if v < 0.8 else "#ffa500")
                    for v in df_near["pcr"].tolist()
                ],
                name="PCR per Strike",
            ))
            fig_pcr.add_hline(y=1.0, line_color="#888", line_dash="dot",
                              annotation_text="PCR=1 (neutral)")
            fig_pcr.add_vline(x=spot, line_color="#ffa500", line_dash="dash")
            fig_pcr.update_layout(
                title="Put-Call Ratio by Strike (>1.2 = bullish support, <0.8 = bearish)",
                height=250,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_pcr, use_container_width=True)

            # Detailed table — top 20 strikes by total OI
            st.markdown("#### Top Strikes by Total OI")
            df_table = df_near.copy()
            df_table["CE OI"]    = df_table["CE_openInterest"].fillna(0).astype(int)
            df_table["PE OI"]    = df_table["PE_openInterest"].fillna(0).astype(int)
            df_table["CE LTP"]   = df_table.get("CE_lastPrice", pd.Series(0, index=df_table.index)).fillna(0).round(1)
            df_table["PE LTP"]   = df_table.get("PE_lastPrice", pd.Series(0, index=df_table.index)).fillna(0).round(1)
            df_table["Total OI"] = df_table["CE OI"] + df_table["PE OI"]
            df_table["PCR"]      = df_table["pcr"].round(2)
            df_table["Dist"]     = (df_table["strikePrice"] - spot).round(0).astype(int)
            df_display = df_table[["strikePrice","Dist","CE OI","CE LTP","PE OI","PE LTP","Total OI","PCR"]]\
                .rename(columns={"strikePrice":"Strike"})\
                .sort_values("Total OI", ascending=False)\
                .head(20)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        else:
            st.info("Option chain data not available. Run engine during market hours (9:15–15:30 IST).")

    st.markdown("---")

    # ── FII Tape Events ──────────────────────────────────────────────────────
    st.markdown("### 🎬 Live Tape — Smart Money Moves")
    st.caption("OI delta events: WRITE = fresh shorts | BUY = fresh longs | UNWIND = covering")

    # Tape is populated by FIITape.tick() in the engine loop
    # Read from /tmp/fii_tape.json if engine is running, else show static analysis
    tape_events = []
    try:
        with open("/tmp/fii_tape.json") as _f:
            tape_events = json.load(_f)
    except Exception:
        pass

    if tape_events:
        rows_tape = []
        for e in tape_events[:50]:
            action_emoji = {
                "WRITE":        "✍️ WRITE",
                "BUY":          "📈 BUY",
                "UNWIND_WRITE": "↩️ UNWIND SHORT",
                "UNWIND_BUY":   "↪️ UNWIND LONG",
            }.get(e.get("action",""), e.get("action",""))
            sig_emoji = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"⚪"}.get(e.get("significance",""),"")
            rows_tape.append({
                "Time":          e.get("timestamp",""),
                "Sig":           f"{sig_emoji} {e.get('significance','')}",
                "Who":           e.get("inferred_participant",""),
                "Smart $":       "✅" if e.get("smart_money") else "",
                "Strike":        f"{e.get('strike',0):.0f}",
                "Type":          e.get("type",""),
                "Action":        action_emoji,
                "ΔOI":           f"{e.get('oi_change',0):+,}",
                "LTP":           f"₹{e.get('ltp',0):.1f}",
                "Description":   e.get("desc",""),
            })
        df_tape = pd.DataFrame(rows_tape)

        def style_tape_action(val):
            if "WRITE" in str(val):    return "color:#ff4b4b;font-weight:bold"
            if "BUY" in str(val):      return "color:#00d4aa;font-weight:bold"
            if "UNWIND" in str(val):   return "color:#ffa500"
            return ""

        def style_tape_who(val):
            if val == "FII": return "color:#7eb8f7;font-weight:bold"
            if val == "DII": return "color:#a78bfa;font-weight:bold"
            if val == "PRO": return "color:#fbbf24;font-weight:bold"
            return ""

        st.dataframe(
            df_tape.style
                .map(style_tape_action, subset=["Action"])
                .map(style_tape_who,    subset=["Who"]),
            use_container_width=True, hide_index=True,
        )
    else:
        # Show static OI change analysis from current chain snapshot
        chain_data2 = fetch_fii_tape_data()
        df2 = chain_data2.get("df") if isinstance(chain_data2, dict) else None
        spot2 = chain_data2.get("spot", 0) if isinstance(chain_data2, dict) else 0

        if df2 is not None and not df2.empty:
            st.info(
                "Live tape requires engine loop running with FIITape.tick(). "
                "Showing current OI snapshot instead — start `python3 loop_engine.py` for real-time tape."
            )
            # Show top 10 CE + PE OI events as approximate tape
            df2 = df2.copy()
            df2["CE OI"] = df2["CE_openInterest"].fillna(0).astype(int)
            df2["PE OI"] = df2["PE_openInterest"].fillna(0).astype(int)
            df2["Dist"]  = (df2["strikePrice"] - spot2).round(0)

            top_ce = df2.nlargest(8, "CE OI")[["strikePrice","Dist","CE OI"]].copy()
            top_ce["Type"]   = "CE"
            top_ce["Note"]   = top_ce.apply(
                lambda r: f"OTM CE wall @ {r['strikePrice']:.0f} — writers likely capping upside"
                          if r["Dist"] > 0 else f"ITM CE @ {r['strikePrice']:.0f}", axis=1)
            top_pe = df2.nlargest(8, "PE OI")[["strikePrice","Dist","PE OI"]].copy()
            top_pe["Type"]   = "PE"
            top_pe["Note"]   = top_pe.apply(
                lambda r: f"OTM PE wall @ {r['strikePrice']:.0f} — writers defending support"
                          if r["Dist"] < 0 else f"ITM PE @ {r['strikePrice']:.0f}", axis=1)

            st.markdown("**Top CE OI Strikes (resistance walls)**")
            st.dataframe(top_ce.rename(columns={"strikePrice":"Strike","CE OI":"OI"})[["Strike","Dist","OI","Note"]],
                         use_container_width=True, hide_index=True)
            st.markdown("**Top PE OI Strikes (support walls)**")
            st.dataframe(top_pe.rename(columns={"strikePrice":"Strike","PE OI":"OI"})[["Strike","Dist","OI","Note"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info(
                "No option chain data. Start the engine during market hours for live FII tape:\n"
                "`python3 loop_engine.py`"
            )

    st.markdown("---")

    # ── DOM / Order Book ─────────────────────────────────────────────────────
    st.markdown("### 📊 DOM — Depth of Market")
    st.caption("Live bid/ask ladder from Dhan L2 (200-level) or NSE fallback (5-level)")

    @st.cache_data(ttl=5)
    def fetch_dom(symbol: str = "NIFTY"):
        try:
            from core.data.dhan_feed import get_best_l2_feed
            feed = get_best_l2_feed()
            depth = feed.get_depth(symbol)
            if depth:
                return {
                    "bids": depth.bids[:15],
                    "asks": depth.asks[:15],
                    "ltp":  depth.ltp,
                    "imbalance": depth.imbalance(),
                }
        except Exception as e:
            return {"error": str(e)}
        return {}

    dom_sym = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "NIFTY_FUT"], key="dom_sym")
    dom = fetch_dom(dom_sym)

    if dom and "error" not in dom and dom.get("bids"):
        bids = dom["bids"]
        asks = dom["asks"]
        ltp  = dom.get("ltp", 0)
        imb  = dom.get("imbalance", 0)

        # Imbalance gauge
        imb_pct = imb * 100
        imb_color = "#00d4aa" if imb > 0.1 else ("#ff4b4b" if imb < -0.1 else "#ffa500")
        st.markdown(
            f"<div style='background:#1e2130;padding:0.8rem 1.2rem;border-radius:8px;"
            f"border-left:4px solid {imb_color};margin-bottom:1rem;'>"
            f"<b>LTP:</b> ₹{ltp:,.2f} &nbsp;|&nbsp; "
            f"<b>Order Book Imbalance:</b> "
            f"<span style='color:{imb_color};font-weight:bold;font-size:1.2rem;'>{imb_pct:+.1f}%</span>"
            f" ({'BID heavy — buyers dominating' if imb > 0.1 else ('ASK heavy — sellers dominating' if imb < -0.1 else 'Balanced')})"
            f"</div>",
            unsafe_allow_html=True,
        )

        # DOM ladder
        dc1, dc2 = st.columns(2)

        with dc1:
            st.markdown("**🟢 BIDS (Buyers)**")
            bid_rows = [{"Price": f"₹{b.get('price',0):,.2f}",
                         "Qty":   f"{b.get('qty',0):,}",
                         "Orders": b.get("orders", "—")} for b in bids]
            bid_df = pd.DataFrame(bid_rows)
            st.dataframe(bid_df, use_container_width=True, hide_index=True)

        with dc2:
            st.markdown("**🔴 ASKS (Sellers)**")
            ask_rows = [{"Price": f"₹{a.get('price',0):,.2f}",
                         "Qty":   f"{a.get('qty',0):,}",
                         "Orders": a.get("orders", "—")} for a in asks]
            ask_df = pd.DataFrame(ask_rows)
            st.dataframe(ask_df, use_container_width=True, hide_index=True)

        # DOM heatmap
        all_prices = [b.get("price",0) for b in bids] + [a.get("price",0) for a in asks]
        bid_qtys   = [b.get("qty",0)   for b in bids] + [0]*len(asks)
        ask_qtys   = [0]*len(bids)                      + [a.get("qty",0) for a in asks]

        fig_dom = go.Figure()
        fig_dom.add_trace(go.Bar(
            x=[f"₹{p:,.0f}" for p in all_prices],
            y=bid_qtys, name="Bid Qty", marker_color="#00d4aa", opacity=0.8,
        ))
        fig_dom.add_trace(go.Bar(
            x=[f"₹{p:,.0f}" for p in all_prices],
            y=[-q for q in ask_qtys], name="Ask Qty", marker_color="#ff4b4b", opacity=0.8,
        ))
        fig_dom.add_hline(y=0, line_color="#555")
        fig_dom.update_layout(
            title="DOM Ladder — Bid vs Ask Volume",
            barmode="overlay", height=300,
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
            margin=dict(t=40, b=20),
            xaxis_title="Price Level", yaxis_title="Qty (bid +, ask −)",
        )
        st.plotly_chart(fig_dom, use_container_width=True)

    elif dom and "error" in dom:
        st.warning(f"DOM feed: {dom['error']}")
        st.info(
            "Configure Dhan credentials in `.env` for 200-level depth:\n"
            "```\nDHAN_CLIENT_ID=...\nDHAN_ACCESS_TOKEN=...\n```\n"
            "Or set GDF credentials for 5-level depth:\n"
            "```\nGDF_USER=...\nGDF_PASS=...\n```"
        )
    else:
        st.info("DOM data not available. Configure L2 feed credentials in `.env`")

    st.markdown("---")

    # ── Footprint Heatmap ────────────────────────────────────────────────────
    st.markdown("### 🔥 Footprint Heatmap — Volume at Price")
    st.caption("Approximate buy vs sell volume per price level per bar (from OHLCV data)")

    @st.cache_data(ttl=120)
    def fetch_footprint():
        try:
            from core.order_flow.fii_tape import FootprintEngine
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            df_fp = ticker.history(period="2d", interval="5m")
            if df_fp.empty:
                return None, None
            df_fp.columns = [c.lower() for c in df_fp.columns]
            engine = FootprintEngine(tick_size=50.0)
            bars   = engine.from_ohlcv(df_fp, bar_minutes=5)
            heatmap = engine.heatmap_data(bars)
            return bars, heatmap
        except Exception as e:
            return None, {"error": str(e)}

    fp_bars, fp_heatmap = fetch_footprint()

    if fp_heatmap and "error" not in fp_heatmap:
        prices     = fp_heatmap.get("y", [])
        timestamps = fp_heatmap.get("x", [])
        buy_vol    = fp_heatmap.get("z_buy", [])
        sell_vol   = fp_heatmap.get("z_sell", [])
        delta      = fp_heatmap.get("z_delta", [])

        if prices and timestamps and delta:
            # Delta heatmap
            fig_fp = go.Figure(data=go.Heatmap(
                z=delta,
                x=timestamps,
                y=prices,
                colorscale=[
                    [0.0, "#ff4b4b"],
                    [0.5, "#0e1117"],
                    [1.0, "#00d4aa"],
                ],
                colorbar=dict(title="Delta<br>(Buy-Sell)"),
                zmid=0,
            ))
            fig_fp.update_layout(
                title="Footprint Delta Heatmap — Green=Buy dominant, Red=Sell dominant",
                height=400,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                margin=dict(t=40, b=20),
                xaxis_title="Time Bar", yaxis_title="Price Level",
            )
            st.plotly_chart(fig_fp, use_container_width=True)

            # POC overlay
            if fp_bars:
                try:
                    from core.order_flow.fii_tape import FootprintEngine as _FP
                    poc_data = _FP().poc_line(fp_bars)   # poc_line works on already-built bars
                    if poc_data:
                        pocs = [p.get("poc",0) for p in poc_data]
                        ts2  = [p.get("ts","") for p in poc_data]
                        fig_poc = go.Figure()
                        fig_poc.add_trace(go.Scatter(
                            x=ts2, y=pocs, mode="lines+markers",
                            line=dict(color="#ffa500", width=2),
                            marker=dict(size=5),
                            name="POC (Point of Control)",
                        ))
                        fig_poc.update_layout(
                            title="POC Line — Price with max traded volume per bar",
                            height=200,
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white",
                            margin=dict(t=40, b=10),
                        )
                        st.plotly_chart(fig_poc, use_container_width=True)
                except Exception:
                    pass
        else:
            st.info("Footprint data is empty. Market may be closed.")
    elif fp_heatmap and "error" in fp_heatmap:
        st.warning(f"Footprint: {fp_heatmap['error']}")
    else:
        st.info("Footprint data not available.")

    st.markdown("---")

    # ── Strategy Guide ────────────────────────────────────────────────────────
    with st.expander("📖 How to Read This Tab"):
        st.markdown("""
**OI Wall (Resistance & Support)**
- **Max CE OI strike** = where institutions have most call shorts → price magnet / resistance
- **Max PE OI strike** = where institutions have most put shorts → strong support
- **Max Pain** = where total OI is highest → market tends to expire near this level
- **PCR > 1.2** = put writers dominant → bullish support | **PCR < 0.8** = call writers dominant → bearish cap

**Tape Events (WRITE / BUY / UNWIND)**
| Action | OI | Price | Meaning |
|---|---|---|---|
| WRITE | ↑ | ↓ | Fresh short opened (selling premium) |
| BUY | ↑ | ↑ | Fresh long opened (buying premium) |
| UNWIND_WRITE | ↓ | ↑ | Short-seller covering (bullish signal) |
| UNWIND_BUY | ↓ | ↓ | Long buyer exiting (bearish signal) |

**FII OTM Writing Pattern**
- FII writing OTM CE (above spot) = they expect price to stay below → **resistance cap**
- FII writing OTM PE (below spot) = they expect price to stay above → **support floor**
- When both CE and PE have large OI → market stuck in range until expiry

**DOM (Depth of Market)**
- Large bid queue = buyers waiting → support level
- Large ask queue = sellers waiting → resistance level
- Imbalance > +10% = buy pressure dominating → bullish
- Imbalance < -10% = sell pressure dominating → bearish

**Footprint Chart**
- Green cells = buy volume dominant at that price level
- Red cells = sell volume dominant at that price level
- POC (Point of Control) = price level with most volume = likely anchor/magnet
        """)



# ══ TAB 12: ORDER FLOW STRATEGY (v3 — 40-pt scoring) ════════════════════════
with tabs[11]:

    FILTER_WINDOW_MIN = 30   # minutes — highlight events this close

    # ── helpers ──────────────────────────────────────────────────────────────
    def _of_load_signal():
        try:
            with open("/tmp/xauusd_of_signal.json") as f:
                return json.load(f)
        except Exception:
            return None

    def _of_action_color(action):
        return {"BUY": "#00d4aa", "SELL": "#ff4b4b"}.get(action, "#ffa500")

    def _of_action_emoji(action):
        return {"BUY": "🟢", "SELL": "🔴"}.get(action, "⏳")

    # ── page header ──────────────────────────────────────────────────────────
    st.markdown("""
<h2 style='margin-bottom:0;'>🧠 Order Flow Strategy <span style='font-size:0.7rem;color:#888;'>v3 · 40-pt scoring</span></h2>
<p style='color:#888;margin-top:4px;font-size:0.9rem;'>
SMC (OB + FVG + BOS/CHoCH + Liquidity) · Volume Profile · CVD ·
Tape Reader · OTE · Premium/Discount · EQH/EQL · ICT Killzones · Self-Learning
</p>
""", unsafe_allow_html=True)

    # ── action buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns([1,1,4])
    with btn1:
        run_now = st.button("▶ Run Now", key="of_run_v3", type="primary")
    with btn2:
        if st.button("⟳ Refresh", key="of_refresh_v3"):
            st.cache_data.clear()
            st.rerun()
    with btn3:
        st.caption("Engine: `python3 xauusd/of_engine.py` — signals fire at 22/40 during London/NY killzones")

    if run_now:
        with st.spinner("Running 40-point order flow analysis + news feed (~20s)..."):
            try:
                from xauusd.of_strategy import analyze_order_flow
                from xauusd.news_feed import get_news_context as _get_news
                _s   = analyze_order_flow()
                _nws = _get_news(force=True)
                import json as _j
                _data = {k: getattr(_s,k) if not callable(getattr(_s,k)) else None
                         for k in _s.__dataclass_fields__ if k != "timestamp"}
                _data.update({
                    "timestamp":  str(_s.timestamp), "max_score": _s.max_score,
                    "news_sentiment":       _nws.get("sentiment","NEUTRAL"),
                    "news_sentiment_score": _nws.get("sentiment_score",0.0),
                    "news_filter":          _nws.get("news_filter",False),
                    "news_filter_reason":   _nws.get("news_filter_reason",""),
                    "news_headlines":       _nws.get("headlines",[])[:6],
                    "news_calendar":        _nws.get("calendar",[])[:8],
                    "news_fetched_at":      _nws.get("fetched_at",""),
                })
                with open("/tmp/xauusd_of_signal.json","w") as _f:
                    _j.dump(_data, _f, indent=2, default=str)
                st.success(f"{_of_action_emoji(_s.action)} {_s.action} {_s.strength} — score {_s.score}/{_s.max_score} | News: {_nws['sentiment']}")
                st.rerun()
            except Exception as _e:
                st.error(f"Error: {_e}")

    st.divider()

    of_sig = _of_load_signal()

    if of_sig is None:
        st.info("No signal yet — click **Run Now** or start the engine.")
        st.code("python3 xauusd/of_engine.py", language="bash")
        st.stop()

    action   = of_sig.get("action","WAIT")
    strength = of_sig.get("strength","WAIT")
    score    = of_sig.get("score",0)
    max_sc   = of_sig.get("max_score",40)
    conf     = of_sig.get("confidence",0)
    entry    = of_sig.get("entry",0)
    sl       = of_sig.get("stop_loss",0)
    tp1      = of_sig.get("target1",0)
    tp2      = of_sig.get("target2",0)
    tp3      = of_sig.get("target3",0)
    rr       = of_sig.get("risk_reward",0)
    ts_str   = (of_sig.get("timestamp","") or "")[:19].replace("T"," ")
    reasons  = of_sig.get("reasons",[])

    act_col  = _of_action_color(action)
    act_emoji= _of_action_emoji(action)
    risk_pts = abs(entry - sl) if sl else 0

    # ══ SECTION 1: SIGNAL CARD ═══════════════════════════════════════════════
    kz   = of_sig.get("killzone","") or "No killzone"
    htf  = of_sig.get("htf_bias","")
    tape_bias = of_sig.get("cvd_bias","") or "NEUTRAL"

    st.markdown(f"""
<div style='background:#1a1d2e;border-radius:14px;padding:1.4rem 1.8rem;
border-left:6px solid {act_col};margin-bottom:1.2rem;'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <span style='color:{act_col};font-size:2rem;font-weight:800;letter-spacing:1px;'>
        {act_emoji} {action} {strength}
      </span>
      <span style='color:#888;font-size:0.9rem;margin-left:12px;'>{ts_str} UTC</span>
    </div>
    <div style='text-align:right;'>
      <span style='color:white;font-size:1.6rem;font-weight:700;'>{score}/{max_sc}</span>
      <span style='color:#888;font-size:0.85rem;'> pts ({conf:.0%} conf)</span>
    </div>
  </div>
  <div style='margin-top:0.8rem;display:flex;gap:2rem;flex-wrap:wrap;'>
    <span style='color:#aaa;'>📍 KZ: <b style='color:white;'>{kz}</b></span>
    <span style='color:#aaa;'>📊 HTF: <b style='color:white;'>{htf}</b></span>
    <span style='color:#aaa;'>🌊 Tape: <b style='color:white;'>{tape_bias}</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

    # Score progress bar
    score_pct = score / max_sc
    bar_color = act_col if score_pct >= 0.55 else "#ffa500"
    st.markdown(f"""
<div style='background:#1e2130;border-radius:8px;height:12px;margin-bottom:0.3rem;overflow:hidden;'>
  <div style='background:{bar_color};height:100%;width:{score_pct*100:.1f}%;border-radius:8px;
  transition:width 0.5s ease;'></div>
</div>
<p style='color:#888;font-size:0.8rem;margin-top:0;'>
  Score: {score}/{max_sc} — Need 22 to trade, 28 for STRONG | Threshold line: ━━━━━━━━━━
</p>
""", unsafe_allow_html=True)

    # ══ SECTION 2: TRADE LEVELS ══════════════════════════════════════════════
    if action in ("BUY","SELL"):
        st.markdown("### Trade Levels")
        tc1,tc2,tc3,tc4,tc5,tc6 = st.columns(6)
        tc1.metric("Entry",         f"${entry:.2f}")
        tc2.metric("Stop Loss",     f"${sl:.2f}",
                   delta=f"-{risk_pts:.2f} pts", delta_color="inverse")
        tc3.metric("TP1  (1.5R)",   f"${tp1:.2f}",
                   delta=f"+{abs(tp1-entry):.2f}" if action=="BUY" else f"-{abs(tp1-entry):.2f}")
        tc4.metric("TP2  (2.5R)",   f"${tp2:.2f}",
                   delta=f"+{abs(tp2-entry):.2f}" if action=="BUY" else f"-{abs(tp2-entry):.2f}")
        tc5.metric("TP3  (4R)",     f"${tp3:.2f}",
                   delta=f"+{abs(tp3-entry):.2f}" if action=="BUY" else f"-{abs(tp3-entry):.2f}")
        tc6.metric("Risk:Reward",   f"1:{rr:.1f}")

        # Trade plan guide
        st.markdown(f"""
<div style='background:#1a1d2e;border-radius:10px;padding:1rem 1.4rem;margin-top:0.5rem;
border:1px solid #2d3250;font-size:0.9rem;'>
  <b style='color:{act_col};'>Trade Plan:</b>
  &nbsp; Enter @ <b>${entry:.2f}</b>
  &nbsp;|&nbsp; SL @ <b>${sl:.2f}</b> ({risk_pts:.1f} pts risk)
  &nbsp;|&nbsp; Take 50% off @ TP1 <b>${tp1:.2f}</b>, move SL to BE
  &nbsp;|&nbsp; Let rest run to TP2 <b>${tp2:.2f}</b>
  &nbsp;|&nbsp; Trail stop for TP3 <b>${tp3:.2f}</b>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ══ SECTION 3: MARKET CONTEXT (clean grid) ═══════════════════════════════
    st.markdown("### Market Context")

    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        st.markdown("**Structure & Levels**")
        struct = of_sig.get("structure","") or "—"
        sc = "#00d4aa" if "UP" in struct else ("#ff4b4b" if "DOWN" in struct else "#888")
        st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;'>"
                    f"<span style='color:#888;font-size:0.8rem;'>4H Structure</span><br>"
                    f"<b style='color:{sc};font-size:1.1rem;'>{struct}</b>"
                    f"</div>", unsafe_allow_html=True)

        poc = of_sig.get("poc",0); vah = of_sig.get("vah",0); val = of_sig.get("val",0)
        vwap = of_sig.get("vwap",0)
        for lbl, val_v, col in [("POC", poc, "#ffa500"),("VAH",vah,"#00d4aa"),("VAL",val,"#ff4b4b"),("VWAP",vwap,"#60a5fa")]:
            if val_v:
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.25rem 0.5rem;"
                            f"background:#1e2130;border-radius:6px;margin-bottom:3px;'>"
                            f"<span style='color:#888;font-size:0.85rem;'>{lbl}</span>"
                            f"<b style='color:{col};'>${val_v:.2f}</b></div>",
                            unsafe_allow_html=True)

    with cc2:
        st.markdown("**Order Blocks & FVGs**")
        ob_t = of_sig.get("ob_type",""); ob_l = of_sig.get("ob_level",0)
        fvg_t = of_sig.get("fvg_type",""); fvg_top = of_sig.get("fvg_top",0); fvg_bot = of_sig.get("fvg_bot",0)

        ob_c = "#00d4aa" if ob_t=="BULLISH" else ("#ff4b4b" if ob_t=="BEARISH" else "#888")
        st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;'>"
                    f"<span style='color:#888;font-size:0.8rem;'>Order Block</span><br>"
                    f"<b style='color:{ob_c};'>{ob_t or 'None near price'}</b>"
                    f"{'<br><span style="color:#888;font-size:0.8rem;">@ $' + str(round(ob_l,2)) + '</span>' if ob_l else ''}"
                    f"</div>", unsafe_allow_html=True)

        fvg_c = "#00d4aa" if fvg_t=="BULLISH" else ("#ff4b4b" if fvg_t=="BEARISH" else "#888")
        st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.7rem 1rem;'>"
                    f"<span style='color:#888;font-size:0.8rem;'>Fair Value Gap</span><br>"
                    f"<b style='color:{fvg_c};'>{fvg_t or 'None filling'}</b>"
                    f"{'<br><span style="color:#888;font-size:0.8rem;">[$' + str(round(fvg_bot,2)) + ' – $' + str(round(fvg_top,2)) + ']</span>' if fvg_top else ''}"
                    f"</div>", unsafe_allow_html=True)

        liq_swept = of_sig.get("liq_swept", False)
        liq_lvl   = of_sig.get("liq_level", 0)
        if liq_swept:
            st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.5rem 1rem;border:1px solid #fbbf24;margin-top:0.5rem;'>"
                        f"<span style='color:#fbbf24;font-weight:bold;'>⚡ Liquidity swept @ ${liq_lvl:.2f}</span>"
                        f"</div>", unsafe_allow_html=True)

    with cc3:
        st.markdown("**Tape Reading**")
        tape_col = {"STRONGLY_BULLISH":"#00d4aa","BULLISH":"#4ade80",
                    "STRONGLY_BEARISH":"#ff4b4b","BEARISH":"#f87171"}.get(tape_bias,"#888")
        buy_pct = of_sig.get("confidence",0.5) * 100   # placeholder

        st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.5rem;'>"
                    f"<span style='color:#888;font-size:0.8rem;'>Tape Bias</span><br>"
                    f"<b style='color:{tape_col};font-size:1.1rem;'>{tape_bias}</b>"
                    f"</div>", unsafe_allow_html=True)

        cvd_bias = of_sig.get("cvd_bias","")
        cvd_c = "#00d4aa" if "BULL" in cvd_bias else ("#ff4b4b" if "BEAR" in cvd_bias else "#888")
        st.markdown(f"<div style='background:#1e2130;border-radius:8px;padding:0.7rem 1rem;'>"
                    f"<span style='color:#888;font-size:0.8rem;'>CVD / Delta</span><br>"
                    f"<b style='color:{cvd_c};'>{cvd_bias or 'NEUTRAL'}</b>"
                    f"</div>", unsafe_allow_html=True)

    st.divider()

    # ══ SECTION 4: WHY THIS SIGNAL ═══════════════════════════════════════════
    st.markdown("### Why This Signal — Score Breakdown")

    if reasons:
        # Show as colored badges
        badges_html = ""
        for r in reasons:
            r_low = r.lower()
            if any(k in r_low for k in ["ob","order block"]): bg="#7c3aed"
            elif any(k in r_low for k in ["fvg","fair value"]): bg="#0369a1"
            elif any(k in r_low for k in ["liquidity","liq","sweep","eqh","eql"]): bg="#b45309"
            elif any(k in r_low for k in ["ote","optimal trade"]): bg="#065f46"
            elif any(k in r_low for k in ["tape","cvd","delta","absorb","climax"]): bg="#831843"
            elif any(k in r_low for k in ["killzone","london","ny am","silver bullet","asian"]): bg="#1e3a5f"
            elif any(k in r_low for k in ["vwap"]): bg="#1a3a1a"
            elif any(k in r_low for k in ["poc","vah","val","volume profile","discount","premium"]): bg="#3a2a0a"
            elif any(k in r_low for k in ["bos","choch","bias","structure"]): bg="#2a1a3a"
            else: bg="#1e2130"
            badges_html += (f"<span style='display:inline-block;background:{bg};color:white;"
                            f"border-radius:20px;padding:4px 12px;margin:3px;font-size:0.82rem;"
                            f"border:1px solid rgba(255,255,255,0.1);'>{r}</span>")
        st.markdown(f"<div style='line-height:2;'>{badges_html}</div>", unsafe_allow_html=True)
    else:
        st.caption("No active confluence — waiting for setup.")

    st.divider()

    # ══ SECTION 4b: GOLD NEWS FEED + ECONOMIC CALENDAR ═══════════════════════
    st.markdown("### Gold News + Economic Calendar")

    # Read news from signal JSON (updated by engine) or fetch fresh
    news_sent    = of_sig.get("news_sentiment", "NEUTRAL")
    news_score   = of_sig.get("news_sentiment_score", 0.0)
    news_filter  = of_sig.get("news_filter", False)
    news_reason  = of_sig.get("news_filter_reason", "")
    news_heads   = of_sig.get("news_headlines", [])
    news_cal     = of_sig.get("news_calendar", [])
    news_fetched = of_sig.get("news_fetched_at", "")

    nc1, nc2, nc3, nc4 = st.columns(4)
    sent_col = {"BULLISH": "#00d4aa", "BEARISH": "#ff4b4b"}.get(news_sent, "#ffa500")
    nc1.metric("News Sentiment", news_sent)
    nc2.metric("Sentiment Score", f"{news_score:+.2f}")
    nc3.metric("Headlines (4h)", len(news_heads))
    nc4.metric("USD Events (8h)", len(news_cal))

    if news_filter:
        st.markdown(f"""
<div style='background:#3a1010;border:1px solid #ff4b4b;border-radius:8px;padding:0.6rem 1.2rem;margin:0.5rem 0;'>
  ⚠️ <b style='color:#ff4b4b;'>NEWS FILTER ACTIVE:</b>
  <span style='color:#fca5a5;'> {news_reason}</span>
  <span style='color:#888;font-size:0.85rem;'> — Avoid entering new trades!</span>
</div>""", unsafe_allow_html=True)

    nl, nr = st.columns(2)

    with nl:
        if news_heads:
            st.markdown("**Recent Gold Headlines**")
            for h in news_heads[:5]:
                s_val  = h.get("sentiment", 0)
                age    = h.get("age_min", 0)
                title  = h.get("title", "")[:70]
                pub    = h.get("publisher", "")
                if s_val > 0.1:    s_icon, s_col = "↑", "#00d4aa"
                elif s_val < -0.1: s_icon, s_col = "↓", "#ff4b4b"
                else:              s_icon, s_col = "→", "#888"
                st.markdown(
                    f"<div style='background:#1a1d2e;border-radius:6px;padding:0.4rem 0.8rem;"
                    f"margin-bottom:4px;border-left:3px solid {s_col};'>"
                    f"<span style='color:{s_col};font-weight:bold;'>{s_icon}</span> "
                    f"<span style='font-size:0.85rem;'>{title}</span>"
                    f"<br><span style='color:#888;font-size:0.75rem;'>{pub} · {age}m ago</span>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            if news_fetched:
                st.caption(f"No headlines in last 4h (fetched {news_fetched})")
            else:
                st.caption("Run engine to load news — click Run Now or start `python3 xauusd/of_engine.py`")

    with nr:
        if news_cal:
            st.markdown("**Upcoming USD Events**")
            for ev in news_cal[:6]:
                ma       = ev.get("minutes_away", 0)
                title    = ev.get("title", "")
                impact   = ev.get("impact", "")
                when     = "NOW" if ma <= 0 else (f"in {ma}m" if ma < 60 else f"in {ma//60}h {ma%60}m")
                is_mover = ev.get("is_gold_mover", False)
                imp_col  = "#ff4b4b" if impact == "High" else "#ffa500"
                star     = " ⭐" if is_mover else ""
                border   = imp_col if ma <= FILTER_WINDOW_MIN else "#2d3250"
                st.markdown(
                    f"<div style='background:#1a1d2e;border-radius:6px;padding:0.35rem 0.8rem;"
                    f"margin-bottom:4px;border-left:3px solid {border};'>"
                    f"<span style='color:{imp_col};font-size:0.75rem;font-weight:bold;'>[{impact}]</span> "
                    f"<span style='font-size:0.85rem;'>{title}{star}</span>"
                    f"<br><span style='color:#888;font-size:0.75rem;'>{ev.get('date','')[:16]} UTC · {when}</span>"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            st.caption("No high-impact USD events in the next 8h.")

    if news_fetched:
        st.caption(f"News last fetched: {news_fetched}")

    st.divider()

    # ══ SECTION 5: LIVE TAPE FEED ════════════════════════════════════════════
    st.markdown("### Live Tape Feed")

    @st.cache_data(ttl=60)
    def _fetch_tape_data():
        try:
            from xauusd.of_strategy import _fetch
            from xauusd.tape_reader import analyze_tape
            df5 = _fetch("5m","3d")
            tape_res = analyze_tape(df5)
            return tape_res, df5
        except Exception as _e:
            return {"error": str(_e), "tape_events":[]}, None

    tape_res, df5_data = _fetch_tape_data()

    if "error" in tape_res:
        st.warning(f"Tape: {tape_res['error']}")
    else:
        # Tape stat row
        tp1c, tp2c, tp3c, tp4c, tp5c = st.columns(5)
        tbias = tape_res.get("tape_bias","NEUTRAL")
        tbias_col = {"STRONGLY_BULLISH":"#00d4aa","BULLISH":"#4ade80","NEUTRAL":"#888",
                     "BEARISH":"#f87171","STRONGLY_BEARISH":"#ff4b4b"}.get(tbias,"#888")
        tp1c.metric("Tape Bias", tbias)
        tp2c.metric("Buy Pressure", f"{tape_res.get('buy_pressure',50):.0f}%")
        tp3c.metric("Delta 5-bar", f"{tape_res.get('delta_5m',0):+.0f}")
        tp4c.metric("Speed", tape_res.get("tape_speed","—"))
        tp5c.metric("Price", f"${tape_res.get('last_price',0):,.2f}")

        # Events feed
        events = tape_res.get("tape_events",[])
        if events:
            st.markdown("**Recent Tape Events:**")
            for ev in events:
                ev_col = "#00d4aa" if any(k in ev.upper() for k in ["BULL","BUY","ABSORB"]) else "#ff4b4b"
                st.markdown(f"<div style='background:#1e2130;border-radius:6px;padding:0.4rem 0.8rem;"
                            f"margin-bottom:4px;border-left:3px solid {ev_col};font-size:0.85rem;'>"
                            f"{ev}</div>", unsafe_allow_html=True)

        # Delta per bar chart
        ts_list   = tape_res.get("timestamps",[])
        delta_list= tape_res.get("delta_series",[])
        if ts_list and delta_list:
            fig_tape = go.Figure()
            colors_d = ["#00d4aa" if d >= 0 else "#ff4b4b" for d in delta_list]
            fig_tape.add_trace(go.Bar(
                x=ts_list, y=delta_list, name="Delta per bar",
                marker_color=colors_d,
            ))
            fig_tape.add_hline(y=0, line_color="#555", line_width=1)
            fig_tape.update_layout(
                title="Buy–Sell Delta per 5m Bar (green=buyers won, red=sellers won)",
                height=200, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font_color="white", margin=dict(t=35,b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_tape, use_container_width=True)

    st.divider()

    # ══ SECTION 6: LIVE CHART (VWAP + VP + Levels) ═══════════════════════════
    st.markdown("### Chart — VWAP · Volume Profile · OB/FVG · Trade Levels")

    @st.cache_data(ttl=180)
    def _fetch_of_chart():
        try:
            from xauusd.of_strategy import _fetch, compute_vwap, compute_volume_profile
            df = _fetch("15m","3d")
            vp_d  = compute_volume_profile(df.tail(96))
            vwap_d= compute_vwap(df)
            return df, vp_d, vwap_d
        except Exception as _e:
            return None, {}, {}

    df_oc, vp_oc, vwap_oc = _fetch_of_chart()

    if df_oc is not None and not df_oc.empty:
        tail = df_oc.tail(80)
        fig_c = go.Figure()

        # Candlestick
        fig_c.add_trace(go.Candlestick(
            x=tail["timestamp"], open=tail["open"], high=tail["high"],
            low=tail["low"],  close=tail["close"], name="XAUUSD 15m",
            increasing_line_color="#00d4aa", decreasing_line_color="#ff4b4b",
            increasing_fillcolor="#00d4aa", decreasing_fillcolor="#ff4b4b",
        ))

        # VWAP bands
        for lbl, val_v, col, dash in [
            ("VWAP",     vwap_oc.get("vwap",0),         "#ffa500", "solid"),
            ("+1σ",      vwap_oc.get("vwap_upper1",0),  "#60a5fa", "dash"),
            ("-1σ",      vwap_oc.get("vwap_lower1",0),  "#60a5fa", "dash"),
            ("+2σ",      vwap_oc.get("vwap_upper2",0),  "#f87171", "dot"),
            ("-2σ",      vwap_oc.get("vwap_lower2",0),  "#4ade80", "dot"),
        ]:
            if val_v:
                fig_c.add_hline(y=val_v, line_color=col, line_dash=dash, line_width=1,
                                annotation_text=f"{lbl} {val_v:.0f}", annotation_font_size=10,
                                annotation_position="right")

        # VP levels
        for lbl, val_v, col in [
            ("POC", vp_oc.get("poc",0), "#ffa500"),
            ("VAH", vp_oc.get("vah",0), "#00d4aa"),
            ("VAL", vp_oc.get("val",0), "#ff4b4b"),
        ]:
            if val_v:
                fig_c.add_hline(y=val_v, line_color=col, line_dash="longdash", line_width=2,
                                annotation_text=f"{lbl} {val_v:.0f}", annotation_font_size=11,
                                annotation_position="left")

        # OB zone
        if of_sig.get("ob_level") and of_sig.get("ob_type"):
            atr_v  = of_sig.get("atr",10)
            ob_mid = of_sig.get("ob_level",0)
            ob_c2  = "rgba(0,212,170,0.12)" if of_sig.get("ob_type")=="BULLISH" else "rgba(255,75,75,0.12)"
            fig_c.add_hrect(y0=ob_mid-atr_v*0.35, y1=ob_mid+atr_v*0.35,
                            fillcolor=ob_c2, line_width=0,
                            annotation_text=f"{of_sig['ob_type']} OB", annotation_font_size=10)

        # FVG zone
        if of_sig.get("fvg_top") and of_sig.get("fvg_bot"):
            fig_c.add_hrect(y0=of_sig["fvg_bot"], y1=of_sig["fvg_top"],
                            fillcolor="rgba(251,191,36,0.12)",
                            line_color="rgba(251,191,36,0.4)", line_width=1,
                            annotation_text="FVG", annotation_font_size=10)

        # Trade levels
        if action in ("BUY","SELL"):
            for lbl, pv, col, dash in [
                ("Entry",  entry, "#ffffff", "dash"),
                ("SL",     sl,    "#ff4b4b", "dot"),
                ("TP1",    tp1,   "#4ade80", "dot"),
                ("TP2",    tp2,   "#00d4aa", "dot"),
            ]:
                if pv:
                    fig_c.add_hline(y=pv, line_color=col, line_dash=dash, line_width=1.5,
                                    annotation_text=f"{lbl} {pv:.2f}",
                                    annotation_position="bottom right" if lbl=="SL" else "top right",
                                    annotation_font_size=11)

        fig_c.update_layout(
            height=520, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="white", xaxis_rangeslider_visible=False,
            title=f"XAUUSD 15m — {action} {strength} | Score {score}/{max_sc}",
            margin=dict(t=45,b=15,l=50,r=110),
        )
        fig_c.update_xaxis(showgrid=False)
        fig_c.update_yaxis(showgrid=True, gridcolor="#1e2130")
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()

    # ══ SECTION 7: SELF-LEARNING PERFORMANCE ═════════════════════════════════
    st.markdown("### Self-Learning — Factor Performance")
    st.caption("The model learns from every trade — winning factors gain weight, losing factors are reduced.")

    try:
        from xauusd.score_learner import get_factor_performance, summary as learner_summary
        ls = learner_summary()
        ll1,ll2,ll3,ll4 = st.columns(4)
        ll1.metric("Factors Tracked",   ls.get("factors_tracked",0))
        ll2.metric("Factors Adjusted",  ls.get("factors_adjusted",0))
        ll3.metric("Last Updated",      (ls.get("last_updated","never") or "")[:10])
        ll4.metric("Avg Trades/Factor", ls.get("total_trades_recorded",0))

        perf = get_factor_performance()
        if perf:
            df_perf = pd.DataFrame(perf)
            def style_wr(val):
                try:
                    v = float(str(val).rstrip("%"))
                    if v >= 65: return "color:#00d4aa;font-weight:bold"
                    if v <= 40: return "color:#ff4b4b;font-weight:bold"
                    return "color:#ffa500"
                except: return ""
            st.dataframe(
                df_perf.style.map(style_wr, subset=["Win Rate"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No factor data yet — trades will be logged automatically as signals fire and SL/TP is hit.")
    except Exception as _e:
        st.caption(f"Learner: {_e}")

    st.divider()

    # ══ SECTION 8: OF HISTORY ════════════════════════════════════════════════
    st.markdown("### Signal History")

    @st.cache_data(ttl=30)
    def _load_of_hist():
        try:
            p = Path(__file__).parent.parent / "data" / "xauusd_of_trades.json"
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return []

    of_hist = _load_of_hist()
    if of_hist:
        rows_ofh = []
        for h in reversed(of_hist[-30:]):
            action_h = h.get("action","")
            rows_ofh.append({
                "Time":     (h.get("timestamp","") or "")[:16].replace("T"," "),
                "Signal":   action_h,
                "Strength": h.get("strength",""),
                "Score":    f"{h.get('score',0)}/{h.get('max_score',40)}",
                "Entry":    f"${h.get('entry',0):.2f}" if h.get("entry") else "—",
                "HTF":      h.get("htf_bias",""),
                "Killzone": h.get("killzone","") or "—",
                "Reasons":  " · ".join((h.get("reasons") or [])[:2]),
            })
        df_ofh = pd.DataFrame(rows_ofh)
        def _style_sig(v):
            if v=="BUY":  return "color:#00d4aa;font-weight:bold"
            if v=="SELL": return "color:#ff4b4b;font-weight:bold"
            return "color:#ffa500"
        st.dataframe(
            df_ofh.style.map(_style_sig, subset=["Signal"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No history yet — start the engine for continuous logging.")

    # ══ SECTION 9: QUICK REFERENCE ═══════════════════════════════════════════
    with st.expander("Quick Reference — Score Layers"):
        st.markdown("""
| Layer | Pts | When it fires |
|---|---|---|
| 4H HTF Bias | 0-5 | 4H trend + EMA stack + 4H/1H BOS aligned |
| Order Block | 0-4 | Price tapping 4H/1H/15m OB in trade direction |
| Fair Value Gap | 0-4 | Unmitigated FVG being filled + HTF FVG + Inversion FVG |
| Liquidity Sweep | 0-3 | EQH/EQL swept (bull/bear trap) + multi-TF sweep |
| OTE Zone | 0-3 | Price in 61.8–70.5% retracement of last swing |
| CVD Divergence | 0-3 | Price up but sellers absorbing (or vice versa) |
| Tape Bias | 0-3 | Strongly bullish/bearish tape + climax/absorption |
| Premium/Discount | 0-2 | Buy in discount (<50% of 4H range), sell in premium |
| Volume Profile | 0-3 | At POC, VAH, or VAL |
| Killzone | 0-3 | NY Silver Bullet > London KZ > NY AM > Asian |
| VWAP | 0-3 | At ±2σ (extreme) or ±1σ + direction aligned |
| Displacement | 0-2 | Large body candle preceding entry |
| Prev Day/Week H/L | 0-2 | Price at PDH/PDL or PWH/PWL |

**Trade fires:** ≥ 22/40 (55%) · **STRONG:** ≥ 28/40 (70%)
**SL:** Below/above OB extreme ±0.3×ATR · **TP1:** 1.5R · **TP2:** 2.5R or key level · **TP3:** 4R
""")
