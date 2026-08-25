# AI-Trade — Full Project Overview

## What This Is
An algorithmic trading system for **Nifty F&O** (Indian market) and **XAUUSD** (Gold), combining ML-based signals, live data feeds, technical analysis, options Greeks, Telegram alerts, and a hedging strategy for zero-loss protection.

---

## Architecture

```
ai-trade/
├── core/                    # Nifty F&O engine
│   ├── agent/
│   │   ├── ml_agent.py      # XGBoost + LightGBM + Ollama ensemble
│   │   ├── trading_agent.py # TradeDecision dataclass + signal_message()
│   │   └── explainer.py     # SHAP TreeExplainer — WHY each decision was made
│   ├── order_flow/
│   │   └── oi_tracker.py    # NSE option chain OI + tape reader
│   ├── memory/
│   │   └── vector_store.py  # Qdrant vector DB — stores market states
│   ├── options/
│   │   └── greeks.py        # BSM Greeks (py_vollib + mibian fallback)
│   ├── alerts/
│   │   └── telegram_bot.py  # Telegram signal cards (HTML formatted)
│   ├── data/
│   │   ├── nse_scraper.py   # NSE live option chain / PCR / VIX
│   │   ├── historical.py    # yfinance OHLCV
│   │   └── nse_participant.py # FII/DII data via nsepython
│   ├── brokers/
│   │   ├── paper.py         # Paper trading
│   │   └── zerodha.py       # Zerodha Kite Connect
│   ├── strategies/          # Scalp, ORB, Swing, Positional, FnO
│   ├── risk/
│   │   └── manager.py       # Daily P&L limits, position sizing
│   └── backtest/
│       └── engine.py        # Bar-by-bar backtester
│
├── xauusd/                  # XAUUSD (Gold) engine
│   ├── data.py              # GC=F price + DXY/10Y/VIX macro data
│   ├── strategy.py          # 1H trend + 15m entry + macro scoring
│   ├── engine.py            # Live loop — polls every 60s, all sessions
│   ├── dashboard.py         # Streamlit dashboard (port 8501)
│   └── india.py             # NSE stock scan + Nifty hedge signal
│
├── vendors/                 # 7 cloned GitHub repos
│   ├── ai-trader/           # OptionsFlowDetector, SignalGenerator
│   ├── algo-strategies/     # Zerodha short straddle / iron fly
│   ├── nifty-flow-analysis/ # OI divergence engine
│   ├── nse-oi-analyzer/     # Option chain analyzer
│   ├── nse-oi-analysis/     # OI data extraction
│   ├── non-directional-strategy/ # Non-directional options backtest
│   └── jugaad-data/         # NSE historical data
│
├── dashboard/
│   └── app.py               # Nifty dashboard (Streamlit)
├── loop_engine.py           # Nifty 24/7 loop
├── backtest_trainer.py      # Historical Nifty backtest + Optuna tuning
├── main.py                  # CLI entry point
└── config/config.yaml       # All settings
```

---

## Running Everything

```bash
# Nifty loop engine (24/7 — polls tape every 60s, trains at 15:35)
python3 loop_engine.py NIFTY

# XAUUSD live engine (all sessions — London, NY, Asia)
python3 xauusd/engine.py

# XAUUSD dashboard (ngrok URL: https://felicita-reliant-erich.ngrok-free.dev)
streamlit run xauusd/dashboard.py --server.port 8501

# Nifty dashboard
streamlit run dashboard/app.py --server.port 8501

# Backtest + retrain ML model on real Nifty data
python3 backtest_trainer.py

# Monitor logs
tail -f /tmp/xauusd.log
tail -f /tmp/loop_engine.log
```

---

## XAUUSD Strategy

| Layer | What it checks |
|-------|---------------|
| **1H Trend** | EMA 21/55/200 stack — only trade with macro direction |
| **15m Entry** | MACD crossover, RSI 40-65, price vs EMA21 |
| **Macro** | DXY direction (inverse), 10Y yield, VIX spike |
| **Session** | London (07-16 UTC) + NY (13-21 UTC) — full signal; Asia needs score ≥7 |
| **Score** | 0-10; need ≥6 to trigger a trade |
| **SL/TP** | 1.5x ATR stop, 2.5x ATR target → R:R ≈ 1:1.67 |

---

## Zero-Loss Hedging System (XAUUSD)

**Strategy 1 — Breakeven Trail:**
- Enter trade; when price hits +1R, move SL to entry
- Worst case from that point = $0 loss

**Strategy 2 — Lock & Hedge:**
- If trade goes -0.5R, open counter-position same size
- One leg always profits; close both at net P&L ≥ 0

**Strategy 3 — Trail Stop:**
- Trail SL by 0.5× ATR below each new high (BUY) or above each new low (SELL)
- Locks in more profit as trade runs

---

## Nifty F&O System

- **ML Model**: XGBoost + LightGBM ensemble (78 features, 13,069 training samples)
- **SHAP explainability**: Shows top 5 features driving each decision
- **Options Greeks**: Delta, Gamma, Theta/day, Vega, IV, break-even via BSM
- **Vector memory**: Qdrant stores market states every 30min, records outcomes after 30min for labeling
- **EOD retrain**: At 15:35 IST, retrains model on today's real outcomes + Optuna 30-trial tuning
- **Telegram**: Signal card sent on every new BUY_CE / BUY_PE / SELL_STRADDLE

---

## India Market Tab (Dashboard)

- **NSE Stock Scan**: 20 liquid stocks scored on EMA trend + RSI + volume spike + 5D momentum + news sentiment
- **Nifty Hedge Signal**: RSI-based hedge — BUY CE when oversold, BUY PE when overbought
- **Signals cached 15 minutes** (don't flip on every refresh)

---

## Infrastructure

| Service | Port | Purpose |
|---------|------|---------|
| Streamlit Dashboard | 8501 | Main UI (ngrok tunneled) |
| Qdrant Vector DB | 6333 | Market state memory |
| Redis | 6379 | Caching |
| Ollama | 11434 | Local LLM (llama3.2:3b) |
| ngrok | 4040 | Public URL access |

---

## GitHub Repos Already Integrated (vendors/)

| Repo | Author | Used For |
|------|--------|---------|
| ai-trader | aaryansinha16 | OptionsFlowDetector, SignalGenerator |
| algo-strategies | buzzsubash | Zerodha straddle execution |
| nifty-flow-analysis | raval137 | OI divergence signals |
| nse-oi-analyzer | VarunS2002 | Option chain parsing |
| nse-oi-analysis | HawkEyeCoding | OI data extraction |
| non-directional-strategy | g-ravity | Backtest framework |
| jugaad-data | jugaad-py | NSE historical data |

---

## Potential Repos to Add

| Repo | What it adds |
|------|-------------|
| `twopirllc/pandas-ta` | 130+ TA indicators (already using some) |
| `kernc/backtesting.py` | Better backtest framework with Optuna integration |
| `goldmansachs/gs-quant` | Institutional-grade risk/portfolio analytics |
| `tensortrade-org/tensortrade` | RL-based trading agent for Gold |
| `philipperemy/deep-learning-for-finance` | LSTM price prediction |
| `nickmccullum/algorithmic-trading-python` | Gold momentum strategies |
| `polakowo/vectorbt` | Vectorized backtesting (fast, 1000x faster than bar-by-bar) |

---

## Telegram Setup

- Bot Token: configured in `.env`
- Chat ID: 1094319146
- Sends: BUY/SELL signals with entry/SL/target/Greeks, WAIT-to-trade transitions, EOD summary

---

## Data Sources

| Data | Source | Freshness |
|------|--------|-----------|
| Gold price (XAUUSD) | yfinance GC=F 1m bars | ~10 min delay |
| DXY / 10Y / VIX | yfinance daily | 1-day delay |
| Nifty option chain | NSE website scraper | ~real-time |
| FII/DII data | nsepython | Daily |
| Stock news | yfinance .news | ~hourly |
