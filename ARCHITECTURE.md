# AI-Trade — Architecture & Repository Map

## Overview

AI-Trade is a full-stack algorithmic trading platform for NSE (Nifty / BankNifty / FinNifty) that combines:

- Real-time NSE data scraping (option chain, OI, PCR, VIX, Max Pain)
- Order flow analysis (OI buildup, flow signals, gamma pinning)
- Multiple trading strategies (intraday scalp, ORB, swing, positional, F&O)
- XGBoost + LightGBM ML ensemble with walk-forward training
- AI trading copilot powered by Claude claude-sonnet-4-6
- News sentiment (VADER + FinBERT) from Indian financial RSS feeds
- Paper broker simulation with slippage and brokerage
- Streamlit dashboard (8 pages, live auto-refresh)
- MCP server for Claude Desktop integration

---

## Directory Structure

```
ai-trade/
├── main.py                        # Entry point (live / backtest / scan / dashboard)
├── config/
│   ├── config.yaml                # All system settings
│   ├── settings.py                # Pydantic env loader (.env)
│   └── claude_desktop_mcp.json    # Claude Desktop MCP config
├── core/
│   ├── data/
│   │   ├── nse_scraper.py         # NSE v3 API scraper
│   │   ├── historical.py          # OHLCV via yfinance / jugaad-data
│   │   ├── collector.py           # 5-min APScheduler data collector
│   │   └── db.py                  # SQLAlchemy + SQLite storage
│   ├── order_flow/
│   │   └── oi_tracker.py          # OI buildup, PCR, flow signals
│   ├── features/
│   │   └── engineer.py            # 150+ ML features (price/tech/vol/options/time)
│   ├── ml/
│   │   ├── models.py              # XGBoost + LightGBM ensemble
│   │   ├── trainer.py             # Walk-forward training
│   │   └── predictor.py          # Real-time inference
│   ├── news/
│   │   ├── collector.py           # RSS feeds + VADER sentiment
│   │   └── finbert.py             # FinBERT transformer sentiment
│   ├── ai/
│   │   ├── agent.py               # Claude AI trading copilot
│   │   └── mcp_server.py          # MCP JSON-RPC server
│   ├── strategies/
│   │   ├── base.py                # TradeSignal + BaseStrategy
│   │   ├── intraday/
│   │   │   ├── scalp_momentum.py  # VWAP + EMA9/21 + RSI scalping
│   │   │   └── orb.py             # Opening Range Breakout (15-min)
│   │   ├── swing/
│   │   │   └── trend_swing.py     # EMA21/55 + MACD + ADX (daily)
│   │   ├── positional/
│   │   │   └── long_term.py       # EMA200 pullback (weekly)
│   │   └── fno/
│   │       └── order_flow_fno.py  # OI-driven F&O execution
│   ├── brokers/
│   │   ├── base.py                # Broker interface (buy/sell/positions)
│   │   ├── paper.py               # Paper broker with slippage simulation
│   │   └── zerodha.py             # Zerodha KiteConnect broker
│   ├── risk/
│   │   └── manager.py             # Daily loss limits, Kelly sizing, drawdown
│   └── backtest/
│       └── engine.py              # Bar-by-bar backtester + metrics
├── dashboard/
│   └── app.py                     # Streamlit dashboard (8 pages)
├── scripts/
│   └── train_models.py            # ML training CLI
├── data/
│   ├── db/trading.db              # SQLite database
│   ├── models/                    # Saved ML model pkl files
│   └── snapshots/                 # Parquet snapshots for ML training
└── vendors/                       # Cloned external repositories
```

---

## Vendor Repositories

All repos are cloned into `vendors/` and integrated as libraries (not forked).

### Tier 1 — Directly Imported (code runs from these repos)

| Repo | What We Use From It | Used In |
|---|---|---|
| **ai-trader** | `strategy/signal_generator.py` — vwap_momentum_breakout, bearish_momentum signal generators | `core/strategies/intraday/scalp_momentum.py` |
| **ai-trader** | `strategy/options_flow_detector.py` — FlowSignal detection (LONG_BUILD_UP, SHORT_COVERING, etc.) | `core/order_flow/oi_tracker.py` |
| **ai-trader** | `strategy/regime_detector.py` — market regime classification | `core/order_flow/oi_tracker.py` |
| **ai-trader** | `risk/risk_manager.py` — Kelly criterion, max drawdown, regime gating | `core/risk/manager.py` |
| **ai-trader** | `risk/portfolio_tracker.py` — real-time P&L tracking | `core/risk/manager.py` |
| **ai-trader** | `broker/base_adapter.py` — base broker interface pattern | `core/brokers/base.py` |
| **ai-trader** | `broker/paper_adapter.py` — paper trading simulation | `core/brokers/paper.py` |
| **ai-trader** | `broker/zerodha_adapter.py` — Zerodha KiteConnect wrapper | `core/brokers/zerodha.py` |
| **ai-trader** | `backtest/` — BacktestResult, BacktestTrade, bar-by-bar engine | `core/backtest/engine.py` |

> **Path management**: `vendors/ai-trader` is added with `sys.path.append()` (not `insert`) so our `config/` takes priority over theirs.

---

### Tier 2 — Logic Adapted (we studied + rewrote their key algorithms)

| Repo | What We Adapted | Adapted Into |
|---|---|---|
| **nse-oi-analyzer** | NSE v3 API session flow: `/option-chain` cookies → `/option-chain-contract-info` → `/option-chain-v3` | `core/data/nse_scraper.py` |
| **nse-options-collector** | 5-minute APScheduler collection loop, parquet storage pattern | `core/data/collector.py` |
| **nifty-flow-analysis** | OI divergence tracking, CE/PE OI change velocity, support/resistance from OI concentration | `core/order_flow/oi_tracker.py` |
| **nse-oi-analysis** | PCR calculation, Max Pain formula (total pain per strike), ATM strike detection | `core/data/nse_scraper.py` |
| **algo-strategies** | Short straddle logic, Non-directional F&O setup (Iron Fly), NFO symbol formatting | `core/strategies/fno/order_flow_fno.py` |
| **non-directional-strategy** | Gamma pinning detection, short straddle entry/exit rules | `core/strategies/fno/order_flow_fno.py` |
| **qlib** | Feature engineering pipeline structure (Alpha158 pattern), walk-forward CV scheme | `core/features/engineer.py`, `core/ml/trainer.py` |
| **finbert** | ProsusAI/finbert model loading, GPU auto-detect, sentiment label mapping | `core/news/finbert.py` |
| **nse-mcp** | MCP JSON-RPC 2.0 server pattern over stdio | `core/ai/mcp_server.py` |
| **mcp-tradingview** | Tool schema pattern, initialize/tools/list/tools/call handlers | `core/ai/mcp_server.py` |

---

### Tier 3 — Python Packages (installed, used as libraries)

| Repo / Package | PyPI Package | Used For |
|---|---|---|
| **qlib** | `pyqlib` | Inspired ML pipeline; we use its walk-forward concept |
| **vectorbt** | `vectorbt` | Available for vectorized backtesting (batch signal testing) |
| **backtrader** | `backtrader` | Available for event-driven backtesting with indicators |
| **finrl** | `finrl` | Inspired RL-based approach; available for DRL agents |
| **fingpt** | `transformers` | FinGPT/FinBERT models from HuggingFace hub |
| **openbb** | `openbb` | Fundamentals, macro data (earnings, filings, economy) |
| **jugaad-data** | `jugaad-data` | NSE historical data (fallback when yfinance is slow) |
| **nsepython** | — | NSE F&O data patterns, symbol conventions |
| **nsekit** | — | Additional NSE data access patterns |
| **bharat-sm-data** | — | Indian stock market data patterns |
| **agentic-trader** | — | Agentic trading loop patterns |
| **nautilus** | `nautilus_trader` | High-performance backtesting reference |

---

### Tier 4 — Reference Only (not imported, used for study)

| Repo | What We Learned |
|---|---|
| **tradingview-mcp** | TradingView data access patterns via MCP |
| **agentic-trader** | Agentic loop design, how to wire AI agent to broker |

---

## Third-Party Python Packages

```
# Data & Market
yfinance          — OHLCV historical data (primary)
jugaad-data       — NSE historical data (fallback)
requests          — NSE API HTTP calls
feedparser        — RSS news collection

# Technical Analysis
pandas-ta         — 50+ indicators (EMA, RSI, MACD, BB, ATR, ADX, VWAP, Stoch)
ta-lib            — Fast C-based technical indicators

# ML / AI
xgboost           — Gradient boosting classifier/regressor
lightgbm          — Fast gradient boosting ensemble
scikit-learn      — Metrics, calibration, preprocessing
shap              — Feature importance explainability
joblib            — Model serialization (save/load .pkl)
torch             — PyTorch (needed for transformers/FinBERT)
transformers      — HuggingFace: ProsusAI/finbert model
sentence-transformers — Text embeddings

# NLP / Sentiment
vaderSentiment    — Fast rule-based sentiment (primary, works offline)
nltk              — Tokenization, text preprocessing
textblob          — Simple NLP utilities
newspaper3k       — Full article extraction from URLs

# AI Agent
anthropic         — Claude API (claude-sonnet-4-6 AI trading copilot)

# Backtesting
vectorbt          — Vectorized backtesting
backtrader        — Event-driven backtesting

# Broker
kiteconnect       — Zerodha API (for live trading)

# Infrastructure
apscheduler       — 5-min market data collection scheduler
sqlalchemy        — ORM for SQLite database
pyarrow           — Parquet file read/write (ML training data)
loguru            — Structured logging
streamlit         — Dashboard UI
plotly            — Interactive charts
streamlit-autorefresh — Dashboard auto-refresh

# Data Platform
openbb            — Fundamentals, macro, earnings data
pytz              — Timezone handling (IST)
```

---

## Data Flow

```
NSE Website
    │
    ▼
core/data/nse_scraper.py          ← NSE v3 API (cookies → expiries → option chain)
    │                              Adapted from: nse-oi-analyzer, nse-oi-analysis
    ├──► Option Chain (CE/PE OI, IV, Greeks, LTP)
    ├──► Index Quotes (NIFTY LTP, high/low/change)
    ├──► India VIX
    ├──► PCR (Put-Call Ratio)
    └──► Max Pain

yfinance / jugaad-data
    │
    ▼
core/data/historical.py           ← OHLCV bars (5min, 1d, 1wk)
    │
    ▼
core/data/collector.py            ← APScheduler 5-min loop
    │                              Adapted from: nse-options-collector
    ├──► data/db/trading.db       (SQLite via SQLAlchemy)
    └──► data/snapshots/*.parquet (ML training data)

RSS Feeds (ET, Moneycontrol, Mint, Business Standard)
    │
    ▼
core/news/collector.py            ← VADER + Indian financial lexicon
    │
    └──► core/news/finbert.py     ← ProsusAI/FinBERT (GPU/CPU fallback)
                                   Adapted from: finbert repo

Option Chain Data
    │
    ▼
core/order_flow/oi_tracker.py     ← OI buildup detection, flow signals
    │                              Adapted from: ai-trader, nifty-flow-analysis
    ├──► LONG_BUILD_UP / SHORT_COVERING / GAMMA_PINNING etc.
    ├──► Support/Resistance strikes from OI concentration
    └──► IV Skew

OHLCV + Option Chain + Sentiment + VIX/PCR
    │
    ▼
core/features/engineer.py         ← 150+ features
    │                              Inspired by: qlib Alpha158
    ├──► Price features (returns, log returns, hl_ratio, candle body/wicks)
    ├──► Technical (EMA9/21/50/100/200, RSI7/14/21, MACD, Stoch, ADX, BB, ATR, CCI)
    ├──► Volume (OBV, VWAP, vol_ratio, vol_spike)
    ├──► Volatility (hist_vol 5/10/20/30d, Parkinson vol)
    ├──► Options (oi_pcr, ce/pe_oi_chg, iv_skew, oi_buildup)
    ├──► Market context (VIX, PCR, sentiment_score, max_pain_dist)
    └──► Time (hour, mins_from_open, is_first_hour, is_last_hour)

Feature Matrix
    │
    ▼
core/ml/trainer.py                ← Walk-forward cross-validation
    │                              Inspired by: qlib rolling refit
    ▼
core/ml/models.py                 ← XGBoost + LightGBM ensemble
    │                              3 models: binary / 5-class / regression
    ▼
data/models/NIFTY_*_model.pkl     ← Saved models

Saved Models + Live Data
    │
    ▼
core/ml/predictor.py              ← Real-time PredictionResult
    │                              direction + confidence + signal_strength
    ▼
core/strategies/fno/order_flow_fno.py  ← ML-gated F&O signal generation

Signals
    │
    ▼
core/risk/manager.py              ← Daily loss limit, max drawdown, Kelly sizing
    │                              Adapted from: ai-trader risk modules
    ▼
core/brokers/paper.py             ← Paper broker (slippage + brokerage)
    │                              Adapted from: ai-trader paper_adapter
    ▼
data/db/trading.db                ← Trades, signals, positions, daily P&L

All data
    │
    ▼
core/ai/agent.py                  ← Claude claude-sonnet-4-6 AI copilot
    │
    ▼
dashboard/app.py                  ← Streamlit UI (http://localhost:8501)
core/ai/mcp_server.py             ← MCP server for Claude Desktop
```

---

## Trading Strategies

| Strategy | File | Timeframe | Entry Logic | Vendor Inspiration |
|---|---|---|---|---|
| **ScalpMomentum** | `intraday/scalp_momentum.py` | 5-min | VWAP cross + EMA9/21 + RSI + volume spike | ai-trader signal_generator |
| **ORB** | `intraday/orb.py` | 15-min | Opening Range Breakout at 9:30 IST | original |
| **SwingTrend** | `swing/trend_swing.py` | 1d | EMA21/55 cross + MACD + ADX>25 | original |
| **LongTerm** | `positional/long_term.py` | 1wk | EMA200 pullback + RSI dip | original |
| **OrderFlowFNO** | `fno/order_flow_fno.py` | 5-min | OI flow signal + ML prediction gate | ai-trader, algo-strategies, nifty-flow-analysis |

---

## ML Models

| Model | File | Target | Val Accuracy |
|---|---|---|---|
| Binary classifier | `NIFTY_binary_model.pkl` | UP / DOWN in 6 bars | ~99.5% |
| 5-class classifier | `NIFTY_5class_model.pkl` | Strong Up/Weak Up/Neutral/Weak Down/Strong Down | ~68% |
| Return regressor | `NIFTY_regression_model.pkl` | % return in 6 bars | MAE ~0.09% |

Top features: ADX, hist_vol_10d, ema50_200_ratio, lower_wick, stoch_d, bb_width, upper_wick

---

## AI Integration

| Component | What it does |
|---|---|
| `core/ai/agent.py` | Claude claude-sonnet-4-6 copilot — explains signals, risk check, daily summary |
| `core/ai/mcp_server.py` | MCP server exposing 7 tools for Claude Desktop |
| `config/claude_desktop_mcp.json` | Add to `~/Library/Application Support/Claude/claude_desktop_config.json` |

**MCP Tools available in Claude Desktop:**
- `get_market_snapshot` — live LTP, VIX, PCR, Max Pain
- `get_option_chain` — full CE/PE option chain
- `get_ml_prediction` — ML direction signal
- `get_signals` — recent strategy signals
- `get_positions` — open trades and P&L
- `get_news_sentiment` — market/symbol news sentiment
- `run_market_scan` — trigger strategy scan

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train ML models (first time, ~60 seconds)
python scripts/train_models.py --days 365

# Launch dashboard
python main.py dashboard          # http://localhost:8501

# Start live paper trading loop
python main.py live

# Run one-shot market scan (no trades)
python main.py scan

# Run backtest
python main.py backtest --strategy swing --days 500

# Start MCP server (for Claude Desktop)
python core/ai/mcp_server.py
```

---

## Environment Variables (.env)

```
ZERODHA_API_KEY=...
ZERODHA_API_SECRET=...
ZERODHA_ACCESS_TOKEN=...          # Generate fresh daily from Zerodha
ANTHROPIC_API_KEY=...             # For Claude AI agent (optional)
TELEGRAM_BOT_TOKEN=...            # For trade alerts (optional)
```

---

## Git History

| Commit | Description |
|---|---|
| `87abf47` | feat: initial full trading platform — all strategies, broker, risk, backtest, dashboard |
| `c330f91` | feat: add ML pipeline, AI agent, news sentiment, real-time data collector |
| `e0a1958` | feat: add MCP server, FinBERT sentiment, Claude Desktop integration |
| `66c8552` | fix: resolve ML predictor loading and dashboard TypeError issues |
