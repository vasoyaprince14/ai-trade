# AI-Trade | Nifty Algorithmic Trading Platform

A full-featured trading platform for NSE F&O — combining multiple open-source repos into one unified system.

---

## Architecture

```
ai-trade/
├── vendors/                    # Cloned GitHub repos (sources)
│   ├── ai-trader/             # Full-stack AI trading system (aaryansinha16)
│   ├── algo-strategies/       # NSE F&O strategies (buzzsubash)
│   ├── nifty-flow-analysis/   # Institutional OI flow tracker (raval137)
│   ├── nse-oi-analyzer/       # Option chain analyzer (VarunS2002)
│   ├── nse-oi-analysis/       # OI data + analysis (HawkEyeCoding)
│   ├── non-directional-strategy/ # Non-directional options (g-ravity)
│   ├── jugaad-data/           # NSE historical data library
│   └── agentic-trader/        # AI agentic trader
│
├── core/                      # Our integration/orchestration layer
│   ├── data/
│   │   ├── nse_scraper.py     # NSE live data (option chain, quotes, PCR)
│   │   ├── historical.py      # Historical OHLCV (jugaad + yfinance)
│   │   └── db.py              # SQLite database (trades, signals, OI)
│   ├── order_flow/
│   │   └── oi_tracker.py      # OI tracker + ai-trader FlowDetector
│   ├── strategies/
│   │   ├── base.py            # BaseStrategy interface
│   │   ├── intraday/
│   │   │   ├── scalp_momentum.py  # VWAP+EMA+RSI (5-min)
│   │   │   └── orb.py             # Opening Range Breakout (15-min)
│   │   ├── swing/
│   │   │   └── trend_swing.py     # EMA21/55 + MACD (daily)
│   │   ├── positional/
│   │   │   └── long_term.py       # EMA200 (weekly)
│   │   └── fno/
│   │       └── order_flow_fno.py  # OI flow → straddle/directional
│   ├── brokers/
│   │   ├── base.py            # Abstract broker interface
│   │   ├── paper.py           # Paper trading (no real money)
│   │   └── zerodha.py         # Zerodha Kite Connect
│   ├── risk/
│   │   └── manager.py         # Risk limits, position sizing
│   └── backtest/
│       └── engine.py          # Bar-by-bar backtesting + metrics
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── config/
│   ├── config.yaml            # All settings
│   └── settings.py            # Pydantic settings + .env loader
│
├── data/
│   └── db/trading.db          # SQLite database
├── logs/                      # Daily log files
├── main.py                    # Entry point
└── requirements.txt
```

---

## Integrated Repositories

| Repo | What We Use |
|------|-------------|
| [aaryansinha16/AI-trader](https://github.com/aaryansinha16/AI-trader) | OptionsFlowDetector, SignalGenerator, RiskManager, BacktestEngine, PaperAdapter, ZerodhaAdapter |
| [buzzsubash/algo_trading_strategies_india](https://github.com/buzzsubash/algo_trading_strategies_india) | Short straddle, iron fly strategies for Zerodha |
| [raval137/NIFTY-BANKNIFTY-CALL-PUT-Live-Market-Analysis](https://github.com/raval137/NIFTY-BANKNIFTY-CALL-PUT-Live-Market-Analysis) | OI divergence engine, NSE API polling with anti-block |
| [VarunS2002/Python-NSE-Option-Chain-Analyzer](https://github.com/VarunS2002/Python-NSE-Option-Chain-Analyzer) | Option chain analysis, continuous refresh logic |
| [HawkEyeCoding/nse-oi-analysis](https://github.com/HawkEyeCoding/nse-oi-analysis) | OI data extraction & intraday trend generation |
| [g-ravity/non-directional-options-strategy](https://github.com/g-ravity/non-directional-options-strategy) | Non-directional strategy backtesting logic |
| [jugaad-py/jugaad-data](https://github.com/jugaad-py/jugaad-data) | NSE historical data (bhavcopy, F&O, index data) |

---

## Strategies

### Intraday (5-min / 15-min)
- **Scalp Momentum**: VWAP + EMA9/21 crossover + RSI filter + Volume spike → Buy ATM Call/Put
- **ORB**: 15-min Opening Range Breakout → Buy Call/Put on breakout/breakdown

### Swing (Daily)
- **Trend Following**: EMA21/55 + MACD crossover + ADX strength → 3-20 day holds

### Positional (Weekly)
- **Long Term**: EMA200 trend filter + pullback entry → 4-52 week holds

### F&O
- **Order Flow FNO**:
  - Gamma Pinning + High IV → Short Straddle / Iron Fly
  - OI Long Build-Up + Bullish PCR → Buy ATM Call
  - OI Short Build-Up + Bearish PCR → Buy ATM Put
  - Max Pain divergence → Reversion trade

---

## Quick Start

### 1. Install Dependencies
```bash
cd ai-trade
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your broker credentials
# Edit config/config.yaml for strategy settings
```

### 3. Run Market Scan (no trades)
```bash
python main.py scan
```

### 4. Launch Dashboard
```bash
python main.py dashboard
# Opens at http://localhost:8501
```

### 5. Paper Trading
Set `mode: paper` in `config/config.yaml`, then:
```bash
python main.py live
```

### 6. Backtest
```bash
python main.py backtest --strategy swing --days 500
python main.py backtest --strategy all --days 365
```

---

## Configuration (config/config.yaml)

Key settings:
```yaml
app:
  mode: "paper"          # paper | live

brokers:
  default: "paper"       # paper | zerodha | angel

risk:
  max_capital_per_trade_pct: 5   # 5% per trade
  max_daily_loss_pct: 2          # Stop at 2% daily loss
  max_drawdown_pct: 15           # Stop at 15% drawdown
```

---

## Broker Setup

### Zerodha
1. Create app at https://developers.kite.trade
2. Add to `.env`: `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`
3. Generate access token daily (or set up TOTP automation)
4. Set `brokers.default: zerodha` in `config.yaml`

### Angel One
1. Create app at https://smartapi.angelbroking.com
2. Add to `.env`: `ANGEL_API_KEY`, `ANGEL_CLIENT_ID`, `ANGEL_PASSWORD`
3. Set `brokers.default: angel` in `config.yaml`

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**.
- Always start with paper trading
- Never risk money you can't afford to lose
- Live F&O trading requires SEBI compliance for algo trading
- Backtested results do not guarantee future performance
