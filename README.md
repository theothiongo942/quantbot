# ⚡ QuantBot — Institutional-Grade Crypto Scalping System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Binance](https://img.shields.io/badge/Exchange-Binance%20Futures-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square)

A fully modular, institutional-grade quantitative trading bot built for Binance Futures. Combines 11 technical indicators into a single conviction score, supports dynamic leverage tiers, ATR-based risk management, and runs a live Flask dashboard for real-time monitoring and control.

Supports both **scalping** (meme coins) and **swing trading** (BTC, ETH, BNB) simultaneously on the same capital pool.

---

## 🖥️ Live Dashboard

![Dashboard Preview](https://i.imgur.com/placeholder.png)

- Real-time candlestick charts powered by TradingView Lightweight Charts
- Live open positions with unrealised PnL updating every 3 seconds
- Signal scanner showing conviction scores for all symbols
- Auto pilot toggle to start and stop trading instantly
- Manual position close buttons
- Trade log with every action the bot takes
- Kill switch indicator that turns red if daily drawdown limit is hit

---

## 🏗️ Architecture

```
Binance WebSocket
       │
       ▼
┌─────────────────┐
│  Module 1       │  Real-time price data, order books,
│  Data Feed      │  VWAP, funding rates, aggr trades
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module 2       │  11 indicators → weighted conviction
│  Signal Engine  │  score → BUY / SELL / NEUTRAL
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐ ┌──────────────┐
│Module 3│ │  Module 4    │
│  Meme  │ │    Risk      │
│Scanner │ │   Manager    │
└────┬───┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            │
            ▼
┌─────────────────┐
│  Module 5       │  Paper trading executor
│  Executor       │  (LiveExecutor available)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module 6       │  Equity curve, win rate,
│  Portfolio      │  PnL tracking, trade stats
│  Tracker        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Module 7       │  Flask web dashboard
│  Dashboard      │  http://localhost:5000
└─────────────────┘
```

---

## 📊 Signal Engine — 11 Indicators

Each indicator returns a score from -1.0 (strong sell) to +1.0 (strong buy). Scores are weighted and combined into a single conviction score from 0 to 1.

| Indicator | Weight | What it measures |
|---|---|---|
| RSI | 10% | Overbought / oversold conditions |
| EMA Crossover | 12% | Trend direction and momentum |
| MACD | 12% | Trend strength and crossovers |
| Bollinger Bands | 8% | Price relative to volatility range |
| Stochastic | 8% | Short term momentum |
| ATR Filter | 5% | Volatility environment |
| VWAP | 10% | Price vs institutional average |
| Order Flow | 15% | Real time buy vs sell pressure |
| Volume Profile | 8% | Unusual volume activity |
| Momentum Divergence | 7% | Price vs RSI divergence |
| Funding Rate | 5% | Market sentiment and crowding |

**Conviction → Leverage mapping:**
```
0.52 - 0.64  →  10x leverage
0.65 - 0.79  →  20x leverage
0.80+        →  50x leverage
```

---

## 🛡️ Risk Management

- **2% risk per trade** — position sized by ATR stop distance
- **Hard margin cap** — $500 maximum margin per trade regardless of price
- **Max 10 simultaneous positions**
- **8% daily drawdown kill switch** — halts all trading automatically
- **Anti-liquidation protection** — closes position if margin ratio drops below 20%
- **ATR-based stop loss and take profit** — adapts to current volatility

---

## 🎯 Trading Modes

### Scalp Mode (Meme Coins)
Targets fast 0.1% to 0.3% moves on PEPE, DOGE, SHIB, WIF, BONK.
- Tight stops at 0.5x ATR
- Quick take profit at 1.5x ATR
- Full position close at TP — capital recycled immediately
- Scans every 3 seconds

### Swing Mode (Core Coins)
Targets larger moves on BTC, ETH, BNB.
- Wider stops at 1.0x ATR
- Take profit at 2.5x ATR
- Partial close at 1.5x ATR to lock in profits

Both modes run simultaneously on the same capital pool.

---

## 📁 Project Structure

```
quantbot/
├── config/
│   ├── __init__.py
│   ├── settings.py          ← all tuneable parameters
│   └── logging_config.py    ← structured coloured logging
├── modules/
│   ├── __init__.py
│   ├── data_feed.py         ← WebSocket + REST data feed
│   ├── signal_engine.py     ← 11 indicator scoring system
│   ├── meme_scanner.py      ← meme coin opportunity scanner
│   ├── risk_manager.py      ← position sizing and risk control
│   ├── executor.py          ← paper and live trade execution
│   ├── portfolio_tracker.py ← equity curve and performance stats
│   └── dashboard.py         ← Flask web dashboard
├── tests/
│   ├── test_data_feed.py    ← 41 tests
│   ├── test_signal_engine.py ← 51 tests
│   ├── test_meme_scanner.py ← 36 tests
│   ├── test_risk_manager.py ← 37 tests
│   ├── test_executor.py     ← 34 tests
│   └── test_portfolio_tracker.py ← 39 tests
├── logs/
└── main.py                  ← entry point
```

**Total: 238 tests across 6 modules — all passing**

---

## ⚙️ Installation

**Requirements:**
- Python 3.12
- Binance account (free — no funds required for paper trading)

**Install dependencies:**
```bash
pip install pandas numpy websocket-client requests flask
```

**Configure API keys:**

Copy `config/settings.py` and add your Binance API keys:
```python
BINANCE_API_KEY    = "your_api_key"
BINANCE_API_SECRET = "your_api_secret"
```

> Your API keys only need **read permission** for paper trading. Never enable withdrawal permissions.

---

## 🚀 Running the Bot

```bash
python main.py
```

Then open your browser at:
```
http://localhost:5000
```

Refresh the page, turn on **AUTO PILOT** and the bot starts trading.

---

## 🧪 Running Tests

Run all tests for a single module:
```bash
py -3.12 tests\test_data_feed.py
py -3.12 tests\test_signal_engine.py
py -3.12 tests\test_meme_scanner.py
py -3.12 tests\test_risk_manager.py
py -3.12 tests\test_executor.py
py -3.12 tests\test_portfolio_tracker.py
```

---

## 🔄 Switching to Live Trading

When you are ready to trade with real money:

1. Enable **Futures trading** on your Binance account
2. Set `USE_TESTNET = True` in `settings.py` first
3. Test on Binance testnet for at least 2 weeks
4. In `main.py` replace:
```python
from modules.data_feed import DataFeed        # already done
from modules.executor import LiveExecutor     # swap PaperExecutor
```
5. Start with small capital — $500 maximum until proven profitable

> ⚠️ Leverage trading carries significant risk. Only trade with money you can afford to lose. Paper trade for at least 4 to 6 weeks before going live.

---

## 📈 Symbols Tracked

| Category | Symbols |
|---|---|
| Core | BTCUSDT, ETHUSDT, BNBUSDT |
| Meme | 1000PEPEUSDT, DOGEUSDT, 1000SHIBUSDT, WIFUSDT, 1000BONKUSDT |

---

## 🛠️ Key Configuration Parameters

All parameters are in `config/settings.py`:

```python
CAPITAL_RISK_PER_TRADE    = 0.005   # 0.5% of equity per trade
MAX_SIMULTANEOUS_POSITIONS = 10      # max open trades at once
DAILY_DRAWDOWN_KILL_PCT   = 0.08    # 8% drawdown kills trading
MAX_MARGIN_PER_TRADE_USD  = 500     # hard cap per position
SCALP_ATR_TP_MULT         = 1.5     # scalp take profit multiplier
SWING_ATR_TP_MULT         = 2.5     # swing take profit multiplier
```

---

## 📜 License

MIT License — free to use, modify and distribute.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance of paper trading does not guarantee future results in live trading. The authors are not responsible for any financial losses incurred from using this software.

---

Built with Python 🐍 | Powered by Binance Futures API
