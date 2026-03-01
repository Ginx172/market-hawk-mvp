# 🦅 Market Hawk MVP

**AI-Powered Multi-Agent Trading Intelligence System**

Market Hawk is a production-grade algorithmic trading system that combines machine learning, natural language processing, and real-time market analysis through a coordinated multi-agent architecture. The system orchestrates 5 specialized AI agents through a central Brain to generate consensus-driven trading decisions.

---

## Architecture

```
                    ┌─────────────────────┐
                    │    🧠 THE BRAIN     │
                    │  Central Orchestrator│
                    │  Weighted Consensus  │
                    └──────────┬──────────┘
                               │
          ┌────────────┬───────┼───────┬────────────┐
          │            │       │       │            │
    ┌─────▼─────┐ ┌───▼───┐ ┌─▼──┐ ┌──▼──┐ ┌──────▼──────┐
    │ Knowledge │ │  ML   │ │News│ │Sec. │ │    Risk     │
    │  Advisor  │ │Signal │ │Ana.│ │Guard│ │   Manager   │
    │  (RAG)    │ │Engine │ │    │ │     │ │  (Kelly)    │
    └───────────┘ └───────┘ └────┘ └─────┘ └─────────────┘
     140K chunks   CatBoost  LLM   Anomaly   Quarter-Kelly
     320+ books    Ensemble  Sent.  Detect.   Position Size
```

## Agents

| Agent | Role | Technology | Function |
|-------|------|-----------|----------|
| **Knowledge Advisor** | VOTER | ChromaDB + RAG + Ollama | Retrieves insights from 140,512 text chunks across 320+ trading books |
| **ML Signal Engine** | VOTER | CatBoost Ensemble | 2-model ensemble generating BUY/SELL/HOLD signals from 60 technical features |
| **News Analyzer** | VOTER | RSS + LLM Sentiment | Real-time news sentiment analysis using qwen3:8b via Ollama |
| **Security Guard** | VOTER | Statistical Anomaly Detection | Detects volume spikes, price gaps, volatility explosions, spread anomalies |
| **Risk Manager** | GATEKEEPER | Quarter-Kelly Criterion | Position sizing, stop-loss/take-profit, portfolio risk limits — can veto trades |

## How It Works

1. **Data Pipeline** fetches live OHLCV data via yfinance and engineers 60 technical features (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, session flags, etc.)

2. **Brain** sends the market context to all voting agents simultaneously

3. Each agent returns a recommendation (BUY/SELL/HOLD) with a confidence score

4. **Brain** calculates weighted consensus:
   - ML Signal Engine: 35% weight
   - Knowledge Advisor: 25% weight
   - Risk Manager: 20% weight
   - News Analyzer: 15% weight
   - Security Guard: 5% weight

5. If consensus exceeds threshold (0.60), the **Risk Manager** gates the final decision with position sizing via Quarter-Kelly criterion

6. All decisions are logged to JSONL audit trail for analysis

## Paper Trading

The system includes a fully automated paper trading loop:

```bash
# Single scan across 13 symbols
python trading/paper_trader.py --once

# Continuous hourly scanning
python trading/paper_trader.py --interval 60

# Check portfolio status
python trading/paper_trader.py --status
```

**Watchlist**: AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA, BTC, ETH, GOLD, SILVER, SPY, QQQ

**Portfolio features**: $100K virtual capital, automatic stop-loss/take-profit, position tracking, win rate, drawdown monitoring, trade history logging.

## ML Models

Located in the external model repository, the system supports 9 trained models:

| Model | Accuracy | Notes |
|-------|----------|-------|
| catboost_v2 (native .cbm) | Walk-forward TBD | Primary model, 300 iterations |
| CatBoost CLEAN | 75.3% | Rebuilt without data leakage |
| CatBoost ULTRA | 75.0% | Hyperparameter optimized |
| CatBoost Commodities | 82% | Per-category specialist |
| CatBoost Indices | 76% | Per-category specialist |
| CatBoost Crypto | 56% | Needs improvement |

All models use 56-60 features calculated from OHLCV data using standard technical analysis indicators.

## Knowledge Base

The Knowledge Advisor draws from a ChromaDB vector database containing:
- **140,512 text chunks** from 320+ algorithmic trading documents
- Sources include: quantitative finance textbooks, trading strategy guides, risk management literature, market microstructure research
- Embedding model: nomic-embed-text
- Retrieval: Maximum Marginal Relevance (k=15, fetch_k=60, λ=0.7)
- LLM: qwen3:8b via Ollama for response generation

## Project Structure

```
Market_Hawk_3/
├── brain/
│   └── orchestrator.py          # Central Brain — consensus engine
├── agents/
│   ├── knowledge_advisor/
│   │   └── rag_engine.py        # RAG with 140K chunks
│   ├── ml_signal_engine/
│   │   └── catboost_predictor.py # 9-model registry + ensemble
│   ├── news_analyzer/
│   │   └── news_sentiment.py    # RSS + LLM sentiment
│   ├── risk_manager/
│   │   └── kelly_criterion.py   # Quarter-Kelly position sizing
│   └── security_guard/
│       └── anomaly_detector.py  # Volume/price/volatility anomalies
├── data/
│   └── market_data_fetcher.py   # yfinance + 60-feature engineering
├── trading/
│   └── paper_trader.py          # Automated paper trading loop
├── config/
│   └── settings.py              # Global configuration
├── scripts/
│   ├── test_end_to_end.py       # Full 5-agent integration test
│   └── test_brain_2agents.py    # Brain unit test
└── logs/
    ├── decisions.jsonl           # Brain decision audit trail
    ├── paper_trades.jsonl        # Paper trading log
    └── paper_portfolio.json      # Portfolio state (persistent)
```

## Tech Stack

- **Python 3.12** — Core runtime
- **CatBoost** — Gradient boosting ML models
- **ChromaDB** — Vector database for knowledge retrieval
- **LangChain** — RAG pipeline orchestration
- **Ollama** — Local LLM inference (qwen3:8b)
- **yfinance** — Live market data
- **feedparser** — RSS news feeds
- **NumPy / Pandas** — Feature engineering

## Hardware Requirements

Optimized for:
- **CPU**: Intel i7-9700F (8 cores) or equivalent
- **GPU**: NVIDIA GTX 1070 8GB VRAM (for local LLM inference)
- **RAM**: 64GB DDR4
- **Storage**: SSD recommended for ChromaDB performance

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Ginx172/market-hawk-mvp.git
cd market-hawk-mvp

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install catboost chromadb langchain langchain-ollama
pip install yfinance feedparser numpy pandas

# Ensure Ollama is running with qwen3:8b
ollama pull qwen3:8b

# Run full 5-agent test
python scripts/test_end_to_end.py

# Start paper trading
python trading/paper_trader.py --once
```

## Sample Output

```
🦅 MARKET HAWK MVP — FULL 5-AGENT LIVE DECISION

  🎯 NVDA (tech_stocks)
  Price: $177.16 | Data: 416 candles | Features: 60

  ⚪ DECISION: HOLD
  Consensus:  +0.3722 (threshold: 0.60)

  Agent Votes:
    ⚪ knowledge_advisor  | HOLD | conf=0.50
    🟢 ml_signal_engine   | BUY  | conf=0.52
    🟢 news_analyzer      | BUY  | conf=0.77
    ⚪ security_guard     | HOLD | conf=0.00

  🚨 SILVER [HIGH] price_gap: 3.50%
  🚨 QQQ [MEDIUM] volume_spike: Z-score 4.2
```

## Roadmap

- [x] Brain Orchestrator with weighted consensus
- [x] Knowledge Advisor (140K chunks RAG)
- [x] ML Signal Engine (CatBoost ensemble)
- [x] News Analyzer (RSS + LLM sentiment)
- [x] Security Guard (anomaly detection)
- [x] Risk Manager (Quarter-Kelly)
- [x] Data Pipeline (60 features)
- [x] Paper Trading Loop
- [ ] Streamlit Dashboard
- [ ] Walk-forward model validation
- [ ] Backtesting engine
- [ ] Live broker integration (IBKR/Alpaca)
- [ ] Multi-timeframe analysis

## License

This project is proprietary software. All rights reserved.

## Author

Developed as part of an AI-driven algorithmic trading research initiative, combining expertise in financial markets, machine learning, and multi-agent systems.
