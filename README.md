# 🦅 Market Hawk MVP — AI Multi-Agent Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: MVP Development](https://img.shields.io/badge/Status-MVP%20Development-orange)]()

## 🎯 Executive Summary

**Market Hawk** is an AI-powered multi-agent trading system that combines:
- **76.47% accuracy** walk-forward validated CatBoost model (47K samples)
- **140K+ knowledge chunks** from 280+ trading books via RAG (ChromaDB)
- **7 specialized AI agents** coordinated by a central "Brain" orchestrator
- **200+ symbols** across 4 continents (US, UK, EU, Asia + Forex + Crypto)
- **Smart Money Concepts** (Order Blocks, FVG, Liquidity Zones, BOS)
- **Multimodal pipeline** (chart images + text analysis via LLaVA fine-tuning)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   THE BRAIN                         │
│            (Central Orchestrator)                    │
│       Weighted Consensus Decision Engine            │
├─────────┬──────────┬──────────┬────────────────────┤
│         │          │          │                      │
▼         ▼          ▼          ▼                      ▼
┌───────┐┌────────┐┌────────┐┌──────────┐┌──────────────┐
│Knowledge││ML Signal││ Risk   ││ Security ││  Continuous  │
│Advisor ││ Engine  ││Manager ││  Guard   ││  Learning    │
│ (RAG)  ││(CatBoost)││(Kelly) ││(Cyber)  ││  Agent       │
└───────┘└────────┘└────────┘└──────────┘└──────────────┘
     │         │         │          │             │
     ▼         ▼         ▼          ▼             ▼
┌──────────────────────────────────────────────────────┐
│              EXECUTOR + ACTIVITY LOGGER              │
│         (Order Execution + Full Audit Trail)         │
└──────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (MVP)                │
│   Live Signals │ P&L Curve │ RAG Chat │ Agent Log    │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Market_Hawk_3/
├── README.md                    # This file
├── ROADMAP.md                   # Development roadmap & priorities
├── ARCHITECTURE.md              # Detailed system architecture
├── INVESTOR_BRIEF.md            # Investor presentation summary
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
│
├── config/                      # Global configuration
│   ├── settings.py              # All settings, paths, hardware profile
│   └── agent_weights.py         # Agent voting weights
│
├── brain/                       # THE BRAIN — Central Orchestrator
│   └── orchestrator.py          # Weighted consensus, audit trail
│
├── agents/                      # Specialized AI Agents
│   ├── knowledge_advisor/       # RAG from 280+ trading books
│   │   └── rag_engine.py        # ChromaDB + MMR retrieval
│   ├── ml_signal_engine/        # CatBoost 76.47% model
│   │   └── catboost_predictor.py
│   ├── risk_manager/            # Kelly Criterion + position sizing
│   │   └── kelly_criterion.py
│   ├── security_guard/          # Multi-layer cybersecurity
│   ├── continuous_learner/      # Self-improvement from history
│   └── news_analyzer/           # Sentiment & event detection
│
├── executor/                    # Trade execution
│   └── broker_adapters/         # ccxt, Alpaca adapters
│
├── data/                        # Data fetching & preprocessing
├── dashboard/                   # Streamlit MVP dashboard
│   ├── app.py                   # Main app (5 pages)
│   ├── pages/                   # Individual page modules
│   └── components/              # Reusable UI components
│
├── models/                      # ML model storage
│   ├── trained/                 # Checkpoints (.cbm, .pkl)
│   └── configs/                 # Hyperparameter configs
│
├── knowledge_base/              # RAG vector store
│   ├── chromadb/                # ChromaDB persistence
│   ├── documents/               # Source document index
│   └── metadata/                # Document stats
│
├── logs/                        # Application & decision logs
├── tests/                       # Test suite
└── scripts/                     # Utility scripts
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Ginx172/Market_Hawk_3.git
cd Market_Hawk_3

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
copy .env.example .env
# Edit .env with your API keys

# Run the dashboard
streamlit run dashboard/app.py
```

---

## 📊 Validated Results

| Metric | Value |
|--------|-------|
| Model Accuracy (Walk-Forward) | **76.47%** |
| Validation Samples | 47,000 |
| Asset Coverage | 200+ symbols |
| Markets | US, UK, EU, Asia, Forex, Crypto |
| Knowledge Base | 140K+ chunks from 280+ books |
| Target Knowledge Base | 1,600+ documents (in progress) |

---

## 🔧 Hardware Optimization

Optimized for:
- **CPU**: Intel i7-9700F (8 cores)
- **GPU**: NVIDIA GeForce GTX 1070 (8GB VRAM)
- **RAM**: 64GB DDR4

---

## 📚 Knowledge Sources

The Knowledge Advisor is built on a comprehensive trading library including:
- Smart Money Concepts & Order Flow
- Technical Analysis (classical + modern)
- Trading Psychology & Risk Management
- Quantitative Finance & Algorithmic Trading
- Market Microstructure
- Options & Derivatives strategies

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

**Phase 1 (Current):** Module scaffolding + Knowledge Advisor integration
**Phase 2:** ML Signal Engine + Brain Orchestrator
**Phase 3:** Risk Manager + Paper Trading
**Phase 4:** Dashboard MVP for investor demo
**Phase 5:** Continuous Learning + Security hardening

---

## 📋 Related Repositories (Legacy)

This repository consolidates work from:
- [market-hawk](https://github.com/Ginx172/market-hawk) — Original architecture & CatBoost model
- [Live_trading_automated_AI_model](https://github.com/Ginx172/Live_trading_automated_AI_model) — Multimodal pipeline & CLI agent
- [Ultimate_Trade_Agentic_RAG](https://github.com/Ginx172/Ultimate_Trade_Agentic_RAG) — Agentic RAG system

**Educational references (forks):**
- [Harvard-Algorithmic-Trading-with-AI](https://github.com/Ginx172/Harvard-Algorithmic-Trading-with-AI) — RBI methodology
- [machine-learning-for-trading](https://github.com/Ginx172/machine-learning-for-trading) — Stefan Jansen's ML4Trading

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Gigi** — AI Trading Systems Developer
- Background: Economics & Finance (Danubius University), Military Operations, NHS Healthcare
- Trading experience: Since late 1990s (Bucharest Stock Exchange), Forex since 2016
- Currently: Level 3 Data Science bootcamp + AI Trading R&D
