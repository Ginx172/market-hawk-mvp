# Market Hawk 3 — AI Trading System

## Project Overview

Market Hawk 3 is a Python-based AI trading system with multi-agent architecture,
ML models (CatBoost, XGBoost, BiLSTM, Transformer, GNN, VAE, RL), and a RAG
knowledge base (~140K chunks from 280+ trading books).

**GitHub:** https://github.com/Ginx172/market-hawk-mvp
**Status:** Active development — MVP phase
**Owner:** Gigi (Gigel Dumitru)

## Tech Stack

- **Language:** Python 3.10+
- **ML Frameworks:** PyTorch, CatBoost, XGBoost, scikit-learn
- **Data:** pandas, numpy, polars (where performance matters)
- **API/Backend:** FastAPI
- **Database:** SQLite (dev), PostgreSQL (planned)
- **RAG:** FAISS / ChromaDB for vector storage
- **Broker APIs:** Interactive Brokers (TWS API), Alpaca (planned)
- **GPU:** NVIDIA GTX 1070 8GB VRAM — always optimize for this constraint

## Hardware Constraints — CRITICAL

All code MUST be optimized for this hardware:
- CPU: Intel i7-9700F (8 cores, no hyperthreading)
- GPU: NVIDIA GTX 1070 (8GB VRAM DDR5)
- RAM: 64GB DDR4 ~2666MHz
- Storage: Multiple drives (K: for dev, G: for models/data)

**Rules:**
- Never load models > 6GB into VRAM (leave headroom)
- Use mixed precision (fp16) for training when possible
- Process large datasets in chunks (max 10K rows per batch for GPU ops)
- Always call torch.cuda.empty_cache() and gc.collect() after large operations
- Use DataLoader with num_workers=4 (half of 8 cores)
- Monitor VRAM usage before allocating tensors

## Architecture

```
Market_Hawk_3/
├── src/
│   ├── core/           # Strategy engine, portfolio manager, risk manager
│   ├── agents/         # Multi-agent system (analyst, trader, risk agents)
│   ├── ml/             # ML models, training pipelines, feature engineering
│   ├── data/           # Data fetching, cleaning, storage
│   ├── indicators/     # Technical indicators (RSI, MACD, Bollinger, custom)
│   ├── broker/         # Broker API integrations (IB, Alpaca)
│   ├── rag/            # RAG knowledge base, embeddings, retrieval
│   ├── backtest/       # Backtesting engine
│   └── utils/          # Logging, config, memory management, helpers
├── config/             # YAML/JSON configs (NO credentials here)
├── data/               # Market data, processed datasets
├── models/             # Saved models, checkpoints
├── logs/               # Logging output
├── tests/              # pytest tests
├── scripts/            # Standalone utility scripts
└── notebooks/          # Jupyter notebooks for exploration
```

## Coding Standards

- **Level:** Doctoral/enterprise-grade Python
- **Style:** PEP 8 strict, type hints on ALL functions
- **Docstrings:** Google-style, detailed
- **Error handling:** Never silent fails — always log + raise or handle
- **Logging:** Use Python `logging` module, structured JSON logs preferred
- **Comments:** In English for code, Romanian OK for inline notes
- **Testing:** pytest, minimum 80% coverage on critical modules

## Mandatory Script Components

Every new script or module MUST include:
1. **Pause/Resume** — Save state on interrupt, resume from last checkpoint
2. **Memory Management** — torch.cuda.empty_cache(), gc.collect() in loops
3. **Chunk Processing** — Never load full dataset in memory
4. **Progress Bars** — tqdm with ETA, samples/sec, memory usage
5. **Timer** — Track elapsed time, estimate completion
6. **Logging** — File + console, with performance metrics
7. **Checkpointing** — Auto-save every 100 epochs/iterations, keep last 10

## Common Commands

```bash
# Run tests
python -m pytest tests/ -v

# Type checking
python -m mypy src/ --ignore-missing-imports

# Lint
python -m flake8 src/ --max-line-length=120

# Run backtest
python scripts/run_backtest.py --config config/backtest_config.yaml

# Start API server
python -m uvicorn src.api.main:app --reload --port 8000
```

## Security Rules — NEVER VIOLATE

- NEVER commit .env files, API keys, or credentials
- NEVER hardcode broker passwords, API keys, or tokens
- Use environment variables or encrypted config for all secrets
- .env and secrets/ are in .gitignore AND .claudeignore
- Validate all external API inputs before processing

## Key Decisions

- CatBoost is the primary production model (best results on our data)
- Backtesting engine must support walk-forward optimization
- All data timestamps in UTC
- Use decimal.Decimal for money/price calculations, never float
- Feature engineering pipeline must be reproducible (seed everything)

## Current Priorities (March 2026)

1. Credentials security — move all secrets to env vars
2. Modular architecture — clean separation of concerns
3. Backtesting engine — fully functional for investor demos
4. CatBoost model optimization — hyperparameter tuning
5. Data visualization dashboard — Plotly/Dash

## Known Issues

- Some modules are stubs (declared but not implemented) — audit found ~10-15% genuine implementation
- Code scattered across C:, G:, J:, K: drives — consolidation ongoing to K: + GitHub
- RAG knowledge base needs re-indexing after chunk format changes
- No CI/CD pipeline yet

## When Modifying Code

- Always check if a function is a stub before building on it
- Prefer modifying existing files over creating new ones
- Run tests after changes: `python -m pytest tests/ -v`
- Keep commits small and focused (one feature/fix per commit)
- Write descriptive commit messages in English
