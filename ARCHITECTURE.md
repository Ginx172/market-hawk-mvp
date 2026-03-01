# 🏗️ Market Hawk MVP — System Architecture

## Overview

Market Hawk is a **multi-agent AI trading system** where specialized agents collaborate
through a central Brain orchestrator to make trading decisions. Each agent has domain
expertise and votes on decisions with configurable weights.

---

## Agent Architecture

### 1. THE BRAIN (Orchestrator)
**Role:** Central coordinator, final decision maker

```
Input Sources:
  - User queries (chat/CLI)
  - Market data feeds (scheduled)
  - Alert triggers (price levels, news events)

Process:
  1. Receive input/trigger
  2. Route to relevant agents
  3. Collect agent responses
  4. Apply weighted consensus
  5. Risk Manager veto check
  6. Execute or reject
  7. Log decision + reasoning

Output:
  - Trading decision (BUY/SELL/HOLD)
  - Position size
  - Entry/Exit levels
  - Confidence score
  - Full reasoning chain
```

### 2. KNOWLEDGE ADVISOR (RAG)
**Role:** Domain expertise from trading literature
**Tech:** ChromaDB + sentence-transformers + MMR retrieval
**Data:** 140K chunks (280 books), expanding to 1,600+ documents

### 3. ML SIGNAL ENGINE
**Role:** Quantitative predictions from market data
**Tech:** CatBoost (76.47% accuracy) + feature engineering
**Data:** 200+ symbols, multi-timeframe

### 4. RISK MANAGER
**Role:** Position sizing, risk limits, portfolio protection
**Tech:** Kelly Criterion + portfolio optimization

### 5. SECURITY GUARD
**Role:** Cybersecurity, anomaly detection, system integrity

### 6. CONTINUOUS LEARNING AGENT
**Role:** Self-improvement from own trading history

### 7. NEWS ANALYZER
**Role:** Sentiment and event detection from news feeds

---

## Decision Flow

```
Market Data / User Query
         │
         ▼
    ┌─────────┐
    │  BRAIN   │ ◄── Security Guard (validates input)
    └────┬────┘
         │
    ┌────┴────────────────────┐
    │    Fan-out to Agents    │
    ├─────────┬───────┬───────┤
    ▼         ▼       ▼       ▼
 Knowledge  ML     Risk    News
 Advisor   Signal  Check   Analyzer
    │         │       │       │
    └────┬────┘       │       │
         │            │       │
    ┌────┴────────────┴───────┘
    │   Weighted Consensus     │
    └─────────┬───────────────┘
              │
    ┌─────────▼───────────┐
    │   Risk Manager Veto  │
    └─────────┬───────────┘
              │
         ┌────┴────┐
         │APPROVED?│
         └──┬───┬──┘
          Yes   No
           │     │
           ▼     ▼
       Execute  Log &
       Order    Alert
           │
           ▼
    Activity Logger
    (Full Audit Trail)
           │
           ▼
    Continuous Learner
    (Track Outcome)
```

---

## Consensus Engine

```
Default Weights:
  ML Signal Engine:     0.35  (highest - quantitative backbone)
  Knowledge Advisor:    0.25  (literature-backed reasoning)
  Risk Manager:         0.20  (risk assessment)
  News Analyzer:        0.15  (sentiment context)
  Security Guard:       0.05  (anomaly flag)

Consensus = Σ(agent_confidence × agent_weight × direction)
Threshold = 0.60 (configurable)

If consensus >= threshold AND risk_manager.approved → EXECUTE
Else → HOLD + log reasoning
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Model | CatBoost + scikit-learn |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Dashboard | Streamlit |
| Data | yfinance, ccxt, pandas |
| GPU | PyTorch + CUDA |
| Local LLM | Ollama (qwen3:8b, DeepSeek) |
| OCR | Tesseract + pdf2image |
| Execution | ccxt (multi-exchange), Alpaca API |
| Logging | Python logging + JSONL audit |
| Testing | pytest |

---

## Hardware Optimization Profile

```python
HARDWARE = {
    "cpu": "Intel i7-9700F",
    "cpu_cores": 8,
    "gpu": "NVIDIA GTX 1070",
    "gpu_vram_gb": 8,
    "ram_gb": 64,
    "ram_type": "DDR4-2666",
    "optimization": {
        "batch_size_default": 32,
        "num_workers": 6,
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "chunk_size_mb": 512,
        "max_gpu_memory_fraction": 0.85,
    }
}
```

---

## Local Data Locations (Reference)

| Data | Location |
|------|----------|
| Trading Documents (1,600+) | `J:\E-Books\Trading Database` |
| Multimodal Dataset | `C:\_AI\Market_Hawk\_mm_dataset` |
| Image Kit | `C:\_AI\Market_Hawk\_mm_image_kit` |
| Pair Dataset | `C:\_AI\Market_Hawk\_mm_pair_dataset` |
| Datasets | `C:\_AI\Market_Hawk\datasets` |
| Market Hawk v2 | `C:\_AI\Market_Hawk_2` |
| Additional Work | `G:\` (various) |
| THIS REPO | `K:\_DEV_MVP_2026\Market_Hawk_3` |

---

## Security Model

```
Layer 1: API Key Management — Encrypted .env + keyring
Layer 2: Connection Security — TLS, rate limiting, circuit breakers
Layer 3: Trading Anomaly Detection — Volume, manipulation, slippage
Layer 4: System Integrity — Audit log hash chain, decision replay
```
