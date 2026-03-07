"""
SCRIPT NAME: settings.py
====================================
Execution Location: K:\\_DEV_MVP_2026\\Market_Hawk_3\\config\\
Purpose: Global configuration for Market Hawk MVP
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Last Modified: 2026-03-01
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# LOCAL DATA PATHS (read from .env, fallback to defaults)
# ============================================================

LOCAL_DATA_PATHS = {
    "trading_documents": os.environ.get(
        "MH_PATH_TRADING_DOCS", r"J:\E-Books\Trading Database"
    ),
    "mm_dataset": os.environ.get(
        "MH_PATH_MM_DATASET", r"C:\_AI\Market_Hawk\_mm_dataset"
    ),
    "mm_image_kit": os.environ.get(
        "MH_PATH_MM_IMAGE_KIT", r"C:\_AI\Market_Hawk\_mm_image_kit"
    ),
    "mm_pair_dataset": os.environ.get(
        "MH_PATH_MM_PAIR_DATASET", r"C:\_AI\Market_Hawk\_mm_pair_dataset"
    ),
    "datasets": os.environ.get(
        "MH_PATH_DATASETS", r"C:\_AI\Market_Hawk\datasets"
    ),
    "market_hawk_v2": os.environ.get(
        "MH_PATH_MARKET_HAWK_V2", r"C:\_AI\Market_Hawk_2"
    ),
}

# ============================================================
# EXISTING CHROMADB (from previous work)
# ============================================================

EXISTING_CHROMADB_PATH = os.environ.get(
    "MH_CHROMADB_PATH",
    r"K:\_DEV_MVP_2026\Agent_Trading_AI\AgentTradingAI\baza_date_vectoriala_v2",
)


# ============================================================
# HARDWARE PROFILE
# ============================================================

@dataclass
class HardwareProfile:
    """Hardware optimization settings for i7-9700F + GTX 1070 + 64GB DDR4."""
    cpu_name: str = "Intel i7-9700F"
    cpu_cores: int = 8
    gpu_name: str = "NVIDIA GTX 1070"
    gpu_vram_gb: int = 8
    ram_gb: int = 64
    ram_type: str = "DDR4-2666"
    # Derived optimization settings
    num_workers: int = 6          # cpu_cores - 2
    batch_size: int = 32          # Safe default for 8GB VRAM
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    chunk_size_mb: int = 512
    max_gpu_memory_fraction: float = 0.85
    prefetch_factor: int = 2


HARDWARE = HardwareProfile()


# ============================================================
# AGENT CONFIGURATION
# ============================================================

@dataclass
class AgentConfig:
    """Configuration for each agent in the multi-agent system."""
    name: str
    weight: float
    enabled: bool = True
    timeout_seconds: int = 30


AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "ml_signal_engine": AgentConfig(
        name="ML Signal Engine",
        weight=0.35,
    ),
    "knowledge_advisor": AgentConfig(
        name="Knowledge Advisor",
        weight=0.25,
    ),
    "risk_manager": AgentConfig(
        name="Risk Manager",
        weight=0.20,
    ),
    "news_analyzer": AgentConfig(
        name="News Analyzer",
        weight=0.15,
    ),
    "security_guard": AgentConfig(
        name="Security Guard",
        weight=0.05,
    ),
}

CONSENSUS_THRESHOLD = 0.60  # Minimum weighted consensus to execute


# ============================================================
# KNOWLEDGE ADVISOR CONFIG
# ============================================================

@dataclass
class RAGConfig:
    """Configuration for the Knowledge Advisor RAG system."""
    # Points to EXISTING ChromaDB v2 (130K+ chunks, 1GB)
    chromadb_path: str = EXISTING_CHROMADB_PATH
    collection_name: str = "algo_trading"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    n_results: int = 15
    fetch_k: int = 60
    mmr_lambda: float = 0.7       # 0.7 = balance relevance vs diversity
    retrieval_type: str = "mmr"   # Maximum Marginal Relevance
    # LLM for response generation
    llm_model: str = "qwen3:8b"
    ollama_host: str = "http://localhost:11434"


RAG_CONFIG = RAGConfig()


# ============================================================
# ML MODEL CONFIG
# ============================================================

@dataclass
class MLConfig:
    """Configuration for the ML Signal Engine."""
    model_path: str = os.environ.get(
        "MH_ML_MODEL_PATH",
        r"G:\......................AI_Models\catboost_v2_advanced_20250724_011145.cbm",
    )
    models_base_dir: str = os.environ.get(
        "MH_ML_MODELS_DIR",
        r"G:\......................AI_Models",
    )
    confidence_threshold: float = 0.65
    features_version: str = "v2"
    supported_timeframes: list = field(default_factory=lambda: ["1h", "4h", "1d"])


ML_CONFIG = MLConfig()


# ============================================================
# RISK MANAGEMENT CONFIG
# ============================================================

@dataclass
class RiskConfig:
    """Configuration for the Risk Manager."""
    max_position_pct: float = 0.02      # Max 2% of portfolio per trade
    max_portfolio_risk: float = 0.06    # Max 6% total portfolio risk
    max_drawdown_pct: float = 0.15      # 15% max drawdown trigger
    kelly_fraction: float = 0.25        # Quarter-Kelly for safety
    max_correlated_positions: int = 3
    default_stop_loss_pct: float = 0.02
    default_take_profit_pct: float = 0.04


RISK_CONFIG = RiskConfig()


# ============================================================
# LOGGING CONFIG
# ============================================================

LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = LOGS_DIR / "market_hawk.log"
DECISION_LOG = LOGS_DIR / "decisions.jsonl"


# ============================================================
# API KEYS (from .env)
# ============================================================

def get_api_key(key_name: str) -> Optional[str]:
    """Safely retrieve API key from environment."""
    return os.environ.get(key_name)
