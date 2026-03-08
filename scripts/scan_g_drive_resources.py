"""
SCRIPT NAME: scan_g_drive_resources.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Scan and catalog AI resources on G: drive for Market Hawk 3 integration.
Hardware Optimization: Intel i7-9700F, 64GB DDR4 — I/O bound, no GPU needed.
Creation Date: 2026-03-08

Scans directories on G: drive, catalogs all resources (models, datasets, code,
books), and generates structured reports for integration planning.

Output:
    - docs/G_DRIVE_RESOURCE_CATALOG.json  — machine-readable catalog
    - docs/G_DRIVE_INTEGRATION_PLAN.md    — human-readable integration plan

Usage:
    # Dry run (list directories without scanning)
    python scripts/scan_g_drive_resources.py --dry-run

    # Full scan
    python scripts/scan_g_drive_resources.py

    # Skip hash calculation (faster)
    python scripts/scan_g_drive_resources.py --skip-hash

    # Resume interrupted scan
    python scripts/scan_g_drive_resources.py --resume
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import LOGS_DIR, setup_logging

logger = logging.getLogger("market_hawk.g_drive_scan")

# ============================================================
# DIRECTORIES TO SCAN
# ============================================================

SCAN_DIRECTORIES: List[str] = [
    r"G:\......................AI_Models",
    r"G:\AI_Models",
    r"G:\..........Market_Hawk_AI_Agent_Live_Trading",
    r"G:\.....DATABASE\dataset",
    r"G:\...Altele\Technical_Analysis_Charts",
    r"G:\_kit_multimodal",
    r"G:\_kit_multimodal_minimal",
    r"G:\ai_data_consolidator",
    r"G:\K\AI_Trading_Processed",
    r"G:\Kit Multimodal",
    r"G:\KIT_TRAINING_BOOKS",
    r"G:\KIT_TRAINING_REPORTS",
    r"G:\L40_CHECKPOINTS",
    r"G:\L40_GPU_SUPERKIT",
    r"G:\MASTER_AI_TRADING_PACKAGE",
    r"G:\Modele AI curate",
    r"G:\ULTRA_KIT_ANTENAMENT_AI",
    r"G:\transfer_runpod",
    r"G:\..............JSON",
    r"G:\.................CSV",
    r"G:\.............METADATA",
    r"G:\.........PARQUET",
    r"G:\trading_db",
]

# ============================================================
# FILE TYPE CATEGORIES
# ============================================================

FILE_CATEGORIES = {
    "models": {".pt", ".pth", ".pkl", ".cbm", ".onnx", ".safetensors",
               ".h5", ".keras", ".joblib", ".bin", ".model"},
    "data": {".csv", ".parquet", ".json", ".feather", ".arrow",
             ".tsv", ".xlsx", ".xls", ".hdf5", ".h5"},
    "code": {".py", ".ipynb", ".r", ".jl"},
    "books": {".pdf", ".epub", ".djvu", ".mobi"},
    "images": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".tiff"},
    "configs": {".yaml", ".yml", ".toml", ".ini", ".cfg"},
    "text": {".txt", ".md", ".rst", ".log"},
}

# Reverse lookup: extension -> category
_EXT_TO_CATEGORY: Dict[str, str] = {}
for _cat, _exts in FILE_CATEGORIES.items():
    for _ext in _exts:
        _EXT_TO_CATEGORY[_ext] = _cat

# ============================================================
# OUTPUT PATHS
# ============================================================

CATALOG_PATH = _PROJECT_ROOT / "docs" / "G_DRIVE_RESOURCE_CATALOG.json"
PLAN_PATH = _PROJECT_ROOT / "docs" / "G_DRIVE_INTEGRATION_PLAN.md"
CHECKPOINT_PATH = _PROJECT_ROOT / "logs" / "g_drive_scan_checkpoint.json"
LOG_PATH = LOGS_DIR / "g_drive_scan.log"

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FileInfo:
    """Metadata for a single file."""
    path: str
    name: str
    extension: str
    category: str
    size_bytes: int
    modified: str  # ISO format
    sha256: Optional[str] = None
    # Model-specific
    model_type: Optional[str] = None
    # Data-specific
    columns: Optional[List[str]] = None
    row_count: Optional[int] = None
    sample_rows: Optional[List[Dict]] = None


@dataclass
class DirectoryScan:
    """Scan results for a single directory."""
    path: str
    exists: bool
    total_files: int = 0
    total_size_bytes: int = 0
    file_type_counts: Dict[str, int] = field(default_factory=dict)
    file_type_sizes: Dict[str, int] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    category_sizes: Dict[str, int] = field(default_factory=dict)
    top_files: List[Dict] = field(default_factory=list)  # Top 20 by size
    tree: Dict[str, Any] = field(default_factory=dict)  # 2-level tree
    files: List[Dict] = field(default_factory=list)  # All file metadata
    scan_time_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class ScanCatalog:
    """Complete scan results."""
    generated: str
    scan_dirs_total: int
    scan_dirs_found: int
    total_files: int
    total_size_bytes: int
    total_size_human: str
    category_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    directories: List[Dict] = field(default_factory=list)
    duplicates: List[Dict] = field(default_factory=list)
    scan_time_sec: float = 0.0


# ============================================================
# SCANNER
# ============================================================

class GDriveScanner:
    """Scan G: drive directories and catalog resources.

    Features:
        - Checkpoint/resume for interrupted scans
        - Progress bar per directory (tqdm)
        - Memory-efficient: samples CSV/Parquet without full load
        - Duplicate detection via SHA-256 hash
        - 2-level directory tree structure

    Args:
        directories: List of directory paths to scan.
        max_sample_rows: Max rows to sample from data files.
        skip_hash: Skip SHA-256 calculation (faster).
        checkpoint_path: Path for checkpoint file.
    """

    def __init__(self,
                 directories: List[str],
                 max_sample_rows: int = 5,
                 skip_hash: bool = False,
                 checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        self.directories = directories
        self.max_sample_rows = max_sample_rows
        self.skip_hash = skip_hash
        self.checkpoint_path = checkpoint_path
        self._hash_map: Dict[str, List[str]] = defaultdict(list)
        self._completed_dirs: Set[str] = set()
        self._results: List[DirectoryScan] = []

    # ---------- CHECKPOINT ----------

    def save_checkpoint(self) -> None:
        """Save scan progress for resume."""
        data = {
            "completed_dirs": list(self._completed_dirs),
            "results": [asdict(r) for r in self._results],
            "hash_map": dict(self._hash_map),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Checkpoint saved: %d dirs complete", len(self._completed_dirs))

    def load_checkpoint(self) -> bool:
        """Load previous checkpoint. Returns True if loaded."""
        if not self.checkpoint_path.exists():
            return False
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self._completed_dirs = set(data.get("completed_dirs", []))
            self._hash_map = defaultdict(list, data.get("hash_map", {}))
            # Reconstruct DirectoryScan objects
            for rd in data.get("results", []):
                ds = DirectoryScan(
                    path=rd["path"], exists=rd["exists"],
                    total_files=rd.get("total_files", 0),
                    total_size_bytes=rd.get("total_size_bytes", 0),
                    file_type_counts=rd.get("file_type_counts", {}),
                    file_type_sizes=rd.get("file_type_sizes", {}),
                    category_counts=rd.get("category_counts", {}),
                    category_sizes=rd.get("category_sizes", {}),
                    top_files=rd.get("top_files", []),
                    tree=rd.get("tree", {}),
                    files=rd.get("files", []),
                    scan_time_sec=rd.get("scan_time_sec", 0),
                    errors=rd.get("errors", []),
                )
                self._results.append(ds)
            logger.info("Checkpoint loaded: %d dirs already scanned",
                        len(self._completed_dirs))
            return True
        except Exception:
            logger.exception("Failed to load checkpoint, starting fresh")
            return False

    # ---------- SCANNING ----------

    def scan_all(self, resume: bool = False) -> ScanCatalog:
        """Scan all directories. Returns complete catalog.

        Args:
            resume: If True, load checkpoint and skip already-scanned dirs.
        """
        if resume:
            self.load_checkpoint()

        try:
            from tqdm import tqdm
            dir_iter = tqdm(self.directories, desc="Directories", unit="dir")
        except ImportError:
            dir_iter = self.directories

        t0 = time.time()

        for dir_path in dir_iter:
            if dir_path in self._completed_dirs:
                logger.info("Skipping (already scanned): %s", dir_path)
                continue

            result = self._scan_directory(dir_path)
            self._results.append(result)
            self._completed_dirs.add(dir_path)

            # Checkpoint after each directory
            self.save_checkpoint()
            gc.collect()

        elapsed = time.time() - t0

        # Build catalog
        catalog = self._build_catalog(elapsed)

        # Find duplicates
        if not self.skip_hash:
            catalog.duplicates = self._find_duplicates()

        return catalog

    def _scan_directory(self, dir_path: str) -> DirectoryScan:
        """Scan a single directory recursively."""
        p = Path(dir_path)
        result = DirectoryScan(path=dir_path, exists=p.exists())

        if not p.exists():
            logger.warning("Directory does not exist: %s", dir_path)
            return result

        logger.info("Scanning: %s", dir_path)
        t0 = time.time()

        file_type_counts: Dict[str, int] = defaultdict(int)
        file_type_sizes: Dict[str, int] = defaultdict(int)
        category_counts: Dict[str, int] = defaultdict(int)
        category_sizes: Dict[str, int] = defaultdict(int)
        all_files: List[Tuple[int, str, str]] = []  # (size, path, ext)
        file_metadata: List[Dict] = []

        # Walk directory — chunk-friendly for 100k+ files
        file_count = 0
        batch: List[Dict] = []
        batch_size = 1000

        try:
            for root, dirs, files in os.walk(p):
                for fname in files:
                    file_count += 1
                    fpath = Path(root) / fname
                    ext = fpath.suffix.lower()
                    category = _EXT_TO_CATEGORY.get(ext, "other")

                    try:
                        stat = fpath.stat()
                        size = stat.st_size
                        mtime = datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat()
                    except (OSError, PermissionError) as e:
                        result.errors.append(f"{fpath}: {e}")
                        continue

                    file_type_counts[ext] += 1
                    file_type_sizes[ext] += size
                    category_counts[category] += 1
                    category_sizes[category] += size
                    result.total_size_bytes += size

                    all_files.append((size, str(fpath), ext))

                    # Build file info for catalog
                    info: Dict[str, Any] = {
                        "path": str(fpath),
                        "name": fname,
                        "extension": ext,
                        "category": category,
                        "size_bytes": size,
                        "modified": mtime,
                    }

                    # Hash (for non-huge files, <500MB)
                    if not self.skip_hash and size < 500 * 1024 * 1024:
                        try:
                            h = self._file_hash(fpath)
                            info["sha256"] = h
                            self._hash_map[h].append(str(fpath))
                        except (OSError, PermissionError):
                            pass

                    # Model metadata
                    if category == "models" and size < 2 * 1024 * 1024 * 1024:
                        meta = self._extract_model_metadata(fpath, ext)
                        if meta:
                            info["model_type"] = meta

                    # Data file sampling
                    if ext in (".csv", ".parquet") and size > 0:
                        sample = self._sample_data_file(fpath, ext)
                        if sample:
                            info.update(sample)

                    batch.append(info)

                    # Flush batch to avoid memory bloat
                    if len(batch) >= batch_size:
                        file_metadata.extend(batch)
                        batch.clear()
                        gc.collect()

        except PermissionError as e:
            result.errors.append(f"Permission denied: {dir_path}: {e}")
            logger.warning("Permission denied scanning %s", dir_path)
        except Exception:
            logger.exception("Error scanning %s", dir_path)

        # Flush remaining batch
        file_metadata.extend(batch)
        batch.clear()

        result.total_files = file_count
        result.file_type_counts = dict(file_type_counts)
        result.file_type_sizes = dict(file_type_sizes)
        result.category_counts = dict(category_counts)
        result.category_sizes = dict(category_sizes)

        # Top 20 largest files
        all_files.sort(key=lambda x: x[0], reverse=True)
        result.top_files = [
            {"path": fp, "size_bytes": sz, "extension": ext,
             "size_human": self._human_size(sz)}
            for sz, fp, ext in all_files[:20]
        ]

        # 2-level directory tree
        result.tree = self._build_tree(p, max_depth=2)

        # Store file metadata
        result.files = file_metadata

        result.scan_time_sec = time.time() - t0
        logger.info("Scanned %s: %d files, %s, %.1fs",
                     dir_path, file_count,
                     self._human_size(result.total_size_bytes),
                     result.scan_time_sec)

        return result

    # ---------- METADATA EXTRACTION ----------

    @staticmethod
    def _extract_model_metadata(fpath: Path, ext: str) -> Optional[str]:
        """Extract model type info without loading the full model."""
        try:
            if ext == ".cbm":
                return "CatBoost"
            if ext == ".onnx":
                return "ONNX"
            if ext == ".safetensors":
                return "SafeTensors"
            if ext in (".pt", ".pth"):
                return "PyTorch"
            if ext == ".h5":
                return "HDF5/Keras"
            if ext in (".pkl", ".joblib"):
                # Peek at pickle header to detect type (fara deserializare)
                with open(fpath, "rb") as f:
                    header = f.read(256)
                # Heuristic: search for class names in pickle stream
                header_str = header.decode("latin-1", errors="ignore")
                if "catboost" in header_str.lower():
                    return "CatBoost (pickle)"
                if "xgboost" in header_str.lower() or "XGBClassifier" in header_str:
                    return "XGBoost (pickle)"
                if "lightgbm" in header_str.lower():
                    return "LightGBM (pickle)"
                if "sklearn" in header_str.lower():
                    return "sklearn (pickle)"
                if "scaler" in header_str.lower():
                    return "Scaler (pickle)"
                return "pickle (unknown)"
            return None
        except Exception:
            return None

    def _sample_data_file(self, fpath: Path, ext: str) -> Optional[Dict]:
        """Sample first N rows from CSV/Parquet without loading entire file."""
        try:
            import pandas as pd

            if ext == ".csv":
                # nrows limiteaza citirea
                df = pd.read_csv(fpath, nrows=self.max_sample_rows + 1,
                                 encoding="utf-8", on_bad_lines="skip")
                # Estimate total rows from file size + avg row size
                if len(df) > 0:
                    avg_row_bytes = fpath.stat().st_size / max(1, len(df))
                    estimated_rows = int(fpath.stat().st_size / max(1, avg_row_bytes))
                else:
                    estimated_rows = 0

                return {
                    "columns": list(df.columns),
                    "row_count": estimated_rows,
                    "sample_rows": df.head(self.max_sample_rows).to_dict(
                        orient="records"
                    ),
                }

            if ext == ".parquet":
                # Parquet stores row count in metadata — no full scan
                pf = pd.read_parquet(fpath, engine="pyarrow")
                row_count = len(pf)
                sample = pf.head(self.max_sample_rows)

                result = {
                    "columns": list(pf.columns),
                    "row_count": row_count,
                    "sample_rows": sample.to_dict(orient="records"),
                }
                del pf
                gc.collect()
                return result

        except Exception as e:
            logger.debug("Could not sample %s: %s", fpath, e)
        return None

    # ---------- HASHING ----------

    @staticmethod
    def _file_hash(fpath: Path) -> str:
        """Compute SHA-256 hash reading in 64KB chunks."""
        sha256 = hashlib.sha256()
        with open(fpath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    def _find_duplicates(self) -> List[Dict]:
        """Find files with identical SHA-256 hashes."""
        duplicates = []
        for h, paths in self._hash_map.items():
            if len(paths) > 1:
                duplicates.append({
                    "sha256": h,
                    "count": len(paths),
                    "paths": paths,
                })
        duplicates.sort(key=lambda x: x["count"], reverse=True)
        logger.info("Found %d duplicate groups", len(duplicates))
        return duplicates

    # ---------- TREE BUILDING ----------

    @staticmethod
    def _build_tree(root: Path, max_depth: int = 2) -> Dict[str, Any]:
        """Build directory tree structure to max_depth levels."""
        tree: Dict[str, Any] = {}
        if not root.exists():
            return tree

        try:
            for item in sorted(root.iterdir()):
                if item.is_dir():
                    subtree: Dict[str, Any] = {"_type": "dir"}
                    if max_depth > 1:
                        try:
                            children = []
                            for child in sorted(item.iterdir()):
                                children.append({
                                    "name": child.name,
                                    "is_dir": child.is_dir(),
                                })
                                if len(children) >= 50:
                                    children.append({"name": "...", "is_dir": False})
                                    break
                            subtree["children"] = children
                        except PermissionError:
                            subtree["children"] = [{"name": "(permission denied)", "is_dir": False}]
                    tree[item.name] = subtree
                else:
                    try:
                        tree[item.name] = {
                            "_type": "file",
                            "size": item.stat().st_size,
                        }
                    except OSError:
                        tree[item.name] = {"_type": "file", "size": 0}
        except PermissionError:
            tree["(permission denied)"] = {}

        return tree

    # ---------- CATALOG BUILDING ----------

    def _build_catalog(self, elapsed: float) -> ScanCatalog:
        """Aggregate all directory scans into a catalog."""
        total_files = 0
        total_size = 0
        category_agg: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"count": 0, "size_bytes": 0}
        )

        for ds in self._results:
            total_files += ds.total_files
            total_size += ds.total_size_bytes
            for cat, count in ds.category_counts.items():
                category_agg[cat]["count"] += count
            for cat, size in ds.category_sizes.items():
                category_agg[cat]["size_bytes"] += size

        # Add human-readable sizes
        for cat_info in category_agg.values():
            cat_info["size_human"] = self._human_size(cat_info["size_bytes"])

        found = sum(1 for ds in self._results if ds.exists)

        return ScanCatalog(
            generated=datetime.now(timezone.utc).isoformat(),
            scan_dirs_total=len(self.directories),
            scan_dirs_found=found,
            total_files=total_files,
            total_size_bytes=total_size,
            total_size_human=self._human_size(total_size),
            category_summary=dict(category_agg),
            directories=[asdict(ds) for ds in self._results],
            scan_time_sec=elapsed,
        )

    # ---------- UTILITIES ----------

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable string."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"


# ============================================================
# INTEGRATION PLAN GENERATOR
# ============================================================

# Mapping: resource pattern -> Market Hawk module
_INTEGRATION_MAP = [
    # Models
    ("catboost", "agents/ml_signal_engine — primary model"),
    ("xgboost", "agents/ml_signal_engine — ensemble member"),
    ("lightgbm", "agents/ml_signal_engine — ensemble member"),
    ("lstm", "agents/ml_signal_engine — sequential model"),
    ("transformer", "agents/ml_signal_engine — attention model"),
    ("finbert", "agents/news_analyzer — sentiment model"),
    ("gnn", "agents/ml_signal_engine — graph model"),
    ("vae", "agents/security_guard — anomaly detection"),
    ("rl", "brain/orchestrator — reinforcement learning"),
    ("scaler", "agents/ml_signal_engine — feature preprocessing"),
    ("onnx", "agents/ml_signal_engine — optimized inference"),
    # Data
    ("ohlcv", "backtesting/data_loader — price data"),
    ("forex", "backtesting/data_loader — forex pairs"),
    ("crypto", "backtesting/data_loader — crypto pairs"),
    ("stock", "backtesting/data_loader — equities"),
    ("sentiment", "agents/news_analyzer — sentiment data"),
    ("feature", "data/market_data_fetcher — engineered features"),
    ("indicator", "data/market_data_fetcher — TA indicators"),
    ("order_book", "data/market_data_fetcher — order flow"),
    # Books/Reports
    ("trading", "agents/knowledge_advisor — RAG knowledge base"),
    ("technical_analysis", "agents/knowledge_advisor — TA reference"),
    ("risk_management", "agents/knowledge_advisor — risk reference"),
    # Code
    (".py", "reference code — audit for reusable components"),
    (".ipynb", "reference notebooks — audit for training pipelines"),
]

# Effort estimation heuristic
_EFFORT_MAP = {
    "models": "LOW (load + validate + register in MODEL_REGISTRY)",
    "data": "MEDIUM (schema mapping + validation + data_loader integration)",
    "code": "HIGH (audit + refactor + test coverage)",
    "books": "MEDIUM (chunk + embed + add to RAG knowledge base)",
    "images": "LOW (chart analysis training data)",
    "configs": "LOW (review + merge)",
}


def generate_integration_plan(catalog: ScanCatalog,
                              output_path: Path) -> None:
    """Generate human-readable integration plan from scan catalog."""
    lines = [
        "# G: Drive Resource Integration Plan",
        "",
        f"**Generated**: {catalog.generated}",
        f"**Directories scanned**: {catalog.scan_dirs_found}/{catalog.scan_dirs_total}",
        f"**Total files**: {catalog.total_files:,}",
        f"**Total size**: {catalog.total_size_human}",
        f"**Scan time**: {catalog.scan_time_sec:.1f}s",
        "",
        "## Category Summary",
        "",
        "| Category | Files | Size | Effort |",
        "|----------|-------|------|--------|",
    ]

    for cat in ("models", "data", "code", "books", "images", "configs", "text", "other"):
        info = catalog.category_summary.get(cat, {})
        count = info.get("count", 0)
        size_h = info.get("size_human", "0 B")
        effort = _EFFORT_MAP.get(cat, "VARIES")
        if count > 0:
            lines.append(f"| {cat} | {count:,} | {size_h} | {effort} |")

    # Per-directory inventory
    lines.extend(["", "## Directory Inventory", ""])

    for ds_dict in catalog.directories:
        ds_path = ds_dict["path"]
        exists = ds_dict["exists"]
        if not exists:
            lines.extend([f"### {ds_path}", "", "**NOT FOUND**", ""])
            continue

        total = ds_dict["total_files"]
        size_h = GDriveScanner._human_size(ds_dict["total_size_bytes"])
        lines.extend([
            f"### {ds_path}",
            "",
            f"- **Files**: {total:,}",
            f"- **Size**: {size_h}",
            f"- **Scan time**: {ds_dict.get('scan_time_sec', 0):.1f}s",
        ])

        # Category breakdown
        cat_counts = ds_dict.get("category_counts", {})
        if cat_counts:
            breakdown = ", ".join(
                f"{cat}: {cnt}" for cat, cnt in
                sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
            )
            lines.append(f"- **Breakdown**: {breakdown}")

        # Top 5 largest files
        top = ds_dict.get("top_files", [])[:5]
        if top:
            lines.extend(["", "**Largest files:**", ""])
            for tf in top:
                name = Path(tf["path"]).name
                lines.append(f"  - `{name}` ({tf['size_human']})")

        # Errors
        errors = ds_dict.get("errors", [])
        if errors:
            lines.extend(["", f"**Errors**: {len(errors)} files could not be read"])

        lines.append("")

    # Integration recommendations
    lines.extend([
        "## Integration Recommendations",
        "",
        "### Priority 1: Direct Integration (LOW effort)",
        "",
    ])

    model_files = []
    data_files = []
    book_files = []

    for ds_dict in catalog.directories:
        for fi in ds_dict.get("files", []):
            cat = fi.get("category", "other")
            if cat == "models":
                model_files.append(fi)
            elif cat == "data":
                data_files.append(fi)
            elif cat == "books":
                book_files.append(fi)

    # Model recommendations
    if model_files:
        lines.append("**ML Models** (register in MODEL_REGISTRY):")
        lines.append("")
        lines.append("| File | Size | Type | Target Module |")
        lines.append("|------|------|------|---------------|")
        # Show top 20 by size
        model_files.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)
        for mf in model_files[:20]:
            name = Path(mf["path"]).name
            size_h = GDriveScanner._human_size(mf.get("size_bytes", 0))
            mtype = mf.get("model_type", "unknown")
            # Find matching integration target
            target = "agents/ml_signal_engine"
            name_lower = name.lower()
            for pattern, mod in _INTEGRATION_MAP:
                if pattern in name_lower:
                    target = mod
                    break
            lines.append(f"| `{name}` | {size_h} | {mtype} | {target} |")
        lines.append("")

    # Data recommendations
    if data_files:
        lines.extend([
            "### Priority 2: Data Integration (MEDIUM effort)",
            "",
            "**Datasets** (integrate via backtesting/data_loader):",
            "",
        ])
        data_files.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)
        for df_info in data_files[:15]:
            name = Path(df_info["path"]).name
            size_h = GDriveScanner._human_size(df_info.get("size_bytes", 0))
            cols = df_info.get("columns")
            col_str = f" — cols: {', '.join(cols[:5])}..." if cols else ""
            rows = df_info.get("row_count")
            row_str = f" ({rows:,} rows)" if rows else ""
            lines.append(f"  - `{name}` ({size_h}{row_str}){col_str}")
        lines.append("")

    # Book recommendations
    if book_files:
        lines.extend([
            "### Priority 3: Knowledge Base (MEDIUM effort)",
            "",
            f"**Books/Reports**: {len(book_files)} files for RAG knowledge base",
            f"  - Target: agents/knowledge_advisor (ChromaDB)",
            f"  - Pipeline: chunk -> embed (nomic-embed-text) -> index",
            "",
        ])

    # Duplicates
    if catalog.duplicates:
        lines.extend([
            "## Duplicate Files",
            "",
            f"Found **{len(catalog.duplicates)}** groups of identical files:",
            "",
        ])
        for dup in catalog.duplicates[:10]:
            lines.append(f"- **{dup['count']} copies** (SHA-256: {dup['sha256'][:12]}...):")
            for dp in dup["paths"][:5]:
                lines.append(f"  - `{dp}`")
            if len(dup["paths"]) > 5:
                lines.append(f"  - ... and {len(dup['paths']) - 5} more")
        lines.append("")

    lines.extend([
        "---",
        "*Generated by scripts/scan_g_drive_resources.py*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Integration plan saved: %s", output_path)


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan G: drive resources for Market Hawk 3 integration",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List directories without scanning",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted scan from checkpoint",
    )
    parser.add_argument(
        "--skip-hash", action="store_true",
        help="Skip SHA-256 hash calculation (faster)",
    )
    parser.add_argument(
        "--max-sample-rows", type=int, default=5,
        help="Max rows to sample from CSV/Parquet (default: 5)",
    )
    return parser.parse_args()


def dry_run_report() -> None:
    """Print scan plan without executing."""
    print("\n" + "=" * 60)
    print("DRY RUN -- G: Drive Resource Scan Plan")
    print("=" * 60)
    print(f"\n  Directories to scan: {len(SCAN_DIRECTORIES)}")
    print()
    for d in SCAN_DIRECTORIES:
        exists = Path(d).exists()
        status = "EXISTS" if exists else "NOT FOUND"
        print(f"    [{status:>9}] {d}")
    found = sum(1 for d in SCAN_DIRECTORIES if Path(d).exists())
    print(f"\n  Found: {found}/{len(SCAN_DIRECTORIES)}")
    print(f"  Output: {CATALOG_PATH}")
    print(f"           {PLAN_PATH}")
    print(f"  Log:    {LOG_PATH}")
    print("\n" + "=" * 60)
    print("Remove --dry-run to execute scan.")
    print("=" * 60 + "\n")


def main() -> None:
    setup_logging(logging.INFO)

    # Dedicated file handler for scan log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s"
    ))
    logging.getLogger("market_hawk.g_drive_scan").addHandler(fh)

    args = parse_args()

    if args.dry_run:
        dry_run_report()
        return

    logger.info("=" * 60)
    logger.info("G: DRIVE RESOURCE SCAN START")
    logger.info("=" * 60)

    scanner = GDriveScanner(
        directories=SCAN_DIRECTORIES,
        max_sample_rows=args.max_sample_rows,
        skip_hash=args.skip_hash,
    )

    catalog = scanner.scan_all(resume=args.resume)

    # Save JSON catalog
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog_dict = asdict(catalog)
    CATALOG_PATH.write_text(
        json.dumps(catalog_dict, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Catalog saved: %s (%.1f MB)",
                CATALOG_PATH, CATALOG_PATH.stat().st_size / 1e6)

    # Generate integration plan
    generate_integration_plan(catalog, PLAN_PATH)

    logger.info("=" * 60)
    logger.info("SCAN COMPLETE: %d files, %s, %.1fs",
                catalog.total_files, catalog.total_size_human,
                catalog.scan_time_sec)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
