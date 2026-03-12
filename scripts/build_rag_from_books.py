"""
SCRIPT NAME: build_rag_from_books.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Expand RAG Knowledge Base from 671+ trading books into ChromaDB.
Hardware Optimization: Intel i7-9700F, 64GB DDR4
Creation Date: 2026-03-08

5-step pipeline:
    Step 1: Scan & Catalog — find all books across J:\\E-Books, G:\\KIT_TRAINING_BOOKS
    Step 2: Text Extraction — pdfplumber (primary) + PyPDF2 (fallback)
    Step 3: Intelligent Chunking — chapter/section detection, 500-1000 tokens
    Step 4: Embedding & Indexing — nomic-embed-text via Ollama -> ChromaDB
    Step 5: Verification — query test + stats

Connects to EXISTING ChromaDB v2 at:
    K:\\_DEV_MVP_2026\\Agent_Trading_AI\\AgentTradingAI\\baza_date_vectoriala_v2\\
    Collection: "algo_trading" — 130K+ chunks from 270+ books
    Embeddings: nomic-embed-text (via Ollama)

DEPENDENCIES:
    pip install chromadb langchain-chroma langchain-ollama pdfplumber PyPDF2 tqdm

Usage:
    # Run all steps
    python scripts/build_rag_from_books.py

    # Single step
    python scripts/build_rag_from_books.py --step 1
    python scripts/build_rag_from_books.py --step 4 --batch-size 50

    # Dry run (no writes to ChromaDB)
    python scripts/build_rag_from_books.py --dry-run

    # Resume from checkpoint
    python scripts/build_rag_from_books.py --resume

    # Custom ChromaDB path
    python scripts/build_rag_from_books.py --chromadb-path /path/to/chromadb

    # Custom chunk size
    python scripts/build_rag_from_books.py --chunk-size 800 --chunk-overlap 150
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    DATA_DIR,
    EXISTING_CHROMADB_PATH,
    LOGS_DIR,
    RAG_CONFIG,
    setup_logging,
)

logger = logging.getLogger("market_hawk.build_rag")

# ============================================================
# CONSTANTS
# ============================================================

CATALOG_FILE = DATA_DIR / "rag_book_catalog.json"
EXTRACTED_DIR = DATA_DIR / "rag_extracted_texts"
CHUNKS_DIR = DATA_DIR / "rag_chunks"
CHECKPOINT_FILE = LOGS_DIR / "rag_build_checkpoint.json"
REPORT_FILE = Path(_PROJECT_ROOT) / "docs" / "RAG_BUILD_REPORT.md"

# Scan directories
BOOK_SCAN_PATHS: List[Path] = [
    Path(r"J:\E-Books"),
    Path(r"G:\KIT_TRAINING_BOOKS"),
]

# Suportate pentru extracere text
BOOK_EXTENSIONS: Set[str] = {".pdf", ".epub", ".djvu", ".docx"}

# Directoare non-trading de exclus (NHS, nursing, medical etc.)
EXCLUDE_DIRS: Set[str] = {
    "nhs", "nursing", "medical", "health", "anatomy", "physiology",
    "biology", "chemistry", "physics", "dentistry", "veterinary",
    "recipes", "cooking", "fitness", "diet", "nutrition",
    "photography", "music", "art", "crafts",
    "recycle bin", "$recycle.bin", "system volume information",
}

# Cuvinte cheie care indica continut relevant trading
TRADING_KEYWORDS: Set[str] = {
    "trading", "trader", "market", "stock", "forex", "crypto",
    "options", "futures", "commodities", "investing", "investment",
    "portfolio", "hedge", "arbitrage", "algorithmic", "algo",
    "technical analysis", "fundamental analysis", "candlestick",
    "fibonacci", "elliott wave", "price action", "order flow",
    "risk management", "money management", "quantitative", "quant",
    "backtest", "machine learning", "data science", "statistics",
    "probability", "stochastic", "volatility", "derivative",
    "bond", "equity", "fixed income", "etf", "mutual fund",
    "behavioral finance", "economics", "macroeconomics",
    "microstructure", "smart money", "ict", "wyckoff",
    "supply demand", "support resistance", "breakout", "momentum",
    "mean reversion", "trend following", "scalping", "swing",
    "day trading", "position trading", "fintech", "blockchain",
    "python", "programming", "deep learning", "neural network",
    "time series", "regression", "classification", "nlp",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit",
}

# Dimensiune maxima fisier PDF (500 MB)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024

# Batch size implicit pentru ChromaDB upsert
DEFAULT_BATCH_SIZE = 100

# Chunk defaults
DEFAULT_CHUNK_SIZE = 800       # tokens (aprox 4 chars/token)
DEFAULT_CHUNK_OVERLAP = 150    # tokens overlap


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BookEntry:
    """Single book in the catalog."""
    filepath: str
    filename: str
    extension: str
    size_bytes: int
    directory: str
    title_normalized: str
    is_trading_relevant: bool = True
    sha256: str = ""
    pages_estimated: int = 0


@dataclass
class ExtractionResult:
    """Result from text extraction of a single book."""
    filepath: str
    pages_extracted: int
    total_chars: int
    method: str  # "pdfplumber", "pypdf2", "fallback"
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class ChunkResult:
    """Result from chunking a single book."""
    filepath: str
    total_chunks: int
    avg_chunk_tokens: float
    chapters_detected: int


@dataclass
class PipelineStats:
    """Aggregated pipeline statistics."""
    total_books_found: int = 0
    books_scanned: int = 0
    books_extracted: int = 0
    books_skipped_exists: int = 0
    books_skipped_error: int = 0
    total_chunks_created: int = 0
    total_chunks_indexed: int = 0
    chunks_already_in_db: int = 0
    total_pages: int = 0
    elapsed_seconds: float = 0.0


# ============================================================
# INTERRUPT HANDLER
# ============================================================

_interrupted = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C — set flag for graceful shutdown."""
    global _interrupted
    _interrupted = True
    logger.warning("Interrupt received — saving checkpoint and exiting gracefully...")


signal.signal(signal.SIGINT, _signal_handler)


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """Save/load progress for pause/resume support."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.path = checkpoint_path
        self.data: Dict[str, Any] = {
            "step": 0,
            "processed_files": [],
            "stats": {},
            "last_updated": "",
        }

    def load(self) -> bool:
        """Load checkpoint from disk. Returns True if checkpoint exists."""
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info("Checkpoint loaded — step=%d, %d files processed",
                        self.data.get("step", 0),
                        len(self.data.get("processed_files", [])))
            return True
        return False

    def save(self, step: int, processed_files: List[str],
             stats: Dict[str, Any] = None):
        """Save checkpoint to disk."""
        self.data = {
            "step": step,
            "processed_files": processed_files,
            "stats": stats or {},
            "last_updated": datetime.now().isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=_json_default)
        logger.debug("Checkpoint saved — step=%d, %d files",
                     step, len(processed_files))

    def get_processed(self) -> Set[str]:
        """Return set of already-processed file paths."""
        return set(self.data.get("processed_files", []))

    def clear(self):
        """Remove checkpoint file."""
        if self.path.exists():
            self.path.unlink()
            logger.info("Checkpoint cleared")


def _json_default(obj):
    """JSON serializer for numpy types and Path objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ============================================================
# STEP 1: SCAN & CATALOG
# ============================================================

def normalize_title(filename: str) -> str:
    """
    Normalize book filename into a clean title.

    Removes:
        - File extension
        - Numeric IDs/hashes (Z-Library, PDFDrive etc.)
        - Parenthetical metadata (year, edition)
        - Special characters

    Args:
        filename: Raw filename (e.g., "Technical_Analysis_2020_Z-Lib.pdf")

    Returns:
        Normalized title (e.g., "technical analysis")
    """
    name = Path(filename).stem

    # Elimina Z-Library / PDFDrive markers
    name = re.sub(r"[-_]?z[-_]?lib(rary)?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[-_]?pdfdrive", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[-_]?libgen", "", name, flags=re.IGNORECASE)

    # Elimina hash-uri si ID-uri lungi
    name = re.sub(r"[a-f0-9]{8,}", "", name)

    # Elimina paranteze cu continut (editii, ani)
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\[[^\]]*\]", "", name)

    # Normalizeaza separatori
    name = re.sub(r"[_\-]+", " ", name)

    # Elimina numere izolate (IDs)
    name = re.sub(r"\b\d{5,}\b", "", name)

    # Curata spatii multiple
    name = re.sub(r"\s+", " ", name).strip().lower()

    return name


def is_trading_relevant(filepath: Path) -> bool:
    """
    Determine if a book is trading/finance relevant.

    Checks directory path and filename against keyword lists.

    Args:
        filepath: Full path to the book file.

    Returns:
        True if the book appears trading-relevant.
    """
    path_lower = str(filepath).lower()

    # Exclude daca e in director non-trading
    for exc in EXCLUDE_DIRS:
        if exc in path_lower:
            return False

    # Verifica cuvinte cheie in cale + filename
    name_lower = filepath.stem.lower().replace("_", " ").replace("-", " ")
    combined = f"{path_lower} {name_lower}"

    for kw in TRADING_KEYWORDS:
        if kw in combined:
            return True

    # Daca e in J:\E-Books directorul principal, presupunem relevant
    # (user-ul a spus ca 99% sunt trading)
    for scan_path in BOOK_SCAN_PATHS:
        if str(scan_path).lower() in path_lower:
            return True

    return False


def compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file (first 1MB for speed)."""
    sha = hashlib.sha256()
    bytes_read = 0
    max_bytes = 1024 * 1024  # 1 MB
    try:
        with open(filepath, "rb") as f:
            while bytes_read < max_bytes:
                data = f.read(chunk_size)
                if not data:
                    break
                sha.update(data)
                bytes_read += len(data)
    except OSError:
        return ""
    return sha.hexdigest()


def step1_scan_catalog(scan_paths: List[Path] = None,
                       resume_from: Set[str] = None) -> List[BookEntry]:
    """
    Step 1: Scan directories and build book catalog.

    Args:
        scan_paths: List of root directories to scan.
        resume_from: Set of already-processed file paths to skip.

    Returns:
        List of BookEntry objects for all discovered books.
    """
    from tqdm import tqdm

    scan_paths = scan_paths or BOOK_SCAN_PATHS
    resume_from = resume_from or set()
    catalog: List[BookEntry] = []
    skipped = 0

    logger.info("=" * 60)
    logger.info("STEP 1: Scan & Catalog")
    logger.info("=" * 60)

    for scan_root in scan_paths:
        if not scan_root.exists():
            logger.warning("Scan path does not exist: %s", scan_root)
            continue

        logger.info("Scanning: %s", scan_root)

        # Colecteaza toate fisierele
        all_files = []
        for ext in BOOK_EXTENSIONS:
            all_files.extend(scan_root.rglob(f"*{ext}"))

        logger.info("  Found %d book files", len(all_files))

        for fpath in tqdm(all_files, desc=f"Cataloging {scan_root.name}",
                          unit="book"):
            if _interrupted:
                logger.warning("Interrupted during catalog scan")
                break

            fpath_str = str(fpath)

            if fpath_str in resume_from:
                skipped += 1
                continue

            try:
                stat = fpath.stat()

                # Skip fisiere prea mari
                if stat.st_size > MAX_FILE_SIZE_BYTES:
                    logger.debug("Skipping oversized file: %s (%.1f MB)",
                                 fpath.name, stat.st_size / 1024 / 1024)
                    continue

                # Skip fisiere goale
                if stat.st_size < 1024:
                    continue

                entry = BookEntry(
                    filepath=fpath_str,
                    filename=fpath.name,
                    extension=fpath.suffix.lower(),
                    size_bytes=stat.st_size,
                    directory=str(fpath.parent),
                    title_normalized=normalize_title(fpath.name),
                    is_trading_relevant=is_trading_relevant(fpath),
                    sha256=compute_file_hash(fpath),
                )
                catalog.append(entry)

            except OSError as e:
                logger.debug("Cannot access %s: %s", fpath, e)

    # Deduplicare pe hash
    seen_hashes: Set[str] = set()
    unique_catalog: List[BookEntry] = []
    duplicates = 0
    for entry in catalog:
        if entry.sha256 and entry.sha256 in seen_hashes:
            duplicates += 1
            continue
        if entry.sha256:
            seen_hashes.add(entry.sha256)
        unique_catalog.append(entry)

    # Filtreaza doar trading-relevant
    trading_catalog = [e for e in unique_catalog if e.is_trading_relevant]

    logger.info("-" * 40)
    logger.info("Catalog summary:")
    logger.info("  Total files found: %d", len(catalog))
    logger.info("  Duplicates removed: %d", duplicates)
    logger.info("  Non-trading filtered: %d", len(unique_catalog) - len(trading_catalog))
    logger.info("  Trading books cataloged: %d", len(trading_catalog))
    if skipped:
        logger.info("  Skipped (already processed): %d", skipped)

    # Salveaza catalog
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog_data = [asdict(e) for e in trading_catalog]
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2, default=_json_default)
    logger.info("  Catalog saved: %s", CATALOG_FILE)

    return trading_catalog


# ============================================================
# STEP 2: TEXT EXTRACTION
# ============================================================

def extract_text_pdfplumber(filepath: str) -> Tuple[List[str], str]:
    """
    Extract text from PDF using pdfplumber (primary method).

    Args:
        filepath: Path to PDF file.

    Returns:
        Tuple of (list of page texts, method name).

    Raises:
        ImportError: If pdfplumber not installed.
        Exception: If extraction fails.
    """
    import pdfplumber

    pages_text: List[str] = []

    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

    return pages_text, "pdfplumber"


def extract_text_pypdf2(filepath: str) -> Tuple[List[str], str]:
    """
    Extract text from PDF using PyPDF2 (fallback method).

    Args:
        filepath: Path to PDF file.

    Returns:
        Tuple of (list of page texts, method name).
    """
    from PyPDF2 import PdfReader

    pages_text: List[str] = []
    reader = PdfReader(filepath)

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())

    return pages_text, "pypdf2"


def extract_text_from_book(filepath: str) -> Tuple[List[str], str]:
    """
    Extract text from a book file. Tries pdfplumber first, then PyPDF2.

    Args:
        filepath: Path to book file.

    Returns:
        Tuple of (list of page texts, extraction method used).

    Raises:
        ValueError: If file format is not supported.
    """
    ext = Path(filepath).suffix.lower()

    if ext != ".pdf":
        raise ValueError(f"Unsupported format for extraction: {ext}. "
                         f"Only PDF supported. EPUB/DJVU/DOCX not yet implemented.")

    # Incearca pdfplumber
    try:
        pages, method = extract_text_pdfplumber(filepath)
        if pages and sum(len(p) for p in pages) > 100:
            return pages, method
    except Exception as e:
        logger.debug("pdfplumber failed for %s: %s", Path(filepath).name, e)

    # Fallback pe PyPDF2
    try:
        pages, method = extract_text_pypdf2(filepath)
        if pages and sum(len(p) for p in pages) > 100:
            return pages, method
    except Exception as e:
        logger.debug("PyPDF2 failed for %s: %s", Path(filepath).name, e)

    return [], "failed"


def step2_extract_texts(catalog: List[BookEntry],
                        processed_files: Set[str] = None,
                        checkpoint_mgr: CheckpointManager = None
                        ) -> Tuple[Dict[str, List[str]], PipelineStats]:
    """
    Step 2: Extract text from all cataloged books.

    Args:
        catalog: List of BookEntry from step 1.
        processed_files: Set of already-processed filepaths.
        checkpoint_mgr: For saving progress during extraction.

    Returns:
        Tuple of (filepath -> list of page texts, stats).
    """
    from tqdm import tqdm

    processed_files = processed_files or set()
    stats = PipelineStats(total_books_found=len(catalog))
    texts: Dict[str, List[str]] = {}
    processed_list = list(processed_files)

    logger.info("=" * 60)
    logger.info("STEP 2: Text Extraction")
    logger.info("=" * 60)

    # Filtreaza doar PDF-uri (celelalte formate nu sunt suportate inca)
    pdf_catalog = [e for e in catalog if e.extension == ".pdf"]
    non_pdf = len(catalog) - len(pdf_catalog)
    if non_pdf > 0:
        logger.info("  Skipping %d non-PDF files (EPUB/DJVU/DOCX not yet supported)",
                     non_pdf)

    # Filtreaza deja procesate
    to_process = [e for e in pdf_catalog if e.filepath not in processed_files]
    stats.books_skipped_exists = len(pdf_catalog) - len(to_process)

    if stats.books_skipped_exists > 0:
        logger.info("  Resuming — %d already extracted, %d remaining",
                     stats.books_skipped_exists, len(to_process))

    # Incarca texte existente salvate anterior
    if EXTRACTED_DIR.exists():
        for saved_file in EXTRACTED_DIR.glob("*.json"):
            try:
                with open(saved_file, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                    texts[saved_data["filepath"]] = saved_data["pages"]
            except (json.JSONDecodeError, KeyError):
                pass

    start_time = time.time()

    pbar = tqdm(to_process, desc="Extracting text", unit="book",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for entry in pbar:
        if _interrupted:
            logger.warning("Interrupted during extraction")
            if checkpoint_mgr:
                checkpoint_mgr.save(2, processed_list, asdict(stats))
            break

        pbar.set_postfix_str(f"{entry.filename[:40]}...")

        try:
            pages, method = extract_text_from_book(entry.filepath)

            if not pages:
                stats.books_skipped_error += 1
                logger.debug("No text extracted: %s", entry.filename)
                continue

            texts[entry.filepath] = pages
            stats.books_extracted += 1
            stats.total_pages += len(pages)

            # Salveaza text extras individual (pentru resume)
            EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', entry.filename)
            save_path = EXTRACTED_DIR / f"{safe_name}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "filepath": entry.filepath,
                    "filename": entry.filename,
                    "method": method,
                    "pages": pages,
                    "page_count": len(pages),
                }, f, default=_json_default)

            processed_list.append(entry.filepath)

            # Checkpoint la fiecare 50 carti
            if stats.books_extracted % 50 == 0:
                if checkpoint_mgr:
                    checkpoint_mgr.save(2, processed_list, asdict(stats))
                gc.collect()

        except Exception as e:
            stats.books_skipped_error += 1
            logger.debug("Extraction error for %s: %s", entry.filename, e)

    stats.elapsed_seconds = time.time() - start_time
    stats.books_scanned = stats.books_extracted + stats.books_skipped_error

    logger.info("-" * 40)
    logger.info("Extraction summary:")
    logger.info("  Books extracted: %d", stats.books_extracted)
    logger.info("  Total pages: %d", stats.total_pages)
    logger.info("  Errors/empty: %d", stats.books_skipped_error)
    logger.info("  Elapsed: %.1fs", stats.elapsed_seconds)

    return texts, stats


# ============================================================
# STEP 3: INTELLIGENT CHUNKING
# ============================================================

# Pattern-uri pentru detectia capitolelor si sectiunilor
CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^part\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^section\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE),  # "1. Introduction"
    re.compile(r"^[A-Z][A-Z\s]{10,}$", re.MULTILINE),  # ALL-CAPS heading
]


def detect_chapters(text: str) -> List[int]:
    """
    Detect chapter/section boundaries in text.

    Args:
        text: Full text content.

    Returns:
        Sorted list of character positions where chapters start.
    """
    boundaries: Set[int] = set()

    for pattern in CHAPTER_PATTERNS:
        for match in pattern.finditer(text):
            boundaries.add(match.start())

    return sorted(boundaries)


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars per token for English)."""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
               ) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with chapter awareness.

    Prefers splitting at paragraph boundaries. Respects chapter boundaries
    when detected. Each chunk includes metadata about position.

    Args:
        text: Full text to chunk.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        List of dicts with keys: text, start_char, end_char, chunk_idx,
        chapter_idx, token_estimate.
    """
    if not text.strip():
        return []

    # Char equivalents
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    # Detecteaza capitole
    chapter_starts = detect_chapters(text)
    chapter_starts = [0] + [c for c in chapter_starts if c > 0]

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    # Daca nu am detectat capitole, chunk liniar
    if len(chapter_starts) <= 1:
        chapter_starts = [0]

    for ch_idx, ch_start in enumerate(chapter_starts):
        # Determina sfarsitul capitolului
        if ch_idx + 1 < len(chapter_starts):
            ch_end = chapter_starts[ch_idx + 1]
        else:
            ch_end = len(text)

        chapter_text = text[ch_start:ch_end]
        if not chapter_text.strip():
            continue

        # Chunk in interiorul capitolului
        pos = 0
        while pos < len(chapter_text):
            end_pos = min(pos + char_size, len(chapter_text))

            # Incearca sa taie la sfarsit de paragraf
            if end_pos < len(chapter_text):
                # Cauta ultimul "\n\n" in intervalul [pos + char_size//2, end_pos]
                search_start = max(pos + char_size // 2, pos)
                para_break = chapter_text.rfind("\n\n", search_start, end_pos + 200)
                if para_break > search_start:
                    end_pos = para_break + 2  # include \n\n

                # Daca nu gasim paragraf, cauta sfarsit de propozitie
                elif end_pos < len(chapter_text):
                    sent_break = chapter_text.rfind(". ", search_start, end_pos + 100)
                    if sent_break > search_start:
                        end_pos = sent_break + 2

            chunk_text_raw = chapter_text[pos:end_pos].strip()

            if chunk_text_raw and len(chunk_text_raw) > 50:
                chunks.append({
                    "text": chunk_text_raw,
                    "start_char": ch_start + pos,
                    "end_char": ch_start + end_pos,
                    "chunk_idx": chunk_idx,
                    "chapter_idx": ch_idx,
                    "token_estimate": estimate_tokens(chunk_text_raw),
                })
                chunk_idx += 1

            # Avanseaza cu overlap
            pos = end_pos - char_overlap
            if pos <= 0 and end_pos >= len(chapter_text):
                break
            if pos >= len(chapter_text):
                break

    return chunks


def step3_chunk_texts(texts: Dict[str, List[str]],
                      catalog: List[BookEntry],
                      chunk_size: int = DEFAULT_CHUNK_SIZE,
                      chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                      ) -> Tuple[List[Dict[str, Any]], List[ChunkResult]]:
    """
    Step 3: Chunk extracted texts with metadata.

    Args:
        texts: filepath -> list of page texts (from step 2).
        catalog: Book catalog for metadata enrichment.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between chunks.

    Returns:
        Tuple of (all chunks with metadata, per-book chunk results).
    """
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("STEP 3: Intelligent Chunking")
    logger.info("=" * 60)
    logger.info("  Chunk size: %d tokens, overlap: %d tokens", chunk_size, chunk_overlap)

    # Index catalog pe filepath
    catalog_index: Dict[str, BookEntry] = {e.filepath: e for e in catalog}

    all_chunks: List[Dict[str, Any]] = []
    results: List[ChunkResult] = []

    for filepath, pages in tqdm(texts.items(), desc="Chunking", unit="book"):
        if _interrupted:
            break

        # Combina paginile intr-un text continuu
        full_text = "\n\n".join(pages)

        if not full_text.strip():
            continue

        # Chunk
        book_chunks = chunk_text(full_text, chunk_size, chunk_overlap)

        if not book_chunks:
            continue

        # Adauga metadata
        entry = catalog_index.get(filepath)
        book_name = entry.filename if entry else Path(filepath).name
        book_title = entry.title_normalized if entry else normalize_title(book_name)

        chapters_detected = len(set(c["chapter_idx"] for c in book_chunks))

        for chunk in book_chunks:
            # Genereaza ID unic pentru chunk (hash continut + sursa)
            chunk_id = hashlib.sha256(
                f"{filepath}:{chunk['chunk_idx']}:{chunk['text'][:100]}".encode()
            ).hexdigest()[:16]

            chunk["id"] = f"rag_{chunk_id}"
            chunk["source"] = book_name
            chunk["source_path"] = filepath
            chunk["book_title"] = book_title
            chunk["directory"] = str(Path(filepath).parent)

            all_chunks.append(chunk)

        avg_tokens = np.mean([c["token_estimate"] for c in book_chunks])
        results.append(ChunkResult(
            filepath=filepath,
            total_chunks=len(book_chunks),
            avg_chunk_tokens=float(avg_tokens),
            chapters_detected=chapters_detected,
        ))

    # Stats
    total_chunks = len(all_chunks)
    if total_chunks > 0:
        token_counts = [c["token_estimate"] for c in all_chunks]
        logger.info("-" * 40)
        logger.info("Chunking summary:")
        logger.info("  Total chunks: %d", total_chunks)
        logger.info("  Avg tokens/chunk: %.0f", np.mean(token_counts))
        logger.info("  Min/Max tokens: %d / %d", min(token_counts), max(token_counts))
        logger.info("  Books chunked: %d", len(results))
    else:
        logger.warning("No chunks generated!")

    return all_chunks, results


# ============================================================
# STEP 4: EMBEDDING & INDEXING
# ============================================================

def step4_index_to_chromadb(chunks: List[Dict[str, Any]],
                            chromadb_path: str = None,
                            collection_name: str = None,
                            batch_size: int = DEFAULT_BATCH_SIZE,
                            dry_run: bool = False,
                            checkpoint_mgr: CheckpointManager = None,
                            processed_ids: Set[str] = None,
                            ) -> PipelineStats:
    """
    Step 4: Embed and index chunks into ChromaDB.

    Uses nomic-embed-text via Ollama (same as existing RAG).

    Args:
        chunks: List of chunk dicts from step 3.
        chromadb_path: Path to ChromaDB persist directory.
        collection_name: ChromaDB collection name.
        batch_size: Number of chunks per upsert batch.
        dry_run: If True, skip actual writes.
        checkpoint_mgr: For saving progress.
        processed_ids: Set of chunk IDs already indexed.

    Returns:
        PipelineStats with indexing metrics.
    """
    from tqdm import tqdm

    chromadb_path = chromadb_path or EXISTING_CHROMADB_PATH
    collection_name = collection_name or RAG_CONFIG.collection_name
    processed_ids = processed_ids or set()

    stats = PipelineStats(total_chunks_created=len(chunks))

    logger.info("=" * 60)
    logger.info("STEP 4: Embedding & Indexing to ChromaDB")
    logger.info("=" * 60)
    logger.info("  ChromaDB: %s", chromadb_path)
    logger.info("  Collection: %s", collection_name)
    logger.info("  Chunks to index: %d", len(chunks))
    logger.info("  Batch size: %d", batch_size)

    if dry_run:
        logger.info("  *** DRY RUN — no writes to ChromaDB ***")
        stats.total_chunks_indexed = len(chunks)
        return stats

    # Filtreaza chunk-uri deja indexate
    new_chunks = [c for c in chunks if c["id"] not in processed_ids]
    stats.chunks_already_in_db = len(chunks) - len(new_chunks)

    if stats.chunks_already_in_db > 0:
        logger.info("  Already indexed: %d, remaining: %d",
                     stats.chunks_already_in_db, len(new_chunks))

    if not new_chunks:
        logger.info("  All chunks already indexed!")
        return stats

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        # Conecteaza la ChromaDB
        client = chromadb.PersistentClient(path=chromadb_path)

        # Embedding function — nomic-embed-text via Ollama
        # Folosim DefaultEmbeddingFunction daca Ollama nu e disponibil,
        # dar preferinta e OllamaEmbeddingFunction
        try:
            # Incearca OllamaEmbeddingFunction
            ollama_ef = embedding_functions.OllamaEmbeddingFunction(
                url=f"{RAG_CONFIG.ollama_host}/api/embeddings",
                model_name=RAG_CONFIG.embedding_model,
            )
            # Test rapid
            test_result = ollama_ef(["test"])
            if test_result and len(test_result[0]) > 0:
                ef = ollama_ef
                logger.info("  Using Ollama embedding: %s", RAG_CONFIG.embedding_model)
            else:
                raise RuntimeError("Empty embedding result")
        except Exception as e:
            logger.warning("  Ollama embedding unavailable: %s", e)
            logger.warning("  Falling back to default embedding (all-MiniLM-L6-v2)")
            ef = embedding_functions.DefaultEmbeddingFunction()

        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = collection.count()
        logger.info("  Existing chunks in collection: %d", existing_count)

        # Index in batches
        start_time = time.time()
        indexed = 0
        processed_list = list(processed_ids)

        pbar = tqdm(range(0, len(new_chunks), batch_size),
                    desc="Indexing", unit="batch",
                    total=(len(new_chunks) + batch_size - 1) // batch_size)

        for batch_start in pbar:
            if _interrupted:
                logger.warning("Interrupted during indexing")
                if checkpoint_mgr:
                    checkpoint_mgr.save(4, processed_list, asdict(stats))
                break

            batch = new_chunks[batch_start:batch_start + batch_size]

            ids = [c["id"] for c in batch]
            documents = [c["text"] for c in batch]
            metadatas = []
            for c in batch:
                metadatas.append({
                    "source": c.get("source", "unknown"),
                    "book_title": c.get("book_title", ""),
                    "chunk_idx": c.get("chunk_idx", 0),
                    "chapter_idx": c.get("chapter_idx", 0),
                    "start_char": c.get("start_char", 0),
                    "end_char": c.get("end_char", 0),
                    "token_estimate": c.get("token_estimate", 0),
                    "source_path": c.get("directory", ""),
                    "pipeline": "build_rag_from_books",
                    "indexed_at": datetime.now().isoformat(),
                })

            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                indexed += len(batch)
                processed_list.extend(ids)
            except Exception as e:
                logger.error("Batch upsert failed at %d: %s", batch_start, e)
                # Incearca individual
                for i, (cid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                    try:
                        collection.upsert(ids=[cid], documents=[doc],
                                          metadatas=[meta])
                        indexed += 1
                        processed_list.append(cid)
                    except Exception as ie:
                        logger.debug("Individual upsert failed: %s — %s", cid, ie)

            # Checkpoint la fiecare 10 batch-uri
            if (batch_start // batch_size) % 10 == 0 and checkpoint_mgr:
                stats.total_chunks_indexed = indexed
                checkpoint_mgr.save(4, processed_list, asdict(stats))

            pbar.set_postfix_str(f"indexed={indexed}")

            # Elibereaza memorie
            if (batch_start // batch_size) % 50 == 0:
                gc.collect()

        stats.total_chunks_indexed = indexed
        stats.elapsed_seconds = time.time() - start_time

        final_count = collection.count()

        logger.info("-" * 40)
        logger.info("Indexing summary:")
        logger.info("  Chunks indexed: %d", indexed)
        logger.info("  Collection total: %d (was %d)", final_count, existing_count)
        logger.info("  New chunks added: %d", final_count - existing_count)
        logger.info("  Elapsed: %.1fs", stats.elapsed_seconds)

    except ImportError as e:
        logger.error("Missing dependency: %s", e)
        logger.error("Install: pip install chromadb")
        raise
    except Exception:
        logger.exception("Indexing failed")
        raise

    return stats


# ============================================================
# STEP 5: VERIFICATION
# ============================================================

def step5_verify(chromadb_path: str = None,
                 collection_name: str = None) -> Dict[str, Any]:
    """
    Step 5: Verify the updated knowledge base.

    Runs test queries and reports statistics.

    Args:
        chromadb_path: Path to ChromaDB directory.
        collection_name: Collection name to verify.

    Returns:
        Dict with verification results.
    """
    chromadb_path = chromadb_path or EXISTING_CHROMADB_PATH
    collection_name = collection_name or RAG_CONFIG.collection_name

    logger.info("=" * 60)
    logger.info("STEP 5: Verification")
    logger.info("=" * 60)

    results: Dict[str, Any] = {
        "status": "unknown",
        "total_chunks": 0,
        "unique_sources": 0,
        "test_queries": [],
    }

    try:
        import chromadb

        client = chromadb.PersistentClient(path=chromadb_path)
        collection = client.get_collection(name=collection_name)

        total = collection.count()
        results["total_chunks"] = total
        logger.info("  Total chunks in collection: %d", total)

        # Interogheaza surse unice
        # ChromaDB nu suporta DISTINCT, facem sample
        sample_size = min(total, 10000)
        sample = collection.get(
            limit=sample_size,
            include=["metadatas"],
        )
        sources = set()
        pipeline_sources = Counter()
        for meta in sample["metadatas"]:
            src = meta.get("source", "unknown")
            sources.add(src)
            pipeline = meta.get("pipeline", "legacy")
            pipeline_sources[pipeline] += 1

        results["unique_sources"] = len(sources)
        results["pipeline_breakdown"] = dict(pipeline_sources)

        logger.info("  Unique sources (from sample): %d", len(sources))
        logger.info("  Pipeline breakdown: %s", dict(pipeline_sources))

        # Test queries
        test_queries = [
            "What are the best entry criteria for order block trades?",
            "How to manage risk in volatile markets?",
            "Explain the concept of smart money accumulation",
            "What is the Kelly criterion for position sizing?",
            "How does mean reversion strategy work?",
        ]

        for query in test_queries:
            start_time = time.time()
            query_results = collection.query(
                query_texts=[query],
                n_results=3,
            )
            elapsed = time.time() - start_time

            top_sources = []
            if query_results["metadatas"]:
                top_sources = [
                    m.get("source", "?") for m in query_results["metadatas"][0]
                ]

            results["test_queries"].append({
                "query": query,
                "results_count": len(query_results["ids"][0]) if query_results["ids"] else 0,
                "top_sources": top_sources,
                "latency_ms": round(elapsed * 1000, 1),
            })

            logger.info("  Query: '%s...'", query[:50])
            logger.info("    Results: %d, Latency: %.1fms",
                         len(query_results["ids"][0]) if query_results["ids"] else 0,
                         elapsed * 1000)
            for src in top_sources:
                logger.info("    Source: %s", src)

        results["status"] = "ok"

    except ImportError:
        logger.error("chromadb not installed — cannot verify")
        results["status"] = "error_dependency"
    except Exception as e:
        logger.exception("Verification failed")
        results["status"] = f"error: {str(e)}"

    return results


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(catalog: List[BookEntry],
                    stats: PipelineStats,
                    verification: Dict[str, Any]) -> str:
    """
    Generate Markdown report of the RAG build pipeline run.

    Args:
        catalog: Book catalog from step 1.
        stats: Pipeline statistics.
        verification: Verification results from step 5.

    Returns:
        Markdown report string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Distributia pe extensii
    ext_counts = Counter(e.extension for e in catalog)

    report = f"""# RAG Build Report

**Generated:** {now}
**Pipeline:** `scripts/build_rag_from_books.py`

## Catalog

| Metric | Value |
|--------|-------|
| Books found | {stats.total_books_found} |
| Books extracted | {stats.books_extracted} |
| Extraction errors | {stats.books_skipped_error} |
| Total pages | {stats.total_pages} |

### By format

| Extension | Count |
|-----------|-------|
"""
    for ext, count in ext_counts.most_common():
        report += f"| {ext} | {count} |\n"

    report += f"""
## Chunking & Indexing

| Metric | Value |
|--------|-------|
| Total chunks created | {stats.total_chunks_created} |
| Chunks indexed | {stats.total_chunks_indexed} |
| Already in DB | {stats.chunks_already_in_db} |
| Elapsed | {stats.elapsed_seconds:.1f}s |

## Verification

| Metric | Value |
|--------|-------|
| Status | {verification.get('status', 'N/A')} |
| Total chunks in collection | {verification.get('total_chunks', 'N/A')} |
| Unique sources | {verification.get('unique_sources', 'N/A')} |

### Test Queries

"""
    for tq in verification.get("test_queries", []):
        report += f"**Q:** {tq['query']}\n"
        report += f"- Results: {tq['results_count']}, Latency: {tq['latency_ms']}ms\n"
        for src in tq.get("top_sources", []):
            report += f"  - {src}\n"
        report += "\n"

    return report


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """
    Run the full RAG build pipeline (or a single step).

    Args:
        args: Parsed command-line arguments.
    """
    global _interrupted

    start_time = time.time()

    checkpoint = CheckpointManager()
    processed_files: Set[str] = set()
    processed_ids: Set[str] = set()

    # Resume support
    if args.resume:
        if checkpoint.load():
            processed_files = checkpoint.get_processed()
            logger.info("Resuming from step %d with %d processed files",
                        checkpoint.data.get("step", 0), len(processed_files))
        else:
            logger.info("No checkpoint found — starting fresh")

    catalog: List[BookEntry] = []
    texts: Dict[str, List[str]] = {}
    chunks: List[Dict[str, Any]] = []
    stats = PipelineStats()
    verification: Dict[str, Any] = {}

    run_step = args.step  # 0 = all

    # ----- STEP 1 -----
    if run_step in (0, 1):
        catalog = step1_scan_catalog(resume_from=processed_files)
        checkpoint.save(1, [e.filepath for e in catalog])

        if _interrupted:
            return
    elif CATALOG_FILE.exists():
        # Incarca catalog existent
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            catalog_data = json.load(f)
        catalog = [BookEntry(**d) for d in catalog_data]
        logger.info("Loaded existing catalog: %d books", len(catalog))

    # ----- STEP 2 -----
    if run_step in (0, 2):
        if not catalog:
            logger.error("No catalog available. Run step 1 first.")
            return

        texts, stats = step2_extract_texts(
            catalog, processed_files, checkpoint
        )

        if _interrupted:
            return
    elif EXTRACTED_DIR.exists():
        # Incarca texte extrase
        for saved_file in EXTRACTED_DIR.glob("*.json"):
            try:
                with open(saved_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    texts[data["filepath"]] = data["pages"]
            except (json.JSONDecodeError, KeyError):
                pass
        logger.info("Loaded %d extracted texts from cache", len(texts))

    # ----- STEP 3 -----
    if run_step in (0, 3):
        if not texts:
            logger.error("No texts available. Run step 2 first.")
            return

        chunks, chunk_results = step3_chunk_texts(
            texts, catalog,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        stats.total_chunks_created = len(chunks)

        # Salveaza chunks (pentru restart step 4)
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        chunks_file = CHUNKS_DIR / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, default=_json_default)
        logger.info("Chunks saved: %s (%d chunks)", chunks_file, len(chunks))

        if _interrupted:
            return
    elif (CHUNKS_DIR / "all_chunks.json").exists():
        # Incarca chunks existente
        with open(CHUNKS_DIR / "all_chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info("Loaded %d chunks from cache", len(chunks))

    # ----- STEP 4 -----
    if run_step in (0, 4):
        if not chunks:
            logger.error("No chunks available. Run step 3 first.")
            return

        stats_step4 = step4_index_to_chromadb(
            chunks,
            chromadb_path=args.chromadb_path,
            collection_name=args.collection,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            checkpoint_mgr=checkpoint,
            processed_ids=processed_ids,
        )
        stats.total_chunks_indexed = stats_step4.total_chunks_indexed
        stats.chunks_already_in_db = stats_step4.chunks_already_in_db

        if _interrupted:
            return

    # ----- STEP 5 -----
    if run_step in (0, 5):
        if args.dry_run:
            logger.info("Skipping verification in dry-run mode")
            verification = {"status": "dry_run", "total_chunks": 0}
        else:
            verification = step5_verify(
                chromadb_path=args.chromadb_path,
                collection_name=args.collection,
            )

    # ----- REPORT -----
    elapsed_total = time.time() - start_time
    stats.elapsed_seconds = elapsed_total

    if catalog or verification:
        report = generate_report(catalog, stats, verification)
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Report saved: %s", REPORT_FILE)

    # Cleanup checkpoint on successful completion
    if not _interrupted and run_step == 0:
        checkpoint.clear()

    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed_total)
    logger.info("=" * 60)

    gc.collect()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the RAG build pipeline."""
    parser = argparse.ArgumentParser(
        description="Build/expand RAG Knowledge Base from trading books into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/build_rag_from_books.py

  # Scan & catalog only
  python scripts/build_rag_from_books.py --step 1

  # Extract text only
  python scripts/build_rag_from_books.py --step 2

  # Dry run (no writes)
  python scripts/build_rag_from_books.py --dry-run

  # Resume from checkpoint
  python scripts/build_rag_from_books.py --resume

  # Custom settings
  python scripts/build_rag_from_books.py --chunk-size 600 --chunk-overlap 100 --batch-size 200
        """,
    )

    parser.add_argument(
        "--step", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
        help="Run a single step (0=all, 1=catalog, 2=extract, 3=chunk, 4=index, 5=verify)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing to ChromaDB",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--chromadb-path", type=str, default=EXISTING_CHROMADB_PATH,
        help=f"ChromaDB persist directory (default: {EXISTING_CHROMADB_PATH})",
    )
    parser.add_argument(
        "--collection", type=str, default=RAG_CONFIG.collection_name,
        help=f"ChromaDB collection name (default: {RAG_CONFIG.collection_name})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in tokens (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in tokens (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"ChromaDB upsert batch size (default: {DEFAULT_BATCH_SIZE})",
    )

    return parser


if __name__ == "__main__":
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RAG Build Pipeline — Market Hawk 3")
    logger.info("=" * 60)
    logger.info("  Step: %s", "ALL" if args.step == 0 else str(args.step))
    logger.info("  Dry run: %s", args.dry_run)
    logger.info("  Resume: %s", args.resume)
    logger.info("  ChromaDB: %s", args.chromadb_path)
    logger.info("  Collection: %s", args.collection)
    logger.info("  Chunk size: %d tokens", args.chunk_size)
    logger.info("  Chunk overlap: %d tokens", args.chunk_overlap)
    logger.info("  Batch size: %d", args.batch_size)

    run_pipeline(args)
