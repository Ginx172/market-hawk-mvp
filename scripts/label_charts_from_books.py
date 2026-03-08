"""
SCRIPT NAME: label_charts_from_books.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Auto-label chart images by mapping book titles to topic directories.
Hardware Optimization: Intel i7-9700F, 64GB DDR4
Creation Date: 2026-03-08

Leverages the J:\\E-Books directory structure for labeling:
    If an image was extracted from "Fibonacci Trading.pdf" and that PDF
    lives in J:\\E-Books\\...\\Fibonacci\\, ALL images from that book
    get label "fibonacci".

4-step pipeline:
    Step 1: Scan J:\\E-Books, build book_filename -> topic mapping
    Step 2: Parse image filenames, extract book titles, fuzzy-match
    Step 3: Apply labels + subsample mega-book (Al Brooks) to max 2000
    Step 4: Update manifest CSV + generate report

Usage:
    # Run all steps
    python scripts/label_charts_from_books.py

    # Single step
    python scripts/label_charts_from_books.py --step 1
    python scripts/label_charts_from_books.py --step 2
    python scripts/label_charts_from_books.py --step 3 --max-per-book 2000
    python scripts/label_charts_from_books.py --step 4

    # Dry run (no writes)
    python scripts/label_charts_from_books.py --dry-run

    # Resume
    python scripts/label_charts_from_books.py --resume
"""

import argparse
import difflib
import gc
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, LOGS_DIR, setup_logging

logger = logging.getLogger("market_hawk.label_charts_from_books")

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MANIFEST = DATA_DIR / "chart_dataset_manifest.csv"
BOOK_TOPIC_MAPPING_FILE = DATA_DIR / "book_topic_mapping.json"
IMAGE_BOOK_MAPPING_FILE = DATA_DIR / "image_book_mapping.json"
REPORT_OUTPUT = Path(_PROJECT_ROOT) / "docs" / "BOOK_LABELING_REPORT.md"
CHECKPOINT_FILE = LOGS_DIR / "book_labeling_checkpoint.json"

# J:\E-Books base path
EBOOKS_BASE = Path(r"J:\E-Books")

# Book file extensions to scan
BOOK_EXTENSIONS: Set[str] = {".pdf", ".epub", ".docx", ".chm", ".mobi"}

# Trading-relevant topic directories (filter out NHS/nursing/medical)
# Matched against normalized directory names
TRADING_TOPICS: Dict[str, str] = {
    # Directory name (normalized) -> canonical label
    "algo trading&ai_trading": "algo_trading",
    "analiza-pitchfork": "pitchfork",
    "bollinger bands": "bollinger_bands",
    "breakout": "breakout",
    "charts": "chart_analysis",
    "crypto": "crypto",
    "day trading": "day_trading",
    "day trading university": "day_trading",
    "economic indicators": "economic_indicators",
    "elliot waves": "elliott_wave",
    "fibonacci": "fibonacci",
    "fire_base_trading_plan": "trading_plan",
    "fundamental analysis": "fundamental_analysis",
    "futures_options_market": "futures_options",
    "gann": "gann",
    "harmonics": "harmonics",
    "hedging": "hedging",
    "high frequency trading": "hft",
    "ichimoku": "ichimoku",
    "indices": "indices",
    "investment banking": "investment",
    "macd": "macd",
    "market_microstructure": "market_microstructure",
    "median line study": "median_line",
    "momentum": "momentum",
    "momoentum": "momentum",
    "moving_average": "moving_average",
    "order_blocks": "order_blocks",
    "planuri de trading": "trading_plan",
    "position_sizing": "position_sizing",
    "price_action": "price_action",
    "price_volume": "price_volume",
    "pullback": "pullback",
    "quants": "quants",
    "renko_charts": "renko",
    "reversal trading": "reversal",
    "risk_management": "risk_management",
    "01_risk_management": "risk_management",
    "rsi": "rsi",
    "scalping": "scalping",
    "sentiment_trading": "sentiment",
    "set-up-trading set up": "trading_setup",
    "short_terms(quick_time_frames)": "short_term",
    "smart money": "smart_money",
    "sniper trading": "sniper_trading",
    "strategies": "strategies",
    "stochastics": "stochastics",
    "supply & demand": "supply_demand",
    "swing": "swing_trading",
    "ta": "technical_analysis",
    "trading journal": "trading_journal",
    "trading psychology": "trading_psychology",
    "trading system": "trading_system",
    "trend line": "trend",
    "volatility": "volatility",
    "wyckoff trading": "wyckoff",
    "investitii": "investment",
    "05_options_futures": "futures_options",
    "03_market_microstructure": "market_microstructure",
    "infografic": "chart_analysis",
    "forex & dumitrel rada": "forex",
    "forex profit accelerator": "forex",
    "forex trading collection": "forex",
    ".rag_trading": "technical_analysis",
    "........trading": "technical_analysis",
    ".......cursuri": "technical_analysis",
    "...categorii carti": "technical_analysis",
    "coding_trading": "algo_trading",
    "documente ptr trading": "technical_analysis",
    "de pe discord": "technical_analysis",
    "pro trade group": "technical_analysis",
    "cursuri": "technical_analysis",
    "altele": "other",
    "arhive": "other",
}

# Fuzzy matching threshold
FUZZY_THRESHOLD: float = 0.60

# Default max images per single book (mega-book subsampling)
DEFAULT_MAX_PER_BOOK: int = 2000

# Format prefixes in image filenames
FORMAT_PREFIXES = ("PDF_", "EPUB_", "DOCX_", "CHM_", "MOBI_", "HTML_")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LabelingStats:
    """Statistics for the book-based labeling run."""
    total_images: int = 0
    total_traceable: int = 0
    total_anonymous: int = 0
    books_matched: int = 0
    books_unmatched: int = 0
    images_labeled: int = 0
    images_unclassified: int = 0
    images_subsampled_out: int = 0
    topic_distribution: Dict[str, int] = field(default_factory=dict)
    book_match_details: Dict[str, str] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "BOOK-BASED LABELING REPORT",
            "=" * 60,
            f"Total images:           {self.total_images:>8,}",
            f"Traceable to a book:    {self.total_traceable:>8,}",
            f"Anonymous:              {self.total_anonymous:>8,}",
            f"Books matched to topic: {self.books_matched:>8,}",
            f"Books unmatched:        {self.books_unmatched:>8,}",
            f"Images labeled:         {self.images_labeled:>8,}",
            f"Images unclassified:    {self.images_unclassified:>8,}",
            f"Images subsampled out:  {self.images_subsampled_out:>8,}",
            f"Elapsed time:           {self.elapsed_sec:>8.1f}s",
            "",
            "--- Topic Distribution ---",
        ]
        for topic, count in sorted(
            self.topic_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {topic:<30s} {count:>8,}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# STEP 1: BUILD BOOK -> TOPIC MAPPING
# ============================================================

def build_book_topic_mapping(ebooks_base: Path) -> Dict[str, str]:
    """Scan J:\\E-Books and build a mapping of book filenames to topics.

    Walks the directory tree (max 3 levels deep), identifies trading-relevant
    topic directories, and maps each book file to its topic.

    Args:
        ebooks_base: Base path to the e-books library.

    Returns:
        Dict mapping book_filename -> canonical_topic_label.
    """
    t0 = time.time()

    if not ebooks_base.exists():
        logger.error("E-Books directory not found: %s", ebooks_base)
        return {}

    mapping: Dict[str, str] = {}
    scanned_dirs = 0
    skipped_dirs = 0

    for root, dirs, files in os.walk(ebooks_base):
        depth = len(Path(root).relative_to(ebooks_base).parts)
        if depth > 3:
            dirs.clear()  # Stop deeper recursion
            continue

        scanned_dirs += 1

        # Get all parent directory names for topic matching
        rel_parts = Path(root).relative_to(ebooks_base).parts
        topic = None

        # Try each directory level for a trading topic match
        for part in reversed(rel_parts):
            part_lower = part.lower().strip()
            if part_lower in TRADING_TOPICS:
                topic = TRADING_TOPICS[part_lower]
                break

        if topic is None:
            skipped_dirs += 1
            continue

        # Map all book files in this directory
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in BOOK_EXTENSIONS:
                mapping[fname] = topic

    # Save mapping
    BOOK_TOPIC_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BOOK_TOPIC_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(
        "Step 1 COMPLETE in %.1fs: %d books mapped to topics "
        "(%d dirs scanned, %d non-trading dirs skipped)",
        elapsed, len(mapping), scanned_dirs, skipped_dirs,
    )

    # Topic distribution
    topic_counts: Counter = Counter(mapping.values())
    logger.info("Topic distribution from J:\\E-Books:")
    for topic, count in topic_counts.most_common():
        logger.info("  %-30s %4d books", topic, count)

    return mapping


# ============================================================
# STEP 2: PARSE IMAGE FILENAMES + FUZZY MATCH
# ============================================================

def normalize_book_title(title: str) -> str:
    """Normalize a book title for fuzzy matching.

    Removes numeric IDs, format prefixes, special characters,
    and converts to lowercase.

    Args:
        title: Raw book title extracted from filename.

    Returns:
        Normalized title string.
    """
    # Remove leading numeric IDs (e.g., "551414104-")
    title = re.sub(r"^\d{5,}-", "", title)
    # Remove trailing format markers
    title = re.sub(r"-PDFDrive$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"-PDFConverted-\d+$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(Z-Library\)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(\s*PDFDrive\s*\)$", "", title, flags=re.IGNORECASE)
    # Replace hyphens/underscores with spaces
    title = title.replace("-", " ").replace("_", " ")
    # Remove extra whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title.lower()


def extract_book_title_from_filename(filename: str) -> Optional[str]:
    """Extract the book title from an image filename.

    Parses the naming convention: {FORMAT}_{BookTitle}_{page}_{imageIndex}.png

    Args:
        filename: Image filename (without directory).

    Returns:
        Extracted book title, or None if not parseable.
    """
    # Strip format prefix
    title = filename
    for prefix in FORMAT_PREFIXES:
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    else:
        return None  # No known prefix

    # Remove image index suffix: _iNN.ext or _pNNN_iNN.ext or _pNNN_imgNN.ext
    title = re.sub(r"_p\d+_i(?:mg)?\d+\.\w+$", "", title)
    title = re.sub(r"_i\d+\.\w+$", "", title)
    # Remove file extension if still present
    title = re.sub(r"\.\w+$", "", title)

    if not title:
        return None

    return title


def fuzzy_match_book(
    image_title: str,
    book_titles: List[str],
    book_titles_normalized: List[str],
    threshold: float = FUZZY_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """Fuzzy-match an image's book title against the book library.

    Uses difflib.SequenceMatcher for string similarity.

    Args:
        image_title: Normalized book title from image filename.
        book_titles: Original book filenames from the library.
        book_titles_normalized: Pre-normalized book titles for matching.
        threshold: Minimum similarity score to accept a match.

    Returns:
        Tuple of (matched_book_filename, similarity_score) or None.
    """
    best_score = 0.0
    best_match = None

    image_norm = normalize_book_title(image_title)
    if not image_norm or len(image_norm) < 5:
        return None

    for orig, norm in zip(book_titles, book_titles_normalized):
        # Quick reject: if length difference is too large
        len_ratio = len(image_norm) / max(len(norm), 1)
        if len_ratio < 0.15 or len_ratio > 6.0:
            continue

        score = difflib.SequenceMatcher(None, image_norm, norm).ratio()

        # Boost: if one is a substring of the other (common with truncated titles)
        if image_norm in norm or norm in image_norm:
            shorter = min(len(image_norm), len(norm))
            longer = max(len(image_norm), len(norm))
            containment = shorter / longer
            # Weighted: sequence similarity + containment bonus
            score = max(score, 0.5 + containment * 0.5)

        if score > best_score:
            best_score = score
            best_match = orig

    if best_score >= threshold and best_match is not None:
        return (best_match, best_score)
    return None


def build_image_book_mapping(
    manifest_path: Path,
    book_topic_mapping: Dict[str, str],
    resume: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Parse image filenames and fuzzy-match to book library.

    Args:
        manifest_path: Path to the chart dataset manifest CSV.
        book_topic_mapping: Dict of book_filename -> topic.
        resume: Whether to resume from checkpoint.

    Returns:
        Dict mapping image_path -> {"book_title": str, "book_file": str, "topic": str}.
    """
    t0 = time.time()

    # Optional tqdm
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Resume support
    if resume and IMAGE_BOOK_MAPPING_FILE.exists():
        with open(IMAGE_BOOK_MAPPING_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info("Resumed: %d images already mapped", len(existing))
        return existing

    df = pd.read_csv(manifest_path)
    logger.info("Manifest: %d images", len(df))

    # Pre-compute normalized book titles for fuzzy matching
    book_filenames = list(book_topic_mapping.keys())
    book_normalized = [
        normalize_book_title(Path(bf).stem) for bf in book_filenames
    ]
    logger.info("Book library: %d books for fuzzy matching", len(book_filenames))

    # Cache: raw book title -> (matched_file, topic) to avoid re-matching
    title_cache: Dict[str, Optional[Tuple[str, str]]] = {}

    image_mapping: Dict[str, Dict[str, str]] = {}
    match_count = 0
    no_match_count = 0
    no_title_count = 0

    rows = list(df.iterrows())
    iterator = tqdm(rows, desc="Matching books", unit="img",
                    mininterval=2.0, ncols=100) if use_tqdm else rows

    for idx, row in iterator:
        img_path = row["image_path"]
        fname = Path(img_path).name

        # Extract book title from filename
        raw_title = extract_book_title_from_filename(fname)
        if raw_title is None:
            no_title_count += 1
            continue

        # Check cache first
        if raw_title in title_cache:
            cached = title_cache[raw_title]
            if cached is not None:
                image_mapping[img_path] = {
                    "book_title": raw_title,
                    "book_file": cached[0],
                    "topic": cached[1],
                }
                match_count += 1
            else:
                image_mapping[img_path] = {
                    "book_title": raw_title,
                    "book_file": "",
                    "topic": "book_unclassified",
                }
                no_match_count += 1
            continue

        # Fuzzy match
        result = fuzzy_match_book(
            raw_title, book_filenames, book_normalized, FUZZY_THRESHOLD,
        )

        if result is not None:
            matched_file, score = result
            topic = book_topic_mapping[matched_file]
            title_cache[raw_title] = (matched_file, topic)
            image_mapping[img_path] = {
                "book_title": raw_title,
                "book_file": matched_file,
                "topic": topic,
            }
            match_count += 1
        else:
            title_cache[raw_title] = None
            image_mapping[img_path] = {
                "book_title": raw_title,
                "book_file": "",
                "topic": "book_unclassified",
            }
            no_match_count += 1

    # Save mapping
    IMAGE_BOOK_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_BOOK_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(image_mapping, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(
        "Step 2 COMPLETE in %.1fs: %d matched, %d unmatched, %d no title "
        "(%d unique titles cached)",
        elapsed, match_count, no_match_count, no_title_count,
        len(title_cache),
    )

    return image_mapping


# ============================================================
# STEP 3: APPLY LABELS + SUBSAMPLE MEGA-BOOK
# ============================================================

def apply_labels(
    manifest_path: Path,
    image_mapping: Dict[str, Dict[str, str]],
    max_per_book: int = DEFAULT_MAX_PER_BOOK,
    dry_run: bool = False,
) -> LabelingStats:
    """Apply book-based labels to manifest and subsample mega-books.

    Args:
        manifest_path: Path to the manifest CSV.
        image_mapping: Dict of image_path -> {book_title, book_file, topic}.
        max_per_book: Maximum images to keep per single book title.
        dry_run: If True, compute stats but don't write.

    Returns:
        LabelingStats with results.
    """
    t0 = time.time()
    stats = LabelingStats()

    df = pd.read_csv(manifest_path)
    stats.total_images = len(df)

    # Count images per raw book title for subsampling
    book_image_counts: Counter = Counter()
    for img_path, info in image_mapping.items():
        book_image_counts[info["book_title"]] += 1

    # Log mega-books
    mega_books = {
        title: count for title, count in book_image_counts.items()
        if count > max_per_book
    }
    if mega_books:
        logger.info("Mega-books to subsample (max %d per book):", max_per_book)
        for title, count in sorted(mega_books.items(), key=lambda x: -x[1]):
            display = title[:70] + "..." if len(title) > 70 else title
            logger.info("  %6d -> %d | %s", count, max_per_book, display)

    # Build subsampling sets: for each mega-book, select random indices to KEEP
    rng = np.random.RandomState(42)
    book_keep_sets: Dict[str, Set[str]] = {}
    for title, count in mega_books.items():
        # Collect all image paths for this book
        book_paths = [
            p for p, info in image_mapping.items()
            if info["book_title"] == title
        ]
        keep_indices = rng.choice(len(book_paths), max_per_book, replace=False)
        book_keep_sets[title] = {book_paths[i] for i in keep_indices}

    # Apply labels
    book_titles_col: List[str] = []
    book_topics_col: List[str] = []
    new_labels: List[str] = []
    keep_mask: List[bool] = []

    unique_matched_books: Set[str] = set()
    unique_unmatched_books: Set[str] = set()

    for _, row in df.iterrows():
        img_path = row["image_path"]

        if img_path in image_mapping:
            info = image_mapping[img_path]
            book_title = info["book_title"]
            topic = info["topic"]
            book_file = info["book_file"]
            stats.total_traceable += 1

            book_titles_col.append(book_title)
            book_topics_col.append(topic)

            if book_file:
                unique_matched_books.add(book_title)
            else:
                unique_unmatched_books.add(book_title)

            # Check subsampling
            if book_title in book_keep_sets:
                if img_path not in book_keep_sets[book_title]:
                    keep_mask.append(False)
                    stats.images_subsampled_out += 1
                    new_labels.append(row["label"])  # Won't matter, row dropped
                    continue

            keep_mask.append(True)

            # Apply topic label (only if we have a real topic, not unclassified)
            if topic != "book_unclassified":
                new_labels.append(topic)
                stats.images_labeled += 1
                stats.topic_distribution[topic] = (
                    stats.topic_distribution.get(topic, 0) + 1
                )
            else:
                new_labels.append(row["label"])  # Preserve existing
                stats.images_unclassified += 1
        else:
            # Anonymous / kit_multimodal
            stats.total_anonymous += 1
            book_titles_col.append("")
            book_topics_col.append("")
            new_labels.append(row["label"])  # Preserve existing
            keep_mask.append(True)

    stats.books_matched = len(unique_matched_books)
    stats.books_unmatched = len(unique_unmatched_books)

    # Update DataFrame
    df["book_title"] = book_titles_col
    df["book_topic"] = book_topics_col
    df["label"] = new_labels

    # Apply subsampling filter
    df_filtered = df[keep_mask].reset_index(drop=True)
    logger.info(
        "Subsampling: %d -> %d images (%d removed)",
        len(df), len(df_filtered), stats.images_subsampled_out,
    )

    stats.elapsed_sec = time.time() - t0

    if dry_run:
        logger.info("DRY RUN -- no files written")
    else:
        # Save updated manifest
        df_filtered.to_csv(manifest_path, index=False)
        logger.info("Manifest updated: %s (%d rows)", manifest_path, len(df_filtered))

    return stats


# ============================================================
# STEP 4: GENERATE REPORT
# ============================================================

def generate_report(stats: LabelingStats) -> None:
    """Generate markdown report with labeling statistics.

    Args:
        stats: Final labeling statistics.
    """
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Book-Based Auto-Labeling Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Source:** `J:\\E-Books` directory structure",
        "",
        "## Method",
        "",
        "Images extracted from trading books carry the book title in their filename.",
        "By matching book titles to their topic directory in J:\\E-Books, we assign",
        "topic-level labels to all images from that book.",
        "",
        "## Results Summary",
        "",
        "| Metric | Count |",
        "|---|---|",
        f"| Total images | {stats.total_images:,} |",
        f"| Traceable to a book | {stats.total_traceable:,} |",
        f"| Anonymous (no book ref) | {stats.total_anonymous:,} |",
        f"| Books matched to topic | {stats.books_matched:,} |",
        f"| Books unmatched | {stats.books_unmatched:,} |",
        f"| Images labeled | {stats.images_labeled:,} |",
        f"| Images unclassified | {stats.images_unclassified:,} |",
        f"| Images subsampled out | {stats.images_subsampled_out:,} |",
        "",
        "## Topic Distribution (labeled images)",
        "",
        "| Topic | Count |",
        "|---|---|",
    ]

    for topic, count in sorted(
        stats.topic_distribution.items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {topic} | {count:,} |")

    lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review topic assignments for accuracy",
        "2. Run clustering on remaining unlabeled/anonymous images",
        "3. Train vision model with updated labels",
        "",
    ])

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Report saved: %s", REPORT_OUTPUT)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label chart images from book->topic directory mapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Steps:
    1: Build book->topic mapping from J:\\E-Books
    2: Parse image filenames, fuzzy-match to books
    3: Apply labels + subsample mega-books
    4: Generate report

Examples:
    python scripts/label_charts_from_books.py
    python scripts/label_charts_from_books.py --step 1
    python scripts/label_charts_from_books.py --step 3 --max-per-book 1000
    python scripts/label_charts_from_books.py --dry-run
""",
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a single step (1-4). If omitted, runs all steps.",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help="Path to chart dataset manifest CSV",
    )
    parser.add_argument(
        "--ebooks-dir", type=str, default=str(EBOOKS_BASE),
        help="Path to e-books library (default: J:\\E-Books)",
    )
    parser.add_argument(
        "--max-per-book", type=int, default=DEFAULT_MAX_PER_BOOK,
        help=f"Max images per book after subsampling (default: {DEFAULT_MAX_PER_BOOK})",
    )
    parser.add_argument(
        "--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD,
        help=f"Fuzzy match threshold (default: {FUZZY_THRESHOLD})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved intermediate files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute stats but don't update manifest",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging(logging.INFO)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BOOK-BASED CHART LABELING PIPELINE")
    logger.info("=" * 60)
    logger.info("Manifest:        %s", args.manifest)
    logger.info("E-Books dir:     %s", args.ebooks_dir)
    logger.info("Max per book:    %d", args.max_per_book)
    logger.info("Fuzzy threshold: %.2f", args.fuzzy_threshold)
    logger.info("Dry run:         %s", args.dry_run)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    # Determine steps
    if args.step is not None:
        steps = [args.step]
    else:
        steps = [1, 2, 3, 4]

    t_total = time.time()

    book_topic_mapping: Dict[str, str] = {}
    image_mapping: Dict[str, Dict[str, str]] = {}

    for step in steps:
        logger.info("-" * 40)
        logger.info("STEP %d", step)
        logger.info("-" * 40)

        if step == 1:
            book_topic_mapping = build_book_topic_mapping(
                Path(args.ebooks_dir),
            )

        elif step == 2:
            # Load book_topic_mapping if not already in memory
            if not book_topic_mapping:
                if BOOK_TOPIC_MAPPING_FILE.exists():
                    with open(BOOK_TOPIC_MAPPING_FILE, "r", encoding="utf-8") as f:
                        book_topic_mapping = json.load(f)
                    logger.info("Loaded book_topic_mapping: %d books",
                                len(book_topic_mapping))
                else:
                    logger.error("Run --step 1 first to build book_topic_mapping")
                    sys.exit(1)

            image_mapping = build_image_book_mapping(
                manifest_path=manifest_path,
                book_topic_mapping=book_topic_mapping,
                resume=args.resume,
            )

        elif step == 3:
            # Load image_mapping if not already in memory
            if not image_mapping:
                if IMAGE_BOOK_MAPPING_FILE.exists():
                    with open(IMAGE_BOOK_MAPPING_FILE, "r", encoding="utf-8") as f:
                        image_mapping = json.load(f)
                    logger.info("Loaded image_book_mapping: %d images",
                                len(image_mapping))
                else:
                    logger.error("Run --step 2 first to build image_book_mapping")
                    sys.exit(1)

            stats = apply_labels(
                manifest_path=manifest_path,
                image_mapping=image_mapping,
                max_per_book=args.max_per_book,
                dry_run=args.dry_run,
            )

            summary = stats.summary()
            logger.info("\n%s", summary)
            print(summary)

        elif step == 4:
            # Generate report (uses stats from step 3, or recompute)
            # Quick recompute from manifest
            if manifest_path.exists():
                df = pd.read_csv(manifest_path)
                stats = LabelingStats(total_images=len(df))

                if "book_topic" in df.columns:
                    has_topic = df["book_topic"].notna() & (df["book_topic"] != "")
                    stats.total_traceable = has_topic.sum()
                    stats.total_anonymous = (~has_topic).sum()

                    topic_dist = df.loc[
                        has_topic & (df["book_topic"] != "book_unclassified"),
                        "book_topic"
                    ].value_counts()
                    stats.topic_distribution = dict(topic_dist)
                    stats.images_labeled = int(topic_dist.sum())
                    unclass = (df["book_topic"] == "book_unclassified").sum()
                    stats.images_unclassified = int(unclass)

                generate_report(stats)
            else:
                logger.error("Manifest not found, cannot generate report")

        gc.collect()

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs (steps: %s)", total_elapsed, steps)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
