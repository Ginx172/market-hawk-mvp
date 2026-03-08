"""
SCRIPT NAME: prepare_chart_dataset.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Scan and catalog chart images from G: drive for vision model training.
Hardware Optimization: Intel i7-9700F, 64GB DDR4
Creation Date: 2026-03-08

Scans two image sources on G: drive:
    - G:\\.....DATABASE\\dataset\\charts (~80,567 images)
    - G:\\_kit_multimodal (subdirectories with chart images ~15,622)

Generates a manifest CSV with metadata for each valid image:
    image_path, label, category, width, height, file_size, source_dir

Features:
    - Pattern classification from filenames/directory structure
    - Corrupt image detection (PIL validation)
    - Size filtering (skip < 50x50 px, skip > 50MB)
    - Checkpoint/resume support
    - Progress bar with ETA
    - Does NOT copy images -- only catalogs paths

Usage:
    # Full scan with defaults
    python scripts/prepare_chart_dataset.py

    # Custom output path
    python scripts/prepare_chart_dataset.py --output data/my_manifest.csv

    # Resume from checkpoint
    python scripts/prepare_chart_dataset.py --resume

    # Dry run (scan but don't write CSV)
    python scripts/prepare_chart_dataset.py --dry-run
"""

import argparse
import csv
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, LOGS_DIR, setup_logging

logger = logging.getLogger("market_hawk.prepare_chart_dataset")

# ============================================================
# CONSTANTS
# ============================================================

# Image sources on G: drive
IMAGE_SOURCES: Dict[str, str] = {
    "charts_db": r"G:\.....DATABASE\dataset\charts",
    "kit_multimodal": r"G:\_kit_multimodal",
}

# Supported image extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}

# Size constraints
MIN_DIMENSION_PX: int = 50
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Pattern classification keywords (lowercase)
PATTERN_KEYWORDS: Dict[str, List[str]] = {
    "head_and_shoulders": [
        "head_and_shoulders", "headandshoulders", "h&s", "head_shoulder",
        "hs_pattern", "inverse_head",
    ],
    "double_top": [
        "double_top", "doubletop", "double-top", "dbl_top",
    ],
    "double_bottom": [
        "double_bottom", "doublebottom", "double-bottom", "dbl_bottom",
    ],
    "triangle": [
        "triangle", "ascending_triangle", "descending_triangle",
        "symmetrical_triangle", "triunghi",
    ],
    "channel": [
        "channel", "ascending_channel", "descending_channel",
        "price_channel", "canal",
    ],
    "flag": [
        "flag", "bull_flag", "bear_flag", "pennant",
    ],
    "wedge": [
        "wedge", "rising_wedge", "falling_wedge",
    ],
    "cup_and_handle": [
        "cup_and_handle", "cupandhandle", "cup_handle",
    ],
    "support_resistance": [
        "support", "resistance", "s_r", "sr_level",
    ],
    "trend": [
        "uptrend", "downtrend", "trend", "trendline",
    ],
    "candlestick": [
        "candlestick", "doji", "hammer", "engulfing", "marubozu",
        "spinning_top", "morning_star", "evening_star", "harami",
    ],
    "fibonacci": [
        "fibonacci", "fib_", "fibo", "retracement",
    ],
    "elliott_wave": [
        "elliott", "wave_count", "impulse_wave", "corrective_wave",
    ],
}

# Direction classification keywords (lowercase)
DIRECTION_KEYWORDS: Dict[str, List[str]] = {
    "bullish": ["bullish", "bull", "long", "buy", "up", "ascending", "rising"],
    "bearish": ["bearish", "bear", "short", "sell", "down", "descending", "falling"],
    "neutral": ["neutral", "sideways", "range", "consolidation", "flat"],
}

# Default output path
DEFAULT_OUTPUT = DATA_DIR / "chart_dataset_manifest.csv"
DEFAULT_CHECKPOINT = LOGS_DIR / "chart_dataset_checkpoint.json"

# CSV header
CSV_HEADER: List[str] = [
    "image_path", "label", "category", "direction",
    "width", "height", "file_size", "source_dir",
]


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ScanStats:
    """Tracks scanning statistics."""
    total_files_found: int = 0
    valid_images: int = 0
    skipped_corrupt: int = 0
    skipped_too_small: int = 0
    skipped_too_large: int = 0
    skipped_not_image: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    direction_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        """Format a human-readable summary."""
        lines = [
            "=" * 60,
            "CHART DATASET SCAN REPORT",
            "=" * 60,
            f"Total files found:      {self.total_files_found:>8,}",
            f"Valid images:           {self.valid_images:>8,}",
            f"Skipped (corrupt):      {self.skipped_corrupt:>8,}",
            f"Skipped (too small):    {self.skipped_too_small:>8,}",
            f"Skipped (too large):    {self.skipped_too_large:>8,}",
            f"Skipped (not image):    {self.skipped_not_image:>8,}",
            f"Elapsed time:           {self.elapsed_sec:>8.1f}s",
            "",
            "--- Label Distribution ---",
        ]
        for label, count in sorted(
            self.label_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {label:<30s} {count:>8,}")
        lines.append("")
        lines.append("--- Direction Distribution ---")
        for direction, count in sorted(
            self.direction_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {direction:<30s} {count:>8,}")
        lines.append("")
        lines.append("--- Source Distribution ---")
        for source, count in sorted(
            self.source_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {source:<30s} {count:>8,}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class ImageRecord:
    """Single image metadata record."""
    image_path: str
    label: str
    category: str
    direction: str
    width: int
    height: int
    file_size: int
    source_dir: str


# ============================================================
# CLASSIFICATION LOGIC
# ============================================================

def classify_pattern(file_path: Path) -> str:
    """Classify chart pattern from filename and parent directory names.

    Args:
        file_path: Path to the image file.

    Returns:
        Pattern label string, or "unlabeled" if no pattern detected.
    """
    # Combine filename + parent dir names for keyword matching
    # Normalize separators: underscores and hyphens -> spaces for word boundary matching
    raw = (
        file_path.stem.lower()
        + " "
        + file_path.parent.name.lower()
        + " "
        + (file_path.parent.parent.name.lower()
           if file_path.parent.parent else "")
    )
    search_text = raw.replace("_", " ").replace("-", " ")

    for pattern_name, keywords in PATTERN_KEYWORDS.items():
        for kw in keywords:
            # Normalize keyword separators too
            kw_normalized = kw.replace("_", " ").replace("-", " ")
            if kw_normalized in search_text:
                return pattern_name

    return "unlabeled"


def classify_direction(file_path: Path) -> str:
    """Classify bullish/bearish/neutral direction from filename and directory.

    Args:
        file_path: Path to the image file.

    Returns:
        Direction string: "bullish", "bearish", "neutral", or "unknown".
    """
    raw = (
        file_path.stem.lower()
        + " "
        + file_path.parent.name.lower()
    )
    search_text = raw.replace("_", " ").replace("-", " ")

    for direction, keywords in DIRECTION_KEYWORDS.items():
        for kw in keywords:
            # Match whole word boundaries to avoid false positives
            # e.g., "uptrend" should not match "up" in "setup"
            kw_normalized = kw.replace("_", " ").replace("-", " ")
            if re.search(r'\b' + re.escape(kw_normalized) + r'\b', search_text):
                return direction

    return "unknown"


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(file_path: Path) -> Optional[Tuple[int, int]]:
    """Validate that a file is a readable image and return dimensions.

    Args:
        file_path: Path to the image file.

    Returns:
        (width, height) tuple if valid, None if corrupt or unreadable.
    """
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        # Re-open after verify (verify can invalidate the file object)
        with Image.open(file_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    checkpoint_path: Path,
    processed_files: Set[str],
    records: List[ImageRecord],
    stats: ScanStats,
) -> None:
    """Save scan progress to checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file.
        processed_files: Set of already-processed file paths.
        records: List of valid image records so far.
        stats: Current scan statistics.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "processed_count": len(processed_files),
        "valid_count": len(records),
        "processed_files": list(processed_files),
        "stats": {
            "total_files_found": stats.total_files_found,
            "valid_images": stats.valid_images,
            "skipped_corrupt": stats.skipped_corrupt,
            "skipped_too_small": stats.skipped_too_small,
            "skipped_too_large": stats.skipped_too_large,
            "skipped_not_image": stats.skipped_not_image,
        },
        "records": [
            {
                "image_path": r.image_path,
                "label": r.label,
                "category": r.category,
                "direction": r.direction,
                "width": r.width,
                "height": r.height,
                "file_size": r.file_size,
                "source_dir": r.source_dir,
            }
            for r in records
        ],
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Checkpoint saved: %d processed, %d valid",
                len(processed_files), len(records))


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Set[str], List[ImageRecord], ScanStats]:
    """Load scan progress from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file.

    Returns:
        Tuple of (processed_files set, records list, stats).
    """
    if not checkpoint_path.exists():
        return set(), [], ScanStats()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_files = set(data.get("processed_files", []))
    stats_data = data.get("stats", {})
    stats = ScanStats(
        total_files_found=stats_data.get("total_files_found", 0),
        valid_images=stats_data.get("valid_images", 0),
        skipped_corrupt=stats_data.get("skipped_corrupt", 0),
        skipped_too_small=stats_data.get("skipped_too_small", 0),
        skipped_too_large=stats_data.get("skipped_too_large", 0),
        skipped_not_image=stats_data.get("skipped_not_image", 0),
    )
    records = [
        ImageRecord(**r) for r in data.get("records", [])
    ]

    logger.info("Checkpoint loaded: %d processed, %d valid (from %s)",
                len(processed_files), len(records),
                data.get("timestamp", "unknown"))
    return processed_files, records, stats


# ============================================================
# SCANNER
# ============================================================

def collect_image_paths(source_dirs: Dict[str, str]) -> List[Tuple[Path, str]]:
    """Walk source directories and collect all candidate image file paths.

    Args:
        source_dirs: Dict mapping source_name -> directory path.

    Returns:
        List of (file_path, source_name) tuples.
    """
    candidates: List[Tuple[Path, str]] = []

    for source_name, dir_path_str in source_dirs.items():
        dir_path = Path(dir_path_str)
        if not dir_path.exists():
            logger.warning("Source directory not found: %s (%s)",
                           dir_path, source_name)
            continue

        logger.info("Scanning source '%s': %s", source_name, dir_path)
        file_count = 0
        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    candidates.append((Path(root) / fname, source_name))
                    file_count += 1

        logger.info("  Found %d candidate image files in '%s'",
                     file_count, source_name)

    return candidates


def process_image(
    file_path: Path,
    source_name: str,
    stats: ScanStats,
) -> Optional[ImageRecord]:
    """Validate and classify a single image file.

    Args:
        file_path: Path to the image.
        source_name: Name of the source directory.
        stats: ScanStats to update in-place.

    Returns:
        ImageRecord if valid, None if skipped.
    """
    stats.total_files_found += 1

    # Check file size
    try:
        file_size = file_path.stat().st_size
    except OSError:
        stats.skipped_corrupt += 1
        return None

    if file_size > MAX_FILE_SIZE_BYTES:
        stats.skipped_too_large += 1
        return None

    # Validate image and get dimensions
    dims = validate_image(file_path)
    if dims is None:
        stats.skipped_corrupt += 1
        return None

    width, height = dims
    if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
        stats.skipped_too_small += 1
        return None

    # Classify
    label = classify_pattern(file_path)
    direction = classify_direction(file_path)

    # Category = label if labeled, else direction, else "other"
    if label != "unlabeled":
        category = label
    elif direction != "unknown":
        category = direction
    else:
        category = "unlabeled"

    # Update distribution stats
    stats.valid_images += 1
    stats.label_distribution[label] = (
        stats.label_distribution.get(label, 0) + 1
    )
    stats.direction_distribution[direction] = (
        stats.direction_distribution.get(direction, 0) + 1
    )
    stats.source_distribution[source_name] = (
        stats.source_distribution.get(source_name, 0) + 1
    )

    return ImageRecord(
        image_path=str(file_path),
        label=label,
        category=category,
        direction=direction,
        width=width,
        height=height,
        file_size=file_size,
        source_dir=source_name,
    )


def scan_and_catalog(
    source_dirs: Dict[str, str],
    output_path: Path,
    checkpoint_path: Path,
    resume: bool = False,
    dry_run: bool = False,
    checkpoint_interval: int = 1000,
) -> ScanStats:
    """Main scan pipeline: collect, validate, classify, write manifest CSV.

    Args:
        source_dirs: Dict mapping source_name -> directory path.
        output_path: Path for the output manifest CSV.
        checkpoint_path: Path for checkpoint file.
        resume: Whether to resume from checkpoint.
        dry_run: If True, scan but don't write final CSV.
        checkpoint_interval: Save checkpoint every N processed files.

    Returns:
        ScanStats with final statistics.
    """
    t0 = time.time()

    # Optional tqdm import
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        logger.info("tqdm not available, using basic progress logging")

    # Resume or fresh start
    if resume:
        processed_files, records, stats = load_checkpoint(checkpoint_path)
        logger.info("Resuming from checkpoint: %d files already processed",
                     len(processed_files))
    else:
        processed_files: Set[str] = set()
        records: List[ImageRecord] = []
        stats = ScanStats()

    # Collect all candidate paths
    logger.info("Collecting image paths from %d sources...", len(source_dirs))
    all_candidates = collect_image_paths(source_dirs)
    total_candidates = len(all_candidates)
    logger.info("Total candidates: %d", total_candidates)

    # Filter out already-processed
    pending = [
        (fp, src) for fp, src in all_candidates
        if str(fp) not in processed_files
    ]
    logger.info("Pending (not yet processed): %d", len(pending))

    if not pending:
        logger.info("Nothing to process. All files already scanned.")
        stats.elapsed_sec = time.time() - t0
        return stats

    # Process with progress tracking
    iterator = (
        tqdm(pending, desc="Scanning images", unit="img",
             mininterval=2.0, ncols=100)
        if use_tqdm
        else pending
    )

    processed_in_session = 0
    for file_path, source_name in iterator:
        record = process_image(file_path, source_name, stats)
        if record is not None:
            records.append(record)

        processed_files.add(str(file_path))
        processed_in_session += 1

        # Periodic checkpoint
        if processed_in_session % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, processed_files, records, stats)
            gc.collect()

        # Basic progress logging fallback
        if not use_tqdm and processed_in_session % 5000 == 0:
            logger.info(
                "Progress: %d/%d processed, %d valid",
                processed_in_session, len(pending), stats.valid_images,
            )

    stats.elapsed_sec = time.time() - t0

    # Save final checkpoint
    save_checkpoint(checkpoint_path, processed_files, records, stats)

    # Write manifest CSV
    if dry_run:
        logger.info("DRY RUN -- skipping CSV write")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for rec in records:
                writer.writerow([
                    rec.image_path,
                    rec.label,
                    rec.category,
                    rec.direction,
                    rec.width,
                    rec.height,
                    rec.file_size,
                    rec.source_dir,
                ])
        logger.info("Manifest CSV written: %s (%d rows)",
                     output_path, len(records))

    # Print summary
    summary = stats.summary()
    logger.info("\n%s", summary)
    print(summary)

    return stats


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan and catalog chart images for vision model training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Full scan
    python scripts/prepare_chart_dataset.py

    # Custom output
    python scripts/prepare_chart_dataset.py --output data/custom_manifest.csv

    # Resume interrupted scan
    python scripts/prepare_chart_dataset.py --resume

    # Dry run
    python scripts/prepare_chart_dataset.py --dry-run
""",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Output CSV path (default: data/chart_dataset_manifest.csv)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint file path (default: logs/chart_dataset_checkpoint.json)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report stats but don't write CSV",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1000,
        help="Save checkpoint every N files (default: 1000)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging(logging.INFO)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CHART DATASET PREPARATION")
    logger.info("=" * 60)
    logger.info("Output:     %s", args.output)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Resume:     %s", args.resume)
    logger.info("Dry run:    %s", args.dry_run)

    # Validate PIL availability
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        logger.error("Pillow is required: pip install Pillow")
        sys.exit(1)

    stats = scan_and_catalog(
        source_dirs=IMAGE_SOURCES,
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        resume=args.resume,
        dry_run=args.dry_run,
        checkpoint_interval=args.checkpoint_interval,
    )

    logger.info("DONE in %.1fs -- %d valid images cataloged",
                stats.elapsed_sec, stats.valid_images)


if __name__ == "__main__":
    main()
