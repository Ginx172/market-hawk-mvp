"""
SCRIPT NAME: auto_label_charts.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: CLIP-based zero-shot classification for unlabeled chart images.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-08

Uses OpenAI CLIP (ViT-B/32, ~600MB VRAM) to classify ~94K unlabeled chart
images into pattern categories via zero-shot text-image similarity.

Pipeline:
    1. Load CLIP model (ViT-B/32) on GPU
    2. Encode text prompts for all pattern categories (one-time)
    3. Process unlabeled images in batches (default 64)
    4. Compute cosine similarity between each image and all text prompts
    5. Assign label based on confidence margin:
       - HIGH (margin > 0.10): auto-label, trusted
       - MEDIUM (margin 0.05-0.10): auto-label, flagged for review
       - LOW (margin < 0.05): remains "unlabeled"
    6. Update manifest CSV with new columns
    7. Generate review queue CSV and labeling report

Dependencies:
    pip install open-clip-torch

Usage:
    # Full auto-labeling run
    python scripts/auto_label_charts.py

    # Custom confidence threshold
    python scripts/auto_label_charts.py --high-threshold 0.12 --low-threshold 0.06

    # Custom batch size (reduce if OOM)
    python scripts/auto_label_charts.py --batch-size 32

    # Resume from checkpoint
    python scripts/auto_label_charts.py --resume

    # Dry run (process but don't update manifest)
    python scripts/auto_label_charts.py --dry-run
"""

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, LOGS_DIR, MODELS_DIR, setup_logging

logger = logging.getLogger("market_hawk.auto_label_charts")

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MANIFEST = DATA_DIR / "chart_dataset_manifest.csv"
DEFAULT_CHECKPOINT = LOGS_DIR / "auto_label_checkpoint.json"
REVIEW_QUEUE_OUTPUT = DATA_DIR / "review_queue.csv"
REPORT_OUTPUT = Path(_PROJECT_ROOT) / "docs" / "AUTO_LABELING_REPORT.md"

# CLIP model config -- ViT-B/32 fits comfortably in 8GB VRAM (~600MB)
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"

# Text prompts mapped to pattern labels
# Each prompt describes a chart pattern as CLIP would understand it
PATTERN_PROMPTS: Dict[str, str] = {
    "head_and_shoulders": "a stock chart showing head and shoulders pattern",
    "double_top": "a stock chart showing double top pattern",
    "double_bottom": "a stock chart showing double bottom pattern",
    "ascending_triangle": "a stock chart showing ascending triangle pattern",
    "descending_triangle": "a stock chart showing descending triangle pattern",
    "bullish_flag": "a stock chart showing bullish flag pattern",
    "bearish_flag": "a stock chart showing bearish flag pattern",
    "cup_and_handle": "a stock chart showing cup and handle pattern",
    "wedge": "a stock chart showing wedge pattern",
    "channel": "a stock chart showing channel pattern",
    "support_resistance": "a stock chart showing support and resistance levels",
    "fibonacci": "a stock chart showing fibonacci retracement",
    "candlestick": "a stock chart showing candlestick pattern",
    "trend": "a stock chart showing trend line analysis",
    "elliott_wave": "a stock chart showing elliott wave pattern",
    "generic_chart": "a generic stock price chart with no specific pattern",
}

# Confidence thresholds
DEFAULT_HIGH_THRESHOLD: float = 0.10   # margin > this = trusted auto-label
DEFAULT_LOW_THRESHOLD: float = 0.05    # margin < this = remains unlabeled

# Batch processing
DEFAULT_BATCH_SIZE: int = 64
CHECKPOINT_INTERVAL: int = 1000


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LabelResult:
    """Classification result for a single image."""
    image_path: str
    clip_label: str
    clip_confidence: float   # max similarity score
    clip_margin: float       # max - second_max
    review_status: str       # "trusted", "needs_review", "unlabeled"


@dataclass
class LabelingStats:
    """Aggregate statistics for the labeling run."""
    total_processed: int = 0
    total_skipped_already_labeled: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_margin: float = 0.0
    elapsed_sec: float = 0.0

    @property
    def auto_labeled(self) -> int:
        """Total images that received a label (high + medium)."""
        return self.high_confidence + self.medium_confidence

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "CLIP AUTO-LABELING REPORT",
            "=" * 60,
            f"Total processed:        {self.total_processed:>8,}",
            f"Skipped (pre-labeled):  {self.total_skipped_already_labeled:>8,}",
            f"HIGH confidence:        {self.high_confidence:>8,}  (trusted)",
            f"MEDIUM confidence:      {self.medium_confidence:>8,}  (needs review)",
            f"LOW confidence:         {self.low_confidence:>8,}  (unlabeled)",
            f"Total auto-labeled:     {self.auto_labeled:>8,}",
            f"Avg confidence:         {self.avg_confidence:>8.4f}",
            f"Avg margin:             {self.avg_margin:>8.4f}",
            f"Elapsed time:           {self.elapsed_sec:>8.1f}s",
            "",
            "--- Label Distribution (auto-labeled) ---",
        ]
        for label, count in sorted(
            self.label_distribution.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {label:<30s} {count:>8,}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# CLIP MODEL LOADER
# ============================================================

def load_clip_model(
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    """Load CLIP model and preprocessing transform.

    Args:
        device: Torch device (cuda or cpu).

    Returns:
        Tuple of (model, preprocess, tokenizer).
    """
    import open_clip

    logger.info("Loading CLIP model: %s (pretrained=%s)",
                CLIP_MODEL_NAME, CLIP_PRETRAINED)

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device,
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

    model.eval()

    # Log VRAM usage
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info("CLIP model loaded on GPU: %.0f MB VRAM", vram_mb)

    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Dict[str, str],
    device: torch.device,
) -> Tuple[torch.Tensor, List[str]]:
    """Encode all text prompts into CLIP embeddings.

    Args:
        model: CLIP model.
        tokenizer: CLIP tokenizer.
        prompts: Dict mapping label -> text prompt.
        device: Torch device.

    Returns:
        Tuple of (text_features tensor [N, dim], ordered label names list).
    """
    labels = list(prompts.keys())
    texts = [prompts[lbl] for lbl in labels]

    tokens = tokenizer(texts).to(device)
    text_features = model.encode_text(tokens)
    # L2 normalize for cosine similarity
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logger.info("Encoded %d text prompts (dim=%d)", len(labels), text_features.shape[1])
    return text_features, labels


# ============================================================
# IMAGE BATCH PROCESSING
# ============================================================

@torch.no_grad()
def classify_image_batch(
    model: Any,
    preprocess: Any,
    image_paths: List[str],
    text_features: torch.Tensor,
    label_names: List[str],
    device: torch.device,
    high_threshold: float,
    low_threshold: float,
) -> List[LabelResult]:
    """Classify a batch of images using CLIP zero-shot.

    Args:
        model: CLIP model.
        preprocess: CLIP image preprocessing transform.
        image_paths: List of image file paths.
        text_features: Pre-encoded text prompt features [N_prompts, dim].
        label_names: Ordered list of label names matching text_features.
        device: Torch device.
        high_threshold: Margin threshold for HIGH confidence.
        low_threshold: Margin threshold for LOW confidence.

    Returns:
        List of LabelResult for each image.
    """
    from PIL import Image

    results: List[LabelResult] = []
    valid_images: List[torch.Tensor] = []
    valid_indices: List[int] = []

    # Load and preprocess images, skip failures
    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img)
            valid_images.append(img_tensor)
            valid_indices.append(i)
        except Exception as e:
            logger.debug("Failed to load image %s: %s", img_path, e)
            results.append(LabelResult(
                image_path=img_path,
                clip_label="unlabeled",
                clip_confidence=0.0,
                clip_margin=0.0,
                review_status="error",
            ))

    if not valid_images:
        # Fill remaining with error results
        for i in range(len(image_paths)):
            if i not in valid_indices:
                pass  # Already added above
        return results

    # Stack into batch tensor
    batch_tensor = torch.stack(valid_images).to(device)

    # Encode images
    image_features = model.encode_image(batch_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity: [batch, n_prompts]
    similarity = (image_features @ text_features.T).cpu().numpy()

    # Process each image result
    result_map: Dict[int, LabelResult] = {}
    for batch_idx, orig_idx in enumerate(valid_indices):
        scores = similarity[batch_idx]

        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        best_score = float(scores[best_idx])
        second_score = float(scores[second_idx])
        margin = best_score - second_score
        best_label = label_names[best_idx]

        # Determine confidence tier
        if margin >= high_threshold:
            review_status = "trusted"
            assigned_label = best_label
        elif margin >= low_threshold:
            review_status = "needs_review"
            assigned_label = best_label
        else:
            review_status = "unlabeled"
            assigned_label = "unlabeled"

        result_map[orig_idx] = LabelResult(
            image_path=image_paths[orig_idx],
            clip_label=assigned_label,
            clip_confidence=best_score,
            clip_margin=margin,
            review_status=review_status,
        )

    # Build final ordered results
    final_results: List[LabelResult] = []
    for i in range(len(image_paths)):
        if i in result_map:
            final_results.append(result_map[i])
        else:
            # Already added as error above, find it
            for r in results:
                if r.image_path == image_paths[i]:
                    final_results.append(r)
                    break

    # Free GPU memory
    del batch_tensor, image_features, similarity
    torch.cuda.empty_cache()

    return final_results


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(
    checkpoint_path: Path,
    processed_paths: Set[str],
    all_results: List[LabelResult],
    stats: LabelingStats,
) -> None:
    """Save labeling progress to checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file.
        processed_paths: Set of processed image paths.
        all_results: All label results so far.
        stats: Current labeling statistics.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "processed_count": len(processed_paths),
        "results_count": len(all_results),
        "processed_paths": list(processed_paths),
        "stats": {
            "total_processed": stats.total_processed,
            "total_skipped_already_labeled": stats.total_skipped_already_labeled,
            "high_confidence": stats.high_confidence,
            "medium_confidence": stats.medium_confidence,
            "low_confidence": stats.low_confidence,
        },
        "results": [
            {
                "image_path": r.image_path,
                "clip_label": r.clip_label,
                "clip_confidence": r.clip_confidence,
                "clip_margin": r.clip_margin,
                "review_status": r.review_status,
            }
            for r in all_results
        ],
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=_json_default)
    logger.info("Checkpoint saved: %d processed, %d results",
                len(processed_paths), len(all_results))


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Set[str], List[LabelResult], LabelingStats]:
    """Load labeling progress from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint JSON file.

    Returns:
        Tuple of (processed_paths set, results list, stats).
    """
    if not checkpoint_path.exists():
        return set(), [], LabelingStats()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_paths = set(data.get("processed_paths", []))
    stats_data = data.get("stats", {})
    stats = LabelingStats(
        total_processed=stats_data.get("total_processed", 0),
        total_skipped_already_labeled=stats_data.get("total_skipped_already_labeled", 0),
        high_confidence=stats_data.get("high_confidence", 0),
        medium_confidence=stats_data.get("medium_confidence", 0),
        low_confidence=stats_data.get("low_confidence", 0),
    )
    results = [
        LabelResult(**r) for r in data.get("results", [])
    ]

    logger.info("Checkpoint loaded: %d processed, %d results (from %s)",
                len(processed_paths), len(results),
                data.get("timestamp", "unknown"))
    return processed_paths, results, stats


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    stats: LabelingStats,
    report_path: Path,
    manifest_path: Path,
    high_threshold: float,
    low_threshold: float,
) -> None:
    """Generate markdown report with labeling statistics.

    Args:
        stats: Final labeling statistics.
        report_path: Output path for the markdown report.
        manifest_path: Path to the updated manifest CSV.
        high_threshold: HIGH confidence margin threshold used.
        low_threshold: LOW confidence margin threshold used.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = stats.total_processed
    pct_high = (stats.high_confidence / max(total, 1)) * 100
    pct_med = (stats.medium_confidence / max(total, 1)) * 100
    pct_low = (stats.low_confidence / max(total, 1)) * 100

    lines = [
        "# Auto-Labeling Report -- CLIP Zero-Shot Classification",
        "",
        f"**Generated:** {timestamp}",
        f"**Model:** CLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})",
        f"**Manifest:** `{manifest_path}`",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| HIGH confidence threshold | margin > {high_threshold} |",
        f"| MEDIUM confidence threshold | margin {low_threshold} - {high_threshold} |",
        f"| LOW confidence threshold | margin < {low_threshold} |",
        f"| Text prompts | {len(PATTERN_PROMPTS)} categories |",
        "",
        "## Results Summary",
        "",
        f"| Metric | Count | % |",
        f"|---|---|---|",
        f"| Total processed | {total:,} | 100% |",
        f"| Pre-labeled (skipped) | {stats.total_skipped_already_labeled:,} | - |",
        f"| HIGH confidence (trusted) | {stats.high_confidence:,} | {pct_high:.1f}% |",
        f"| MEDIUM confidence (needs review) | {stats.medium_confidence:,} | {pct_med:.1f}% |",
        f"| LOW confidence (unlabeled) | {stats.low_confidence:,} | {pct_low:.1f}% |",
        f"| Total auto-labeled | {stats.auto_labeled:,} | {(pct_high + pct_med):.1f}% |",
        "",
        f"**Average confidence score:** {stats.avg_confidence:.4f}",
        f"**Average margin:** {stats.avg_margin:.4f}",
        f"**Elapsed time:** {stats.elapsed_sec:.1f}s",
        "",
        "## Label Distribution (auto-labeled only)",
        "",
        "| Pattern | Count |",
        "|---|---|",
    ]

    for label, count in sorted(
        stats.label_distribution.items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {label} | {count:,} |")

    lines.extend([
        "",
        "## Text Prompts Used",
        "",
        "| Label | Prompt |",
        "|---|---|",
    ])
    for label, prompt in PATTERN_PROMPTS.items():
        lines.append(f"| {label} | {prompt} |")

    lines.extend([
        "",
        "## Next Steps",
        "",
        "1. Review images in `data/review_queue.csv` (MEDIUM confidence)",
        "2. Manually verify a sample from HIGH confidence labels",
        "3. Retrain vision model with updated labels",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Report saved: %s", report_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_auto_labeling(
    manifest_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    high_threshold: float,
    low_threshold: float,
    resume: bool = False,
    dry_run: bool = False,
) -> LabelingStats:
    """Main auto-labeling pipeline.

    Args:
        manifest_path: Path to the chart dataset manifest CSV.
        checkpoint_path: Path to checkpoint file.
        batch_size: Number of images per batch.
        high_threshold: Margin threshold for HIGH confidence.
        low_threshold: Margin threshold for LOW confidence.
        resume: Whether to resume from checkpoint.
        dry_run: If True, process but don't update manifest.

    Returns:
        LabelingStats with final statistics.
    """
    t0 = time.time()

    # Optional tqdm
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Load manifest
    df = pd.read_csv(manifest_path)
    logger.info("Manifest loaded: %d rows", len(df))

    # Identify unlabeled images (skip already-labeled from Phase 1)
    unlabeled_mask = df["label"] == "unlabeled"
    unlabeled_paths = df.loc[unlabeled_mask, "image_path"].tolist()
    already_labeled_count = (~unlabeled_mask).sum()
    logger.info("Unlabeled images: %d (already labeled: %d)",
                len(unlabeled_paths), already_labeled_count)

    # Resume or fresh start
    if resume:
        processed_paths, all_results, stats = load_checkpoint(checkpoint_path)
        stats.total_skipped_already_labeled = already_labeled_count
    else:
        processed_paths: Set[str] = set()
        all_results: List[LabelResult] = []
        stats = LabelingStats(total_skipped_already_labeled=already_labeled_count)

    # Filter out already-processed
    pending = [p for p in unlabeled_paths if p not in processed_paths]
    logger.info("Pending: %d images", len(pending))

    if not pending:
        logger.info("Nothing to process.")
        stats.elapsed_sec = time.time() - t0
        return stats

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (%.1f GB VRAM)",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_mem / 1e9)

    # Load CLIP
    model, preprocess, tokenizer = load_clip_model(device)

    # Encode text prompts (one-time)
    text_features, label_names = encode_text_prompts(
        model, tokenizer, PATTERN_PROMPTS, device,
    )

    # Process in batches
    total_batches = (len(pending) + batch_size - 1) // batch_size
    confidence_sum = 0.0
    margin_sum = 0.0

    # Carry forward existing sums from checkpoint
    if resume and stats.total_processed > 0:
        confidence_sum = stats.avg_confidence * stats.total_processed
        margin_sum = stats.avg_margin * stats.total_processed

    if use_tqdm:
        pbar = tqdm(total=len(pending), desc="CLIP labeling", unit="img",
                     mininterval=2.0, ncols=100)
    else:
        pbar = None

    processed_in_session = 0
    for batch_start in range(0, len(pending), batch_size):
        batch_paths = pending[batch_start:batch_start + batch_size]

        batch_results = classify_image_batch(
            model=model,
            preprocess=preprocess,
            image_paths=batch_paths,
            text_features=text_features,
            label_names=label_names,
            device=device,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

        for result in batch_results:
            all_results.append(result)
            processed_paths.add(result.image_path)
            stats.total_processed += 1
            confidence_sum += result.clip_confidence
            margin_sum += result.clip_margin

            if result.review_status == "trusted":
                stats.high_confidence += 1
                stats.label_distribution[result.clip_label] = (
                    stats.label_distribution.get(result.clip_label, 0) + 1
                )
            elif result.review_status == "needs_review":
                stats.medium_confidence += 1
                stats.label_distribution[result.clip_label] = (
                    stats.label_distribution.get(result.clip_label, 0) + 1
                )
            else:
                stats.low_confidence += 1

        processed_in_session += len(batch_paths)

        if pbar is not None:
            pbar.update(len(batch_paths))

        # Periodic checkpoint
        if processed_in_session % CHECKPOINT_INTERVAL < batch_size:
            stats.avg_confidence = confidence_sum / max(stats.total_processed, 1)
            stats.avg_margin = margin_sum / max(stats.total_processed, 1)
            save_checkpoint(checkpoint_path, processed_paths, all_results, stats)
            gc.collect()

        # Basic progress logging fallback
        if not use_tqdm and processed_in_session % 5000 < batch_size:
            logger.info(
                "Progress: %d/%d | HIGH=%d MEDIUM=%d LOW=%d",
                processed_in_session, len(pending),
                stats.high_confidence, stats.medium_confidence, stats.low_confidence,
            )

    if pbar is not None:
        pbar.close()

    # Final stats
    stats.avg_confidence = confidence_sum / max(stats.total_processed, 1)
    stats.avg_margin = margin_sum / max(stats.total_processed, 1)
    stats.elapsed_sec = time.time() - t0

    # Save final checkpoint
    save_checkpoint(checkpoint_path, processed_paths, all_results, stats)

    # Free CLIP model
    del model, preprocess, tokenizer, text_features
    torch.cuda.empty_cache()
    gc.collect()

    if dry_run:
        logger.info("DRY RUN -- skipping manifest update")
        summary = stats.summary()
        logger.info("\n%s", summary)
        print(summary)
        return stats

    # Update manifest CSV with CLIP results
    logger.info("Updating manifest CSV...")
    result_map: Dict[str, LabelResult] = {r.image_path: r for r in all_results}

    # Initialize new columns
    clip_labels = []
    clip_confidences = []
    clip_margins = []
    review_statuses = []
    updated_labels = []
    updated_categories = []

    for _, row in df.iterrows():
        path = row["image_path"]
        original_label = row["label"]

        if path in result_map:
            r = result_map[path]
            clip_labels.append(r.clip_label)
            clip_confidences.append(round(r.clip_confidence, 6))
            clip_margins.append(round(r.clip_margin, 6))
            review_statuses.append(r.review_status)

            # Update label/category only if CLIP assigned a non-unlabeled label
            if r.clip_label != "unlabeled" and r.review_status in ("trusted", "needs_review"):
                updated_labels.append(r.clip_label)
                updated_categories.append(r.clip_label)
            else:
                updated_labels.append(original_label)
                updated_categories.append(row["category"])
        else:
            # Already labeled from Phase 1, preserve original
            clip_labels.append("")
            clip_confidences.append(np.nan)
            clip_margins.append(np.nan)
            review_statuses.append("pre_labeled")
            updated_labels.append(original_label)
            updated_categories.append(row["category"])

    df["clip_label"] = clip_labels
    df["clip_confidence"] = clip_confidences
    df["clip_margin"] = clip_margins
    df["review_status"] = review_statuses
    df["label"] = updated_labels
    df["category"] = updated_categories

    df.to_csv(manifest_path, index=False)
    logger.info("Manifest updated: %s", manifest_path)

    # Generate review queue CSV (MEDIUM confidence images)
    review_df = df[df["review_status"] == "needs_review"].copy()
    review_df = review_df.sort_values("clip_margin", ascending=True)
    review_df.to_csv(REVIEW_QUEUE_OUTPUT, index=False)
    logger.info("Review queue saved: %s (%d images)",
                REVIEW_QUEUE_OUTPUT, len(review_df))

    # Generate report
    generate_report(
        stats=stats,
        report_path=REPORT_OUTPUT,
        manifest_path=manifest_path,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )

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
        description="CLIP-based zero-shot auto-labeling for chart images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Full auto-labeling
    python scripts/auto_label_charts.py

    # Custom thresholds
    python scripts/auto_label_charts.py --high-threshold 0.12 --low-threshold 0.06

    # Smaller batch size (if OOM)
    python scripts/auto_label_charts.py --batch-size 32

    # Resume interrupted run
    python scripts/auto_label_charts.py --resume

    # Dry run
    python scripts/auto_label_charts.py --dry-run
""",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help="Path to chart dataset manifest CSV",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint file path",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Images per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD,
        help=f"HIGH confidence margin threshold (default: {DEFAULT_HIGH_THRESHOLD})",
    )
    parser.add_argument(
        "--low-threshold", type=float, default=DEFAULT_LOW_THRESHOLD,
        help=f"LOW confidence margin threshold (default: {DEFAULT_LOW_THRESHOLD})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process but don't update manifest CSV",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging(logging.INFO)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CLIP AUTO-LABELING PIPELINE")
    logger.info("=" * 60)
    logger.info("Manifest:        %s", args.manifest)
    logger.info("Batch size:      %d", args.batch_size)
    logger.info("HIGH threshold:  %.2f", args.high_threshold)
    logger.info("LOW threshold:   %.2f", args.low_threshold)
    logger.info("Resume:          %s", args.resume)
    logger.info("Dry run:         %s", args.dry_run)

    # Validate open_clip
    try:
        import open_clip  # noqa: F401
    except ImportError:
        logger.error("open-clip-torch is required: pip install open-clip-torch")
        sys.exit(1)

    # Validate manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.error("Run prepare_chart_dataset.py first.")
        sys.exit(1)

    stats = run_auto_labeling(
        manifest_path=manifest_path,
        checkpoint_path=Path(args.checkpoint),
        batch_size=args.batch_size,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    logger.info("DONE in %.1fs -- %d images auto-labeled",
                stats.elapsed_sec, stats.auto_labeled)


if __name__ == "__main__":
    main()
