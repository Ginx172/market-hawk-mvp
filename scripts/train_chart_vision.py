"""
SCRIPT NAME: train_chart_vision.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: Train chart pattern recognition vision model (ResNet50 transfer learning).
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-08

Orchestrates the full training pipeline:
    1. Load manifest CSV (from prepare_chart_dataset.py)
    2. Split into train/val/test (70/15/15)
    3. Create ChartPatternDataset instances with appropriate transforms
    4. Train ChartPatternClassifier via ChartTrainer
    5. Evaluate on test set
    6. Save model + training report + update model_hashes.json

Usage:
    # Dry run (show data stats, no training)
    python scripts/train_chart_vision.py --dry-run

    # Train with defaults
    python scripts/train_chart_vision.py

    # Custom parameters
    python scripts/train_chart_vision.py --epochs 30 --batch-size 16 --lr 5e-5

    # Resume from checkpoint
    python scripts/train_chart_vision.py --resume models/vision/chart_vision_ep010_acc0.7500.pt

    # Custom manifest and output
    python scripts/train_chart_vision.py --manifest data/custom_manifest.csv --output-dir models/vision_v2
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    DATA_DIR, HARDWARE, MODELS_DIR, setup_logging,
)
from ml.vision_model import (
    ChartPatternClassifier,
    ChartPatternDataset,
    ChartTrainer,
    TrainingReport,
    get_train_transforms,
    get_val_transforms,
)

logger = logging.getLogger("market_hawk.train_chart_vision")

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MANIFEST = DATA_DIR / "chart_dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "vision"
MODEL_HASHES_FILE = MODELS_DIR / "model_hashes.json"

# Split ratios
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15


# ============================================================
# DATA SPLITTING
# ============================================================

def split_manifest(
    manifest_path: Path,
    label_column: str = "label",
    filter_unlabeled: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest CSV into train/val/test DataFrames.

    Uses stratified random split (70/15/15) to maintain label distribution.

    Args:
        manifest_path: Path to the manifest CSV.
        label_column: Column to use as stratification target.
        filter_unlabeled: If True, remove rows where label == "unlabeled".
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df = pd.read_csv(manifest_path)
    logger.info("Manifest loaded: %d total rows", len(df))

    if filter_unlabeled:
        before = len(df)
        df = df[df[label_column] != "unlabeled"].reset_index(drop=True)
        logger.info("Filtered unlabeled: %d -> %d rows", before, len(df))

    if len(df) == 0:
        raise ValueError("No labeled samples found in manifest")

    # Count per class
    class_counts = df[label_column].value_counts()
    logger.info("Class distribution:\n%s", class_counts.to_string())

    # Remove classes with < 3 samples (need at least 1 per split)
    min_samples = 3
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    removed = class_counts[class_counts < min_samples]
    if len(removed) > 0:
        logger.warning(
            "Removing %d classes with < %d samples: %s",
            len(removed), min_samples, removed.index.tolist(),
        )
        df = df[df[label_column].isin(valid_classes)].reset_index(drop=True)

    # Stratified split
    try:
        from sklearn.model_selection import train_test_split

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, test_size=(VAL_RATIO + TEST_RATIO),
            stratify=df[label_column],
            random_state=random_seed,
        )

        # Second split: val vs test
        relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test,
            stratify=temp_df[label_column],
            random_state=random_seed,
        )
    except ValueError as e:
        logger.warning(
            "Stratified split failed (%s), falling back to random split", e,
        )
        np.random.seed(random_seed)
        indices = np.random.permutation(len(df))
        n_train = int(len(df) * TRAIN_RATIO)
        n_val = int(len(df) * VAL_RATIO)
        train_df = df.iloc[indices[:n_train]].reset_index(drop=True)
        val_df = df.iloc[indices[n_train:n_train + n_val]].reset_index(drop=True)
        test_df = df.iloc[indices[n_train + n_val:]].reset_index(drop=True)

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    return train_df, val_df, test_df


def write_split_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Write split DataFrames to separate CSV files.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        output_dir: Directory to write split CSVs.

    Returns:
        Tuple of (train_csv_path, val_csv_path, test_csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_manifest.csv"
    val_path = output_dir / "val_manifest.csv"
    test_path = output_dir / "test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Split manifests saved to %s", output_dir)
    return train_path, val_path, test_path


# ============================================================
# MODEL HASH INTEGRATION
# ============================================================

def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file, reading in 8KB chunks.

    Args:
        path: Path to the file.

    Returns:
        Hex digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def update_model_hashes(model_path: Path, model_name: str) -> None:
    """Add or update model entry in model_hashes.json.

    Args:
        model_path: Path to the model file.
        model_name: Registry name for the model.
    """
    hashes: Dict[str, Any] = {}
    if MODEL_HASHES_FILE.exists():
        with open(MODEL_HASHES_FILE, "r", encoding="utf-8") as f:
            hashes = json.load(f)

    file_hash = compute_file_hash(model_path)
    hashes[model_name] = {
        "path": str(model_path),
        "sha256": file_hash,
    }

    with open(MODEL_HASHES_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)

    logger.info("model_hashes.json updated: %s -> %s", model_name, file_hash[:16])


# ============================================================
# TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_test_set(
    model: ChartPatternClassifier,
    test_dataset: ChartPatternDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, float]:
    """Evaluate trained model on test set.

    Args:
        model: Trained ChartPatternClassifier.
        test_dataset: Test dataset.
        device: Torch device.
        batch_size: Batch size for evaluation.
        num_workers: DataLoader workers.

    Returns:
        Dict with accuracy, precision, recall, f1 metrics.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    model.eval()
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds: List[int] = []
    all_labels: List[int] = []
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    results = {
        "test_accuracy": accuracy_score(all_labels, all_preds),
        "test_precision": precision_score(
            all_labels, all_preds, average="weighted", zero_division=0,
        ),
        "test_recall": recall_score(
            all_labels, all_preds, average="weighted", zero_division=0,
        ),
        "test_f1": f1_score(
            all_labels, all_preds, average="weighted", zero_division=0,
        ),
    }

    logger.info(
        "Test results: acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
        results["test_accuracy"], results["test_precision"],
        results["test_recall"], results["test_f1"],
    )
    return results


# ============================================================
# DRY RUN
# ============================================================

def dry_run_report(args: argparse.Namespace) -> None:
    """Print configuration and dataset stats without training.

    Args:
        args: Parsed CLI arguments.
    """
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        print("Run prepare_chart_dataset.py first.")
        return

    df = pd.read_csv(manifest_path)
    labeled = df[df["label"] != "unlabeled"]

    print("=" * 60)
    print("CHART VISION TRAINING -- DRY RUN")
    print("=" * 60)
    print(f"Manifest:        {manifest_path}")
    print(f"Total rows:      {len(df):,}")
    print(f"Labeled rows:    {len(labeled):,}")
    print(f"Unlabeled rows:  {len(df) - len(labeled):,}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print()

    # Split preview
    if len(labeled) > 0:
        n_train = int(len(labeled) * TRAIN_RATIO)
        n_val = int(len(labeled) * VAL_RATIO)
        n_test = len(labeled) - n_train - n_val
        print(f"Planned split:   train={n_train:,} / val={n_val:,} / test={n_test:,}")
        print()

    # Class distribution
    print("--- Label Distribution ---")
    for label, count in labeled["label"].value_counts().items():
        print(f"  {label:<30s} {count:>8,}")

    # Hardware
    print()
    print("--- Hardware ---")
    print(f"  GPU available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU name:        {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM:            {vram:.1f} GB")
    print(f"  CPU cores:       {HARDWARE.cpu_cores}")
    print(f"  RAM:             {HARDWARE.ram_gb} GB")

    # Model size estimate
    print()
    n_classes = labeled["label"].nunique()
    print(f"  Model classes:   {n_classes}")
    print(f"  ResNet50 params: ~23.5M total, ~8.5M trainable (layer4 + head)")
    print(f"  Est. VRAM usage: ~2.5-3.5 GB (fp16, batch_size={args.batch_size})")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train chart pattern recognition vision model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Dry run
    python scripts/train_chart_vision.py --dry-run

    # Train with defaults
    python scripts/train_chart_vision.py

    # Custom settings
    python scripts/train_chart_vision.py --epochs 30 --batch-size 16 --lr 5e-5

    # Resume from checkpoint
    python scripts/train_chart_vision.py --resume models/vision/chart_vision_ep010_acc0.7500.pt
""",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help="Path to chart dataset manifest CSV",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=HARDWARE.batch_size,
        help=f"Batch size (default: {HARDWARE.batch_size})",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for model checkpoints and reports",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration and data stats without training",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--label-column", type=str, default="label",
        help="Column name to use as classification target (default: label)",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Entry point."""
    setup_logging(logging.INFO)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CHART VISION MODEL TRAINING")
    logger.info("=" * 60)

    # Dry run
    if args.dry_run:
        dry_run_report(args)
        return

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Validate manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.error("Run prepare_chart_dataset.py first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Step 1: Split data
    logger.info("Step 1/4: Splitting dataset...")
    train_df, val_df, test_df = split_manifest(
        manifest_path,
        label_column=args.label_column,
        random_seed=args.seed,
    )
    train_csv, val_csv, test_csv = write_split_manifests(
        train_df, val_df, test_df, output_dir,
    )

    # Step 2: Create datasets
    logger.info("Step 2/4: Creating datasets...")
    # Build label mapping from training set
    unique_labels = sorted(train_df[args.label_column].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(label_to_idx)
    logger.info("Number of classes: %d", num_classes)
    logger.info("Classes: %s", unique_labels)

    # Save label mapping
    label_map_path = output_dir / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    train_dataset = ChartPatternDataset(
        manifest_path=str(train_csv),
        label_column=args.label_column,
        transform=get_train_transforms(),
        label_to_idx=label_to_idx,
        filter_unlabeled=False,  # Already filtered in split
    )
    val_dataset = ChartPatternDataset(
        manifest_path=str(val_csv),
        label_column=args.label_column,
        transform=get_val_transforms(),
        label_to_idx=label_to_idx,
        filter_unlabeled=False,
    )
    test_dataset = ChartPatternDataset(
        manifest_path=str(test_csv),
        label_column=args.label_column,
        transform=get_val_transforms(),
        label_to_idx=label_to_idx,
        filter_unlabeled=False,
    )

    # Step 3: Train
    logger.info("Step 3/4: Training model...")
    model = ChartPatternClassifier(
        num_classes=num_classes,
        pretrained=True,
    )
    logger.info(
        "Model: %d total params, %d trainable",
        model.get_total_params(), model.get_trainable_params(),
    )

    trainer = ChartTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )

    report = trainer.train(resume_checkpoint=args.resume)

    # Step 4: Evaluate on test set
    logger.info("Step 4/4: Evaluating on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_results = evaluate_test_set(
        model=model,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Update report with test metrics
    report.test_samples = len(test_dataset)
    test_report_path = output_dir / "test_results.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # Update model_hashes.json
    best_model_path = output_dir / "chart_vision_best.pt"
    if best_model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"chart_vision_{timestamp}"
        update_model_hashes(best_model_path, model_name)

    total_elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE in %.1fs", total_elapsed)
    logger.info("Best val accuracy: %.4f (epoch %d)",
                report.best_val_accuracy, report.best_epoch)
    logger.info("Test accuracy:     %.4f", test_results["test_accuracy"])
    logger.info("Test F1:           %.4f", test_results["test_f1"])
    logger.info("Model saved:       %s", best_model_path)
    logger.info("=" * 60)

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
