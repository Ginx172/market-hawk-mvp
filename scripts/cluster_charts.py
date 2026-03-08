"""
SCRIPT NAME: cluster_charts.py
====================================
Execution Location: market-hawk-mvp/scripts/
Purpose: CLIP embedding extraction + K-Means clustering for chart auto-labeling.
Hardware Optimization: Intel i7-9700F, NVIDIA GTX 1070 8GB VRAM, 64GB DDR4
Creation Date: 2026-03-08

4-step pipeline for semi-supervised chart labeling via visual clustering:

    Step 1: Extract CLIP ViT-B/32 embeddings for all unlabeled images
            -> data/chart_embeddings.npy (N x 512, ~183 MB for 94K images)
            -> data/chart_embedding_paths.json

    Step 2: PCA reduction (512 -> 50) + K-Means clustering (k=15)
            -> data/chart_clusters.csv (image_path, cluster_id, distance_to_centroid)
            Optional: UMAP 2D projection if umap-learn is installed

    Step 3: Generate review samples (10 closest to centroid per cluster)
            -> data/cluster_samples/cluster_00/ ... cluster_14/
            -> data/cluster_review.html (visual grid for manual inspection)

    Step 4: Apply manual labels from data/cluster_labels.json
            -> Updates data/chart_dataset_manifest.csv

Each step is independent and can be run separately via --step flag.

Usage:
    # Run all steps
    python scripts/cluster_charts.py

    # Run single step
    python scripts/cluster_charts.py --step 1
    python scripts/cluster_charts.py --step 2 --n-clusters 20
    python scripts/cluster_charts.py --step 3
    python scripts/cluster_charts.py --step 4

    # Custom batch size for embedding extraction
    python scripts/cluster_charts.py --step 1 --batch-size 32

    # Resume embedding extraction from checkpoint
    python scripts/cluster_charts.py --step 1 --resume

    # Dry run (no file writes)
    python scripts/cluster_charts.py --step 4 --dry-run
"""

import argparse
import gc
import html
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Garanteaza ca project root e in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, LOGS_DIR, setup_logging

logger = logging.getLogger("market_hawk.cluster_charts")

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MANIFEST = DATA_DIR / "chart_dataset_manifest.csv"
EMBEDDINGS_FILE = DATA_DIR / "chart_embeddings.npy"
EMBEDDING_PATHS_FILE = DATA_DIR / "chart_embedding_paths.json"
CLUSTERS_FILE = DATA_DIR / "chart_clusters.csv"
CLUSTER_SAMPLES_DIR = DATA_DIR / "cluster_samples"
CLUSTER_REVIEW_HTML = DATA_DIR / "cluster_review.html"
CLUSTER_LABELS_FILE = DATA_DIR / "cluster_labels.json"
EMBEDDING_CHECKPOINT = LOGS_DIR / "cluster_embedding_checkpoint.json"

# CLIP config
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
EMBEDDING_DIM: int = 512

# Defaults
DEFAULT_N_CLUSTERS: int = 15
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_PCA_COMPONENTS: int = 50
DEFAULT_SAMPLES_PER_CLUSTER: int = 10
CHECKPOINT_INTERVAL: int = 5000


# ============================================================
# STEP 1: EXTRACT CLIP EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(
    manifest_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
) -> None:
    """Extract CLIP image embeddings for all unlabeled chart images.

    Processes images in batches through CLIP ViT-B/32 encoder,
    saving L2-normalized 512-dim embeddings to a numpy file.

    Args:
        manifest_path: Path to the chart dataset manifest CSV.
        batch_size: Images per batch for CLIP encoding.
        resume: Whether to resume from checkpoint.
    """
    import open_clip
    from PIL import Image

    t0 = time.time()

    # Optional tqdm
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Load manifest -- process unlabeled images only
    df = pd.read_csv(manifest_path)
    unlabeled_mask = df["label"] == "unlabeled"
    image_paths = df.loc[unlabeled_mask, "image_path"].tolist()
    logger.info("Total unlabeled images to embed: %d", len(image_paths))

    # Resume support
    start_idx = 0
    existing_embeddings: Optional[np.ndarray] = None
    existing_paths: List[str] = []

    if resume and EMBEDDING_CHECKPOINT.exists():
        with open(EMBEDDING_CHECKPOINT, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("processed_count", 0)
        existing_paths = ckpt.get("processed_paths", [])
        if EMBEDDINGS_FILE.exists() and start_idx > 0:
            existing_embeddings = np.load(EMBEDDINGS_FILE)
            logger.info("Resuming from checkpoint: %d already processed", start_idx)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load CLIP model
    logger.info("Loading CLIP model: %s (%s)", CLIP_MODEL_NAME, CLIP_PRETRAINED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device,
    )
    model.eval()

    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        logger.info("CLIP loaded: %.0f MB VRAM", vram_mb)

    # Prepare storage
    pending_paths = image_paths[start_idx:]
    total_pending = len(pending_paths)
    logger.info("Pending: %d images (batch_size=%d)", total_pending, batch_size)

    # Pre-allocate numpy array for new embeddings
    new_embeddings = np.zeros((total_pending, EMBEDDING_DIM), dtype=np.float32)
    new_valid_count = 0
    new_valid_paths: List[str] = []

    # Progress
    if use_tqdm:
        pbar = tqdm(total=total_pending, desc="Extracting embeddings",
                    unit="img", mininterval=2.0, ncols=100)
    else:
        pbar = None

    processed_in_session = 0

    for batch_start in range(0, total_pending, batch_size):
        batch_paths = pending_paths[batch_start:batch_start + batch_size]

        # Load and preprocess images
        batch_tensors: List[torch.Tensor] = []
        batch_valid_paths: List[str] = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)
                batch_valid_paths.append(img_path)
            except Exception as e:
                logger.debug("Skip unreadable image: %s (%s)", img_path, e)

        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to(device)
            features = model.encode_image(batch_tensor)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
            features_np = features.cpu().numpy()

            for i, path in enumerate(batch_valid_paths):
                new_embeddings[new_valid_count] = features_np[i]
                new_valid_count += 1
                new_valid_paths.append(path)

            del batch_tensor, features, features_np
            torch.cuda.empty_cache()

        processed_in_session += len(batch_paths)
        if pbar is not None:
            pbar.update(len(batch_paths))

        # Checkpoint
        if processed_in_session % CHECKPOINT_INTERVAL < batch_size:
            _save_embedding_checkpoint(
                start_idx + processed_in_session,
                existing_paths + new_valid_paths,
                existing_embeddings,
                new_embeddings[:new_valid_count],
            )
            gc.collect()

        # Fallback progress
        if not use_tqdm and processed_in_session % 10000 < batch_size:
            elapsed = time.time() - t0
            rate = processed_in_session / max(elapsed, 1)
            logger.info("Progress: %d/%d (%.1f img/s)",
                        processed_in_session, total_pending, rate)

    if pbar is not None:
        pbar.close()

    # Free model
    del model, preprocess
    torch.cuda.empty_cache()
    gc.collect()

    # Trim and combine embeddings
    new_embeddings = new_embeddings[:new_valid_count]
    all_paths = existing_paths + new_valid_paths

    if existing_embeddings is not None and len(existing_embeddings) > 0:
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        all_embeddings = new_embeddings

    # Save
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, all_embeddings)
    with open(EMBEDDING_PATHS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_paths, f)

    # Final checkpoint
    _save_embedding_checkpoint(
        start_idx + processed_in_session, all_paths, None, None,
    )

    elapsed = time.time() - t0
    logger.info(
        "Step 1 COMPLETE: %d embeddings extracted (shape=%s) in %.1fs",
        len(all_embeddings), all_embeddings.shape, elapsed,
    )
    logger.info("Saved: %s (%.1f MB)", EMBEDDINGS_FILE,
                all_embeddings.nbytes / (1024 * 1024))


def _save_embedding_checkpoint(
    processed_count: int,
    processed_paths: List[str],
    existing_embeddings: Optional[np.ndarray],
    new_embeddings: Optional[np.ndarray],
) -> None:
    """Save embedding extraction progress.

    Args:
        processed_count: Total images processed so far.
        processed_paths: All valid image paths processed.
        existing_embeddings: Previously saved embeddings (or None).
        new_embeddings: Newly extracted embeddings (or None).
    """
    EMBEDDING_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    # Save intermediate embeddings
    if new_embeddings is not None and len(new_embeddings) > 0:
        if existing_embeddings is not None and len(existing_embeddings) > 0:
            combined = np.vstack([existing_embeddings, new_embeddings])
        else:
            combined = new_embeddings
        np.save(EMBEDDINGS_FILE, combined)

    # Save checkpoint metadata
    ckpt = {
        "processed_count": processed_count,
        "processed_paths": processed_paths,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(EMBEDDING_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(ckpt, f)

    logger.info("Embedding checkpoint: %d processed", processed_count)


# ============================================================
# STEP 2: PCA + K-MEANS CLUSTERING
# ============================================================

def cluster_embeddings(
    n_clusters: int = DEFAULT_N_CLUSTERS,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
) -> None:
    """Reduce dimensionality with PCA and cluster with K-Means.

    Args:
        n_clusters: Number of K-Means clusters.
        pca_components: PCA target dimensions before clustering.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    t0 = time.time()

    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        logger.error("Embeddings not found: %s. Run --step 1 first.", EMBEDDINGS_FILE)
        sys.exit(1)
    if not EMBEDDING_PATHS_FILE.exists():
        logger.error("Embedding paths not found: %s. Run --step 1 first.",
                      EMBEDDING_PATHS_FILE)
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE)
    with open(EMBEDDING_PATHS_FILE, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    logger.info("Loaded embeddings: shape=%s, paths=%d",
                embeddings.shape, len(image_paths))
    assert len(embeddings) == len(image_paths), "Embeddings/paths count mismatch"

    # PCA reduction
    logger.info("PCA: %d -> %d dimensions...", embeddings.shape[1], pca_components)
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA variance explained: %.2f%%", variance_explained * 100)

    # K-Means clustering
    logger.info("K-Means clustering: k=%d...", n_clusters)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=0,
    )
    cluster_ids = kmeans.fit_predict(embeddings_pca)

    # Compute distances to centroid
    distances = np.zeros(len(embeddings_pca), dtype=np.float32)
    for i in range(len(embeddings_pca)):
        centroid = kmeans.cluster_centers_[cluster_ids[i]]
        distances[i] = np.linalg.norm(embeddings_pca[i] - centroid)

    # Silhouette score (sample for speed if > 10K)
    sample_size = min(len(embeddings_pca), 10000)
    if sample_size < len(embeddings_pca):
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(embeddings_pca), sample_size, replace=False)
        sil_score = silhouette_score(
            embeddings_pca[sample_idx], cluster_ids[sample_idx],
        )
    else:
        sil_score = silhouette_score(embeddings_pca, cluster_ids)
    logger.info("Silhouette score: %.4f", sil_score)

    # Cluster size distribution
    unique, counts = np.unique(cluster_ids, return_counts=True)
    logger.info("Cluster sizes:")
    for cid, cnt in zip(unique, counts):
        logger.info("  Cluster %02d: %6d images", cid, cnt)

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        "image_path": image_paths,
        "cluster_id": cluster_ids,
        "distance_to_centroid": np.round(distances, 6),
    })
    cluster_df.to_csv(CLUSTERS_FILE, index=False)
    logger.info("Cluster assignments saved: %s", CLUSTERS_FILE)

    # Optional: UMAP 2D projection for visualization
    _try_umap_projection(embeddings_pca, cluster_ids)

    elapsed = time.time() - t0
    logger.info("Step 2 COMPLETE in %.1fs (silhouette=%.4f)", elapsed, sil_score)


def _try_umap_projection(
    embeddings_pca: np.ndarray,
    cluster_ids: np.ndarray,
) -> None:
    """Attempt UMAP 2D projection and save coordinates.

    Args:
        embeddings_pca: PCA-reduced embeddings.
        cluster_ids: Cluster assignments.
    """
    try:
        import umap

        logger.info("UMAP 2D projection (this may take a few minutes)...")
        reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
        )
        coords_2d = reducer.fit_transform(embeddings_pca)

        umap_df = pd.DataFrame({
            "umap_x": coords_2d[:, 0],
            "umap_y": coords_2d[:, 1],
            "cluster_id": cluster_ids,
        })
        umap_path = DATA_DIR / "chart_umap_2d.csv"
        umap_df.to_csv(umap_path, index=False)
        logger.info("UMAP 2D saved: %s", umap_path)

    except ImportError:
        logger.info("umap-learn not installed, skipping UMAP projection. "
                     "Install with: pip install umap-learn")
    except Exception as e:
        logger.warning("UMAP failed: %s", e)


# ============================================================
# STEP 3: GENERATE REVIEW SAMPLES
# ============================================================

def generate_review_samples(
    samples_per_cluster: int = DEFAULT_SAMPLES_PER_CLUSTER,
) -> None:
    """Select representative images per cluster and generate HTML review page.

    For each cluster, copies the N images closest to the centroid
    into data/cluster_samples/cluster_XX/ directories, and generates
    an HTML page for visual review.

    Args:
        samples_per_cluster: Number of representative images per cluster.
    """
    t0 = time.time()

    if not CLUSTERS_FILE.exists():
        logger.error("Cluster assignments not found: %s. Run --step 2 first.",
                      CLUSTERS_FILE)
        sys.exit(1)

    cluster_df = pd.read_csv(CLUSTERS_FILE)
    n_clusters = cluster_df["cluster_id"].nunique()
    logger.info("Loaded %d cluster assignments (%d clusters)",
                len(cluster_df), n_clusters)

    # Clean previous samples
    if CLUSTER_SAMPLES_DIR.exists():
        shutil.rmtree(CLUSTER_SAMPLES_DIR)
    CLUSTER_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # HTML page header
    html_parts: List[str] = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Chart Cluster Review</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; "
        "margin: 20px; }",
        "h1 { color: #e94560; }",
        "h2 { color: #0f3460; background: #eee; padding: 8px 16px; "
        "border-radius: 4px; margin-top: 40px; }",
        ".cluster-info { color: #aaa; margin-bottom: 10px; }",
        ".grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }",
        ".grid img { width: 220px; height: 165px; object-fit: contain; "
        "border: 2px solid #333; border-radius: 4px; background: #fff; }",
        ".grid .img-card { text-align: center; }",
        ".grid .img-card p { font-size: 11px; color: #888; margin: 4px 0; "
        "word-break: break-all; max-width: 220px; }",
        "table { border-collapse: collapse; margin: 20px 0; }",
        "td, th { border: 1px solid #444; padding: 6px 12px; text-align: left; }",
        "th { background: #16213e; }",
        ".instructions { background: #16213e; padding: 16px; border-radius: 8px; "
        "margin-bottom: 30px; }",
        ".instructions code { background: #0f3460; padding: 2px 6px; "
        "border-radius: 3px; }",
        "</style>",
        "</head><body>",
        "<h1>Chart Cluster Review</h1>",
        "<div class='instructions'>",
        "<p><strong>Instructions:</strong></p>",
        "<ol>",
        "<li>Review the representative images for each cluster below.</li>",
        "<li>Decide which chart pattern label fits each cluster.</li>",
        "<li>Edit <code>data/cluster_labels.json</code> with your assignments:</li>",
        "</ol>",
        '<pre>{\n  "0": "head_and_shoulders",\n  "1": "double_top",\n'
        '  "2": "generic_chart",\n  ...\n}</pre>',
        "<p>Then run: <code>python scripts/cluster_charts.py --step 4</code></p>",
        "</div>",
        "<h2>Cluster Overview</h2>",
        "<table><tr><th>Cluster</th><th>Images</th>"
        "<th>Avg Distance</th><th>Min Distance</th></tr>",
    ]

    # Summary table rows
    for cid in sorted(cluster_df["cluster_id"].unique()):
        c_data = cluster_df[cluster_df["cluster_id"] == cid]
        count = len(c_data)
        avg_dist = c_data["distance_to_centroid"].mean()
        min_dist = c_data["distance_to_centroid"].min()
        html_parts.append(
            f"<tr><td>Cluster {cid:02d}</td><td>{count:,}</td>"
            f"<td>{avg_dist:.4f}</td><td>{min_dist:.4f}</td></tr>"
        )
    html_parts.append("</table>")

    # Per-cluster sample grids
    for cid in sorted(cluster_df["cluster_id"].unique()):
        c_data = cluster_df[cluster_df["cluster_id"] == cid].copy()
        c_data = c_data.sort_values("distance_to_centroid", ascending=True)
        c_count = len(c_data)

        # Select top N closest to centroid
        samples = c_data.head(samples_per_cluster)

        # Create cluster sample directory
        cluster_dir = CLUSTER_SAMPLES_DIR / f"cluster_{cid:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        html_parts.append(
            f"<h2>Cluster {cid:02d} ({c_count:,} images)</h2>"
        )
        html_parts.append(
            f"<div class='cluster-info'>Avg distance to centroid: "
            f"{c_data['distance_to_centroid'].mean():.4f}</div>"
        )
        html_parts.append("<div class='grid'>")

        for _, row in samples.iterrows():
            src_path = Path(row["image_path"])
            dist = row["distance_to_centroid"]

            # Copy sample image
            if src_path.exists():
                dst_path = cluster_dir / src_path.name
                # Avoid name collisions
                if dst_path.exists():
                    dst_path = cluster_dir / f"{src_path.stem}_{dist:.4f}{src_path.suffix}"
                try:
                    shutil.copy2(src_path, dst_path)
                except OSError as e:
                    logger.warning("Failed to copy %s: %s", src_path, e)
                    continue

                # Use relative path for HTML (from data/ directory)
                rel_path = dst_path.relative_to(DATA_DIR)
                fname_escaped = html.escape(src_path.name)
                html_parts.append(
                    f"<div class='img-card'>"
                    f"<img src='{rel_path}' alt='{fname_escaped}' "
                    f"loading='lazy'>"
                    f"<p>{fname_escaped}</p>"
                    f"<p>dist: {dist:.4f}</p>"
                    f"</div>"
                )
            else:
                logger.debug("Image not found: %s", src_path)

        html_parts.append("</div>")

    # HTML footer
    html_parts.extend([
        f"<hr><p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{n_clusters} clusters | {len(cluster_df):,} images</p>",
        "</body></html>",
    ])

    # Write HTML
    CLUSTER_REVIEW_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_REVIEW_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    logger.info("Review HTML saved: %s", CLUSTER_REVIEW_HTML)

    # Also generate an empty cluster_labels.json template if it doesn't exist
    if not CLUSTER_LABELS_FILE.exists():
        template: Dict[str, str] = {}
        for cid in sorted(cluster_df["cluster_id"].unique()):
            template[str(cid)] = ""
        with open(CLUSTER_LABELS_FILE, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        logger.info("Label template created: %s (fill in labels and run --step 4)",
                     CLUSTER_LABELS_FILE)

    elapsed = time.time() - t0
    logger.info("Step 3 COMPLETE in %.1fs (%d clusters, %d samples each)",
                elapsed, n_clusters, samples_per_cluster)


# ============================================================
# STEP 4: APPLY MANUAL LABELS
# ============================================================

def apply_cluster_labels(
    manifest_path: Path,
    dry_run: bool = False,
) -> None:
    """Apply manual cluster labels to the manifest CSV.

    Reads cluster_labels.json (user-created mapping of cluster_id -> label),
    joins with cluster_clusters.csv, and updates the manifest.

    Args:
        manifest_path: Path to the chart dataset manifest CSV.
        dry_run: If True, show what would change but don't write.
    """
    t0 = time.time()

    # Validate inputs
    if not CLUSTER_LABELS_FILE.exists():
        logger.error(
            "Cluster labels file not found: %s\n"
            "Create this file with cluster_id -> label mapping, e.g.:\n"
            '  {"0": "head_and_shoulders", "1": "double_top", ...}\n'
            "Review data/cluster_review.html to decide labels.",
            CLUSTER_LABELS_FILE,
        )
        sys.exit(1)

    if not CLUSTERS_FILE.exists():
        logger.error("Cluster assignments not found: %s. Run --step 2 first.",
                      CLUSTERS_FILE)
        sys.exit(1)

    # Load cluster labels
    with open(CLUSTER_LABELS_FILE, "r", encoding="utf-8") as f:
        cluster_labels_raw = json.load(f)

    # Convert keys to int, skip empty labels
    cluster_labels: Dict[int, str] = {}
    for cid_str, label in cluster_labels_raw.items():
        label = label.strip()
        if label:
            cluster_labels[int(cid_str)] = label

    if not cluster_labels:
        logger.error("No labels found in %s. Fill in labels and retry.",
                      CLUSTER_LABELS_FILE)
        sys.exit(1)

    logger.info("Loaded %d cluster labels from %s",
                len(cluster_labels), CLUSTER_LABELS_FILE)
    for cid, label in sorted(cluster_labels.items()):
        logger.info("  Cluster %02d -> %s", cid, label)

    # Load cluster assignments
    cluster_df = pd.read_csv(CLUSTERS_FILE)
    logger.info("Loaded %d cluster assignments", len(cluster_df))

    # Build image_path -> label mapping
    path_to_label: Dict[str, str] = {}
    for _, row in cluster_df.iterrows():
        cid = int(row["cluster_id"])
        if cid in cluster_labels:
            path_to_label[row["image_path"]] = cluster_labels[cid]

    logger.info("Images to label: %d", len(path_to_label))

    # Load manifest
    df = pd.read_csv(manifest_path)
    logger.info("Manifest loaded: %d rows", len(df))

    # Count changes
    changes = 0
    for idx, row in df.iterrows():
        if row["image_path"] in path_to_label:
            new_label = path_to_label[row["image_path"]]
            if row["label"] != new_label:
                changes += 1
                if not dry_run:
                    df.at[idx, "label"] = new_label
                    df.at[idx, "category"] = new_label
                    df.at[idx, "review_status"] = "cluster_labeled"

    logger.info("Labels to apply: %d changes", changes)

    if dry_run:
        logger.info("DRY RUN -- no changes written")
        # Show distribution preview
        preview: Dict[str, int] = {}
        for label in path_to_label.values():
            preview[label] = preview.get(label, 0) + 1
        logger.info("Preview label distribution:")
        for label, count in sorted(preview.items(), key=lambda x: -x[1]):
            logger.info("  %-30s %8d", label, count)
    else:
        df.to_csv(manifest_path, index=False)
        logger.info("Manifest updated: %s (%d labels applied)", manifest_path, changes)

    elapsed = time.time() - t0
    logger.info("Step 4 COMPLETE in %.1fs", elapsed)

    # Final label distribution
    if not dry_run:
        label_dist = df["label"].value_counts()
        logger.info("Final label distribution:")
        for label, count in label_dist.items():
            logger.info("  %-30s %8d", label, count)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLIP embedding clustering pipeline for chart auto-labeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Steps:
    1: Extract CLIP embeddings (data/chart_embeddings.npy)
    2: PCA + K-Means clustering (data/chart_clusters.csv)
    3: Generate review samples (data/cluster_review.html)
    4: Apply labels from data/cluster_labels.json

Examples:
    # All steps
    python scripts/cluster_charts.py

    # Single step
    python scripts/cluster_charts.py --step 1
    python scripts/cluster_charts.py --step 2 --n-clusters 20
    python scripts/cluster_charts.py --step 3
    python scripts/cluster_charts.py --step 4

    # Resume embedding extraction
    python scripts/cluster_charts.py --step 1 --resume

    # Dry run step 4
    python scripts/cluster_charts.py --step 4 --dry-run
""",
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a single step (1-4). If omitted, runs steps 1-3.",
    )
    parser.add_argument(
        "--manifest", type=str, default=str(DEFAULT_MANIFEST),
        help="Path to chart dataset manifest CSV",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
        help=f"Number of K-Means clusters (default: {DEFAULT_N_CLUSTERS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding extraction (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--samples-per-cluster", type=int, default=DEFAULT_SAMPLES_PER_CLUSTER,
        help=f"Review samples per cluster (default: {DEFAULT_SAMPLES_PER_CLUSTER})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume embedding extraction from checkpoint",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show changes without writing (step 4 only)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    setup_logging(logging.INFO)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CHART CLUSTERING PIPELINE")
    logger.info("=" * 60)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    # Validate open_clip for steps that need it
    if args.step in (1, None):
        try:
            import open_clip  # noqa: F401
        except ImportError:
            logger.error("open-clip-torch is required: pip install open-clip-torch")
            sys.exit(1)

    # Determine which steps to run
    if args.step is not None:
        steps = [args.step]
    else:
        # Default: run steps 1-3 (step 4 requires manual label file)
        steps = [1, 2, 3]

    t_total = time.time()

    for step in steps:
        logger.info("-" * 40)
        logger.info("STEP %d", step)
        logger.info("-" * 40)

        if step == 1:
            extract_embeddings(
                manifest_path=manifest_path,
                batch_size=args.batch_size,
                resume=args.resume,
            )
        elif step == 2:
            cluster_embeddings(
                n_clusters=args.n_clusters,
                pca_components=DEFAULT_PCA_COMPONENTS,
            )
        elif step == 3:
            generate_review_samples(
                samples_per_cluster=args.samples_per_cluster,
            )
        elif step == 4:
            apply_cluster_labels(
                manifest_path=manifest_path,
                dry_run=args.dry_run,
            )

        gc.collect()

    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs (steps: %s)",
                total_elapsed, steps)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
