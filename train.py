"""
AQP1 Training Script
=====================
Trains a YOLO segmentation model to detect two regions per overlay image:
    Class 0  overlay_outer   →  total width (120 mm) + total length (370 mm)
    Class 1  dark_core       →  dark area width (80 mm) + dark area height (340 mm)

Workflow
---------
  python train.py            → auto-annotates ALL images, splits dataset, trains model
  python main.py             → server loads new model automatically

  Everything is fully automatic — no manual annotation required.
  On each run train.py will:
    Phase 1  Auto-annotate every image in the training source folder:
               - Keep any existing confirmed labels untouched
               - Use the current best model (if one exists) for images above --conf
               - Fall back to classical CV for new/uncertain images
    Phase 2  Build the 85/15 train/val split and write data.yaml
    Phase 3  Train YOLO and save the best weights to models/

Checkpoint naming (mirrors ADP4 pattern):
    Every epoch  →  models/aqp1_S4_e012_20260413_1430.pt
    Final best   →  models/aqp1_S4_50ep_4h12m_20260413_1430.pt
    On Ctrl+C    →  models/aqp1_S4_e012_INTERRUPTED_20260413_1430.pt

Usage
------
  # Standard run (auto-detects device, 50 epochs):
  python train.py --size S4

  # Allow more RAM on a larger machine:
  python train.py --size S4 --ram-limit 8.0

  # Longer run with a stronger base model:
  python train.py --size S4 --epochs 100 --model yolo11s-seg.pt

  # CUDA GPU (overrides auto-detect):
  python train.py --size S4 --device 0
"""

import argparse
import csv
import gc
import json
import logging
import os
import re
import shutil
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# 0. HEADLESS AUTO-ANNOTATION
#    Runs fully automatically — no UI, no clicks.
#
#    Per-image priority:
#      1. Existing confirmed label (.txt already present)  → keep untouched
#      2. Trained model (if available)                     → accept if conf ≥ threshold
#      3. Classical CV                                     → column-intensity profiling
#
#    Classical CV strategy for AQP1 overlay images:
#      outer: scan edge columns/rows for the first non-black pixel (bright edges)
#      core : column-wise mean intensity profile inside the outer ROI —
#             the pad core is the darker band in the horizontal centre
# ==============================================================================

def _cv_detect_outer(gray: np.ndarray):
    """Return (x1,y1,x2,y2) of the non-black product region."""
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    def _first_bright_col(arr, thr=25):
        for i, v in enumerate(arr):
            if v > thr: return i
        return 0
    def _last_bright_col(arr, thr=25):
        for i in range(len(arr)-1, -1, -1):
            if arr[i] > thr: return i
        return len(arr)-1

    # Column profile — mean over middle 60 % of rows to avoid top/bottom edge noise
    r0, r1 = H // 5, 4 * H // 5
    col_mean = blur[r0:r1, :].mean(axis=0)
    x1 = _first_bright_col(col_mean)
    x2 = _last_bright_col(col_mean)

    # Row profile — mean over detected product columns
    c0, c1 = max(0, x1 + (x2-x1)//5), min(W, x2 - (x2-x1)//5)
    row_mean = blur[:, c0:c1].mean(axis=1)
    y1 = _first_bright_col(row_mean, thr=15)
    y2 = _last_bright_col(row_mean, thr=15)

    # Safety margins — YOLO bounding boxes should hug the content, not the image edge
    pad = max(2, min(W, H) // 80)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)
    return x1, y1, x2, y2


def _cv_detect_core(gray: np.ndarray, outer: tuple):
    """
    Return (x1,y1,x2,y2) of the pad core inside `outer`.

    Uses a column-intensity profile within the outer ROI:
    the pad core is the darker horizontal band in the centre.
    We find where the smoothed column mean drops below a threshold,
    giving the left and right boundaries of the core.
    Vertical extent is taken from a row-intensity threshold within that
    column span.
    """
    H, W = gray.shape
    ox1, oy1, ox2, oy2 = outer
    roi = gray[oy1:oy2, ox1:ox2].astype(float)
    rH, rW = roi.shape

    if rH < 10 or rW < 10:
        # Fallback: centre quarter
        return (ox1 + rW//4, oy1 + rH//20,
                ox2 - rW//4, oy2 - rH//20)

    # ── Horizontal (width) detection ──────────────────────────────
    # Smooth column-mean profile and find the dip in the centre
    col_mean = cv2.GaussianBlur(roi, (1, min(rH|1, 31)), 0).mean(axis=0)
    # Normalise to 0-1
    mn, mx = col_mean.min(), col_mean.max()
    if mx - mn < 5:   # flat image — fallback
        cx1 = ox1 + rW // 4
        cx2 = ox2 - rW // 4
    else:
        col_norm = (col_mean - mn) / (mx - mn)
        # Threshold: pixels darker than 55 % of the normalised range
        dark_mask = col_norm < 0.55
        # Keep only the longest run of dark columns in the centre 80 %
        margin = rW // 10
        dark_mask[:margin]     = False
        dark_mask[rW-margin:]  = False

        # Find contiguous dark runs
        dark_cols = np.where(dark_mask)[0]
        if len(dark_cols) < 4:
            cx1 = ox1 + rW // 4
            cx2 = ox2 - rW // 4
        else:
            # Longest run
            runs, cur = [], [dark_cols[0]]
            for c in dark_cols[1:]:
                if c == cur[-1] + 1:
                    cur.append(c)
                else:
                    runs.append(cur)
                    cur = [c]
            runs.append(cur)
            best = max(runs, key=len)
            cx1 = ox1 + best[0]
            cx2 = ox1 + best[-1]
            # Small outward pad (the core boundary is slightly lighter than the interior)
            edge_pad = max(2, rW // 30)
            cx1 = max(ox1, cx1 - edge_pad)
            cx2 = min(ox2, cx2 + edge_pad)

    # ── Vertical (height) detection ───────────────────────────────
    col_slice = gray[oy1:oy2, cx1:cx2].astype(float)
    row_mean = col_slice.mean(axis=1)
    mn2, mx2 = row_mean.min(), row_mean.max()
    if mx2 - mn2 < 5:
        cy1, cy2 = oy1 + rH // 20, oy2 - rH // 20
    else:
        row_norm = (row_mean - mn2) / (mx2 - mn2)
        dark_rows = np.where(row_norm < 0.65)[0]
        if len(dark_rows) < 4:
            cy1, cy2 = oy1 + rH // 20, oy2 - rH // 20
        else:
            cy1 = oy1 + dark_rows[0]
            cy2 = oy1 + dark_rows[-1]
            v_pad = max(2, rH // 40)
            cy1 = max(oy1, cy1 - v_pad)
            cy2 = min(oy2, cy2 + v_pad)

    return cx1, cy1, cx2, cy2


def _yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int,
               W: int, H: int) -> str:
    """Convert absolute bbox to YOLO bounding-box format (class cx cy w h)."""
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def headless_annotate_all(
    size:            str,
    model_path:      str | None = None,
    conf_threshold:  float      = 0.60,
    overwrite:       bool       = False,
) -> int:
    """
    Annotate every image in cfg.TRAINING_SOURCES[size] without any human input.

    Args:
        size:           Product size key, e.g. "S4".
        model_path:     Path to an existing .pt to use for model-assisted labelling.
                        When None (or the file doesn't exist) falls back to CV only.
        conf_threshold: Minimum YOLO confidence to accept a model detection.
        overwrite:      If True, re-annotate images that already have a label.

    Returns:
        Number of images that now have a label file.
    """
    images_dir = Path(cfg.TRAINING_SOURCES.get(size, ""))
    labels_dir = Path(cfg.DATASET_DIR) / "train" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        log.error("Images folder not found: %s", images_dir)
        sys.exit(1)

    images = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png")) +
        list(images_dir.glob("*.tiff"))
    )
    if not images:
        log.error("No images found in %s", images_dir)
        sys.exit(1)

    log.info("Auto-annotation: %d images found in %s", len(images), images_dir)

    # ── Load model if available ───────────────────────────────────────────────
    yolo_model = None
    if model_path and Path(model_path).exists():
        try:
            yolo_model = YOLO(model_path)
            log.info("Model-assisted annotation: %s  (conf ≥ %.2f)", model_path, conf_threshold)
        except Exception as e:
            log.warning("Could not load model for annotation: %s — using CV only", e)

    # ── Annotation counters ───────────────────────────────────────────────────
    n_kept      = 0  # existing labels left untouched
    n_model     = 0  # annotated by model
    n_cv        = 0  # annotated by classical CV
    n_failed    = 0  # could not annotate

    for i, img_path in enumerate(images):
        lbl_path = labels_dir / (img_path.stem + ".txt")

        # 1. Keep existing label unless overwrite requested
        if lbl_path.exists() and not overwrite:
            n_kept += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning("[%d/%d] Cannot read %s — skipped", i+1, len(images), img_path.name)
            n_failed += 1
            continue

        H, W = frame.shape[:2]
        outer_box = core_box = None

        # 2. Try model-assisted annotation
        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                if results and results[0].boxes is not None:
                    detections = {0: [], 1: []}
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        conf   = float(box.conf[0].item())
                        if cls_id in detections and conf >= conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections[cls_id].append((conf, int(x1), int(y1), int(x2), int(y2)))
                    for cls_id in detections:
                        if detections[cls_id]:
                            best = max(detections[cls_id], key=lambda d: d[0])
                            if cls_id == 0:
                                outer_box = best[1:]
                            else:
                                core_box  = best[1:]
            except Exception as e:
                log.warning("[%d/%d] Model inference failed: %s", i+1, len(images), e)

        # 3. Fall back to classical CV for any missing box
        if outer_box is None or core_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if outer_box is None:
                outer_box = _cv_detect_outer(gray)
            if core_box is None:
                core_box = _cv_detect_core(gray, outer_box)

            if outer_box is None or core_box is None:
                log.warning("[%d/%d] CV detection failed for %s", i+1, len(images), img_path.name)
                n_failed += 1
                continue

            if yolo_model is not None:
                n_cv += 1   # model ran but we used CV for at least one class
            else:
                n_cv += 1
        else:
            n_model += 1

        # 4. Write YOLO label
        with open(lbl_path, "w") as f:
            f.write(_yolo_line(0, *outer_box, W, H) + "\n")
            f.write(_yolo_line(1, *core_box,  W, H) + "\n")

        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            log.info("  Annotated %d / %d  (model: %d  cv: %d  kept: %d  failed: %d)",
                     i + 1, len(images), n_model, n_cv, n_kept, n_failed)

    total_labelled = len([p for p in labels_dir.glob("*.txt")])
    log.info("=" * 60)
    log.info("Auto-annotation complete:")
    log.info("  Kept existing : %d", n_kept)
    log.info("  Model-labelled: %d", n_model)
    log.info("  CV-labelled   : %d", n_cv)
    log.info("  Failed        : %d", n_failed)
    log.info("  Total labelled: %d / %d", total_labelled, len(images))
    log.info("=" * 60)
    return total_labelled


# ==============================================================================
# 1. DATASET PREPARATION
#    Splits annotated images into train / val sets and writes data.yaml
# ==============================================================================

def prepare_dataset(size: str, val_split: float = 0.15) -> Path:
    """
    Reads annotated images + labels from auto_annotate.py output, splits
    them 85/15 train/val, copies into training_data/{train,val}/{images,labels}/
    and writes data.yaml.

    Returns the path to data.yaml.
    """
    import random

    src_images = Path(cfg.TRAINING_SOURCES.get(size, ""))
    src_labels = Path(cfg.DATASET_DIR) / "train" / "labels"   # output of auto_annotate.py

    if not src_images.exists():
        log.error("Training images folder not found: %s", src_images)
        sys.exit(1)

    # Gather images that have a matching label file
    all_images = sorted(src_images.glob("*.jpg")) + sorted(src_images.glob("*.png"))
    labelled = [p for p in all_images if (src_labels / (p.stem + ".txt")).exists()]

    if not labelled:
        log.error(
            "No labelled images found.\n"
            "  Images: %s\n"
            "  Labels: %s\n"
            "Run auto_annotate.py first to create labels.",
            src_images, src_labels,
        )
        sys.exit(1)

    log.info("Labelled images found: %d / %d", len(labelled), len(all_images))

    random.seed(42)
    random.shuffle(labelled)
    n_val   = max(1, int(len(labelled) * val_split))
    n_train = len(labelled) - n_val
    splits  = {"train": labelled[:n_train], "val": labelled[n_train:]}

    log.info("Split → train: %d  |  val: %d", n_train, n_val)

    # Create output dirs
    ds_root = Path(cfg.DATASET_DIR)
    for split in ("train", "val"):
        (ds_root / split / "images").mkdir(parents=True, exist_ok=True)
        (ds_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files
    for split, files in splits.items():
        for img_path in files:
            lbl_path = src_labels / (img_path.stem + ".txt")
            dst_img = ds_root / split / "images" / img_path.name
            dst_lbl = ds_root / split / "labels" / (img_path.stem + ".txt")
            if img_path.resolve() != dst_img.resolve():
                shutil.copy2(img_path, dst_img)
            if lbl_path.resolve() != dst_lbl.resolve():
                shutil.copy2(lbl_path, dst_lbl)

    # Write data.yaml
    data_yaml_path = Path(cfg.BASE_DIR) / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.dump(
            {
                "path":  str(ds_root.absolute()),
                "train": str((ds_root / "train" / "images").absolute()),
                "val":   str((ds_root / "val"   / "images").absolute()),
                "nc":    cfg.NUM_CLASSES,
                "names": {k: v for k, v in cfg.CLASS_NAMES.items()},
            },
            f,
        )
    log.info("data.yaml written → %s", data_yaml_path)
    return data_yaml_path


# ==============================================================================
# 2. SYSTEM RESOURCE CHECK
# ==============================================================================

def check_resources(ram_limit_gb: float) -> bool:
    mem          = psutil.virtual_memory()
    swap         = psutil.swap_memory()
    total_gb     = mem.total     / 1e9
    available_gb = mem.available / 1e9
    swap_used_gb = swap.used     / 1e9

    log.info("RAM total   : %.1f GB  |  free: %.1f GB", total_gb, available_gb)
    log.info("Swap in use : %.2f GB", swap_used_gb)
    log.info("CPU cores   : %d", os.cpu_count() or 1)
    log.info("RAM budget  : %.1f GB", ram_limit_gb)

    if swap_used_gb > 1.0:
        log.warning("Swap already %.1f GB — close other apps before training.", swap_used_gb)

    min_free = ram_limit_gb * 0.5
    if available_gb < min_free:
        log.error(
            "Only %.1f GB free — need %.1f GB minimum. Close apps and retry.",
            available_gb, min_free,
        )
        return False

    if available_gb < ram_limit_gb:
        log.warning(
            "Free RAM (%.1f GB) is below budget (%.1f GB) — proceed with caution.",
            available_gb, ram_limit_gb,
        )

    return True


def log_memory():
    try:
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        msg = f"Process RAM: {rss:.2f} GB"
        if torch.backends.mps.is_available():
            mps = torch.mps.driver_allocated_memory() / 1e9
            msg += f"  |  MPS: {mps:.2f} GB"
        elif torch.cuda.is_available():
            cuda = torch.cuda.memory_allocated() / 1e9
            msg += f"  |  CUDA: {cuda:.2f} GB"
        log.info(msg)
    except Exception:
        pass


# ==============================================================================
# 3. DEVICE AUTO-SELECTION
# ==============================================================================

def select_device(ram_limit_gb: float) -> tuple[str, int, int, int]:
    """
    Returns (device_str, imgsz, batch_size, dataloader_workers).

    MPS RAM tiers (Apple Silicon unified memory):
      ≤ 4 GB  →  imgsz=640  batch=2
      ≤ 6 GB  →  imgsz=1024 batch=1
      ≤ 8 GB  →  imgsz=1024 batch=2
      > 8 GB  →  imgsz=1024 batch=4

    AQP1 images are 1024×880, so training at imgsz=1024 matches
    production exactly and avoids the accuracy hit from downscaling.
    """
    if torch.backends.mps.is_available():
        log.info("Apple Silicon MPS detected — hardware acceleration enabled")
        if ram_limit_gb <= 4.0:
            imgsz, batch = 640, 2
        elif ram_limit_gb <= 6.0:
            imgsz, batch = 1024, 1
        elif ram_limit_gb <= 8.0:
            imgsz, batch = 1024, 2
        else:
            imgsz, batch = 1024, 4
        log.info("MPS: imgsz=%d  batch=%d  (RAM budget: %.1f GB)", imgsz, batch, ram_limit_gb)
        return "mps", imgsz, batch, 0   # workers=0 required on MPS

    elif torch.cuda.is_available():
        name   = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("CUDA GPU: %s (%.1f GB VRAM)", name, mem_gb)
        batch   = 16 if mem_gb >= 16 else (8 if mem_gb >= 8 else 4)
        workers = min(4, os.cpu_count() or 4)
        return "0", 1024, batch, workers

    else:
        log.warning("No GPU — CPU only training will be slow (~40+ hours for 50 epochs).")
        return "cpu", 1024, 2, 0


def estimate_time(device: str, epochs: int, imgsz: int,
                  n_train: int, batch: int) -> float:
    iters = n_train / batch
    if device == "mps":
        spi = 0.08 * (imgsz / 640) ** 1.5
    elif device.isdigit():
        spi = 0.02 * (imgsz / 640) ** 1.5
    else:
        spi = 1.5  * (imgsz / 640) ** 1.5
    return (iters * spi * epochs) / 3600


# ==============================================================================
# 4. AUTO-UPDATE config.py PRIMARY_MODEL
# ==============================================================================

def update_primary_model(model_filename: str, timestamp: str) -> None:
    config_path = Path(cfg.BASE_DIR) / "config.py"
    if not config_path.exists():
        return
    text = config_path.read_text()
    text = re.sub(
        r'^PRIMARY_MODEL\s*=\s*".*"',
        f'PRIMARY_MODEL = "{model_filename}"',
        text, count=1, flags=re.MULTILINE,
    )
    text = re.sub(
        r'^MODEL_SOURCE\s*=\s*".*"',
        f'MODEL_SOURCE = "Training run ({timestamp})"',
        text, count=1, flags=re.MULTILINE,
    )
    config_path.write_text(text)
    log.info("config.py updated → PRIMARY_MODEL = \"%s\"", model_filename)


# ==============================================================================
# 5. MAIN TRAINING FUNCTION
# ==============================================================================

def train(
    size:             str   = "S4",
    model_name:       str   = "yolo11n-seg.pt",
    epochs:           int   = 50,
    imgsz:            int   = None,
    batch_size:       int   = None,
    device:           str   = None,
    ram_limit_gb:     float = 4.0,
    val_split:        float = 0.15,
    experiment_name:  str   = "aqp1_optimal",
    project_dir:      str   = "runs/segment",
    skip_dataset:     bool  = False,
    auto_annotate:    bool  = True,
    conf_threshold:   float = 0.60,
) -> None:

    log.info("=" * 70)
    log.info("AQP1 TRAINING  —  size: %s", size)
    log.info("=" * 70)

    # ── Resource check ───────────────────────────────────────────────────────
    if not check_resources(ram_limit_gb):
        log.error("Aborting — insufficient RAM.")
        return

    # ── CPU threads ──────────────────────────────────────────────────────────
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(min(4, cpu_count))
    log.info("PyTorch threads: %d", cpu_count)

    # ── Device ───────────────────────────────────────────────────────────────
    auto_device, auto_imgsz, auto_batch, workers = select_device(ram_limit_gb)
    device     = device     or auto_device
    imgsz      = imgsz      or auto_imgsz
    batch_size = batch_size or auto_batch

    # ── Auto-annotation ───────────────────────────────────────────────────────
    if auto_annotate and not skip_dataset:
        log.info("=" * 70)
        log.info("PHASE 1 — AUTO-ANNOTATION")
        log.info("=" * 70)

        # Use the current best model (if one exists) to assist annotation
        assist_model = None
        latest_pt    = Path(cfg.MODELS_DIR) / "aqp1_latest.pt"
        primary_pt   = Path(cfg.MODELS_DIR) / cfg.PRIMARY_MODEL
        for candidate in [latest_pt, primary_pt]:
            if candidate.exists() and candidate.stat().st_size > 100_000:
                assist_model = str(candidate)
                log.info("Using existing model for annotation assistance: %s", candidate.name)
                break
        if assist_model is None:
            log.info("No existing model found — using classical CV for all images")

        headless_annotate_all(
            size           = size,
            model_path     = assist_model,
            conf_threshold = conf_threshold,
            overwrite      = False,   # never overwrite confirmed human labels
        )
        log.info("=" * 70)
        log.info("PHASE 2 — TRAINING")
        log.info("=" * 70)

    # ── Dataset ──────────────────────────────────────────────────────────────
    if skip_dataset:
        data_yaml = Path(cfg.BASE_DIR) / "data.yaml"
        if not data_yaml.exists():
            log.error("data.yaml not found and --skip-dataset was set.")
            return
    else:
        data_yaml = prepare_dataset(size, val_split=val_split)

    # Count images for time estimate
    ds_root = Path(cfg.DATASET_DIR)
    train_imgs = list((ds_root / "train" / "images").glob("*.jpg")) + \
                 list((ds_root / "train" / "images").glob("*.png"))
    val_imgs   = list((ds_root / "val"   / "images").glob("*.jpg")) + \
                 list((ds_root / "val"   / "images").glob("*.png"))

    log.info("Train images : %d", len(train_imgs))
    log.info("Val images   : %d", len(val_imgs))
    log.info("Device       : %s", device)
    log.info("Base model   : %s", model_name)
    log.info("imgsz        : %d  (AQP1 images: 1024×880)", imgsz)
    log.info("Batch        : %d", batch_size)
    log.info("Epochs       : %d", epochs)
    log.info("RAM budget   : %.1f GB", ram_limit_gb)
    est = estimate_time(device, epochs, imgsz, len(train_imgs), batch_size)
    log.info("Est. runtime : %.1f hours", est)
    log.info("=" * 70)

    # ── Load base model ───────────────────────────────────────────────────────
    log.info("Loading base model: %s", model_name)
    try:
        model = YOLO(model_name)
    except Exception as e:
        log.error("Could not load model: %s", e)
        return

    # ── Checkpoint infrastructure ─────────────────────────────────────────────
    models_dir = Path(cfg.MODELS_DIR)
    models_dir.mkdir(exist_ok=True)

    start_time = datetime.now()
    start_ts   = start_time.strftime("%Y%m%d_%H%M")

    def _ckpt_name(epoch: int, interrupted: bool = False) -> str:
        tag = "_INTERRUPTED" if interrupted else ""
        return f"aqp1_{size}_e{epoch:03d}{tag}_{start_ts}.pt"

    def _emergency_save(signum=None, frame=None):
        """Save whatever is available on SIGINT / SIGTERM."""
        for search in [
            Path(project_dir) / experiment_name,
            Path(project_dir) / (experiment_name + "2"),
        ]:
            for pt in ["last.pt", "best.pt"]:
                candidate = search / "weights" / pt
                if candidate.exists() and candidate.stat().st_size > 100_000:
                    epoch_guess = getattr(_emergency_save, "_last_epoch", 0)
                    dest = models_dir / _ckpt_name(epoch_guess, interrupted=True)
                    shutil.copy2(str(candidate), str(dest))
                    log.warning("[INTERRUPT] Saved → %s", dest.name)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    def _on_epoch_end(trainer):
        """Per-epoch callback: copy weights + log memory."""
        try:
            epoch = int(getattr(trainer, "epoch", 0))
            _emergency_save._last_epoch = epoch
            last_pt = Path(trainer.save_dir) / "weights" / "last.pt"
            if last_pt.exists() and last_pt.stat().st_size > 100_000:
                ckpt_name = _ckpt_name(epoch)
                dest      = models_dir / ckpt_name
                shutil.copy2(str(last_pt), str(dest))
                shutil.copy2(str(last_pt), str(models_dir / "aqp1_latest.pt"))
                log.info("[CHECKPOINT] Epoch %03d → %s", epoch, ckpt_name)
        except Exception as e:
            log.warning("[CHECKPOINT] Could not save: %s", e)
        log_memory()

    model.add_callback("on_train_epoch_end", _on_epoch_end)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    train_args = {
        "data":          str(data_yaml),
        "epochs":        epochs,
        "imgsz":         imgsz,
        "batch":         batch_size,
        "device":        device,
        "workers":       workers,
        "project":       project_dir,
        "name":          experiment_name,

        "patience":      20,
        "save":          True,
        "save_period":   1,
        "val":           True,
        "verbose":       True,
        "plots":         True,
        "seed":          42,

        "optimizer":     "AdamW",
        "lr0":           0.001,
        "lrf":           0.01,
        "weight_decay":  0.0005,
        "warmup_epochs": 3,

        # Augmentation tuned for stationary overhead camera (conveyor belt)
        "hsv_h":         0.01,    # very subtle hue shift (near-greyscale images)
        "hsv_s":         0.3,
        "hsv_v":         0.4,
        "degrees":       5.0,     # slight rotation (camera is fixed but not perfectly level)
        "translate":     0.05,
        "scale":         0.2,
        "shear":         1.0,
        "perspective":   0.0005,
        "flipud":        0.0,     # product is always same orientation on conveyor
        "fliplr":        0.5,
        "mosaic":        0.7,
        "mixup":         0.0,
        "copy_paste":    0.05,

        "iou":           0.45,
        "cos_lr":        True,
        "close_mosaic":  10,
        "cache":         False,   # never cache — saves RAM
        "amp":           True,
        "fraction":      1.0,
    }

    log.info("Starting training…")
    log.info("Checkpoints: models/aqp1_%s_e###_%s.pt", size, start_ts)
    log.info("Press Ctrl+C at any time — current weights will be saved.")
    log.info("=" * 70)

    # ── Train ────────────────────────────────────────────────────────────────
    best_pt = None
    try:
        results  = model.train(**train_args)
        end_time = datetime.now()
        duration = end_time - start_time

        total_min    = int(duration.total_seconds() // 60)
        hours, mins  = total_min // 60, total_min % 60
        duration_str = f"{hours}h{mins:02d}m"
        duration_hr  = duration.total_seconds() / 3600

        log.info("=" * 70)
        log.info("TRAINING COMPLETE — %s", duration_str)
        log.info("=" * 70)

        # ── Locate weights ────────────────────────────────────────────────
        run_dir = Path(results.save_dir) if (results and hasattr(results, "save_dir")) \
                  else Path(project_dir) / experiment_name
        weights_dir = run_dir / "weights"

        yolo_best = weights_dir / "best.pt"
        if yolo_best.exists():
            best_pt = yolo_best
            log.info("best.pt: %s (%.1f MB)", yolo_best, yolo_best.stat().st_size / 1e6)

        if not best_pt:
            fallback = weights_dir / "best_manual.pt"
            model.save(str(fallback))
            best_pt = fallback
            log.warning("best.pt missing — manual save: %s", fallback)

        # ── Read metrics from results.csv ─────────────────────────────────
        actual_epochs   = epochs
        metrics_summary = {}
        results_csv     = run_dir / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                actual_epochs = int(float(last.get("epoch", epochs)))
                metrics_summary = {
                    "final_epoch":      actual_epochs,
                    "mAP50_box":        round(float(last.get("metrics/mAP50(B)",    0)), 4),
                    "mAP50_95_box":     round(float(last.get("metrics/mAP50-95(B)", 0)), 4),
                    "mAP50_mask":       round(float(last.get("metrics/mAP50(M)",    0)), 4),
                    "mAP50_95_mask":    round(float(last.get("metrics/mAP50-95(M)", 0)), 4),
                    "seg_loss_val":     round(float(last.get("val/seg_loss",        0)), 4),
                }
                log.info("Final metrics:")
                for k, v in metrics_summary.items():
                    log.info("  %-22s: %s", k, v)

                seg   = metrics_summary["seg_loss_val"]
                mmask = metrics_summary["mAP50_95_mask"]
                if   seg < 0.5 and mmask > 0.88:
                    quality = "EXCELLENT — ready for production"
                elif seg < 0.8 and mmask > 0.80:
                    quality = "GOOD — acceptable for production"
                elif seg < 1.0 and mmask > 0.75:
                    quality = "FAIR — measurements may have ±5mm error"
                else:
                    quality = "POOR — needs more data or epochs"
                log.info("  %-22s: %s", "quality", quality)

        # ── Save final model ──────────────────────────────────────────────
        final_name = f"aqp1_{size}_{actual_epochs}ep_{duration_str}_{start_ts}.pt"
        final_dest = models_dir / final_name
        shutil.copy2(str(best_pt), str(final_dest))
        shutil.copy2(str(best_pt), str(models_dir / "aqp1_latest.pt"))
        log.info("Saved → models/%s", final_name)
        log.info("Latest pointer → aqp1_latest.pt")

        # ── Summary JSON ──────────────────────────────────────────────────
        summary = {
            "status":          "completed",
            "size":            size,
            "model":           model_name,
            "epochs_trained":  actual_epochs,
            "imgsz":           imgsz,
            "batch_size":      batch_size,
            "device":          device,
            "duration_str":    duration_str,
            "duration_hours":  round(duration_hr, 2),
            "timestamp_start": str(start_time),
            "timestamp_end":   str(end_time),
            "train_images":    len(train_imgs),
            "val_images":      len(val_imgs),
            "metrics":         metrics_summary,
            "weights":         str(final_dest),
        }
        summary_path = run_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Summary → %s", summary_path)

        # ── Auto-update config.py ─────────────────────────────────────────
        update_primary_model(final_name, start_ts)

        log.info("=" * 70)
        log.info("SUCCESS")
        log.info("Model  →  models/%s", final_name)
        log.info("config.py PRIMARY_MODEL updated automatically")
        log.info("Next   →  python main.py  (new model loads on startup)")
        log.info("=" * 70)

    except KeyboardInterrupt:
        log.warning("Training interrupted — partial weights saved.")
    except Exception as e:
        log.error("Training failed: %s", e)
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log_memory()


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AQP1 YOLO segmentation training — detects overlay_outer + dark_core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical workflow:
  1. python auto_annotate.py --size S4        # label images
  2. python train.py --size S4               # train model
  3. python main.py                          # run server with new model

Checkpoint naming:
  Every epoch  →  models/aqp1_S4_e012_20260413_1430.pt
  Final best   →  models/aqp1_S4_50ep_4h12m_20260413_1430.pt
  On Ctrl+C    →  models/aqp1_S4_e012_INTERRUPTED_20260413_1430.pt

Examples:
  python train.py --size S4
  python train.py --size S4 --epochs 100 --model yolo11s-seg.pt
  python train.py --size S4 --ram-limit 8.0
  python train.py --size S4 --device 0          # force CUDA
  python train.py --size S4 --skip-dataset      # skip split (reuse existing)
        """
    )
    parser.add_argument("--size",         default="S4",
                        help="Product size: S4 / S5 / S6")
    parser.add_argument("--model",        default="yolo11n-seg.pt",
                        help="Base YOLO model (yolo11n-seg.pt / yolo11s-seg.pt)")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--imgsz",        type=int,   default=None,
                        help="Image size override (default: auto from device)")
    parser.add_argument("--batch",        type=int,   default=None,
                        help="Batch size override (default: auto)")
    parser.add_argument("--device",       default=None,
                        help="Device: mps | cpu | 0 (CUDA). Auto-detected if omitted.")
    parser.add_argument("--ram-limit",    type=float, default=4.0,
                        help="RAM budget in GB (default: 4.0)")
    parser.add_argument("--val-split",    type=float, default=0.15,
                        help="Fraction of data held out for validation (default: 0.15)")
    parser.add_argument("--name",         default="aqp1_optimal",
                        help="Experiment name for runs/ subfolder")
    parser.add_argument("--skip-dataset",      action="store_true",
                        help="Skip dataset split (reuse existing training_data/)")
    parser.add_argument("--no-auto-annotate", action="store_true",
                        help="Skip auto-annotation (use existing labels only)")
    parser.add_argument("--conf",             type=float, default=0.60,
                        help="Min model confidence to accept an auto-annotation (default: 0.60)")
    args = parser.parse_args()

    train(
        size            = args.size.upper(),
        model_name      = args.model,
        epochs          = args.epochs,
        imgsz           = args.imgsz,
        batch_size      = args.batch,
        device          = args.device,
        ram_limit_gb    = args.ram_limit,
        val_split       = args.val_split,
        experiment_name = args.name,
        skip_dataset    = args.skip_dataset,
        auto_annotate   = not args.no_auto_annotate,
        conf_threshold  = args.conf,
    )
