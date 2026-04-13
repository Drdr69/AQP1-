"""
AQP1 Project Configuration
===========================
Single source of truth for all paths, sizes, calibration values, and defaults.
"""

import os
from datetime import datetime, timedelta

# ==============================================================================
# DIRECTORY & PATH MANAGEMENT
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Network Source
NETWORK_BASE      = r"\\BlissOC110.bcc.pg.com\Images\PBQA\Overlay"
NETWORK_SUBFOLDER = "VISAQP1"

# Local Storage
CACHE_DIR    = os.path.join(BASE_DIR, "static", "cache")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
DATASET_DIR  = os.path.join(BASE_DIR, "training_data")

# ==============================================================================
# ENGINE DEFAULTS
# ==============================================================================

DEFAULT_SCAN_INTERVAL = 3600   # 1 hour between automated measurements
DEFAULT_BATCH_SIZE    = 8      # images shown per results table

# Size tokens — used to scan stream folders regardless of naming prefix
# e.g. "CLPN_S4[]", "SHPN_S6[]", etc.
SIZE_TOKENS = ["S4", "S5", "S6", "S7"]

# ==============================================================================
# PRODUCT SPECIFICATIONS  (reference measurements per size, all in mm)
# ==============================================================================

SIZE_SPECS = {
    "S4": {
        "total_width_mm":      120.0,
        "dark_area_width_mm":   80.0,
        "total_length_mm":     370.0,
        "dark_area_height_mm": 340.0,
        # Pixel-to-mm calibration — derived from 1024×880 image:
        #   horizontal: 120 mm ÷ ~512 product px ≈ 0.234 mm/px
        #   vertical:   370 mm ÷  880 px          ≈ 0.420 mm/px
        "scale_x_mm_per_px": 120.0 / 512.0,
        "scale_y_mm_per_px": 370.0 / 880.0,
        "image_w_px": 1024,
        "image_h_px": 880,
    },
    "S5": {},   # TODO: populate once S5 spec is confirmed
    "S6": {},   # TODO: populate once S6 spec is confirmed
}

# Default tolerance (±mm) used for pass/fail evaluation
DEFAULT_TOLERANCE_MM = 2.0

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Updated automatically by train.py after each successful training run
PRIMARY_MODEL = "aqp1_S4_46ep_0h20m_20260413_1225.pt"
MODEL_SOURCE = "Training run (20260413_1225)"

# YOLO class definitions (two regions annotated per image)
CLASS_NAMES = {
    0: "overlay_outer",   # full product boundary  →  total width (120mm) + height (370mm)
    1: "dark_core",       # inner dark region      →  dark width  ( 80mm) + height (340mm)
}
NUM_CLASSES = len(CLASS_NAMES)

# ==============================================================================
# TRAINING DATA PATHS  (per size)
# ==============================================================================

TRAINING_SOURCES = {
    "S4": os.path.join(BASE_DIR, "Training Data S4", "AQP1 S4 images"),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def ensure_dirs():
    """Create all required local directories if they don't exist."""
    for d in [CACHE_DIR, MODELS_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)


def get_date_folder(delta: int = 0) -> str:
    """
    Return the date-level UNC directory for today (or delta days ago).

    Structure: NETWORK_BASE / YYYY_MM / DD
    Example:   \\BlissOC110.bcc.pg.com\Images\PBQA\Overlay\2026_04\13
    """
    active_date = datetime.now() - timedelta(days=delta)
    year_month  = active_date.strftime("%Y_%m")   # "2026_04"
    day         = active_date.strftime("%d")        # "13"
    return os.path.join(NETWORK_BASE, year_month, day)


def get_network_path(size: str = "S4", delta: int = 0) -> str:
    """
    Build the full UNC path for a given size and date offset.

    Example: \\BlissOC110.bcc.pg.com\Images\PBQA\Overlay\2026_04\13\CLPN_S4[]\VISAQP1
    """
    date_base   = get_date_folder(delta)
    size_folder = f"CLPN_{size}[]"
    return os.path.join(date_base, size_folder, NETWORK_SUBFOLDER)
