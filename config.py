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
NETWORK_BASE      = r"\\BlissOC110.bcc.pg.com\Images\PBQA\Processed"
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
SIZE_TOKENS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

# ==============================================================================
# PRODUCT SPECIFICATIONS  (reference measurements per size, all in mm)
#
#  Class 0  overlay_outer  — full pad boundary
#      total_width_mm  = 120      total_length_mm   = 370
#
#  Class 1  pad_core       — inner core region (dark area)
#      core_width_mm   = 110      core_height_mm    = 340
#
#  Class 2  s_wrap         — S-Wrap zone at top of pad
#      s_wrap_height_mm = 15      (width = same as overlay_outer)
#
# ==============================================================================

SIZE_SPECS = {
    "S4": {
        # Overlay outer
        "total_width_mm":    120.0,
        "total_length_mm":   370.0,

        # Pad core (dark inner region)
        "core_width_mm":     110.0,
        "core_height_mm":    340.0,

        # S-Wrap (top and bottom zones — same target height)
        "s_wrap_height_mm":   15.0,   # applies to both top and bottom

        # Pixel-to-mm calibration falls back to these when outer box not detected.
        # In practice the self-calibrating path (120 / outer_w_px) is used instead.
        "scale_x_mm_per_px": 120.0 / 512.0,
        "scale_y_mm_per_px": 370.0 / 880.0,

        "image_w_px": 1024,
        "image_h_px": 880,
    },
    "S5": {},
    "S6": {},
    "S7": {},
}

# Default tolerance (±mm) used for pass/fail evaluation
DEFAULT_TOLERANCE_MM = 2.0

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Updated automatically by train.py after each successful training run
PRIMARY_MODEL = ""   # set automatically by train.py after training
MODEL_SOURCE  = ""

# YOLO class definitions — 3 regions annotated per image
CLASS_NAMES = {
    0: "overlay_outer",   # full pad boundary       → total width (120mm) + length (370mm)
    1: "pad_core",        # inner core region       → core width  (110mm) + height (340mm)
    2: "s_wrap",          # S-Wrap zone at top      → s_wrap height (15mm)
}
NUM_CLASSES = len(CLASS_NAMES)

# ==============================================================================
# TRAINING DATA PATHS  (per size)
# ==============================================================================

TRAINING_SOURCES = {
    "S4": os.path.join(BASE_DIR, "Training Data S4"),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def ensure_dirs():
    """Create all required local directories if they don't exist."""
    for d in [CACHE_DIR, MODELS_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)


def get_date_folder(delta: int = 0) -> str:
    active_date = datetime.now() - timedelta(days=delta)
    return os.path.join(NETWORK_BASE, active_date.strftime("%Y_%m_%d"))


def get_network_path(size: str = "S4", delta: int = 0) -> str:
    date_base   = get_date_folder(delta)
    size_folder = f"CLPN_{size}[]"
    return os.path.join(date_base, size_folder, NETWORK_SUBFOLDER)
