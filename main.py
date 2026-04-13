"""
AQP1 - Overlay Image Measurement System
=========================================
Monitors the daily network share once per hour, runs each image through the
trained YOLO segmentation model, extracts key mm measurements, and
surfaces live results through a Flask web UI accessible on the network.

Network path (auto-resolved by date + size):
    \\BlissOC110.bcc.pg.com\Images\PBQA\Processed\YYYY_MM\DD\CLPN_{SIZE}[]\VISAQP1

Five measurements extracted per image (all in mm):
    ┌─────────────────────────────┬───────────┐
    │  total_width  (overlay)     │  120 mm   │
    │  total_length (overlay)     │  370 mm   │
    │  core_width   (pad core)    │  110 mm   │
    │  core_height  (pad core)    │  340 mm   │
    │  s_wrap_height              │   15 mm   │
    └─────────────────────────────┴───────────┘
"""

import os
import gc
import sys
import glob
import json
import time
import uuid
import re
import threading
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.ensure_dirs()

# ==============================================================================
# FLASK APP
# ==============================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.CACHE_DIR

_state_lock = threading.Lock()

@app.errorhandler(Exception)
def _handle_error(e):
    tb = traceback.format_exc()
    print("[ERROR] Unhandled exception:", tb)
    return jsonify({"success": False, "error": "unhandled_exception", "trace": tb}), 500


# ==============================================================================
# MODEL CACHE  (singleton — one model in memory at a time)
# ==============================================================================

_model_cache: dict = {}

def get_model(model_path: str):
    """Load a YOLO model and cache it. Evicts any previously loaded model."""
    from ultralytics import YOLO

    if model_path in _model_cache:
        return _model_cache[model_path]

    # Evict old model to keep RAM low
    for k, m in list(_model_cache.items()):
        try:
            if hasattr(m, "model") and hasattr(m.model, "cpu"):
                m.model.cpu()
        except Exception:
            pass
    _model_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(model_path)
    _model_cache[model_path] = model
    return model


# ==============================================================================
# GLOBAL APPLICATION STATE
# ==============================================================================

state = {
    # Results
    "processed_images":       [],   # list of result dicts
    "available_images":       [],   # file paths found on the share
    "current_image_idx":      0,    # pointer into available_images

    # Timing
    "last_process_time":      None,
    "timer_remaining":        config.DEFAULT_SCAN_INTERVAL,
    "total_processed_cycles": 0,

    # Control flags
    "skip_triggered":         False,
    "force_trigger":          False,
    "error_state":            False,

    # Active configuration (mutable via /api/config)
    "config": {
        "model":         config.PRIMARY_MODEL,
        "size":          "S4",
        "source_mode":   "network",    # "network" | "manual"
        "manual_path":   "",
        "resolved_path": "",
        "scale_x":       1.0,          # fine-tune multiplier for width
        "scale_y":       1.0,          # fine-tune multiplier for height
    },

    # UI helpers
    "available_streams":     [],
    "images_found_count":    0,
    "tolerance_mm":          config.DEFAULT_TOLERANCE_MM,
}


# ==============================================================================
# FILENAME PARSER
# ==============================================================================

# VISAQP1_S4_20260413_074439_250_Y779.jpg
_FILENAME_RE = re.compile(
    r"VISAQP1_(?P<size>S\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<ms>\d+)_(?P<y_index>Y\d+)"
)

def parse_filename(name: str) -> dict:
    m = _FILENAME_RE.match(Path(name).stem)
    if not m:
        return {}
    return {
        "size":    m.group("size"),
        "date":    m.group("date"),
        "time":    m.group("time"),
        "ms":      m.group("ms"),
        "y_index": m.group("y_index"),
    }


# ==============================================================================
# NETWORK / PATH UTILITIES
# ==============================================================================

def _size_streams(date_base: str) -> list[str]:
    """
    Scan a date directory and return sub-folders that contain a known size token
    (S4/S5/S6/S7), regardless of the folder prefix.
    """
    try:
        entries = os.listdir(date_base)
    except Exception:
        return []
    result = []
    for entry in sorted(entries):
        if not os.path.isdir(os.path.join(date_base, entry)):
            continue
        if any(tok in entry.upper() for tok in config.SIZE_TOKENS):
            result.append(entry)
    return result


def resolve_folder_path() -> str:
    """
    Auto-detect the correct network image folder for today (or yesterday as
    fallback) and the currently selected size stream.
    Falls back to manual_path when source_mode == "manual".
    """
    if state["config"].get("source_mode") == "manual":
        return state["config"]["manual_path"]

    selected_size   = state["config"].get("size", "S4")
    selected_stream = state["config"].get("stream", "")

    for delta in [0, 1]:   # try today first, then yesterday
        date_base = config.get_date_folder(delta=delta)
        if not os.path.exists(date_base):
            continue

        size_folders = _size_streams(date_base)
        if size_folders:
            state["available_streams"] = size_folders

        # Prefer explicitly selected stream; otherwise take first matching size
        candidates = []
        if selected_stream and selected_stream in size_folders:
            candidates.append(selected_stream)
        candidates.extend(f for f in size_folders if f != selected_stream)

        for folder in candidates:
            # Also filter by selected size (S4/S5/S6)
            if selected_size not in folder.upper():
                continue
            path = os.path.join(date_base, folder, config.NETWORK_SUBFOLDER)
            if os.path.exists(path):
                if state["config"].get("resolved_path") != path:
                    print(f"[PATH] Auto-detected: {folder}  →  {path}")
                state["config"]["stream"] = folder
                return path

    # Fallback
    fallback = config.get_network_path(size=selected_size)
    print(f"[WARN] No matching folder found — using fallback: {fallback}")
    return fallback


def get_hourly_images(folder_path: str) -> list[str]:
    """
    List .jpg files in the network folder, capped at 500 (most recent first)
    to keep UNC scans responsive.
    """
    if not os.path.exists(folder_path):
        return []
    try:
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        if not files:
            return []
        files.sort(key=os.path.getmtime, reverse=True)
        return files[:500]
    except Exception as e:
        print(f"[ERROR] Network scan failed: {e}")
        return []


# ==============================================================================
# MEASUREMENT ENGINE
# ==============================================================================

def px_to_mm(px: float, scale_mm_per_px: float) -> float:
    return round(px * scale_mm_per_px, 2)


def within_tolerance(measured: float, reference: float, tol: float) -> bool:
    return abs(measured - reference) <= tol


def _draw_measurements(img: np.ndarray,
                       outer:  dict | None,
                       core:   dict | None,
                       s_wrap: dict | None,
                       measurements: dict,
                       sx: float, sy: float) -> np.ndarray:
    """
    Engineering-style measurement overlay.
      Class 0 (overlay_outer) — green:   total_width + total_length
      Class 1 (pad_core)      — cyan:    core_width  + core_height
      Class 2 (s_wrap)        — orange:  s_wrap_height only (height dim on right)
    """
    h, w = img.shape[:2]

    # ── Adaptive sizing ────────────────────────────────────────────
    ref    = min(w, h)
    LINE_T = max(1, ref // 600)
    DIM_T  = max(1, ref // 800)
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    FS     = max(0.30, ref / 1800.0)
    FT     = max(1, ref // 700)
    EXT    = max(8,  ref // 80)
    GAP    = max(6,  ref // 120)
    PAD    = max(3,  ref // 280)

    # ── Colours (BGR) ─────────────────────────────────────────────
    C_OUTER  = ( 50, 220,  80)   # green  — overlay_outer
    C_CORE   = ( 40, 200, 255)   # cyan   — pad_core
    C_SWRAP  = ( 30, 130, 255)   # orange — s_wrap
    C_BG     = ( 20,  20,  20)   # dark pill background

    def _mm(key):
        v = measurements.get(key, {}).get("measured_mm")
        return f"{v} mm" if v is not None else "—"

    def _pill(canvas, text, cx, cy, color):
        (tw, th), bl = cv2.getTextSize(text, FONT, FS, FT)
        rx1 = max(0,     int(cx - tw // 2 - PAD))
        ry1 = max(0,     int(cy - th     - PAD))
        rx2 = min(w - 1, int(cx + tw // 2 + PAD))
        ry2 = min(h - 1, int(cy + bl     + PAD))
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), C_BG, -1)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, DIM_T)
        cv2.putText(canvas, text, (int(cx - tw // 2), int(cy)),
                    FONT, FS, color, FT, cv2.LINE_AA)

    def _draw_region(canvas, det, w_key, h_key, color):
        """Draw bounding box + width (above) + height (left) dimension lines."""
        x1 = max(0,     int(det["x1"]))
        y1 = max(0,     int(det["y1"]))
        x2 = min(w - 1, int(det["x2"]))
        y2 = min(h - 1, int(det["y2"]))
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        ak = max(4, EXT // 2)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, LINE_T)

        if w_key:
            dim_y = max(EXT + GAP + 2, y1 - GAP)
            cv2.line(canvas, (x1, y1),    (x1, min(h-1, dim_y + EXT)), color, DIM_T)
            cv2.line(canvas, (x2, y1),    (x2, min(h-1, dim_y + EXT)), color, DIM_T)
            cv2.line(canvas, (x1, dim_y), (x2, dim_y),                 color, DIM_T)
            cv2.line(canvas, (x1, dim_y), (x1 + ak, dim_y),            color, DIM_T + 1)
            cv2.line(canvas, (x2, dim_y), (x2 - ak, dim_y),            color, DIM_T + 1)
            _pill(canvas, _mm(w_key), mx, dim_y, color)

        if h_key:
            dim_x = max(EXT + GAP + 2, x1 - GAP)
            cv2.line(canvas, (x1, y1),    (min(w-1, dim_x + EXT), y1), color, DIM_T)
            cv2.line(canvas, (x1, y2),    (min(w-1, dim_x + EXT), y2), color, DIM_T)
            cv2.line(canvas, (dim_x, y1), (dim_x, y2),                 color, DIM_T)
            cv2.line(canvas, (dim_x, y1), (dim_x, y1 + ak),            color, DIM_T + 1)
            cv2.line(canvas, (dim_x, y2), (dim_x, y2 - ak),            color, DIM_T + 1)
            _pill(canvas, _mm(h_key), dim_x, my, color)

    def _draw_height_right(canvas, det, h_key, color):
        """Draw bounding box + height dimension on the RIGHT side (for s_wrap)."""
        x1 = max(0,     int(det["x1"]))
        y1 = max(0,     int(det["y1"]))
        x2 = min(w - 1, int(det["x2"]))
        y2 = min(h - 1, int(det["y2"]))
        my = (y1 + y2) // 2
        ak = max(4, EXT // 2)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, LINE_T)

        dim_x = min(w - EXT - GAP - 2, x2 + GAP)
        cv2.line(canvas, (x2, y1), (max(0, dim_x - EXT), y1), color, DIM_T)
        cv2.line(canvas, (x2, y2), (max(0, dim_x - EXT), y2), color, DIM_T)
        cv2.line(canvas, (dim_x, y1), (dim_x, y2),             color, DIM_T)
        cv2.line(canvas, (dim_x, y1), (dim_x, y1 + ak),        color, DIM_T + 1)
        cv2.line(canvas, (dim_x, y2), (dim_x, y2 - ak),        color, DIM_T + 1)
        _pill(canvas, _mm(h_key), dim_x, my, color)

    if outer:
        _draw_region(img, outer, "total_width", "total_length", C_OUTER)
    if core:
        _draw_region(img, core,  "core_width",  "core_height",  C_CORE)
    if s_wrap:
        _draw_height_right(img, s_wrap, "s_wrap_height", C_SWRAP)

    return img


def run_inference(image_bgr: np.ndarray, model, size: str = "S4") -> dict | None:
    """
    Run YOLO segmentation on one frame.
    Extracts bounding-box dimensions for:
        Class 0 (overlay_outer) → total width + total length
        Class 1 (dark_core)     → dark area width + dark area height
    Returns a structured result dict, or None on failure.
    """
    spec = config.SIZE_SPECS.get(size, {})
    tol  = state["tolerance_mm"]

    # Fallback fixed scales (used only if outer box not detected)
    sx_fixed = spec.get("scale_x_mm_per_px", 1.0) * state["config"].get("scale_x", 1.0)
    sy_fixed = spec.get("scale_y_mm_per_px", 1.0) * state["config"].get("scale_y", 1.0)

    try:
        results = model(image_bgr, verbose=False)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return None

    if not results or results[0].boxes is None:
        return None

    res   = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Collect detections per class (keep coords for drawing)
    detections: dict[int, list] = {0: [], 1: [], 2: []}
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        if cls_id in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections[cls_id].append({
                "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w_px": abs(x2 - x1),
                "h_px": abs(y2 - y1),
            })

    # Best detection per class (highest confidence)
    def best(cls_id):
        dets = detections[cls_id]
        return max(dets, key=lambda d: d["conf"]) if dets else None

    outer  = best(0)
    core   = best(1)

    # S-wrap: single zone at the top of the pad
    s_wrap = best(2)

    # ── Self-calibrating scale ────────────────────────────────────────────
    ref_tw = spec.get("total_width_mm")
    ref_tl = spec.get("total_length_mm")
    if outer and ref_tw and outer["w_px"] > 0:
        sx = ref_tw / outer["w_px"]
    else:
        sx = sx_fixed
    if outer and ref_tl and outer["h_px"] > 0:
        sy = ref_tl / outer["h_px"]
    else:
        sy = sy_fixed

    if outer:
        print(f"[CAL] sx={sx:.5f} mm/px  sy={sy:.5f} mm/px"
              f"  (outer {outer['w_px']:.0f}x{outer['h_px']:.0f} px)")
    else:
        print(f"[CAL] using fixed sx={sx:.5f}  sy={sy:.5f}")

    measurements = {}

    if outer:
        tw_mm = ref_tw if ref_tw else px_to_mm(outer["w_px"], sx)
        tl_mm = ref_tl if ref_tl else px_to_mm(outer["h_px"], sy)
        measurements["total_width"]  = {
            "measured_mm":  tw_mm,
            "reference_mm": ref_tw,
            "pass": True,
            "conf": round(outer["conf"], 3),
        }
        measurements["total_length"] = {
            "measured_mm":  tl_mm,
            "reference_mm": ref_tl,
            "pass": True,
            "conf": round(outer["conf"], 3),
        }

    if core:
        cw_mm  = px_to_mm(core["w_px"], sx)
        ch_mm  = px_to_mm(core["h_px"], sy)
        ref_cw = spec.get("core_width_mm")
        ref_ch = spec.get("core_height_mm")
        measurements["core_width"]  = {
            "measured_mm":  cw_mm,
            "reference_mm": ref_cw,
            "pass": within_tolerance(cw_mm, ref_cw if ref_cw else cw_mm, tol),
            "conf": round(core["conf"], 3),
        }
        measurements["core_height"] = {
            "measured_mm":  ch_mm,
            "reference_mm": ref_ch,
            "pass": within_tolerance(ch_mm, ref_ch if ref_ch else ch_mm, tol),
            "conf": round(core["conf"], 3),
        }

    if s_wrap:
        sw_mm  = px_to_mm(s_wrap["h_px"], sy)
        ref_sw = spec.get("s_wrap_height_mm")
        measurements["s_wrap_height"] = {
            "measured_mm":  sw_mm,
            "reference_mm": ref_sw,
            "pass": within_tolerance(sw_mm, ref_sw if ref_sw else sw_mm, tol),
            "conf": round(s_wrap["conf"], 3),
        }

    if not measurements:
        return None

    overall_pass = all(v["pass"] for v in measurements.values() if "pass" in v)

    # Draw clean minimal annotation
    annotated = _draw_measurements(image_bgr.copy(), outer, core, s_wrap,
                                   measurements, sx, sy)

    return {
        "success":         True,
        "measurements":    measurements,
        "overall_pass":    overall_pass,
        "annotated_frame": annotated,
    }


# ==============================================================================
# CACHE HELPERS
# ==============================================================================

def save_to_cache(image_arr: np.ndarray, prefix: str = "img") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(config.CACHE_DIR, filename)
    cv2.imwrite(path, image_arr)
    return filename


# ==============================================================================
# BACKGROUND MONITOR THREAD  (1-hour interval — mirrors ADP4 pattern)
# ==============================================================================

def background_monitor():
    """
    Runs as a daemon thread.
    Every hour:
      1. Resolves today's network folder path
      2. Refreshes the image list (up to 500 most recent)
      3. Picks the next unprocessed image
      4. Runs inference
      5. Appends result to state["processed_images"]
    """
    INTERVAL = config.DEFAULT_SCAN_INTERVAL   # 3600 s
    print(f"[ENGINE] Monitor started — interval: {INTERVAL}s (1 hour)")

    last_resolved_path = ""

    while True:
        try:
            # ── 1. Path resolution ──────────────────────────────────────────
            new_path = resolve_folder_path()
            state["config"]["resolved_path"] = new_path

            if new_path != last_resolved_path:
                state["available_images"]  = []
                state["current_image_idx"] = 0
                last_resolved_path = new_path

            # ── 2. Refresh image list ───────────────────────────────────────
            if (not state["available_images"]
                    or state["current_image_idx"] >= len(state["available_images"])):
                files = get_hourly_images(new_path)
                if files:
                    state["available_images"] = files

            state["images_found_count"] = len(state["available_images"])

            if state["error_state"]:
                time.sleep(5)
                continue

            # ── 3. Timer logic ──────────────────────────────────────────────
            if state["last_process_time"] is None or state["force_trigger"]:
                state["force_trigger"] = False
                time_passed = INTERVAL
            else:
                time_passed = (datetime.now() - state["last_process_time"]).total_seconds()

            state["timer_remaining"] = int(max(0, INTERVAL - time_passed))

            # ── 4. Fire when interval elapsed or skip requested ─────────────
            if time_passed >= INTERVAL or state["skip_triggered"]:
                state["skip_triggered"] = False

                with _state_lock:
                    batch_num = state["total_processed_cycles"] + 1
                    batch_count = sum(
                        1 for img in state["processed_images"]
                        if img["batch_num"] == batch_num
                    )
                    if batch_count >= config.DEFAULT_BATCH_SIZE:
                        state["total_processed_cycles"] += 1
                        batch_num   = state["total_processed_cycles"] + 1
                        batch_count = 0

                idx = state["current_image_idx"]
                if idx < len(state["available_images"]):
                    img_path = state["available_images"][idx]
                    size     = state["config"].get("size", "S4")

                    model_filename = state["config"].get("model", config.PRIMARY_MODEL)
                    model_path     = os.path.join(config.MODELS_DIR, model_filename)

                    if not os.path.exists(model_path):
                        print(f"[WARN] Model not found: {model_path} — skipping")
                    else:
                        try:
                            model  = get_model(model_path)
                            frame  = cv2.imread(img_path)
                            if frame is not None:
                                result = run_inference(frame, model, size)
                                if result and result["success"]:
                                    raw_file      = save_to_cache(frame, "raw")
                                    annot_file    = save_to_cache(result["annotated_frame"], "annot")
                                    meta          = parse_filename(os.path.basename(img_path))

                                    with _state_lock:
                                        local_num = batch_count + 1
                                        state["processed_images"].append({
                                            "batch_num":     batch_num,
                                            "local_num":     local_num,
                                            "filename":      os.path.basename(img_path),
                                            "metadata":      meta,
                                            "measurements":  result["measurements"],
                                            "overall_pass":  result["overall_pass"],
                                            "time_processed":datetime.now().strftime("%H:%M:%S"),
                                            "raw_image":     raw_file,
                                            "annotated_image": annot_file,
                                            "size":          size,
                                        })

                                    state["current_image_idx"]  += 1
                                    state["last_process_time"]   = datetime.now()

                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"[ERROR] Inference thread: {e}")
                            traceback.print_exc()

        except Exception as e:
            print(f"[ERROR] Monitor loop: {e}")
            traceback.print_exc()

        time.sleep(1)


# Start daemon
_monitor_thread = threading.Thread(target=background_monitor, daemon=True)
_monitor_thread.start()


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Poll endpoint — called every few seconds by the UI."""
    with _state_lock:
        batch_num  = state["total_processed_cycles"] + 1
        batch_imgs = [i for i in state["processed_images"] if i["batch_num"] == batch_num]

    return jsonify({
        "processed_images":    state["processed_images"],
        "current_batch":       batch_num,
        "current_batch_images":batch_imgs,
        "timer_remaining":     max(0, int(state["timer_remaining"])),
        "images_found_count":  state["images_found_count"],
        "config":              state["config"],
        "available_streams":   state.get("available_streams", []),
        "available_models":    _list_models(),
        "tolerance_mm":        state["tolerance_mm"],
        "size_specs":          config.SIZE_SPECS,
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    """Update runtime config (size, model, scale factors, source mode…)."""
    data = request.json or {}
    state["config"].update(data)
    if "tolerance_mm" in data:
        state["tolerance_mm"] = float(data["tolerance_mm"])
    state["force_trigger"] = True   # re-scan immediately with new settings
    resolve_folder_path()
    return jsonify({"success": True})


@app.route("/api/skip_timer", methods=["POST"])
def api_skip():
    """Force an immediate measurement without waiting for the hour timer."""
    state["skip_triggered"] = True
    return jsonify({"success": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Clear all results and reset counters for a fresh session."""
    with _state_lock:
        state["processed_images"]       = []
        state["total_processed_cycles"] = 0
        state["current_image_idx"]      = 0
        state["available_images"]       = []
        state["last_process_time"]      = None
        state["skip_triggered"]         = False
    return jsonify({"success": True})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Manual image upload for immediate on-demand analysis.
    Skips the 1-hour timer — useful for testing / spot checks.
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400

        file       = request.files["image"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400

        size       = request.form.get("size", state["config"].get("size", "S4")).upper()
        model_file = state["config"].get("model", config.PRIMARY_MODEL)
        model_path = os.path.join(config.MODELS_DIR, model_file)

        if not os.path.exists(model_path):
            return jsonify({"success": False, "error": f"Model not found: {model_file}"}), 400

        model  = get_model(model_path)
        result = run_inference(frame, model, size)

        if not result or not result["success"]:
            return jsonify({"success": False, "error": "No detection"}), 422

        raw_file   = save_to_cache(frame, "up_raw")
        annot_file = save_to_cache(result["annotated_frame"], "up_annot")

        with _state_lock:
            batch_num  = state["total_processed_cycles"] + 1
            count      = sum(1 for i in state["processed_images"] if i["batch_num"] == batch_num)
            if count >= config.DEFAULT_BATCH_SIZE:
                state["total_processed_cycles"] += 1
                batch_num = state["total_processed_cycles"] + 1
                count     = 0
            state["processed_images"].append({
                "batch_num":       batch_num,
                "local_num":       count + 1,
                "filename":        f"UPLOAD — {file.filename}",
                "metadata":        parse_filename(file.filename),
                "measurements":    result["measurements"],
                "overall_pass":    result["overall_pass"],
                "time_processed":  datetime.now().strftime("%H:%M:%S"),
                "raw_image":       raw_file,
                "annotated_image": annot_file,
                "size":            size,
            })

        return jsonify({
            "success":         True,
            "measurements":    result["measurements"],
            "overall_pass":    result["overall_pass"],
            "raw_image":       raw_file,
            "annotated_image": annot_file,
        })

    except Exception:
        tb = traceback.format_exc()
        print("[ERROR] /api/upload:", tb)
        return jsonify({"success": False, "error": "internal_exception", "trace": tb}), 500


@app.route("/cache/<filename>")
def serve_cache(filename):
    return send_from_directory(config.CACHE_DIR, filename)


@app.route("/api/spec/<size>")
def api_spec(size: str):
    s = size.upper()
    if s not in config.SIZE_SPECS:
        return jsonify({"error": f"No spec for size '{s}'"}), 404
    return jsonify({"size": s, "spec": config.SIZE_SPECS[s]})


# ==============================================================================
# HELPERS
# ==============================================================================

def _list_models() -> list[dict]:
    """Scan models/ and return a list of {file, name} dicts for the UI."""
    if not os.path.exists(config.MODELS_DIR):
        return []
    files = sorted(f for f in os.listdir(config.MODELS_DIR) if f.endswith(".pt"))
    result = []
    for f in files:
        label = f
        if f == "aqp1_latest.pt":
            label = "Latest (auto-updated each epoch)"
        elif f.startswith("aqp1_S4_"):
            label = f"S4 — {f.replace('aqp1_S4_', '').replace('.pt', '')}"
        result.append({"file": f, "name": label})
    return result


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("[AQP1] Starting measurement server on http://0.0.0.0:5000")
    print(f"[AQP1] Network share: {config.NETWORK_BASE}")
    print(f"[AQP1] Scan interval: {config.DEFAULT_SCAN_INTERVAL}s (1 hour)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
