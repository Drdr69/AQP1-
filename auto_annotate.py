"""
AQP1 Auto-Annotation Tool
==========================
Fixed overhead camera + consistent product position means annotations are
nearly identical across all images.

Workflow
---------
  Image 1  : auto-detection pre-fills both boxes.  Adjust if needed, ENTER to confirm.
  Image 2+ : last confirmed positions are re-used automatically (template propagation).
             Just press ENTER to accept, or drag a corner if something looks off.

Each confirmed annotation saves TWO measurements in one box:
  Class 0  overlay_outer  (cyan)   →  total width  (120 mm)  +  total length  (370 mm)
  Class 1  dark_core      (green)  →  dark width   ( 80 mm)  +  dark height   (340 mm)

Controls
---------
  LEFT CLICK   drag nearest corner (snap distance 25 px)
  TAB          switch active region (outer ↔ core)
  ENTER / C    save labels and advance
  A            re-run auto-detection on this image (overrides template)
  S            skip image (keep existing label if present)
  R            restore template positions on this image
  Q / ESC      quit  (prints resume index)

Usage
------
  python auto_annotate.py --size S4
  python auto_annotate.py --size S4 --start 40     # resume
  python auto_annotate.py --images path/ --labels path/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ──────────────────────────────────────────────────────────────────────────────
# Visual constants
# ──────────────────────────────────────────────────────────────────────────────
COLORS     = {0: (0, 220, 255), 1: (50, 255, 50)}   # cyan, green
FONT       = cv2.FONT_HERSHEY_SIMPLEX
WINDOW     = "AQP1 Auto-Annotator"
CORNER_R   = 9
SNAP_DIST  = 25

TEMPLATE_FILE = Path(cfg.BASE_DIR) / "annotation_template.json"


# ──────────────────────────────────────────────────────────────────────────────
# Template persistence  (saves last confirmed positions to disk)
# ──────────────────────────────────────────────────────────────────────────────

def save_template(outer_pts, core_pts) -> None:
    data = {"outer": outer_pts, "core": core_pts}
    TEMPLATE_FILE.write_text(json.dumps(data, indent=2))


def load_template() -> tuple[list | None, list | None]:
    if not TEMPLATE_FILE.exists():
        return None, None
    try:
        data = json.loads(TEMPLATE_FILE.read_text())
        outer = [tuple(p) for p in data["outer"]]
        core  = [tuple(p) for p in data["core"]]
        return outer, core
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Auto-detection  (used only when no template exists yet)
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess(img: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blur)


def _bright_edges_h(row, threshold=80):
    W = len(row)
    lx = next((x for x in range(W)         if row[x] > threshold), W // 10)
    rx = next((x for x in range(W-1,-1,-1) if row[x] > threshold), W * 9 // 10)
    return lx, rx


def _bright_edges_v(col, threshold=60):
    H = len(col)
    ty = next((y for y in range(H)         if col[y] > threshold), H // 20)
    by = next((y for y in range(H-1,-1,-1) if col[y] > threshold), H * 19 // 20)
    return ty, by


def auto_detect_outer(gray):
    H, W = gray.shape
    lxs = [_bright_edges_h(gray[r].astype(int))[0] for r in [H//4, H//2, 3*H//4]]
    rxs = [_bright_edges_h(gray[r].astype(int))[1] for r in [H//4, H//2, 3*H//4]]
    tys = [_bright_edges_v(gray[:,c].astype(int))[0] for c in [W//4, W//2, 3*W//4]]
    bys = [_bright_edges_v(gray[:,c].astype(int))[1] for c in [W//4, W//2, 3*W//4]]
    x0, x1 = max(0, int(np.median(lxs))), min(W-1, int(np.median(rxs)))
    y0, y1 = max(0, int(np.median(tys))), min(H-1, int(np.median(bys)))
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]


def auto_detect_core(gray, outer_pts):
    H, W = gray.shape
    ox0 = max(0, min(p[0] for p in outer_pts))
    ox1 = min(W, max(p[0] for p in outer_pts))
    oy0 = max(0, min(p[1] for p in outer_pts))
    oy1 = min(H, max(p[1] for p in outer_pts))
    roi = gray[oy0:oy1, ox0:ox1]
    if roi.size == 0:
        cx0,cx1 = ox0+(ox1-ox0)//4, ox1-(ox1-ox0)//4
        cy0,cy1 = oy0+(oy1-oy0)//20, oy1-(oy1-oy0)//20
        return [(cx0,cy0),(cx1,cy0),(cx1,cy1),(cx0,cy1)]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        rx,ry,rw,rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cx0,cy0,cx1,cy1 = ox0+rx, oy0+ry, ox0+rx+rw, oy0+ry+rh
    else:
        cx0,cx1 = ox0+(ox1-ox0)//5, ox1-(ox1-ox0)//5
        cy0,cy1 = oy0+(oy1-oy0)//20, oy1-(oy1-oy0)//20
    cx0,cx1 = max(0,cx0), min(W-1,cx1)
    cy0,cy1 = max(0,cy0), min(H-1,cy1)
    return [(cx0,cy0),(cx1,cy0),(cx1,cy1),(cx0,cy1)]


def auto_detect(img):
    gray  = _preprocess(img)
    outer = auto_detect_outer(gray)
    core  = auto_detect_core(gray, outer)
    return outer, core


# ──────────────────────────────────────────────────────────────────────────────
# Label I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_labels(label_path, outer_pts, core_pts, img_w, img_h):
    def fmt(pts):
        vals = []
        for x, y in pts:
            vals += [x/img_w, y/img_h]
        return " ".join(f"{v:.6f}" for v in vals)
    with open(label_path, "w") as f:
        f.write(f"0 {fmt(outer_pts)}\n")
        f.write(f"1 {fmt(core_pts)}\n")


def load_labels(label_path, img_w, img_h):
    if not label_path.exists():
        return None, None
    outer, core = None, None
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
                pts  = [(int(vals[i]*img_w), int(vals[i+1]*img_h)) for i in range(0,8,2)]
                if cls == 0: outer = pts
                if cls == 1: core  = pts
    except Exception as e:
        print(f"  [WARN] Could not parse {label_path.name}: {e}")
    return outer, core


# ──────────────────────────────────────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────────────────────────────────────

def draw_region(canvas, pts, color, active, label, scale):
    sp  = [(int(x*scale), int(y*scale)) for x,y in pts]
    arr = np.array(sp, dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [arr], color)
    cv2.addWeighted(overlay, 0.30 if active else 0.10, canvas, 0.70 if active else 0.90, 0, canvas)
    cv2.polylines(canvas, [arr], True, color, 2 if active else 1)
    for i, p in enumerate(sp):
        r = CORNER_R if active else 5
        cv2.circle(canvas, p, r, color, -1)
        cv2.circle(canvas, p, r, (0,0,0), 1)
        lbl = ["TL","TR","BR","BL"][i]
        cv2.putText(canvas, lbl, (p[0]+10, p[1]-8), FONT, 0.5, (0,0,0), 3)
        cv2.putText(canvas, lbl, (p[0]+10, p[1]-8), FONT, 0.5, color, 1)
    cv2.putText(canvas, label, (sp[0][0]+4, sp[0][1]+18), FONT, 0.60, (0,0,0), 3)
    cv2.putText(canvas, label, (sp[0][0]+4, sp[0][1]+18), FONT, 0.60, color, 1)


def draw_hud(canvas, filename, idx, total, active_region, using_template):
    region_name = ["overlay_outer [0]", "dark_core [1]"][active_region]
    src_tag     = "TEMPLATE" if using_template else "AUTO-DETECT"
    lines = [
        f"Image {idx+1}/{total}:  {filename}  [{src_tag}]",
        f"Active: {region_name}  (TAB to switch)",
        "LClick=move corner  A=auto-detect  R=restore template  ENTER=save  S=skip  Q=quit",
    ]
    for i, line in enumerate(lines):
        y = 26 + i*26
        cv2.putText(canvas, line, (8, y), FONT, 0.60, (0,0,0), 3)
        cv2.putText(canvas, line, (8, y), FONT, 0.60, (255,255,255), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def run(images_dir: Path, labels_dir: Path, start_idx: int = 0) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not images:
        print(f"[ERROR] No images found in {images_dir}")
        return

    total = len(images)
    print(f"Found {total} images. Starting at index {start_idx}.")

    # Load any previously saved template
    tmpl_outer, tmpl_core = load_template()
    if tmpl_outer:
        print(f"[TEMPLATE] Loaded from {TEMPLATE_FILE.name} — will be used as default positions.")
    else:
        print("[TEMPLATE] None saved yet — first image will use auto-detection.")
    print()

    # ── Mouse state ──────────────────────────────────────────────────────────
    drag_region = None
    drag_corner = None
    disp_scale  = 1.0
    outer_pts   = []
    core_pts    = []
    active_region  = 0
    using_template = False

    def _nearest(x_d, y_d):
        best_d, best_r, best_c = SNAP_DIST, None, None
        for ri, pts in enumerate([outer_pts, core_pts]):
            for ci, (px, py) in enumerate(pts):
                d = ((px*disp_scale - x_d)**2 + (py*disp_scale - y_d)**2)**0.5
                if d < best_d:
                    best_d, best_r, best_c = d, ri, ci
        return best_r, best_c

    def on_mouse(event, x, y, flags, param):
        nonlocal drag_region, drag_corner, outer_pts, core_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            r, c = _nearest(x, y)
            if r is not None:
                drag_region, drag_corner = r, c
        elif event == cv2.EVENT_MOUSEMOVE and drag_region is not None:
            fx, fy = int(x/disp_scale), int(y/disp_scale)
            if drag_region == 0: outer_pts[drag_corner] = (fx, fy)
            else:                core_pts[drag_corner]  = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP:
            drag_region = drag_corner = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 800)
    cv2.setMouseCallback(WINDOW, on_mouse)

    idx = start_idx
    while idx < total:
        img_path = images[idx]
        lbl_path = labels_dir / (img_path.stem + ".txt")

        base = cv2.imread(str(img_path))
        if base is None:
            print(f"[SKIP] Cannot read {img_path.name}")
            idx += 1
            continue

        img_h, img_w = base.shape[:2]
        disp_scale   = min(1100/img_w, 800/img_h)

        # ── Decide starting positions for this image ─────────────────────────
        #   Priority: existing saved label > template > auto-detect
        saved_outer, saved_core = load_labels(lbl_path, img_w, img_h)

        if saved_outer and saved_core:
            outer_pts, core_pts = saved_outer, saved_core
            using_template = False
            print(f"  [{idx+1:>4}/{total}] {img_path.name}  — existing label loaded")
        elif tmpl_outer and tmpl_core:
            outer_pts, core_pts = list(tmpl_outer), list(tmpl_core)
            using_template = True
            print(f"  [{idx+1:>4}/{total}] {img_path.name}  — template applied")
        else:
            outer_pts, core_pts = auto_detect(base)
            using_template = False
            print(f"  [{idx+1:>4}/{total}] {img_path.name}  — auto-detected")

        # Keep a copy of the detection/template so R can restore it
        restore_outer = list(outer_pts)
        restore_core  = list(core_pts)
        restore_sw_top = list(sw_top_pts) if sw_top_pts else []
        restore_sw_bot = list(sw_bot_pts) if sw_bot_pts else []
        active_region = 0

        # ── Per-image interaction loop ────────────────────────────────────────
        while True:
            dw, dh = int(img_w*disp_scale), int(img_h*disp_scale)
            canvas = cv2.resize(base, (dw, dh))

            draw_region(canvas, outer_pts, COLORS[0], active_region==0,
                        "overlay_outer  (120mm wide × 370mm tall)", disp_scale)
            draw_region(canvas, core_pts,  COLORS[1], active_region==1,
                        "dark_core  (80mm wide × 340mm tall)",      disp_scale)
            draw_hud(canvas, img_path.name, idx, total, active_region, using_template)

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord("c")):       # ENTER / C — save
                save_labels(lbl_path, outer_pts, core_pts, img_w, img_h)
                # Persist positions as new template for all following images
                tmpl_outer, tmpl_core = list(outer_pts), list(core_pts)
                save_template(outer_pts, core_pts)
                print(f"    [SAVED] {lbl_path.name}  →  template updated")
                idx += 1
                break

            elif key == ord("s"):           # Skip
                print(f"    [SKIP]  {img_path.name}")
                idx += 1
                break

            elif key == ord("a"):           # Re-run auto-detection
                outer_pts, core_pts, sw_top_pts, sw_bot_pts = auto_detect(base)
                restore_outer, restore_core = list(outer_pts), list(core_pts)
                restore_sw_top, restore_sw_bot = list(sw_top_pts), list(sw_bot_pts)
                using_template = False
                print(f"    [AUTO]  Re-detected regions")

            elif key == ord("r"):           # Restore template / auto-detect positions
                outer_pts, core_pts = list(restore_outer), list(restore_core)
                sw_top_pts, sw_bot_pts = list(restore_sw_top), list(restore_sw_bot)

            elif key == ord("\t"):          # TAB — switch active region
                active_region = 1 - active_region

            elif key in (ord("q"), 27):     # Q / ESC — quit
                cv2.destroyAllWindows()
                print(f"\nStopped at index {idx}. Resume with --start {idx}")
                return

    cv2.destroyAllWindows()
    print(f"\nAll {total} images processed.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AQP1 auto-annotator — template propagation across identical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How it works:
  • First image uses auto-detection to pre-fill both boxes.
  • After you confirm (ENTER), those exact pixel positions are saved as a
    template in annotation_template.json.
  • Every subsequent image opens with the template already applied.
  • Just press ENTER to accept, or drag a corner if something looks off.
  • Any correction you save immediately updates the template for all
    images that follow.

Each box captures BOTH dimensions:
  Cyan  overlay_outer  →  width (120mm)  AND  height (370mm)
  Green dark_core      →  width  (80mm)  AND  height (340mm)

Examples:
  python auto_annotate.py --size S4
  python auto_annotate.py --size S4 --start 40
        """
    )
    parser.add_argument("--size",   default="S4")
    parser.add_argument("--images", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--start",  type=int, default=0)
    args = parser.parse_args()

    size = args.size.upper()

    images_dir = Path(args.images) if args.images else Path(cfg.TRAINING_SOURCES.get(size, ""))
    labels_dir = Path(args.labels) if args.labels else Path(cfg.DATASET_DIR) / "train" / "labels"

    if not images_dir.exists():
        print(f"[ERROR] Images folder not found: {images_dir}")
        sys.exit(1)

    run(images_dir, labels_dir, start_idx=args.start)
