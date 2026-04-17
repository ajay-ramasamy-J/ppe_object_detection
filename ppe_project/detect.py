"""
PPE Object Detection - Multi-Model Testing
Supports: YOLOv8, YOLOv9, YOLOv11, YOLOv12 (ultralytics)
- Green box  → PPE item PRESENT
- Red box    → PPE item ABSENT  (if your model has "no_helmet", "no_vest" etc. classes)

Usage:
    python ppe_detection.py

Folder structure expected:
    ppe_detection.py
    models/
        yolov8_ppe.pt
        yolov9_ppe.pt
        yolov11_ppe.pt
        yolov12_ppe.pt   ← "yolov26" mapped here; adjust path below
    test_images/
        image1.jpg
        image2.jpg
        image3.jpg
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# ① CONFIGURATION  – edit paths to match yours
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "YOLOv8":  "runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",
    "YOLOv9":  "runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "YOLOv11": "runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
    "YOLOv26": "runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",   # ← rename to your v26 weight file
}

TEST_IMAGES = [
    "ppe_project/test1/img1.png",
    "ppe_project/test1/img2.png",
    "ppe_project/test1/img3.png",
]

OUTPUT_DIR = Path("output_results")

# Detection threshold
CONF_THRESHOLD = 0.3

# ── Class-name → colour mapping ──────────────────────────────────────────────
# Classes whose names contain any keyword in ABSENT_KEYWORDS → RED  (missing PPE)
# Everything else                                            → GREEN (PPE present)
ABSENT_KEYWORDS = [
    "no_", "no-", "without", "missing", "absent",
    "none", "unprotected",
]

GREEN = (0, 200, 0)
RED   = (0, 0, 220)

# ─────────────────────────────────────────────
# ② HELPERS
# ─────────────────────────────────────────────

def is_absent(class_name: str) -> bool:
    """Return True if the detected class represents a MISSING PPE item."""
    name = class_name.lower()
    return any(kw in name for kw in ABSENT_KEYWORDS)


def draw_detections(image: np.ndarray, results, model_name: str) -> np.ndarray:
    """
    Draw bounding boxes on *image* from ultralytics Results object.
    Green = PPE present | Red = PPE absent
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Header banner
    banner_h = 36
    cv2.rectangle(annotated, (0, 0), (w, banner_h), (30, 30, 30), -1)
    cv2.putText(annotated, f"  {model_name}", (8, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.putText(annotated, "No detections", (10, banner_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return annotated

    names = results[0].names   # {0: 'helmet', 1: 'no_helmet', ...}

    for box in boxes:
        conf  = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id     = int(box.cls[0])
        cls_name   = names.get(cls_id, str(cls_id))
        colour     = RED if is_absent(cls_name) else GREEN
        thickness  = 2

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        # Label background
        label      = f"{cls_name}  {conf:.0%}"
        font_scale = 0.52
        font_thick = 1
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        label_y    = max(y1 - 6, lh + 6)
        cv2.rectangle(annotated,
                      (x1, label_y - lh - baseline - 2),
                      (x1 + lw + 4, label_y + 2),
                      colour, -1)
        cv2.putText(annotated, label,
                    (x1 + 2, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thick, cv2.LINE_AA)

    # Legend (bottom-left)
    legend_y = h - 10
    cv2.circle(annotated, (14, legend_y - 4), 7, GREEN, -1)
    cv2.putText(annotated, "Present", (26, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    cv2.circle(annotated, (100, legend_y - 4), 7, RED, -1)
    cv2.putText(annotated, "Absent", (112, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

    return annotated


def make_grid(images: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Tile a list of images into a grid (all resized to same height)."""
    target_h = 480
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(img, (new_w, target_h)))

    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]
        # Pad last row if uneven
        while len(row_imgs) < cols:
            blank = np.zeros_like(row_imgs[0])
            row_imgs.append(blank)
        rows.append(np.hstack(row_imgs))

    return np.vstack(rows)


# ─────────────────────────────────────────────
# ③ MAIN
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n── Loading models ──────────────────────────────")
    models = {}
    for name, path in MODEL_PATHS.items():
        if not Path(path).exists():
            print(f"  [SKIP] {name}: file not found → {path}")
            continue
        print(f"  Loading {name} …", end=" ", flush=True)
        models[name] = YOLO(path)
        print("OK")

    if not models:
        print("\n⚠  No model files found. Check MODEL_PATHS in the script.")
        return

    # Process each image
    for img_path in TEST_IMAGES:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"\n[SKIP] Image not found: {img_path}")
            continue

        print(f"\n── Processing: {img_path.name} ─────────────────")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ✗ Could not read image.")
            continue

        per_model_outputs = []

        for model_name, model in models.items():
            print(f"  {model_name} … ", end="", flush=True)

            results = model(image, conf=CONF_THRESHOLD, verbose=False)
            annotated = draw_detections(image, results, model_name)

            # Save individual result
            out_name = f"{img_path.stem}_{model_name}.jpg"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"saved → {out_path}")

            per_model_outputs.append(annotated)

        # Save comparison grid (all models side-by-side for this image)
        if len(per_model_outputs) > 1:
            grid = make_grid(per_model_outputs, cols=2)
            grid_path = OUTPUT_DIR / f"{img_path.stem}_ALL_MODELS.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"  Grid saved → {grid_path}")

    print("\n✔  Done! Results in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()