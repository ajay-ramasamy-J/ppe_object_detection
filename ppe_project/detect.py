import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
# ① CONFIG
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "YOLOv8":  "runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",
    "YOLOv9":  "runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "YOLOv11": "runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
    "YOLOv26": "runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",
}

TEST_IMAGES = [
    "ppe_project/test1/img4.jpg",
    "ppe_project/test1/img5.jpg",
    "ppe_project/test1/img6.jpg",
]

OUTPUT_DIR = Path("output_results")

CONF_THRESHOLD = 0.3

# ─────────────────────────────────────────────
# ② CLASS GROUPING (IMPORTANT FIX)
# ─────────────────────────────────────────────

ABSENT_CLASSES = {
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest"
}

IGNORE_CLASSES = {
    "Person",
    "Ladder",
    "Safety Cone",
    "Fall-Detected"
}

# Colors
GREEN = (0, 200, 0)     # PPE present
RED   = (0, 0, 220)     # PPE missing
GRAY  = (180, 180, 180) # Neutral objects

# ─────────────────────────────────────────────
# ③ HELPERS
# ─────────────────────────────────────────────

def get_color(class_name: str):
    if class_name in IGNORE_CLASSES:
        return GRAY
    elif class_name in ABSENT_CLASSES:
        return RED
    else:
        return GREEN


def draw_detections(image, results, model_name):
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Header
    cv2.rectangle(annotated, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(annotated, f"{model_name}", (10, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.putText(annotated, "No detections", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return annotated

    names = results[0].names

    for box in boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))

        color = get_color(cls_name)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{cls_name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label,
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(annotated,
                      (x1, y1 - th - 6),
                      (x1 + tw + 4, y1),
                      color, -1)

        cv2.putText(annotated, label,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Legend
    legend_y = h - 10

    cv2.circle(annotated, (15, legend_y), 6, GREEN, -1)
    cv2.putText(annotated, "Present", (25, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    cv2.circle(annotated, (100, legend_y), 6, RED, -1)
    cv2.putText(annotated, "Absent", (110, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

    cv2.circle(annotated, (180, legend_y), 6, GRAY, -1)
    cv2.putText(annotated, "Neutral", (190, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)

    return annotated


def make_grid(images, cols=2):
    target_h = 480
    resized = []

    for img in images:
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(img, (new_w, target_h)))

    rows = []
    for i in range(0, len(resized), cols):
        row = resized[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(row))

    return np.vstack(rows)

# ─────────────────────────────────────────────
# ④ MAIN
# ─────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading models...")
    models = {}

    for name, path in MODEL_PATHS.items():
        if Path(path).exists():
            models[name] = YOLO(path)
            print(f"{name} loaded")
        else:
            print(f"{name} not found")

    for img_path in TEST_IMAGES:
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image not found: {img_path}")
            continue

        outputs = []

        for model_name, model in models.items():
            results = model(img, conf=CONF_THRESHOLD, verbose=False)

            annotated = draw_detections(img, results, model_name)

            save_path = OUTPUT_DIR / f"{Path(img_path).stem}_{model_name}.jpg"
            cv2.imwrite(str(save_path), annotated)

            outputs.append(annotated)

        # Grid comparison
        if len(outputs) > 1:
            grid = make_grid(outputs)
            grid_path = OUTPUT_DIR / f"{Path(img_path).stem}_comparison.jpg"
            cv2.imwrite(str(grid_path), grid)

    print("Done! Check output_results folder.")


if __name__ == "__main__":
    main()