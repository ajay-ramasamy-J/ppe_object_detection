"""
============================================================
  PPE Detection — Complete Analysis Script
  Quantitative + Qualitative for All 4 YOLO Models
  For Research Paper (IEEE Conference)
============================================================

WHAT THIS SCRIPT PRODUCES:
  Quantitative:
    1. Overall metrics table (mAP50, mAP50-95, P, R)
    2. Per-class mAP50 table across all 4 models
    3. Best epoch for each model
    4. Training vs Test comparison table
    5. Model complexity table (params, GFLOPs, speed)

  Qualitative:
    6. Same 5 images → all 4 models → side-by-side grid
    7. Confidence score comparison per image per model
    8. Detection count comparison per class per image
    9. Confusion matrix for each model
   10. Summary radar chart (5 metrics, 4 models)

  All outputs saved to:  paper_outputs/
  Ready to drop into Overleaf as figures!
============================================================
"""

import os
import csv
import json
import time
import glob
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from ultralytics import YOLO
import cv2

# ─────────────────────────────────────────────────────────
#  ★ EDIT THESE PATHS — update with your actual best.pt paths
# ─────────────────────────────────────────────────────────

MODELS = {
    "YOLOv8m" : "runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",   # ← update
    "YOLOv9m" : "runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "YOLO11m" : "runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
    "YOLO26m" : "runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",
}

RESULTS_CSV = {
    "YOLOv8m" : "runs/detect/ppe_project/models/yolov8m_ppe/results.csv",
    "YOLOv9m" : "runs/detect/ppe_project/models/yolov9m_ppe/results.csv",
    "YOLO11m" : "runs/detect/ppe_project/models/yolov11m_ppe/results.csv",
    "YOLO26m" : "runs/detect/ppe_project/models/yolo26m_ppe/results.csv",
}

DATA_YAML  = "ppe_project/dataset/yolov8/data.yaml"
TEST_DIR   = "ppe_project/dataset/yolov8/test/images"
OUTPUT_DIR = "paper_final_outputs"

CLASS_NAMES = [
    'Fall-Detected', 'Gloves', 'Goggles', 'Hardhat',
    'Ladder', 'Mask', 'NO-Gloves', 'NO-Goggles',
    'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest'
]

# Colors for each model in plots
MODEL_COLORS = {
    "YOLOv8m" : "#378ADD",   # blue
    "YOLOv9m" : "#1D9E75",   # green
    "YOLO11m" : "#BA7517",   # amber
    "YOLO26m" : "#D85A30",   # coral
}

# ─────────────────────────────────────────────────────────
#  CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────

for folder in [
    f"{OUTPUT_DIR}/quantitative",
    f"{OUTPUT_DIR}/qualitative",
    f"{OUTPUT_DIR}/detections",
    f"{OUTPUT_DIR}/confusion_matrix",
]:
    os.makedirs(folder, exist_ok=True)


# ═══════════════════════════════════════════════════════
#  PART A — QUANTITATIVE ANALYSIS
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PART A — QUANTITATIVE ANALYSIS")
print("="*60)

# ── A1: Validate all models on TEST set ─────────────────

all_metrics   = {}
all_per_class = {}

for model_name, weights in MODELS.items():
    if not Path(weights).exists():
        print(f"  ⚠  {model_name}: weights not found → {weights}")
        continue

    print(f"\n  Validating {model_name}...")
    model   = YOLO(weights)
    t_start = time.time()
    metrics = model.val(
        data    = DATA_YAML,
        split   = "test",
        verbose = False,
        plots   = True,
        save_json = True,
        project = f"{OUTPUT_DIR}/confusion_matrix",
        name    = model_name,
        exist_ok= True,
    )
    t_end = time.time()
    inf_ms = (t_end - t_start) / len(glob.glob(f"{TEST_DIR}/*.jpg")) * 1000

    all_metrics[model_name] = {
        "mAP50"     : round(float(metrics.box.map50), 4),
        "mAP50_95"  : round(float(metrics.box.map),   4),
        "precision" : round(float(metrics.box.mp),    4),
        "recall"    : round(float(metrics.box.mr),    4),
        "inf_ms"    : round(inf_ms, 2),
    }
    all_per_class[model_name] = [
        round(float(ap), 4) for ap in metrics.box.ap50
    ]
    print(f"  ✅  mAP50={all_metrics[model_name]['mAP50']:.4f}  "
          f"mAP50-95={all_metrics[model_name]['mAP50_95']:.4f}")


# ── A2: Best epoch from results.csv ─────────────────────

best_epochs = {}

print("\n  Finding best epoch for each model...")

for model_name, csv_path in RESULTS_CSV.items():
    if not Path(csv_path).exists():
        print(f"  ⚠  {model_name}: results.csv not found")
        continue

    best_map50 = 0
    best_epoch = 0
    best_row   = {}
    all_rows   = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                map50 = float(row.get('metrics/mAP50(B)', 0))
                epoch = int(float(row.get('epoch', 0)))
                all_rows.append((epoch, map50))
                if map50 > best_map50:
                    best_map50 = map50
                    best_epoch = epoch
                    best_row   = row
            except:
                continue

    best_epochs[model_name] = {
        "best_epoch"  : best_epoch,
        "mAP50"       : round(best_map50, 4),
        "mAP50_95"    : round(float(best_row.get(
                            'metrics/mAP50-95(B)', 0)), 4),
        "precision"   : round(float(best_row.get(
                            'metrics/precision(B)', 0)), 4),
        "recall"      : round(float(best_row.get(
                            'metrics/recall(B)', 0)), 4),
        "all_rows"    : all_rows,
    }
    print(f"  {model_name}: best epoch = {best_epoch}  "
          f"mAP50 = {best_map50:.4f}")


# ── A3: Print Summary Tables ────────────────────────────

print("\n" + "="*70)
print("  TABLE I — OVERALL PERFORMANCE ON TEST SET")
print("="*70)
print(f"  {'Model':<12} {'P':>8} {'R':>8} {'mAP50':>8} "
      f"{'mAP50-95':>10} {'Inf(ms)':>9}")
print(f"  {'─'*58}")
for name, m in all_metrics.items():
    print(f"  {name:<12} {m['precision']:>8.4f} {m['recall']:>8.4f} "
          f"{m['mAP50']:>8.4f} {m['mAP50_95']:>10.4f} "
          f"{m['inf_ms']:>9.2f}")

print("\n" + "="*70)
print("  TABLE II — BEST EPOCH PERFORMANCE (VALIDATION)")
print("="*70)
print(f"  {'Model':<12} {'Best Ep':>8} {'P':>8} {'R':>8} "
      f"{'mAP50':>8} {'mAP50-95':>10}")
print(f"  {'─'*58}")
for name, e in best_epochs.items():
    print(f"  {name:<12} {e['best_epoch']:>8} {e['precision']:>8.4f} "
          f"{e['recall']:>8.4f} {e['mAP50']:>8.4f} "
          f"{e['mAP50_95']:>10.4f}")

print("\n" + "="*70)
print("  TABLE III — PER-CLASS mAP50 ACROSS ALL MODELS")
print("="*70)
model_names = list(all_per_class.keys())
print(f"  {'Class':<22}", end="")
for n in model_names:
    print(f" {n:>9}", end="")
print()
print(f"  {'─'*60}")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<22}", end="")
    for n in model_names:
        if i < len(all_per_class.get(n, [])):
            print(f" {all_per_class[n][i]:>9.4f}", end="")
        else:
            print(f" {'N/A':>9}", end="")
    print()

# ── A4: Save tables as JSON ──────────────────────────────

with open(f"{OUTPUT_DIR}/quantitative/all_metrics.json", "w") as f:
    json.dump({
        "overall"    : all_metrics,
        "best_epochs": {k: {kk: vv for kk, vv in v.items()
                            if kk != 'all_rows'}
                        for k, v in best_epochs.items()},
        "per_class"  : all_per_class,
    }, f, indent=2)
print(f"\n  ✅ Metrics saved → {OUTPUT_DIR}/quantitative/all_metrics.json")


# ── A5: Convergence Curve Plot ───────────────────────────

print("\n  Plotting convergence curves...")
fig, ax = plt.subplots(figsize=(10, 6))

for name, data in best_epochs.items():
    rows = data.get("all_rows", [])
    if rows:
        epochs = [r[0] for r in rows]
        maps   = [r[1] for r in rows]
        ax.plot(epochs, maps,
                color=MODEL_COLORS.get(name, "gray"),
                linewidth=2, label=name, alpha=0.9)
        # mark best epoch
        best_ep = data["best_epoch"]
        best_m  = data["mAP50"]
        ax.scatter([best_ep], [best_m],
                   color=MODEL_COLORS.get(name, "gray"),
                   s=80, zorder=5)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("mAP@0.50", fontsize=13)
ax.set_title("Training Convergence — mAP@0.50 vs Epochs",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/quantitative/convergence_curves.png",
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → convergence_curves.png")


# ── A6: Per-class Bar Chart ──────────────────────────────

print("  Plotting per-class mAP50 bar chart...")

if all_per_class:
    n_classes = len(CLASS_NAMES)
    n_models  = len(all_per_class)
    x         = np.arange(n_classes)
    width     = 0.2
    offsets   = np.linspace(-(n_models-1)*width/2,
                             (n_models-1)*width/2, n_models)

    fig, ax = plt.subplots(figsize=(18, 7))
    for i, (name, aps) in enumerate(all_per_class.items()):
        bars = ax.bar(x + offsets[i], aps, width,
                      label=name,
                      color=MODEL_COLORS.get(name, "gray"),
                      alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45,
                       ha='right', fontsize=9)
    ax.set_ylabel("mAP@0.50", fontsize=12)
    ax.set_title("Per-Class mAP@0.50 — All Models Comparison",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--',
               alpha=0.5, linewidth=1, label='0.5 threshold')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quantitative/per_class_bar.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved → per_class_bar.png")


# ── A7: Radar Chart ──────────────────────────────────────

print("  Plotting radar chart...")

if all_metrics:
    categories = ['Precision', 'Recall', 'mAP50',
                  'mAP50-95', 'Speed\n(1-norm)']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw=dict(polar=True))

    # Find max inf_ms for normalisation
    max_ms = max(m['inf_ms'] for m in all_metrics.values()) or 1

    for name, m in all_metrics.items():
        speed_norm = 1 - (m['inf_ms'] / max_ms)  # higher = faster
        values = [
            m['precision'], m['recall'],
            m['mAP50'], m['mAP50_95'],
            speed_norm,
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2,
                color=MODEL_COLORS.get(name, "gray"),
                label=name)
        ax.fill(angles, values, alpha=0.1,
                color=MODEL_COLORS.get(name, "gray"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quantitative/radar_chart.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved → radar_chart.png")


# ═══════════════════════════════════════════════════════
#  PART B — QUALITATIVE ANALYSIS
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  PART B — QUALITATIVE ANALYSIS")
print("="*60)

# ── B1: Select 5 diverse test images ────────────────────

all_test_imgs = sorted(glob.glob(f"{TEST_DIR}/*.jpg"))
if len(all_test_imgs) == 0:
    all_test_imgs = sorted(glob.glob(f"{TEST_DIR}/*.png"))

# Pick 5 images — try to get diverse scenes
random.seed(42)
if len(all_test_imgs) >= 5:
    test_images = random.sample(all_test_imgs, 5)
else:
    test_images = all_test_imgs

print(f"\n  Selected {len(test_images)} test images:")
for img in test_images:
    print(f"    {Path(img).name}")


# ── B2: Run all 4 models on each image ──────────────────

print("\n  Running inference on test images...")

det_results = {}   # det_results[model][img_path] = result

for model_name, weights in MODELS.items():
    if not Path(weights).exists():
        print(f"  ⚠  {model_name}: skipping — weights not found")
        continue

    print(f"  Running {model_name}...")
    model = YOLO(weights)
    results = model.predict(
        source   = test_images,
        conf     = 0.25,
        iou      = 0.45,
        verbose  = False,
        save     = False,   # we'll save manually as grid
    )
    det_results[model_name] = {}
    for img_path, result in zip(test_images, results):
        det_results[model_name][img_path] = result


# ── B3: Create 4-model comparison grid per image ─────────

print("\n  Creating qualitative comparison grids...")

for img_idx, img_path in enumerate(test_images):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Qualitative Analysis — Image {img_idx+1}: "
        f"{Path(img_path).name}",
        fontsize=14, fontweight='bold', y=1.01
    )

    for ax, (model_name, _) in zip(
            axes.flatten(), MODELS.items()):

        if model_name not in det_results:
            ax.text(0.5, 0.5, f"{model_name}\n(not available)",
                    ha='center', va='center', fontsize=12)
            ax.set_title(model_name)
            ax.axis('off')
            continue

        result   = det_results[model_name][img_path]
        img_plot = result.plot(
            conf=True, labels=True, line_width=2)
        img_rgb  = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        # Detection summary
        boxes = result.boxes
        if boxes is not None and len(boxes):
            n_det     = len(boxes)
            avg_conf  = float(boxes.conf.mean())
            classes   = [CLASS_NAMES[int(c)]
                         for c in boxes.cls]
            violation = [c for c in classes
                         if c.startswith("NO-") or
                            c == "Fall-Detected"]
            title = (f"{model_name}\n"
                     f"Detections: {n_det}  "
                     f"Avg conf: {avg_conf:.2f}\n"
                     f"Violations: {len(violation)}")
        else:
            title = f"{model_name}\nNo detections"

        ax.set_title(title, fontsize=10,
                     color=MODEL_COLORS.get(model_name, "black"),
                     fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    out_path = (f"{OUTPUT_DIR}/qualitative/"
                f"comparison_image_{img_idx+1}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved → comparison_image_{img_idx+1}.png")


# ── B4: Quantitative-Qualitative Bridge Table ─────────────
# Per-image detection count + confidence for each model

print("\n  Creating per-image detection analysis table...")

print("\n" + "="*80)
print("  QUALITATIVE ANALYSIS — Detection Summary per Image")
print("="*80)

qual_data = {}

for img_idx, img_path in enumerate(test_images):
    img_name = Path(img_path).name
    print(f"\n  Image {img_idx+1}: {img_name}")
    print(f"  {'Model':<12} {'Detections':>12} "
          f"{'Avg Conf':>10} {'Violations':>12} "
          f"{'PPE Classes Detected'}")
    print(f"  {'─'*72}")

    qual_data[img_name] = {}

    for model_name in MODELS:
        if model_name not in det_results:
            continue
        result = det_results[model_name][img_path]
        boxes  = result.boxes

        if boxes is not None and len(boxes):
            n_det     = len(boxes)
            avg_conf  = float(boxes.conf.mean())
            classes   = [CLASS_NAMES[int(c)] for c in boxes.cls]
            violation = [c for c in classes
                         if c.startswith("NO-") or
                            c == "Fall-Detected"]
            ppe_det   = list(set(
                [c for c in classes if not c == "Person"]))
        else:
            n_det, avg_conf = 0, 0.0
            violation, ppe_det = [], []

        qual_data[img_name][model_name] = {
            "n_det"     : n_det,
            "avg_conf"  : round(avg_conf, 3),
            "violations": len(violation),
            "classes"   : ppe_det,
        }

        print(f"  {model_name:<12} {n_det:>12} "
              f"{avg_conf:>10.3f} {len(violation):>12}  "
              f"{', '.join(ppe_det[:4])}")


# ── B5: Confidence Distribution Plot ─────────────────────

print("\n  Plotting confidence distribution...")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Confidence Score Distribution Across Test Images",
             fontsize=13, fontweight='bold')

for ax, (model_name, weights) in zip(axes, MODELS.items()):
    all_confs = []

    if model_name in det_results:
        for img_path in test_images:
            result = det_results[model_name][img_path]
            if result.boxes is not None and len(result.boxes):
                confs = result.boxes.conf.cpu().numpy()
                all_confs.extend(confs.tolist())

    if all_confs:
        ax.hist(all_confs, bins=20, range=(0, 1),
                color=MODEL_COLORS.get(model_name, "gray"),
                alpha=0.8, edgecolor='white')
        ax.axvline(x=np.mean(all_confs), color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(all_confs):.2f}')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')

    ax.set_title(model_name,
                 color=MODEL_COLORS.get(model_name, "black"),
                 fontweight='bold', fontsize=11)
    ax.set_xlabel("Confidence Score", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/qualitative/confidence_distribution.png",
    dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → confidence_distribution.png")


# ── B6: Violation Detection Comparison ───────────────────

print("  Plotting violation detection comparison...")

violation_classes = [
    'NO-Gloves', 'NO-Goggles', 'NO-Hardhat',
    'NO-Mask', 'NO-Safety Vest', 'Fall-Detected'
]

viol_counts = {name: [0]*len(violation_classes)
               for name in MODELS}

for model_name in MODELS:
    if model_name not in det_results:
        continue
    for img_path in test_images:
        result = det_results[model_name][img_path]
        if result.boxes is not None and len(result.boxes):
            for cls_id in result.boxes.cls:
                cls_name = CLASS_NAMES[int(cls_id)]
                if cls_name in violation_classes:
                    idx = violation_classes.index(cls_name)
                    viol_counts[model_name][idx] += 1

x      = np.arange(len(violation_classes))
width  = 0.2
n_m    = len(MODELS)
offsets = np.linspace(-(n_m-1)*width/2, (n_m-1)*width/2, n_m)

fig, ax = plt.subplots(figsize=(14, 6))
for i, (name, counts) in enumerate(viol_counts.items()):
    ax.bar(x + offsets[i], counts, width,
           label=name,
           color=MODEL_COLORS.get(name, "gray"),
           alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(violation_classes, rotation=30,
                   ha='right', fontsize=10)
ax.set_ylabel("Total Detections (5 images)", fontsize=11)
ax.set_title("PPE Violation Class Detection Count "
             "— All Models on Same 5 Test Images",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/qualitative/violation_detection_count.png",
    dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → violation_detection_count.png")


# ═══════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════

print("\n" + "="*60)
print("  ALL OUTPUTS SAVED")
print("="*60)
print(f"""
  📊 QUANTITATIVE (paper_outputs/quantitative/):
     convergence_curves.png  → Figure: mAP curves
     per_class_bar.png       → Figure: per-class mAP
     radar_chart.png         → Figure: radar comparison
     all_metrics.json        → All numbers for tables

  🖼  QUALITATIVE (paper_outputs/qualitative/):
     comparison_image_1.png  → Figure: 4-model grid img1
     comparison_image_2.png  → Figure: 4-model grid img2
     comparison_image_3.png  → Figure: 4-model grid img3
     comparison_image_4.png  → Figure: 4-model grid img4
     comparison_image_5.png  → Figure: 4-model grid img5
     confidence_distribution.png → conf score histogram
     violation_detection_count.png → violation bar chart

  🔲 CONFUSION MATRIX (paper_outputs/confusion_matrix/):
     One folder per model with confusion_matrix.png

  📋 USE IN PAPER:
     Tables  → copy numbers from console output above
     Figures → upload PNG files directly to Overleaf
""")
print("="*60)