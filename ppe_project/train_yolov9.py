"""
============================================================
  PPE Detection — YOLOv9m Training (Lightning AI T4)
  Platform : Lightning AI  |  GPU: T4 (16 GB)
  Dataset  : 44,000+ images
  Target   : mAP50 ≥ 0.85

  WHY FASTER THAN COLAB:
  ┌─────────────────┬──────────────┬──────────────────────┐
  │ Setting         │ Colab        │ Lightning AI (this)  │
  ├─────────────────┼──────────────┼──────────────────────┤
  │ workers         │ 2 (limited)  │ 4 (more CPU cores)   │
  │ batch           │ 16           │ 16 (T4 safe)         │
  │ cache           │ True (slow)  │ auto-detect          │
  │ time/epoch      │ ~14–16 mins  │ ~8–10 mins (target)  │
  │ freeze          │ 10           │ 10 (stable start)    │
  └─────────────────┴──────────────┴──────────────────────┘

  YOLOv9m ARCHITECTURE (GELAN + PGI):
  - GELAN  → better gradient flow than v8/v11
  - PGI    → Programmable Gradient Information
  - RepNCSPELAN4 blocks → efficient feature extraction
  - Typically +1–2% mAP over YOLOv8m on PPE datasets
============================================================
"""

import os
import time
import json
import torch
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────
#  GPU SETUP
# ─────────────────────────────────────────────────────────

print("=" * 54)
if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU  : {gpu}")
    print(f"  VRAM : {vram:.1f} GB")
    DEVICE = 0
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.empty_cache()
else:
    print("  GPU  : NOT FOUND")
    DEVICE = "cpu"
    vram   = 0
print("=" * 54)


# ─────────────────────────────────────────────────────────
#  PATHS  ★ only edit if your layout differs
# ─────────────────────────────────────────────────────────

DATA_YAML   = "ppe_project/dataset/yolov9/data.yaml"
MODEL_NAME  = "yolov9m_ppe"
PROJECT_DIR = "ppe_project/models"
RESULTS_DIR = "ppe_project/results"


# ─────────────────────────────────────────────────────────
#  AUTO BATCH — based on VRAM
#  YOLOv9m @ 640:
#  T4  (15.6GB) → batch=16 safe (v9m is heavier than v8m)
#  A100(40GB)   → batch=32
#  Lesson from v8m: batch=32 on T4 → 12.5GB mem → slow
#  Keep batch=16 for stable training
# ─────────────────────────────────────────────────────────

if vram >= 35:      # A100
    BATCH = 32
elif vram >= 15:    # T4
    BATCH = 16
else:
    BATCH = 8

print(f"  VRAM {vram:.1f}GB → batch={BATCH}")


# ─────────────────────────────────────────────────────────
#  AUTO CACHE
#  Lightning AI T4 free RAM ≈ 12–15GB
#  44k×640px needs ~22GB for ram cache → disk only
# ─────────────────────────────────────────────────────────

try:
    import psutil
    free_ram = psutil.virtual_memory().available / 1e9
except ImportError:
    free_ram = 0

CACHE = "ram" if free_ram > 22 else True
print(f"  RAM  {free_ram:.1f}GB → cache='{CACHE}'")
print("=" * 54 + "\n")


# ─────────────────────────────────────────────────────────
#  LOAD YOLOv9m
#  Auto-downloads on first run (~39MB)
# ─────────────────────────────────────────────────────────

model = YOLO("yolov9m.pt")

print(f"  Model   : YOLOv9m (GELAN + PGI)")
print(f"  Data    : {DATA_YAML}")
print(f"  Output  : {PROJECT_DIR}/{MODEL_NAME}")
print(f"  Epochs  : 100  |  Patience : 20")
print(f"  Batch   : {BATCH}  |  ImgSz   : 640")
print(f"  Workers : 4    |  Cache   : {CACHE}")
print("=" * 54 + "\n")


# ─────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────

start = time.time()

results = model.train(
    data        = DATA_YAML,
    epochs      = 100,
    imgsz       = 640,
    project     = PROJECT_DIR,
    name        = MODEL_NAME,
    device      = DEVICE,
    exist_ok    = True,

    # ── Batch & Workers ──────────────────────────────────
    # workers=4 is the key fix vs Colab (was 2)
    # Lightning AI gives 4 CPU cores → 2x faster data loading
    batch       = BATCH,
    workers     = 4,           # ★ was 2 on Colab → now 4

    # ── Precision ────────────────────────────────────────
    amp         = True,        # FP16 → ~35% faster on T4

    # ── Cache ────────────────────────────────────────────
    cache       = CACHE,

    # ── Early Stopping ───────────────────────────────────
    patience    = 20,

    # ── Checkpointing ────────────────────────────────────
    save        = True,
    save_period = 10,          # every 10 epochs on Lightning AI
                               # (no disconnect risk unlike Colab)

    # ── Freeze Strategy ──────────────────────────────────
    # freeze=10 proven to work well:
    # → GPU mem stays ~5GB (not 12.5GB)
    # → epoch 1 starts at 0.55+ mAP (not 0.37)
    # → stable early training
    freeze      = 10,

    # ── Optimizer ────────────────────────────────────────
    # SGD is the right choice for YOLOv9
    # YOLOv9 GELAN paper used SGD — better convergence
    # than AdamW for this specific architecture
    optimizer       = "SGD",
    lr0             = 0.01,
    lrf             = 0.01,    # cosine → final LR = 1e-4
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 3.0,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,

    # ── LR Schedule ──────────────────────────────────────
    cos_lr      = True,

    # ── Augmentation ─────────────────────────────────────
    mosaic      = 1.0,
    mixup       = 0.1,
    copy_paste  = 0.1,
    close_mosaic= 10,

    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    degrees     = 5.0,
    translate   = 0.1,
    scale       = 0.5,
    fliplr      = 0.5,
    flipud      = 0.0,

    # ── Loss Weights ─────────────────────────────────────
    box         = 7.5,
    cls         = 0.5,
    dfl         = 1.5,

    # ── Misc ─────────────────────────────────────────────
    dropout     = 0.0,
    verbose     = True,
    plots       = True,
    val         = True,
)


# ─────────────────────────────────────────────────────────
#  SAVE RESULTS
# ─────────────────────────────────────────────────────────

elapsed = round((time.time() - start) / 60, 2)

r = {
    "model"       : MODEL_NAME,
    "mAP50"       : float(results.results_dict.get("metrics/mAP50(B)",     0)),
    "mAP50_95"    : float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    "precision"   : float(results.results_dict.get("metrics/precision(B)", 0)),
    "recall"      : float(results.results_dict.get("metrics/recall(B)",    0)),
    "time_mins"   : elapsed,
    "best_weights": f"{PROJECT_DIR}/{MODEL_NAME}/weights/best.pt",
}

os.makedirs(RESULTS_DIR, exist_ok=True)
json_path = f"{RESULTS_DIR}/{MODEL_NAME}_results.json"
with open(json_path, "w") as f:
    json.dump(r, f, indent=2)


# ─────────────────────────────────────────────────────────
#  FINAL SUMMARY — full leaderboard
# ─────────────────────────────────────────────────────────

ref = {
    "YOLO11m (50ep)" : 0.7897,
    "YOLOv8m (35ep)" : 0.774,
    "YOLO26m (~44ep)": 0.790,
}

elapsed_hrs = elapsed / 60

print(f"\n{'='*54}")
print(f"  YOLOV9M COMPLETE!")
print(f"{'='*54}")
print(f"  {'Metric':<20} {'Value':>10}")
print(f"  {'─'*34}")
print(f"  {'mAP50':<20} {r['mAP50']:>10.4f}")
print(f"  {'mAP50-95':<20} {r['mAP50_95']:>10.4f}")
print(f"  {'Precision':<20} {r['precision']:>10.4f}")
print(f"  {'Recall':<20} {r['recall']:>10.4f}")
print(f"  {'Time (mins)':<20} {elapsed:>10.1f}")
print(f"  {'Time (hrs)':<20} {elapsed_hrs:>10.1f}")
print(f"  {'─'*34}")

# Leaderboard
print(f"\n  {'─'*46}")
print(f"  {'Model':<22} {'mAP50':>8}  {'Rank'}")
print(f"  {'─'*46}")
all_scores = {**ref, "YOLOv9m (this)": r["mAP50"]}
for i, (name, score) in enumerate(
    sorted(all_scores.items(), key=lambda x: x[1], reverse=True), 1
):
    crown = " 👑" if i == 1 else ""
    print(f"  {name:<22} {score:>8.4f}  #{i}{crown}")
print(f"  {'─'*46}")
print(f"\n  Best weights → {r['best_weights']}")
print(f"  Results JSON → {json_path}")
print(f"{'='*54}\n")