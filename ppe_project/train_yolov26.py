"""
============================================================
  PPE Detection — YOLO26m Training (100 Epochs)
  Platform : Lightning AI  |  GPU: T4 (16 GB)
  Dataset  : 44,000+ images
  Target   : mAP50 ≥ 0.85

  WHY YOLO26m is different from v8 / v11:
  ┌─────────────────────────────────────────────────────┐
  │  YOLO26 KEY INNOVATIONS                             │
  │  ✅ NMS-Free  — no post-processing step             │
  │  ✅ No DFL    — simpler head, faster CPU inference  │
  │  ✅ MuSGD     — LLM-inspired optimizer              │
  │  ✅ ProgLoss  — progressive loss balancing          │
  │  ✅ STAL      — small target aware label assignment │
  │  ✅ ~43% faster CPU inference than YOLO11           │
  └─────────────────────────────────────────────────────┘

  YOLO26 vs your other models:
  ┌──────────┬────────┬──────────┬──────────────────────┐
  │ Model    │ Params │ mAP50    │ Speed per epoch (T4) │
  ├──────────┼────────┼──────────┼──────────────────────┤
  │ YOLO11m  │  20M   │  0.7897  │  ~8 mins (512 img)   │
  │ YOLOv8m  │  25M   │  ~0.774  │  ~10 mins            │
  │ YOLO26m  │  20M   │  0.85+?  │  ~7–8 mins           │
  └──────────┴────────┴──────────┴──────────────────────┘

  T4 OPTIMISED SETTINGS:
  batch=16   → safe ceiling for YOLO26m @ 640 on T4
  amp=True   → FP16 mixed precision (~35% speed boost)
  freeze=10  → stable early training like v8m fix
  cos_lr     → smooth convergence
  patience=20→ early stopping
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
    print("  GPU  : NOT FOUND — CPU only")
    DEVICE = "cpu"
    vram   = 0
print("=" * 54)


# ─────────────────────────────────────────────────────────
#  PATHS  ★ edit if your layout differs
# ─────────────────────────────────────────────────────────

DATA_YAML   = "ppe_project/dataset/yolov26/data.yaml"
MODEL_NAME  = "yolo26m_ppe"
PROJECT_DIR = "ppe_project/models"
RESULTS_DIR = "ppe_project/results"


# ─────────────────────────────────────────────────────────
#  AUTO BATCH SIZE based on real VRAM
#  YOLO26m @ 640 on T4 (15.6GB):
#  batch=16 is safe → GPU mem ~5–6GB (same as v8m fix)
#  batch=32 only if VRAM ≥ 24GB (A100)
# ─────────────────────────────────────────────────────────

if vram >= 35:      # A100
    BATCH = 32
elif vram >= 15:    # T4
    BATCH = 32
else:
    BATCH = 8

print(f"  VRAM {vram:.1f}GB → batch={BATCH}")


# ─────────────────────────────────────────────────────────
#  AUTO CACHE
#  Lightning AI T4 free RAM ≈ 12–15GB
#  44k×640px needs ~22GB for ram cache → disk cache only
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
#  LOAD YOLO26m
#
#  YOLO26 model options (small → large):
#    yolo26n.pt → Nano   (fastest, edge devices)
#    yolo26s.pt → Small
#    yolo26m.pt → Medium (recommended ✅ for T4)
#    yolo26l.pt → Large  (needs 24GB+ VRAM)
#    yolo26x.pt → XLarge (needs A100)
#
#  yolo26m.pt auto-downloads on first run
# ─────────────────────────────────────────────────────────

model = YOLO("yolo26m.pt")

print(f"  Model   : YOLO26m (NMS-free, MuSGD, ProgLoss)")
print(f"  Data    : {DATA_YAML}")
print(f"  Output  : {PROJECT_DIR}/{MODEL_NAME}")
print(f"  Epochs  : 100  |  Patience : 20")
print(f"  Batch   : {BATCH}  |  ImgSz   : 640")
print(f"  Cache   : {CACHE}")
print("=" * 54 + "\n")


# ─────────────────────────────────────────────────────────
#  TRAINING — YOLO26m optimised for PPE on T4
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
    # batch=16 proven safe on T4 for 640px (from v8m fix).
    # YOLO26m has no DFL head → slightly lighter than v8m
    # but keep 16 to avoid the 12.5G GPU mem issue we saw.
    batch       = BATCH,
    workers     = 4,

    # ── Precision ────────────────────────────────────────
    # amp=True critical for T4 — FP16 on Tensor Cores
    # YOLO26's simplified head (no DFL, no NMS) makes
    # AMP even more effective than on v8/v11
    amp         = True,

    # ── Cache ────────────────────────────────────────────
    cache       = CACHE,

    # ── Early Stopping ───────────────────────────────────
    patience    = 20,

    # ── Checkpointing ────────────────────────────────────
    save        = True,
    save_period = 10,

    # ── Freeze Strategy ──────────────────────────────────
    # Same fix that worked for v8m:
    # freeze=10 → GPU mem stays at ~5GB (not 12.5GB)
    # epoch 1 mAP starts at 0.55+ instead of 0.37
    # backbone already knows edges/shapes from COCO pretraining
    freeze      = 10,

    # ── Optimizer ────────────────────────────────────────
    # YOLO26 introduced MuSGD internally but the Ultralytics
    # API still uses standard optimizers externally.
    # AdamW works best for fine-tuning YOLO26 on custom data.
    # SGD used in original YOLO26 paper for COCO training
    # but AdamW converges faster on domain-specific PPE data.
    optimizer       = "AdamW",
    lr0             = 0.001,    # fine-tune LR — not too aggressive
    lrf             = 0.01,     # cosine decay → final LR = 1e-5
    momentum        = 0.937,
    weight_decay    = 0.0005,
    warmup_epochs   = 3.0,      # 3 epoch warmup — YOLO26 converges fast
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,

    # ── LR Schedule ──────────────────────────────────────
    cos_lr      = True,         # cosine decay — smoother convergence

    # ── Augmentation ─────────────────────────────────────
    # YOLO26 with freeze=10 → safe to use full mosaic
    # from epoch 1 (backbone stable, only head trains early)
    # STAL (Small Target Aware) in YOLO26 benefits greatly
    # from mosaic — creates more small-object scenarios
    mosaic      = 1.0,          # essential for crowded PPE scenes
    mixup       = 0.1,
    copy_paste  = 0.1,          # helps rare PPE classes
    close_mosaic= 10,           # disable last 10 epochs for clean tail

    hsv_h       = 0.015,        # hue — construction site lighting varies
    hsv_s       = 0.7,
    hsv_v       = 0.4,          # brightness — night cams, shadows, flash
    degrees     = 5.0,
    translate   = 0.1,
    scale       = 0.5,
    fliplr      = 0.5,
    flipud      = 0.0,          # never — people always upright
    shear       = 0.0,

    # ── Loss Weights ─────────────────────────────────────
    # YOLO26 removed DFL so dfl weight has no effect
    # box and cls remain same as v8/v11
    box         = 7.5,
    cls         = 0.5,

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
#  FINAL SUMMARY — full comparison across all your models
# ─────────────────────────────────────────────────────────

models_ref = {
    "YOLO11m (50ep)" : 0.7897,
    "YOLOv8m (35ep)" : 0.774,
}

yolo11m_mins = 410.78
delta_time   = elapsed - yolo11m_mins

print(f"\n{'='*54}")
print(f"  YOLO26M TRAINING COMPLETE!")
print(f"{'='*54}")
print(f"  {'Metric':<20} {'Value':>10}")
print(f"  {'─'*34}")
print(f"  {'mAP50':<20} {r['mAP50']:>10.4f}")
print(f"  {'mAP50-95':<20} {r['mAP50_95']:>10.4f}")
print(f"  {'Precision':<20} {r['precision']:>10.4f}")
print(f"  {'Recall':<20} {r['recall']:>10.4f}")
print(f"  {'Time (mins)':<20} {elapsed:>10.1f}")
print(f"  {'─'*34}")

# Full leaderboard
print(f"\n  {'─'*46}")
print(f"  {'Model':<20} {'mAP50':>8}  {'Winner'}")
print(f"  {'─'*46}")
all_results = {**models_ref, "YOLO26m (this)": r["mAP50"]}
best_map    = max(all_results.values())
for name, score in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    star = "  ✅ BEST" if score == best_map else ""
    print(f"  {name:<20} {score:>8.4f}{star}")
print(f"  {'─'*46}")

print(f"\n  Best weights → {r['best_weights']}")
print(f"  Results JSON → {json_path}")
print(f"{'='*54}\n")