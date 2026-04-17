"""
=============================================================
  FILE: train_all.py
  Trains YOLOv8m, YOLOv9m, YOLO11m, YOLO26m with weighted
  BCE loss in a single script run.

  Usage:
      python train_all.py

  Prerequisites:
      - weighted_trainer.py in the same directory
      - data.yaml configured correctly
      - pretrained weights in weights/ folder
      - GPU with ≥ 8 GB VRAM (16 GB recommended for batch=16)

  Outputs:
      runs/weighted/<model_name>/weights/best.pt   (per model)
      results/train_summary.json                   (all metrics)
=============================================================
"""

import os
import json
import time
import traceback
from multiprocessing import freeze_support

import torch
from ultralytics import YOLO

from weighted import WeightedBCETrainer

# ── Paths & settings ──────────────────────────────────────
DATA_YAML  = "C:/Users/metav/Desktop/project/ppe_object_detection/ppe_project/Dataset/data.yaml"
EPOCHS     = 50
BATCH      = 16         # Increased to use more of 12GB VRAM
IMG_SIZE   = 640
PATIENCE   = 15          # early-stop: 15/50 is proportional
DEVICE     = 0
PROJECT    = "C:/Users/metav/Desktop/project/ppe_object_detection/runs/weighted"
SEED       = 42
CHECKPOINT_DIR = "checkpoints"  # Directory for saving training checkpoints
SAVE_PERIOD = 5         # Save checkpoint every N epochs

# ── Model weights — point to your existing best.pt ────────
MODELS = {
    "yolov8m":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",
    "yolov9m":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "yolo11m":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
    "yolo26m":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",
}

# ── Shared hyperparameters ────────────────────────────────
SHARED_ARGS = dict(
    data         = DATA_YAML,
    epochs       = EPOCHS,
    batch        = BATCH,
    imgsz        = IMG_SIZE,
    patience     = PATIENCE,
    device       = DEVICE,
    optimizer    = "AdamW",
    lr0          = 0.0005,      # lower LR since fine-tuning from best.pt
    lrf          = 1e-5,
    weight_decay = 0.0005,
    warmup_epochs= 3,
    cos_lr       = True,
    mosaic       = 1.0,
    mixup        = 0.1,
    copy_paste   = 0.1,
    close_mosaic = 10,
    amp          = True,        # mixed precision = faster + less VRAM
    workers      = 0,           # Windows-safe single-process dataloader
    cache        = False,        # cache images in RAM for speed
    seed         = SEED,
    verbose      = True,
    save_period  = SAVE_PERIOD, # Save checkpoint every N epochs
    save         = True,        # Enable saving of checkpoints
    trainer      = WeightedBCETrainer,
)

# ─────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    summary = {}

    print("\n" + "=" * 60)
    print("  WEIGHTED BCE TRAINING  —  4 MODELS")
    print(f"  Epochs : {EPOCHS}   Patience : {PATIENCE}")
    print(f"  Batch  : {BATCH}    ImgSize  : {IMG_SIZE}")
    print(f"  Device : {DEVICE}")
    print("=" * 60)

    for model_key, weight_path in MODELS.items():

        print(f"\n{'=' * 60}")
        print(f"  [{model_key.upper()}]  {weight_path}")
        print(f"{'=' * 60}\n")

        if not os.path.exists(weight_path):
            msg = f"Weight file not found: {weight_path}"
            print(f"  [SKIP] {msg}")
            summary[model_key] = {"status": "skipped", "error": msg}
            continue

        t0 = time.time()

        try:
            model = YOLO(weight_path)

            results = model.train(
                **SHARED_ARGS,
                project = PROJECT,
                name    = f"{model_key}_weighted",
            )

            elapsed = (time.time() - t0) / 3600
            rd      = results.results_dict

            metrics = {
                "status":      "success",
                "map50":       float(rd.get("metrics/mAP50(B)",     0)),
                "map50_95":    float(rd.get("metrics/mAP50-95(B)",  0)),
                "precision":   float(rd.get("metrics/precision(B)", 0)),
                "recall":      float(rd.get("metrics/recall(B)",    0)),
                "best_epoch":  int(results.best_epoch)
                               if hasattr(results, "best_epoch") else -1,
                "train_hours": round(elapsed, 3),
                "save_dir":    str(results.save_dir),
                "best_pt":     str(os.path.join(results.save_dir,
                                                 "weights", "best.pt")),
                "checkpoints_dir": str(os.path.join(results.save_dir,
                                                     "weights")),
                "save_period": SAVE_PERIOD,
            }
            summary[model_key] = metrics

            print(f"\n  ✓  {model_key} training complete")
            print(f"     mAP@0.50      : {metrics['map50']:.4f}")
            print(f"     mAP@0.50:0.95 : {metrics['map50_95']:.4f}")
            print(f"     Precision     : {metrics['precision']:.4f}")
            print(f"     Recall        : {metrics['recall']:.4f}")
            print(f"     Best epoch    : {metrics['best_epoch']}")
            print(f"     Duration      : {elapsed:.2f} hrs")
            print(f"     Best weights  : {metrics['best_pt']}")
            print(f"     Checkpoints   : {metrics['checkpoints_dir']}")

        except Exception as exc:
            elapsed = (time.time() - t0) / 3600
            tb      = traceback.format_exc()
            print(f"\n  [ERROR] {model_key} failed after {elapsed:.2f} hrs")
            print(tb)
            summary[model_key] = {
                "status":      "failed",
                "error":       str(exc),
                "train_hours": round(elapsed, 3),
            }

        finally:
            # Free GPU memory before the next run
            try:
                del model
            except NameError:
                pass
            torch.cuda.empty_cache()

    # ── Save training summary ─────────────────────────────────
    out_path = "results/train_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<12} {'Status':<10} {'mAP50':>8} {'mAP50-95':>10} {'Hours':>7}")
    print("  " + "-" * 52)
    for m, r in summary.items():
        if r.get("status") == "success":
            print(f"  {m:<12} {'✓':^10} {r['map50']:>8.4f} {r['map50_95']:>10.4f} {r['train_hours']:>7.2f}")
        else:
            print(f"  {m:<12} {r.get('status','?'):^10} {'—':>8} {'—':>10} {r.get('train_hours', 0):>7.2f}")

    print(f"\n  Results saved to : {out_path}")
    print("  Next step        : python evaluate_all.py")
    print("=" * 60)


if __name__ == "__main__":
    freeze_support()
    main()
