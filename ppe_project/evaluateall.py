"""
=============================================================
  FILE: evaluate_all.py
  Evaluates ALL 8 model variants on the test split:
    — 4 baseline models  (weights/yolov*_best.pt)
    — 4 weighted models  (runs/weighted/<name>/weights/best.pt)

  Saves:
    results/eval_results.json   (full per-class AP50)
    results/eval_summary.csv    (human-readable table)

  Usage:
      python evaluate_all.py

  Run AFTER train_all.py completes.
=============================================================
"""

import json
import os
from multiprocessing import freeze_support
import csv

from ultralytics import YOLO

import torch

# ── Config ────────────────────────────────────────────────
DATA_YAML = "C:/Users/metav/Desktop/project/ppe_object_detection/ppe_project/Dataset/data.yaml"
IMG_SIZE  = 640
DEVICE    = 0

# Violation classes to highlight in the report
KEY_CLASSES = {"NO-Safety Vest", "NO-Hardhat", "NO-Mask", "NO-Gloves"}

# ── Model registry ────────────────────────────────────────
MODELS = {
    # Baselines — your already-trained models
    "yolov8m_baseline":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov8m_ppe/weights/best.pt",
    "yolov9m_baseline":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov9m_ppe/weights/best.pt",
    "yolo11m_baseline":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolov11m_ppe/weights/best.pt",
    "yolo26m_baseline":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/detect/ppe_project/models/yolo26m_ppe/weights/best.pt",
    # Weighted — produced by train_all.py
    "yolov8m_weighted":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/weighted/yolov8m_weighted/weights/best.pt",
    "yolov9m_weighted":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/weighted/yolov9m_weighted/weights/best.pt",
    "yolo11m_weighted":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/weighted/yolo11m_weighted/weights/best.pt",
    "yolo26m_weighted":  "C:/Users/metav/Desktop/project/ppe_object_detection/runs/weighted/yolo26m_weighted/weights/best.pt",
}

os.makedirs("C:/Users/metav/Desktop/project/ppe_object_detection/ppe_project/results", exist_ok=True)

# ─────────────────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────────────────

def main():
    all_results = {}

    for run_name, weight_path in MODELS.items():

        if not os.path.exists(weight_path):
            print(f"[SKIP] {run_name}: {weight_path} not found")
            all_results[run_name] = {"status": "missing"}
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {run_name}")
        print(f"  Weights   : {weight_path}")
        print(f"{'=' * 60}")

        model = YOLO(weight_path)

        try:
            metrics = model.val(
                data    = DATA_YAML,
                split   = "test",
                imgsz   = IMG_SIZE,
                batch   = 16,
                device  = DEVICE,
                workers = 0,
                verbose = False,
            )

            # ── Parse class names ──────────────────────────────
            raw_names = metrics.names
            class_names = (
                [raw_names[k] for k in sorted(raw_names)]
                if isinstance(raw_names, dict)
                else list(raw_names)
            )

            # ── Per-class AP50 ─────────────────────────────────
            ap50_list = (
                metrics.box.ap50.tolist()
                if hasattr(metrics.box, "ap50")
                else []
            )

            per_class = {
                cls: round(ap, 4)
                for cls, ap in zip(class_names, ap50_list)
            }

            record = {
                "status":    "success",
                "map50":     round(float(metrics.box.map50), 4),
                "map50_95":  round(float(metrics.box.map),   4),
                "precision": round(float(metrics.box.mp),    4),
                "recall":    round(float(metrics.box.mr),    4),
                "per_class": per_class,
            }
            all_results[run_name] = record

            print(f"\n  mAP@0.50      : {record['map50']:.4f}")
            print(f"  mAP@0.50:0.95 : {record['map50_95']:.4f}")
            print(f"  Precision     : {record['precision']:.4f}")
            print(f"  Recall        : {record['recall']:.4f}")
            print(f"\n  Per-class AP@0.50:")
            for cls, ap in per_class.items():
                flag = "  ← KEY VIOLATION" if cls in KEY_CLASSES else ""
                weak = " [WEAK]" if ap < 0.50 else ""
                print(f"    {cls:<22} {ap:.4f}{weak}{flag}")

        except Exception as exc:
            print(f"  [ERROR] {run_name}: {exc}")
            all_results[run_name] = {"status": "error", "error": str(exc)}

        finally:
            del model
            torch.cuda.empty_cache()

    # ── Save full JSON ────────────────────────────────────────
    json_path = "C:/Users/metav/Desktop/project/ppe_object_detection/ppe_project/results/eval_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to: {json_path}")

    # ── Build comparison table & CSV ──────────────────────────
    MODEL_BASES = ["yolov8m", "yolov9m", "yolo11m", "yolo26m"]
    OVERALL_COLS = ["map50", "map50_95", "precision", "recall"]

    csv_path = "C:/Users/metav/Desktop/project/ppe_object_detection/ppe_project/results/eval_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # ── Section 1: Overall metrics ────────────────────────
        writer.writerow(["=== OVERALL METRICS ==="])
        writer.writerow(["Model", "Variant", "mAP@0.50", "mAP@0.50:0.95",
                         "Precision", "Recall"])

        for base in MODEL_BASES:
            for variant in ("baseline", "weighted"):
                key = f"{base}_{variant}"
                r   = all_results.get(key, {})
                if r.get("status") != "success":
                    writer.writerow([base, variant, "N/A", "N/A", "N/A", "N/A"])
                else:
                    writer.writerow([
                        base, variant,
                        r["map50"], r["map50_95"],
                        r["precision"], r["recall"],
                    ])
            writer.writerow([])  # blank row between models

        # ── Section 2: Per-class AP50 for violation classes ───
        writer.writerow([])
        writer.writerow(["=== VIOLATION CLASS AP@0.50: BASELINE vs WEIGHTED ==="])
        writer.writerow(["Class", "Model", "Baseline", "Weighted",
                         "Delta", "% Gain"])

        for cls in sorted(KEY_CLASSES):
            for base in MODEL_BASES:
                b_ap = (all_results.get(f"{base}_baseline", {})
                                    .get("per_class", {}).get(cls))
                w_ap = (all_results.get(f"{base}_weighted", {})
                                    .get("per_class", {}).get(cls))

                if b_ap is not None and w_ap is not None:
                    delta  = round(w_ap - b_ap, 4)
                    pct    = round((delta / b_ap) * 100, 1) if b_ap > 0 else 0
                    writer.writerow([cls, base, b_ap, w_ap, delta, f"{pct}%"])
                else:
                    writer.writerow([cls, base, "N/A", "N/A", "—", "—"])
            writer.writerow([])

    print(f"  Summary CSV saved to : {csv_path}")

    # ─────────────────────────────────────────────────────────
    # CONSOLE REPORT
    # ─────────────────────────────────────────────────────────

    SEP = "=" * 74

    print(f"\n{SEP}")
    print("  OVERALL mAP@0.50 — BASELINE vs WEIGHTED")
    print(SEP)
    print(f"  {'Model':<12} {'Baseline':>10} {'Weighted':>10} {'Δ mAP50':>10}")
    print("  " + "-" * 46)
    for base in MODEL_BASES:
        b = all_results.get(f"{base}_baseline", {}).get("map50")
        w = all_results.get(f"{base}_weighted", {}).get("map50")
        if b is not None and w is not None:
            print(f"  {base:<12} {b:>10.4f} {w:>10.4f} {w-b:>+10.4f}")
        else:
            print(f"  {base:<12} {'N/A':>10} {'N/A':>10} {'—':>10}")

    print(f"\n{SEP}")
    print("  VIOLATION CLASS AP@0.50 — BASELINE vs WEIGHTED (Δ)")
    print(SEP)
    print(f"  {'Class':<22} {'Model':<12} {'Base':>7} {'Wtd':>7} {'Δ':>7} {'%':>7}")
    print("  " + "-" * 62)

    for cls in sorted(KEY_CLASSES):
        first = True
        for base in MODEL_BASES:
            b_ap = (all_results.get(f"{base}_baseline", {})
                                .get("per_class", {}).get(cls))
            w_ap = (all_results.get(f"{base}_weighted", {})
                                .get("per_class", {}).get(cls))
            label = cls if first else ""
            first = False

            if b_ap is not None and w_ap is not None:
                delta = w_ap - b_ap
                pct   = (delta / b_ap) * 100 if b_ap > 0 else 0
                gain  = f"{pct:+.1f}%"
                print(f"  {label:<22} {base:<12} {b_ap:>7.3f} {w_ap:>7.3f} "
                      f"{delta:>+7.3f} {gain:>7}")
            else:
                print(f"  {label:<22} {base:<12} {'N/A':>7} {'N/A':>7} {'—':>7} {'—':>7}")
        print()

    print(SEP)
    print("  Next step: python generate_report.py")
    print(SEP)

if __name__ == "__main__":
    freeze_support()
    main()
