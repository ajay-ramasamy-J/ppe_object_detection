"""
=============================================================
  FILE: weighted_trainer.py
  Shared custom trainer — patched BCE with per-class weights.
  Import this in train_all.py and evaluate_all.py.
=============================================================
"""

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer


# ── Per-class positive weights ────────────────────────────
# Keys MUST exactly match your data.yaml class names.
CLASS_WEIGHT_DICT = {
    "NO-Safety Vest": 4.0,
    "NO-Gloves":      2.5,
    "NO-Hardhat":     2.0,
    "NO-Mask":        2.0,
    "Safety Vest":    1.5,
}


def build_pos_weight(class_names: list[str], weight_dict: dict, device) -> torch.Tensor:
    """Return a (nc,) pos_weight tensor for BCEWithLogitsLoss."""
    nc = len(class_names)
    w  = torch.ones(nc, device=device)
    print("\n[WeightedTrainer] Per-class BCE weights:")
    for i, name in enumerate(class_names):
        if name in weight_dict:
            w[i] = weight_dict[name]
            print(f"  ✓  [{i:2d}] '{name}'  →  {w[i]:.1f}")
        else:
            print(f"       [{i:2d}] '{name}'  →  1.0 (default)")
    return w


# ── Helper: try every known patch point ──────────────────

def _patch_bce(obj, pw: torch.Tensor, label: str) -> bool:
    """
    Attempt to replace obj.bce with a weighted BCEWithLogitsLoss.
    Returns True on success.
    """
    if obj is not None and hasattr(obj, "bce"):
        obj.bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        print(f"[WeightedTrainer] ✓ Patched via {label}")
        return True
    return False


class WeightedBCETrainer(DetectionTrainer):
    """
    Drop-in DetectionTrainer subclass.
    Injects per-class pos_weight into BCEWithLogitsLoss.
    Compatible with YOLOv8, YOLOv9, YOLO11, YOLO26.
    """

    # ── Called by Ultralytics after model & criterion are built ──

    def _setup_criterion(self):
        super()._setup_criterion()
        self._inject_weights(caller="_setup_criterion")

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self._inject_weights(caller="_setup_train")

    # ── Called every step (lazy fallback for late-init models) ──

    def optimizer_step(self):
        if not getattr(self, "_bce_patched", False):
            self._inject_weights(caller="optimizer_step (lazy)")
        super().optimizer_step()

    # ── Core patch logic ──────────────────────────────────

    def _inject_weights(self, caller: str = ""):
        if getattr(self, "_bce_patched", False):
            return  # already done

        try:
            device      = next(self.model.parameters()).device
            names_raw   = self.data.get("names", {})
            class_names = (
                [names_raw[k] for k in sorted(names_raw)]
                if isinstance(names_raw, dict)
                else list(names_raw)
            )
            pw = build_pos_weight(class_names, CLASS_WEIGHT_DICT, device)
        except Exception as e:
            print(f"[WeightedTrainer] Could not build weight tensor: {e}")
            return

        patched = (
            # Attempt 1 — model's own loss object (YOLO11 / newer)
            _patch_bce(
                getattr(getattr(self.model, "model", None), "__getitem__", lambda i: None)(-1)
                if hasattr(getattr(self.model, "model", None), "__getitem__") else None,
                pw, "model.model[-1].loss"
            )
            or _patch_bce(getattr(self.model.model[-1], "loss", None), pw, "model.model[-1].loss")
            # Attempt 2 — self.criterion  (most common)
            or _patch_bce(getattr(self, "criterion", None), pw, "self.criterion")
            # Attempt 3 — model.criterion (YOLOv9 / v8)
            or _patch_bce(getattr(self.model, "criterion", None), pw, "model.criterion")
        )

        if not patched:
            print(f"[WeightedTrainer] ✗ Could not patch BCE (called from: {caller}).")
            print("  Weighted loss is NOT active — training continues with standard BCE.")
            print("  If this persists, check your Ultralytics version.")
        else:
            self._bce_patched = True