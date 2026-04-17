import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_paths = {
    "YOLOv8":  "runs/detect/ppe_project/models/yolov8m_ppe/results.csv",
    "YOLOv9":  "runs/detect/ppe_project/models/yolov9m_ppe/results.csv",
    "YOLOv11": "runs/detect/ppe_project/models/yolov11m_ppe/results.csv",
    "YOLOv26": "runs/detect/ppe_project/models/yolo26m_ppe/results.csv",
}

# ✅ Only 4 metrics — no Speed
METRICS = ["Precision", "Recall", "mAP50", "mAP50-95"]

MODEL_STYLES = {
    "YOLOv8":  {"color": "#2196F3", "linestyle": "-",  "marker": "o"},  # blue
    "YOLOv9":  {"color": "#FF5722", "linestyle": "-",  "marker": "o"},  # deep orange
    "YOLOv11": {"color": "#FFC107", "linestyle": "-",  "marker": "o"},  # amber
    "YOLOv26": {"color": "#4CAF50", "linestyle": "-",  "marker": "o"},  # green
}

def get_best_metrics(path):
    df = pd.read_csv(path)
    best_idx = df["metrics/mAP50-95(B)"].idxmax()
    best = df.loc[best_idx]
    return [
        best["metrics/precision(B)"],
        best["metrics/recall(B)"],
        best["metrics/mAP50(B)"],
        best["metrics/mAP50-95(B)"],
    ]

model_results = {m: get_best_metrics(p) for m, p in model_paths.items()}

N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRICS, fontsize=12)

for model_name, values in model_results.items():
    vals = values + values[:1]
    style = MODEL_STYLES[model_name]

    ax.plot(
        angles, vals,
        linewidth=2.5,                    # ✅ uniform width for all
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=6,
        color=style["color"],
        label=model_name,
        zorder=3                          # ✅ all lines drawn on top equally
    )
    ax.fill(
        angles, vals,
        alpha=0.07,                       # ✅ very light fill so lines stay visible
        color=style["color"]
    )

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.3, 1.15),
    fontsize=10,
    framealpha=0.8
)

plt.title("Model Performance Radar Chart", size=14, pad=15)
plt.tight_layout()
plt.savefig("ppe_radar_chart2.png", dpi=300, bbox_inches="tight")
plt.show()