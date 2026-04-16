"""
=============================================================
  FILE: generate_report.py
  Reads results/eval_results.json and produces:
    results/report.html   — rich, self-contained HTML report

  Usage:
      python generate_report.py

  Run AFTER evaluate_all.py completes.
=============================================================
"""

import json
import os
from datetime import datetime

# ── Load results ──────────────────────────────────────────
JSON_PATH   = "results/eval_results.json"
OUTPUT_PATH = "results/report.html"

with open(JSON_PATH) as f:
    data = json.load(f)

MODEL_BASES    = ["yolov8m", "yolov9m", "yolo11m", "yolo26m"]
KEY_CLASSES    = ["NO-Safety Vest", "NO-Hardhat", "NO-Mask", "NO-Gloves"]
OVERALL_METRICS = [
    ("mAP@0.50",       "map50"),
    ("mAP@0.50:0.95",  "map50_95"),
    ("Precision",      "precision"),
    ("Recall",         "recall"),
]

now = datetime.now().strftime("%Y-%m-%d %H:%M")

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def safe(key, run, sub=None):
    r = data.get(run, {})
    if sub:
        return r.get("per_class", {}).get(sub)
    return r.get(key)

def fmt(v, pct=False):
    if v is None:
        return "<td class='na'>—</td>"
    color = ""
    if isinstance(v, float):
        if v >= 0.75:
            color = " class='hi'"
        elif v < 0.50:
            color = " class='lo'"
    val = f"{v*100:.1f}" if pct else f"{v:.4f}"
    return f"<td{color}>{val}</td>"

def delta_cell(b, w):
    if b is None or w is None:
        return "<td class='na'>—</td>"
    d   = w - b
    pct = (d / b * 100) if b > 0 else 0
    cls = "pos-delta" if d > 0 else ("neg-delta" if d < 0 else "")
    sign = "+" if d >= 0 else ""
    return (
        f"<td class='{cls}'>{sign}{d:.4f}<br>"
        f"<span class='sub'>{sign}{pct:.1f}%</span></td>"
    )

# ─────────────────────────────────────────────────────────
# BUILD HTML
# ─────────────────────────────────────────────────────────

def overall_table():
    rows = []
    for base in MODEL_BASES:
        bk = f"{base}_baseline"
        wk = f"{base}_weighted"
        b  = data.get(bk, {})
        w  = data.get(wk, {})
        ok = b.get("status") == "success"
        wok = w.get("status") == "success"

        def cell(run_data, key):
            v = run_data.get(key) if run_data.get("status") == "success" else None
            return fmt(v)

        rows.append(f"""
        <tr>
          <td class="model-name">{base}</td>
          {"".join(cell(b, k) for _, k in OVERALL_METRICS)}
          {"".join(cell(w, k) for _, k in OVERALL_METRICS)}
          {delta_cell(b.get('map50') if ok else None, w.get('map50') if wok else None)}
        </tr>""")

    hdrs_base    = "".join(f"<th>{lbl}</th>" for lbl, _ in OVERALL_METRICS)
    hdrs_wtd     = hdrs_base

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="{len(OVERALL_METRICS)}" class="group-hdr baseline-hdr">Baseline</th>
          <th colspan="{len(OVERALL_METRICS)}" class="group-hdr weighted-hdr">Weighted</th>
          <th rowspan="2">ΔmAP@0.50</th>
        </tr>
        <tr>{hdrs_base}{hdrs_wtd}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def violation_table():
    rows = []
    for cls in KEY_CLASSES:
        first = True
        for base in MODEL_BASES:
            bk = f"{base}_baseline"
            wk = f"{base}_weighted"
            b_ap = safe(None, bk, cls)
            w_ap = safe(None, wk, cls)
            cls_cell = (
                f'<td class="cls-name" rowspan="{len(MODEL_BASES)}">{cls}</td>'
                if first else ""
            )
            first = False
            rows.append(f"""
            <tr>
              {cls_cell}
              <td class="model-name">{base}</td>
              {fmt(b_ap)}
              {fmt(w_ap)}
              {delta_cell(b_ap, w_ap)}
            </tr>""")
        rows.append('<tr class="sep"><td colspan="5"></td></tr>')

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Violation Class</th>
          <th>Model</th>
          <th class="baseline-hdr">Baseline AP@0.50</th>
          <th class="weighted-hdr">Weighted AP@0.50</th>
          <th>Δ (abs / %)</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def perclass_table():
    # Collect all class names from first successful model
    all_cls = []
    for key, r in data.items():
        if r.get("status") == "success" and r.get("per_class"):
            all_cls = list(r["per_class"].keys())
            break

    if not all_cls:
        return "<p>No per-class data available.</p>"

    cols = []
    for base in MODEL_BASES:
        bk = f"{base}_baseline"
        wk = f"{base}_weighted"
        cols.append((base, bk, wk))

    header_model = "".join(
        f'<th colspan="2" class="group-hdr">{base}</th>'
        for base, _, _ in cols
    )
    header_sub = "".join(
        '<th class="baseline-hdr">Base</th><th class="weighted-hdr">Wtd</th>'
        for _ in cols
    )

    rows = []
    for cls in all_cls:
        is_key = cls in KEY_CLASSES
        row_cls = ' class="key-row"' if is_key else ""
        label   = f'{cls} <span class="key-badge">KEY</span>' if is_key else cls
        cells   = "".join(
            fmt(data.get(bk, {}).get("per_class", {}).get(cls)) +
            fmt(data.get(wk, {}).get("per_class", {}).get(cls))
            for _, bk, wk in cols
        )
        rows.append(f"<tr{row_cls}><td class='cls-name'>{label}</td>{cells}</tr>")

    return f"""
    <table class="data-table wide">
      <thead>
        <tr><th rowspan="2">Class</th>{header_model}</tr>
        <tr>{header_sub}</tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────
# FULL HTML DOCUMENT
# ─────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPE Detection — Weighted Loss Report</title>
<style>
  :root {{
    --bg:         #0d1117;
    --surface:    #161b22;
    --surface2:   #1c2230;
    --border:     #30363d;
    --text:       #e6edf3;
    --muted:      #8b949e;
    --baseline:   #388bfd;
    --weighted:   #3fb950;
    --delta-pos:  #3fb950;
    --delta-neg:  #f85149;
    --hi:         #3fb950;
    --lo:         #f85149;
    --key:        #d29922;
    --accent:     #58a6ff;
    --font-mono:  'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    --font-sans:  'Inter', 'Segoe UI', system-ui, sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.6;
    padding: 0 0 80px;
  }}

  /* ── Header ── */
  header {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #1a2535 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 40px 36px;
  }}
  header h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: var(--text);
  }}
  header h1 span {{ color: var(--accent); }}
  header .meta {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 13px;
    font-family: var(--font-mono);
  }}
  header .pills {{
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
  }}
  .pill {{
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
  }}
  .pill-blue  {{ background: rgba(56, 139, 253, .15); color: var(--baseline); border: 1px solid rgba(56,139,253,.3); }}
  .pill-green {{ background: rgba(63, 185, 80, .15);  color: var(--weighted); border: 1px solid rgba(63,185,80,.3); }}
  .pill-gold  {{ background: rgba(210, 153, 34, .15); color: var(--key);      border: 1px solid rgba(210,153,34,.3); }}

  /* ── Layout ── */
  main {{ max-width: 1280px; margin: 0 auto; padding: 40px 24px 0; }}

  section {{ margin-bottom: 52px; }}
  section h2 {{
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  section h2::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 4px;
  }}
  .section-desc {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 18px;
  }}

  /* ── Tables ── */
  .table-wrap {{ overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }}

  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    font-size: 13px;
  }}
  table.data-table.wide {{ min-width: 960px; }}

  thead tr {{ background: var(--surface2); }}
  thead th {{
    padding: 10px 14px;
    text-align: center;
    font-weight: 600;
    font-size: 12px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}
  th.group-hdr  {{ font-size: 13px; border-bottom: none; padding-bottom: 2px; }}
  th.baseline-hdr {{ color: var(--baseline); }}
  th.weighted-hdr {{ color: var(--weighted); }}

  tbody tr {{ border-bottom: 1px solid rgba(48,54,61,.6); transition: background .15s; }}
  tbody tr:hover {{ background: var(--surface2); }}
  tbody tr:last-child {{ border-bottom: none; }}
  tr.sep td {{ padding: 0; height: 6px; background: var(--bg); border: none; }}
  tr.key-row {{ background: rgba(210, 153, 34, .05); }}

  td {{
    padding: 9px 14px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--text);
  }}
  td.model-name, td.cls-name {{
    text-align: left;
    font-family: var(--font-sans);
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
  }}
  td.hi {{ color: var(--hi); font-weight: 600; }}
  td.lo {{ color: var(--lo); }}
  td.na {{ color: var(--muted); }}
  td.pos-delta {{ color: var(--delta-pos); font-weight: 600; }}
  td.neg-delta {{ color: var(--delta-neg); }}
  .sub {{ font-size: 10px; opacity: .7; }}

  .key-badge {{
    background: rgba(210,153,34,.2);
    color: var(--key);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
    font-family: var(--font-sans);
    margin-left: 4px;
    vertical-align: middle;
  }}

  /* ── Legend ── */
  .legend {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    font-size: 12px;
    color: var(--muted);
  }}
  .legend-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 2px;
    margin-right: 5px;
    vertical-align: middle;
  }}

  /* ── Footer ── */
  footer {{
    margin-top: 60px;
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--font-mono);
    padding: 0 20px;
  }}
</style>
</head>
<body>

<header>
  <h1>PPE Detection — <span>Weighted Loss Experiment</span></h1>
  <p class="meta">Generated: {now}  ·  Models: YOLOv8m · YOLOv9m · YOLO11m · YOLO26m</p>
  <div class="pills">
    <span class="pill pill-blue">Baseline (standard BCE)</span>
    <span class="pill pill-green">Weighted (per-class BCE)</span>
    <span class="pill pill-gold">4 violation classes tracked</span>
  </div>
</header>

<main>

  <!-- ── SECTION 1: OVERALL ── -->
  <section>
    <h2>Overall Metrics — All Models</h2>
    <p class="section-desc">
      Comparison of mAP, Precision, and Recall on the test split.
      ΔmAP@0.50 = Weighted − Baseline.
    </p>
    <div class="legend">
      <span><span class="legend-dot" style="background:var(--baseline)"></span>Baseline</span>
      <span><span class="legend-dot" style="background:var(--weighted)"></span>Weighted</span>
      <span><span class="legend-dot" style="background:var(--hi)"></span>≥ 0.75</span>
      <span><span class="legend-dot" style="background:var(--lo)"></span>&lt; 0.50</span>
    </div>
    <div class="table-wrap">{overall_table()}</div>
  </section>

  <!-- ── SECTION 2: VIOLATION CLASSES ── -->
  <section>
    <h2>Violation Class AP@0.50 — Baseline vs Weighted</h2>
    <p class="section-desc">
      Per-class Average Precision at IoU=0.50 for the four safety-critical
      violation categories that received elevated BCE weights.
    </p>
    <div class="table-wrap">{violation_table()}</div>
  </section>

  <!-- ── SECTION 3: FULL PER-CLASS ── -->
  <section>
    <h2>Full Per-Class AP@0.50</h2>
    <p class="section-desc">
      Every class across all 8 model variants (4 baselines + 4 weighted).
      Key violation classes are highlighted in gold.
    </p>
    <div class="table-wrap">{perclass_table()}</div>
  </section>

</main>

<footer>
  PPE Detection Experiment Report · {now} ·
  Baseline = standard BCE · Weighted = per-class pos_weight BCE
</footer>

</body>
</html>"""

# ── Write file ────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n✓ Report written to: {OUTPUT_PATH}")
print("  Open in a browser to view the full comparison table.")