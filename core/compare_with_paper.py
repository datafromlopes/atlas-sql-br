# coding=utf-8
# Copyright (C) 2026  Diego Lopes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# =============================================================================
#  Side-by-side comparison of a reproduction run against Table 3 of the SBBD
#  2026 paper (base vs fine-tuned Llama-3.1-8B-Instruct, 195 comparable pairs).
#
#     uv run python core/compare_with_paper.py --experiment 6
#
#  Reads results/summary_v{N}.json (written by evaluate_sql_preds.py) and writes
#  results/paper_comparison_v{N}.md, also printed to the console.
# =============================================================================
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import PROJECT_PATH  # noqa: E402

# Table 3 of the paper. Percent metrics are stored here as fractions.
PAPER = {
    "string_exact":         {"base": 0.000, "finetuned": 0.000, "label": "Exact Match (%)", "pct": True},
    "string_similarity":    {"base": 0.076, "finetuned": 0.302, "label": "String Similarity"},
    "token_precision":      {"base": 0.211, "finetuned": 0.668, "label": "Token Precision"},
    "token_recall":         {"base": 0.416, "finetuned": 0.617, "label": "Token Recall"},
    "token_f1":             {"base": 0.223, "finetuned": 0.623, "label": "Token F1"},
    "structural_precision": {"base": 0.298, "finetuned": 0.351, "label": "Structural Precision"},
    "structural_recall":    {"base": 0.223, "finetuned": 0.308, "label": "Structural Recall"},
    "structural_f1":        {"base": 0.244, "finetuned": 0.320, "label": "Structural F1"},
    "component_jaccard":    {"base": 0.212, "finetuned": 0.233, "label": "Component Jaccard"},
    "geospatial_precision": {"base": 0.267, "finetuned": 0.593, "label": "Geospatial Precision"},
    "geospatial_recall":    {"base": 0.120, "finetuned": 0.470, "label": "Geospatial Recall"},
    "geospatial_f1":        {"base": 0.166, "finetuned": 0.524, "label": "Geospatial F1"},
    "spatial_exact_match":  {"base": 0.000, "finetuned": 0.0769, "label": "Spatial Exact Match (%)", "pct": True},
}
PAPER_COUNTS = {"base": {"tp": 77, "fp": 211}, "finetuned": {"tp": 301, "fp": 207}}
NEW_METRICS = [("execution_accuracy", "Execution Accuracy"), ("executable_rate", "Executable Rate")]


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:.2f}" if pct else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Compare a reproduction with Table 3 of the paper.")
    ap.add_argument("--experiment", type=int, default=6)
    ap.add_argument("--results-dir", default=str(Path(PROJECT_PATH) / "results"))
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    spath = rdir / f"summary_v{args.experiment}.json"
    if not spath.exists():
        sys.exit(f"{spath} not found. Run evaluate_sql_preds.py first.")
    s = json.loads(spath.read_text(encoding="utf-8"))
    ov = s["overall"]

    # structural precision/recall live inside the per-row structural_f1 dict and are
    # not aggregated by the evaluator; the F1 is. Mark them unavailable if absent.
    lines = [f"# Reproduction v{args.experiment} vs. paper (Table 3)", "",
             f"Records averaged: paper 195 comparable of 196 | reproduction {s['n']} of "
             f"{s['n_total']} ({s['n_skipped_no_sql']} without SQL from either model)", "",
             "| Metric | Paper base | Repro base | Δ base | Paper FT | Repro FT | Δ FT |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, spec in PAPER.items():
        pct = spec.get("pct", False)
        rb, rf = ov["base"].get(key), ov["finetuned"].get(key)
        db = None if rb is None else rb - spec["base"]
        df = None if rf is None else rf - spec["finetuned"]
        lines.append(f"| {spec['label']} | {fmt(spec['base'], pct)} | {fmt(rb, pct)} | {fmt(db, pct)} "
                     f"| {fmt(spec['finetuned'], pct)} | {fmt(rf, pct)} | {fmt(df, pct)} |")
    lines += ["", "## Spatial function call counts", "",
              "| Model | Paper TP | Repro TP | Paper FP | Repro FP | Repro FN |", "|---|---:|---:|---:|---:|---:|"]
    for m in ("base", "finetuned"):
        c = ov[m].get("geospatial_counts", {})
        lines.append(f"| {m} | {PAPER_COUNTS[m]['tp']} | {c.get('tp', 'n/a')} | {PAPER_COUNTS[m]['fp']} "
                     f"| {c.get('fp', 'n/a')} | {c.get('fn', 'n/a')} |")
    lines += ["", "## New metrics (not in the paper): execution against the reference database", "",
              "| Metric | Base | Fine-tuned | Δ |", "|---|---:|---:|---:|"]
    for key, label in NEW_METRICS:
        b, f = ov["base"].get(key), ov["finetuned"].get(key)
        d = None if (b is None or f is None) else f - b
        lines.append(f"| {label} | {fmt(b)} | {fmt(f)} | {fmt(d)} |")
    if s.get("by_level"):
        lines += ["", "## By complexity tier (fine-tuned)", "",
                  "| Tier | n | Exec. Acc. | Token F1 | Structural F1 | Geospatial F1 | Spatial EM |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for lvl, per in s["by_level"].items():
            f = per["finetuned"]
            lines.append(f"| {lvl} | {f['n']} | {fmt(f.get('execution_accuracy'))} | {fmt(f.get('token_f1'))} "
                         f"| {fmt(f.get('structural_f1'))} | {fmt(f.get('geospatial_f1'))} "
                         f"| {fmt(f.get('spatial_exact_match'), True)} |")
    out = rdir / f"paper_comparison_v{args.experiment}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
