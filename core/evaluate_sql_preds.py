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
#  evaluate_sql_preds.py — validates generated SQL against the gold.
#
#  Consumes the prediction files (one per experiment) produced by
#  generate_sql_preds.py, in the format:
#     {sql_validation_id, nivel, question, sql_code_gold,
#      sql_code_base, sql_code_finetuned}
#
#  Uses the project's sql_validation.py framework, applying ALL its metrics via
#  validate_sql (ast equivalence, structural F1, execution accuracy, component
#  matching per clause, string matching and the failure taxonomy). On top of that
#  it runs EXPLAIN on Postgres to flag whether each SQL is executable.
#
#  Output: ONE consolidated JSON report (all experiments) saved to
#  results/experimento_reports.json — SUMMARIES ONLY (per metric, global and per
#  level, with the delta and the fine-tuned failure taxonomy). The per-row detail
#  (slim, no result sets) is written ONLY when --save-details is passed, one file
#  per experiment, so the consolidated report never grows huge again.
#
#  Usage:
#     # score ALL predictions_v*.json found in the default predictions dir:
#     uv run python evaluate_sql_preds.py --dsn "postgresql://user:pass@host:5432/db"
#     # or a specific folder / specific files:
#     uv run python evaluate_sql_preds.py --preds-dir preds --dsn "..."
#     uv run python evaluate_sql_preds.py --predictions preds/predictions_v1.json --dsn "..."
#
#  Without --dsn, tries libpq environment variables (PGHOST, PGDATABASE, PGUSER,
#  PGPASSWORD). With --no-db, skips execution and reports only AST diagnostics.
# =============================================================================
from __future__ import annotations

import os
import re
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# Allow importing sql_validation.py from the same folder or the project root.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from sql_validation import validate_sql  # noqa: E402
from utils import PROJECT_PATH, PREDS_DIR # noqa: E402
from utils.utils import Logger  # noqa: E402

logger = Logger("scoring").setup_logging()

GOLD = "sql_code_gold"
BASE = "sql_code_base"
FT = "sql_code_finetuned"
PREDICTORS = [("base", BASE), ("finetuned", FT)]

PREDS_DIR = Path(PREDS_DIR)

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)

# Where generate_sql_preds.py writes predictions_v{N}.json (one per experiment).
_VER_RE = re.compile(r"_v(\d+)\.json$", re.IGNORECASE)


def _version_key(p: Path):
    """Sort by experiment version (v0, v1, …, v10) instead of alphabetically."""
    m = _VER_RE.search(p.name)
    return (0, int(m.group(1))) if m else (1, p.name.lower())


def discover_prediction_files(preds_dir: Path) -> list[Path]:
    """Find every predictions_v*.json in preds_dir, de-duplicated and ordered by version."""
    files = {p.resolve() for p in preds_dir.glob("predictions_v*.json")}
    return sorted(files, key=_version_key)


def _experiment_label(path: Path) -> str:
    """e.g. predictions_v3.json -> 'v3' (falls back to the file stem)."""
    m = _VER_RE.search(path.name)
    return f"v{m.group(1)}" if m else path.stem

# Wall-clock cadence for the progress bar (seconds between updates).
_PROGRESS_INTERVAL_S = 2.0


def _progress_bar(fraction: float, width: int = 24) -> str:
    """Textual progress bar that works in logs / non-TTY."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "─" * (width - filled)


# ═══════════════════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════════════════
def open_connection(dsn: str | None, timeout_ms: int):
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError("psycopg2 not installed: pip install psycopg2-binary")
    # options sets statement_timeout — prevents a bad generated SQL from hanging scoring.
    conn = psycopg2.connect(dsn or "", options=f"-c statement_timeout={timeout_ms}")
    conn.autocommit = False
    return conn


def resolve_mode(gold_sql: str, mode: str) -> str:
    """compare-mode 'auto': uses 'list' when the gold has ORDER BY (order matters,
    e.g. rankings), otherwise 'multiset'. Other values are passed through as-is."""
    if mode != "auto":
        return mode
    return "list" if _ORDER_BY_RE.search(gold_sql or "") else "multiset"


def explain_ok(conn, sql: str) -> tuple[bool, str | None]:
    """Is the SQL executable? Validate with EXPLAIN (plan only — no execution),
    inside a SAVEPOINT so a bad query never poisons the outer transaction."""
    sql = (sql or "").strip()
    if not sql:
        return False, "empty sql"
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT _explain_sp")
            try:
                cur.execute(f"EXPLAIN {sql.rstrip(';')}")
                cur.fetchall()
                cur.execute("RELEASE SAVEPOINT _explain_sp")
                return True, None
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT _explain_sp")
                return False, str(e).strip().splitlines()[0]
    except Exception as e:
        return False, str(e).strip().splitlines()[0]


# ═══════════════════════════════════════════════════════════════════════════
# Scoring of a single predictions file
# ═══════════════════════════════════════════════════════════════════════════
def score_file(path: Path, conn, compare_mode: str, use_db: bool) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    detailed = []
    total = len(rows)

    # Accumulators: per model (base/finetuned), global and per level.
    agg = {m: defaultdict(list) for m, _ in PREDICTORS}
    by_level = {m: defaultdict(lambda: defaultdict(list)) for m, _ in PREDICTORS}
    ft_failures = Counter()
    gold_exec_fail = 0
    row_errors = 0

    last_tick = time.monotonic()

    for i, r in enumerate(rows, 1):
        gold = r.get(GOLD, "") or ""
        level = r.get("nivel", r.get("level", "")) or ""
        mode = resolve_mode(gold, compare_mode)

        # Is the GOLD itself executable? (EXPLAIN). A non-executable gold makes the
        # execution comparison meaningless for this row, so we exclude it.
        gold_ok, gold_err = (explain_ok(conn, gold) if use_db else (None, None))
        if use_db and not gold_ok:
            gold_exec_fail += 1

        row_out = {"id": r.get("sql_validation_id"), "nivel": level,
                   "compare_mode": mode, "gold_executable": gold_ok,
                   "gold_explain_error": gold_err, "models": {}}

        for mname, key in PREDICTORS:
            pred = r.get(key, "") or ""

            try:
                # ALL sql_validation metrics in one call: ast equivalence, structural
                # F1, execution accuracy (when conn given), component matching per
                # clause, string matching and the failure taxonomy.
                full = json.loads(validate_sql(pred, gold, conn=conn if use_db else None,
                                               compare_mode=mode))
                sm = full["summary"]

                # executability of the GENERATED SQL (EXPLAIN, plan only)
                pred_ok, pred_err = (explain_ok(conn, pred) if use_db else (None, None))
            except Exception as exc:
                # Never let a single prediction abort the whole evaluation: record
                # the failure for this row/model and move on.
                row_errors += 1
                logger.warning(f"row {i} [{mname}] scoring error: {exc}")
                full = {"summary": {"ast_equivalent": False, "execution_match": False,
                                    "component_avg_jaccard": 0.0, "string_exact": False,
                                    "string_similarity": 0.0, "failures": ["scoring_error"]},
                        "structural_f1": {}}
                sm = full["summary"]
                pred_ok, pred_err = (False, str(exc))

            exec_match = sm.get("execution_match")
            ast_eq = bool(sm.get("ast_equivalent"))
            comp_j = float(sm.get("component_avg_jaccard") or 0.0)
            str_ex = bool(sm.get("string_exact"))
            str_sim = float(sm.get("string_similarity") or 0.0)
            sf1 = float((full.get("structural_f1") or {}).get("f1") or 0.0)

            # execution counts only when the gold is executable (valid comparison)
            if use_db and exec_match is not None and gold_ok:
                agg[mname]["exec"].append(1 if exec_match else 0)
                by_level[mname][level]["exec"].append(1 if exec_match else 0)
            if use_db and pred_ok is not None:
                agg[mname]["executable"].append(1 if pred_ok else 0)
                by_level[mname][level]["executable"].append(1 if pred_ok else 0)
            agg[mname]["ast"].append(1 if ast_eq else 0)
            agg[mname]["comp"].append(comp_j)
            agg[mname]["str_exact"].append(1 if str_ex else 0)
            agg[mname]["str_sim"].append(str_sim)
            agg[mname]["sf1"].append(sf1)
            by_level[mname][level]["ast"].append(1 if ast_eq else 0)

            if mname == "finetuned":
                ft_failures.update(sm.get("failures", []))

            # Slim per-row detail: numbers and flags only. The full validate_sql
            # payload carries predicted_rows/gold_rows (entire result sets) — those
            # are NEVER stored, otherwise the report balloons to gigabytes.
            row_out["models"][mname] = {
                "execution_match": exec_match,
                "executable": pred_ok,
                "ast_equivalent": ast_eq,
                "structural_f1": round(sf1, 4),
                "component_jaccard": round(comp_j, 4),
                "string_exact": str_ex,
                "string_similarity": round(str_sim, 4),
                "failures": sm.get("failures", []),
                "explain_error": pred_err,
            }

        detailed.append(row_out)

        # ── progress (throttled; always emit the final row) ──────────────────
        now = time.monotonic()
        if now - last_tick >= _PROGRESS_INTERVAL_S or i == total:
            frac = i / total if total else 1.0
            if use_db:
                running = (f" | base✓={sum(agg['base']['exec'])} "
                           f"ft✓={sum(agg['finetuned']['exec'])}"
                           + (f" gold✗={gold_exec_fail}" if gold_exec_fail else ""))
            else:
                running = (f" | base ast✓={sum(agg['base']['ast'])} "
                           f"ft ast✓={sum(agg['finetuned']['ast'])}")
            logger.info(f"[{_progress_bar(frac)}] {frac * 100:5.1f}% "
                        f"row {i}/{total}{running}")
            last_tick = now

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    summary = {"file": str(path), "experiment": _experiment_label(path),
               "n": len(rows), "db_used": use_db, "compare_mode": compare_mode,
               "gold_exec_failures": gold_exec_fail, "row_errors": row_errors,
               "overall": {}, "by_level": {},
               "failures_finetuned": dict(ft_failures.most_common())}

    for mname, _ in PREDICTORS:
        summary["overall"][mname] = {
            "execution_accuracy": mean(agg[mname]["exec"]),
            "executable_rate": mean(agg[mname]["executable"]),
            "ast_equivalence": mean(agg[mname]["ast"]),
            "structural_f1": mean(agg[mname]["sf1"]),
            "component_jaccard": mean(agg[mname]["comp"]),
            "string_exact": mean(agg[mname]["str_exact"]),
            "string_similarity": mean(agg[mname]["str_sim"]),
        }

    levels = sorted({(r.get("nivel", r.get("level", "")) or "") for r in rows})
    for lvl in levels:
        summary["by_level"][lvl] = {}
        for mname, _ in PREDICTORS:
            summary["by_level"][lvl][mname] = {
                "execution_accuracy": mean(by_level[mname][lvl]["exec"]),
                "executable_rate": mean(by_level[mname][lvl]["executable"]),
                "ast_equivalence": mean(by_level[mname][lvl]["ast"]),
            }

    # delta = the study number (fine-tuned − base) on the headline metric
    def delta(metric, scope_base, scope_ft):
        b, f = scope_base.get(metric), scope_ft.get(metric)
        return round(f - b, 4) if (b is not None and f is not None) else None

    metric_key = "execution_accuracy" if use_db else "ast_equivalence"
    summary["dataset_value_delta"] = {
        "metric": metric_key,
        "overall": delta(metric_key, summary["overall"]["base"], summary["overall"]["finetuned"]),
        "by_level": {lvl: delta(metric_key, summary["by_level"][lvl]["base"],
                                summary["by_level"][lvl]["finetuned"]) for lvl in levels},
    }
    return {"summary": summary, "rows": detailed}


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════
def print_report(rep: dict):
    s = rep["summary"]
    mk = s["dataset_value_delta"]["metric"]
    logger.banner(f"{Path(s['file']).name}   (n={s['n']}, db={s['db_used']}, headline={mk})", width=70)
    if s["gold_exec_failures"]:
        logger.warning(f"{s['gold_exec_failures']} gold(s) did NOT execute — check schema/DSN.")

    def fmt(v):
        return "  n/a" if v is None else f"{v:6.3f}"

    logger.info(f"  {'metric':<22}{'base':>10}{'fine-tuned':>14}{'Δ':>10}")
    for m in ("execution_accuracy", "executable_rate", "ast_equivalence",
              "structural_f1", "component_jaccard", "string_exact", "string_similarity"):
        b = s["overall"]["base"][m]; f = s["overall"]["finetuned"][m]
        d = (round(f - b, 4) if (b is not None and f is not None) else None)
        logger.info(f"  {m:<22}{fmt(b):>10}{fmt(f):>14}{fmt(d):>10}")

    logger.info(f"  {mk} per level (base → fine-tuned, Δ):")
    for lvl, dv in s["dataset_value_delta"]["by_level"].items():
        b = s["by_level"][lvl]["base"][mk]; f = s["by_level"][lvl]["finetuned"][mk]
        logger.info(f"    {lvl or 'NA':<16}{fmt(b)} → {fmt(f)}   Δ={fmt(dv)}")

    logger.info(f"DATASET VALUE (Δ {mk} global): {fmt(s['dataset_value_delta']['overall'])}")

    if s["failures_finetuned"]:
        logger.info("Most common failures (fine-tuned):")
        for k, v in list(s["failures_finetuned"].items())[:8]:
            logger.info(f"    {k:<24} {v}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Validate generated SQL vs gold (all experiments).")
    ap.add_argument("--predictions", nargs="*", default=None,
                    help="Specific predictions_v*.json files. If omitted, ALL files in "
                         "--preds-dir are scored.")
    ap.add_argument("--preds-dir", default=None,
                    help=f"Folder scanned for predictions_v*.json when --predictions is "
                         f"omitted (default: {PREDS_DIR}).")
    ap.add_argument("--dsn", default=None,
                    help="Postgres DSN. Without it, uses libpq environment variables.")
    ap.add_argument("--compare-mode", default="auto",
                    choices=["set", "multiset", "list", "auto"],
                    help="Row comparison mode (default: auto).")
    ap.add_argument("--statement-timeout-ms", type=int, default=15000)
    ap.add_argument("--no-db", action="store_true",
                    help="Skip execution; report only AST/string metrics.")
    ap.add_argument("--save-details", action="store_true",
                    help="Also write per-row detail (slim, no result sets) to "
                         "details_v{N}.json — off by default to keep output small.")
    ap.add_argument("--out-dir", default=None,
                    help=f"Folder for the consolidated report "
                         f"(default: {Path(PROJECT_PATH) / 'results'}).")
    args = ap.parse_args()

    # ── Resolve which prediction files to score ──────────────────────────────
    if args.predictions:
        seen, paths = set(), []
        for p in args.predictions:                       # explicit list: dedup, then order
            rp = Path(p).resolve()
            if rp not in seen:
                seen.add(rp); paths.append(Path(p))
        paths.sort(key=_version_key)
    else:
        preds_dir = Path(args.preds_dir) if args.preds_dir else PREDS_DIR
        if not preds_dir.exists():
            logger.error(f"Predictions dir not found: {preds_dir} "
                         f"(pass --preds-dir or --predictions).")
            sys.exit(1)
        paths = discover_prediction_files(preds_dir)
        if not paths:
            logger.error(f"No predictions_v*.json found in {preds_dir}.")
            sys.exit(1)
        logger.info(f"Discovered {len(paths)} prediction file(s) in {preds_dir}: "
                    + ", ".join(p.name for p in paths))

    use_db = not args.no_db
    conn = None
    if use_db:
        try:
            conn = open_connection(args.dsn, args.statement_timeout_ms)
            logger.info(f"Connected to Postgres (statement_timeout={args.statement_timeout_ms}ms).")
        except Exception as e:
            logger.error(f"Failed to connect: {e}. Run with --no-db for AST-only metrics.")
            sys.exit(1)

    reports = []          # cross-experiment summaries (the consolidated report)
    detail_paths = []
    out_dir = Path(args.out_dir) if args.out_dir else (Path(PROJECT_PATH) / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in paths:
            if not path.exists():
                logger.warning(f"File not found, skipping: {path}")
                continue
            logger.info(f"Evaluating {path} …")
            rep = score_file(path, conn, args.compare_mode, use_db)
            print_report(rep)
            reports.append(rep["summary"])

            # Per-row detail is large; write it (slim, no result sets) only on request,
            # one file per experiment — never inside the consolidated summary report.
            if args.save_details:
                label = rep["summary"]["experiment"]
                dpath = out_dir / f"details_{label}.json"
                dpath.write_text(json.dumps(rep["rows"], ensure_ascii=False, indent=2, default=str),
                                 encoding="utf-8")
                detail_paths.append(dpath)
                logger.info(f"  details saved -> {dpath.resolve()}")
    finally:
        if conn is not None:
            conn.rollback()  # ensure nothing was committed
            conn.close()

    # ── Consolidated report (SUMMARIES ONLY) -> results/experimento_reports.json ──
    out_path = out_dir / "experiments_reports.json"
    consolidated = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_used": use_db,
        "compare_mode": args.compare_mode,
        "experiments": {s["experiment"]: s for s in reports},   # summary per experiment
    }
    out_path.write_text(json.dumps(consolidated, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Consolidated report saved -> {out_path.resolve()} ({size_kb:.1f} KB)")
    if detail_paths:
        logger.info(f"Per-experiment details: {len(detail_paths)} file(s) in {out_dir}")

    # final cross-experiment comparison
    if len(reports) > 1:
        mk = reports[0]["dataset_value_delta"]["metric"]
        logger.banner(f"CROSS-EXPERIMENT COMPARISON ({mk})", width=70)
        logger.info(f"  {'experiment':<32}{'base':>10}{'fine-tuned':>14}{'Δ':>10}")
        for s in reports:
            b = s["overall"]["base"][mk]; f = s["overall"]["finetuned"][mk]
            d = s["dataset_value_delta"]["overall"]
            fb = "  n/a" if b is None else f"{b:6.3f}"
            ff = "  n/a" if f is None else f"{f:6.3f}"
            fd = "  n/a" if d is None else f"{d:6.3f}"
            logger.info(f"  {s.get('experiment', Path(s['file']).name):<32}{fb:>10}{ff:>14}{fd:>10}")


if __name__ == "__main__":
    main()