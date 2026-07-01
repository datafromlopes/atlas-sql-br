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
#  check_sql_sintax.py — check that every gold SQL is VALID on Postgres using
#  EXPLAIN (plan only, no execution) and record the invalid ones into a CSV for
#  manual review.
#
#  Why EXPLAIN (without ANALYZE): the planner parses and binds the query —
#  it resolves table/column names, functions and types — WITHOUT running it or
#  materializing any rows, so validation is near-instant.
#  Trade-off: EXPLAIN catches structural errors (syntax, missing table/column,
#  unknown function, type mismatch) but NOT execution-only errors (division by
#  zero, bad runtime casts) and does NOT measure query speed (no 'timeout'
#  signal). Use --execute if you also want to run each query and time it.
#
#  Read-only: the session is forced to read-only and a statement_timeout caps
#  each query, so nothing can change data or hang the run.
#
#  Usage:
#     uv run python check_sql_sintax.py \
#         --dataset atlas_sql_br.csv \
#         --dsn "postgresql://user:pass@localhost:5432/schools" \
#         --out failed_sql.csv
#
#  Reads .csv or .parquet (by extension). Without --dsn, libpq environment
#  variables are used (PGHOST, PGDATABASE, PGUSER, PGPASSWORD). If the password
#  has special chars (e.g. '@'), prefer the env vars or percent-encode it
#  ('@' -> '%40').
# =============================================================================
from __future__ import annotations

import csv
import time
import argparse
from pathlib import Path

import polars as pl


def first_existing(df: pl.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_dataframe(path: str) -> pl.DataFrame:
    p = path.lower()
    if p.endswith((".csv", ".tsv")):
        return pl.read_csv(path, separator="\t" if p.endswith(".tsv") else ",")
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate each gold SQL on Postgres via EXPLAIN.")
    ap.add_argument("--dataset", required=True, help="Path to the dataset (.csv or .parquet).")
    ap.add_argument("--dsn", default=None, help="Postgres DSN. Without it, uses libpq env vars.")
    ap.add_argument("--sql-col", default="sql_code", help="Column with the gold SQL.")
    ap.add_argument("--id-col", default="id", help="Identifier column (recorded in the CSV).")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all",
                    help="If a 'train' column exists, validate only train(==1)/test(==0)/all.")
    ap.add_argument("--statement-timeout-ms", type=int, default=30000,
                    help="Per-query timeout (mostly relevant with --execute).")
    ap.add_argument("--execute", action="store_true",
                    help="Also EXECUTE each query (slower) instead of EXPLAIN-only.")
    ap.add_argument("--out", default="failed_sql.csv", help="CSV with the failing rows.")
    ap.add_argument("--limit", type=int, default=None, help="Validate only the first N rows (debug).")
    args = ap.parse_args()

    import psycopg2
    from psycopg2 import errors as pg_errors

    # ── load dataset ─────────────────────────────────────────────────────────
    df = load_dataframe(args.dataset)
    if args.sql_col not in df.columns:
        raise SystemExit(f"Column '{args.sql_col}' not found. Available: {df.columns}")

    # optional split filter (dataset uses train: 1=train, 0=test)
    if args.split != "all" and "train" in df.columns:
        want = 1 if args.split == "train" else 0
        df = df.filter(pl.col("train").cast(pl.Int64) == want)

    id_col = args.id_col if args.id_col in df.columns else None
    level_col = first_existing(df, "level", "nivel")
    if args.limit:
        df = df.head(args.limit)
    total = len(df)
    mode = "EXECUTE" if args.execute else "EXPLAIN (plan only)"
    print(f"[info] dataset={args.dataset} split={args.split} rows={total} mode={mode} "
          f"sql_col={args.sql_col} id_col={id_col} level_col={level_col}")

    # ── connect (read-only + statement timeout) ──────────────────────────────
    opts = f"-c statement_timeout={args.statement_timeout_ms} -c default_transaction_read_only=on"
    conn = psycopg2.connect(args.dsn or "", options=opts)
    conn.autocommit = True  # each statement is its own tx; an error aborts only it
    print(f"[info] connected (statement_timeout={args.statement_timeout_ms}ms, read-only)")

    sqls = df[args.sql_col].to_list()
    ids = df[id_col].to_list() if id_col else list(range(total))
    levels = df[level_col].to_list() if level_col else [""] * total

    failures = []  # rows that did not validate
    n_ok = n_timeout = n_error = 0
    last_tick = time.monotonic()

    with conn.cursor() as cur:
        for i, (sql, rid, lvl) in enumerate(zip(sqls, ids, levels)):
            sql = (sql or "").strip()
            if not sql:
                n_error += 1
                failures.append((i, rid, lvl, "empty", 0.0, "empty sql"))
                continue

            # EXPLAIN: plan only (parse/bind, no execution). Strip trailing ';'
            # so it sits cleanly inside "EXPLAIN <stmt>".
            stmt = sql if args.execute else f"EXPLAIN {sql.rstrip(';')}"

            t0 = time.monotonic()
            try:
                cur.execute(stmt)
                if args.execute and cur.description is None:
                    pass                       # non-row statement; nothing to fetch
                else:
                    cur.fetchall()             # consume rows (query result or EXPLAIN plan)
                n_ok += 1
            except pg_errors.QueryCanceled as e:  # statement_timeout
                n_timeout += 1
                failures.append((i, rid, lvl, "timeout", round(time.monotonic() - t0, 2),
                                 str(e).strip().splitlines()[0]))
            except psycopg2.Error as e:           # syntax / missing object / type / runtime
                n_error += 1
                failures.append((i, rid, lvl, "error", round(time.monotonic() - t0, 2),
                                 str(e).strip().splitlines()[0]))

            now = time.monotonic()
            if now - last_tick >= 2.0 or i + 1 == total:
                done = i + 1
                print(f"[{done:>5}/{total}] ok={n_ok} timeout={n_timeout} error={n_error}",
                      flush=True)
                last_tick = now

    conn.close()

    # ── write failures CSV ───────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "id", "level", "status", "elapsed_s", "error", "sql"])
        for i, rid, lvl, status, elapsed, err in failures:
            w.writerow([i, rid, lvl, status, elapsed, err, sqls[i]])

    print(f"\n[done] {total} validated | ok={n_ok} | timeout={n_timeout} | error={n_error}")
    print(f"[done] {len(failures)} failing rows -> {out_path.resolve()}")


if __name__ == "__main__":
    main()