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
"""
benchmark_preview.py  —  SQL Validation Framework preview (no database).

Updated for the sqlglot version of sql_validation:
  • Cases that used to break the hand-rolled parser now parse (CTE + window).
  • Shows canonical equivalence despite reformatting / commuted AND.
  • Surfaces real parse_error (invalid SQL).
  • Covers EVERY FailureType.
  • Demonstrates the Execution Accuracy deterministic tie-break via an offline
    simulation (no connection and no real tables). Real execution against
    Postgres is done by score_predictions.py.
"""
import re
import json

try:
    from core.sql_validation import (
        validate_sql, FailureType, _order_key_indices, _compare_rows,
    )
except ImportError:  # allows running with sql_validation.py in the same folder
    from sql_validation import (
        validate_sql, FailureType, _order_key_indices, _compare_rows,
    )

from utils.utils import Logger

logger = Logger("benchmark").setup_logging()

# ── ANSI ─────────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; YELLOW = "\033[33m"; RED = "\033[31m"
CYAN = "\033[36m"; WHITE = "\033[97m"
W = 72
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _vlen(s: str) -> int:
    return len(_ANSI.sub("", s))


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _vlen(s))


# ── Failure classification by severity (covers EVERY FailureType) ─────────────
_SOFT = {"missing_distinct", "wrong_order_by", "wrong_limit",
         "wrong_join_condition", "wrong_condition_value", "execution_error"}
_HARD = {"wrong_aggregation", "wrong_columns", "wrong_table", "missing_join",
         "wrong_join_type", "missing_group_by", "missing_having",
         "missing_subquery", "wrong_set_op", "wrong_condition_op", "parse_error"}


def _failure_icon(f: str) -> str:
    if f == "correct":
        return f"{GREEN}✔ correct{RESET}"
    if f in _SOFT:
        return f"{YELLOW}⚠ {f}{RESET}"
    if f in _HARD:
        return f"{RED}✘ {f}{RESET}"
    return f"{DIM}? {f}{RESET}"


# ── Formatters ───────────────────────────────────────────────────────────────
def header(text: str, color: str):
    logger.info(f"\n{color}{BOLD}┌{'─' * (W - 2)}┐{RESET}")
    logger.info(f"{color}{BOLD}│  {_pad(text, W - 6)}  │{RESET}")
    logger.info(f"{color}{BOLD}└{'─' * (W - 2)}┘{RESET}")


def section(title: str):
    logger.info(f"\n  {CYAN}{BOLD}{title}{RESET}")
    logger.info(f"  {DIM}{'─' * (W - 4)}{RESET}")


def badge(label: str, value, ok: bool | None = None):
    color = CYAN if ok is None else (GREEN if ok else RED)
    logger.info(f"  {WHITE}{label:<28}{RESET}  {color}{BOLD}{value}{RESET}")


def failures_line(failures: list[str]):
    rendered = "   ".join(_failure_icon(f) for f in failures) or f"{DIM}—{RESET}"
    logger.info(f"  {WHITE}{'Failures':<28}{RESET}  {rendered}")


def components_table(components: dict):
    section("Component Matching (per clause)")
    logger.info(f"  {DIM}{_pad('Clause', 14)} {_pad('Match', 8)} {'Jaccard':>8}{RESET}")
    logger.info(f"  {DIM}{'─'*14} {'─'*8} {'─'*8}{RESET}")
    for clause, data in components.items():
        icon = f"{GREEN}✔{RESET}" if data["match"] else f"{RED}✘{RESET}"
        jc = data["jaccard"]
        jcolor = GREEN if jc >= 0.8 else (YELLOW if jc >= 0.4 else RED)
        logger.info(f"  {WHITE}{_pad(clause, 14)}{RESET} {_pad(icon, 8)} {jcolor}{jc:>8.4f}{RESET}")


# ── Cases ──────────────────────────────────────────────────────────────────────
CASES = [
    {
        "label": "CASE 1 — PRED IDENTICAL TO GOLD",
        "color": GREEN,
        "description": "Points within a 5 km radius using ST_DWithin and a ::geography cast.",
        "gold": """
            SELECT p.id, p.nome,
                   ST_Distance(p.geom, ST_MakePoint(-43.1729, -22.9068)::geography) AS distancia
            FROM pontos_interesse p
            WHERE ST_DWithin(p.geom::geography,
                             ST_MakePoint(-43.1729, -22.9068)::geography, 5000)
            ORDER BY distancia ASC LIMIT 10;
        """,
        "pred": """
            SELECT p.id, p.nome,
                   ST_Distance(p.geom, ST_MakePoint(-43.1729, -22.9068)::geography) AS distancia
            FROM pontos_interesse p
            WHERE ST_DWithin(p.geom::geography,
                             ST_MakePoint(-43.1729, -22.9068)::geography, 5000)
            ORDER BY distancia ASC LIMIT 10;
        """,
    },
    {
        "label": "CASE 2 — EQUIVALENT DESPITE REFORMATTING",
        "color": GREEN,
        "description": "Same semantics: commuted AND, different case and spacing. "
                       "The sqlglot canonical form should recognize them as equivalent.",
        "gold": """
            SELECT mu.nm_mun, COUNT(*) AS total
            FROM escola e
            JOIN municipio mu ON ST_Contains(mu.geometry, e.geometry)
            WHERE mu.cd_uf = '13' AND e.tp_depende = 2
            GROUP BY mu.nm_mun
            ORDER BY total DESC;
        """,
        "pred": """
            select mu.nm_mun, count(*) AS total
            from escola e
            join municipio mu on st_contains(mu.geometry, e.geometry)
            where e.tp_depende = 2 and mu.cd_uf = '13'
            group by mu.nm_mun
            order by total desc
        """,
    },
    {
        "label": "CASE 3 — CTE + WINDOW (used to break the parser)",
        "color": CYAN,
        "description": "Very Hard: CTE + RANK() OVER. The old hand-rolled parser "
                       "ignored the OVER body; sqlglot parses it fully.",
        "gold": """
            WITH contagem AS (
                SELECT mu.nm_mun, COUNT(*) AS qtd
                FROM escola e
                JOIN municipio mu ON ST_Contains(mu.geometry, e.geometry)
                GROUP BY mu.nm_mun
            )
            SELECT nm_mun, qtd, RANK() OVER (ORDER BY qtd DESC) AS posicao
            FROM contagem ORDER BY posicao;
        """,
        "pred": """
            with contagem as (
                select mu.nm_mun, count(*) as qtd
                from escola e
                join municipio mu on st_contains(mu.geometry, e.geometry)
                group by mu.nm_mun
            )
            select nm_mun, qtd, rank() over (order by qtd desc) as posicao
            from contagem order by posicao
        """,
    },
    {
        "label": "CASE 4 — PRED SLIGHTLY DIFFERENT",
        "color": YELLOW,
        "description": "Polygon intersection: swaps ST_Intersects for ST_Within "
                       "and drops the DISTINCT.",
        "gold": """
            SELECT DISTINCT b.id, b.nome,
                   SUM(ST_Area(ST_Intersection(b.geom, z.geom))) AS area_intersecao
            FROM bairros b
            JOIN zonas_risco z ON ST_Intersects(b.geom, z.geom)
            WHERE z.nivel = 'alto'
            GROUP BY b.id, b.nome
            HAVING SUM(ST_Area(ST_Intersection(b.geom, z.geom))) > 1000
            ORDER BY area_intersecao DESC;
        """,
        "pred": """
            SELECT b.id, b.nome,
                   SUM(ST_Area(ST_Intersection(b.geom, z.geom))) AS area_intersecao
            FROM bairros b
            JOIN zonas_risco z ON ST_Within(b.geom, z.geom)
            WHERE z.nivel = 'alto'
            GROUP BY b.id, b.nome
            HAVING SUM(ST_Area(ST_Intersection(b.geom, z.geom))) > 1000
            ORDER BY area_intersecao DESC;
        """,
    },
    {
        "label": "CASE 5 — PRED COMPLETELY DIFFERENT",
        "color": RED,
        "description": "Per-municipality count with a CTE: pred ignores the CTE, JOIN, HAVING and ORDER BY.",
        "gold": """
            WITH municipios_afetados AS (
                SELECT DISTINCT m.id, m.nome, m.geom
                FROM municipios m
                JOIN ocorrencias o ON ST_Contains(m.geom, o.geom)
                WHERE o.data >= '2024-01-01'
            )
            SELECT ma.nome, COUNT(o.id) AS total_ocorrencias
            FROM municipios_afetados ma
            JOIN ocorrencias o ON ST_Contains(ma.geom, o.geom)
            GROUP BY ma.nome
            HAVING COUNT(o.id) > 10
            ORDER BY total_ocorrencias DESC LIMIT 5;
        """,
        "pred": "SELECT nome FROM municipios LIMIT 5;",
    },
    {
        "label": "CASE 6 — INVALID SQL (real parse_error)",
        "color": RED,
        "description": "With sqlglot, parse_error means genuinely invalid SQL "
                       "— not a parser limitation.",
        "gold": "SELECT nm_mun FROM municipio WHERE cd_uf = '35';",
        "pred": "SELECT FROM WHERE GROUP municipio;",
    },
]


def print_case(case: dict):
    result = json.loads(validate_sql(predicted=case["pred"], gold=case["gold"]))
    s = result["summary"]

    header(case["label"], case["color"])
    logger.info(f"\n  {DIM}{case['description']}{RESET}")

    section("Summary")
    badge("AST Equivalent", s["ast_equivalent"], ok=s["ast_equivalent"])
    perr = result["ast"].get("parse_error")
    if perr:
        badge("Parse Error", perr, ok=False)
    badge("String Exact Match", s["string_exact"], ok=s["string_exact"])
    badge("String Similarity", f"{s['string_similarity']:.4f}", ok=s["string_similarity"] >= 0.9)
    if perr:
        badge("Component Avg Jaccard", "n/a (parse error)", ok=False)
    else:
        badge("Component Avg Jaccard", f"{s['component_avg_jaccard']:.4f}",
              ok=s["component_avg_jaccard"] >= 0.9)

    sf = result["structural_f1"]
    if sf.get("error"):
        badge("Structural F1", "n/a (parse error)", ok=False)
    else:
        badge("Structural F1",
              f"{sf['f1']:.4f}  (P={sf['precision']:.4f}  R={sf['recall']:.4f})",
              ok=sf["f1"] >= 0.9)

    logger.info("")
    failures_line(s["failures"])

    if not result["ast"].get("parse_error"):
        components_table(result["components"])

    section("String Matching — Token F1")
    tf = result["string"]["token_f1"]
    badge("Precision", f"{tf['precision']:.4f}", ok=tf["precision"] >= 0.9)
    badge("Recall", f"{tf['recall']:.4f}", ok=tf["recall"] >= 0.9)
    badge("F1", f"{tf['f1']:.4f}", ok=tf["f1"] >= 0.9)


# ── Offline demo of the Execution Accuracy deterministic tie-break ─────────────
def execution_tiebreak_demo():
    header("EXECUTION ACCURACY — deterministic tie-break (offline simulation)", CYAN)
    gold = ("SELECT mu.nm_mun, COUNT(*) AS qtd FROM escola e "
            "JOIN municipio mu ON ST_Contains(mu.geometry, e.geometry) "
            "GROUP BY mu.nm_mun ORDER BY qtd DESC;")
    ki = _order_key_indices(gold)
    logger.info(f"\n  {DIM}gold has ORDER BY qtd → ordering key mapped to column {ki}{RESET}")

    gold_rows = [("Manaus", 10), ("Tefé", 5), ("Coari", 5)]
    scenarios = [
        ("identical rows",              [("Manaus", 10), ("Tefé", 5), ("Coari", 5)]),
        ("tie reordered (Tefé<->Coari)", [("Manaus", 10), ("Coari", 5), ("Tefé", 5)]),
        ("PRIMARY order swapped",       [("Tefé", 5), ("Manaus", 10), ("Coari", 5)]),
        ("different row set",           [("Manaus", 10), ("Tefé", 5)]),
    ]

    section("scenario x comparison mode")
    logger.info(f"  {DIM}{_pad('scenario', 34)}{_pad('multiset', 11)}{_pad('list', 9)}"
          f"{'ordered_robust':>16}{RESET}")
    logger.info(f"  {DIM}{'─'*34}{'─'*11}{'─'*9}{'─'*16}{RESET}")

    def cell(b):
        return f"{GREEN}✔{RESET}" if b else f"{RED}✘{RESET}"

    for name, pred_rows in scenarios:
        ms = _compare_rows(pred_rows, gold_rows, "multiset", ki)
        ls = _compare_rows(pred_rows, gold_rows, "list", ki)
        orb = _compare_rows(pred_rows, gold_rows, "ordered_robust", ki)
        logger.info(f"  {WHITE}{_pad(name, 34)}{RESET}{_pad(cell(ms), 11)}"
              f"{_pad(cell(ls), 9)}{_pad(cell(orb), 16)}")

    logger.info(f"\n  {DIM}Reading: 'ordered_robust' validates the ORDER BY order but tolerates "
          f"reordering{RESET}")
    logger.info(f"  {DIM}among ties — without the multiset false positive nor the "
          f"list false negative.{RESET}")


def run_benchmark():
    logger.info(f"\n{BOLD}{WHITE}{'=' * W}")
    logger.info("  SQL VALIDATION FRAMEWORK  —  benchmark preview (sqlglot)")
    logger.info(f"{'=' * W}{RESET}")

    for case in CASES:
        print_case(case)

    execution_tiebreak_demo()

    logger.info(f"\n{DIM}{'─' * W}")
    logger.info(f"  {len(CASES)} cases  |  mode: AST + string + components (no database)")
    logger.info(f"  Real Execution Accuracy (vs Postgres) → use score_predictions.py")
    logger.info(f"{'─' * W}{RESET}\n")


if __name__ == "__main__":
    run_benchmark()