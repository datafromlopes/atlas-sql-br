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
benchmark_preview.py  —  Prévia/benchmark do SQL Validation Framework
com queries geoespaciais de exemplo (sem conexão com banco de dados).
"""

import json
from core.sql_validation import validate_sql

# ── ANSI colors ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"
BG_DARK = "\033[48;5;235m"

# ── Cases ────────────────────────────────────────────────────────────────────

CASES = [
    {
        "label": "CASE 1 — PRED IDENTICAL TO GOLD",
        "color": GREEN,
        "description": "Pontos dentro de um raio de 5 km com ST_DWithin.",
        "gold": """
            SELECT p.id, p.nome,
                   ST_Distance(p.geom, ST_MakePoint(-43.1729, -22.9068)::geography) AS distancia
            FROM pontos_interesse p
            WHERE ST_DWithin(
                p.geom::geography,
                ST_MakePoint(-43.1729, -22.9068)::geography,
                5000
            )
            ORDER BY distancia ASC
            LIMIT 10;
        """,
        "pred": """
            SELECT p.id, p.nome,
                   ST_Distance(p.geom, ST_MakePoint(-43.1729, -22.9068)::geography) AS distancia
            FROM pontos_interesse p
            WHERE ST_DWithin(
                p.geom::geography,
                ST_MakePoint(-43.1729, -22.9068)::geography,
                5000
            )
            ORDER BY distancia ASC
            LIMIT 10;
        """,
    },
    {
        "label": "CASE 2 — PRED SLIGHTLY DIFFERENT",
        "color": YELLOW,
        "description": "Interseção de polígonos: pred troca ST_Intersects por ST_Within e perde o DISTINCT.",
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
        "label": "CASE 3 — PRED COMPLETELY DIFFERENT",
        "color": RED,
        "description": "Contagem por município com CTE geoespacial: pred ignora CTE, JOIN, HAVING e ORDER BY.",
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
            ORDER BY total_ocorrencias DESC
            LIMIT 5;
        """,
        "pred": """
            SELECT nome
            FROM municipios
            LIMIT 5;
        """,
    },
]

# ── Formatters ───────────────────────────────────────────────────────────────

W = 72

def header(text: str, color: str):
    pad = W - len(text) - 4
    print(f"\n{color}{BOLD}┌{'─' * (W - 2)}┐{RESET}")
    print(f"{color}{BOLD}│  {text}{' ' * pad}  │{RESET}")
    print(f"{color}{BOLD}└{'─' * (W - 2)}┘{RESET}")

def section(title: str):
    print(f"\n  {CYAN}{BOLD}{title}{RESET}")
    print(f"  {DIM}{'─' * (W - 4)}{RESET}")

def badge(label: str, value, ok: bool | None = None):
    if ok is True:   color = GREEN
    elif ok is False: color = RED
    else:             color = CYAN
    val_str = str(value)
    print(f"  {WHITE}{label:<28}{RESET}  {color}{BOLD}{val_str}{RESET}")

def failures_line(failures: list[str], case_color: str):
    icons = {
        "correct":              f"{GREEN}✔  correct{RESET}",
        "wrong_join_condition": f"{YELLOW}⚠  wrong_join_condition{RESET}",
        "missing_distinct":     f"{YELLOW}⚠  missing_distinct{RESET}",
        "wrong_columns":        f"{RED}✘  wrong_columns{RESET}",
        "wrong_table":          f"{RED}✘  wrong_table{RESET}",
        "missing_group_by":     f"{RED}✘  missing_group_by{RESET}",
        "missing_having":       f"{RED}✘  missing_having{RESET}",
        "wrong_order_by":       f"{RED}✘  wrong_order_by{RESET}",
        "wrong_limit":          f"{RED}✘  wrong_limit{RESET}",
        "missing_subquery":     f"{RED}✘  missing_subquery{RESET}",
        "wrong_set_op":         f"{RED}✘  wrong_set_op{RESET}",
        "parse_error":          f"{RED}✘  parse_error{RESET}",
        "unknown":              f"{DIM}?  unknown{RESET}",
    }
    rendered = "  |  ".join(icons.get(f, f"{DIM}{f}{RESET}") for f in failures)
    print(f"  {WHITE}{'Failures':<28}{RESET}  {rendered}")

def components_table(components: dict):
    section("Component Matching (por cláusula)")
    print(f"  {DIM}{'Clause':<14} {'Match':<8} {'Jaccard':>8}{RESET}")
    print(f"  {DIM}{'─'*14} {'─'*8} {'─'*8}{RESET}")
    for clause, data in components.items():
        match_icon = f"{GREEN}✔{RESET}" if data["match"] else f"{RED}✘{RESET}"
        jaccard_color = GREEN if data["jaccard"] >= 0.8 else (YELLOW if data["jaccard"] >= 0.4 else RED)
        print(f"  {WHITE}{clause:<14}{RESET} {match_icon:<16} {jaccard_color}{data['jaccard']:>8.4f}{RESET}")

def print_case(i: int, case: dict):
    result = json.loads(validate_sql(predicted=case["pred"], gold=case["gold"]))
    s      = result["summary"]
    color  = case["color"]

    header(case["label"], color)
    print(f"\n  {DIM}{case['description']}{RESET}")

    # ── Summary ──────────────────────────────────────────────────────────
    section("Resumo")
    badge("AST Equivalent",        s["ast_equivalent"],        ok=s["ast_equivalent"])
    badge("String Exact Match",    s["string_exact"],          ok=s["string_exact"])
    badge("String Similarity",     f"{s['string_similarity']:.4f}",
          ok=s["string_similarity"] >= 0.9)
    badge("Component Avg Jaccard", f"{s['component_avg_jaccard']:.4f}",
          ok=s["component_avg_jaccard"] >= 0.9)

    sf = result["structural_f1"]
    badge("Structural F1",         f"{sf['f1']:.4f}  "
          f"(P={sf['precision']:.4f}  R={sf['recall']:.4f})",
          ok=sf["f1"] >= 0.9)

    # ── Failures ─────────────────────────────────────────────────────────
    print()
    failures_line(s["failures"], color)

    # ── Component table ───────────────────────────────────────────────────
    components_table(result["components"])

    # ── Token F1 ─────────────────────────────────────────────────────────
    section("String Matching — Token F1")
    tf = result["string"]["token_f1"]
    badge("Precision", f"{tf['precision']:.4f}", ok=tf["precision"] >= 0.9)
    badge("Recall",    f"{tf['recall']:.4f}",    ok=tf["recall"]    >= 0.9)
    badge("F1",        f"{tf['f1']:.4f}",        ok=tf["f1"]        >= 0.9)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark():
    print(f"\n{BOLD}{WHITE}{'=' * W}")
    print(f"  SQL VALIDATION FRAMEWORK  —  benchmark preview")
    print(f"{'=' * W}{RESET}")

    for i, case in enumerate(CASES, 1):
        print_case(i, case)

    print(f"\n{DIM}{'─' * W}")
    print(f"  {len(CASES)} casos avaliados  |  modo: AST + string (sem conexão com banco)")
    print(f"{'─' * W}{RESET}\n")


if __name__ == "__main__":
    run_benchmark()