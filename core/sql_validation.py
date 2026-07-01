# coding=utf-8
# Copyright (C) 2026  Diego Lopes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""
sql_validation.py  —  Research-grade SQL Validation Framework (sqlglot edition)
================================================================================
Mudou em relação à versão anterior:
  • O parser artesanal (lexer + recursive-descent + AST próprios) foi SUBSTITUÍDO
    pelo parser do sqlglot (dialeto postgres). Isso elimina a classe inteira de
    `parse_error` espúrios em PostGIS (ST_*, ::cast), window functions, CTE
    recursiva, UNION etc. Se o sqlglot não parseia, o SQL é genuinamente inválido.
  • Execution Accuracy ganhou o modo "auto"/"ordered_robust": quando o gold tem
    ORDER BY, a comparação valida a ordem MAS tolera reordenação dentro de
    empates (compara o conjunto de linhas + a sequência das chaves de ordenação).

Limite que NÃO tem conserto (é teórico): equivalência de SQL é indecidível em
geral. A canônica de AST sempre terá falsos-negativos (ex.: IN-subquery vs JOIN).
Por isso ela é DIAGNÓSTICO; o veredito é a Execution Accuracy.

Interface pública (inalterada):
  parse_sql, ast_canonical_equivalence, structural_f1, execution_accuracy,
  component_matching, component_matching_score, string_matching,
  classify_failure, validate_sql, FailureType, ComponentMatchResult
"""
from __future__ import annotations

import re
import sys
import json
import difflib
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

try:
    from sqlglot.optimizer.simplify import simplify as _sg_simplify
except Exception:  # pragma: no cover
    _sg_simplify = None
try:
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers as _sg_norm_ids
except Exception:  # pragma: no cover
    _sg_norm_ids = None

try:
    import psycopg2  # noqa: F401
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

PgConnection = Any
PgDSN = str
CompareMode = str  # "set" | "multiset" | "list" | "ordered_robust" | "auto"
DIALECT = "postgres"

# A model in a greedy-decoding loop can emit a hugely nested string that blows up
# the recursive sqlglot parser. Two guards: reject obviously-degenerate output by
# length before parsing, and give the parser modest extra stack headroom for
# legitimately deep SQL (CTEs, windows) — capped to stay well clear of a real
# interpreter stack overflow.
_MAX_SQL_CHARS = 20000
_PARSE_RECURSION_LIMIT = 4000


class ParseError(Exception):
    """Mantido por compatibilidade — encapsula falhas do sqlglot."""


# ═══════════════════════════════════════════════════════════════════════════
# 1 — PARSING (sqlglot)
# ═══════════════════════════════════════════════════════════════════════════
def parse_sql(sql: str) -> exp.Expression:
    """Parseia SQL no dialeto postgres. Levanta ParseError em SQL inválido.

    Resiliente a predições degeneradas: strings absurdamente longas são rejeitadas
    sem parsear, e um estouro de recursão do parser vira ParseError (em vez de
    derrubar a avaliação inteira)."""
    if not sql or not sql.strip():
        raise ParseError("empty statement")
    if len(sql) > _MAX_SQL_CHARS:
        raise ParseError(f"sql too long ({len(sql)} chars) — likely a degenerate generation")

    prev_limit = sys.getrecursionlimit()
    try:
        if prev_limit < _PARSE_RECURSION_LIMIT:
            sys.setrecursionlimit(_PARSE_RECURSION_LIMIT)
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except RecursionError as e:
        raise ParseError("recursion limit exceeded while parsing (degenerate SQL)") from e
    except SqlglotError as e:
        raise ParseError(str(e)) from e
    finally:
        sys.setrecursionlimit(prev_limit)
    if tree is None:
        raise ParseError("empty statement")
    return tree


def _primary_select(tree: exp.Expression) -> Optional[exp.Select]:
    """Retorna o Select de topo (ou o primeiro, em UNION/CTE)."""
    if isinstance(tree, exp.Select):
        return tree
    return tree.find(exp.Select)


# ═══════════════════════════════════════════════════════════════════════════
# 2 — CANONICALIZAÇÃO  (normaliza identificadores, booleanos e comutatividade)
# ═══════════════════════════════════════════════════════════════════════════
def _sort_commutative(tree: exp.Expression) -> exp.Expression:
    """Ordena listas comutativas: projeções (sem ORDER BY), GROUP BY e listas IN."""
    for sel in tree.find_all(exp.Select):
        if not sel.args.get("order") and sel.expressions:
            sel.set("expressions", sorted(sel.expressions, key=lambda e: e.sql(dialect=DIALECT)))
        grp = sel.args.get("group")
        if grp is not None and grp.expressions:
            grp.set("expressions", sorted(grp.expressions, key=lambda e: e.sql(dialect=DIALECT)))
    for in_ in tree.find_all(exp.In):
        vals = in_.args.get("expressions")
        if vals:
            in_.set("expressions", sorted(vals, key=lambda e: e.sql(dialect=DIALECT)))
    return tree


def _canonical_sql(sql: str) -> str:
    """Forma canônica textual de uma query (para equivalência de AST)."""
    tree = parse_sql(sql)
    if _sg_norm_ids is not None:
        try:
            tree = _sg_norm_ids(tree, dialect=DIALECT)   # minúsculas em ids não-aspeados
        except Exception:
            pass
    if _sg_simplify is not None:
        try:
            tree = _sg_simplify(tree)                     # normaliza AND/OR, constantes
        except Exception:
            pass
    tree = _sort_commutative(tree)
    return tree.sql(dialect=DIALECT, normalize=True)


def ast_canonical_equivalence(predicted: str, gold: str) -> dict[str, Any]:
    """Compara as formas canônicas (sqlglot) de duas queries."""
    result = {"equivalent": False, "pred_canonical": None,
              "gold_canonical": None, "parse_error": None}
    try:
        pc = _canonical_sql(predicted)
        gc = _canonical_sql(gold)
        result["pred_canonical"] = pc
        result["gold_canonical"] = gc
        result["equivalent"] = pc == gc
    except ParseError as e:
        result["parse_error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3 — STRUCTURAL F1  (precision/recall sobre nós da AST)
# ═══════════════════════════════════════════════════════════════════════════
def _node_signature(node: exp.Expression) -> str:
    t = type(node).__name__
    if isinstance(node, exp.Column):
        return f"Column:{node.name.lower()}"
    if isinstance(node, exp.Identifier):
        return f"Identifier:{str(node.this).lower()}"
    if isinstance(node, exp.Literal):
        return f"Literal:{node.this}"
    if isinstance(node, (exp.Func, exp.Anonymous)):
        name = (node.sql_name() if hasattr(node, "sql_name") else t)
        return f"Func:{str(name).lower()}"
    return f"Node:{t}"


def _collect_signatures(tree: exp.Expression) -> list[str]:
    return [_node_signature(n) for n in tree.find_all(exp.Expression)]


def structural_f1(predicted: str, gold: str) -> dict[str, Any]:
    try:
        pt = parse_sql(predicted)
        gt = parse_sql(gold)
    except ParseError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": True}

    pc = Counter(_collect_signatures(pt))
    gc = Counter(_collect_signatures(gt))
    common = sum(min(pc[k], gc[k]) for k in gc)
    tp, tg = sum(pc.values()), sum(gc.values())
    precision = common / tp if tp else 0.0
    recall = common / tg if tg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "pred_nodes": tp, "gold_nodes": tg, "common": common}


# ═══════════════════════════════════════════════════════════════════════════
# 4 — EXECUTION ACCURACY  (com desempate determinístico)
# ═══════════════════════════════════════════════════════════════════════════
def _pg_connect(dsn: PgDSN) -> PgConnection:
    if not _PG_AVAILABLE:
        raise RuntimeError("psycopg2 não encontrado. Instale: pip install psycopg2-binary")
    import psycopg2
    return psycopg2.connect(dsn)


def _run_query(conn: PgConnection, sql: str):
    """Executa em SAVEPOINT para isolar erros sem invalidar a transação externa."""
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT _sqlval_sp")
            try:
                cur.execute(sql)
                rows = [tuple(r) for r in cur.fetchall()]
                cur.execute("RELEASE SAVEPOINT _sqlval_sp")
                return rows, None
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT _sqlval_sp")
                return None, str(e)
    except Exception as e:
        return None, str(e)


def _order_key_indices(gold_sql: str) -> Optional[list[int]]:
    """Mapeia as colunas do ORDER BY do gold para posições no resultado.
    Retorna None quando não é mapeável (ex.: SELECT *, ORDER BY por expressão
    fora do SELECT, ou query com UNION) — nesse caso cai-se em multiset."""
    try:
        tree = parse_sql(gold_sql)
    except ParseError:
        return None
    if not isinstance(tree, exp.Select):
        return None
    order = tree.args.get("order")
    if order is None or not order.expressions:
        return None
    proj = [p.alias_or_name for p in tree.expressions]
    if any((not p) or p == "*" for p in proj):
        return None
    proj_lower = [p.lower() for p in proj]
    idxs: list[int] = []
    for ordered in order.expressions:           # exp.Ordered
        target = ordered.this
        name = (target.alias_or_name or target.sql(dialect=DIALECT)).lower()
        if name in proj_lower:
            idxs.append(proj_lower.index(name))
        else:
            return None
    return idxs


def _safe_counter_eq(a: list, b: list) -> bool:
    try:
        return Counter(a) == Counter(b)
    except TypeError:  # linhas com tipos não-hasheáveis
        return sorted(map(repr, a)) == sorted(map(repr, b))


def _compare_rows(pred, gold, mode: str, key_idx: Optional[list[int]]) -> bool:
    if pred is None or gold is None:
        return False
    if mode == "set":
        try:
            return frozenset(pred) == frozenset(gold)
        except TypeError:
            return _safe_counter_eq(pred, gold)
    if mode == "list":
        return pred == gold
    multiset_eq = _safe_counter_eq(pred, gold)
    if mode == "multiset":
        return multiset_eq
    if mode == "ordered_robust":
        if not multiset_eq:
            return False
        if not key_idx:
            return multiset_eq                  # sem chave mapeável → multiset
        try:
            pk = [tuple(r[i] for i in key_idx) for r in pred]
            gk = [tuple(r[i] for i in key_idx) for r in gold]
        except (IndexError, TypeError):
            return multiset_eq
        return pk == gk                          # mesma sequência de chaves (tolera empates)
    return multiset_eq


def execution_accuracy(
    predicted: str,
    gold: str,
    dsn: Optional[PgDSN] = None,
    conn: Optional[PgConnection] = None,
    compare_mode: CompareMode = "multiset",
) -> dict[str, Any]:
    """
    Execution Accuracy via PostgreSQL.

    compare_mode:
        'set'            → ignora duplicatas e ordem
        'multiset'       → respeita duplicatas, ignora ordem            (default)
        'list'           → respeita ordem e duplicatas (estrito)
        'ordered_robust' → respeita a ordem do ORDER BY, mas tolera
                           reordenação entre linhas empatadas
        'auto'           → 'ordered_robust' se o gold tem ORDER BY mapeável,
                           senão 'multiset'                              (recomendado)
    """
    managed = False
    if dsn:
        conn = _pg_connect(dsn)
        conn.autocommit = False
        managed = True
    elif conn is None:
        raise ValueError("Forneça 'dsn' ou 'conn' para a Execution Accuracy.")

    # resolve o modo efetivo
    key_idx = None
    effective = compare_mode
    if compare_mode in ("auto", "ordered_robust"):
        key_idx = _order_key_indices(gold)
        if compare_mode == "auto":
            effective = "ordered_robust" if key_idx else "multiset"

    result: dict[str, Any] = dict(
        match=False, compare_mode=compare_mode, effective_mode=effective,
        predicted_rows=None, gold_rows=None, predicted_error=None, gold_error=None,
    )
    try:
        pr, pe = _run_query(conn, predicted)
        gr, ge = _run_query(conn, gold)
        result.update(predicted_rows=pr, gold_rows=gr, predicted_error=pe, gold_error=ge)
        if pr is not None and gr is not None:
            result["match"] = _compare_rows(pr, gr, effective, key_idx)
    finally:
        if managed:
            conn.rollback()
            conn.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5 — COMPONENT MATCHING  (por cláusula, sobre a AST do sqlglot)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ComponentMatchResult:
    clause: str
    predicted: Any
    gold: Any
    match: bool
    jaccard: float


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _node_sql(n: exp.Expression) -> str:
    return n.sql(dialect=DIALECT)


def _clause_nodes(sel: Optional[exp.Select]) -> dict[str, list[exp.Expression]]:
    if sel is None:
        return {k: [] for k in ("select", "from", "joins", "where",
                                "group_by", "having", "order_by", "limit")}
    frm = sel.args.get("from")
    grp = sel.args.get("group")
    order = sel.args.get("order")
    where = sel.args.get("where")
    having = sel.args.get("having")
    limit = sel.args.get("limit")
    return {
        "select":   list(sel.expressions),
        "from":     list(frm.find_all(exp.Table)) if frm else [],
        "joins":    list(sel.args.get("joins") or []),
        "where":    [where.this] if where else [],
        "group_by": list(grp.expressions) if grp else [],
        "having":   [having.this] if having else [],
        "order_by": list(order.expressions) if order else [],
        "limit":    [limit] if limit else [],
    }


def component_matching(predicted: str, gold: str) -> dict[str, ComponentMatchResult]:
    try:
        pa = _primary_select(parse_sql(predicted))
        ga = _primary_select(parse_sql(gold))
    except ParseError:
        return {"parse_error": ComponentMatchResult("parse_error", None, None, False, 0.0)}

    pc = _clause_nodes(pa)
    gc = _clause_nodes(ga)
    results: dict[str, ComponentMatchResult] = {}

    ordered_clauses = {"joins", "order_by"}
    for clause in ("select", "from", "joins", "where", "group_by", "having", "order_by", "limit"):
        p_nodes, g_nodes = pc[clause], gc[clause]
        if clause in ordered_clauses:
            ps = [_node_sql(n) for n in p_nodes]
            gs = [_node_sql(n) for n in g_nodes]
            match = ps == gs
            pcnt, gcnt = Counter(ps), Counter(gs)
            common = sum(min(pcnt[k], gcnt[k]) for k in gcnt)
            union = sum((pcnt | gcnt).values())
            jac = common / union if union else 1.0
        else:
            ps_set = {_node_sql(n) for n in p_nodes}
            gs_set = {_node_sql(n) for n in g_nodes}
            match = ps_set == gs_set
            jac = _jaccard(ps_set, gs_set)
        results[clause] = ComponentMatchResult(clause, p_nodes, g_nodes, match, round(jac, 4))
    return results


def component_matching_score(predicted: str, gold: str) -> float:
    cm = component_matching(predicted, gold)
    relevant = [r for r in cm.values()
                if isinstance(r, ComponentMatchResult) and r.gold]
    return round(sum(r.jaccard for r in relevant) / len(relevant), 4) if relevant else 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 6 — STRING MATCHING
# ═══════════════════════════════════════════════════════════════════════════
def _norm(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return re.sub(r"\s+", " ", sql.strip().lower()).rstrip(";").strip()


def _tok(sql: str) -> list[str]:
    return re.findall(r"[\w\.\*]+|<>|!=|>=|<=|[(),=<>!;]", _norm(sql))


def string_matching(predicted: str, gold: str) -> dict[str, Any]:
    pn, gn = _norm(predicted), _norm(gold)
    pt, gt = _tok(predicted), _tok(gold)
    pc, gc = Counter(pt), Counter(gt)
    common = sum(min(pc[t], gc[t]) for t in gc)
    tp, tg = sum(pc.values()), sum(gc.values())
    prec = common / tp if tp else 0.0
    rec = common / tg if tg else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"exact": pn == gn,
            "similarity": round(difflib.SequenceMatcher(None, pn, gn).ratio(), 4),
            "token_f1": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}}


# ═══════════════════════════════════════════════════════════════════════════
# 7 — FAILURE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════
class FailureType(Enum):
    CORRECT = "correct"
    WRONG_AGGREGATION = "wrong_aggregation"
    WRONG_CONDITION_VALUE = "wrong_condition_value"
    WRONG_CONDITION_OP = "wrong_condition_op"
    MISSING_JOIN = "missing_join"
    WRONG_JOIN_TYPE = "wrong_join_type"
    WRONG_JOIN_CONDITION = "wrong_join_condition"
    MISSING_GROUPBY = "missing_group_by"
    MISSING_HAVING = "missing_having"
    WRONG_COLUMNS = "wrong_columns"
    WRONG_TABLE = "wrong_table"
    MISSING_DISTINCT = "missing_distinct"
    WRONG_ORDERBY = "wrong_order_by"
    WRONG_LIMIT = "wrong_limit"
    MISSING_SUBQUERY = "missing_subquery"
    WRONG_SET_OP = "wrong_set_op"
    PARSE_ERROR = "parse_error"
    EXECUTION_ERROR = "execution_error"
    UNKNOWN = "unknown"


def _agg_func_names(sel: exp.Select) -> set[str]:
    names = set()
    for proj in sel.expressions:
        for fn in proj.find_all(exp.AggFunc):
            names.add(type(fn).__name__.lower())
    return names


def _has_subquery(sel: exp.Select) -> bool:
    frm = sel.args.get("from")
    in_from = bool(frm and frm.find(exp.Subquery))
    in_select = any(p.find(exp.Subquery) or p.find(exp.Select) for p in sel.expressions)
    return in_from or in_select


def _set_op(tree: exp.Expression) -> Optional[str]:
    node = tree.find(exp.Union, exp.Intersect, exp.Except)
    return type(node).__name__.lower() if node else None


def classify_failure(predicted: str, gold: str,
                     ex_result: Optional[dict] = None) -> list[FailureType]:
    try:
        pt = parse_sql(predicted)
        gt = parse_sql(gold)
    except ParseError:
        return [FailureType.PARSE_ERROR]

    failures: list[FailureType] = []
    if ex_result and (ex_result.get("predicted_error") or ex_result.get("gold_error")):
        failures.append(FailureType.EXECUTION_ERROR)

    if _canonical_sql(predicted) == _canonical_sql(gold):
        return [FailureType.CORRECT]

    pa, ga = _primary_select(pt), _primary_select(gt)
    cm = component_matching(predicted, gold)

    def differs(name: str) -> bool:
        r = cm.get(name)
        return r is not None and not r.match

    if pa is not None and ga is not None:
        if bool(pa.args.get("distinct")) != bool(ga.args.get("distinct")):
            failures.append(FailureType.MISSING_DISTINCT)

        if differs("select"):
            if _agg_func_names(pa) != _agg_func_names(ga):
                failures.append(FailureType.WRONG_AGGREGATION)
            else:
                failures.append(FailureType.WRONG_COLUMNS)

        if differs("from"):
            failures.append(FailureType.WRONG_TABLE)

        if differs("joins"):
            pj = pa.args.get("joins") or []
            gj = ga.args.get("joins") or []
            p_types = {(j.args.get("side") or j.args.get("kind") or "inner").lower() for j in pj}
            g_types = {(j.args.get("side") or j.args.get("kind") or "inner").lower() for j in gj}
            if len(pj) != len(gj):
                failures.append(FailureType.MISSING_JOIN)
            elif p_types != g_types:
                failures.append(FailureType.WRONG_JOIN_TYPE)
            else:
                failures.append(FailureType.WRONG_JOIN_CONDITION)

        if differs("where"):
            p_ops = {type(n).__name__ for n in (pa.args.get("where").find_all(exp.Connector)
                                                if pa.args.get("where") else [])}
            g_ops = {type(n).__name__ for n in (ga.args.get("where").find_all(exp.Connector)
                                                if ga.args.get("where") else [])}
            failures.append(FailureType.WRONG_CONDITION_OP if p_ops != g_ops
                            else FailureType.WRONG_CONDITION_VALUE)

        if differs("group_by"):
            failures.append(FailureType.MISSING_GROUPBY)
        if differs("having"):
            failures.append(FailureType.MISSING_HAVING)
        if differs("order_by"):
            failures.append(FailureType.WRONG_ORDERBY)
        if differs("limit"):
            failures.append(FailureType.WRONG_LIMIT)

        if _has_subquery(ga) and not _has_subquery(pa):
            failures.append(FailureType.MISSING_SUBQUERY)

    if _set_op(gt) != _set_op(pt):
        failures.append(FailureType.WRONG_SET_OP)

    return failures or [FailureType.UNKNOWN]


# ═══════════════════════════════════════════════════════════════════════════
# 8 — RELATÓRIO CONSOLIDADO
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SQLValidationReport:
    predicted: str
    gold: str
    ast_equivalent: bool = False
    ast_parse_error: Optional[str] = None
    structural_f1: dict = field(default_factory=dict)
    ex_match: Optional[bool] = None
    ex_compare_mode: str = "multiset"
    ex_predicted_rows: Optional[list] = None
    ex_gold_rows: Optional[list] = None
    ex_error: Optional[str] = None
    component_results: dict = field(default_factory=dict)
    component_avg_jaccard: float = 0.0
    string: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)


def validate_sql(
    predicted: str,
    gold: str,
    dsn: Optional[PgDSN] = None,
    conn: Optional[PgConnection] = None,
    compare_mode: CompareMode = "multiset",
) -> str:
    """Roda todas as métricas e retorna um JSON normalizado (string)."""
    r = SQLValidationReport(predicted=predicted, gold=gold)

    ast = ast_canonical_equivalence(predicted, gold)
    r.ast_equivalent = ast["equivalent"]
    r.ast_parse_error = ast.get("parse_error")

    r.structural_f1 = structural_f1(predicted, gold)

    ex_result = None
    if dsn or conn:
        ex_result = execution_accuracy(predicted, gold, dsn=dsn, conn=conn,
                                       compare_mode=compare_mode)
        r.ex_match = ex_result["match"]
        r.ex_compare_mode = ex_result.get("effective_mode", ex_result["compare_mode"])
        r.ex_predicted_rows = ex_result["predicted_rows"]
        r.ex_gold_rows = ex_result["gold_rows"]
        r.ex_error = ex_result.get("predicted_error") or ex_result.get("gold_error")

    cm = component_matching(predicted, gold)
    r.component_results = {k: {"match": v.match, "jaccard": v.jaccard}
                           for k, v in cm.items() if isinstance(v, ComponentMatchResult)}
    r.component_avg_jaccard = component_matching_score(predicted, gold)

    r.string = string_matching(predicted, gold)
    r.failures = [f.value for f in classify_failure(predicted, gold, ex_result)]

    payload = {
        "summary": {
            "ast_equivalent": r.ast_equivalent,
            "execution_match": r.ex_match,
            "component_avg_jaccard": round(r.component_avg_jaccard, 4),
            "string_exact": r.string.get("exact"),
            "string_similarity": r.string.get("similarity"),
            "failures": r.failures,
        },
        "ast": {"equivalent": r.ast_equivalent, "parse_error": r.ast_parse_error},
        "structural_f1": r.structural_f1,
        "execution": {
            "match": r.ex_match,
            "compare_mode": r.ex_compare_mode,
            "predicted_rows": r.ex_predicted_rows,
            "gold_rows": r.ex_gold_rows,
            "error": r.ex_error,
        } if r.ex_match is not None else None,
        "components": r.component_results,
        "string": r.string,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)