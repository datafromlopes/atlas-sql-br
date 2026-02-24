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
sql_validation.py  —  Research-grade SQL Validation Framework
=================================================================
Metrics:
  1. AST Canonical Equivalence  (custom recursive-descent parser)
  2. Logical Normalization       WHERE/HAVING in canonical CNF form
  3. Execution Accuracy          modes: set / multiset / list
  4. Structural F1               precision/recall over AST nodes
  5. Component Matching          per-clause, AST-aware
  6. String Matching             exact / similarity / token-F1
  7. Failure Taxonomy            automatic error classification

Parser covers:
  • SELECT DISTINCT / *, aliases, aggregate functions (COUNT, SUM, AVG…)
  • Subqueries in SELECT, FROM and WHERE
  • CTEs  (WITH … AS (…), WITH RECURSIVE)
  • JOINs: INNER / LEFT / RIGHT / FULL / CROSS / NATURAL + ON / USING
  • WHERE / HAVING with nested AND/OR/NOT/IN/BETWEEN/LIKE/EXISTS conditions
  • GROUP BY / ORDER BY (ASC/DESC) / LIMIT / OFFSET
  • Set operators: UNION / INTERSECT / EXCEPT  (ALL)
  • CASE WHEN … THEN … ELSE … END expressions
"""
from __future__ import annotations

import re, difflib, json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal, Optional

try:
    import psycopg2
    import psycopg2.extras
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

# PgConnection = psycopg2.extensions.connection (Any to avoid import errors when driver is absent)
PgConnection = Any
PgDSN = str   # "host=localhost dbname=mydb user=postgres password=secret"
              # or URL  "postgresql://user:pass@host:5432/dbname"


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — LEXER
# ═══════════════════════════════════════════════════════════════════════════════

class TT(Enum):
    KW=auto(); ID=auto(); NUM=auto(); STR=auto()
    LP=auto(); RP=auto(); COMMA=auto(); DOT=auto()
    STAR=auto(); SEMI=auto(); OP=auto(); EOF=auto()

KEYWORDS = {
    "select","from","where","group","by","having","order","limit","offset",
    "distinct","all","as","on","and","or","not","in","between","like","is",
    "null","true","false","exists","case","when","then","else","end",
    "join","inner","outer","left","right","full","cross","natural","using",
    "union","intersect","except","with","recursive","asc","desc",
    "count","sum","avg","min","max","coalesce","cast","over","partition",
    "rows","range","unbounded","preceding","following","current","row",
}

@dataclass
class Token:
    type: TT; value: str; pos: int = 0
    def __repr__(self): return f"<{self.type.name}:{self.value!r}>"

def lex(sql: str) -> list[Token]:
    toks, i, n = [], 0, len(sql)
    while i < n:
        if sql[i].isspace(): i += 1; continue
        if sql[i:i+2]=="--":
            while i<n and sql[i]!="\n": i+=1
            continue
        if sql[i:i+2]=="/*":
            e=sql.find("*/",i+2); i=e+2 if e!=-1 else n; continue
        # string literals
        if sql[i] in ("'",'"','`'):
            q=sql[i]; j=i+1
            while j<n:
                if sql[j]==q:
                    if j+1<n and sql[j+1]==q: j+=2
                    else: j+=1; break
                j+=1
            toks.append(Token(TT.STR, sql[i:j].lower(), i)); i=j; continue
        # numeric literals
        if sql[i].isdigit() or (sql[i]=='.' and i+1<n and sql[i+1].isdigit()):
            j=i
            while j<n and (sql[j].isdigit() or sql[j] in '.eE'): j+=1
            toks.append(Token(TT.NUM, sql[i:j], i)); i=j; continue
        # two-character operators
        if sql[i:i+2] in ("<>","!=",">=","<=","||"):
            toks.append(Token(TT.OP, sql[i:i+2], i)); i+=2; continue
        # single-character operators
        if sql[i] in "=<>+-%/&|^~!":
            toks.append(Token(TT.OP, sql[i], i)); i+=1; continue
        P={'(':TT.LP,')':TT.RP,',':TT.COMMA,'.':TT.DOT,'*':TT.STAR,';':TT.SEMI}
        if sql[i] in P:
            toks.append(Token(P[sql[i]], sql[i], i)); i+=1; continue
        if sql[i].isalpha() or sql[i]=='_':
            j=i
            while j<n and (sql[j].isalnum() or sql[j]=='_'): j+=1
            w=sql[i:j].lower()
            toks.append(Token(TT.KW if w in KEYWORDS else TT.ID, w, i)); i=j; continue
        i+=1
    toks.append(Token(TT.EOF,"",n))
    return toks


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — AST NODES
# ═══════════════════════════════════════════════════════════════════════════════

class ASTNode:
    def canonical(self) -> Any: raise NotImplementedError
    def node_type(self) -> str: return self.__class__.__name__

@dataclass
class Literal(ASTNode):
    value: str
    def canonical(self): return ("lit", self.value)

@dataclass
class Identifier(ASTNode):
    parts: list[str]
    alias: Optional[str] = None
    def canonical(self): return ("id", ".".join(self.parts))

@dataclass
class StarExpr(ASTNode):
    table: Optional[str] = None
    def canonical(self): return ("star", self.table)

@dataclass
class FuncCall(ASTNode):
    name: str
    args: list[ASTNode]
    distinct: bool = False
    alias: Optional[str] = None
    def canonical(self):
        ac = tuple(a.canonical() for a in self.args)
        return ("func", self.name, self.distinct, ac)

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    def canonical(self):
        lc, rc = self.left.canonical(), self.right.canonical()
        if self.op in ("=","<>","!=") and str(lc) > str(rc): lc, rc = rc, lc
        return ("binop", self.op, lc, rc)

@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode
    def canonical(self): return ("unary", self.op, self.operand.canonical())

@dataclass
class BoolOp(ASTNode):
    """AND/OR node with N operands — canonical form sorts operands (commutativity)."""
    op: str   # "and" | "or"
    operands: list[ASTNode]
    def canonical(self):
        parts = tuple(sorted(str(o.canonical()) for o in self.operands))
        return ("bool", self.op, parts)

@dataclass
class InExpr(ASTNode):
    expr: ASTNode
    values: list[ASTNode]
    negated: bool = False
    def canonical(self):
        vc = tuple(sorted(str(v.canonical()) for v in self.values))
        return ("in", self.negated, self.expr.canonical(), vc)

@dataclass
class BetweenExpr(ASTNode):
    expr: ASTNode
    low: ASTNode
    high: ASTNode
    negated: bool = False
    def canonical(self):
        return ("between", self.negated,
                self.expr.canonical(), self.low.canonical(), self.high.canonical())

@dataclass
class ExistsExpr(ASTNode):
    subquery: "SelectStmt"
    negated: bool = False
    def canonical(self): return ("exists", self.negated, self.subquery.canonical())

@dataclass
class CaseExpr(ASTNode):
    operand: Optional[ASTNode]
    whens: list[tuple[ASTNode, ASTNode]]
    else_: Optional[ASTNode]
    def canonical(self):
        wc = tuple((w.canonical(), t.canonical()) for w, t in self.whens)
        return ("case", self.operand.canonical() if self.operand else None,
                wc, self.else_.canonical() if self.else_ else None)

@dataclass
class OrderByItem(ASTNode):
    expr: ASTNode
    direction: str = "asc"
    def canonical(self): return ("orderitem", self.expr.canonical(), self.direction)

@dataclass
class JoinClause(ASTNode):
    join_type: str        # inner/left/right/full/cross/natural
    table: ASTNode        # Identifier ou Subquery
    condition: Optional[ASTNode] = None   # ON expr ou USING (cols)
    def canonical(self):
        return ("join", self.join_type,
                self.table.canonical(),
                self.condition.canonical() if self.condition else None)

@dataclass
class UsingExpr(ASTNode):
    columns: list[str]
    def canonical(self): return ("using", tuple(sorted(self.columns)))

@dataclass
class CTE(ASTNode):
    name: str
    query: "SelectStmt"
    def canonical(self): return ("cte", self.name, self.query.canonical())

@dataclass
class SubqueryExpr(ASTNode):
    """Subquery used as a scalar expression or in a FROM clause."""
    query: "SelectStmt"
    alias: Optional[str] = None
    def canonical(self): return ("subq", self.query.canonical())

@dataclass
class SetOp(ASTNode):
    """Set operation node: UNION / INTERSECT / EXCEPT."""
    op: str
    all: bool
    left: "SelectStmt"
    right: "SelectStmt"
    def canonical(self):
        return ("setop", self.op, self.all,
                self.left.canonical(), self.right.canonical())

@dataclass
class SelectStmt(ASTNode):
    ctes: list[CTE] = field(default_factory=list)
    distinct: bool = False
    columns: list[ASTNode] = field(default_factory=list)
    from_: list[ASTNode] = field(default_factory=list)
    joins: list[JoinClause] = field(default_factory=list)
    where: Optional[ASTNode] = None
    group_by: list[ASTNode] = field(default_factory=list)
    having: Optional[ASTNode] = None
    order_by: list[OrderByItem] = field(default_factory=list)
    limit: Optional[ASTNode] = None
    offset: Optional[ASTNode] = None
    set_op: Optional[SetOp] = None

    def canonical(self) -> tuple:
        # SELECT list: sort when there is no ORDER BY (projections are commutative without ORDER BY)
        cols_c = tuple(sorted(str(c.canonical()) for c in self.columns)
                       if not self.order_by else
                       (str(c.canonical()) for c in self.columns))
        # FROM: sort tables (JOINs change semantics, but plain aliases without ON are commutative)
        from_c = tuple(sorted(str(f.canonical()) for f in self.from_))
        # JOIN clauses: preserve order (LEFT JOIN is not commutative)
        joins_c = tuple(j.canonical() for j in self.joins)
        # GROUP BY: sort (commutative)
        gb_c = tuple(sorted(str(g.canonical()) for g in self.group_by))
        # ORDER BY: preserve order
        ob_c = tuple(o.canonical() for o in self.order_by)
        # CTEs: sort by name
        cte_c = tuple(sorted(str(c.canonical()) for c in self.ctes))

        return (
            "select",
            cte_c,
            self.distinct,
            cols_c,
            from_c,
            joins_c,
            self.where.canonical() if self.where else None,
            gb_c,
            self.having.canonical() if self.having else None,
            ob_c,
            self.limit.canonical() if self.limit else None,
            self.offset.canonical() if self.offset else None,
            self.set_op.canonical() if self.set_op else None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — PARSER RECURSIVE-DESCENT
# ═══════════════════════════════════════════════════════════════════════════════

class ParseError(Exception): pass

class Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.pos  = 0

    # ── utilities ───────────────────────────────────────────────────────────

    def peek(self, offset: int = 0) -> Token:
        i = self.pos + offset
        return self.toks[i] if i < len(self.toks) else self.toks[-1]

    def cur(self) -> Token: return self.peek()

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if t.type != TT.EOF: self.pos += 1
        return t

    def expect(self, *vals) -> Token:
        t = self.cur()
        if t.value not in vals:
            raise ParseError(f"Expected {vals}, found {t!r} at pos {t.pos}")
        return self.advance()

    def match(self, *vals) -> bool:
        if self.cur().value in vals: self.advance(); return True
        return False

    def match_seq(self, *vals) -> bool:
        """Try to consume a sequence of keywords, advancing only if all match."""
        for i, v in enumerate(vals):
            if self.peek(i).value != v: return False
        for _ in vals: self.advance()
        return True

    def cur_is(self, *vals) -> bool:
        return self.cur().value in vals

    def cur_type(self, *types) -> bool:
        return self.cur().type in types

    # ── entry point ─────────────────────────────────────────────────────────

    def parse(self) -> SelectStmt:
        stmt = self.parse_query()
        self.match(";")
        return stmt

    def parse_query(self) -> SelectStmt:
        ctes: list[CTE] = []
        if self.cur_is("with"):
            ctes = self.parse_ctes()
        stmt = self.parse_select_core()
        stmt.ctes = ctes
        # set operations: UNION / INTERSECT / EXCEPT
        while self.cur_is("union","intersect","except"):
            op = self.advance().value
            all_ = self.match("all")
            right = self.parse_select_core()
            stmt = SelectStmt(set_op=SetOp(op, all_, stmt, right))
        return stmt

    def parse_ctes(self) -> list[CTE]:
        self.advance()  # consume WITH keyword
        self.match("recursive")
        ctes = []
        while True:
            name = self.advance().value  # CTE alias name
            self.expect("as")
            self.expect("(")
            q = self.parse_query()
            self.expect(")")
            ctes.append(CTE(name, q))
            if not self.match(","): break
        return ctes

    def parse_select_core(self) -> SelectStmt:
        self.expect("select")
        distinct = self.match("distinct")
        if self.match("all"): distinct = False

        cols = self.parse_select_list()
        from_, joins = [], []
        where = group_by = having = order_by = limit = offset = None

        if self.cur_is("from"):
            self.advance()
            from_, joins = self.parse_from()

        if self.cur_is("where"):
            self.advance()
            where = self.parse_condition()

        if self.match_seq("group","by"):
            group_by = self.parse_expr_list()

        if self.cur_is("having"):
            self.advance()
            having = self.parse_condition()

        if self.match_seq("order","by"):
            order_by = self.parse_order_list()

        if self.cur_is("limit"):
            self.advance()
            limit = self.parse_primary()
            if self.cur_is("offset"):
                self.advance(); offset = self.parse_primary()

        return SelectStmt(
            distinct=distinct, columns=cols,
            from_=from_, joins=joins,
            where=where, group_by=group_by or [],
            having=having, order_by=order_by or [],
            limit=limit, offset=offset,
        )

    # ── SELECT list ─────────────────────────────────────────────────────────

    def parse_select_list(self) -> list[ASTNode]:
        items = [self.parse_select_item()]
        while self.match(","): items.append(self.parse_select_item())
        return items

    def parse_select_item(self) -> ASTNode:
        expr = self.parse_expr()
        alias = None
        if self.cur_is("as"):
            self.advance()
            alias = self.advance().value
        elif self.cur().type in (TT.ID, TT.STR) and not self.cur_is(*list(KEYWORDS)):
            alias = self.advance().value
        if hasattr(expr, "alias"): expr.alias = alias
        return expr

    # ── FROM + JOINs ────────────────────────────────────────────────────────

    def parse_from(self) -> tuple[list[ASTNode], list[JoinClause]]:
        tables = [self.parse_table_ref()]
        while self.match(","):
            tables.append(self.parse_table_ref())

        joins: list[JoinClause] = []
        while self.cur_is("join","inner","left","right","full","cross","natural"):
            jt = self.parse_join_type()
            tbl = self.parse_table_ref()
            cond = None
            if self.cur_is("on"):
                self.advance(); cond = self.parse_condition()
            elif self.cur_is("using"):
                self.advance(); self.expect("(")
                cols = [self.advance().value]
                while self.match(","): cols.append(self.advance().value)
                self.expect(")"); cond = UsingExpr(cols)
            joins.append(JoinClause(jt, tbl, cond))
        return tables, joins

    def parse_join_type(self) -> str:
        if self.cur_is("join"):   self.advance(); return "inner"
        if self.cur_is("inner"):  self.advance(); self.expect("join"); return "inner"
        if self.cur_is("cross"):  self.advance(); self.expect("join"); return "cross"
        if self.cur_is("natural"):self.advance(); self.match("join"); return "natural"
        side = self.advance().value   # left / right / full
        self.match("outer"); self.expect("join")
        return side

    def parse_table_ref(self) -> ASTNode:
        if self.cur().type == TT.LP:
            self.advance()
            sub = self.parse_query(); self.expect(")")
            alias = None
            if self.cur_is("as"): self.advance()
            if self.cur().type in (TT.ID,) and not self.cur_is(*list(KEYWORDS)):
                alias = self.advance().value
            return SubqueryExpr(sub, alias)
        parts = [self.advance().value]
        while self.cur().type == TT.DOT: self.advance(); parts.append(self.advance().value)
        alias = None
        if self.cur_is("as"): self.advance(); alias = self.advance().value
        elif self.cur().type == TT.ID and not self.cur_is(*list(KEYWORDS)):
            alias = self.advance().value
        node = Identifier(parts, alias)
        return node

    # ── ORDER BY ────────────────────────────────────────────────────────────

    def parse_order_list(self) -> list[OrderByItem]:
        items = [self.parse_order_item()]
        while self.match(","): items.append(self.parse_order_item())
        return items

    def parse_order_item(self) -> OrderByItem:
        expr = self.parse_expr()
        direction = "asc"
        if self.cur_is("asc"):  self.advance(); direction = "asc"
        elif self.cur_is("desc"): self.advance(); direction = "desc"
        return OrderByItem(expr, direction)

    def parse_expr_list(self) -> list[ASTNode]:
        items = [self.parse_expr()]
        while self.match(","): items.append(self.parse_expr())
        return items

    # ── EXPRESSIONS ─────────────────────────────────────────────────────────

    def parse_condition(self) -> ASTNode:
        return self.parse_or()

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        if not self.cur_is("or"): return left
        operands = [left]
        while self.cur_is("or"):
            self.advance(); operands.append(self.parse_and())
        return BoolOp("or", operands)

    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        if not self.cur_is("and"): return left
        operands = [left]
        while self.cur_is("and"):
            self.advance(); operands.append(self.parse_not())
        return BoolOp("and", operands)

    def parse_not(self) -> ASTNode:
        if self.cur_is("not"):
            self.advance(); return UnaryOp("not", self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_expr()
        # IS NULL / IS NOT NULL predicates
        if self.cur_is("is"):
            self.advance()
            neg = self.match("not")
            self.expect("null")
            return UnaryOp("is_not_null" if neg else "is_null", left)
        # BETWEEN predicate
        if self.cur_is("between") or (self.cur_is("not") and self.peek(1).value=="between"):
            neg = self.match("not"); self.expect("between")
            low = self.parse_expr(); self.expect("and"); high = self.parse_expr()
            return BetweenExpr(left, low, high, neg)
        # IN predicate (value list or subquery)
        if self.cur_is("in") or (self.cur_is("not") and self.peek(1).value=="in"):
            neg = self.match("not"); self.expect("in"); self.expect("(")
            if self.cur_is("select"):
                sub = self.parse_query(); self.expect(")")
                vals: list[ASTNode] = [SubqueryExpr(sub)]
            else:
                vals = self.parse_expr_list(); self.expect(")")
            return InExpr(left, vals, neg)
        # LIKE / NOT LIKE predicate
        if self.cur_is("like") or (self.cur_is("not") and self.peek(1).value=="like"):
            neg = self.match("not"); self.expect("like")
            pattern = self.parse_expr()
            return BinOp("not_like" if neg else "like", left, pattern)
        # binary comparison operators (=, <>, !=, <, >, <=, >=)
        if self.cur().type == TT.OP:
            op = self.advance().value
            right = self.parse_expr()
            return BinOp(op, left, right)
        return left

    def parse_expr(self) -> ASTNode:
        return self.parse_add()

    def parse_add(self) -> ASTNode:
        left = self.parse_mul()
        while self.cur().type==TT.OP and self.cur().value in ("+","-","||"):
            op=self.advance().value; right=self.parse_mul()
            left=BinOp(op,left,right)
        return left

    def parse_mul(self) -> ASTNode:
        left = self.parse_unary()
        while self.cur().type==TT.OP and self.cur().value in ("*","/","%"):
            op=self.advance().value; right=self.parse_unary()
            left=BinOp(op,left,right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.cur().type==TT.OP and self.cur().value=="-":
            self.advance(); return UnaryOp("-", self.parse_primary())
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        t = self.cur()

        # parenthesised expression or scalar subquery
        if t.type == TT.LP:
            self.advance()
            if self.cur_is("select"):
                sub = self.parse_query(); self.expect(")")
                return SubqueryExpr(sub)
            expr = self.parse_condition(); self.expect(")")
            return expr

        # EXISTS subquery
        if t.value == "exists":
            self.advance(); self.expect("(")
            sub = self.parse_query(); self.expect(")")
            return ExistsExpr(sub)

        # CASE expression
        if t.value == "case":
            return self.parse_case()

        # CAST expression
        if t.value == "cast":
            self.advance(); self.expect("(")
            expr = self.parse_expr()
            self.expect("as"); dtype = self.advance().value
            self.expect(")"); return FuncCall("cast", [expr, Literal(dtype)])

        # boolean / null literals
        if t.value in ("null","true","false"):
            self.advance(); return Literal(t.value)

        # numeric and string literals
        if t.type in (TT.NUM, TT.STR):
            self.advance(); return Literal(t.value)

        # bare star wildcard expression
        if t.type == TT.STAR:
            self.advance(); return StarExpr()

        # function call or plain identifier
        if t.type in (TT.ID, TT.KW):
            name = self.advance().value
            # qualified name: table.col or table.*
            if self.cur().type == TT.DOT:
                self.advance()
                if self.cur().type == TT.STAR:
                    self.advance(); return StarExpr(name)
                col = self.advance().value
                return Identifier([name, col])
            # parenthesised argument list → function call
            if self.cur().type == TT.LP:
                self.advance()
                distinct = self.match("distinct")
                args: list[ASTNode] = []
                if self.cur().type == TT.STAR:
                    self.advance(); args = [StarExpr()]
                elif not self.cur_is(")"):
                    args = self.parse_expr_list()
                self.expect(")")
                # OVER (window function) — skip window body, keep sentinel
                if self.cur_is("over"):
                    self.advance(); self.expect("(")
                    depth = 1
                    while depth:
                        if self.cur().type==TT.LP: depth+=1
                        elif self.cur().type==TT.RP: depth-=1
                        if depth: self.advance()
                    self.advance()
                return FuncCall(name, args, distinct)
            return Identifier([name])

        raise ParseError(f"Unexpected token: {t!r}")

    def parse_case(self) -> CaseExpr:
        self.expect("case")
        operand = None
        if not self.cur_is("when"): operand = self.parse_expr()
        whens = []
        while self.cur_is("when"):
            self.advance(); cond = self.parse_condition()
            self.expect("then"); result = self.parse_expr()
            whens.append((cond, result))
        else_ = None
        if self.cur_is("else"): self.advance(); else_ = self.parse_expr()
        self.expect("end")
        return CaseExpr(operand, whens, else_)


def parse_sql(sql: str) -> SelectStmt:
    """Public entry point for the SQL parser. Returns a SelectStmt AST."""
    tokens = lex(sql.strip())
    parser = Parser(tokens)
    return parser.parse()


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — AST CANONICAL EQUIVALENCE
# ═══════════════════════════════════════════════════════════════════════════════

def ast_canonical_equivalence(predicted: str, gold: str) -> dict[str, Any]:
    """
    Compare the canonical forms of two SQL ASTs.
    Handles commutativity of SELECT list (without ORDER BY), AND/OR operands,
    JOIN equivalence with ON conditions, and alias normalization.
    """
    result: dict[str, Any] = {
        "equivalent": False,
        "pred_canonical": None,
        "gold_canonical": None,
        "parse_error": None,
    }
    try:
        pred_ast = parse_sql(predicted)
        gold_ast = parse_sql(gold)
        pc = pred_ast.canonical()
        gc = gold_ast.canonical()
        result["pred_canonical"] = pc
        result["gold_canonical"] = gc
        result["equivalent"] = pc == gc
    except ParseError as e:
        result["parse_error"] = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — STRUCTURAL F1  (precision / recall over AST nodes)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_nodes(node: Any, nodes: list) -> None:
    """Recursively traverse a canonical tuple and collect all leaf and internal nodes."""
    if isinstance(node, tuple):
        nodes.append(node[:2])          # node type + name
        for child in node[1:]:
            _collect_nodes(child, nodes)
    elif isinstance(node, (list, tuple)):
        for item in node: _collect_nodes(item, nodes)


def structural_f1(predicted: str, gold: str) -> dict[str, float]:
    """
    Compare two ASTs as multisets of nodes.
    Returns precision, recall, and F1 score.
    """
    try:
        pc = parse_sql(predicted).canonical()
        gc = parse_sql(gold).canonical()
    except ParseError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": True}

    pred_nodes: list = []; gold_nodes: list = []
    _collect_nodes(pc, pred_nodes); _collect_nodes(gc, gold_nodes)

    pc_count = Counter(str(n) for n in pred_nodes)
    gc_count = Counter(str(n) for n in gold_nodes)

    common = sum(min(pc_count[k], gc_count[k]) for k in gc_count)
    tp, tg = sum(pc_count.values()), sum(gc_count.values())
    precision = common / tp if tp else 0.0
    recall    = common / tg if tg else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {"precision": round(precision,4), "recall": round(recall,4),
            "f1": round(f1,4), "pred_nodes": tp, "gold_nodes": tg, "common": common}


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — EXECUTION ACCURACY  (multiset-aware)
# ═══════════════════════════════════════════════════════════════════════════════

CompareMode = str  # "set" | "multiset" | "list"


# ── PostgreSQL connection helpers ────────────────────────────────────────────

def _pg_connect(dsn: PgDSN) -> PgConnection:
    """Open a psycopg2 connection from a DSN keyword string or a postgresql:// URL."""
    if not _PG_AVAILABLE:
        raise RuntimeError(
            "psycopg2 not found. Install it with: pip install psycopg2-binary"
        )
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        return psycopg2.connect(dsn)
    return psycopg2.connect(dsn)


def _run_query(conn: PgConnection, sql: str) -> tuple[Optional[list[tuple]], Optional[str]]:
    """
    Execute a read-only query against PostgreSQL using a SAVEPOINT to
    isolate errors without invalidating the outer transaction.
    Returns (rows, error_message).
    """
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


def execution_accuracy(
    predicted: str,
    gold: str,
    dsn: Optional[PgDSN] = None,
    conn: Optional[PgConnection] = None,
    compare_mode: CompareMode = "multiset",
) -> dict[str, Any]:
    """
    Execution Accuracy using PostgreSQL.

    Parameters
    ----------
    dsn          : DSN keyword string or postgresql:// URL (connection is opened and closed)
    conn         : already-open psycopg2 connection (not closed by the framework)
    compare_mode :
        'set'      → frozenset — ignores duplicates (may produce false positives)
        'multiset' → Counter   — respects duplicates, ignores order  ← default
        'list'     → list      — respects both order and duplicates (use with ORDER BY)

    DSN examples
    ------------
        dsn = "host=localhost port=5432 dbname=spider user=postgres password=secret"
        dsn = "postgresql://postgres:secret@localhost:5432/spider"
    """
    managed = False
    if dsn:
        conn = _pg_connect(dsn)
        conn.autocommit = False
        managed = True
    elif conn is None:
        raise ValueError("Provide either 'dsn' or 'conn' to run Execution Accuracy.")

    result: dict[str, Any] = dict(
        match=False, compare_mode=compare_mode,
        predicted_rows=None, gold_rows=None,
        predicted_error=None, gold_error=None,
    )
    try:
        pr, pe = _run_query(conn, predicted)
        gr, ge = _run_query(conn, gold)
        result.update(predicted_rows=pr, gold_rows=gr,
                      predicted_error=pe, gold_error=ge)
        if pr is not None and gr is not None:
            if compare_mode == "set":
                result["match"] = frozenset(pr) == frozenset(gr)
            elif compare_mode == "multiset":
                result["match"] = Counter(pr) == Counter(gr)
            else:
                result["match"] = pr == gr
    finally:
        if managed:
            conn.rollback()   # ensure no accidental DDL is committed
            conn.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — COMPONENT MATCHING  (AST-aware, per clause)
# ═══════════════════════════════════════════════════════════════════════════════

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

def _canonical_set(nodes: list[ASTNode]) -> set[str]:
    return {str(n.canonical()) for n in nodes}

def component_matching(predicted: str, gold: str) -> dict[str, ComponentMatchResult]:
    """
    Component Matching based on the real AST — no regex splitting.
    Each clause is compared via the canonical() form of its nodes.
    Aggregate functions with internal commas are no longer split incorrectly.
    """
    results: dict[str, ComponentMatchResult] = {}

    try:
        pa = parse_sql(predicted)
        ga = parse_sql(gold)
    except ParseError as e:
        return {"parse_error": ComponentMatchResult("parse_error", None, None, False, 0.0)}

    def cmp(name: str, p_nodes: list, g_nodes: list, ordered=False):
        if ordered:
            ps = [str(n.canonical()) for n in p_nodes]
            gs = [str(n.canonical()) for n in g_nodes]
            match = ps == gs
            # Jaccard over multiset to measure overlap
            pc, gc = Counter(ps), Counter(gs)
            common = sum(min(pc[k], gc[k]) for k in gc)
            union = sum((pc | gc).values())
            jaccard = common / union if union else 1.0
        else:
            ps_set = _canonical_set(p_nodes) if p_nodes else set()
            gs_set = _canonical_set(g_nodes) if g_nodes else set()
            match = ps_set == gs_set
            jaccard = _jaccard(ps_set, gs_set)
        results[name] = ComponentMatchResult(name, p_nodes, g_nodes, match, round(jaccard,4))

    # SELECT clause
    cmp("select", pa.columns, ga.columns)
    # FROM clause
    cmp("from", pa.from_, ga.from_)
    # JOIN clauses (order-sensitive)
    pj = [Literal(str(j.canonical())) for j in pa.joins]
    gj = [Literal(str(j.canonical())) for j in ga.joins]
    cmp("joins", pj, gj, ordered=True)
    # WHERE condition clause
    pw = [pa.where] if pa.where else []
    gw = [ga.where] if ga.where else []
    cmp("where", pw, gw)
    # GROUP BY clause
    cmp("group_by", pa.group_by, ga.group_by)
    # HAVING clause
    ph = [pa.having] if pa.having else []
    gh = [ga.having] if ga.having else []
    cmp("having", ph, gh)
    # ORDER BY clause (order-sensitive)
    cmp("order_by", pa.order_by, ga.order_by, ordered=True)
    # LIMIT clause
    pl = [pa.limit] if pa.limit else []
    gl = [ga.limit] if ga.limit else []
    cmp("limit", pl, gl)

    return results

def component_matching_score(predicted: str, gold: str) -> float:
    cm = component_matching(predicted, gold)
    relevant = [r for r in cm.values()
                if isinstance(r, ComponentMatchResult) and r.gold]
    return sum(r.jaccard for r in relevant) / len(relevant) if relevant else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — STRING MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(sql: str) -> str:
    sql = re.sub(r"--[^\n]*","",sql)
    sql = re.sub(r"/\*.*?\*/","",sql,flags=re.DOTALL)
    return re.sub(r"\s+"," ",sql.strip().lower()).rstrip(";").strip()

def _tok(sql: str) -> list[str]:
    return re.findall(r"[\w\.\*]+|<>|!=|>=|<=|[(),=<>!;]", _norm(sql))

def string_matching(predicted: str, gold: str) -> dict[str, Any]:
    pn, gn = _norm(predicted), _norm(gold)
    pt, gt = _tok(predicted), _tok(gold)
    pc, gc = Counter(pt), Counter(gt)
    common = sum(min(pc[t], gc[t]) for t in gc)
    tp, tg = sum(pc.values()), sum(gc.values())
    prec = common/tp if tp else 0.0
    rec  = common/tg if tg else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {
        "exact":      pn == gn,
        "similarity": round(difflib.SequenceMatcher(None, pn, gn).ratio(), 4),
        "token_f1":   {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9 — FAILURE TAXONOMY  (classificação automática de erros)
# ═══════════════════════════════════════════════════════════════════════════════

class FailureType(Enum):
    CORRECT                 = "correct"
    WRONG_AGGREGATION       = "wrong_aggregation"        # e.g. COUNT vs SUM
    WRONG_CONDITION_VALUE   = "wrong_condition_value"    # e.g. salary > 5000 vs 3000
    WRONG_CONDITION_OP      = "wrong_condition_op"       # e.g. AND vs OR
    MISSING_JOIN            = "missing_join"
    WRONG_JOIN_TYPE         = "wrong_join_type"          # e.g. LEFT vs INNER
    WRONG_JOIN_CONDITION    = "wrong_join_condition"
    MISSING_GROUPBY         = "missing_group_by"
    MISSING_HAVING          = "missing_having"
    WRONG_COLUMNS           = "wrong_columns"
    WRONG_TABLE             = "wrong_table"
    MISSING_DISTINCT        = "missing_distinct"
    WRONG_ORDERBY           = "wrong_order_by"
    WRONG_LIMIT             = "wrong_limit"
    MISSING_SUBQUERY        = "missing_subquery"
    WRONG_SET_OP            = "wrong_set_op"
    PARSE_ERROR             = "parse_error"
    EXECUTION_ERROR         = "execution_error"
    UNKNOWN                 = "unknown"

def classify_failure(predicted: str, gold: str,
                     ex_result: Optional[dict] = None) -> list[FailureType]:
    """
    Return a list of detected FailureType values.
    Uses the real AST for precise diagnosis.
    """
    failures: list[FailureType] = []

    try:
        pa = parse_sql(predicted)
        ga = parse_sql(gold)
    except ParseError:
        return [FailureType.PARSE_ERROR]

    if ex_result and (ex_result.get("predicted_error") or ex_result.get("gold_error")):
        failures.append(FailureType.EXECUTION_ERROR)

    # short-circuit: if ASTs are identical, it's correct
    if pa.canonical() == ga.canonical():
        return [FailureType.CORRECT]

    cm = component_matching(predicted, gold)

    def clause_differs(name: str) -> bool:  # helper: True when predicted != gold for clause
        r = cm.get(name)
        return r is not None and not r.match

    # DISTINCT flag mismatch
    if pa.distinct != ga.distinct:
        failures.append(FailureType.MISSING_DISTINCT)

    # SELECT columns
    if clause_differs("select"):
        # check whether aggregate functions differ
        p_funcs = {n.name for n in pa.columns if isinstance(n, FuncCall)}
        g_funcs = {n.name for n in ga.columns if isinstance(n, FuncCall)}
        if p_funcs != g_funcs and (p_funcs | g_funcs):
            failures.append(FailureType.WRONG_AGGREGATION)
        else:
            failures.append(FailureType.WRONG_COLUMNS)

    # FROM tables
    if clause_differs("from"):
        failures.append(FailureType.WRONG_TABLE)

    # JOIN clauses
    if clause_differs("joins"):
        p_jtypes = {j.join_type for j in pa.joins}
        g_jtypes = {j.join_type for j in ga.joins}
        if len(pa.joins) != len(ga.joins):
            failures.append(FailureType.MISSING_JOIN)
        elif p_jtypes != g_jtypes:
            failures.append(FailureType.WRONG_JOIN_TYPE)
        else:
            failures.append(FailureType.WRONG_JOIN_CONDITION)

    # WHERE condition
    if clause_differs("where"):
        pw_str = str(pa.where.canonical()) if pa.where else ""
        gw_str = str(ga.where.canonical()) if ga.where else ""
        # detect whether logical operator changed or only operand values changed
        p_ops = set(re.findall(r"'bool', '(\w+)'", pw_str))
        g_ops = set(re.findall(r"'bool', '(\w+)'", gw_str))
        if p_ops != g_ops:
            failures.append(FailureType.WRONG_CONDITION_OP)
        else:
            failures.append(FailureType.WRONG_CONDITION_VALUE)

    # GROUP BY
    if clause_differs("group_by"):
        failures.append(FailureType.MISSING_GROUPBY)

    # HAVING
    if clause_differs("having"):
        failures.append(FailureType.MISSING_HAVING)

    # ORDER BY
    if clause_differs("order_by"):
        failures.append(FailureType.WRONG_ORDERBY)

    # LIMIT
    if clause_differs("limit"):
        failures.append(FailureType.WRONG_LIMIT)

    # subquery present in gold but missing in predicted
    def has_subq(stmt: SelectStmt) -> bool:
        return any(isinstance(f, SubqueryExpr) for f in stmt.from_) or \
               any(isinstance(f, SubqueryExpr) for f in stmt.columns)
    if has_subq(ga) and not has_subq(pa):
        failures.append(FailureType.MISSING_SUBQUERY)

    # set operations (UNION / INTERSECT / EXCEPT)
    if (ga.set_op is None) != (pa.set_op is None):
        failures.append(FailureType.WRONG_SET_OP)
    elif ga.set_op and pa.set_op and ga.set_op.op != pa.set_op.op:
        failures.append(FailureType.WRONG_SET_OP)

    return failures or [FailureType.UNKNOWN]


# ═══════════════════════════════════════════════════════════════════════════════
# 10 — RELATÓRIO CONSOLIDADO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SQLValidationReport:
    predicted: str
    gold: str
    # AST canonical equivalence
    ast_equivalent: bool = False
    ast_parse_error: Optional[str] = None
    # Structural F1 over AST nodes
    structural_f1: dict = field(default_factory=dict)
    # Execution Accuracy (PostgreSQL)
    ex_match: Optional[bool] = None
    ex_compare_mode: str = "multiset"
    ex_predicted_rows: Optional[list] = None
    ex_gold_rows: Optional[list] = None
    ex_error: Optional[str] = None
    # Component Matching per clause
    component_results: dict = field(default_factory=dict)
    component_avg_jaccard: float = 0.0
    # String Matching metrics
    string: dict = field(default_factory=dict)
    # Failure Taxonomy labels
    failures: list = field(default_factory=list)


def validate_sql(
    predicted: str,
    gold: str,
    dsn: Optional[PgDSN] = None,
    conn: Optional[PgConnection] = None,
    compare_mode: CompareMode = "multiset",
) -> SQLValidationReport:
    r = SQLValidationReport(predicted=predicted, gold=gold)

    # AST canonical equivalence
    ast = ast_canonical_equivalence(predicted, gold)
    r.ast_equivalent  = ast["equivalent"]
    r.ast_parse_error = ast.get("parse_error")

    # Structural F1 over AST nodes
    r.structural_f1 = structural_f1(predicted, gold)

    # Execution Accuracy (optional — requires db connection)
    ex_result = None
    if dsn or conn:
        ex_result = execution_accuracy(predicted, gold, dsn=dsn,
                                       conn=conn, compare_mode=compare_mode)
        r.ex_match           = ex_result["match"]
        r.ex_compare_mode    = ex_result["compare_mode"]
        r.ex_predicted_rows  = ex_result["predicted_rows"]
        r.ex_gold_rows       = ex_result["gold_rows"]
        r.ex_error           = ex_result.get("predicted_error") or ex_result.get("gold_error")

    # Component Matching per clause
    cm = component_matching(predicted, gold)
    r.component_results     = {k: {"match": v.match, "jaccard": v.jaccard}
                                for k, v in cm.items()
                                if isinstance(v, ComponentMatchResult)}
    r.component_avg_jaccard = component_matching_score(predicted, gold)

    # String Matching metrics
    r.string = string_matching(predicted, gold)

    # Failure Taxonomy
    r.failures = [f.value for f in classify_failure(predicted, gold, ex_result)]

    return r