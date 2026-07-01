# AtlasSQL-BR

A Brazilian Portuguese geospatial **Text-to-SQL** dataset and fine-tuning pipeline, built on the **CulturaEduca** territorial data of Brazilian public schools. The project pairs natural language questions (NLQs) in Portuguese with executable **PostGIS** queries over a PostgreSQL database, and provides the tooling to fine-tune compact language models to generate that SQL and to evaluate them by **execution** against a live database.

CulturaEduca is a mapping platform created in partnership with the Brazilian Ministry of Culture (MinC) to store, analyze, and georeference the *educational territory* of public schools, supporting pedagogical projects, community action, and public policy. AtlasSQL-BR turns that territorial data into a benchmark for geospatial SQL generation, aimed at questions relevant to **public management** — budget allocation, improvement of the public school network, and analysis of school distribution and infrastructure.

---

## Overview

The task: given a question `q` in Portuguese and the CulturaEduca database schema `S`, learn a function `f : (q, S) → s`, where `s` is an executable PostGIS query whose result answers `q`.

The pipeline has three stages, each backed by a script in `core/`:

1. **Train** — fine-tune a base model on the dataset (`finetuning_strategies.py`).
2. **Generate** — produce SQL predictions on the held-out set (`generate_sql_preds.py`).
3. **Evaluate** — score predictions against the gold SQL, by execution and by structure (`evaluate_sql_preds.py` + `sql_validation.py`).

Experiments are configured through YAML files under `experiments/` and tracked with **MLflow**.

---

## Repository structure

```
ATLAS_SQL_BR/
├── core/                          # pipeline scripts
│   ├── benchmark_preview.py
│   ├── check_sql_sintax.py
│   ├── evaluate_sql_preds.py      # score predictions vs gold (execution + structure)
│   ├── finetuning_strategies.py   # training entry point (LoRA / IA3 / none)
│   ├── generate_sql_preds.py      # generate predictions on the held-out set
│   └── sql_validation.py          # metrics framework (sqlglot-based)
├── data/
│   ├── article_released/          # paper dataset — mirror of the HF `paper-released` branch
│   │   ├── train.parquet
│   │   └── validation.parquet
│   ├── predictions/               # predictions_v{0..5}.json (one per experiment)
│   ├── atlas_sql_br.csv           # main dataset (train / internal-val split)
│   ├── atlas_sql_br.parquet
│   ├── atlas_sql_br_validation.csv        # held-out evaluation set
│   └── atlas_sql_br_validation.parquet
├── experiments/                   # exp-v{0..5}.yaml experiment configs
├── results/
│   └── experiments_reports.json   # consolidated evaluation report
├── utils/                         # shared paths, variables, logging
│   ├── global_variables.py
│   └── utils.py
├── config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Dataset

Each record is a question-SQL pair with the following relation schema:

```
dataset(id: string, question: string, level: string, sql_code: string, train: integer)
```

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier of the pair, prefixed by complexity tier. |
| `question` | string | Natural language question in Brazilian Portuguese, phrased without database jargon. |
| `level` | string | Complexity tier: `Fácil`, `Médio`, `Difícil`, `Muito Difícil`. |
| `sql_code` | string | Gold PostGIS/SQL query, executable against the reference database. |
| `train` | integer | Partition flag: `1` = training pool, `0` = held-out. |

**Partition.** The dataset is split into a training pool and an internal validation slice (for early stopping) in `atlas_sql_br.csv`, plus a separate held-out evaluation set in `atlas_sql_br_validation.csv`. The held-out set shares no question and no SQL query with the training pool.

**Published version.** The dataset released with the paper is publicly available on the Hugging Face Hub at [`datafromlopes/atlas-sql-br`](https://huggingface.co/datasets/datafromlopes/atlas-sql-br) (Portuguese, Text-to-SQL, MIT license). The exact snapshot used in the paper lives on the [`paper-released`](https://huggingface.co/datasets/datafromlopes/atlas-sql-br/tree/paper-released) branch, whose `train.parquet` and `validation.parquet` are mirrored under `data/article_version/` for reproducibility. The `atlas_sql_br*` files at the root of `data/` are the working version of the dataset maintained in this repository.

### Complexity tiers

| Tier | Structural criteria |
|---|---|
| Fácil (Easy) | ≤1 JOIN; no aggregation; no subquery; direct spatial filter |
| Médio (Medium) | 2–3 JOINs; no aggregation; multiple spatial conditions |
| Difícil (Hard) | 3–4 JOINs; ≥1 aggregation (COUNT, SUM, AVG); multiple spatial functions |
| Muito Difícil (Very Hard) | 4+ JOINs; multiple aggregations; subqueries/CTEs; window functions; CASE WHEN; possible UNION |

---

## Database

The database is PostgreSQL + PostGIS. All geometries use **SRID 4674 (SIRGAS 2000)**, a geographic (degree-based) reference system.

### Tables

**IBGE (official Brazilian geography):**
`pais`, `regiao`, `unidade_federativa`, `rg_intermediaria`, `rg_imediata`, `municipio`, `distrito`, `subdistrito`, `setor` — administrative boundary polygons across the territorial hierarchy.

**CulturaEduca (thematic facilities):**
`escola` (school point geometry + identifiers), `microdados_ed_basica` (hundreds of school-census attributes, 1:1 with `escola`), and public facility points `cras`, `creas`, `centro_pop`, `biblioteca`.

### PostGIS conventions (golden rules)

These conventions keep the gold SQL executable and are worth following when authoring new pairs:

- Geometry column is named `geometry` in **every** table (not `geom`).
- SRID 4674 is geographic, so **distance/proximity** needs a `::geography` cast to get meters (`ST_Distance`, `ST_DWithin`).
- **Metric area**: `ST_Area(ST_Transform(geometry, 5880)) / 1e6` for km² (5880 = Brazil Polyconic). `municipio.area_km2` is precomputed.
- **Topological predicates** (`ST_Contains`, `ST_Within`, `ST_Intersects`) work directly on 4674 without transformation.
- Rounding: use `ROUND(value::numeric, n)` — `ROUND(double precision, n)` does not exist in PostgreSQL.
- Public school filter: `escola.tp_depende IN (1, 2, 3)` (1=Federal, 2=State, 3=Municipal; 4=Private).
- Location type: `escola.tp_localiz` — 1=Urban, 2=Rural.
- Infrastructure flags in `microdados_ed_basica` use the `in_*` prefix: 1=present, 0=absent.
- School ↔ microdata join: `escola.cd_entidade = microdados_ed_basica.cd_entidade` (1:1).

The full schema is in `database_structure.sql`.

---

## Fine-tuning

Six experiments are defined by pairing a base model with a PEFT method. Baselines (`none`) are evaluated untrained; the others adapt with LoRA or IA³.

| Exp. | Base model | PEFT |
|---|---|---|
| v0 | Llama-3.2-3B-Instruct | none (baseline) |
| v1 | Llama-3.2-1B-Instruct | LoRA |
| v2 | Qwen2.5-Coder-1.5B-Instruct | none (baseline) |
| v3 | Qwen2.5-Coder-1.5B-Instruct | LoRA |
| v4 | Qwen2.5-Coder-1.5B-Instruct | IA³ |
| v5 | Llama-3.2-1B-Instruct | IA³ |

All configurations share the training protocol (defined per experiment in `experiments/exp-v{N}.yaml`): up to 10 epochs, effective batch of 16 (batch 8 × grad-accum 2), AdamW, learning rate 1e-4, cosine schedule with 0.06 warmup, weight decay 0.01, and early stopping on validation loss (patience 5). Only decoder-only (causal) models are supported — Llama and Qwen. Training runs on Apple Silicon (MPS backend) in `fp32`; experiments are tracked with MLflow against a local PostgreSQL backend.

The prompt format is identical in training and generation:

```
Pergunta: {question}
SQL:
```

---

## Evaluation

Predictions are scored both by **execution** (the verdict) and by **structure/string** (diagnostics), plus a **failure taxonomy**. All metrics are computed by `sql_validation.py`, which parses SQL with **sqlglot** (postgres dialect).

| Metric | What it measures |
|---|---|
| **Execution Accuracy (EX)** | Predicted result set matches the gold result set on the live database (order-sensitive when the gold has `ORDER BY`, multiset otherwise). |
| **Executable Rate** | Fraction of predictions the database accepts and plans without error (`EXPLAIN`, plan-only). |
| **Structural F1** | Precision/recall/F1 over AST node multisets; insensitive to aliases and clause reordering. |
| **Geospatial Function F1** | Precision/recall/F1 over the multiset of PostGIS `ST_*` calls, extracted from the AST. Detects wrong spatial-predicate choices (e.g. `ST_Intersects` instead of `ST_Within`), with a per-function TP/FP/FN breakdown. |
| **Component Jaccard** | Per-clause Jaccard (SELECT, FROM, JOIN, WHERE, …), averaged over non-empty gold clauses. |
| **String Similarity / Exact Match** | Surface-form measures over normalized tokens (reported as a conservative lower bound). |
| **Failure taxonomy** | Categorizes each error (wrong condition value, wrong join condition, missing join, parse error, …). |

Execution Accuracy is the metric of record; string-level metrics can produce false negatives for spatially equivalent formulations, so they are reported as lower bounds.

The consolidated report is written to `results/experiments_reports.json`, containing per-experiment summaries — overall metrics, per-level breakdown, the geospatial per-function table, and the fine-tuned failure taxonomy.

---

## Usage

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
# install dependencies
uv sync
```

### 1. Train (one experiment at a time)

```bash
uv run python core/finetuning_strategies.py --experiment_version 3
```

Baseline experiments (`train_method: none`, i.e. v0 and v2) have nothing to train and are skipped here; their base-model generation is handled by the generation step.

### 2. Generate predictions on the held-out set

```bash
uv run python core/generate_sql_preds.py --experiment_version 3
```

Writes `data/predictions/predictions_v3.json`.

### 3. Evaluate

```bash
# score all predictions_v*.json against the live database
uv run python core/evaluate_sql_preds.py --dsn "postgresql://user:pass@host:5432/db"

# a specific file, or AST/structure-only (no database)
uv run python core/evaluate_sql_preds.py --predictions data/predictions/predictions_v3.json --dsn "..."
uv run python core/evaluate_sql_preds.py --no-db
```

Writes the consolidated report to `results/experiments_reports.json`. Pass `--save-details` to also write per-row detail (`details_v{N}.json`), which is off by default to keep the report small.

> Note: if the database password contains `@`, URL-encode it as `%40` in the DSN.

---

## Requirements

- Python (see `.python-version`)
- PostgreSQL + PostGIS with the AtlasSQL-BR schema loaded
- Key libraries: `sqlglot`, `psycopg2`, `transformers`, `peft`, `torch`, `polars`, `mlflow`

---

## Project context

AtlasSQL-BR is developed within the CulturaEduca partnership with the Institutional Evaluation Center of FEUSP (School of Education, University of São Paulo). By turning georeferenced educational-territory data into a Text-to-SQL benchmark, it supports data-informed pedagogical projects, field-work (*estágio*) planning, and the formulation of educational public policy — enabling non-experts to query complex geospatial data about the public school network in natural language.

## Dataset citation

The AtlasSQL-BR dataset is published on the Hugging Face Hub with DOI `10.57967/hf/8931`:

> Diego O. Lopes, Kelly R. Braghetto. *AtlasSQL-BR: A Brazilian Portuguese Geospatial Text-to-SQL Dataset*. Hugging Face, https://huggingface.co/datasets/datafromlopes/atlas-sql-br

## License

See the `LICENSE` file in the repository root. The published dataset on Hugging Face is released under the MIT license.