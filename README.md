# AtlasSQL-BR

**A Brazilian Portuguese Geospatial Text-to-SQL Dataset with Spatial Hierarchies**

![AtlasSQL-BR](atlas-sql-br-banner.png)

[![Paper](https://img.shields.io/badge/Paper-SBBD%202026-1e3a8a)](https://doi.org/10.5753/sbbd.2026.249149)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-atlas--sql--br-yellow)](https://huggingface.co/datasets/datafromlopes/atlas-sql-br)
[![Dataset DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F8931-blue)](https://doi.org/10.57967/hf/8931)
[![License: GPL-3.0](https://img.shields.io/badge/Code-GPL--3.0-green)](LICENSE)
[![License: MIT](https://img.shields.io/badge/Dataset-MIT-green)](https://huggingface.co/datasets/datafromlopes/atlas-sql-br)

> Diego O. Lopes, Kelly R. Braghetto. **AtlasSQL-BR: A Brazilian Portuguese Geospatial Text-to-SQL Dataset with Spatial Hierarchies.** In: *Proceedings of the 41st Brazilian Symposium on Databases (SBBD 2026)*, pp. 85-98. SBC, 2026. [doi:10.5753/sbbd.2026.249149](https://doi.org/10.5753/sbbd.2026.249149)

AtlasSQL-BR pairs natural language questions in Brazilian Portuguese with executable **PostGIS** queries over real public data: the Brazilian school census and the official IBGE territorial hierarchy (country, region, state, municipality, district, census sector). This repository contains the dataset, the fine-tuning pipeline for compact LLMs, and an execution-based evaluation framework.

**Why it exists.** Text-to-SQL benchmarks such as Spider and BIRD have almost no spatial queries and are overwhelmingly in English. Geospatial questions need predicates like `ST_Within`, `ST_DWithin` and `ST_Touches`, correct SRID handling and hierarchy-aware joins, none of which general models learn from existing data. AtlasSQL-BR fills that gap for Portuguese.

```text
Pergunta: Quais escolas públicas estão a até 2 km da Avenida Paulista?
SQL:      SELECT e.no_entidade
          FROM escola e
          WHERE e.tp_depende IN (1, 2, 3)
            AND ST_DWithin(e.geometry::geography, (...)::geography, 2000);
```

---

## Table of contents

- [Quick start](#quick-start)
- [Dataset](#dataset)
- [Database](#database)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Project context](#project-context)
- [Citation](#citation)
- [License](#license)

---

## Quick start

```bash
# 1. install dependencies (uses uv)
uv sync

# 2. fine-tune one experiment (see experiments/exp-v*.yaml)
uv run python core/finetuning_strategies.py --experiment_version 3

# 3. generate SQL for the held-out set
uv run python core/generate_sql_preds.py --experiment_version 3

# 4. evaluate by execution against a live PostGIS database
uv run python core/evaluate_sql_preds.py --dsn "postgresql://user:pass@host:5432/db"

# or structure-only, no database needed
uv run python core/evaluate_sql_preds.py --no-db
```

To use only the dataset, skip the pipeline and load it from the Hub:

```python
from datasets import load_dataset
ds = load_dataset("datafromlopes/atlas-sql-br")           # working version
ds = load_dataset("datafromlopes/atlas-sql-br", revision="paper-released")  # exact paper snapshot
```

Requirements: Python (see `.python-version`), PostgreSQL + PostGIS with the AtlasSQL-BR schema loaded (`database_structure.sql`), and the libraries in `pyproject.toml` (`transformers`, `peft`, `torch`, `sqlglot`, `psycopg2`, `polars`, `mlflow`).

---

## Dataset

Each record is a question-SQL pair:

| Field      | Type    | Description                                                                 |
| ---------- | ------- | --------------------------------------------------------------------------- |
| `id`       | string  | Unique identifier, prefixed by complexity tier.                             |
| `question` | string  | Question in Brazilian Portuguese, written without database jargon.          |
| `level`    | string  | Complexity tier: `Fácil`, `Médio`, `Difícil`, `Muito Difícil`.              |
| `sql_code` | string  | Gold PostGIS/SQL query, executable against the reference database.          |
| `train`    | integer | `1` = training pool, `0` = held-out.                                        |

**Splits.** `data/atlas_sql_br.*` holds the training pool plus an internal validation slice used for early stopping. `data/atlas_sql_br_validation.*` is the held-out evaluation set and shares no question and no SQL with the training pool. The exact snapshot used in the paper is mirrored under `data/article_released/` and lives on the [`paper-released`](https://huggingface.co/datasets/datafromlopes/atlas-sql-br/tree/paper-released) branch on the Hub.

**Complexity tiers**

| Tier          | Structural criteria                                                                          |
| ------------- | -------------------------------------------------------------------------------------------- |
| Fácil         | at most 1 JOIN, no aggregation, no subquery, direct spatial filter                           |
| Médio         | 2 to 3 JOINs, no aggregation, multiple spatial conditions                                    |
| Difícil       | 3 to 4 JOINs, at least one aggregation (COUNT, SUM, AVG), multiple spatial functions         |
| Muito Difícil | 4+ JOINs, multiple aggregations, subqueries or CTEs, window functions, CASE WHEN, UNION      |

**Spatial coverage.** Questions exercise containment (`ST_Contains`, `ST_Within`), proximity and buffers (`ST_DWithin`, `ST_Distance`), intersection (`ST_Intersects`) and adjacency (`ST_Touches`), combined with joins across the IBGE territorial hierarchy.

**Augmentation.** To increase linguistic diversity while keeping the SQL unchanged, questions were augmented with controlled token deletion, insertion, synonym replacement and back-translation, guided by TF-IDF relevance scoring so that semantically load-bearing terms are preserved.

---

## Database

PostgreSQL + PostGIS. All geometries use **SRID 4674 (SIRGAS 2000)**, a geographic, degree-based reference system.

**IBGE tables (official territorial hierarchy):** `pais`, `regiao`, `unidade_federativa`, `rg_intermediaria`, `rg_imediata`, `municipio`, `distrito`, `subdistrito`, `setor`.

**CulturaEduca tables (thematic facilities):** `escola` (school points and identifiers), `microdados_ed_basica` (school-census attributes, 1:1 with `escola`), and public facility points `cras`, `creas`, `centro_pop`, `biblioteca`.

**PostGIS conventions used in the gold SQL**

- Geometry column is named `geometry` in every table.
- SRID 4674 is geographic, so distance and proximity require a `::geography` cast to get meters (`ST_Distance`, `ST_DWithin`).
- Metric area: `ST_Area(ST_Transform(geometry, 5880)) / 1e6` for km² (5880 = Brazil Polyconic). `municipio.area_km2` is precomputed.
- Topological predicates (`ST_Contains`, `ST_Within`, `ST_Intersects`) work directly on 4674.
- Rounding: `ROUND(value::numeric, n)`; `ROUND(double precision, n)` does not exist in PostgreSQL.
- Public school filter: `escola.tp_depende IN (1, 2, 3)` (1 Federal, 2 State, 3 Municipal; 4 Private).
- Location type: `escola.tp_localiz` (1 Urban, 2 Rural).
- Infrastructure flags in `microdados_ed_basica` use the `in_*` prefix (1 present, 0 absent).
- School to microdata join: `escola.cd_entidade = microdados_ed_basica.cd_entidade`.

Full schema: `database_structure.sql`.

---

## Fine-tuning

Experiments pair a compact base model with a PEFT method. Baselines are evaluated untrained.

| Exp. | Base model                  | Method          |
| ---- | --------------------------- | --------------- |
| v0   | Llama-3.2-3B-Instruct       | none (baseline) |
| v1   | Llama-3.2-1B-Instruct       | LoRA            |
| v2   | Qwen2.5-Coder-1.5B-Instruct | none (baseline) |
| v3   | Qwen2.5-Coder-1.5B-Instruct | LoRA            |
| v4   | Qwen2.5-Coder-1.5B-Instruct | IA³             |
| v5   | Llama-3.2-1B-Instruct       | IA³             |

Shared protocol (per-experiment YAML in `experiments/`): up to 10 epochs, effective batch 16 (8 × grad-accum 2), AdamW, lr 1e-4, cosine schedule with 0.06 warmup, weight decay 0.01, early stopping on validation loss (patience 5). Decoder-only models only (Llama, Qwen). Training runs on Apple Silicon (MPS) in fp32; runs are tracked with MLflow on a local PostgreSQL backend.

Prompt format, identical in training and generation:

```text
Pergunta: {question}
SQL:
```

---

## Evaluation

Predictions are scored by **execution** (the verdict) and by **structure** (diagnostics), plus a failure taxonomy. Everything is computed in `core/sql_validation.py` with `sqlglot` (postgres dialect).

| Metric                         | What it measures                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **Execution Accuracy (EX)**    | Predicted result set equals the gold result set on the live database (order-sensitive when gold has `ORDER BY`).   |
| **Executable Rate**            | Fraction of predictions the database plans without error (`EXPLAIN` only).                                         |
| **Structural F1**              | Precision, recall and F1 over AST node multisets; alias- and order-insensitive.                                    |
| **Geospatial Function F1**     | Precision, recall and F1 over the multiset of `ST_*` calls, with a per-function breakdown. Catches wrong predicate choice, e.g. `ST_Intersects` instead of `ST_Within`. |
| **Component Jaccard**          | Per-clause Jaccard (SELECT, FROM, JOIN, WHERE, ...) averaged over non-empty gold clauses.                           |
| **String similarity / EM**     | Surface-level measures over normalized tokens, reported as a conservative lower bound.                             |
| **Failure taxonomy**           | Categorizes each error: wrong condition value, wrong join condition, missing join, parse error, and so on.          |

Execution Accuracy is the metric of record. The consolidated report is written to `results/experiments_reports.json`.

---

## Results

<!-- TODO: fill from results/experiments_reports.json -->

| Exp. | Model                       | Method | EX (%) | Executable (%) | Geo-F1 |
| ---- | --------------------------- | ------ | ------ | -------------- | ------ |
| v0   | Llama-3.2-3B-Instruct       | none   |        |                |        |
| v1   | Llama-3.2-1B-Instruct       | LoRA   |        |                |        |
| v2   | Qwen2.5-Coder-1.5B-Instruct | none   |        |                |        |
| v3   | Qwen2.5-Coder-1.5B-Instruct | LoRA   |        |                |        |
| v4   | Qwen2.5-Coder-1.5B-Instruct | IA³    |        |                |        |
| v5   | Llama-3.2-1B-Instruct       | IA³    |        |                |        |

Per-tier breakdown, the per-function geospatial table and the failure taxonomy are in `results/experiments_reports.json`. See the paper for the full analysis.

---

## Repository structure

```text
atlas-sql-br/
├── core/                          # pipeline
│   ├── finetuning_strategies.py   # training entry point (LoRA / IA3 / none)
│   ├── generate_sql_preds.py      # generate predictions on the held-out set
│   ├── evaluate_sql_preds.py      # score predictions vs gold (execution + structure)
│   ├── sql_validation.py          # metrics framework (sqlglot-based)
│   ├── check_sql_sintax.py
│   └── benchmark_preview.py
├── data/
│   ├── article_released/          # paper snapshot (mirror of the HF paper-released branch)
│   ├── predictions/               # predictions_v{0..5}.json
│   ├── atlas_sql_br.{csv,parquet}            # training pool + internal validation
│   └── atlas_sql_br_validation.{csv,parquet} # held-out evaluation set
├── experiments/                   # exp-v{0..5}.yaml
├── results/experiments_reports.json
├── utils/
├── config.yaml
└── pyproject.toml
```

---

## Project context

AtlasSQL-BR is developed within the [CulturaEduca](https://culturaeduca.cc) partnership with the Institutional Evaluation Center of FEUSP (School of Education, University of São Paulo). CulturaEduca is a mapping platform created with the Brazilian Ministry of Culture to georeference the educational territory of public schools. Turning that data into a Text-to-SQL benchmark lets non-experts ask questions about the public school network in natural language, supporting pedagogical projects, field-work planning and educational public policy.

This is the research artifact of my MSc at IME-USP, advised by Prof. Kelly R. Braghetto.

---

## Citation

Paper:

```bibtex
@inproceedings{lopes2026atlassql,
  title     = {AtlasSQL-BR: A Brazilian Portuguese Geospatial Text-to-SQL Dataset with Spatial Hierarchies},
  author    = {Lopes, Diego O. and Braghetto, Kelly R.},
  booktitle = {Proceedings of the 41st Brazilian Symposium on Databases (SBBD)},
  pages     = {85--98},
  year      = {2026},
  publisher = {Sociedade Brasileira de Computa\c{c}\~{a}o},
  doi       = {10.5753/sbbd.2026.249149}
}
```

Dataset:

```bibtex
@misc{lopes2026atlassql_dataset,
  title     = {AtlasSQL-BR: A Brazilian Portuguese Geospatial Text-to-SQL Dataset},
  author    = {Lopes, Diego O. and Braghetto, Kelly R.},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/8931},
  url       = {https://huggingface.co/datasets/datafromlopes/atlas-sql-br}
}
```

---

## License

Code: GPL-3.0 (see `LICENSE`). Dataset: MIT, on the Hugging Face Hub.
