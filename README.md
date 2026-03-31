# VaultTech — Forging Line Cycle Time Analysis

Real industrial case study: analyzing per-piece cycle times on a steel forging line that produces track chain links for heavy earthmoving equipment.

## Overview

Each piece travels ~58 seconds through 4 zones: **Furnace → Main Press (4 strikes) → Auxiliary Press → Quench Bath**. The dataset contains ~180k raw PLC signal readings across 4 die matrices. An XGBoost model predicts total bath time after only the 2nd strike (~18s into the process), enabling real-time delay alerts.

See `docs/FINAL_PROJECT.md` for the full task list and evaluation rubric.

## Prerequisites

- **Python** ≥ 3.13
- **uv** — [astral.sh/uv](https://docs.astral.sh/uv/)
- **Docker** + Docker Compose

## Quick start

```bash
# 1. Initialize your git repository
git init
git add -A
git commit -m "initial: student starter template"

# 2. Install dependencies
uv sync

# 3. Create .env and start PostgreSQL + Flyway
cp infra/.env.example infra/.env
cd infra && docker compose up -d && cd ..

# 4. Seed the database with raw data
uv run python scripts/seed.py --env infra/.env

# 5. Launch JupyterLab (notebooks on :8888)
uv run lab

# 6. Launch Streamlit app (dashboard on :8501)
uv run app
```

## Project structure

```
├── app/                    Streamlit dashboard
├── data/
│   ├── *.csv.gz            Raw PLC data (for seeding PostgreSQL)
│   └── gold/               Generated parquet (gitignored)
├── deploy/                 SageMaker deployment script + README
├── docs/                   Project documentation (01–05) + FINAL_PROJECT.md
├── infra/                  docker-compose.yml + .env for local stack
├── migrations/
│   ├── sql/                Flyway migrations (V001–V008)
│   └── DATABASE.md         Full schema reference
├── models/                 Trained XGBoost model + metadata (generated)
├── notebooks/              Analysis pipeline (00–05)
├── scripts/                seed.py (data loader)
├── src/vaultech_analysis/  Python package (inference, app/lab launchers)
└── tests/                  pytest tests (inference, causes, SageMaker)
```

## Data architecture

Medallion pattern: **Bronze → Silver → Gold**

| Layer | Storage | Content |
|---|---|---|
| Bronze | PostgreSQL `bronze` schema | Raw PLC signal/value pairs (~1.4M rows) |
| Silver | PostgreSQL `silver.clean_pieces` | Cleaned, 1 row/piece (~169k rows) |
| Gold | `data/gold/pieces.parquet` | Enriched with partial times + production runs |

Pipeline: `01_bronze_to_silver.ipynb` → `02_silver_to_gold.ipynb` → `03_build_clean_dataset.ipynb` (quality gate)

## Useful commands

| Command | What it does |
|---|---|
| `uv sync` | Install/update dependencies |
| `uv run lab` | Launch JupyterLab |
| `uv run app` | Launch Streamlit dashboard |
| `uv run pytest` | Run all tests |
| `uv run pytest tests/test_app_e2e.py -v` | E2E browser tests (local app) |
| `VAULTECH_E2E_DOCKER=1 uv run pytest tests/test_app_e2e.py -v` | E2E browser tests (Docker container) |
| `uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5` | CLI prediction |

## Documentation

Read in order:

1. `docs/01_TheProduct.md` — What's being forged
2. `docs/02_ManufacturingProcess.md` — 4-zone line, timing, signals
3. `docs/03_CleaningUpData.md` — Cleaning rules and justification
4. `docs/04_DataArchitecture.md` — Medallion architecture
5. `docs/05_PredictiveModel.md` — XGBoost feature selection and training

## Technologies

| Category | Tools |
|---|---|
| Language | Python ≥ 3.13 |
| Package management | uv |
| Database | PostgreSQL 16, Flyway (versioned SQL migrations — schema changes are tracked in `migrations/sql/V001–V008*.sql` and applied automatically on `docker compose up`) |
| Data processing | Pandas, PyArrow (Parquet) |
| Machine learning | XGBoost, scikit-learn |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |
| Cloud (AWS) | SageMaker (Model Registry, real-time endpoints), ECR, ECS/Fargate, S3 |
| Testing | pytest |
| Versioning | commitizen (conventional commits) |
