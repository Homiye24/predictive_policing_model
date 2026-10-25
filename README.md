# Zone 17 Predictive Policing Research Implementation

This repository implements the research described in `aim_of_project.md` and
`CHAPTER THREE.docx`. It is intended for offline research and decision
support—not automated policing decisions or operational deployment.

The planned system enriches structured crime records with features from
`Incident_Summary`, aggregates incidents by division/outpost, time, and crime
category, and compares a hybrid GAT-GRU-Transformer model with selected
statistical and machine-learning baselines. It forecasts incident counts for
each location/time/category cell and ranks the cells by predicted risk; a
separate binary hotspot classifier is not required. Explainability methods
interpret model outputs but do not establish causal effects.

## Repository layout

```text
.
├── configs/                  # Dataset, target, and experiment settings
├── data/
│   ├── raw/                   # Private source records; never commit these
│   └── processed/             # Local derived data; never commit these
├── docs/
│   ├── architecture.md        # Module responsibilities and constraints
│   ├── data_pipeline.md       # Ingestion and transformation behavior
│   └── PROJECT_MAP.md         # Progress, decisions, pipeline, and open items
├── .github/skills/             # Reusable project-specific Copilot skills
├── .github/workflows/          # GitHub Actions CI
├── outputs/                   # Local experiment artifacts; never commit these
├── src/predictive_policing/
│   ├── data/                  # Intake, cleaning, and temporal splits
│   ├── enrichment/            # Narrative NLP and structured feature fusion
│   ├── spatial_temporal/      # Aggregation, geocoding, graphs, sequences
│   ├── models/                # Hybrid model and baselines
│   ├── explainability/        # SHAP, permutation, and attention analysis
│   └── evaluation/            # Metrics, comparisons, and ranked forecasts
└── tests/
```

## Getting started

Install [uv](https://docs.astral.sh/uv/), then sync the project and its
development tools. `uv sync` creates the project environment and lockfile:

```powershell
uv sync --group dev
```

Review `configs/default.yaml`, then validate its settings:

```powershell
uv run predictive-policing config-check --config configs/default.yaml
```

To extract, validate, and transform the configured local CSV or Excel workbook:

```powershell
uv run predictive-policing data-prepare --config configs/default.yaml
```

The command writes cleaned records and a counts-only quality report to the
configured absolute paths under `data/processed/`. It does not geocode addresses or
send records to external services. Review the configured source-column
mappings and quality report before using the prepared records downstream.
Missing crime categories are retained with `model_crime_category` set to
`unknown`; the original source value remains unchanged.

The primary experiment forecasts one 24-hour interval ahead using the
24-hour aggregation. The 3-day and weekly aggregations remain exploratory
one-step comparisons, forecasting the next 3 days and next week respectively.
The Chapter Three look-back is 12 intervals at each resolution.

Put source data at the configured local path or update the config. Never add
incident-level records, identifying information, credentials, or generated
data to version control. The `.gitignore` excludes local data and outputs.

## Development

Run tests and lint checks through the managed environment:

```powershell
uv run pytest
uv run ruff check .
```

GitHub Actions runs the locked environment, tests, lint, and Copilot skill
metadata checks for pushes and pull requests.

## Project skills

Reusable Copilot Agent Skills are stored under `.github/skills/` and provide
project-specific guidance for data engineering, narrative NLP, geospatial
graphs, forecasting, deep learning, explainability, privacy, and research
reproducibility.

Optional dependencies are grouped by module:

```powershell
uv sync --group dev --extra nlp --extra deep-learning --extra explainability --extra visualization
```

PyTorch and PyTorch Geometric installation can depend on the operating system
and accelerator; consult their official installation guidance if a compatible
wheel is not resolved automatically.

## Research safeguards

- Use chronological train/validation/test splits and fit preprocessing only
  on training data to prevent temporal leakage.
- Exclude reporting-time and post-incident fields when unavailable at the
  forecast cutoff.
- Rank division/outpost × future interval × crime-category cells by predicted
  incident counts; there is no separate binary hotspot target.
- Treat SHAP and attention as model interpretation, not evidence of causation.
- Do not send sensitive incident addresses to an external geocoding provider
  without approval.
- Do not use predictions for individual-level decisions or as a substitute
  for human judgment and due process.

## Current status

The configurable CSV ingestion, cleaning, time-feature, and quality-report
stage is implemented and covered by synthetic tests. Narrative enrichment,
interval aggregation, graph construction, model training, explainability,
and baseline evaluation remain to be implemented and validated against the
approved dataset and experiment design.
