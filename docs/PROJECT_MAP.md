# Project map and progress

**Last updated:** 2026-10-25  
**Purpose:** Persistent overview of the research implementation, decisions,
repository structure, and next work. Update this file as decisions are made
and implementation stages are completed.

## Research goal

Develop an explainable spatial-temporal crime forecasting decision-support
model for Police Zone 17 (Ondo and Ekiti). Enrich the source records,
particularly `Incident_Summary`, then predict incident counts by
**division/outpost × future time interval × crime category**. Rank these
location/time/category cells by predicted count to present likely hotspots.
A separate binary hotspot-classification target is not required.

## Agreed decisions

- Study area: Police Zone 17, Ondo and Ekiti State Commands.
- Use three aggregation resolutions: **24 hours, 3 days, and 1 week**. Run
  and report each resolution separately.
- Define prediction output and hotspot ranking dimensions as
  **location (division/outpost), future time interval, and crime category**.
- Preserve missing crime categories in a model-facing `unknown` category;
  retain the original blank source value and report the missingness.
- Keep original recorded fields distinguishable from NLP-enriched values.
- Use SHAP and attention to interpret model behavior, not to claim causation.
- Keep source incident data private and out of version control.

## Methodology-derived settings and confirmed forecast design

- Chapter Three specifies a **12-interval look-back**. At the three chosen
  resolutions this means 12 days, 36 days, or 12 weeks of history,
  respectively. These are separate experiments, not equivalent real-time
  durations.
- **Confirmed primary horizon:** one 24-hour interval ahead with the 24-hour
  aggregation. The primary model predicts counts for the next 24-hour
  location/time/category cells.
- Keep 3-day and weekly aggregations as exploratory comparisons. A one-step
  horizon at those resolutions predicts the next 3 days and next week,
  respectively. They are not 24-hour forecasts and must be reported
  separately from the primary result.
- Use chronological train/validation/test splits, initially 70/15/15, with
  all learned preprocessing fitted on training data only.
- Geocode addresses during preprocessing only after confirming an approved
  provider or source. Do not send sensitive incident addresses externally
  without authorization; validate coordinates against the administrative
  hierarchy and known locations.

## Planned sequential pipeline

1. **Data audit and intake** — inspect source columns, record coverage,
   missingness, duplicates, label quality, coordinates, and address fields.
2. **Cleaning and canonicalization** — standardize date/time, offence labels,
   commands, divisions/outposts, and location text; retain provenance.
3. **Narrative enrichment** — extract crime type/subcategory candidates,
   weapons, modus operandi, and other useful entities/signals from incident
   narratives. Preserve original values and enrichment confidence/review
   state. Add independently sourced holiday/festive calendar features.
4. **Structured feature engineering and fusion** — encode valid structured
   fields, cyclical time, and NLP features. Exclude unavailable-at-forecast
   and target-derived information.
5. **Spatial-temporal construction** — aggregate by division/outpost,
   time interval, and crime category; construct a validated spatial graph;
   build look-back sequences and future count targets.
6. **Chronological splitting** — partition in time, prevent leakage across
   preprocessing and target construction, and reserve test data for final
   evaluation.
7. **Baseline experiments** — ARIMA, Random Forest, XGBoost, SVM, KNN,
   standalone GRU, standalone MHSA, standalone GAT, and GCN. Include a
   persistence/historical-average reference if approved.
8. **Hybrid training** — train GAT-GRU-Transformer count forecasting,
   tune on validation data, and compare fairly with baselines.
9. **Evaluation and interpretation** — report count metrics for each
   resolution, rank location/time/category forecasts, produce maps, and
   apply SHAP, permutation importance, and attention analysis.
10. **Reproducible reporting** — preserve configuration, split dates,
    feature definitions, model settings, metrics, plots, and limitations.

## Repository map

| Path | Role | Progress |
| --- | --- | --- |
| `configs/default.yaml` | Data columns, aggregation windows, forecast target/horizon, model settings | `unknown` category; primary 24h one-step horizon confirmed |
| `data/raw/` | Private source records | Scaffold only; local data ignored by Git |
| `data/processed/` | Cleaned and enriched local data | Scaffold only |
| `src/predictive_policing/data/` | CSV intake, audit, cleaning, time features, chronological splits | Initial pipeline implemented and tested |
| `docs/data_pipeline.md` | Data preparation contract, outputs, and known limitations | Documented |
| `src/predictive_policing/enrichment/` | NLP, structured features, fusion | Package boundary created; implementation pending |
| `src/predictive_policing/spatial_temporal/` | Aggregation, geocoding integration, graph, sequences | Package boundary created; implementation pending |
| `src/predictive_policing/models/` | Hybrid and baseline models | Package boundary created; implementation pending |
| `src/predictive_policing/evaluation/` | Metrics, comparisons, ranking outputs | Package boundary created; implementation pending |
| `src/predictive_policing/explainability/` | SHAP, permutation, attention interpretations | Package boundary created; implementation pending |
| `docs/architecture.md` | Module responsibilities and design constraints | Initial architecture documented |
| `.github/skills/` | Reusable Copilot guidance for eight project skill areas | Initial skill set created |
| `.github/workflows/ci.yml` | Push/PR CI for locked install, tests, lint, and skill checks | Implemented; hosted run starts after push/PR |
| `tests/` | Automated checks | Configuration tests in place |
| `uv.lock` | Reproducible Python environment | Created by `uv sync` |

## Open decisions

- Confirm whether to include a persistence/historical-average baseline in
  addition to the requested models.
- Approve geocoding source/provider and privacy process before processing
  incident addresses with an external service.
- Decide how to resolve disagreements between recorded crime labels and
  narrative-derived label candidates, including whether human review is
  available.
- Confirm available data fields, coordinate coverage, and calendar source
  before finalizing feature engineering.

## Current status and next milestone

**Status:** Local Excel ingestion and transformation have been run on
`crimedata.xlsx`. The configured mapping is based on the `Crime_Data`
worksheet headers. Quality report: 29,555 source rows; 43 exact duplicate
rows removed; 29,512 rows prepared; 0 invalid occurrence timestamps; 3
source-missing crime categories mapped to model value `unknown`; 1 missing
coordinate pair; 16 rows participating
in repeated `Case_ID` values; 24 rows participating in repeated `Entry_No`
values; and 6,723 reports timestamped before occurrence. Both identifier
candidates repeat after exact-row deduplication, so neither should be treated
as a unique incident key without clarification. Review timestamp
inconsistencies before using reporting delay. Raw and prepared incident files
are ignored by Git. Missing crime categories are preserved as `unknown`.
The primary forecast is confirmed as 24 hours ahead; the 3-day and weekly
resolutions remain exploratory one-step comparisons. No predictive model has
been implemented.

**Next milestone:** Review repeated incident IDs and timestamp quality with
the data owner; then implement the interval aggregation and graph-ready
spatial-temporal representation. The absolute data paths in the configuration
must be updated if the project moves to another machine.

## Progress update convention

When a pipeline stage advances, update this map with its status (`planned`,
`in progress`, `done`, or `blocked`), relevant paths, validation performed,
and any new decisions. Keep confirmed decisions separate from proposals.
