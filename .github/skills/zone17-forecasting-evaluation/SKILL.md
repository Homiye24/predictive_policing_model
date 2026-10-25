---
name: zone17-forecasting-evaluation
description: Use when defining temporal windows and targets, building forecasts, creating chronological splits, evaluating models, or ranking predicted Zone 17 risk cells.
---

# Zone 17 forecasting and evaluation

## Research target

Forecast incident counts for each
**division/outpost × future time interval × crime category** cell. Rank cells
by predicted counts to present likely hotspots. Do not add a separate binary
hotspot classifier unless the research objective changes.

## Temporal setup

- Run the configured resolutions as separate experiments: `24h`, `3d`, and
  `1w`.
- Chapter Three specifies 12 look-back intervals. This corresponds to
  12 days, 36 days, and 12 weeks respectively; report each resolution
  separately.
- The confirmed primary experiment is one 24-hour interval ahead with the
  24-hour aggregation.
- The 3-day and weekly aggregations are exploratory one-step comparisons
  (next 3 days and next week). Do not describe them as 24-hour forecasts.
- Keep horizon and primary resolution configurable; report the resolutions
  separately.
- Define interval boundaries, timezone, incomplete boundary periods, and
  missing/no-record intervals explicitly.

## Evaluation workflow

1. Build target counts by location, interval, and category. Ensure unobserved
   cells are treated consistently and not confused with missing records.
2. Use chronological train/validation/test periods (initially 70/15/15, as in
   Chapter Three). Fit data transformations on training data only.
3. Ensure each model receives information available at the same forecast
   cutoff and predicts the same target and horizon.
4. Tune on validation data. Do not use test results for model selection.
5. Report MAE, RMSE, and R-squared as in Chapter Three; explain edge cases
   such as all-zero targets or undefined R-squared. Do not report MAPE for
   sparse count targets with zeros.
6. Report results separately by aggregation resolution and, where feasible,
   by crime category and location. Include uncertainty or repeated-run
   variation when implemented.
7. Rank forecast cells by predicted count and clearly label interval,
   category, location, and forecast time. Describe this as relative predicted
   volume/risk, not certainty.

## Guardrails

- Do not let target-period incidents, narratives, reporting outcomes, or
  preprocessing statistics leak into input features.
- Include a simple persistence/historical-average reference if approved; do
  not omit a naive comparison without explaining why.
- Compare all baselines and the hybrid under identical splits, target
  definitions, and evaluation conventions.
- Keep empirical metrics distinct from recommendations or operational claims.
