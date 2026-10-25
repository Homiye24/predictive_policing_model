---
name: zone17-data-engineering
description: Use when inspecting, loading, validating, cleaning, splitting, or documenting the Zone 17 incident dataset and its structured fields.
---

# Zone 17 data engineering

## Objective

Create traceable, leakage-safe incident tables for the Zone 17 predictive
policing research. Keep raw files immutable and all incident data local.

## Workflow

1. Read the configured input path and column mappings from
   `configs/default.yaml`; do not assume column names beyond the configuration.
2. Profile row counts, schema, date coverage, missingness, duplicate IDs,
   category frequencies, coordinate validity, and reporting delays.
3. Normalize timestamps and category/administrative labels while retaining
   source values or a provenance record for every transformation.
4. Apply documented duplicate and missing-data rules. Do not silently discard
   records or invent coordinates.
5. Build chronological train/validation/test partitions. Fit encoders,
   imputers, and scalers on training data only, then apply them to later
   partitions.
6. Save derived data only below `data/processed/`; keep raw and derived
   incident-level data out of Git.
7. Add tests for parsing, mappings, missing values, duplicate handling,
   chronology, and leakage-sensitive transformations.

## Guardrails

- Do not use incident IDs, post-incident action/status, reporting delay, or
  any other field as a predictor unless it is available at the forecast
  cutoff and its use is justified.
- Distinguish missing from unknown and from an actual negative value.
- Make data-quality failures visible with actionable errors or a report;
  do not silently coerce bad values into plausible ones.
- Never print or commit raw narratives or personally identifying details.

## Expected outputs

A cleaned, documented dataset; a quality summary with no unnecessary
identifiers; stable category/column mappings; and reproducible chronological
split boundaries.
