# Data ingestion and transformation

## Run

1. Copy the authorized source CSV to the configured local path.
2. Verify every source column mapping and the local timezone in
   `configs/default.yaml`.
3. Run `uv run predictive-policing data-prepare --config configs/default.yaml`.
4. Review `data/processed/data_quality_report.json` before downstream use.

The reader accepts UTF-8/UTF-8-BOM CSV by default and XLSX/XLSM workbooks using
the configured worksheet. It does not infer unknown schemas, geocode
addresses, or contact external services. The current configuration points to
the local workbook using an absolute Windows path; update it if the project
is moved to another machine.

## Transformation contract

- Source columns are retained unchanged; standardized model-facing fields
  use the `model_` prefix.
- Exact duplicate rows are removed. Conflicting rows with duplicate case IDs
  or entry numbers are retained and counted for review; neither candidate is
  assumed unique.
- Required occurrence timestamps are parsed in the configured IANA timezone
  when naive, then stored as UTC. Invalid values remain present as missing
  `model_occurred_at` and are counted; chronological splitting refuses to
  proceed while timestamps are missing.
- Division/outpost and category values are whitespace-normalized and
  case-folded in model-facing fields. Missing categories are represented as
  the configured `unknown` category in `model_crime_category`; source values
  remain unchanged. The quality report separately records source-missing
  category rows and rows assigned to the unknown bucket. This does not
  harmonize synonyms or establish a validated crime taxonomy.
- Coordinates are numeric/range-checked and retained only as valid pairs.
  Missing or invalid coordinates are reported; no location is inferred.
- Reporting delay and cyclical hour, weekday, month, and day-of-year
  features are derived. Negative reporting delays are set missing and
  reported. Features use local time for the configured timezone.
- Reporting delay must not be used as a forecast input without a separate
  availability and quality review; many source rows have occurrence/report
  timestamp ordering that produces a negative delay.
- No records are dropped for missing timestamps, coordinates, or categories.
  Downstream stages must explicitly define treatment; the count-only quality
  report makes these cases visible.
- Prepared CSV and JSON report writes use temporary files and atomic
  replacement. Input, prepared, and report paths must be distinct.

## Temporal split helper

`split_chronologically` assigns distinct timestamps in chronological order
and keeps every row with the same timestamp in one partition. Use it on the
aggregated interval dataset for the research train/validation/test split,
not on raw incidents where distinct events within one interval may have
different timestamps.

## Not yet included

This stage does not perform NLP enrichment, taxonomy mapping, holiday
calendar joins, learned imputers/encoders/scalers, spatial graph construction,
aggregation into the 24-hour/3-day/weekly targets, or model fitting. Learned
preprocessing must be fitted on training data only when those stages are
implemented.
