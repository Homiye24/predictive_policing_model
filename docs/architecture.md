# Implementation architecture

The pipeline is divided into packages with explicit responsibilities. Modules
should exchange pandas data frames for tabular stages and typed PyTorch tensors
or graph objects at the modeling boundary. Keep configuration and file I/O at
the pipeline edges; avoid making package modules read project files implicitly.

| Package | Responsibility | Expected output |
| --- | --- | --- |
| `data` | Load source files, map configured columns, clean records, and create chronological splits | Validated incident frames and train/validation/test ranges |
| `enrichment` | Process incident narratives, engineer structured features, and fuse feature blocks | Feature matrix with documented feature names |
| `spatial_temporal` | Aggregate incidents by division and interval, construct the location graph, and create look-back sequences | Time-indexed node features, graph, and forecast targets |
| `models` | Implement the hybrid GAT-GRU-Transformer and selected baselines, predicting incident counts for each location/time/category cell | Fitted predictors and count forecasts |
| `evaluation` | Apply the same temporal experiment protocol and count metrics at each aggregation resolution; rank predicted location/time/category cells | Comparable metrics, ranked forecasts, and saved result tables |
| `explainability` | Generate model attention summaries, permutation importance, and SHAP attributions | Global and per-forecast model explanations |

The CLI should orchestrate these packages; it should not contain modeling or
preprocessing logic. Predict incident counts and derive the hotspot/risk
ranking from those predictions; do not train a separate binary hotspot
classifier. Share preprocessing and spatial-temporal representations where
their definitions are genuinely identical.

The project uses three separately evaluated aggregation resolutions: 24 hours,
3 days, and 1 week. The primary experiment is the 24-hour resolution with a
one-interval-ahead horizon (predict the next 24 hours). The 3-day and weekly
resolutions are exploratory one-step comparisons whose horizons are the next
3 days and next week; do not describe these as 24-hour forecasts. The
prediction target is incident count for each
division/outpost, future interval, and crime category. A ranked list of these
cells is the hotspot/risk output; a separate binary hotspot classifier is not
required. Record the horizon in the experiment config. The Chapter Three
look-back setting is 12 intervals,
interpreted separately at each resolution (12 days, 36 days, or 12 weeks).
These are different elapsed histories, so report results separately by
resolution.

NLP-extracted labels and attributes should remain distinguishable from the
original recorded fields, with confidence and review status retained where
possible. SHAP and attention explain model behavior; they do not establish
that a factor causes crime. Geocoding is a preprocessing dependency, not a
live model dependency; do not send sensitive incident addresses to an
external provider without approval.
