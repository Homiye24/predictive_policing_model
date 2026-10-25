---
name: zone17-deep-learning
description: Use when implementing, training, debugging, or comparing the Zone 17 hybrid GAT-GRU-Transformer and deep-learning baselines.
---

# Zone 17 deep-learning models

## Objective

Implement the chapter's spatial-temporal model with clear module boundaries
and compare it fairly with requested baselines.

## Model plan

- **Hybrid:** GAT encodes spatial relationships per interval; GRU models
  shorter-range temporal dynamics; Transformer/MHSA models longer-range
  dependencies; a count-regression head predicts future counts per
  location/time/category cell.
- **Requested baselines:** standalone GRU, standalone MHSA, standalone GAT,
  and GCN, alongside non-neural ARIMA, Random Forest, XGBoost, SVM, and KNN.
- Keep output/target shapes, category handling, graph node order, and temporal
  alignment explicit and tested.

## Implementation workflow

1. Confirm target, horizon, aggregation resolution, look-back, feature
   schema, graph, and split dates from configuration.
2. Build tensor/dataset adapters separately from model definitions.
3. Implement and test each component on synthetic small inputs before
   training on research data.
4. Use validation-only tuning and record seeds, parameters, dependency
   versions, checkpoints, and metrics.
5. Follow Chapter Three's initial regularization approach: dropout, weight
   decay, early stopping, and gradient clipping. Tune on validation data.
6. Run baselines under the same information cutoff and split protocol.
7. Check shapes, finite losses, gradients, empty/missing nodes, batch
   ordering, and reproducibility.

## Guardrails

- Do not encode future or target-derived information in model inputs.
- Do not compare models trained on different target construction or test
  periods as if the comparison were controlled.
- Keep model code free of implicit file reads and hidden global settings.
- Choose CPU/GPU dependencies deliberately; keep optional ML dependencies
  isolated in project extras and lock the environment with `uv`.
- A good training score alone is not evidence of useful forecasting; inspect
  naive baselines, sparse-target behavior, and temporal generalization.
