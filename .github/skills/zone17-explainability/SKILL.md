---
name: zone17-explainability
description: Use when implementing or interpreting SHAP, permutation importance, GAT attention, temporal attention, or other explanations for Zone 17 model forecasts.
---

# Zone 17 explainability

## Objective

Describe which inputs influence model forecasts, at both population and
individual forecast-cell level, using the Chapter Three methods.

## Methods

- **GAT attention:** summarize influential neighboring nodes, with the
  explicit caveat that attention weights are not definitive explanations.
- **Permutation importance:** shuffle features using an evaluation design
  that preserves temporal/spatial structure as much as practical; quantify
  changes in held-out error.
- **SHAP:** use an appropriate documented explainer/surrogate for the deep
  model; state what model is explained, background data, output scale, and
  limitations.

## Workflow

1. Explain the exact forecast output: category, location, interval, horizon,
   and model version.
2. Compute global and local explanations using data that does not leak into
   model fitting or test-based feature selection.
3. Test explanation stability across samples, seeds, and aggregation
   resolutions where feasible.
4. Check that feature names and transformations map back to understandable
   source concepts.
5. Present explanations with plots/tables and concise limitations.

## Non-causality requirement

SHAP, feature importance, and attention describe model behavior and
associations. **They do not provide evidence that a factor causes crime.**
Do not write causal claims from these outputs. Causal questions require a
separate research design, assumptions, and evidence.

## Sensitive interpretation

Avoid presenting an attribution as evidence against an individual or
community. Explain uncertainty, data-quality issues, confounding, and the
risk that historical policing and recording patterns are reflected in model
outputs. Do not turn scores into automated enforcement decisions.
