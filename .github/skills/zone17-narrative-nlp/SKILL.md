---
name: zone17-narrative-nlp
description: Use when extracting or evaluating crime categories, subcategories, weapons, modus operandi, or other structured features from Zone 17 incident narratives.
---

# Zone 17 incident narrative NLP

## Objective

Enrich structured crime records using `Incident_Summary` (or the configured
equivalent) and preserve the distinction between recorded facts and
machine-extracted candidates.

## Workflow

1. Confirm the configured narrative and target columns and inspect a
   privacy-safe sample before selecting extraction methods.
2. Establish a text-cleaning pipeline that preserves meaningful crime terms,
   weapon names, negation, and contextual phrases.
3. Extract candidate crime type and subcategory, weapon, modus operandi, and
   other justified entities/signals. Use TF-IDF, rule-based indicators, and
   LDA features as specified in Chapter Three; treat embeddings as an
   optional extension.
4. Store extracted fields separately from original labels. Where supported,
   include confidence, method/version, and review status. Do not overwrite
   recorded categories automatically.
5. Evaluate extraction against a manually reviewed sample using
   class-appropriate metrics and a documented annotation guide. Review
   disagreement and low-confidence cases.
6. Fit corpus-dependent transformations (vocabulary, TF-IDF, topic models)
   on training data only to prevent information leakage.
7. Test empty, short, misspelled, ambiguous, multilingual, and conflicting
   narratives.

## Guardrails

- A narrative classifier predicting the recorded crime label is a diagnostic,
  not a source of ground-truth labels and not part of the forecasting model
  unless explicitly approved.
- Do not infer a weapon or offence as fact from ambiguous wording. Preserve
  uncertainty.
- Do not feed target-period narrative text into a forecast made before that
  narrative would exist.
- Keep original narratives and any sensitive text out of logs, tests, and
  version control.

## Expected outputs

An auditable feature table joined by a stable internal record key, a feature
dictionary, extraction quality results, and tests that verify provenance and
train-only fitting.
