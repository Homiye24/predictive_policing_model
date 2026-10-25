---
name: zone17-research-reproducibility
description: Use when planning or documenting project stages, changing methodology decisions, running experiments, or updating the project progress map.
---

# Zone 17 research workflow and reproducibility

## Source of truth

Consult `aim_of_project.md`, `CHAPTER THREE.docx`,
`docs/architecture.md`, `docs/PROJECT_MAP.md`, and `configs/default.yaml`
before changing research assumptions. Keep documented user decisions
distinct from assistant recommendations and unresolved questions.

## Workflow

1. Work through the pipeline in dependency order: data audit; cleaning;
   enrichment; spatial-temporal representation; target construction;
   chronological splits; baselines and hybrid model; evaluation;
   explanation; reporting.
2. For each stage, state inputs, outputs, assumptions, validation, and
   dependencies on earlier decisions.
3. Record final experiment choices in configuration, not only in prose.
   Current confirmed aggregation windows are 24-hour, 3-day, and weekly;
   the forecast lead time remains to be confirmed. The 12-interval look-back
   is interpreted separately at each resolution.
4. Make changes modular and add focused tests. Use `uv` for dependency
   management, preserve `uv.lock`, and run tests and lint checks.
5. Record data version/quality summary, split dates, feature definitions,
   seeds, dependency versions, model settings, metrics, and known limitations
   for every reported experiment.
6. Update `docs/PROJECT_MAP.md` after meaningful decisions or completed
   milestones: revise status, affected paths, validation, open items, and
   next milestone. Never mark work complete without verification.

## Research integrity

- Keep the reserved test period out of tuning and feature selection.
- Distinguish observed results from expectations or recommendations.
- Report negative, weak, or inconclusive results honestly.
- Describe SHAP/attention as model interpretation, not causal evidence.
- Keep generated examples and reports free of sensitive incident-level data.
