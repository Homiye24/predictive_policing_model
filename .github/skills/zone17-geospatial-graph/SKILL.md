---
name: zone17-geospatial-graph
description: Use when validating coordinates, geocoding approved location text, mapping Zone 17 administrative units, or building the spatial graph.
---

# Zone 17 geospatial analysis and graph construction

## Objective

Represent the study geography at division/outpost level and construct a
validated spatial graph that supports the spatial-temporal models.

## Workflow

1. Standardize state, area command, division/outpost, and incident-location
   labels; maintain an explicit mapping from source labels to canonical nodes.
2. Validate coordinates against the expected Nigerian region, coordinate
   reference system, administrative hierarchy, and known locations.
3. Prefer approved authoritative gazetteers or supplied coordinates. If
   geocoding is authorized, use it as a preprocessing step, record source,
   match quality, and timestamp, and manually inspect ambiguous matches.
4. Do not infer an exact incident location from a division centroid. Keep
   incident coordinates, address matches, and administrative centroids as
   different provenance types.
5. Build nodes for the agreed division/outpost units. Construct candidate
   edges using the methodology's centroid Haversine distance and Gaussian
   kernel with a documented bandwidth and cutoff. Review whether
   administrative adjacency should supplement distance-based edges.
6. Check graph connectivity, isolated nodes, duplicate nodes, self-loops,
   symmetry/weight rules, and coordinate coverage. Save graph metadata and
   node mappings.
7. Test distance calculations, cutoff behavior, edge weights, and missing or
   invalid coordinate handling.

## Privacy and provider constraints

- Never send sensitive incident addresses to Google Maps or another external
  geocoding service without explicit institutional/legal approval and an
  approved data-handling process.
- Prefer local or approved authoritative sources where possible.
- Do not make a live external API call part of model inference.
- Review provider terms, billing, retention, and permitted research use
  before any approved integration.

## Expected outputs

A canonical node registry, validated coordinate/provenance table, graph
definition with documented parameters, and a data-quality report for
geocoding and spatial coverage.
