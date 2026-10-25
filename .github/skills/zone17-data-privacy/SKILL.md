---
name: zone17-data-privacy
description: Use whenever handling, profiling, testing, geocoding, logging, sharing, or versioning sensitive Zone 17 incident records or narratives.
---

# Zone 17 data privacy and security

## Default handling

Treat incident records, addresses, narratives, coordinates, and derived
record-level outputs as sensitive unless the data owner explicitly classifies
them otherwise.

- Keep source and derived incident-level data in local ignored data
  directories; never commit them.
- Never paste sensitive records into prompts, public issues, external
  services, logs, test fixtures, or generated documentation.
- Use synthetic or carefully de-identified examples in tests and screenshots.
- Minimize columns and retention in intermediate artifacts; protect local
  copies using the institution's approved controls.
- Avoid exposing incident IDs, exact addresses, or rare combinations in
  plots and progress reports.

## External services

Before sending addresses, narratives, or coordinates to a geocoder, cloud
model, or other external service, stop and obtain explicit data-owner and
institutional approval. Confirm the legal basis, provider terms, retention,
region, security, and permitted research use. Prefer approved local data
sources. Do not implement external upload as a default or silently fall back
to it.

## Development practices

- Keep credentials in environment/local secret stores, never source control.
- Validate outputs for identifying fields before export.
- Make errors explicit without printing sensitive input values.
- When in doubt about authorization or disclosure risk, pause the transfer
  and request clarification rather than assuming permission.
