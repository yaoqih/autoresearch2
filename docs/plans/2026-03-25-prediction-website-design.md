# Prediction Website Design

**Date:** 2026-03-25

**Goal:** Provide a clean read-only website for daily predictions and realized validation using the canonical date-keyed archives already produced by the batch pipeline.

## Product Scope

The website is display-only:

- no login
- no admin panel
- no manual trigger buttons
- no database

Deployment assumptions:

- the website runs on a Node-capable host
- the website and the quant artifacts are on the same machine
- the site can directly read `artifacts/predictions/*.json`

## Primary Navigation

The website is date-driven:

- `/`
  - latest official prediction
- `/history`
  - historical list of predictions by date
- `/predictions/[date]`
  - one-day detail page with top candidates and realized validation

This matches the archive design where one official prediction exists per trading day and same-day reruns overwrite that day's canonical record.

## Chosen Framework

Use Astro with Node SSR.

Reasons:

- the site is primarily content and data display
- the pages can be server-rendered directly from JSON files
- there is no need for a client-heavy framework
- Astro allows a small, readable codebase with direct filesystem access on the server

Use the official Node adapter in standalone mode so the built server can run directly with:

- `node ./dist/server/entry.mjs`

## Data Source

The site reads only the canonical website-facing artifacts:

- `artifacts/predictions/latest.json`
- `artifacts/predictions/index.json`
- `artifacts/predictions/daily/YYYY-MM-DD.json`

Raw run archives remain useful for audit but are not part of the website read path.

To avoid hardcoding a single deployment layout, the site should resolve the prediction archive directory in this order:

1. `QUANT_PREDICTIONS_DIR` environment variable
2. repo-relative fallback `../artifacts/predictions` from the `web/` project directory

## Rendering Strategy

Use SSR for all pages.

Why:

- latest and validation status can change without a rebuild
- deployment stays simple because the Node process and data live together
- the site remains stateless and filesystem-backed

## Error Handling

The site should degrade gracefully:

- if `latest.json` is missing, show an empty-state homepage
- if `index.json` is missing, show an empty history page
- if a per-day file is missing, return a proper 404 page
- if a prediction is still pending, display that clearly instead of showing empty metrics

## Visual Direction

Avoid generic dashboard styling.

Direction:

- editorial, finance-report feel
- warm neutral background instead of flat white
- serif-forward typography for titles and numbers
- restrained accents in deep green / copper / slate
- minimal but intentional motion

This should read like a daily market briefing, not an admin panel.

## Deployment Flow

The quant pipeline continues writing JSON artifacts. The website process simply reads them on request.

Operational flow:

1. run `predict` or `daily`
2. canonical JSON files update
3. the Astro SSR site serves the new data on the next request

No data sync job and no frontend rebuild are required for daily updates.
