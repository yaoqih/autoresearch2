# Prediction Website Archive Design

**Date:** 2026-03-25

**Goal:** Make prediction artifacts directly consumable by a date-driven website that shows today's pick and historical realized performance.

## Context

The current archive layout is optimized for offline runs:

- `artifacts/predictions/<archive_id>/prediction.json`
- `artifacts/predictions/<archive_id>/top_candidates.csv`
- `artifacts/validation/history.csv`

This is awkward for the website because the primary key is `archive_id`, while the website's primary navigation is `as_of_date`. The current layout also splits current prediction detail and realized validation into separate locations, forcing the frontend or backend to join data from directories and CSV files.

The product decision is now explicit:

- one model only
- one official prediction per trading day
- reruns on the same day should overwrite the official website record

## Requirements

The website should be able to read:

- the latest prediction quickly
- the full historical list in date-descending order
- a single day's full detail, including top candidates and realized validation once available

The training and prediction pipeline should still keep enough raw artifacts for audit and debugging.

## Chosen Approach

Keep raw run archives, but add a canonical date-keyed archive layer for website consumption.

### Raw run archive

Keep the existing per-run archive:

- `artifacts/predictions/<archive_id>/prediction.json`
- `artifacts/predictions/<archive_id>/top_candidates.csv`

This remains useful for audit, reproducibility, and debugging.

### Canonical website archive

Add normalized website-facing artifacts:

- `artifacts/predictions/daily/YYYY-MM-DD.json`
- `artifacts/predictions/index.json`
- `artifacts/predictions/latest.json`

These files become the source of truth for the website.

## Canonical Daily Schema

Each `daily/YYYY-MM-DD.json` file should contain:

- `as_of_date`
- `archive_id`
- `prediction_name`
- `model_created_at`
- `entry_date`
- `exit_date`
- `status`
- `universe_size`
- `selected`
- `top_candidates`
- `validation`
- `summary`
- `updated_at`

Key points:

- `status` is `pending` or `validated`
- `selected` is a compact object for the chosen stock
- `validation` contains the full realized validation payload when available
- `summary` duplicates the small set of fields needed by list pages, so the frontend does not need to parse nested objects

## Validation Semantics

Validation should operate on canonical daily records rather than raw run archives.

That means:

- one validation result per `as_of_date`
- same-day reruns overwrite the canonical record before validation
- validation history should reflect the canonical daily record, not every raw rerun

For compatibility, `artifacts/validation/history.csv` is still maintained, but its logical granularity becomes one row per official day-level prediction.

## Legacy Compatibility

Some existing predictions may only exist as raw run archives. The new code should be able to backfill canonical daily files from legacy per-run archives.

Rule:

- group legacy raw prediction payloads by `as_of_date`
- choose the latest `archive_id` for that date as the canonical daily record
- materialize `daily/YYYY-MM-DD.json` from that record if it does not already exist

This allows old predictions to appear on the website without a one-off migration script.

## Index Files

`index.json` should contain a lightweight list of records sorted by `as_of_date` descending. Each record should include:

- `as_of_date`
- `status`
- `selected_code`
- `selected_score`
- `entry_date`
- `exit_date`
- `archive_id`
- `hit`
- `alpha`
- `selected_return`
- `updated_at`

`latest.json` should be the most recent item from the same canonical set.

## Pipeline Changes

### `predict`

- write the existing raw run archive
- write or overwrite `daily/<as_of_date>.json`
- refresh `index.json`
- refresh `latest.json`

### `validate`

- ensure canonical daily files exist, including legacy backfill
- validate pending canonical daily files
- update the matching daily file in place
- refresh `index.json`
- refresh `latest.json`
- continue writing `history.csv`

## Why This Design

This design matches the product's true key: trading date.

It keeps auditability without forcing the website to understand execution-oriented storage. The frontend gets stable, direct-read artifacts, and the batch pipeline keeps enough detail to diagnose reruns or compare raw outputs if needed.
