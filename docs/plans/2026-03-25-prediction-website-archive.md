# Prediction Website Archive Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a date-keyed prediction archive for the website while preserving raw run artifacts and validation history compatibility.

**Architecture:** `predict` continues to emit raw per-run artifacts, but also writes a canonical day-level JSON file. `validate` backfills canonical daily files from legacy raw archives, validates the canonical records, and refreshes lightweight list artifacts for the website.

**Tech Stack:** Python, pandas, pathlib, unittest, existing JSON/CSV file utilities

---

### Task 1: Add tests for canonical prediction artifacts

**Files:**
- Modify: `tests/test_prediction_validation.py`
- Test: `tests/test_prediction_validation.py`

**Step 1: Write the failing test**

Add a test that runs `predict_pipeline(...)` and expects:

- `artifacts/predictions/daily/<as_of_date>.json` exists
- `artifacts/predictions/index.json` exists
- `artifacts/predictions/latest.json` exists
- the daily JSON contains `selected`, `top_candidates`, and `status=pending`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction_validation.py -k canonical -v`

Expected: FAIL because canonical website artifacts are not written yet.

**Step 3: Write minimal implementation**

Implement helper code that writes canonical daily/index/latest JSON files during prediction.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction_validation.py -k canonical -v`

Expected: PASS

### Task 2: Add tests for validation refresh and history compatibility

**Files:**
- Modify: `tests/test_prediction_validation.py`
- Test: `tests/test_prediction_validation.py`

**Step 1: Write the failing test**

Extend coverage so that after `validate_pipeline(...)`:

- canonical daily JSON becomes `status=validated`
- canonical daily JSON contains `validation` and `summary`
- `index.json` reflects realized metrics
- `history.csv` still exists
- repeated `validate_pipeline(...)` does not append duplicates for the same day

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_prediction_validation.py -v`

Expected: FAIL because validation does not update canonical artifacts yet.

**Step 3: Write minimal implementation**

Update validation flow to validate daily canonical records and upsert history rows.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_prediction_validation.py -v`

Expected: PASS

### Task 3: Add path helpers and archive sync helpers

**Files:**
- Modify: `src/quant_impl/settings.py`
- Create or Modify: `src/quant_impl/pipelines/validate.py`
- Modify: `src/quant_impl/pipelines/predict.py`

**Step 1: Write the failing test**

Add a small settings test for the new canonical archive paths.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings.py -k prediction -v`

Expected: FAIL because the new path helpers do not exist.

**Step 3: Write minimal implementation**

Add config helpers for:

- daily archive dir
- per-day archive path
- prediction index path
- prediction latest path

Add helper code to:

- build canonical payloads
- load legacy raw prediction files
- backfill missing day-level files
- rebuild list artifacts

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_settings.py -k prediction -v`

Expected: PASS

### Task 4: Update docs

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

No automated doc test. Capture the behavior change in README after code is green.

**Step 2: Write minimal implementation**

Document:

- raw run archive
- canonical day-level archive
- website-facing artifacts
- validate behavior

**Step 3: Verify manually**

Read the updated README sections to confirm the paths and semantics match code.

### Task 5: Final verification

**Files:**
- Verify only

**Step 1: Run focused tests**

Run:

- `pytest tests/test_prediction_validation.py -v`
- `pytest tests/test_settings.py -v`

Expected: all pass

**Step 2: Run broader regression tests if needed**

Run: `pytest tests/test_train_pipeline.py -v`

Expected: no regression in adjacent behavior

**Step 3: Review git diff**

Run: `git diff --stat`

Expected: changes are limited to prediction/archive/validation/docs/tests
