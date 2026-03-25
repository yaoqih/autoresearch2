# Top10 Validation Fallback And History Charts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Top10 fallback-aware validation backfill and redesign the prediction website so users can inspect each candidate's realized return plus daily-return and NAV charts.

**Architecture:** Extend validation so the canonical archive becomes the single source of truth for executed-result summaries and candidate-level realized outcomes. Then update the Astro site to read the richer archive, compute history chart series server-side, and render inline SVG charts plus denser detail views.

**Tech Stack:** Python, unittest, pandas, PyTorch, Astro, Node test runner, inline SVG

---

### Task 1: Lock fallback validation behavior with failing tests

**Files:**
- Modify: `tests/test_prediction_validation.py`
- Reference: `tests/helpers.py`

**Step 1: Write the failing test for fallback execution**

Add a test that creates a prediction with at least two candidates, patches the realized map so Rank 1 has `open_limit_day1=True`, Rank 2 is tradeable, and asserts:

```python
self.assertEqual(validated_daily["validation"]["executed_rank"], 2)
self.assertEqual(validated_daily["validation"]["executed_code"], validated_daily["top_candidates"][1]["code"])
self.assertAlmostEqual(validated_daily["validation"]["selected_return"], expected_rank2_return, places=8)
```

**Step 2: Run the targeted test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_prediction_validation.PredictionValidationTest.test_validate_pipeline_falls_back_to_next_tradeable_top_candidate -v`
Expected: FAIL because validation currently always uses model Top1.

**Step 3: Write the failing test for all-top10-blocked behavior**

Add a test that patches all Top candidates to `open_limit_day1=True` and asserts:

```python
self.assertEqual(validated_daily["validation"]["all_top10_blocked"], 1)
self.assertEqual(validated_daily["validation"]["selected_return"], 0.0)
self.assertIsNone(validated_daily["validation"]["executed_code"])
```

**Step 4: Run the targeted test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_prediction_validation.PredictionValidationTest.test_validate_pipeline_sets_zero_return_when_top10_all_blocked -v`
Expected: FAIL because the zero-return fallback is not implemented.

### Task 2: Lock candidate-level archive backfill with a failing test

**Files:**
- Modify: `tests/test_prediction_validation.py`

**Step 1: Write the failing archive-structure test**

Assert that after validation:

```python
candidate = validated_daily["top_candidates"][0]
self.assertIn("validation", candidate)
self.assertIn("strict_open_return", candidate["validation"])
self.assertIn("tradeable", candidate["validation"])
self.assertIn("executed", candidate["validation"])
```

Also assert `history.csv` contains `executed_code` and `executed_rank`.

**Step 2: Run the targeted test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_prediction_validation.PredictionValidationTest.test_validation_backfills_candidate_level_returns_into_archive -v`
Expected: FAIL because candidate-level validation is not written yet.

### Task 3: Implement fallback-aware validation

**Files:**
- Modify: `src/quant_impl/pipelines/validate.py`
- Modify: `src/quant_impl/pipelines/prediction_archive.py`

**Step 1: Add helper(s) to enrich Top candidates**

Create a small helper that:

- walks `payload["top_candidates"][:10]`
- merges realized detail fields onto each candidate
- marks the first tradeable candidate as executed
- returns the enriched list plus executed summary

**Step 2: Switch top-level validation to executed-result semantics**

Populate:

```python
row["executed_code"] = ...
row["executed_rank"] = ...
row["fallback_applied"] = ...
row["all_top10_blocked"] = ...
row["schema_version"] = 2
```

and use executed return for `selected_return`, `alpha`, and `hit`.

**Step 3: Ensure validated archives are rewritten when schema changes**

Update the equality guard so previously validated payloads with missing candidate validation or mismatched `schema_version` are rewritten.

**Step 4: Run targeted Python tests**

Run: `PYTHONPATH=src python -m unittest tests.test_prediction_validation -v`
Expected: PASS

### Task 4: Extend website data utilities with chart series and executed summaries

**Files:**
- Modify: `web/src/lib/predictions.mjs`
- Modify: `web/tests/predictions.test.mjs`

**Step 1: Write failing web tests**

Add tests for:

- computing daily-return series from history
- computing NAV series with first point fixed at `1`
- preferring `executed_code` when present

**Step 2: Run the web test command to verify failure**

Run: `npm --prefix web test`
Expected: FAIL because chart helpers and executed-summary helpers do not exist yet.

**Step 3: Implement minimal helpers**

Add utilities such as:

```js
export function buildHistorySeries(index) { ... }
export function buildNavSeries(index) { ... }
export function displayExecutedCode(item) { ... }
```

**Step 4: Re-run web tests**

Run: `npm --prefix web test`
Expected: PASS

### Task 5: Redesign history and detail pages

**Files:**
- Modify: `web/src/pages/history.astro`
- Modify: `web/src/pages/predictions/[date].astro`
- Modify: `web/src/pages/index.astro`
- Modify: `web/src/styles/global.css`

**Step 1: Update history page to render charts and executed-result cards**

Use the new helpers to render:

- daily-return SVG card
- NAV SVG card
- updated history cards with fallback badge / executed-code-first copy

**Step 2: Update prediction detail page**

Replace the simple Top list with a table-like candidate panel showing realized outcomes and executed marker.

**Step 3: Update landing page**

Use executed summaries in the latest card and recent-history cards.

**Step 4: Run web tests and build**

Run: `npm --prefix web test`
Expected: PASS

Run: `npm --prefix web build`
Expected: PASS

### Task 6: Run focused regression verification

**Files:**
- No code changes expected

**Step 1: Run focused Python regressions**

Run: `PYTHONPATH=src python -m unittest tests.test_prediction_validation tests.test_train_pipeline tests.test_settings -v`
Expected: PASS

**Step 2: Run web verification**

Run: `npm --prefix web test`
Expected: PASS

Run: `npm --prefix web build`
Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans/2026-03-26-top10-validation-fallback-and-history-charts-design.md \
        docs/plans/2026-03-26-top10-validation-fallback-and-history-charts.md \
        src/quant_impl/pipelines/validate.py \
        src/quant_impl/pipelines/prediction_archive.py \
        tests/test_prediction_validation.py \
        web/src/lib/predictions.mjs \
        web/src/pages/history.astro \
        web/src/pages/index.astro \
        web/src/pages/predictions/[date].astro \
        web/src/styles/global.css \
        web/tests/predictions.test.mjs
git commit -m "feat: add fallback-aware validation insights to prediction site"
```
