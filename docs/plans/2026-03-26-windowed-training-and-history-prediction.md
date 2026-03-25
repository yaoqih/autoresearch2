# Windowed Training And History Prediction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit deployment training windows and a history prediction pipeline that can backfill validated results for recent dates while keeping existing default workflows compatible.

**Architecture:** Keep the existing single-day training, prediction, and validation primitives as the source of truth. Add thin date-window resolution helpers, a new batch prediction pipeline that loops over the current single-day predictor, and minimal CLI plumbing so both user workflows can be expressed with first-class commands instead of ad hoc scripts.

**Tech Stack:** Python, unittest, argparse, pandas, PyTorch, existing `quant_impl` pipeline code

---

### Task 1: Lock explicit deployment windows with failing tests

**Files:**
- Modify: `tests/test_train_pipeline.py`
- Reference: `tests/helpers.py`

**Step 1: Write the failing test**

```python
def test_train_pipeline_uses_explicit_deployment_date_window() -> None:
    result = train_pipeline(
        config,
        device="cpu",
        profile="full",
        force_prepare=True,
        deploy_only=True,
        deployment_start_date="2020-02-12",
        deployment_end_date="2020-03-10",
    )
    self.assertEqual(result["deployment_fit"]["train_start_date"], "2020-02-12")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py::TrainPipelineTest::test_train_pipeline_uses_explicit_deployment_date_window -q`
Expected: FAIL because `train_pipeline()` does not accept explicit deployment dates yet.

### Task 2: Lock anchor-date lookback training with a failing test

**Files:**
- Modify: `tests/test_train_pipeline.py`

**Step 1: Write the failing test**

```python
def test_train_pipeline_uses_anchor_date_lookback_window() -> None:
    result = train_pipeline(
        config,
        device="cpu",
        profile="full",
        force_prepare=True,
        deploy_only=True,
        deployment_anchor_date="2020-04-07",
        deployment_lookback_years=1,
    )
    self.assertEqual(result["deployment_window"]["anchor_date"], "2020-04-07")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py::TrainPipelineTest::test_train_pipeline_uses_anchor_date_lookback_window -q`
Expected: FAIL because lookback-based deployment windows are not implemented yet.

### Task 3: Implement deployment window resolution

**Files:**
- Modify: `src/quant_impl/pipelines/train.py`
- Optionally modify: `src/quant_impl/data/market.py`

**Step 1: Add window parsing helpers**

```python
def resolve_day_range_indices(bundle, *, start_date=None, end_date=None, anchor_date=None, lookback_years=None):
    ...
```

**Step 2: Apply helper to deployment training**

```python
deployment_train_days = list(range(window_start, window_end))
deployment_valid_indices = []
```

**Step 3: Persist deployment window metadata**

```python
result["deployment_window"] = {...}
artifact["deployment_window"] = {...}
```

**Step 4: Run targeted tests**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py -q`
Expected: PASS

### Task 4: Lock history prediction with validation using a failing test

**Files:**
- Modify: `tests/test_prediction_validation.py`

**Step 1: Write the failing test**

```python
def test_predict_history_pipeline_backfills_validation_for_multiple_days() -> None:
    result = predict_history_pipeline(
        config,
        device="cpu",
        start_date=bundle["dates"][-6],
        end_date=bundle["dates"][-5],
        validate=True,
    )
    self.assertEqual(result["predicted"], 2)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_prediction_validation.py::PredictionValidationTest::test_predict_history_pipeline_backfills_validation_for_multiple_days -q`
Expected: FAIL because no history prediction pipeline exists yet.

### Task 5: Implement history prediction pipeline and CLI plumbing

**Files:**
- Create: `src/quant_impl/pipelines/predict_history.py`
- Modify: `src/quant_impl/pipelines/__init__.py`
- Modify: `src/quant_impl/cli.py`

**Step 1: Add batch predictor**

```python
for as_of_date in resolved_dates:
    predict_pipeline(config, device=device, as_of_date=as_of_date, limit_stocks=limit_stocks)
```

**Step 2: Optionally validate after prediction**

```python
validation_result = validate_pipeline(config) if validate else None
```

**Step 3: Return compact summary**

```python
return {"predicted": len(predictions), "dates": resolved_dates, "validation": validation_result}
```

**Step 4: Wire CLI**

```python
predict_history = subparsers.add_parser("predict-history", ...)
```

**Step 5: Run targeted tests**

Run: `PYTHONPATH=src python -m pytest tests/test_prediction_validation.py -q`
Expected: PASS

### Task 6: Update README examples and command docs

**Files:**
- Modify: `README.md`

**Step 1: Document new train window flags**

```markdown
train --deploy-only --deployment-start-date 2021-01-01 --deployment-end-date 2025-12-31
```

**Step 2: Document new history prediction command**

```markdown
predict-history --anchor-date 2026-03-26 --lookback-months 3 --validate
```

**Step 3: Re-run focused regression suite**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py tests/test_prediction_validation.py tests/test_settings.py -q`
Expected: PASS
