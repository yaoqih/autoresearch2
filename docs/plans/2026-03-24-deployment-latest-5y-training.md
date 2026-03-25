# Deployment Latest 5Y Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep walk-forward cross-validation unchanged while switching the final deployment model to the latest rolling training window, a fixed 5 epochs, and no early stopping.

**Architecture:** Reuse the existing walk-forward evaluation path as-is. Add deployment-only training overrides in the training pipeline so the saved model is trained on the latest rolling window instead of nearly all history, and persist deployment window metadata plus deployment-specific training settings for traceability.

**Tech Stack:** Python, unittest, PyTorch, existing `quant_impl` pipeline code

---

### Task 1: Lock deployment behavior with a failing test

**Files:**
- Modify: `tests/test_train_pipeline.py`
- Reference: `tests/helpers.py`

**Step 1: Write the failing test**

```python
def test_train_pipeline_uses_latest_rolling_window_for_deployment() -> None:
    ...
    self.assertEqual(result["deployment_fit"]["train_start_index"], expected_train_start)
    self.assertEqual(len(result["deployment_fit"]["history"]), 5)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py -q`
Expected: FAIL because deployment training still starts from index `0` and does not run 5 epochs.

### Task 2: Implement deployment-only overrides

**Files:**
- Modify: `src/quant_impl/pipelines/train.py`

**Step 1: Add minimal implementation**

```python
deployment_train_start = max(0, deployment_valid_start - runtime_config.data.train_days)
deployment_train_days = list(range(deployment_train_start, deployment_valid_start))
deployment_model, deployment_fit = fit_one_window(..., epochs_override=5, enable_early_stopping=False)
```

**Step 2: Persist metadata**

```python
fit_summary["train_start_index"] = train_day_indices[0]
artifact["deployment_training_config"] = {...}
```

**Step 3: Run targeted tests**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py tests/test_champion_spec.py -q`
Expected: PASS

### Task 3: Update user-facing docs

**Files:**
- Modify: `README.md`

**Step 1: Document deployment behavior**

```markdown
- walk-forward evaluation stays `5y train + 1y valid + 1y holdout`
- final deployment model retrains on the latest rolling 5y window with fixed 5 epochs and no early stopping
```

**Step 2: Re-run relevant tests**

Run: `PYTHONPATH=src python -m pytest tests/test_train_pipeline.py tests/test_champion_spec.py -q`
Expected: PASS
