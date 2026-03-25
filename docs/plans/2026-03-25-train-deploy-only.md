# Train Deploy-Only Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit `train --deploy-only` mode that skips walk-forward cross-validation and trains only the final deployment model using the latest 5-year window with no validation split.

**Architecture:** The CLI will keep exposing the boolean `--deploy-only` flag and pass it into `train_pipeline()`. Inside the pipeline, fold construction and CV summary generation stay bypassed in deploy-only mode, and the deployment training branch switches to a pure train-only latest-5-year window: `valid_day_indices=[]`, no best-valid checkpoint selection, and final weights from the last configured epoch. The result payload and saved artifact will continue to record deploy-only mode.

**Tech Stack:** Python 3.12, argparse, PyTorch, unittest.

---

### Task 1: Lock Deploy-Only Behavior With Tests

**Files:**
- Modify: `tests/test_train_pipeline.py`

**Step 1: Write the failing tests**

Add tests covering:
- `train_pipeline(..., deploy_only=True)` still skips fold execution and still produces a deployment model
- deploy-only uses the latest 5-year train window with `valid_days == 0`
- deploy-only records `best_valid_metrics is None`
- deploy-only uses the final epoch weights rather than best-valid checkpoint selection

**Step 2: Run test to verify it fails**

Run:
- `PYTHONPATH=src python -m unittest tests.test_train_pipeline`

Expected:
- FAIL because deploy-only still reserves a validation split and still reloads the best-valid checkpoint.

### Task 2: Implement CLI And Pipeline Support

**Files:**
- Modify: `src/quant_impl/pipelines/train.py`
- Modify: `README.md`

**Step 1: Change deploy-only training semantics**

Update `train_pipeline()` to:
- keep default CV behavior unchanged
- in deploy-only mode, build `deployment_train_days` from the latest `train_days`
- in deploy-only mode, pass `valid_day_indices=[]`
- make `fit_one_window()` support a “use final epoch weights” path when no validation selection is desired
- persist the resulting zero-valid deployment fit summary in the result/artifact

**Step 2: Run focused tests**

Run:
- `PYTHONPATH=src python -m unittest tests.test_train_pipeline`

Expected:
- PASS

### Task 3: Verify Runtime Behavior

**Files:**
- Modify: `README.md` if usage docs need sync

**Step 1: Run syntax and smoke verification**

Run:
- `python -m py_compile src/quant_impl/pipelines/train.py`
- `PYTHONPATH=src python -m quant_impl.cli train --device cpu --profile screen --deploy-only --limit-stocks 50`

Expected:
- compile check exits 0
- training logs skip fold execution and go straight to deployment training with `valid_days=0`

**Step 2: Summarize**

Report:
- how to invoke deploy-only mode
- that default `train` behavior is unchanged
- that deploy-only now uses the latest 5-year train-only window and final epoch weights
