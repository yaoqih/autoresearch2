# Environment Proxy Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a config-controlled switch that decides whether the downloader inherits environment proxy variables.

**Architecture:** Extend `DownloadSettings` with a `use_env_proxy` boolean, thread it through the daily pipeline into `download_stock.py`, and let `HttpClient` map it onto `requests.Session().trust_env`. Keep KDL proxy-pool behavior unchanged.

**Tech Stack:** Python, `requests`, `unittest`, YAML config loading

---

### Task 1: Lock behavior with tests

**Files:**
- Modify: `tests/test_download_stock.py`
- Modify: `tests/test_logging.py`

**Step 1: Write the failing tests**

Add tests that cover:
- `HttpClient(..., use_env_proxy=False)` sets `session.trust_env` to `False`
- `load_config()` reads `download.use_env_proxy`
- `run_download_step()` forwards the flag to `download_stock.py`

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock tests.test_logging`

Expected: FAIL because the new field/argument does not exist yet.

### Task 2: Thread the config through the downloader

**Files:**
- Modify: `src/quant_impl/settings.py`
- Modify: `src/quant_impl/pipelines/daily.py`
- Modify: `download_stock.py`
- Modify: `configs/default.yaml`

**Step 1: Write minimal implementation**

Make these changes:
- add `DownloadSettings.use_env_proxy`
- add `use_env_proxy` to the default YAML
- pass `--use-env-proxy` or `--no-use-env-proxy` from `run_download_step`
- add matching CLI flags in `download_stock.py`
- set `requests.Session().trust_env` from the CLI flag

**Step 2: Run targeted tests**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock tests.test_logging`

Expected: PASS

### Task 3: Verify final behavior

**Files:**
- None

**Step 1: Run focused regression checks**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock tests.test_logging`

Expected: PASS with the new default config path still working.
