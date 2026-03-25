# Juliang Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace KDL proxy handling with a Juliang-only single-lease proxy manager, keep cookie warmup optional and off by default, and classify proxy-layer versus target-layer failures.

**Architecture:** Thread Juliang settings from YAML into `download_stock.py`, replace the KDL pool with a shared Juliang lease manager, and let all worker threads reuse the active proxy until it expires or fails. Preserve the existing downloader/request flow and make cookie warmup a best-effort optional add-on rather than a hard dependency.

**Tech Stack:** Python, `requests`, `python-dotenv`, unittest, YAML config loading, Node.js cookie warmer

---

### Task 1: Lock config and CLI plumbing with tests

**Files:**
- Modify: `tests/test_settings.py`
- Modify: `tests/test_logging.py`

**Step 1: Write the failing tests**

Add tests that verify:

- `load_config()` reads Juliang fields from `download`
- default config disables `eastmoney_cookie_warmup`
- `run_download_step()` forwards Juliang CLI flags and secrets/plumbing fields
- `run_download_step()` no longer forwards KDL flags

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_settings tests.test_logging`

Expected: FAIL because the Juliang settings and CLI forwarding do not exist yet.

### Task 2: Lock downloader behavior with tests

**Files:**
- Modify: `tests/test_download_stock.py`

**Step 1: Write the failing tests**

Add tests that verify:

- Juliang sign generation is deterministic
- a Juliang lease manager reuses the same active proxy until refresh is required
- proxy metadata contains auth credentials in request format
- a proxy transport exception is classified as `proxy_transport`
- a target response with anti-bot markers is classified as `target_site`
- cookie warmup stays optional when disabled

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock`

Expected: FAIL because the Juliang lease manager and failure categorization do not exist yet.

### Task 3: Add Juliang configuration and daily pipeline plumbing

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/quant_impl/settings.py`
- Modify: `src/quant_impl/pipelines/daily.py`

**Step 1: Write minimal implementation**

Make these changes:

- add Juliang-specific settings to `DownloadSettings`
- set `eastmoney_cookie_warmup: false` in the default config
- add Juliang defaults to the default YAML
- forward Juliang args from `run_download_step()`
- remove KDL-specific argument forwarding

**Step 2: Run focused tests**

Run: `PYTHONPATH=src python -m unittest tests.test_settings tests.test_logging`

Expected: PASS

### Task 4: Replace KDL runtime logic with Juliang single-lease reuse

**Files:**
- Modify: `download_stock.py`

**Step 1: Write minimal implementation**

Make these changes:

- remove the KDL import and KDL-only config fields
- add a Juliang config dataclass
- add a Juliang API helper that signs and requests `dynamic/getips`
- add a shared lease manager that keeps one active proxy
- reuse one active proxy across threads until expiry or invalidation
- keep cookie warmup optional and able to reuse the active proxy
- classify failures into proxy transport vs target site
- update logging/detail strings to include the failure category

**Step 2: Run focused tests**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock`

Expected: PASS

### Task 5: Update documentation

**Files:**
- Modify: `README.md`

**Step 1: Write minimal documentation**

Document:

- Juliang-only environment variables
- single-proxy high-reuse behavior
- cookie warmup now optional and off by default
- how to interpret proxy transport vs target site failures

**Step 2: Run focused tests**

Run: `PYTHONPATH=src python -m unittest tests.test_settings tests.test_logging tests.test_download_stock`

Expected: PASS

### Task 6: Final verification

**Files:**
- None

**Step 1: Run fresh regression checks**

Run: `PYTHONPATH=src python -m unittest tests.test_settings tests.test_logging tests.test_download_stock`

Expected: PASS with no KDL references remaining in downloader config flow.
