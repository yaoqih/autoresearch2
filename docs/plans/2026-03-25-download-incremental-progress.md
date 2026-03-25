# Download Incremental Resume And Progress Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change the downloader to resume from each local parquet's next trading day instead of re-downloading the full symbol, and add a zero-dependency console progress bar.

**Architecture:** The downloader will plan work per symbol using local parquet state plus the trading calendar. Missing files start from the global start date, existing files resume from the next trading session after their latest local row, and completed files are skipped. Downloaded tail data is merged back into the existing parquet by date and written atomically through the current store path. Progress rendering stays separate from logging and only affects interactive console output.

**Tech Stack:** Python 3.10, pandas, pyarrow parquet IO, requests, unittest.

---

### Task 1: Lock Incremental Resume Behavior With Tests

**Files:**
- Modify: `tests/test_download_stock.py`
- Test: `tests/test_download_stock.py`

**Step 1: Write the failing tests**

Add tests covering:
- existing parquet resumes from the next trading session
- existing parquet at or beyond required end date is skipped
- incremental fetch merges old and new rows by date
- progress renderer emits a useful progress line

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock`
Expected: FAIL because incremental planning and progress classes do not exist yet.

### Task 2: Implement Per-Symbol Incremental Planning

**Files:**
- Modify: `download_stock.py`
- Test: `tests/test_download_stock.py`

**Step 1: Add planning primitives**

Implement:
- per-symbol local latest-date lookup in `OutputTracker`
- trading-calendar helper to get the next session after a date
- a small download plan object carrying `instrument` and `request_start_date`

**Step 2: Use the plan in crawler execution**

Update `StockCrawler` so it:
- skips symbols whose local latest date already covers the required end date
- requests only the missing tail for stale symbols
- preserves `--force` full redownload behavior

**Step 3: Merge fetched tail data into existing parquet**

Load existing parquet when resuming, append new rows, de-duplicate by `date`, sort, and overwrite the file.

**Step 4: Run tests**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock`
Expected: incremental resume tests pass.

### Task 3: Add Console Progress Output

**Files:**
- Modify: `download_stock.py`
- Test: `tests/test_download_stock.py`

**Step 1: Add a lightweight progress renderer**

Implement a zero-dependency progress helper that writes a single updating line to stdout only when console output is enabled and interactive.

**Step 2: Integrate progress updates**

Update `StockCrawler.run()` to advance progress after each completed symbol and close the progress line cleanly on exit.

**Step 3: Run targeted tests**

Run: `PYTHONPATH=src python -m unittest tests.test_download_stock`
Expected: progress test passes and no existing downloader tests regress.

### Task 4: Verify End To End

**Files:**
- Modify: `download_stock.py`
- Modify: `README.md` if behavior docs need sync

**Step 1: Run syntax and focused verification**

Run:
- `PYTHONPATH=src python -m unittest tests.test_download_stock tests.test_settings tests.test_logging`
- `python -m py_compile download_stock.py src/quant_impl/pipelines/daily.py src/quant_impl/settings.py`

Expected:
- all targeted tests pass
- compile check exits 0

**Step 2: Summarize runtime behavior**

Report:
- resume now uses local max date plus next trading session
- missing files still start from the configured global start date
- progress bar is console-only and log-safe
