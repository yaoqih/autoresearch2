# Eastmoney Cookie Warmer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a lightweight Eastmoney cookie warmer that uses a system Chrome/Chromium plus `puppeteer-core`, caches the minimum required cookies, and lets the Python downloader reuse them.

**Architecture:** Keep browser automation isolated in a small Node CLI so the Python downloader stays focused on HTTP downloads. The downloader reads a cached cookie header when available, invokes the warmer only when needed, and reuses the existing proxy/non-proxy request flow without changing proxy-pool scheduling.

**Tech Stack:** Python, Node.js, `puppeteer-core`, unittest, YAML config, README documentation.

---

### Task 1: Add failing tests for config and command wiring

**Files:**
- Modify: `tests/test_logging.py`
- Modify: `tests/test_settings.py`

**Step 1: Write the failing test**

Add tests that assert:
- `load_config()` reads cookie warmer settings from YAML.
- `run_download_step()` forwards cookie warmer arguments to `download_stock.py`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_logging tests.test_settings`

Expected: FAIL because the new settings and command flags do not exist yet.

**Step 3: Write minimal implementation**

Add the new download settings dataclass fields and wire them into the download pipeline command construction.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_logging tests.test_settings`

Expected: PASS

### Task 2: Add failing tests for cookie cache and request integration

**Files:**
- Modify: `tests/test_download_stock.py`
- Modify: `download_stock.py`

**Step 1: Write the failing test**

Add tests that assert:
- a cookie manager loads a fresh cached cookie header without spawning Node,
- a stale or missing cache triggers the warmer command,
- `HttpClient` injects the warmed cookie header into outgoing requests.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_download_stock`

Expected: FAIL because the cookie manager and integration do not exist yet.

**Step 3: Write minimal implementation**

Add:
- a cookie warmer config object,
- a cache-backed cookie manager,
- optional integration into `HttpClient`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_download_stock`

Expected: PASS

### Task 3: Add the lightweight Node warmer

**Files:**
- Create: `tools/eastmoney_cookie_warmer.mjs`
- Create: `package.json`

**Step 1: Write the failing test**

Use a syntax-level verification step for the new CLI script and keep Python tests responsible for the integration boundary.

**Step 2: Run test to verify it fails**

Run: `node --check tools/eastmoney_cookie_warmer.mjs`

Expected: FAIL before the script exists.

**Step 3: Write minimal implementation**

Implement a Node CLI that:
- launches system Chrome/Chromium via `puppeteer-core`,
- blocks non-essential assets,
- waits for `nid18` and `nid18_create_time`,
- prints compact JSON to stdout.

**Step 4: Run test to verify it passes**

Run: `node --check tools/eastmoney_cookie_warmer.mjs`

Expected: PASS

### Task 4: Update docs for install and usage

**Files:**
- Modify: `README.md`
- Modify: `configs/default.yaml`

**Step 1: Write the failing test**

Use manual verification for the documentation changes and config defaults.

**Step 2: Run test to verify it fails**

Inspect README and config for missing cookie warmer instructions and fields.

**Step 3: Write minimal implementation**

Document:
- how to install Chromium/Chrome on macOS, Ubuntu/Debian, and RHEL-family systems,
- how to configure and run the cookie warmer,
- how it behaves with direct connections and proxy setups.

**Step 4: Run test to verify it passes**

Re-read README and config for completeness and consistency with the implementation.

### Task 5: Final verification

**Files:**
- Verify modified files only

**Step 1: Run focused tests**

Run:
- `python -m unittest tests.test_settings tests.test_logging tests.test_download_stock`
- `node --check tools/eastmoney_cookie_warmer.mjs`

**Step 2: Run one integration smoke**

Run a mocked or non-networked smoke for the Python downloader boundary to confirm the warmer command is wired correctly.

**Step 3: Summarize residual risk**

Call out that live Eastmoney/browser verification still depends on local Chrome plus network availability.
