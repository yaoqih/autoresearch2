# Prediction Website Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a read-only Astro SSR website that reads canonical prediction JSON files directly from the local filesystem on the server.

**Architecture:** Create a standalone `web/` Astro project with a small server-side data access layer, three SSR routes, and a shared layout. The site reads `latest.json`, `index.json`, and per-day canonical JSON files directly from `artifacts/predictions`.

**Tech Stack:** Astro, @astrojs/node, Node filesystem APIs, CSS variables, built-in `node:test`

---

### Task 1: Add failing tests for the data access layer

**Files:**
- Create: `web/tests/predictions.test.mjs`
- Test: `web/tests/predictions.test.mjs`

**Step 1: Write the failing test**

Add tests covering:

- loading `latest.json`
- loading `index.json`
- loading `daily/YYYY-MM-DD.json`
- graceful empty behavior when files are missing

**Step 2: Run test to verify it fails**

Run: `node --test web/tests/predictions.test.mjs`

Expected: FAIL because the data loader module does not exist yet.

**Step 3: Write minimal implementation**

Create the filesystem-backed loader used by the Astro pages.

**Step 4: Run test to verify it passes**

Run: `node --test web/tests/predictions.test.mjs`

Expected: PASS

### Task 2: Scaffold the Astro SSR app

**Files:**
- Create: `web/package.json`
- Create: `web/astro.config.mjs`
- Create: `web/src/pages/index.astro`
- Create: `web/src/pages/history.astro`
- Create: `web/src/pages/predictions/[date].astro`
- Create: shared layout and style files under `web/src/`

**Step 1: Write the failing test**

No isolated unit test. Build verification will be the gate.

**Step 2: Write minimal implementation**

Set up:

- Astro with Node standalone adapter
- server-side routes
- shared layout
- homepage, history page, and date-detail page

**Step 3: Run build to verify it succeeds**

Run: `npm --prefix web run build`

Expected: build succeeds and emits a standalone server in `web/dist/server/entry.mjs`

### Task 3: Add deployment and dev scripts

**Files:**
- Modify: `package.json`
- Modify: `README.md`

**Step 1: Write minimal implementation**

Add convenience scripts from the repo root for:

- install website deps
- dev server
- build
- preview/start

Document environment variable `QUANT_PREDICTIONS_DIR`.

**Step 2: Verify**

Run the documented commands or inspect them for correctness after the site build passes.

### Task 4: Final verification

**Files:**
- Verify only

**Step 1: Run data-layer tests**

Run: `node --test web/tests/predictions.test.mjs`

Expected: PASS

**Step 2: Run site build**

Run: `npm --prefix web run build`

Expected: PASS

**Step 3: Run Python regression tests affected by archive changes**

Run: `pytest tests/test_settings.py tests/test_prediction_validation.py tests/test_train_pipeline.py -v`

Expected: PASS

**Step 4: Review diff**

Run: `git diff --stat`

Expected: changes cover web app files, docs, and any root script wiring only.
