import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";

import {
  buildDailyReturnSeries,
  buildNavSeries,
  buildTonghuashunUrl,
  displayPrimaryCode,
  displayStockCode,
  loadLatestPrediction,
  loadPredictionDetail,
  loadPredictionIndex,
} from "../src/lib/predictions.mjs";

async function writeJson(filePath, payload) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(payload, null, 2), "utf8");
}

test("loads latest, history index, and daily detail from canonical archive files", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "quant-web-"));

  try {
    const predictionsDir = path.join(root, "predictions");
    const latest = {
      as_of_date: "2026-03-24",
      status: "pending",
      selected: { code: "SZ002309", score: 3.26 },
    };
    const index = [
      {
        as_of_date: "2026-03-24",
        status: "pending",
        selected_code: "SZ002309",
      },
    ];
    const detail = {
      as_of_date: "2026-03-24",
      status: "validated",
      selected: { code: "SZ002309", score: 3.26 },
      validation: { alpha: 0.0214, selected_return: 0.0341 },
      top_candidates: [{ rank: 1, code: "SZ002309", score: 3.26 }],
    };

    await writeJson(path.join(predictionsDir, "latest.json"), latest);
    await writeJson(path.join(predictionsDir, "index.json"), index);
    await writeJson(path.join(predictionsDir, "daily", "2026-03-24.json"), detail);

    const loadedLatest = await loadLatestPrediction(predictionsDir);
    const loadedIndex = await loadPredictionIndex(predictionsDir);
    const loadedDetail = await loadPredictionDetail("2026-03-24", predictionsDir);

    assert.deepEqual(loadedLatest, latest);
    assert.deepEqual(loadedIndex, index);
    assert.deepEqual(loadedDetail, detail);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("returns empty states when canonical prediction files are missing", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "quant-web-empty-"));

  try {
    const predictionsDir = path.join(root, "predictions");

    const loadedLatest = await loadLatestPrediction(predictionsDir);
    const loadedIndex = await loadPredictionIndex(predictionsDir);
    const loadedDetail = await loadPredictionDetail("2026-03-24", predictionsDir);

    assert.equal(loadedLatest, null);
    assert.deepEqual(loadedIndex, []);
    assert.equal(loadedDetail, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("treats empty latest.json as missing latest prediction", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "quant-web-empty-latest-"));

  try {
    const predictionsDir = path.join(root, "predictions");
    await writeJson(path.join(predictionsDir, "latest.json"), {});

    const loadedLatest = await loadLatestPrediction(predictionsDir);

    assert.equal(loadedLatest, null);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("falls back to cwd-adjacent artifacts directory when no explicit path is provided", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "quant-web-cwd-"));
  const originalCwd = process.cwd();

  try {
    const appDir = path.join(root, "app");
    const predictionsDir = path.join(root, "artifacts", "predictions");
    const latest = {
      as_of_date: "2026-03-24",
      status: "pending",
      selected: { code: "SZ002309", score: 3.26 },
    };

    await mkdir(appDir, { recursive: true });
    await writeJson(path.join(predictionsDir, "latest.json"), latest);
    process.chdir(appDir);

    const loadedLatest = await loadLatestPrediction();

    assert.deepEqual(loadedLatest, latest);
  } finally {
    process.chdir(originalCwd);
    await rm(root, { recursive: true, force: true });
  }
});

test("builds Tonghuashun links from exchange-prefixed stock codes", () => {
  assert.equal(buildTonghuashunUrl("SZ301363"), "https://stockpage.10jqka.com.cn/301363/");
  assert.equal(buildTonghuashunUrl("SH600758"), "https://stockpage.10jqka.com.cn/600758/");
  assert.equal(displayStockCode("SZ301363"), "301363");
  assert.equal(displayStockCode("SH600758"), "600758");
});

test("prefers executed code and builds chronological performance series", () => {
  const history = [
    {
      as_of_date: "2026-03-24",
      status: "validated",
      selected_return: 0.02,
      selected_code: "SZ000001",
      executed_code: "SZ000002",
    },
    {
      as_of_date: "2026-03-21",
      status: "validated",
      selected_return: -0.01,
      selected_code: "SZ000003",
    },
    {
      as_of_date: "2026-03-25",
      status: "pending",
      selected_return: null,
      selected_code: "SZ000004",
    },
  ];

  const dailySeries = buildDailyReturnSeries(history);
  const navSeries = buildNavSeries(history);

  assert.equal(displayPrimaryCode(history[0]), "SZ000002");
  assert.equal(displayPrimaryCode(history[1]), "SZ000003");
  assert.deepEqual(
    dailySeries.map((item) => [item.date, item.value]),
    [
      ["2026-03-21", -0.01],
      ["2026-03-24", 0.02],
    ],
  );
  assert.deepEqual(
    navSeries.map((item) => [item.date, item.value]),
    [
      ["2026-03-21", 1],
      ["2026-03-24", 1.02],
    ],
  );
});
