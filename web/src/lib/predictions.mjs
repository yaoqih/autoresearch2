import path from "node:path";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";

function discoverPredictionsDir() {
  const cwd = process.cwd();
  const candidates = [
    path.resolve(cwd, "artifacts/predictions"),
    path.resolve(cwd, "../artifacts/predictions"),
    path.resolve(cwd, "../../artifacts/predictions"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[0];
}

function resolvePredictionsDir(predictionsDir) {
  if (predictionsDir) {
    return path.resolve(predictionsDir);
  }
  if (process.env.QUANT_PREDICTIONS_DIR) {
    return path.resolve(process.env.QUANT_PREDICTIONS_DIR);
  }
  return discoverPredictionsDir();
}

async function readJsonFile(filePath) {
  try {
    const raw = await readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

export async function loadPredictionIndex(predictionsDir) {
  const root = resolvePredictionsDir(predictionsDir);
  const payload = await readJsonFile(path.join(root, "index.json"));
  return Array.isArray(payload) ? payload : [];
}

export async function loadPredictionDetail(date, predictionsDir) {
  const root = resolvePredictionsDir(predictionsDir);
  return readJsonFile(path.join(root, "daily", `${date}.json`));
}

export async function loadLatestPrediction(predictionsDir) {
  const root = resolvePredictionsDir(predictionsDir);
  const latest = await readJsonFile(path.join(root, "latest.json"));
  if (latest && latest.as_of_date) {
    return latest;
  }
  const index = await loadPredictionIndex(root);
  if (!index.length || !index[0]?.as_of_date) {
    return null;
  }
  return loadPredictionDetail(index[0].as_of_date, root);
}

export function predictionDetailPath(date) {
  return `/predictions/${date}`;
}

export function displayPrimaryCode(item) {
  if (!item || typeof item !== "object") {
    return "";
  }
  const executedCode = item.validation?.executed_code ?? item.executed_code;
  if (executedCode) {
    return String(executedCode);
  }
  const selectedCode = item.selected?.code ?? item.selected_code;
  return selectedCode ? String(selectedCode) : "";
}

export function executionSummary(item) {
  if (!item || typeof item !== "object") {
    return {
      modelCode: null,
      executedCode: null,
      executedRank: null,
      fallbackApplied: false,
      fallbackWindowSize: 10,
      allFallbackBlocked: false,
      allTop10Blocked: false,
    };
  }
  const validation = item.validation ?? item;
  const fallbackWindowSize = Number(validation.fallback_window_size ?? item.fallback_window_size ?? 10);
  const allFallbackBlocked = Boolean(validation.all_fallback_blocked ?? item.all_fallback_blocked);
  const rawAllTop10Blocked = validation.all_top10_blocked ?? item.all_top10_blocked;
  return {
    modelCode: item.selected?.code ?? item.selected_code ?? null,
    executedCode: validation.executed_code ?? item.executed_code ?? null,
    executedRank: validation.executed_rank ?? item.executed_rank ?? null,
    fallbackApplied: Boolean(validation.fallback_applied ?? item.fallback_applied),
    fallbackWindowSize,
    allFallbackBlocked,
    allTop10Blocked: rawAllTop10Blocked === undefined ? allFallbackBlocked : Boolean(rawAllTop10Blocked),
  };
}

export function displayStockCode(code) {
  if (!code) {
    return "";
  }
  return String(code).replace(/^(SZ|SH|BJ)/i, "");
}

export function buildTonghuashunUrl(code) {
  const display = displayStockCode(code);
  if (!display) {
    return "https://stockpage.10jqka.com.cn/";
  }
  return `https://stockpage.10jqka.com.cn/${display}/`;
}

export function formatDisplayDate(dateText) {
  if (!dateText) {
    return "N/A";
  }
  const parsed = new Date(`${dateText}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) {
    return dateText;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "long",
    day: "numeric",
    weekday: "short",
    timeZone: "Asia/Shanghai",
  }).format(parsed);
}

export function formatSignedPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "待验证";
  }
  const numeric = Number(value) * 100;
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(2)}%`;
}

export function statusLabel(status) {
  return status === "validated" ? "已验证" : "待验证";
}

export function buildDailyReturnSeries(index) {
  if (!Array.isArray(index)) {
    return [];
  }
  return index
    .filter((item) => item?.selected_return !== null && item?.selected_return !== undefined)
    .filter((item) => Number.isFinite(Number(item.selected_return)))
    .map((item) => ({
      date: String(item.as_of_date),
      value: Number(item.selected_return),
    }))
    .sort((left, right) => left.date.localeCompare(right.date));
}

export function buildNavSeries(index) {
  const dailySeries = buildDailyReturnSeries(index);
  let nav = 1;
  return dailySeries.map((item, indexValue) => {
    if (indexValue === 0) {
      return {
        date: item.date,
        value: nav,
      };
    }
    nav *= 1 + item.value;
    return {
      date: item.date,
      value: nav,
    };
  });
}
