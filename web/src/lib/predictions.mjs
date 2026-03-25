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
