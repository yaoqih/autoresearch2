#!/usr/bin/env node

import fs from "node:fs";
import puppeteer from "puppeteer-core";

const DEFAULT_TIMEOUT_MS = 15000;
const TARGET_URL = "https://quote.eastmoney.com/";
const REQUIRED_COOKIES = ["nid18", "nid18_create_time"];
const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36";

function parseArgs(argv) {
  const parsed = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      throw new Error(`Unexpected argument: ${token}`);
    }
    const key = token.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed.set(key, true);
      continue;
    }
    parsed.set(key, next);
    index += 1;
  }
  return parsed;
}

function toPositiveInt(value, fallbackValue) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallbackValue;
  }
  return Math.trunc(numeric);
}

function detectBrowserPath() {
  const candidates = [
    process.env.PUPPETEER_EXECUTABLE_PATH,
    process.env.EASTMONEY_BROWSER_PATH,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? null;
}

async function blockNonEssentialAssets(page) {
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const type = request.resourceType();
    if (["image", "font", "media", "manifest", "texttrack", "object"].includes(type)) {
      request.abort().catch(() => {});
      return;
    }
    request.continue().catch(() => {});
  });
}

async function warmCookies({ browserPath, proxy, timeoutMs }) {
  const launchArgs = [
    "--no-first-run",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
  ];
  if (proxy) {
    launchArgs.push(`--proxy-server=${proxy}`);
  }
  if (process.platform === "linux") {
    launchArgs.push("--no-sandbox", "--disable-setuid-sandbox");
  }

  const browser = await puppeteer.launch({
    executablePath: browserPath,
    headless: "new",
    args: launchArgs,
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });
    await page.setUserAgent(USER_AGENT);
    await blockNonEssentialAssets(page);
    await page.goto(TARGET_URL, { waitUntil: "domcontentloaded", timeout: timeoutMs });
    await page.waitForFunction(
      (requiredCookies) =>
        requiredCookies.every((name) => document.cookie.includes(`${name}=`)),
      { timeout: timeoutMs },
      REQUIRED_COOKIES,
    );
    const cookies = await page.cookies(TARGET_URL, "https://push2his.eastmoney.com/");
    const cookieMap = new Map(cookies.map((entry) => [entry.name, entry.value]));
    const missing = REQUIRED_COOKIES.filter((name) => !cookieMap.get(name));
    if (missing.length > 0) {
      throw new Error(`Missing required cookies: ${missing.join(", ")}`);
    }
    return {
      nid18: cookieMap.get("nid18"),
      nid18_create_time: cookieMap.get("nid18_create_time"),
      fetched_at: new Date().toISOString(),
      browser_path: browserPath,
    };
  } finally {
    await browser.close();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const timeoutMs = toPositiveInt(args.get("timeout-ms"), DEFAULT_TIMEOUT_MS);
  const browserPath = args.get("browser-path") || detectBrowserPath();
  const proxy = args.get("proxy") || null;

  if (!browserPath) {
    throw new Error("No Chrome/Chromium executable found. Pass --browser-path or set EASTMONEY_BROWSER_PATH.");
  }

  const payload = await warmCookies({ browserPath, proxy, timeoutMs });
  process.stdout.write(`${JSON.stringify(payload)}\n`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
});
