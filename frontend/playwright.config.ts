import fs from "fs";
import os from "os";
import path from "path";
import { defineConfig } from "@playwright/test";

const PORT = Number(process.env.E2E_PORT || 8088);
const BASE_URL = process.env.E2E_BASE_URL || `http://127.0.0.1:${PORT}`;
const AUDIT_DIR = process.env.DATA_DIR || "/tmp/matrika_playwright_audit";
const AUDIT_DB = process.env.DB_PATH || `${AUDIT_DIR}/matrika.db`;

function resolveChromiumExecutable(): string | undefined {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  }
  const cacheDir = path.join(os.homedir(), "Library", "Caches", "ms-playwright");
  if (fs.existsSync(cacheDir)) {
    const entries = fs.readdirSync(cacheDir).sort().reverse();
    for (const entry of entries) {
      if (entry.startsWith("chromium_headless_shell-")) {
        const candidate = path.join(
          cacheDir,
          entry,
          "chrome-headless-shell-mac-arm64",
          "chrome-headless-shell"
        );
        if (fs.existsSync(candidate)) return candidate;
      }
      if (entry.startsWith("chromium-")) {
        const candidate = path.join(
          cacheDir,
          entry,
          "chrome-mac-arm64",
          "Google Chrome for Testing.app",
          "Contents",
          "MacOS",
          "Google Chrome for Testing"
        );
        if (fs.existsSync(candidate)) return candidate;
      }
    }
  }
  const systemChrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  if (fs.existsSync(systemChrome)) return systemChrome;
  return undefined;
}

const executablePath = resolveChromiumExecutable();

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
    headless: true,
    launchOptions: executablePath ? { executablePath } : undefined,
  },
  webServer: {
    command: `cd .. && SQLITE_JOURNAL_MODE=DELETE DATA_DIR=${AUDIT_DIR} DB_PATH=${AUDIT_DB} AI_BASE_URL=http://localhost:1234/v1 AI_MODELS_URL=http://localhost:1234/api/v0/models FREELLM_BASE_URL=https://api.openai.com/v1 ./.venv/bin/python -m backend.tests.seed_e2e_db && SQLITE_JOURNAL_MODE=DELETE DATA_DIR=${AUDIT_DIR} DB_PATH=${AUDIT_DB} AI_BASE_URL=http://localhost:1234/v1 AI_MODELS_URL=http://localhost:1234/api/v0/models FREELLM_BASE_URL=https://api.openai.com/v1 ./.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port ${PORT}`,
    url: `${BASE_URL}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 45_000,
  },
});
