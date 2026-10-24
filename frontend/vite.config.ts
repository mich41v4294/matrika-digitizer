import { execSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

function resolveGitCommitFromDir(startDir: string): string | null {
  let current = resolve(startDir);
  for (let depth = 0; depth < 4; depth++) {
    const dotGit = join(current, ".git");
    if (existsSync(dotGit)) {
      try {
        let gitDir = dotGit;
        let commonDir = dotGit;
        const st = statSync(dotGit);
        if (st.isFile()) {
          const content = readFileSync(dotGit, "utf-8").trim();
          if (content.startsWith("gitdir:")) {
            const relGitDir = content.slice("gitdir:".length).trim();
            gitDir = resolve(current, relGitDir);
            commonDir = gitDir;
            const commonDirFile = join(gitDir, "commondir");
            if (existsSync(commonDirFile)) {
              const relCommon = readFileSync(commonDirFile, "utf-8").trim();
              if (relCommon) {
                commonDir = resolve(gitDir, relCommon);
              }
            }
          }
        }
        const headFile = join(gitDir, "HEAD");
        if (existsSync(headFile)) {
          const head = readFileSync(headFile, "utf-8").trim();
          if (!head.startsWith("ref:")) {
            return head.slice(0, 7);
          }
          const refPath = head.slice("ref:".length).trim();
          for (const base of [gitDir, commonDir]) {
            const looseRef = join(base, refPath);
            if (existsSync(looseRef)) {
              const sha = readFileSync(looseRef, "utf-8").trim();
              if (sha) return sha.slice(0, 7);
            }
          }
          for (const base of [commonDir, gitDir]) {
            const packedRefs = join(base, "packed-refs");
            if (existsSync(packedRefs)) {
              const lines = readFileSync(packedRefs, "utf-8").split(/\r?\n/);
              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("^")) continue;
                const [sha, ref] = trimmed.split(/\s+/);
                if (ref === refPath && sha) {
                  return sha.slice(0, 7);
                }
              }
            }
          }
        }
      } catch {
        // ignore and continue fallback
      }
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

function getBuildCommitHash(): string {
  const proc = (
    globalThis as {
      process?: { env?: Record<string, string | undefined>; cwd?: () => string };
    }
  ).process;
  const env = proc?.env || {};
  for (const key of [
    "VITE_COMMIT_HASH",
    "BUILD_COMMIT_HASH",
    "GIT_COMMIT",
    "COMMIT_SHA",
    "GITHUB_SHA",
  ]) {
    const val = (env[key] || "").trim();
    if (val && val !== "dev" && val !== "unknown") {
      return /^[0-9a-f]{8,40}$/i.test(val) ? val.slice(0, 7) : val;
    }
  }

  const cwd = proc?.cwd ? proc.cwd() : ".";
  try {
    const out = execSync("git rev-parse --short=7 HEAD", {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      cwd,
    }).trim();
    if (out) return out;
  } catch {
    // git CLI unavailable or not a git working tree; fall back to reading .git/HEAD
  }

  return resolveGitCommitFromDir(cwd) || "dev";
}

const buildCommitHash = getBuildCommitHash();

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __BUILD_COMMIT_HASH__: JSON.stringify(buildCommitHash),
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor": ["react", "react-dom", "react-router-dom"],
          "icons-vendor": ["lucide-react"],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
      "/media": "http://localhost:8080",
      "/ws": { target: "ws://localhost:8080", ws: true },
    },
  },
});
