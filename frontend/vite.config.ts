import { execSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { get as httpsGet } from "node:https";
import { dirname, join, resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const GITHUB_REPO = "mich41v4294/matrika-digitizer";
const HEX_SHA_RE = /^[0-9a-f]{7,40}$/i;

function normalizeCommitHash(raw: string | null | undefined): string | null {
  const val = (raw || "").trim();
  if (!val || val === "dev" || val === "unknown" || val.includes("$Format:")) {
    return null;
  }
  return HEX_SHA_RE.test(val) ? val.slice(0, 7).toLowerCase() : val;
}

function extract40HexFromRefFile(filePath: string): string | null {
  if (!existsSync(filePath)) return null;
  try {
    const content = readFileSync(filePath, "utf-8").trim();
    const match = content.match(/\b([0-9a-f]{40})\b/i);
    if (match?.[1]) return match[1].slice(0, 7).toLowerCase();
  } catch {
    // ignore
  }
  return null;
}

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
            const direct = normalizeCommitHash(head);
            if (direct) return direct;
          } else {
            const refPath = head.slice("ref:".length).trim();
            const candidateRefs = [refPath];
            if (refPath.startsWith("refs/heads/")) {
              const branch = refPath.slice("refs/heads/".length);
              candidateRefs.push(`refs/remotes/origin/${branch}`);
            }
            for (const ref of candidateRefs) {
              for (const base of [gitDir, commonDir]) {
                const looseRef = join(base, ref);
                if (existsSync(looseRef)) {
                  const sha = normalizeCommitHash(readFileSync(looseRef, "utf-8"));
                  if (sha) return sha;
                }
              }
              for (const base of [commonDir, gitDir]) {
                const packedRefs = join(base, "packed-refs");
                if (existsSync(packedRefs)) {
                  const lines = readFileSync(packedRefs, "utf-8").split(/\r?\n/);
                  for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("^")) continue;
                    const [sha, r] = trimmed.split(/\s+/);
                    if (r === ref && sha) {
                      const norm = normalizeCommitHash(sha);
                      if (norm) return norm;
                    }
                  }
                }
              }
            }
          }
        }
        for (const base of [gitDir, commonDir]) {
          for (const fallbackFile of ["ORIG_HEAD", "FETCH_HEAD"]) {
            const sha = extract40HexFromRefFile(join(base, fallbackFile));
            if (sha) return sha;
          }
          const headLog = join(base, "logs", "HEAD");
          if (existsSync(headLog)) {
            const lines = readFileSync(headLog, "utf-8").trim().split(/\r?\n/);
            for (let i = lines.length - 1; i >= 0; i--) {
              const parts = lines[i].trim().split(/\s+/);
              if (parts.length >= 2 && HEX_SHA_RE.test(parts[1])) {
                return parts[1].slice(0, 7).toLowerCase();
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

function resolveArchiveOrCachedCommit(startDir: string): string | null {
  let current = resolve(startDir);
  for (let depth = 0; depth < 4; depth++) {
    for (const rel of [
      "_build_info.py",
      join("backend", "app", "_build_info.py"),
      ".build-commit",
    ]) {
      const candidate = join(current, rel);
      if (existsSync(candidate)) {
        try {
          const text = readFileSync(candidate, "utf-8").trim();
          const m = text.match(/ARCHIVE_COMMIT_HASH\s*=\s*["']([^"']+)["']/);
          if (m?.[1]) {
            const norm = normalizeCommitHash(m[1]);
            if (norm) return norm;
          } else {
            const norm = normalizeCommitHash(text);
            if (norm) return norm;
          }
        } catch {
          // ignore
        }
      }
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

function httpsGetText(url: string, headers: Record<string, string>, timeoutMs = 3000): Promise<string | null> {
  return new Promise((resolvePromise) => {
    try {
      const req = httpsGet(url, { headers, timeout: timeoutMs }, (res) => {
        if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
          resolvePromise(null);
          return;
        }
        res.setEncoding("utf-8");
        let body = "";
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          resolvePromise(body);
        });
        res.on("error", () => {
          resolvePromise(null);
        });
      });
      req.on("timeout", () => {
        req.destroy();
        resolvePromise(null);
      });
      req.on("error", () => {
        resolvePromise(null);
      });
    } catch {
      resolvePromise(null);
    }
  });
}

async function fetchRemoteGitHubCommitHash(): Promise<string | null> {
  const infoRefs = await httpsGetText(
    `https://github.com/${GITHUB_REPO}.git/info/refs?service=git-upload-pack`,
    { "User-Agent": "matrika-digitizer-build" }
  );
  if (infoRefs) {
    const mainMatch =
      infoRefs.match(/([0-9a-f]{40})\s+refs\/heads\/main\b/i) ||
      infoRefs.match(/([0-9a-f]{40})\s+HEAD\b/i);
    if (mainMatch?.[1]) {
      return mainMatch[1].slice(0, 7).toLowerCase();
    }
  }

  const rawSha = await httpsGetText(
    `https://api.github.com/repos/${GITHUB_REPO}/commits/main`,
    {
      Accept: "application/vnd.github.sha",
      "User-Agent": "matrika-digitizer-build",
    }
  );
  return normalizeCommitHash(rawSha);
}

async function getBuildCommitHash(): Promise<string> {
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
    "SOURCE_COMMIT",
    "COMMIT_SHA",
    "GITHUB_SHA",
    "CI_COMMIT_SHA",
    "COOLIFY_GIT_COMMIT_SHA",
    "PORTAINER_GIT_COMMIT",
    "RENDER_GIT_COMMIT",
    "RAILWAY_GIT_COMMIT_SHA",
    "VERCEL_GIT_COMMIT_SHA",
  ]) {
    const norm = normalizeCommitHash(env[key]);
    if (norm) return norm;
  }

  const cwd = proc?.cwd ? proc.cwd() : ".";
  try {
    const out = execSync("git rev-parse --short=7 HEAD", {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "ignore"],
      cwd,
    }).trim();
    const norm = normalizeCommitHash(out);
    if (norm) return norm;
  } catch {
    // git CLI unavailable or not a git working tree
  }

  const fromGitDir = resolveGitCommitFromDir(cwd);
  if (fromGitDir) return fromGitDir;

  const fromArchive = resolveArchiveOrCachedCommit(cwd);
  if (fromArchive) return fromArchive;

  const fromRemote = await fetchRemoteGitHubCommitHash();
  if (fromRemote) return fromRemote;

  return "dev";
}

export default defineConfig(async () => {
  const buildCommitHash = await getBuildCommitHash();

  return {
    plugins: [
      react(),
      tailwindcss(),
      {
        name: "emit-build-commit-file",
        closeBundle() {
          try {
            mkdirSync("dist", { recursive: true });
            writeFileSync(join("dist", ".build-commit"), `${buildCommitHash}\n`, "utf-8");
          } catch {
            // ignore write errors in read-only environments
          }
        },
      },
    ],
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
  };
});
