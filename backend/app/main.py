from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func
from sqlmodel import Session, select
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.app.api import census, debug, familysearch, questions, records, scans, uploads, ws
from backend.app.config import BASE_REPO_DIR, settings
from backend.app.db.models import (
    CensusSheet,
    FamilySearchMatch,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
)
from backend.app.db.session import engine, init_db
from backend.app.mcp.familysearch_client import fs_mcp_manager, read_fs_env
from backend.app.pipeline.ai_client import ai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield
    await fs_mcp_manager.close()


app = FastAPI(
    title="Matrika OCR & FamilySearch Digitizer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(GZipMiddleware, minimum_size=1000)

allowed_origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]
if settings.pangolin_public_url:
    allowed_origins.append(settings.pangolin_public_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptionalApiKeyMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            configured_key = (settings.app_api_key or "").strip()
            if configured_key:
                request = Request(scope, receive=receive)
                path = request.url.path
                method = request.method.upper()
                is_public_feature_flags_get = method == "GET" and path in {
                    "/api/debug/feature-flags",
                    "/api/feature-flags",
                }
                needs_auth = (path.startswith("/api/debug") and not is_public_feature_flags_get) or (
                    path.startswith("/api/") and method in {"POST", "PUT", "PATCH", "DELETE"}
                )
                if needs_auth:
                    provided = (request.headers.get("X-API-Key") or "").strip()
                    if not provided:
                        auth_header = (request.headers.get("Authorization") or "").strip()
                        if auth_header.lower().startswith("bearer "):
                            provided = auth_header[7:].strip()
                    if provided != configured_key:
                        resp = JSONResponse(
                            status_code=401,
                            content={"detail": "Invalid or missing API key."},
                        )
                        await resp(scope, receive, send)
                        return
        await self.app(scope, receive, send)


app.add_middleware(OptionalApiKeyMiddleware)

settings.ensure_dirs()
app.mount(
    "/media/uploads/slovakiana",
    StaticFiles(directory=str(settings.census_data_dir)),
    name="census-uploads",
)
app.mount(
    "/media/uploads/matrika",
    StaticFiles(directory=str(settings.matrika_data_dir)),
    name="matrika-uploads",
)
app.mount("/media/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
app.mount("/media/crops", StaticFiles(directory=str(settings.crops_dir)), name="crops")

app.include_router(uploads.router)
app.include_router(scans.router)
app.include_router(questions.router)
app.include_router(records.router)
app.include_router(census.router)
app.include_router(familysearch.router)
app.include_router(debug.router)
app.include_router(ws.router)


_GITHUB_REPO = "mich41v4294/matrika-digitizer"
_HEX_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_CACHED_BUILD_COMMIT: str | None = None


def _normalize_commit_hash(raw: str | None) -> str | None:
    val = (raw or "").strip()
    if not val or val in ("dev", "unknown") or "$Format:" in val:
        return None
    if _HEX_SHA_RE.fullmatch(val):
        return val[:7].lower()
    return val


def _extract_40_hex_from_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        m = re.search(r"\b([0-9a-fA-F]{40})\b", path.read_text(encoding="utf-8"))
        if m:
            return m.group(1)[:7].lower()
    except OSError:
        pass
    return None


def _resolve_git_commit_from_dir(start_dir: Path) -> str | None:
    current = start_dir.resolve()
    for _ in range(4):
        dot_git = current / ".git"
        if dot_git.exists():
            try:
                git_dir = dot_git
                common_dir = dot_git
                if dot_git.is_file():
                    content = dot_git.read_text(encoding="utf-8").strip()
                    if content.startswith("gitdir:"):
                        rel_git = content[len("gitdir:") :].strip()
                        git_dir = (current / rel_git).resolve()
                        common_dir = git_dir
                        commondir_file = git_dir / "commondir"
                        if commondir_file.is_file():
                            rel_common = commondir_file.read_text(encoding="utf-8").strip()
                            if rel_common:
                                common_dir = (git_dir / rel_common).resolve()
                head_file = git_dir / "HEAD"
                if head_file.is_file():
                    head = head_file.read_text(encoding="utf-8").strip()
                    if not head.startswith("ref:"):
                        norm = _normalize_commit_hash(head)
                        if norm:
                            return norm
                    else:
                        ref_path = head[len("ref:") :].strip()
                        candidate_refs = [ref_path]
                        if ref_path.startswith("refs/heads/"):
                            branch = ref_path[len("refs/heads/") :]
                            candidate_refs.append(f"refs/remotes/origin/{branch}")
                        for ref in candidate_refs:
                            for base in (git_dir, common_dir):
                                loose_ref = base / ref
                                if loose_ref.is_file():
                                    norm = _normalize_commit_hash(
                                        loose_ref.read_text(encoding="utf-8")
                                    )
                                    if norm:
                                        return norm
                            for base in (common_dir, git_dir):
                                packed_refs = base / "packed-refs"
                                if packed_refs.is_file():
                                    for line in packed_refs.read_text(
                                        encoding="utf-8"
                                    ).splitlines():
                                        stripped = line.strip()
                                        if (
                                            not stripped
                                            or stripped.startswith("#")
                                            or stripped.startswith("^")
                                        ):
                                            continue
                                        parts = stripped.split()
                                        if len(parts) >= 2 and parts[1] == ref and parts[0]:
                                            norm = _normalize_commit_hash(parts[0])
                                            if norm:
                                                return norm
                for base in (git_dir, common_dir):
                    for fallback_name in ("ORIG_HEAD", "FETCH_HEAD"):
                        sha = _extract_40_hex_from_file(base / fallback_name)
                        if sha:
                            return sha
                    head_log = base / "logs" / "HEAD"
                    if head_log.is_file():
                        for line in reversed(
                            head_log.read_text(encoding="utf-8").splitlines()
                        ):
                            parts = line.strip().split()
                            if len(parts) >= 2 and _HEX_SHA_RE.fullmatch(parts[1]):
                                return parts[1][:7].lower()
            except OSError:
                pass
        if current.parent == current:
            break
        current = current.parent
    return None


def _fetch_remote_github_commit_hash() -> str | None:
    for url, headers, pattern in (
        (
            f"https://github.com/{_GITHUB_REPO}.git/info/refs?service=git-upload-pack",
            {"User-Agent": "matrika-digitizer"},
            r"([0-9a-fA-F]{40})\s+(?:refs/heads/main|HEAD)\b",
        ),
        (
            f"https://api.github.com/repos/{_GITHUB_REPO}/commits/main",
            {
                "Accept": "application/vnd.github.sha",
                "User-Agent": "matrika-digitizer",
            },
            r"\b([0-9a-fA-F]{40})\b",
        ),
    ):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                m = re.search(pattern, body)
                if m:
                    return m.group(1)[:7].lower()
        except Exception:
            continue
    return None


def get_build_commit_hash() -> str:
    global _CACHED_BUILD_COMMIT
    if _CACHED_BUILD_COMMIT:
        return _CACHED_BUILD_COMMIT

    for key in (
        "BUILD_COMMIT_HASH",
        "VITE_COMMIT_HASH",
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
    ):
        norm = _normalize_commit_hash(os.environ.get(key))
        if norm:
            _CACHED_BUILD_COMMIT = norm
            return norm

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=str(BASE_REPO_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        norm = _normalize_commit_hash(out)
        if norm:
            _CACHED_BUILD_COMMIT = norm
            return norm
    except Exception:
        pass

    from_git_dir = _resolve_git_commit_from_dir(BASE_REPO_DIR)
    if from_git_dir:
        _CACHED_BUILD_COMMIT = from_git_dir
        return from_git_dir

    try:
        from backend.app._build_info import ARCHIVE_COMMIT_HASH

        norm = _normalize_commit_hash(ARCHIVE_COMMIT_HASH)
        if norm:
            _CACHED_BUILD_COMMIT = norm
            return norm
    except Exception:
        pass

    for commit_file in (
        settings.frontend_dist_dir / ".build-commit",
        BASE_REPO_DIR / "frontend" / "dist" / ".build-commit",
        BASE_REPO_DIR / ".build-commit",
    ):
        if commit_file.is_file():
            try:
                norm = _normalize_commit_hash(commit_file.read_text(encoding="utf-8"))
                if norm:
                    _CACHED_BUILD_COMMIT = norm
                    return norm
            except OSError:
                pass

    from_remote = _fetch_remote_github_commit_hash()
    if from_remote:
        _CACHED_BUILD_COMMIT = from_remote
        return from_remote

    return "dev"


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "pangolin_url": settings.pangolin_public_url,
        "build_commit": get_build_commit_hash(),
    }


@app.get("/api/status")
async def system_status() -> dict[str, Any]:
    ai_info = await ai_client.get_models_status()
    fs_env = read_fs_env()
    counts_payload: dict[str, int] | None = None
    try:
        import json as _json
        import matrika_db_rs as _rust_db

        if settings.db_path and Path(settings.db_path).exists():
            counts_payload = _json.loads(_rust_db.system_status_counts_rs(str(settings.db_path)))
    except Exception:
        counts_payload = None

    if counts_payload is None:
        with Session(engine) as session:
            scans_count = session.exec(select(func.count()).select_from(ScanImage)).one()
            rows_count = session.exec(select(func.count()).select_from(RecordRow)).one()
            facts_count = session.exec(select(func.count()).select_from(PersonFact)).one()
            open_q_count = session.exec(
                select(func.count())
                .select_from(QuestionItem)
                .where(QuestionItem.status == "open")
            ).one()
            matched_fs_filter = (
                FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
                func.trim(FamilySearchMatch.fs_person_id) != "",
                FamilySearchMatch.confidence_score > 0,
            )
            fs_row = session.exec(
                select(
                    func.count(FamilySearchMatch.id),
                    func.coalesce(
                        func.sum(
                            case(
                                (FamilySearchMatch.reviewed_by_user == False, 1),  # noqa: E712
                                else_=0,
                            )
                        ),
                        0,
                    ),
                ).where(*matched_fs_filter)
            ).one()
            new_fs_count = int(fs_row[0] or 0)
            unreviewed_fs_count = int(fs_row[1] or 0)
            census_sheets_count = session.exec(select(func.count()).select_from(CensusSheet)).one()
            counts_payload = {
                "scans": scans_count,
                "rows": rows_count,
                "facts": facts_count,
                "open_questions": open_q_count,
                "new_to_familysearch": new_fs_count,
                "unreviewed_matches": unreviewed_fs_count,
                "fs_pending_matches": unreviewed_fs_count,
                "fs_matched": new_fs_count,
                "census_sheets": census_sheets_count,
            }

    from backend.app.pipeline.task_queue import task_queue

    q_snap = task_queue.get_snapshot()
    ws.ws_manager.latest_state["queued_count"] = q_snap["counts"]["queued"]
    ws.ws_manager.latest_state["active_task"] = q_snap["active_task"]
    ws.ws_manager.latest_state["queued_tasks"] = q_snap["queued_tasks"]

    return {
        "pangolin_public_url": settings.pangolin_public_url,
        "build_commit": get_build_commit_hash(),
        "ai_server": ai_info,
        "familysearch": {
            "has_token": bool(fs_env.get("FS_ACCESS_TOKEN")),
            "token_valid": bool(
                fs_env.get("FS_ACCESS_TOKEN")
                and fs_mcp_manager._cached_auth_status
                and fs_mcp_manager._cached_auth_status.get("token_valid")
            ),
            "sync_enabled": bool(
                fs_env.get("FS_ACCESS_TOKEN")
                and fs_mcp_manager._cached_auth_status
                and fs_mcp_manager._cached_auth_status.get("token_valid")
            ),
            "environment": fs_env.get("FS_ENVIRONMENT", "production"),
        },
        "counts": counts_payload,
        "pipeline_state": ws.ws_manager.latest_state,
        "task_queue": q_snap,
        "feature_flags": settings.get_feature_flags(),
    }


@app.get("/api/feature-flags")
def get_public_feature_flags() -> dict[str, Any]:
    return {"flags": settings.get_feature_flags()}


if settings.frontend_dist_dir.exists():
    assets_dir = settings.frontend_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path in ("api", "media") or full_path.startswith(("api/", "media/")):
            raise HTTPException(status_code=404, detail="Not found")
        dist_root = settings.frontend_dist_dir.resolve()
        candidate = (dist_root / full_path).resolve()
        if full_path and candidate.is_relative_to(dist_root) and candidate.is_file():
            return FileResponse(str(candidate))
        index_html = dist_root / "index.html"
        if index_html.is_file():
            return FileResponse(
                str(index_html),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )
        return {"message": "Frontend not built yet."}
