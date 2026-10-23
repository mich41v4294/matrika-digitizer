from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func
from sqlmodel import Session, select
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.app.api import census, debug, familysearch, questions, records, scans, uploads, ws
from backend.app.config import settings
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


@app.middleware("http")
async def verify_optional_api_key(request: Request, call_next):
    configured_key = (settings.app_api_key or "").strip()
    if configured_key:
        path = request.url.path
        method = request.method.upper()
        needs_auth = path.startswith("/api/debug") or (
            path.startswith("/api/") and method in {"POST", "PUT", "PATCH", "DELETE"}
        )
        if needs_auth:
            provided = (request.headers.get("X-API-Key") or "").strip()
            if not provided:
                auth_header = (request.headers.get("Authorization") or "").strip()
                if auth_header.lower().startswith("bearer "):
                    provided = auth_header[7:].strip()
            if provided != configured_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
    return await call_next(request)

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


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "pangolin_url": settings.pangolin_public_url}


@app.get("/api/status")
async def system_status() -> dict[str, Any]:
    ai_info = await ai_client.get_models_status()
    fs_env = read_fs_env()
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

    return {
        "pangolin_public_url": settings.pangolin_public_url,
        "ai_server": ai_info,
        "familysearch": {
            "has_token": bool(fs_env.get("FS_ACCESS_TOKEN")),
            "environment": fs_env.get("FS_ENVIRONMENT", "production"),
        },
        "counts": {
            "scans": scans_count,
            "rows": rows_count,
            "facts": facts_count,
            "open_questions": open_q_count,
            "new_to_familysearch": new_fs_count,
            "unreviewed_matches": unreviewed_fs_count,
            "fs_pending_matches": unreviewed_fs_count,
            "fs_matched": new_fs_count,
            "census_sheets": census_sheets_count,
        },
        "pipeline_state": ws.ws_manager.latest_state,
    }


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
