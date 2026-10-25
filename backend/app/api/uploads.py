from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.db.models import Batch, ScanImage
from backend.app.db.session import SessionDep
from backend.app.pipeline.orchestrator import process_single_scan_pipeline
from backend.app.pipeline.task_queue import task_queue

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_FILE_BYTES = 50 * 1024 * 1024  # 50 MB per scan


def _is_valid_image_magic_bytes(content: bytes) -> bool:
    if len(content) < 12:
        return False
    if content.startswith(b"\xff\xd8\xff"):
        return True
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return True
    return False


def _slugify_folder_part(text: str | None, fallback: str = "matrika", max_len: int = 48) -> str:
    if not text:
        return fallback
    norm = unicodedata.normalize("NFKD", str(text))
    ascii_str = "".join(ch for ch in norm if not unicodedata.combining(ch)).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_str).strip("_")
    return (slug[:max_len].rstrip("_")) or fallback


def _get_matrika_root() -> Path:
    root = getattr(settings, "matrika_data_dir", None) or (Path(settings.upload_dir) / "matrika")
    p = Path(root).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _build_batch_storage_dir(batch: Batch) -> Path:
    """
    Returns (and creates) a human-browsable directory for `batch` inside `settings.matrika_data_dir`:
    `<matrika_data_dir>/<YYYY-MM-DD>_batch_<batch.id:04d>_<slug>/`
    Reuses any existing folder matching `*_batch_<batch.id:04d>_*` so multi-chunk uploads
    always land in the same directory.
    """
    matrika_root = _get_matrika_root()
    bid = int(batch.id or 0)
    pattern = f"*_batch_{bid:04d}_*"
    existing = sorted(p for p in matrika_root.glob(pattern) if p.is_dir())
    if existing:
        return existing[0]

    created_dt = getattr(batch, "created_at", None) or datetime.now(timezone.utc)
    date_prefix = created_dt.strftime("%Y-%m-%d")
    slug_parts: list[str] = []
    if batch.parish_or_book:
        slug_parts.append(_slugify_folder_part(batch.parish_or_book, fallback="", max_len=28))
    slug_parts.append(_slugify_folder_part(batch.name, fallback="matrika", max_len=36))
    if batch.record_type_hint and batch.record_type_hint != "auto":
        slug_parts.append(_slugify_folder_part(batch.record_type_hint, fallback="", max_len=16))
    combined_slug = "_".join(p for p in slug_parts if p)[:64] or "matrika"

    folder_name = f"{date_prefix}_batch_{bid:04d}_{combined_slug}"
    batch_dir = (matrika_root / folder_name).resolve()
    if not batch_dir.is_relative_to(matrika_root):
        raise HTTPException(status_code=400, detail="Invalid batch storage directory path.")
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def _write_batch_manifest(session: Session, batch: Batch, batch_dir: Path) -> None:
    """Writes a self-documenting `batch_manifest.json` inside `batch_dir` on the NFS share."""
    try:
        scans = session.exec(
            select(ScanImage)
            .where(ScanImage.batch_id == batch.id)
            .order_by(ScanImage.id.asc())
        ).all()
        manifest = {
            "batch_id": batch.id,
            "batch_name": batch.name,
            "parish_or_book": batch.parish_or_book,
            "record_type_hint": batch.record_type_hint,
            "created_at": (
                batch.created_at.isoformat()
                if getattr(batch, "created_at", None) is not None
                else None
            ),
            "total_scans": len(scans),
            "scans": [
                {
                    "scan_id": s.id,
                    "filename": s.filename,
                    "stored_file": Path(s.original_path).name if s.original_path else s.filename,
                    "detected_type": s.detected_type,
                    "detected_year": s.detected_year,
                    "status": s.status,
                }
                for s in scans
            ],
        }
        manifest_path = batch_dir / "batch_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


class CreateBatchRequest(BaseModel):
    name: str = "Nahraná matrika"
    parish_or_book: str | None = None
    record_type_hint: str = "auto"
    watchlist_filter: str | None = None


class LoadSamplesRequest(BaseModel):
    limit: int | None = None
    auto_process_first: bool = False
    max_rows_first: int = 3
    watchlist_filter: str | None = None


async def _validate_and_save_upload(
    uf: UploadFile,
    batch: Batch,
    idx: int,
    ts: int,
) -> tuple[str, Path, Path]:
    raw_name = Path(uf.filename or f"scan_{idx}.jpg").name
    ext = Path(raw_name).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}' for '{raw_name}'. Allowed: .jpg, .jpeg, .png, .webp",
        )
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(raw_name).stem)[:80] or f"scan_{idx}"
    safe_name = f"{safe_stem}{ext}"
    content = await uf.read(MAX_UPLOAD_FILE_BYTES + 1)
    if len(content) > MAX_UPLOAD_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{raw_name}' exceeds maximum allowed size of 50 MB.",
        )
    if not _is_valid_image_magic_bytes(content):
        raise HTTPException(
            status_code=400,
            detail=f"File '{raw_name}' is not a valid JPEG, PNG, or WebP image.",
        )

    batch_dir = _build_batch_storage_dir(batch)
    matrika_root = _get_matrika_root()
    dest_name = f"{idx:03d}_{safe_name}"
    dest_path = (batch_dir / dest_name).resolve()
    if dest_path.exists():
        dest_name = f"{idx:03d}_{ts}_{safe_name}"
        dest_path = (batch_dir / dest_name).resolve()
    if not dest_path.is_relative_to(matrika_root):
        raise HTTPException(status_code=400, detail="Invalid upload destination path.")
    dest_path.write_bytes(content)
    return raw_name, dest_path, batch_dir


@router.post("/batches")
def create_upload_batch(req: CreateBatchRequest, session: SessionDep) -> dict[str, Any]:
    settings.ensure_dirs()
    batch = Batch(
        name=req.name or "Nahraná matrika",
        parish_or_book=req.parish_or_book or None,
        record_type_hint=req.record_type_hint or "auto",
        watchlist_filter=req.watchlist_filter or None,
        total_scans=0,
        status="uploaded",
    )
    session.add(batch)
    session.commit()
    session.refresh(batch)
    return {"batch_id": batch.id, "batch_name": batch.name}


@router.post("/chunk")
async def upload_scan_chunk(
    session: SessionDep,
    batch_id: int = Form(...),
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    settings.ensure_dirs()
    batch = session.get(Batch, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    ts = int(time.time())
    start_idx = (batch.total_scans or 0) + 1
    new_scans: list[ScanImage] = []
    batch_dir: Path | None = None
    for offset, uf in enumerate(files):
        idx = start_idx + offset
        raw_name, dest_path, batch_dir = await _validate_and_save_upload(uf, batch, idx, ts)
        scan = ScanImage(
            batch_id=batch.id,
            filename=raw_name,
            original_path=str(dest_path),
            detected_type=batch.record_type_hint if batch.record_type_hint != "auto" else "baptism",
            status="uploaded",
        )
        session.add(scan)
        new_scans.append(scan)

    batch.total_scans = (batch.total_scans or 0) + len(new_scans)
    session.add(batch)
    session.commit()
    for scan in new_scans:
        session.refresh(scan)

    if batch_dir is not None:
        _write_batch_manifest(session, batch, batch_dir)

    return {
        "batch_id": batch.id,
        "uploaded_count": len(new_scans),
        "total_scans": batch.total_scans,
        "batch_folder": batch_dir.name if batch_dir else None,
        "scans": [{"id": s.id, "filename": s.filename} for s in new_scans],
    }


@router.post("/load-samples")
async def load_sample_scans(
    req: LoadSamplesRequest,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    settings.ensure_dirs()
    if not settings.sample_dir.exists():
        raise HTTPException(status_code=404, detail=f"Samples directory not found at {settings.sample_dir}")

    sample_files = sorted(
        [
            p
            for p in settings.sample_dir.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
        ]
    )
    if req.limit:
        sample_files = sample_files[: req.limit]

    batch = Batch(
        name="Ukážkové matriky (Samples 1802–1896)",
        parish_or_book="Slovenské a uhorské cirkevné matriky",
        record_type_hint="auto",
        watchlist_filter=req.watchlist_filter,
        total_scans=len(sample_files),
        status="uploaded",
    )
    session.add(batch)
    session.commit()
    session.refresh(batch)

    batch_dir = _build_batch_storage_dir(batch)
    new_scans: list[ScanImage] = []
    for idx, sp in enumerate(sample_files, start=1):
        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", sp.stem)[:80] or f"sample_{idx}"
        dest_name = f"{idx:03d}_{safe_stem}{sp.suffix.lower()}"
        dest_path = batch_dir / dest_name
        shutil.copy2(sp, dest_path)

        scan = ScanImage(
            batch_id=batch.id,
            filename=sp.name,
            original_path=str(dest_path),
            status="uploaded",
        )
        session.add(scan)
        new_scans.append(scan)

    session.commit()
    for scan in new_scans:
        session.refresh(scan)
    _write_batch_manifest(session, batch, batch_dir)
    created_scans = [{"id": s.id, "filename": s.filename} for s in new_scans]

    if req.auto_process_first and created_scans:
        first_id = created_scans[0]["id"]
        first_name = created_scans[0]["filename"]
        task_queue.enqueue_background(
            lambda: process_single_scan_pipeline(
                first_id,
                max_rows=req.max_rows_first,
                run_mcp=True,
                watchlist_filter=req.watchlist_filter,
            ),
            task_type="matrika_scan_ocr",
            category="ocr",
            title_sk=f"AI OCR ukážkového skenu: {first_name}",
            title_en=f"Sample scan AI OCR: {first_name}",
            initial_stage="layout",
            active_model=settings.model_ocr,
            target_url=f"/knihy?scan={first_id}",
            metadata={"scan_id": first_id},
        )

    return {
        "batch_id": batch.id,
        "uploaded_count": len(created_scans),
        "batch_folder": batch_dir.name,
        "scans": created_scans,
    }
