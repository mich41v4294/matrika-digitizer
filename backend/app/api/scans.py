from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from backend.app.config import settings
from backend.app.db.models import Batch, FamilySearchMatch, PersonFact, QuestionItem, RecordRow, ScanImage
from backend.app.db.session import SessionDep
from backend.app.pipeline.orchestrator import (
    process_batch_stage_queued,
    process_single_scan_pipeline,
    run_layout_for_scan_sync,
)

router = APIRouter(prefix="/api/scans", tags=["scans"])


class ProcessScanRequest(BaseModel):
    max_rows: int | None = None
    run_mcp: bool = True
    watchlist_filter: str | None = None
    background: bool = True
    only_undone: bool = True


_RESOLVED_ROOTS_CACHE: tuple[
    tuple[Path | None, Path | None, Path | None],
    tuple[Path | None, Path | None, Path | None],
] | None = None


def _get_resolved_media_roots() -> tuple[Path | None, Path | None, Path | None]:
    global _RESOLVED_ROOTS_CACHE
    raw_c = getattr(settings, "census_data_dir", None)
    raw_m = getattr(settings, "matrika_data_dir", None)
    raw_u = getattr(settings, "upload_dir", None)
    key = (
        Path(raw_c) if raw_c else None,
        Path(raw_m) if raw_m else None,
        Path(raw_u) if raw_u else None,
    )
    if _RESOLVED_ROOTS_CACHE is not None and _RESOLVED_ROOTS_CACHE[0] == key:
        return _RESOLVED_ROOTS_CACHE[1]
    res = (
        key[0].resolve() if key[0] else None,
        key[1].resolve() if key[1] else None,
        key[2].resolve() if key[2] else None,
    )
    _RESOLVED_ROOTS_CACHE = (key, res)
    return res


def resolve_scan_media_url(
    scan_or_path: ScanImage | str | None,
    filename: str | None = None,
) -> str | None:
    if scan_or_path is None and not filename:
        return None
    if isinstance(scan_or_path, ScanImage):
        raw = str(scan_or_path.original_path or "").strip()
        filename = filename or scan_or_path.filename
    else:
        raw = str(scan_or_path or "").strip()
    if not raw:
        return f"/media/uploads/{filename}" if filename else None
    if raw.startswith("/media/") or raw.startswith("http://") or raw.startswith("https://"):
        return raw
    p = Path(raw)
    try:
        p_res = p.resolve()
    except OSError:
        p_res = p
    census_root, matrika_root, upload_root = _get_resolved_media_roots()
    if census_root:
        try:
            rel_c = p_res.relative_to(census_root)
            return f"/media/uploads/slovakiana/{rel_c.as_posix()}"
        except ValueError:
            pass
    if matrika_root:
        try:
            rel_m = p_res.relative_to(matrika_root)
            return f"/media/uploads/matrika/{rel_m.as_posix()}"
        except ValueError:
            pass
    if upload_root:
        try:
            rel_u = p_res.relative_to(upload_root)
            return f"/media/uploads/{rel_u.as_posix()}"
        except ValueError:
            pass
    if "slovakiana/" in raw:
        idx = raw.find("slovakiana/")
        return f"/media/uploads/{raw[idx:]}"
    if "matrika/" in raw:
        idx = raw.find("matrika/")
        return f"/media/uploads/{raw[idx:]}"
    return f"/media/uploads/{filename or p.name}"


def _scan_to_dict(scan: ScanImage, rows_count: int = 0, open_questions: int = 0) -> dict[str, Any]:
    return {
        "id": scan.id,
        "batch_id": scan.batch_id,
        "census_sheet_id": scan.census_sheet_id,
        "page_side": scan.page_side,
        "filename": scan.filename,
        "original_url": resolve_scan_media_url(scan),
        "overlay_url": scan.overlay_path,
        "header_crop_url": scan.header_crop_path,
        "width": scan.width,
        "height": scan.height,
        "spine_x": scan.spine_x,
        "detected_year": scan.detected_year,
        "detected_type": scan.detected_type,
        "header_raw_ocr": scan.header_raw_ocr,
        "status": scan.status,
        "error_message": scan.error_message,
        "rows_count": rows_count,
        "open_questions": open_questions,
        "created_at": scan.created_at,
    }


@router.get("")
def list_scans(
    session: SessionDep,
    scan_type: str | None = None,
    is_census: bool | None = None,
) -> dict[str, Any]:
    stmt = select(ScanImage).order_by(ScanImage.id.desc())
    if scan_type == "matrika" or is_census is False:
        stmt = stmt.where(
            ScanImage.census_sheet_id.is_(None),  # type: ignore[union-attr]
            ScanImage.detected_type != "census",
        )
    elif scan_type == "census" or is_census is True:
        stmt = stmt.where(
            (ScanImage.census_sheet_id.is_not(None))  # type: ignore[union-attr]
            | (ScanImage.detected_type == "census")
        )
    scans = session.exec(stmt).all()
    batches = session.exec(select(Batch).order_by(Batch.id.desc())).all()
    row_counts = dict(
        session.exec(
            select(RecordRow.scan_id, func.count(RecordRow.id)).group_by(RecordRow.scan_id)
        ).all()
    )
    open_q_counts = dict(
        session.exec(
            select(QuestionItem.scan_id, func.count(QuestionItem.id))
            .where(QuestionItem.status == "open")
            .group_by(QuestionItem.scan_id)
        ).all()
    )
    out_scans = [
        _scan_to_dict(s, int(row_counts.get(s.id, 0)), int(open_q_counts.get(s.id, 0)))
        for s in scans
    ]
    return {"batches": batches, "scans": out_scans}


@router.get("/{scan_id}")
def get_scan_detail(scan_id: int, session: SessionDep) -> dict[str, Any]:
    scan = session.get(ScanImage, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    rows = session.exec(
        select(RecordRow).where(RecordRow.scan_id == scan_id).order_by(RecordRow.row_index)
    ).all()
    facts = session.exec(select(PersonFact).where(PersonFact.scan_id == scan_id)).all()
    fact_by_row = {f.row_id: f for f in facts}
    fact_ids = [f.id for f in facts if f.id is not None]
    matches_by_fact: dict[int, FamilySearchMatch] = {}
    if fact_ids:
        for m in session.exec(
            select(FamilySearchMatch).where(
                FamilySearchMatch.person_fact_id.in_(fact_ids),
                FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
                func.trim(FamilySearchMatch.fs_person_id) != "",
                FamilySearchMatch.confidence_score > 0,
            )
        ).all():
            if m.fs_person_id and str(m.fs_person_id).strip():
                matches_by_fact[m.person_fact_id] = m
    questions = session.exec(select(QuestionItem).where(QuestionItem.scan_id == scan_id)).all()
    q_by_row: dict[int, list[QuestionItem]] = {}
    for q in questions:
        q_by_row.setdefault(q.row_id, []).append(q)

    grid_meta = json.loads(scan.grid_metadata_json or "{}")
    grid_row_by_idx = {
        gr.get("row_index"): gr
        for gr in grid_meta.get("rows", [])
        if isinstance(gr, dict) and gr.get("row_index") is not None
    }

    rows_payload = []
    for r in rows:
        fact = fact_by_row.get(r.id)
        fs_match = matches_by_fact.get(fact.id) if (fact and fact.id is not None) else None
        gr_info = grid_row_by_idx.get(r.row_index, {})
        rows_payload.append({
            "id": r.id,
            "row_index": r.row_index,
            "entry_number": r.entry_number,
            "y_start": r.y_start,
            "y_end": r.y_end,
            "angle_left_deg": gr_info.get("angle_left_deg", grid_meta.get("skew_left_deg", 0.0)),
            "angle_right_deg": gr_info.get("angle_right_deg", grid_meta.get("skew_right_deg", 0.0)),
            "left_quad": gr_info.get("left_quad", []),
            "right_quad": gr_info.get("right_quad", []),
            "stitched_crop_url": r.stitched_crop_path,
            "left_crop_url": r.left_crop_path,
            "right_crop_url": r.right_crop_path,
            "cell_crops": json.loads(r.cell_crops_json or "[]"),
            "raw_ocr_left": r.raw_ocr_left,
            "raw_ocr_right": r.raw_ocr_right,
            "raw_ocr_combined": r.raw_ocr_combined,
            "decision_action": r.decision_action,
            "decision_confidence": r.decision_confidence,
            "decision_reason": r.decision_reason,
            "has_ditto_marks": r.has_ditto_marks,
            "normalized": json.loads(r.normalized_json or "{}"),
            "status": r.status,
            "fact": fact,
            "fs_match": fs_match,
            "questions": q_by_row.get(r.id, []),
        })

    return {
        "scan": _scan_to_dict(scan, len(rows), sum(1 for q in questions if q.status == "open")),
        "grid_metadata": grid_meta,
        "rows": rows_payload,
    }


@router.post("/{scan_id}/slice")
def slice_scan_only(scan_id: int, session: SessionDep) -> dict[str, Any]:
    scan = session.get(ScanImage, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    rows = run_layout_for_scan_sync(session, scan)
    return {
        "scan_id": scan.id,
        "rows_sliced": len(rows),
        "overlay_url": scan.overlay_path,
        "header_crop_url": scan.header_crop_path,
    }


@router.post("/{scan_id}/process")
async def trigger_scan_pipeline(
    scan_id: int,
    req: ProcessScanRequest,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    scan = session.get(ScanImage, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if req.background:
        background_tasks.add_task(
            process_single_scan_pipeline,
            scan_id,
            max_rows=req.max_rows,
            run_mcp=req.run_mcp,
            watchlist_filter=req.watchlist_filter,
        )
        return {"status": "started_in_background", "scan_id": scan_id}

    res = await process_single_scan_pipeline(
        scan_id,
        max_rows=req.max_rows,
        run_mcp=req.run_mcp,
        watchlist_filter=req.watchlist_filter,
    )
    return {"status": "completed", **res}


@router.post("/batches/{batch_id}/process")
async def trigger_batch_pipeline(
    batch_id: int,
    req: ProcessScanRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    background_tasks.add_task(
        process_batch_stage_queued,
        batch_id,
        max_rows_per_scan=req.max_rows,
        run_mcp=req.run_mcp,
        only_undone=req.only_undone,
    )
    return {"status": "batch_started", "batch_id": batch_id}
