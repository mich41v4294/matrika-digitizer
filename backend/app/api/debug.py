from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from backend.app.api.ws import ws_manager
from backend.app.config import settings
from backend.app.db.models import (
    CensusDownloadJob,
    FamilySearchMatch,
    OCRFeedback,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.db.session import SessionDep, engine
from backend.app.mcp.familysearch_client import fs_mcp_manager
from backend.app.pipeline.ai_client import ai_client
from backend.app.pipeline.decision_stage import (
    evaluate_census_row_deterministic,
    run_decision_for_row,
)
from backend.app.pipeline.extraction_stage import (
    reconcile_census_household_relationships,
    resolve_row_extraction_crop,
    run_census_extraction_for_row,
    run_extraction_for_row,
)
from backend.app.pipeline.feedback import (
    build_llm_debug_report_payload,
    delete_ocr_feedback_fast,
    format_llm_debug_report_markdown,
    get_feedback_statistics,
    persist_llm_debug_report_files,
    reset_ocr_feedback,
    serialize_ocr_feedback,
    submit_ocr_feedback_fast,
    upsert_ocr_feedback,
)
from backend.app.pipeline.ocr_stage import media_rel_to_path, run_ocr_for_row
from backend.app.pipeline.task_queue import task_queue

router = APIRouter(prefix="/api/debug", tags=["debug"])


class SubmitFeedbackRequest(BaseModel):
    verdict: str  # "correct" | "incorrect"
    comment: str | None = None
    expected_correction: str | None = None
    target_type: str = "row"
    row_id: int | None = None
    scan_id: int | None = None
    census_sheet_id: int | None = None
    person_fact_id: int | None = None
    overview_house_id: int | None = None
    municipal_district_id: int | None = None



class RerunStageRequest(BaseModel):
    row_id: int
    stage: str  # ocr, decision, extract, mcp, all


class CustomPromptRequest(BaseModel):
    model: str
    provider: str | None = None
    prompt: str
    system_prompt: str | None = None
    row_id: int | None = None
    use_row_image: bool = False
    max_tokens: int = 800


class AIConfigUpdateRequest(BaseModel):
    provider: str | None = None
    freellm_model: str | None = None
    model: str | None = None
    local_model: str | None = None
    stage: str | None = None
    stages: dict[str, Any] | None = None
    preset: str | None = None
    local_max_concurrent: int | None = None


@router.get("/ai-config")
async def get_ai_config() -> dict[str, Any]:
    cfg = settings.get_ai_runtime_config()
    models_info = await ai_client.get_models_status()
    return {
        **cfg,
        "online": models_info.get("online", False),
        "local_online": models_info.get("local_online", False),
        "local_models": models_info.get("local_models", []),
        "freellm_online": models_info.get("freellm_online", False),
        "freellm_models": models_info.get("freellm_models", []),
        "models": models_info.get("models", []),
        "stages": cfg.get("stages", models_info.get("stages", {})),
        "concurrency": ai_client.get_concurrency_state(),
        "ai_server": models_info,
    }


@router.post("/ai-config")
async def update_ai_config(req: AIConfigUpdateRequest) -> dict[str, Any]:
    cfg = settings.set_ai_runtime_config(
        provider=req.provider,
        freellm_model=req.freellm_model,
        stage=req.stage,
        model=req.model,
        local_model=req.local_model,
        stages=req.stages,
        preset=req.preset,
        local_max_concurrent=req.local_max_concurrent,
    )
    if req.local_max_concurrent is not None:
        ai_client.notify_concurrency_limit_changed()
    models_info = await ai_client.get_models_status()
    return {
        **cfg,
        "online": models_info.get("online", False),
        "local_online": models_info.get("local_online", False),
        "local_models": models_info.get("local_models", []),
        "freellm_online": models_info.get("freellm_online", False),
        "freellm_models": models_info.get("freellm_models", []),
        "models": models_info.get("models", []),
        "stages": cfg.get("stages", models_info.get("stages", {})),
        "concurrency": ai_client.get_concurrency_state(),
        "ai_server": models_info,
    }


@router.get("/overview")
async def get_debug_overview(session: SessionDep) -> dict[str, Any]:
    models_info = await ai_client.get_models_status()
    fs_status = await fs_mcp_manager.get_auth_status()
    q_snap = task_queue.get_snapshot()
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "get_debug_overview_stats_and_traces_rs"):
            ov_rs = json.loads(_rust_db.get_debug_overview_stats_and_traces_rs(db_p, 80))
            return {
                "pangolin_public_url": settings.pangolin_public_url,
                "ai_server": models_info,
                "familysearch_mcp": fs_status,
                "pipeline_state": ws_manager.latest_state,
                "task_queue": q_snap,
                "stats": ov_rs["stats"],
                "feedback_stats": get_feedback_statistics(session),
                "recent_traces": ov_rs["recent_traces"],
            }
    except Exception:
        pass
    traces = session.exec(select(WorkflowTrace).order_by(WorkflowTrace.id.desc()).limit(80)).all()

    total_scans = session.exec(select(func.count()).select_from(ScanImage)).one()
    total_rows = session.exec(select(func.count()).select_from(RecordRow)).one()
    total_facts = session.exec(select(func.count()).select_from(PersonFact)).one()
    open_questions = session.exec(
        select(func.count()).select_from(QuestionItem).where(QuestionItem.status == "open")
    ).one()
    total_matches = session.exec(
        select(func.count())
        .select_from(FamilySearchMatch)
        .where(
            FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
            func.trim(FamilySearchMatch.fs_person_id) != "",
            FamilySearchMatch.confidence_score > 0,
        )
    ).one()

    return {
        "pangolin_public_url": settings.pangolin_public_url,
        "ai_server": models_info,
        "familysearch_mcp": fs_status,
        "pipeline_state": ws_manager.latest_state,
        "task_queue": q_snap,
        "stats": {
            "total_scans": total_scans,
            "total_rows": total_rows,
            "total_facts": total_facts,
            "open_questions": open_questions,
            "total_matches": total_matches,
            "traces_logged": len(traces),
        },
        "feedback_stats": get_feedback_statistics(session),
        "recent_traces": traces,
    }


@router.get("/tasks")
def get_debug_tasks() -> dict[str, Any]:
    return task_queue.get_snapshot()


@router.post("/tasks/{task_id}/cancel")
@router.delete("/tasks/{task_id}")
def cancel_debug_task(task_id: str, session: SessionDep) -> dict[str, Any]:
    cancelled = task_queue.cancel_task(task_id)
    if cancelled is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found in queue")
    job_id = (cancelled.get("metadata") or {}).get("job_id")
    if job_id is not None:
        j = session.get(CensusDownloadJob, int(job_id))
        if j and j.status == "running":
            j.status = "paused"
            session.add(j)
            session.commit()
    return {
        "status": "cancelled",
        "task": cancelled,
        "task_queue": task_queue.get_snapshot(),
    }


@router.post("/tasks/{task_id}/prioritize")
def prioritize_debug_task(task_id: str) -> dict[str, Any]:
    updated = task_queue.prioritize_task(task_id)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found in queue")
    return {
        "status": "prioritized",
        "task": updated,
        "task_queue": task_queue.get_snapshot(),
    }


@router.post("/tasks/clear-queued")
def clear_debug_queued_tasks(session: SessionDep) -> dict[str, Any]:
    queued_before = task_queue.get_queued_tasks_list()
    cnt = task_queue.clear_queued_tasks()
    for qt in queued_before:
        jid = (qt.get("metadata") or {}).get("job_id")
        if jid is not None:
            j = session.get(CensusDownloadJob, int(jid))
            if j and j.status == "running":
                j.status = "paused"
                session.add(j)
    session.commit()
    return {
        "status": "cleared_queued",
        "cleared_count": cnt,
        "task_queue": task_queue.get_snapshot(),
    }


@router.post("/tasks/clear-history")
def clear_debug_task_history() -> dict[str, Any]:
    cnt = task_queue.clear_history()
    return {
        "status": "cleared_history",
        "cleared_count": cnt,
        "task_queue": task_queue.get_snapshot(),
    }


@router.get("/feedback/stats")
def get_debug_feedback_stats(session: SessionDep) -> dict[str, Any]:
    return get_feedback_statistics(session)


@router.get("/feedback")
def list_debug_feedback(
    session: SessionDep,
    verdict: str | None = None,
    scan_id: int | None = None,
    row_id: int | None = None,
    census_sheet_id: int | None = None,
    model_combo: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    stmt = select(OCRFeedback).order_by(OCRFeedback.updated_at.desc(), OCRFeedback.id.desc())
    if verdict in ("correct", "incorrect"):
        stmt = stmt.where(OCRFeedback.verdict == verdict)
    if scan_id is not None:
        stmt = stmt.where(OCRFeedback.scan_id == scan_id)
    if row_id is not None:
        stmt = stmt.where(OCRFeedback.row_id == row_id)
    if census_sheet_id is not None:
        stmt = stmt.where(OCRFeedback.census_sheet_id == census_sheet_id)
    if model_combo:
        stmt = stmt.where(OCRFeedback.model_combo == model_combo)

    rows = list(session.exec(stmt.limit(limit)).all())
    report_files = persist_llm_debug_report_files(session)
    report_payload = build_llm_debug_report_payload(
        session,
        verdict_filter=verdict if verdict in ("correct", "incorrect") else None,
    )
    report_md = format_llm_debug_report_markdown(report_payload)

    return {
        "feedback": [serialize_ocr_feedback(r, include_traces=True) for r in rows],
        "stats": report_payload["statistics"],
        "report_markdown": report_md,
        "report_files": report_files,
    }


@router.post("/feedback")
def submit_debug_feedback(
    req: SubmitFeedbackRequest,
    session: SessionDep,
    light: bool = False,
) -> dict[str, Any]:
    return submit_ocr_feedback_fast(
        session,
        verdict=req.verdict,
        comment=req.comment,
        expected_correction=req.expected_correction,
        target_type=req.target_type,
        row_id=req.row_id,
        scan_id=req.scan_id,
        census_sheet_id=req.census_sheet_id,
        person_fact_id=req.person_fact_id,
        overview_house_id=req.overview_house_id,
        municipal_district_id=req.municipal_district_id,
        light=light,
    )


@router.delete("/feedback/{feedback_id}")
def delete_debug_feedback(
    feedback_id: int,
    session: SessionDep,
    light: bool = False,
) -> dict[str, Any]:
    res = delete_ocr_feedback_fast(session, feedback_id, light=light)
    if res is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return res


@router.get("/feedback/export")
def export_debug_feedback_file(
    session: SessionDep,
    format: str = "md",
    verdict: str | None = None,
    feedback_id: int | None = None,
) -> Response:
    persist_llm_debug_report_files(session)
    payload = build_llm_debug_report_payload(
        session,
        verdict_filter=verdict if verdict in ("correct", "incorrect") else None,
        feedback_id=feedback_id,
    )
    fmt = format.strip().lower()
    suffix = f"_{verdict}" if verdict in ("correct", "incorrect") else (f"_item_{feedback_id}" if feedback_id else "")
    if fmt == "json":
        clean_payload = {k: v for k, v in payload.items() if k != "_cached_markdown"}
        content = json.dumps(clean_payload, ensure_ascii=False, indent=2)
        filename = f"ocr_feedback_debug_report{suffix}.json"
        return Response(
            content=content,
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    md_text = format_llm_debug_report_markdown(payload)
    filename = f"ocr_feedback_debug_report{suffix}.md"
    return PlainTextResponse(
        content=md_text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )



@router.get("/traces")
def list_workflow_traces(
    session: SessionDep,
    scan_id: int | None = None,
    row_id: int | None = None,
    stage: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "list_workflow_traces_rs"):
            return json.loads(
                _rust_db.list_workflow_traces_rs(
                    db_p,
                    scan_id=scan_id,
                    row_id=row_id,
                    stage=stage,
                    limit=limit,
                )
            )
    except Exception:
        pass
    stmt = select(WorkflowTrace).order_by(WorkflowTrace.id.desc())
    if scan_id is not None:
        stmt = stmt.where(WorkflowTrace.scan_id == scan_id)
    if row_id is not None:
        stmt = stmt.where(WorkflowTrace.row_id == row_id)
    if stage:
        if stage == "ocr":
            stmt = stmt.where(
                WorkflowTrace.stage.in_(
                    ["ocr", "ocr_header", "ocr_left", "ocr_right", "census_overview_ocr"]
                )
            )
        elif stage == "extract":
            stmt = stmt.where(
                WorkflowTrace.stage.in_(
                    ["extract", "census_extract", "census_overview_extract"]
                )
            )
        elif stage == "layout":
            stmt = stmt.where(WorkflowTrace.stage.in_(["layout", "census_layout"]))
        elif stage in ("mcp_search", "mcp_record"):
            stmt = stmt.where(WorkflowTrace.stage.in_(["mcp_record", "mcp_search"]))
        else:
            stmt = stmt.where(WorkflowTrace.stage == stage)
    traces = session.exec(stmt.limit(limit)).all()
    return {"traces": traces}


@router.post("/rerun-stage")
async def rerun_single_row_stage(
    req: RerunStageRequest,
    session: SessionDep,
) -> dict[str, Any]:
    if req.stage not in ("all", "ocr", "decision", "extract", "mcp"):
        raise HTTPException(status_code=400, detail=f"Unknown stage {req.stage}")
    row_check = session.get(RecordRow, req.row_id)
    if not row_check:
        raise HTTPException(status_code=404, detail="Row not found")
    scan_check = session.get(ScanImage, row_check.scan_id)
    if not scan_check:
        raise HTTPException(status_code=404, detail="Scan not found")

    row_id = int(row_check.id or req.row_id)
    row_idx = row_check.row_index
    scan_id = int(scan_check.id or row_check.scan_id)
    scan_fname = scan_check.filename
    sheet_id = scan_check.census_sheet_id
    bind = session.get_bind() or engine

    reset_ocr_feedback(session, row_id=row_id)

    async def _do_rerun() -> dict[str, Any]:
        from sqlmodel import Session

        with Session(bind) as task_session:
            row = task_session.get(RecordRow, row_id)
            if not row:
                raise HTTPException(status_code=404, detail="Row not found")
            scan = task_session.get(ScanImage, scan_id)
            if not scan:
                raise HTTPException(status_code=404, detail="Scan not found")

            is_census = bool(
                scan.detected_type == "census"
                or row.record_type == "census"
                or scan.census_sheet_id is not None
            )

            await ws_manager.broadcast(
                "state",
                {
                    "current_sheet_id": scan.census_sheet_id,
                    "current_scan_id": scan.id,
                    "current_row_id": row.id,
                    "current_row_index": row.row_index,
                    "total_rows": 1,
                    "active_stage": "ocr" if req.stage in ("all", "ocr") else req.stage,
                    "active_model": settings.model_ocr if req.stage in ("all", "ocr") else settings.model_extract,
                    "progress_pct": 35,
                    "message_sk": f"Opakujem fázu '{req.stage}' pre riadok #{row.row_index} ({scan.filename})...",
                    "message_en": f"Re-running stage '{req.stage}' for row #{row.row_index} ({scan.filename})...",
                },
            )

            try:
                skip_row_ocr = bool(settings.get_feature_flags().get("skip_row_ocr", True))
                if req.stage == "all":
                    ocr_out: dict[str, Any] = {}
                    if not skip_row_ocr:
                        ocr_out = await run_ocr_for_row(task_session, scan, row)
                    else:
                        row.raw_ocr_left = None
                        row.raw_ocr_right = None
                        row.raw_ocr_combined = None
                        task_session.add(row)
                        task_session.commit()
                    if is_census:
                        await ws_manager.broadcast(
                            "state",
                            {
                                "current_sheet_id": scan.census_sheet_id,
                                "current_scan_id": scan.id,
                                "current_row_id": row.id,
                                "current_row_index": row.row_index,
                                "total_rows": 1,
                                "active_stage": "extract",
                                "active_model": settings.model_extract,
                                "progress_pct": 75,
                                "message_sk": f"Extrahujem údaje riadku #{row.row_index} ({scan.filename})...",
                                "message_en": f"Extracting row #{row.row_index} ({scan.filename})...",
                            },
                        )
                        fact = await run_census_extraction_for_row(
                            task_session, scan, row, skip_row_ocr=skip_row_ocr
                        )
                        if scan.census_sheet_id:
                            reconcile_census_household_relationships(task_session, scan.census_sheet_id)
                            if fact:
                                task_session.refresh(fact)
                        evaluate_census_row_deterministic(task_session, scan, row, fact, log_trace=False)
                        out_payload: dict[str, Any] = {
                            "stage": "all",
                            "ocr": ocr_out,
                            "fact": fact,
                            "skipped_decision": True,
                            "skipped_row_ocr": skip_row_ocr,
                        }
                    else:
                        dec_out = (
                            await run_decision_for_row(task_session, scan, row)
                            if not skip_row_ocr
                            else {"action": "auto_accept", "confidence": 0.95}
                        )
                        fact = await run_extraction_for_row(
                            task_session, scan, row, skip_row_ocr=skip_row_ocr
                        )
                        out_payload = {
                            "stage": "all",
                            "ocr": ocr_out,
                            "decision": dec_out,
                            "fact": fact,
                            "skipped_row_ocr": skip_row_ocr,
                        }
                elif req.stage == "ocr":
                    out = await run_ocr_for_row(task_session, scan, row)
                    out_payload = {"stage": "ocr", "result": out}
                elif req.stage == "decision":
                    out = await run_decision_for_row(task_session, scan, row)
                    out_payload = {"stage": "decision", "result": out}
                elif req.stage == "extract":
                    if is_census:
                        fact = await run_census_extraction_for_row(task_session, scan, row)
                        if scan.census_sheet_id:
                            reconcile_census_household_relationships(task_session, scan.census_sheet_id)
                            if fact:
                                task_session.refresh(fact)
                        evaluate_census_row_deterministic(task_session, scan, row, fact, log_trace=False)
                    else:
                        fact = await run_extraction_for_row(task_session, scan, row)
                    out_payload = {"stage": "extract", "fact": fact}
                elif req.stage == "mcp":
                    token_valid, auth_status = await fs_mcp_manager.ensure_valid_token()
                    if not token_valid:
                        has_tok = bool(auth_status.get("has_token_configured"))
                        prob = auth_status.get("token_problem")
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"FamilySearch OAuth token je neplatný alebo exspirovaný ({prob})."
                                if has_tok and prob
                                else "FamilySearch OAuth token je neplatný alebo exspirovaný."
                                if has_tok
                                else "FamilySearch OAuth token nie je vložený. Vyhľadávanie osôb cez FamilySearch MCP vyžaduje platný token."
                            ),
                        )
                    fact = task_session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()
                    if not fact:
                        if is_census:
                            fact = await run_census_extraction_for_row(task_session, scan, row)
                            if scan.census_sheet_id:
                                reconcile_census_household_relationships(task_session, scan.census_sheet_id)
                                if fact:
                                    task_session.refresh(fact)
                            evaluate_census_row_deterministic(task_session, scan, row, fact, log_trace=False)
                        else:
                            fact = await run_extraction_for_row(task_session, scan, row)
                    match = await fs_mcp_manager.match_person_fact(task_session, fact, row)
                    out_payload = {"stage": "mcp", "match": match}
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown stage {req.stage}")

                done_sk = f"Hotovo! Fáza '{req.stage}' pre riadok #{row.row_index} ({scan.filename}) dokončená."
                done_en = f"Completed stage '{req.stage}' for row #{row.row_index} ({scan.filename})."
                await ws_manager.broadcast(
                    "state",
                    {
                        "current_sheet_id": scan.census_sheet_id,
                        "current_scan_id": scan.id,
                        "current_row_id": row.id,
                        "active_stage": "idle",
                        "active_model": None,
                        "progress_pct": 100,
                        "message_sk": done_sk,
                        "message_en": done_en,
                    },
                )
                await ws_manager.emit_completion(
                    {
                        "status": "completed",
                        "kind": "row_rerun",
                        "sheet_id": scan.census_sheet_id,
                        "scan_id": scan.id,
                        "scan_filename": scan.filename,
                        "rows_processed": 1,
                        "title_sk": f"Riadok #{row.row_index} prečítaný ({req.stage})",
                        "title_en": f"Row #{row.row_index} completed ({req.stage})",
                        "message_sk": done_sk,
                        "message_en": done_en,
                        "target_url": f"/debug/{scan.id}?row={row.id}",
                    }
                )
                return out_payload
            except HTTPException:
                raise
            except Exception as exc:
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "idle",
                        "active_model": None,
                        "progress_pct": 0,
                        "message_sk": f"Chyba pri opakovaní riadku #{row.row_index}: {exc}",
                        "message_en": f"Error re-running row #{row.row_index}: {exc}",
                    },
                )
                raise

    return await task_queue.run_or_enqueue(
        _do_rerun,
        task_type="debug_rerun_stage",
        category="debug",
        title_sk=f"Opakovanie fázy '{req.stage}' – riadok #{row_idx} ({scan_fname})",
        title_en=f"Re-run stage '{req.stage}' – row #{row_idx} ({scan_fname})",
        initial_stage="ocr" if req.stage in ("all", "ocr") else req.stage,
        active_model=settings.model_ocr if req.stage in ("all", "ocr") else settings.model_extract,
        target_url=f"/debug/{scan_id}?row={row_id}",
        metadata={"scan_id": scan_id, "row_id": row_id, "sheet_id": sheet_id},
    )


@router.post("/custom-prompt")
async def run_custom_debug_prompt(
    req: CustomPromptRequest,
    session: SessionDep,
) -> dict[str, Any]:
    img_path = None
    img_rel = None
    scan_id = None
    if req.row_id:
        row = session.get(RecordRow, req.row_id)
        if row:
            scan_id = row.scan_id
            if req.use_row_image:
                img_path, img_rel = await asyncio.to_thread(resolve_row_extraction_crop, row)

    requested_model = req.model
    if req.provider and ":" not in requested_model:
        requested_model = f"{req.provider.strip().lower()}:{requested_model}"

    res = await ai_client.chat_completion(
        stage="debug_playground",
        model=requested_model,
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        image_path=img_path,
        image_rel_url=img_rel,
        scan_id=scan_id,
        row_id=req.row_id,
        max_tokens=req.max_tokens,
    )
    return res


@router.get("/feature-flags")
def get_feature_flags() -> dict[str, Any]:
    return {"flags": settings.get_feature_flags()}


@router.post("/feature-flags")
async def update_feature_flags(payload: dict[str, Any]) -> dict[str, Any]:
    updates = payload.get("flags") if isinstance(payload.get("flags"), dict) else payload
    flags = settings.set_feature_flags(updates if isinstance(updates, dict) else {})
    await ws_manager.broadcast("feature_flags", flags)
    return {"flags": flags}
