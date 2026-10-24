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
    FamilySearchMatch,
    OCRFeedback,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.db.session import SessionDep
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
    format_llm_debug_report_markdown,
    get_feedback_statistics,
    persist_llm_debug_report_files,
    serialize_ocr_feedback,
    upsert_ocr_feedback,
)
from backend.app.pipeline.ocr_stage import media_rel_to_path, run_ocr_for_row

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
    )
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
        "ai_server": models_info,
    }


@router.get("/overview")
async def get_debug_overview(session: SessionDep) -> dict[str, Any]:
    models_info = await ai_client.get_models_status()
    fs_status = await fs_mcp_manager.get_auth_status()
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
    report_payload = build_llm_debug_report_payload(
        session,
        verdict_filter=verdict if verdict in ("correct", "incorrect") else None,
    )
    report_md = format_llm_debug_report_markdown(report_payload)
    report_files = persist_llm_debug_report_files(session)

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
) -> dict[str, Any]:
    fb = upsert_ocr_feedback(
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
    )
    report_payload = build_llm_debug_report_payload(session)
    report_md = format_llm_debug_report_markdown(report_payload)
    return {
        "status": "ok",
        "feedback": serialize_ocr_feedback(fb, include_traces=True),
        "stats": report_payload["statistics"],
        "report_markdown": report_md,
        "report_files": report_payload["statistics"]["summary"]["report_files"],
    }


@router.delete("/feedback/{feedback_id}")
def delete_debug_feedback(
    feedback_id: int,
    session: SessionDep,
) -> dict[str, Any]:
    fb = session.get(OCRFeedback, feedback_id)
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    session.delete(fb)
    session.commit()
    report_files = persist_llm_debug_report_files(session)
    report_payload = build_llm_debug_report_payload(session)
    return {
        "status": "deleted",
        "deleted_id": feedback_id,
        "stats": report_payload["statistics"],
        "report_markdown": format_llm_debug_report_markdown(report_payload),
        "report_files": report_files,
    }


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
        content = json.dumps(payload, ensure_ascii=False, indent=2)
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
    row = session.get(RecordRow, req.row_id)
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")
    scan = session.get(ScanImage, row.scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    is_census = bool(
        scan.detected_type == "census"
        or row.record_type == "census"
        or scan.census_sheet_id is not None
    )

    if req.stage == "all":
        ocr_out = await run_ocr_for_row(session, scan, row)
        if is_census:
            fact = await run_census_extraction_for_row(session, scan, row)
            if scan.census_sheet_id:
                reconcile_census_household_relationships(session, scan.census_sheet_id)
                if fact:
                    session.refresh(fact)
            evaluate_census_row_deterministic(session, scan, row, fact, log_trace=False)
            return {"stage": "all", "ocr": ocr_out, "fact": fact, "skipped_decision": True}
        dec_out = await run_decision_for_row(session, scan, row)
        fact = await run_extraction_for_row(session, scan, row)
        return {"stage": "all", "ocr": ocr_out, "decision": dec_out, "fact": fact}

    if req.stage == "ocr":
        out = await run_ocr_for_row(session, scan, row)
        return {"stage": "ocr", "result": out}
    if req.stage == "decision":
        out = await run_decision_for_row(session, scan, row)
        return {"stage": "decision", "result": out}
    if req.stage == "extract":
        if is_census:
            fact = await run_census_extraction_for_row(session, scan, row)
            if scan.census_sheet_id:
                reconcile_census_household_relationships(session, scan.census_sheet_id)
                if fact:
                    session.refresh(fact)
            evaluate_census_row_deterministic(session, scan, row, fact, log_trace=False)
        else:
            fact = await run_extraction_for_row(session, scan, row)
        return {"stage": "extract", "fact": fact}
    if req.stage == "mcp":
        fact = session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()
        if not fact:
            if is_census:
                fact = await run_census_extraction_for_row(session, scan, row)
                if scan.census_sheet_id:
                    reconcile_census_household_relationships(session, scan.census_sheet_id)
                    if fact:
                        session.refresh(fact)
                evaluate_census_row_deterministic(session, scan, row, fact, log_trace=False)
            else:
                fact = await run_extraction_for_row(session, scan, row)
        match = await fs_mcp_manager.match_person_fact(session, fact, row)
        return {"stage": "mcp", "match": match}

    raise HTTPException(status_code=400, detail=f"Unknown stage {req.stage}")


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
def update_feature_flags(payload: dict[str, Any]) -> dict[str, Any]:
    updates = payload.get("flags") if isinstance(payload.get("flags"), dict) else payload
    flags = settings.set_feature_flags(updates if isinstance(updates, dict) else {})
    return {"flags": flags}
