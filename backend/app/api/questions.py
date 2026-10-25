from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from backend.app.db.models import PersonFact, QuestionItem, RecordRow, ScanImage
from backend.app.db.session import SessionDep
from backend.app.mcp.familysearch_client import (
    fs_mcp_manager,
    schedule_auto_fs_match,
    trigger_auto_fs_match_for_fact,
)
from backend.app.pipeline.decision_stage import (
    _collect_neighbor_crops,
    build_slovak_question_details,
)
from backend.app.pipeline.lexicon import (
    apply_person_fact_search_fields,
    compute_person_fact_search_fields,
    parse_observationes_events,
)
from backend.app.pipeline.schemas import apply_human_answer_to_fields
from backend.app.search.fts import index_person_fact

router = APIRouter(prefix="/api/questions", tags=["questions"])


class AnswerQuestionRequest(BaseModel):
    answer: str
    answered_by: str = "Reviewer"


@router.get("")
def list_questions(
    session: SessionDep,
    status: str | None = None,
) -> dict[str, Any]:
    stmt = select(QuestionItem).order_by(QuestionItem.id.asc())
    if status:
        stmt = stmt.where(QuestionItem.status == status)
    items = session.exec(stmt).all()

    row_ids = list({q.row_id for q in items if q.row_id is not None})
    scan_ids = list({q.scan_id for q in items if q.scan_id is not None})
    rows_by_id: dict[int, RecordRow] = (
        {r.id: r for r in session.exec(select(RecordRow).where(RecordRow.id.in_(row_ids))).all() if r.id is not None}
        if row_ids
        else {}
    )
    scans_by_id: dict[int, ScanImage] = (
        {s.id: s for s in session.exec(select(ScanImage).where(ScanImage.id.in_(scan_ids))).all() if s.id is not None}
        if scan_ids
        else {}
    )
    facts_by_row_id: dict[int, PersonFact] = {}
    if row_ids:
        for pf in session.exec(select(PersonFact).where(PersonFact.row_id.in_(row_ids))).all():
            if pf.row_id not in facts_by_row_id:
                facts_by_row_id[pf.row_id] = pf

    # Pre-fetch all rows for the involved scans in one query and lazily cache parsed contexts
    scan_rows_map: dict[int, list[RecordRow]] = {}
    parsed_ctx_cache: dict[tuple[int, str], dict[str, str]] = {}
    from backend.app.pipeline.decision_stage import parse_row_ocr_context

    def _get_row_ctx(r_obj: RecordRow) -> dict[str, str]:
        if not r_obj.raw_ocr_combined and not r_obj.raw_ocr_left and not r_obj.raw_ocr_right:
            return {}
        ck = (int(r_obj.id or 0), r_obj.raw_ocr_combined or "")
        if r_obj.id is not None and ck in parsed_ctx_cache:
            return parsed_ctx_cache[ck]
        ctx_val = parse_row_ocr_context(r_obj.raw_ocr_left, r_obj.raw_ocr_right, r_obj.raw_ocr_combined)
        if r_obj.id is not None:
            parsed_ctx_cache[ck] = ctx_val
        return ctx_val

    if scan_ids:
        all_scan_rows = session.exec(
            select(RecordRow)
            .where(RecordRow.scan_id.in_(scan_ids))
            .order_by(RecordRow.row_index)
        ).all()
        for sr in all_scan_rows:
            scan_rows_map.setdefault(sr.scan_id, []).append(sr)

    enriched: list[dict[str, Any]] = []
    for q in items:
        row = rows_by_id.get(q.row_id)
        scan = scans_by_id.get(q.scan_id)
        fact = facts_by_row_id.get(q.row_id)

        stored_cands = json.loads(q.candidates_json or "[]")
        details = (
            build_slovak_question_details(
                scan=scan,
                row=row,
                uncertain_field=q.field_name,
                llm_question_sk=q.question_sk,
                llm_reason=q.question_en,
                llm_candidates=stored_cands,
                has_ditto=bool(row.has_ditto_marks if row else False),
                parsed_ctx=_get_row_ctx(row),
            )
            if row
            else {
                "question_sk": q.question_sk,
                "description_sk": q.question_en or "",
                "candidates": stored_cands,
                "row_context": None,
            }
        )

        if row and q.scan_id in scan_rows_map:
            neighbor_crops = []
            for n in scan_rows_map[q.scan_id]:
                if n.id == q.row_id or not n.raw_ocr_combined:
                    continue
                cells = json.loads(n.cell_crops_json or "[]")
                crop_url = n.stitched_crop_path
                for c in cells:
                    if c.get("key") == q.field_name:
                        crop_url = c.get("crop_url")
                        break
                n_ctx = _get_row_ctx(n)
                summary_parts = []
                if n_ctx.get("entry_number"):
                    summary_parts.append(f"č. {n_ctx['entry_number']}")
                if n_ctx.get("child_name"):
                    summary_parts.append(f"dieťa: {n_ctx['child_name']}")
                if n_ctx.get("parents_text"):
                    summary_parts.append(f"rodičia: {n_ctx['parents_text']}")
                if n_ctx.get("village_text"):
                    summary_parts.append(f"obec: {n_ctx['village_text']}")
                readable_snippet = (
                    " • ".join(summary_parts)
                    if summary_parts
                    else ((n.raw_ocr_left or "") + " | " + (n.raw_ocr_right or ""))[:140].strip(" |")
                )
                neighbor_crops.append({
                    "row_index": n.row_index,
                    "crop_url": crop_url,
                    "confirmed_text": readable_snippet,
                })
                if len(neighbor_crops) >= 3:
                    break
        else:
            neighbor_crops = (
                _collect_neighbor_crops(session, q.scan_id, q.row_id, q.field_name)
                if row
                else json.loads(q.neighbor_crops_json or "[]")
            )

        enriched.append({
            "id": q.id,
            "row_id": q.row_id,
            "scan_id": q.scan_id,
            "scan_filename": scan.filename if scan else f"Sken #{q.scan_id}",
            "detected_year": (scan.detected_year if scan and scan.detected_year else None)
            or (str(fact.event_year) if fact and fact.event_year else None),
            "row_index": row.row_index if row else 0,
            "entry_number": (fact.entry_number if fact and fact.entry_number else None)
            or (row.entry_number if row and row.entry_number else None)
            or (details["row_context"]["entry_number"] if details.get("row_context") else None),
            "field_name": q.field_name,
            "question_sk": details["question_sk"],
            "description_sk": details["description_sk"],
            "question_en": q.question_en or details["description_sk"],
            "row_context": details.get("row_context"),
            "crop_url": q.crop_path or (row.stitched_crop_path if row else None),
            "stitched_row_url": row.stitched_crop_path if row else None,
            "neighbor_crops": neighbor_crops,
            "confirmed_candidates": details["candidates"],
            "candidates": details["candidates"],
            "raw_ocr_snippet": q.raw_ocr_snippet,
            "status": q.status,
            "selected_answer": q.selected_answer,
            "answered_by": q.answered_by,
            "answered_at": q.answered_at,
        })

    status_counts = dict(
        session.exec(
            select(QuestionItem.status, func.count(QuestionItem.id)).group_by(QuestionItem.status)
        ).all()
    )
    counts = {
        "open": int(status_counts.get("open", 0)),
        "answered": int(status_counts.get("answered", 0)),
        "skipped_for_admin": int(status_counts.get("skipped_for_admin", 0)),
    }
    return {"counts": counts, "questions": enriched}


@router.post("/{question_id}/answer")
async def answer_question(
    question_id: int,
    req: AnswerQuestionRequest,
    session: SessionDep,
) -> dict[str, Any]:
    q = session.get(QuestionItem, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    q.selected_answer = req.answer.strip()
    q.answered_by = req.answered_by
    q.answered_at = datetime.now(timezone.utc)
    q.status = "answered"
    session.add(q)
    session.commit()

    row = session.get(RecordRow, q.row_id)
    scan = session.get(ScanImage, q.scan_id)
    fact = session.exec(select(PersonFact).where(PersonFact.row_id == q.row_id)).first()

    if fact is None and row is not None and scan is not None:
        from backend.app.pipeline.extraction_stage import (
            run_census_extraction_for_row,
            run_extraction_for_row,
        )

        is_census = bool(scan.census_sheet_id or scan.detected_type == "census")
        if is_census:
            fact = await run_census_extraction_for_row(session, scan, row)
        else:
            fact = await run_extraction_for_row(session, scan, row)

    if fact and q.selected_answer:
        fact_dict: dict[str, Any] = {
            "entry_number": fact.entry_number,
            "birth_or_event_date": fact.birth_or_event_date,
            "baptism_or_burial_date": fact.baptism_or_burial_date,
            "given_name_raw": fact.given_name_raw,
            "given_name_sk": fact.given_name_sk,
            "given_name_hu": fact.given_name_hu,
            "surname_raw": fact.surname_raw,
            "surname_normalized": fact.surname_normalized,
            "sex": fact.sex,
            "father_name": fact.father_name,
            "mother_name": fact.mother_name,
            "mother_maiden_surname": fact.mother_maiden_surname,
            "social_status_or_occupation": fact.social_status_or_occupation,
            "village_raw": fact.village_raw,
            "village_sk": fact.village_sk,
            "village_hu": fact.village_hu,
            "house_number": fact.house_number,
            "observationes_raw": fact.observationes_raw,
        }
        apply_human_answer_to_fields(fact_dict, q.field_name, q.selected_answer, req.answered_by)
        for k, v in fact_dict.items():
            if hasattr(fact, k) and v is not None:
                setattr(fact, k, v)
        obs_events = parse_observationes_events(fact.observationes_raw)
        fact.observationes_events_json = json.dumps(obs_events, ensure_ascii=False)
        apply_person_fact_search_fields(fact)
        session.add(fact)
        session.commit()
        session.refresh(fact)
        index_person_fact(session, fact)

    if row:
        row.status = "extracted"
        session.add(row)

    session.commit()

    if fact and fact.id is not None:
        schedule_auto_fs_match(fact.id, trigger_auto_fs_match_for_fact)

    return {"status": "ok", "question_id": q.id, "selected_answer": q.selected_answer}


@router.post("/{question_id}/skip")
def skip_question_for_admin(question_id: int, session: SessionDep) -> dict[str, Any]:
    q = session.get(QuestionItem, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    q.status = "skipped_for_admin"
    session.add(q)
    session.commit()
    return {"status": "skipped_for_admin", "question_id": q.id}
