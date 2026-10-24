from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlmodel import select

from backend.app.api.scans import resolve_scan_media_url as _resolve_scan_media_url
from backend.app.config import settings
from backend.app.db.models import (
    CensusHousehold,
    CensusSheet,
    FamilySearchMatch,
    LocalRecordMatch,
    PersonFact,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.db.session import SessionDep
from backend.app.mcp.familysearch_client import (
    _ensure_split_spouse_in_fs_url,
    _format_fact_person_label,
    _is_nameless_fact,
    build_action_packet_and_new_items,
    build_census_action_packet,
    cleanup_unmatched_familysearch_rows,
    fs_mcp_manager,
    write_fs_env,
)

router = APIRouter(prefix="/api/familysearch", tags=["familysearch"])


class TokenRequest(BaseModel):
    access_token: str = Field(min_length=1, max_length=4096, pattern=r"^[^\r\n\x00]+$")
    client_id: str | None = Field(default=None, max_length=256, pattern=r"^[^\r\n\x00]*$")
    app_key: str | None = Field(default=None, max_length=256, pattern=r"^[^\r\n\x00]*$")
    environment: Literal["production", "beta", "integration"] = "production"


class PlaceTestRequest(BaseModel):
    name: str
    year: int = 1884


class ReviewMatchRequest(BaseModel):
    reviewed: bool = True
    notes: str | None = None


@router.get("/status")
@router.get("/token-status")
async def get_fs_status(session: SessionDep) -> dict[str, Any]:
    auth_status = await fs_mcp_manager.get_auth_status()
    progress = fs_mcp_manager.get_progress_state(session)
    return {
        **auth_status,
        **progress,
        "auth_mode": auth_status.get("mode", "public_places_and_action_packets"),
        "mode": progress["mode"],
        "progress": progress,
    }


@router.get("/progress")
def get_fs_progress(session: SessionDep) -> dict[str, Any]:
    progress = fs_mcp_manager.get_progress_state(session)
    return {**progress, "progress": progress}


@router.post("/token")
async def save_fs_token(req: TokenRequest, session: SessionDep) -> dict[str, Any]:
    updates = {
        "FS_ACCESS_TOKEN": req.access_token.strip(),
        "FS_ENVIRONMENT": req.environment,
    }
    if req.client_id or req.app_key:
        updates["FS_CLIENT_ID"] = (req.client_id or req.app_key or "").strip()
    write_fs_env(updates)
    auth_status = await fs_mcp_manager.get_auth_status(force=True)
    progress = fs_mcp_manager.get_progress_state(session)
    return {
        **auth_status,
        **progress,
        "auth_mode": auth_status.get("mode", "public_places_and_action_packets"),
        "mode": progress["mode"],
        "progress": progress,
    }


@router.post("/test-place")
async def test_fs_place(req: PlaceTestRequest) -> dict[str, Any]:
    return await fs_mcp_manager.call_tool(
        "search_places_at_date",
        {"name": req.name, "year": req.year},
        stage="mcp_place",
    )


def _clean_what_agrees(raw_agrees: list[Any]) -> list[Any]:
    """Filter out self-referential census strings or '? ?' placeholders from what_agrees."""
    cleaned: list[Any] = []
    for item in raw_agrees or []:
        if isinstance(item, str):
            s = item.strip()
            if not s or "? ?" in s or s.startswith("Sčítací hárok"):
                continue
            cleaned.append(item)
        elif isinstance(item, dict):
            val = str(item.get("value") or item.get("fs_val") or item.get("matrika_val") or "").strip()
            field_sk = str(item.get("field_sk") or "").strip()
            if "? ?" in val or val.startswith("Sčítací hárok") or field_sk.startswith("Sčítací hárok"):
                continue
            cleaned.append(item)
    return cleaned


@router.get("/matches")
def list_fs_matches(session: SessionDep, scan_id: int | None = None) -> dict[str, Any]:
    stmt = (
        select(FamilySearchMatch)
        .where(
            FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
            func.trim(FamilySearchMatch.fs_person_id) != "",
            FamilySearchMatch.confidence_score > 0,
        )
        .order_by(FamilySearchMatch.id.desc())
    )
    matches = [
        m
        for m in session.exec(stmt).all()
        if m.fs_person_id and str(m.fs_person_id).strip() and (m.confidence_score or 0.0) > 0
    ]

    fact_ids = list({m.person_fact_id for m in matches if m.person_fact_id is not None})
    row_ids = list({m.row_id for m in matches if m.row_id is not None})
    facts_by_id = (
        {f.id: f for f in session.exec(select(PersonFact).where(PersonFact.id.in_(fact_ids))).all()}
        if fact_ids
        else {}
    )
    for f in facts_by_id.values():
        if f.row_id and f.row_id not in row_ids:
            row_ids.append(f.row_id)

    rows_by_id = (
        {r.id: r for r in session.exec(select(RecordRow).where(RecordRow.id.in_(row_ids))).all()}
        if row_ids
        else {}
    )
    scan_ids = list({f.scan_id for f in facts_by_id.values() if f.scan_id is not None})
    scans_by_id = (
        {s.id: s for s in session.exec(select(ScanImage).where(ScanImage.id.in_(scan_ids))).all()}
        if scan_ids
        else {}
    )
    sheet_ids = list({f.census_sheet_id for f in facts_by_id.values() if f.census_sheet_id is not None})
    sheets_by_id = (
        {cs.id: cs for cs in session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all()}
        if sheet_ids
        else {}
    )
    households_by_sheet: dict[int, CensusHousehold] = {}
    households_by_id: dict[int, CensusHousehold] = {}
    if sheet_ids:
        for hh in session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id.in_(sheet_ids))
            .order_by(CensusHousehold.id.asc())
        ).all():
            if hh.id is not None:
                households_by_id[hh.id] = hh
            if hh.census_sheet_id not in households_by_sheet or (hh.dwelling_unit or "I") == "I":
                households_by_sheet[hh.census_sheet_id] = hh

    peers_by_sheet: dict[int, list[PersonFact]] = {}
    if sheet_ids:
        for pf in session.exec(
            select(PersonFact).where(PersonFact.census_sheet_id.in_(sheet_ids))
        ).all():
            if pf.census_sheet_id is not None:
                peers_by_sheet.setdefault(pf.census_sheet_id, []).append(pf)

    local_matches_by_census_fact: dict[int, list[LocalRecordMatch]] = {}
    if fact_ids:
        for lm in session.exec(
            select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id.in_(fact_ids))
        ).all():
            local_matches_by_census_fact.setdefault(lm.census_fact_id, []).append(lm)

    items: list[dict[str, Any]] = []
    for m in matches:
        fact = facts_by_id.get(m.person_fact_id)
        if fact is None or _is_nameless_fact(fact):
            continue

        sheet = sheets_by_id.get(fact.census_sheet_id) if fact.census_sheet_id else None
        if sheet and getattr(sheet, "sheet_type", "household") in ("overview", "municipal_overview"):
            continue

        if scan_id and fact.scan_id != scan_id:
            continue

        row = rows_by_id.get(m.row_id) or (rows_by_id.get(fact.row_id) if fact.row_id else None)
        scan = scans_by_id.get(fact.scan_id) if fact.scan_id else None

        stored_packet = json.loads(m.action_packet_json or "{}")
        stored_new = json.loads(m.what_is_new_json or "[]")
        std_place = (
            fact.fs_place_standardized
            or stored_packet.get("standardized_place")
            or stored_packet.get("birth_place_modern_sk")
        )
        raw_fs_url = m.fs_url or stored_packet.get("fs_search_url") or stored_packet.get("familysearch_search_url") or ""
        sheet_peers = peers_by_sheet.get(fact.census_sheet_id) if fact.census_sheet_id else None
        fixed_fs_url = _ensure_split_spouse_in_fs_url(raw_fs_url, fact, session, peers=sheet_peers)

        if fact.record_type == "census":
            household = (
                households_by_id.get(fact.census_household_id)
                if fact.census_household_id
                else None
            ) or (households_by_sheet.get(fact.census_sheet_id) if fact.census_sheet_id else None)
            local_matches = local_matches_by_census_fact.get(fact.id or 0, [])
            fresh_packet, fresh_new = build_census_action_packet(
                fact=fact,
                sheet=sheet,
                household=household,
                place_std=std_place,
                fs_url=fixed_fs_url,
                local_matches=local_matches,
            )
            action_packet = {**stored_packet, **fresh_packet}
            what_is_new = fresh_new if fresh_new else stored_new
        else:
            fresh_packet, fresh_new = build_action_packet_and_new_items(
                fact,
                std_place,
                fixed_fs_url,
                session=session,
            )
            action_packet = {**stored_packet, **fresh_packet}
            what_is_new = fresh_new if fresh_new else stored_new

        what_agrees = _clean_what_agrees(json.loads(m.what_agrees_json or "[]"))

        items.append({
            "id": m.id,
            "fact_id": m.person_fact_id,
            "person_fact_id": m.person_fact_id,
            "row_id": m.row_id or (row.id if row else None),
            "scan_id": fact.scan_id,
            "scan_filename": scan.filename if scan else None,
            "scan_url": _resolve_scan_media_url(scan),
            "overlay_url": scan.overlay_path if scan else None,
            "stitched_crop_url": row.stitched_crop_path if row else None,
            "y_start": row.y_start if row else None,
            "y_end": row.y_end if row else None,
            "scan_width": scan.width if scan else None,
            "scan_height": scan.height if scan else None,
            "status": m.match_status,
            "match_status": m.match_status,
            "fs_person_id": m.fs_person_id,
            "fs_person_name": action_packet.get("person_name_display"),
            "fs_record_ark": m.fs_record_ark,
            "fs_url": fixed_fs_url,
            "confidence_score": m.confidence_score,
            "standardized_place": action_packet.get("birth_place_modern_sk") or std_place,
            "summary_sk": m.summary_sk,
            "explanation_sk": m.summary_sk,
            "what_agrees": what_agrees,
            "what_differs": json.loads(m.what_differs_json or "[]"),
            "what_is_new": what_is_new,
            "action_packet": action_packet,
            "reviewed_by_user": m.reviewed_by_user,
            "user_notes": m.user_notes,
            "fact": fact.model_dump() if fact else None,
        })

    progress = fs_mcp_manager.get_progress_state(session, refresh_counts=False)
    counts = {
        "total": len(items),
        "matched": len(items),
        "new_person": sum(1 for x in items if x["match_status"] == "new_person"),
        "existing_needs_facts": sum(1 for x in items if x["match_status"] == "existing_needs_facts"),
        "confirmed_match": sum(1 for x in items if x["match_status"] == "confirmed_match"),
        "reviewed": sum(1 for x in items if x["reviewed_by_user"]),
        "unreviewed": sum(1 for x in items if not x["reviewed_by_user"]),
        "total_facts_in_db": progress["total_facts_in_db"],
        "total_matched_in_db": progress["total_matched_in_db"],
        "reviewed_count": progress["reviewed_count"],
    }
    return {"counts": counts, "matches": items, "progress": progress}


@router.post("/matches/{fact_id}/refresh")
async def refresh_fs_match(fact_id: int, session: SessionDep) -> dict[str, Any]:
    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    row = session.get(RecordRow, fact.row_id) if fact.row_id else None
    person_label = _format_fact_person_label(fact)
    await fs_mcp_manager.start_progress(
        mode="auto",
        total=1,
        current_person=person_label,
        session=session,
    )
    try:
        match = await fs_mcp_manager.match_person_fact(session, fact, row)
        is_matched = bool(match and (match.fs_person_id or "").strip())
        await fs_mcp_manager.step_progress(
            increment_processed=1,
            increment_matched=1 if is_matched else 0,
            current_person=person_label,
            session=session,
        )
        progress = await fs_mcp_manager.finish_progress(session=session)
    except Exception as exc:
        await fs_mcp_manager.finish_progress(session=session, error=str(exc))
        raise
    if not match or not (match.fs_person_id or "").strip():
        return {"status": "unmatched", "match_id": None, "fs_status": "unmatched", "progress": progress}
    return {
        "status": "ok",
        "match_id": match.id,
        "fs_person_id": match.fs_person_id,
        "fs_status": match.match_status,
        "progress": progress,
    }


@router.post("/match-row/{row_id}")
async def match_row_endpoint(row_id: int, session: SessionDep) -> dict[str, Any]:
    row = session.get(RecordRow, row_id)
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")
    fact = session.exec(select(PersonFact).where(PersonFact.row_id == row_id)).first()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found for row")
    person_label = _format_fact_person_label(fact)
    await fs_mcp_manager.start_progress(
        mode="auto",
        total=1,
        current_person=person_label,
        session=session,
    )
    try:
        match = await fs_mcp_manager.match_person_fact(session, fact, row)
        is_matched = bool(match and (match.fs_person_id or "").strip())
        await fs_mcp_manager.step_progress(
            increment_processed=1,
            increment_matched=1 if is_matched else 0,
            current_person=person_label,
            session=session,
        )
        progress = await fs_mcp_manager.finish_progress(session=session)
    except Exception as exc:
        await fs_mcp_manager.finish_progress(session=session, error=str(exc))
        raise
    if not match or not (match.fs_person_id or "").strip():
        return {"status": "unmatched", "match_id": None, "fs_status": "unmatched", "progress": progress}
    return {
        "status": "ok",
        "match_id": match.id,
        "fs_person_id": match.fs_person_id,
        "fs_status": match.match_status,
        "progress": progress,
    }


@router.post("/matches/{match_id}/review")
async def mark_match_reviewed(
    match_id: int,
    req: ReviewMatchRequest,
    session: SessionDep,
) -> dict[str, Any]:
    match = session.get(FamilySearchMatch, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    match.reviewed_by_user = req.reviewed
    if req.notes is not None:
        match.user_notes = req.notes
    session.add(match)
    session.commit()
    progress = await fs_mcp_manager.broadcast_progress(session=session, refresh_counts=True)
    return {
        "status": "ok",
        "match_id": match.id,
        "reviewed_by_user": match.reviewed_by_user,
        "progress": progress,
    }


@router.post("/sync-all")
@router.post("/rematch")
async def sync_all_fs_matches(session: SessionDep) -> dict[str, Any]:
    cleanup_unmatched_familysearch_rows(session)
    facts = list(session.exec(select(PersonFact).order_by(PersonFact.id.asc())).all())

    # Pre-load census sheets so we can accurately count non-nameless, non-overview people to check
    sheet_ids = list({f.census_sheet_id for f in facts if f.census_sheet_id is not None})
    sheets_by_id = (
        {cs.id: cs for cs in session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all()}
        if sheet_ids
        else {}
    )

    valid_facts_to_check: list[PersonFact] = []
    seen_sheet_ids: set[int] = set()
    for fact in facts:
        if _is_nameless_fact(fact):
            continue
        if fact.record_type == "census" and fact.census_sheet_id:
            if fact.census_sheet_id in seen_sheet_ids:
                continue
            seen_sheet_ids.add(fact.census_sheet_id)
            sheet = sheets_by_id.get(fact.census_sheet_id)
            if sheet and getattr(sheet, "sheet_type", "household") not in ("overview", "municipal_overview"):
                sheet_facts = [
                    sf
                    for sf in facts
                    if sf.census_sheet_id == sheet.id and not _is_nameless_fact(sf)
                ]
                valid_facts_to_check.extend(sheet_facts)
            continue
        valid_facts_to_check.append(fact)

    total_to_check = len(valid_facts_to_check)
    first_person = _format_fact_person_label(valid_facts_to_check[0]) if valid_facts_to_check else ""
    await fs_mcp_manager.start_progress(
        mode="rematch",
        total=total_to_check,
        current_person=first_person,
        session=session,
    )

    synced = 0
    synced_sheet_ids: set[int] = set()
    try:
        for fact in facts:
            if _is_nameless_fact(fact):
                continue
            if fact.record_type == "census" and fact.census_sheet_id:
                if fact.census_sheet_id not in synced_sheet_ids:
                    synced_sheet_ids.add(fact.census_sheet_id)
                    sheet = sheets_by_id.get(fact.census_sheet_id) or session.get(
                        CensusSheet, fact.census_sheet_id
                    )
                    if sheet and getattr(sheet, "sheet_type", "household") not in ("overview", "municipal_overview"):
                        synced += await fs_mcp_manager.match_census_sheet(
                            session,
                            sheet,
                            manage_progress=False,
                        )
                continue
            person_label = _format_fact_person_label(fact)
            await fs_mcp_manager.step_progress(current_person=person_label, session=session)
            row = session.get(RecordRow, fact.row_id) if fact.row_id else None
            m = await fs_mcp_manager.match_person_fact(session, fact, row)
            is_matched = bool(m and (m.fs_person_id or "").strip())
            if is_matched:
                synced += 1
            await fs_mcp_manager.step_progress(
                increment_processed=1,
                increment_matched=1 if is_matched else 0,
                current_person=person_label,
                session=session,
            )
        cleanup_unmatched_familysearch_rows(session)
        progress = await fs_mcp_manager.finish_progress(session=session)
    except Exception as exc:
        await fs_mcp_manager.finish_progress(session=session, error=str(exc))
        raise
    return {
        "status": "ok",
        "synced_count": synced,
        "matched_count": synced,
        "progress": progress,
    }


rematch_all_fs_matches = sync_all_fs_matches


@router.get("/logs")
def list_mcp_logs(session: SessionDep, limit: int = 50) -> dict[str, Any]:
    logs = session.exec(
        select(WorkflowTrace)
        .where(WorkflowTrace.stage.in_(["mcp_place", "mcp_record", "mcp_auth", "mcp_compare"]))
        .order_by(WorkflowTrace.id.desc())
        .limit(limit)
    ).all()
    return {"logs": [l.model_dump() for l in logs]}
