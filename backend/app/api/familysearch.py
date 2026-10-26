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
from backend.app.db.session import SessionDep, engine
from backend.app.mcp.familysearch_client import (
    _build_candidate_fs_url,
    _ensure_split_spouse_in_fs_url,
    _format_fact_person_label,
    _is_nameless_fact,
    apply_candidate_selection_to_match,
    build_action_packet_and_new_items,
    build_census_action_packet,
    cleanup_unmatched_familysearch_rows,
    evaluate_fs_match_candidates,
    fs_mcp_manager,
    is_valid_fs_identifier,
    normalize_fs_identifier,
    write_fs_env,
)
from backend.app.pipeline.task_queue import task_queue

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
    selected_fs_person_ids: list[str] | None = None


class SelectMatchCandidatesRequest(BaseModel):
    selected_fs_person_ids: list[str] | None = None
    toggle_fs_person_id: str | None = None
    add_candidate_id: str | None = None
    add_candidate_name: str | None = None


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


def _hydrate_candidates_for_match(
    m: FamilySearchMatch,
    fact: PersonFact,
    action_packet: dict[str, Any],
    what_agrees: list[Any],
    what_differs: list[Any],
    fixed_fs_url: str,
    std_place: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Ensures every match item has a normalized list of `candidates` (each with a `XXXX-YYY` identifier)
    and a list of `selected_fs_person_ids` (`["XXXX-YYY", ...]`).
    """
    raw_candidates: list[Any] = []
    try:
        parsed_c = json.loads(getattr(m, "candidates_json", None) or "[]")
        if isinstance(parsed_c, list):
            raw_candidates = parsed_c
    except Exception:
        raw_candidates = []

    if not raw_candidates and isinstance(action_packet.get("candidates"), list):
        raw_candidates = list(action_packet["candidates"])

    if not raw_candidates and m.raw_mcp_results_json and m.raw_mcp_results_json != "{}":
        try:
            raw_mcp = json.loads(m.raw_mcp_results_json)
            rec_search = raw_mcp.get("record_search") if isinstance(raw_mcp, dict) else None
            hits = (
                (
                    rec_search.get("results")
                    or rec_search.get("matches")
                    or rec_search.get("persons")
                    or rec_search.get("records")
                )
                if isinstance(rec_search, dict)
                else None
            )
            if isinstance(hits, list) and hits:
                raw_candidates = evaluate_fs_match_candidates(
                    fact,
                    hits,
                    place_std=std_place,
                    is_census=(fact.record_type == "census"),
                    default_fs_url=fixed_fs_url,
                )
        except Exception:
            pass

    primary_id = normalize_fs_identifier(m.fs_person_id) or str(m.fs_person_id or "").strip()
    person_display = (
        action_packet.get("person_name_display")
        or f"{fact.given_name_sk or fact.given_name_raw or '?'} {fact.surname_raw or fact.surname_normalized or '?'}".strip()
    )
    birth_year = fact.birth_year_est or fact.event_year
    birth_date_str = action_packet.get("birth_date") or fact.birth_or_event_date or (str(birth_year) if birth_year else None)

    normalized_cands: list[dict[str, Any]] = []
    seen_cids: set[str] = set()
    for idx_c, c in enumerate(raw_candidates):
        if not isinstance(c, dict):
            continue
        cid = normalize_fs_identifier(c, fallback_seed=f"{c.get('name') or idx_c}")
        if not cid or cid in seen_cids:
            continue
        seen_cids.add(cid)
        c_ark = str(c.get("fs_record_ark") or c.get("ark") or "").strip() or None
        c_url = str(c.get("fs_url") or "").strip() or _build_candidate_fs_url(cid, c_ark, fixed_fs_url)
        normalized_cands.append(
            {
                "fs_person_id": cid,
                "fs_id": cid,
                "fs_record_ark": c_ark,
                "fs_url": c_url,
                "name": str(c.get("name") or person_display),
                "sex": c.get("sex") or fact.sex,
                "lifespan": c.get("lifespan") or (f"*{birth_year}" if birth_year else None),
                "birth_date": c.get("birth_date") or birth_date_str,
                "birth_year": c.get("birth_year") or birth_year,
                "birth_place": c.get("birth_place") or std_place or fact.village_sk or fact.village_raw,
                "collection": c.get("collection"),
                "relatives_summary": c.get("relatives_summary"),
                "confidence_score": round(float(c.get("confidence_score") or m.confidence_score or 0.88), 2),
                "what_agrees": _clean_what_agrees(c.get("what_agrees") or what_agrees),
                "what_differs": c.get("what_differs") if isinstance(c.get("what_differs"), list) else what_differs,
                "selected": bool(c.get("selected", False)),
            }
        )

    if primary_id and primary_id not in seen_cids:
        seen_cids.add(primary_id)
        rels_parts = [
            f"Otec: {fact.father_name}" if fact.father_name else None,
            f"Matka: {fact.mother_name}" if fact.mother_name else None,
            f"Manžel/ka: {fact.spouse_name}" if fact.spouse_name else None,
        ]
        normalized_cands.insert(
            0,
            {
                "fs_person_id": primary_id,
                "fs_id": primary_id,
                "fs_record_ark": m.fs_record_ark,
                "fs_url": _build_candidate_fs_url(primary_id, m.fs_record_ark, fixed_fs_url),
                "name": str(person_display),
                "sex": fact.sex,
                "lifespan": f"*{birth_year}" if birth_year else None,
                "birth_date": birth_date_str,
                "birth_year": birth_year,
                "birth_place": std_place or fact.village_sk or fact.village_raw,
                "collection": None,
                "relatives_summary": ", ".join([p for p in rels_parts if p]) or None,
                "confidence_score": round(float(m.confidence_score or 0.88), 2),
                "what_agrees": what_agrees,
                "what_differs": what_differs,
                "selected": True,
            },
        )

    raw_selected: list[Any] = []
    try:
        parsed_s = json.loads(getattr(m, "selected_fs_person_ids_json", None) or "[]")
        if isinstance(parsed_s, list):
            raw_selected = parsed_s
    except Exception:
        raw_selected = []
    if not raw_selected and isinstance(action_packet.get("selected_fs_person_ids"), list):
        raw_selected = list(action_packet["selected_fs_person_ids"])

    selected_ids: list[str] = []
    for rs in raw_selected:
        sid = normalize_fs_identifier(rs)
        if sid and sid in seen_cids and sid not in selected_ids:
            selected_ids.append(sid)
    if not selected_ids:
        for c in normalized_cands:
            if c.get("selected") and c["fs_person_id"] not in selected_ids:
                selected_ids.append(c["fs_person_id"])
    if not selected_ids and primary_id:
        selected_ids = [primary_id]
    elif not selected_ids and normalized_cands:
        selected_ids = [normalized_cands[0]["fs_person_id"]]

    for c in normalized_cands:
        c["selected"] = c["fs_person_id"] in selected_ids

    return normalized_cands, selected_ids


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
        what_differs = json.loads(m.what_differs_json or "[]")
        candidates, selected_fs_person_ids = _hydrate_candidates_for_match(
            m=m,
            fact=fact,
            action_packet=action_packet,
            what_agrees=what_agrees,
            what_differs=what_differs,
            fixed_fs_url=fixed_fs_url,
            std_place=std_place,
        )
        primary_fs_id = selected_fs_person_ids[0] if selected_fs_person_ids else normalize_fs_identifier(m.fs_person_id)
        action_packet["candidates"] = candidates
        action_packet["selected_fs_person_ids"] = selected_fs_person_ids

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
            "fs_person_id": primary_fs_id,
            "selected_fs_person_ids": selected_fs_person_ids,
            "candidates": candidates,
            "fs_person_name": action_packet.get("person_name_display"),
            "fs_record_ark": m.fs_record_ark,
            "fs_url": fixed_fs_url,
            "confidence_score": m.confidence_score,
            "standardized_place": action_packet.get("birth_place_modern_sk") or std_place,
            "summary_sk": m.summary_sk,
            "explanation_sk": m.summary_sk,
            "what_agrees": what_agrees,
            "what_differs": what_differs,
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


async def _require_valid_fs_token() -> dict[str, Any]:
    token_valid, auth_status = await fs_mcp_manager.ensure_valid_token()
    if not token_valid:
        has_token = bool(auth_status.get("has_token_configured"))
        problem = auth_status.get("token_problem")
        if has_token:
            detail = (
                f"FamilySearch OAuth token je neplatný alebo exspirovaný ({problem})."
                if problem
                else "FamilySearch OAuth token je neplatný alebo exspirovaný. Zadajte platný token."
            )
        else:
            detail = "FamilySearch OAuth token nie je vložený. Vyhľadávanie a párovanie osôb cez FamilySearch MCP vyžaduje platný token."
        raise HTTPException(status_code=400, detail=detail)
    return auth_status


@router.post("/matches/{fact_id}/refresh")
async def refresh_fs_match(fact_id: int, session: SessionDep) -> dict[str, Any]:
    await _require_valid_fs_token()
    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    person_label = _format_fact_person_label(fact)
    bind = session.get_bind() or engine

    async def _do_refresh() -> dict[str, Any]:
        from sqlmodel import Session
        from backend.app.api.ws import ws_manager

        with Session(bind) as task_session:
            t_fact = task_session.get(PersonFact, fact_id)
            if not t_fact:
                raise HTTPException(status_code=404, detail="Fact not found")
            t_row = task_session.get(RecordRow, t_fact.row_id) if t_fact.row_id else None
            await fs_mcp_manager.start_progress(
                mode="auto",
                total=1,
                current_person=person_label,
                session=task_session,
            )
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "mcp_match",
                    "active_model": "familysearch-mcp",
                    "current_scan_id": t_fact.scan_id,
                    "current_row_id": t_fact.row_id,
                    "progress_pct": 50,
                    "message_sk": f"Obnovujem FamilySearch párovanie pre {person_label}…",
                    "message_en": f"Refreshing FamilySearch pairing for {person_label}…",
                },
            )
            try:
                match = await fs_mcp_manager.match_person_fact(task_session, t_fact, t_row)
                is_matched = bool(match and (match.fs_person_id or "").strip())
                await fs_mcp_manager.step_progress(
                    increment_processed=1,
                    increment_matched=1 if is_matched else 0,
                    current_person=person_label,
                    session=task_session,
                )
                progress = await fs_mcp_manager.finish_progress(session=task_session)
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "idle",
                        "active_model": None,
                        "current_scan_id": t_fact.scan_id,
                        "current_row_id": t_fact.row_id,
                        "progress_pct": 100,
                        "message_sk": f"FamilySearch párovanie dokončené ({person_label}).",
                        "message_en": f"FamilySearch pairing completed ({person_label}).",
                    },
                )
            except Exception as exc:
                await fs_mcp_manager.finish_progress(session=task_session, error=str(exc))
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "idle",
                        "active_model": None,
                        "progress_pct": 0,
                        "message_sk": f"Chyba pri FamilySearch párovaní ({person_label}): {exc}",
                        "message_en": f"FamilySearch pairing error ({person_label}): {exc}",
                    },
                )
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

    return await task_queue.run_or_enqueue(
        _do_refresh,
        task_type="fs_match_single",
        category="familysearch",
        title_sk=f"Párovanie s FamilySearch: {person_label}",
        title_en=f"FamilySearch pairing: {person_label}",
        initial_stage="mcp_match",
        active_model="familysearch-mcp",
        target_url="/familysearch",
        metadata={"fact_id": fact_id, "scan_id": fact.scan_id, "row_id": fact.row_id},
    )


@router.post("/match-row/{row_id}")
async def match_row_endpoint(row_id: int, session: SessionDep) -> dict[str, Any]:
    row = session.get(RecordRow, row_id)
    if not row:
        raise HTTPException(status_code=404, detail="Row not found")
    fact = session.exec(select(PersonFact).where(PersonFact.row_id == row_id)).first()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found for row")
    fact_id = int(fact.id or 0)
    return await refresh_fs_match(fact_id, session)


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
    if req.selected_fs_person_ids is not None:
        fact = session.get(PersonFact, match.person_fact_id)
        sheet = session.get(CensusSheet, fact.census_sheet_id) if (fact and fact.census_sheet_id) else None
        local_matches = (
            list(session.exec(select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id == fact.id)).all())
            if (fact and fact.id)
            else []
        )
        stored_packet = json.loads(match.action_packet_json or "{}")
        what_agrees = _clean_what_agrees(json.loads(match.what_agrees_json or "[]"))
        what_differs = json.loads(match.what_differs_json or "[]")
        cands, _ = _hydrate_candidates_for_match(
            m=match,
            fact=fact or PersonFact(row_id=0, scan_id=0),
            action_packet=stored_packet,
            what_agrees=what_agrees,
            what_differs=what_differs,
            fixed_fs_url=match.fs_url or "",
            std_place=fact.fs_place_standardized if fact else None,
        )
        apply_candidate_selection_to_match(
            match,
            cands,
            selected_ids=req.selected_fs_person_ids,
            fact=fact,
            sheet=sheet,
            local_matches=local_matches,
        )
    session.add(match)
    session.commit()
    progress = await fs_mcp_manager.broadcast_progress(session=session, refresh_counts=True)
    return {
        "status": "ok",
        "match_id": match.id,
        "reviewed_by_user": match.reviewed_by_user,
        "fs_person_id": match.fs_person_id,
        "selected_fs_person_ids": json.loads(match.selected_fs_person_ids_json or "[]"),
        "progress": progress,
    }


@router.post("/matches/{match_id}/candidates")
@router.post("/matches/{match_id}/select-candidates")
async def update_match_candidates_selection(
    match_id: int,
    req: SelectMatchCandidatesRequest,
    session: SessionDep,
) -> dict[str, Any]:
    """
    Updates the selected FamilySearch match candidates (`selected_fs_person_ids`, supporting
    multiple selections of possible matches, toggling a single candidate, or adding a custom
    `XXXX-YYY` FamilySearch identifier).
    """
    match = session.get(FamilySearchMatch, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    fact = session.get(PersonFact, match.person_fact_id)
    sheet = session.get(CensusSheet, fact.census_sheet_id) if (fact and fact.census_sheet_id) else None
    local_matches = (
        list(session.exec(select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id == fact.id)).all())
        if (fact and fact.id)
        else []
    )
    stored_packet = json.loads(match.action_packet_json or "{}")
    what_agrees = _clean_what_agrees(json.loads(match.what_agrees_json or "[]"))
    what_differs = json.loads(match.what_differs_json or "[]")
    cands, current_selected = _hydrate_candidates_for_match(
        m=match,
        fact=fact or PersonFact(row_id=0, scan_id=0),
        action_packet=stored_packet,
        what_agrees=what_agrees,
        what_differs=what_differs,
        fixed_fs_url=match.fs_url or "",
        std_place=fact.fs_place_standardized if fact else None,
    )

    if req.add_candidate_id and str(req.add_candidate_id).strip():
        raw_add = str(req.add_candidate_id).strip()
        if not is_valid_fs_identifier(raw_add):
            norm_add = normalize_fs_identifier(raw_add)
            if not is_valid_fs_identifier(norm_add):
                raise HTTPException(
                    status_code=400,
                    detail="Neplatný identifikátor FamilySearch. Použite formát XXXX-YYY (napr. KWCZ-001).",
                )
            add_id = norm_add
        else:
            add_id = raw_add.upper()

        if not any(c.get("fs_person_id") == add_id for c in cands):
            person_display = (
                (req.add_candidate_name or "").strip()
                or stored_packet.get("person_name_display")
                or (
                    f"{fact.given_name_sk or fact.given_name_raw or '?'} {fact.surname_raw or fact.surname_normalized or '?'}".strip()
                    if fact
                    else add_id
                )
            )
            birth_year = (fact.birth_year_est or fact.event_year) if fact else None
            cands.append(
                {
                    "fs_person_id": add_id,
                    "fs_id": add_id,
                    "fs_record_ark": None,
                    "fs_url": _build_candidate_fs_url(add_id, None, match.fs_url or ""),
                    "name": person_display,
                    "sex": fact.sex if fact else None,
                    "lifespan": f"*{birth_year}" if birth_year else None,
                    "birth_date": (fact.birth_or_event_date or (str(birth_year) if birth_year else None)) if fact else None,
                    "birth_year": birth_year,
                    "birth_place": (fact.fs_place_standardized or fact.village_sk or fact.village_raw) if fact else None,
                    "collection": None,
                    "relatives_summary": None,
                    "confidence_score": 0.90,
                    "what_agrees": [
                        {"field_sk": "Meno v strome / zázname FS", "value": f"{person_display} ({add_id})"}
                    ],
                    "what_differs": [],
                    "selected": True,
                }
            )
        if add_id not in current_selected:
            current_selected.append(add_id)

    next_selected = list(current_selected)
    if req.selected_fs_person_ids is not None:
        normalized_req_ids: list[str] = []
        for raw_sid in req.selected_fs_person_ids:
            sid = normalize_fs_identifier(raw_sid)
            if sid and sid not in normalized_req_ids:
                normalized_req_ids.append(sid)
        if not normalized_req_ids:
            raise HTTPException(
                status_code=400,
                detail="Vyberte aspoň jedného kandidáta zhody (XXXX-YYY).",
            )
        # Ensure any valid XXXX-YYY in selected_fs_person_ids that wasn't in cands is added to cands
        for sid in normalized_req_ids:
            if not any(c.get("fs_person_id") == sid for c in cands):
                p_disp = stored_packet.get("person_name_display") or sid
                cands.append(
                    {
                        "fs_person_id": sid,
                        "fs_id": sid,
                        "fs_record_ark": None,
                        "fs_url": _build_candidate_fs_url(sid, None, match.fs_url or ""),
                        "name": p_disp,
                        "confidence_score": round(float(match.confidence_score or 0.88), 2),
                        "what_agrees": [{"field_sk": "Meno v strome / zázname FS", "value": f"{p_disp} ({sid})"}],
                        "what_differs": [],
                        "selected": True,
                    }
                )
        next_selected = normalized_req_ids
    elif req.toggle_fs_person_id and str(req.toggle_fs_person_id).strip():
        toggle_id = normalize_fs_identifier(req.toggle_fs_person_id)
        if not toggle_id:
            raise HTTPException(status_code=400, detail="Neplatný identifikátor kandidáta (XXXX-YYY).")
        if toggle_id in next_selected:
            if len(next_selected) > 1:
                next_selected = [sid for sid in next_selected if sid != toggle_id]
        else:
            next_selected.append(toggle_id)

    updated_cands, final_selected = apply_candidate_selection_to_match(
        match,
        cands,
        selected_ids=next_selected,
        fact=fact,
        sheet=sheet,
        local_matches=local_matches,
    )
    session.add(match)
    session.commit()
    session.refresh(match)

    return {
        "status": "ok",
        "match_id": match.id,
        "match_status": match.match_status,
        "fs_person_id": match.fs_person_id,
        "selected_fs_person_ids": final_selected,
        "candidates": updated_cands,
        "confidence_score": match.confidence_score,
        "summary_sk": match.summary_sk,
        "explanation_sk": match.summary_sk,
        "what_agrees": _clean_what_agrees(json.loads(match.what_agrees_json or "[]")),
        "what_differs": json.loads(match.what_differs_json or "[]"),
    }


@router.post("/sync-all")
@router.post("/rematch")
async def sync_all_fs_matches(
    session: SessionDep,
    background: bool = False,
) -> dict[str, Any]:
    await _require_valid_fs_token()
    bind = session.get_bind() or engine

    async def _run_sync_all() -> dict[str, Any]:
        import time as _time
        from sqlmodel import Session
        from backend.app.api.ws import ws_manager

        t0 = _time.perf_counter()
        with Session(bind) as task_session:
            cleanup_unmatched_familysearch_rows(task_session)
            facts = list(task_session.exec(select(PersonFact).order_by(PersonFact.id.asc())).all())

            sheet_ids = list({f.census_sheet_id for f in facts if f.census_sheet_id is not None})
            sheets_by_id = (
                {cs.id: cs for cs in task_session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all()}
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
                session=task_session,
            )
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "mcp_match",
                    "active_model": "familysearch-mcp",
                    "current_row_index": 0,
                    "total_rows": total_to_check,
                    "progress_pct": 10,
                    "message_sk": f"Spúšťam párovanie s FamilySearch MCP ({total_to_check} osôb)…",
                    "message_en": f"Starting FamilySearch MCP pairing ({total_to_check} persons)…",
                },
            )

            synced = 0
            synced_sheet_ids: set[int] = set()
            try:
                for fact in facts:
                    if task_queue.is_current_task_cancelled():
                        raise RuntimeError("FamilySearch pairing cancelled")
                    if _is_nameless_fact(fact):
                        continue
                    if fact.record_type == "census" and fact.census_sheet_id:
                        if fact.census_sheet_id not in synced_sheet_ids:
                            synced_sheet_ids.add(fact.census_sheet_id)
                            sheet = sheets_by_id.get(fact.census_sheet_id) or task_session.get(
                                CensusSheet, fact.census_sheet_id
                            )
                            if sheet and getattr(sheet, "sheet_type", "household") not in ("overview", "municipal_overview"):
                                synced += await fs_mcp_manager.match_census_sheet(
                                    task_session,
                                    sheet,
                                    manage_progress=False,
                                )
                        continue
                    person_label = _format_fact_person_label(fact)
                    await fs_mcp_manager.step_progress(current_person=person_label, session=task_session)
                    proc_now = int(fs_mcp_manager.progress_state.get("processed") or 0) + 1
                    pct_now = min(98, max(10, int(100 * proc_now / max(1, total_to_check))))
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "mcp_match",
                            "active_model": "familysearch-mcp",
                            "current_scan_id": fact.scan_id,
                            "current_row_id": fact.row_id,
                            "current_row_index": proc_now,
                            "total_rows": total_to_check,
                            "progress_pct": pct_now,
                            "message_sk": f"Párujem s FamilySearch ({proc_now}/{total_to_check}): {person_label}…",
                            "message_en": f"Matching with FamilySearch ({proc_now}/{total_to_check}): {person_label}…",
                        },
                    )
                    row = task_session.get(RecordRow, fact.row_id) if fact.row_id else None
                    m = await fs_mcp_manager.match_person_fact(task_session, fact, row)
                    is_matched = bool(m and (m.fs_person_id or "").strip())
                    if is_matched:
                        synced += 1
                    await fs_mcp_manager.step_progress(
                        increment_processed=1,
                        increment_matched=1 if is_matched else 0,
                        current_person=person_label,
                        session=task_session,
                    )
                cleanup_unmatched_familysearch_rows(task_session)
                progress = await fs_mcp_manager.finish_progress(session=task_session)
                dur_sec = max(1, int(round(_time.perf_counter() - t0)))
                done_sk = f"FamilySearch párovanie dokončené ({synced} zhôd z {total_to_check} skontrolovaných osôb)."
                done_en = f"FamilySearch pairing completed ({synced} matches out of {total_to_check} checked persons)."
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "idle",
                        "active_model": None,
                        "current_row_index": total_to_check,
                        "total_rows": total_to_check,
                        "progress_pct": 100,
                        "message_sk": done_sk,
                        "message_en": done_en,
                    },
                )
                await ws_manager.emit_completion(
                    {
                        "status": "completed",
                        "kind": "familysearch_rematch",
                        "rows_processed": total_to_check,
                        "matched_count": synced,
                        "duration_sec": dur_sec,
                        "title_sk": "FamilySearch párovanie dokončené",
                        "title_en": "FamilySearch pairing completed",
                        "message_sk": done_sk,
                        "message_en": done_en,
                        "target_url": "/familysearch",
                    }
                )
            except Exception as exc:
                await fs_mcp_manager.finish_progress(session=task_session, error=str(exc))
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "idle",
                        "active_model": None,
                        "progress_pct": 0,
                        "message_sk": f"Chyba pri párovaní s FamilySearch: {exc}",
                        "message_en": f"Error during FamilySearch pairing: {exc}",
                    },
                )
                raise
            return {
                "status": "ok",
                "synced_count": synced,
                "matched_count": synced,
                "progress": progress,
            }

    if background:
        queued_task = task_queue.enqueue_background(
            _run_sync_all,
            task_type="fs_rematch_all",
            category="familysearch",
            title_sk="Párovanie s FamilySearch MCP (všetky osoby)",
            title_en="FamilySearch MCP Pairing (all persons)",
            initial_stage="mcp_match",
            active_model="familysearch-mcp",
            target_url="/familysearch",
        )
        return {
            "status": "queued" if queued_task.get("status") == "queued" else "started",
            "task": queued_task,
            "progress": fs_mcp_manager.get_progress_state(session),
        }

    return await task_queue.run_or_enqueue(
        _run_sync_all,
        task_type="fs_rematch_all",
        category="familysearch",
        title_sk="Párovanie s FamilySearch MCP (všetky osoby)",
        title_en="FamilySearch MCP Pairing (all persons)",
        initial_stage="mcp_match",
        active_model="familysearch-mcp",
        target_url="/familysearch",
    )


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
