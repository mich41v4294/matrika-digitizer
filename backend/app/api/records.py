from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from backend.app.db.models import CensusOverviewHouse, CensusSheet, FamilySearchMatch, PersonFact, RecordRow, ScanImage
from backend.app.db.session import SessionDep, delete_person_fact_cascade
from backend.app.export.gedcom import export_facts_to_csv, export_facts_to_gedcom
from backend.app.mcp.familysearch_client import (
    fs_mcp_manager,
    schedule_auto_fs_match,
    trigger_auto_fs_match_for_fact,
)
from backend.app.pipeline.feedback import build_feedback_lookup_maps
from backend.app.pipeline.lexicon import (
    apply_person_fact_search_fields,
    canonical_municipality_key,
    compute_person_fact_search_fields,
    normalize_house_number,
    parse_observationes_events,
)
from backend.app.pipeline.schemas import (
    validate_event_year,
    validate_house_number,
    validate_matrika_date,
    validate_person_name,
    validate_sex,
)
from backend.app.search.fts import (
    get_search_progress,
    index_person_fact,
    rebuild_all_fts,
    search_unified,
)

router = APIRouter(prefix="/api/records", tags=["records"])


class UpdateFactRequest(BaseModel):
    entry_number: str | None = None
    given_name_sk: str | None = None
    given_name_raw: str | None = None
    surname_raw: str | None = None
    surname_normalized: str | None = None
    sex: str | None = None
    birth_or_event_date: str | None = None
    baptism_or_burial_date: str | None = None
    event_year: int | None = None
    birth_year_est: int | None = None
    father_name: str | None = None
    mother_name: str | None = None
    mother_maiden_surname: str | None = None
    village_raw: str | None = None
    village_sk: str | None = None
    village_hu: str | None = None
    house_number: str | None = None
    godparents_or_witnesses: str | None = None
    relation_to_head: str | None = None
    social_status_or_occupation: str | None = None
    birth_place_raw: str | None = None
    observationes_raw: str | None = None
    spouse_name: str | None = None


class CreateFactRequest(UpdateFactRequest):
    scan_id: int | None = None
    row_id: int | None = None
    census_sheet_id: int | None = None
    record_type: str = "baptism"


@router.get("")
@router.get("/search")
def list_records(
    session: SessionDep,
    q: str | None = None,
    surname: str | None = None,
    record_type: str | None = None,
    scan_id: int | None = None,
    village: str | None = None,
    street: str | None = None,
    house_number: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    include_catalog_sheets: bool = True,
    fuzzy: bool = True,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    res = search_unified(
        session,
        q=q,
        surname=surname,
        record_type=record_type,
        municipality=village,
        street=street,
        house_number=house_number,
        year_from=year_from,
        year_to=year_to,
        scan_id=scan_id,
        include_catalog_sheets=include_catalog_sheets,
        fuzzy=fuzzy,
        limit=limit,
        offset=offset,
    )
    rec_items = res.get("records", [])
    rec_fact_ids = [item["fact"].id for item in rec_items if item.get("fact") and item["fact"].id is not None]
    fb_maps = build_feedback_lookup_maps(session, fact_ids=rec_fact_ids) if rec_fact_ids else {"by_fact": {}, "by_row": {}}
    for item in rec_items:
        f = item["fact"]
        item["observationes_events"] = json.loads(f.observationes_events_json or "[]")
        item["ocr_feedback"] = (
            fb_maps["by_fact"].get(f.id or 0)
            or fb_maps["by_row"].get(f.row_id or 0)
        )
    cat_sheets_raw = res.get("catalog_sheets", [])
    cat_owner_by_id: dict[int, str] = {}
    cat_owner_by_muni_year_hn: dict[tuple[str, int, str], str] = {}
    if cat_sheets_raw:
        cat_years = list({int(s.census_year or 1930) for s in cat_sheets_raw})
        for ovh in session.exec(
            select(CensusOverviewHouse).where(CensusOverviewHouse.census_year.in_(cat_years))  # type: ignore[union-attr]
        ).all():
            if not ovh.owner_name:
                continue
            if ovh.matched_sheet_id and ovh.matched_sheet_id not in cat_owner_by_id:
                cat_owner_by_id[ovh.matched_sheet_id] = ovh.owner_name
            m_k = canonical_municipality_key(ovh.municipality)
            hn_k = normalize_house_number(ovh.house_number)
            if m_k and hn_k:
                key_mh = (m_k, int(ovh.census_year or 1930), hn_k)
                if key_mh not in cat_owner_by_muni_year_hn:
                    cat_owner_by_muni_year_hn[key_mh] = ovh.owner_name

    res["catalog_sheets"] = [
        {
            **s.model_dump(),
            "household_owner": (
                s.household_owner
                or cat_owner_by_id.get(s.id or 0)
                or cat_owner_by_muni_year_hn.get(
                    (
                        canonical_municipality_key(s.municipality),
                        int(s.census_year or 1930),
                        normalize_house_number(s.house_number),
                    )
                )
            ),
            "slovakiana_id": s.slovakiana_key,
            "district_name": s.district,
            "municipality_name": s.municipality,
            "street_name": s.street_raw,
            "house_number_raw": s.house_number,
            "house_number_conscription": s.house_number,
            "pipeline_status": s.download_status,
        }
        for s in cat_sheets_raw
    ]
    return res


search_records_endpoint = list_records


@router.get("/search-progress")
def get_records_search_progress() -> dict[str, Any]:
    return get_search_progress()


@router.post("/reindex")
def reindex_records_fts(session: SessionDep) -> dict[str, Any]:
    counts = rebuild_all_fts(session)
    return {"status": "reindexed", **counts}


@router.post("")
async def create_record(
    req: CreateFactRequest,
    session: SessionDep,
) -> dict[str, Any]:
    scan_id = req.scan_id or 0
    row_id = req.row_id
    if not row_id:
        row = RecordRow(
            scan_id=scan_id,
            row_index=1,
            y_start=0,
            y_end=100,
            status="extracted",
            record_type=req.record_type,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        row_id = row.id or 0

    fact = PersonFact(
        row_id=row_id,
        scan_id=scan_id,
        census_sheet_id=req.census_sheet_id,
        record_type=req.record_type,
    )
    session.add(fact)
    session.commit()
    session.refresh(fact)

    update_payload = req.model_dump(
        exclude={"scan_id", "row_id", "census_sheet_id", "record_type"},
        exclude_unset=True,
    )
    res = await update_record(fact.id or 0, UpdateFactRequest(**update_payload), session)
    return {"status": "created", "fact": res["fact"]}


@router.put("/{fact_id}")
async def update_record(
    fact_id: int,
    req: UpdateFactRequest,
    session: SessionDep,
) -> dict[str, Any]:
    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="PersonFact not found")

    update_data = req.model_dump(exclude_unset=True)
    if "birth_or_event_date" in update_data and update_data["birth_or_event_date"]:
        clean_b, _ = validate_matrika_date(update_data["birth_or_event_date"], entry_number=fact.entry_number)
        if clean_b:
            update_data["birth_or_event_date"] = clean_b
    if "baptism_or_burial_date" in update_data and update_data["baptism_or_burial_date"]:
        clean_bp, _ = validate_matrika_date(update_data["baptism_or_burial_date"], entry_number=fact.entry_number)
        if clean_bp:
            update_data["baptism_or_burial_date"] = clean_bp
    if "father_name" in update_data and update_data["father_name"]:
        clean_f, _ = validate_person_name(update_data["father_name"], role="father")
        if clean_f:
            update_data["father_name"] = clean_f
    if "mother_name" in update_data and update_data["mother_name"]:
        clean_m, _ = validate_person_name(update_data["mother_name"], role="mother")
        if clean_m:
            update_data["mother_name"] = clean_m
    if "event_year" in update_data and update_data["event_year"] is not None:
        update_data["event_year"] = validate_event_year(update_data["event_year"], fact.event_year)
    if "house_number" in update_data and update_data["house_number"] is not None:
        update_data["house_number"] = validate_house_number(update_data["house_number"], fact.village_raw)

    for k, v in update_data.items():
        setattr(fact, k, v)
    if "observationes_raw" in update_data:
        obs_events = parse_observationes_events(fact.observationes_raw)
        fact.observationes_events_json = json.dumps(obs_events, ensure_ascii=False)
    if fact.given_name_raw:
        fact.sex = validate_sex(fact.sex, fact.given_name_raw)

    apply_person_fact_search_fields(fact)

    session.add(fact)
    session.flush()
    index_person_fact(session, fact)
    session.commit()
    session.refresh(fact)

    schedule_auto_fs_match(fact.id, trigger_auto_fs_match_for_fact)

    return {"status": "updated", "fact": fact}


@router.delete("/{fact_id}")
def delete_record(fact_id: int, session: SessionDep) -> dict[str, Any]:
    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="PersonFact not found")
    census_sheet_id = fact.census_sheet_id
    deleted = delete_person_fact_cascade(session, fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="PersonFact not found")
    if census_sheet_id:
        try:
            from backend.app.pipeline.extraction_stage import reconcile_census_household_relationships

            reconcile_census_household_relationships(session, census_sheet_id)
        except Exception:
            pass
    return {"status": "deleted", "fact_id": fact_id}


def _query_export_facts(session: SessionDep, only_new_to_fs: bool) -> list[PersonFact]:
    stmt = select(PersonFact)
    if only_new_to_fs:
        matched_subq = (
            select(FamilySearchMatch.person_fact_id)
            .where(
                FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
                func.trim(FamilySearchMatch.fs_person_id) != "",
                FamilySearchMatch.confidence_score > 0,
            )
            .distinct()
        )
        stmt = stmt.where(PersonFact.id.in_(matched_subq))  # type: ignore[union-attr]
    return list(session.exec(stmt.order_by(PersonFact.id.asc())).all())


@router.get("/export/csv", response_class=PlainTextResponse)
def download_csv(session: SessionDep, only_new_to_fs: bool = False) -> PlainTextResponse:
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "export_facts_to_csv_rs"):
            csv_text = _rust_db.export_facts_to_csv_rs(db_p, only_new_to_fs)
            return PlainTextResponse(
                content=csv_text,
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="matrika_records.csv"'},
            )
    except Exception:
        pass
    facts = _query_export_facts(session, only_new_to_fs)

    sheet_ids = list({f.census_sheet_id for f in facts if f.census_sheet_id is not None})
    sheets_by_id = (
        {cs.id: cs for cs in session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all() if cs.id is not None}
        if sheet_ids
        else {}
    )
    csv_text = export_facts_to_csv(facts, sheets_by_id=sheets_by_id)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="matrika_records.csv"'},
    )


@router.get("/export/gedcom", response_class=PlainTextResponse)
def download_gedcom(session: SessionDep, only_new_to_fs: bool = False) -> PlainTextResponse:
    try:
        from datetime import datetime, timezone
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "export_facts_to_gedcom_rs"):
            today_str = datetime.now(timezone.utc).strftime("%d %b %Y").upper()
            ged_text = _rust_db.export_facts_to_gedcom_rs(db_p, today_str, only_new_to_fs)
            return PlainTextResponse(
                content=ged_text,
                media_type="text/plain; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="matrika_export.ged"'},
            )
    except Exception:
        pass
    facts = _query_export_facts(session, only_new_to_fs)

    sheet_ids = list({f.census_sheet_id for f in facts if f.census_sheet_id is not None})
    sheets_by_id = (
        {cs.id: cs for cs in session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all() if cs.id is not None}
        if sheet_ids
        else {}
    )
    ged_text = export_facts_to_gedcom(facts, sheets_by_id=sheets_by_id)
    return PlainTextResponse(
        content=ged_text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="matrika_export.ged"'},
    )
