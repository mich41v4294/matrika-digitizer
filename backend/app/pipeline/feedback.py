from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.db.models import (
    CensusMunicipalOverviewDistrict,
    CensusOverviewHouse,
    CensusSheet,
    OCRFeedback,
    PersonFact,
    RecordRow,
    ScanImage,
    WorkflowTrace,
    utcnow,
)

try:
    import matrika_db_rs as _rust_db
except ImportError:
    _rust_db = None


def _session_file_db_path(session: Session) -> str | None:
    if _rust_db is None:
        return None
    try:
        if session.new or session.dirty or session.deleted:
            return None
        if session.in_transaction():
            raw_conn = getattr(
                getattr(session.connection(), "connection", None),
                "dbapi_connection",
                None,
            )
            if raw_conn is not None and getattr(raw_conn, "in_transaction", False):
                return None
        bind = session.get_bind()
        db_name = getattr(getattr(bind, "url", None), "database", None)
        if not db_name or db_name == ":memory:" or str(db_name).startswith("file::memory:"):
            return None
        p = Path(str(db_name))
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


def _rust_model_defaults_args() -> tuple[str, str, str, str, list[str]]:
    r_ocr = settings.resolve_stage_ai("ocr")
    r_dec = settings.resolve_stage_ai("decision")
    r_ext = settings.resolve_stage_ai("extract")
    r_lay = settings.resolve_stage_ai("layout")
    known = [
        settings.local_model_ocr,
        settings.local_model_decision,
        settings.local_model_extract,
        settings.local_model_layout,
    ]
    return (
        f"{r_ocr['provider']}:{r_ocr['model']}",
        f"{r_dec['provider']}:{r_dec['model']}",
        f"{r_ext['provider']}:{r_ext['model']}",
        f"{r_lay['provider']}:{r_lay['model']}",
        known,
    )


def normalize_model_tag(raw: str | None, stage_hint: str = "extract") -> str:
    val = str(raw or "").strip()
    if not val:
        resolved = settings.resolve_stage_ai(stage_hint)
        return f"{resolved['provider']}:{resolved['model']}"
    if ":" in val:
        return val
    low = val.lower()
    known_local = {
        settings.local_model_ocr.lower(),
        settings.local_model_decision.lower(),
        settings.local_model_extract.lower(),
        settings.local_model_layout.lower(),
    }
    if low in known_local or any(tok in low for tok in ("mlx", "saluki", "glm-ocr", "d1-3b")):
        return f"local:{val}"
    return f"freellm:{val}"


def canonical_stage_group(stage: str | None) -> str:
    s = str(stage or "").strip().lower()
    if s in ("layout", "census_layout") or "layout" in s:
        return "layout"
    if s in ("ocr", "ocr_header", "ocr_left", "ocr_right", "census_overview_ocr") or s.startswith("ocr"):
        return "ocr"
    if s == "decision" or "decision" in s:
        return "decision"
    if s in ("extract", "census_extract", "census_overview_extract") or "extract" in s:
        return "extract"
    if s.startswith("mcp"):
        return "mcp"
    return "other"


def format_model_combo(
    ocr_model: str,
    extract_model: str,
    decision_model: str | None = None,
    *,
    is_census: bool = True,
) -> str:
    ocr_m = normalize_model_tag(ocr_model, "ocr")
    ext_m = normalize_model_tag(extract_model, "extract")
    dec_m = (decision_model or "").strip()
    has_llm_decision = bool(
        dec_m
        and not is_census
        and not dec_m.startswith(("deterministic:", "skipped", "none"))
    )
    if has_llm_decision:
        dec_norm = normalize_model_tag(dec_m, "decision")
        return f"OCR: {ocr_m} + Decision: {dec_norm} + Extract: {ext_m}"
    return f"OCR: {ocr_m} + Extract: {ext_m}"


def _serialize_trace_for_snapshot(t: WorkflowTrace) -> dict[str, Any]:
    parsed_obj: Any = {}
    try:
        parsed_obj = json.loads(t.parsed_json or "{}")
    except Exception:
        parsed_obj = t.parsed_json
    return {
        "id": t.id,
        "scan_id": t.scan_id,
        "row_id": t.row_id,
        "stage": t.stage,
        "canonical_stage": canonical_stage_group(t.stage),
        "model_or_tool": t.model_or_tool,
        "status": t.status,
        "duration_ms": int(t.duration_ms or 0),
        "prompt_tokens": int(t.prompt_tokens or 0),
        "completion_tokens": int(t.completion_tokens or 0),
        "reasoning_tokens": int(t.reasoning_tokens or 0),
        "image_path": t.image_path,
        "prompt_text": t.prompt_text,
        "reasoning_content": t.reasoning_content,
        "response_text": t.response_text,
        "parsed_json": parsed_obj,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


def collect_row_execution_context(
    session: Session,
    *,
    target_type: str = "row",
    row_id: int | None = None,
    scan_id: int | None = None,
    census_sheet_id: int | None = None,
    person_fact_id: int | None = None,
    overview_house_id: int | None = None,
    municipal_district_id: int | None = None,
) -> dict[str, Any]:
    fact: PersonFact | None = session.get(PersonFact, person_fact_id) if person_fact_id else None
    house: CensusOverviewHouse | None = (
        session.get(CensusOverviewHouse, overview_house_id) if overview_house_id else None
    )
    dist: CensusMunicipalOverviewDistrict | None = (
        session.get(CensusMunicipalOverviewDistrict, municipal_district_id)
        if municipal_district_id
        else None
    )

    if fact:
        row_id = row_id or fact.row_id
        scan_id = scan_id or fact.scan_id
        census_sheet_id = census_sheet_id or fact.census_sheet_id
        if target_type == "row":
            target_type = "person_fact"

    if house:
        scan_id = scan_id or house.scan_id
        census_sheet_id = census_sheet_id or house.census_sheet_id
        if target_type == "row":
            target_type = "overview_house"
        if not row_id and scan_id:
            cand_rows = session.exec(
                select(RecordRow).where(RecordRow.scan_id == scan_id).order_by(RecordRow.row_index.asc())
            ).all()
            for r_cand in cand_rows:
                if house.stitched_crop_path and r_cand.stitched_crop_path == house.stitched_crop_path:
                    row_id = r_cand.id
                    break
                if r_cand.row_index == house.row_index:
                    row_id = r_cand.id
                    break

    if dist:
        scan_id = scan_id or dist.scan_id
        census_sheet_id = census_sheet_id or dist.census_sheet_id
        if target_type == "row":
            target_type = "municipal_district"
        if not row_id and scan_id:
            cand_rows = session.exec(
                select(RecordRow).where(RecordRow.scan_id == scan_id).order_by(RecordRow.row_index.asc())
            ).all()
            for r_cand in cand_rows:
                if dist.stitched_crop_path and r_cand.stitched_crop_path == dist.stitched_crop_path:
                    row_id = r_cand.id
                    break
                if r_cand.row_index == dist.row_index:
                    row_id = r_cand.id
                    break

    row: RecordRow | None = session.get(RecordRow, row_id) if row_id else None
    if row:
        scan_id = scan_id or row.scan_id
        if not fact and row.id is not None:
            fact = session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()
            if fact and fact.id is not None:
                person_fact_id = fact.id
                census_sheet_id = census_sheet_id or fact.census_sheet_id

    scan: ScanImage | None = session.get(ScanImage, scan_id) if scan_id else None
    if scan and not census_sheet_id:
        census_sheet_id = scan.census_sheet_id

    sheet: CensusSheet | None = session.get(CensusSheet, census_sheet_id) if census_sheet_id else None
    if sheet and not scan:
        fallback_sid = sheet.front_scan_id or sheet.back_scan_id
        if fallback_sid:
            scan = session.get(ScanImage, fallback_sid)
            scan_id = scan.id if scan else None

    if row and sheet and not fact and not house and not dist:
        if sheet.sheet_type == "overview":
            house = session.exec(
                select(CensusOverviewHouse).where(
                    CensusOverviewHouse.census_sheet_id == sheet.id,
                    (CensusOverviewHouse.stitched_crop_path == row.stitched_crop_path)
                    | (CensusOverviewHouse.row_index == row.row_index),
                )
            ).first()
            if house and house.id is not None:
                overview_house_id = house.id
        elif sheet.sheet_type == "municipal_overview":
            dist = session.exec(
                select(CensusMunicipalOverviewDistrict).where(
                    CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id,
                    (CensusMunicipalOverviewDistrict.stitched_crop_path == row.stitched_crop_path)
                    | (CensusMunicipalOverviewDistrict.row_index == row.row_index),
                )
            ).first()
            if dist and dist.id is not None:
                municipal_district_id = dist.id

    is_census = bool(
        census_sheet_id is not None
        or (scan and scan.detected_type == "census")
        or (row and row.record_type == "census")
        or (fact and fact.record_type == "census")
    )

    scan_rows_count = 1
    if scan_id:
        all_scan_rows = session.exec(select(RecordRow.id).where(RecordRow.scan_id == scan_id)).all()
        scan_rows_count = max(1, len(all_scan_rows))

    # Gather relevant WorkflowTrace entries
    row_traces: list[WorkflowTrace] = []
    if row_id:
        row_traces = list(
            session.exec(
                select(WorkflowTrace)
                .where(WorkflowTrace.row_id == row_id)
                .order_by(WorkflowTrace.id.asc())
            ).all()
        )

    scan_traces: list[WorkflowTrace] = []
    if scan_id:
        scan_traces = list(
            session.exec(
                select(WorkflowTrace)
                .where(WorkflowTrace.scan_id == scan_id)
                .order_by(WorkflowTrace.id.asc())
            ).all()
        )

    # Also match traces by crop path if row_id was not set on cached overview traces
    if row and scan_traces:
        row_crops = {
            c
            for c in (row.left_crop_path, row.right_crop_path, row.stitched_crop_path)
            if c
        }
        try:
            for item in json.loads(row.cell_crops_json or "[]"):
                if isinstance(item, dict) and item.get("crop_url"):
                    row_crops.add(str(item["crop_url"]))
        except Exception:
            pass
        seen_tids = {t.id for t in row_traces if t.id is not None}
        for st in scan_traces:
            if st.id not in seen_tids and st.image_path and st.image_path in row_crops:
                row_traces.append(st)
                if st.id is not None:
                    seen_tids.add(st.id)
        row_traces.sort(key=lambda x: x.id or 0)

    # 1. Layout step (scan-level)
    layout_traces = [
        t for t in scan_traces if canonical_stage_group(t.stage) == "layout"
    ]
    latest_layout_trace = layout_traces[-1] if layout_traces else None
    scan_layout_ms = int(latest_layout_trace.duration_ms or 0) if latest_layout_trace else 0
    layout_duration_ms = round(scan_layout_ms / max(1, scan_rows_count)) if scan_layout_ms > 0 else 0
    layout_model = (
        latest_layout_trace.model_or_tool
        if latest_layout_trace
        else ("opencv:census_sheet_slicer" if is_census else "opencv:angled_quad_row_slicer")
    )

    # 2. OCR step (header + row left/right/cells)
    header_ocr_traces = [t for t in scan_traces if t.stage == "ocr_header"]
    latest_header_ocr = header_ocr_traces[-1] if header_ocr_traces else None
    scan_ocr_header_ms = int(latest_header_ocr.duration_ms or 0) if latest_header_ocr else 0
    ocr_header_ms = round(scan_ocr_header_ms / max(1, scan_rows_count)) if scan_ocr_header_ms > 0 else 0

    ocr_row_candidates = [
        t
        for t in row_traces
        if canonical_stage_group(t.stage) == "ocr" and t.stage != "ocr_header"
    ]
    # Deduplicate by (stage, image_path) keeping the latest run
    dedup_ocr_map: dict[tuple[str, str], WorkflowTrace] = {}
    for t in ocr_row_candidates:
        dedup_ocr_map[(t.stage, t.image_path or "")] = t
    latest_ocr_traces = list(dedup_ocr_map.values())

    ocr_left_ms = sum(
        int(t.duration_ms or 0)
        for t in latest_ocr_traces
        if t.stage in ("ocr_left", "ocr", "census_overview_ocr")
    )
    ocr_right_ms = sum(
        int(t.duration_ms or 0)
        for t in latest_ocr_traces
        if t.stage == "ocr_right"
    )
    ocr_duration_ms = ocr_left_ms + ocr_right_ms + ocr_header_ms

    raw_ocr_model: str | None = None
    if latest_ocr_traces:
        raw_ocr_model = latest_ocr_traces[-1].model_or_tool
    elif latest_header_ocr:
        raw_ocr_model = latest_header_ocr.model_or_tool
    else:
        any_scan_ocr = [t for t in scan_traces if canonical_stage_group(t.stage) == "ocr"]
        if any_scan_ocr:
            raw_ocr_model = any_scan_ocr[-1].model_or_tool
    ocr_model = normalize_model_tag(raw_ocr_model, "ocr")

    # 3. Decision step
    decision_candidates = [t for t in row_traces if canonical_stage_group(t.stage) == "decision"]
    latest_decision_trace = decision_candidates[-1] if decision_candidates else None
    decision_duration_ms = int(latest_decision_trace.duration_ms or 0) if latest_decision_trace else 0
    if latest_decision_trace:
        decision_model = (
            latest_decision_trace.model_or_tool
            if latest_decision_trace.model_or_tool.startswith("deterministic:")
            else normalize_model_tag(latest_decision_trace.model_or_tool, "decision")
        )
    else:
        decision_model = (
            "deterministic:census_gazetteer"
            if is_census
            else normalize_model_tag(None, "decision")
        )

    # 4. Extraction step
    extract_candidates = [t for t in row_traces if canonical_stage_group(t.stage) == "extract"]
    latest_extract_trace = extract_candidates[-1] if extract_candidates else None
    extract_duration_ms = int(latest_extract_trace.duration_ms or 0) if latest_extract_trace else 0
    raw_extract_model: str | None = None
    if latest_extract_trace:
        raw_extract_model = latest_extract_trace.model_or_tool
    else:
        any_scan_ext = [t for t in scan_traces if canonical_stage_group(t.stage) == "extract"]
        if any_scan_ext:
            raw_extract_model = any_scan_ext[-1].model_or_tool
    extract_model = normalize_model_tag(raw_extract_model, "extract")

    # 5. MCP step
    mcp_candidates = [t for t in row_traces if canonical_stage_group(t.stage) == "mcp"]
    if not mcp_candidates and scan_traces and not row_id:
        mcp_candidates = [t for t in scan_traces if canonical_stage_group(t.stage) == "mcp"]
    dedup_mcp_map: dict[str, WorkflowTrace] = {}
    for t in mcp_candidates:
        dedup_mcp_map[f"{t.stage}:{t.model_or_tool}"] = t
    latest_mcp_traces = list(dedup_mcp_map.values())
    mcp_duration_ms = sum(int(t.duration_ms or 0) for t in latest_mcp_traces)
    mcp_model = latest_mcp_traces[-1].model_or_tool if latest_mcp_traces else None

    total_duration_ms = (
        layout_duration_ms
        + ocr_duration_ms
        + decision_duration_ms
        + extract_duration_ms
        + mcp_duration_ms
    )

    model_combo = format_model_combo(
        ocr_model,
        extract_model,
        decision_model,
        is_census=is_census,
    )

    combined_traces_for_row: list[WorkflowTrace] = []
    for t in [
        *([latest_layout_trace] if latest_layout_trace else []),
        *([latest_header_ocr] if latest_header_ocr else []),
        *latest_ocr_traces,
        *([latest_decision_trace] if latest_decision_trace else []),
        *([latest_extract_trace] if latest_extract_trace else []),
        *latest_mcp_traces,
    ]:
        if t is not None:
            combined_traces_for_row.append(t)

    prompt_tokens = sum(int(t.prompt_tokens or 0) for t in combined_traces_for_row)
    completion_tokens = sum(int(t.completion_tokens or 0) for t in combined_traces_for_row)
    reasoning_tokens = sum(int(t.reasoning_tokens or 0) for t in combined_traces_for_row)

    stage_breakdown = {
        "layout_ms": layout_duration_ms,
        "scan_layout_ms": scan_layout_ms,
        "ocr_ms": ocr_duration_ms,
        "ocr_header_ms": ocr_header_ms,
        "scan_ocr_header_ms": scan_ocr_header_ms,
        "ocr_left_ms": ocr_left_ms,
        "ocr_right_ms": ocr_right_ms,
        "ocr_calls_count": len(latest_ocr_traces) + (1 if latest_header_ocr else 0),
        "decision_ms": decision_duration_ms,
        "extract_ms": extract_duration_ms,
        "mcp_ms": mcp_duration_ms,
        "total_ms": total_duration_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "scan_rows_count": scan_rows_count,
    }

    # Build raw OCR snapshot & extracted snapshot
    raw_ocr_snapshot = (
        (row.raw_ocr_combined if row and row.raw_ocr_combined else None)
        or (row.raw_ocr_left if row and row.raw_ocr_left else None)
        or (house.raw_ocr_text if house and house.raw_ocr_text else None)
        or (dist.raw_ocr_text if dist and dist.raw_ocr_text else None)
        or (scan.header_raw_ocr if scan and scan.header_raw_ocr else None)
    )

    crop_url = (
        (row.stitched_crop_path if row and row.stitched_crop_path else None)
        or (house.stitched_crop_path if house and house.stitched_crop_path else None)
        or (dist.stitched_crop_path if dist and dist.stitched_crop_path else None)
        or (row.left_crop_path if row and row.left_crop_path else None)
        or (scan.header_crop_path if scan and scan.header_crop_path else None)
    )

    extracted_snapshot: dict[str, Any] = {
        "target_type": target_type,
        "is_census": is_census,
        "scan": {
            "id": scan.id if scan else scan_id,
            "filename": scan.filename if scan else None,
            "page_side": scan.page_side if scan else None,
            "original_path": scan.original_path if scan else None,
            "overlay_path": scan.overlay_path if scan else None,
            "header_crop_path": scan.header_crop_path if scan else None,
            "detected_year": scan.detected_year if scan else None,
            "detected_type": scan.detected_type if scan else None,
        },
        "census_sheet": (
            {
                "id": sheet.id,
                "slovakiana_key": sheet.slovakiana_key,
                "census_year": sheet.census_year,
                "district": sheet.district,
                "municipality": sheet.municipality,
                "house_number": sheet.house_number,
                "sheet_number": sheet.sheet_number,
                "sheet_type": sheet.sheet_type,
                "obvod_number": sheet.obvod_number,
                "commissioner_name": sheet.commissioner_name,
            }
            if sheet
            else None
        ),
        "row": (
            {
                "id": row.id,
                "row_index": row.row_index,
                "entry_number": row.entry_number,
                "y_start": row.y_start,
                "y_end": row.y_end,
                "stitched_crop_path": row.stitched_crop_path,
                "left_crop_path": row.left_crop_path,
                "right_crop_path": row.right_crop_path,
                "cell_crops": json.loads(row.cell_crops_json or "[]"),
                "raw_ocr_left": row.raw_ocr_left,
                "raw_ocr_right": row.raw_ocr_right,
                "raw_ocr_combined": row.raw_ocr_combined,
                "decision_action": row.decision_action,
                "decision_confidence": row.decision_confidence,
                "normalized": json.loads(row.normalized_json or "{}"),
            }
            if row
            else None
        ),
    }

    if fact:
        extracted_snapshot["person_fact"] = {
            "id": fact.id,
            "record_type": fact.record_type,
            "entry_number": fact.entry_number,
            "person_row_number": fact.person_row_number,
            "surname_raw": fact.surname_raw,
            "surname_normalized": fact.surname_normalized,
            "given_name_raw": fact.given_name_raw,
            "given_name_sk": fact.given_name_sk,
            "mother_maiden_surname": fact.mother_maiden_surname,
            "relation_to_head": fact.relation_to_head,
            "sex": fact.sex,
            "birth_or_event_date": fact.birth_or_event_date,
            "baptism_or_burial_date": fact.baptism_or_burial_date,
            "birth_year_est": fact.birth_year_est,
            "event_year": fact.event_year,
            "marital_status": fact.marital_status,
            "birth_place_raw": fact.birth_place_raw,
            "birth_place_municipality": fact.birth_place_municipality,
            "birth_place_district": fact.birth_place_district,
            "birth_place_country": fact.birth_place_country,
            "home_jurisdiction": fact.home_jurisdiction,
            "citizenship": fact.citizenship,
            "nationality": fact.nationality,
            "religion": fact.religion,
            "literacy": fact.literacy,
            "social_status_or_occupation": fact.social_status_or_occupation,
            "occupation_position": fact.occupation_position,
            "employer_workplace": fact.employer_workplace,
            "father_name": fact.father_name,
            "mother_name": fact.mother_name,
            "village_raw": fact.village_raw,
            "house_number": fact.house_number,
            "observationes_raw": fact.observationes_raw,
        }
    if house:
        extracted_snapshot["overview_house"] = {
            "id": house.id,
            "row_index": house.row_index,
            "ordinal_number": house.ordinal_number,
            "settlement_name": house.settlement_name,
            "house_number": house.house_number,
            "is_inhabited": house.is_inhabited,
            "owner_name": house.owner_name,
            "household_count": house.household_count,
            "present_persons_count": house.present_persons_count,
            "remarks": house.remarks,
            "raw_ocr_text": house.raw_ocr_text,
        }
    if dist:
        extracted_snapshot["municipal_district"] = {
            "id": dist.id,
            "row_index": dist.row_index,
            "ordinal_number": dist.ordinal_number,
            "settlement_name": dist.settlement_name,
            "obvod_number": dist.obvod_number,
            "commissioner_name": dist.commissioner_name,
            "houses_total": dist.houses_total,
            "houses_inhabited": dist.houses_inhabited,
            "houses_uninhabited": dist.houses_uninhabited,
            "household_count": dist.household_count,
            "present_persons_count": dist.present_persons_count,
            "is_summary_row": dist.is_summary_row,
            "raw_ocr_text": dist.raw_ocr_text,
        }

    traces_snapshot = [_serialize_trace_for_snapshot(t) for t in combined_traces_for_row]

    return {
        "target_type": target_type,
        "row_id": row.id if row else row_id,
        "scan_id": scan.id if scan else scan_id,
        "census_sheet_id": sheet.id if sheet else census_sheet_id,
        "person_fact_id": fact.id if fact else person_fact_id,
        "overview_house_id": house.id if house else overview_house_id,
        "municipal_district_id": dist.id if dist else municipal_district_id,
        "model_combo": model_combo,
        "layout_model": layout_model,
        "ocr_model": ocr_model,
        "decision_model": decision_model,
        "extract_model": extract_model,
        "mcp_model": mcp_model,
        "total_duration_ms": total_duration_ms,
        "layout_duration_ms": layout_duration_ms,
        "ocr_duration_ms": ocr_duration_ms,
        "decision_duration_ms": decision_duration_ms,
        "extract_duration_ms": extract_duration_ms,
        "mcp_duration_ms": mcp_duration_ms,
        "raw_ocr_snapshot": raw_ocr_snapshot,
        "extracted_snapshot": extracted_snapshot,
        "stage_breakdown": stage_breakdown,
        "traces_snapshot": traces_snapshot,
        "crop_url": crop_url,
    }


def upsert_ocr_feedback(
    session: Session,
    *,
    verdict: str,
    comment: str | None = None,
    expected_correction: str | None = None,
    target_type: str = "row",
    row_id: int | None = None,
    scan_id: int | None = None,
    census_sheet_id: int | None = None,
    person_fact_id: int | None = None,
    overview_house_id: int | None = None,
    municipal_district_id: int | None = None,
) -> OCRFeedback:
    norm_verdict = "incorrect" if str(verdict).strip().lower() in ("incorrect", "wrong", "bad", "false", "0", "error") else "correct"
    ctx = collect_row_execution_context(
        session,
        target_type=target_type,
        row_id=row_id,
        scan_id=scan_id,
        census_sheet_id=census_sheet_id,
        person_fact_id=person_fact_id,
        overview_house_id=overview_house_id,
        municipal_district_id=municipal_district_id,
    )

    resolved_row_id = ctx["row_id"]
    resolved_fact_id = ctx["person_fact_id"]
    resolved_house_id = ctx["overview_house_id"]
    resolved_dist_id = ctx["municipal_district_id"]
    resolved_scan_id = ctx["scan_id"]
    resolved_sheet_id = ctx["census_sheet_id"]
    model_combo = ctx["model_combo"]

    existing: OCRFeedback | None = None
    if resolved_fact_id:
        existing = session.exec(
            select(OCRFeedback).where(
                OCRFeedback.person_fact_id == resolved_fact_id,
                OCRFeedback.model_combo == model_combo,
            )
        ).first()
    if not existing and resolved_house_id:
        existing = session.exec(
            select(OCRFeedback).where(
                OCRFeedback.overview_house_id == resolved_house_id,
                OCRFeedback.model_combo == model_combo,
            )
        ).first()
    if not existing and resolved_dist_id:
        existing = session.exec(
            select(OCRFeedback).where(
                OCRFeedback.municipal_district_id == resolved_dist_id,
                OCRFeedback.model_combo == model_combo,
            )
        ).first()
    if not existing and resolved_row_id:
        existing = session.exec(
            select(OCRFeedback).where(
                OCRFeedback.row_id == resolved_row_id,
                OCRFeedback.model_combo == model_combo,
            )
        ).first()

    if not existing:
        existing = OCRFeedback(
            target_type=ctx["target_type"],
            row_id=resolved_row_id,
            scan_id=resolved_scan_id,
            census_sheet_id=resolved_sheet_id,
            person_fact_id=resolved_fact_id,
            overview_house_id=resolved_house_id,
            municipal_district_id=resolved_dist_id,
            model_combo=model_combo,
        )

    existing.target_type = ctx["target_type"]
    existing.row_id = resolved_row_id
    existing.scan_id = resolved_scan_id
    existing.census_sheet_id = resolved_sheet_id
    existing.person_fact_id = resolved_fact_id
    existing.overview_house_id = resolved_house_id
    existing.municipal_district_id = resolved_dist_id
    existing.verdict = norm_verdict
    if comment is not None:
        existing.comment = comment.strip() or None
    if expected_correction is not None:
        existing.expected_correction = expected_correction.strip() or None
    existing.model_combo = model_combo
    existing.layout_model = ctx["layout_model"]
    existing.ocr_model = ctx["ocr_model"]
    existing.decision_model = ctx["decision_model"]
    existing.extract_model = ctx["extract_model"]
    existing.mcp_model = ctx["mcp_model"]
    existing.total_duration_ms = ctx["total_duration_ms"]
    existing.layout_duration_ms = ctx["layout_duration_ms"]
    existing.ocr_duration_ms = ctx["ocr_duration_ms"]
    existing.decision_duration_ms = ctx["decision_duration_ms"]
    existing.extract_duration_ms = ctx["extract_duration_ms"]
    existing.mcp_duration_ms = ctx["mcp_duration_ms"]
    existing.raw_ocr_snapshot = ctx["raw_ocr_snapshot"]
    existing.extracted_snapshot_json = json.dumps(ctx["extracted_snapshot"], ensure_ascii=False)
    existing.stage_breakdown_json = json.dumps(ctx["stage_breakdown"], ensure_ascii=False)
    existing.traces_snapshot_json = json.dumps(ctx["traces_snapshot"], ensure_ascii=False)
    existing.crop_url = ctx["crop_url"]
    existing.updated_at = utcnow()

    session.add(existing)
    session.commit()
    session.refresh(existing)
    persist_llm_debug_report_files(session)
    return existing


def submit_ocr_feedback_fast(
    session: Session,
    *,
    verdict: str,
    comment: str | None = None,
    expected_correction: str | None = None,
    target_type: str = "row",
    row_id: int | None = None,
    scan_id: int | None = None,
    census_sheet_id: int | None = None,
    person_fact_id: int | None = None,
    overview_house_id: int | None = None,
    municipal_district_id: int | None = None,
    light: bool = False,
) -> dict[str, Any]:
    fb = upsert_ocr_feedback(
        session,
        verdict=verdict,
        comment=comment,
        expected_correction=expected_correction,
        target_type=target_type,
        row_id=row_id,
        scan_id=scan_id,
        census_sheet_id=census_sheet_id,
        person_fact_id=person_fact_id,
        overview_house_id=overview_house_id,
        municipal_district_id=municipal_district_id,
    )
    report_payload = build_llm_debug_report_payload(session)
    report_md = format_llm_debug_report_markdown(report_payload)
    out: dict[str, Any] = {
        "status": "ok",
        "feedback": serialize_ocr_feedback(fb, include_traces=True),
        "stats": report_payload["statistics"],
        "report_files": report_payload["statistics"]["summary"]["report_files"],
    }
    if not light:
        out["report_markdown"] = report_md
    return out


def delete_ocr_feedback_fast(
    session: Session,
    feedback_id: int,
    *,
    light: bool = False,
) -> dict[str, Any] | None:
    fb = session.get(OCRFeedback, feedback_id)
    if not fb:
        return None
    session.delete(fb)
    session.commit()
    report_files = persist_llm_debug_report_files(session)
    report_payload = build_llm_debug_report_payload(session)
    out: dict[str, Any] = {
        "status": "deleted",
        "deleted_id": feedback_id,
        "stats": report_payload["statistics"],
        "report_files": report_files,
    }
    if not light:
        out["report_markdown"] = format_llm_debug_report_markdown(report_payload)
    return out


def serialize_ocr_feedback(fb: OCRFeedback, *, include_traces: bool = True) -> dict[str, Any]:
    extracted_obj: dict[str, Any] = {}
    stage_breakdown: dict[str, Any] = {}
    traces_list: list[dict[str, Any]] = []
    try:
        extracted_obj = json.loads(fb.extracted_snapshot_json or "{}")
    except Exception:
        extracted_obj = {}
    try:
        stage_breakdown = json.loads(fb.stage_breakdown_json or "{}")
    except Exception:
        stage_breakdown = {}
    if include_traces:
        try:
            traces_list = json.loads(fb.traces_snapshot_json or "[]")
        except Exception:
            traces_list = []

    # Build a concise human-readable label for the row/person/house
    label_parts: list[str] = []
    cs_info = extracted_obj.get("census_sheet") or {}
    sc_info = extracted_obj.get("scan") or {}
    row_info = extracted_obj.get("row") or {}
    pf_info = extracted_obj.get("person_fact") or {}
    oh_info = extracted_obj.get("overview_house") or {}
    md_info = extracted_obj.get("municipal_district") or {}

    if cs_info.get("slovakiana_key"):
        label_parts.append(str(cs_info["slovakiana_key"]))
    elif sc_info.get("filename"):
        label_parts.append(str(sc_info["filename"]))

    if pf_info:
        p_name = f"{pf_info.get('given_name_raw') or ''} {pf_info.get('surname_raw') or ''}".strip()
        r_num = pf_info.get("person_row_number") or pf_info.get("entry_number") or row_info.get("row_index")
        if r_num:
            label_parts.append(f"Row #{r_num}")
        if p_name:
            label_parts.append(p_name)
    elif oh_info:
        label_parts.append(
            f"Dom č. {oh_info.get('house_number') or '—'} ({oh_info.get('owner_name') or '—'})"
        )
    elif md_info:
        label_parts.append(
            f"Obvod {md_info.get('obvod_number') or 'Spolu'} ({md_info.get('commissioner_name') or '—'})"
        )
    elif row_info.get("row_index") is not None:
        label_parts.append(f"Row #{row_info.get('row_index')}")

    out: dict[str, Any] = {
        "id": fb.id,
        "target_type": fb.target_type,
        "row_id": fb.row_id,
        "scan_id": fb.scan_id,
        "census_sheet_id": fb.census_sheet_id,
        "person_fact_id": fb.person_fact_id,
        "overview_house_id": fb.overview_house_id,
        "municipal_district_id": fb.municipal_district_id,
        "verdict": fb.verdict,
        "comment": fb.comment,
        "expected_correction": fb.expected_correction,
        "display_label": " · ".join(label_parts) if label_parts else f"Row #{fb.row_id or fb.id}",
        "model_combo": fb.model_combo,
        "layout_model": fb.layout_model,
        "ocr_model": fb.ocr_model,
        "decision_model": fb.decision_model,
        "extract_model": fb.extract_model,
        "mcp_model": fb.mcp_model,
        "total_duration_ms": fb.total_duration_ms,
        "layout_duration_ms": fb.layout_duration_ms,
        "ocr_duration_ms": fb.ocr_duration_ms,
        "decision_duration_ms": fb.decision_duration_ms,
        "extract_duration_ms": fb.extract_duration_ms,
        "mcp_duration_ms": fb.mcp_duration_ms,
        "raw_ocr_snapshot": fb.raw_ocr_snapshot,
        "extracted_snapshot": extracted_obj,
        "stage_breakdown": stage_breakdown,
        "crop_url": fb.crop_url,
        "created_at": fb.created_at.isoformat() if fb.created_at else None,
        "updated_at": fb.updated_at.isoformat() if fb.updated_at else None,
    }
    if include_traces:
        out["traces_snapshot"] = traces_list
    return out


def reset_ocr_feedback(
    session: Session,
    *,
    census_sheet_id: int | None = None,
    scan_id: int | None = None,
    scan_ids: list[int] | None = None,
    row_id: int | None = None,
    row_ids: list[int] | None = None,
    person_fact_id: int | None = None,
    overview_house_id: int | None = None,
    municipal_district_id: int | None = None,
    commit: bool = True,
    persist_report: bool = True,
) -> int:
    """
    Deletes all `OCRFeedback` records (verdicts and comments) associated with the specified
    sheet, scan(s), row(s), person fact, overview house, or municipal overview district,
    and refreshes the workspace LLM debug report files.
    """
    fb_candidates = list(
        session.exec(
            select(
                OCRFeedback.id,
                OCRFeedback.census_sheet_id,
                OCRFeedback.scan_id,
                OCRFeedback.row_id,
                OCRFeedback.person_fact_id,
                OCRFeedback.overview_house_id,
                OCRFeedback.municipal_district_id,
            )
        ).all()
    )
    if not fb_candidates:
        return 0

    target_sheet_ids: set[int] = {census_sheet_id} if census_sheet_id else set()
    target_scan_ids: set[int] = set(scan_ids or [])
    if scan_id:
        target_scan_ids.add(scan_id)
    target_row_ids: set[int] = set(row_ids or [])
    if row_id:
        target_row_ids.add(row_id)
    target_fact_ids: set[int] = {person_fact_id} if person_fact_id else set()
    target_house_ids: set[int] = {overview_house_id} if overview_house_id else set()
    target_dist_ids: set[int] = {municipal_district_id} if municipal_district_id else set()

    # Expand from census_sheet_id -> all scans, rows, facts, houses, districts of that sheet
    if target_sheet_ids:
        for sid in list(target_sheet_ids):
            sh = session.get(CensusSheet, sid)
            if sh:
                if sh.front_scan_id:
                    target_scan_ids.add(sh.front_scan_id)
                if sh.back_scan_id:
                    target_scan_ids.add(sh.back_scan_id)
            for sc_id in session.exec(
                select(ScanImage.id).where(ScanImage.census_sheet_id == sid)
            ).all():
                if sc_id is not None:
                    target_scan_ids.add(sc_id)
            for f_id in session.exec(
                select(PersonFact.id).where(PersonFact.census_sheet_id == sid)
            ).all():
                if f_id is not None:
                    target_fact_ids.add(f_id)
            for h_id in session.exec(
                select(CensusOverviewHouse.id).where(CensusOverviewHouse.census_sheet_id == sid)
            ).all():
                if h_id is not None:
                    target_house_ids.add(h_id)
            for d_id in session.exec(
                select(CensusMunicipalOverviewDistrict.id).where(
                    CensusMunicipalOverviewDistrict.census_sheet_id == sid
                )
            ).all():
                if d_id is not None:
                    target_dist_ids.add(d_id)

    # Expand from scan_ids -> all rows, facts, houses, districts on those scans (only when not scoped to specific row_ids)
    if target_scan_ids and not (row_id or row_ids):
        for r_id in session.exec(
            select(RecordRow.id).where(RecordRow.scan_id.in_(list(target_scan_ids)))  # type: ignore[union-attr]
        ).all():
            if r_id is not None:
                target_row_ids.add(r_id)
        for f_id in session.exec(
            select(PersonFact.id).where(PersonFact.scan_id.in_(list(target_scan_ids)))  # type: ignore[union-attr]
        ).all():
            if f_id is not None:
                target_fact_ids.add(f_id)
        for h_id in session.exec(
            select(CensusOverviewHouse.id).where(
                CensusOverviewHouse.scan_id.in_(list(target_scan_ids))  # type: ignore[union-attr]
            )
        ).all():
            if h_id is not None:
                target_house_ids.add(h_id)
        for d_id in session.exec(
            select(CensusMunicipalOverviewDistrict.id).where(
                CensusMunicipalOverviewDistrict.scan_id.in_(list(target_scan_ids))  # type: ignore[union-attr]
            )
        ).all():
            if d_id is not None:
                target_dist_ids.add(d_id)

    # Expand from row_ids -> facts, overview houses, municipal districts on those rows
    if target_row_ids:
        for f_id in session.exec(
            select(PersonFact.id).where(PersonFact.row_id.in_(list(target_row_ids)))  # type: ignore[union-attr]
        ).all():
            if f_id is not None:
                target_fact_ids.add(f_id)
        row_objs = list(
            session.exec(
                select(RecordRow.id, RecordRow.scan_id, RecordRow.row_index, RecordRow.stitched_crop_path).where(
                    RecordRow.id.in_(list(target_row_ids))  # type: ignore[union-attr]
                )
            ).all()
        )
        for _r_id, r_scan_id, r_row_index, r_stitched_crop in row_objs:
            if r_scan_id:
                for ovh in session.exec(
                    select(CensusOverviewHouse).where(CensusOverviewHouse.scan_id == r_scan_id)
                ).all():
                    if ovh.id is not None and (
                        (r_stitched_crop and ovh.stitched_crop_path == r_stitched_crop)
                        or ovh.row_index == r_row_index
                    ):
                        target_house_ids.add(ovh.id)
                for md in session.exec(
                    select(CensusMunicipalOverviewDistrict).where(
                        CensusMunicipalOverviewDistrict.scan_id == r_scan_id
                    )
                ).all():
                    if md.id is not None and (
                        (r_stitched_crop and md.stitched_crop_path == r_stitched_crop)
                        or md.row_index == r_row_index
                    ):
                        target_dist_ids.add(md.id)

    ids_to_delete: list[int] = []
    for fb_id, fb_cs_id, fb_sc_id, fb_r_id, fb_pf_id, fb_oh_id, fb_md_id in fb_candidates:
        if fb_id is None:
            continue
        should_delete = False
        if target_sheet_ids and fb_cs_id in target_sheet_ids:
            should_delete = True
        elif (
            target_scan_ids
            and not (row_id or row_ids)
            and fb_sc_id in target_scan_ids
        ):
            should_delete = True
        elif target_row_ids and fb_r_id in target_row_ids:
            should_delete = True
        elif target_fact_ids and fb_pf_id in target_fact_ids:
            should_delete = True
        elif target_house_ids and fb_oh_id in target_house_ids:
            should_delete = True
        elif target_dist_ids and fb_md_id in target_dist_ids:
            should_delete = True

        if should_delete:
            ids_to_delete.append(int(fb_id))

    deleted_count = len(ids_to_delete)
    if deleted_count > 0:
        from sqlalchemy import delete as sa_delete

        for obj in list(session.identity_map.values()):
            if isinstance(obj, OCRFeedback) and obj.id in ids_to_delete:
                session.expunge(obj)
        session.exec(
            sa_delete(OCRFeedback)
            .where(OCRFeedback.id.in_(ids_to_delete))  # type: ignore[union-attr]
            .execution_options(synchronize_session=False)
        )
        if commit:
            session.commit()
            if persist_report:
                persist_llm_debug_report_files(session)
        else:
            session.flush()

    return deleted_count


def build_feedback_lookup_maps(
    session: Session,
    *,
    census_sheet_id: int | None = None,
    scan_ids: list[int] | None = None,
    row_ids: list[int] | None = None,
    fact_ids: list[int] | None = None,
) -> dict[str, dict[int, dict[str, Any]]]:
    db_p = _session_file_db_path(session)
    if db_p and _rust_db is not None:
        try:
            raw_json = _rust_db.build_feedback_lookup_maps_rs(
                db_p,
                census_sheet_id=census_sheet_id,
                scan_ids=[int(x) for x in scan_ids if x is not None] if scan_ids else None,
                row_ids=[int(x) for x in row_ids if x is not None] if row_ids else None,
                fact_ids=[int(x) for x in fact_ids if x is not None] if fact_ids else None,
            )
            parsed = json.loads(raw_json)
            return {
                "by_row": {int(k): v for k, v in (parsed.get("by_row") or {}).items()},
                "by_fact": {int(k): v for k, v in (parsed.get("by_fact") or {}).items()},
                "by_house": {int(k): v for k, v in (parsed.get("by_house") or {}).items()},
                "by_dist": {int(k): v for k, v in (parsed.get("by_dist") or {}).items()},
            }
        except Exception:
            pass

    stmt = select(OCRFeedback).order_by(OCRFeedback.updated_at.desc(), OCRFeedback.id.desc())
    if census_sheet_id is not None and scan_ids:
        stmt = stmt.where(
            (OCRFeedback.census_sheet_id == census_sheet_id)
            | (OCRFeedback.scan_id.in_(scan_ids))  # type: ignore[union-attr]
        )
    elif census_sheet_id is not None:
        stmt = stmt.where(OCRFeedback.census_sheet_id == census_sheet_id)
    elif scan_ids:
        stmt = stmt.where(OCRFeedback.scan_id.in_(scan_ids))  # type: ignore[union-attr]
    elif row_ids:
        stmt = stmt.where(OCRFeedback.row_id.in_(row_ids))  # type: ignore[union-attr]
    elif fact_ids:
        stmt = stmt.where(OCRFeedback.person_fact_id.in_(fact_ids))  # type: ignore[union-attr]

    fb_rows = list(session.exec(stmt).all())
    by_row: dict[int, dict[str, Any]] = {}
    by_fact: dict[int, dict[str, Any]] = {}
    by_house: dict[int, dict[str, Any]] = {}
    by_dist: dict[int, dict[str, Any]] = {}

    for fb in fb_rows:
        serialized = serialize_ocr_feedback(fb, include_traces=False)
        if fb.row_id and fb.row_id not in by_row:
            by_row[fb.row_id] = serialized
        if fb.person_fact_id and fb.person_fact_id not in by_fact:
            by_fact[fb.person_fact_id] = serialized
        if fb.overview_house_id and fb.overview_house_id not in by_house:
            by_house[fb.overview_house_id] = serialized
        if fb.municipal_district_id and fb.municipal_district_id not in by_dist:
            by_dist[fb.municipal_district_id] = serialized

    return {
        "by_row": by_row,
        "by_fact": by_fact,
        "by_house": by_house,
        "by_dist": by_dist,
    }


def get_feedback_statistics(session: Session) -> dict[str, Any]:
    db_p = _session_file_db_path(session)
    if db_p and _rust_db is not None:
        try:
            d_ocr, d_dec, d_ext, d_lay, known = _rust_model_defaults_args()
            raw = _rust_db.get_feedback_statistics_rs(
                db_p,
                str(settings.data_dir),
                d_ocr,
                d_dec,
                d_ext,
                d_lay,
                known,
            )
            return json.loads(raw)
        except Exception:
            pass

    all_feedback = list(
        session.exec(select(OCRFeedback).order_by(OCRFeedback.id.desc())).all()
    )
    all_traces = list(
        session.exec(select(WorkflowTrace).order_by(WorkflowTrace.id.asc())).all()
    )

    # Scan-level layout and header OCR traces
    scan_layout_map: dict[int, WorkflowTrace] = {}
    scan_header_ocr_map: dict[int, WorkflowTrace] = {}
    scan_row_ids_map: dict[int, set[int]] = {}
    row_traces_map: dict[int, list[WorkflowTrace]] = {}

    for t in all_traces:
        if t.scan_id is not None and t.row_id is not None:
            scan_row_ids_map.setdefault(t.scan_id, set()).add(t.row_id)
        if t.row_id is not None:
            row_traces_map.setdefault(t.row_id, []).append(t)
        elif t.scan_id is not None:
            cg = canonical_stage_group(t.stage)
            if cg == "layout":
                scan_layout_map[t.scan_id] = t
            elif t.stage == "ocr_header":
                scan_header_ocr_map[t.scan_id] = t

    # Group row-level execution timing samples by model_combo
    combo_buckets: dict[str, dict[str, Any]] = {}

    def _ensure_combo_bucket(
        combo_key: str,
        *,
        ocr_m: str,
        dec_m: str | None,
        ext_m: str,
        lay_m: str | None,
    ) -> dict[str, Any]:
        if combo_key not in combo_buckets:
            combo_buckets[combo_key] = {
                "model_combo": combo_key,
                "ocr_model": ocr_m,
                "decision_model": dec_m,
                "extract_model": ext_m,
                "layout_model": lay_m or "opencv:angled_quad_row_slicer",
                "correct_count": 0,
                "incorrect_count": 0,
                "total_rated": 0,
                "timing_samples": [],
                "seen_row_keys": set(),
            }
        return combo_buckets[combo_key]

    # Map row_id -> latest feedback verdicts for individual step accuracy
    row_verdicts: dict[int, list[str]] = {}
    for fb in all_feedback:
        if fb.row_id is not None:
            row_verdicts.setdefault(fb.row_id, []).append(fb.verdict)

    # 1. Process all row_traces_map into combo_buckets
    for r_id, r_traces in row_traces_map.items():
        if not r_traces:
            continue
        s_id = r_traces[-1].scan_id
        s_rows_cnt = max(1, len(scan_row_ids_map.get(s_id or 0, {r_id})))

        lay_tr = scan_layout_map.get(s_id or 0) if s_id else None
        scan_lay_ms = int(lay_tr.duration_ms or 0) if lay_tr else 0
        lay_ms = round(scan_lay_ms / s_rows_cnt) if scan_lay_ms > 0 else 0
        lay_m = lay_tr.model_or_tool if lay_tr else "opencv:angled_quad_row_slicer"

        hdr_tr = scan_header_ocr_map.get(s_id or 0) if s_id else None
        scan_hdr_ms = int(hdr_tr.duration_ms or 0) if hdr_tr else 0
        hdr_ms = round(scan_hdr_ms / s_rows_cnt) if scan_hdr_ms > 0 else 0

        dedup_ocr: dict[tuple[str, str], WorkflowTrace] = {}
        for t in r_traces:
            if canonical_stage_group(t.stage) == "ocr" and t.stage != "ocr_header":
                dedup_ocr[(t.stage, t.image_path or "")] = t
        ocr_list = list(dedup_ocr.values())
        ocr_l_ms = sum(
            int(t.duration_ms or 0)
            for t in ocr_list
            if t.stage in ("ocr_left", "ocr", "census_overview_ocr")
        )
        ocr_r_ms = sum(
            int(t.duration_ms or 0) for t in ocr_list if t.stage == "ocr_right"
        )
        ocr_tot_ms = ocr_l_ms + ocr_r_ms + hdr_ms
        ocr_m = normalize_model_tag(
            ocr_list[-1].model_or_tool
            if ocr_list
            else (hdr_tr.model_or_tool if hdr_tr else None),
            "ocr",
        )

        dec_list = [t for t in r_traces if canonical_stage_group(t.stage) == "decision"]
        dec_tr = dec_list[-1] if dec_list else None
        dec_ms = int(dec_tr.duration_ms or 0) if dec_tr else 0
        dec_m = (
            dec_tr.model_or_tool
            if (dec_tr and dec_tr.model_or_tool.startswith("deterministic:"))
            else (normalize_model_tag(dec_tr.model_or_tool, "decision") if dec_tr else "deterministic:census_gazetteer")
        )
        is_census_row = bool(not dec_tr or dec_m.startswith("deterministic:"))

        ext_list = [t for t in r_traces if canonical_stage_group(t.stage) == "extract"]
        ext_tr = ext_list[-1] if ext_list else None
        ext_ms = int(ext_tr.duration_ms or 0) if ext_tr else 0
        ext_m = normalize_model_tag(ext_tr.model_or_tool if ext_tr else None, "extract")

        mcp_list = [t for t in r_traces if canonical_stage_group(t.stage) == "mcp"]
        mcp_ms = sum(int(t.duration_ms or 0) for t in mcp_list)

        tot_ms = lay_ms + ocr_tot_ms + dec_ms + ext_ms + mcp_ms
        combo_key = format_model_combo(ocr_m, ext_m, dec_m, is_census=is_census_row)

        p_tok = sum(int(t.prompt_tokens or 0) for t in r_traces)
        c_tok = sum(int(t.completion_tokens or 0) for t in r_traces)
        r_tok = sum(int(t.reasoning_tokens or 0) for t in r_traces)

        bucket = _ensure_combo_bucket(
            combo_key,
            ocr_m=ocr_m,
            dec_m=dec_m,
            ext_m=ext_m,
            lay_m=lay_m,
        )
        bucket["seen_row_keys"].add((r_id, combo_key))
        bucket["timing_samples"].append(
            {
                "layout_ms": lay_ms,
                "scan_layout_ms": scan_lay_ms,
                "ocr_ms": ocr_tot_ms,
                "ocr_header_ms": hdr_ms,
                "ocr_left_ms": ocr_l_ms,
                "ocr_right_ms": ocr_r_ms,
                "decision_ms": dec_ms,
                "extract_ms": ext_ms,
                "mcp_ms": mcp_ms,
                "total_ms": tot_ms,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "reasoning_tokens": r_tok,
            }
        )

    # 2. Fold all OCRFeedback entries into combo_buckets
    for fb in all_feedback:
        ocr_m = normalize_model_tag(fb.ocr_model, "ocr")
        ext_m = normalize_model_tag(fb.extract_model, "extract")
        dec_m = fb.decision_model or "deterministic:census_gazetteer"
        combo_key = fb.model_combo or format_model_combo(
            ocr_m,
            ext_m,
            dec_m,
            is_census=dec_m.startswith("deterministic:"),
        )
        bucket = _ensure_combo_bucket(
            combo_key,
            ocr_m=ocr_m,
            dec_m=dec_m,
            ext_m=ext_m,
            lay_m=fb.layout_model,
        )
        bucket["total_rated"] += 1
        if fb.verdict == "correct":
            bucket["correct_count"] += 1
        else:
            bucket["incorrect_count"] += 1

        sb: dict[str, Any] = {}
        try:
            sb = json.loads(fb.stage_breakdown_json or "{}")
        except Exception:
            sb = {}
        sample_key = (fb.row_id or -int(fb.id or 0), combo_key)
        if sample_key not in bucket["seen_row_keys"] and (
            fb.total_duration_ms > 0 or not bucket["timing_samples"]
        ):
            bucket["seen_row_keys"].add(sample_key)
            bucket["timing_samples"].append(
                {
                    "layout_ms": int(fb.layout_duration_ms or sb.get("layout_ms") or 0),
                    "scan_layout_ms": int(sb.get("scan_layout_ms") or fb.layout_duration_ms or 0),
                    "ocr_ms": int(fb.ocr_duration_ms or sb.get("ocr_ms") or 0),
                    "ocr_header_ms": int(sb.get("ocr_header_ms") or 0),
                    "ocr_left_ms": int(sb.get("ocr_left_ms") or fb.ocr_duration_ms or 0),
                    "ocr_right_ms": int(sb.get("ocr_right_ms") or 0),
                    "decision_ms": int(fb.decision_duration_ms or sb.get("decision_ms") or 0),
                    "extract_ms": int(fb.extract_duration_ms or sb.get("extract_ms") or 0),
                    "mcp_ms": int(fb.mcp_duration_ms or sb.get("mcp_ms") or 0),
                    "total_ms": int(fb.total_duration_ms or sb.get("total_ms") or 0),
                    "prompt_tokens": int(sb.get("prompt_tokens") or 0),
                    "completion_tokens": int(sb.get("completion_tokens") or 0),
                    "reasoning_tokens": int(sb.get("reasoning_tokens") or 0),
                }
            )

    # Ensure active configured combo is represented
    active_ocr_m = normalize_model_tag(None, "ocr")
    active_ext_m = normalize_model_tag(None, "extract")
    active_dec_m = normalize_model_tag(None, "decision")
    active_census_combo = format_model_combo(active_ocr_m, active_ext_m, None, is_census=True)
    active_matrika_combo = format_model_combo(active_ocr_m, active_ext_m, active_dec_m, is_census=False)

    if not combo_buckets:
        _ensure_combo_bucket(
            active_census_combo,
            ocr_m=active_ocr_m,
            dec_m="deterministic:census_gazetteer",
            ext_m=active_ext_m,
            lay_m="opencv:census_sheet_slicer",
        )

    def _avg(vals: list[int]) -> int:
        return round(sum(vals) / len(vals)) if vals else 0

    model_combos_out: list[dict[str, Any]] = []
    all_samples_flat: list[dict[str, int]] = []

    for combo_key, b in combo_buckets.items():
        samples = b["timing_samples"]
        non_zero_samples = [s for s in samples if s["total_ms"] > 0] or samples
        all_samples_flat.extend(non_zero_samples)

        tot_rated = int(b["total_rated"])
        corr = int(b["correct_count"])
        incorr = int(b["incorrect_count"])
        rate_pct = round(100.0 * corr / tot_rated, 1) if tot_rated > 0 else None

        tot_ms_list = [s["total_ms"] for s in non_zero_samples]
        model_combos_out.append(
            {
                "model_combo": combo_key,
                "ocr_model": b["ocr_model"],
                "decision_model": b["decision_model"],
                "extract_model": b["extract_model"],
                "layout_model": b["layout_model"],
                "is_active_combo": combo_key in (active_census_combo, active_matrika_combo),
                "total_rated": tot_rated,
                "correct_count": corr,
                "incorrect_count": incorr,
                "correctness_rate_pct": rate_pct,
                "rows_processed": max(len(samples), tot_rated),
                "avg_total_ms": _avg(tot_ms_list),
                "min_total_ms": min(tot_ms_list) if tot_ms_list else 0,
                "max_total_ms": max(tot_ms_list) if tot_ms_list else 0,
                "steps": {
                    "layout_ms": _avg([s["layout_ms"] for s in non_zero_samples]),
                    "scan_layout_ms": _avg([s["scan_layout_ms"] for s in non_zero_samples]),
                    "ocr_ms": _avg([s["ocr_ms"] for s in non_zero_samples]),
                    "ocr_header_ms": _avg([s["ocr_header_ms"] for s in non_zero_samples]),
                    "ocr_left_ms": _avg([s["ocr_left_ms"] for s in non_zero_samples]),
                    "ocr_right_ms": _avg([s["ocr_right_ms"] for s in non_zero_samples]),
                    "decision_ms": _avg([s["decision_ms"] for s in non_zero_samples]),
                    "extract_ms": _avg([s["extract_ms"] for s in non_zero_samples]),
                    "mcp_ms": _avg([s["mcp_ms"] for s in non_zero_samples]),
                },
                "avg_tokens": {
                    "prompt_tokens": _avg([s["prompt_tokens"] for s in non_zero_samples]),
                    "completion_tokens": _avg([s["completion_tokens"] for s in non_zero_samples]),
                    "reasoning_tokens": _avg([s["reasoning_tokens"] for s in non_zero_samples]),
                },
            }
        )

    model_combos_out.sort(
        key=lambda c: (
            0 if c["total_rated"] > 0 else 1,
            0 if c["is_active_combo"] else 1,
            -int(c["total_rated"]),
            -int(c["rows_processed"]),
        )
    )

    # 3. Individual Step Breakdown across all traces & feedback
    step_groups: dict[tuple[str, str], dict[str, Any]] = {}
    STAGE_ORDER = {
        "layout": 1,
        "census_layout": 1,
        "ocr_header": 2,
        "ocr_left": 3,
        "ocr_right": 4,
        "ocr": 5,
        "census_overview_ocr": 5,
        "decision": 6,
        "extract": 7,
        "census_extract": 7,
        "census_overview_extract": 7,
        "mcp_place": 8,
        "mcp_record": 9,
        "mcp_compare": 10,
    }

    for t in all_traces:
        if t.stage == "debug_playground":
            continue
        cg = canonical_stage_group(t.stage)
        norm_mod = (
            t.model_or_tool
            if (cg in ("layout", "mcp") or t.model_or_tool.startswith(("opencv:", "deterministic:")))
            else normalize_model_tag(t.model_or_tool, cg if cg in ("ocr", "decision", "extract") else "extract")
        )
        key = (t.stage, norm_mod)
        if key not in step_groups:
            step_groups[key] = {
                "stage": t.stage,
                "canonical_stage": cg,
                "model_or_tool": norm_mod,
                "durations": [],
                "ok_count": 0,
                "error_count": 0,
                "prompt_tokens": [],
                "completion_tokens": [],
                "reasoning_tokens": [],
                "row_ids": set(),
            }
        sg = step_groups[key]
        sg["durations"].append(int(t.duration_ms or 0))
        if t.status in ("ok", "success"):
            sg["ok_count"] += 1
        else:
            sg["error_count"] += 1
        sg["prompt_tokens"].append(int(t.prompt_tokens or 0))
        sg["completion_tokens"].append(int(t.completion_tokens or 0))
        sg["reasoning_tokens"].append(int(t.reasoning_tokens or 0))
        if t.row_id is not None:
            sg["row_ids"].add(t.row_id)
        elif t.scan_id is not None:
            for rid in scan_row_ids_map.get(t.scan_id, set()):
                sg["row_ids"].add(rid)

    individual_steps_out: list[dict[str, Any]] = []
    for (st_name, mod_name), sg in step_groups.items():
        durs = sg["durations"]
        corr_r = 0
        incorr_r = 0
        for rid in sg["row_ids"]:
            for v in row_verdicts.get(rid, []):
                if v == "correct":
                    corr_r += 1
                else:
                    incorr_r += 1
        tot_r = corr_r + incorr_r
        individual_steps_out.append(
            {
                "stage": st_name,
                "canonical_stage": sg["canonical_stage"],
                "model_or_tool": mod_name,
                "calls_count": len(durs),
                "ok_count": sg["ok_count"],
                "error_count": sg["error_count"],
                "avg_duration_ms": _avg(durs),
                "min_duration_ms": min(durs) if durs else 0,
                "max_duration_ms": max(durs) if durs else 0,
                "total_duration_ms": sum(durs),
                "avg_prompt_tokens": _avg(sg["prompt_tokens"]),
                "avg_completion_tokens": _avg(sg["completion_tokens"]),
                "avg_reasoning_tokens": _avg(sg["reasoning_tokens"]),
                "rated_rows_count": tot_r,
                "correct_rows_count": corr_r,
                "incorrect_rows_count": incorr_r,
                "correctness_rate_pct": round(100.0 * corr_r / tot_r, 1) if tot_r > 0 else None,
            }
        )

    individual_steps_out.sort(
        key=lambda s: (STAGE_ORDER.get(s["stage"], 99), s["stage"], -int(s["calls_count"]))
    )

    total_fb = len(all_feedback)
    total_corr = sum(1 for fb in all_feedback if fb.verdict == "correct")
    total_incorr = total_fb - total_corr
    overall_rate = round(100.0 * total_corr / total_fb, 1) if total_fb > 0 else None

    return {
        "summary": {
            "total_feedback": total_fb,
            "correct_count": total_corr,
            "incorrect_count": total_incorr,
            "correctness_rate_pct": overall_rate,
            "total_rows_analyzed": len(row_traces_map),
            "avg_total_duration_ms": _avg([s["total_ms"] for s in all_samples_flat]),
            "avg_steps_ms": {
                "layout_ms": _avg([s["layout_ms"] for s in all_samples_flat]),
                "ocr_ms": _avg([s["ocr_ms"] for s in all_samples_flat]),
                "ocr_header_ms": _avg([s["ocr_header_ms"] for s in all_samples_flat]),
                "ocr_left_ms": _avg([s["ocr_left_ms"] for s in all_samples_flat]),
                "ocr_right_ms": _avg([s["ocr_right_ms"] for s in all_samples_flat]),
                "decision_ms": _avg([s["decision_ms"] for s in all_samples_flat]),
                "extract_ms": _avg([s["extract_ms"] for s in all_samples_flat]),
                "mcp_ms": _avg([s["mcp_ms"] for s in all_samples_flat]),
            },
            "report_files": {
                "markdown_path": str(settings.data_dir / "ocr_feedback_debug_report.md"),
                "json_path": str(settings.data_dir / "ocr_feedback_debug_report.json"),
                "relative_markdown_path": "data/ocr_feedback_debug_report.md",
                "relative_json_path": "data/ocr_feedback_debug_report.json",
            },
        },
        "model_combos": model_combos_out,
        "individual_steps": individual_steps_out,
    }


def build_llm_debug_report_payload(
    session: Session,
    *,
    verdict_filter: str | None = None,
    feedback_id: int | None = None,
) -> dict[str, Any]:
    db_p = _session_file_db_path(session)
    if db_p and _rust_db is not None:
        try:
            d_ocr, d_dec, d_ext, d_lay, known = _rust_model_defaults_args()
            raw = _rust_db.build_and_persist_llm_debug_report_rs(
                db_p,
                str(settings.data_dir),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(settings.get_ai_runtime_config(), ensure_ascii=False),
                d_ocr,
                d_dec,
                d_ext,
                d_lay,
                known,
                verdict_filter=verdict_filter if verdict_filter in ("correct", "incorrect") else None,
                feedback_id=feedback_id,
                persist_files=False,
            )
            res = json.loads(raw)
            payload = res["payload"]
            payload["_cached_markdown"] = res["markdown"]
            return payload
        except Exception:
            pass

    stmt = select(OCRFeedback).order_by(
        # Show incorrect items first so the LLM immediately sees failure cases
        OCRFeedback.verdict.desc(),
        OCRFeedback.updated_at.desc(),
        OCRFeedback.id.desc(),
    )
    if feedback_id is not None:
        stmt = stmt.where(OCRFeedback.id == feedback_id)
    elif verdict_filter in ("correct", "incorrect"):
        stmt = stmt.where(OCRFeedback.verdict == verdict_filter)

    fb_items = list(session.exec(stmt).all())
    stats = get_feedback_statistics(session)
    serialized_items = [serialize_ocr_feedback(fb, include_traces=True) for fb in fb_items]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ai_runtime_config": settings.get_ai_runtime_config(),
        "statistics": stats,
        "feedback_count": len(serialized_items),
        "feedback_items": serialized_items,
    }


def format_llm_debug_report_markdown(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("_cached_markdown"), str):
        return str(payload["_cached_markdown"])
    stats = payload.get("statistics") or {}
    summary = stats.get("summary") or {}
    combos = stats.get("model_combos") or []
    steps = stats.get("individual_steps") or []
    items = payload.get("feedback_items") or []

    overall_rate_str = (
        f"{summary['correctness_rate_pct']}%"
        if summary.get("correctness_rate_pct") is not None
        else "N/A"
    )
    lines: list[str] = [
        "# Matrika & Census Digitizer — OCR Feedback & LLM Debugging Report",
        f"- **Generated At**: `{payload.get('generated_at')}`",
        "- **Workspace Report Files**: `data/ocr_feedback_debug_report.md` | `data/ocr_feedback_debug_report.json`",
        f"- **Total Feedback Items**: `{summary.get('total_feedback', 0)}` "
        f"(`{summary.get('correct_count', 0)}` CORRECT, `{summary.get('incorrect_count', 0)}` INCORRECT)",
        f"- **Overall Correctness Rate**: `{overall_rate_str}`",
        f"- **Average Total Row Time**: `{summary.get('avg_total_duration_ms', 0)} ms`",
        "",
        "## 1. Model Combination Correctness & Average Step Timing",
        "",
        "| Model Combination | Correctness Rate | Correct / Rated | Rows | Avg Total (ms) | Layout (ms) | OCR Total (ms) | OCR Left / Right / Hdr (ms) | Decision (ms) | Extract (ms) | MCP (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for c in combos:
        rate_str = (
            f"**{c['correctness_rate_pct']}%**"
            if c.get("correctness_rate_pct") is not None
            else "— (unrated)"
        )
        st = c.get("steps") or {}
        lines.append(
            f"| `{c.get('model_combo')}` | {rate_str} | "
            f"{c.get('correct_count', 0)} / {c.get('total_rated', 0)} "
            f"({c.get('incorrect_count', 0)} err) | {c.get('rows_processed', 0)} | "
            f"**{c.get('avg_total_ms', 0)} ms** | {st.get('layout_ms', 0)} ms | "
            f"{st.get('ocr_ms', 0)} ms | "
            f"L:{st.get('ocr_left_ms', 0)} / R:{st.get('ocr_right_ms', 0)} / H:{st.get('ocr_header_ms', 0)} | "
            f"{st.get('decision_ms', 0)} ms | {st.get('extract_ms', 0)} ms | {st.get('mcp_ms', 0)} ms |"
        )

    lines.extend(
        [
            "",
            "## 2. Individual Pipeline Steps Breakdown",
            "",
            "| Stage | Canonical Step | Model / Tool | Calls | Avg Time (ms) | Min / Max (ms) | Avg Tokens (In / Out / Reason) | Rated Row Accuracy |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for s in steps:
        s_rate = (
            f"{s['correctness_rate_pct']}% ({s['correct_rows_count']}/{s['rated_rows_count']})"
            if s.get("correctness_rate_pct") is not None
            else "—"
        )
        lines.append(
            f"| `{s.get('stage')}` | `{s.get('canonical_stage')}` | `{s.get('model_or_tool')}` | "
            f"{s.get('calls_count', 0)} | **{s.get('avg_duration_ms', 0)} ms** | "
            f"{s.get('min_duration_ms', 0)} / {s.get('max_duration_ms', 0)} ms | "
            f"{s.get('avg_prompt_tokens', 0)} / {s.get('avg_completion_tokens', 0)} / {s.get('avg_reasoning_tokens', 0)} | "
            f"{s_rate} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Detailed User Feedback Entries ({len(items)} items)",
            "",
        ]
    )

    if not items:
        lines.append("_No OCR feedback entries recorded yet._")
    else:
        for idx, item in enumerate(items, start=1):
            verdict_upper = str(item.get("verdict") or "correct").upper()
            ext_snap = item.get("extracted_snapshot") or {}
            sc_info = ext_snap.get("scan") or {}
            cs_info = ext_snap.get("census_sheet") or {}
            row_info = ext_snap.get("row") or {}
            sb = item.get("stage_breakdown") or {}
            traces = item.get("traces_snapshot") or []

            lines.extend(
                [
                    f"### {idx}. [{verdict_upper}] {item.get('display_label')} (Feedback ID #{item.get('id')})",
                    f"- **Verdict**: `{verdict_upper}`",
                    f"- **User Comment / Issue Description**: {item.get('comment') or '_None provided_'}",
                ]
            )
            if item.get("expected_correction"):
                lines.append(f"- **Expected Correction**: `{item.get('expected_correction')}`")

            lines.extend(
                [
                    f"- **Updated At**: `{item.get('updated_at')}`",
                    f"- **Target**: `target_type={item.get('target_type')}`, `row_id={item.get('row_id')}`, "
                    f"`scan_id={item.get('scan_id')}`, `census_sheet_id={item.get('census_sheet_id')}`, "
                    f"`person_fact_id={item.get('person_fact_id')}`, `overview_house_id={item.get('overview_house_id')}`, "
                    f"`municipal_district_id={item.get('municipal_district_id')}`",
                    f"- **Model Combo**: `{item.get('model_combo')}`",
                    f"  - **Layout**: `{item.get('layout_model')}` (`{item.get('layout_duration_ms', 0)} ms`, scan total: `{sb.get('scan_layout_ms', 0)} ms`)",
                    f"  - **OCR**: `{item.get('ocr_model')}` (`{item.get('ocr_duration_ms', 0)} ms` — left: `{sb.get('ocr_left_ms', 0)} ms`, right: `{sb.get('ocr_right_ms', 0)} ms`, header: `{sb.get('ocr_header_ms', 0)} ms`)",
                    f"  - **Decision**: `{item.get('decision_model')}` (`{item.get('decision_duration_ms', 0)} ms`)",
                    f"  - **Extract**: `{item.get('extract_model')}` (`{item.get('extract_duration_ms', 0)} ms`)",
                    f"  - **MCP**: `{item.get('mcp_model') or 'none'}` (`{item.get('mcp_duration_ms', 0)} ms`)",
                    f"  - **Total Row Processing Time**: `{item.get('total_duration_ms', 0)} ms` "
                    f"(tokens: prompt=`{sb.get('prompt_tokens', 0)}`, completion=`{sb.get('completion_tokens', 0)}`, reasoning=`{sb.get('reasoning_tokens', 0)}`)",
                ]
            )

            if cs_info:
                lines.append(
                    f"- **Census Sheet Context**: key=`{cs_info.get('slovakiana_key')}`, "
                    f"year=`{cs_info.get('census_year')}`, municipality=`{cs_info.get('municipality')}`, "
                    f"district=`{cs_info.get('district')}`, house_number=`{cs_info.get('house_number')}`, "
                    f"sheet_type=`{cs_info.get('sheet_type')}`"
                )
            if sc_info:
                lines.append(
                    f"- **Scan Files**: filename=`{sc_info.get('filename')}`, "
                    f"original_path=`{sc_info.get('original_path')}`, overlay_path=`{sc_info.get('overlay_path')}`"
                )
            if row_info or item.get("crop_url"):
                lines.append(
                    f"- **Row Crops**: stitched=`{row_info.get('stitched_crop_path') or item.get('crop_url')}`, "
                    f"left=`{row_info.get('left_crop_path')}`, right=`{row_info.get('right_crop_path')}`"
                )

            lines.extend(
                [
                    "",
                    "#### Raw OCR Text",
                    "```text",
                    str(item.get("raw_ocr_snapshot") or "(empty)"),
                    "```",
                    "",
                    "#### Extracted Structured Data (Snapshot JSON)",
                    "```json",
                    json.dumps(
                        {
                            k: v
                            for k, v in ext_snap.items()
                            if k in ("person_fact", "overview_house", "municipal_district", "row") and v is not None
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                ]
            )

            if traces:
                lines.extend(["", f"#### Pipeline Stage Traces ({len(traces)} calls)"])
                for tr in traces:
                    lines.extend(
                        [
                            f"- **Trace #{tr.get('id')}** — `{tr.get('stage')}` · `{tr.get('model_or_tool')}` · "
                            f"`{tr.get('duration_ms')} ms` · status=`{tr.get('status')}` · image=`{tr.get('image_path')}`",
                            "  - **Prompt**:",
                            "    ```text",
                            "    " + str(tr.get("prompt_text") or "").replace("\n", "\n    "),
                            "    ```",
                        ]
                    )
                    if tr.get("reasoning_content"):
                        lines.extend(
                            [
                                "  - **Reasoning**:",
                                "    ```text",
                                "    " + str(tr.get("reasoning_content") or "").replace("\n", "\n    "),
                                "    ```",
                            ]
                        )
                    lines.extend(
                        [
                            "  - **Response**:",
                            "    ```text",
                            "    " + str(tr.get("response_text") or "").replace("\n", "\n    "),
                            "    ```",
                        ]
                    )
            lines.append("")

    return "\n".join(lines)


def persist_llm_debug_report_files(session: Session) -> dict[str, str]:
    db_p = _session_file_db_path(session)
    if db_p and _rust_db is not None:
        try:
            d_ocr, d_dec, d_ext, d_lay, known = _rust_model_defaults_args()
            _rust_db.build_and_persist_llm_debug_report_rs(
                db_p,
                str(settings.data_dir),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(settings.get_ai_runtime_config(), ensure_ascii=False),
                d_ocr,
                d_dec,
                d_ext,
                d_lay,
                known,
                verdict_filter=None,
                feedback_id=None,
                persist_files=True,
            )
            data_dir = Path(settings.data_dir)
            return {
                "markdown_path": str(data_dir / "ocr_feedback_debug_report.md"),
                "json_path": str(data_dir / "ocr_feedback_debug_report.json"),
                "relative_markdown_path": "data/ocr_feedback_debug_report.md",
                "relative_json_path": "data/ocr_feedback_debug_report.json",
            }
        except Exception:
            pass

    payload = build_llm_debug_report_payload(session)
    md_text = format_llm_debug_report_markdown(payload)
    clean_payload = {k: v for k, v in payload.items() if k != "_cached_markdown"}
    json_text = json.dumps(clean_payload, ensure_ascii=False, indent=2)

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    md_path = data_dir / "ocr_feedback_debug_report.md"
    json_path = data_dir / "ocr_feedback_debug_report.json"

    try:
        md_path.write_text(md_text, encoding="utf-8")
        json_path.write_text(json_text, encoding="utf-8")
    except OSError:
        pass

    return {
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "relative_markdown_path": "data/ocr_feedback_debug_report.md",
        "relative_json_path": "data/ocr_feedback_debug_report.json",
    }
