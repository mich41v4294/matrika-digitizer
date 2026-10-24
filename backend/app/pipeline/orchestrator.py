from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from backend.app.api.ws import ws_manager
from backend.app.config import settings
from backend.app.db.models import (
    Batch,
    CensusSheet,
    FamilySearchMatch,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.db.session import engine
from backend.app.mcp.familysearch_client import fs_mcp_manager
from backend.app.pipeline.census_layout import (
    detect_is_municipal_overview_sheet,
    detect_is_overview_sheet,
    slice_census_overview_sheet,
    slice_census_sheet,
)
from backend.app.pipeline.decision_stage import (
    evaluate_census_row_deterministic,
    run_decision_for_row,
)
from backend.app.pipeline.extraction_stage import (
    extract_census_municipal_overview_sheet,
    extract_census_overview_sheet,
    reconcile_census_household_relationships,
    run_census_extraction_for_row,
    run_extraction_for_row,
)
from backend.app.pipeline.layout import LayoutResult, slice_matrika_spread
from backend.app.pipeline.ocr_stage import run_ocr_for_row, run_ocr_for_scan_header


def _matches_watchlist(fact: PersonFact, watchlist_filter: str | None) -> bool:
    if not watchlist_filter or not watchlist_filter.strip():
        return True
    terms = [t.strip().lower() for t in watchlist_filter.split(",") if t.strip()]
    if not terms:
        return True
    haystack = " ".join([
        fact.surname_raw or "",
        fact.surname_normalized or "",
        fact.mother_maiden_surname or "",
        fact.father_name or "",
        fact.mother_name or "",
        fact.village_raw or "",
        fact.village_sk or "",
        fact.village_hu or "",
    ]).lower()
    return any(term in haystack for term in terms)


def _persist_layout_result(
    session: Session,
    scan: ScanImage,
    layout_res: LayoutResult,
    dur_ms: int,
) -> list[RecordRow]:
    if scan.detected_type == "census":
        layout_res.spine_x = 0
        layout_res.spine_offset_px = 0.0
    scan.width = layout_res.width
    scan.height = layout_res.height
    scan.spine_x = layout_res.spine_x
    scan.header_crop_path = layout_res.header_crop_rel
    scan.overlay_path = layout_res.overlay_rel
    scan.grid_metadata_json = layout_res.to_json()
    scan.status = "sliced"
    scan.error_message = None
    session.add(scan)
    session.commit()

    # Update existing rows by row_index in-place (preserving row IDs) and remove orphaned child records
    existing_rows = session.exec(
        select(RecordRow).where(RecordRow.scan_id == scan.id).order_by(RecordRow.row_index)
    ).all()
    existing_by_idx = {er.row_index: er for er in existing_rows}
    new_indices = {r_info.row_index for r_info in layout_res.rows}
    for er in existing_rows:
        if er.row_index not in new_indices:
            for m in session.exec(
                select(FamilySearchMatch).where(FamilySearchMatch.row_id == er.id)
            ).all():
                session.delete(m)
            for pf in session.exec(select(PersonFact).where(PersonFact.row_id == er.id)).all():
                session.delete(pf)
            for qi in session.exec(select(QuestionItem).where(QuestionItem.row_id == er.id)).all():
                session.delete(qi)
            session.delete(er)
    session.commit()

    created_rows: list[RecordRow] = []
    for r_info in layout_res.rows:
        row = existing_by_idx.get(r_info.row_index)
        if row is None:
            row = RecordRow(
                scan_id=scan.id,
                row_index=r_info.row_index,
                status="sliced",
            )
        row.y_start = r_info.y_start
        row.y_end = r_info.y_end
        row.left_crop_path = r_info.left_crop_rel
        row.right_crop_path = r_info.right_crop_rel
        row.stitched_crop_path = r_info.stitched_crop_rel
        row.cell_crops_json = json.dumps(r_info.cell_crops, ensure_ascii=False)
        session.add(row)
        created_rows.append(row)
    session.commit()
    for r in created_rows:
        session.refresh(r)

    trace = WorkflowTrace(
        scan_id=scan.id,
        stage="layout",
        model_or_tool="opencv:angled_quad_row_slicer",
        prompt_text=f"Image {scan.filename} ({layout_res.width}x{layout_res.height}), spine_x={layout_res.spine_x}, mode={layout_res.slicing_mode}",
        image_path=layout_res.overlay_rel,
        response_text=(
            f"Sliced {len(created_rows)} angled rows ({layout_res.slicing_mode}), "
            f"angle_L={layout_res.skew_left_deg:+.2f}deg, angle_R={layout_res.skew_right_deg:+.2f}deg, "
            f"spine_offset={layout_res.spine_offset_px:+.1f}px"
        ),
        parsed_json=json.dumps(
            {
                "spine_x": layout_res.spine_x,
                "rows_count": len(created_rows),
                "skew_left_deg": layout_res.skew_left_deg,
                "skew_right_deg": layout_res.skew_right_deg,
                "col_tilt_left_deg": layout_res.col_tilt_left_deg,
                "col_tilt_right_deg": layout_res.col_tilt_right_deg,
                "spine_offset_px": layout_res.spine_offset_px,
                "slicing_mode": layout_res.slicing_mode,
            }
        ),
        status="ok",
        duration_ms=dur_ms,
    )
    session.add(trace)
    session.commit()
    return created_rows


def run_layout_for_scan_sync(session: Session, scan: ScanImage) -> list[RecordRow]:
    t0 = time.perf_counter()
    orig_path = Path(scan.original_path)
    if scan.detected_type == "census":
        sheet = session.get(CensusSheet, scan.census_sheet_id) if scan.census_sheet_id else None
        census_year = sheet.census_year if sheet else 1930
        is_blackened = bool(sheet and sheet.is_blackened)
        is_overview = bool(sheet and (sheet.sheet_type in ("overview", "municipal_overview") or not sheet.house_number))
        layout_res = slice_census_sheet(
            orig_path,
            scan.id,
            settings.crops_dir,
            page_side=scan.page_side or "front",
            census_year=census_year,
            is_blackened=is_blackened,
            is_overview=is_overview,
        )
    else:
        layout_res = slice_matrika_spread(orig_path, scan.id, settings.crops_dir)
    dur_ms = int((time.perf_counter() - t0) * 1000)
    return _persist_layout_result(session, scan, layout_res, dur_ms)


async def run_layout_for_scan_async(session: Session, scan: ScanImage) -> list[RecordRow]:
    t0 = time.perf_counter()
    orig_path = Path(scan.original_path)
    if scan.detected_type == "census":
        sheet = session.get(CensusSheet, scan.census_sheet_id) if scan.census_sheet_id else None
        census_year = sheet.census_year if sheet else 1930
        is_blackened = bool(sheet and sheet.is_blackened)
        is_overview = bool(sheet and (sheet.sheet_type in ("overview", "municipal_overview") or not sheet.house_number))
        layout_res = await asyncio.to_thread(
            slice_census_sheet,
            orig_path,
            scan.id,
            settings.crops_dir,
            page_side=scan.page_side or "front",
            census_year=census_year,
            is_blackened=is_blackened,
            is_overview=is_overview,
        )
    else:
        layout_res = await asyncio.to_thread(
            slice_matrika_spread, orig_path, scan.id, settings.crops_dir
        )
    dur_ms = int((time.perf_counter() - t0) * 1000)
    return _persist_layout_result(session, scan, layout_res, dur_ms)


async def process_single_scan_pipeline(
    scan_id: int,
    *,
    max_rows: int | None = None,
    run_mcp: bool = True,
    watchlist_filter: str | None = None,
) -> dict[str, Any]:
    """
    Runs the stage-batched pipeline on a single scan:
    Stage 1: OpenCV Layout & Row Slicing
    Stage 2: All GLM-OCR calls (header + rows) while glm-ocr stays warm in VRAM
    Stage 3: All d1-3b-mlx decision/triage calls while d1-3b-mlx stays warm
    Stage 4: All underdog-saluki-27b-1.0 extraction calls while saluki stays warm
    Stage 5: FamilySearch MCP place + record matching
    """
    with Session(engine) as session:
        scan = session.get(ScanImage, scan_id)
        if not scan:
            raise ValueError(f"Scan {scan_id} not found")

        try:
            scan.error_message = None
            session.add(scan)
            session.commit()

            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "layout",
                    "active_model": (
                        "opencv:census_sheet_slicer"
                        if scan.detected_type == "census"
                        else "opencv:two_page_spread_slicer"
                    ),
                    "current_scan_id": scan.id,
                    "progress_pct": 10,
                    "message_sk": (
                        f"Rozdeľujem riadky sčítacieho hárku ({scan.filename})..."
                        if scan.detected_type == "census"
                        else f"Rozdeľujem dvojstranu a riadky matriky ({scan.filename})..."
                    ),
                    "message_en": (
                        f"Slicing census sheet rows ({scan.filename})..."
                        if scan.detected_type == "census"
                        else f"Slicing two-page spread and rows ({scan.filename})..."
                    ),
                },
            )

            rows = session.exec(
                select(RecordRow).where(RecordRow.scan_id == scan.id).order_by(RecordRow.row_index)
            ).all()
            if not rows:
                rows = await run_layout_for_scan_async(session, scan)

            target_rows = rows[:max_rows] if max_rows else rows

            # STAGE 2: GLM-OCR Batch
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "ocr",
                    "active_model": settings.model_ocr,
                    "current_scan_id": scan.id,
                    "progress_pct": 25,
                    "message_sk": f"Čítam rukopis pomocou {settings.model_ocr} ({len(target_rows)} riadkov)...",
                    "message_en": f"Running GLM-OCR batch on header + {len(target_rows)} rows...",
                },
            )
            await run_ocr_for_scan_header(session, scan)
            for idx, row in enumerate(target_rows, start=1):
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "ocr",
                        "active_model": settings.model_ocr,
                        "current_scan_id": scan.id,
                        "current_row_id": row.id,
                        "progress_pct": 25 + int(25 * idx / max(1, len(target_rows))),
                        "message_sk": f"OCR čítanie riadku {idx}/{len(target_rows)} ({scan.filename})...",
                        "message_en": f"GLM-OCR row {idx}/{len(target_rows)}...",
                    },
                )
                await run_ocr_for_row(session, scan, row)

            scan.status = "ocr_done"
            session.add(scan)
            session.commit()

            is_census = bool(scan.detected_type == "census" or scan.census_sheet_id is not None)

            # STAGE 3: Decision Model (d1-3b-mlx) Batch — ONLY for 19th-century church matrikas!
            # Census sheets skip d1-3b-mlx completely and go straight from Cell OCR -> Census Extraction
            # (with composite cell crop + 100km gazetteer + household reconciliation).
            if not is_census:
                prev_ocr = None
                for idx, row in enumerate(target_rows, start=1):
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "decision",
                            "active_model": settings.model_decision,
                            "current_scan_id": scan.id,
                            "current_row_id": row.id,
                            "progress_pct": 50 + int(20 * idx / max(1, len(target_rows))),
                            "message_sk": f"Kontrola čitateľnosti ({settings.model_decision}) – riadok {idx}/{len(target_rows)}...",
                            "message_en": f"Decision model triage row {idx}/{len(target_rows)}...",
                        },
                    )
                    await run_decision_for_row(session, scan, row, prev_ocr)
                    prev_ocr = row.raw_ocr_combined

                scan.status = "triaged"
                session.add(scan)
                session.commit()
            else:
                for row in target_rows:
                    row.decision_action = "auto_accept"
                    row.decision_confidence = 0.95
                    row.status = "auto_accepted"
                    session.add(row)
                session.commit()

            # STAGE 4: Genealogical Extraction & Normalization Batch
            prev_fact: PersonFact | None = None
            extracted_facts: list[tuple[PersonFact, RecordRow]] = []
            for idx, row in enumerate(target_rows, start=1):
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "extract",
                        "active_model": settings.model_extract,
                        "current_scan_id": scan.id,
                        "current_row_id": row.id,
                        "progress_pct": 70 + int(18 * idx / max(1, len(target_rows))),
                        "message_sk": f"Extrakcia mien, obcí a poznámok ({settings.model_extract}) – riadok {idx}/{len(target_rows)}...",
                        "message_en": f"Extracting structured facts row {idx}/{len(target_rows)}...",
                    },
                )
                if is_census:
                    fact = await run_census_extraction_for_row(session, scan, row, prev_fact)
                else:
                    fact = await run_extraction_for_row(session, scan, row, prev_fact)
                if fact is not None:
                    prev_fact = fact
                    extracted_facts.append((fact, row))

            if is_census and scan.census_sheet_id:
                reconcile_census_household_relationships(session, scan.census_sheet_id)
            if is_census:
                for fact, row in extracted_facts:
                    session.refresh(fact)
                    evaluate_census_row_deterministic(session, scan, row, fact, log_trace=False)

            scan.status = "extracted"
            session.add(scan)
            session.commit()

            # STAGE 5: FamilySearch MCP Matching
            matched_count = 0
            if run_mcp:
                if scan.detected_type == "census" and scan.census_sheet_id:
                    sheet = session.get(CensusSheet, scan.census_sheet_id)
                    if sheet:
                        matched_count = await fs_mcp_manager.match_census_sheet(session, sheet)
                else:
                    for idx, (fact, row) in enumerate(extracted_facts, start=1):
                        if not _matches_watchlist(fact, watchlist_filter):
                            continue
                        await ws_manager.broadcast(
                            "state",
                            {
                                "active_stage": "mcp_match",
                                "active_model": "familysearch-mcp",
                                "current_scan_id": scan.id,
                                "current_row_id": row.id,
                                "progress_pct": 88 + int(11 * idx / max(1, len(extracted_facts))),
                                "message_sk": f"Porovnávam s FamilySearch ({fact.given_name_sk or ''} {fact.surname_normalized or ''})...",
                                "message_en": f"Matching row {idx}/{len(extracted_facts)} via familysearch-mcp...",
                            },
                        )
                        m_res = await fs_mcp_manager.match_person_fact(session, fact, row)
                        if m_res is not None and (m_res.fs_person_id or "").strip():
                            matched_count += 1

            scan.status = "matched" if run_mcp else "extracted"
            session.add(scan)
            session.commit()

            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "current_scan_id": scan.id,
                    "current_row_id": None,
                    "progress_pct": 100,
                    "message_sk": f"Hotovo! Spracovaných {len(target_rows)} riadkov zo skenu {scan.filename}.",
                    "message_en": f"Completed {len(target_rows)} rows for {scan.filename}.",
                },
            )

            return {
                "scan_id": scan.id,
                "rows_processed": len(target_rows),
                "matched_count": matched_count,
                "detected_year": scan.detected_year,
                "detected_type": scan.detected_type,
            }
        except Exception as exc:
            scan.status = "error"
            scan.error_message = str(exc)[:500]
            session.add(scan)
            session.commit()
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "current_scan_id": scan.id,
                    "current_row_id": None,
                    "progress_pct": 0,
                    "message_sk": f"Chyba pri spracovaní {scan.filename}: {exc}",
                    "message_en": f"Error processing {scan.filename}: {exc}",
                },
            )
            return {
                "scan_id": scan.id,
                "status": "error",
                "error": str(exc),
            }


async def process_batch_stage_queued(
    batch_id: int,
    *,
    max_rows_per_scan: int | None = None,
    run_mcp: bool = True,
    only_undone: bool = True,
) -> dict[str, Any]:
    """
    Strictly stage-batches an entire multi-image Batch across ALL not-done scans:
    1. All pending scans -> OpenCV Layout Slicing
    2. All pending scans -> GLM-OCR (no model swap during entire OCR phase!)
    3. All pending scans -> d1-3b-mlx Decision Model (no model swap during entire triage phase!)
    4. All pending scans -> underdog-saluki-27b-1.0 Extractor (no model swap during entire extraction phase!)
    5. All pending scans -> familysearch-mcp Matching
    """
    with Session(engine) as session:
        batch = session.get(Batch, batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        all_scans = list(
            session.exec(
                select(ScanImage).where(ScanImage.batch_id == batch_id).order_by(ScanImage.id.asc())
            ).all()
        )
        if not all_scans:
            return {"batch_id": batch_id, "scans": 0, "scans_processed": 0, "scans_skipped": 0}

        if only_undone:
            scans = [
                s
                for s in all_scans
                if (s.status or "").strip() not in ("extracted", "matched")
            ]
        else:
            scans = list(all_scans)

        skipped_count = len(all_scans) - len(scans)
        batch.total_scans = len(all_scans)
        batch.processed_scans = skipped_count
        session.add(batch)
        session.commit()

        if not scans:
            batch.status = "completed"
            batch.processed_scans = len(all_scans)
            session.add(batch)
            session.commit()
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "progress_pct": 100,
                    "message_sk": f"Všetky skeny v dávke '{batch.name}' sú už spracované ({skipped_count} hotových).",
                    "message_en": f"All scans in batch '{batch.name}' are already processed ({skipped_count} done).",
                },
            )
            return {
                "batch_id": batch_id,
                "scans_processed": 0,
                "scans_skipped": skipped_count,
                "facts_extracted": 0,
            }

        try:
            batch.status = "slicing"
            session.add(batch)
            session.commit()

            # Phase 1: Layout Slicing across all pending scans
            scan_rows_map: dict[int, list[RecordRow]] = {}
            for s in scans:
                rows = session.exec(
                    select(RecordRow).where(RecordRow.scan_id == s.id).order_by(RecordRow.row_index)
                ).all()
                if not rows:
                    rows = await run_layout_for_scan_async(session, s)
                scan_rows_map[s.id] = rows[:max_rows_per_scan] if max_rows_per_scan else rows

            # Phase 2: All GLM-OCR calls across all pending scans
            batch.status = "ocr_stage"
            session.add(batch)
            session.commit()
            for s in scans:
                target_rows = scan_rows_map[s.id]
                if (
                    only_undone
                    and s.status in ("ocr_done", "triaged")
                    and target_rows
                    and all(bool(r.raw_ocr_combined) for r in target_rows)
                ):
                    continue
                if not (only_undone and s.header_raw_ocr):
                    await run_ocr_for_scan_header(session, s)
                for r in target_rows:
                    if only_undone and r.raw_ocr_combined and r.status in (
                        "ocr_done",
                        "triaged",
                        "auto_accepted",
                        "extracted",
                        "matched",
                    ):
                        continue
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "ocr",
                            "active_model": settings.model_ocr,
                            "current_scan_id": s.id,
                            "current_row_id": r.id,
                            "message_sk": f"[Dávka OCR] {s.filename} – riadok {r.row_index}",
                            "message_en": f"[Batch OCR] {s.filename} – row {r.row_index}",
                        },
                    )
                    await run_ocr_for_row(session, s, r)
                if s.status not in ("triaged", "extracted", "matched"):
                    s.status = "ocr_done"
                session.add(s)
                session.commit()

            # Phase 3: All d1-3b-mlx calls across non-census church matrika scans
            batch.status = "decision_stage"
            session.add(batch)
            session.commit()
            for s in scans:
                if only_undone and s.status == "triaged":
                    continue
                s_is_census = bool(s.detected_type == "census" or s.census_sheet_id is not None)
                if s_is_census:
                    for r in scan_rows_map[s.id]:
                        r.decision_action = "auto_accept"
                        r.decision_confidence = 0.95
                        r.status = "auto_accepted"
                        session.add(r)
                    s.status = "triaged"
                    session.add(s)
                    session.commit()
                    continue
                prev_ocr = None
                for r in scan_rows_map[s.id]:
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "decision",
                            "active_model": settings.model_decision,
                            "current_scan_id": s.id,
                            "current_row_id": r.id,
                            "message_sk": f"[Dávka Triage] {s.filename} – riadok {r.row_index}",
                            "message_en": f"[Batch Triage] {s.filename} – row {r.row_index}",
                        },
                    )
                    await run_decision_for_row(session, s, r, prev_ocr)
                    prev_ocr = r.raw_ocr_combined
                s.status = "triaged"
                session.add(s)
                session.commit()

            # Phase 4: All underdog-saluki-27b-1.0 calls across all pending scans
            batch.status = "extract_stage"
            session.add(batch)
            session.commit()
            all_facts: list[tuple[PersonFact, RecordRow]] = []
            for idx_s, s in enumerate(scans, start=1):
                s_is_census = bool(s.detected_type == "census" or s.census_sheet_id is not None)
                prev_fact = None
                scan_facts: list[tuple[PersonFact, RecordRow]] = []
                for r in scan_rows_map[s.id]:
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "extract",
                            "active_model": settings.model_extract,
                            "current_scan_id": s.id,
                            "current_row_id": r.id,
                            "message_sk": f"[Dávka Extrakcia] {s.filename} – riadok {r.row_index}",
                            "message_en": f"[Batch Extract] {s.filename} – row {r.row_index}",
                        },
                    )
                    if s_is_census:
                        fact = await run_census_extraction_for_row(session, s, r, prev_fact)
                    else:
                        fact = await run_extraction_for_row(session, s, r, prev_fact)
                    if fact is not None:
                        prev_fact = fact
                        all_facts.append((fact, r))
                        scan_facts.append((fact, r))
                if s_is_census and s.census_sheet_id:
                    reconcile_census_household_relationships(session, s.census_sheet_id)
                if s_is_census:
                    for fact, r in scan_facts:
                        session.refresh(fact)
                        evaluate_census_row_deterministic(session, s, r, fact, log_trace=False)
                s.status = "extracted"
                session.add(s)
                if s.census_sheet_id:
                    parent_sheet = session.get(CensusSheet, s.census_sheet_id)
                    if parent_sheet:
                        parent_sheet.download_status = "matched" if run_mcp else "extracted"
                        session.add(parent_sheet)
                batch.processed_scans = skipped_count + idx_s
                session.add(batch)
                session.commit()

            # Phase 5: FamilySearch MCP matching
            if run_mcp:
                batch.status = "matching"
                session.add(batch)
                session.commit()
                for fact, r in all_facts:
                    if _matches_watchlist(fact, batch.watchlist_filter):
                        await fs_mcp_manager.match_person_fact(session, fact, r)
                for s in scans:
                    s.status = "matched"
                    session.add(s)
                session.commit()

            batch.status = "completed"
            batch.processed_scans = len(all_scans)
            session.add(batch)
            session.commit()

            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "progress_pct": 100,
                    "message_sk": f"Dávka '{batch.name}' bola úspešne spracovaná!",
                    "message_en": f"Batch '{batch.name}' completed!",
                },
            )
            return {
                "batch_id": batch_id,
                "scans_processed": len(scans),
                "scans_skipped": skipped_count,
                "facts_extracted": len(all_facts),
            }
        except Exception as exc:
            batch.status = "error"
            session.add(batch)
            session.commit()
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "progress_pct": 0,
                    "message_sk": f"Chyba pri spracovaní dávky '{batch.name}': {exc}",
                    "message_en": f"Error processing batch '{batch.name}': {exc}",
                },
            )
            return {"batch_id": batch_id, "status": "error", "error": str(exc)}


async def process_census_sheet_pipeline(
    sheet_id: int,
    *,
    max_rows: int | None = None,
    run_mcp: bool = True,
) -> dict[str, Any]:
    """
    Runs the full pipeline on a CensusSheet (front scan `_v` and optional back scan `_r`
    if overflow household members exist), reconciles household relationships across rows,
    and cross-matches against both FamilySearch and local parish registers.
    Automatically detects and extracts `Prehľad sčítacieho obvodu` (`sheet_type == 'overview'`).
    """
    with Session(engine) as session:
        sheet = session.get(CensusSheet, sheet_id)
        if not sheet:
            raise ValueError(f"CensusSheet {sheet_id} not found")
        front_scan_id = sheet.front_scan_id
        back_scan_id = sheet.back_scan_id
        enable_overview = bool(settings.get_feature_flags().get("census_overview_sheets", True))
        if not sheet.house_number and sheet.sheet_type == "overview":
            sheet.sheet_type = "overview"
            session.add(sheet)
            session.commit()

        if enable_overview and sheet.sheet_type == "municipal_overview":
            front_scan = session.get(ScanImage, front_scan_id) if front_scan_id else None
            back_scan = session.get(ScanImage, back_scan_id) if back_scan_id else None
            districts = await extract_census_municipal_overview_sheet(session, sheet, front_scan, back_scan)
            return {
                "census_sheet_id": sheet.id,
                "slovakiana_key": sheet.slovakiana_key,
                "sheet_type": "municipal_overview",
                "rows_processed": len(districts),
                "municipal_districts_count": len([d for d in districts if not d.is_summary_row]),
                "download_status": sheet.download_status,
            }

        if enable_overview and sheet.sheet_type == "overview":
            front_scan = session.get(ScanImage, front_scan_id) if front_scan_id else None
            back_scan = session.get(ScanImage, back_scan_id) if back_scan_id else None
            houses = await extract_census_overview_sheet(session, sheet, front_scan, back_scan)
            session.refresh(sheet)
            return {
                "census_sheet_id": sheet.id,
                "slovakiana_key": sheet.slovakiana_key,
                "sheet_type": sheet.sheet_type,
                "rows_processed": len(houses),
                "overview_houses_count": len(houses),
                "download_status": sheet.download_status,
            }

    if not front_scan_id:
        raise ValueError(f"CensusSheet {sheet_id} has no front_scan_id downloaded yet")

    # Quick header check on front scan before running full household pipeline
    with Session(engine) as session:
        sheet = session.get(CensusSheet, sheet_id)
        front_scan = session.get(ScanImage, front_scan_id)
        back_scan = session.get(ScanImage, back_scan_id) if back_scan_id else None
        enable_overview = bool(settings.get_feature_flags().get("census_overview_sheets", True))
        if sheet and front_scan:
            if not front_scan.header_crop_path:
                await run_layout_for_scan_async(session, front_scan)
            await run_ocr_for_scan_header(session, front_scan)
            session.refresh(sheet)
            if enable_overview and (
                sheet.sheet_type == "municipal_overview"
                or detect_is_municipal_overview_sheet(front_scan.header_raw_ocr or "", sheet.slovakiana_key)
            ):
                districts = await extract_census_municipal_overview_sheet(session, sheet, front_scan, back_scan)
                return {
                    "census_sheet_id": sheet.id,
                    "slovakiana_key": sheet.slovakiana_key,
                    "sheet_type": "municipal_overview",
                    "rows_processed": len(districts),
                    "municipal_districts_count": len([d for d in districts if not d.is_summary_row]),
                    "download_status": sheet.download_status,
                }
            if enable_overview and (
                sheet.sheet_type == "overview"
                or detect_is_overview_sheet(front_scan.header_raw_ocr or "", sheet.slovakiana_key)
            ):
                houses = await extract_census_overview_sheet(session, sheet, front_scan, back_scan)
                session.refresh(sheet)
                return {
                    "census_sheet_id": sheet.id,
                    "slovakiana_key": sheet.slovakiana_key,
                    "sheet_type": sheet.sheet_type,
                    "rows_processed": len(houses),
                    "overview_houses_count": len(houses),
                    "download_status": sheet.download_status,
                }

    res_front = await process_single_scan_pipeline(
        front_scan_id,
        max_rows=max_rows,
        run_mcp=run_mcp,
    )
    total_rows_processed = int(res_front.get("rows_processed", 0))

    if back_scan_id:
        should_process_back = False
        with Session(engine) as session:
            sheet = session.get(CensusSheet, sheet_id)
            back_scan = session.get(ScanImage, back_scan_id)
            if (
                sheet
                and sheet.sheet_type not in ("overview", "municipal_overview")
                and back_scan
                and back_scan.original_path
                and Path(back_scan.original_path).exists()
            ):
                back_rows = await run_layout_for_scan_async(session, back_scan)
                should_process_back = len(back_rows) > 0
        if should_process_back:
            res_back = await process_single_scan_pipeline(
                back_scan_id,
                max_rows=max_rows,
                run_mcp=run_mcp,
            )
            total_rows_processed += int(res_back.get("rows_processed", 0))

    with Session(engine) as session:
        sheet = session.get(CensusSheet, sheet_id)
        if sheet:
            enable_overview = bool(settings.get_feature_flags().get("census_overview_sheets", True))
            if enable_overview and sheet.sheet_type == "municipal_overview":
                front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
                back_scan = session.get(ScanImage, sheet.back_scan_id) if sheet.back_scan_id else None
                districts = await extract_census_municipal_overview_sheet(session, sheet, front_scan, back_scan)
                return {
                    "census_sheet_id": sheet.id,
                    "slovakiana_key": sheet.slovakiana_key,
                    "sheet_type": "municipal_overview",
                    "rows_processed": len(districts),
                    "municipal_districts_count": len([d for d in districts if not d.is_summary_row]),
                    "download_status": sheet.download_status,
                }
            if enable_overview and sheet.sheet_type == "overview":
                front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
                back_scan = session.get(ScanImage, sheet.back_scan_id) if sheet.back_scan_id else None
                houses = await extract_census_overview_sheet(session, sheet, front_scan, back_scan)
                session.refresh(sheet)
                return {
                    "census_sheet_id": sheet.id,
                    "slovakiana_key": sheet.slovakiana_key,
                    "sheet_type": sheet.sheet_type,
                    "rows_processed": len(houses),
                    "overview_houses_count": len(houses),
                    "download_status": sheet.download_status,
                }
            household = reconcile_census_household_relationships(session, sheet.id)
            if run_mcp:
                await fs_mcp_manager.match_census_sheet(session, sheet)
            sheet.download_status = "matched" if run_mcp else "extracted"
            session.add(sheet)
            session.commit()
            return {
                "census_sheet_id": sheet.id,
                "slovakiana_key": sheet.slovakiana_key,
                "sheet_type": sheet.sheet_type,
                "rows_processed": total_rows_processed,
                "household_id": household.id if household else None,
                "download_status": sheet.download_status,
            }
    return res_front

