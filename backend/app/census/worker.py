"""
Two-Tier Slovakiana Census Worker:
- Tier 1: Resumable catalog harvester that paginates /censussheet/search across 1930 & 1940
  and indexes sheets into SQLite + FTS5.
- Tier 2: Prioritized scan downloader & pipeline orchestrator that fetches high-resolution
  front (_v) and back (_r) scans, slices the 24-column grid, runs OCR/Decision/Extraction,
  reconciles household relationships across rows, and cross-matches with FamilySearch & local Matrika.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from sqlmodel import Session, select

from backend.app.api.ws import ws_manager
from backend.app.census.slovakiana_client import slovakiana_client
from backend.app.config import settings
from backend.app.db.models import (
    Batch,
    CensusDownloadJob,
    CensusSheet,
    ScanImage,
    utcnow,
)
from backend.app.db.session import engine
from backend.app.pipeline.lexicon import (
    canonical_municipality_display,
    canonical_municipality_key,
    fold_text_diacritics,
)
from backend.app.search.fts import bulk_index_census_sheets, index_census_sheet

_ACTIVE_JOB_FLAGS: dict[int, bool] = {}
_ACTIVE_JOB_RUNTIME: dict[int, dict[str, Any]] = {}


def get_job_runtime_info(job_id: int) -> dict[str, Any] | None:
    return _ACTIVE_JOB_RUNTIME.get(job_id)

DONE_PIPELINE_STATUSES: tuple[str, ...] = ("ocr_done", "extracted", "matched")
DONE_DOWNLOAD_ONLY_STATUSES: tuple[str, ...] = (
    "downloaded",
    "sliced",
    "ocr_done",
    "extracted",
    "matched",
)


def is_census_sheet_done(
    sheet: CensusSheet,
    *,
    run_ocr: bool = True,
    overview_only: bool = False,
) -> bool:
    """Returns True if a CensusSheet has already completed the target batch stage."""
    status = (sheet.download_status or "").strip()
    if overview_only:
        is_potential_ov = bool(
            sheet.sheet_type in ("municipal_overview", "overview")
            or not (sheet.house_number and str(sheet.house_number).strip())
        )
        if not is_potential_ov:
            return True
        if status in DONE_PIPELINE_STATUSES:
            return True
        if status == "sliced" and sheet.sheet_type == "household" and sheet.front_scan_id:
            return True
        return False
    done_set = DONE_PIPELINE_STATUSES if run_ocr else DONE_DOWNLOAD_ONLY_STATUSES
    return status in done_set


def _sheet_matches_muni_filter(sheet: CensusSheet, muni_value: str | None) -> bool:
    if not muni_value or not muni_value.strip():
        return True
    m_val = muni_value.strip()
    m_folded = fold_text_diacritics(m_val)
    m_canon = canonical_municipality_key(m_val)
    s_canon = canonical_municipality_key(sheet.municipality)
    s_folded = fold_text_diacritics(sheet.municipality or sheet.municipality_folded or "")
    if sheet.municipality_key and sheet.municipality_key == m_val:
        return True
    if m_canon and s_canon == m_canon:
        return True
    if m_folded and m_folded in s_folded:
        return True
    if m_canon and m_canon in s_canon:
        return True
    return False


def _find_matching_census_sheets(
    session: Session,
    filters: dict[str, Any],
    *,
    only_undone: bool = True,
    run_ocr: bool = True,
    overview_only: bool = False,
) -> list[CensusSheet]:
    """
    Finds CensusSheet rows in the local DB matching the job filter criteria.
    When `only_undone=True`, excludes sheets that have already been processed
    (`download_status` in `DONE_PIPELINE_STATUSES` when `run_ocr=True`, or
    `DONE_DOWNLOAD_ONLY_STATUSES` when `run_ocr=False`), and orders remaining
    sheets deterministically (overview sheets first, already-downloaded sheets next,
    then by census_year, house_number_int, and id).
    """
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "find_matching_census_sheets_rs"):
            matched_ids = _rust_db.find_matching_census_sheets_rs(
                db_p,
                json.dumps(filters, ensure_ascii=False),
                only_undone=only_undone,
                run_ocr=run_ocr,
                overview_only=overview_only,
            )
            if not matched_ids:
                return []
            sheets_by_id: dict[int, CensusSheet] = {}
            for start in range(0, len(matched_ids), 450):
                chunk = matched_ids[start : start + 450]
                for s in session.exec(select(CensusSheet).where(CensusSheet.id.in_(chunk))).all():  # type: ignore[union-attr]
                    if s.id is not None:
                        sheets_by_id[s.id] = s
            return [sheets_by_id[sid] for sid in matched_ids if sid in sheets_by_id]
    except Exception:
        pass

    raw_years = filters.get("years") or [1930]
    years = [int(raw_years[0]) if raw_years and int(raw_years[0]) == 1940 else 1930]
    include_blackened = bool(filters.get("include_blackened", True))
    category_keys = [
        str(k).strip() for k in (filters.get("category_keys") or []) if str(k).strip()
    ]
    muni_keys = [k for k in category_keys if len(k) >= 4]
    dist_keys = [k for k in category_keys if len(k) == 3]
    municipality = str(filters.get("municipality") or "").strip()
    district = str(filters.get("district") or "").strip()
    search_text = str(filters.get("search_text") or "").strip()

    stmt = select(CensusSheet).where(CensusSheet.has_images == True)  # noqa: E712
    if years:
        stmt = stmt.where(CensusSheet.census_year.in_(years))  # type: ignore[union-attr]
    if not include_blackened:
        stmt = stmt.where(CensusSheet.is_blackened == False)  # noqa: E712
    if only_undone and not overview_only:
        done_statuses = list(
            DONE_PIPELINE_STATUSES if run_ocr else DONE_DOWNLOAD_ONLY_STATUSES
        )
        stmt = stmt.where(
            (CensusSheet.download_status.is_(None))  # type: ignore[union-attr]
            | (CensusSheet.download_status.notin_(done_statuses))  # type: ignore[union-attr]
        )

    all_rows = list(session.exec(stmt).all())
    matched: list[CensusSheet] = []

    eff_muni_name = municipality or (search_text if muni_keys and search_text else "")
    eff_muni_canon = canonical_municipality_key(eff_muni_name) if eff_muni_name else ""
    search_canon = canonical_municipality_key(search_text) if search_text else ""
    search_folded = fold_text_diacritics(search_text) if search_text else ""
    dist_folded = fold_text_diacritics(district) if district else ""

    for s in all_rows:
        if overview_only:
            is_potential_ov = bool(
                s.sheet_type in ("municipal_overview", "overview")
                or not (s.house_number and str(s.house_number).strip())
            )
            if not is_potential_ov:
                continue
            if only_undone and is_census_sheet_done(s, run_ocr=run_ocr, overview_only=True):
                continue
        if muni_keys or municipality:
            key_ok = bool(muni_keys and s.municipality_key in muni_keys)
            name_ok = bool(eff_muni_name and _sheet_matches_muni_filter(s, eff_muni_name))
            if not (key_ok or name_ok):
                continue
        elif dist_keys or district:
            d_key_ok = bool(dist_keys and s.district_key in dist_keys)
            s_dist_f = fold_text_diacritics(s.district or s.district_folded or "")
            d_name_ok = bool(dist_folded and dist_folded in s_dist_f)
            if not (d_key_ok or d_name_ok):
                continue

        if search_text:
            # Skip redundant text check if search_text simply repeats the matched municipality name
            s_muni_canon = canonical_municipality_key(s.municipality)
            is_same_muni_query = bool(
                (eff_muni_canon and search_canon == eff_muni_canon)
                or (search_canon and s_muni_canon and search_canon == s_muni_canon)
            )
            if not is_same_muni_query:
                haystack = fold_text_diacritics(
                    " ".join(
                        str(x)
                        for x in (
                            s.title_raw,
                            s.municipality,
                            s.street_raw,
                            s.house_number,
                            s.slovakiana_key,
                            s.household_owner,
                        )
                        if x
                    )
                )
                if search_folded not in haystack:
                    continue

        matched.append(s)

    matched.sort(
        key=lambda s: (
            0 if (s.sheet_type in ("municipal_overview", "overview") or not s.house_number) else 1,
            0
            if s.download_status in ("downloaded", "sliced", "queued")
            else (2 if s.download_status == "error" else 1),
            s.census_year or 0,
            s.house_number_int if s.house_number_int is not None else 999999,
            s.id or 0,
        )
    )
    return matched


def is_job_running_in_memory(job_id: int) -> bool:
    return bool(_ACTIVE_JOB_FLAGS.get(job_id, False))


def request_job_pause(job_id: int) -> None:
    _ACTIVE_JOB_FLAGS[job_id] = False


def upsert_census_sheet_from_dict(
    session: Session,
    data: dict[str, Any],
    existing_by_key: dict[str, CensusSheet] | None = None,
) -> CensusSheet:
    """Upserts a CensusSheet row from parsed Slovakiana search or detail metadata."""
    key = data["slovakiana_key"]
    sheet = (
        existing_by_key.get(key)
        if existing_by_key is not None
        else session.exec(select(CensusSheet).where(CensusSheet.slovakiana_key == key)).first()
    )
    if not sheet:
        sheet = CensusSheet(slovakiana_key=key)
        if existing_by_key is not None:
            existing_by_key[key] = sheet

    for field in (
        "digital_object_id",
        "urn_nbn",
        "census_year",
        "district",
        "district_key",
        "district_folded",
        "municipality",
        "municipality_key",
        "municipality_folded",
        "street_raw",
        "house_number",
        "house_number_int",
        "box_number",
        "sheet_number",
        "title_raw",
        "is_blackened",
        "has_images",
        "thumbnail_id",
    ):
        val = data.get(field)
        if val is not None:
            if isinstance(val, str) and not val.strip() and getattr(sheet, field, None):
                continue
            setattr(sheet, field, val)

    if sheet.municipality:
        sheet.municipality = canonical_municipality_display(sheet.municipality) or sheet.municipality
        sheet.municipality_folded = canonical_municipality_key(sheet.municipality)

    if "attributes" in data:
        sheet.attributes_json = json.dumps(data["attributes"], ensure_ascii=False)
    if "images" in data:
        sheet.image_urls_json = json.dumps(data["images"], ensure_ascii=False)
    sheet.updated_at = utcnow()
    session.add(sheet)
    return sheet


async def run_catalog_harvest_job(
    job_id: int,
    *,
    max_pages: int | None = None,
    page_size: int = 200,
    finalize_job: bool = True,
) -> dict[str, Any]:
    """
    Tier 1 Catalog Harvester: Paginates Slovakiana /censussheet/search, upserts CensusSheet rows,
    and indexes them into FTS5. Checkpoints cursor_page after every page for crash/pause resumability.
    """
    _ACTIVE_JOB_FLAGS[job_id] = True
    with Session(engine) as session:
        job = session.get(CensusDownloadJob, job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        filters = json.loads(job.filter_json or "{}")
        raw_years = filters.get("years") or [1930]
        years = [int(raw_years[0]) if raw_years and int(raw_years[0]) == 1940 else 1930]
        category_keys = [
            str(k).strip() for k in (filters.get("category_keys") or []) if str(k).strip()
        ]
        muni_keys = [k for k in category_keys if len(k) >= 4]
        dist_keys = [k for k in category_keys if len(k) == 3]
        municipality = str(filters.get("municipality") or "").strip()
        search_text = str(filters.get("search_text") or "").strip()
        # If a specific Slovakiana municipality category key is provided and search_text merely
        # repeats the municipality name, omit searchText so Slovakiana filters strictly by categoryKey.
        effective_search_text = search_text
        if (
            muni_keys
            and municipality
            and search_text
            and canonical_municipality_key(search_text) == canonical_municipality_key(municipality)
        ):
            effective_search_text = ""
        include_blackened = bool(filters.get("include_blackened", True))
        only_unblackened = not include_blackened
        start_page = int(job.cursor_page or 0)
        job.status = "running"
        job.updated_at = utcnow()
        session.add(job)
        session.commit()

    page = start_page
    pages_processed = 0
    total_harvested = 0
    last_total_remote = 0
    has_more = False

    try:
        while _ACTIVE_JOB_FLAGS.get(job_id, True):
            if max_pages is not None and pages_processed >= max_pages:
                break

            res = await slovakiana_client.search_sheets(
                page=page,
                size=page_size,
                only_unblackened=only_unblackened,
                years=years,
                category_keys=category_keys,
                search_text=effective_search_text,
            )
            items = res.get("items") or []
            total_remote = int(res.get("total") or 0)
            last_total_remote = total_remote
            if finalize_job or len(items) <= 40:
                await slovakiana_client.download_thumbnails_batch(items)
            else:
                priority_items = [
                    it
                    for it in items
                    if not (it.get("house_number") and str(it.get("house_number")).strip())
                ][:40]
                if priority_items:
                    await slovakiana_client.download_thumbnails_batch(priority_items)

            with Session(engine) as session:
                job = session.get(CensusDownloadJob, job_id)
                if not job or job.status == "paused":
                    break
                if finalize_job:
                    job.total_items = total_remote
                page_keys = [
                    str(it.get("slovakiana_key") or "")
                    for it in items
                    if it.get("slovakiana_key")
                ]
                existing_map: dict[str, CensusSheet] = (
                    {
                        s.slovakiana_key: s
                        for s in session.exec(
                            select(CensusSheet).where(CensusSheet.slovakiana_key.in_(page_keys))  # type: ignore[union-attr]
                        ).all()
                    }
                    if page_keys
                    else {}
                )
                upserted_sheets: list[CensusSheet] = []
                for item in items:
                    if len(muni_keys) == 1 and not item.get("municipality_key"):
                        item["municipality_key"] = muni_keys[0]
                    if len(dist_keys) == 1 and not item.get("district_key"):
                        item["district_key"] = dist_keys[0]
                    sh = upsert_census_sheet_from_dict(session, item, existing_by_key=existing_map)
                    upserted_sheets.append(sh)
                session.flush()
                bulk_index_census_sheets(session, upserted_sheets)

                if finalize_job:
                    job.completed_items += len(items)
                elif job.id in _ACTIVE_JOB_RUNTIME:
                    _ACTIVE_JOB_RUNTIME[job.id]["cataloged_sheets"] = (
                        int(_ACTIVE_JOB_RUNTIME[job.id].get("cataloged_sheets") or 0) + len(items)
                    )
                    _ACTIVE_JOB_RUNTIME[job.id]["remote_total"] = total_remote
                job.cursor_page = page + 1
                job.updated_at = utcnow()
                session.add(job)
                session.commit()

                pct = min(
                    99,
                    int(
                        100
                        * (job.completed_items if finalize_job else ((page + 1) * len(items)))
                        / max(1, total_remote)
                    ),
                )
                await ws_manager.broadcast(
                    "state",
                    {
                        "job_id": job_id,
                        "active_stage": "census_harvest",
                        "active_model": "slovakiana-wcm-api",
                        "progress_pct": pct,
                        "message_sk": f"Sťahujem katalóg sčítacích hárkov Slovakiana: {min(total_remote, (page + 1) * page_size):,} / {total_remote:,} (strana {page + 1})",
                        "message_en": f"Harvesting Slovakiana census catalog: {min(total_remote, (page + 1) * page_size):,} / {total_remote:,} (page {page + 1})",
                    },
                )

            total_harvested += len(items)
            pages_processed += 1
            page += 1
            has_more = bool(
                items and len(items) >= page_size and (page * page_size) < total_remote
            )
            if not has_more:
                break

        if finalize_job:
            with Session(engine) as session:
                job = session.get(CensusDownloadJob, job_id)
                if job:
                    if _ACTIVE_JOB_FLAGS.get(job_id, True):
                        job.status = "completed"
                    else:
                        job.status = "paused"
                    job.updated_at = utcnow()
                    session.add(job)
                    session.commit()

            done_sk = f"Katalóg sčítacích hárkov aktualizovaný ({total_harvested:,} hárkov)."
            done_en = f"Census catalog harvest finished ({total_harvested:,} sheets)."
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "progress_pct": 100,
                    "message_sk": done_sk,
                    "message_en": done_en,
                },
            )
            if total_harvested > 0:
                await ws_manager.emit_completion(
                    {
                        "status": "completed",
                        "kind": "census_harvest",
                        "job_id": job_id,
                        "rows_processed": total_harvested,
                        "title_sk": "Katalóg Slovakiana stiahnutý",
                        "title_en": "Slovakiana catalog harvested",
                        "message_sk": done_sk,
                        "message_en": done_en,
                        "target_url": "/scitacie-harky",
                    }
                )
        return {
            "job_id": job_id,
            "harvested": total_harvested,
            "pages": pages_processed,
            "has_more": has_more,
            "total_remote": last_total_remote,
        }
    except Exception as exc:
        if finalize_job:
            with Session(engine) as session:
                job = session.get(CensusDownloadJob, job_id)
                if job:
                    job.status = "error"
                    job.error_message = str(exc)
                    job.updated_at = utcnow()
                    session.add(job)
                    session.commit()
            await ws_manager.broadcast(
                "state",
                {
                    "active_stage": "idle",
                    "active_model": None,
                    "progress_pct": 0,
                    "message_sk": f"Chyba pri sťahovaní katalógu Slovakiana: {exc}",
                    "message_en": f"Slovakiana catalog harvest error: {exc}",
                },
            )
        return {"job_id": job_id, "error": str(exc), "has_more": False}
    finally:
        if finalize_job:
            _ACTIVE_JOB_FLAGS.pop(job_id, None)
            _ACTIVE_JOB_RUNTIME.pop(job_id, None)


async def download_and_process_single_census_sheet(
    sheet_id: int,
    *,
    run_pipeline: bool = True,
    run_mcp: bool = True,
    max_rows: int | None = None,
    force_sheet_type: str | None = None,
    finalize_idle: bool = True,
    overview_only: bool = False,
) -> dict[str, Any]:
    """
    Tier 2 Single-Sheet Materializer:
    1. Fetches /culturalobject/{key} and /digitalobject/{doid} from Slovakiana.
    2. Downloads front (_v) and back (_r) high-res JPEG scans to data/uploads/slovakiana/...
    3. Creates/links ScanImage rows (detected_type='census') and runs the fine-tuned census pipeline.
    """
    with Session(engine) as session:
        sheet = session.get(CensusSheet, sheet_id)
        if not sheet:
            return {"error": f"CensusSheet {sheet_id} not found"}
        if run_pipeline:
            from backend.app.pipeline.feedback import reset_ocr_feedback

            reset_ocr_feedback(session, census_sheet_id=sheet_id)
        slovakiana_key = sheet.slovakiana_key
        sheet_label = (
            f"{sheet.municipality or ''} č. {sheet.house_number or sheet.sheet_number or slovakiana_key}".strip()
        )
        initial_scan_id = sheet.front_scan_id

    await ws_manager.broadcast(
        "state",
        {
            "current_sheet_id": sheet_id,
            "current_slovakiana_key": slovakiana_key,
            "current_sheet_label": sheet_label,
            "active_stage": "census_download",
            "active_model": "slovakiana-iiif",
            "current_scan_id": initial_scan_id,
            "current_row_id": None,
            "current_row_index": None,
            "total_rows": None,
            "progress_pct": 5,
            "message_sk": f"Sťahujem metadátový záznam a skeny zo Slovakiany ({sheet_label})…",
            "message_en": f"Fetching metadata and scans from Slovakiana ({sheet_label})…",
        },
    )

    detail = await slovakiana_client.fetch_sheet_detail(slovakiana_key)

    with Session(engine) as session:
        sheet = session.get(CensusSheet, sheet_id)
        if not sheet:
            return {"error": f"CensusSheet {sheet_id} not found"}
        sheet = upsert_census_sheet_from_dict(session, detail)
        index_census_sheet(session, sheet)
        session.commit()
        session.refresh(sheet)

        images = detail.get("images") or []
        if not images:
            sheet.download_status = "cataloged"
            sheet.has_images = False
            session.add(sheet)
            session.commit()
            if finalize_idle:
                await ws_manager.broadcast(
                    "state",
                    {
                        "current_sheet_id": sheet_id,
                        "current_slovakiana_key": slovakiana_key,
                        "current_sheet_label": sheet_label,
                        "active_stage": "idle",
                        "active_model": None,
                        "current_scan_id": None,
                        "current_row_id": None,
                        "progress_pct": 100,
                        "message_sk": f"Hárok {sheet_label} nemá v Slovakiane priradené skeny.",
                        "message_en": f"Sheet {sheet_label} has no scan images on Slovakiana.",
                    },
                )
            return {"sheet_id": sheet_id, "status": "no_images"}

        # Ensure a Batch exists for this municipality & census year
        batch_name = f"Sčítanie {sheet.census_year} – {sheet.municipality or 'Slovensko'}"
        batch = session.exec(select(Batch).where(Batch.name == batch_name)).first()
        if not batch:
            batch = Batch(
                name=batch_name,
                parish_or_book=f"{sheet.municipality} ({sheet.district or 'SK'}) {sheet.census_year}",
                record_type_hint="census",
                status="uploaded",
            )
            session.add(batch)
            session.commit()
            session.refresh(batch)

        census_year = int(sheet.census_year)
        batch_id = batch.id
        thumbnail_id = sheet.thumbnail_id
        muni_slug = (sheet.municipality_folded or "obec").replace(" ", "_")[:40]
        dest_dir = (
            Path(settings.census_data_dir)
            / str(census_year)
            / f"{sheet.district_key or 'okr'}_{muni_slug}"
        )

    await slovakiana_client.download_thumbnail(
        slovakiana_key,
        thumbnail_id,
        images[0].get("thumbnail_url") if images else None,
    )

    downloaded_scan_ids: list[int] = []
    used_sides: set[str] = set()
    for idx, img_desc in enumerate(images):
        raw_side = str(
            img_desc.get("page_side")
            or ("front" if idx == 0 else ("back" if idx == 1 else f"page_{idx + 1}"))
        ).strip()
        side = raw_side if raw_side not in used_sides else f"page_{idx + 1}"
        used_sides.add(side)

        fname = f"{slovakiana_key}_{side}.jpg"
        dest_path = dest_dir / fname
        legacy_path = dest_dir / f"census_{census_year}_{slovakiana_key}_{side}.jpg"
        if legacy_path.exists() and not dest_path.exists():
            legacy_path.rename(dest_path)
        await ws_manager.broadcast(
            "state",
            {
                "current_sheet_id": sheet_id,
                "current_slovakiana_key": slovakiana_key,
                "current_sheet_label": sheet_label,
                "active_stage": "census_download",
                "active_model": "slovakiana-iiif",
                "current_scan_id": initial_scan_id,
                "current_row_id": None,
                "progress_pct": 6 + int(6 * (idx + 1) / max(1, len(images))),
                "message_sk": f"Overujem / sťahujem sken strany {idx + 1}/{len(images)} ({side}) pre {sheet_label}…",
                "message_en": f"Checking / downloading scan page {idx + 1}/{len(images)} ({side}) for {sheet_label}…",
            },
        )
        await slovakiana_client.download_image_to_jpeg(img_desc, dest_path)

        with Session(engine) as session:
            sheet = session.get(CensusSheet, sheet_id)
            if not sheet:
                break
            if side == "front":
                existing_scan_id = sheet.front_scan_id
            elif side == "back":
                existing_scan_id = sheet.back_scan_id
            else:
                existing_scan_id = None
            scan = session.get(ScanImage, existing_scan_id) if existing_scan_id else None
            if not scan:
                scan = session.exec(
                    select(ScanImage).where(
                        ScanImage.census_sheet_id == sheet.id,
                        ScanImage.page_side == side,
                    )
                ).first()
            if not scan:
                scan = ScanImage(
                    batch_id=batch_id,
                    census_sheet_id=sheet.id,
                    page_side=side,
                    filename=fname,
                    original_path=str(dest_path),
                    detected_year=str(census_year),
                    detected_type="census",
                    status="uploaded",
                )
                session.add(scan)
                session.commit()
                session.refresh(scan)
            else:
                scan.page_side = side
                scan.filename = fname
                scan.original_path = str(dest_path)
                scan.detected_year = str(census_year)
                scan.detected_type = "census"
                session.add(scan)
                session.commit()
                session.refresh(scan)

            if side == "front":
                sheet.front_scan_id = scan.id
            elif side == "back":
                sheet.back_scan_id = scan.id

            sheet.download_status = "downloaded"
            sheet.updated_at = utcnow()
            session.add(sheet)
            session.commit()
            if scan.id is not None:
                downloaded_scan_ids.append(scan.id)

    for jid, rt in list(_ACTIVE_JOB_RUNTIME.items()):
        if rt.get("current_sheet_id") == sheet_id:
            rt["downloaded_sheets"] = int(rt.get("downloaded_sheets") or 0) + 1

    if run_pipeline and downloaded_scan_ids:
        if overview_only:
            with Session(engine) as session:
                sh_check = session.get(CensusSheet, sheet_id)
                is_potential_ov = bool(
                    sh_check
                    and (
                        sh_check.sheet_type in ("municipal_overview", "overview")
                        or not (sh_check.house_number and str(sh_check.house_number).strip())
                    )
                )
            if not is_potential_ov:
                return {
                    "sheet_id": sheet_id,
                    "slovakiana_key": slovakiana_key,
                    "scan_ids": downloaded_scan_ids,
                    "status": "downloaded",
                    "skipped_household_ocr": True,
                }
        from backend.app.pipeline.orchestrator import process_census_sheet_pipeline

        pipeline_res = await process_census_sheet_pipeline(
            sheet_id,
            max_rows=max_rows,
            run_mcp=run_mcp,
            force_sheet_type=force_sheet_type,
            finalize_idle=finalize_idle,
            overview_only=overview_only,
        )
        return {
            "sheet_id": sheet_id,
            "slovakiana_key": slovakiana_key,
            "scan_ids": downloaded_scan_ids,
            "pipeline": pipeline_res,
        }

    if finalize_idle:
        done_sk = f"Hotovo: stiahnuté skeny pre hárok {sheet_label}."
        done_en = f"Done: downloaded scans for sheet {sheet_label}."
        await ws_manager.broadcast(
            "state",
            {
                "current_sheet_id": sheet_id,
                "current_slovakiana_key": slovakiana_key,
                "current_sheet_label": sheet_label,
                "active_stage": "idle",
                "active_model": None,
                "current_scan_id": downloaded_scan_ids[0] if downloaded_scan_ids else None,
                "current_row_id": None,
                "progress_pct": 100,
                "message_sk": done_sk,
                "message_en": done_en,
            },
        )
        await ws_manager.emit_completion(
            {
                "status": "completed",
                "kind": "census_sheet",
                "sheet_id": sheet_id,
                "slovakiana_key": slovakiana_key,
                "sheet_label": sheet_label,
                "title_sk": "Skeny hárku stiahnuté",
                "title_en": "Sheet scans downloaded",
                "message_sk": done_sk,
                "message_en": done_en,
                "target_url": f"/scitacie-harky/{slovakiana_key or sheet_id}",
            }
        )
    return {
        "sheet_id": sheet_id,
        "slovakiana_key": slovakiana_key,
        "scan_ids": downloaded_scan_ids,
        "status": "downloaded",
    }


async def run_download_and_pipeline_job(job_id: int) -> dict[str, Any]:
    """
    Tier 2 Batch Worker: Finds matching NOT-DONE CensusSheet records (or harvests additional
    catalog pages from Slovakiana until `limit_sheets` undone sheets are found or Slovakiana
    has no more pages), downloads their front/back high-res scans, and optionally runs the
    fine-tuned census pipeline.
    """
    _ACTIVE_JOB_FLAGS[job_id] = True
    with Session(engine) as session:
        job = session.get(CensusDownloadJob, job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        filters = json.loads(job.filter_json or "{}")
        limit_sheets = int(filters.get("limit_sheets") or 50)
        run_ocr = bool(filters.get("auto_ocr", job.job_type == "full_pipeline"))
        run_mcp = bool(filters.get("run_mcp", True))
        only_undone = bool(filters.get("only_undone", True))
        overview_only = bool(filters.get("overview_only", False))
        max_rows = filters.get("max_rows")

        undone_matches = _find_matching_census_sheets(
            session,
            filters,
            only_undone=only_undone,
            run_ocr=run_ocr,
            overview_only=overview_only,
        )
        all_matches = _find_matching_census_sheets(
            session,
            filters,
            only_undone=False,
            run_ocr=run_ocr,
            overview_only=overview_only,
        )
        existing_total_count = len(all_matches)
        already_done_count = max(0, existing_total_count - len(undone_matches))

    _ACTIVE_JOB_RUNTIME[job_id] = {
        "job_id": job_id,
        "downloaded_sheets": 0,
        "processed_sheets": 0,
        "total_sheets": len(undone_matches[:limit_sheets]),
        "current_sheet_id": None,
        "current_slovakiana_key": None,
        "current_sheet_label": None,
        "current_sheet_index": 0,
    }

    # If the local DB has fewer than `limit_sheets` undone sheets (or when `overview_only=True`
    # and we need the full village catalog so overview sheets at the end of the remote pagination
    # are discovered), harvest catalog pages from Slovakiana.
    harvest_attempts = 0
    max_harvest_pages = 25
    has_more_remote = True
    while (
        _ACTIVE_JOB_FLAGS.get(job_id, True)
        and (len(undone_matches) < limit_sheets or (overview_only and has_more_remote))
        and harvest_attempts < max_harvest_pages
    ):
        harvest_page_size = min(200, max(50, limit_sheets + existing_total_count))
        harvest_res = await run_catalog_harvest_job(
            job_id,
            max_pages=1,
            page_size=harvest_page_size,
            finalize_job=False,
        )
        harvest_attempts += 1
        has_more_remote = bool(harvest_res.get("has_more", False))
        with Session(engine) as session:
            undone_matches = _find_matching_census_sheets(
                session,
                filters,
                only_undone=only_undone,
                run_ocr=run_ocr,
                overview_only=overview_only,
            )
            all_matches = _find_matching_census_sheets(
                session,
                filters,
                only_undone=False,
                run_ocr=run_ocr,
                overview_only=overview_only,
            )
            existing_total_count = len(all_matches)
            already_done_count = max(0, existing_total_count - len(undone_matches))

        if (
            harvest_res.get("error")
            or int(harvest_res.get("harvested") or 0) == 0
            or not has_more_remote
        ):
            break

    candidates = undone_matches[:limit_sheets]
    _ACTIVE_JOB_RUNTIME[job_id]["total_sheets"] = len(candidates)

    with Session(engine) as session:
        job = session.get(CensusDownloadJob, job_id)
        if job:
            job.status = "running"
            job.total_items = len(candidates)
            job.completed_items = 0
            job.updated_at = utcnow()
            session.add(job)
            session.commit()

    processed = 0
    failed = 0
    for idx, sheet in enumerate(candidates):
        if not _ACTIVE_JOB_FLAGS.get(job_id, True):
            break
        if sheet.id is None:
            continue
        sheet_label = f"{sheet.municipality or ''} č. {sheet.house_number or sheet.sheet_number or sheet.slovakiana_key}".strip()
        _ACTIVE_JOB_RUNTIME[job_id].update(
            {
                "current_sheet_id": sheet.id,
                "current_slovakiana_key": sheet.slovakiana_key,
                "current_sheet_label": sheet_label,
                "current_sheet_index": idx + 1,
                "total_sheets": len(candidates),
            }
        )
        try:
            await ws_manager.broadcast(
                "state",
                {
                    "job_id": job_id,
                    "job_sheet_index": idx + 1,
                    "job_total_sheets": len(candidates),
                    "current_sheet_id": sheet.id,
                    "current_slovakiana_key": sheet.slovakiana_key,
                    "current_sheet_label": sheet_label,
                    "active_stage": "census_download",
                    "active_model": "slovakiana-iiif",
                    "progress_pct": max(5, int(100 * idx / max(1, len(candidates)))),
                    "message_sk": f"Sťahujem a spracúvam hárok {idx + 1}/{len(candidates)}: {sheet_label}",
                    "message_en": f"Downloading & processing sheet {idx + 1}/{len(candidates)}: {sheet_label}",
                },
            )
            dp_kwargs: dict[str, Any] = {
                "run_pipeline": run_ocr,
                "run_mcp": run_mcp,
                "max_rows": max_rows,
            }
            if overview_only:
                dp_kwargs["overview_only"] = overview_only
            await download_and_prepare_or_process(
                sheet.id,
                **dp_kwargs,
            )
            processed += 1
            _ACTIVE_JOB_RUNTIME[job_id]["processed_sheets"] = processed
            _ACTIVE_JOB_RUNTIME[job_id]["downloaded_sheets"] = max(
                int(_ACTIVE_JOB_RUNTIME[job_id].get("downloaded_sheets") or 0),
                processed,
            )
        except Exception as exc:
            failed += 1
            with Session(engine) as session:
                sh = session.get(CensusSheet, sheet.id)
                if sh:
                    sh.download_status = "error"
                    sh.error_message = str(exc)
                    session.add(sh)
                    session.commit()

        with Session(engine) as session:
            job = session.get(CensusDownloadJob, job_id)
            if job:
                job.completed_items = processed
                job.failed_items = failed
                job.updated_at = utcnow()
                session.add(job)
                session.commit()

    was_not_paused = _ACTIVE_JOB_FLAGS.get(job_id, True)
    with Session(engine) as session:
        job = session.get(CensusDownloadJob, job_id)
        if job:
            job.status = "completed" if was_not_paused else "paused"
            job.updated_at = utcnow()
            session.add(job)
            session.commit()

    _ACTIVE_JOB_FLAGS.pop(job_id, None)
    _ACTIVE_JOB_RUNTIME.pop(job_id, None)

    done_msg_sk = (
        f"Všetky vybrané sčítacie hárky sú už spracované ({already_done_count} hotových, 0 zostávajúcich)."
        if processed == 0 and failed == 0 and already_done_count > 0
        else f"Hotovo: stiahnutých a spracovaných {processed} sčítacích hárkov."
    )
    done_msg_en = (
        f"All selected census sheets are already processed ({already_done_count} done, 0 remaining)."
        if processed == 0 and failed == 0 and already_done_count > 0
        else f"Done: downloaded and processed {processed} census sheets."
    )

    await ws_manager.broadcast(
        "state",
        {
            "active_stage": "idle",
            "active_model": None,
            "current_sheet_id": None,
            "current_slovakiana_key": None,
            "current_sheet_label": None,
            "progress_pct": 100,
            "message_sk": done_msg_sk,
            "message_en": done_msg_en,
        },
    )
    if was_not_paused:
        await ws_manager.emit_completion(
            {
                "status": "completed" if failed == 0 or processed > 0 else "error",
                "kind": "census_job",
                "job_id": job_id,
                "rows_processed": processed,
                "title_sk": "Hromadné OCR dokončené" if run_ocr else "Hromadné sťahovanie dokončené",
                "title_en": "Batch census OCR completed" if run_ocr else "Batch census download completed",
                "message_sk": done_msg_sk,
                "message_en": done_msg_en,
                "target_url": "/scitacie-harky",
            }
        )
    return {
        "job_id": job_id,
        "processed": processed,
        "failed": failed,
        "skipped_done": already_done_count,
    }


async def download_and_prepare_or_process(
    sheet_id: int,
    *,
    run_pipeline: bool = True,
    run_mcp: bool = True,
    max_rows: int | None = None,
    overview_only: bool = False,
) -> dict[str, Any]:
    return await download_and_process_single_census_sheet(
        sheet_id,
        run_pipeline=run_pipeline,
        run_mcp=run_mcp,
        max_rows=max_rows,
        finalize_idle=False,
        overview_only=overview_only,
    )
