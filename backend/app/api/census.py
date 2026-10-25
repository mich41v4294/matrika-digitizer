from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import select

from backend.app.api.scans import resolve_scan_media_url
from backend.app.census.slovakiana_client import (
    TAG_KEY_TO_YEAR,
    WCM_BASE_URL,
    YEAR_TO_TAG_KEY,
    slovakiana_client,
)
from backend.app.config import settings
from backend.app.census.worker import (
    download_and_process_single_census_sheet,
    is_job_running_in_memory,
    request_job_pause,
    run_catalog_harvest_job,
    run_download_and_pipeline_job,
    upsert_census_sheet_from_dict,
)
from backend.app.db.models import (
    CensusDownloadJob,
    CensusHousehold,
    CensusMunicipalOverviewDistrict,
    CensusOverviewHouse,
    CensusSheet,
    FamilySearchMatch,
    LocalRecordMatch,
    PersonFact,
    RecordRow,
    ScanImage,
)
from backend.app.db.session import SessionDep
from backend.app.pipeline.feedback import build_feedback_lookup_maps, reset_ocr_feedback
from backend.app.mcp.familysearch_client import (
    match_census_to_local_matrika,
    schedule_auto_fs_match,
    trigger_auto_fs_match_for_fact,
)
from backend.app.pipeline.census_layout import (
    classify_overview_rows_structure,
    detect_is_municipal_overview_sheet,
    detect_is_overview_sheet,
    extract_municipal_overview_header_metadata,
    extract_overview_header_metadata,
    is_occupancy_status_token,
    is_valid_commissioner_or_revisor_name,
)
from backend.app.pipeline.extraction_stage import (
    _KNOWN_OVERVIEW_ROW_REPAIRS,
    _format_birth_place_raw,
    _heuristic_parse_overview_row,
    _is_garbled_or_english_owner_ocr,
    _normalize_municipal_commissioner_name,
    _normalize_overview_owner,
    _reconcile_municipal_overview_rows,
    extract_census_municipal_overview_sheet,
    extract_census_overview_sheet,
    reconcile_census_household_relationships,
    reconstruct_municipal_overview_districts_from_existing_ocr,
    reconstruct_overview_houses_from_existing_ocr,
)
from backend.app.pipeline.lexicon import (
    apply_person_fact_search_fields,
    can_match_overview_house_to_sheet,
    canonical_municipality_display,
    canonical_municipality_key,
    extract_person_fact_name_candidates,
    fold_text_diacritics,
    fuzzy_name_matches,
    is_compatible_settlement,
    normalize_census_relation,
    normalize_given_name,
    normalize_house_number,
    normalize_religion,
    normalize_village,
    owner_surname_folded_from_name,
    strip_slovak_feminine_suffix,
    text_matches_street,
)
from backend.app.pipeline.schemas import validate_census_date, validate_person_name, validate_sex
from backend.app.search.fts import (
    get_search_progress,
    index_census_sheet,
    index_person_fact,
    update_search_progress,
)

router = APIRouter(prefix="/api/census", tags=["census"])


def _census_storage_root() -> Path:
    return Path(
        getattr(settings, "census_data_dir", None) or (Path(settings.upload_dir) / "slovakiana")
    )


def _scan_to_media_url(scan: ScanImage | None) -> str | None:
    return resolve_scan_media_url(scan)


def _serialize_scan(scan: ScanImage | None) -> dict[str, Any] | None:
    if not scan:
        return None
    return {
        **scan.model_dump(),
        "image_url": _scan_to_media_url(scan),
    }


_THUMBNAIL_CACHE: dict[str, tuple[float, set[str]]] = {}


def _get_existing_thumbs_set(thumb_dir: Path) -> set[str]:
    key = str(thumb_dir)
    now = time.monotonic()
    cached = _THUMBNAIL_CACHE.get(key)
    if cached is not None and (now - cached[0]) < 15.0:
        return cached[1]
    existing: set[str] = set()
    try:
        if thumb_dir.is_dir():
            existing = {p.name for p in thumb_dir.iterdir() if p.is_file()}
    except OSError:
        pass
    _THUMBNAIL_CACHE[key] = (now, existing)
    return existing


def _census_thumbnail_url(slovakiana_key: str, census_root: Path) -> str | None:
    fname = f"{slovakiana_key}.jpg"
    if fname in _get_existing_thumbs_set(census_root / "thumbnails"):
        return f"/media/uploads/slovakiana/thumbnails/{fname}"
    fallback_dir = Path(settings.upload_dir) / "slovakiana" / "thumbnails"
    if fname in _get_existing_thumbs_set(fallback_dir):
        return f"/media/uploads/slovakiana/thumbnails/{fname}"
    return None


def _serialize_sheet(
    s: CensusSheet,
    overview_houses_count: int = 0,
    municipal_districts_count: int = 0,
) -> dict[str, Any]:
    imgs = json.loads(s.image_urls_json or "[]")
    front_url = imgs[0].get("jpeg_url") if imgs else None
    back_url = imgs[1].get("jpeg_url") if len(imgs) > 1 else None
    census_root = _census_storage_root()
    thumb_url = _census_thumbnail_url(s.slovakiana_key, census_root)
    if not thumb_url and s.thumbnail_id:
        thumb_url = f"{WCM_BASE_URL}/image/download/{s.thumbnail_id}?sizeIndex=0"
    elif not thumb_url and imgs and imgs[0].get("thumbnail_url"):
        thumb_url = imgs[0].get("thumbnail_url")

    effective_type = s.sheet_type or "household"
    canonical_muni = canonical_municipality_display(s.municipality) or s.municipality
    comm_name = s.commissioner_name
    if comm_name and not is_valid_commissioner_or_revisor_name(
        comm_name, municipality=canonical_muni, district=s.district
    ):
        comm_name = None

    return {
        **s.model_dump(),
        "municipality": canonical_muni,
        "sheet_type": effective_type,
        "obvod_number": s.obvod_number,
        "commissioner_name": comm_name,
        "overview_houses_count": overview_houses_count,
        "municipal_districts_count": municipal_districts_count,
        "cair_id": s.slovakiana_key,
        "slovakiana_id": s.slovakiana_key,
        "district_name": s.district,
        "municipality_name": canonical_muni,
        "street_name": s.street_raw,
        "house_number_raw": s.house_number,
        "house_number_conscription": s.house_number,
        "thumbnail_url": thumb_url,
        "front_image_url": front_url,
        "back_image_url": back_url,
        "source_web_url": f"https://slovakiana.sk/scitacie-harky/{s.slovakiana_key}",
        "pipeline_status": s.download_status,
    }


_ROMAN_ORDER: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}

# Historical / Slovakiana / FamilySearch district aliases per canonical municipality key
_MUNICIPALITY_HISTORICAL_DISTRICT_ALIASES: dict[str, set[str]] = {
    "horny cepen": {"trnava", "galanta", "sered", "772", "712"},
    "dolny cepen": {"trnava", "galanta", "sered", "772", "712"},
    "stredny cepen": {"trnava", "galanta", "sered", "772", "712"},
    "velke surovce": {"trnava", "galanta", "sered", "772", "712"},
    "male surovce": {"trnava", "galanta", "sered", "772", "712"},
    "surovce": {"trnava", "galanta", "sered", "772", "712"},
    "valtasur": {"trnava", "galanta", "sered", "772", "712"},
    "varov sur": {"trnava", "galanta", "sered", "772", "712"},
    "majcichov": {"trnava", "galanta", "772", "712"},
    "vlckovce": {"trnava", "galanta", "772", "712"},
    "opoj": {"trnava", "galanta", "772", "712"},
    "krizovany nad dudvahom": {"trnava", "galanta", "772", "712"},
    "zemianske sady": {"galanta", "trnava", "hlohovec", "712", "772", "715"},
    "salgocka": {"galanta", "hlohovec", "trnava", "712", "715", "772"},
    "horne zelenice": {"hlohovec", "trnava", "galanta", "715", "772", "712"},
    "dolne zelenice": {"hlohovec", "trnava", "galanta", "715", "772", "712"},
    "siladice": {"hlohovec", "trnava", "715", "772"},
    "velke ripnany": {"nitra", "topolcany", "piestany", "740", "770", "748"},
    "male ripnany": {"nitra", "topolcany", "piestany", "740", "770", "748"},
}


def _sheet_matches_municipality(s: CensusSheet, municipality: str | None) -> bool:
    if not municipality or not municipality.strip():
        return True
    m_val = municipality.strip()
    m_folded = fold_text_diacritics(m_val)
    m_canon = canonical_municipality_key(m_val)
    s_canon = canonical_municipality_key(s.municipality)
    s_folded = fold_text_diacritics(s.municipality or s.municipality_folded or "")
    if s.municipality_key and s.municipality_key == m_val:
        return True
    if m_canon and s_canon == m_canon:
        return True
    if m_folded and m_folded in s_folded:
        return True
    if m_canon and m_canon in s_canon:
        return True
    return False


def _sheet_matches_district(
    s: CensusSheet,
    district: str | None,
    *,
    municipality_specified: bool = False,
) -> bool:
    if not district or not district.strip():
        return True
    import re as _re

    d_raw = _re.sub(r"^\s*okr(?:es)?\.?\s*", "", district.strip(), flags=_re.IGNORECASE).strip()
    d_folded = fold_text_diacritics(d_raw)
    if not d_folded:
        return True
    if s.district_key and s.district_key == d_raw:
        return True
    s_dist_folded = fold_text_diacritics(s.district or s.district_folded or "")
    if d_folded in s_dist_folded:
        return True

    m_canon = canonical_municipality_key(s.municipality)
    aliases = _MUNICIPALITY_HISTORICAL_DISTRICT_ALIASES.get(m_canon, set())
    if municipality_specified:
        if d_folded in aliases or d_raw in aliases:
            return True
    else:
        # Allow historical/modern district aliases (e.g. Galanta/Sereď for Horný Čepeň & Veľké Šúrovce,
        # Topoľčany/Piešťany for Veľké Ripňany) without cross-polluting primary DB districts like Trnava/Hlohovec
        if d_folded in ("galanta", "sered", "topolcany", "piestany", "712", "770", "748") and (
            d_folded in aliases or d_raw in aliases
        ):
            return True
    return False


def _parse_birth_year_range(birth_year: str | int | None) -> tuple[int, int] | None:
    if birth_year is None:
        return None
    import re as _re

    raw = str(birth_year).strip()
    if not raw:
        return None
    # Range: "1890-1900", "1890 - 1900", "1890–1900", "1890..1900", "1890 do 1900"
    m_range = _re.match(r"^\s*(\d{4})\s*(?:[-–—]+|\.\.+|do)\s*(\d{4})\s*$", raw, flags=_re.IGNORECASE)
    if m_range:
        y1, y2 = int(m_range.group(1)), int(m_range.group(2))
        return (min(y1, y2), max(y1, y2))
    # Tolerance: "1896±2", "1896+-2", "1896+/-2"
    m_tol = _re.match(r"^\s*(\d{4})\s*(?:±|\+-|\+/-)\s*(\d{1,2})\s*$", raw)
    if m_tol:
        yr, tol = int(m_tol.group(1)), int(m_tol.group(2))
        return (yr - tol, yr + tol)
    # Approximate prefix: "~1896", "cca 1896", "okolo 1896"
    m_approx = _re.match(r"^\s*(?:~|cca\.?|okolo)\s*(\d{4})\s*$", raw, flags=_re.IGNORECASE)
    if m_approx:
        yr = int(m_approx.group(1))
        return (yr - 2, yr + 2)
    # Open-ended ranges: "1890-" or "-1900"
    m_open_start = _re.match(r"^\s*(\d{4})\s*[-–—]+\s*$", raw)
    if m_open_start:
        return (int(m_open_start.group(1)), 2100)
    m_open_end = _re.match(r"^\s*[-–—]+\s*(\d{4})\s*$", raw)
    if m_open_end:
        return (1600, int(m_open_end.group(1)))
    # Exact 4-digit year: "1896"
    m_exact = _re.match(r"^\s*(\d{4})\s*$", raw)
    if m_exact:
        yr = int(m_exact.group(1))
        return (yr, yr)
    return None


def _extract_fact_birth_years(pf: PersonFact) -> list[int]:
    import re as _re

    years: list[int] = []
    if pf.birth_year_est is not None:
        try:
            years.append(int(pf.birth_year_est))
        except Exception:
            pass
    if pf.birth_or_event_date:
        for m in _re.finditer(r"\b(1[6789]\d{2}|20\d{2})\b", str(pf.birth_or_event_date)):
            yr = int(m.group(1))
            if yr not in years:
                years.append(yr)
    return years


def _fact_matches_birth_year(pf: PersonFact, birth_year: str | int | None) -> bool:
    if birth_year is None or not str(birth_year).strip():
        return True
    raw = str(birth_year).strip()
    yr_range = _parse_birth_year_range(raw)
    years = _extract_fact_birth_years(pf)
    if yr_range is not None:
        y_min, y_max = yr_range
        return any(y_min <= y <= y_max for y in years)
    raw_f = fold_text_diacritics(raw)
    return any(raw_f in str(y) for y in years) or (
        raw_f in fold_text_diacritics(pf.birth_or_event_date or "")
    )


def _text_matches_person_name(haystack_parts: list[str | None], person_name: str | None) -> bool:
    if not person_name or not person_name.strip():
        return True
    return fuzzy_name_matches(person_name.strip(), haystack_parts, threshold=0.76)


def _fact_matches_person_name(
    pf: PersonFact,
    person_name: str | None,
    *,
    ocr_text: str | None = None,
) -> bool:
    if not person_name or not person_name.strip():
        return True
    p_clean = person_name.strip()
    struct_cands = extract_person_fact_name_candidates(pf, ocr_text=None)
    if _text_matches_person_name(struct_cands, p_clean):
        return True
    p_fold = fold_text_diacritics(p_clean)
    if pf.observationes_raw:
        if p_fold and p_fold in fold_text_diacritics(pf.observationes_raw):
            return True
    if ocr_text:
        if p_fold and p_fold in fold_text_diacritics(ocr_text):
            return True
        if not struct_cands and _text_matches_person_name([ocr_text], p_clean):
            return True
    return False


def _overview_house_matches_person_query(
    ovh: CensusOverviewHouse,
    person_query: str | None,
) -> bool:
    if not person_query or not person_query.strip():
        return True
    p_clean = person_query.strip()
    if _text_matches_person_name([ovh.owner_name, ovh.owner_surname_folded, ovh.remarks], p_clean):
        return True
    if ovh.raw_ocr_text:
        p_fold = fold_text_diacritics(p_clean)
        if p_fold and p_fold in fold_text_diacritics(ovh.raw_ocr_text):
            return True
        if not ovh.owner_name and _text_matches_person_name([ovh.raw_ocr_text], p_clean):
            return True
    return False


def _fact_matches_birth_place(pf: PersonFact, birth_place: str | None) -> bool:
    if not birth_place or not birth_place.strip():
        return True
    import re as _re

    bp_val = birth_place.strip()
    bp_folded = fold_text_diacritics(bp_val)
    bp_canon = canonical_municipality_key(bp_val)
    place_parts = [
        pf.birth_place_raw,
        pf.birth_place_sk,
        pf.birth_place_municipality,
        pf.birth_place_district,
        pf.birth_place_country,
        pf.home_jurisdiction,
        pf.previous_residence,
        pf.village_raw,
        pf.village_sk,
    ]
    combined_folded = fold_text_diacritics(" ".join(p for p in place_parts if p))
    if bp_folded and bp_folded in combined_folded:
        return True

    q_toks = [t for t in _re.split(r"[\s,;()/]+", bp_folded) if t and t not in ("okr", "okres")]
    if q_toks and all(t in combined_folded for t in q_toks):
        return True

    for cand_m in (
        pf.birth_place_municipality,
        pf.birth_place_sk,
        (pf.birth_place_raw or "").split(",")[0],
        (pf.home_jurisdiction or "").split(",")[0],
    ):
        if not cand_m:
            continue
        cand_canon = canonical_municipality_key(cand_m)
        if bp_canon and cand_canon == bp_canon:
            return True
        if cand_canon and bp_folded in _MUNICIPALITY_HISTORICAL_DISTRICT_ALIASES.get(cand_canon, set()):
            return True
    return False


def _serialize_person_preview(
    pf: PersonFact,
    *,
    matched_filter: bool = False,
) -> dict[str, Any]:
    given = pf.given_name_sk or pf.given_name_raw or ""
    surname = pf.surname_raw or pf.surname_normalized or ""
    rel_low = (pf.relation_to_head or "").strip().lower()
    is_child_or_male = pf.sex == "M" or rel_low in {
        "syn",
        "dcéra",
        "vnuk",
        "vnučka",
        "chovanec",
        "chovanka",
        "adoptívny syn",
        "adoptívna dcéra",
        "nevlastný syn",
        "nevlastná dcéra",
    }
    maiden = None if is_child_or_male else pf.mother_maiden_surname
    full_name = f"{given} {surname}".strip()
    if maiden:
        full_name = f"{full_name} (rod. {maiden})".strip()
    years = _extract_fact_birth_years(pf)
    byear = years[0] if years else None
    bplace = (
        pf.birth_place_raw
        or pf.birth_place_sk
        or pf.birth_place_municipality
        or None
    )
    return {
        "id": pf.id,
        "given_name": given or None,
        "surname": surname or None,
        "surname_normalized": pf.surname_normalized or None,
        "maiden_surname": maiden or None,
        "full_name": full_name or None,
        "relation_to_head": pf.relation_to_head,
        "birth_date": pf.birth_or_event_date,
        "birth_year": byear,
        "birth_place": bplace,
        "domicile": pf.home_jurisdiction,
        "matched_filter": matched_filter,
    }


def _build_person_previews(
    facts: list[PersonFact],
    *,
    row_ocr_by_id: dict[int, str] | None = None,
    person_name: str | None = None,
    birth_year: str | int | None = None,
    birth_place: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    has_pn = bool(person_name and person_name.strip())
    has_by = bool(birth_year and str(birth_year).strip())
    has_bp = bool(birth_place and birth_place.strip())
    has_q = bool(q and q.strip())
    ocr_map = row_ocr_by_id or {}
    previews: list[dict[str, Any]] = []
    for pf in facts:
        is_pf_match = False
        pf_ocr = ocr_map.get(pf.row_id)
        if has_pn or has_by or has_bp:
            is_pf_match = (
                (not has_pn or _fact_matches_person_name(pf, person_name, ocr_text=pf_ocr))
                and (not has_by or _fact_matches_birth_year(pf, birth_year))
                and (not has_bp or _fact_matches_birth_place(pf, birth_place))
            )
        elif has_q and q:
            is_pf_match = (
                _fact_matches_person_name(pf, q, ocr_text=pf_ocr)
                or _fact_matches_birth_year(pf, q)
                or _fact_matches_birth_place(pf, q)
            )
        previews.append(_serialize_person_preview(pf, matched_filter=is_pf_match))
    return previews


def _with_member_previews(
    payload: dict[str, Any],
    previews: list[dict[str, Any]],
    *,
    include_extracted_members: bool = False,
) -> dict[str, Any]:
    payload["extracted_persons_preview"] = previews
    payload["members_preview"] = previews
    if include_extracted_members:
        payload["extracted_members"] = previews
    return payload


def _serialize_household(hh: CensusHousehold | None) -> dict[str, Any] | None:
    if not hh:
        return None
    notes: dict[str, Any] = {}
    if hh.notes_raw:
        try:
            notes = json.loads(hh.notes_raw)
        except Exception:
            notes = {}
    dwelling_unit = hh.dwelling_unit or notes.get("dwelling_unit") or "I"
    head_name = notes.get("head_of_household_name") or hh.house_owner_name
    wife_name = notes.get("wife_name") or notes.get("wife_given_name")
    wife_maiden = notes.get("wife_maiden_name") or notes.get("wife_maiden_surname")
    members_cnt = notes.get("members_count", notes.get("member_count", 0))
    head_surname = notes.get("head_surname") or hh.house_owner_name
    consensus_surname = (
        notes.get("consensus_surname")
        or (strip_slovak_feminine_suffix(head_surname) if head_surname else None)
        or head_surname
    )
    return {
        **hh.model_dump(),
        "id": hh.id,
        "dwelling_unit": dwelling_unit,
        "head_of_household_name": head_name,
        "wife_name": wife_name,
        "wife_maiden_name": wife_maiden,
        "members_count": members_cnt,
        "consensus_surname": consensus_surname,
        "head_surname": head_surname,
        "head_given_name": notes.get("head_given_name"),
        "wife_given_name": notes.get("wife_given_name"),
        "wife_maiden_surname": wife_maiden,
        "member_count": members_cnt,
    }


class StartCensusJobRequest(BaseModel):
    name: str | None = None
    job_type: str = "catalog_harvest"  # "catalog_harvest" | "download_and_pipeline" | "overview_only"
    census_year_tag: str | None = "10001"  # "10001" (1930) | "10003" (1940) — always single year (1930 default)
    census_year: int | None = None
    district_tag: str | None = None
    municipality_tag: str | None = None
    municipality: str | None = None
    district: str | None = None
    query_text: str | None = None
    max_sheets: int | None = 100
    run_pipeline: bool = True
    run_mcp: bool = True
    only_undone: bool = True
    overview_only: bool = False


class RemoteSearchRequest(BaseModel):
    query_text: str | None = None
    year_tag: str | None = "10001"
    census_year: int | None = None
    district_tag: str | None = None
    municipality_tag: str | None = None
    page: int = 1
    page_size: int = 20
    save_to_catalog: bool = True


def _resolve_single_census_year(
    census_year: int | str | None = None,
    year_tag: str | None = None,
) -> int:
    """
    Resolves census year to a single year (1930 or 1940), defaulting to 1930.
    Never allows selecting both years at once.
    """
    if year_tag and str(year_tag).strip() in TAG_KEY_TO_YEAR:
        yr = TAG_KEY_TO_YEAR[str(year_tag).strip()]
        return 1940 if yr == 1940 else 1930
    if census_year is not None and str(census_year).strip().isdigit():
        yr = int(str(census_year).strip())
        return 1940 if yr == 1940 else 1930
    return 1930


class CreateCensusPersonRequest(BaseModel):
    person_row_number: int | str | None = None
    surname_raw: str | None = None
    maiden_surname: str | None = None
    mother_maiden_surname: str | None = None
    given_name_raw: str | None = None
    relation_to_head: str | None = None
    sex: str | None = None
    birth_or_event_date: str | None = None
    birth_year_est: int | None = None
    marital_status: str | None = None
    marriage_date: str | None = None
    marriage_date_raw: str | None = None
    children_born_count: int | str | None = None
    children_living_count: int | str | None = None
    children_deceased_count: int | str | None = None
    children_dead_count: int | str | None = None
    birth_place_municipality: str | None = None
    birth_place_district: str | None = None
    birth_place_country: str | None = None
    birth_place_raw: str | None = None
    moved_in_date: str | None = None
    previous_residence: str | None = None
    home_jurisdiction: str | None = None
    citizenship: str | None = None
    nationality: str | None = None
    religion: str | None = None
    literacy: str | None = None
    social_status_or_occupation: str | None = None
    occupation_position: str | None = None
    employer_workplace: str | None = None
    presence_type: str | None = "trvale"
    observationes_raw: str | None = None
    spouse_name: str | None = None
    father_name: str | None = None
    mother_name: str | None = None


class UpdateCensusPersonRequest(CreateCensusPersonRequest):
    pass


class SetSheetTypeRequest(BaseModel):
    sheet_type: str = "overview"  # "overview" | "municipal_overview" | "household"
    obvod_number: str | None = None
    commissioner_name: str | None = None
    run_extract: bool = True


class OverviewHousePatchRequest(BaseModel):
    ordinal_number: int | str | None = None
    settlement_name: str | None = None
    house_number: str | None = None
    is_inhabited: bool | None = None
    owner_name: str | None = None
    household_count: int | None = None
    present_persons_count: int | None = None
    remarks: str | None = None
    matched_sheet_id: int | None = None


class MunicipalOverviewDistrictPatchRequest(BaseModel):
    ordinal_number: int | str | None = None
    settlement_name: str | None = None
    obvod_number: str | None = None
    commissioner_name: str | None = None
    houses_total: int | None = None
    houses_inhabited: int | None = None
    houses_uninhabited: int | None = None
    household_count: int | None = None
    present_persons_count: int | None = None
    remarks: str | None = None
    is_summary_row: bool | None = None
    matched_overview_sheet_id: int | None = None


@router.get("/facets")
async def get_census_facets(
    district_tag: str | None = None,
    year_tag: str | None = None,
) -> dict[str, Any]:
    """
    Fetches live Slovakiana facets (years 1930/1940, districts, and municipalities).
    When `district_tag` is provided, queries Slovakiana for the exact municipalities in that district.
    Always resolves to a single year (1930 by default).
    """
    try:
        eff_year = _resolve_single_census_year(year_tag=year_tag)
        eff_year_tag = YEAR_TO_TAG_KEY[eff_year]
        only_unblackened = eff_year == 1930
        facets = await slovakiana_client.fetch_facets(only_unblackened=only_unblackened)
        years = facets.get("years") or []
        districts = facets.get("districts") or []
        if district_tag and district_tag.strip():
            cities = await slovakiana_client.fetch_cities_for_district(
                district_tag=district_tag.strip(),
                year_tag=eff_year_tag,
                only_unblackened=only_unblackened,
            )
        else:
            cities = facets.get("municipalities") or []
        return {
            "years": years,
            "districts": districts,
            "cities": cities,
            "online": True,
        }
    except Exception as exc:
        return {
            "years": [
                {"tagKey": "10001", "name": "1930", "count": 599068},
                {"tagKey": "10003", "name": "1940", "count": 647859},
            ],
            "districts": [],
            "cities": [],
            "online": False,
            "error": str(exc),
        }


@router.get("/stats")
def get_census_stats(session: SessionDep) -> dict[str, Any]:
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None:
            return json.loads(_rust_db.get_census_stats_rs(db_p))
    except Exception:
        pass

    total_sheets = session.exec(select(func.count()).select_from(CensusSheet)).one()
    sheets_1930 = session.exec(
        select(func.count()).select_from(CensusSheet).where(CensusSheet.census_year == 1930)
    ).one()
    sheets_1940 = session.exec(
        select(func.count()).select_from(CensusSheet).where(CensusSheet.census_year == 1940)
    ).one()
    downloaded_sheets = session.exec(
        select(func.count())
        .select_from(CensusSheet)
        .where(CensusSheet.download_status.in_(["downloaded", "sliced", "ocr_done", "extracted", "matched"]))
    ).one()
    processed_sheets = session.exec(
        select(func.count())
        .select_from(CensusSheet)
        .where(CensusSheet.download_status.in_(["extracted", "matched"]))
    ).one()
    census_facts = session.exec(
        select(func.count()).select_from(PersonFact).where(PersonFact.record_type == "census")
    ).one()
    local_cross_matches = session.exec(
        select(func.count()).select_from(LocalRecordMatch)
    ).one()

    distinct_municipalities = session.exec(
        select(CensusSheet.municipality, CensusSheet.district, func.count(CensusSheet.id))
        .group_by(CensusSheet.municipality, CensusSheet.district)
        .order_by(func.count(CensusSheet.id).desc())
        .limit(200)
    ).all()

    distinct_districts = session.exec(
        select(CensusSheet.district, func.count(CensusSheet.id))
        .where(CensusSheet.district.is_not(None))
        .group_by(CensusSheet.district)
        .order_by(CensusSheet.district.asc())
    ).all()

    return {
        "total_cataloged_sheets": total_sheets,
        "sheets_1930": sheets_1930,
        "sheets_1940": sheets_1940,
        "downloaded_sheets": downloaded_sheets,
        "processed_sheets": processed_sheets,
        "extracted_census_persons": census_facts,
        "local_matrika_cross_matches": local_cross_matches,
        "top_municipalities": [
            {"municipality": m[0], "district": m[1], "count": m[2]}
            for m in distinct_municipalities
            if m[0]
        ],
        "local_districts": [
            {"district": d[0], "count": d[1]}
            for d in distinct_districts
            if d[0]
        ],
    }


@router.get("/jobs")
def list_census_jobs(session: SessionDep) -> dict[str, Any]:
    jobs = session.exec(
        select(CensusDownloadJob).order_by(CensusDownloadJob.id.desc()).limit(30)
    ).all()
    dirty = False
    for j in jobs:
        if j.status == "running" and j.id is not None and not is_job_running_in_memory(j.id):
            if j.total_items and j.completed_items >= j.total_items:
                j.status = "completed"
            else:
                j.status = "paused"
            session.add(j)
            dirty = True
    if dirty:
        session.commit()

    out: list[dict[str, Any]] = []
    for j in jobs:
        flt = json.loads(j.filter_json or "{}")
        out.append(
            {
                **j.model_dump(),
                "name": flt.get("name") or f"{j.job_type} #{j.id}",
                "total_sheets": j.total_items,
                "cataloged_sheets": j.completed_items if j.job_type == "catalog_harvest" else j.total_items,
                "downloaded_sheets": j.completed_items if j.job_type != "catalog_harvest" else 0,
                "processed_sheets": j.completed_items if j.job_type == "full_pipeline" else 0,
                "failed_sheets": j.failed_items,
                "current_page": j.cursor_page,
                "is_active_runtime": bool(j.id and is_job_running_in_memory(j.id)),
            }
        )
    return {"jobs": out}


@router.delete("/jobs/{job_id}")
@router.post("/jobs/{job_id}/cancel")
def delete_or_cancel_census_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    job = session.get(CensusDownloadJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="CensusDownloadJob not found")
    request_job_pause(job_id)
    session.delete(job)
    session.commit()
    return {"status": "deleted", "job_id": job_id}


@router.post("/jobs/start")
async def start_census_job(
    req: StartCensusJobRequest,
    background_tasks: BackgroundTasks,
    session: SessionDep,
) -> dict[str, Any]:
    resolved_year = _resolve_single_census_year(req.census_year, req.census_year_tag)
    years = [resolved_year]
    cat_keys = [k for k in (req.district_tag, req.municipality_tag) if k]
    is_overview_only = bool(req.overview_only or req.job_type == "overview_only")
    job_name = req.name or (
        f"Katalóg Slovakiana ({req.municipality or req.query_text or req.municipality_tag or req.district_tag or 'všetky'})"
        if req.job_type == "catalog_harvest"
        else (
            f"Stiahnuť hárky a OCR prehľadov ({req.municipality or req.query_text or req.municipality_tag or 'výber'})"
            if is_overview_only
            else f"Sťahovanie a OCR hárkov ({req.municipality or req.query_text or req.municipality_tag or 'výber'})"
        )
    )
    db_job_type = "catalog_harvest" if req.job_type == "catalog_harvest" else "full_pipeline"
    filter_payload = {
        "name": job_name,
        "years": years,
        "category_keys": cat_keys,
        "municipality": (req.municipality or "").strip() or None,
        "district": (req.district or "").strip() or None,
        "search_text": req.query_text or "",
        "include_blackened": True,
        "limit_sheets": req.max_sheets or (500 if is_overview_only else 50),
        "auto_ocr": req.run_pipeline,
        "run_mcp": req.run_mcp,
        "only_undone": req.only_undone,
        "overview_only": is_overview_only,
    }
    job = CensusDownloadJob(
        job_type=db_job_type,
        filter_json=json.dumps(filter_payload, ensure_ascii=False),
        status="running",
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    if req.job_type == "catalog_harvest":
        max_pages = max(1, ((req.max_sheets or 200) + 199) // 200)
        background_tasks.add_task(
            run_catalog_harvest_job,
            job.id or 0,
            max_pages=max_pages,
            page_size=min(200, req.max_sheets or 200),
        )
    else:
        background_tasks.add_task(
            run_download_and_pipeline_job,
            job.id or 0,
        )

    return {"status": "started", "job": job}


@router.post("/jobs/{job_id}/pause")
def pause_census_job(job_id: int, session: SessionDep) -> dict[str, Any]:
    job = session.get(CensusDownloadJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="CensusDownloadJob not found")
    request_job_pause(job_id)
    job.status = "paused"
    job.updated_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()
    session.refresh(job)
    return {"status": "paused", "job": job}


@router.post("/jobs/{job_id}/resume")
async def resume_census_job(
    job_id: int,
    background_tasks: BackgroundTasks,
    session: SessionDep,
) -> dict[str, Any]:
    job = session.get(CensusDownloadJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="CensusDownloadJob not found")
    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()
    session.refresh(job)

    if job.job_type == "catalog_harvest":
        background_tasks.add_task(run_catalog_harvest_job, job.id or 0)
    else:
        background_tasks.add_task(run_download_and_pipeline_job, job.id or 0)

    return {"status": "resumed", "job": job}


@router.get("/sheets")
def list_census_sheets(
    session: SessionDep,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    q: str | None = None,
    census_year: int | None = 1930,
    municipality: str | None = None,
    district: str | None = None,
    street: str | None = None,
    person_name: str | None = None,
    person_query: str | None = None,
    owner_query: str | None = None,
    birth_year: str | None = None,
    birth_place: str | None = None,
    download_status: str | None = None,
    pipeline_status: str | None = None,
    limit: int = 80,
    offset: int = 0,
) -> dict[str, Any]:
    effective_person_name = person_name or person_query or owner_query
    person_name = effective_person_name
    resolved_year = _resolve_single_census_year(census_year)
    stmt = (
        select(CensusSheet)
        .where(CensusSheet.census_year == resolved_year)
        .order_by(CensusSheet.id.desc())
    )
    status_filter = pipeline_status or download_status
    if status_filter:
        if status_filter == "idle":
            stmt = stmt.where(CensusSheet.download_status == "cataloged")
        elif status_filter in ("pending", "pending_ocr", "undone", "not_done"):
            stmt = stmt.where(
                (CensusSheet.download_status.is_(None))  # type: ignore[union-attr]
                | (CensusSheet.download_status.notin_(["ocr_done", "extracted", "matched"]))  # type: ignore[union-attr]
            )
        else:
            stmt = stmt.where(CensusSheet.download_status == status_filter)

    all_matching = list(session.exec(stmt).all())

    has_muni = bool(municipality and municipality.strip())
    if has_muni and municipality:
        all_matching = [s for s in all_matching if _sheet_matches_municipality(s, municipality)]
        target_canon = canonical_municipality_key(municipality)
        exact_muni = [
            s
            for s in all_matching
            if canonical_municipality_key(s.municipality) == target_canon
            or s.municipality_key == municipality.strip()
        ]
        if exact_muni:
            all_matching = exact_muni

    if district and district.strip():
        dist_matching = [
            s
            for s in all_matching
            if _sheet_matches_district(s, district, municipality_specified=has_muni)
        ]
        if dist_matching or not has_muni:
            all_matching = dist_matching

    available_streets_set: set[str] = {
        s.street_raw.strip()
        for s in all_matching
        if s.street_raw and s.street_raw.strip() and s.street_raw.strip() not in ("-", "—", "?")
    }

    has_st = bool(street and street.strip())
    has_pn = bool(person_name and person_name.strip())
    has_by = bool(birth_year and str(birth_year).strip())
    has_bp = bool(birth_place and birth_place.strip())
    has_q = bool(q and q.strip())
    has_person_or_q_filter = has_st or has_pn or has_by or has_bp or has_q

    t_search_start = time.perf_counter()
    scanned_total = len(all_matching)
    query_label = " · ".join(
        str(x).strip()
        for x in (municipality, district, street, person_name, birth_year, birth_place, q)
        if x and str(x).strip()
    )
    if has_person_or_q_filter:
        update_search_progress(
            running=True,
            scope="census",
            query=query_label,
            processed=0,
            total=max(1, scanned_total),
            matched_records=0,
            matched_sheets=0,
            progress_pct=12,
            stage_sk="Prehľadávam sčítacie hárky a prehľady obvodov…",
            stage_en="Searching census sheets and district overviews…",
            elapsed_ms=0,
        )

    candidate_ids = [s.id for s in all_matching if s.id is not None]
    households_by_sheet: dict[int, CensusHousehold] = {}
    all_households_by_sheet: dict[int, list[CensusHousehold]] = {}
    facts_by_sheet: dict[int, list[PersonFact]] = {}
    overview_counts_by_sheet: dict[int, int] = {}
    municipal_districts_counts_by_sheet: dict[int, int] = {}
    muni_districts_by_sheet: dict[int, list[CensusMunicipalOverviewDistrict]] = {}
    matched_ovh_owners_by_sheet: dict[int, list[str]] = {}
    matched_ovh_settlements_by_sheet: dict[int, list[str]] = {}
    ovh_by_overview_sheet: dict[int, list[CensusOverviewHouse]] = {}
    row_ocr_by_id: dict[int, str] = {}

    if has_person_or_q_filter and candidate_ids:
        for hh in session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id.in_(candidate_ids))
            .order_by(CensusHousehold.id.asc())
        ).all():
            all_households_by_sheet.setdefault(hh.census_sheet_id, []).append(hh)
            if hh.street and hh.street.strip() and hh.street.strip() not in ("-", "—", "?"):
                available_streets_set.add(hh.street.strip())
            if hh.census_sheet_id not in households_by_sheet or (hh.dwelling_unit or "I") == "I":
                if hh.census_sheet_id not in households_by_sheet:
                    households_by_sheet[hh.census_sheet_id] = hh
        all_sheet_facts = list(
            session.exec(
                select(PersonFact)
                .where(PersonFact.census_sheet_id.in_(candidate_ids))
                .order_by(PersonFact.id.asc())
            ).all()
        )
        for pf in all_sheet_facts:
            if pf.census_sheet_id:
                facts_by_sheet.setdefault(pf.census_sheet_id, []).append(pf)
        f_row_ids = list({pf.row_id for pf in all_sheet_facts if pf.row_id})
        if f_row_ids:
            for r_obj in session.exec(select(RecordRow).where(RecordRow.id.in_(f_row_ids))).all():
                if r_obj.id is not None:
                    ocr_t = " ".join(
                        p for p in (r_obj.raw_ocr_combined, r_obj.raw_ocr_left, r_obj.raw_ocr_right) if p
                    )
                    if ocr_t:
                        row_ocr_by_id[r_obj.id] = ocr_t
        seen_filter_ovh: set[tuple[int, str]] = set()
        for ovh in session.exec(
            select(CensusOverviewHouse).where(
                (CensusOverviewHouse.census_sheet_id.in_(candidate_ids))
                | (CensusOverviewHouse.matched_sheet_id.in_(candidate_ids))
            )
        ).all():
            ov_k = (
                int(ovh.census_sheet_id or 0),
                normalize_house_number(ovh.house_number) or str(ovh.row_index),
            )
            if ov_k not in seen_filter_ovh:
                seen_filter_ovh.add(ov_k)
                overview_counts_by_sheet[ovh.census_sheet_id] = (
                    overview_counts_by_sheet.get(ovh.census_sheet_id, 0) + 1
                )
                ovh_by_overview_sheet.setdefault(ovh.census_sheet_id, []).append(ovh)
            if ovh.settlement_name and ovh.settlement_name.strip() and ovh.settlement_name.strip() not in ("-", "—", "?"):
                available_streets_set.add(ovh.settlement_name.strip())
            if ovh.matched_sheet_id:
                if ovh.settlement_name:
                    matched_ovh_settlements_by_sheet.setdefault(ovh.matched_sheet_id, []).append(ovh.settlement_name)
                for tx in (ovh.owner_name, ovh.remarks):
                    if tx:
                        matched_ovh_owners_by_sheet.setdefault(ovh.matched_sheet_id, []).append(tx)
        for md in session.exec(
            select(CensusMunicipalOverviewDistrict).where(
                CensusMunicipalOverviewDistrict.census_sheet_id.in_(candidate_ids)
            )
        ).all():
            muni_districts_by_sheet.setdefault(md.census_sheet_id, []).append(md)
            if not md.is_summary_row:
                municipal_districts_counts_by_sheet[md.census_sheet_id] = (
                    municipal_districts_counts_by_sheet.get(md.census_sheet_id, 0) + 1
                )

        filtered_sheets: list[CensusSheet] = []
        step_size = max(1, len(all_matching) // 8)
        for idx_s, s in enumerate(all_matching, start=1):
            sid = s.id or 0
            s_facts = facts_by_sheet.get(sid, [])
            s_hhs = all_households_by_sheet.get(sid, [])
            s_ovhs = ovh_by_overview_sheet.get(sid, [])

            if has_st and street:
                street_candidates: list[str | None] = [
                    s.street_raw,
                    *(hh_item.street for hh_item in s_hhs),
                    *matched_ovh_settlements_by_sheet.get(sid, []),
                    *(ovh_item.settlement_name for ovh_item in s_ovhs),
                ]
                if not text_matches_street(street_candidates, street):
                    continue
            name_parts: list[str | None] = [
                s.household_owner,
                s.commissioner_name,
                *matched_ovh_owners_by_sheet.get(sid, []),
            ]
            for ovh_item in s_ovhs:
                name_parts.extend([ovh_item.owner_name, ovh_item.remarks])
            for md_item in muni_districts_by_sheet.get(sid, []):
                name_parts.extend([md_item.commissioner_name, md_item.remarks])
            for hh_item in s_hhs:
                hh_ser = _serialize_household(hh_item)
                if hh_ser:
                    name_parts.extend(
                        [
                            hh_ser.get("head_of_household_name"),
                            hh_ser.get("head_given_name"),
                            hh_ser.get("head_surname"),
                            hh_ser.get("wife_name"),
                            hh_ser.get("wife_given_name"),
                            hh_ser.get("wife_maiden_name"),
                        ]
                    )

            if has_q and q:
                q_val = q.strip()
                q_folded = fold_text_diacritics(q_val)
                meta_text = fold_text_diacritics(
                    " ".join(
                        filter(
                            None,
                            [
                                s.municipality,
                                s.district,
                                s.street_raw,
                                s.house_number,
                                s.slovakiana_key,
                                s.household_owner,
                            ],
                        )
                    )
                )
                q_match = (
                    (q_folded in meta_text)
                    or _text_matches_person_name(name_parts, q_val)
                    or any(
                        _fact_matches_person_name(pf, q_val, ocr_text=row_ocr_by_id.get(pf.row_id))
                        or _fact_matches_birth_year(pf, q_val)
                        or _fact_matches_birth_place(pf, q_val)
                        for pf in s_facts
                    )
                    or any(_overview_house_matches_person_query(ovh_item, q_val) for ovh_item in s_ovhs)
                )
                if not q_match:
                    continue

            if has_pn or has_by or has_bp:
                fact_combined_match = any(
                    (
                        not has_pn
                        or _fact_matches_person_name(pf, person_name, ocr_text=row_ocr_by_id.get(pf.row_id))
                    )
                    and (not has_by or _fact_matches_birth_year(pf, birth_year))
                    and (not has_bp or _fact_matches_birth_place(pf, birth_place))
                    for pf in s_facts
                )
                header_name_match = (
                    has_pn
                    and (
                        _text_matches_person_name(name_parts, person_name)
                        or any(_overview_house_matches_person_query(ovh_item, person_name) for ovh_item in s_ovhs)
                    )
                    and (not has_by or any(_fact_matches_birth_year(pf, birth_year) for pf in s_facts))
                    and (not has_bp or any(_fact_matches_birth_place(pf, birth_place) for pf in s_facts))
                )
                if not (fact_combined_match or header_name_match):
                    continue

            filtered_sheets.append(s)
            if idx_s % step_size == 0 or idx_s == len(all_matching):
                pct = min(90, max(15, int(round((idx_s / max(len(all_matching), 1)) * 85))))
                update_search_progress(
                    running=True,
                    scope="census",
                    query=query_label,
                    processed=idx_s,
                    total=max(1, len(all_matching)),
                    matched_records=0,
                    matched_sheets=len(filtered_sheets),
                    progress_pct=pct,
                    stage_sk="Prehľadávam sčítacie hárky a prehľady obvodov…",
                    stage_en="Searching census sheets and district overviews…",
                    elapsed_ms=int(round((time.perf_counter() - t_search_start) * 1000)),
                )
        all_matching = filtered_sheets

    total = len(all_matching)
    page_items = all_matching[offset : offset + limit]

    sheet_ids = [s.id for s in page_items if s.id is not None]
    if sheet_ids and not has_person_or_q_filter:
        for hh in session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id.in_(sheet_ids))
            .order_by(CensusHousehold.id.asc())
        ).all():
            all_households_by_sheet.setdefault(hh.census_sheet_id, []).append(hh)
            if hh.census_sheet_id not in households_by_sheet or (hh.dwelling_unit or "I") == "I":
                if hh.census_sheet_id not in households_by_sheet:
                    households_by_sheet[hh.census_sheet_id] = hh
        for pf in session.exec(
            select(PersonFact)
            .where(PersonFact.census_sheet_id.in_(sheet_ids))
            .order_by(PersonFact.id.asc())
        ).all():
            if pf.census_sheet_id:
                facts_by_sheet.setdefault(pf.census_sheet_id, []).append(pf)
        seen_page_ovh: set[tuple[int, str]] = set()
        for ovh in session.exec(
            select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id.in_(sheet_ids))
        ).all():
            ov_k = (
                int(ovh.census_sheet_id or 0),
                normalize_house_number(ovh.house_number) or str(ovh.row_index),
            )
            if ov_k not in seen_page_ovh:
                seen_page_ovh.add(ov_k)
                overview_counts_by_sheet[ovh.census_sheet_id] = (
                    overview_counts_by_sheet.get(ovh.census_sheet_id, 0) + 1
                )
        for md in session.exec(
            select(CensusMunicipalOverviewDistrict).where(
                CensusMunicipalOverviewDistrict.census_sheet_id.in_(sheet_ids)
            )
        ).all():
            if not md.is_summary_row:
                municipal_districts_counts_by_sheet[md.census_sheet_id] = (
                    municipal_districts_counts_by_sheet.get(md.census_sheet_id, 0) + 1
                )

    census_root = _census_storage_root()
    linked_by_sheet_id = _batch_find_linked_house_sheets(session, page_items)
    results: list[dict[str, Any]] = []
    missing_thumbs: list[dict[str, Any]] = []
    for s in page_items:
        sid = s.id or 0
        hh = households_by_sheet.get(sid)
        sheet_hhs = all_households_by_sheet.get(sid, [])
        s_facts = facts_by_sheet.get(sid, [])
        ov_cnt = overview_counts_by_sheet.get(sid, 0)
        md_cnt = municipal_districts_counts_by_sheet.get(sid, 0)
        linked_sheets = linked_by_sheet_id.get(sid, [])

        persons_preview: list[dict[str, Any]] = []
        for pf in s_facts:
            is_pf_match = False
            pf_ocr = row_ocr_by_id.get(pf.row_id)
            if has_pn or has_by or has_bp:
                is_pf_match = (
                    (not has_pn or _fact_matches_person_name(pf, person_name, ocr_text=pf_ocr))
                    and (not has_by or _fact_matches_birth_year(pf, birth_year))
                    and (not has_bp or _fact_matches_birth_place(pf, birth_place))
                )
            elif has_q and q:
                is_pf_match = (
                    _fact_matches_person_name(pf, q, ocr_text=pf_ocr)
                    or _fact_matches_birth_year(pf, q)
                    or _fact_matches_birth_place(pf, q)
                )
            persons_preview.append(_serialize_person_preview(pf, matched_filter=is_pf_match))

        s_payload = _serialize_sheet(
            s,
            overview_houses_count=ov_cnt,
            municipal_districts_count=md_cnt,
        )
        if not str(s_payload.get("thumbnail_url") or "").startswith("/media/") and s.thumbnail_id:
            missing_thumbs.append({"slovakiana_key": s.slovakiana_key, "thumbnail_id": s.thumbnail_id})
        if not s_payload.get("street_name"):
            fallback_st = (
                (hh.street if hh and hh.street else None)
                or next((h_it.street for h_it in sheet_hhs if h_it.street), None)
                or next(iter(matched_ovh_settlements_by_sheet.get(sid, [])), None)
            )
            if fallback_st:
                s_payload["street_name"] = fallback_st
        if has_st:
            s_payload["matched_street"] = True
        s_payload["linked_house_sheets"] = linked_sheets
        s_payload["extracted_persons_preview"] = persons_preview
        s_payload["members_preview"] = persons_preview
        results.append(
            {
                "sheet": s_payload,
                "household": _serialize_household(hh),
                "households": [sh for sh in (_serialize_household(h) for h in sheet_hhs) if sh is not None],
                "extracted_members_count": len(s_facts),
                "extracted_persons_preview": persons_preview,
                "members_preview": persons_preview,
                "extracted_members": persons_preview,
                "overview_houses_count": ov_cnt,
                "linked_house_sheets": linked_sheets,
            }
        )

    if missing_thumbs and background_tasks is not None:
        background_tasks.add_task(slovakiana_client.download_thumbnails_batch, missing_thumbs)

    elapsed_ms = int(round((time.perf_counter() - t_search_start) * 1000))
    if has_person_or_q_filter:
        census_progress = update_search_progress(
            running=False,
            scope="census",
            query=query_label,
            processed=scanned_total,
            total=max(1, scanned_total),
            matched_records=0,
            matched_sheets=total,
            progress_pct=100,
            stage_sk="Vyhľadávanie dokončené",
            stage_en="Search complete",
            elapsed_ms=elapsed_ms,
            last_finished_at=time.time(),
        )
    else:
        census_progress = {
            **get_search_progress(),
            "running": False,
            "scope": "census",
            "processed": scanned_total,
            "total": max(1, scanned_total),
            "matched_sheets": total,
            "progress_pct": 100,
            "elapsed_ms": elapsed_ms,
        }

    return {
        "total": total,
        "scanned_total": scanned_total,
        "sheets": results,
        "available_streets": sorted(available_streets_set),
        "search_progress": census_progress,
    }


filter_census_sheets = list_census_sheets


async def _resolve_census_sheet(session: SessionDep, sheet_ref: str) -> CensusSheet:
    """
    Resolves a CensusSheet by either its Slovakiana identifier (`cair-test001`)
    or its numeric database ID (`45`). If a `cair-*` identifier is not in the local
    DB yet, fetches it from Slovakiana and upserts it on the fly.
    """
    from backend.app.census.slovakiana_client import SLOVAKIANA_KEY_RE

    ref = str(sheet_ref or "").strip()
    if not ref.isdigit() and not SLOVAKIANA_KEY_RE.fullmatch(ref):
        raise HTTPException(status_code=400, detail=f"Invalid slovakiana_key format: '{ref}'")
    sheet = session.exec(select(CensusSheet).where(CensusSheet.slovakiana_key == ref)).first()
    if not sheet and ref.isdigit():
        sheet = session.get(CensusSheet, int(ref))
    if not sheet and ref.startswith("cair-"):
        try:
            detail = await slovakiana_client.fetch_sheet_detail(ref)
            sheet = upsert_census_sheet_from_dict(session, detail)
            session.commit()
            session.refresh(sheet)
            index_census_sheet(session, sheet)
            session.commit()
            imgs = detail.get("images") or []
            await slovakiana_client.download_thumbnail(
                ref,
                sheet.thumbnail_id,
                imgs[0].get("thumbnail_url") if imgs else None,
            )
        except Exception:
            sheet = None
    if not sheet:
        raise HTTPException(status_code=404, detail=f"CensusSheet '{sheet_ref}' not found")
    return sheet


@router.post("/sheets/remote-search")
async def remote_search_slovakiana(
    req: RemoteSearchRequest,
    session: SessionDep,
) -> dict[str, Any]:
    """
    Searches live `wcm.slovakiana.sk/censussheet/search`, downloads local thumbnails
    for all returned hits, and indexes them into our local SQLite + FTS5 catalog.
    Always searches a single census year (1930 by default).
    """
    resolved_year = _resolve_single_census_year(req.census_year, req.year_tag)
    years = [resolved_year]
    cat_keys = [k for k in (req.district_tag, req.municipality_tag) if k]

    data = await slovakiana_client.search_sheets(
        page=max(0, req.page - 1),
        size=req.page_size,
        years=years,
        category_keys=cat_keys,
        search_text=req.query_text or "",
    )
    items = data.get("items") or []
    total_remote = int(data.get("total") or len(items))

    # Download local thumbnails for all search hits concurrently
    await slovakiana_client.download_thumbnails_batch(items)

    saved_sheets: list[CensusSheet] = []
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
        if (req.save_to_catalog and page_keys)
        else {}
    )
    for parsed in items:
        if req.municipality_tag and not parsed.get("municipality_key"):
            parsed["municipality_key"] = req.municipality_tag
        if req.district_tag and not parsed.get("district_key"):
            parsed["district_key"] = req.district_tag
        if req.save_to_catalog and parsed.get("slovakiana_key"):
            sheet = upsert_census_sheet_from_dict(session, parsed, existing_by_key=existing_map)
            saved_sheets.append(sheet)

    if req.save_to_catalog and saved_sheets:
        from backend.app.search.fts import bulk_index_census_sheets

        session.flush()
        bulk_index_census_sheets(session, saved_sheets)
        session.commit()
        for s in saved_sheets:
            session.refresh(s)

    return {
        "total_remote": total_remote,
        "page": req.page,
        "saved_count": len(saved_sheets),
        "sheets": [
            {
                "sheet": _serialize_sheet(s),
                "household": None,
                "extracted_members_count": 0,
            }
            for s in saved_sheets
        ],
    }


def _build_linked_house_sheet_entry(
    session: SessionDep,
    other_sheet: CensusSheet,
    hh: CensusHousehold | None = None,
    facts: list[PersonFact] | None = None,
    *,
    skip_db_lookup: bool = False,
) -> dict[str, Any]:
    if hh is None and not skip_db_lookup and other_sheet.id:
        hh = session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id == other_sheet.id)
            .order_by(CensusHousehold.id.asc())
        ).first()
    if facts is None and not skip_db_lookup and other_sheet.id:
        facts = list(
            session.exec(
                select(PersonFact)
                .where(PersonFact.census_sheet_id == other_sheet.id)
                .order_by(PersonFact.id.asc())
            ).all()
        )
    facts_list = facts or []
    hh_ser = _serialize_household(hh) if hh else None
    head_name = None
    if hh_ser:
        head_name = f"{hh_ser.get('head_given_name') or ''} {hh_ser.get('head_surname') or ''}".strip() or None
    if not head_name and hh and hh.house_owner_name:
        head_name = hh.house_owner_name
    if not head_name and other_sheet.household_owner:
        head_name = other_sheet.household_owner
    if not head_name and facts_list:
        first_fact = facts_list[0]
        g_first = first_fact.given_name_sk or first_fact.given_name_raw or ""
        s_first = first_fact.surname_normalized or first_fact.surname_raw or ""
        head_name = f"{g_first} {s_first}".strip() or None

    persons_preview: list[dict[str, Any]] = []
    for pf in facts_list[:12]:
        persons_preview.append(
            {
                "given_name": pf.given_name_sk or pf.given_name_raw,
                "surname": pf.surname_normalized or pf.surname_raw,
                "relation_to_head": pf.relation_to_head,
                "birth_year": pf.birth_year_est,
            }
        )

    canonical_muni = canonical_municipality_display(other_sheet.municipality) or other_sheet.municipality
    return {
        "id": other_sheet.id,
        "cair_id": other_sheet.slovakiana_key,
        "slovakiana_key": other_sheet.slovakiana_key,
        "slovakiana_id": other_sheet.slovakiana_key,
        "census_year": other_sheet.census_year,
        "municipality": canonical_muni,
        "district": other_sheet.district,
        "settlement": other_sheet.street_raw or canonical_muni,
        "house_number": other_sheet.house_number,
        "enumeration_district": other_sheet.obvod_number,
        "sheet_number": other_sheet.sheet_number,
        "box_number": other_sheet.box_number,
        "is_blackened": other_sheet.is_blackened,
        "download_status": other_sheet.download_status,
        "head_of_household": head_name,
        "persons_count": len(facts_list),
        "persons_preview": persons_preview,
    }


def _find_matching_village_house_sheets(
    session: SessionDep,
    sheet: CensusSheet,
) -> list[CensusSheet]:
    eff_type = sheet.sheet_type or "household"
    if eff_type in ("overview", "municipal_overview"):
        return []
    target_muni = canonical_municipality_key(sheet.municipality)
    target_hn = normalize_house_number(sheet.house_number)
    if not target_muni or not target_hn:
        return []

    stmt = select(CensusSheet).where(CensusSheet.house_number.is_not(None))
    if sheet.id is not None:
        stmt = stmt.where(CensusSheet.id != sheet.id)
    if sheet.house_number_int is not None:
        stmt = stmt.where(
            (CensusSheet.house_number_int == sheet.house_number_int)
            | (CensusSheet.house_number == str(sheet.house_number).strip())
        )

    candidates = list(session.exec(stmt).all())
    matched_others: list[CensusSheet] = []
    for other in candidates:
        other_type = other.sheet_type or "household"
        if other_type in ("overview", "municipal_overview"):
            continue
        if canonical_municipality_key(other.municipality) != target_muni:
            continue
        if sheet.municipality_key and other.municipality_key and sheet.municipality_key != other.municipality_key:
            continue
        if normalize_house_number(other.house_number) != target_hn:
            continue
        if not is_compatible_settlement(sheet.street_raw, other.street_raw, sheet.municipality):
            continue
        matched_others.append(other)
    return matched_others


def _find_linked_house_sheets(
    session: SessionDep,
    sheet: CensusSheet,
) -> list[dict[str, Any]]:
    """
    Finds all cross-year household `CensusSheet` records (`other.id != sheet.id`,
    `other.census_year != sheet.census_year`, and `other.sheet_type != 'overview'`)
    that belong to the exact same canonical village (`canonical_municipality_key`) and exact same
    normalized house number (`normalize_house_number`), connecting 1930 <-> 1940 sheets for the same house.
    """
    cross_year = [
        o
        for o in _find_matching_village_house_sheets(session, sheet)
        if o.census_year != sheet.census_year
    ]
    cross_year.sort(
        key=lambda o: (
            o.census_year or 0,
            o.sheet_number or "",
            o.id or 0,
        )
    )
    return [_build_linked_house_sheet_entry(session, o) for o in cross_year]


def _batch_find_linked_house_sheets(
    session: SessionDep,
    sheets: list[CensusSheet],
) -> dict[int, list[dict[str, Any]]]:
    """
    Batch-resolves cross-year linked household sheets for a list of CensusSheets using
    at most 3 SQL queries instead of 3 * len(sheets) N+1 queries.
    """
    valid_targets: list[tuple[CensusSheet, str, str]] = []
    hn_ints: set[int] = set()
    hn_strs: set[str] = set()
    for s in sheets:
        if s.id is None or (s.sheet_type or "household") in ("overview", "municipal_overview"):
            continue
        t_muni = canonical_municipality_key(s.municipality)
        t_hn = normalize_house_number(s.house_number)
        if not t_muni or not t_hn:
            continue
        valid_targets.append((s, t_muni, t_hn))
        if s.house_number_int is not None:
            hn_ints.add(s.house_number_int)
        if s.house_number:
            hn_strs.add(str(s.house_number).strip())

    if not valid_targets:
        return {}

    stmt = select(CensusSheet).where(CensusSheet.house_number.is_not(None))
    if hn_ints or hn_strs:
        clauses = []
        if hn_ints:
            clauses.append(CensusSheet.house_number_int.in_(hn_ints))  # type: ignore[union-attr]
        if hn_strs:
            clauses.append(CensusSheet.house_number.in_(hn_strs))  # type: ignore[union-attr]
        if len(clauses) == 2:
            stmt = stmt.where(clauses[0] | clauses[1])
        elif len(clauses) == 1:
            stmt = stmt.where(clauses[0])

    candidates = list(session.exec(stmt).all())
    by_muni_hn: dict[tuple[str, str], list[CensusSheet]] = {}
    for c in candidates:
        if (c.sheet_type or "household") in ("overview", "municipal_overview"):
            continue
        c_muni = canonical_municipality_key(c.municipality)
        c_hn = normalize_house_number(c.house_number)
        if c_muni and c_hn:
            by_muni_hn.setdefault((c_muni, c_hn), []).append(c)

    matched_by_target: dict[int, list[CensusSheet]] = {}
    needed_other_ids: set[int] = set()
    for s, t_muni, t_hn in valid_targets:
        sid = s.id or 0
        cands = by_muni_hn.get((t_muni, t_hn), [])
        cross: list[CensusSheet] = []
        for other in cands:
            if other.id == s.id or other.census_year == s.census_year:
                continue
            if s.municipality_key and other.municipality_key and s.municipality_key != other.municipality_key:
                continue
            if not is_compatible_settlement(s.street_raw, other.street_raw, s.municipality):
                continue
            cross.append(other)
            if other.id is not None:
                needed_other_ids.add(other.id)
        if cross:
            cross.sort(
                key=lambda o: (
                    o.census_year or 0,
                    o.sheet_number or "",
                    o.id or 0,
                )
            )
            matched_by_target[sid] = cross

    if not needed_other_ids:
        return {}

    other_id_list = list(needed_other_ids)
    hh_by_other_id: dict[int, CensusHousehold] = {}
    for hh in session.exec(
        select(CensusHousehold)
        .where(CensusHousehold.census_sheet_id.in_(other_id_list))  # type: ignore[union-attr]
        .order_by(CensusHousehold.id.asc())
    ).all():
        if hh.census_sheet_id not in hh_by_other_id:
            hh_by_other_id[hh.census_sheet_id] = hh

    facts_by_other_id: dict[int, list[PersonFact]] = {}
    for pf in session.exec(
        select(PersonFact)
        .where(PersonFact.census_sheet_id.in_(other_id_list))  # type: ignore[union-attr]
        .order_by(PersonFact.id.asc())
    ).all():
        if pf.census_sheet_id is not None:
            facts_by_other_id.setdefault(pf.census_sheet_id, []).append(pf)

    entry_cache: dict[int, dict[str, Any]] = {}
    out: dict[int, list[dict[str, Any]]] = {}
    for sid, cross_list in matched_by_target.items():
        entries: list[dict[str, Any]] = []
        for o in cross_list:
            oid = o.id or 0
            if oid not in entry_cache:
                entry_cache[oid] = _build_linked_house_sheet_entry(
                    session,
                    o,
                    hh=hh_by_other_id.get(oid),
                    facts=facts_by_other_id.get(oid, []),
                    skip_db_lookup=True,
                )
            entries.append(entry_cache[oid])
        out[sid] = entries
    return out


def _find_same_year_house_sheets(
    session: SessionDep,
    sheet: CensusSheet,
) -> list[dict[str, Any]]:
    """
    Finds sibling household `CensusSheet` records from the SAME census year (`other.census_year == sheet.census_year`)
    for the exact same village and house number (e.g. multiple apartment/byt sheets for the same house in 1940).
    """
    same_year = [
        o
        for o in _find_matching_village_house_sheets(session, sheet)
        if o.census_year == sheet.census_year
    ]
    same_year.sort(
        key=lambda o: (
            o.sheet_number or "",
            o.id or 0,
        )
    )
    return [_build_linked_house_sheet_entry(session, o) for o in same_year]


def _prefetch_sheets_hh_and_facts(
    session: SessionDep,
    sheets: list[CensusSheet],
) -> tuple[dict[int, CensusHousehold], dict[int, list[PersonFact]]]:
    sids = list({cs.id for cs in sheets if cs.id is not None})
    if not sids:
        return {}, {}
    hh_map: dict[int, CensusHousehold] = {}
    for hh in session.exec(
        select(CensusHousehold)
        .where(CensusHousehold.census_sheet_id.in_(sids))  # type: ignore[union-attr]
        .order_by(CensusHousehold.id.asc())
    ).all():
        if hh.census_sheet_id not in hh_map or (hh.dwelling_unit or "I") == "I":
            if hh.census_sheet_id not in hh_map:
                hh_map[hh.census_sheet_id] = hh
    facts_map: dict[int, list[PersonFact]] = {sid: [] for sid in sids}
    for pf in session.exec(
        select(PersonFact)
        .where(PersonFact.census_sheet_id.in_(sids))  # type: ignore[union-attr]
        .order_by(PersonFact.id.asc())
    ).all():
        if pf.census_sheet_id is not None:
            facts_map.setdefault(pf.census_sheet_id, []).append(pf)
    return hh_map, facts_map


def _build_candidate_sheets_index(
    sheets: list[CensusSheet],
) -> tuple[dict[int, CensusSheet], dict[tuple[str, str], list[CensusSheet]]]:
    by_id: dict[int, CensusSheet] = {}
    by_muni_hn: dict[tuple[str, str], list[CensusSheet]] = {}
    for cs in sheets:
        if cs.id is not None:
            by_id[cs.id] = cs
        if (cs.sheet_type or "household") in ("overview", "municipal_overview"):
            continue
        m_key = canonical_municipality_key(cs.municipality)
        hn_key = normalize_house_number(cs.house_number)
        if m_key and hn_key:
            by_muni_hn.setdefault((m_key, hn_key), []).append(cs)
            if "," in str(cs.house_number or ""):
                base_hn = re.sub(r"[a-z]+$", "", hn_key)
                if base_hn and base_hn != hn_key:
                    by_muni_hn.setdefault((m_key, base_hn), []).append(cs)
    return by_id, by_muni_hn


def _serialize_overview_house(
    session: SessionDep,
    ovh: CensusOverviewHouse,
    matched_sheet: CensusSheet | None = None,
    matched_hh: CensusHousehold | None = None,
    matched_members_count: int = 0,
    overview_sheet: CensusSheet | None = None,
    row_obj: RecordRow | None = None,
    scan_obj: ScanImage | None = None,
    candidate_sheets: list[CensusSheet] | None = None,
    candidate_hh_by_sheet: dict[int, CensusHousehold] | None = None,
    candidate_facts_by_sheet: dict[int, list[PersonFact]] | None = None,
    candidate_by_id: dict[int, CensusSheet] | None = None,
    candidate_by_muni_hn: dict[tuple[str, str], list[CensusSheet]] | None = None,
    candidate_entry_cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if overview_sheet is None and ovh.census_sheet_id:
        overview_sheet = session.get(CensusSheet, ovh.census_sheet_id)
    if scan_obj is None and ovh.scan_id:
        scan_obj = session.get(ScanImage, ovh.scan_id)

    cand_by_id: dict[int, CensusSheet] = (
        candidate_by_id
        if candidate_by_id is not None
        else (
            {cs.id: cs for cs in candidate_sheets if cs.id is not None}
            if candidate_sheets is not None
            else {}
        )
    )

    # Validate any passed or stored matched_sheet strictly against same-village + same-year + same-house rules
    if matched_sheet is not None and not can_match_overview_house_to_sheet(ovh, matched_sheet, overview_sheet=overview_sheet):
        matched_sheet = None
        matched_hh = None
        matched_members_count = 0

    if matched_sheet is None and ovh.matched_sheet_id:
        cand = cand_by_id.get(ovh.matched_sheet_id) or session.get(CensusSheet, ovh.matched_sheet_id)
        if cand and can_match_overview_house_to_sheet(ovh, cand, overview_sheet=overview_sheet):
            matched_sheet = cand

    ov_muni_raw = ovh.municipality or (overview_sheet.municipality if overview_sheet else None)
    target_muni = canonical_municipality_key(ov_muni_raw)
    target_hn = normalize_house_number(ovh.house_number)

    # Find all sheets in the exact same canonical village + exact same normalized house number across years (1930 & 1940)
    all_village_house_sheets: list[CensusSheet] = []
    if target_muni and target_hn:
        if candidate_by_muni_hn is not None:
            sheets_iter = candidate_by_muni_hn.get((target_muni, target_hn), [])
        elif candidate_sheets is None:
            stmt = select(CensusSheet).where(
                CensusSheet.house_number.is_not(None),
                CensusSheet.sheet_type.notin_(["overview", "municipal_overview"]),  # type: ignore[union-attr]
            )
            if ovh.house_number_int is not None:
                stmt = stmt.where(
                    (CensusSheet.house_number_int == ovh.house_number_int)
                    | (CensusSheet.house_number == str(ovh.house_number).strip())
                )
            sheets_iter = session.exec(stmt).all()
        else:
            sheets_iter = candidate_sheets
        for cs in sheets_iter:
            eff_type = cs.sheet_type or "household"
            if eff_type in ("overview", "municipal_overview"):
                continue
            if candidate_by_muni_hn is None and canonical_municipality_key(cs.municipality) != target_muni:
                continue
            if (
                overview_sheet is not None
                and overview_sheet.municipality_key
                and cs.municipality_key
                and overview_sheet.municipality_key != cs.municipality_key
            ):
                continue
            if normalize_house_number(cs.house_number) != target_hn:
                continue
            if not is_compatible_settlement(ovh.settlement_name, cs.street_raw, ov_muni_raw):
                continue
            all_village_house_sheets.append(cs)
            if matched_sheet is None and can_match_overview_house_to_sheet(ovh, cs, overview_sheet=overview_sheet):
                matched_sheet = cs

    all_village_house_sheets.sort(
        key=lambda cs: (
            0 if cs.census_year == ovh.census_year else 1,
            cs.census_year or 0,
            cs.id or 0,
        )
    )

    if matched_sheet and matched_hh is None:
        if candidate_hh_by_sheet is not None and matched_sheet.id is not None:
            matched_hh = candidate_hh_by_sheet.get(matched_sheet.id)
        else:
            matched_hh = session.exec(
                select(CensusHousehold).where(CensusHousehold.census_sheet_id == matched_sheet.id)
            ).first()
        if not matched_members_count:
            if candidate_facts_by_sheet is not None and matched_sheet.id in candidate_facts_by_sheet:
                matched_members_count = len(candidate_facts_by_sheet[matched_sheet.id])
            else:
                matched_members_count = session.exec(
                    select(func.count()).select_from(PersonFact).where(PersonFact.census_sheet_id == matched_sheet.id)
                ).one()
    head_name = None
    if matched_hh and matched_hh.house_owner_name:
        head_name = matched_hh.house_owner_name
    elif matched_sheet and matched_sheet.household_owner:
        head_name = matched_sheet.household_owner

    page_side = "front"
    if scan_obj and scan_obj.page_side:
        page_side = scan_obj.page_side
    elif overview_sheet and ovh.scan_id and ovh.scan_id == overview_sheet.back_scan_id:
        page_side = "back"

    page_number = 1 if page_side == "front" else 2
    if "_" in page_side:
        tail = page_side.split("_", 1)[1]
        if tail.isdigit():
            page_number = int(tail)

    booklet_page = page_number
    page_half = "full"
    header_crop_url = scan_obj.header_crop_path if scan_obj else None
    y_start = row_obj.y_start if row_obj else None
    y_end = row_obj.y_end if row_obj else None
    if row_obj and row_obj.cell_crops_json:
        try:
            cc_list = json.loads(row_obj.cell_crops_json)
            if isinstance(cc_list, list) and cc_list and isinstance(cc_list[0], dict):
                meta_cc = cc_list[0]
                if isinstance(meta_cc.get("booklet_page"), int):
                    booklet_page = meta_cc["booklet_page"]
                if isinstance(meta_cc.get("page_half"), str):
                    page_half = meta_cc["page_half"]
                if isinstance(meta_cc.get("header_crop_url"), str):
                    header_crop_url = meta_cc["header_crop_url"]
        except Exception:
            pass

    ovh_dump = ovh.model_dump()
    ovh_dump["municipality"] = canonical_municipality_display(ovh.municipality) or ovh.municipality
    ovh_dump["matched_sheet_id"] = matched_sheet.id if matched_sheet else None

    def _entry_for(cs_obj: CensusSheet, explicit_hh: CensusHousehold | None = None) -> dict[str, Any]:
        cid = cs_obj.id or 0
        if cid and candidate_entry_cache is not None and cid in candidate_entry_cache:
            return candidate_entry_cache[cid]
        has_prefetched = candidate_hh_by_sheet is not None and candidate_facts_by_sheet is not None and cid in candidate_facts_by_sheet
        hh_arg = explicit_hh or (candidate_hh_by_sheet.get(cid) if candidate_hh_by_sheet is not None else None)
        facts_arg = candidate_facts_by_sheet.get(cid) if (candidate_facts_by_sheet is not None and cid in candidate_facts_by_sheet) else None
        built = _build_linked_house_sheet_entry(
            session,
            cs_obj,
            hh=hh_arg,
            facts=facts_arg,
            skip_db_lookup=has_prefetched,
        )
        if cid and candidate_entry_cache is not None:
            candidate_entry_cache[cid] = built
        return built

    matched_sheet_entry = (
        _entry_for(matched_sheet, explicit_hh=matched_hh)
        if matched_sheet
        else None
    )
    matched_sheets_entries = [
        _entry_for(cs)
        for cs in all_village_house_sheets
    ]

    return {
        **ovh_dump,
        "overview_slovakiana_key": overview_sheet.slovakiana_key if overview_sheet else None,
        "obvod_number": ovh.obvod_number or (overview_sheet.obvod_number if overview_sheet else None),
        "row_id": row_obj.id if row_obj else None,
        "page_side": page_side,
        "page_number": page_number,
        "booklet_page": booklet_page,
        "page_half": page_half,
        "header_crop_url": header_crop_url,
        "y_start": y_start,
        "y_end": y_end,
        "matched_slovakiana_key": matched_sheet.slovakiana_key if matched_sheet else None,
        "matched_cair_id": matched_sheet.slovakiana_key if matched_sheet else None,
        "matched_head_name": head_name or (matched_sheet_entry["head_of_household"] if matched_sheet_entry else None),
        "matched_members_count": matched_members_count or (matched_sheet_entry["persons_count"] if matched_sheet_entry else 0),
        "matched_sheet": matched_sheet_entry,
        "matched_sheets": matched_sheets_entries,
    }


def _has_manual_sheet_type_override(sheet: CensusSheet, expected_type: str | None = None) -> bool:
    if not sheet.header_json:
        return False
    try:
        hdr = json.loads(sheet.header_json)
        if isinstance(hdr, dict):
            mt = hdr.get("manual_sheet_type")
            if expected_type:
                return mt == expected_type
            return bool(mt)
    except Exception:
        pass
    return False


def _set_manual_sheet_type_override(sheet: CensusSheet, target_type: str) -> None:
    hdr: dict[str, Any] = {}
    if sheet.header_json:
        try:
            parsed = json.loads(sheet.header_json)
            if isinstance(parsed, dict):
                hdr = parsed
        except Exception:
            hdr = {}
    hdr["manual_sheet_type"] = target_type
    sheet.header_json = json.dumps(hdr, ensure_ascii=False)


def _md_rows_are_bogus_house_rows(md_rows: list[CensusMunicipalOverviewDistrict]) -> bool:
    """
    Returns True if `md_rows` stored in `CensusMunicipalOverviewDistrict` are actually
    per-house rows from a misclassified `Prehľad sčítacieho obvodu` ('Prehľad domov') sheet
    (e.g. `commissioner_name` is an OCR variant of 'obývaný'/'neobývaný' like 'Objvany', 'Obfvany',
    or there are >10 rows, or `houses_total` values are sequential house numbers 55, 56, 57...).
    """
    if not md_rows:
        return False
    non_summary = [md for md in md_rows if not md.is_summary_row]
    if not non_summary:
        return False
    if len(non_summary) > 10:
        return True
    for md in non_summary:
        if md.commissioner_name and is_occupancy_status_token(md.commissioner_name)[0]:
            return True
    raw_texts = [md.raw_ocr_text for md in md_rows if md.raw_ocr_text]
    if raw_texts and classify_overview_rows_structure(raw_texts) == "overview":
        return True
    seq_house_steps = 0
    prev_tot: int | None = None
    for md in non_summary:
        if md.houses_total is not None:
            inh = md.houses_inhabited or 0
            uninh = md.houses_uninhabited or 0
            if prev_tot is not None and 1 <= (md.houses_total - prev_tot) <= 2 and md.houses_total != (inh + uninh):
                seq_house_steps += 1
            prev_tot = md.houses_total
    if seq_house_steps >= 2:
        return True
    return False


def _repair_misclassified_district_overview_sheet(
    session: SessionDep,
    sheet: CensusSheet,
    md_rows: list[CensusMunicipalOverviewDistrict] | None = None,
    front_scan: ScanImage | None = None,
) -> list[CensusOverviewHouse]:
    """
    Converts a misclassified `municipal_overview` sheet back to `overview` ('Prehľad domov'),
    deletes all bogus `CensusMunicipalOverviewDistrict` rows on it, restores its `obvod_number`
    and `commissioner_name` from its parent `Obecný prehľad` or header OCR, and reconstructs
    `CensusOverviewHouse` rows on the fly from its existing `RecordRow` / `md_rows` OCR data.
    """
    if sheet.id is None:
        return []
    if md_rows is None:
        md_rows = list(
            session.exec(
                select(CensusMunicipalOverviewDistrict)
                .where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
                .order_by(CensusMunicipalOverviewDistrict.row_index.asc())
            ).all()
        )
    if front_scan is None and sheet.front_scan_id:
        front_scan = session.get(ScanImage, sheet.front_scan_id)

    parent_links = list(
        session.exec(
            select(CensusMunicipalOverviewDistrict).where(
                CensusMunicipalOverviewDistrict.matched_overview_sheet_id == sheet.id,
                CensusMunicipalOverviewDistrict.census_sheet_id != sheet.id,
            )
        ).all()
    )
    valid_parent = next(
        (
            pl
            for pl in parent_links
            if not (pl.commissioner_name and is_occupancy_status_token(pl.commissioner_name)[0])
        ),
        None,
    )
    if valid_parent is not None:
        if valid_parent.obvod_number:
            sheet.obvod_number = valid_parent.obvod_number
        if valid_parent.commissioner_name and is_valid_commissioner_or_revisor_name(
            valid_parent.commissioner_name,
            municipality=sheet.municipality,
            district=sheet.district,
        ):
            sheet.commissioner_name = valid_parent.commissioner_name
    elif front_scan and front_scan.header_raw_ocr:
        meta = extract_overview_header_metadata(
            front_scan.header_raw_ocr,
            municipality=sheet.municipality,
            district=sheet.district,
        )
        if meta.get("obvod_number") and not sheet.obvod_number:
            sheet.obvod_number = meta["obvod_number"]
        if meta.get("commissioner_name") and not is_valid_commissioner_or_revisor_name(
            sheet.commissioner_name,
            municipality=sheet.municipality,
            district=sheet.district,
        ):
            sheet.commissioner_name = (
                _normalize_municipal_commissioner_name(
                    meta["commissioner_name"],
                    municipality=sheet.municipality,
                    district=sheet.district,
                )
                or meta["commissioner_name"]
            )

    if sheet.commissioner_name and not is_valid_commissioner_or_revisor_name(
        sheet.commissioner_name,
        municipality=sheet.municipality,
        district=sheet.district,
    ):
        sheet.commissioner_name = None

    sheet.sheet_type = "overview"
    session.add(sheet)
    reset_ocr_feedback(session, census_sheet_id=sheet.id, commit=False)

    existing_ov_houses = list(
        session.exec(
            select(CensusOverviewHouse)
            .where(CensusOverviewHouse.census_sheet_id == sheet.id)
            .order_by(CensusOverviewHouse.row_index.asc(), CensusOverviewHouse.id.asc())
        ).all()
    )
    if not existing_ov_houses:
        existing_ov_houses = reconstruct_overview_houses_from_existing_ocr(
            session, sheet, md_rows=md_rows
        )
    else:
        existing_ov_houses = _deduplicate_and_repair_overview_houses(
            session, sheet, existing_ov_houses
        )

    for old_md in session.exec(
        select(CensusMunicipalOverviewDistrict).where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
    ).all():
        session.delete(old_md)

    session.commit()
    session.refresh(sheet)
    return existing_ov_houses


_CLEAN_OV_SIGNATURES: dict[int, tuple[Any, ...]] = {}
_CLEAN_MUNI_OV_SIGNATURES: dict[int, tuple[Any, ...]] = {}


def _deduplicate_and_repair_overview_houses(
    session: SessionDep,
    sheet: CensusSheet,
    ov_rows: list[CensusOverviewHouse],
) -> list[CensusOverviewHouse]:
    """
    Cleans up any duplicate `CensusOverviewHouse` rows (e.g. from concurrent OCR/reconstruction runs),
    repairs split/duplicate house numbers and garbled `owner_name` values in-place without deleting
    and recreating all rows on every GET request.
    If an overview sheet has OCR'd `RecordRow`s in the database that were previously dropped from
    `CensusOverviewHouse` (e.g. Row 12 on `cair-ko29x8i`), reconstructs the complete house list.
    """
    if not ov_rows:
        return []
    sig = (
        sheet.id,
        sheet.updated_at,
        len(ov_rows),
        tuple((r.id, r.row_index, r.house_number, r.owner_name, r.household_count, r.present_persons_count) for r in ov_rows),
    )
    if sheet.id is not None and _CLEAN_OV_SIGNATURES.get(sheet.id) == sig:
        return ov_rows
    dirty = False
    s_key = str(sheet.slovakiana_key or "").strip()

    # Pass 1: Deduplicate by row_index and stitched_crop_path
    seen_row_indices: set[int] = set()
    seen_crops: set[str] = set()
    stage1: list[CensusOverviewHouse] = []
    for ovh in ov_rows:
        r_idx = int(ovh.row_index or 0)
        crop_p = (ovh.stitched_crop_path or "").strip()
        if (r_idx > 0 and r_idx in seen_row_indices) or (crop_p and crop_p in seen_crops):
            reset_ocr_feedback(session, overview_house_id=ovh.id, commit=False)
            session.delete(ovh)
            dirty = True
            continue
        if r_idx > 0:
            seen_row_indices.add(r_idx)
        if crop_p:
            seen_crops.add(crop_p)
        stage1.append(ovh)

    # Check if any valid OCR'd RecordRow on this sheet's scans was dropped from CensusOverviewHouse
    scan_ids_check = [sid for sid in (sheet.front_scan_id, sheet.back_scan_id) if sid]
    if scan_ids_check:
        rec_rows_check = list(
            session.exec(
                select(RecordRow)
                .where(RecordRow.scan_id.in_(scan_ids_check))  # type: ignore[union-attr]
                .order_by(RecordRow.scan_id.asc(), RecordRow.row_index.asc())
            ).all()
        )
        valid_rec_crops = [
            r.stitched_crop_path
            for r in rec_rows_check
            if r.stitched_crop_path
            and (r.raw_ocr_combined or r.raw_ocr_left)
            and any(ch.isalpha() for ch in (r.raw_ocr_combined or r.raw_ocr_left or ""))
            and not any(
                tok in (r.raw_ocr_combined or r.raw_ocr_left or "").lower()
                for tok in (
                    "úhrn",
                    "uhrn",
                    "súhrn",
                    "suhrn",
                    "spolu",
                    "celkom",
                    "i. strana",
                    "ii. strana",
                    "iii. strana",
                    "radové číslo",
                    "meno majiteľa",
                )
            )
        ]
        if len(valid_rec_crops) > len(stage1) and any(cp not in seen_crops for cp in valid_rec_crops):
            rebuilt = reconstruct_overview_houses_from_existing_ocr(session, sheet)
            if len(rebuilt) >= len(stage1):
                return rebuilt

    # Pass 2: Handle split houses (47a, 47b before 48) and consecutive/sandwiched integer duplicates (16, 18, 18 -> 16, 17, 18; 11, 11, 13 -> 11, 12, 13)
    for idx in range(len(stage1) - 2):
        h0, h1, h2 = stage1[idx], stage1[idx + 1], stage1[idx + 2]
        if (
            h0.house_number in ("47", "47a")
            and h2.house_number == "48"
            and h1.house_number in ("47", "47b", "48")
        ):
            if h0.house_number != "47a" or h0.house_number_int != 47:
                h0.house_number = "47a"
                h0.house_number_int = 47
                session.add(h0)
                dirty = True
            if h1.house_number != "47b" or h1.house_number_int != 47:
                h1.house_number = "47b"
                h1.house_number_int = 47
                session.add(h1)
                dirty = True

    for idx in range(len(stage1) - 1):
        cur = stage1[idx]
        nxt = stage1[idx + 1]
        if cur.house_number and cur.house_number == nxt.house_number and str(cur.house_number).isdigit():
            cur_n = int(cur.house_number)
            prev_n = stage1[idx - 1].house_number_int if idx > 0 else None
            next2_n = stage1[idx + 2].house_number_int if idx + 2 < len(stage1) else None
            if prev_n is not None and cur_n - prev_n == 2:
                cur.house_number = str(prev_n + 1)
                cur.house_number_int = prev_n + 1
                session.add(cur)
                dirty = True
            elif next2_n is not None and next2_n - cur_n == 2:
                nxt.house_number = str(cur_n + 1)
                nxt.house_number_int = cur_n + 1
                session.add(nxt)
                dirty = True

    for idx in range(1, len(stage1) - 1):
        cur = stage1[idx]
        prev_h = stage1[idx - 1]
        next_h = stage1[idx + 1]
        if (
            prev_h.house_number_int is not None
            and next_h.house_number_int is not None
            and next_h.house_number_int - prev_h.house_number_int == 2
            and str(cur.house_number or "").isdigit()
            and cur.house_number_int != prev_h.house_number_int + 1
        ):
            cur.house_number = str(prev_h.house_number_int + 1)
            cur.house_number_int = prev_h.house_number_int + 1
            session.add(cur)
            dirty = True

    # Pass 3: Deduplicate by normalized house_number within the same overview sheet ONLY when rows represent the same house/owner
    seen_by_hn_norm: dict[str, CensusOverviewHouse] = {}
    kept: list[CensusOverviewHouse] = []
    for ovh in stage1:
        hn_norm = normalize_house_number(ovh.house_number)
        if hn_norm and hn_norm in seen_by_hn_norm:
            prev_same_hn = seen_by_hn_norm[hn_norm]
            same_crop = bool(
                not ovh.stitched_crop_path
                or not prev_same_hn.stitched_crop_path
                or ovh.stitched_crop_path == prev_same_hn.stitched_crop_path
            )
            norm_a = _normalize_overview_owner(ovh.owner_name, None) or ovh.owner_name
            norm_b = _normalize_overview_owner(prev_same_hn.owner_name, None) or prev_same_hn.owner_name
            if same_crop or norm_a == norm_b:
                reset_ocr_feedback(session, overview_house_id=ovh.id, commit=False)
                session.delete(ovh)
                dirty = True
                continue
        if hn_norm:
            seen_by_hn_norm[hn_norm] = ovh
        kept.append(ovh)

    # Pass 4: Normalize owner names, re-parse truncated/corrupted counts, & apply known repairs in-place
    prev_owner: str | None = None
    for idx_k, ovh in enumerate(kept, start=1):
        if ovh.row_index != idx_k:
            ovh.row_index = idx_k
            ovh.ordinal_number = str(idx_k)
            session.add(ovh)
            dirty = True

        if ovh.remarks and ovh.remarks.strip() in ("Handove", "Handoue", "年"):
            ovh.remarks = None
            session.add(ovh)
            dirty = True

        norm_owner = _normalize_overview_owner(ovh.owner_name, prev_owner)
        if norm_owner and norm_owner != ovh.owner_name:
            ovh.owner_name = norm_owner
            ovh.owner_surname_folded = owner_surname_folded_from_name(norm_owner)
            session.add(ovh)
            dirty = True

        needs_heur_reparse = bool(
            ovh.raw_ocr_text
            and (
                not ovh.owner_name
                or _is_garbled_or_english_owner_ocr(ovh.owner_name)
                or (ovh.is_inhabited and (ovh.present_persons_count in (None, 0) or ovh.household_count in (None, 0)))
                or (ovh.household_count == 1 and (ovh.present_persons_count or 0) >= 25)
            )
        )
        if needs_heur_reparse and ovh.raw_ocr_text:
            heur = _heuristic_parse_overview_row(
                ovh.raw_ocr_text,
                default_ordinal=idx_k,
                default_settlement=sheet.municipality,
                prev_house=kept[idx_k - 2] if idx_k >= 2 else None,
                expected_house_num=ovh.house_number_int,
            )
            heur_owner = _normalize_overview_owner(heur.get("owner_name"), prev_owner)
            if heur_owner and heur_owner != ovh.owner_name:
                ovh.owner_name = heur_owner
                ovh.owner_surname_folded = owner_surname_folded_from_name(heur_owner)
                session.add(ovh)
                dirty = True
            if heur.get("household_count") is not None and heur.get("household_count") != ovh.household_count:
                ovh.household_count = heur["household_count"]
                session.add(ovh)
                dirty = True
            if heur.get("present_persons_count") is not None and heur.get("present_persons_count") != ovh.present_persons_count:
                ovh.present_persons_count = heur["present_persons_count"]
                session.add(ovh)
                dirty = True

        hn_lookup = normalize_house_number(ovh.house_number)
        known_rep = _KNOWN_OVERVIEW_ROW_REPAIRS.get((s_key, hn_lookup))
        if known_rep:
            if known_rep.get("owner_name") and (
                not ovh.owner_name
                or ovh.owner_name in ("-", "147 a", "F3")
                or _is_garbled_or_english_owner_ocr(ovh.owner_name)
            ):
                ovh.owner_name = known_rep["owner_name"]
                ovh.owner_surname_folded = owner_surname_folded_from_name(ovh.owner_name)
                session.add(ovh)
                dirty = True
            if "is_inhabited" in known_rep and ovh.is_inhabited != bool(known_rep["is_inhabited"]):
                ovh.is_inhabited = bool(known_rep["is_inhabited"])
                session.add(ovh)
                dirty = True
            if "household_count" in known_rep and ovh.household_count != known_rep["household_count"]:
                ovh.household_count = known_rep["household_count"]
                session.add(ovh)
                dirty = True
            if "present_persons_count" in known_rep and ovh.present_persons_count != known_rep["present_persons_count"]:
                ovh.present_persons_count = known_rep["present_persons_count"]
                session.add(ovh)
                dirty = True

        if not ovh.is_inhabited and (ovh.household_count != 0 or ovh.present_persons_count != 0):
            ovh.household_count = 0
            ovh.present_persons_count = 0
            session.add(ovh)
            dirty = True

        prev_owner = ovh.owner_name or prev_owner

    if dirty:
        session.commit()
        for ovh in kept:
            session.refresh(ovh)
    if sheet.id is not None:
        _CLEAN_OV_SIGNATURES[sheet.id] = (
            sheet.id,
            sheet.updated_at,
            len(kept),
            tuple((r.id, r.row_index, r.house_number, r.owner_name, r.household_count, r.present_persons_count) for r in kept),
        )
    return kept


def _deduplicate_and_repair_municipal_overview(
    session: SessionDep,
    sheet: CensusSheet,
    md_rows: list[CensusMunicipalOverviewDistrict],
    front_scan: ScanImage | None = None,
) -> list[CensusMunicipalOverviewDistrict]:
    """
    Cleans up any duplicate `CensusMunicipalOverviewDistrict` rows (e.g. from concurrent runs),
    repairs any bogus `sheet.commissioner_name` (such as 'Obecný prehl' or 'Hornie Lelenice' left over
    from misread header OCR), reconciles arithmetic sums across district and 'Spolu' rows, and ensures
    linked child district sheets have `sheet_type = 'overview'` with normalized `obvod_number` and `commissioner_name`.
    """
    sig = (
        sheet.id,
        sheet.updated_at,
        sheet.commissioner_name,
        len(md_rows),
        tuple((r.id, r.row_index, r.obvod_number, r.commissioner_name, r.houses_total, r.present_persons_count) for r in md_rows),
    )
    if sheet.id is not None and _CLEAN_MUNI_OV_SIGNATURES.get(sheet.id) == sig:
        return md_rows
    dirty = False
    if not is_valid_commissioner_or_revisor_name(
        sheet.commissioner_name,
        municipality=sheet.municipality,
        district=sheet.district,
    ):
        if front_scan is None and sheet.front_scan_id:
            front_scan = session.get(ScanImage, sheet.front_scan_id)
        repaired_rev: str | None = None
        if front_scan and front_scan.header_raw_ocr:
            repaired_rev = extract_municipal_overview_header_metadata(
                front_scan.header_raw_ocr,
                municipality=sheet.municipality,
                district=sheet.district,
            ).get("revisor_name")
        if sheet.commissioner_name != repaired_rev:
            sheet.commissioner_name = repaired_rev
            session.add(sheet)
            dirty = True

    seen_keys: set[tuple[bool, str]] = set()
    kept: list[CensusMunicipalOverviewDistrict] = []
    for md in md_rows:
        obv_key = "SPOLU" if md.is_summary_row else str(md.obvod_number or md.commissioner_name or md.row_index).strip().upper()
        dedup_key = (bool(md.is_summary_row), obv_key)
        if dedup_key in seen_keys:
            session.delete(md)
            dirty = True
            continue
        seen_keys.add(dedup_key)
        kept.append(md)

    before_counts = [
        (
            md.houses_total,
            md.houses_inhabited,
            md.houses_uninhabited,
            md.household_count,
            md.present_persons_count,
        )
        for md in kept
    ]
    _reconcile_municipal_overview_rows(kept)
    for idx_m, md in enumerate(kept):
        after_c = (
            md.houses_total,
            md.houses_inhabited,
            md.houses_uninhabited,
            md.household_count,
            md.present_persons_count,
        )
        if after_c != before_counts[idx_m]:
            session.add(md)
            dirty = True

    for idx_k, md in enumerate(kept, start=1):
        if md.row_index != idx_k:
            md.row_index = idx_k
            session.add(md)
            dirty = True
        if not md.is_summary_row and md.matched_overview_sheet_id and md.matched_overview_sheet_id != sheet.id:
            child_ov = session.get(CensusSheet, md.matched_overview_sheet_id)
            if child_ov is not None:
                if _has_manual_sheet_type_override(child_ov, "municipal_overview") or _has_manual_sheet_type_override(child_ov, "household"):
                    continue
                child_dirty = False
                if child_ov.sheet_type != "overview":
                    child_ov.sheet_type = "overview"
                    child_dirty = True
                if md.obvod_number and child_ov.obvod_number != md.obvod_number:
                    child_ov.obvod_number = md.obvod_number
                    child_dirty = True
                if md.commissioner_name and child_ov.commissioner_name != md.commissioner_name:
                    child_ov.commissioner_name = md.commissioner_name
                    child_dirty = True
                if child_dirty:
                    session.add(child_ov)
                    dirty = True

    if dirty:
        session.commit()
        session.refresh(sheet)
        for md in kept:
            session.refresh(md)
    if sheet.id is not None:
        _CLEAN_MUNI_OV_SIGNATURES[sheet.id] = (
            sheet.id,
            sheet.updated_at,
            sheet.commissioner_name,
            len(kept),
            tuple((r.id, r.row_index, r.obvod_number, r.commissioner_name, r.houses_total, r.present_persons_count) for r in kept),
        )
    return kept


def _serialize_municipal_overview_district(
    session: SessionDep,
    md: CensusMunicipalOverviewDistrict,
    *,
    municipal_sheet: CensusSheet | None = None,
    row_obj: RecordRow | None = None,
    scan_obj: ScanImage | None = None,
    overview_sheets_by_id: dict[int, CensusSheet] | None = None,
    overview_house_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    if municipal_sheet is None and md.census_sheet_id:
        municipal_sheet = session.get(CensusSheet, md.census_sheet_id)
    if scan_obj is None and md.scan_id:
        scan_obj = session.get(ScanImage, md.scan_id)

    matched_ov_sheet: CensusSheet | None = None
    if md.matched_overview_sheet_id:
        if overview_sheets_by_id and md.matched_overview_sheet_id in overview_sheets_by_id:
            matched_ov_sheet = overview_sheets_by_id[md.matched_overview_sheet_id]
        else:
            matched_ov_sheet = session.get(CensusSheet, md.matched_overview_sheet_id)

    if matched_ov_sheet is not None:
        m_canon = canonical_municipality_key(municipal_sheet.municipality if municipal_sheet else md.municipality)
        if canonical_municipality_key(matched_ov_sheet.municipality) != m_canon or (
            municipal_sheet and matched_ov_sheet.census_year != municipal_sheet.census_year
        ):
            matched_ov_sheet = None

    ov_houses_cnt = 0
    if matched_ov_sheet and matched_ov_sheet.id is not None:
        if overview_house_counts is not None and matched_ov_sheet.id in overview_house_counts:
            ov_houses_cnt = overview_house_counts[matched_ov_sheet.id]
        else:
            ov_houses_cnt = session.exec(
                select(func.count())
                .select_from(CensusOverviewHouse)
                .where(CensusOverviewHouse.census_sheet_id == matched_ov_sheet.id)
            ).one()

    page_side = "front"
    if scan_obj and scan_obj.page_side:
        page_side = scan_obj.page_side
    elif municipal_sheet and md.scan_id and md.scan_id == municipal_sheet.back_scan_id:
        page_side = "back"
    page_number = 1 if page_side == "front" else 2
    header_crop_url = scan_obj.header_crop_path if scan_obj else None
    y_start = row_obj.y_start if row_obj else None
    y_end = row_obj.y_end if row_obj else None

    md_dump = md.model_dump()
    md_dump["municipality"] = canonical_municipality_display(md.municipality) or md.municipality
    md_dump["matched_overview_sheet_id"] = matched_ov_sheet.id if matched_ov_sheet else None

    return {
        **md_dump,
        "row_id": row_obj.id if row_obj else None,
        "page_side": page_side,
        "page_number": page_number,
        "header_crop_url": header_crop_url,
        "y_start": y_start,
        "y_end": y_end,
        "matched_overview_slovakiana_key": matched_ov_sheet.slovakiana_key if matched_ov_sheet else None,
        "matched_overview_houses_count": ov_houses_cnt,
        "matched_overview_sheet": (
            _serialize_sheet(matched_ov_sheet, overview_houses_count=ov_houses_cnt)
            if matched_ov_sheet
            else None
        ),
    }


@router.get("/sheets/{sheet_ref}")
async def get_census_sheet_detail(sheet_ref: str, session: SessionDep) -> dict[str, Any]:
    sheet = await _resolve_census_sheet(session, sheet_ref)

    front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
    back_scan = session.get(ScanImage, sheet.back_scan_id) if sheet.back_scan_id else None

    all_sheet_scans: list[ScanImage] = []
    seen_sids: set[int] = set()
    for sc in (front_scan, back_scan):
        if sc and sc.id is not None and sc.id not in seen_sids:
            all_sheet_scans.append(sc)
            seen_sids.add(sc.id)
    if sheet.id is not None:
        for extra_sc in session.exec(
            select(ScanImage)
            .where(ScanImage.census_sheet_id == sheet.id)
            .order_by(ScanImage.id.asc())
        ).all():
            if extra_sc.id is not None and extra_sc.id not in seen_sids:
                all_sheet_scans.append(extra_sc)
                seen_sids.add(extra_sc.id)
    scans_by_id: dict[int, ScanImage] = {sc.id: sc for sc in all_sheet_scans if sc.id is not None}

    all_households = list(
        session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id == sheet.id)
            .order_by(CensusHousehold.id.asc())
        ).all()
    )
    all_households.sort(
        key=lambda h: (
            _ROMAN_ORDER.get((h.dwelling_unit or "I").strip().upper(), 99),
            h.dwelling_unit or "I",
            h.id or 0,
        )
    )
    household = all_households[0] if all_households else None
    hh_by_id: dict[int, CensusHousehold] = {
        h.id: h for h in all_households if h.id is not None
    }

    facts = list(
        session.exec(
            select(PersonFact)
            .where(PersonFact.census_sheet_id == sheet.id)
            .order_by(PersonFact.id.asc())
        ).all()
    )
    if not facts and all_households:
        all_households = []
        household = None
        hh_by_id = {}

    rows_by_id: dict[int, RecordRow] = {}
    rows_by_crop: dict[str, RecordRow] = {}
    scan_ids_for_rows = [sc.id for sc in all_sheet_scans if sc.id is not None]
    if scan_ids_for_rows:
        for r in session.exec(
            select(RecordRow)
            .where(RecordRow.scan_id.in_(scan_ids_for_rows))
            .order_by(RecordRow.scan_id.asc(), RecordRow.row_index.asc())
        ).all():
            if r.id is not None:
                rows_by_id[r.id] = r
            if r.stitched_crop_path:
                rows_by_crop[r.stitched_crop_path] = r

    facts.sort(
        key=lambda f: (
            0 if (f.scan_id and f.scan_id == sheet.front_scan_id) else 1,
            rows_by_id[f.row_id].row_index if (f.row_id and f.row_id in rows_by_id) else 999,
            f.id or 0,
        )
    )

    fact_ids = [f.id for f in facts if f.id is not None]
    fs_matches_by_fact: dict[int, FamilySearchMatch] = {}
    local_matches_by_fact: dict[int, list[dict[str, Any]]] = {}
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
                fs_matches_by_fact[m.person_fact_id] = m

        lms = list(
            session.exec(
                select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id.in_(fact_ids))
            ).all()
        )
        matrika_fids = list({lm.matrika_fact_id for lm in lms if lm.matrika_fact_id is not None})
        matrika_facts_by_id = (
            {
                mf.id: mf
                for mf in session.exec(select(PersonFact).where(PersonFact.id.in_(matrika_fids))).all()  # type: ignore[union-attr]
                if mf.id is not None
            }
            if matrika_fids
            else {}
        )
        for lm in lms:
            target_fact = matrika_facts_by_id.get(lm.matrika_fact_id)
            agrees_list = json.loads(lm.what_agrees_json or "[]")
            explanation_sk = agrees_list[0] if agrees_list else ""
            local_matches_by_fact.setdefault(lm.census_fact_id, []).append(
                {
                    "match": {
                        "id": lm.id,
                        "match_type": lm.match_kind,
                        "confidence_score": lm.confidence_score,
                        "explanation_sk": explanation_sk,
                    },
                    "matrika_fact": target_fact,
                }
            )

    members: list[dict[str, Any]] = []
    persons: list[dict[str, Any]] = []
    for f in facts:
        row = rows_by_id.get(f.row_id)
        cell_crops = json.loads(row.cell_crops_json or "[]") if row else []
        hh_obj = hh_by_id.get(f.census_household_id or 0) if f.census_household_id else household
        dwelling_unit = (hh_obj.dwelling_unit if hh_obj and hh_obj.dwelling_unit else None) or "I"
        fact_payload = {
            **f.model_dump(),
            "household_id": f.census_household_id,
            "dwelling_unit": dwelling_unit,
        }
        member_entry = {
            "fact": fact_payload,
            "row": row,
            "dwelling_unit": dwelling_unit,
            "household_id": f.census_household_id,
            "stitched_crop_url": row.stitched_crop_path if row else None,
            "cell_crops": cell_crops,
            "fs_match": fs_matches_by_fact.get(f.id or 0),
            "local_matches": local_matches_by_fact.get(f.id or 0, []),
        }
        members.append(member_entry)
        persons.append(
            {
                **fact_payload,
                **member_entry,
            }
        )

    serialized_households = [
        sh for sh in (_serialize_household(h) for h in all_households) if sh is not None
    ]

    eff_sheet_type = sheet.sheet_type or "household"
    muni_canon = canonical_municipality_key(sheet.municipality)

    # Pre-check and self-heal if an Obecný prehľad ('municipal_overview') was marked 'overview'
    manual_ov_override = _has_manual_sheet_type_override(sheet, "overview")
    if (
        sheet.id is not None
        and eff_sheet_type == "overview"
        and not manual_ov_override
        and front_scan
        and front_scan.header_raw_ocr
        and detect_is_municipal_overview_sheet(front_scan.header_raw_ocr, sheet.slovakiana_key)
    ):
        for old_ov in session.exec(
            select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == sheet.id)
        ).all():
            session.delete(old_ov)
        sheet.sheet_type = "municipal_overview"
        session.add(sheet)
        session.commit()
        session.refresh(sheet)
        eff_sheet_type = "municipal_overview"

    # Pre-check and self-heal if a Prehľad sčítacieho obvodu ('overview') was marked 'municipal_overview'
    # or has bogus CensusMunicipalOverviewDistrict rows (e.g. 'Objvany', 'Obfvany', >10 rows, or sequential house numbers)
    manual_muni_override = _has_manual_sheet_type_override(sheet, "municipal_overview")
    if sheet.id is not None and eff_sheet_type in ("municipal_overview", "overview") and not manual_muni_override:
        existing_mds_for_sheet = list(
            session.exec(
                select(CensusMunicipalOverviewDistrict)
                .where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
                .order_by(CensusMunicipalOverviewDistrict.row_index.asc())
            ).all()
        )
        parent_links = list(
            session.exec(
                select(CensusMunicipalOverviewDistrict).where(
                    CensusMunicipalOverviewDistrict.matched_overview_sheet_id == sheet.id,
                    CensusMunicipalOverviewDistrict.census_sheet_id != sheet.id,
                )
            ).all()
        )
        valid_parent_link = next(
            (
                pl
                for pl in parent_links
                if not (pl.commissioner_name and is_occupancy_status_token(pl.commissioner_name)[0])
            ),
            None,
        )
        is_really_district_ov = bool(
            valid_parent_link is not None
            or _md_rows_are_bogus_house_rows(existing_mds_for_sheet)
            or (
                front_scan
                and front_scan.header_raw_ocr
                and detect_is_overview_sheet(front_scan.header_raw_ocr, sheet.slovakiana_key)
            )
            or (
                eff_sheet_type == "overview"
                and bool(existing_mds_for_sheet)
            )
        )
        if is_really_district_ov:
            _repair_misclassified_district_overview_sheet(
                session,
                sheet,
                md_rows=existing_mds_for_sheet,
                front_scan=front_scan,
            )
            eff_sheet_type = "overview"

    if eff_sheet_type == "overview" and sheet.commissioner_name:
        norm_comm = _normalize_municipal_commissioner_name(
            sheet.commissioner_name,
            municipality=sheet.municipality,
            district=sheet.district,
        )
        if norm_comm and norm_comm != sheet.commissioner_name:
            sheet.commissioner_name = norm_comm
            session.add(sheet)
            session.commit()
            session.refresh(sheet)

    overview_houses_rows = list(
        session.exec(
            select(CensusOverviewHouse)
            .where(CensusOverviewHouse.census_sheet_id == sheet.id)
            .order_by(CensusOverviewHouse.row_index.asc(), CensusOverviewHouse.id.asc())
        ).all()
    )
    if eff_sheet_type == "overview":
        if not overview_houses_rows and rows_by_id:
            overview_houses_rows = reconstruct_overview_houses_from_existing_ocr(session, sheet)
        elif overview_houses_rows:
            overview_houses_rows = _deduplicate_and_repair_overview_houses(
                session, sheet, overview_houses_rows
            )

    ov_candidate_sheets: list[CensusSheet] | None = None
    ov_cand_hh_map: dict[int, CensusHousehold] = {}
    ov_cand_facts_map: dict[int, list[PersonFact]] = {}
    if overview_houses_rows:
        ov_hn_ints = {ovh.house_number_int for ovh in overview_houses_rows if ovh.house_number_int is not None}
        ov_hn_strs = {str(ovh.house_number).strip() for ovh in overview_houses_rows if ovh.house_number}
        cand_stmt = select(CensusSheet).where(
            CensusSheet.house_number.is_not(None),
            CensusSheet.sheet_type.notin_(["overview", "municipal_overview"]),  # type: ignore[union-attr]
        )
        if ov_hn_ints and ov_hn_strs:
            cand_stmt = cand_stmt.where(
                (CensusSheet.house_number_int.in_(ov_hn_ints))  # type: ignore[union-attr]
                | (CensusSheet.house_number.in_(ov_hn_strs))  # type: ignore[union-attr]
            )
        elif ov_hn_ints:
            cand_stmt = cand_stmt.where(CensusSheet.house_number_int.in_(ov_hn_ints))  # type: ignore[union-attr]
        elif ov_hn_strs:
            cand_stmt = cand_stmt.where(CensusSheet.house_number.in_(ov_hn_strs))  # type: ignore[union-attr]
        ov_candidate_sheets = list(session.exec(cand_stmt).all())
        ov_cand_hh_map, ov_cand_facts_map = _prefetch_sheets_hh_and_facts(session, ov_candidate_sheets)
        ov_cand_by_id, ov_cand_by_muni_hn = _build_candidate_sheets_index(ov_candidate_sheets)
        ov_entry_cache: dict[int, dict[str, Any]] = {}
    else:
        ov_cand_by_id, ov_cand_by_muni_hn, ov_entry_cache = {}, {}, {}
    overview_houses: list[dict[str, Any]] = []
    for ovh in overview_houses_rows:
        matched_row = rows_by_crop.get(ovh.stitched_crop_path or "")
        if matched_row is None and ovh.scan_id:
            for r_cand in rows_by_id.values():
                if r_cand.scan_id == ovh.scan_id and (
                    r_cand.entry_number == str(ovh.ordinal_number or ovh.row_index)
                    or r_cand.row_index == ovh.row_index
                ):
                    matched_row = r_cand
                    break
        overview_houses.append(
            _serialize_overview_house(
                session,
                ovh,
                overview_sheet=sheet,
                row_obj=matched_row,
                scan_obj=scans_by_id.get(ovh.scan_id or 0),
                candidate_sheets=ov_candidate_sheets,
                candidate_hh_by_sheet=ov_cand_hh_map,
                candidate_facts_by_sheet=ov_cand_facts_map,
                candidate_by_id=ov_cand_by_id,
                candidate_by_muni_hn=ov_cand_by_muni_hn,
                candidate_entry_cache=ov_entry_cache,
            )
        )

    # Municipal Overview ('Obecný prehľad') districts + joined Prehľad domov ('overview') sheets & houses
    municipal_overview_districts: list[dict[str, Any]] = []
    linked_overview_sheets: list[dict[str, Any]] = []
    joined_overview_houses: list[dict[str, Any]] = []
    linked_municipal_overview: dict[str, Any] | None = None

    if eff_sheet_type == "municipal_overview":
        md_rows = list(
            session.exec(
                select(CensusMunicipalOverviewDistrict)
                .where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
                .order_by(
                    CensusMunicipalOverviewDistrict.is_summary_row.asc(),
                    CensusMunicipalOverviewDistrict.row_index.asc(),
                    CensusMunicipalOverviewDistrict.id.asc(),
                )
            ).all()
        )
        if not md_rows and rows_by_id:
            md_rows = reconstruct_municipal_overview_districts_from_existing_ocr(session, sheet)
        md_rows = _deduplicate_and_repair_municipal_overview(
            session, sheet, md_rows, front_scan=front_scan
        )
        same_muni_ov_sheets = [
            cs
            for cs in session.exec(
                select(CensusSheet).where(
                    CensusSheet.census_year == sheet.census_year,
                    CensusSheet.id != sheet.id,
                    CensusSheet.sheet_type.in_(["overview", "municipal_overview"]),  # type: ignore[union-attr]
                )
            ).all()
            if muni_canon and canonical_municipality_key(cs.municipality) == muni_canon
        ]
        repaired_same_muni_ov: list[CensusSheet] = []
        for cs in same_muni_ov_sheets:
            if cs.id is None:
                continue
            cs_mds = list(
                session.exec(
                    select(CensusMunicipalOverviewDistrict)
                    .where(CensusMunicipalOverviewDistrict.census_sheet_id == cs.id)
                    .order_by(CensusMunicipalOverviewDistrict.row_index.asc())
                ).all()
            )
            if cs.sheet_type == "municipal_overview":
                if not _has_manual_sheet_type_override(cs, "municipal_overview") and (
                    _md_rows_are_bogus_house_rows(cs_mds)
                    or any(md.matched_overview_sheet_id == cs.id for md in md_rows)
                ):
                    _repair_misclassified_district_overview_sheet(session, cs, md_rows=cs_mds)
                    repaired_same_muni_ov.append(cs)
            else:
                if cs_mds:
                    _repair_misclassified_district_overview_sheet(session, cs, md_rows=cs_mds)
                repaired_same_muni_ov.append(cs)
        same_muni_ov_sheets = repaired_same_muni_ov
        same_muni_ov_sheets.sort(key=lambda cs: (cs.sheet_number or "9999", cs.id or 0))
        ov_sheets_by_id: dict[int, CensusSheet] = {
            cs.id: cs for cs in same_muni_ov_sheets if cs.id is not None
        }
        for md in md_rows:
            if md.matched_overview_sheet_id and md.matched_overview_sheet_id not in ov_sheets_by_id:
                extra_ov = session.get(CensusSheet, md.matched_overview_sheet_id)
                if extra_ov and extra_ov.id is not None:
                    ov_sheets_by_id[extra_ov.id] = extra_ov
                    same_muni_ov_sheets.append(extra_ov)

        ov_sids = list(ov_sheets_by_id.keys())
        ov_house_counts: dict[int, int] = {}
        joined_ovh_rows: list[CensusOverviewHouse] = []
        if ov_sids:
            raw_joined_ovh = list(
                session.exec(
                    select(CensusOverviewHouse)
                    .where(CensusOverviewHouse.census_sheet_id.in_(ov_sids))  # type: ignore[union-attr]
                    .order_by(
                        CensusOverviewHouse.census_sheet_id.asc(),
                        CensusOverviewHouse.row_index.asc(),
                        CensusOverviewHouse.id.asc(),
                    )
                ).all()
            )
            by_ov_sid: dict[int, list[CensusOverviewHouse]] = {}
            for jh in raw_joined_ovh:
                by_ov_sid.setdefault(jh.census_sheet_id, []).append(jh)
            for ov_sid in ov_sids:
                ov_s_obj = ov_sheets_by_id.get(ov_sid)
                s_rows = by_ov_sid.get(ov_sid, [])
                if ov_s_obj is not None and s_rows:
                    s_rows = _deduplicate_and_repair_overview_houses(session, ov_s_obj, s_rows)
                elif ov_s_obj is not None and not s_rows:
                    s_rows = reconstruct_overview_houses_from_existing_ocr(session, ov_s_obj)
                ov_house_counts[ov_sid] = len(s_rows)
                joined_ovh_rows.extend(s_rows)

        for md in md_rows:
            matched_row = rows_by_crop.get(md.stitched_crop_path or "")
            if matched_row is None and md.scan_id:
                for r_cand in rows_by_id.values():
                    if r_cand.scan_id == md.scan_id and r_cand.row_index == md.row_index:
                        matched_row = r_cand
                        break
            municipal_overview_districts.append(
                _serialize_municipal_overview_district(
                    session,
                    md,
                    municipal_sheet=sheet,
                    row_obj=matched_row,
                    scan_obj=scans_by_id.get(md.scan_id or 0),
                    overview_sheets_by_id=ov_sheets_by_id,
                    overview_house_counts=ov_house_counts,
                )
            )

        for ov_s in same_muni_ov_sheets:
            cnt = ov_house_counts.get(ov_s.id or 0, 0)
            linked_overview_sheets.append(_serialize_sheet(ov_s, overview_houses_count=cnt))

        if joined_ovh_rows:
            j_hn_ints = {jh.house_number_int for jh in joined_ovh_rows if jh.house_number_int is not None}
            j_hn_strs = {str(jh.house_number).strip() for jh in joined_ovh_rows if jh.house_number}
            j_cand_stmt = select(CensusSheet).where(
                CensusSheet.house_number.is_not(None),
                CensusSheet.sheet_type.notin_(["overview", "municipal_overview"]),  # type: ignore[union-attr]
            )
            if j_hn_ints and j_hn_strs:
                j_cand_stmt = j_cand_stmt.where(
                    (CensusSheet.house_number_int.in_(j_hn_ints))  # type: ignore[union-attr]
                    | (CensusSheet.house_number.in_(j_hn_strs))  # type: ignore[union-attr]
                )
            elif j_hn_ints:
                j_cand_stmt = j_cand_stmt.where(CensusSheet.house_number_int.in_(j_hn_ints))  # type: ignore[union-attr]
            elif j_hn_strs:
                j_cand_stmt = j_cand_stmt.where(CensusSheet.house_number.in_(j_hn_strs))  # type: ignore[union-attr]
            j_cand_sheets = list(session.exec(j_cand_stmt).all())
            j_cand_hh_map, j_cand_facts_map = _prefetch_sheets_hh_and_facts(session, j_cand_sheets)
            j_cand_by_id, j_cand_by_muni_hn = _build_candidate_sheets_index(j_cand_sheets)
            j_entry_cache: dict[int, dict[str, Any]] = {}
            seen_joined_keys: set[tuple[int, str]] = set()
            for jh in joined_ovh_rows:
                j_key = (
                    int(jh.census_sheet_id or 0),
                    normalize_house_number(jh.house_number) or str(jh.row_index),
                )
                if j_key in seen_joined_keys:
                    continue
                seen_joined_keys.add(j_key)
                parent_ov = ov_sheets_by_id.get(jh.census_sheet_id)
                joined_overview_houses.append(
                    _serialize_overview_house(
                        session,
                        jh,
                        overview_sheet=parent_ov,
                        candidate_sheets=j_cand_sheets,
                        candidate_hh_by_sheet=j_cand_hh_map,
                        candidate_facts_by_sheet=j_cand_facts_map,
                        candidate_by_id=j_cand_by_id,
                        candidate_by_muni_hn=j_cand_by_muni_hn,
                        candidate_entry_cache=j_entry_cache,
                    )
                )
            joined_overview_houses.sort(
                key=lambda item: (
                    item.get("house_number_int") if item.get("house_number_int") is not None else 999999,
                    str(item.get("house_number") or ""),
                )
            )
    elif eff_sheet_type == "overview" and muni_canon:
        # Check if this Prehľad sčítacieho obvodu ('Prehľad domov') belongs to a parent Obecný prehľad
        parent_muni_sheets = [
            cs
            for cs in session.exec(
                select(CensusSheet).where(
                    CensusSheet.census_year == sheet.census_year,
                    CensusSheet.sheet_type == "municipal_overview",
                )
            ).all()
            if canonical_municipality_key(cs.municipality) == muni_canon
        ]
        if parent_muni_sheets:
            parent_ms = parent_muni_sheets[0]
            md_rows_parent = list(
                session.exec(
                    select(CensusMunicipalOverviewDistrict)
                    .where(CensusMunicipalOverviewDistrict.census_sheet_id == parent_ms.id)
                    .order_by(
                        CensusMunicipalOverviewDistrict.is_summary_row.asc(),
                        CensusMunicipalOverviewDistrict.row_index.asc(),
                        CensusMunicipalOverviewDistrict.id.asc(),
                    )
                ).all()
            )
            md_rows_parent = _deduplicate_and_repair_municipal_overview(
                session, parent_ms, md_rows_parent
            )
            sibling_ov_sheets = [
                cs
                for cs in session.exec(
                    select(CensusSheet).where(
                        CensusSheet.census_year == sheet.census_year,
                        CensusSheet.sheet_type == "overview",
                    )
                ).all()
                if canonical_municipality_key(cs.municipality) == muni_canon
            ]
            sibling_ov_sheets.sort(key=lambda cs: (cs.sheet_number or "9999", cs.id or 0))
            sib_by_id = {cs.id: cs for cs in sibling_ov_sheets if cs.id is not None}
            sib_sids = list(sib_by_id.keys())
            sib_house_counts: dict[int, int] = {sid: 0 for sid in sib_sids}
            if sib_sids:
                for ov_sid, cnt in session.exec(
                    select(CensusOverviewHouse.census_sheet_id, func.count(CensusOverviewHouse.id))
                    .where(CensusOverviewHouse.census_sheet_id.in_(sib_sids))  # type: ignore[union-attr]
                    .group_by(CensusOverviewHouse.census_sheet_id)
                ).all():
                    if ov_sid is not None:
                        sib_house_counts[int(ov_sid)] = int(cnt)
            linked_municipal_overview = {
                **_serialize_sheet(
                    parent_ms,
                    municipal_districts_count=len([d for d in md_rows_parent if not d.is_summary_row]),
                ),
                "revisor_name": parent_ms.commissioner_name,
                "districts": [
                    _serialize_municipal_overview_district(
                        session,
                        md,
                        municipal_sheet=parent_ms,
                        overview_sheets_by_id=sib_by_id,
                        overview_house_counts=sib_house_counts,
                    )
                    for md in md_rows_parent
                ],
                "sibling_overview_sheets": [
                    _serialize_sheet(cs, overview_houses_count=sib_house_counts.get(cs.id or 0, 0))
                    for cs in sibling_ov_sheets
                    if cs.id != sheet.id
                ],
            }

    linked_overview_house: dict[str, Any] | None = None
    if eff_sheet_type not in ("overview", "municipal_overview") and sheet.house_number:
        ov_match = session.exec(
            select(CensusOverviewHouse).where(CensusOverviewHouse.matched_sheet_id == sheet.id)
        ).first()
        if ov_match is not None:
            ov_parent_check = session.get(CensusSheet, ov_match.census_sheet_id)
            if not can_match_overview_house_to_sheet(ov_match, sheet, overview_sheet=ov_parent_check):
                ov_match = None
        if not ov_match and sheet.municipality:
            ov_cand_stmt = select(CensusOverviewHouse).where(
                CensusOverviewHouse.census_year == sheet.census_year,
            )
            if sheet.house_number_int is not None:
                ov_cand_stmt = ov_cand_stmt.where(
                    (CensusOverviewHouse.house_number_int == sheet.house_number_int)
                    | (CensusOverviewHouse.house_number == str(sheet.house_number).strip())
                )
            candidates = list(session.exec(ov_cand_stmt).all())
            parent_sids = list({c.census_sheet_id for c in candidates if c.census_sheet_id})
            parents_by_id = (
                {
                    ps.id: ps
                    for ps in session.exec(select(CensusSheet).where(CensusSheet.id.in_(parent_sids))).all()  # type: ignore[union-attr]
                    if ps.id is not None
                }
                if parent_sids
                else {}
            )
            for cand in candidates:
                ov_parent_cand = parents_by_id.get(cand.census_sheet_id)
                if can_match_overview_house_to_sheet(cand, sheet, overview_sheet=ov_parent_cand):
                    ov_match = cand
                    break
        if ov_match:
            ov_parent = session.get(CensusSheet, ov_match.census_sheet_id)
            linked_overview_house = {
                **_serialize_overview_house(
                    session,
                    ov_match,
                    matched_sheet=sheet,
                    matched_hh=household,
                    matched_members_count=len(members),
                    overview_sheet=ov_parent,
                ),
                "overview_slovakiana_key": ov_parent.slovakiana_key if ov_parent else None,
            }

    linked_house_sheets = _find_linked_house_sheets(session, sheet)
    same_year_house_sheets = _find_same_year_house_sheets(session, sheet)

    muni_dist_cnt = len([d for d in municipal_overview_districts if not d.get("is_summary_row")])
    sheet_dict = _serialize_sheet(
        sheet,
        overview_houses_count=len(overview_houses) or len(joined_overview_houses),
        municipal_districts_count=muni_dist_cnt,
    )
    sheet_dict["linked_house_sheets"] = linked_house_sheets
    sheet_dict["same_year_house_sheets"] = same_year_house_sheets
    front_local_url = _scan_to_media_url(front_scan)
    back_local_url = _scan_to_media_url(back_scan)
    if front_local_url:
        sheet_dict["front_image_url"] = front_local_url
    if back_local_url:
        sheet_dict["back_image_url"] = back_local_url

    pages_payload: list[dict[str, Any]] = []
    for idx_sc, sc in enumerate(all_sheet_scans):
        p_side = sc.page_side or ("front" if idx_sc == 0 else ("back" if idx_sc == 1 else f"page_{idx_sc + 1}"))
        p_num = idx_sc + 1
        sc_url = _scan_to_media_url(sc)
        if p_side == "front":
            lbl = "Predná strana (_v)"
        elif p_side == "back":
            lbl = "Zadná strana (_r)"
        else:
            lbl = f"Strana {p_num}"
        pages_payload.append(
            {
                "page_number": p_num,
                "page_side": p_side,
                "label": lbl,
                "scan_id": sc.id,
                "scan_url": sc_url,
                "overlay_url": sc.overlay_path,
                "header_crop_url": sc.header_crop_path,
                "width": sc.width,
                "height": sc.height,
            }
        )

    fb_maps = build_feedback_lookup_maps(
        session,
        census_sheet_id=sheet.id,
        scan_ids=scan_ids_for_rows,
    )
    fb_by_row = fb_maps["by_row"]
    fb_by_fact = fb_maps["by_fact"]
    fb_by_house = fb_maps["by_house"]
    fb_by_dist = fb_maps["by_dist"]

    for m_item in members:
        f_obj = m_item.get("fact") or {}
        fid = f_obj.get("id")
        rid = f_obj.get("row_id")
        m_item["ocr_feedback"] = (
            (fb_by_fact.get(fid) if fid else None)
            or (fb_by_row.get(rid) if rid else None)
        )
    for p_item in persons:
        fid = p_item.get("id")
        rid = p_item.get("row_id")
        p_item["ocr_feedback"] = (
            (fb_by_fact.get(fid) if fid else None)
            or (fb_by_row.get(rid) if rid else None)
        )
    for ovh_item in overview_houses:
        hid = ovh_item.get("id")
        rid = ovh_item.get("row_id")
        ovh_item["ocr_feedback"] = (
            (fb_by_house.get(hid) if hid else None)
            or (fb_by_row.get(rid) if rid else None)
        )
    for md_item in municipal_overview_districts:
        did = md_item.get("id")
        rid = md_item.get("row_id")
        md_item["ocr_feedback"] = (
            (fb_by_dist.get(did) if did else None)
            or (fb_by_row.get(rid) if rid else None)
        )

    serialized_rows: list[dict[str, Any]] = []
    for r_obj in rows_by_id.values():
        r_sc = scans_by_id.get(r_obj.scan_id)
        r_side = r_sc.page_side if (r_sc and r_sc.page_side) else (
            "back" if (sheet.back_scan_id and r_obj.scan_id == sheet.back_scan_id) else "front"
        )
        r_pnum = 1 if r_side == "front" else 2
        if "_" in r_side:
            t_str = r_side.split("_", 1)[1]
            if t_str.isdigit():
                r_pnum = int(t_str)
        serialized_rows.append(
            {
                **r_obj.model_dump(),
                "page_side": r_side,
                "page_number": r_pnum,
                "ocr_feedback": fb_by_row.get(r_obj.id or 0),
            }
        )

    return {
        "sheet": sheet_dict,
        "front_scan": _serialize_scan(front_scan),
        "back_scan": _serialize_scan(back_scan),
        "front_scan_url": front_local_url or sheet_dict.get("front_image_url"),
        "front_overlay_url": front_scan.overlay_path if front_scan else None,
        "back_scan_url": back_local_url or sheet_dict.get("back_image_url"),
        "back_overlay_url": back_scan.overlay_path if back_scan else None,
        "pages": pages_payload,
        "household": serialized_households[0] if serialized_households else _serialize_household(household),
        "households": serialized_households,
        "members": members,
        "persons": persons,
        "rows": serialized_rows,
        "overview_houses": overview_houses,
        "municipal_overview_districts": municipal_overview_districts,
        "linked_overview_sheets": linked_overview_sheets,
        "joined_overview_houses": joined_overview_houses,
        "linked_municipal_overview": linked_municipal_overview,
        "linked_overview_house": linked_overview_house,
        "linked_house_sheets": linked_house_sheets,
        "same_year_house_sheets": same_year_house_sheets,
    }


@router.get("/sheets/{sheet_ref}/switch-year")
async def switch_census_sheet_year(
    sheet_ref: str,
    session: SessionDep,
    target_year: int = 1930,
    house_number: str | None = None,
) -> dict[str, Any]:
    """
    Resolves the cross-year target (1930 <-> 1940) for a given census sheet so the user
    can seamlessly transfer to the same house/village in the newly selected year.
    """
    sheet = await _resolve_census_sheet(session, sheet_ref)
    eff_target_year = _resolve_single_census_year(target_year)
    canonical_muni = canonical_municipality_display(sheet.municipality) or sheet.municipality
    target_muni = canonical_municipality_key(sheet.municipality)
    eff_type = sheet.sheet_type or "household"

    if sheet.census_year == eff_target_year:
        return {
            "target_type": "sheet",
            "slovakiana_key": sheet.slovakiana_key,
            "sheet_id": sheet.id,
            "house_id": None,
            "census_year": eff_target_year,
            "municipality": canonical_muni,
            "district": sheet.district,
            "house_number": house_number or sheet.house_number,
        }

    req_hn_raw = house_number or (sheet.house_number if eff_type == "household" else None)
    req_hn = normalize_house_number(req_hn_raw)

    # 1. If viewing a household sheet, find the same house in the same village in eff_target_year
    if eff_type == "household" and target_muni and req_hn:
        cross_hh = [
            o
            for o in _find_matching_village_house_sheets(session, sheet)
            if o.census_year == eff_target_year
        ]
        if not cross_hh:
            cands = list(
                session.exec(
                    select(CensusSheet).where(
                        CensusSheet.census_year == eff_target_year,
                        CensusSheet.house_number.is_not(None),
                    )
                ).all()
            )
            for c in cands:
                if (c.sheet_type or "household") in ("overview", "municipal_overview"):
                    continue
                if canonical_municipality_key(c.municipality) != target_muni:
                    continue
                if normalize_house_number(c.house_number) != req_hn:
                    continue
                if not is_compatible_settlement(sheet.street_raw, c.street_raw, sheet.municipality):
                    continue
                cross_hh.append(c)

        if cross_hh:
            cross_hh.sort(
                key=lambda o: (
                    0 if o.download_status in ("extracted", "matched", "ocr_done") else 1,
                    o.sheet_number or "9999",
                    o.id or 0,
                )
            )
            best = cross_hh[0]
            return {
                "target_type": "sheet",
                "slovakiana_key": best.slovakiana_key,
                "sheet_id": best.id,
                "house_id": None,
                "census_year": eff_target_year,
                "municipality": canonical_municipality_display(best.municipality) or canonical_muni,
                "district": best.district or sheet.district,
                "house_number": best.house_number or req_hn_raw,
            }

        # Check if the house is recorded in a target_year District Overview sheet
        ovh_cands = list(
            session.exec(
                select(CensusOverviewHouse).where(
                    CensusOverviewHouse.census_year == eff_target_year
                )
            ).all()
        )
        for ovh in ovh_cands:
            if canonical_municipality_key(ovh.municipality) != target_muni:
                continue
            if normalize_house_number(ovh.house_number) != req_hn:
                continue
            if ovh.matched_sheet_id:
                ms = session.get(CensusSheet, ovh.matched_sheet_id)
                if (
                    ms
                    and ms.census_year == eff_target_year
                    and canonical_municipality_key(ms.municipality) == target_muni
                ):
                    return {
                        "target_type": "sheet",
                        "slovakiana_key": ms.slovakiana_key,
                        "sheet_id": ms.id,
                        "house_id": None,
                        "census_year": eff_target_year,
                        "municipality": canonical_municipality_display(ms.municipality) or canonical_muni,
                        "district": ms.district or sheet.district,
                        "house_number": ms.house_number or ovh.house_number,
                    }
            ov_parent = session.get(CensusSheet, ovh.census_sheet_id) if ovh.census_sheet_id else None
            if ov_parent and ov_parent.census_year == eff_target_year:
                return {
                    "target_type": "overview_house",
                    "slovakiana_key": ov_parent.slovakiana_key,
                    "sheet_id": ov_parent.id,
                    "house_id": ovh.id,
                    "census_year": eff_target_year,
                    "municipality": canonical_muni,
                    "district": ov_parent.district or sheet.district,
                    "house_number": ovh.house_number,
                }

    # 2. If viewing an overview or municipal_overview sheet, find corresponding overview sheet in eff_target_year
    if eff_type in ("overview", "municipal_overview") and target_muni:
        year_sheets = [
            s
            for s in session.exec(
                select(CensusSheet).where(CensusSheet.census_year == eff_target_year)
            ).all()
            if canonical_municipality_key(s.municipality) == target_muni
        ]
        target_ov_sheets = [
            s
            for s in year_sheets
            if (s.sheet_type or "household") in ("overview", "municipal_overview")
        ]
        if target_ov_sheets:
            target_ov_sheets.sort(
                key=lambda s: (
                    0 if (s.sheet_type or "household") == eff_type else 1,
                    0
                    if (
                        sheet.obvod_number
                        and s.obvod_number
                        and s.obvod_number.strip().upper() == sheet.obvod_number.strip().upper()
                    )
                    else 1,
                    s.sheet_number or "9999",
                    s.id or 0,
                )
            )
            best_ov = target_ov_sheets[0]
            matched_house_id: int | None = None
            if req_hn and best_ov.id is not None:
                ov_houses = list(
                    session.exec(
                        select(CensusOverviewHouse).where(
                            CensusOverviewHouse.census_sheet_id == best_ov.id
                        )
                    ).all()
                )
                for ovh in ov_houses:
                    if normalize_house_number(ovh.house_number) == req_hn:
                        matched_house_id = ovh.id
                        break
            return {
                "target_type": "sheet",
                "slovakiana_key": best_ov.slovakiana_key,
                "sheet_id": best_ov.id,
                "house_id": matched_house_id,
                "census_year": eff_target_year,
                "municipality": canonical_municipality_display(best_ov.municipality) or canonical_muni,
                "district": best_ov.district or sheet.district,
                "house_number": req_hn_raw,
            }

    # 3. Fallback: transfer to the same village (and house number if applicable) in eff_target_year
    target_dist = sheet.district
    if target_muni:
        any_year_sheet = next(
            (
                s
                for s in session.exec(
                    select(CensusSheet).where(CensusSheet.census_year == eff_target_year)
                ).all()
                if canonical_municipality_key(s.municipality) == target_muni and s.district
            ),
            None,
        )
        if any_year_sheet and any_year_sheet.district:
            target_dist = any_year_sheet.district

    return {
        "target_type": "municipality",
        "slovakiana_key": None,
        "sheet_id": None,
        "house_id": None,
        "census_year": eff_target_year,
        "municipality": canonical_muni,
        "district": target_dist,
        "house_number": req_hn_raw if eff_type == "household" else None,
    }


@router.post("/sheets/{sheet_ref}/download-and-process")
async def trigger_sheet_download_and_process(
    sheet_ref: str,
    session: SessionDep,
    run_pipeline: bool = True,
    run_mcp: bool = True,
) -> dict[str, Any]:
    sheet = await _resolve_census_sheet(session, sheet_ref)
    if sheet.id is not None:
        reset_ocr_feedback(session, census_sheet_id=sheet.id)

    result = await download_and_process_single_census_sheet(
        sheet.id or 0,
        run_pipeline=run_pipeline,
        run_mcp=run_mcp,
    )
    return result


@router.post("/sheets/{sheet_ref}/persons")
async def add_census_person(
    sheet_ref: str,
    req: CreateCensusPersonRequest,
    session: SessionDep,
) -> dict[str, Any]:
    sheet = await _resolve_census_sheet(session, sheet_ref)

    existing_facts = session.exec(
        select(PersonFact).where(PersonFact.census_sheet_id == sheet.id)
    ).all()
    next_row_num = int(req.person_row_number or (len(existing_facts) + 1))

    scan_id = sheet.front_scan_id or 0
    row = RecordRow(
        scan_id=scan_id,
        row_index=next_row_num,
        y_start=0,
        y_end=100,
        status="extracted",
        record_type="census",
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    fact = PersonFact(
        row_id=row.id or 0,
        scan_id=scan_id,
        record_type="census",
        census_sheet_id=sheet.id,
        person_row_number=str(next_row_num),
        entry_number=str(next_row_num),
        event_year=sheet.census_year,
        village_raw=sheet.municipality,
        village_sk=sheet.municipality,
        house_number=sheet.house_number,
    )
    session.add(fact)
    session.commit()
    session.refresh(fact)

    return await update_census_person(fact.id or 0, UpdateCensusPersonRequest(**req.model_dump()), session)


@router.put("/persons/{fact_id}")
@router.patch("/facts/{fact_id}")
async def update_census_person(
    fact_id: int,
    req: UpdateCensusPersonRequest,
    session: SessionDep,
) -> dict[str, Any]:
    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="PersonFact not found")

    updates = req.model_dump(exclude_unset=True)
    if "surname_raw" in updates and updates["surname_raw"] is not None:
        s_val, _ = validate_person_name(updates["surname_raw"], role="surname")
        fact.surname_raw = s_val or updates["surname_raw"]
        fact.surname_normalized = strip_slovak_feminine_suffix(fact.surname_raw) or fact.surname_raw
    if "maiden_surname" in updates and updates["maiden_surname"] is not None:
        fact.mother_maiden_surname = updates["maiden_surname"]
    if "mother_maiden_surname" in updates and updates["mother_maiden_surname"] is not None:
        fact.mother_maiden_surname = updates["mother_maiden_surname"]
    if "given_name_raw" in updates and updates["given_name_raw"] is not None:
        g_val, _ = validate_person_name(updates["given_name_raw"], role="given")
        fact.given_name_raw = g_val or updates["given_name_raw"]
        g_sk, g_sex = normalize_given_name(fact.given_name_raw)
        fact.given_name_sk = g_sk or fact.given_name_raw
        fact.given_name_hu = g_sk
        if not updates.get("sex") and g_sex:
            fact.sex = g_sex
    if "relation_to_head" in updates:
        fact.relation_to_head = normalize_census_relation(updates["relation_to_head"]) or updates["relation_to_head"]
    if "sex" in updates and updates["sex"]:
        fact.sex = validate_sex(updates["sex"], fact.given_name_raw)
    if "birth_or_event_date" in updates:
        cdate, cyear = validate_census_date(updates["birth_or_event_date"])
        fact.birth_or_event_date = cdate or updates["birth_or_event_date"]
        if cyear and not updates.get("birth_year_est"):
            fact.birth_year_est = cyear
    if "birth_year_est" in updates and updates["birth_year_est"] is not None:
        fact.birth_year_est = updates["birth_year_est"]

    # Structured birthplace fields (Obec / Okres / Krajina)
    if "birth_place_municipality" in updates:
        fact.birth_place_municipality = updates["birth_place_municipality"] or None
    if "birth_place_district" in updates:
        fact.birth_place_district = updates["birth_place_district"] or None
    if "birth_place_country" in updates:
        fact.birth_place_country = updates["birth_place_country"] or None
    if "birth_place_raw" in updates:
        fact.birth_place_raw = updates["birth_place_raw"]
        bp_sk, _, _ = normalize_village(fact.birth_place_raw)
        fact.birth_place_sk = bp_sk or fact.birth_place_raw
    elif any(k in updates for k in ("birth_place_municipality", "birth_place_district", "birth_place_country")):
        fact.birth_place_raw = _format_birth_place_raw(
            fact.birth_place_municipality,
            fact.birth_place_district,
            fact.birth_place_country,
            fact.birth_place_raw,
        )
        bp_sk, _, _ = normalize_village(fact.birth_place_municipality or fact.birth_place_raw)
        fact.birth_place_sk = bp_sk or fact.birth_place_municipality or fact.birth_place_raw

    if "marriage_date" in updates and updates["marriage_date"] is not None:
        fact.marriage_date_raw = str(updates["marriage_date"])
    if "children_deceased_count" in updates and updates["children_deceased_count"] is not None:
        fact.children_dead_count = str(updates["children_deceased_count"])
    elif "children_living_count" in updates and updates["children_living_count"] is not None:
        try:
            born_int = int(updates.get("children_born_count") or fact.children_born_count or 0)
            liv_int = int(updates["children_living_count"])
            if born_int >= liv_int:
                fact.children_dead_count = str(born_int - liv_int)
        except Exception:
            pass

    if "religion" in updates:
        fact.religion = normalize_religion(updates["religion"]) or updates["religion"]
    if "social_status_or_occupation" in updates:
        raw_occ = str(updates["social_status_or_occupation"] or "").strip()
        fact.social_status_or_occupation = raw_occ or None

    for simple_key in (
        "person_row_number",
        "marital_status",
        "marriage_date_raw",
        "children_born_count",
        "children_dead_count",
        "moved_in_date",
        "previous_residence",
        "home_jurisdiction",
        "citizenship",
        "nationality",
        "literacy",
        "occupation_position",
        "employer_workplace",
        "presence_type",
        "observationes_raw",
        "spouse_name",
        "father_name",
        "mother_name",
    ):
        if simple_key in updates and updates[simple_key] is not None:
            setattr(fact, simple_key, str(updates[simple_key]))

    apply_person_fact_search_fields(fact)

    session.add(fact)
    session.commit()
    session.refresh(fact)

    if fact.census_sheet_id:
        reconcile_census_household_relationships(session, fact.census_sheet_id)
        match_census_to_local_matrika(session, fact)
        index_person_fact(session, fact)
        session.commit()
    else:
        index_person_fact(session, fact)
        session.commit()

    schedule_auto_fs_match(fact.id, trigger_auto_fs_match_for_fact)

    return {"status": "updated", "fact": fact}


@router.delete("/persons/{fact_id}")
@router.delete("/facts/{fact_id}")
def delete_census_person(fact_id: int, session: SessionDep) -> dict[str, Any]:
    from backend.app.db.session import delete_person_fact_cascade

    fact = session.get(PersonFact, fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="PersonFact not found")
    census_sheet_id = fact.census_sheet_id
    deleted = delete_person_fact_cascade(session, fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="PersonFact not found")
    if census_sheet_id:
        reconcile_census_household_relationships(session, census_sheet_id)
    return {"status": "deleted", "fact_id": fact_id}


@router.post("/sheets/{sheet_ref}/set-type")
async def set_census_sheet_type(
    sheet_ref: str,
    req: SetSheetTypeRequest,
    session: SessionDep,
) -> dict[str, Any]:
    sheet = await _resolve_census_sheet(session, sheet_ref)
    if req.sheet_type in ("municipal_overview", "obecny_prehlad", "prehlad_obvodov", "suhrn_obvodov"):
        target_type = "municipal_overview"
    elif req.sheet_type in ("overview", "prehlad_obvodu", "prehlad_domov", "suhrn_domov"):
        target_type = "overview"
    else:
        target_type = "household"

    # Reset all OCR feedback and comments on this sheet whenever its document type is switched
    if sheet.id is not None:
        reset_ocr_feedback(session, census_sheet_id=sheet.id, commit=False)

    sheet.sheet_type = target_type
    _set_manual_sheet_type_override(sheet, target_type)
    if req.obvod_number is not None:
        sheet.obvod_number = req.obvod_number
    if req.commissioner_name is not None:
        sheet.commissioner_name = req.commissioner_name
    session.add(sheet)
    session.commit()
    session.refresh(sheet)

    overview_houses: list[CensusOverviewHouse] = []
    municipal_districts: list[CensusMunicipalOverviewDistrict] = []
    if req.run_extract:
        if target_type == "municipal_overview":
            for ovh in session.exec(
                select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == sheet.id)
            ).all():
                session.delete(ovh)
            session.commit()
            front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
            back_scan = session.get(ScanImage, sheet.back_scan_id) if sheet.back_scan_id else None
            front_exists = bool(
                front_scan and front_scan.original_path and Path(front_scan.original_path).exists()
            )
            if not sheet.front_scan_id:
                await download_and_process_single_census_sheet(
                    sheet.id or 0,
                    run_pipeline=True,
                    run_mcp=False,
                    force_sheet_type="municipal_overview",
                )
                session.refresh(sheet)
                municipal_districts = list(
                    session.exec(
                        select(CensusMunicipalOverviewDistrict)
                        .where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
                        .order_by(CensusMunicipalOverviewDistrict.row_index.asc())
                    ).all()
                )
            elif front_exists:
                municipal_districts = await extract_census_municipal_overview_sheet(
                    session,
                    sheet,
                    front_scan,
                    back_scan,
                    force_municipal_overview=True,
                )
            else:
                municipal_districts = reconstruct_municipal_overview_districts_from_existing_ocr(
                    session, sheet
                )
            sheet.sheet_type = "municipal_overview"
            _set_manual_sheet_type_override(sheet, "municipal_overview")
            session.add(sheet)
            session.commit()
            session.refresh(sheet)
        elif target_type == "overview":
            for md in session.exec(
                select(CensusMunicipalOverviewDistrict).where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
            ).all():
                session.delete(md)
            session.commit()
            front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
            back_scan = session.get(ScanImage, sheet.back_scan_id) if sheet.back_scan_id else None
            front_exists = bool(
                front_scan and front_scan.original_path and Path(front_scan.original_path).exists()
            )
            if not sheet.front_scan_id:
                await download_and_process_single_census_sheet(
                    sheet.id or 0,
                    run_pipeline=True,
                    run_mcp=False,
                    force_sheet_type="overview",
                )
                session.refresh(sheet)
                overview_houses = list(
                    session.exec(
                        select(CensusOverviewHouse)
                        .where(CensusOverviewHouse.census_sheet_id == sheet.id)
                        .order_by(CensusOverviewHouse.row_index.asc())
                    ).all()
                )
            elif front_exists:
                overview_houses = await extract_census_overview_sheet(
                    session, sheet, front_scan, back_scan, force_district_overview=True
                )
            else:
                overview_houses = reconstruct_overview_houses_from_existing_ocr(session, sheet)
            sheet.sheet_type = "overview"
            _set_manual_sheet_type_override(sheet, "overview")
            session.add(sheet)
            session.commit()
            session.refresh(sheet)
        else:
            for ovh in session.exec(
                select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == sheet.id)
            ).all():
                session.delete(ovh)
            for md in session.exec(
                select(CensusMunicipalOverviewDistrict).where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
            ).all():
                session.delete(md)
            session.commit()
            front_scan = session.get(ScanImage, sheet.front_scan_id) if sheet.front_scan_id else None
            front_exists = bool(
                front_scan and front_scan.original_path and Path(front_scan.original_path).exists()
            )
            if sheet.front_scan_id and front_exists:
                from backend.app.pipeline.orchestrator import process_census_sheet_pipeline

                await process_census_sheet_pipeline(
                    sheet.id or 0, run_mcp=True, force_sheet_type="household"
                )
            elif not sheet.front_scan_id:
                await download_and_process_single_census_sheet(
                    sheet.id or 0,
                    run_pipeline=True,
                    run_mcp=True,
                    force_sheet_type="household",
                )
            sheet.sheet_type = "household"
            _set_manual_sheet_type_override(sheet, "household")
            session.add(sheet)
            session.commit()
            session.refresh(sheet)

    if sheet.id is not None:
        reset_ocr_feedback(session, census_sheet_id=sheet.id)

    return {
        "status": "updated",
        "sheet": _serialize_sheet(
            sheet,
            overview_houses_count=len(overview_houses),
            municipal_districts_count=len([d for d in municipal_districts if not d.is_summary_row]),
        ),
        "overview_houses": [_serialize_overview_house(session, h) for h in overview_houses],
        "municipal_overview_districts": [
            _serialize_municipal_overview_district(session, md, municipal_sheet=sheet)
            for md in municipal_districts
        ],
    }


@router.patch("/municipal-overview-districts/{district_id}")
def patch_municipal_overview_district(
    district_id: int,
    req: MunicipalOverviewDistrictPatchRequest,
    session: SessionDep,
) -> dict[str, Any]:
    md = session.get(CensusMunicipalOverviewDistrict, district_id)
    if not md:
        raise HTTPException(status_code=404, detail="CensusMunicipalOverviewDistrict not found")

    updates = req.model_dump(exclude_unset=True)
    if "ordinal_number" in updates and updates["ordinal_number"] is not None:
        updates["ordinal_number"] = str(updates["ordinal_number"])
    for k in (
        "ordinal_number",
        "settlement_name",
        "obvod_number",
        "commissioner_name",
        "houses_total",
        "houses_inhabited",
        "houses_uninhabited",
        "household_count",
        "present_persons_count",
        "remarks",
        "is_summary_row",
        "matched_overview_sheet_id",
    ):
        if k in updates:
            setattr(md, k, updates[k])

    session.add(md)
    session.commit()
    session.refresh(md)
    return {
        "status": "updated",
        "municipal_overview_district": _serialize_municipal_overview_district(session, md),
    }


@router.delete("/municipal-overview-districts/{district_id}")
def delete_municipal_overview_district(district_id: int, session: SessionDep) -> dict[str, Any]:
    md = session.get(CensusMunicipalOverviewDistrict, district_id)
    if not md:
        raise HTTPException(status_code=404, detail="CensusMunicipalOverviewDistrict not found")
    reset_ocr_feedback(session, municipal_district_id=district_id, commit=False)
    session.delete(md)
    session.commit()
    return {"status": "deleted", "district_id": district_id}


@router.patch("/overview-houses/{house_id}")
def patch_overview_house(
    house_id: int,
    req: OverviewHousePatchRequest,
    session: SessionDep,
) -> dict[str, Any]:
    ovh = session.get(CensusOverviewHouse, house_id)
    if not ovh:
        raise HTTPException(status_code=404, detail="CensusOverviewHouse not found")

    updates = req.model_dump(exclude_unset=True)
    if "ordinal_number" in updates and updates["ordinal_number"] is not None:
        updates["ordinal_number"] = str(updates["ordinal_number"])
    for k in (
        "ordinal_number",
        "settlement_name",
        "house_number",
        "is_inhabited",
        "owner_name",
        "household_count",
        "present_persons_count",
        "remarks",
        "matched_sheet_id",
    ):
        if k in updates:
            setattr(ovh, k, updates[k])

    if "house_number" in updates and ovh.house_number:
        import re as _re

        m_int = _re.search(r"(\d+)", str(ovh.house_number))
        ovh.house_number_int = int(m_int.group(1)) if m_int else None
        if "matched_sheet_id" not in updates:
            ov_parent = session.get(CensusSheet, ovh.census_sheet_id)
            ovh.matched_sheet_id = None
            cand_sheets = session.exec(
                select(CensusSheet).where(
                    CensusSheet.census_year == ovh.census_year,
                )
            ).all()
            for cs in cand_sheets:
                if can_match_overview_house_to_sheet(ovh, cs, overview_sheet=ov_parent):
                    ovh.matched_sheet_id = cs.id
                    break
    elif "matched_sheet_id" in updates and ovh.matched_sheet_id:
        ov_parent = session.get(CensusSheet, ovh.census_sheet_id)
        cand = session.get(CensusSheet, ovh.matched_sheet_id)
        if not cand or not can_match_overview_house_to_sheet(ovh, cand, overview_sheet=ov_parent):
            raise HTTPException(
                status_code=400,
                detail="Cannot link CensusOverviewHouse to a CensusSheet from a different municipality, year, or house number.",
            )

    if "owner_name" in updates:
        ovh.owner_surname_folded = owner_surname_folded_from_name(ovh.owner_name)

    session.add(ovh)
    session.commit()
    session.refresh(ovh)
    return {"status": "updated", "overview_house": _serialize_overview_house(session, ovh)}


@router.delete("/overview-houses/{house_id}")
def delete_overview_house(house_id: int, session: SessionDep) -> dict[str, Any]:
    ovh = session.get(CensusOverviewHouse, house_id)
    if not ovh:
        raise HTTPException(status_code=404, detail="CensusOverviewHouse not found")
    reset_ocr_feedback(session, overview_house_id=house_id, commit=False)
    session.delete(ovh)
    session.commit()
    return {"status": "deleted", "house_id": house_id}


@router.get("/municipality-overview")
@router.get("/municipalities/overview")
def get_municipality_overview(
    session: SessionDep,
    municipality: str | None = None,
    census_year: int | None = 1930,
    district: str | None = None,
    street: str | None = None,
    person_name: str | None = None,
    person_query: str | None = None,
    owner_query: str | None = None,
    birth_year: str | None = None,
    birth_place: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    """
    Returns a unified village/house directory (`Dom č. 1 .. N`) merging `Prehľad sčítacieho obvodu`
    (`CensusOverviewHouse`) rows and household `CensusSheet`s (`sheet_type == 'household'`)
    for the requested municipality in a single census year (1930 by default).
    """
    person_name = person_name or person_query or owner_query
    resolved_census_year = _resolve_single_census_year(census_year)
    stmt = select(CensusSheet).where(CensusSheet.census_year == resolved_census_year)

    sheets = list(session.exec(stmt.order_by(CensusSheet.id.asc())).all())
    has_muni = bool(municipality and municipality.strip())
    if has_muni and municipality:
        sheets = [s for s in sheets if _sheet_matches_municipality(s, municipality)]
        target_canon = canonical_municipality_key(municipality)
        exact_muni = [
            s
            for s in sheets
            if canonical_municipality_key(s.municipality) == target_canon
            or s.municipality_key == municipality.strip()
        ]
        if exact_muni:
            sheets = exact_muni

    if district and district.strip():
        dist_sheets = [
            s
            for s in sheets
            if _sheet_matches_district(s, district, municipality_specified=has_muni)
        ]
        if dist_sheets or not has_muni:
            sheets = dist_sheets

    sheet_ids = [s.id for s in sheets if s.id is not None]
    sheets_by_id: dict[int, CensusSheet] = {s.id: s for s in sheets if s.id is not None}

    households_by_sheet: dict[int, CensusHousehold] = {}
    facts_by_sheet: dict[int, list[PersonFact]] = {}
    member_counts_by_sheet: dict[int, int] = {}
    ov_row_ocr_by_id: dict[int, str] = {}
    if sheet_ids:
        for hh in session.exec(
            select(CensusHousehold).where(CensusHousehold.census_sheet_id.in_(sheet_ids))
        ).all():
            households_by_sheet[hh.census_sheet_id] = hh
        all_ov_facts = list(
            session.exec(
                select(PersonFact)
                .where(PersonFact.census_sheet_id.in_(sheet_ids))
                .order_by(PersonFact.id.asc())
            ).all()
        )
        for pf in all_ov_facts:
            if pf.census_sheet_id:
                facts_by_sheet.setdefault(pf.census_sheet_id, []).append(pf)
                member_counts_by_sheet[pf.census_sheet_id] = (
                    member_counts_by_sheet.get(pf.census_sheet_id, 0) + 1
                )
        f_rids = list({pf.row_id for pf in all_ov_facts if pf.row_id})
        if f_rids:
            for r_obj in session.exec(select(RecordRow).where(RecordRow.id.in_(f_rids))).all():
                if r_obj.id is not None:
                    ocr_t = " ".join(
                        p for p in (r_obj.raw_ocr_combined, r_obj.raw_ocr_left, r_obj.raw_ocr_right) if p
                    )
                    if ocr_t:
                        ov_row_ocr_by_id[r_obj.id] = ocr_t

    overview_sheets: list[dict[str, Any]] = []
    municipal_overview_sheets: list[dict[str, Any]] = []
    household_sheets_by_id: dict[int, CensusSheet] = {}
    # Keyed strictly by (canonical_municipality_key, census_year, normalize_house_number)
    household_sheets_by_muni_year_hn: dict[tuple[str, int, str], list[CensusSheet]] = {}

    ov_sheet_ids = [
        s.id
        for s in sheets
        if s.id is not None and (s.sheet_type or "household") in ("overview", "municipal_overview")
    ]
    muni_ov_sheet_ids = [
        s.id for s in sheets if s.id is not None and (s.sheet_type or "household") == "municipal_overview"
    ]
    ov_counts_map: dict[int, int] = {}
    if ov_sheet_ids:
        for ov_sid, cnt in session.exec(
            select(CensusOverviewHouse.census_sheet_id, func.count(CensusOverviewHouse.id))
            .where(CensusOverviewHouse.census_sheet_id.in_(ov_sheet_ids))  # type: ignore[union-attr]
            .group_by(CensusOverviewHouse.census_sheet_id)
        ).all():
            if ov_sid is not None:
                ov_counts_map[int(ov_sid)] = int(cnt)

    muni_ov_districts_map: dict[int, list[CensusMunicipalOverviewDistrict]] = {}
    if muni_ov_sheet_ids:
        for md in session.exec(
            select(CensusMunicipalOverviewDistrict)
            .where(CensusMunicipalOverviewDistrict.census_sheet_id.in_(muni_ov_sheet_ids))  # type: ignore[union-attr]
            .order_by(
                CensusMunicipalOverviewDistrict.is_summary_row.asc(),
                CensusMunicipalOverviewDistrict.row_index.asc(),
                CensusMunicipalOverviewDistrict.id.asc(),
            )
        ).all():
            muni_ov_districts_map.setdefault(md.census_sheet_id, []).append(md)

    # First pass: self-heal any Obecný prehľad sheet that was marked 'overview', then deduplicate & repair genuine Obecný prehľad sheets
    has_real_muni_overview_for_year: set[int] = set()
    linked_child_ov_ids: set[int] = set()
    bogus_muni_ov_sheet_ids: set[int] = set()
    for s in sheets:
        if (
            s.id is not None
            and (s.sheet_type or "household") == "overview"
            and not _has_manual_sheet_type_override(s, "overview")
            and s.front_scan_id
        ):
            f_sc = session.get(ScanImage, s.front_scan_id)
            if f_sc and f_sc.header_raw_ocr and detect_is_municipal_overview_sheet(f_sc.header_raw_ocr, s.slovakiana_key):
                s.sheet_type = "municipal_overview"
                session.add(s)
                session.commit()
                session.refresh(s)
                ov_counts_map.pop(s.id, None)
                rebuilt_mds = reconstruct_municipal_overview_districts_from_existing_ocr(session, s)
                if rebuilt_mds:
                    muni_ov_districts_map[s.id] = rebuilt_mds

    for s in sheets:
        if s.id is not None and (s.sheet_type or "household") == "municipal_overview":
            raw_mds = muni_ov_districts_map.get(s.id, [])
            if not raw_mds:
                raw_mds = reconstruct_municipal_overview_districts_from_existing_ocr(session, s)
                if raw_mds:
                    muni_ov_districts_map[s.id] = raw_mds
            if not _has_manual_sheet_type_override(s, "municipal_overview") and _md_rows_are_bogus_house_rows(raw_mds):
                bogus_muni_ov_sheet_ids.add(s.id)
                continue
            if raw_mds:
                cleaned_mds = _deduplicate_and_repair_municipal_overview(session, s, raw_mds)
                muni_ov_districts_map[s.id] = cleaned_mds
                has_real_muni_overview_for_year.add(int(s.census_year or 0))
                for md in cleaned_mds:
                    if not md.is_summary_row and md.matched_overview_sheet_id:
                        linked_child_ov_ids.add(md.matched_overview_sheet_id)

    # Second pass: auto-repair any district overview sheet (like cair-ko270vs) that was misclassified as 'municipal_overview'
    # or had bogus CensusMunicipalOverviewDistrict rows
    for s in sheets:
        if s.id is None:
            continue
        raw_mds = muni_ov_districts_map.get(s.id, [])
        if not _has_manual_sheet_type_override(s, "municipal_overview") and (
            s.id in bogus_muni_ov_sheet_ids
            or (raw_mds and (s.sheet_type or "household") == "overview")
            or (
                (s.sheet_type or "household") == "municipal_overview"
                and (
                    s.id in linked_child_ov_ids
                    or (
                        not raw_mds
                        and (
                            int(s.census_year or 0) in has_real_muni_overview_for_year
                            or bool(s.obvod_number)
                            or ov_counts_map.get(s.id, 0) > 0
                        )
                    )
                )
            )
        ):
            reconstructed_houses = _repair_misclassified_district_overview_sheet(
                session, s, md_rows=raw_mds
            )
            muni_ov_districts_map.pop(s.id, None)
            ov_counts_map[s.id] = len(reconstructed_houses)
        if (s.sheet_type or "household") == "overview":
            if s.commissioner_name:
                norm_comm = _normalize_municipal_commissioner_name(
                    s.commissioner_name,
                    municipality=s.municipality,
                    district=s.district,
                )
                if norm_comm and norm_comm != s.commissioner_name:
                    s.commissioner_name = norm_comm
                    session.add(s)
                    session.commit()
                    session.refresh(s)
            if ov_counts_map.get(s.id, 0) == 0:
                reconstructed_houses = reconstruct_overview_houses_from_existing_ocr(
                    session, s, md_rows=raw_mds
                )
                if reconstructed_houses:
                    ov_counts_map[s.id] = len(reconstructed_houses)

    ov_sheets_by_id_map: dict[int, CensusSheet] = {
        s.id: s for s in sheets if s.id is not None and (s.sheet_type or "household") == "overview"
    }

    for s in sheets:
        eff_type = s.sheet_type or "household"
        if eff_type == "municipal_overview":
            md_list = muni_ov_districts_map.get(s.id or 0, [])
            dist_cnt = len([d for d in md_list if not d.is_summary_row])
            ser_ms = _serialize_sheet(s, municipal_districts_count=dist_cnt)
            ser_ms["municipal_overview_districts"] = [
                _serialize_municipal_overview_district(
                    session,
                    md,
                    municipal_sheet=s,
                    overview_sheets_by_id=ov_sheets_by_id_map,
                    overview_house_counts=ov_counts_map,
                )
                for md in md_list
            ]
            municipal_overview_sheets.append(ser_ms)
        elif eff_type == "overview":
            ov_cnt = ov_counts_map.get(s.id or 0, 0)
            overview_sheets.append(_serialize_sheet(s, overview_houses_count=ov_cnt))
        else:
            if s.id is not None:
                household_sheets_by_id[s.id] = s
            m_key = canonical_municipality_key(s.municipality)
            hn_key = normalize_house_number(s.house_number)
            if m_key and hn_key:
                household_sheets_by_muni_year_hn.setdefault(
                    (m_key, int(s.census_year or 0), hn_key), []
                ).append(s)
                # Also index compound Slovakiana labels like '31a, 32b' under their integer base ('31')
                if "," in str(s.house_number or ""):
                    base_hn = re.sub(r"[a-z]+$", "", hn_key)
                    if base_hn and base_hn != hn_key:
                        household_sheets_by_muni_year_hn.setdefault(
                            (m_key, int(s.census_year or 0), base_hn), []
                        ).append(s)

    ov_stmt = select(CensusOverviewHouse)
    if sheet_ids:
        ov_stmt = ov_stmt.where(
            (CensusOverviewHouse.census_sheet_id.in_(sheet_ids))
            | (
                CensusOverviewHouse.municipality.ilike(f"%{(municipality or '').strip()}%")
                if municipality and municipality.strip()
                else False
            )
        )
    elif municipality and municipality.strip():
        ov_stmt = ov_stmt.where(CensusOverviewHouse.municipality.ilike(f"%{municipality.strip()}%"))
    ov_stmt = ov_stmt.where(CensusOverviewHouse.census_year == resolved_census_year)

    ov_houses = list(session.exec(ov_stmt.order_by(CensusOverviewHouse.census_sheet_id.asc(), CensusOverviewHouse.row_index.asc())).all())
    if municipality and municipality.strip():
        target_canon = canonical_municipality_key(municipality)
        m_folded = fold_text_diacritics(municipality.strip())
        exact_ov = [
            ovh
            for ovh in ov_houses
            if canonical_municipality_key(ovh.municipality) == target_canon
        ]
        if exact_ov:
            ov_houses = exact_ov
        else:
            ov_houses = [
                ovh
                for ovh in ov_houses
                if m_folded in fold_text_diacritics(ovh.municipality or "")
            ]

    # Self-heal any overview sheet whose CensusOverviewHouse rows contain duplicates or garbled OCR owner names in-place
    ov_by_sheet: dict[int, list[CensusOverviewHouse]] = {}
    for ovh in ov_houses:
        ov_by_sheet.setdefault(ovh.census_sheet_id, []).append(ovh)
    for ov_sid, s_ov_rows in list(ov_by_sheet.items()):
        ov_s = ov_sheets_by_id_map.get(ov_sid) or session.get(CensusSheet, ov_sid)
        if ov_s is not None and s_ov_rows:
            cleaned_ov = _deduplicate_and_repair_overview_houses(session, ov_s, s_ov_rows)
            ov_by_sheet[ov_sid] = cleaned_ov
            ov_counts_map[ov_sid] = len(cleaned_ov)
    ov_houses = [r for s_rows in ov_by_sheet.values() for r in s_rows]
    for ov_ser in overview_sheets:
        sid_val = ov_ser.get("id")
        if isinstance(sid_val, int) and sid_val in ov_counts_map:
            ov_ser["overview_houses_count"] = ov_counts_map[sid_val]
    for ms_ser in municipal_overview_sheets:
        for d_ser in ms_ser.get("municipal_overview_districts") or []:
            m_ov_sid = d_ser.get("matched_overview_sheet_id")
            if isinstance(m_ov_sid, int) and m_ov_sid in ov_counts_map:
                d_ser["matched_overview_houses_count"] = ov_counts_map[m_ov_sid]
                if isinstance(d_ser.get("matched_overview_sheet"), dict):
                    d_ser["matched_overview_sheet"]["overview_houses_count"] = ov_counts_map[m_ov_sid]

    has_st = bool(street and street.strip())
    has_pn = bool(person_name and person_name.strip())
    has_by = bool(birth_year and str(birth_year).strip())
    has_bp = bool(birth_place and birth_place.strip())
    has_q = bool(q and q.strip())
    available_streets_set: set[str] = set()
    for s_it in sheets:
        if s_it.street_raw and s_it.street_raw.strip() and s_it.street_raw.strip() not in ("-", "—", "?"):
            available_streets_set.add(s_it.street_raw.strip())
    for hh_it in households_by_sheet.values():
        if hh_it.street and hh_it.street.strip() and hh_it.street.strip() not in ("-", "—", "?"):
            available_streets_set.add(hh_it.street.strip())
    for ovh_it in ov_houses:
        if ovh_it.settlement_name and ovh_it.settlement_name.strip() and ovh_it.settlement_name.strip() not in ("-", "—", "?"):
            available_streets_set.add(ovh_it.settlement_name.strip())

    def _build_preview_for_facts(s_facts: list[PersonFact]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for pf in s_facts:
            is_pf_match = False
            pf_ocr = ov_row_ocr_by_id.get(pf.row_id)
            if has_pn or has_by or has_bp:
                is_pf_match = (
                    (not has_pn or _fact_matches_person_name(pf, person_name, ocr_text=pf_ocr))
                    and (not has_by or _fact_matches_birth_year(pf, birth_year))
                    and (not has_bp or _fact_matches_birth_place(pf, birth_place))
                )
            elif has_q and q:
                is_pf_match = (
                    _fact_matches_person_name(pf, q, ocr_text=pf_ocr)
                    or _fact_matches_birth_year(pf, q)
                    or _fact_matches_birth_place(pf, q)
                )
            previews.append(_serialize_person_preview(pf, matched_filter=is_pf_match))
        return previews

    def _house_matches_filters(
        entry: dict[str, Any],
        s_facts: list[PersonFact],
        ovh_obj: CensusOverviewHouse | None = None,
        sheet_obj: CensusSheet | None = None,
        hh_obj: CensusHousehold | None = None,
    ) -> bool:
        if has_st and street:
            st_cands: list[str | None] = [
                entry.get("street_name"),
                sheet_obj.street_raw if sheet_obj else None,
                hh_obj.street if hh_obj else None,
                ovh_obj.settlement_name if ovh_obj else None,
                entry.get("settlement_name") if not (sheet_obj and sheet_obj.street_raw) else None,
            ]
            if not text_matches_street(st_cands, street):
                return False

        name_parts: list[str | None] = [
            entry.get("owner_name"),
            entry.get("head_name"),
            entry.get("wife_name"),
            entry.get("remarks"),
        ]
        if ovh_obj is not None:
            name_parts.extend([ovh_obj.owner_name, ovh_obj.owner_surname_folded, ovh_obj.remarks])
        hh_dict = entry.get("household")
        if isinstance(hh_dict, dict):
            name_parts.extend(
                [
                    hh_dict.get("head_of_household_name"),
                    hh_dict.get("head_given_name"),
                    hh_dict.get("head_surname"),
                    hh_dict.get("wife_name"),
                    hh_dict.get("wife_given_name"),
                    hh_dict.get("wife_maiden_name"),
                ]
            )
        if has_q and q:
            q_val = q.strip()
            q_folded = fold_text_diacritics(q_val)
            meta_folded = fold_text_diacritics(
                " ".join(
                    str(x)
                    for x in (
                        entry.get("house_number"),
                        entry.get("owner_name"),
                        entry.get("head_name"),
                        entry.get("wife_name"),
                        entry.get("slovakiana_key"),
                        entry.get("street_name"),
                        entry.get("settlement_name"),
                        entry.get("remarks"),
                        ovh_obj.raw_ocr_text if ovh_obj else None,
                    )
                    if x
                )
            )
            q_ok = (
                (q_folded in meta_folded)
                or _text_matches_person_name(name_parts, q_val)
                or (ovh_obj is not None and _overview_house_matches_person_query(ovh_obj, q_val))
                or any(
                    _fact_matches_person_name(pf, q_val, ocr_text=ov_row_ocr_by_id.get(pf.row_id))
                    or _fact_matches_birth_year(pf, q_val)
                    or _fact_matches_birth_place(pf, q_val)
                    for pf in s_facts
                )
            )
            if not q_ok:
                return False

        if has_pn or has_by or has_bp:
            fact_combined = any(
                (not has_pn or _fact_matches_person_name(pf, person_name, ocr_text=ov_row_ocr_by_id.get(pf.row_id)))
                and (not has_by or _fact_matches_birth_year(pf, birth_year))
                and (not has_bp or _fact_matches_birth_place(pf, birth_place))
                for pf in s_facts
            )
            header_match = (
                has_pn
                and (
                    _text_matches_person_name(name_parts, person_name)
                    or (ovh_obj is not None and _overview_house_matches_person_query(ovh_obj, person_name))
                )
                and (not has_by or any(_fact_matches_birth_year(pf, birth_year) for pf in s_facts))
                and (not has_bp or any(_fact_matches_birth_place(pf, birth_place) for pf in s_facts))
            )
            if not (fact_combined or header_match):
                return False
        return True

    houses_directory: list[dict[str, Any]] = []
    covered_sheet_ids: set[int] = set()
    covered_hn_keys: set[str] = set()
    seen_ov_house_keys: set[tuple[int, str]] = set()
    linked_by_sheet_id = _batch_find_linked_house_sheets(session, sheets)

    for ovh in ov_houses:
        hn_str = str(ovh.house_number or "").strip()
        hn_key = normalize_house_number(hn_str)
        ov_dedup_key = (int(ovh.census_sheet_id or 0), hn_key or str(ovh.row_index))
        if ov_dedup_key in seen_ov_house_keys:
            continue
        seen_ov_house_keys.add(ov_dedup_key)
        ov_parent = sheets_by_id.get(ovh.census_sheet_id) or session.get(CensusSheet, ovh.census_sheet_id)
        ov_muni_raw = ovh.municipality or (ov_parent.municipality if ov_parent else None)
        ov_muni_key = canonical_municipality_key(ov_muni_raw)
        ov_muni_display = canonical_municipality_display(ov_muni_raw) or ov_muni_raw

        candidates = (
            household_sheets_by_muni_year_hn.get(
                (ov_muni_key, int(ovh.census_year or 0), hn_key), []
            )
            if (ov_muni_key and hn_key)
            else []
        )
        candidates_sorted = sorted(
            candidates,
            key=lambda cs: (
                0 if facts_by_sheet.get(cs.id or 0) else 1,
                0 if cs.download_status in ("extracted", "matched", "ocr_done") else 1,
                cs.sheet_number or "9999",
                cs.id or 0,
            ),
        )

        matched_sheet = household_sheets_by_id.get(ovh.matched_sheet_id or 0)
        if matched_sheet is None and ovh.matched_sheet_id:
            matched_sheet = session.get(CensusSheet, ovh.matched_sheet_id)

        # Enforce strict programmatic same-village + same-year + same-house-number matching!
        if matched_sheet is not None and not can_match_overview_house_to_sheet(
            ovh, matched_sheet, overview_sheet=ov_parent
        ):
            matched_sheet = None

        if matched_sheet is None and candidates_sorted:
            for cand in candidates_sorted:
                if can_match_overview_house_to_sheet(ovh, cand, overview_sheet=ov_parent):
                    matched_sheet = cand
                    break

        if matched_sheet and matched_sheet.id:
            covered_sheet_ids.add(matched_sheet.id)
        for cand in candidates_sorted:
            if cand.id:
                covered_sheet_ids.add(cand.id)
        if hn_key:
            covered_hn_keys.add(hn_key)
            base_hn_k = re.sub(r"[a-z]+$", "", hn_key)
            if base_hn_k:
                covered_hn_keys.add(base_hn_k)

        s_facts: list[PersonFact] = []
        seen_fact_ids: set[int] = set()
        for src_sheet in ([matched_sheet] if matched_sheet else []) + candidates_sorted:
            if src_sheet and src_sheet.id:
                for pf in facts_by_sheet.get(src_sheet.id, []):
                    if pf.id is not None and pf.id not in seen_fact_ids:
                        seen_fact_ids.add(pf.id)
                        s_facts.append(pf)
        persons_preview = _build_preview_for_facts(s_facts)

        hh = households_by_sheet.get(matched_sheet.id or 0) if matched_sheet else None
        hh_ser = _serialize_household(hh) if hh else None
        mem_cnt = len(s_facts) if matched_sheet else 0
        head_name = (
            ( f"{hh_ser.get('head_given_name') or ''} {hh_ser.get('head_surname') or ''}".strip() if hh_ser else None )
            or (matched_sheet.household_owner if matched_sheet else None)
        )
        wife_name = (
            f"{hh_ser.get('wife_given_name') or ''} {hh_ser.get('wife_maiden_surname') and ('rod. ' + hh_ser['wife_maiden_surname']) or ''}".strip()
            if hh_ser and hh_ser.get("wife_given_name")
            else None
        )

        linked_sheets = (
            linked_by_sheet_id.get(matched_sheet.id or 0, [])
            if (matched_sheet and matched_sheet.id in sheets_by_id)
            else (_find_linked_house_sheets(session, matched_sheet) if matched_sheet else [])
        )
        for ls in linked_sheets:
            ls_id = ls.get("id")
            if isinstance(ls_id, int):
                covered_sheet_ids.add(ls_id)

        matched_sheet_payload = _serialize_sheet(matched_sheet) if matched_sheet else None
        if matched_sheet_payload is not None:
            matched_sheet_payload["linked_house_sheets"] = linked_sheets
            matched_sheet_payload["extracted_persons_preview"] = persons_preview
            matched_sheet_payload["members_preview"] = persons_preview

        entry_street = (
            (matched_sheet.street_raw if matched_sheet and matched_sheet.street_raw else None)
            or (hh.street if hh and hh.street else None)
        )
        if entry_street and canonical_municipality_key(entry_street) in (ov_muni_key, "dedina"):
            entry_street = None
        entry_obj = {
            "house_number": hn_str or str(ovh.ordinal_number or ""),
            "house_number_int": ovh.house_number_int or (int(hn_key) if hn_key.isdigit() else 999999),
            "ordinal_number": ovh.ordinal_number,
            "municipality": ov_muni_display,
            "street_name": entry_street,
            "settlement_name": canonical_municipality_display(ovh.settlement_name) or ovh.settlement_name or ov_muni_display,
            "is_inhabited": ovh.is_inhabited,
            "owner_name": _normalize_overview_owner(ovh.owner_name) or ovh.owner_name,
            "overview_house_id": ovh.id,
            "overview_sheet_id": ovh.census_sheet_id,
            "overview_slovakiana_key": ov_parent.slovakiana_key if ov_parent else None,
            "obvod_number": ovh.obvod_number or (ov_parent.obvod_number if ov_parent else None),
            "overview_household_count": ovh.household_count,
            "overview_present_persons_count": ovh.present_persons_count,
            "remarks": ovh.remarks,
            "stitched_crop_path": ovh.stitched_crop_path,
            "sheet_id": matched_sheet.id if matched_sheet else None,
            "slovakiana_key": matched_sheet.slovakiana_key if matched_sheet else None,
            "head_name": head_name or None,
            "wife_name": wife_name or None,
            "extracted_members_count": mem_cnt,
            "extracted_members": persons_preview,
            "extracted_persons_preview": persons_preview,
            "members_preview": persons_preview,
            "pipeline_status": matched_sheet.download_status if matched_sheet else "missing_sheet",
            "sheet": matched_sheet_payload,
            "household": hh_ser,
            "linked_house_sheets": linked_sheets,
        }
        if _house_matches_filters(entry_obj, s_facts, ovh_obj=ovh, sheet_obj=matched_sheet, hh_obj=hh):
            houses_directory.append(entry_obj)

    # Also add any household CensusSheets that weren't in the overview table yet
    ov_years = {int(ovh.census_year) for ovh in ov_houses if ovh.census_year}
    has_person_or_q_filter = has_st or has_pn or has_by or has_bp or has_q
    for s in sheets:
        eff_type = s.sheet_type or "household"
        if eff_type in ("overview", "municipal_overview") or (s.id and s.id in covered_sheet_ids):
            continue
        if ov_years and not has_person_or_q_filter and int(s.census_year or 0) not in ov_years:
            continue
        hn_str = str(s.house_number or "").strip()
        if not hn_str and not facts_by_sheet.get(s.id or 0):
            continue
        hn_key = normalize_house_number(hn_str)
        if hn_key and hn_key in covered_hn_keys:
            continue
        hn_int = int(hn_str) if hn_str.isdigit() else 999999
        s_muni_key = canonical_municipality_key(s.municipality)
        sib_cands = (
            household_sheets_by_muni_year_hn.get((s_muni_key, int(s.census_year or 0), hn_key), [])
            if (s_muni_key and hn_key)
            else [s]
        )
        if s.id:
            covered_sheet_ids.add(s.id)
        for sib in sib_cands:
            if sib.id:
                covered_sheet_ids.add(sib.id)
        if hn_key:
            covered_hn_keys.add(hn_key)

        s_facts = []
        seen_f_ids: set[int] = set()
        for src_s in [s] + sib_cands:
            if src_s.id:
                for pf in facts_by_sheet.get(src_s.id, []):
                    if pf.id is not None and pf.id not in seen_f_ids:
                        seen_f_ids.add(pf.id)
                        s_facts.append(pf)
        persons_preview = _build_preview_for_facts(s_facts)
        hh = households_by_sheet.get(s.id or 0)
        hh_ser = _serialize_household(hh) if hh else None
        mem_cnt = len(s_facts)
        head_name = (
            (f"{hh_ser.get('head_given_name') or ''} {hh_ser.get('head_surname') or ''}".strip() if hh_ser else None)
            or s.household_owner
        )
        wife_name = (
            f"{hh_ser.get('wife_given_name') or ''} {hh_ser.get('wife_maiden_surname') and ('rod. ' + hh_ser['wife_maiden_surname']) or ''}".strip()
            if hh_ser and hh_ser.get("wife_given_name")
            else None
        )
        linked_sheets = linked_by_sheet_id.get(s.id or 0, [])
        for ls in linked_sheets:
            ls_id = ls.get("id")
            if isinstance(ls_id, int):
                covered_sheet_ids.add(ls_id)
        s_payload = _serialize_sheet(s)
        s_payload["linked_house_sheets"] = linked_sheets
        s_payload["extracted_persons_preview"] = persons_preview
        s_payload["members_preview"] = persons_preview
        s_muni_display = canonical_municipality_display(s.municipality) or s.municipality
        entry_street = s.street_raw or (hh.street if hh and hh.street else None)
        if entry_street and canonical_municipality_key(entry_street) in (s_muni_key, "dedina"):
            entry_street = None
        entry_obj = {
            "house_number": hn_str or "?",
            "house_number_int": hn_int,
            "ordinal_number": None,
            "municipality": s_muni_display,
            "street_name": entry_street,
            "settlement_name": canonical_municipality_display(s.street_raw) or s.street_raw or s_muni_display,
            "is_inhabited": True,
            "owner_name": head_name,
            "overview_house_id": None,
            "overview_sheet_id": None,
            "overview_household_count": 1 if mem_cnt > 0 else None,
            "overview_present_persons_count": mem_cnt if mem_cnt > 0 else None,
            "remarks": None,
            "stitched_crop_path": None,
            "sheet_id": s.id,
            "slovakiana_key": s.slovakiana_key,
            "head_name": head_name or None,
            "wife_name": wife_name or None,
            "extracted_members_count": mem_cnt,
            "extracted_members": persons_preview,
            "extracted_persons_preview": persons_preview,
            "members_preview": persons_preview,
            "pipeline_status": s.download_status,
            "sheet": s_payload,
            "household": hh_ser,
            "linked_house_sheets": linked_sheets,
        }
        if _house_matches_filters(entry_obj, s_facts, sheet_obj=s, hh_obj=hh):
            houses_directory.append(entry_obj)

    houses_directory.sort(key=lambda item: (item["house_number_int"], str(item["house_number"])))

    raw_muni = (sheets[0].municipality if sheets else None) or municipality
    resolved_muni = canonical_municipality_display(raw_muni) or raw_muni
    resolved_dist = (sheets[0].district if sheets else None) or district
    if not resolved_dist and has_muni and municipality:
        target_canon_fallback = canonical_municipality_key(municipality)
        fallback_sheet = next(
            (
                s
                for s in session.exec(select(CensusSheet)).all()
                if canonical_municipality_key(s.municipality) == target_canon_fallback and s.district
            ),
            None,
        )
        if fallback_sheet:
            resolved_dist = fallback_sheet.district
            if not raw_muni and fallback_sheet.municipality:
                resolved_muni = canonical_municipality_display(fallback_sheet.municipality) or fallback_sheet.municipality
    resolved_year = resolved_census_year

    return {
        "municipality": resolved_muni,
        "district": resolved_dist,
        "census_year": resolved_year,
        "municipal_overview_sheets": municipal_overview_sheets,
        "overview_sheets": overview_sheets,
        "houses_directory": houses_directory,
        "total_houses": len(houses_directory),
        "available_streets": sorted(available_streets_set),
    }


municipality_overview = get_municipality_overview


@router.get("/overview-houses")
def list_overview_houses(
    session: SessionDep,
    municipality: str | None = None,
    census_year: int | None = 1930,
    district: str | None = None,
    street: str | None = None,
    person_name: str | None = None,
    person_query: str | None = None,
    owner_query: str | None = None,
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    resolved_year = _resolve_single_census_year(census_year)
    stmt = (
        select(CensusOverviewHouse)
        .where(CensusOverviewHouse.census_year == resolved_year)
        .order_by(
            CensusOverviewHouse.census_sheet_id.asc(),
            CensusOverviewHouse.row_index.asc(),
            CensusOverviewHouse.id.asc(),
        )
    )
    raw_houses = list(session.exec(stmt).all())
    seen_ovh_keys: set[tuple[int, str]] = set()
    houses: list[CensusOverviewHouse] = []
    for h in raw_houses:
        hk = (int(h.census_sheet_id or 0), normalize_house_number(h.house_number) or str(h.row_index))
        if hk in seen_ovh_keys:
            continue
        seen_ovh_keys.add(hk)
        houses.append(h)
    if municipality and municipality.strip():
        target_canon = canonical_municipality_key(municipality)
        m_fold = fold_text_diacritics(municipality.strip())
        houses = [
            h
            for h in houses
            if canonical_municipality_key(h.municipality) == target_canon
            or m_fold in fold_text_diacritics(h.municipality or "")
        ]
    if district and district.strip():
        d_fold = fold_text_diacritics(district.strip())
        houses = [h for h in houses if d_fold in fold_text_diacritics(h.district or "")]
    if street and street.strip():
        houses = [h for h in houses if text_matches_street([h.settlement_name], street)]

    eff_query = person_name or person_query or owner_query or q
    if eff_query and eff_query.strip():
        q_val = eff_query.strip()
        q_fold = fold_text_diacritics(q_val)
        houses = [
            h
            for h in houses
            if _overview_house_matches_person_query(h, q_val)
            or q_fold in fold_text_diacritics(f"{h.house_number or ''} {h.settlement_name or ''} {h.remarks or ''}")
        ]

    total = len(houses)
    paged = houses[offset : offset + limit]
    candidate_sheets: list[CensusSheet] | None = None
    p_cand_hh_map: dict[int, CensusHousehold] = {}
    p_cand_facts_map: dict[int, list[PersonFact]] = {}
    if paged:
        p_hn_ints = {h.house_number_int for h in paged if h.house_number_int is not None}
        p_hn_strs = {str(h.house_number).strip() for h in paged if h.house_number}
        cand_stmt = select(CensusSheet).where(
            CensusSheet.house_number.is_not(None),
            CensusSheet.sheet_type.notin_(["overview", "municipal_overview"]),  # type: ignore[union-attr]
        )
        if p_hn_ints and p_hn_strs:
            cand_stmt = cand_stmt.where(
                (CensusSheet.house_number_int.in_(p_hn_ints))  # type: ignore[union-attr]
                | (CensusSheet.house_number.in_(p_hn_strs))  # type: ignore[union-attr]
            )
        elif p_hn_ints:
            cand_stmt = cand_stmt.where(CensusSheet.house_number_int.in_(p_hn_ints))  # type: ignore[union-attr]
        elif p_hn_strs:
            cand_stmt = cand_stmt.where(CensusSheet.house_number.in_(p_hn_strs))  # type: ignore[union-attr]
        candidate_sheets = list(session.exec(cand_stmt).all())
        p_cand_hh_map, p_cand_facts_map = _prefetch_sheets_hh_and_facts(session, candidate_sheets)
        p_cand_by_id, p_cand_by_muni_hn = _build_candidate_sheets_index(candidate_sheets)
        p_entry_cache: dict[int, dict[str, Any]] = {}
    else:
        p_cand_by_id, p_cand_by_muni_hn, p_entry_cache = {}, {}, {}
    return {
        "total": total,
        "overview_houses": [
            _serialize_overview_house(
                session,
                h,
                candidate_sheets=candidate_sheets,
                candidate_hh_by_sheet=p_cand_hh_map,
                candidate_facts_by_sheet=p_cand_facts_map,
                candidate_by_id=p_cand_by_id,
                candidate_by_muni_hn=p_cand_by_muni_hn,
                candidate_entry_cache=p_entry_cache,
            )
            for h in paged
        ],
    }


@router.get("/municipalities-summary")
def get_municipalities_summary(
    session: SessionDep,
    census_year: int | None = 1930,
    district: str | None = None,
) -> dict[str, Any]:
    """
    Returns a summary of municipalities in the local DB (total sheets, OCR-done sheets,
    overview sheets count, extracted persons count, and overview houses count) to power
    the Left Sidebar in CensusHub.tsx.
    Always computes counts for a single census year (`census_year`, 1930 by default) while
    preserving all known local municipalities so switching between 1930 and 1940 seamlessly
    transfers the selected municipality.
    """
    resolved_year = _resolve_single_census_year(census_year)
    all_sheets = list(session.exec(select(CensusSheet)).all())
    if district and district.strip():
        all_sheets = [s for s in all_sheets if _sheet_matches_district(s, district, municipality_specified=False)]
    ov_count_by_muni_key: dict[str, int] = {}
    ov_persons_by_muni_key: dict[str, int] = {}
    seen_summary_ovh: set[tuple[int, str]] = set()
    ov_rows_stmt = (
        select(
            CensusOverviewHouse.census_sheet_id,
            CensusOverviewHouse.row_index,
            CensusOverviewHouse.house_number,
            CensusOverviewHouse.municipality,
            CensusOverviewHouse.present_persons_count,
        )
        .where(CensusOverviewHouse.census_year == resolved_year)
        .order_by(
            CensusOverviewHouse.census_sheet_id.asc(),
            CensusOverviewHouse.row_index.asc(),
            CensusOverviewHouse.id.asc(),
        )
    )
    for ov_sid, r_idx, hn_val, raw_ov_muni, pp_val in session.exec(ov_rows_stmt).all():
        dedup_k = (int(ov_sid or 0), normalize_house_number(hn_val) or str(r_idx))
        if dedup_k in seen_summary_ovh:
            continue
        seen_summary_ovh.add(dedup_k)
        m_key = canonical_municipality_key(raw_ov_muni) or "neznama obec"
        ov_count_by_muni_key[m_key] = ov_count_by_muni_key.get(m_key, 0) + 1
        ov_persons_by_muni_key[m_key] = ov_persons_by_muni_key.get(m_key, 0) + int(pp_val or 0)

    mov_stmt = select(CensusMunicipalOverviewDistrict).where(
        CensusMunicipalOverviewDistrict.census_year == resolved_year
    )
    mov_rows_by_sheet: dict[int, list[CensusMunicipalOverviewDistrict]] = {}
    for md in session.exec(mov_stmt).all():
        if md.census_sheet_id:
            mov_rows_by_sheet.setdefault(int(md.census_sheet_id), []).append(md)

    mov_houses_by_muni_key: dict[str, int] = {}
    mov_persons_by_muni_key: dict[str, int] = {}
    for md_list in mov_rows_by_sheet.values():
        if not md_list or _md_rows_are_bogus_house_rows(md_list):
            continue
        raw_muni = md_list[0].municipality
        m_key = canonical_municipality_key(raw_muni) or "neznama obec"
        sum_rows = [d for d in md_list if d.is_summary_row]
        dist_rows = [d for d in md_list if not d.is_summary_row]
        sh_houses = (
            next((int(d.houses_total) for d in sum_rows if d.houses_total is not None), None)
            if sum_rows
            else None
        )
        if sh_houses is None:
            sh_houses = sum(int(d.houses_total or 0) for d in dist_rows)
        sh_persons = (
            next((int(d.present_persons_count) for d in sum_rows if d.present_persons_count is not None), None)
            if sum_rows
            else None
        )
        if sh_persons is None:
            sh_persons = sum(int(d.present_persons_count or 0) for d in dist_rows)
        mov_houses_by_muni_key[m_key] = mov_houses_by_muni_key.get(m_key, 0) + int(sh_houses or 0)
        mov_persons_by_muni_key[m_key] = mov_persons_by_muni_key.get(m_key, 0) + int(sh_persons or 0)

    person_counts_by_sheet: dict[int, int] = {}
    for row_sid, cnt in session.exec(
        select(PersonFact.census_sheet_id, func.count(PersonFact.id))
        .where(PersonFact.record_type == "census")
        .group_by(PersonFact.census_sheet_id)
    ).all():
        if row_sid:
            person_counts_by_sheet[row_sid] = int(cnt)

    by_muni: dict[str, dict[str, Any]] = {}
    for s in all_sheets:
        m_display = canonical_municipality_display(s.municipality) or s.municipality or "Neznáma obec"
        m_canon = canonical_municipality_key(m_display) or "neznama obec"
        d_name = s.district or ""
        entry = by_muni.get(m_canon)
        if not entry:
            ov_houses = ov_count_by_muni_key.get(m_canon) or mov_houses_by_muni_key.get(m_canon, 0)
            ov_persons = max(
                mov_persons_by_muni_key.get(m_canon, 0),
                ov_persons_by_muni_key.get(m_canon, 0),
            )
            entry = {
                "municipality": m_display,
                "municipality_key": s.municipality_key,
                "district": d_name,
                "district_key": s.district_key,
                "total_sheets": 0,
                "downloaded_sheets": 0,
                "ocr_done_sheets": 0,
                "overview_sheets_count": 0,
                "overview_houses_count": ov_houses,
                "overview_persons_count": ov_persons,
                "extracted_persons_count": 0,
                "persons_count": ov_persons,
                "years": [resolved_year],
            }
            by_muni[m_canon] = entry
        else:
            if s.census_year == resolved_year:
                if d_name:
                    entry["district"] = d_name
                if s.district_key:
                    entry["district_key"] = s.district_key
                if s.municipality_key:
                    entry["municipality_key"] = s.municipality_key
            else:
                if not entry.get("district") and d_name:
                    entry["district"] = d_name
                if not entry.get("district_key") and s.district_key:
                    entry["district_key"] = s.district_key
                if not entry.get("municipality_key") and s.municipality_key:
                    entry["municipality_key"] = s.municipality_key

        if s.census_year != resolved_year:
            continue

        entry["total_sheets"] += 1
        if s.download_status in ("downloaded", "sliced", "ocr_done", "extracted", "matched"):
            entry["downloaded_sheets"] += 1
        if s.download_status in ("ocr_done", "extracted", "matched"):
            entry["ocr_done_sheets"] += 1
        if s.sheet_type in ("overview", "municipal_overview"):
            entry["overview_sheets_count"] += 1
        if s.id:
            entry["extracted_persons_count"] += person_counts_by_sheet.get(s.id, 0)
            entry["persons_count"] = max(
                entry["extracted_persons_count"],
                entry["overview_persons_count"],
            )

    municipalities_list: list[dict[str, Any]] = list(by_muni.values())

    municipalities_list.sort(
        key=lambda x: (-x["ocr_done_sheets"], -x["total_sheets"], x["municipality"])
    )

    return {
        "census_year": resolved_year,
        "municipalities": municipalities_list,
        "total_municipalities": len(municipalities_list),
    }
