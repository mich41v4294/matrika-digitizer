from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.db.models import (
    CensusHousehold,
    CensusOverviewHouse,
    CensusSheet,
    FamilySearchMatch,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.pipeline.ai_client import ai_client
from backend.app.pipeline.ocr_stage import media_rel_to_path, resolve_row_extraction_crop
from backend.app.pipeline.census_layout import (
    detect_is_overview_sheet,
    extract_overview_header_metadata,
    slice_census_overview_sheet,
)
from backend.app.pipeline.lexicon import (
    CENSUS_RELATION_MAP as _RELATION_MAP,
    KNOWN_SLOVAK_SURNAMES,
    SLOVAK_ADJECTIVAL_SURNAME_STEMS,
    VILLAGE_MAP,
    _KNOWN_SURNAME_BY_FOLDED,
    _SURNAME_STEM_A_FOLDED,
    _SURNAME_STEM_O_FOLDED,
    apply_area_period_name_corrections,
    apply_person_fact_search_fields,
    ascii_fold,
    can_match_overview_house_to_sheet,
    canonical_municipality_display,
    canonical_municipality_key,
    compute_person_fact_search_fields,
    extract_maiden_surname,
    feminize_slovak_adjectival_surname,
    get_regional_municipality_gazetteer,
    is_known_given_name,
    is_month_name,
    is_occupation_or_role_token,
    is_slovak_adjectival_surname,
    masculinize_slovak_adjectival_surname,
    normalize_census_birthplace_fields,
    normalize_census_country,
    normalize_census_district,
    normalize_census_municipality,
    normalize_census_relation,
    normalize_given_name,
    normalize_house_number,
    normalize_occupation,
    normalize_religion,
    normalize_village,
    owner_surname_folded_from_name,
    parse_observationes_events,
    sanitize_corrupted_surname_suffix,
    strip_slovak_feminine_suffix,
)
from backend.app.pipeline.schemas import (
    LLMCensusExtractionSchema,
    LLMCensusOverviewRowSchema,
    LLMExtractionSchema,
    ValidatedPersonFactPayload,
    apply_human_answer_to_fields,
    is_header_or_garbage,
    parse_parents_structured,
    validate_census_date,
    validate_entry_number,
    validate_event_year,
    validate_house_number,
    validate_legitimacy,
    validate_matrika_date,
    validate_person_name,
    validate_sex,
    validate_village_raw,
)
from backend.app.search.fts import index_census_sheet, index_person_fact

EXTRACTION_SYSTEM_PROMPT = """You are an expert paleographer and genealogist transcribing 18th-20th century church registers (Matricula Baptisatorum, Copulatorum, Defunctorum).
STRICT TYPE-SAFETY & PALEOGRAPHIC RULES:
1. NEVER translate personal names or surnames, and NEVER change names from their old historical variant to a modern variant!
   - Keep "Andreas" as "Andreas" (NEVER change to Ondrej or András).
   - Keep "Stephanus" as "Stephanus", "Catharina" as "Catharina", "Joannes" as "Joannes".
2. STRICT FIELD TYPES:
   - `birth_or_event_date` and `baptism_or_burial_date`: MUST ONLY contain a day number (1..31) and/or a historical month name (e.g. "18. Augustus", "Julius 28", "24."). NEVER put personal names (like "Paulus", "Andreas") or occupations ("sutor", "boarius") into date fields!
   - `father_name`, `mother_name`, `given_name_raw`, `surname_raw`, `mother_maiden_surname`: MUST ONLY contain personal names and surnames. NEVER put month names (like "Augustus", "Julius"), occupations ("boarius", "opilio", "colonus", "sutor"), or punctuation (".", "/", "1") into name fields!
   - `sex`: MUST be strictly "M" or "F" (or null).
Return ONLY a single raw JSON object (no markdown fences or explanation)."""


def _heuristic_fallback_extract(
    raw_left: str,
    raw_right: str,
    page_year: str | None,
    record_type: str,
    human_override: str | None = None,
) -> dict[str, Any]:
    """
    Deterministic, type-safe parser that strips table header lines, enforces semantic types
    for dates vs. personal names vs. occupations, and preserves exact historical names.
    """
    left_lines = [
        ln.strip()
        for ln in (raw_left or "").splitlines()
        if ln.strip() and not is_header_or_garbage(ln)
    ]
    right_lines = [
        ln.strip()
        for ln in (raw_right or "").splitlines()
        if ln.strip() and not is_header_or_garbage(ln)
    ]

    cells_l: list[str] = []
    for ln in left_lines:
        for c in ln.split("|"):
            c_clean = c.strip()
            if c_clean and not is_header_or_garbage(c_clean):
                cells_l.append(c_clean)

    cells_r: list[str] = []
    for ln in right_lines:
        for c in ln.split("|"):
            c_clean = c.strip()
            if c_clean and not is_header_or_garbage(c_clean):
                cells_r.append(c_clean)

    combined = " | ".join(cells_l + cells_r)
    if human_override:
        combined = f"{combined} | [MANUÁLNA OPRAVA: {human_override}]"

    # 1. Extract sequential entry number (e.g. '51.', '52.', '10a', '6a')
    entry_no: str | None = None
    for c in cells_l[:3]:
        valid_en = validate_entry_number(c)
        if valid_en:
            entry_no = valid_en
            break

    # 2. Extract child's given name from short name cells first (avoiding long parent/godparent cells)
    given_raw: str | None = None
    given_cell: str | None = None
    for max_tokens in (2, 4):
        for c in cells_l:
            c_toks = c.split()
            if len(c_toks) > max_tokens:
                continue
            # Skip cells that contain occupations or parent connectors
            if any(is_occupation_or_role_token(t) for t in c_toks):
                continue
            for tok in c_toks:
                clean_tok = tok.strip(".,;0123456789°")
                if is_known_given_name(clean_tok):
                    given_raw, _ = normalize_given_name(clean_tok)
                    if given_raw:
                        given_cell = c
                        break
            if given_raw:
                break
        if given_raw:
            break

    # 3. Type-safe Date Extraction: only accept cells that pass `validate_matrika_date`
    first_line_cells: list[str] = []
    if left_lines:
        first_line_cells = [
            c.strip()
            for c in left_lines[0].split("|")
            if c.strip() and not is_header_or_garbage(c.strip())
        ]

    date_cells: list[str] = []
    scan_pool = first_line_cells[:5] if len(first_line_cells) >= 3 else cells_l[:6]
    for c in scan_pool:
        if c == given_raw:
            continue
        if c == given_cell:
            day_m = re.match(r"^(\d{1,2})(?:°|\b)", c)
            if day_m:
                d_val, _ = validate_matrika_date(day_m.group(1), entry_number=entry_no)
                if d_val:
                    date_cells.append(d_val)
            continue
        if entry_no and validate_entry_number(c) == entry_no and not is_month_name(c) and not entry_no.lower().endswith("a"):
            continue
        d_val, misplaced_names = validate_matrika_date(c, entry_number=entry_no)
        # Never treat a cell with a misplaced person name (like 'Paulus 23' or 'Kidveny Gaulus sutor') as a date cell
        if d_val and not misplaced_names:
            date_cells.append(d_val)

    # Check if a second line on the left page starts with a month continuation (e.g. 'Majus A°' or 'April')
    if len(left_lines) > 1 and (not date_cells or all(re.match(r"^\d{1,2}\.$", dc) for dc in date_cells)):
        for ln2 in left_lines[1:]:
            first_c2 = ln2.split("|")[0].strip()
            if first_c2:
                d_val2, misplaced2 = validate_matrika_date(first_c2, entry_number=entry_no)
                if d_val2 and not misplaced2 and any(is_month_name(t) for t in d_val2.split()):
                    date_cells.append(d_val2)
                    break

    birth_date: str | None = None
    baptism_date: str | None = None
    if len(date_cells) >= 2:
        # Check if date_cells[0] is a bare day (e.g. '22.') and date_cells[1] is a bare month (e.g. 'Majus')
        if re.match(r"^\d{1,2}\.$", date_cells[0]) and is_month_name(date_cells[1]):
            birth_date = f"{date_cells[0]} {date_cells[1]}"
            baptism_date = date_cells[2] if len(date_cells) >= 3 else None
        elif is_month_name(date_cells[0]) and re.match(r"^\d{1,2}\.$", date_cells[1]):
            birth_date = date_cells[0] if any(ch.isdigit() for ch in date_cells[0]) else f"{date_cells[0]} {date_cells[1].rstrip('.')}"
            baptism_date = date_cells[1] if any(ch.isdigit() for ch in date_cells[0]) else (date_cells[2] if len(date_cells) >= 3 else None)
        else:
            birth_date = date_cells[0]
            baptism_date = date_cells[1]
    elif len(date_cells) == 1:
        birth_date = date_cells[0]

    # 4. Type-safe Parents Extraction
    parent_markers = (
        " et ", "&", "uxor", "utox", "utor", "cath", "ev.", "cv.", "e.a.c", "ev. a.",
        "famul", "famil", "colon", "colm", "columnev", "inquilin", "molitor", "sutor", "boari", "boarins", "opilio", "portal",
    )
    left_parent_candidates: list[str] = []
    for c in cells_l:
        if c == given_raw or c == given_cell:
            continue
        if entry_no and validate_entry_number(c) == entry_no:
            continue
        if is_month_name(c):
            continue
        d_check, d_names = validate_matrika_date(c, entry_number=entry_no)
        if d_check and not d_names:
            continue
        if any(w in c.lower() for w in parent_markers) or len(c) > 12:
            left_parent_candidates.append(c)

    # Prefer the candidate that has explicit parent/religion markers or `<Surname> <GivenName>` structure
    parents_cell = ""
    if left_parent_candidates:
        scored_candidates: list[tuple[int, str]] = []
        for cand in left_parent_candidates:
            score = len(cand)
            low_c = cand.lower()
            if any(m in low_c for m in (" et ", "uxor", "utox", "familii", "coa colm", "columnev", "ev. a.", "cv. a.", "portal. of")):
                score += 40
            parsed_cand = parse_parents_structured(cand)
            if parsed_cand["father_name"] and parsed_cand["mother_name"]:
                score += 30
            elif parsed_cand["father_name"] and parsed_cand["surname_raw"]:
                score += 15
            scored_candidates.append((score, cand))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        parents_cell = scored_candidates[0][1]

    if not parents_cell:
        for c in cells_r:
            if any(w in c.lower() for w in parent_markers):
                parents_cell = c
                break

    parsed_parents = parse_parents_structured(parents_cell)
    father_name = parsed_parents["father_name"]
    mother_name = parsed_parents["mother_name"]
    surname_raw = parsed_parents["surname_raw"]
    mother_maiden = parsed_parents["mother_maiden_surname"]

    # 5. Village & House Number extraction
    village_raw: str | None = None
    if cells_r:
        village_raw = validate_village_raw(cells_r[0], parents_cell)
    if not village_raw:
        low_comb = combined.lower()
        for v_key, (v_orig, _) in VILLAGE_MAP.items():
            if v_key in low_comb:
                village_raw = v_orig
                break

    village_preserved, village_hu, house_no = normalize_village(village_raw)

    # 6. Godparents, Officiant, Observationes
    godparents_raw, _ = validate_person_name(cells_r[1] if len(cells_r) > 1 else None, role="godparents")
    officiant_raw, _ = validate_person_name(cells_r[2] if len(cells_r) > 2 else None, role="officiant")
    obs_candidate = cells_r[3] if len(cells_r) > 3 else (cells_r[-1] if len(cells_r) >= 3 else None)
    obs_raw, _ = validate_person_name(obs_candidate, role="godparents")
    if obs_raw and obs_raw == officiant_raw:
        obs_raw = None

    sex = validate_sex(None, given_raw)
    legitimacy = validate_legitimacy(None, " ".join(cells_l))
    event_year = validate_event_year(page_year, page_year)

    payload: dict[str, Any] = {
        "entry_number": entry_no,
        "event_year": event_year,
        "birth_or_event_date": birth_date,
        "baptism_or_burial_date": baptism_date,
        "given_name_raw": given_raw,
        "given_name_sk": given_raw,
        "given_name_hu": given_raw,
        "surname_raw": surname_raw,
        "surname_normalized": surname_raw,
        "sex": sex,
        "legitimacy": legitimacy,
        "religion": normalize_religion(combined),
        "father_name": father_name,
        "mother_name": mother_name,
        "mother_maiden_surname": mother_maiden,
        "spouse_name": None,
        "social_status_or_occupation": normalize_occupation(combined),
        "village_raw": village_raw,
        "village_sk": village_preserved,
        "village_hu": village_hu,
        "house_number": house_no,
        "godparents_or_witnesses": godparents_raw,
        "officiant": officiant_raw,
        "age_at_event": None,
        "cause_of_death": None,
        "observationes_raw": obs_raw,
        "type_safety_repairs": [],
    }
    area_hint = village_hu or village_preserved or village_raw
    payload["type_safety_repairs"] = apply_area_period_name_corrections(
        payload, area=area_hint, year=event_year
    )
    return payload


def reconcile_extraction_with_type_safety(
    llm_raw: dict[str, Any] | None,
    fallback: dict[str, Any],
    *,
    page_year: str | None = None,
    answered_qs: list[QuestionItem] | None = None,
    has_ditto_marks: bool = False,
    prev_fact: PersonFact | None = None,
) -> dict[str, Any]:
    """
    Validates raw LLM output via `LLMExtractionSchema`, enforces cross-field semantic type safety
    (catching e.g. `birth_or_event_date='Paulus 23'` and `father_name='Augustus'`), and merges
    with the deterministic fallback parser to produce a `ValidatedPersonFactPayload`.
    """
    llm_validated = LLMExtractionSchema.model_validate(llm_raw or {})
    repairs: list[str] = list(llm_validated.type_safety_repairs)

    merged: dict[str, Any] = dict(fallback)

    # 1. Entry Number & Event Year
    merged["entry_number"] = fallback.get("entry_number") or llm_validated.entry_number
    merged["event_year"] = validate_event_year(llm_validated.event_year, page_year or fallback.get("event_year"))

    # 2. Dates (strictly validated; prefer date that includes both day & month over bare day)
    fb_birth = fallback.get("birth_or_event_date")
    llm_birth = llm_validated.birth_or_event_date
    if fb_birth and llm_birth:
        # Prefer whichever date includes a month name (e.g. '18. Augustus' or 'Julius 28' over '18.')
        fb_has_month = any(is_month_name(t) for t in fb_birth.split())
        llm_has_month = any(is_month_name(t) for t in llm_birth.split())
        merged["birth_or_event_date"] = fb_birth if (fb_has_month and not llm_has_month) else llm_birth
    else:
        merged["birth_or_event_date"] = fb_birth or llm_birth

    fb_bapt = fallback.get("baptism_or_burial_date")
    llm_bapt = llm_validated.baptism_or_burial_date
    merged["baptism_or_burial_date"] = fb_bapt or llm_bapt

    # 3. Child Given Name & Sex
    fb_given = fallback.get("given_name_raw")
    llm_given = llm_validated.given_name_raw
    chosen_given = fb_given if (fb_given and is_known_given_name(fb_given)) else (llm_given or fb_given)
    verbatim_given, _ = normalize_given_name(chosen_given)
    merged["given_name_raw"] = verbatim_given
    merged["given_name_sk"] = verbatim_given
    merged["given_name_hu"] = verbatim_given
    merged["sex"] = validate_sex(llm_validated.sex or fallback.get("sex"), verbatim_given)

    # 4. Parents & Surnames (never allow month names, occupations, or single-token truncation when fallback has full name)
    fb_father = fallback.get("father_name")
    llm_father = llm_validated.father_name
    if fb_father:
        merged["father_name"] = fb_father
    else:
        merged["father_name"] = llm_father

    fb_mother = fallback.get("mother_name")
    llm_mother = llm_validated.mother_name
    if fb_mother:
        merged["mother_name"] = fb_mother
    else:
        merged["mother_name"] = llm_mother

    verbatim_surname = fallback.get("surname_raw") or llm_validated.surname_raw
    if not verbatim_surname and merged.get("father_name"):
        s_cand, _ = validate_person_name(merged["father_name"].split()[0], role="surname")
        verbatim_surname = s_cand
    merged["surname_raw"] = verbatim_surname
    merged["surname_normalized"] = verbatim_surname

    merged["mother_maiden_surname"] = (
        fallback.get("mother_maiden_surname") or llm_validated.mother_maiden_surname
    )

    # 5. Village & House Number
    merged["village_raw"] = fallback.get("village_raw") or llm_validated.village_raw
    v_orig, v_hu, h_no = normalize_village(merged.get("village_raw"))
    if v_orig:
        merged["village_sk"] = v_orig
    if v_hu:
        merged["village_hu"] = v_hu
    merged["house_number"] = validate_house_number(
        h_no or llm_validated.house_number or fallback.get("house_number"),
        merged.get("village_raw"),
    )

    # 6. Godparents, Officiant, Observationes (ensure no duplication of parents cell)
    parents_norm = f"{merged.get('father_name') or ''} {merged.get('mother_name') or ''}".lower()
    llm_gp = llm_validated.godparents_or_witnesses
    if llm_gp and merged.get("father_name") and merged["father_name"].lower() in llm_gp.lower():
        repairs.append(f"Rejected godparents_or_witnesses duplicating father_name ({llm_gp!r})")
        llm_gp = None
    merged["godparents_or_witnesses"] = fallback.get("godparents_or_witnesses") or llm_gp

    merged["officiant"] = fallback.get("officiant") or llm_validated.officiant

    llm_obs = llm_validated.observationes_raw
    if llm_obs and merged.get("father_name") and merged["father_name"].lower() in llm_obs.lower():
        repairs.append(f"Rejected observationes_raw duplicating father_name ({llm_obs!r})")
        llm_obs = None
    merged["observationes_raw"] = fallback.get("observationes_raw") or llm_obs

    # 7. Apply verified human corrections type-safely
    for q in answered_qs or []:
        if q.selected_answer:
            apply_human_answer_to_fields(
                merged,
                q.field_name,
                q.selected_answer,
                q.answered_by or "Reviewer",
            )

    # 8. Ditto mark resolution from previous row
    if has_ditto_marks and prev_fact:
        if not merged.get("village_raw") or re.match(
            r'^(?:"|idem|eadem|dtto)\.?$', str(merged.get("village_raw") or ""), re.IGNORECASE
        ):
            merged["village_raw"] = prev_fact.village_raw
            merged["village_sk"] = prev_fact.village_sk
            merged["village_hu"] = prev_fact.village_hu
        if not merged.get("officiant") or re.search(
            r"\b(?:idem|i\s+dem|eadem|dtto)\b", str(merged.get("officiant") or ""), re.IGNORECASE
        ):
            merged["officiant"] = prev_fact.officiant

    # Re-normalize village after human answers & ditto resolution
    v_orig2, v_hu2, h_no2 = normalize_village(merged.get("village_raw"))
    if v_orig2:
        merged["village_sk"] = v_orig2
    if v_hu2:
        merged["village_hu"] = v_hu2
    if h_no2 and not merged.get("house_number"):
        merged["house_number"] = h_no2

    obs_events = parse_observationes_events(merged.get("observationes_raw"))
    merged["observationes_events"] = obs_events
    area_hint = merged.get("village_hu") or merged.get("village_sk") or merged.get("village_raw")
    repairs.extend(
        apply_area_period_name_corrections(
            merged, area=area_hint, year=merged.get("event_year") or page_year
        )
    )
    merged["type_safety_repairs"] = repairs
    if repairs and parents_norm:
        pass

    validated_payload = ValidatedPersonFactPayload.model_validate(merged)
    return validated_payload.model_dump()


async def run_extraction_for_row(
    session: Session,
    scan: ScanImage,
    row: RecordRow,
    prev_fact: PersonFact | None = None,
) -> PersonFact:
    answered_qs = session.exec(
        select(QuestionItem).where(QuestionItem.row_id == row.id, QuestionItem.status == "answered")
    ).all()
    human_notes = "; ".join(
        [f"{q.field_name}: {q.selected_answer}" for q in answered_qs if q.selected_answer]
    )

    fallback = _heuristic_fallback_extract(
        row.raw_ocr_left or "",
        row.raw_ocr_right or "",
        scan.detected_year,
        scan.detected_type,
        human_notes or None,
    )

    user_prompt = (
        f"IMPORTANT: NEVER translate or modernize personal names or surnames. Keep exact historical variants (e.g. keep Andreas as Andreas, Stephanus as Stephanus).\n"
        f"TYPE SAFETY: Dates must ONLY contain day (1..31) and month (e.g. '18. Augustus', 'Julius 28'). NEVER put personal names (e.g. 'Paulus') in dates, and NEVER put month names (e.g. 'Augustus') or punctuation ('.', '/') in person/parent names!\n"
        f"Pre-parsed column hints: entry={fallback.get('entry_number')}, birth_date={fallback.get('birth_or_event_date')}, "
        f"baptism_date={fallback.get('baptism_or_burial_date')}, child={fallback.get('given_name_raw')}, "
        f"father={fallback.get('father_name')}, mother={fallback.get('mother_name')}, village={fallback.get('village_raw')}\n"
        f"Return ONLY raw JSON (no markdown, no explanation) with keys: entry_number, event_year, birth_or_event_date, baptism_or_burial_date, "
        f"given_name_raw, surname_raw, sex, father_name, mother_name, "
        f"mother_maiden_surname, village_raw, house_number, godparents_or_witnesses, "
        f"officiant, observationes_raw from this {scan.detected_type} matrika row (Year {scan.detected_year or '1884'}):\n"
        f"{row.raw_ocr_combined or ''}"
    )
    if human_notes:
        user_prompt += f"\nVerified human correction: {human_notes}"

    res = await ai_client.chat_completion(
        stage="extract",
        model=settings.model_extract,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        prompt=user_prompt,
        scan_id=scan.id,
        row_id=row.id,
        max_tokens=260,
        temperature=0.0,
        timeout_sec=45.0,
        skip_thinking=True,
        response_model=LLMExtractionSchema,
    )

    data = res.get("parsed") or {}
    merged = reconcile_extraction_with_type_safety(
        data,
        fallback,
        page_year=scan.detected_year,
        answered_qs=list(answered_qs),
        has_ditto_marks=bool(row.has_ditto_marks),
        prev_fact=prev_fact,
    )

    row.normalized_json = json.dumps(merged, ensure_ascii=False)
    row.status = "extracted"
    session.add(row)

    fact = session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()
    if not fact:
        fact = PersonFact(row_id=row.id, scan_id=scan.id)

    fact.record_type = scan.detected_type
    fact.entry_number = str(merged.get("entry_number") or row.entry_number or row.row_index)
    fact.event_year = int(merged.get("event_year") or 1884)
    fact.birth_or_event_date = merged.get("birth_or_event_date")
    fact.baptism_or_burial_date = merged.get("baptism_or_burial_date")
    fact.given_name_raw = merged.get("given_name_raw")
    fact.given_name_sk = merged.get("given_name_sk")
    fact.given_name_hu = merged.get("given_name_hu")
    fact.surname_raw = merged.get("surname_raw")
    fact.surname_normalized = merged.get("surname_normalized")
    fact.sex = merged.get("sex")
    fact.legitimacy = merged.get("legitimacy")
    fact.religion = merged.get("religion")
    fact.father_name = merged.get("father_name")
    fact.mother_name = merged.get("mother_name")
    fact.mother_maiden_surname = merged.get("mother_maiden_surname")
    fact.spouse_name = merged.get("spouse_name")
    fact.social_status_or_occupation = merged.get("social_status_or_occupation")
    fact.village_raw = merged.get("village_raw")
    fact.village_sk = merged.get("village_sk")
    fact.village_hu = merged.get("village_hu")
    fact.house_number = str(merged.get("house_number") or "") or None
    fact.godparents_or_witnesses = merged.get("godparents_or_witnesses")
    fact.officiant = merged.get("officiant")
    fact.age_at_event = str(merged.get("age_at_event") or "") or None
    fact.cause_of_death = merged.get("cause_of_death")
    fact.observationes_raw = merged.get("observationes_raw")
    fact.observationes_events_json = json.dumps(merged.get("observationes_events") or [], ensure_ascii=False)

    apply_person_fact_search_fields(fact)

    session.add(fact)
    session.commit()
    session.refresh(fact)
    index_person_fact(session, fact)
    session.commit()
    return fact


CENSUS_EXTRACTION_SYSTEM_PROMPT = """You are an expert Slovak archivist and paleographer transcribing 1930 and 1940 Slovak Census sheets (Sčítací hárok / Popisný hárok, Národný archív SR / Slovakiana).
The table has 24 columns mapped to these exact JSON keys:
1: Byt č. (`apartment_number`: dwelling unit / flat number: "1", "2", "I", "II")
2: Poradové číslo osoby (`person_row_number`: person row number "1".."15")
3: Priezvisko a rodné priezvisko u vydatých žien (`surname_raw`: married/current surname; `maiden_surname`: maiden surname if present after "rod." or null)
4: Krstné meno (`given_name_raw`: given name in standard Slovak orthography)
5: Pomer k hlave domácnosti (`relation_to_head`: relation to head of household — normalize to standard Slovak kinship terms:
   - Head: "hlava domácnosti" ("hlava", "majiteľ bytu", "majitel dom.", "prednosta", "hosp.", "hospodár")
   - Spouse/Partner: "manželka" ("manž.", "žena", "manzelka"), "manžel" ("muž", "manzel"), "družka" ("druzka", "konkubína"), "druh"
   - Children: "syn", "dcéra" ("dcera"), "nevlastný syn" ("pastorok", "nevlastny syn"), "nevlastná dcéra" ("pastorkyňa", "nevlastna dcera"), "adoptívny syn", "adoptívna dcéra", "chovanec" ("chovanica", "chovanka")
   - Parents & In-laws: "matka" ("mať", "mati"), "otec", "svokra" ("matka manželky", "matka manžela", "tchyně", "testiná"), "svokor" ("otec manželky", "otec manžela", "tchán", "test"), "svatka", "svat", "zať" ("zat"), "nevesta" ("snacha")
   - Grandchildren & Grandparents: "vnuk", "vnučka" ("vnucka"), "pravnuk", "pravnučka", "stará matka" ("babka", "stara matka"), "starý otec" ("dedo", "stary otec")
   - Extended family: "brat", "sestra", "švagor" ("svagor"), "švagriná" ("svagrina"), "strýko" ("stryko", "stryc"), "stryná" ("stryna"), "ujo" ("ujec"), "teta" ("tetka"), "synovec", "neter" ("netere"), "bratranec", "sesternica", "príbuzný" ("pribuzny"), "príbuzná" ("pribuzna")
   - Household staff & apprentices: "slúžka" ("sluzka", "služobná"), "sluha" ("čeľadník", "celadnik", "paholok"), "gazdiná" ("gazdina"), "pestúnka", "učeník" ("ucenik"), "tovariš" ("tovaris"), "pomocník" ("pomocnik")
   - Lodgers & guests: "podnájomník" ("podnajomnik", "nájomník"), "podnájomníčka" ("podnajomnicka"), "bývajúci" ("byvajuci", "spolubývajúci"), "hosť" ("host"))
6: Pohlavie (`sex`: "muž." -> "M", "žen." -> "F")
7: Dátum narodenia (`birth_date_raw`: day, month, year as written on the row; `birth_year`: integer 4-digit year or null)
8: Rodinný stav (`marital_status`: "slobodný", "slobodná", "ženatý", "vydatá", "vdovec", "vdova", "rozvedený")
9-10: Dátum sobáša (`marriage_date_raw`), počet živo narodených detí (`children_born_count`) a z nich zomrelých (`children_dead_count`)
11-12: Rodisko structured into 4 flat keys:
   - `birth_place_municipality` (a) Obec — municipality of birth)
   - `birth_place_district` (b) Politický okres — district)
   - `birth_place_country` (c) Krajina / štát — country/province, e.g. "Slovensko")
   - `birth_place_raw` (combined `Obec, okr. Okres, Krajina`)
13: Prisťahoval sa kedy a odkiaľ (`moved_in_date`, `previous_residence`)
14: Domovská príslušnosť (`home_jurisdiction`: municipality/district) and Štátna príslušnosť (`citizenship`: e.g. "Československá", "ČSR")
15: Národnosť (`nationality`: e.g. "slovenská", "maďarská", "nemecká", "rusínska", "židovská")
16: Náboženstvo (`religion`: e.g. "rím. kat.", "ev. a. v.", "gr. kat.", "ref.", "izr.")
17: Znalosť čítania a písania (`literacy`: e.g. "číta a píše")
18-20: Povolanie (`occupation`), postavenie v povolaní (`occupation_position`: e.g. "samostatný", "pomáhajúci člen", "nádenník"), závod/zamestnávateľ (`employer_workplace`)
21-23: Prítomnosť (`presence_type`: "trvale", "dočasne prítomný", "dočasne neprítomný")
24: Poznámka (`notes_raw`: physical defects / Byt č. II / marriage notes)
1930/1940 SLOVAK CURSIVE PALEOGRAPHY RULES:
- Inspect the attached row image carefully when provided and cross-check against the noisy OCR text.
- Cursive capital `T` with a looped top crossbar is often misread by OCR as `S`, `F`, or `Tr`, and feminine suffix `-ová` is often misread as `-orá`, `-ora`, or `-urá`. Always cross-check Row 1's surname against the rest of the household.
- Cursive `Št-` at the start of a given name (where the `Š` caron + `t` loop looks like `H` or `Pl`) is often misread by OCR as `H-` or `St-`.
- Cursive looped capital `I-` is often misread by OCR as `St-`, and `M-` as `H-`.
- Cursive `-nská` (`ns` ligature + `-ká`) is often misread as `-nha'` or `-nhová`, and `nz` is often misread as `ur`.
- Always use proper Slovak given-name and surname orthography with diacritics.
CRITICAL BLANK / UNINHABITED ROW RULE:
- If the row image or OCR text is blank, crossed out, or contains only printed form sub-labels (`a)`, `b)`, `c)`) or a general house note such as `"Neobývaný dom."` with NO handwritten person entry in Columns 2–4, you MUST return `null` for `"surname_raw"`, `"given_name_raw"`, `"relation_to_head"`, `"birth_date_raw"`, `"birth_year"`, and all person fields! NEVER invent, guess, or hallucinate a person on a blank row!
IMPORTANT: If a cell contains ditto marks (`//`, `"`, `„`, `- " -`, `dtto`, `taktiež`), output `"//"` for that subfield so it can inherit from the previous row. Never confuse the person's surname with their birthplace!
REQUIRED JSON SCHEMA KEYS (return ONLY a single flat raw JSON object with these exact keys, no markdown or explanation):
["apartment_number", "person_row_number", "surname_raw", "maiden_surname", "given_name_raw", "relation_to_head", "sex", "birth_date_raw", "birth_year", "marital_status", "marriage_date_raw", "children_born_count", "children_dead_count", "birth_place_raw", "birth_place_municipality", "birth_place_district", "birth_place_country", "citizenship", "moved_in_date", "previous_residence", "home_jurisdiction", "nationality", "religion", "literacy", "occupation", "occupation_position", "employer_workplace", "presence_type", "notes_raw"]"""


_CANONICAL_RELATIONS: set[str] = set(_RELATION_MAP.values())

_FEMALE_RELATIONS: set[str] = {
    "manželka",
    "družka",
    "dcéra",
    "nevlastná dcéra",
    "adoptívna dcéra",
    "matka",
    "svokra",
    "svatka",
    "nevesta",
    "vnučka",
    "pravnučka",
    "stará matka",
    "sestra",
    "švagriná",
    "stryná",
    "teta",
    "neter",
    "sesternica",
    "príbuzná",
    "slúžka",
    "gazdiná",
    "pestúnka",
    "podnájomníčka",
}

_MALE_RELATIONS: set[str] = {
    "manžel",
    "druh",
    "syn",
    "nevlastný syn",
    "adoptívny syn",
    "otec",
    "svokor",
    "svat",
    "zať",
    "vnuk",
    "pravnuk",
    "starý otec",
    "brat",
    "švagor",
    "strýko",
    "ujo",
    "synovec",
    "bratranec",
    "príbuzný",
    "sluha",
    "učeník",
    "tovariš",
    "pomocník",
    "podnájomník",
}

_HEAD_RELATIONS: set[str] = {
    "hlava domácnosti",
    "prednosta domácnosti",
    "hlava",
    "majiteľ domu",
    "majiteľ bytu",
    "majiteľ bytu (hlava)",
    "majiteľ domu (hlava)",
    "hospodár",
}


_DITTO_TOKEN_RE = re.compile(
    r'^(?://+|"+|„+|”+|-\s*["„”//]+\s*-|—\s*["„”//]+\s*—|dtto\.?|detto\.?|taktiež|taktiez|idem|id\.?)$',
    re.IGNORECASE,
)


def _is_census_ditto(val: str | None) -> bool:
    if not val:
        return False
    s = re.sub(r"\s+", " ", str(val)).strip()
    if not s:
        return False
    if _DITTO_TOKEN_RE.match(s):
        return True
    if s in ('"', "„", "”", "//", "///", '-"-', '- " -', "—\"—", "- // -"):
        return True
    return False


def _clean_place_subfield(val: str | None) -> str | None:
    if not val:
        return None
    s = re.sub(r"^\s*(?:\([a-d]\)|[a-d]\))\s*", "", str(val), flags=re.IGNORECASE)
    s = re.sub(r"\s*(?:\([a-d]\)|[a-d]\))\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip(" .,-;:/")
    if not s:
        return None
    if _is_census_ditto(s):
        return "//"
    if re.fullmatch(r"[\d\s./-]+", s):
        return None
    if len(s) < 2 or s.lower() in ("xin", "x", "okr", "okres"):
        return None
    return s


def _parse_birthplace_subfields(
    raw_text: str | None,
    *,
    sheet: CensusSheet | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Parses a census Rodisko string (which may contain a) Obec, b) Okres, c) Krajina
    or comma/slash-separated parts) into (municipality, district, country).
    """
    if not raw_text:
        return None, None, None
    txt = str(raw_text).strip()
    if not txt:
        return None, None, None
    if _is_census_ditto(txt):
        return "//", "//", "//"

    muni: str | None = None
    dist: str | None = None
    country: str | None = None

    m_a = re.search(r"(?:^|\b)a\)\s*(.+?)(?=\s+[bc]\)|[,;\n]|$)", txt, flags=re.IGNORECASE)
    m_b = re.search(r"(?:^|\b)b\)\s*(.+?)(?=\s+c\)|[,;\n]|$)", txt, flags=re.IGNORECASE)
    m_c = re.search(r"(?:^|\b)c\)\s*(.+?)(?=[,;\n]|$)", txt, flags=re.IGNORECASE)
    if m_a or m_b or m_c:
        muni = _clean_place_subfield(m_a.group(1)) if m_a else None
        dist = _clean_place_subfield(m_b.group(1)) if m_b else None
        country = _clean_place_subfield(m_c.group(1)) if m_c else None
        return muni, dist, country

    # Also support trailing "(a)", "(b)", "(c)" markers (e.g. "Vel' Surovee (a)")
    m_ta = re.search(r"^(.+?)\s*\(a\)\s*$", txt, flags=re.IGNORECASE)
    if m_ta:
        return _clean_place_subfield(m_ta.group(1)), None, None

    # Check for "Obec, okr. Okres, Krajina"
    m_okr = re.search(r"\bokr(?:es)?\.?\s*([^,;]+)", txt, flags=re.IGNORECASE)
    if m_okr:
        dist = _clean_place_subfield(m_okr.group(1))
        txt_wo_okr = (txt[: m_okr.start()] + " " + txt[m_okr.end() :]).strip(" ,;-")
        parts = [_clean_place_subfield(p) for p in re.split(r"[,;/]+", txt_wo_okr)]
        parts = [p for p in parts if p]
        if parts:
            muni = parts[0]
        if len(parts) >= 2:
            country = parts[-1]
        return muni, dist, country

    parts = [_clean_place_subfield(p) for p in re.split(r"[,;]+", txt)]
    parts = [p for p in parts if p]
    if len(parts) == 1:
        muni = parts[0]
    elif len(parts) == 2:
        muni, dist = parts[0], parts[1]
        if dist and dist.lower() in ("slovensko", "slovenská", "slovenska", "čsr", "česko-slovensko", "uhorsko", "maďarsko"):
            country = dist
            dist = sheet.district if sheet else None
    elif len(parts) >= 3:
        muni, dist, country = parts[0], parts[1], parts[2]

    return muni, dist, country


def _extract_birthplace_from_right_ocr(
    raw_right: str | None,
    *,
    sheet: CensusSheet | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Extracts and normalizes `(birth_place_municipality, birth_place_district, birth_place_country)`
    directly from `row.raw_ocr_right` (`Cols 11-22`), handling:
    - Standalone pipe-separated rubric markers (`a) | Hrr. Cepen | b) | Tinava | c) | Moreušá`)
    - Blank/ditto `a)` before `b)` (`(a) |  | \\n (b) | Thranna | \\n (c) | Novenha |` -> `muni_raw = "//"`)
    - Inline rubric markers (`a) Hrv. Čepen | b) Tjunava | c) Slovenská` or `Vel' Surovee (a)`)
    - Unlabeled 3-cell birthplace sequences at the start of Col 11 (`Jepen Horni | Truova | Slovenska`)
    """
    if not raw_right:
        return None, None, None

    raw_cells: list[str] = []
    for line in raw_right.splitlines():
        if is_header_or_garbage(line):
            continue
        line_cells: list[str] = []
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs and cs not in ("[ZAČIERNENÉ / REDACTED 1940]", "-"):
                line_cells.append(cs)
        if not line_cells:
            continue
        # Detect when a line has `(a) |  |` (standalone label `a)` with an empty value cell on that line)
        if len(line_cells) == 1 and re.fullmatch(r"\(?[a-c][\)\.,:]?", line_cells[0], flags=re.IGNORECASE):
            letter = re.sub(r"[^a-cA-C]", "", line_cells[0]).lower()
            raw_cells.append(f"{letter}) //")
        else:
            raw_cells.extend(line_cells)

    if not raw_cells:
        return None, None, None

    # Merge standalone rubric marker cells (`a)`, `b)`, `c)`) with the immediately following value cell,
    # NEVER merging `a)` with `b)` when `a)` was followed by an empty cell!
    merged_cells: list[str] = []
    i = 0
    while i < len(raw_cells):
        cur = raw_cells[i]
        if re.fullmatch(r"\(?[a-c][\)\.,:]?", cur, flags=re.IGNORECASE):
            letter = re.sub(r"[^a-cA-C]", "", cur).lower()
            if i + 1 < len(raw_cells) and not re.fullmatch(
                r"\(?[a-cd][\)\.,:]?", raw_cells[i + 1], flags=re.IGNORECASE
            ):
                merged_cells.append(f"{letter}) {raw_cells[i + 1]}")
                i += 2
            else:
                merged_cells.append(f"{letter}) //")
                i += 1
        else:
            merged_cells.append(cur)
            i += 1

    muni_raw: str | None = None
    dist_raw: str | None = None
    country_raw: str | None = None

    # Pass 1: Look for explicit `a)`, `b)`, `c)` or `(a)` cells before citizenship/religion/occupation
    for idx, c in enumerate(merged_cells):
        low = c.lower()
        # Stop scanning birthplace once we hit religion or literacy
        if normalize_religion(c) or any(lit in low for lit in ("číta", "cita", "píše", "pise")):
            break
        if re.match(r"^\(?a[\)\.,:]?\s*(?://|dtto|dto|\"|„)?\s*$", c, flags=re.IGNORECASE):
            if not muni_raw:
                muni_raw = "//"
            continue
        if re.search(r"(?:^|\b)[a-c]\)\s*\S+", c, flags=re.IGNORECASE):
            m_m, m_d, m_c = _parse_birthplace_subfields(c, sheet=sheet)
            if m_m and not muni_raw:
                muni_raw = m_m
            if m_d and not dist_raw:
                dist_raw = m_d
            if m_c and not country_raw:
                country_raw = m_c
        elif re.search(r"\(a\)\s*$", c, flags=re.IGNORECASE) and not muni_raw:
            muni_raw = _clean_place_subfield(c)

    # If `b)` or `c)` was explicitly provided while `a)` was blank/omitted, treat `a)` as ditto `"//"`
    # so Pass 2 NEVER shifts `(b) Thranna` into `a) Obec`!
    if not muni_raw and (dist_raw or country_raw):
        muni_raw = "//"

    # Pass 2: If no explicit `a)`/`b)`/`c)` marker was found, inspect the first 1-3 cells at the start of Col 11
    if not muni_raw:
        pre_relig_cells: list[str] = []
        for c in merged_cells[:6]:
            low = c.lower()
            if normalize_religion(c) or any(lit in low for lit in ("číta", "cita", "píše", "pise")):
                break
            if low in ("čsr", "čs", "čs.", "čsl.", "čsl", "češl.", "češl", "čh", "cil.", "ohl."):
                break
            # Skip moved-in date cells like "1. nov. 1911" or "r. 1919"
            _, c_yr = validate_census_date(c)
            if c_yr or re.search(r"\b1[89]\d{2}\b", c):
                continue
            pre_relig_cells.append(c)

        if pre_relig_cells:
            muni_raw = _clean_place_subfield(pre_relig_cells[0])
            if len(pre_relig_cells) >= 2 and not dist_raw:
                dist_raw = _clean_place_subfield(pre_relig_cells[1])
            if len(pre_relig_cells) >= 3 and not country_raw:
                country_raw = _clean_place_subfield(pre_relig_cells[2])

    sheet_muni = sheet.municipality if sheet else None
    sheet_dist = sheet.district if sheet else None
    if muni_raw == "//":
        _, norm_dist, norm_cnt = normalize_census_birthplace_fields(
            None,
            dist_raw,
            country_raw,
            sheet_municipality=sheet_muni,
            sheet_district=sheet_dist,
        )
        return "//", norm_dist, norm_cnt

    return normalize_census_birthplace_fields(
        muni_raw,
        dist_raw,
        country_raw,
        sheet_municipality=sheet_muni,
        sheet_district=sheet_dist,
    )


def _format_birth_place_raw(
    muni: str | None,
    dist: str | None,
    country: str | None,
    fallback_raw: str | None = None,
) -> str | None:
    clean_m = muni.strip() if muni and not _is_census_ditto(muni) else None
    clean_d = dist.strip() if dist and not _is_census_ditto(dist) else None
    clean_c = country.strip() if country and not _is_census_ditto(country) else None
    if clean_m and clean_d and clean_c:
        return f"{clean_m}, okr. {clean_d}, {clean_c}"
    if clean_m and clean_d:
        return f"{clean_m}, okr. {clean_d}"
    if clean_m and clean_c:
        return f"{clean_m}, {clean_c}"
    if clean_m:
        return clean_m
    if fallback_raw and not _is_census_ditto(fallback_raw):
        return fallback_raw.strip()
    return None


def _heuristic_fallback_census_extract(
    raw_left: str,
    raw_right: str,
    census_year: int = 1930,
    sheet: CensusSheet | None = None,
    row_index: int = 1,
    prev_fact: PersonFact | None = None,
) -> dict[str, Any]:
    """
    Deterministic parser for 1930 and 1940 Slovak census pipe-separated OCR rows.
    Handles both full 1930 sheets and partially-redacted 1940 sheets.
    NEVER calls normalize_village(c) before surname parsing and NEVER assigns the
    household surname to birth_place_raw.
    """
    left_cells: list[str] = []
    for line in (raw_left or "").splitlines():
        if is_header_or_garbage(line):
            continue
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs and cs not in ("[ZAČIERNENÉ / REDACTED 1940]", "-"):
                left_cells.append(cs)

    right_cells: list[str] = []
    for line in (raw_right or "").splitlines():
        if is_header_or_garbage(line):
            continue
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs and cs not in ("[ZAČIERNENÉ / REDACTED 1940]", "-"):
                right_cells.append(cs)

    cells = left_cells + right_cells

    person_row_number: int | None = row_index
    surname_raw: str | None = None
    maiden_surname: str | None = None
    given_name_raw: str | None = None
    relation_to_head: str | None = None
    sex: str | None = None
    birth_date_raw: str | None = None
    birth_year_est: int | None = None
    marital_status: str | None = None
    marriage_date_raw: str | None = None
    children_born_count: int | None = None
    children_dead_count: int | None = None
    birth_place_municipality: str | None = None
    birth_place_district: str | None = None
    birth_place_country: str | None = None
    birth_place_raw: str | None = None
    moved_in_date: str | None = None
    previous_residence: str | None = None
    home_jurisdiction: str | None = None
    citizenship: str | None = None
    nationality: str | None = None
    religion_raw: str | None = None
    literacy: str | None = None
    occupation_raw: str | None = None
    occupation_position: str | None = None
    employer_workplace: str | None = None
    presence_type: str | None = "trvale"
    observationes_raw: str | None = None
    surname_was_inherited: bool = False

    for idx_c, c in enumerate(cells):
        in_right_half = idx_c >= len(left_cells)
        low = c.lower()
        # Row number at start of row (e.g. "1", "2.", "I. 1.")
        if not surname_raw and not given_name_raw and re.fullmatch(r"(?:[IVX]+\.?\s*)?\d{1,2}\.?", c):
            m_num = re.search(r"(\d{1,2})", c)
            if m_num:
                val = int(m_num.group(1))
                if 1 <= val <= 25:
                    person_row_number = val
            continue

        # Ditto mark in left half (for surname or Column 5 relation_to_head such as cursive "33" / "11" / "//") or in right half for birthplace
        if _is_census_ditto(c) or (not in_right_half and (surname_raw or given_name_raw) and c in ("33", "11", '""', "„", '"', "//")):
            if not in_right_half:
                if not surname_raw and prev_fact:
                    surname_raw = prev_fact.surname_raw
                    surname_was_inherited = True
                elif (surname_raw or given_name_raw) and not relation_to_head and prev_fact and prev_fact.relation_to_head:
                    relation_to_head = prev_fact.relation_to_head
            elif in_right_half and not birth_place_raw:
                birth_place_raw = "//"
                birth_place_municipality = "//"
            continue

        # Explicit a) Obec / b) Okres / c) Krajina markers (only in right half before occupation)
        if in_right_half and not occupation_raw and re.search(r"(?:^|\b)[a-c]\)\s*\S+", c, flags=re.IGNORECASE):
            m_muni, m_dist, m_cnt = _parse_birthplace_subfields(c, sheet=sheet)
            if m_muni and not birth_place_municipality:
                birth_place_municipality = m_muni
            if m_dist and not birth_place_district:
                birth_place_district = m_dist
            if m_cnt and not birth_place_country:
                birth_place_country = m_cnt
            continue

        # Relation to head — ONLY in left half (Column 5); right half (Columns 18-20) often mentions "v domácnosti predn. dom." as employer!
        if not in_right_half:
            rel_norm = normalize_census_relation(c)
            if rel_norm in _CANONICAL_RELATIONS and not relation_to_head:
                relation_to_head = rel_norm
                if rel_norm in _FEMALE_RELATIONS:
                    sex = "F"
                elif rel_norm in _MALE_RELATIONS:
                    sex = "M"
                continue

        # Sex token
        if low in (
            "m", "muž", "muž.", "mužské", "munrle", "mutsle", "muríle", "marshé", "nutre",
            "f", "ž", "žen", "žen.", "ženské", "jevrhá", "jevrié", "jevrie", "jeudre", "jurá", "jura",
        ):
            sex = "F" if low.startswith(("ž", "f", "jev", "jeu", "jur")) else "M"
            continue

        # Date of birth (e.g. "15. III. 1892" or "12.5.1904" or "1892")
        c_date, c_year = validate_census_date(c)
        if c_year and not birth_year_est and 1825 <= c_year <= census_year:
            birth_date_raw = c_date
            birth_year_est = c_year
            continue

        # Marital status
        if any(
            ms in low
            for ms in ("slobodn", "ženat", "vydat", "vdov", "rozved", "rozlúč")
        ) and not marital_status:
            marital_status = c
            continue

        # Citizenship (Štátna príslušnosť)
        if low in ("čsr", "čs", "čs.", "československá", "čsl.", "čsl", "češl.", "češl", "slovenská republika", "sr") and not citizenship:
            citizenship = "Československá" if "č" in low else c
            continue

        # Religion
        relig = normalize_religion(c)
        if relig and not religion_raw:
            religion_raw = "rím.-kat." if relig == "rím. kat." else relig
            continue

        # Nationality
        if any(
            nat in low
            for nat in ("slovensk", "maďarsk", "nemeck", "rusínsk", "česk", "židovsk", "rómsk", "cigánsk")
        ) and not nationality:
            nationality = "slovenská" if "slovensk" in low else c
            continue

        # Literacy
        if any(lit in low for lit in ("číta a píše", "cita a pise", "číta aj píše", "čísla a píso", "negramot", "len číta")) and not literacy:
            literacy = "číta aj píše" if ("píše" in low or "pise" in low or "píso" in low) else c
            continue

        # Occupation
        occ = normalize_occupation(c)
        if occ and not occupation_raw:
            occupation_raw = occ
            continue

        # Name / Maiden surname heuristics ("Kováčová rod. Vargová") — ONLY in left half!
        if not in_right_half:
            married_s, maiden_s = extract_maiden_surname(c)
            if maiden_s and not maiden_surname:
                maiden_surname = maiden_s
                if married_s and not surname_raw:
                    surname_raw = married_s
                continue

            tokens = [t.strip(".,;:()") for t in c.split() if t.strip(".,;:()")]
            if tokens and not surname_raw and not given_name_raw:
                if len(tokens) >= 2 and is_known_given_name(tokens[-1]):
                    surname_raw = " ".join(tokens[:-1])
                    given_name_raw = tokens[-1]
                    continue
                elif len(tokens) >= 2 and is_known_given_name(tokens[0]):
                    given_name_raw = tokens[0]
                    surname_raw = " ".join(tokens[1:])
                    continue
                elif len(tokens) == 1:
                    if is_known_given_name(tokens[0]):
                        given_name_raw = tokens[0]
                    elif tokens[0][0].isupper() and len(tokens[0]) >= 2 and not any(ch.isdigit() for ch in tokens[0]):
                        surname_raw = tokens[0]
                    continue
            elif tokens and surname_raw and not given_name_raw:
                t0_low = tokens[0].lower()
                if is_known_given_name(tokens[0]):
                    given_name_raw = " ".join(tokens)
                    continue
                elif t0_low in ("zaw", "zan", "jau", "jan", "zaan"):
                    given_name_raw = "Ján"
                    continue
                elif t0_low in ("doref", "joref", "jozef"):
                    given_name_raw = "Jozef"
                    continue
                elif t0_low in ("manta", "marta"):
                    given_name_raw = "Marta"
                    continue
                elif t0_low in ("marina", "maria", "mária"):
                    given_name_raw = "Mária"
                    continue
                elif t0_low in ("ana", "anna"):
                    given_name_raw = "Anna"
                    continue
                elif t0_low in ("botela", "botella", "etela", "estela"):
                    given_name_raw = "Estela"
                    continue
                elif t0_low in ("aquac", "ignac", "ignác"):
                    given_name_raw = "Ignác"
                    continue
                elif len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) >= 3:
                    given_name_raw = tokens[0].capitalize() if tokens[0].islower() else tokens[0]
                    continue

        # Right-half birthplace cell (never assign the household surname to birth_place_raw!)
        if in_right_half and not birth_place_municipality:
            clean_bp = re.sub(r"^[a-c]\)\s*", "", c, flags=re.IGNORECASE).strip()
            surname_stem = (strip_slovak_feminine_suffix(surname_raw) or "").lower() if surname_raw else ""
            prev_stem = (strip_slovak_feminine_suffix(prev_fact.surname_raw) or "").lower() if (prev_fact and prev_fact.surname_raw) else ""
            bp_stem = (strip_slovak_feminine_suffix(clean_bp) or "").lower()
            is_household_surname = bool(
                clean_bp
                and (
                    (surname_raw and clean_bp.lower() == surname_raw.lower())
                    or (surname_stem and bp_stem == surname_stem)
                    or (prev_stem and bp_stem == prev_stem)
                )
            )
            if (
                clean_bp
                and len(clean_bp) >= 3
                and clean_bp[0].isupper()
                and not is_known_given_name(clean_bp)
                and not is_household_surname
            ):
                m_muni, m_dist, m_cnt = _parse_birthplace_subfields(c, sheet=sheet)
                birth_place_municipality = m_muni or clean_bp
                birth_place_district = birth_place_district or m_dist
                birth_place_country = birth_place_country or m_cnt
                continue

    if surname_raw and not sex and surname_raw.lower().endswith(("ová", "ova", "orá", "ora", "ná", "na", "ká", "ka")):
        sex = "F"

    sex = validate_sex(sex, given_name_raw)

    if relation_to_head in ("syn", "dcéra") and sex in ("M", "F"):
        relation_to_head = "dcéra" if sex == "F" else "syn"

    if not relation_to_head:
        if row_index == 1:
            relation_to_head = "hlava domácnosti"
        elif prev_fact:
            if row_index == 2 and sex == "F" and (
                not birth_year_est
                or not prev_fact.birth_year_est
                or abs(birth_year_est - prev_fact.birth_year_est) <= 22
            ):
                relation_to_head = "manželka"
            elif birth_year_est and prev_fact.birth_year_est and birth_year_est >= prev_fact.birth_year_est + 15:
                relation_to_head = "dcéra" if sex == "F" else "syn"
            elif prev_fact.relation_to_head in ("syn", "dcéra") and (
                not birth_year_est
                or not prev_fact.birth_year_est
                or abs(birth_year_est - prev_fact.birth_year_est) <= 20
            ):
                relation_to_head = "dcéra" if sex == "F" else "syn"

    # Inherit surname from previous household member when relation is wife/child and surname was ditto/omitted
    if not surname_raw and prev_fact and relation_to_head in ("manželka", "syn", "dcéra"):
        surname_raw = prev_fact.surname_raw
        surname_was_inherited = True
    if surname_raw:
        surname_raw = sanitize_corrupted_surname_suffix(surname_raw) or surname_raw
        stem_fb = _surname_stem(surname_raw)
        if stem_fb:
            if surname_was_inherited:
                prev_has_ova = _has_ocr_feminine_ova_ending(
                    prev_fact.surname_raw if prev_fact else surname_raw,
                    stem_fb,
                )
                surname_raw = _inflect_census_surname(
                    stem_fb,
                    sex,
                    relation_to_head,
                    raw_ocr_surname=None,
                    force_ova_on_ditto=prev_has_ova,
                )
            else:
                surname_raw = _inflect_census_surname(
                    stem_fb,
                    sex,
                    relation_to_head,
                    raw_ocr_surname=surname_raw,
                )

    # Extract and normalize Rodisko (a) Obec, b) Okres, c) Krajina using the ~100km regional gazetteer
    ocr_muni, ocr_dist, ocr_cnt = _extract_birthplace_from_right_ocr(raw_right, sheet=sheet)
    if ocr_muni == "//":
        if prev_fact and (prev_fact.birth_place_municipality or prev_fact.birth_place_sk):
            birth_place_municipality = prev_fact.birth_place_municipality or prev_fact.birth_place_sk
        else:
            birth_place_municipality = sheet.municipality if (sheet and row_index == 1) else "//"
    elif ocr_muni and (not birth_place_municipality or birth_place_municipality != "//"):
        birth_place_municipality = ocr_muni
    if ocr_dist and not birth_place_district:
        birth_place_district = ocr_dist
    if ocr_cnt and not birth_place_country:
        birth_place_country = ocr_cnt

    if birth_place_municipality and birth_place_municipality != "//":
        birth_place_municipality, birth_place_district, birth_place_country = normalize_census_birthplace_fields(
            birth_place_municipality,
            birth_place_district,
            birth_place_country,
            sheet_municipality=sheet.municipality if sheet else None,
            sheet_district=sheet.district if sheet else None,
        )

    if not birth_place_raw or birth_place_raw != "//":
        birth_place_raw = _format_birth_place_raw(
            birth_place_municipality,
            birth_place_district,
            birth_place_country,
        )

    village_raw = sheet.municipality if sheet else None
    house_number = sheet.house_number if sheet else None

    return {
        "person_row_number": person_row_number,
        "surname_raw": surname_raw,
        "maiden_surname": maiden_surname,
        "given_name_raw": given_name_raw,
        "relation_to_head": relation_to_head,
        "sex": sex,
        "birth_date_raw": birth_date_raw,
        "birth_year_est": birth_year_est,
        "marital_status": marital_status,
        "marriage_date_raw": marriage_date_raw,
        "children_born_count": children_born_count,
        "children_dead_count": children_dead_count,
        "birth_place_municipality": birth_place_municipality,
        "birth_place_district": birth_place_district,
        "birth_place_country": birth_place_country,
        "birth_place_raw": birth_place_raw,
        "moved_in_date": moved_in_date,
        "previous_residence": previous_residence,
        "home_jurisdiction": home_jurisdiction,
        "citizenship": citizenship,
        "nationality": nationality,
        "religion_raw": religion_raw,
        "literacy": literacy,
        "occupation_raw": occupation_raw,
        "occupation_position": occupation_position,
        "employer_workplace": employer_workplace,
        "presence_type": presence_type,
        "observationes_raw": observationes_raw,
        "village_raw": village_raw,
        "house_number": house_number,
    }


_PROMPT_EXAMPLE_NAMES_FOLDED: set[str] = {
    "kovac",
    "kovacova",
    "vargova",
    "varga",
    "horvath",
    "horvathova",
    "novak",
    "novakova",
    "bystricka",
    "bystricky",
    "kral",
    "kralova",
    "irena",
    "maria",
    "jan",
    "jozef",
    "alzbeta",
    "veronika",
}


def _is_effectively_blank_census_left_ocr(raw_ocr_left: str | None) -> bool:
    """
    Returns True when `raw_ocr_left` contains no meaningful handwritten person entry
    (e.g. only empty cell pipes `|`, printed sub-labels `a)`, `b)`, `c)`, header boilerplate,
    redaction markers, or general house notes like `Neobývaný dom.`).
    """
    if not raw_ocr_left or not raw_ocr_left.strip():
        return True
    cleaned_lines: list[str] = []
    for line in raw_ocr_left.splitlines():
        if is_header_or_garbage(line):
            continue
        # Remove HTML tags, redaction markers, and printed sub-column labels a), b), c)
        s = re.sub(r"<[^>]+>", " ", line)
        s = s.replace("[ZAČIERNENÉ / REDACTED 1940]", " ").replace("[ĽAVÁ STRANA / LEFT]", " ")
        s = re.sub(r"\(\s*[abc]\s*\)|\b[abc]\s*\)", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:neobývaný\s*dom|neobyvany\s*dom|neobývaný|neobyvany)\b", " ", s, flags=re.IGNORECASE)
        cleaned_lines.append(s)
    combined = " ".join(cleaned_lines)
    # Check for any alphabetic token >= 2 letters
    alpha_words = re.findall(r"[A-Za-zÁ-ž]{2,}", combined)
    if alpha_words:
        return False
    return True


def _is_ungrounded_llm_person_name(
    name_val: str | None,
    *,
    raw_ocr_left: str | None,
    prev_fact: PersonFact | None,
    fallback_val: str | None,
    role: str = "surname",
) -> bool:
    """
    Guards against the LLM hallucinating a name on a blank/crossed-out row or copying
    an example name that has zero character overlap with the row's OCR text.
    """
    if not name_val or not str(name_val).strip():
        return False
    s = str(name_val).strip()
    if _is_census_ditto(s):
        return False
    # If the row's left OCR is effectively blank, ANY LLM-invented name is ungrounded
    if _is_effectively_blank_census_left_ocr(raw_ocr_left):
        return True

    s_fold = ascii_fold(s)
    if fallback_val and ascii_fold(fallback_val) == s_fold:
        return False
    if role == "surname" and prev_fact and (prev_fact.surname_raw or prev_fact.surname_normalized):
        prev_s = prev_fact.surname_raw or prev_fact.surname_normalized or ""
        if _paleographic_stem_similarity(s, prev_s) >= 0.65:
            return False

    # Extract alphabetic tokens from raw_ocr_left (excluding printed a), b), c) labels)
    ocr_clean = re.sub(r"\(\s*[abc]\s*\)|\b[abc]\s*\)", " ", raw_ocr_left or "", flags=re.IGNORECASE)
    ocr_tokens = re.findall(r"[A-Za-zÁ-ž]{2,}", ocr_clean)
    if not ocr_tokens:
        return True

    # If the LLM output is one of the common prompt/example names, require paleographic/character overlap with OCR
    s_stem_fold = ascii_fold(strip_slovak_feminine_suffix(s) or s)
    if s_fold in _PROMPT_EXAMPLE_NAMES_FOLDED or s_stem_fold in _PROMPT_EXAMPLE_NAMES_FOLDED:
        best_sim = max(
            (_paleographic_stem_similarity(s, tok) for tok in ocr_tokens),
            default=0.0,
        )
        # Also allow known cursive OCR mappings (e.g. Heter -> Štefan, Havia -> Mária, Zaan/Jaw -> Ján)
        ocr_tokens_fold = {ascii_fold(tok) for tok in ocr_tokens}
        has_known_alias = (
            (s_fold == "stefan" and bool(ocr_tokens_fold & {"heter", "psefan", "plefaw", "siefaw", "srefan", "stefan"}))
            or (s_fold == "maria" and bool(ocr_tokens_fold & {"havia", "marina", "maria"}))
            or (s_fold == "irena" and bool(ocr_tokens_fold & {"steena", "stena", "irena"}))
            or (s_fold == "jan" and bool(ocr_tokens_fold & {"zaan", "zaw", "zan", "jau", "jaw", "jain", "jan"}))
            or (s_fold == "jozef" and bool(ocr_tokens_fold & {"doref", "joref", "josef", "jozef"}))
        )
        if best_sim < 0.42 and not has_known_alias:
            return True

    return False


async def run_census_extraction_for_row(
    session: Session,
    scan: ScanImage,
    row: RecordRow,
    prev_fact: PersonFact | None = None,
) -> PersonFact | None:
    """
    Extracts a single census household row (1930 full or 1940 partially redacted) into a PersonFact.
    Returns None and marks row.status = 'empty_row' if no person name was found.
    """
    sheet = session.get(CensusSheet, scan.census_sheet_id) if scan.census_sheet_id else None
    census_year = sheet.census_year if sheet else (int(scan.detected_year) if (scan.detected_year and scan.detected_year.isdigit()) else 1930)
    is_blackened = bool(sheet and sheet.is_blackened)

    # Automatically resolve prev_fact from previous row of the same scan when not explicitly passed
    if prev_fact is None and row.row_index > 1:
        prev_rows = session.exec(
            select(RecordRow)
            .where(RecordRow.scan_id == scan.id, RecordRow.row_index < row.row_index)
            .order_by(RecordRow.row_index.desc())
        ).all()
        for pr in prev_rows:
            pf = session.exec(select(PersonFact).where(PersonFact.row_id == pr.id)).first()
            if pf is not None:
                prev_fact = pf
                break

    answered_qs = session.exec(
        select(QuestionItem).where(QuestionItem.row_id == row.id, QuestionItem.status == "answered")
    ).all()
    human_notes = "; ".join(
        [f"{q.field_name}: {q.selected_answer}" for q in answered_qs if q.selected_answer]
    )

    left_is_blank = _is_effectively_blank_census_left_ocr(row.raw_ocr_left)
    if left_is_blank and not human_notes:
        row.status = "empty_row"
        row.normalized_json = json.dumps({"status": "empty_row"}, ensure_ascii=False)
        session.add(row)
        for m in session.exec(select(FamilySearchMatch).where(FamilySearchMatch.row_id == row.id)).all():
            session.delete(m)
        for pf in session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).all():
            session.delete(pf)
        session.commit()
        return None

    fallback = _heuristic_fallback_census_extract(
        row.raw_ocr_left or "",
        row.raw_ocr_right or "",
        census_year=census_year,
        sheet=sheet,
        row_index=row.row_index,
        prev_fact=prev_fact,
    )

    data: dict[str, Any] = {}
    if not left_is_blank and (not is_blackened or (row.raw_ocr_left and len(row.raw_ocr_left.strip()) > 5)):
        nearby_villages = get_regional_municipality_gazetteer(
            sheet.municipality if sheet else None,
            sheet.district if sheet else None,
        )[:25]
        user_prompt = (
            f"Transcribe this {census_year} Slovak Census row (Municipality: {sheet.municipality if sheet else 'Unknown'}, "
            f"District: {sheet.district if sheet else 'Unknown'}, "
            f"House: {sheet.house_number if sheet else 'Unknown'}).\n"
            f"Regional ~100km villages in vicinity: {', '.join(nearby_villages)}\n"
            f"Pre-parsed hints: surname={fallback.get('surname_raw')}, given_name={fallback.get('given_name_raw')}, "
            f"relation={fallback.get('relation_to_head')}, birth_date={fallback.get('birth_date_raw')}, "
            f"birth_place_municipality={fallback.get('birth_place_municipality')}, "
            f"birth_place_district={fallback.get('birth_place_district')}, "
            f"birth_place_country={fallback.get('birth_place_country')}, "
            f"occupation={fallback.get('occupation_raw')}\n"
            f"OCR text:\n{row.raw_ocr_combined or ''}"
        )
        if prev_fact and (prev_fact.surname_raw or prev_fact.surname_normalized):
            user_prompt += (
                f"\nHousehold context (previous row): surname={prev_fact.surname_raw or prev_fact.surname_normalized}, "
                f"given_name={prev_fact.given_name_sk or prev_fact.given_name_raw}, "
                f"relation={prev_fact.relation_to_head}, "
                f"birth_place_municipality={prev_fact.birth_place_municipality or prev_fact.birth_place_sk}"
            )
        if human_notes:
            user_prompt += f"\nVerified human correction: {human_notes}"

        extract_image_path, extract_image_rel = await asyncio.to_thread(
            resolve_row_extraction_crop, row
        )

        res = await ai_client.chat_completion(
            stage="extract",
            model=settings.model_extract,
            system_prompt=CENSUS_EXTRACTION_SYSTEM_PROMPT,
            prompt=user_prompt,
            image_path=extract_image_path,
            image_rel_url=extract_image_rel,
            scan_id=scan.id,
            row_id=row.id,
            max_tokens=360,
            temperature=0.0,
            timeout_sec=45.0,
            skip_thinking=True,
            response_model=LLMCensusExtractionSchema,
        )
        data = res.get("parsed") or {}

    raw_s = data.get("surname_raw")
    if _is_ungrounded_llm_person_name(
        raw_s,
        raw_ocr_left=row.raw_ocr_left,
        prev_fact=prev_fact,
        fallback_val=fallback.get("surname_raw"),
        role="surname",
    ):
        raw_s = None
    if _is_census_ditto(raw_s) and prev_fact:
        raw_s = prev_fact.surname_raw
    surname_cand, _ = validate_person_name(raw_s, role="surname")
    surname_raw = surname_cand or fallback.get("surname_raw")
    maiden_cand, _ = validate_person_name(data.get("maiden_surname"), role="surname")
    maiden_surname = maiden_cand or fallback.get("maiden_surname")
    if surname_raw:
        m_sur, maid = extract_maiden_surname(surname_raw)
        if maid and not maiden_surname:
            maiden_surname = maid
            surname_raw = m_sur
        surname_raw = sanitize_corrupted_surname_suffix(surname_raw) or surname_raw

    raw_g = data.get("given_name_raw")
    if _is_ungrounded_llm_person_name(
        raw_g,
        raw_ocr_left=row.raw_ocr_left,
        prev_fact=prev_fact,
        fallback_val=fallback.get("given_name_raw"),
        role="given",
    ):
        raw_g = None
    given_cand, _ = validate_person_name(raw_g, role="given")
    given_name_raw = given_cand or fallback.get("given_name_raw")
    relation_to_head = normalize_census_relation(data.get("relation_to_head")) or fallback.get("relation_to_head")

    # Apply human answers early if they corrected identity_name
    for q in answered_qs:
        if q.selected_answer and q.field_name in ("identity_name", "given_name_raw", "surname_raw"):
            tmp_d = {"given_name_raw": given_name_raw, "surname_raw": surname_raw}
            tmp_d = apply_human_answer_to_fields(tmp_d, q.field_name, q.selected_answer)
            given_name_raw = tmp_d.get("given_name_raw") or given_name_raw
            surname_raw = tmp_d.get("surname_raw") or surname_raw

    # Blank Row Guard: If both given_name_raw and surname_raw are empty / "?" / None, mark empty_row and delete any PersonFact
    clean_given_check = (given_name_raw or "").strip(" ?.-_\"'/,;:")
    clean_surname_check = (surname_raw or "").strip(" ?.-_\"'/,;:")
    if not clean_given_check and not clean_surname_check:
        row.status = "empty_row"
        row.normalized_json = json.dumps({"status": "empty_row"}, ensure_ascii=False)
        session.add(row)
        for m in session.exec(select(FamilySearchMatch).where(FamilySearchMatch.row_id == row.id)).all():
            session.delete(m)
        for pf in session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).all():
            session.delete(pf)
        session.commit()
        return None

    if not surname_raw and prev_fact and (row.has_ditto_marks or relation_to_head in ("manželka", "syn", "dcéra")):
        surname_raw = prev_fact.surname_raw

    surname_norm = strip_slovak_feminine_suffix(surname_raw) if surname_raw else None
    g_sk, g_sex = normalize_given_name(given_name_raw)
    g_hu = g_sk
    sex = validate_sex(data.get("sex") or fallback.get("sex") or g_sex, given_name_raw)

    b_date, b_year = validate_census_date(data.get("birth_date_raw") or fallback.get("birth_date_raw"))
    birth_year_est = data.get("birth_year") or data.get("birth_year_est") or b_year or fallback.get("birth_year_est")

    # Rodisko (a) Obec, (b) Okres, (c) Krajina + Ditto resolution
    bp_muni = data.get("birth_place_municipality") or fallback.get("birth_place_municipality")
    bp_dist = data.get("birth_place_district") or fallback.get("birth_place_district")
    bp_country = data.get("birth_place_country") or fallback.get("birth_place_country")
    raw_bp = data.get("birth_place_raw") or fallback.get("birth_place_raw")

    # Never allow the household surname (e.g. Paraba / Parabová) to be assigned to birthplace!
    surname_stem = (surname_norm or "").lower()
    prev_stem = (strip_slovak_feminine_suffix(prev_fact.surname_raw) or "").lower() if (prev_fact and prev_fact.surname_raw) else ""

    def _reject_surname_as_place(val: str | None) -> str | None:
        if not val:
            return None
        v_clean = val.strip()
        if not v_clean:
            return None
        v_stem = (strip_slovak_feminine_suffix(v_clean) or "").lower()
        if (
            (surname_raw and v_clean.lower() == surname_raw.lower())
            or (surname_stem and v_stem == surname_stem)
            or (prev_stem and v_stem == prev_stem)
        ):
            return None
        return v_clean

    bp_muni = _reject_surname_as_place(bp_muni)
    raw_bp = _reject_surname_as_place(raw_bp)

    if raw_bp and not (bp_muni and bp_dist and bp_country):
        p_m, p_d, p_c = _parse_birthplace_subfields(raw_bp, sheet=sheet)
        bp_muni = bp_muni or _reject_surname_as_place(p_m)
        bp_dist = bp_dist or p_d
        bp_country = bp_country or p_c

    # Check if right OCR explicitly has a blank/ditto `a)` before `b)` (e.g. `(a) |  | \n (b) | Thranna |`)
    ocr_muni_chk, ocr_dist_chk, ocr_cnt_chk = _extract_birthplace_from_right_ocr(
        row.raw_ocr_right, sheet=sheet
    )

    # Ditto inheritance for birth_place_* from prev_fact
    if prev_fact:
        prev_bp_muni = prev_fact.birth_place_municipality or prev_fact.birth_place_sk
        if (
            _is_census_ditto(bp_muni)
            or (_is_census_ditto(raw_bp) and not bp_muni)
            or (ocr_muni_chk == "//" and prev_bp_muni)
        ):
            bp_muni = prev_bp_muni or prev_fact.birth_place_raw
            if ocr_dist_chk and not bp_dist:
                bp_dist = ocr_dist_chk
            if ocr_cnt_chk and not bp_country:
                bp_country = ocr_cnt_chk
        if _is_census_ditto(bp_dist) or (_is_census_ditto(raw_bp) and not bp_dist):
            bp_dist = prev_fact.birth_place_district
        if _is_census_ditto(bp_country) or (_is_census_ditto(raw_bp) and not bp_country):
            bp_country = prev_fact.birth_place_country
        # Children born in the same household whose birthplace was ditto-marked or omitted
        if not bp_muni and (row.has_ditto_marks or relation_to_head in ("syn", "dcéra")):
            bp_muni = prev_fact.birth_place_municipality or (sheet.municipality if sheet else None)
            bp_dist = bp_dist or prev_fact.birth_place_district or (sheet.district if sheet else None)
            bp_country = bp_country or prev_fact.birth_place_country or "Slovensko"

    if bp_dist and _is_census_ditto(bp_dist):
        bp_dist = prev_fact.birth_place_district if prev_fact else None
    if bp_country and _is_census_ditto(bp_country):
        bp_country = prev_fact.birth_place_country if prev_fact else None

    if bp_muni and not _is_census_ditto(bp_muni):
        bp_muni, bp_dist, bp_country = normalize_census_birthplace_fields(
            bp_muni,
            bp_dist,
            bp_country,
            sheet_municipality=sheet.municipality if sheet else None,
            sheet_district=sheet.district if sheet else None,
        )
    else:
        bp_muni = None

    # If birth_place_municipality matches census sheet.municipality and district/country are blank, fill from sheet
    if sheet and bp_muni:
        if ascii_fold(bp_muni) == ascii_fold(sheet.municipality or "") or (
            sheet.municipality and ascii_fold(bp_muni) in ascii_fold(sheet.municipality)
        ):
            if not bp_dist and sheet.district:
                bp_dist = sheet.district
            if not bp_country:
                bp_country = "Slovensko"
    if bp_muni and not bp_country:
        bp_country = "Slovensko"

    birth_place_raw = _format_birth_place_raw(bp_muni, bp_dist, bp_country, raw_bp)
    bp_sk = bp_muni or (normalize_village(birth_place_raw)[0] if birth_place_raw else None) or birth_place_raw

    village_raw = (sheet.municipality if sheet else None) or fallback.get("village_raw")
    v_sk, v_hu, _ = normalize_village(village_raw)
    house_number = (sheet.house_number if sheet else None) or fallback.get("house_number")

    # Ditto inheritance for citizenship, nationality, religion, home_jurisdiction, literacy
    def _resolve_ditto_field(raw_val: str | None, prev_val: str | None, default_val: str | None = None) -> str | None:
        if raw_val and not _is_census_ditto(raw_val):
            return raw_val.strip()
        if prev_val:
            return prev_val
        return default_val

    raw_relig = data.get("religion") or data.get("religion_raw") or fallback.get("religion_raw")
    if _is_census_ditto(raw_relig) and prev_fact:
        religion = prev_fact.religion
    else:
        religion = normalize_religion(raw_relig) or (prev_fact.religion if (prev_fact and not raw_relig and row.row_index > 1) else None)
    if religion in ("rím. kat.", "r. kat.", "rím. kat", "rim. kat."):
        religion = "rím.-kat."

    nationality = _resolve_ditto_field(
        data.get("nationality") or fallback.get("nationality"),
        prev_fact.nationality if prev_fact else None,
        "slovenská" if (sheet and row.row_index == 1 and not is_blackened) else None,
    )
    if nationality and "slovensk" in nationality.lower():
        nationality = "slovenská"

    citizenship = _resolve_ditto_field(
        data.get("citizenship") or fallback.get("citizenship"),
        prev_fact.citizenship if prev_fact else None,
        "Československá" if not is_blackened else None,
    )
    if citizenship and citizenship.lower().strip(".") in ("čsr", "čs", "čsl", "češl", "československá", "československa"):
        citizenship = "Československá"

    default_home_jur = (
        f"{village_raw}, okr. {sheet.district}, Slovensko"
        if (sheet and village_raw and sheet.district and not is_blackened)
        else (village_raw if not is_blackened else None)
    )
    home_jurisdiction = _resolve_ditto_field(
        data.get("home_jurisdiction") or fallback.get("home_jurisdiction"),
        prev_fact.home_jurisdiction if prev_fact else None,
        default_home_jur,
    )
    if (
        home_jurisdiction
        and sheet
        and village_raw
        and sheet.district
        and ascii_fold(home_jurisdiction) == ascii_fold(village_raw)
    ):
        home_jurisdiction = f"{village_raw}, okr. {sheet.district}, Slovensko"

    literacy = _resolve_ditto_field(
        data.get("literacy") or fallback.get("literacy"),
        prev_fact.literacy if prev_fact else None,
    )
    if literacy and any(tok in literacy.lower() for tok in ("číta a píše", "cita a pise", "čísla a píso", "číta aj píše")):
        literacy = "číta aj píše"

    occ_raw = data.get("occupation") or data.get("occupation_raw") or fallback.get("occupation_raw")
    if _is_census_ditto(occ_raw) and prev_fact:
        occ_norm = prev_fact.social_status_or_occupation
    else:
        occ_norm = normalize_occupation(occ_raw) or occ_raw

    occ_pos = _resolve_ditto_field(
        data.get("occupation_position") or fallback.get("occupation_position"),
        prev_fact.occupation_position if (prev_fact and _is_census_ditto(data.get("occupation_position"))) else None,
    )
    emp_work = _resolve_ditto_field(
        data.get("employer_workplace") or fallback.get("employer_workplace"),
        prev_fact.employer_workplace if (prev_fact and _is_census_ditto(data.get("employer_workplace"))) else None,
    )

    merged_dict: dict[str, Any] = {
        "person_row_number": str(row.row_index),
        "surname_raw": surname_raw,
        "surname_normalized": surname_norm or surname_raw,
        "maiden_surname": maiden_surname,
        "given_name_raw": given_name_raw,
        "given_name_sk": g_sk or given_name_raw,
        "given_name_hu": g_hu,
        "relation_to_head": relation_to_head,
        "sex": sex,
        "birth_or_event_date": b_date,
        "birth_year_est": birth_year_est,
        "marital_status": data.get("marital_status") or fallback.get("marital_status"),
        "marriage_date_raw": data.get("marriage_date_raw") or fallback.get("marriage_date_raw"),
        "children_born_count": str(data.get("children_born_count")) if data.get("children_born_count") is not None else fallback.get("children_born_count"),
        "children_dead_count": str(data.get("children_dead_count")) if data.get("children_dead_count") is not None else fallback.get("children_dead_count"),
        "birth_place_municipality": bp_muni,
        "birth_place_district": bp_dist,
        "birth_place_country": bp_country,
        "birth_place_raw": birth_place_raw,
        "birth_place_sk": bp_sk or birth_place_raw,
        "moved_in_date": data.get("moved_in_date") or fallback.get("moved_in_date"),
        "previous_residence": data.get("previous_residence") or fallback.get("previous_residence"),
        "home_jurisdiction": home_jurisdiction,
        "citizenship": citizenship,
        "nationality": nationality,
        "religion": religion,
        "literacy": literacy,
        "social_status_or_occupation": occ_norm,
        "occupation_position": occ_pos,
        "employer_workplace": emp_work,
        "presence_type": data.get("presence_type") or fallback.get("presence_type") or "trvale",
        "observationes_raw": data.get("notes_raw") or data.get("observationes_raw") or fallback.get("observationes_raw"),
        "village_raw": village_raw,
        "village_sk": v_sk or village_raw,
        "village_hu": v_hu,
        "house_number": house_number,
    }

    for q in answered_qs:
        if q.selected_answer:
            merged_dict = apply_human_answer_to_fields(
                merged_dict,
                q.field_name,
                q.selected_answer,
                answered_by=q.answered_by or "Človek",
            )

    row.normalized_json = json.dumps(merged_dict, ensure_ascii=False)
    row.status = "extracted"
    session.add(row)

    fact = session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()
    if not fact:
        fact = PersonFact(row_id=row.id, scan_id=scan.id)

    fact.record_type = "census"
    fact.census_sheet_id = scan.census_sheet_id
    fact.person_row_number = str(merged_dict.get("person_row_number") or row.row_index)
    fact.entry_number = str(fact.person_row_number)
    fact.event_year = int(census_year)
    fact.birth_or_event_date = merged_dict.get("birth_or_event_date")
    fact.birth_year_est = int(merged_dict["birth_year_est"]) if merged_dict.get("birth_year_est") else None
    fact.given_name_raw = merged_dict.get("given_name_raw")
    fact.given_name_sk = merged_dict.get("given_name_sk")
    fact.given_name_hu = merged_dict.get("given_name_hu")
    fact.surname_raw = merged_dict.get("surname_raw")
    fact.surname_normalized = merged_dict.get("surname_normalized")
    fact.sex = merged_dict.get("sex")
    fact.relation_to_head = merged_dict.get("relation_to_head")
    fact.marital_status = merged_dict.get("marital_status")
    fact.marriage_date_raw = merged_dict.get("marriage_date_raw")
    fact.children_born_count = str(merged_dict["children_born_count"]) if merged_dict.get("children_born_count") is not None else None
    fact.children_dead_count = str(merged_dict["children_dead_count"]) if merged_dict.get("children_dead_count") is not None else None
    fact.birth_place_municipality = merged_dict.get("birth_place_municipality")
    fact.birth_place_district = merged_dict.get("birth_place_district")
    fact.birth_place_country = merged_dict.get("birth_place_country")
    fact.birth_place_raw = merged_dict.get("birth_place_raw")
    fact.birth_place_sk = merged_dict.get("birth_place_sk")
    fact.moved_in_date = merged_dict.get("moved_in_date")
    fact.previous_residence = merged_dict.get("previous_residence")
    fact.home_jurisdiction = merged_dict.get("home_jurisdiction")
    fact.citizenship = merged_dict.get("citizenship")
    fact.nationality = merged_dict.get("nationality")
    fact.religion = merged_dict.get("religion")
    fact.literacy = merged_dict.get("literacy")
    fact.social_status_or_occupation = merged_dict.get("social_status_or_occupation")
    fact.occupation_position = merged_dict.get("occupation_position")
    fact.employer_workplace = merged_dict.get("employer_workplace")
    fact.presence_type = merged_dict.get("presence_type")
    fact.observationes_raw = merged_dict.get("observationes_raw")
    fact.village_raw = merged_dict.get("village_raw")
    fact.village_sk = merged_dict.get("village_sk")
    fact.village_hu = merged_dict.get("village_hu")
    fact.house_number = str(merged_dict.get("house_number") or "") or None
    if merged_dict.get("maiden_surname") and fact.relation_to_head in ("manželka", "hlava domácnosti", "matka", "svokra"):
        fact.mother_maiden_surname = merged_dict.get("maiden_surname")

    apply_person_fact_search_fields(fact)

    session.add(fact)
    session.commit()
    session.refresh(fact)
    index_person_fact(session, fact)
    session.commit()
    return fact


_SHARED_HOUSEHOLD_SURNAME_RELATIONS: set[str] = {
    "hlava domácnosti",
    "manželka",
    "manžel",
    "syn",
    "dcéra",
    "vnuk",
    "vnučka",
    "pravnuk",
    "pravnučka",
}


_surname_stem = strip_slovak_feminine_suffix


def _cursive_digraph_fold(folded: str) -> str:
    """
    Applies Slovak/Central European Kurrent/cursive digraph equivalence rules:
    - `nz` <-> `ur`
    - `ni` / `in` <-> `m` (and `mm` -> `m`)
    - `tr` -> `t` at word start
    - cursive initial capital equivalence (`s`/`f` <-> `t` before `om`/`on`, `k` <-> `h`, `b` <-> `p`)
    - final `r` <-> `n` after `a`/`o`/`e`
    - `č` <-> `c` (already folded by ascii_fold)
    """
    res = folded.lower()
    if res.startswith("tr") and len(res) >= 5:
        res = "t" + res[2:]
    res = res.replace("ur", "nz")
    res = res.replace("ni", "m").replace("in", "m")
    res = re.sub(r"([a-z])\1+", r"\1", res)
    if re.match(r"^[sf]om[aeo][nr]$", res):
        res = "t" + res[1:-1] + "n"
    return res


_PALEO_PAIR_COSTS: dict[tuple[str, str], float] = {}
for _a, _b, _cost in [
    ("n", "u", 0.08),
    ("z", "r", 0.08),
    ("n", "r", 0.10),
    ("s", "t", 0.10),
    ("f", "t", 0.12),
    ("h", "k", 0.14),
    ("p", "b", 0.14),
    ("l", "t", 0.12),
    ("c", "k", 0.12),
    ("s", "z", 0.14),
    ("i", "y", 0.12),
    ("i", "j", 0.12),
    ("i", "e", 0.18),
    ("a", "o", 0.16),
    ("e", "o", 0.18),
    ("u", "v", 0.12),
    ("v", "r", 0.12),
    ("m", "n", 0.15),
    ("b", "h", 0.20),
    ("d", "t", 0.18),
    ("g", "q", 0.18),
    ("s", "h", 0.22),
]:
    _PALEO_PAIR_COSTS[(_a, _b)] = _cost
    _PALEO_PAIR_COSTS[(_b, _a)] = _cost


def _weighted_paleo_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = i * 0.75
    for j in range(1, lb + 1):
        dp[0][j] = j * 0.75
    for i in range(1, la + 1):
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            if ca == cb:
                sub_cost = 0.0
            else:
                sub_cost = _PALEO_PAIR_COSTS.get((ca, cb), 0.85)
            dp[i][j] = min(
                dp[i - 1][j] + 0.75,
                dp[i][j - 1] + 0.75,
                dp[i - 1][j - 1] + sub_cost,
            )
    return dp[la][lb]


def _paleographic_stem_similarity(left: str | None, right: str | None) -> float:
    """
    Computes paleographic similarity in [0.0, 1.0] between two surnames or surname stems.
    Models Kurrent/cursive digraph confusions (`nz` <-> `ur`, `n` <-> `u`, `z` <-> `r`,
    `m` <-> `ni`/`in`, `č` <-> `c`, `ľ`/`l` <-> `t`) so cursive misreads of the same stem
    achieve similarity >= 0.85.
    """
    stem_l = _surname_stem(left)
    stem_r = _surname_stem(right)
    if not stem_l or not stem_r:
        return 0.0
    if stem_l == stem_r:
        return 1.0
    if stem_l.lower() == stem_r.lower():
        return 0.99

    f_l = ascii_fold(stem_l)
    f_r = ascii_fold(stem_r)
    if not f_l or not f_r:
        return 0.0
    if f_l == f_r:
        return 0.97

    max_len = max(len(f_l), len(f_r), 1)
    direct_dist = _weighted_paleo_distance(f_l, f_r)
    direct_sim = max(0.0, 1.0 - (direct_dist / max_len))

    dg_l = _cursive_digraph_fold(f_l)
    dg_r = _cursive_digraph_fold(f_r)
    if dg_l == dg_r:
        digraph_sim = 0.95
    else:
        dg_max = max(len(dg_l), len(dg_r), 1)
        dg_dist = _weighted_paleo_distance(dg_l, dg_r)
        digraph_sim = max(0.0, 0.95 - (dg_dist / dg_max))

    return round(max(direct_sim, digraph_sim), 4)


def _has_ocr_feminine_ova_ending(
    raw_ocr_surname: str | None,
    canonical_stem: str | None = None,
) -> bool:
    """
    Returns True if `raw_ocr_surname` as written on the census sheet has a Slovak `-ová`
    suffix (or a known cursive OCR reading of `-ová` such as `-ova`, `-orá`, `-ora`, `-urá`, `-ov'`).
    Returns False when the surname on the scan is written WITHOUT `-ová` (e.g. `Bódi`, `Vinter`,
    `Harsányi`, or an adjectival `-á` surname like `Šťastná`).
    """
    if not raw_ocr_surname:
        return False
    m_sur, _ = extract_maiden_surname(str(raw_ocr_surname).strip())
    s = re.sub(r"^(?:vd\.?|va\.?|vdova|ovd\.?)\s+", "", (m_sur or str(raw_ocr_surname)).strip(), flags=re.IGNORECASE)
    s = re.sub(r"['’`\"„”]+", "", s).strip(" .,;:()/\\")
    if not s or _is_census_ditto(s):
        return False
    low = s.lower()
    low_f = ascii_fold(low)
    stem_f = ascii_fold(canonical_stem or "")
    if stem_f and low_f == stem_f:
        return False
    if low_f in ("vancouver", "vancouverova"):
        return True
    if low.endswith(("ová", "ova", "ovà", "ovä", "ové", "ove", "ovra", "orá", "urá")):
        return True
    if low.endswith(("ora", "ura")) and len(s) >= 5:
        if stem_f and stem_f.endswith(("ora", "ura")) and len(low_f) <= len(stem_f):
            return False
        return True
    if low.endswith(("-ov", "ov")) and len(s) >= 5 and not (stem_f and stem_f.endswith("ov")):
        return True
    return False


def _append_slovak_ova_suffix(stem: str) -> str:
    """
    Appends `-ová` to a Slovak noun-stem surname properly:
    - Never appends `-ová` to adjectival surnames (`Šťastný` -> `Šťastná`, `Bystrický` -> `Bystrická`).
    - Drops final `-a` / `-o` before `-ová` (`Paraba` -> `Parabová`, `Buchta` -> `Buchtová`,
      `Liška` -> `Lišková`, `Vančo` -> `Vančová`, `Šimko` -> `Šimková`).
    - Never creates double suffixes (`*ováová`, `*aová`, `*áová`, `*oová`, `*-ová`).
    """
    clean_stem = sanitize_corrupted_surname_suffix(stem) or stem
    clean_stem = clean_stem.rstrip("-")
    adj_fem = feminize_slovak_adjectival_surname(clean_stem)
    if adj_fem:
        return adj_fem
    low = clean_stem.lower()
    if low.endswith(("ová", "ova")):
        return clean_stem
    if low.endswith(("a", "á", "o", "ó")) and len(clean_stem) >= 4:
        return clean_stem[:-1] + "ová"
    return clean_stem + "ová"


def _inflect_census_surname(
    canonical_stem: str,
    sex: str | None = None,
    relation_to_head: str | None = None,
    raw_ocr_surname: str | None = None,
    *,
    force_ova_on_ditto: bool = False,
) -> str:
    """
    Inflects or preserves a canonical surname for census display without forcefully
    tacking `-ová` onto women's surnames:
    - Male members receive the canonical masculine stem (`Kováč`, `Paraba`, `Šťastný`, `Bystrický`).
    - Female members with an ADJECTIVAL surname (`Šťastný`, `Veselý`, `Bystrický`, `Majerský`, etc.)
      ALWAYS receive the proper feminine adjectival form (`-á` / `-ská` / `-cká`, e.g. `Šťastná`,
      `Bystrická`, `Majerská`), NEVER `-ýová` / `-áová` / `-aová`!
    - Female members with a NOUN-STEM surname:
      - If `raw_ocr_surname` is present on the scan and does NOT have an `-ová` ending
        (e.g. `Bódi`, `Vinter`, `Harsányi`), preserve `stem` WITHOUT `-ová`!
      - Only apply `-ová` (`_append_slovak_ova_suffix(stem)`) when `raw_ocr_surname` actually has
        an `-ová` ending on the scan, or when the row's surname was a ditto/blank inheriting from
        a household where `-ová` is used (`force_ova_on_ditto=True`).
    """
    raw_clean = sanitize_corrupted_surname_suffix(canonical_stem) or (canonical_stem or "").strip()
    stem = (_surname_stem(raw_clean) or raw_clean).strip()
    if not stem:
        return canonical_stem
    rel_low = (relation_to_head or "").strip().lower()
    is_female = (sex == "F") or (rel_low in _FEMALE_RELATIONS and sex != "M")
    if not is_female:
        return stem

    # Rule 1a: Adjectival surnames ALWAYS use -á / -ská / -cká for females, NEVER -ová!
    adj_fem = feminize_slovak_adjectival_surname(stem) or feminize_slovak_adjectival_surname(raw_ocr_surname)
    if adj_fem:
        return adj_fem

    # Rule 1b: For noun-stem surnames, check what was actually written on the scan
    if raw_ocr_surname and not _is_census_ditto(raw_ocr_surname):
        if _has_ocr_feminine_ova_ending(raw_ocr_surname, stem):
            return _append_slovak_ova_suffix(stem)
        # Written surname on the scan has NO -ová -> do NOT forcefully add -ová!
        return stem

    if force_ova_on_ditto:
        return _append_slovak_ova_suffix(stem)
    return stem


def _is_ocr_surname_artifact(stem: str) -> bool:
    """
    Detects phonetic / paleographic OCR artifacts in Slovak surnames such as:
    - `au+r`
    - `hastu` (`Hastua` from `Šťastná`)
    """
    f = ascii_fold(stem)
    if not f:
        return True
    if "aur" in f and f not in ("laurencik", "maurer", "vavro"):
        return True
    if any(pat in f for pat in ("moravecor", "hastu", "opacit")):
        return True
    if any(ch in stem for ch in ("'", "đ", "Đ", "Ś", "ś")):
        return True
    return False


def _record_surname_candidate(
    fact: PersonFact,
    *,
    original_raw: str | None,
    original_normalized: str | None,
    consensus_stem: str,
    inflected_surname: str,
) -> None:
    existing: list[Any] = []
    raw_json = getattr(fact, "surname_candidates_json", None) or "[]"
    try:
        parsed = json.loads(raw_json)
        if isinstance(parsed, list):
            existing = parsed
    except Exception:
        existing = []

    entry = {
        "raw_ocr": original_raw,
        "normalized_ocr": original_normalized,
        "consensus_stem": consensus_stem,
        "resolved_surname": inflected_surname,
        "source": "byt_surname_consensus",
    }
    if not any(isinstance(x, dict) and x.get("raw_ocr") == original_raw and x.get("consensus_stem") == consensus_stem for x in existing):
        existing.append(entry)
    fact.surname_candidates_json = json.dumps(existing, ensure_ascii=False)


def _resolve_byt_surname_consensus(
    byt_facts: list[PersonFact],
    session: Session | None = None,
) -> str | None:
    """
    Resolves surname consensus and paleographic disambiguation within a single Byt (dwelling unit):
    1. Groups all surname stems in `byt_facts` with `_paleographic_stem_similarity >= 0.85`
       into the same surname cluster (e.g. `1x Horváth` + `5x Horvath` -> 1 cluster).
    2. Scores each candidate stem in a cluster by:
       - Known Slovak surname lexicon / verified SQLite DB attestation bonus,
       - Head-of-household (Row 1 of the Byt) clarity bonus,
       - Cluster frequency count (so when no candidate has a lexicon override, 5 vs 1 majority wins!).
    3. Applies the winning canonical surname to all members of that cluster in `byt_facts`,
       inflecting with `-ová` for female members (`Horváth` for male, `Horváthová` for female)
       while preserving the original OCR reading in `surname_candidates_json` (`surname_Candidiates_json`).
    4. Ensures members in the same Byt whose `relation_to_head` implies sharing the head's surname
       (`hlava domácnosti`, `manželka`, `syn`, `dcéra`, `vnuk`, `vnučka`) and whose surname is
       missing, ditto (`//`, `"`), or paleographically close (>= 0.62) to the Byt consensus surname
       automatically receive the Byt consensus surname.
    Returns the primary Byt consensus surname stem (or None).
    """
    if not byt_facts:
        return None

    # Identify head of household in this Byt
    head_fact: PersonFact | None = None
    for f in byt_facts:
        rel = normalize_census_relation(f.relation_to_head) or (f.relation_to_head or "").lower()
        if rel in _HEAD_RELATIONS:
            head_fact = f
            break
    if head_fact is None:
        head_fact = byt_facts[0]

    # Collect (fact_index, fact, stem) for all facts with a non-ditto surname
    fact_stems: list[tuple[int, PersonFact, str]] = []
    for idx, f in enumerate(byt_facts):
        raw_s = f.surname_raw or f.surname_normalized
        if not raw_s or _is_census_ditto(raw_s):
            continue
        st = _surname_stem(raw_s)
        if st:
            fact_stems.append((idx, f, st))

    # Query SQLite for high-confidence surnames attested outside this sheet (scoped to candidate stems)
    db_folded_counts: dict[str, int] = {}
    if session is not None and fact_stems:
        current_sheet_id = byt_facts[0].census_sheet_id
        current_fact_ids = {f.id for f in byt_facts if f.id is not None}
        cand_stems_set = {st for _, _, st in fact_stems}
        for st in list(cand_stems_set):
            st_f = ascii_fold(st)
            if st_f in _KNOWN_SURNAME_BY_FOLDED:
                cand_stems_set.add(_KNOWN_SURNAME_BY_FOLDED[st_f][0])
        cand_folded_set = {ascii_fold(st) for st in cand_stems_set if ascii_fold(st)}
        try:
            from backend.app.pipeline.lexicon import _apply_digraphs

            cand_digraphs = {_apply_digraphs(st) for st in cand_stems_set if _apply_digraphs(st)}
            stmt_other = select(
                PersonFact.surname_normalized,
                PersonFact.surname_raw,
                PersonFact.census_sheet_id,
                PersonFact.id,
            ).where(
                (PersonFact.surname_folded.in_(cand_digraphs | cand_folded_set))  # type: ignore[union-attr]
                | (PersonFact.surname_normalized.in_(cand_stems_set))  # type: ignore[union-attr]
                | (PersonFact.surname_raw.in_(cand_stems_set))  # type: ignore[union-attr]
            )
            for s_norm, s_raw, s_sheet_id, s_id in session.exec(stmt_other).all():
                if s_id in current_fact_ids or (current_sheet_id and s_sheet_id == current_sheet_id):
                    continue
                st = _surname_stem(s_norm or s_raw)
                if st and not _is_ocr_surname_artifact(st):
                    fd = ascii_fold(st)
                    if fd:
                        db_folded_counts[fd] = db_folded_counts.get(fd, 0) + 1
        except Exception:
            pass

    # Build clusters of paleographically similar stems (similarity >= 0.85)
    clusters: list[list[tuple[int, PersonFact, str]]] = []
    for item in fact_stems:
        _, _, st = item
        placed = False
        for cluster in clusters:
            if any(_paleographic_stem_similarity(st, c_st) >= 0.85 for _, _, c_st in cluster):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    def _score_candidate_stem(cand: str, cluster_items: list[tuple[int, PersonFact, str]]) -> float:
        cand_f = ascii_fold(cand)
        # 1. Cluster frequency count (1.0 per matching occurrence in the Byt)
        exact_count = sum(1 for _, _, st in cluster_items if st.lower() == cand.lower())
        folded_count = sum(1 for _, _, st in cluster_items if ascii_fold(st) == cand_f)
        freq_score = float(folded_count) + 0.15 * float(exact_count)

        # 2. Head-of-household (Row 1 of the Byt) clarity bonus (+1.5 so tie goes to Head, while 5 vs 1 non-lexicon goes to majority 5)
        head_bonus = 0.0
        for idx, f, st in cluster_items:
            if (idx == 0 or (head_fact and f.id == head_fact.id)) and ascii_fold(st) == cand_f:
                head_bonus = 1.5
                break

        # 3. Curated Slovak surname lexicon & DB attestation bonus
        lex_bonus = 0.0
        if cand in KNOWN_SLOVAK_SURNAMES:
            lex_bonus += KNOWN_SLOVAK_SURNAMES[cand]
        elif cand_f in _KNOWN_SURNAME_BY_FOLDED:
            lex_bonus += _KNOWN_SURNAME_BY_FOLDED[cand_f][1] - 0.3

        if cand_f in db_folded_counts and not _is_ocr_surname_artifact(cand):
            lex_bonus += min(2.5, 0.8 * db_folded_counts[cand_f])

        # 4. Phonetic / paleographic validity & diacritic precision
        artifact_penalty = -3.5 if _is_ocr_surname_artifact(cand) else 0.0
        diacritic_bonus = 0.35 if any(ord(c) > 127 for c in cand) and not _is_ocr_surname_artifact(cand) else 0.0

        return freq_score + head_bonus + lex_bonus + artifact_penalty + diacritic_bonus

    cluster_winners: list[tuple[str, list[tuple[int, PersonFact, str]], bool]] = []
    for cluster in clusters:
        # Candidate pool: all distinct stems in the cluster + any known lexicon canonical form matching a stem in the cluster
        cand_pool: list[str] = []
        for _, _, st in cluster:
            if st not in cand_pool:
                cand_pool.append(st)
            st_f = ascii_fold(st)
            if st_f in _KNOWN_SURNAME_BY_FOLDED:
                known_canonical = _KNOWN_SURNAME_BY_FOLDED[st_f][0]
                if known_canonical not in cand_pool:
                    cand_pool.append(known_canonical)

        best_stem = max(cand_pool, key=lambda c: _score_candidate_stem(c, cluster))
        has_head = any(idx == 0 or (head_fact and f.id == head_fact.id) for idx, f, _ in cluster)
        cluster_winners.append((best_stem, cluster, has_head))

        # Apply winning canonical surname to all members of this cluster
        cluster_has_ova = any(_has_ocr_feminine_ova_ending(f.surname_raw, best_stem) for _, f, _ in cluster)
        for _, f, orig_st in cluster:
            orig_raw = f.surname_raw
            orig_norm = f.surname_normalized
            inflected = _inflect_census_surname(
                best_stem,
                f.sex,
                f.relation_to_head,
                raw_ocr_surname=orig_raw,
                force_ova_on_ditto=cluster_has_ova,
            )
            if orig_raw != inflected or orig_norm != best_stem:
                _record_surname_candidate(
                    f,
                    original_raw=orig_raw,
                    original_normalized=orig_norm,
                    consensus_stem=best_stem,
                    inflected_surname=inflected,
                )
            f.surname_normalized = best_stem
            f.surname_raw = inflected

    # Determine the primary Byt consensus surname (head's cluster winner, or largest cluster winner)
    byt_consensus_stem: str | None = None
    for winner_stem, cluster, has_head in cluster_winners:
        if has_head:
            byt_consensus_stem = winner_stem
            break
    if not byt_consensus_stem and cluster_winners:
        cluster_winners.sort(key=lambda x: len(x[1]), reverse=True)
        byt_consensus_stem = cluster_winners[0][0]

    # Step 3b: Byt Majority Family Dominance rule:
    # When >= 3 core family members (hlava domácnosti, manželka, manžel, syn, dcéra) in the same Byt
    # share a clear majority surname stem (e.g. 5x Kováč/Kováčová) and 1 core family member
    # (even Row 1 hlava domácnosti!) has an outlier OCR surname (like 1x Kowacs), unify all core
    # family members (including Row 1 hlava domácnosti) to the majority surname!
    # Do NOT overwrite legitimately distinct surnames (e.g. married daughter with different surname)!
    _core_family_relations = {"hlava domácnosti", "manželka", "manžel", "syn", "dcéra"}
    core_stem_counts: dict[str, int] = {}
    for f in byt_facts:
        rel_n = normalize_census_relation(f.relation_to_head) or (f.relation_to_head or "").strip().lower()
        if rel_n in _core_family_relations and f.surname_normalized:
            core_stem_counts[f.surname_normalized] = core_stem_counts.get(f.surname_normalized, 0) + 1
    if core_stem_counts:
        top_stem, top_count = max(core_stem_counts.items(), key=lambda kv: kv[1])
        other_max = max((cnt for st_k, cnt in core_stem_counts.items() if st_k != top_stem), default=0)
        if top_count >= 3 and top_count >= 3 * max(other_max, 1):
            byt_consensus_stem = top_stem
            byt_has_ova = any(_has_ocr_feminine_ova_ending(bf.surname_raw, top_stem) for bf in byt_facts)
            for f in byt_facts:
                rel_n = normalize_census_relation(f.relation_to_head) or (f.relation_to_head or "").strip().lower()
                if rel_n in _core_family_relations and f.surname_normalized != top_stem:
                    cur_raw = f.surname_raw
                    cur_norm = f.surname_normalized
                    is_outlier_ocr = (
                        not cur_norm
                        or _is_ocr_surname_artifact(cur_norm)
                        or _paleographic_stem_similarity(cur_norm, top_stem) >= 0.62
                    )
                    if not is_outlier_ocr or core_stem_counts.get(cur_norm, 0) >= 2:
                        continue
                    inflected = _inflect_census_surname(
                        top_stem,
                        f.sex,
                        rel_n,
                        raw_ocr_surname=cur_raw,
                        force_ova_on_ditto=byt_has_ova,
                    )
                    _record_surname_candidate(
                        f,
                        original_raw=cur_raw,
                        original_normalized=cur_norm,
                        consensus_stem=top_stem,
                        inflected_surname=inflected,
                    )
                    f.surname_normalized = top_stem
                    f.surname_raw = inflected

    # Step 4: Ensure members in the same Byt whose relation_to_head implies sharing the head's surname
    # and whose surname is missing, ditto, or paleographically close (>= 0.62) receive the Byt consensus surname
    if byt_consensus_stem:
        byt_has_ova = any(_has_ocr_feminine_ova_ending(bf.surname_raw, byt_consensus_stem) for bf in byt_facts)
        stem_counts_all: dict[str, int] = {}
        for bf in byt_facts:
            if bf.surname_normalized:
                stem_counts_all[bf.surname_normalized] = stem_counts_all.get(bf.surname_normalized, 0) + 1
        for f in byt_facts:
            rel_norm = normalize_census_relation(f.relation_to_head) or (f.relation_to_head or "").strip().lower()
            if rel_norm in _SHARED_HOUSEHOLD_SURNAME_RELATIONS:
                cur_raw = f.surname_raw
                cur_norm = f.surname_normalized
                is_missing_or_ditto = not cur_raw or _is_census_ditto(cur_raw)
                # Do not overwrite a non-head surname cluster that appears >= 2 times in the Byt (e.g. married daughter + son-in-law + grandchildren)
                if not is_missing_or_ditto and cur_norm and cur_norm != byt_consensus_stem:
                    if stem_counts_all.get(cur_norm, 0) >= 2 and not _is_ocr_surname_artifact(cur_norm):
                        continue
                sim = _paleographic_stem_similarity(cur_norm or cur_raw, byt_consensus_stem) if not is_missing_or_ditto else 1.0
                if is_missing_or_ditto or sim >= 0.62:
                    inflected = _inflect_census_surname(
                        byt_consensus_stem,
                        f.sex,
                        rel_norm,
                        raw_ocr_surname=None if is_missing_or_ditto else cur_raw,
                        force_ova_on_ditto=byt_has_ova,
                    )
                    if cur_raw != inflected or cur_norm != byt_consensus_stem:
                        _record_surname_candidate(
                            f,
                            original_raw=cur_raw,
                            original_normalized=cur_norm,
                            consensus_stem=byt_consensus_stem,
                            inflected_surname=inflected,
                        )
                    f.surname_normalized = byt_consensus_stem
                    f.surname_raw = inflected

    # Update search fields on all facts in the Byt
    for f in byt_facts:
        apply_person_fact_search_fields(f)

    return byt_consensus_stem


_ROMAN_NUMERALS: list[tuple[int, str]] = [
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]
_ROMAN_TO_INT: dict[str, int] = {
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


def _int_to_roman(n: int) -> str:
    if n <= 0:
        return "I"
    res = []
    rem = n
    for val, sym in _ROMAN_NUMERALS:
        while rem >= val:
            res.append(sym)
            rem -= val
    return "".join(res) or "I"


def _detect_explicit_byt_marker(fact: PersonFact, row: RecordRow | None) -> str | None:
    """
    Checks `fact.observationes_raw`, `row.raw_ocr_left`, and `row.raw_ocr_combined`
    for an explicit dwelling unit marker such as:
    - `Byt č. II`, `Byt II`, `II. byt`, `2. byt`
    - Leading Column 1 (`Byt č.`) + Column 2 (`Poradové číslo osoby`) pipe pattern
      e.g. `2 | 7 | Opacity | Anton` or `2 | 4 | Bodi | Jaw` where Column 1 >= 2.
    Returns the Roman numeral (`"II"`, `"III"`, ...) if detected, else None.
    """
    texts = [
        fact.observationes_raw or "",
        (row.raw_ocr_left if row else "") or "",
        (row.raw_ocr_combined if row else "") or "",
    ]
    combined = " \n ".join(texts)

    # Check textual Byt markers: "Byt č. II", "II. byt", "2. byt", "Byt 2"
    m_txt = re.search(
        r"(?:\bbyt\s*(?:č\.?|c\.?|číslo)?\s*([IVX]{1,4}|\d{1,2})\b|\b([IVX]{1,4}|\d{1,2})\.\s*byt\b)",
        combined,
        flags=re.IGNORECASE,
    )
    if m_txt:
        token = (m_txt.group(1) or m_txt.group(2) or "").strip().upper()
        if token in _ROMAN_TO_INT and _ROMAN_TO_INT[token] >= 2:
            return token
        if token.isdigit() and 2 <= int(token) <= 10:
            return _int_to_roman(int(token))

    # Check Column 1 | Column 2 leading pipe pattern in raw_ocr_left: e.g. "2 | 7 | Opacity | Anton"
    raw_left = (row.raw_ocr_left if row else "") or ""
    for line in raw_left.splitlines():
        if is_header_or_garbage(line):
            continue
        m_col = re.match(r"^\s*([IVX]{1,4}|\d{1,2})\.?\s*\|\s*(\d{1,2})\.?\s*\|\s*([A-Za-zÁ-ž])", line)
        if m_col:
            col1 = m_col.group(1).strip().upper()
            if col1 in _ROMAN_TO_INT and _ROMAN_TO_INT[col1] >= 2:
                return col1
            if col1.isdigit() and 2 <= int(col1) <= 10:
                return _int_to_roman(int(col1))
        break

    return None


def _enrich_census_fact_from_ocr(fact: PersonFact, row: RecordRow | None) -> None:
    """
    Normalizes `relation_to_head`, cleans common cursive OCR misreads on `given_name_raw`,
    and fills missing `given_name_raw` / `sex` from `row.raw_ocr_left` when available.
    """
    if fact.relation_to_head:
        fact.relation_to_head = normalize_census_relation(fact.relation_to_head)

    raw_left = (row.raw_ocr_left if row else "") or ""
    left_cells: list[str] = []
    for line in raw_left.splitlines():
        if is_header_or_garbage(line):
            continue
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs and cs not in ("[ZAČIERNENÉ / REDACTED 1940]", "-"):
                left_cells.append(cs)

    # Check if relation_to_head was missed or if OCR cells in left half contain a more specific kinship role (e.g. manželka, pastorok / pastorkyňa)
    row_idx = row.row_index if row else None
    explicit_byt = _detect_explicit_byt_marker(fact, row)
    p_num_clean = str(fact.person_row_number or "").strip(" .")
    is_subsequent_in_same_byt = bool(
        (row_idx is not None and row_idx > 1 or p_num_clean not in ("", "1", "I"))
        and not explicit_byt
        and p_num_clean not in ("1", "I")
    )

    for c in left_cells:
        rel_c = normalize_census_relation(c)
        if rel_c in _CANONICAL_RELATIONS:
            if (
                not fact.relation_to_head
                or fact.relation_to_head not in _CANONICAL_RELATIONS
                or (fact.relation_to_head in ("syn", "dcéra") and rel_c not in ("syn", "dcéra", "hlava domácnosti"))
                or (
                    is_subsequent_in_same_byt
                    and fact.relation_to_head in _HEAD_RELATIONS
                    and rel_c not in _HEAD_RELATIONS
                )
            ):
                fact.relation_to_head = rel_c
                break

    # Fix known cursive given-name OCR readings or recover missing given_name_raw from Column 4
    given_ocr_fixes = {
        "doref": "Jozef",
        "joref": "Jozef",
        "zaan": "Ján",
        "zaw": "Ján",
        "zan": "Ján",
        "jau": "Ján",
        "jaw": "Ján",
        "jain": "Ján",
        "manta": "Marta",
        "ferdinand": "Ferdinand",
        "marina": "Mária",
        "havia": "Mária",
        "haqdalena": "Magdaléna",
        "heter": "Štefan",
        "steena": "Irena",
        "stena": "Irena",
        "vivencia": "Vincencia",
        "mikulas": "Mikuláš",
        "ana": "Anna",
        "botela": "Estela",
        "botella": "Estela",
        "etela": "Estela",
        "aquac": "Ignác",
        "ignac": "Ignác",
        "rozalia": "Rozália",
    }
    if not fact.surname_raw and left_cells:
        for c in left_cells:
            c_clean = re.sub(r"^(?:vd\.?|va\.?|vdova)\s+", "", c.strip(), flags=re.IGNORECASE).strip(".,;: ")
            if c_clean != c.strip() and len(c_clean) >= 3 and c_clean[0].isupper():
                stem_c = _surname_stem(c_clean) or c_clean
                fact.surname_normalized = stem_c
                fact.surname_raw = _inflect_census_surname(
                    stem_c,
                    fact.sex,
                    fact.relation_to_head,
                    raw_ocr_surname=c_clean,
                )
                break
    elif fact.surname_raw:
        sanitized_raw = sanitize_corrupted_surname_suffix(fact.surname_raw) or fact.surname_raw
        stem_cur = _surname_stem(sanitized_raw) or _surname_stem(fact.surname_normalized)
        # Check if raw_ocr_left has the actual scan reading without a forced -ová suffix
        scan_ocr_token: str | None = sanitized_raw
        if left_cells and stem_cur and not is_slovak_adjectival_surname(stem_cur):
            for c in left_cells:
                c_clean = re.sub(r"^(?:vd\.?|va\.?|vdova|ovd\.?)\s+", "", c.strip(), flags=re.IGNORECASE).strip(".,;:'\" ")
                if c_clean and (_surname_stem(c_clean) == stem_cur or _paleographic_stem_similarity(_surname_stem(c_clean) or c_clean, stem_cur) >= 0.80):
                    if not _has_ocr_feminine_ova_ending(c_clean, stem_cur):
                        scan_ocr_token = c_clean
                        break
        if stem_cur:
            inflected_cur = _inflect_census_surname(
                stem_cur,
                fact.sex,
                fact.relation_to_head,
                raw_ocr_surname=scan_ocr_token,
            )
            if stem_cur != fact.surname_normalized or inflected_cur != fact.surname_raw:
                orig_r, orig_n = fact.surname_raw, fact.surname_normalized
                if orig_r != inflected_cur and not orig_r.lower().endswith((stem_cur.lower() + "ová", stem_cur.lower() + "ova")):
                    _record_surname_candidate(
                        fact,
                        original_raw=orig_r,
                        original_normalized=orig_n,
                        consensus_stem=stem_cur,
                        inflected_surname=inflected_cur,
                    )
                fact.surname_normalized = stem_cur
                fact.surname_raw = inflected_cur
    if not fact.given_name_raw and left_cells:
        for idx_c, c in enumerate(left_cells):
            c_low = c.strip(".,;: ").lower()
            if c_low in given_ocr_fixes:
                fact.given_name_raw = given_ocr_fixes[c_low]
                break
            if fact.surname_raw and idx_c > 0 and left_cells[idx_c - 1].strip().lower() == fact.surname_raw.strip().lower():
                if is_known_given_name(c):
                    fact.given_name_raw = c.strip()
                    break

    # Also repair if an earlier run mapped "Heter" -> "Peter" while raw OCR / given_name_hu had "Heter"
    if (
        (fact.given_name_hu and fact.given_name_hu.strip().lower() in given_ocr_fixes)
        or (
            fact.given_name_raw == "Peter"
            and any(c.strip(".,;: ").lower() == "heter" for c in left_cells)
        )
    ):
        fix_key = (
            fact.given_name_hu.strip().lower()
            if (fact.given_name_hu and fact.given_name_hu.strip().lower() in given_ocr_fixes)
            else "heter"
        )
        fixed_gn = given_ocr_fixes[fix_key]
        fact.given_name_raw = fixed_gn
        fact.given_name_sk = fixed_gn
        fact.given_name_hu = fixed_gn

    if fact.given_name_raw:
        g_low = fact.given_name_raw.strip(".,;: ").lower()
        if g_low in given_ocr_fixes:
            fixed_gn = given_ocr_fixes[g_low]
            fact.given_name_raw = fixed_gn
            fact.given_name_sk = fixed_gn
            fact.given_name_hu = fixed_gn
        elif fact.given_name_raw.islower():
            fact.given_name_raw = fact.given_name_raw.capitalize()
            fact.given_name_sk = fact.given_name_raw
        else:
            g_sk, g_hu = normalize_given_name(fact.given_name_raw)
            if g_sk:
                fact.given_name_sk = g_sk
            if g_hu and (not fact.given_name_hu or fact.given_name_hu.strip().lower() in given_ocr_fixes):
                fact.given_name_hu = g_hu

    if fact.marital_status and fact.marital_status.strip().lower() == "rdovec":
        fact.marital_status = "vdovec"

    # Infer or repair sex from relation_to_head, given_name, or OCR cell
    rel_n = fact.relation_to_head or ""
    if rel_n in _FEMALE_RELATIONS:
        fact.sex = "F"
    elif rel_n in _MALE_RELATIONS:
        fact.sex = "M"
    else:
        v_sex = validate_sex(None, fact.given_name_sk or fact.given_name_raw)
        if v_sex in ("M", "F"):
            fact.sex = v_sex
        elif not fact.sex:
            for c in left_cells:
                cl = c.lower()
                if cl in ("munrle", "mutsle", "muríle", "marshé", "murfle", "munhé", "murshé", "mužské", "muž.", "muž"):
                    fact.sex = "M"
                    break
                if cl in ("jevrhá", "furhe'", "jeurhe", "jeurla", "ženské", "žen.", "žena"):
                    fact.sex = "F"
                    break

    # Recover missing marital_status from left OCR cells (cursive OCR misreads)
    if not fact.marital_status and left_cells:
        for c in left_cells:
            cl = c.strip(".,;:'\" ").lower()
            if cl in ("fenatz", "fenaty", "zenaty", "ženatý", "žen.", "zen."):
                fact.marital_status = "ženatý"
                break
            if cl in ("nydata", "vydata", "vydatá", "vyd."):
                fact.marital_status = "vydatá"
                break
            if cl in (
                "shob núa",
                "shob nua",
                "nobrida",
                "no borna",
                "sobo sá",
                "sobo sa",
                "slobodna",
                "slobodná",
                "slob.",
            ):
                fact.marital_status = "slobodná" if fact.sex == "F" else "slobodný"
                break
            if cl in ("notary", "slobodny", "slobodný"):
                fact.marital_status = "slobodná" if fact.sex == "F" else "slobodný"
                break
            if cl in ("vdova", "vd."):
                fact.marital_status = "vdova" if fact.sex == "F" else "vdovec"
                break
            if cl in ("vdovec", "rdovec"):
                fact.marital_status = "vdovec"
                break

    # Recover religion, literacy, and occupation fields from right OCR cells
    raw_right = (row.raw_ocr_right if row else "") or ""
    right_cells: list[str] = []
    for line in raw_right.splitlines():
        if is_header_or_garbage(line):
            continue
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs and cs not in ("[ZAČIERNENÉ / REDACTED 1940]", "-"):
                right_cells.append(cs)

    if not fact.religion and right_cells:
        for c in right_cells:
            relig_c = normalize_religion(c)
            if relig_c:
                fact.religion = relig_c
                break

    if not fact.literacy and right_cells:
        for c in right_cells:
            cl = c.lower()
            if any(
                tok in cl
                for tok in (
                    "číta a píše",
                    "cita a pise",
                    "číta aj píše",
                    "citat a priat",
                    "citat a print",
                    "citrt a print",
                    "čítať a písať",
                )
            ):
                fact.literacy = "číta aj píše"
                break

    if not fact.social_status_or_occupation and right_cells:
        joined_right = " | ".join(right_cells).lower()
        muni_label = (fact.village_sk or fact.village_raw or "").strip()
        if "pol. hosp" in joined_right or "poľ. hosp" in joined_right or "poľnohosp" in joined_right:
            fact.social_status_or_occupation = "poľnohospodárstvo"
            if ("volnik" in joined_right or "roľník" in joined_right or "rolnik" in joined_right) and not fact.occupation_position:
                fact.occupation_position = "roľník, majiteľ" if ("mafitel" in joined_right or "majitel" in joined_right or "majiteľ" in joined_right) else "roľník"
            if ("blasnej" in joined_right or "vlastnej" in joined_right) and not fact.employer_workplace:
                fact.employer_workplace = f"na vlastnej usadlosti, {muni_label}" if muni_label else "na vlastnej usadlosti"
        elif "domäneutti" in joined_right or "v domácnosti pred" in joined_right or "a) dom" in joined_right:
            fact.social_status_or_occupation = "v domácnosti"
            if not fact.occupation_position:
                fact.occupation_position = "pomáhajúci člen"
            if not fact.employer_workplace:
                fact.employer_workplace = "v domácnosti prednostu dom."
        elif "žiačka" in joined_right or "ziacka" in joined_right or "ľud. škola" in joined_right:
            fact.social_status_or_occupation = "žiačka" if fact.sex == "F" else "žiak"
            if not fact.employer_workplace and ("škola" in joined_right or "skola" in joined_right):
                fact.employer_workplace = f"št. ľud. škola, {muni_label}" if muni_label else "št. ľud. škola"


def reconcile_census_household_relationships(
    session: Session,
    census_sheet_id: int | CensusSheet,
) -> CensusHousehold | None:
    """
    Segments all extracted PersonFact rows of a CensusSheet (ordered by `row.row_index`)
    into one or more Byty (dwelling units: `I`, `II`, `III`, ...), resolves Byt surname
    consensus & paleographic disambiguation for each Byt, creates/updates one
    `CensusHousehold` per Byt (`dwelling_unit="I"`, `"II"`, etc.), assigns
    `fact.household_id = household.id`, and populates `father_name`, `mother_name`,
    `mother_maiden_surname`, and `spouse_name` within each Byt.
    """
    if isinstance(census_sheet_id, CensusSheet):
        sheet = census_sheet_id
        sheet_id = sheet.id
    else:
        sheet_id = int(census_sheet_id)
        sheet = session.get(CensusSheet, sheet_id)
    if not sheet or sheet_id is None:
        return None

    facts = list(
        session.exec(
            select(PersonFact)
            .where(PersonFact.census_sheet_id == sheet_id)
            .order_by(PersonFact.id.asc())
        ).all()
    )
    if not facts:
        for stale_hh in session.exec(
            select(CensusHousehold).where(CensusHousehold.census_sheet_id == sheet_id)
        ).all():
            session.delete(stale_hh)
        matched_ovh: CensusOverviewHouse | None = None
        if sheet.sheet_type != "overview" and sheet.municipality and sheet.house_number:
            ov_stmt = select(CensusOverviewHouse).where(
                CensusOverviewHouse.census_year == sheet.census_year,
            )
            if sheet.house_number_int is not None:
                ov_stmt = ov_stmt.where(
                    (CensusOverviewHouse.house_number_int == sheet.house_number_int)
                    | (CensusOverviewHouse.house_number == str(sheet.house_number).strip())
                )
            ov_houses = session.exec(ov_stmt).all()
            for ovh in ov_houses:
                if can_match_overview_house_to_sheet(ovh, sheet):
                    matched_ovh = ovh
                    if ovh.matched_sheet_id != sheet.id:
                        ovh.matched_sheet_id = sheet.id
                        session.add(ovh)
                    break
        if matched_ovh and matched_ovh.owner_name:
            if matched_ovh.is_inhabited is False:
                sheet.household_owner = f"{matched_ovh.owner_name} (neobývaný dom)"
            else:
                sheet.household_owner = matched_ovh.owner_name
            if not sheet.street_raw and matched_ovh.settlement_name:
                sheet.street_raw = matched_ovh.settlement_name
        else:
            sheet.household_owner = None
        if sheet.municipality:
            norm_sheet_muni, _, _ = normalize_census_municipality(sheet.municipality, sheet_municipality=None)
            if norm_sheet_muni and norm_sheet_muni != sheet.municipality:
                sheet.municipality = norm_sheet_muni
        session.add(sheet)
        session.commit()
        index_census_sheet(session, sheet)
        session.commit()
        return None

    row_ids = list({f.row_id for f in facts if f.row_id})
    row_by_id: dict[int, RecordRow] = (
        {
            r_obj.id: r_obj
            for r_obj in session.exec(select(RecordRow).where(RecordRow.id.in_(row_ids))).all()  # type: ignore[union-attr]
            if r_obj.id is not None
        }
        if row_ids
        else {}
    )

    def _fact_sort_key(f: PersonFact) -> tuple[int, int]:
        r_obj = row_by_id.get(f.row_id) if f.row_id else None
        if r_obj is not None:
            return (r_obj.row_index, f.id or 0)
        try:
            p_num = int(re.sub(r"\D+", "", str(f.person_row_number or "")) or "999")
        except ValueError:
            p_num = 999
        return (p_num, f.id or 0)

    facts.sort(key=_fact_sort_key)

    for f in facts:
        _enrich_census_fact_from_ocr(f, row_by_id.get(f.row_id) if f.row_id else None)

    feature_flags = settings.get_feature_flags()
    enable_multi_byt = bool(feature_flags.get("census_multi_byt", True))
    enable_surname_consensus = bool(feature_flags.get("census_surname_consensus", True))

    # Segment ordered facts into distinct Byty (dwelling units: I, II, III, ...)
    byt_groups: list[tuple[str, list[PersonFact]]] = []
    current_unit = "I"
    current_byt_index = 1
    current_group: list[PersonFact] = []

    for idx, f in enumerate(facts):
        r_obj = row_by_id.get(f.row_id) if f.row_id else None
        if idx == 0 or not enable_multi_byt:
            current_group.append(f)
            continue

        explicit_marker = _detect_explicit_byt_marker(f, r_obj)
        rel_norm = normalize_census_relation(f.relation_to_head) or (f.relation_to_head or "").strip().lower()
        p_num_str = str(f.person_row_number or "").strip(" .")
        is_row_num_restart = p_num_str in ("1", "I") and len(current_group) >= 1
        is_explicit_head = rel_norm in _HEAD_RELATIONS

        # Check if Column 5 in left_cells actually has an explicit head-of-household phrase
        left_has_head_phrase = False
        if r_obj and r_obj.raw_ocr_left:
            for line in r_obj.raw_ocr_left.splitlines():
                if is_header_or_garbage(line):
                    continue
                for cell_txt in line.split("|"):
                    if normalize_census_relation(cell_txt.strip()) in _HEAD_RELATIONS:
                        left_has_head_phrase = True
                        break

        # Guard against treating a ditto/continuation row or same-family / sequential row without Byt marker as a new Byt
        first_head = current_group[0]
        shares_group_surname = any(
            _paleographic_stem_similarity(f.surname_raw or f.surname_normalized, gf.surname_raw or gf.surname_normalized) >= 0.78
            for gf in current_group
            if (gf.surname_raw or gf.surname_normalized)
        )
        is_ditto_or_continuation = bool(
            _is_census_ditto(f.surname_raw)
            or (
                not explicit_marker
                and not is_row_num_restart
                and (shares_group_surname or not left_has_head_phrase)
            )
            or (
                not explicit_marker
                and not is_row_num_restart
                and f.sex == "F"
                and f.mother_maiden_surname
                and _paleographic_stem_similarity(f.surname_raw, first_head.surname_raw) >= 0.85
            )
            or (
                not explicit_marker
                and not is_row_num_restart
                and f.given_name_raw
                and first_head.given_name_raw
                and ascii_fold(f.given_name_raw) == ascii_fold(first_head.given_name_raw)
                and _paleographic_stem_similarity(f.surname_raw, first_head.surname_raw) >= 0.85
            )
        )

        # If a subsequent row within the same Byt was mistakenly labeled "hlava domácnosti", demote it to spouse/child
        if not explicit_marker and not is_row_num_restart and is_explicit_head and is_ditto_or_continuation:
            if len(current_group) == 1 and f.sex == "F":
                f.relation_to_head = "manželka"
            elif first_head.birth_year_est and f.birth_year_est and f.birth_year_est >= first_head.birth_year_est + 15:
                f.relation_to_head = "dcéra" if f.sex == "F" else "syn"
            else:
                f.relation_to_head = "manželka" if f.sex == "F" else "syn"
            rel_norm = f.relation_to_head
            is_explicit_head = False

        if (explicit_marker or is_explicit_head or is_row_num_restart) and not is_ditto_or_continuation:
            byt_groups.append((current_unit, current_group))
            current_byt_index += 1
            if explicit_marker and _ROMAN_TO_INT.get(explicit_marker, 0) >= current_byt_index:
                current_unit = explicit_marker
                current_byt_index = _ROMAN_TO_INT[explicit_marker]
            else:
                current_unit = _int_to_roman(current_byt_index)
            current_group = [f]
        else:
            current_group.append(f)

    if current_group:
        byt_groups.append((current_unit, current_group))

    existing_households = list(
        session.exec(
            select(CensusHousehold)
            .where(CensusHousehold.census_sheet_id == sheet_id)
            .order_by(CensusHousehold.id.asc())
        ).all()
    )

    active_households: list[CensusHousehold] = []
    for byt_pos, (dwelling_unit, byt_facts) in enumerate(byt_groups):
        # 1. Resolve Byt surname consensus & paleographic disambiguation for this Byt
        consensus_surname = (
            _resolve_byt_surname_consensus(byt_facts, session)
            if enable_surname_consensus
            else None
        )

        # 2. Create or reuse CensusHousehold for this Byt
        if byt_pos < len(existing_households):
            household = existing_households[byt_pos]
        else:
            household = CensusHousehold(census_sheet_id=sheet_id)
            session.add(household)
            session.flush()

        household.dwelling_unit = dwelling_unit
        household.apartment_number = str(_ROMAN_TO_INT.get(dwelling_unit, byt_pos + 1))
        household.street = sheet.street_raw or sheet.municipality
        household.house_number = sheet.house_number

        head_fact: PersonFact | None = None
        wife_fact: PersonFact | None = None
        for f in byt_facts:
            rel = (f.relation_to_head or "").lower()
            if rel in _HEAD_RELATIONS and not head_fact:
                head_fact = f
            elif rel in ("manželka", "družka") and not wife_fact:
                wife_fact = f

        if not head_fact and byt_facts:
            head_fact = byt_facts[0]
            if not head_fact.relation_to_head:
                head_fact.relation_to_head = "hlava domácnosti"

        if (
            not wife_fact
            and len(byt_facts) >= 2
            and head_fact
            and head_fact.sex == "M"
            and byt_facts[0].id == head_fact.id
            and not byt_facts[1].relation_to_head
            and byt_facts[1].sex == "F"
            and (
                not byt_facts[1].birth_year_est
                or not head_fact.birth_year_est
                or abs(byt_facts[1].birth_year_est - head_fact.birth_year_est) <= 22
            )
        ):
            byt_facts[1].relation_to_head = "manželka"
            wife_fact = byt_facts[1]

        if not consensus_surname and head_fact:
            consensus_surname = _surname_stem(head_fact.surname_normalized or head_fact.surname_raw)

        head_given = (head_fact.given_name_sk or head_fact.given_name_raw) if head_fact else None
        head_surname = (head_fact.surname_raw or consensus_surname) if head_fact else consensus_surname
        head_full = f"{head_given or ''} {head_surname or ''}".strip() if (head_given or head_surname) else None

        wife_given = (wife_fact.given_name_sk or wife_fact.given_name_raw) if wife_fact else None
        wife_surname = wife_fact.surname_raw if wife_fact else None
        wife_full = f"{wife_given or ''} {wife_surname or ''}".strip() if (wife_given or wife_surname) else None
        wife_maiden = wife_fact.mother_maiden_surname if wife_fact else None

        household.head_person_fact_id = head_fact.id if head_fact else None
        household.house_owner_name = head_full
        household.notes_raw = json.dumps(
            {
                "dwelling_unit": dwelling_unit,
                "consensus_surname": consensus_surname,
                "head_of_household_name": head_full,
                "wife_name": wife_full,
                "wife_maiden_name": wife_maiden,
                "head_surname": head_surname,
                "head_given_name": head_given,
                "wife_given_name": wife_given,
                "wife_maiden_surname": wife_maiden,
                "member_count": len(byt_facts),
                "members_count": len(byt_facts),
            },
            ensure_ascii=False,
        )
        session.add(household)
        session.flush()
        active_households.append(household)

        if byt_pos == 0 and head_full:
            sheet.household_owner = head_full
            session.add(sheet)

        if sheet.municipality:
            norm_sheet_muni, _, _ = normalize_census_municipality(sheet.municipality, sheet_municipality=None)
            if norm_sheet_muni and norm_sheet_muni != sheet.municipality:
                sheet.municipality = norm_sheet_muni
                session.add(sheet)

        byt_consensus_religion = next(
            (bf.religion for bf in byt_facts if bf.religion and not _is_census_ditto(bf.religion)),
            None,
        )

        prev_f: PersonFact | None = None
        byt_has_ova = any(_has_ocr_feminine_ova_ending(bf.surname_raw, consensus_surname) for bf in byt_facts)
        for idx_f, f in enumerate(byt_facts):
            f.household_id = household.id
            f.census_household_id = household.id
            rel = (f.relation_to_head or "").lower()
            if not rel and head_fact and f.id != head_fact.id and (not wife_fact or f.id != wife_fact.id):
                if head_fact.birth_year_est and f.birth_year_est and f.birth_year_est >= head_fact.birth_year_est + 15:
                    f.relation_to_head = "dcéra" if f.sex == "F" else "syn"
                    rel = f.relation_to_head
                elif prev_f and prev_f.relation_to_head in ("syn", "dcéra"):
                    f.relation_to_head = "dcéra" if f.sex == "F" else "syn"
                    rel = f.relation_to_head

            if not f.surname_raw and consensus_surname and rel in _SHARED_HOUSEHOLD_SURNAME_RELATIONS:
                f.surname_normalized = consensus_surname
                f.surname_raw = _inflect_census_surname(
                    consensus_surname,
                    f.sex,
                    rel,
                    raw_ocr_surname=None,
                    force_ova_on_ditto=byt_has_ova,
                )
            elif f.surname_normalized and f.surname_raw:
                expected_inflected = _inflect_census_surname(
                    f.surname_normalized,
                    f.sex,
                    rel,
                    raw_ocr_surname=f.surname_raw,
                )
                if expected_inflected and expected_inflected != f.surname_raw:
                    f.surname_raw = expected_inflected

            if head_fact and f.id == head_fact.id:
                f.spouse_name = wife_full
                f.father_name = None
                f.mother_name = None
                if not f.marital_status and wife_fact:
                    f.marital_status = "vydatá" if f.sex == "F" else "ženatý"
            elif wife_fact and f.id == wife_fact.id:
                f.spouse_name = head_full
                f.father_name = None
                f.mother_name = None
                if not f.marital_status:
                    f.marital_status = "vydatá"
            elif rel in ("syn", "dcéra", "nevlastný syn", "nevlastná dcéra", "adoptívny syn", "adoptívna dcéra"):
                if head_fact and (head_fact.sex == "M" or not wife_fact) and rel in ("syn", "dcéra", "adoptívny syn", "adoptívna dcéra"):
                    f.father_name = head_given
                if wife_fact:
                    f.mother_name = wife_given
                    if wife_maiden:
                        f.mother_maiden_surname = wife_maiden or f.mother_maiden_surname
                elif head_fact and head_fact.sex == "F":
                    f.mother_name = head_given
                    f.father_name = None
                if not f.marital_status and f.birth_year_est and sheet.census_year and (sheet.census_year - f.birth_year_est) <= 16:
                    f.marital_status = "slobodná" if f.sex == "F" else "slobodný"
                if (
                    not f.social_status_or_occupation
                    and f.birth_year_est
                    and sheet.census_year
                    and (sheet.census_year - f.birth_year_est) <= 14
                ):
                    f.social_status_or_occupation = "v domácnosti"
                    if not f.occupation_position:
                        f.occupation_position = "dieťa"
            else:
                # Non-child relatives (e.g. svokra, švagriná, slúžka) should not inherit head/wife as parents
                if f.father_name in (head_given, wife_given):
                    f.father_name = None
                if f.mother_name in (head_given, wife_given):
                    f.mother_name = None

            if (not f.religion or _is_census_ditto(f.religion)) and byt_consensus_religion:
                f.religion = byt_consensus_religion

            if f.home_jurisdiction:
                f.home_jurisdiction = re.sub(r"\bHorný\s+Čepen\b", "Horný Čepeň", f.home_jurisdiction, flags=re.IGNORECASE)
                f.home_jurisdiction = re.sub(r"\bDolný\s+Čepen\b", "Dolný Čepeň", f.home_jurisdiction, flags=re.IGNORECASE)
                f.home_jurisdiction = re.sub(r"\bStredný\s+Čepen\b", "Stredný Čepeň", f.home_jurisdiction, flags=re.IGNORECASE)
            if f.village_raw and re.search(r"\b(?:Horný|Dolný|Stredný)\s+Čepen\b", f.village_raw, flags=re.IGNORECASE):
                f.village_raw = re.sub(r"Čepen\b", "Čepeň", f.village_raw)
            if f.village_sk and re.search(r"\b(?:Horný|Dolný|Stredný)\s+Čepen\b", f.village_sk, flags=re.IGNORECASE):
                f.village_sk = re.sub(r"Čepen\b", "Čepeň", f.village_sk)
            if f.fs_place_standardized and re.search(r"\b(?:Horný|Dolný|Stredný)\s+Čepen\b", f.fs_place_standardized, flags=re.IGNORECASE):
                f.fs_place_standardized = re.sub(r"Čepen\b", "Čepeň", f.fs_place_standardized)

            # Re-extract & normalize Rodisko (a) Obec, (b) Okres, (c) Krajina from right OCR + regional ~100km gazetteer
            r_obj = row_by_id.get(f.row_id) if f.row_id else None
            ocr_muni, ocr_dist, ocr_cnt = (
                _extract_birthplace_from_right_ocr(r_obj.raw_ocr_right, sheet=sheet)
                if (r_obj and r_obj.raw_ocr_right)
                else (None, None, None)
            )
            if ocr_muni == "//":
                cur_muni_norm = "//"
                _, cur_dist_norm, cur_cnt_norm = normalize_census_birthplace_fields(
                    None,
                    f.birth_place_district or ocr_dist,
                    f.birth_place_country or ocr_cnt,
                    sheet_municipality=sheet.municipality,
                    sheet_district=sheet.district,
                )
            else:
                cur_muni_norm, cur_dist_norm, cur_cnt_norm = normalize_census_birthplace_fields(
                    f.birth_place_municipality,
                    f.birth_place_district,
                    f.birth_place_country,
                    sheet_municipality=sheet.municipality,
                    sheet_district=sheet.district,
                )
            f.birth_place_municipality = cur_muni_norm or ocr_muni
            f.birth_place_district = cur_dist_norm or ocr_dist
            f.birth_place_country = cur_cnt_norm or ocr_cnt
            if f.birth_place_municipality and f.birth_place_municipality != "//":
                f.birth_place_municipality, f.birth_place_district, f.birth_place_country = (
                    normalize_census_birthplace_fields(
                        f.birth_place_municipality,
                        f.birth_place_district,
                        f.birth_place_country,
                        sheet_municipality=sheet.municipality,
                        sheet_district=sheet.district,
                    )
                )

            # Ditto & household inheritance within this Byt
            donor = prev_f or head_fact
            if idx_f > 0 and donor and donor.id != f.id:
                for attr in ("citizenship", "nationality", "religion", "home_jurisdiction", "literacy"):
                    val = getattr(f, attr, None)
                    if not val or _is_census_ditto(val):
                        donor_val = getattr(donor, attr, None) or (getattr(head_fact, attr, None) if head_fact else None)
                        if donor_val and not _is_census_ditto(donor_val):
                            setattr(f, attr, donor_val)

                # Birthplace ditto inheritance
                if _is_census_ditto(f.birth_place_municipality) or _is_census_ditto(f.birth_place_raw):
                    f.birth_place_municipality = donor.birth_place_municipality or donor.birth_place_sk
                if _is_census_ditto(f.birth_place_district):
                    f.birth_place_district = donor.birth_place_district
                if _is_census_ditto(f.birth_place_country):
                    f.birth_place_country = donor.birth_place_country

                if not f.birth_place_municipality and rel in ("syn", "dcéra"):
                    f.birth_place_municipality = donor.birth_place_municipality or sheet.municipality
                    f.birth_place_district = f.birth_place_district or donor.birth_place_district or sheet.district
                    f.birth_place_country = f.birth_place_country or donor.birth_place_country or "Slovensko"

            # Fill district/country when birth_place_municipality matches sheet.municipality
            if f.birth_place_municipality:
                f.birth_place_municipality, f.birth_place_district, f.birth_place_country = (
                    normalize_census_birthplace_fields(
                        f.birth_place_municipality,
                        f.birth_place_district,
                        f.birth_place_country,
                        sheet_municipality=sheet.municipality,
                        sheet_district=sheet.district,
                    )
                )
                if sheet.municipality and ascii_fold(f.birth_place_municipality) == ascii_fold(sheet.municipality):
                    if not f.birth_place_district and sheet.district:
                        f.birth_place_district = sheet.district
                    if not f.birth_place_country:
                        f.birth_place_country = "Slovensko"
                if not f.birth_place_country:
                    f.birth_place_country = "Slovensko"
                f.birth_place_raw = _format_birth_place_raw(
                    f.birth_place_municipality,
                    f.birth_place_district,
                    f.birth_place_country,
                    f.birth_place_raw,
                )
                f.birth_place_sk = f.birth_place_municipality
            elif not sheet.is_blackened:
                f.birth_place_raw = None
                f.birth_place_sk = None
                f.birth_place_district = None
                f.birth_place_country = None

            # Keep RecordRow.normalized_json synchronized with reconciled PersonFact
            if r_obj and r_obj.normalized_json:
                try:
                    nd = json.loads(r_obj.normalized_json)
                    if isinstance(nd, dict):
                        nd["surname_raw"] = f.surname_raw
                        nd["surname_normalized"] = f.surname_normalized
                        nd["given_name_raw"] = f.given_name_raw
                        nd["given_name_sk"] = f.given_name_sk
                        nd["relation_to_head"] = f.relation_to_head
                        nd["sex"] = f.sex
                        nd["dwelling_unit"] = dwelling_unit
                        nd["birth_place_municipality"] = f.birth_place_municipality
                        nd["birth_place_district"] = f.birth_place_district
                        nd["birth_place_country"] = f.birth_place_country
                        nd["birth_place_raw"] = f.birth_place_raw
                        r_obj.normalized_json = json.dumps(nd, ensure_ascii=False)
                        session.add(r_obj)
                except Exception:
                    pass

            # Refresh existing FamilySearchMatch action packet & summary if present and matched
            if f.id is not None:
                existing_fs_match = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == f.id)
                ).first()
                if existing_fs_match is not None and not (existing_fs_match.fs_person_id or "").strip():
                    session.delete(existing_fs_match)
                    existing_fs_match = None
                if existing_fs_match is not None:
                    try:
                        from urllib.parse import quote_plus
                        from backend.app.db.models import LocalRecordMatch
                        from backend.app.mcp.familysearch_client import (
                            _split_spouse_name,
                            build_census_action_packet,
                        )

                        place_std = (
                            f.fs_place_standardized
                            or f"{sheet.municipality}, {sheet.district or ''}, Slovakia".strip(", ")
                        )
                        given_fs = f.given_name_sk or f.given_name_raw or ""
                        surname_fs = f.surname_normalized or f.surname_raw or ""
                        birth_year_fs = f.birth_year_est
                        birth_place_fs = (
                            getattr(f, "birth_place_municipality", None)
                            or f.birth_place_sk
                            or f.birth_place_raw
                            or place_std
                        )
                        spouse_given, spouse_surname = _split_spouse_name(f, session)
                        fs_search_params = [
                            f"q.givenName={quote_plus(given_fs)}" if given_fs else "",
                            f"q.surname={quote_plus(surname_fs)}" if surname_fs else "",
                            f"q.birthLikePlace={quote_plus(birth_place_fs)}" if birth_place_fs else "",
                            f"q.birthLikeDate.from={birth_year_fs - 2}" if birth_year_fs else "",
                            f"q.birthLikeDate.to={birth_year_fs + 2}" if birth_year_fs else "",
                            f"q.residencePlace={quote_plus(place_std)}",
                            f"q.residenceDate.from={sheet.census_year}",
                            f"q.residenceDate.to={sheet.census_year}",
                            f"q.fatherGivenName={quote_plus(f.father_name)}" if f.father_name else "",
                            f"q.motherGivenName={quote_plus(f.mother_name)}" if f.mother_name else "",
                            f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                            f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
                        ]
                        fs_url = "https://www.familysearch.org/search/record/results?" + "&".join(
                            [p for p in fs_search_params if p]
                        )
                        local_matches = list(
                            session.exec(
                                select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id == f.id)
                            ).all()
                        )
                        action_packet, what_is_new = build_census_action_packet(
                            fact=f,
                            sheet=sheet,
                            household=household,
                            place_std=place_std,
                            fs_url=fs_url,
                            local_matches=local_matches,
                        )
                        summary_sk = (
                            f"Sčítací hárok {sheet.census_year} ({sheet.municipality}, dom č. {sheet.house_number or '—'}): "
                            f"{given_fs or '?'} {surname_fs or '?'} ({f.relation_to_head or 'člen domácnosti'})"
                        )
                        if local_matches:
                            summary_sk += f" – prepojené s lokálnou matrikou ({len(local_matches)} zhodných záznamov)."
                        existing_fs_match.fs_url = fs_url
                        existing_fs_match.summary_sk = summary_sk
                        existing_fs_match.what_is_new_json = json.dumps(what_is_new, ensure_ascii=False)
                        existing_fs_match.action_packet_json = json.dumps(action_packet, ensure_ascii=False)
                        session.add(existing_fs_match)
                    except Exception:
                        pass

            prev_f = f
            session.add(f)
            index_person_fact(session, f)

    # Delete any stale extra CensusHousehold records from prior runs
    if len(existing_households) > len(active_households):
        for stale_hh in existing_households[len(active_households) :]:
            session.delete(stale_hh)

    # Link any CensusOverviewHouse in the same municipality & census_year with matching house_number
    if sheet.sheet_type != "overview" and sheet.municipality and sheet.house_number:
        ov_stmt = select(CensusOverviewHouse).where(
            CensusOverviewHouse.census_year == sheet.census_year,
        )
        if sheet.house_number_int is not None:
            ov_stmt = ov_stmt.where(
                (CensusOverviewHouse.house_number_int == sheet.house_number_int)
                | (CensusOverviewHouse.house_number == str(sheet.house_number).strip())
            )
        ov_houses = session.exec(ov_stmt).all()
        for ovh in ov_houses:
            if can_match_overview_house_to_sheet(ovh, sheet) and ovh.matched_sheet_id != sheet.id:
                ovh.matched_sheet_id = sheet.id
                session.add(ovh)

    session.commit()
    index_census_sheet(session, sheet)
    session.commit()
    primary_household = active_households[0] if active_households else None
    if primary_household:
        session.refresh(primary_household)
    return primary_household


CENSUS_OVERVIEW_SYSTEM_PROMPT = """You are an expert Slovak archivist transcribing a 'Prehľad sčítacieho obvodu' (Census Enumeration District Overview table, 1930/1940 Slovak Census).
The table has 8 columns:
1: Radové číslo (ordinal row number 1, 2, 3...)
2: Meno osady (dediny, samoty, ulice, námestia) (settlement / village / street name e.g. "Veľké Šurovce", or `- " -` / `//` ditto mark)
3: Číslo domu (house number e.g. "1", "2", "18")
4: Či je dom obývaný alebo neobývaný (is_inhabited: true if "obývaný" or "1", false if "neobývaný" / "0")
5: Meno majiteľa domu (house owner name e.g. "Ján Šulák", "Karol Gálik", "Jozef Paraba", or `- " -` / `//` ditto mark if same owner as previous house)
6: Počet bytových strán (domácností) (household_count: integer e.g. 1, 2)
7: Počet prítomných osôb (present_persons_count: integer e.g. 4, 6, 8)
8: Poznámka (remarks e.g. "obchod", "škola", "neobývaný")
Return ONLY raw JSON starting with { and ending with }."""


_OVERVIEW_GIVEN_NAME_FIXES: dict[str, str] = {
    "zdun": "Ján",
    "zoim": "Ján",
    "zvon": "Ján",
    "zou": "Ján",
    "zan": "Ján",
    "zán": "Ján",
    "zain": "Ján",
    "zaun": "Ján",
    "jam": "Ján",
    "yan": "Ján",
    "john": "Ján",
    "jane": "Ján",
    "jano": "Ján",
    "jan": "Ján",
    "ján": "Ján",
    "zoref": "Jozef",
    "goref": "Jozef",
    "jored": "Jozef",
    "joref": "Jozef",
    "josef": "Jozef",
    "doref": "Jozef",
    "gef": "Jozef",
    "jozef": "Jozef",
    "zakub": "Jakub",
    "zakul": "Jakub",
    "jakub": "Jakub",
    "tomas": "Tomáš",
    "tomáš": "Tomáš",
    "andrej": "Andrej",
    "adam": "Adam",
    "matej": "Matej",
    "martin": "Martin",
    "michal": "Michal",
    "stefan": "Štefan",
    "štefan": "Štefan",
    "pavel": "Pavel",
    "pavol": "Pavol",
    "anton": "Anton",
    "frantisek": "František",
    "františek": "František",
    "karol": "Karol",
    "imrich": "Imrich",
    "ondrej": "Ondrej",
    "juraj": "Juraj",
    "peter": "Peter",
    "ignac": "Ignác",
    "ignác": "Ignác",
    "eduard": "Eduard",
    "bedrich": "Bedrich",
    "maria": "Mária",
    "mária": "Mária",
    "anna": "Anna",
    "katarina": "Katarína",
    "katarína": "Katarína",
    "katerina": "Katarína",
    "materina": "Katarína",
    "alzbeta": "Alžbeta",
    "alžbeta": "Alžbeta",
    "zuzana": "Zuzana",
    "adela": "Adela",
    "albina": "Albína",
    "albína": "Albína",
    "rora": "Róza",
    "roza": "Róza",
    "róza": "Róza",
    "karolina": "Karolína",
    "karolína": "Karolína",
    "gisela": "Gizela",
    "gizela": "Gizela",
}


def _normalize_overview_owner(raw_owner: str | None, prev_owner: str | None = None) -> str | None:
    if not raw_owner:
        return None
    s = raw_owner.strip()
    low = s.lower().strip(".,;:")
    if _is_census_ditto(s) or low in ("dtto", "dto", "tto", "dt0", "dtu", "dtt", "-11-", "- 11 -"):
        return prev_owner

    # Clean common cursive OCR phrase misreads and restore standard Slovak feminine/patronymic suffixes
    s = re.sub(r"\bmate\s+of\b", "Matej", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*'\s*$", "", s)
    s = re.sub(r"([a-záčďéíľĺňóôŕšťúýž])'\s+", r"\1á ", s)
    s = re.sub(r"([A-Za-záčďéíľĺňóôŕšťúýž]{3,})ova\b", r"\1ová", s)
    s = re.sub(r"([A-Za-záčďéíľĺňóôŕšťúýž]{3,})ov[ií][cć]\b", r"\1ovič", s)
    s = re.sub(r"\bLom[aá]'\s+aprava\b|\bLomna\s+sprava\b|\bLomis'\s+správa\b|\bhenná\s+oprava\b|\bLosna'\s+správa\b", "Lesná správa", s, flags=re.IGNORECASE)
    if "ministerstvo" in s.lower() and "škol" in s.lower():
        return "Ministerstvo školstva – štát. ľud. škola"

    tokens = s.split()
    if tokens:
        first_clean = tokens[0].strip(".,;:()").lower()
        if first_clean in _OVERVIEW_GIVEN_NAME_FIXES and len(tokens) <= 4:
            tokens[0] = _OVERVIEW_GIVEN_NAME_FIXES[first_clean]
            if len(tokens) >= 2 and tokens[1][:1].islower() and tokens[1].lower() not in ("a", "rod.", "vd.", "ml.", "st."):
                tokens[1] = tokens[1][:1].upper() + tokens[1][1:]
            s = " ".join(tokens)
        elif s[:1].islower() and not s.lower().startswith("vd."):
            s = s[:1].upper() + s[1:]
    return s if s.lower().endswith(("st.", "ml.")) else s.rstrip(".:,")


def _heuristic_parse_overview_row(
    raw_text: str,
    *,
    default_ordinal: int,
    default_settlement: str | None,
    prev_house: CensusOverviewHouse | None = None,
) -> dict[str, Any]:
    cells: list[str] = []
    for line in (raw_text or "").splitlines():
        for c in line.split("|"):
            cs = re.sub(r"\s+", " ", c).strip()
            if cs:
                cells.append(cs)

    ordinal_number: int | None = default_ordinal
    settlement_name: str | None = None
    house_number: str | None = None
    is_inhabited: bool = True
    owner_name: str | None = None
    household_count: int | None = None
    present_persons_count: int | None = None
    remarks: str | None = None

    ints_before_owner: list[str] = []
    ints_after_owner: list[int] = []
    for idx, c in enumerate(cells):
        low = c.lower().strip(".,;:")
        if any(hdr in low for hdr in ("radové číslo", "meno osady", "číslo domu", "meno majiteľa", "počet bytových", "úhrn", "súhrn")):
            continue
        if _is_census_ditto(c) or low in ("-11-", "- 11 -", "-ii-", "dtto", "dto", "tto", "dt0", "dtu", "dtt"):
            if low in ("dtto", "dto", "tto", "dt0", "dtu", "dtt") and prev_house:
                owner_name = prev_house.owner_name
            elif not settlement_name and not owner_name and not ints_before_owner:
                settlement_name = (prev_house.settlement_name if prev_house else None) or default_settlement
            elif not owner_name and prev_house:
                owner_name = prev_house.owner_name
            continue
        if "neobýv" in low or "neobyv" in low:
            is_inhabited = False
            continue
        if "obýv" in low or "obyv" in low:
            is_inhabited = True
            continue
        if c in ("-", "—", "/", "+", "√", "%"):
            continue

        # Handle merged two-integer cells like "1. 10" or "1.8" after owner_name
        if owner_name and re.fullmatch(r"\d{1,2}\s*[.,]\s*\d{1,3}\.?", c):
            m_pair = re.findall(r"\d+", c)
            if len(m_pair) == 2:
                ints_after_owner.extend([int(m_pair[0]), int(m_pair[1])])
                continue

        if re.fullmatch(r"[\[{/.+-√✓]*\d{1,4}[a-zA-Z]?[/.+-√✓}\]]*", c):
            m_d = re.search(r"(\d+)", c)
            if m_d:
                val_int = int(m_d.group(1))
                if not owner_name:
                    ints_before_owner.append(str(val_int))
                else:
                    ints_after_owner.append(val_int)
                continue

        # Text cell: either settlement name or owner name or remarks
        if not settlement_name and not owner_name and (
            (default_settlement and ascii_fold(default_settlement) in ascii_fold(c))
            or any(
                v in low
                for v in (
                    "dedina",
                    "medina",
                    "skupina domov",
                    "strážny",
                    "stražny",
                    "strazny",
                    "mlyn",
                    "mljn",
                    "osada",
                    "ulica",
                    "dvor",
                    "majer",
                    "samota",
                    "kopanice",
                    "laz",
                    "lazy",
                    "potok",
                    "dolina",
                )
            )
        ):
            if "medina" in low or "dedina" in low:
                settlement_name = "dedina"
            elif "stražn" in low or "strážn" in low or "strazn" in low:
                settlement_name = "žel. strážny domček"
            elif "mljn" in low or "mlyn" in low:
                settlement_name = "mlyn"
            else:
                settlement_name = c.rstrip(".")
            continue
        if not owner_name and any(ch.isalpha() for ch in c) and len(c) >= 3:
            owner_name = _normalize_overview_owner(c, prev_house.owner_name if prev_house else None)
            continue
        if owner_name and any(ch.isalpha() for ch in c):
            remarks = f"{remarks}; {c}" if remarks else c

    house_number = str(default_ordinal)
    if ints_before_owner:
        for cand_str in ints_before_owner:
            if cand_str.isdigit():
                c_val = int(cand_str)
                if c_val == default_ordinal or (default_ordinal > 1 and abs(c_val - default_ordinal) <= 2):
                    house_number = str(c_val)
                    break

    if ints_after_owner:
        household_count = ints_after_owner[0]
        if len(ints_after_owner) >= 2:
            # Guard against 3-digit auditor page running totals (e.g. 341, 477) when household_count <= 4
            p_cand = ints_after_owner[1]
            if p_cand > 60 and (household_count or 1) <= 5 and len(ints_after_owner) >= 3 and ints_after_owner[2] <= 40:
                present_persons_count = ints_after_owner[2]
            elif p_cand <= 80:
                present_persons_count = p_cand

    if not settlement_name:
        settlement_name = (prev_house.settlement_name if prev_house else None) or default_settlement or "dedina"

    return {
        "ordinal_number": ordinal_number,
        "settlement_name": settlement_name,
        "house_number": house_number,
        "is_inhabited": is_inhabited,
        "owner_name": owner_name,
        "household_count": household_count,
        "present_persons_count": present_persons_count,
        "remarks": remarks,
    }


async def extract_census_overview_sheet(
    session: Session,
    sheet: CensusSheet,
    front_scan: ScanImage | None = None,
    back_scan: ScanImage | None = None,
) -> list[CensusOverviewHouse]:
    """
    Slices, OCRs, and extracts a 'Prehľad sčítacieho obvodu' sheet across ALL its pages
    (front _v, back _r, and any additional page ScanImages, including 4-page folded bifolio spreads)
    into CensusOverviewHouse records and RecordRow records, deleting any bogus household/PersonFact
    records on that sheet and cross-linking matched_sheet_id to household CensusSheets in the same municipality.
    Stage-batched: all GLM-OCR calls execute first while glm-ocr stays warm in VRAM,
    followed by structured extraction.
    """
    from pathlib import Path
    from backend.app.db.models import WorkflowTrace
    from backend.app.pipeline.ocr_stage import html_table_to_pipe_text, media_rel_to_path

    sheet.sheet_type = "overview"
    session.add(sheet)

    if front_scan is None and sheet.front_scan_id:
        front_scan = session.get(ScanImage, sheet.front_scan_id)
    if back_scan is None and sheet.back_scan_id:
        back_scan = session.get(ScanImage, sheet.back_scan_id)

    # 1. Delete any bogus PersonFact / FamilySearchMatch / CensusHousehold previously created on this overview sheet
    for pf in session.exec(select(PersonFact).where(PersonFact.census_sheet_id == sheet.id)).all():
        for m in session.exec(select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == pf.id)).all():
            session.delete(m)
        session.delete(pf)
    for hh in session.exec(select(CensusHousehold).where(CensusHousehold.census_sheet_id == sheet.id)).all():
        session.delete(hh)
    for old_ov in session.exec(select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == sheet.id)).all():
        session.delete(old_ov)
    session.commit()

    scans_to_process: list[tuple[ScanImage, str]] = []
    seen_scan_ids: set[int] = set()
    if front_scan and front_scan.original_path and Path(front_scan.original_path).exists():
        scans_to_process.append((front_scan, "front"))
        if front_scan.id is not None:
            seen_scan_ids.add(front_scan.id)
    if back_scan and back_scan.original_path and Path(back_scan.original_path).exists():
        scans_to_process.append((back_scan, "back"))
        if back_scan.id is not None:
            seen_scan_ids.add(back_scan.id)

    if sheet.id is not None:
        extra_scans = session.exec(
            select(ScanImage)
            .where(ScanImage.census_sheet_id == sheet.id)
            .order_by(ScanImage.id.asc())
        ).all()
        for es in extra_scans:
            if es.id is not None and es.id not in seen_scan_ids and es.original_path and Path(es.original_path).exists():
                scans_to_process.append((es, es.page_side or f"page_{len(scans_to_process) + 1}"))
                seen_scan_ids.add(es.id)

    # Pre-load all household sheets in the same canonical municipality & census_year for fast house_number matching
    if sheet.municipality:
        sheet.municipality = canonical_municipality_display(sheet.municipality) or sheet.municipality
        sheet.municipality_folded = canonical_municipality_key(sheet.municipality)
        session.add(sheet)
    muni_canon = canonical_municipality_key(sheet.municipality)
    candidate_sheets = session.exec(
        select(CensusSheet).where(
            CensusSheet.census_year == sheet.census_year,
            CensusSheet.id != sheet.id,
        )
    ).all()
    sheet_by_house_num: dict[str, CensusSheet] = {}
    for cs in candidate_sheets:
        if cs.sheet_type == "overview":
            continue
        if not muni_canon or canonical_municipality_key(cs.municipality) != muni_canon:
            continue
        hn_key = normalize_house_number(cs.house_number)
        if hn_key and hn_key not in sheet_by_house_num:
            sheet_by_house_num[hn_key] = cs

    # Stage 1: Slice all scans, persist RecordRows per scan, and run ALL GLM-OCR calls while glm-ocr stays warm in VRAM
    # Queue items: (booklet_page, scan_order, row_index, scan_obj, r_info, record_row, combined_ocr)
    raw_ocr_items: list[tuple[int, int, int, ScanImage, Any, RecordRow, str]] = []

    for scan_order, (scan_obj, page_side) in enumerate(scans_to_process):
        layout_res = await asyncio.to_thread(
            slice_census_overview_sheet,
            Path(scan_obj.original_path),
            scan_obj.id or 0,
            Path(settings.crops_dir),
            page_side=page_side,
            census_year=sheet.census_year,
        )
        scan_obj.width = layout_res.width
        scan_obj.height = layout_res.height
        scan_obj.spine_x = 0
        scan_obj.header_crop_path = layout_res.header_crop_rel
        scan_obj.overlay_path = layout_res.overlay_rel
        scan_obj.grid_metadata_json = layout_res.to_json()
        scan_obj.status = "sliced"
        session.add(scan_obj)
        session.commit()

        # Persist RecordRow rows for this scan so UI and DB row counts reflect all sliced overview rows
        existing_rows = session.exec(
            select(RecordRow).where(RecordRow.scan_id == scan_obj.id).order_by(RecordRow.row_index)
        ).all()
        existing_by_idx = {er.row_index: er for er in existing_rows}
        new_indices = {r_info.row_index for r_info in layout_res.rows}
        for er in existing_rows:
            if er.row_index not in new_indices:
                for qi in session.exec(select(QuestionItem).where(QuestionItem.row_id == er.id)).all():
                    session.delete(qi)
                session.delete(er)
        session.commit()

        scan_record_rows: dict[int, RecordRow] = {}
        for r_info in layout_res.rows:
            r_rec = existing_by_idx.get(r_info.row_index)
            if r_rec is None:
                r_rec = RecordRow(
                    scan_id=scan_obj.id or 0,
                    row_index=r_info.row_index,
                    status="sliced",
                    record_type="census",
                )
            r_rec.y_start = r_info.y_start
            r_rec.y_end = r_info.y_end
            r_rec.left_crop_path = r_info.left_crop_rel
            r_rec.right_crop_path = r_info.right_crop_rel
            r_rec.stitched_crop_path = r_info.stitched_crop_rel
            r_rec.cell_crops_json = json.dumps(r_info.cell_crops, ensure_ascii=False)
            r_rec.record_type = "census"
            session.add(r_rec)
            scan_record_rows[r_info.row_index] = r_rec
        session.commit()
        for r_rec in scan_record_rows.values():
            session.refresh(r_rec)

        # Check existing cached WorkflowTrace OCR on this scan ONLY if grid_metadata_json had identical row count
        cached_traces = session.exec(
            select(WorkflowTrace)
            .where(WorkflowTrace.scan_id == scan_obj.id, WorkflowTrace.status == "ok")
            .order_by(WorkflowTrace.id.desc())
        ).all()
        trace_by_img: dict[str, str] = {}
        for tr in cached_traces:
            if tr.image_path and tr.response_text and tr.image_path not in trace_by_img:
                trace_by_img[tr.image_path] = tr.response_text

        # Header OCR on front page using Text Recognition to capture obvod_number & commissioner_name
        if page_side == "front" and layout_res.header_crop_rel:
            h_path = media_rel_to_path(layout_res.header_crop_rel)
            cached_h = trace_by_img.get(layout_res.header_crop_rel)
            if cached_h and ("prehľad" in cached_h.lower() or "prehlad" in cached_h.lower() or "komisár" in cached_h.lower() or "komisar" in cached_h.lower()):
                h_text = html_table_to_pipe_text(cached_h)
            elif h_path and h_path.exists():
                res_h = await ai_client.chat_completion(
                    stage="ocr_header",
                    model=settings.model_ocr,
                    prompt="Text Recognition:",
                    image_path=h_path,
                    image_rel_url=layout_res.header_crop_rel,
                    scan_id=scan_obj.id,
                    max_tokens=500,
                )
                h_text = html_table_to_pipe_text(res_h["content"])
            else:
                h_text = ""
            if h_text:
                scan_obj.header_raw_ocr = h_text
                meta = extract_overview_header_metadata(h_text)
                if meta.get("obvod_number"):
                    sheet.obvod_number = meta["obvod_number"]
                if meta.get("commissioner_name"):
                    comm = meta["commissioner_name"]
                    if any(k in comm.lower() for k in ("pánošký", "panosky", "ránošký", "ranosky")):
                        comm = "Rudolf Pánovský"
                    sheet.commissioner_name = comm
                session.add(scan_obj)
                session.add(sheet)
                session.commit()

        for r_info in layout_res.rows:
            r_rec = scan_record_rows[r_info.row_index]
            l_path = media_rel_to_path(r_info.left_crop_rel)
            r_path = media_rel_to_path(r_info.right_crop_rel)
            combined_ocr = ""
            cached_row_ocr = html_table_to_pipe_text(trace_by_img.get(r_info.left_crop_rel, "")).strip(" |")
            if cached_row_ocr and r_rec.y_start == r_info.y_start and r_rec.y_end == r_info.y_end:
                combined_ocr = cached_row_ocr
            elif l_path and l_path.exists():
                res_l = await ai_client.chat_completion(
                    stage="ocr_left",
                    model=settings.model_ocr,
                    prompt="Table Recognition:",
                    image_path=l_path,
                    image_rel_url=r_info.left_crop_rel,
                    scan_id=scan_obj.id,
                    row_id=r_rec.id,
                    max_tokens=360,
                )
                combined_ocr = html_table_to_pipe_text(res_l["content"]).strip(" |")

            # Fallback if Table Recognition on full strip missed the owner name or returned empty
            heur_probe = _heuristic_parse_overview_row(
                combined_ocr,
                default_ordinal=r_info.row_index,
                default_settlement=sheet.municipality,
                prev_house=None,
            )
            if (not combined_ocr or not heur_probe.get("owner_name")) and r_path and r_path.exists():
                cached_r_ocr = html_table_to_pipe_text(trace_by_img.get(r_info.right_crop_rel, "")).strip(" |")
                if cached_r_ocr:
                    right_ocr = cached_r_ocr
                else:
                    res_r = await ai_client.chat_completion(
                        stage="ocr_right",
                        model=settings.model_ocr,
                        prompt="Table Recognition:",
                        image_path=r_path,
                        image_rel_url=r_info.right_crop_rel,
                        scan_id=scan_obj.id,
                        row_id=r_rec.id,
                        max_tokens=300,
                    )
                    right_ocr = html_table_to_pipe_text(res_r["content"]).strip(" |")
                if not right_ocr and l_path and l_path.exists():
                    res_txt = await ai_client.chat_completion(
                        stage="ocr_left",
                        model=settings.model_ocr,
                        prompt="Text Recognition:",
                        image_path=l_path,
                        image_rel_url=r_info.left_crop_rel,
                        scan_id=scan_obj.id,
                        row_id=r_rec.id,
                        max_tokens=300,
                    )
                    right_ocr = html_table_to_pipe_text(res_txt["content"]).strip(" |")
                if right_ocr:
                    combined_ocr = f"{combined_ocr} | {right_ocr}".strip(" |") if combined_ocr else right_ocr

            r_rec.raw_ocr_left = combined_ocr
            r_rec.raw_ocr_combined = combined_ocr
            r_rec.status = "ocr_done"
            session.add(r_rec)
            session.commit()

            booklet_page = 1 if page_side == "front" else (2 if page_side == "back" else scan_order + 1)
            if r_info.cell_crops and isinstance(r_info.cell_crops[0], dict):
                bp_val = r_info.cell_crops[0].get("booklet_page")
                if isinstance(bp_val, int):
                    booklet_page = bp_val
            raw_ocr_items.append((booklet_page, scan_order, r_info.row_index, scan_obj, r_info, r_rec, combined_ocr))

        scan_obj.status = "ocr_done"
        session.add(scan_obj)
        session.commit()

    # Sort all rows across all pages in true booklet page order (Page 1 -> Page 2 -> Page 3 -> Page 4 -> ...)
    raw_ocr_items.sort(key=lambda item: (item[0], item[1], item[2]))

    # Stage 2: Extract structured CensusOverviewHouse records in sequential order
    created_houses: list[CensusOverviewHouse] = []
    prev_house: CensusOverviewHouse | None = None
    seq_house_num = 0

    for booklet_page, _scan_order, _scan_row_idx, scan_obj, r_info, r_rec, combined_ocr in raw_ocr_items:
        if not combined_ocr or len(re.sub(r"[^A-Za-z0-9Á-ž]", "", combined_ocr)) < 2:
            continue

        low_comb = combined_ocr.lower()
        if any(
            tok in low_comb
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
                "všetko spolu",
                "vsetko spolu",
            )
        ):
            continue

        expected_ord = seq_house_num + 1
        heur = _heuristic_parse_overview_row(
            combined_ocr,
            default_ordinal=expected_ord,
            default_settlement=sheet.municipality,
            prev_house=prev_house,
        )

        # Skip summary rows that have no owner name and only huge 2-3 digit totals
        if not heur.get("owner_name"):
            nums_in_row = [int(x) for x in re.findall(r"\b\d+\b", combined_ocr)]
            if nums_in_row and max(nums_in_row) >= 80 and not any(ch.isalpha() for ch in combined_ocr):
                continue

        parsed: dict[str, Any] = {}
        if not heur.get("owner_name"):
            prompt = (
                f"Transcribe row #{expected_ord} (House #{heur.get('house_number')}) of Prehľad sčítacieho obvodu (Municipality: {sheet.municipality or 'Unknown'}).\n"
                f"Previous row settlement: {prev_house.settlement_name if prev_house else (sheet.municipality or 'dedina')}, "
                f"Previous row owner: {prev_house.owner_name if prev_house else 'None'}.\n"
                f"Heuristic hints: ordinal={heur.get('ordinal_number')}, house={heur.get('house_number')}, owner={heur.get('owner_name')}\n"
                f"OCR text:\n{combined_ocr}"
            )
            res_ext = await ai_client.chat_completion(
                stage="extract",
                model=settings.model_extract,
                system_prompt=CENSUS_OVERVIEW_SYSTEM_PROMPT,
                prompt=prompt,
                scan_id=scan_obj.id,
                row_id=r_rec.id,
                max_tokens=200,
                temperature=0.0,
                timeout_sec=35.0,
                skip_thinking=True,
                response_model=LLMCensusOverviewRowSchema,
            )
            parsed = res_ext.get("parsed") or {}

        owner = _normalize_overview_owner(
            heur.get("owner_name") or parsed.get("owner_name"),
            prev_house.owner_name if prev_house else None,
        )
        house_no_cand = str(heur.get("house_number") or parsed.get("house_number") or expected_ord).strip()
        if not owner and not re.search(r"\d", house_no_cand):
            continue

        seq_house_num += 1
        ord_num = seq_house_num
        settlement = (
            heur.get("settlement_name")
            or parsed.get("settlement_name")
            or (prev_house.settlement_name if prev_house else (sheet.municipality or "dedina"))
        )
        if _is_census_ditto(settlement) or str(settlement).lower() in ("-11-", "- 11 -", "medina"):
            settlement = (prev_house.settlement_name if prev_house else None) or sheet.municipality or "dedina"

        house_no = str(heur.get("house_number") or parsed.get("house_number") or ord_num).strip()
        if _is_census_ditto(house_no) and prev_house and prev_house.house_number:
            house_no = prev_house.house_number
        if house_no.isdigit() and abs(int(house_no) - ord_num) > 5 and ord_num <= 100:
            house_no = str(ord_num)

        m_int = re.search(r"(\d+)", house_no)
        house_no_int = int(m_int.group(1)) if m_int else None
        hn_lookup = normalize_house_number(house_no)
        matched_sheet = sheet_by_house_num.get(hn_lookup)

        owner_surname_folded = owner_surname_folded_from_name(owner)

        is_inhab = heur.get("is_inhabited", True)

        hh_cnt = heur.get("household_count") if heur.get("household_count") is not None else parsed.get("household_count")
        pp_cnt = heur.get("present_persons_count") if heur.get("present_persons_count") is not None else parsed.get("present_persons_count")

        ov_house = CensusOverviewHouse(
            census_sheet_id=sheet.id or 0,
            scan_id=scan_obj.id,
            row_index=ord_num,
            census_year=sheet.census_year,
            district=sheet.district,
            municipality=canonical_municipality_display(sheet.municipality) or sheet.municipality,
            obvod_number=sheet.obvod_number,
            ordinal_number=str(ord_num),
            settlement_name=settlement,
            house_number=house_no,
            house_number_int=house_no_int,
            is_inhabited=bool(is_inhab),
            owner_name=owner,
            owner_surname_folded=owner_surname_folded,
            household_count=int(hh_cnt) if hh_cnt is not None else None,
            present_persons_count=int(pp_cnt) if pp_cnt is not None else None,
            remarks=heur.get("remarks") or parsed.get("remarks"),
            matched_sheet_id=None,
            stitched_crop_path=r_info.stitched_crop_rel,
            raw_ocr_text=combined_ocr,
            confidence=float(parsed.get("confidence") or 0.92),
        )
        if matched_sheet and can_match_overview_house_to_sheet(ov_house, matched_sheet, overview_sheet=sheet):
            ov_house.matched_sheet_id = matched_sheet.id
        session.add(ov_house)

        r_rec.entry_number = str(ord_num)
        r_rec.status = "extracted"
        r_rec.normalized_json = json.dumps(
            {
                "ordinal_number": ord_num,
                "booklet_page": booklet_page,
                "settlement_name": settlement,
                "house_number": house_no,
                "owner_name": owner,
                "household_count": ov_house.household_count,
                "present_persons_count": ov_house.present_persons_count,
            },
            ensure_ascii=False,
        )
        session.add(r_rec)
        session.commit()
        session.refresh(ov_house)
        created_houses.append(ov_house)
        prev_house = ov_house

    for scan_obj, _ in scans_to_process:
        scan_obj.status = "extracted"
        session.add(scan_obj)
    sheet.download_status = "extracted"
    session.add(sheet)
    session.commit()
    index_census_sheet(session, sheet)
    session.commit()
    return created_houses

