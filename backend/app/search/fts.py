"""
SQLite FTS5 Virtual Table & Multi-Faceted Phonetic Search Engine.
Indexes both 1.25M Slovakiana CensusSheet catalog entries and PersonFact rows
(from parish registers and census household extractions).
"""

from __future__ import annotations

import re
from typing import Any
from sqlalchemy import Engine, text
from sqlmodel import Session, select

from backend.app.db.models import (
    CensusHousehold,
    CensusOverviewHouse,
    CensusSheet,
    FamilySearchMatch,
    LocalRecordMatch,
    PersonFact,
    RecordRow,
    ScanImage,
)
from backend.app.pipeline.lexicon import (
    _apply_digraphs,
    _paleographic_similarity,
    _phonetic_skeleton,
    apply_person_fact_search_fields,
    compute_person_fact_search_fields,
    expand_given_name_equivalents,
    extract_person_fact_name_candidates,
    fold_text_diacritics,
    fuzzy_name_match_score,
    fuzzy_name_matches,
    fuzzy_token_similarity,
    get_given_name_equivalents,
    orthographic_fold,
    parse_house_number_int,
    strip_slovak_feminine_suffix,
    weighted_edit_similarity,
)


def ensure_fts_schema(engine: Engine) -> None:
    """Creates the FTS5 virtual table for unified person & census sheet search if missing."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index_fts USING fts5(
                    entity_type UNINDEXED,
                    entity_id UNINDEXED,
                    names_text,
                    surnames_phonetic,
                    places_text,
                    house_and_sheet,
                    occupation_and_religion,
                    notes_and_ocr,
                    tokenize='unicode61 remove_diacritics 2'
                );
                """
            )
        )


def _build_person_fts_row(fact: PersonFact, ocr_text: str | None = None) -> dict[str, Any]:
    given_tokens = get_given_name_equivalents(fact.given_name_raw or fact.given_name_sk)
    surname_raw = fact.surname_raw or ""
    surname_norm = fact.surname_normalized or surname_raw
    surname_stem = strip_slovak_feminine_suffix(surname_norm) or surname_norm
    maiden = fact.mother_maiden_surname or ""
    maiden_stem = strip_slovak_feminine_suffix(maiden) or maiden

    names_parts = [
        fact.given_name_raw or "",
        fact.given_name_sk or "",
        fact.given_name_hu or "",
        " ".join(given_tokens),
        surname_raw,
        surname_norm,
        surname_stem,
        fact.father_name or "",
        fact.mother_name or "",
        maiden,
        maiden_stem,
        fact.spouse_name or "",
        *extract_person_fact_name_candidates(fact),
    ]
    phonetic_parts = []
    for s in (surname_raw, surname_norm, surname_stem, maiden, maiden_stem, *extract_person_fact_name_candidates(fact)):
        if s:
            phonetic_parts.append(_apply_digraphs(s))
            phonetic_parts.append(orthographic_fold(s))
            phonetic_parts.append(_phonetic_skeleton(s))

    places_parts = [
        fact.village_raw or "",
        fact.village_sk or "",
        fact.village_hu or "",
        fact.fs_place_standardized or "",
        fact.birth_place_raw or "",
        fact.birth_place_sk or "",
        fact.home_jurisdiction or "",
        fact.previous_residence or "",
    ]
    house_parts = [
        fact.house_number or "",
        str(fact.house_number_int or ""),
        fact.entry_number or "",
        fact.person_row_number or "",
        str(fact.event_year or ""),
        str(fact.birth_year_est or ""),
        fact.birth_or_event_date or "",
    ]
    occ_parts = [
        fact.social_status_or_occupation or "",
        fact.occupation_position or "",
        fact.employer_workplace or "",
        fact.religion or "",
        fact.nationality or "",
        fact.relation_to_head or "",
        fact.marital_status or "",
    ]
    notes_parts = [
        fact.observationes_raw or "",
        ocr_text or "",
    ]
    return {
        "entity_type": "person_fact",
        "entity_id": str(fact.id),
        "names_text": " ".join(p for p in names_parts if p),
        "surnames_phonetic": " ".join(p for p in phonetic_parts if p),
        "places_text": " ".join(p for p in places_parts if p),
        "house_and_sheet": " ".join(p for p in house_parts if p),
        "occupation_and_religion": " ".join(p for p in occ_parts if p),
        "notes_and_ocr": " ".join(p for p in notes_parts if p),
    }


def _build_sheet_fts_row(sheet: CensusSheet) -> dict[str, Any]:
    return {
        "entity_type": "census_sheet",
        "entity_id": str(sheet.id),
        "names_text": sheet.household_owner or "",
        "surnames_phonetic": (
            f"{_apply_digraphs(sheet.household_owner)} {_phonetic_skeleton(sheet.household_owner)}"
            if sheet.household_owner
            else ""
        ),
        "places_text": " ".join(
            p
            for p in [
                sheet.municipality,
                sheet.municipality_folded,
                sheet.district or "",
                sheet.district_folded or "",
                sheet.street_raw or "",
            ]
            if p
        ),
        "house_and_sheet": " ".join(
            p
            for p in [
                sheet.house_number or "",
                str(sheet.house_number_int or ""),
                sheet.box_number or "",
                sheet.sheet_number or "",
                str(sheet.census_year),
                sheet.slovakiana_key,
            ]
            if p
        ),
        "occupation_and_religion": "scitaci harok census",
        "notes_and_ocr": sheet.title_raw or "",
    }


def index_person_fact(session: Session, fact: PersonFact, ocr_text: str | None = None) -> None:
    """Updates pre-computed search columns on PersonFact and upserts its FTS5 document."""
    if fact.id is None:
        return
    apply_person_fact_search_fields(fact)
    session.add(fact)

    row_data = _build_person_fts_row(fact, ocr_text=ocr_text)
    session.exec(
        text(
            "DELETE FROM search_index_fts WHERE entity_type = 'person_fact' AND entity_id = :eid"
        ).bindparams(eid=str(fact.id))
    )
    session.exec(
        text(
            """
            INSERT INTO search_index_fts (
                entity_type, entity_id, names_text, surnames_phonetic,
                places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
            ) VALUES (
                :entity_type, :entity_id, :names_text, :surnames_phonetic,
                :places_text, :house_and_sheet, :occupation_and_religion, :notes_and_ocr
            )
            """
        ).bindparams(**row_data)
    )


def index_census_sheet(session: Session, sheet: CensusSheet) -> None:
    """Upserts a CensusSheet into the FTS5 index."""
    if sheet.id is None:
        return
    row_data = _build_sheet_fts_row(sheet)
    session.exec(
        text(
            "DELETE FROM search_index_fts WHERE entity_type = 'census_sheet' AND entity_id = :eid"
        ).bindparams(eid=str(sheet.id))
    )
    session.exec(
        text(
            """
            INSERT INTO search_index_fts (
                entity_type, entity_id, names_text, surnames_phonetic,
                places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
            ) VALUES (
                :entity_type, :entity_id, :names_text, :surnames_phonetic,
                :places_text, :house_and_sheet, :occupation_and_religion, :notes_and_ocr
            )
            """
        ).bindparams(**row_data)
    )


def bulk_index_census_sheets(session: Session, sheets: list[CensusSheet]) -> None:
    """Fast batch upsert of CensusSheet records into FTS5 during catalog harvesting."""
    valid_sheets = [s for s in sheets if s.id is not None]
    if not valid_sheets:
        return
    ids = [str(s.id) for s in valid_sheets]
    placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
    params = {f"id_{i}": eid for i, eid in enumerate(ids)}
    session.exec(
        text(
            f"DELETE FROM search_index_fts WHERE entity_type = 'census_sheet' AND entity_id IN ({placeholders})"
        ).bindparams(**params)
    )
    for sheet in valid_sheets:
        row_data = _build_sheet_fts_row(sheet)
        session.exec(
            text(
                """
                INSERT INTO search_index_fts (
                    entity_type, entity_id, names_text, surnames_phonetic,
                    places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
                ) VALUES (
                    :entity_type, :entity_id, :names_text, :surnames_phonetic,
                    :places_text, :house_and_sheet, :occupation_and_religion, :notes_and_ocr
                )
                """
            ).bindparams(**row_data)
        )


def rebuild_all_fts(session: Session) -> dict[str, int]:
    """Rebuilds the entire FTS5 search index from PersonFact and CensusSheet tables."""
    session.exec(text("DELETE FROM search_index_fts"))
    facts = session.exec(select(PersonFact)).all()
    rows_map = {r.id: r for r in session.exec(select(RecordRow)).all() if r.id is not None}
    for f in facts:
        r = rows_map.get(f.row_id)
        index_person_fact(session, f, ocr_text=r.raw_ocr_combined if r else None)

    sheets = session.exec(select(CensusSheet)).all()
    for s in sheets:
        index_census_sheet(session, s)

    session.commit()
    return {"indexed_person_facts": len(facts), "indexed_census_sheets": len(sheets)}


def build_fts_match_expression(query: str, fuzzy: bool = True) -> str:
    """Builds an FTS5 MATCH query with diacritic stripping, prefix matching, and phonetic/synonym expansion."""
    raw_tokens = re.findall(r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ\-/]+", query)
    clauses: list[str] = []
    for tok in raw_tokens:
        clean = re.sub(r"[^A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]", "", tok)
        if not clean:
            continue
        folded = fold_text_diacritics(clean)
        variants: set[str] = {folded, f"{folded}*"}
        if clean.isalpha() and len(clean) >= 3:
            stem = strip_slovak_feminine_suffix(clean)
            if stem:
                stem_folded = fold_text_diacritics(stem)
                variants.add(stem_folded)
                variants.add(f"{stem_folded}*")
            if fuzzy:
                digraph = _apply_digraphs(stem or clean)
                orth = orthographic_fold(stem or clean)
                skel = _phonetic_skeleton(stem or clean)
                if len(digraph) >= 3:
                    variants.add(digraph)
                    variants.add(f"{digraph}*")
                if len(orth) >= 3:
                    variants.add(orth)
                    variants.add(f"{orth}*")
                if len(skel) >= 3:
                    variants.add(skel)
            equivs = get_given_name_equivalents(clean)
            for eq in equivs:
                eq_f = re.sub(r"[^a-z0-9]", "", fold_text_diacritics(eq))
                if len(eq_f) >= 2:
                    variants.add(eq_f)
        safe_variants = [re.sub(r"[^a-z0-9*]", "", v) for v in variants if re.sub(r"[^a-z0-9]", "", v)]
        if safe_variants:
            clauses.append("(" + " OR ".join(sorted(set(safe_variants))) + ")")
    return " AND ".join(clauses)


def search_unified(
    session: Session,
    *,
    q: str | None = None,
    record_type: str | None = None,
    scan_id: int | None = None,
    census_sheet_id: int | None = None,
    municipality: str | None = None,
    district: str | None = None,
    surname: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    birth_year_from: int | None = None,
    birth_year_to: int | None = None,
    house_number: str | None = None,
    occupation: str | None = None,
    religion: str | None = None,
    fs_status: str | None = None,
    include_catalog_sheets: bool = True,
    fuzzy: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Executes unified FTS5 + B-tree faceted search across PersonFact and CensusSheet."""
    stmt = select(PersonFact)

    if record_type and record_type != "all":
        stmt = stmt.where(PersonFact.record_type == record_type)
    if scan_id is not None:
        stmt = stmt.where(PersonFact.scan_id == scan_id)
    if census_sheet_id is not None:
        stmt = stmt.where(PersonFact.census_sheet_id == census_sheet_id)
    if year_from is not None:
        stmt = stmt.where(PersonFact.event_year >= year_from)
    if year_to is not None:
        stmt = stmt.where(PersonFact.event_year <= year_to)
    if birth_year_from is not None:
        stmt = stmt.where(PersonFact.birth_year_est >= birth_year_from)
    if birth_year_to is not None:
        stmt = stmt.where(PersonFact.birth_year_est <= birth_year_to)

    if municipality and municipality.strip():
        m_fold = fold_text_diacritics(municipality)
        stmt = stmt.where(
            (PersonFact.village_folded.contains(m_fold))  # type: ignore[union-attr]
            | (PersonFact.village_sk.ilike(f"%{municipality.strip()}%"))  # type: ignore[union-attr]
            | (PersonFact.village_raw.ilike(f"%{municipality.strip()}%"))  # type: ignore[union-attr]
        )

    if house_number and house_number.strip():
        h_int = parse_house_number_int(house_number)
        if h_int is not None:
            stmt = stmt.where(
                (PersonFact.house_number_int == h_int)
                | (PersonFact.house_number.ilike(f"%{house_number.strip()}%"))  # type: ignore[union-attr]
            )
        else:
            stmt = stmt.where(PersonFact.house_number.ilike(f"%{house_number.strip()}%"))  # type: ignore[union-attr]

    if occupation and occupation.strip():
        stmt = stmt.where(
            PersonFact.social_status_or_occupation.ilike(f"%{occupation.strip()}%")  # type: ignore[union-attr]
        )

    if religion and religion.strip():
        stmt = stmt.where(PersonFact.religion.ilike(f"%{religion.strip()}%"))  # type: ignore[union-attr]

    if surname and surname.strip() and not fuzzy:
        s_stem = strip_slovak_feminine_suffix(surname.strip()) or surname.strip()
        s_fold = _apply_digraphs(s_stem)
        stmt = stmt.where(
            (PersonFact.surname_folded.contains(s_fold))  # type: ignore[union-attr]
            | (PersonFact.surname_normalized.ilike(f"%{surname.strip()}%"))  # type: ignore[union-attr]
        )

    matched_sheet_ids: list[int] = []
    fact_scores: dict[int, float] = {}
    all_candidates = list(session.exec(stmt.order_by(PersonFact.id.desc())).all())

    if surname and surname.strip() and fuzzy:
        s_query = surname.strip()
        surname_filtered: list[PersonFact] = []
        for f in all_candidates:
            s_cands = [
                f.surname_raw,
                f.surname_normalized,
                f.mother_maiden_surname,
                *extract_person_fact_name_candidates(f),
            ]
            sc = fuzzy_name_match_score(s_query, s_cands, threshold=0.76)
            if sc >= 0.76:
                surname_filtered.append(f)
                if f.id is not None:
                    fact_scores[f.id] = sc
        all_candidates = surname_filtered

    if q and q.strip():
        q_clean = q.strip()
        fts_expr = build_fts_match_expression(q_clean, fuzzy=fuzzy)
        fts_person_ids: list[int] = []
        if fts_expr:
            try:
                rows_fts = session.exec(
                    text(
                        """
                        SELECT entity_type, entity_id
                        FROM search_index_fts
                        WHERE search_index_fts MATCH :expr
                        ORDER BY rank
                        LIMIT 2000
                        """
                    ).bindparams(expr=fts_expr)
                ).all()
                for etype, eid in rows_fts:
                    if etype == "person_fact" and str(eid).isdigit():
                        fts_person_ids.append(int(eid))
                    elif etype == "census_sheet" and str(eid).isdigit():
                        matched_sheet_ids.append(int(eid))
            except Exception:
                pass

        cand_row_ids = list({f.row_id for f in all_candidates if f.row_id})
        row_ocr_map: dict[int, str] = {}
        if cand_row_ids:
            for r_obj in session.exec(select(RecordRow).where(RecordRow.id.in_(cand_row_ids))).all():  # type: ignore[union-attr]
                if r_obj.id is not None:
                    ocr_txt = " ".join(
                        p for p in (r_obj.raw_ocr_combined, r_obj.raw_ocr_left, r_obj.raw_ocr_right) if p
                    )
                    if ocr_txt:
                        row_ocr_map[r_obj.id] = ocr_txt

        needle = fold_text_diacritics(q_clean)
        fts_id_set = set(fts_person_ids)
        threshold = 0.76 if fuzzy else 0.90
        scored_candidates: list[tuple[float, int, PersonFact]] = []

        for f in all_candidates:
            ocr_txt = row_ocr_map.get(f.row_id)
            name_cands = extract_person_fact_name_candidates(f, ocr_text=ocr_txt)
            haystack_parts = [
                *name_cands,
                f.village_raw or "",
                f.village_sk or "",
                f.village_hu or "",
                f.fs_place_standardized or "",
                f.birth_place_raw or "",
                f.birth_place_sk or "",
                f.birth_place_municipality or "",
                f.home_jurisdiction or "",
                f.social_status_or_occupation or "",
                f.occupation_position or "",
                f.religion or "",
                f.observationes_raw or "",
                f.house_number or "",
                str(f.event_year or ""),
                str(f.birth_year_est or ""),
            ]
            haystack = fold_text_diacritics(" ".join(p for p in haystack_parts if p))

            score = 0.0
            if needle and needle in haystack:
                score = 1.0
            else:
                f_score = fuzzy_name_match_score(
                    q_clean,
                    [
                        *name_cands,
                        f.village_raw,
                        f.village_sk,
                        f.birth_place_raw,
                        f.birth_place_sk,
                        f.birth_place_municipality,
                        f.social_status_or_occupation,
                        f.house_number,
                    ],
                    threshold=threshold,
                )
                if f_score >= threshold:
                    score = f_score
                elif f.id in fts_id_set:
                    # Matched via FTS5 non-name columns (notes, occupation, etc.)
                    q_toks = [
                        fold_text_diacritics(t)
                        for t in re.findall(r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+", q_clean)
                        if t
                    ]
                    if q_toks and all(any(qt in w for w in haystack.split()) for qt in q_toks):
                        score = 0.88

            if score >= threshold:
                if f.id is not None:
                    fact_scores[f.id] = score
                if f.census_sheet_id and f.census_sheet_id not in matched_sheet_ids:
                    matched_sheet_ids.append(f.census_sheet_id)
                scored_candidates.append((score, f.id or 0, f))

        scored_candidates.sort(key=lambda item: (round(item[0], 3), item[1]), reverse=True)
        facts = [item[2] for item in scored_candidates]
    else:
        facts = all_candidates

    # Filter by FamilySearch match status if requested
    fact_ids = [f.id for f in facts if f.id is not None]
    fs_matches = (
        session.exec(select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id.in_(fact_ids))).all()  # type: ignore[union-attr]
        if fact_ids
        else []
    )
    fs_map = {
        m.person_fact_id: m
        for m in fs_matches
        if m.fs_person_id and str(m.fs_person_id).strip() and (m.confidence_score or 0.0) > 0
    }
    if fs_status and fs_status != "all":
        facts = [
            f
            for f in facts
            if f.id is not None
            and ((fs_map.get(f.id).match_status if fs_map.get(f.id) else "unmatched") == fs_status)
        ]

    total_facts = len(facts)
    paged_facts = facts[offset : offset + limit]

    row_ids = list({f.row_id for f in paged_facts})
    scan_ids = list({f.scan_id for f in paged_facts})
    rows = {
        r.id: r
        for r in (
            session.exec(select(RecordRow).where(RecordRow.id.in_(row_ids))).all()  # type: ignore[union-attr]
            if row_ids
            else []
        )
    }
    scans = {
        s.id: s
        for s in (
            session.exec(select(ScanImage).where(ScanImage.id.in_(scan_ids))).all()  # type: ignore[union-attr]
            if scan_ids
            else []
        )
    }
    local_matches = (
        session.exec(
            select(LocalRecordMatch).where(
                (LocalRecordMatch.census_fact_id.in_([f.id for f in paged_facts if f.id]))  # type: ignore[union-attr]
                | (LocalRecordMatch.matrika_fact_id.in_([f.id for f in paged_facts if f.id]))  # type: ignore[union-attr]
            )
        ).all()
        if paged_facts
        else []
    )
    local_match_map: dict[int, list[dict[str, Any]]] = {}
    for lm in local_matches:
        local_match_map.setdefault(lm.census_fact_id, []).append(
            {
                "id": lm.id,
                "matched_fact_id": lm.matrika_fact_id,
                "confidence_score": lm.confidence_score,
                "match_kind": lm.match_kind,
            }
        )
        local_match_map.setdefault(lm.matrika_fact_id, []).append(
            {
                "id": lm.id,
                "matched_fact_id": lm.census_fact_id,
                "confidence_score": lm.confidence_score,
                "match_kind": "census",
            }
        )

    records_out: list[dict[str, Any]] = []
    from backend.app.api.scans import resolve_scan_media_url

    for f in paged_facts:
        r = rows.get(f.row_id)
        s = scans.get(f.scan_id)
        fs = fs_map.get(f.id) if f.id is not None else None
        m_score = fact_scores.get(f.id or -1, 1.0)
        records_out.append(
            {
                "fact": f,
                "scan_filename": s.filename if s else None,
                "scan_url": resolve_scan_media_url(s),
                "header_crop_url": s.header_crop_path if s else None,
                "stitched_crop_url": r.stitched_crop_path if r else None,
                "decision_confidence": r.decision_confidence if r else 0.0,
                "fs_match": fs,
                "local_matches": local_match_map.get(f.id or -1, []),
                "match_score": round(m_score, 4),
                "fuzzy_match": bool(m_score < 0.96),
            }
        )

    # Also query cataloged CensusSheets when include_catalog_sheets is enabled
    catalog_sheets_out: list[CensusSheet] = []
    if include_catalog_sheets and record_type in (None, "all", "census"):
        sheet_stmt = select(CensusSheet)
        if municipality and municipality.strip():
            m_fold = fold_text_diacritics(municipality)
            sheet_stmt = sheet_stmt.where(
                (CensusSheet.municipality_folded.contains(m_fold))  # type: ignore[union-attr]
                | (CensusSheet.municipality.ilike(f"%{municipality.strip()}%"))  # type: ignore[union-attr]
            )
        if district and district.strip():
            d_fold = fold_text_diacritics(district)
            sheet_stmt = sheet_stmt.where(
                (CensusSheet.district_folded.contains(d_fold))  # type: ignore[union-attr]
                | (CensusSheet.district.ilike(f"%{district.strip()}%"))  # type: ignore[union-attr]
            )
        if year_from is not None:
            sheet_stmt = sheet_stmt.where(CensusSheet.census_year >= year_from)
        if year_to is not None:
            sheet_stmt = sheet_stmt.where(CensusSheet.census_year <= year_to)
        if house_number and house_number.strip():
            h_int = parse_house_number_int(house_number)
            if h_int is not None:
                sheet_stmt = sheet_stmt.where(CensusSheet.house_number_int == h_int)
            else:
                sheet_stmt = sheet_stmt.where(
                    CensusSheet.house_number.ilike(f"%{house_number.strip()}%")  # type: ignore[union-attr]
                )
        if q and q.strip():
            q_clean = q.strip()
            q_fold = fold_text_diacritics(q_clean)
            if fuzzy:
                # Check CensusOverviewHouse owners & CensusSheet household_owner for fuzzy matches
                for ovh in session.exec(select(CensusOverviewHouse)).all():
                    if fuzzy_name_matches(q_clean, [ovh.owner_name, ovh.raw_ocr_text, ovh.remarks], threshold=0.76):
                        if ovh.census_sheet_id and ovh.census_sheet_id not in matched_sheet_ids:
                            matched_sheet_ids.append(ovh.census_sheet_id)
                        if ovh.matched_sheet_id and ovh.matched_sheet_id not in matched_sheet_ids:
                            matched_sheet_ids.append(ovh.matched_sheet_id)
                for cs_cand in session.exec(
                    select(CensusSheet).where(CensusSheet.household_owner.is_not(None))  # type: ignore[union-attr]
                ).all():
                    if cs_cand.id and cs_cand.id not in matched_sheet_ids:
                        if fuzzy_name_matches(q_clean, [cs_cand.household_owner], threshold=0.76):
                            matched_sheet_ids.append(cs_cand.id)

            if matched_sheet_ids:
                sheet_stmt = sheet_stmt.where(
                    (CensusSheet.id.in_(matched_sheet_ids[:200]))  # type: ignore[union-attr]
                    | (CensusSheet.municipality_folded.contains(q_fold))  # type: ignore[union-attr]
                    | (CensusSheet.title_raw.ilike(f"%{q_clean}%"))  # type: ignore[union-attr]
                    | (CensusSheet.street_raw.ilike(f"%{q_clean}%"))  # type: ignore[union-attr]
                    | (CensusSheet.slovakiana_key == q_clean)
                )
            else:
                sheet_stmt = sheet_stmt.where(
                    (CensusSheet.municipality_folded.contains(q_fold))  # type: ignore[union-attr]
                    | (CensusSheet.title_raw.ilike(f"%{q_clean}%"))  # type: ignore[union-attr]
                    | (CensusSheet.street_raw.ilike(f"%{q_clean}%"))  # type: ignore[union-attr]
                    | (CensusSheet.slovakiana_key == q_clean)
                )
        catalog_sheets_out = list(session.exec(sheet_stmt.order_by(CensusSheet.id.desc()).limit(30)).all())

    return {
        "total": total_facts,
        "records": records_out,
        "catalog_sheets": catalog_sheets_out,
    }


search_unified_records = search_unified
