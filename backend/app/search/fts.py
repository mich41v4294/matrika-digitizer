"""
SQLite FTS5 Virtual Table & Multi-Faceted Phonetic Search Engine.
Indexes both 1.25M Slovakiana CensusSheet catalog entries and PersonFact rows
(from parish registers and census household extractions).
"""

from __future__ import annotations

import re
import time
from typing import Any
from sqlalchemy import Engine, func, text
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
    normalize_street_query,
    orthographic_fold,
    parse_house_number_int,
    strip_slovak_feminine_suffix,
    text_matches_street,
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
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "bulk_index_census_sheets_rs"):
            _rust_db.bulk_index_census_sheets_rs(
                db_p, [int(s.id) for s in valid_sheets if s.id is not None]
            )
            return
    except Exception:
        pass
    ids = [str(s.id) for s in valid_sheets]
    for start in range(0, len(ids), 450):
        chunk_ids = ids[start : start + 450]
        placeholders = ", ".join(f":id_{i}" for i in range(len(chunk_ids)))
        params = {f"id_{i}": eid for i, eid in enumerate(chunk_ids)}
        session.exec(
            text(
                f"DELETE FROM search_index_fts WHERE entity_type = 'census_sheet' AND entity_id IN ({placeholders})"
            ).bindparams(**params)
        )
    rows_data = [_build_sheet_fts_row(sheet) for sheet in valid_sheets]
    insert_stmt = text(
        """
        INSERT INTO search_index_fts (
            entity_type, entity_id, names_text, surnames_phonetic,
            places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
        ) VALUES (
            :entity_type, :entity_id, :names_text, :surnames_phonetic,
            :places_text, :house_and_sheet, :occupation_and_religion, :notes_and_ocr
        )
        """
    )
    session.connection().execute(insert_stmt, rows_data)


def bulk_index_person_facts(
    session: Session,
    facts: list[PersonFact],
    ocr_by_row_id: dict[int, str] | None = None,
) -> None:
    """Fast batch upsert of PersonFact records into FTS5."""
    valid_facts = [f for f in facts if f.id is not None]
    if not valid_facts:
        return
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if (
            db_p
            and ocr_by_row_id is None
            and _rust_db is not None
            and hasattr(_rust_db, "bulk_index_person_facts_rs")
        ):
            for fact in valid_facts:
                apply_person_fact_search_fields(fact)
            _rust_db.bulk_index_person_facts_rs(
                db_p, [int(f.id) for f in valid_facts if f.id is not None]
            )
            return
    except Exception:
        pass
    ids = [str(f.id) for f in valid_facts]
    for start in range(0, len(ids), 450):
        chunk_ids = ids[start : start + 450]
        placeholders = ", ".join(f":id_{i}" for i in range(len(chunk_ids)))
        params = {f"id_{i}": eid for i, eid in enumerate(chunk_ids)}
        session.exec(
            text(
                f"DELETE FROM search_index_fts WHERE entity_type = 'person_fact' AND entity_id IN ({placeholders})"
            ).bindparams(**params)
        )
    ocr_map = ocr_by_row_id or {}
    rows_data: list[dict[str, Any]] = []
    for fact in valid_facts:
        apply_person_fact_search_fields(fact)
        session.add(fact)
        rows_data.append(_build_person_fts_row(fact, ocr_text=ocr_map.get(fact.row_id)))
    insert_stmt = text(
        """
        INSERT INTO search_index_fts (
            entity_type, entity_id, names_text, surnames_phonetic,
            places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
        ) VALUES (
            :entity_type, :entity_id, :names_text, :surnames_phonetic,
            :places_text, :house_and_sheet, :occupation_and_religion, :notes_and_ocr
        )
        """
    )
    session.connection().execute(insert_stmt, rows_data)


def rebuild_all_fts(session: Session) -> dict[str, int]:
    """Rebuilds the entire FTS5 search index from PersonFact and CensusSheet tables."""
    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if (
            db_p
            and _rust_db is not None
            and hasattr(_rust_db, "bulk_index_person_facts_rs")
            and hasattr(_rust_db, "bulk_index_census_sheets_rs")
        ):
            n_facts = int(_rust_db.bulk_index_person_facts_rs(db_p, None))
            n_sheets = int(_rust_db.bulk_index_census_sheets_rs(db_p, None))
            return {"indexed_person_facts": n_facts, "indexed_census_sheets": n_sheets}
    except Exception:
        pass
    session.exec(text("DELETE FROM search_index_fts"))
    facts = list(session.exec(select(PersonFact)).all())
    ocr_by_row_id = {
        r.id: r.raw_ocr_combined
        for r in session.exec(select(RecordRow)).all()
        if r.id is not None and r.raw_ocr_combined
    }
    bulk_index_person_facts(session, facts, ocr_by_row_id=ocr_by_row_id)

    sheets = list(session.exec(select(CensusSheet)).all())
    bulk_index_census_sheets(session, sheets)

    session.commit()
    return {"indexed_person_facts": len(facts), "indexed_census_sheets": len(sheets)}


def build_fts_match_expression(query: str, fuzzy: bool = True) -> str:
    """Builds an FTS5 MATCH query with diacritic stripping, prefix matching, and phonetic/synonym expansion."""
    try:
        from backend.app.pipeline.lexicon import _rust_lex

        if _rust_lex is not None and hasattr(_rust_lex, "build_fts_match_expression_rs"):
            return str(_rust_lex.build_fts_match_expression_rs(query, fuzzy))
    except Exception:
        pass
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


SEARCH_PROGRESS_STATE: dict[str, Any] = {
    "running": False,
    "scope": "unified",
    "query": "",
    "processed": 0,
    "total": 0,
    "matched_records": 0,
    "matched_sheets": 0,
    "progress_pct": 100,
    "stage_sk": "Vyhľadávanie pripravené",
    "stage_en": "Search ready",
    "elapsed_ms": 0,
    "last_finished_at": None,
}


def get_search_progress() -> dict[str, Any]:
    return dict(SEARCH_PROGRESS_STATE)


def update_search_progress(**kwargs: Any) -> dict[str, Any]:
    SEARCH_PROGRESS_STATE.update(kwargs)
    state = dict(SEARCH_PROGRESS_STATE)
    try:
        from backend.app.api.ws import ws_manager

        ws_manager.broadcast_sync("search_progress", state)
    except Exception:
        pass
    return state


def search_unified(
    session: Session,
    *,
    q: str | None = None,
    record_type: str | None = None,
    scan_id: int | None = None,
    census_sheet_id: int | None = None,
    municipality: str | None = None,
    district: str | None = None,
    street: str | None = None,
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
    t_start = time.perf_counter()
    query_parts = [
        p.strip()
        for p in (q, surname, municipality, street, house_number, occupation, religion)
        if p and str(p).strip()
    ]
    query_label = " · ".join(query_parts)
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

    st_matching_sheet_ids: set[int] = set()
    st_matching_hh_ids: set[int] = set()
    if street and street.strip():
        from sqlalchemy import or_

        st_fold = fold_text_diacritics(street.strip())
        st_clean, _ = normalize_street_query(street)
        probe_toks = [t for t in (st_clean or st_fold).split() if len(t) >= 3]
        probe_prefix = probe_toks[0][:4] if probe_toks else ""

        cs_st_stmt = select(CensusSheet).where(CensusSheet.street_raw.is_not(None))  # type: ignore[union-attr]
        hh_st_stmt = select(CensusHousehold).where(CensusHousehold.street.is_not(None))  # type: ignore[union-attr]
        ovh_st_stmt = select(CensusOverviewHouse).where(CensusOverviewHouse.settlement_name.is_not(None))  # type: ignore[union-attr]
        if municipality and municipality.strip():
            m_f = fold_text_diacritics(municipality)
            cs_st_stmt = cs_st_stmt.where(
                (CensusSheet.municipality_folded.contains(m_f))  # type: ignore[union-attr]
                | (CensusSheet.municipality.ilike(f"%{municipality.strip()}%"))  # type: ignore[union-attr]
            )
            ovh_st_stmt = ovh_st_stmt.where(
                CensusOverviewHouse.municipality.ilike(f"%{municipality.strip()}%")  # type: ignore[union-attr]
            )
        if probe_prefix:
            cs_st_stmt = cs_st_stmt.where(
                CensusSheet.street_raw.ilike(f"%{probe_prefix}%")  # type: ignore[union-attr]
                | CensusSheet.street_raw.ilike(f"%{street.strip()[:4]}%")  # type: ignore[union-attr]
            )
            hh_st_stmt = hh_st_stmt.where(
                CensusHousehold.street.ilike(f"%{probe_prefix}%")  # type: ignore[union-attr]
                | CensusHousehold.street.ilike(f"%{street.strip()[:4]}%")  # type: ignore[union-attr]
            )
            ovh_st_stmt = ovh_st_stmt.where(
                CensusOverviewHouse.settlement_name.ilike(f"%{probe_prefix}%")  # type: ignore[union-attr]
                | CensusOverviewHouse.settlement_name.ilike(f"%{street.strip()[:4]}%")  # type: ignore[union-attr]
            )

        for cs_row in session.exec(cs_st_stmt).all():
            if cs_row.id and text_matches_street([cs_row.street_raw], street):
                st_matching_sheet_ids.add(cs_row.id)
        for hh_row in session.exec(hh_st_stmt).all():
            if text_matches_street([hh_row.street], street):
                if hh_row.id:
                    st_matching_hh_ids.add(hh_row.id)
                if hh_row.census_sheet_id:
                    st_matching_sheet_ids.add(hh_row.census_sheet_id)
        for ovh_row in session.exec(ovh_st_stmt).all():
            if text_matches_street([ovh_row.settlement_name], street):
                if ovh_row.census_sheet_id:
                    st_matching_sheet_ids.add(ovh_row.census_sheet_id)
                if ovh_row.matched_sheet_id:
                    st_matching_sheet_ids.add(ovh_row.matched_sheet_id)
        like_pat = f"%{st_clean or st_fold}%"
        raw_like_pat = f"%{street.strip()}%"
        street_clauses = [
            PersonFact.village_folded.contains(st_clean or st_fold),  # type: ignore[union-attr]
            PersonFact.village_raw.ilike(raw_like_pat),  # type: ignore[union-attr]
            PersonFact.village_sk.ilike(raw_like_pat),  # type: ignore[union-attr]
            PersonFact.observationes_raw.ilike(raw_like_pat),  # type: ignore[union-attr]
            PersonFact.observationes_raw.ilike(like_pat),  # type: ignore[union-attr]
        ]
        if st_matching_sheet_ids:
            street_clauses.append(PersonFact.census_sheet_id.in_(st_matching_sheet_ids))  # type: ignore[union-attr]
        if st_matching_hh_ids:
            street_clauses.append(PersonFact.census_household_id.in_(st_matching_hh_ids))  # type: ignore[union-attr]
        stmt = stmt.where(or_(*street_clauses))

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
    matched_Id_set: set[int] = set()
    needs_python_scoring = bool((surname and surname.strip() and fuzzy) or (q and q.strip()))
    needs_fs_filter = bool(fs_status and fs_status != "all")

    if not needs_python_scoring and not needs_fs_filter:
        total_candidates_count = int(
            session.exec(select(func.count()).select_from(stmt.subquery())).one() or 0
        )
        total_facts = total_candidates_count
        paged_facts = list(
            session.exec(stmt.order_by(PersonFact.id.desc()).offset(offset).limit(limit)).all()
        )
        update_search_progress(
            running=True,
            scope="unified",
            query=query_label,
            processed=total_candidates_count,
            total=max(1, total_candidates_count),
            matched_records=total_facts,
            matched_sheets=0,
            progress_pct=80,
            stage_sk="Prehľadávam matriky a sčítacie hárky…",
            stage_en="Searching registers and census sheets…",
            elapsed_ms=int(round((time.perf_counter() - t_start) * 1000)),
        )
    else:
        fts_person_ids: list[int] = []
        if q and q.strip():
            q_clean = q.strip()
            fts_expr = build_fts_match_expression(q_clean, fuzzy=fuzzy)
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
                            sid_int = int(eid)
                            if sid_int not in matched_Id_set:
                                matched_Id_set.add(sid_int)
                                matched_sheet_ids.append(sid_int)
                except Exception:
                    pass

        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        used_rust_search = False
        if db_p and _rust_db is not None and hasattr(_rust_db, "score_person_facts_search_rs"):
            try:
                has_sql_where = bool(
                    (record_type and record_type != "all")
                    or scan_id is not None
                    or census_sheet_id is not None
                    or year_from is not None
                    or year_to is not None
                    or birth_year_from is not None
                    or birth_year_to is not None
                    or (municipality and municipality.strip())
                    or (street and street.strip())
                    or (house_number and house_number.strip())
                    or (occupation and occupation.strip())
                    or (religion and religion.strip())
                    or (surname and surname.strip() and not fuzzy)
                )
                cand_fids: list[int] | None = None
                if has_sql_where:
                    id_stmt = stmt.with_only_columns(PersonFact.id)
                    cand_fids = [int(r) for r in session.exec(id_stmt).all() if r is not None]

                total_candidates_count, scored_tuples = _rust_db.score_person_facts_search_rs(
                    db_p,
                    candidate_fact_ids=cand_fids,
                    surname_query=surname if (surname and surname.strip() and fuzzy) else None,
                    q_query=q if (q and q.strip()) else None,
                    fts_person_ids=fts_person_ids,
                    fuzzy=fuzzy,
                )
                matched_fids: list[int] = []
                for fid_val, sc_val, csid_val in scored_tuples:
                    fid_int = int(fid_val)
                    matched_fids.append(fid_int)
                    fact_scores[fid_int] = float(sc_val)
                    if q and q.strip() and csid_val is not None:
                        csid_int = int(csid_val)
                        if csid_int not in matched_Id_set:
                            matched_Id_set.add(csid_int)
                            matched_sheet_ids.append(csid_int)

                if needs_fs_filter and matched_fids:
                    fs_matches_all_rs: list[FamilySearchMatch] = []
                    for chunk_start in range(0, len(matched_fids), 500):
                        chunk_fids = matched_fids[chunk_start : chunk_start + 500]
                        fs_matches_all_rs.extend(
                            session.exec(
                                select(FamilySearchMatch).where(
                                    FamilySearchMatch.person_fact_id.in_(chunk_fids)  # type: ignore[union-attr]
                                )
                            ).all()
                        )
                    fs_map_rs = {
                        m.person_fact_id: m
                        for m in fs_matches_all_rs
                        if m.fs_person_id and str(m.fs_person_id).strip() and (m.confidence_score or 0.0) > 0
                    }
                    matched_fids = [
                        fid
                        for fid in matched_fids
                        if (fs_map_rs.get(fid).match_status if fs_map_rs.get(fid) else "unmatched") == fs_status
                    ]

                total_facts = len(matched_fids)
                page_fids = matched_fids[offset : offset + limit]
                if page_fids:
                    loaded_page_facts = {
                        pf.id: pf
                        for pf in session.exec(
                            select(PersonFact).where(PersonFact.id.in_(page_fids))  # type: ignore[union-attr]
                        ).all()
                        if pf.id is not None
                    }
                    paged_facts = [loaded_page_facts[fid] for fid in page_fids if fid in loaded_page_facts]
                else:
                    paged_facts = []
                used_rust_search = True
            except Exception:
                used_rust_search = False

        if not used_rust_search:
            all_candidates = list(session.exec(stmt.order_by(PersonFact.id.desc())).all())
            total_candidates_count = len(all_candidates)
            update_search_progress(
                running=True,
                scope="unified",
                query=query_label,
                processed=0,
                total=max(1, total_candidates_count),
                matched_records=0,
                matched_sheets=0,
                progress_pct=10,
                stage_sk="Prehľadávam matriky a sčítacie hárky…",
                stage_en="Searching registers and census sheets…",
                elapsed_ms=0,
            )

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
                cand_row_ids = list({f.row_id for f in all_candidates if f.row_id})
                row_ocr_map: dict[int, str] = {}
                for chunk_start in range(0, len(cand_row_ids), 500):
                    chunk_rids = cand_row_ids[chunk_start : chunk_start + 500]
                    for r_id_val, ocr_comb, ocr_l, ocr_r in session.exec(
                        select(
                            RecordRow.id,
                            RecordRow.raw_ocr_combined,
                            RecordRow.raw_ocr_left,
                            RecordRow.raw_ocr_right,
                        ).where(RecordRow.id.in_(chunk_rids))  # type: ignore[union-attr]
                    ).all():
                        if r_id_val is not None:
                            ocr_txt = " ".join(p for p in (ocr_comb, ocr_l, ocr_r) if p)
                            if ocr_txt:
                                row_ocr_map[int(r_id_val)] = ocr_txt

                needle = fold_text_diacritics(q_clean)
                q_toks = [
                    fold_text_diacritics(t)
                    for t in re.findall(r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+", q_clean)
                    if t
                ]
                fts_id_set = set(fts_person_ids)
                threshold = 0.76 if fuzzy else 0.90
                scored_candidates: list[tuple[float, int, PersonFact]] = []
                step_size = max(1, len(all_candidates) // 8)

                for idx_cand, f in enumerate(all_candidates, start=1):
                    ocr_txt = row_ocr_map.get(f.row_id)
                    struct_name_cands = extract_person_fact_name_candidates(f, ocr_text=None)
                    haystack_parts = [
                        *struct_name_cands,
                        ocr_txt or "",
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
                                *struct_name_cands,
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
                            # Matched via FTS5 non-name columns (notes, occupation, raw OCR, etc.)
                            if q_toks and all(any(qt in w for w in haystack.split()) for qt in q_toks):
                                score = 0.88
                            elif ocr_txt:
                                ocr_score = fuzzy_name_match_score(q_clean, [ocr_txt], threshold=threshold)
                                if ocr_score >= threshold:
                                    score = ocr_score
                        elif ocr_txt and not struct_name_cands:
                            ocr_score = fuzzy_name_match_score(q_clean, [ocr_txt], threshold=threshold)
                            if ocr_score >= threshold:
                                score = ocr_score

                    if score >= threshold:
                        if f.id is not None:
                            fact_scores[f.id] = score
                        if f.census_sheet_id and f.census_sheet_id not in matched_Id_set:
                            matched_Id_set.add(f.census_sheet_id)
                            matched_sheet_ids.append(f.census_sheet_id)
                        scored_candidates.append((score, f.id or 0, f))

                    if idx_cand % step_size == 0 or idx_cand == len(all_candidates):
                        pct = min(85, max(15, int(round((idx_cand / max(len(all_candidates), 1)) * 80))))
                        update_search_progress(
                            running=True,
                            scope="unified",
                            query=query_label,
                            processed=idx_cand,
                            total=max(1, len(all_candidates)),
                            matched_records=len(scored_candidates),
                            matched_sheets=len(matched_sheet_ids),
                            progress_pct=pct,
                            stage_sk="Porovnávam fonetické a OCR varianty mien…",
                            stage_en="Comparing phonetic and OCR name variants…",
                            elapsed_ms=int(round((time.perf_counter() - t_start) * 1000)),
                        )

                scored_candidates.sort(key=lambda item: (round(item[0], 3), item[1]), reverse=True)
                facts = [item[2] for item in scored_candidates]
            else:
                facts = all_candidates

            if needs_fs_filter:
                all_fact_ids = [f.id for f in facts if f.id is not None]
                fs_matches_all: list[FamilySearchMatch] = []
                for chunk_start in range(0, len(all_fact_ids), 500):
                    chunk_fids = all_fact_ids[chunk_start : chunk_start + 500]
                    fs_matches_all.extend(
                        session.exec(select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id.in_(chunk_fids))).all()  # type: ignore[union-attr]
                    )
                fs_map_all = {
                    m.person_fact_id: m
                    for m in fs_matches_all
                    if m.fs_person_id and str(m.fs_person_id).strip() and (m.confidence_score or 0.0) > 0
                }
                facts = [
                    f
                    for f in facts
                    if f.id is not None
                    and ((fs_map_all.get(f.id).match_status if fs_map_all.get(f.id) else "unmatched") == fs_status)
                ]

            total_facts = len(facts)
            paged_facts = facts[offset : offset + limit]

    paged_fact_ids = [f.id for f in paged_facts if f.id is not None]
    fs_matches = (
        session.exec(select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id.in_(paged_fact_ids))).all()  # type: ignore[union-attr]
        if paged_fact_ids
        else []
    )
    fs_map = {
        m.person_fact_id: m
        for m in fs_matches
        if m.fs_person_id and str(m.fs_person_id).strip() and (m.confidence_score or 0.0) > 0
    }

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

    paged_sheet_ids = list({f.census_sheet_id for f in paged_facts if f.census_sheet_id})
    paged_hh_ids = list({f.census_household_id for f in paged_facts if f.census_household_id})
    paged_sheets_map = {
        cs.id: cs
        for cs in (
            session.exec(select(CensusSheet).where(CensusSheet.id.in_(paged_sheet_ids))).all()  # type: ignore[union-attr]
            if paged_sheet_ids
            else []
        )
        if cs.id is not None
    }
    paged_hh_map = {
        hh.id: hh
        for hh in (
            session.exec(select(CensusHousehold).where(CensusHousehold.id.in_(paged_hh_ids))).all()  # type: ignore[union-attr]
            if paged_hh_ids
            else []
        )
        if hh.id is not None
    }

    records_out: list[dict[str, Any]] = []
    from backend.app.api.scans import resolve_scan_media_url

    for f in paged_facts:
        r = rows.get(f.row_id)
        s = scans.get(f.scan_id)
        fs = fs_map.get(f.id) if f.id is not None else None
        cs_obj = paged_sheets_map.get(f.census_sheet_id) if f.census_sheet_id else None
        hh_obj = paged_hh_map.get(f.census_household_id) if f.census_household_id else None
        st_name = (
            (cs_obj.street_raw if cs_obj and cs_obj.street_raw else None)
            or (hh_obj.street if hh_obj and hh_obj.street else None)
        )
        m_score = fact_scores.get(f.id or -1, 1.0)
        records_out.append(
            {
                "fact": f,
                "street_name": st_name,
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
        if street and street.strip():
            st_clean, _ = normalize_street_query(street)
            st_like = f"%{st_clean or street.strip()}%"
            if st_matching_sheet_ids:
                sheet_stmt = sheet_stmt.where(
                    (CensusSheet.id.in_(st_matching_sheet_ids))  # type: ignore[union-attr]
                    | (CensusSheet.street_raw.ilike(st_like))  # type: ignore[union-attr]
                )
            else:
                sheet_stmt = sheet_stmt.where(
                    CensusSheet.street_raw.ilike(st_like)  # type: ignore[union-attr]
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
                used_rust_cat_match = False
                try:
                    from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

                    db_p_cat = _session_file_db_path(session)
                    if (
                        db_p_cat
                        and _rust_db is not None
                        and hasattr(_rust_db, "match_catalog_sheets_and_overview_owners_rs")
                    ):
                        extra_sids = _rust_db.match_catalog_sheets_and_overview_owners_rs(
                            db_p_cat,
                            q_clean,
                            year_from=year_from,
                            year_to=year_to,
                            municipality=municipality,
                            district=district,
                        )
                        for esid in extra_sids:
                            esid_int = int(esid)
                            if esid_int not in matched_Id_set:
                                matched_Id_set.add(esid_int)
                                matched_sheet_ids.append(esid_int)
                        used_rust_cat_match = True
                except Exception:
                    used_rust_cat_match = False

                if not used_rust_cat_match:
                    # Check CensusOverviewHouse owners & CensusSheet household_owner for fuzzy matches (scoped by active filters)
                    ovh_stmt = select(
                        CensusOverviewHouse.census_sheet_id,
                        CensusOverviewHouse.matched_sheet_id,
                        CensusOverviewHouse.owner_name,
                        CensusOverviewHouse.remarks,
                        CensusOverviewHouse.raw_ocr_text,
                    )
                    if year_from is not None:
                        ovh_stmt = ovh_stmt.where(CensusOverviewHouse.census_year >= year_from)
                    if year_to is not None:
                        ovh_stmt = ovh_stmt.where(CensusOverviewHouse.census_year <= year_to)
                    if municipality and municipality.strip():
                        ovh_stmt = ovh_stmt.where(CensusOverviewHouse.municipality.ilike(f"%{municipality.strip()}%"))  # type: ignore[union-attr]
                    for ov_csid, ov_msid, ov_owner, ov_rem, ov_ocr in session.exec(ovh_stmt).all():
                        matched_ovh = fuzzy_name_matches(q_clean, [ov_owner, ov_rem], threshold=0.76) or (
                            bool(ov_ocr and q_fold and q_fold in fold_text_diacritics(ov_ocr))
                        ) or (not ov_owner and bool(ov_ocr) and fuzzy_name_matches(q_clean, [ov_ocr], threshold=0.76))
                        if matched_ovh:
                            if ov_csid and ov_csid not in matched_Id_set:
                                matched_Id_set.add(ov_csid)
                                matched_sheet_ids.append(ov_csid)
                            if ov_msid and ov_msid not in matched_Id_set:
                                matched_Id_set.add(ov_msid)
                                matched_sheet_ids.append(ov_msid)

                    cs_owner_stmt = sheet_stmt.where(CensusSheet.household_owner.is_not(None)).with_only_columns(  # type: ignore[union-attr]
                        CensusSheet.id, CensusSheet.household_owner
                    )
                    for cs_id_val, cs_owner_val in session.connection().execute(cs_owner_stmt).all():
                        if cs_id_val and cs_id_val not in matched_Id_set:
                            if fuzzy_name_matches(q_clean, [cs_owner_val], threshold=0.76):
                                matched_Id_set.add(cs_id_val)
                                matched_sheet_ids.append(cs_id_val)

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

    elapsed_ms = int(round((time.perf_counter() - t_start) * 1000))
    final_progress = update_search_progress(
        running=False,
        scope="unified",
        query=query_label,
        processed=total_candidates_count,
        total=max(1, total_candidates_count),
        matched_records=total_facts,
        matched_sheets=len(catalog_sheets_out),
        progress_pct=100,
        stage_sk="Vyhľadávanie dokončené",
        stage_en="Search complete",
        elapsed_ms=elapsed_ms,
        last_finished_at=time.time(),
    )

    return {
        "total": total_facts,
        "scanned_total": total_candidates_count,
        "records": records_out,
        "catalog_sheets": catalog_sheets_out,
        "search_progress": final_progress,
    }


search_unified_records = search_unified
