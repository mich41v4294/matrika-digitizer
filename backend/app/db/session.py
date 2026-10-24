import os
from typing import Annotated, Any, Generator
from fastapi import Depends
from sqlalchemy import event, text
from sqlmodel import Session, SQLModel, create_engine
from backend.app.config import settings

settings.ensure_dirs()

sqlite_url = f"sqlite:///{settings.db_path}"
engine = create_engine(
    sqlite_url,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    requested_mode = os.environ.get("SQLITE_JOURNAL_MODE", "WAL").strip().upper()
    if requested_mode not in {"WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"}:
        requested_mode = "WAL"
    try:
        cursor.execute(f"PRAGMA journal_mode={requested_mode};")
    except Exception:
        try:
            cursor.execute("PRAGMA journal_mode=DELETE;")
        except Exception:
            pass
    try:
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA temp_store=MEMORY;")
        cursor.execute("PRAGMA cache_size=-64000;")
    except Exception:
        pass
    cursor.close()


_SAFE_COLUMN_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "scanimage": [
        ("census_sheet_id", "INTEGER"),
        ("page_side", "TEXT"),
    ],
    "personfact": [
        ("census_sheet_id", "INTEGER"),
        ("census_household_id", "INTEGER"),
        ("person_row_number", "TEXT"),
        ("relation_to_head", "TEXT"),
        ("marital_status", "TEXT"),
        ("birth_year_est", "INTEGER"),
        ("birth_place_raw", "TEXT"),
        ("birth_place_sk", "TEXT"),
        ("birth_place_municipality", "TEXT"),
        ("birth_place_district", "TEXT"),
        ("birth_place_country", "TEXT"),
        ("citizenship", "TEXT"),
        ("moved_in_date", "TEXT"),
        ("previous_residence", "TEXT"),
        ("home_jurisdiction", "TEXT"),
        ("nationality", "TEXT"),
        ("literacy", "TEXT"),
        ("occupation_position", "TEXT"),
        ("employer_workplace", "TEXT"),
        ("marriage_date_raw", "TEXT"),
        ("children_born_count", "TEXT"),
        ("children_dead_count", "TEXT"),
        ("presence_type", "TEXT"),
        ("surname_candidates_json", "TEXT DEFAULT '[]'"),
        ("surname_folded", "TEXT"),
        ("surname_skeleton", "TEXT"),
        ("given_canonical_id", "TEXT"),
        ("village_folded", "TEXT"),
        ("house_number_int", "INTEGER"),
    ],
    "censussheet": [
        ("sheet_type", "TEXT DEFAULT 'household'"),
        ("obvod_number", "TEXT"),
        ("commissioner_name", "TEXT"),
    ],
    "censushousehold": [
        ("dwelling_unit", "TEXT"),
    ],
}


def _ensure_sqlite_columns() -> None:
    with engine.begin() as conn:
        for table_name, columns in _SAFE_COLUMN_MIGRATIONS.items():
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            if not rows:
                continue
            existing = {r[1] for r in rows}
            for col_name, col_type in columns:
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
        # Ensure default sheet_type on existing rows
        conn.execute(text("UPDATE censussheet SET sheet_type = 'household' WHERE sheet_type IS NULL OR sheet_type = ''"))
        conn.execute(text("UPDATE censushousehold SET dwelling_unit = 'I' WHERE dwelling_unit IS NULL OR dwelling_unit = ''"))
        for idx_sql in (
            "CREATE INDEX IF NOT EXISTS ix_familysearchmatch_fs_person_id ON familysearchmatch (fs_person_id)",
            "CREATE INDEX IF NOT EXISTS ix_familysearchmatch_status_fact ON familysearchmatch (match_status, person_fact_id)",
            "CREATE INDEX IF NOT EXISTS ix_recordrow_scan_row ON recordrow (scan_id, row_index)",
            "CREATE INDEX IF NOT EXISTS ix_personfact_sheet_id ON personfact (census_sheet_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_personfact_type_id ON personfact (record_type, id)",
            "CREATE INDEX IF NOT EXISTS ix_censussheet_year_status_id ON censussheet (census_year, download_status, id)",
            "CREATE INDEX IF NOT EXISTS ix_censusoverviewhouse_sheet_row ON censusoverviewhouse (census_sheet_id, row_index)",
            "CREATE INDEX IF NOT EXISTS ix_censusmunicipaloverviewdistrict_sheet_row ON censusmunicipaloverviewdistrict (census_sheet_id, row_index)",
            "CREATE INDEX IF NOT EXISTS ix_workflowtrace_scan_id ON workflowtrace (scan_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_questionitem_status_id ON questionitem (status, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_verdict_combo ON ocrfeedback (verdict, model_combo)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_row_id ON ocrfeedback (row_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_fact_id ON ocrfeedback (person_fact_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_sheet_id ON ocrfeedback (census_sheet_id, id)",
        ):
            conn.execute(text(idx_sql))
        # Remove any stale unmatched FamilySearchMatch rows (fs_person_id IS NULL or empty)
        try:
            conn.execute(
                text(
                    "DELETE FROM familysearchmatch "
                    "WHERE fs_person_id IS NULL OR trim(fs_person_id) = '' OR confidence_score <= 0"
                )
            )
        except Exception:
            pass


def delete_person_fact_cascade(session: Session, fact_id: int) -> bool:
    """Deletes a PersonFact along with its linked FamilySearchMatch, LocalRecordMatch, and FTS5 entries."""
    from sqlmodel import select as _select
    from backend.app.db.models import FamilySearchMatch, LocalRecordMatch, PersonFact

    fact = session.get(PersonFact, fact_id)
    if not fact:
        return False

    for fsm in session.exec(
        _select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact_id)
    ).all():
        session.delete(fsm)

    for lrm in session.exec(
        _select(LocalRecordMatch).where(
            (LocalRecordMatch.census_fact_id == fact_id)
            | (LocalRecordMatch.matrika_fact_id == fact_id)
        )
    ).all():
        session.delete(lrm)

    try:
        session.exec(
            text(
                "DELETE FROM search_index_fts WHERE entity_type = 'person_fact' AND entity_id = :eid"
            ).bindparams(eid=str(fact_id))
        )
    except Exception:
        pass

    session.delete(fact)
    session.commit()
    return True


def init_db() -> None:
    settings.ensure_dirs()
    from backend.app.db import models as _models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _ensure_sqlite_columns()
    try:
        from backend.app.search.fts import ensure_fts_schema

        ensure_fts_schema(engine)
    except ImportError:
        pass


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
