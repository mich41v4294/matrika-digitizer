import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Annotated, Any, Generator
from fastapi import Depends
from sqlalchemy import event, text
from sqlmodel import Session, SQLModel, create_engine
from backend.app.config import settings

logger = logging.getLogger(__name__)

settings.ensure_dirs()

sqlite_url = f"sqlite:///{settings.db_path}"
engine = create_engine(
    sqlite_url,
    connect_args={"check_same_thread": False, "timeout": 30},
)


def _decode_sqlite_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    try:
        dbapi_connection.text_factory = _decode_sqlite_text
    except Exception:
        pass
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
        cursor.execute("PRAGMA mmap_size=0;")
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
            "CREATE INDEX IF NOT EXISTS ix_workflowtrace_row_id ON workflowtrace (row_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_workflowtrace_stats_cov ON workflowtrace (id, scan_id, row_id, stage, model_or_tool, status, duration_ms, prompt_tokens, completion_tokens, reasoning_tokens, image_path)",
            "CREATE INDEX IF NOT EXISTS ix_questionitem_status_id ON questionitem (status, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_verdict_combo ON ocrfeedback (verdict, model_combo)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_row_id ON ocrfeedback (row_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_fact_id ON ocrfeedback (person_fact_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_ocrfeedback_sheet_id ON ocrfeedback (census_sheet_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_censussheet_year_muni_hn ON censussheet (census_year, municipality_folded, house_number_int)",
            "CREATE INDEX IF NOT EXISTS ix_censussheet_muni_key_year ON censussheet (municipality_key, census_year)",
            "CREATE INDEX IF NOT EXISTS ix_personfact_skel_cid_year ON personfact (record_type, surname_skeleton, given_canonical_id, event_year)",
            "CREATE INDEX IF NOT EXISTS ix_censusoverviewhouse_year_muni ON censusoverviewhouse (census_year, municipality, house_number_int)",
            "CREATE INDEX IF NOT EXISTS ix_censushousehold_sheet_id ON censushousehold (census_sheet_id, id)",
            "CREATE INDEX IF NOT EXISTS ix_localrecordmatch_census_fact ON localrecordmatch (census_fact_id, matrika_fact_id)",
        ):
            conn.execute(text(idx_sql))
        # Remove any stale unmatched FamilySearchMatch rows (fs_person_id IS NULL or empty)
        conn.execute(
            text(
                "DELETE FROM familysearchmatch "
                "WHERE fs_person_id IS NULL OR trim(fs_person_id) = '' OR confidence_score <= 0"
            )
        )
        # Sanitize any legacy non-UTF-8 bytes in ocrfeedback / workflowtrace / censussheet text columns
        _sanitize_non_utf8_sqlite_rows(conn)
        # Reset any downloaded household census sheets that were falsely marked extracted/matched with 0 extracted records
        conn.execute(
            text(
                "UPDATE censussheet "
                "SET download_status = 'downloaded' "
                "WHERE sheet_type = 'household' "
                "AND front_scan_id IS NOT NULL "
                "AND download_status IN ('extracted', 'matched', 'ocr_done') "
                "AND id NOT IN (SELECT DISTINCT census_sheet_id FROM personfact WHERE census_sheet_id IS NOT NULL) "
                "AND id NOT IN (SELECT DISTINCT census_sheet_id FROM censusoverviewhouse WHERE census_sheet_id IS NOT NULL) "
                "AND id NOT IN (SELECT DISTINCT census_sheet_id FROM censusmunicipaloverviewdistrict WHERE census_sheet_id IS NOT NULL)"
            )
        )


def _sanitize_non_utf8_sqlite_rows(conn: Any) -> None:
    import json

    table_cols: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "ocrfeedback",
            (
                "comment",
                "expected_correction",
                "raw_ocr_snapshot",
                "extracted_snapshot_json",
                "stage_breakdown_json",
                "traces_snapshot_json",
            ),
        ),
        (
            "censussheet",
            ("commissioner_name", "household_owner", "title_raw", "header_json"),
        ),
        (
            "scanimage",
            ("header_raw_ocr", "error_message"),
        ),
        (
            "recordrow",
            ("raw_ocr_left", "raw_ocr_right", "raw_ocr_combined", "normalized_json", "decision_json"),
        ),
    )
    for tbl, cols in table_cols:
        select_exprs = ", ".join(["id"] + [f"CAST({c} AS BLOB)" for c in cols])
        rows = conn.execute(text(f"SELECT {select_exprs} FROM {tbl}")).fetchall()
        for r in rows:
            row_id = r[0]
            updates: dict[str, str] = {}
            for idx, col_name in enumerate(cols, start=1):
                raw_val = r[idx]
                if not isinstance(raw_val, (bytes, bytearray, memoryview)):
                    continue
                raw_bytes = bytes(raw_val)
                try:
                    raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    cleaned = raw_bytes.decode("utf-8", errors="replace")
                    if col_name.endswith("_json"):
                        try:
                            json.loads(cleaned)
                        except Exception:
                            cleaned = "[]" if col_name.startswith("traces_") else "{}"
                    updates[col_name] = cleaned
            if updates:
                set_clause = ", ".join(f"{k} = :{k}" for k in updates)
                conn.execute(
                    text(f"UPDATE {tbl} SET {set_clause} WHERE id = :_row_id"),
                    {**updates, "_row_id": row_id},
                )


def delete_person_fact_cascade(session: Session, fact_id: int) -> bool:
    """Deletes a PersonFact along with its linked FamilySearchMatch, LocalRecordMatch, and FTS5 entries."""
    from sqlmodel import select as _select
    from backend.app.db.models import FamilySearchMatch, LocalRecordMatch, PersonFact

    fact = session.get(PersonFact, fact_id)
    if not fact:
        return False

    try:
        from backend.app.pipeline.feedback import reset_ocr_feedback

        reset_ocr_feedback(session, row_id=fact.row_id, person_fact_id=fact_id, commit=False)
    except Exception:
        pass

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


def _clear_all_sqlite_pools(db_path: Path | None = None) -> None:
    try:
        engine.dispose()
    except Exception:
        pass
    try:
        import matrika_db_rs as _rust_db

        if hasattr(_rust_db, "clear_db_pool_rs"):
            _rust_db.clear_db_pool_rs(str(db_path) if db_path else None)
    except Exception:
        pass


def _is_sqlite_db_healthy(db_path: Path) -> bool:
    if not db_path.exists() or db_path.stat().st_size == 0:
        return True
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=15)
        conn.text_factory = _decode_sqlite_text
        res = conn.execute("PRAGMA integrity_check(1);").fetchall()
        if not res or str(res[0][0]).strip().lower() != "ok":
            return False
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute("ROLLBACK;")
        return True
    except Exception as exc:
        logger.warning("SQLite health check failed on %s: %s", db_path, exc)
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _sanitize_salvaged_value(col_name: str, val: Any, col_default: Any = None) -> Any:
    if val is None:
        return col_default
    if isinstance(val, (bytes, bytearray, memoryview)):
        val = bytes(val).decode("utf-8", errors="replace")
    if isinstance(val, str):
        val = val.encode("utf-8", errors="replace").decode("utf-8")
        if col_name.endswith("_json"):
            try:
                json.loads(val)
            except Exception:
                return "[]" if ("traces" in col_name or "crops" in col_name or "candidates" in col_name or "events" in col_name or "urls" in col_name) else "{}"
    return val


def _read_salvageable_table_rows(
    src_conn: sqlite3.Connection,
    tbl_name: str,
    common_cols: list[str],
    col_defaults: dict[str, Any],
) -> list[tuple[Any, ...]]:
    col_csv = ", ".join(common_cols)
    has_id = "id" in common_cols
    id_idx = common_cols.index("id") if has_id else 0
    rows_by_key: dict[Any, tuple[Any, ...]] = {}

    # Pass 1: Forward B-tree scan (ORDER BY rowid ASC)
    asc_complete = False
    try:
        cur = src_conn.cursor()
        cur.execute(f"SELECT {col_csv} FROM {tbl_name} ORDER BY rowid ASC")
        while True:
            r = cur.fetchone()
            if r is None:
                asc_complete = True
                break
            key = r[id_idx] if has_id else len(rows_by_key)
            rows_by_key[key] = tuple(
                _sanitize_salvaged_value(common_cols[i], r[i], col_defaults.get(common_cols[i]))
                for i in range(len(common_cols))
            )
    except Exception as exc:
        logger.warning("Forward salvage scan on %s stopped at %d rows: %s", tbl_name, len(rows_by_key), exc)

    if asc_complete:
        return list(rows_by_key.values())

    # Pass 2: Reverse B-tree scan (ORDER BY rowid DESC) to rescue pages after the damaged page
    try:
        cur_rev = src_conn.cursor()
        cur_rev.execute(f"SELECT {col_csv} FROM {tbl_name} ORDER BY rowid DESC")
        while True:
            r = cur_rev.fetchone()
            if r is None:
                break
            key = r[id_idx] if has_id else f"rev_{len(rows_by_key)}"
            if key not in rows_by_key:
                rows_by_key[key] = tuple(
                    _sanitize_salvaged_value(common_cols[i], r[i], col_defaults.get(common_cols[i]))
                    for i in range(len(common_cols))
                )
    except Exception as exc:
        logger.warning("Reverse salvage scan on %s stopped at %d total rows: %s", tbl_name, len(rows_by_key), exc)

    if not has_id:
        return list(rows_by_key.values())

    # Pass 3: ID-based point-lookup rescue for any rows between ASC and DESC or with single-column overflow damage
    candidate_ids: set[int] = set()
    for order_dir in ("ASC", "DESC"):
        try:
            c_id = src_conn.cursor()
            c_id.execute(f"SELECT id FROM {tbl_name} ORDER BY id {order_dir}")
            while True:
                ir = c_id.fetchone()
                if ir is None:
                    break
                if isinstance(ir[0], int):
                    candidate_ids.add(ir[0])
        except Exception:
            pass

    max_known_id = max(
        [k for k in rows_by_key.keys() if isinstance(k, int)] + list(candidate_ids) + [0]
    )
    try:
        m_row = src_conn.execute(f"SELECT MAX(id) FROM {tbl_name}").fetchone()
        if m_row and isinstance(m_row[0], int) and m_row[0] > max_known_id:
            max_known_id = m_row[0]
    except Exception:
        pass

    if 0 < max_known_id <= 100_000:
        candidate_ids.update(range(1, max_known_id + 1))

    missing_ids = sorted(cid for cid in candidate_ids if cid not in rows_by_key)
    skip_partial_rescue = tbl_name in ("ocrfeedback", "workflowtrace")

    for cid in missing_ids:
        try:
            r = src_conn.execute(
                f"SELECT {col_csv} FROM {tbl_name} WHERE id = ?", (cid,)
            ).fetchone()
            if r is not None:
                rows_by_key[cid] = tuple(
                    _sanitize_salvaged_value(common_cols[i], r[i], col_defaults.get(common_cols[i]))
                    for i in range(len(common_cols))
                )
                continue
        except Exception:
            if skip_partial_rescue:
                continue

        if skip_partial_rescue:
            continue

        # Column-by-column rescue if only one overflow column on this row is damaged
        try:
            exists_chk = src_conn.execute(
                f"SELECT id FROM {tbl_name} WHERE id = ?", (cid,)
            ).fetchone()
            if not exists_chk:
                continue
            rescued_vals: list[Any] = []
            for c_name in common_cols:
                if c_name == "id":
                    rescued_vals.append(cid)
                    continue
                try:
                    c_row = src_conn.execute(
                        f"SELECT {c_name} FROM {tbl_name} WHERE id = ?", (cid,)
                    ).fetchone()
                    raw_v = c_row[0] if c_row else col_defaults.get(c_name)
                    rescued_vals.append(
                        _sanitize_salvaged_value(c_name, raw_v, col_defaults.get(c_name))
                    )
                except Exception:
                    fallback_v = (
                        "[]"
                        if c_name.endswith("_json") and ("crops" in c_name or "candidates" in c_name or "events" in c_name or "urls" in c_name)
                        else ("{}" if c_name.endswith("_json") else col_defaults.get(c_name))
                    )
                    rescued_vals.append(fallback_v)
            rows_by_key[cid] = tuple(rescued_vals)
        except Exception:
            continue

    sorted_keys = sorted(
        rows_by_key.keys(),
        key=lambda k: (0, k) if isinstance(k, int) else (1, str(k)),
    )
    return [rows_by_key[k] for k in sorted_keys]


def _salvage_corrupted_sqlite_db(db_path: Path) -> bool:
    """
    Rebuilds a corrupted SQLite database (`database disk image is malformed`) into a fresh,
    verified SQLite file by salvaging all readable rows table-by-table (using forward scan,
    reverse B-tree scan, and per-ID/per-column rescue), backing up the corrupted file,
    removing stale `-wal`/`-shm` files, and swapping in the clean database.
    """
    _clear_all_sqlite_pools(db_path)
    if not db_path.exists():
        return False

    logger.warning("Starting automatic SQLite salvage recovery for %s ...", db_path)
    salvage_path = db_path.with_name(f"{db_path.name}.salvage_tmp")
    for p in (salvage_path, Path(f"{salvage_path}-wal"), Path(f"{salvage_path}-shm")):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    from backend.app.db import models as _models  # noqa: F401

    salvage_engine = create_engine(
        f"sqlite:///{salvage_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    SQLModel.metadata.create_all(salvage_engine)
    salvage_engine.dispose()

    def _open_src_db() -> sqlite3.Connection:
        c = sqlite3.connect(str(db_path), timeout=30)
        c.text_factory = _decode_sqlite_text
        try:
            c.execute("PRAGMA writable_schema = ON;")
        except Exception:
            pass
        return c

    src_conn: sqlite3.Connection | None = None
    dst_conn: sqlite3.Connection | None = None
    try:
        src_conn = _open_src_db()
        try:
            src_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except Exception:
            # If a corrupted -wal file prevents even reading sqlite_master, move -wal/-shm aside first
            src_conn.close()
            for suffix in ("-wal", "-shm"):
                wp = Path(f"{db_path}{suffix}")
                if wp.exists():
                    try:
                        wp.replace(Path(f"{db_path}.corrupt{suffix}.bak"))
                    except OSError:
                        try:
                            wp.unlink()
                        except OSError:
                            pass
            src_conn = _open_src_db()

        dst_conn = sqlite3.connect(str(salvage_path), timeout=30)
        dst_conn.text_factory = _decode_sqlite_text

        for table in SQLModel.metadata.sorted_tables:
            tbl_name = table.name
            dst_info = dst_conn.execute(f"PRAGMA table_info({tbl_name})").fetchall()
            if not dst_info:
                continue
            dst_cols = [str(r[1]) for r in dst_info]
            col_defaults: dict[str, Any] = {}
            for col_obj in table.columns:
                if col_obj.default is not None and not callable(getattr(col_obj.default, "arg", None)):
                    col_defaults[col_obj.name] = getattr(col_obj.default, "arg", None)

            src_cols: list[str] = []
            try:
                src_info = src_conn.execute(f"PRAGMA table_info({tbl_name})").fetchall()
                src_cols = [str(r[1]) for r in src_info]
            except Exception:
                for cand_c in dst_cols:
                    try:
                        src_conn.execute(f"SELECT {cand_c} FROM {tbl_name} LIMIT 0")
                        src_cols.append(cand_c)
                    except Exception:
                        pass

            common_cols = [c for c in dst_cols if c in src_cols]
            if not common_cols:
                continue

            salvaged_rows = _read_salvageable_table_rows(
                src_conn, tbl_name, common_cols, col_defaults
            )
            if not salvaged_rows:
                continue

            col_csv = ", ".join(common_cols)
            placeholders = ", ".join("?" for _ in common_cols)
            insert_sql = f"INSERT OR REPLACE INTO {tbl_name} ({col_csv}) VALUES ({placeholders})"
            for start in range(0, len(salvaged_rows), 500):
                chunk = salvaged_rows[start : start + 500]
                for row_tuple in chunk:
                    try:
                        dst_conn.execute(insert_sql, row_tuple)
                    except Exception:
                        pass
            dst_conn.commit()
            logger.info("Salvaged %d rows from table '%s'", len(salvaged_rows), tbl_name)

        dst_conn.close()
        dst_conn = None
        src_conn.close()
        src_conn = None

        ts = int(time.time())
        backup_path = db_path.with_name(f"{db_path.name}.corrupt.{ts}.bak")
        try:
            shutil.copy2(db_path, backup_path)
        except Exception:
            pass

        for suffix in ("-wal", "-shm"):
            for base_p in (db_path, salvage_path):
                wp = Path(f"{base_p}{suffix}")
                if wp.exists():
                    try:
                        wp.unlink()
                    except OSError:
                        pass

        salvage_path.replace(db_path)
        _clear_all_sqlite_pools(db_path)
        logger.warning("SQLite salvage recovery completed for %s (backup at %s)", db_path, backup_path)
        return True
    except Exception as exc:
        logger.error("Failed to salvage SQLite DB %s: %s", db_path, exc)
        return False
    finally:
        for c in (src_conn, dst_conn):
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass


def _rebuild_fts_after_salvage() -> None:
    try:
        from backend.app.search.fts import ensure_fts_schema, rebuild_all_fts

        ensure_fts_schema(engine)
        with Session(engine) as session:
            rebuild_all_fts(session)
    except Exception as exc:
        logger.warning("Could not rebuild FTS5 after salvage: %s", exc)


def init_db() -> None:
    settings.ensure_dirs()
    from backend.app.db import models as _models  # noqa: F401

    db_path = Path(settings.db_path) if settings.db_path else None
    salvaged = False
    if db_path and db_path.exists() and not _is_sqlite_db_healthy(db_path):
        salvaged = _salvage_corrupted_sqlite_db(db_path)

    try:
        SQLModel.metadata.create_all(engine)
        _ensure_sqlite_columns()
        try:
            from backend.app.search.fts import ensure_fts_schema

            ensure_fts_schema(engine)
        except ImportError:
            pass
    except Exception as exc:
        if db_path and _salvage_corrupted_sqlite_db(db_path):
            salvaged = True
            SQLModel.metadata.create_all(engine)
            _ensure_sqlite_columns()
            try:
                from backend.app.search.fts import ensure_fts_schema

                ensure_fts_schema(engine)
            except ImportError:
                pass
        else:
            raise exc

    if salvaged:
        _rebuild_fts_after_salvage()


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
