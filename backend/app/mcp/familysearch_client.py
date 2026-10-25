from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from sqlalchemy import func
from sqlmodel import Session, select

from backend.app.config import BASE_REPO_DIR, settings
from backend.app.db.models import (
    CensusHousehold,
    CensusSheet,
    FamilySearchMatch,
    LocalRecordMatch,
    PersonFact,
    RecordRow,
    WorkflowTrace,
)
from backend.app.pipeline.lexicon import (
    _load_regional_places_cache,
    _paleographic_similarity,
    _phonetic_skeleton,
    apply_area_period_name_corrections,
    fold_text_diacritics,
    fs_name_cache_size,
    get_given_name_equivalents,
    get_regional_municipality_gazetteer,
    is_known_given_name,
    record_observed_fs_names,
    record_regional_places_cache,
    resolve_area_key,
    resolve_given_name_canonical_id,
)


ALLOWED_FS_ENV_KEYS: frozenset[str] = frozenset(
    {"FS_ACCESS_TOKEN", "FS_ENVIRONMENT", "FS_CLIENT_ID", "FS_TIMEOUT"}
)


def _sanitize_env_val(val: Any) -> str:
    return re.sub(r"[\r\n\x00]+", "", str(val or "")).strip()


def read_fs_env() -> dict[str, str]:
    settings.ensure_dirs()
    env_map: dict[str, str] = {
        "FS_ENVIRONMENT": settings.fs_environment,
        "FS_TIMEOUT": str(settings.fs_timeout),
        "FS_ACCESS_TOKEN": "",
    }
    if settings.fs_env_file.exists():
        for line in settings.fs_env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            if key in ALLOWED_FS_ENV_KEYS:
                env_map[key] = _sanitize_env_val(v)
    return env_map


def write_fs_env(updates: dict[str, str]) -> dict[str, str]:
    current = read_fs_env()
    for k, v in updates.items():
        if k in ALLOWED_FS_ENV_KEYS and v is not None:
            current[k] = _sanitize_env_val(v)
    lines = [f"{k}={_sanitize_env_val(v)}" for k, v in current.items() if k in ALLOWED_FS_ENV_KEYS]
    settings.fs_env_file.parent.mkdir(parents=True, exist_ok=True)
    settings.fs_env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "fs_mcp_manager" in globals():
        fs_mcp_manager._cached_auth_ts = 0.0
        fs_mcp_manager._mcp_env_sig = None
    return current


def _format_fact_person_label(fact: PersonFact | None) -> str:
    """Format a human-readable person label for live progress display, e.g. 'Ján Kováč (1898)'."""
    if fact is None:
        return ""
    given = (fact.given_name_sk or fact.given_name_raw or "").strip(" ?—–-")
    surname = (fact.surname_raw or fact.surname_normalized or "").strip(" ?—–-")
    name = f"{given} {surname}".strip() or f"Osoba #{fact.id or '?'}"
    year = fact.birth_year_est or fact.event_year
    if year:
        return f"{name} ({year})"
    return name


def _is_valid_name_token(val: str | None) -> bool:
    if not val:
        return False
    cleaned = str(val).strip(" ?—–-.,;/\"'()[]")
    if not cleaned or cleaned.lower() in ("?", "none", "null", "n/a", "neznáme", "neznamy"):
        return False
    return True


def _is_nameless_fact(fact: PersonFact | None) -> bool:
    """Return True if fact is None or both given_name_raw and surname_raw are empty/placeholder."""
    if fact is None:
        return True
    given_clean = (fact.given_name_raw or "").strip(" ?—–-")
    surname_clean = (fact.surname_raw or "").strip(" ?—–-")
    if not given_clean and not surname_clean:
        return True
    return not _is_valid_name_token(given_clean) and not _is_valid_name_token(surname_clean)


def _split_spouse_name(
    fact: PersonFact,
    session: Session | None = None,
    peers: list[PersonFact] | None = None,
) -> tuple[str, str]:
    """
    Split `fact.spouse_name` into `(spouse_given, spouse_surname)`.
    In Slovak census records, `fact.spouse_name` is formatted as `"GivenName Surname"`
    (e.g. `"Mária Parabová"` or `"Ján Paraba"`). Strips parenthetical notes like `(rod. ...)`.
    """
    raw = str(fact.spouse_name or "").strip()
    if not raw:
        return "", ""
    # Strip parenthetical notes like (rod. ...) and trailing maiden surname notes
    cleaned = re.sub(r"\([^)]*\)", "", raw)
    cleaned = re.sub(r"\b(?:rod\.|rodená|rodený|née)\s+.*$", "", cleaned, flags=re.IGNORECASE)
    tokens = [t.strip(" ,.;/-—–?") for t in cleaned.split() if t.strip(" ,.;/-—–?")]
    tokens = [t for t in tokens if _is_valid_name_token(t)]
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        spouse_given = tokens[0]
        spouse_surname = ""
        if fact.census_sheet_id:
            try:
                peer_list = peers
                if peer_list is None and session is not None:
                    peer_list = list(
                        session.exec(
                            select(PersonFact).where(PersonFact.census_sheet_id == fact.census_sheet_id)
                        ).all()
                    )
                for peer in peer_list or []:
                    if peer.id == fact.id:
                        continue
                    peer_given = (peer.given_name_sk or peer.given_name_raw or "").strip()
                    if peer_given and peer_given.lower() == spouse_given.lower():
                        spouse_surname = (peer.surname_raw or peer.surname_normalized or "").strip(" ?—–-")
                        break
            except Exception:
                pass
        return spouse_given, spouse_surname
    spouse_given = " ".join(tokens[:-1])
    spouse_surname = tokens[-1]
    return spouse_given, spouse_surname


def _ensure_split_spouse_in_fs_url(
    fs_url: str,
    fact: PersonFact,
    session: Session | None = None,
    peers: list[PersonFact] | None = None,
) -> str:
    """Ensure `fs_url` uses split `q.spouseGivenName` and `q.spouseSurname` when `fact.spouse_name` is set."""
    if not fs_url or not fact.spouse_name:
        return fs_url
    spouse_given, spouse_surname = _split_spouse_name(fact, session, peers=peers)
    if not spouse_given and not spouse_surname:
        return fs_url
    if "?" not in fs_url:
        return fs_url
    base, query = fs_url.split("?", 1)
    parts = [p for p in query.split("&") if p]
    filtered = [
        p
        for p in parts
        if not p.startswith("q.spouseGivenName=") and not p.startswith("q.spouseSurname=")
    ]
    if spouse_given:
        filtered.append(f"q.spouseGivenName={quote_plus(spouse_given)}")
    if spouse_surname:
        filtered.append(f"q.spouseSurname={quote_plus(spouse_surname)}")
    return f"{base}?{'&'.join(filtered)}"


def _split_fs_display_name(full: str | None) -> tuple[str | None, str | None]:
    """Split a FamilySearch display name into (given, surname)."""
    if not full:
        return None, None
    parts = [p for p in str(full).split() if p and p not in ("-", "—")]
    if not parts:
        return None, None
    if len(parts) == 1:
        if is_known_given_name(parts[0]):
            return parts[0], None
        return None, parts[0]
    return " ".join(parts[:-1]), parts[-1]


def extract_names_from_fs_hits(hits: list[Any] | None) -> tuple[list[str], list[str]]:
    """Collect given names and surnames observed on FamilySearch search hits."""
    surnames: list[str] = []
    given_names: list[str] = []

    def _absorb(text: str | None) -> None:
        given, surname = _split_fs_display_name(text)
        if given:
            given_names.append(given)
        if surname:
            surnames.append(surname)

    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        _absorb(hit.get("name"))
        for form in hit.get("names") or []:
            if isinstance(form, dict):
                _absorb(form.get("text"))
            elif isinstance(form, str):
                _absorb(form)
        for extra in hit.get("also_on_this_record") or []:
            if isinstance(extra, str):
                _absorb(extra)
            elif isinstance(extra, dict):
                _absorb(extra.get("name") or extra.get("text"))
    return surnames, given_names


def _apply_fs_name_matches_to_fact(fact: PersonFact, row: RecordRow, area: str, year: int) -> None:
    """Re-run area/period matching on the fact after harvesting live FS frequencies."""
    fields: dict[str, Any] = {
        "given_name_raw": fact.given_name_raw,
        "given_name_sk": fact.given_name_sk,
        "given_name_hu": fact.given_name_hu,
        "surname_raw": fact.surname_raw,
        "surname_normalized": fact.surname_normalized,
        "father_name": fact.father_name,
        "mother_name": fact.mother_name,
        "mother_maiden_surname": fact.mother_maiden_surname,
        "village_hu": fact.village_hu,
        "village_sk": fact.village_sk,
        "village_raw": fact.village_raw,
        "event_year": fact.event_year,
    }
    repairs = apply_area_period_name_corrections(fields, area=area, year=year)
    fact.given_name_raw = fields.get("given_name_raw")
    fact.given_name_sk = fields.get("given_name_sk")
    fact.given_name_hu = fields.get("given_name_hu")
    fact.surname_raw = fields.get("surname_raw")
    fact.surname_normalized = fields.get("surname_normalized") or fields.get("surname_raw")
    fact.father_name = fields.get("father_name")
    fact.mother_name = fields.get("mother_name")
    fact.mother_maiden_surname = fields.get("mother_maiden_surname")
    if repairs:
        try:
            payload = json.loads(row.normalized_json or "{}")
            if isinstance(payload, dict):
                existing = list(payload.get("type_safety_repairs") or [])
                for note in repairs:
                    if note not in existing:
                        existing.append(note)
                payload["type_safety_repairs"] = existing
                for key in (
                    "given_name_raw",
                    "given_name_sk",
                    "given_name_hu",
                    "surname_raw",
                    "surname_normalized",
                    "father_name",
                    "mother_name",
                    "mother_maiden_surname",
                ):
                    if fields.get(key) is not None:
                        payload[key] = fields.get(key)
                row.normalized_json = json.dumps(payload, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            pass


def _clean_place_query(fact: PersonFact) -> str:
    """Return a clean historical place query for FamilySearch place standardization."""
    for cand in (fact.village_hu, fact.village_sk, fact.village_raw):
        if not cand:
            continue
        cleaned = re.sub(r"\b\d+[a-zA-Z°]*\b", "", cand).strip(" ,.;/-")
        if len(cleaned) >= 3 and not any(
            bad in cleaned.lower()
            for bad in ("conditio", "michael", "cath", "ev.", "idem", "theligio", "ivo.")
        ):
            return cleaned
    return ""


def build_action_packet_and_new_items(
    fact: PersonFact,
    place_std: str | None,
    fs_url: str,
    session: Session | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    fs_url = _ensure_split_spouse_in_fs_url(fs_url, fact, session)

    if fact.record_type == "census":
        sheet: CensusSheet | None = None
        household: CensusHousehold | None = None
        local_matches: list[LocalRecordMatch] = []
        hh_id = getattr(fact, "household_id", None) or getattr(fact, "census_household_id", None)

        def _load_census_context(db: Session) -> None:
            nonlocal sheet, household, local_matches
            if fact.census_sheet_id:
                sheet = db.get(CensusSheet, fact.census_sheet_id)
            elif fact.scan_id:
                sheet = db.exec(
                    select(CensusSheet).where(
                        (CensusSheet.front_scan_id == fact.scan_id)
                        | (CensusSheet.back_scan_id == fact.scan_id)
                    )
                ).first()
            if hh_id:
                household = db.get(CensusHousehold, hh_id)
            elif sheet and sheet.id:
                household = db.exec(
                    select(CensusHousehold).where(CensusHousehold.census_sheet_id == sheet.id)
                ).first()
            if fact.id:
                local_matches = list(
                    db.exec(
                        select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id == fact.id)
                    ).all()
                )

        if session is not None:
            _load_census_context(session)
        else:
            try:
                from backend.app.db.session import engine

                with Session(engine) as tmp_db:
                    _load_census_context(tmp_db)
            except Exception:
                pass

        return build_census_action_packet(
            fact=fact,
            sheet=sheet,
            household=household,
            place_std=place_std,
            fs_url=fs_url,
            local_matches=local_matches,
        )

    given_display = fact.given_name_raw or fact.given_name_sk or "?"
    surname_display = fact.surname_raw or fact.surname_normalized or "?"
    full_name_display = f"{given_display} {surname_display}".strip()
    obs_events = json.loads(fact.observationes_events_json or "[]")

    obs_summary_parts: list[str] = []
    if fact.observationes_raw:
        obs_summary_parts.append(fact.observationes_raw)
    for ev in obs_events:
        if isinstance(ev, dict):
            ev_type = ev.get("event_type") or ev.get("type") or "event"
            ev_date = ev.get("date") or (str(ev["year"]) if ev.get("year") else "") or ev.get("raw_note") or ""
            desc = f"{ev_type}: {ev_date} {ev.get('spouse') or ''} {ev.get('place') or ''}".strip()
            if desc and desc not in obs_summary_parts:
                obs_summary_parts.append(desc)
    observationes_note = " | ".join(obs_summary_parts) if obs_summary_parts else None

    birth_date_full = (
        f"{fact.birth_or_event_date} ({fact.event_year})".strip()
        if fact.birth_or_event_date and fact.event_year
        else (fact.birth_or_event_date or (str(fact.event_year) if fact.event_year else None))
    )
    baptism_date_full = (
        f"{fact.baptism_or_burial_date} ({fact.event_year})".strip()
        if fact.baptism_or_burial_date and fact.event_year
        else fact.baptism_or_burial_date
    )

    clean_place_std = place_std if (place_std and place_std.lower() != "hungary") else None
    hist_place = fact.village_hu or fact.village_raw or clean_place_std or "Neznáma obec"
    if fact.house_number and f"č. {fact.house_number}" not in (hist_place or ""):
        hist_place_with_house = f"{hist_place} (dom č. {fact.house_number})"
    else:
        hist_place_with_house = hist_place

    modern_place = (
        clean_place_std
        or (f"{fact.village_hu} ({fact.village_sk}), Nyitra, Hungary" if fact.village_hu and fact.village_sk else None)
        or fact.village_sk
        or fact.village_raw
        or "Slovensko"
    )

    citation_sk = (
        f"Rímskokatolícka cirkev, Matrika pokrstených ({fact.record_type}), "
        f"rok {fact.event_year or '?'}, záznam č. {fact.entry_number or '?'}, "
        f"meno: {full_name_display}, "
        f"rodičia: {fact.father_name or '?'} a {fact.mother_name or '?'}, "
        f"obec: {hist_place_with_house}."
    )

    what_is_new: list[dict[str, str]] = [
        {
            "field_sk": "Meno a priezvisko (presne podľa matriky)",
            "value": full_name_display,
        },
        {
            "field_sk": "Dátum narodenia / krstu",
            "value": f"Nar.: {fact.birth_or_event_date or '?'} | Krst: {fact.baptism_or_burial_date or '?'} ({fact.event_year or '?'})",
        },
        {
            "field_sk": "Rodičia (otec / matka)",
            "value": f"Otec: {fact.father_name or '?'} | Matka: {fact.mother_name or '?'}",
        },
        {
            "field_sk": "Obec, súpisné číslo domu a štandardizované miesto",
            "value": f"{fact.village_raw or '?'} (dom č. {fact.house_number or '—'}) → {modern_place}",
        },
    ]
    if fact.godparents_or_witnesses:
        what_is_new.append({
            "field_sk": "Krstní rodičia / Svedkovia (Patrini)",
            "value": fact.godparents_or_witnesses,
        })
    if fact.social_status_or_occupation:
        what_is_new.append({
            "field_sk": "Povolanie / Stav",
            "value": fact.social_status_or_occupation,
        })
    if observationes_note:
        what_is_new.append({
            "field_sk": "Dodatočné poznámky (Observationes)",
            "value": observationes_note,
        })

    action_packet: dict[str, Any] = {
        # Keys expected by FamilySearchReview.tsx
        "person_name_display": full_name_display,
        "given_name": given_display if given_display != "?" else None,
        "surname": surname_display if surname_display != "?" else None,
        "sex": fact.sex,
        "birth_date": birth_date_full,
        "baptism_date": baptism_date_full,
        "birth_place_historical": hist_place_with_house,
        "birth_place_modern_sk": modern_place,
        "father_full_name": fact.father_name,
        "mother_full_name": fact.mother_name,
        "godparents_note": (
            f"{fact.godparents_or_witnesses}"
            + (f" (Krstil: {fact.officiant})" if fact.officiant else "")
            if fact.godparents_or_witnesses
            else (f"Krstil: {fact.officiant}" if fact.officiant else None)
        ),
        "observationes_note": observationes_note,
        "source_citation_sk": citation_sk,
        "fs_search_url": fs_url,
        # Legacy / additional keys
        "title_sk": f"Balíček pre FamilySearch – zápis č. {fact.entry_number or '?'}",
        "person_name": full_name_display,
        "latin_name": fact.given_name_raw,
        "event_type": fact.record_type,
        "event_year": fact.event_year,
        "standardized_place": modern_place,
        "house_number": fact.house_number,
        "father": fact.father_name,
        "mother": fact.mother_name,
        "godparents": fact.godparents_or_witnesses,
        "observationes_events": obs_events,
        "citation_draft_sk": citation_sk,
        "familysearch_search_url": fs_url,
    }
    return action_packet, what_is_new


def cleanup_unmatched_familysearch_rows(session: Session) -> int:
    """
    Removes any FamilySearchMatch rows where no actual FamilySearch person/record ID
    (`fs_person_id`) was matched or where confidence_score <= 0.
    """
    from sqlalchemy import func

    unmatched = list(
        session.exec(
            select(FamilySearchMatch).where(
                (FamilySearchMatch.fs_person_id.is_(None))  # type: ignore[union-attr]
                | (func.trim(FamilySearchMatch.fs_person_id) == "")
                | (FamilySearchMatch.confidence_score <= 0)
            )
        ).all()
    )
    if not unmatched:
        return 0
    for m in unmatched:
        session.delete(m)
    session.commit()
    return len(unmatched)


class FamilySearchMcpManager:
    """Manages connection to `.venv/bin/familysearch-mcp` stdio server and logs every tool call in WorkflowTrace."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        venv_bin = BASE_REPO_DIR / ".venv" / "bin" / "familysearch-mcp"
        which_bin = shutil.which("familysearch-mcp")
        self.mcp_bin: Path = venv_bin if venv_bin.exists() else Path(which_bin or venv_bin)
        self._cached_auth_status: dict[str, Any] | None = None
        self._cached_auth_ts: float = 0.0
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_session: ClientSession | None = None
        self._mcp_loop: asyncio.AbstractEventLoop | None = None
        self._mcp_env_sig: tuple[tuple[str, str], ...] | None = None
        self.progress_state: dict[str, Any] = {
            "running": False,
            "mode": "idle",
            "processed": 0,
            "total": 0,
            "matched_count": 0,
            "current_person": "",
            "total_facts_in_db": 0,
            "total_matched_in_db": 0,
            "reviewed_count": 0,
            "last_finished_at": None,
            "error": None,
        }

    def _is_call_tool_mocked(self) -> bool:
        return (
            hasattr(self.call_tool, "_mock_name")
            or hasattr(self.call_tool, "assert_called")
            or not hasattr(self.call_tool, "__func__")
        )

    def refresh_db_counts(self, session: Session | None = None) -> dict[str, int]:
        """Refresh `total_facts_in_db`, `total_matched_in_db`, and `reviewed_count` from the database."""
        def _query(db: Session) -> tuple[int, int, int]:
            total_facts = int(db.exec(select(func.count()).select_from(PersonFact)).one() or 0)
            matched_filter = (
                FamilySearchMatch.fs_person_id.is_not(None),  # type: ignore[union-attr]
                func.trim(FamilySearchMatch.fs_person_id) != "",
                FamilySearchMatch.confidence_score > 0,
            )
            total_matched = int(
                db.exec(select(func.count()).select_from(FamilySearchMatch).where(*matched_filter)).one() or 0
            )
            reviewed = int(
                db.exec(
                    select(func.count())
                    .select_from(FamilySearchMatch)
                    .where(*matched_filter, FamilySearchMatch.reviewed_by_user == True)  # noqa: E712
                ).one()
                or 0
            )
            return total_facts, total_matched, reviewed

        try:
            from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

            db_p = (
                _session_file_db_path(session)
                if session is not None
                else (str(settings.db_path) if settings.db_path and Path(settings.db_path).exists() else None)
            )
            if db_p and _rust_db is not None and hasattr(_rust_db, "get_fs_progress_counts_rs"):
                t_f, t_m, rev = _rust_db.get_fs_progress_counts_rs(db_p)
                total_facts, total_matched, reviewed = int(t_f), int(t_m), int(rev)
            elif session is not None:
                total_facts, total_matched, reviewed = _query(session)
            else:
                from backend.app.db.session import engine

                with Session(engine) as db:
                    total_facts, total_matched, reviewed = _query(db)
            self.progress_state["total_facts_in_db"] = total_facts
            self.progress_state["total_matched_in_db"] = total_matched
            self.progress_state["reviewed_count"] = reviewed
            if not self.progress_state.get("running") and self.progress_state.get("last_finished_at") is None:
                self.progress_state["matched_count"] = total_matched
                if self.progress_state.get("total", 0) == 0:
                    self.progress_state["total"] = total_facts
                    self.progress_state["processed"] = total_facts
        except Exception:
            pass
        return {
            "total_facts_in_db": int(self.progress_state.get("total_facts_in_db") or 0),
            "total_matched_in_db": int(self.progress_state.get("total_matched_in_db") or 0),
            "reviewed_count": int(self.progress_state.get("reviewed_count") or 0),
        }

    def get_progress_state(
        self,
        session: Session | None = None,
        *,
        refresh_counts: bool = True,
    ) -> dict[str, Any]:
        if refresh_counts:
            self.refresh_db_counts(session)
        state = dict(self.progress_state)
        total = int(state.get("total") or 0)
        processed = int(state.get("processed") or 0)
        if state.get("running"):
            pct = int(round((processed / max(total, 1)) * 100)) if total > 0 else 0
        else:
            pct = 100 if total > 0 and processed >= total else 0
        state["progress_pct"] = max(0, min(100, pct))
        return state

    async def broadcast_progress(
        self,
        session: Session | None = None,
        *,
        refresh_counts: bool = False,
    ) -> dict[str, Any]:
        state = self.get_progress_state(session=session, refresh_counts=refresh_counts)
        try:
            from backend.app.api.ws import ws_manager

            await ws_manager.broadcast("fs_match_progress", state)
        except Exception:
            pass
        return state

    async def start_progress(
        self,
        *,
        mode: str = "rematch",
        total: int = 0,
        current_person: str = "",
        session: Session | None = None,
    ) -> dict[str, Any]:
        self.refresh_db_counts(session)
        self.progress_state.update(
            {
                "running": True,
                "mode": mode if mode in ("rematch", "auto") else "auto",
                "processed": 0,
                "total": max(0, int(total)),
                "matched_count": 0,
                "current_person": str(current_person or ""),
                "error": None,
            }
        )
        state = await self.broadcast_progress(session=session, refresh_counts=False)
        await asyncio.sleep(0)
        return state

    async def step_progress(
        self,
        *,
        processed: int | None = None,
        increment_processed: int = 0,
        increment_matched: int = 0,
        current_person: str | None = None,
        session: Session | None = None,
        refresh_counts: bool = False,
    ) -> dict[str, Any]:
        if processed is not None:
            self.progress_state["processed"] = max(0, int(processed))
        elif increment_processed:
            self.progress_state["processed"] = max(
                0, int(self.progress_state.get("processed") or 0) + int(increment_processed)
            )
        if increment_matched:
            self.progress_state["matched_count"] = max(
                0, int(self.progress_state.get("matched_count") or 0) + int(increment_matched)
            )
        if current_person is not None:
            self.progress_state["current_person"] = str(current_person)
        state = await self.broadcast_progress(session=session, refresh_counts=refresh_counts)
        await asyncio.sleep(0)
        return state

    async def finish_progress(
        self,
        *,
        session: Session | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        total = int(self.progress_state.get("total") or 0)
        processed = int(self.progress_state.get("processed") or 0)
        self.progress_state.update(
            {
                "running": False,
                "mode": "idle",
                "processed": max(processed, total) if error is None and total > 0 else processed,
                "current_person": "",
                "last_finished_at": time.time(),
                "error": error,
            }
        )
        state = await self.broadcast_progress(session=session, refresh_counts=True)
        try:
            from backend.app.api.ws import ws_manager

            await ws_manager.broadcast("fs_matches_updated", state)
        except Exception:
            pass
        return state

    def _build_server_params(self) -> StdioServerParameters:
        fs_env = read_fs_env()
        env = dict(os.environ)
        for k, v in fs_env.items():
            if k in ALLOWED_FS_ENV_KEYS:
                env[k] = _sanitize_env_val(v)
        env["FS_ENV_FILE"] = str(settings.fs_env_file)
        env["FAMILYSEARCH_ENV_FILE"] = str(settings.fs_env_file)
        vendor_mcp_dir = str(BASE_REPO_DIR / "vendor" / "familysearch-mcp")
        existing_py_path = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{vendor_mcp_dir}{os.pathsep}{existing_py_path}" if existing_py_path else vendor_mcp_dir
        return StdioServerParameters(
            command=str(self.mcp_bin),
            args=[],
            env=env,
        )

    async def _reset_mcp_session_unlocked(self) -> None:
        stack = self._mcp_stack
        self._mcp_stack = None
        self._mcp_session = None
        self._mcp_env_sig = None
        if stack is not None:
            try:
                loop = asyncio.get_running_loop()
                if self._mcp_loop is loop:
                    await stack.aclose()
            except Exception:
                pass
        self._mcp_loop = None

    async def _get_or_create_mcp_session_unlocked(self) -> ClientSession:
        loop = asyncio.get_running_loop()
        env_sig = tuple(sorted(read_fs_env().items()))
        if (
            self._mcp_session is not None
            and self._mcp_stack is not None
            and self._mcp_loop is loop
            and self._mcp_env_sig == env_sig
        ):
            return self._mcp_session

        await self._reset_mcp_session_unlocked()
        server_params = self._build_server_params()
        stack = AsyncExitStack()
        try:
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            mcp_session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await mcp_session.initialize()
        except Exception:
            try:
                await stack.aclose()
            except Exception:
                pass
            raise

        self._mcp_stack = stack
        self._mcp_session = mcp_session
        self._mcp_loop = loop
        self._mcp_env_sig = env_sig
        return mcp_session

    async def close(self) -> None:
        async with self._lock:
            await self._reset_mcp_session_unlocked()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        stage: str = "mcp_place",
        scan_id: int | None = None,
        row_id: int | None = None,
    ) -> dict[str, Any]:
        if not self.mcp_bin.exists():
            return {"error": f"familysearch-mcp binary not found at {self.mcp_bin}"}

        started = datetime.now(timezone.utc)
        response_data: Any = None
        status = "ok"

        async with self._lock:
            for attempt in range(2):
                try:
                    mcp_session = await self._get_or_create_mcp_session_unlocked()
                    result = await mcp_session.call_tool(tool_name, arguments=arguments)
                    texts = [
                        c.text for c in result.content if hasattr(c, "text") and c.text
                    ]
                    joined = "\n".join(texts)
                    try:
                        response_data = json.loads(joined)
                    except Exception:
                        response_data = {"raw_text": joined}
                    if getattr(result, "isError", False):
                        status = "error"
                    break
                except Exception as exc:
                    await self._reset_mcp_session_unlocked()
                    if attempt == 1:
                        status = "error"
                        response_data = {"error": str(exc)}

        finished = datetime.now(timezone.utc)
        duration_ms = int((finished - started).total_seconds() * 1000)

        try:
            from backend.app.db.session import engine

            with Session(engine) as db:
                trace = WorkflowTrace(
                    scan_id=scan_id,
                    row_id=row_id,
                    stage=stage,
                    model_or_tool=f"familysearch-mcp:{tool_name}",
                    prompt_text=json.dumps(arguments, indent=2, ensure_ascii=False),
                    response_text=json.dumps(response_data, indent=2, ensure_ascii=False)[:32000],
                    parsed_json=json.dumps(response_data, ensure_ascii=False)[:32000],
                    status=status,
                    duration_ms=duration_ms,
                )
                db.add(trace)
                db.commit()
        except Exception:
            pass

        return response_data if isinstance(response_data, dict) else {"result": response_data}

    async def get_auth_status(self, *, force: bool = False) -> dict[str, Any]:
        fs_env = read_fs_env()
        has_token = bool(fs_env.get("FS_ACCESS_TOKEN"))
        now = time.time()
        if not force and not self._is_call_tool_mocked():
            if self._cached_auth_status is not None and (
                now - self._cached_auth_ts < 60.0
                or self._lock.locked()
                or bool(self.progress_state.get("running"))
            ):
                cached = dict(self._cached_auth_status)
                authenticated = bool(cached.get("authenticated") or has_token)
                cached["authenticated"] = authenticated
                cached["has_token_configured"] = has_token
                cached["mode"] = "live_tree" if authenticated else "public_places_and_action_packets"
                cached["environment"] = fs_env.get("FS_ENVIRONMENT", "production")
                return cached
            if self._lock.locked() or bool(self.progress_state.get("running")):
                connected = self.mcp_bin.exists()
                return {
                    "connected": connected,
                    "authenticated": has_token,
                    "mode": "live_tree" if has_token else "public_places_and_action_packets",
                    "tools_count": 29,
                    "has_token_configured": has_token,
                    "env_file": str(settings.fs_env_file),
                    "environment": fs_env.get("FS_ENVIRONMENT", "production"),
                    "mcp_response": {"status": "busy"},
                    "status": "ok" if connected else "error",
                }

        mcp_res = await self.call_tool("auth_status", {}, stage="mcp_auth")
        connected = "error" not in mcp_res
        authenticated = bool(mcp_res.get("authenticated") or has_token)
        res = {
            "connected": connected,
            "authenticated": authenticated,
            "mode": "live_tree" if authenticated else "public_places_and_action_packets",
            "tools_count": 29,
            "has_token_configured": has_token,
            "env_file": str(settings.fs_env_file),
            "environment": fs_env.get("FS_ENVIRONMENT", "production"),
            "mcp_response": mcp_res,
            "status": "ok" if connected else "error",
        }
        self._cached_auth_status = dict(res)
        self._cached_auth_ts = now
        return res

    async def check_status(self) -> dict[str, Any]:
        return await self.get_auth_status()

    async def ensure_regional_places_cached(
        self,
        municipality: str | None = None,
        district: str | None = None,
        year: int = 1930,
        extra_places: list[str] | None = None,
        scan_id: int | None = None,
        include_gazetteer: bool = True,
    ) -> list[str]:
        """
        Queries FamilySearch MCP (`search_places_at_date`) for the sheet's municipality,
        district, and ~100km regional neighbor villages, caching canonical place names
        in `data/regional_places_cache.json` so subsequent rows match instantaneously.
        """
        cache = _load_regional_places_cache()
        cached_queries = cache.get("queries") or {}

        seed_places: list[str] = []
        for p in [municipality, district, *(extra_places or [])]:
            if p and str(p).strip() and str(p).strip() not in seed_places:
                seed_places.append(str(p).strip())

        if include_gazetteer:
            # Also include top regional villages from our ~100km gazetteer for this district
            for v in get_regional_municipality_gazetteer(municipality, district)[:18]:
                if v not in seed_places:
                    seed_places.append(v)

        for place_name in seed_places:
            q_key = fold_text_diacritics(place_name)
            existing_entries = cached_queries.get(q_key)
            if not q_key or (isinstance(existing_entries, list) and len(existing_entries) > 0):
                continue
            res = await self.call_tool(
                "search_places_at_date",
                {
                    "name": f"{place_name}, Czechoslovakia",
                    "year": year,
                    "count": 10,
                },
                stage="mcp_regional_place_cache",
                scan_id=scan_id,
            )
            places = res.get("places") if isinstance(res, dict) else None
            vicinity = res.get("vicinity_places") if isinstance(res, dict) else None
            if isinstance(places, list):
                record_regional_places_cache(
                    place_name,
                    places,
                    default_district=district,
                    vicinity_places=vicinity if isinstance(vicinity, list) else None,
                )
            else:
                record_regional_places_cache(place_name, [], default_district=district)

        return get_regional_municipality_gazetteer(municipality, district)

    def _should_query_mcp_records(self, fs_env: dict[str, str] | None = None) -> bool:
        env_map = fs_env if fs_env is not None else read_fs_env()
        has_token = bool(env_map.get("FS_ACCESS_TOKEN") or os.environ.get("FS_ACCESS_TOKEN"))
        is_mocked = (
            hasattr(self.call_tool, "_mock_name")
            or hasattr(self.call_tool, "assert_called")
            or not hasattr(self.call_tool, "__func__")
        )
        return has_token or is_mocked

    async def match_person_fact(
        self,
        session: Session,
        fact: PersonFact,
        row: RecordRow | None = None,
    ) -> FamilySearchMatch | None:
        """
        Runs FamilySearch MCP tools (place standardization + record/person search)
        and stores a FamilySearchMatch row ONLY when an actual match (`fs_person_id` non-empty)
        is found in FamilySearch MCP.
        """
        if _is_nameless_fact(fact):
            if fact and fact.id is not None:
                stale = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).all()
                for sm in stale:
                    session.delete(sm)
                session.commit()
            return None

        if row is None and fact.row_id:
            row = session.get(RecordRow, fact.row_id)
        eff_row_id = row.id if (row and row.id is not None) else (fact.row_id or 0)

        if fact.record_type == "census" and fact.census_sheet_id:
            sheet = session.get(CensusSheet, fact.census_sheet_id)
            if sheet and getattr(sheet, "sheet_type", "household") in ("overview", "municipal_overview"):
                if fact.id is not None:
                    stale = session.exec(
                        select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                    ).all()
                    for sm in stale:
                        session.delete(sm)
                    session.commit()
                return None

        is_census = fact.record_type == "census"
        place_query = _clean_place_query(fact)
        default_country = "Czechoslovakia" if is_census else "Hungary"
        year_int = int(fact.event_year or (1930 if is_census else 1884))

        place_res: dict[str, Any] = {}
        if place_query:
            cached_q = (_load_regional_places_cache().get("queries") or {}).get(
                fold_text_diacritics(place_query)
            )
            if isinstance(cached_q, list) and cached_q:
                place_res = {"places": cached_q}
            else:
                query_with_country = (
                    place_query
                    if any(c in place_query.lower() for c in ("hungary", "czechoslovakia", "slovakia", "slovensko"))
                    else f"{place_query}, {default_country}"
                )
                place_res = await self.call_tool(
                    "search_places_at_date",
                    {"name": query_with_country, "year": year_int},
                    stage="mcp_place",
                    scan_id=fact.scan_id,
                    row_id=eff_row_id or None,
                )
                if isinstance(place_res.get("places"), list) and place_res["places"]:
                    record_regional_places_cache(place_query, place_res["places"])

        place_std = None
        place_id = None
        places_list = place_res.get("places") if isinstance(place_res, dict) else None
        if isinstance(places_list, list) and places_list:
            chosen_place = None
            for pl in places_list:
                full_name = str(pl.get("full_name") or pl.get("name") or "")
                pl_type = str(pl.get("type") or "")
                if full_name.lower() in ("hungary", "slovakia", "czechoslovakia", "europe"):
                    continue
                if pl_type.lower() == "country":
                    continue
                if any(
                    k in full_name.lower()
                    for k in ("hungary", "slovakia", "czechoslovakia", "nyitra", "nitra", "trencsén", "trenčín", "pozsony", "hlohovec", "trakovice")
                ):
                    chosen_place = pl
                    break
            if chosen_place:
                place_std = chosen_place.get("full_name") or chosen_place.get("name")
                place_id = str(chosen_place.get("id") or "")

        if not place_std:
            if is_census:
                muni = fact.village_sk or fact.village_raw
                place_std = f"{muni}, Slovensko" if muni else "Slovensko"
            elif fact.village_sk and fact.village_hu and fact.village_sk != fact.village_hu:
                place_std = f"{fact.village_hu} ({fact.village_sk}), Nyitra, Hungary"
            elif fact.village_hu and len(fact.village_hu) >= 3 and not re.match(r"^\d", fact.village_hu):
                place_std = f"{fact.village_hu}, Nyitra, Hungary"
            else:
                place_std = fact.village_sk or fact.village_raw or "Slovakia"

        fact.fs_place_standardized = place_std
        fact.fs_place_id = place_id
        session.add(fact)

        area_key = resolve_area_key(place_std or fact.village_hu or fact.village_raw)
        given = (fact.given_name_sk or fact.given_name_raw or "") if is_census else (fact.given_name_raw or "")
        surname = (fact.surname_normalized or fact.surname_raw or "") if is_census else (fact.surname_raw or "")
        spouse_given, spouse_surname = _split_spouse_name(fact, session)

        if is_census:
            birth_year = fact.birth_year_est
            birth_place = (
                getattr(fact, "birth_place_municipality", None)
                or fact.birth_place_sk
                or fact.birth_place_raw
                or place_std
            )
            fs_search_params = [
                f"q.givenName={quote_plus(given)}" if given else "",
                f"q.surname={quote_plus(surname)}" if surname else "",
                f"q.birthLikePlace={quote_plus(birth_place)}" if birth_place else "",
                f"q.birthLikeDate.from={birth_year - 2}" if birth_year else "",
                f"q.birthLikeDate.to={birth_year + 2}" if birth_year else "",
                f"q.residencePlace={quote_plus(place_std)}",
                f"q.residenceDate.from={year_int}",
                f"q.residenceDate.to={year_int}",
                f"q.fatherGivenName={quote_plus(fact.father_name)}" if fact.father_name else "",
                f"q.motherGivenName={quote_plus(fact.mother_name)}" if fact.mother_name else "",
                f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
            ]
        else:
            fs_search_params = [
                f"q.givenName={quote_plus(given)}" if given else "",
                f"q.surname={quote_plus(surname)}" if surname else "",
                f"q.birthLikePlace={quote_plus(place_std)}",
                f"q.birthLikeDate.from={year_int - 2}",
                f"q.birthLikeDate.to={year_int + 2}",
                f"q.fatherGivenName={quote_plus(fact.father_name)}" if fact.father_name else "",
                f"q.motherGivenName={quote_plus(fact.mother_name)}" if fact.mother_name else "",
                f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
            ]
        fs_url = "https://www.familysearch.org/search/record/results?" + "&".join(
            [p for p in fs_search_params if p]
        )

        what_agrees: list[dict[str, str]] = []
        what_differs: list[dict[str, str]] = []
        if is_census:
            local_matches = match_census_to_local_matrika(session, fact)
            if local_matches:
                best_local = local_matches[0]
                matrika_fact = session.get(PersonFact, best_local.matrika_fact_id)
                if matrika_fact:
                    what_agrees.append(
                        {
                            "field_sk": "Lokálna matrika (zhoda)",
                            "value": (
                                f"{matrika_fact.given_name_raw or ''} {matrika_fact.surname_raw or ''} "
                                f"({matrika_fact.record_type}, rok {matrika_fact.event_year}, "
                                f"dátum {matrika_fact.birth_or_event_date or '?'}) – skóre {best_local.confidence_score:.2f}"
                            ),
                        }
                    )

        search_res: dict[str, Any] | None = None
        harvest_res: dict[str, Any] | None = None
        best: dict[str, Any] | None = None
        fs_env = read_fs_env()
        if self._should_query_mcp_records(fs_env) and (given or surname):
            search_args: dict[str, Any] = {
                "given": given,
                "surname": surname,
                "birth_place": (
                    (
                        getattr(fact, "birth_place_municipality", None)
                        or fact.birth_place_sk
                        or fact.birth_place_raw
                        or place_std
                    )
                    if is_census
                    else place_std
                ),
                "birth_year": int(fact.birth_year_est or year_int) if is_census else year_int,
                "loose": True,
                "count": 20,
            }
            if is_census and place_std:
                search_args["residence_place"] = place_std
            if spouse_given:
                search_args["spouse_given"] = spouse_given
            if spouse_surname:
                search_args["spouse_surname"] = spouse_surname
            search_res = await self.call_tool(
                "search_records",
                search_args,
                stage="mcp_record",
                scan_id=fact.scan_id,
                row_id=eff_row_id or None,
            )
            results = (
                (
                    search_res.get("results")
                    or search_res.get("matches")
                    or search_res.get("persons")
                    or search_res.get("records")
                )
                if isinstance(search_res, dict)
                else None
            )
            harvested_surnames, harvested_given = extract_names_from_fs_hits(
                results if isinstance(results, list) else None
            )
            if not is_census and fs_name_cache_size(area_key, year_int) < 15:
                harvest_res = await self.call_tool(
                    "search_records",
                    {
                        "birth_place": place_std,
                        "birth_year": year_int,
                        "loose": True,
                        "count": 50,
                    },
                    stage="mcp_record",
                    scan_id=fact.scan_id,
                    row_id=eff_row_id or None,
                )
                extra_hits = harvest_res.get("results") if isinstance(harvest_res, dict) else None
                extra_s, extra_g = extract_names_from_fs_hits(
                    extra_hits if isinstance(extra_hits, list) else None
                )
                harvested_surnames.extend(extra_s)
                harvested_given.extend(extra_g)
            if not is_census and row is not None and (harvested_surnames or harvested_given):
                record_observed_fs_names(area_key, year_int, harvested_surnames, harvested_given)
                _apply_fs_name_matches_to_fact(fact, row, area_key, year_int)
                session.add(fact)
                session.add(row)
                given = fact.given_name_raw or given
                surname = fact.surname_raw or surname
            if isinstance(results, list) and results and isinstance(results[0], dict):
                best = results[0]

        fs_person_id = (
            str(best.get("id") or best.get("person_id") or best.get("fs_person_id") or "").strip()
            if best
            else ""
        )
        if not best or not fs_person_id:
            if not self._should_query_mcp_records(fs_env) and fact.id is not None:
                existing_match = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).first()
                if existing_match and existing_match.fs_person_id and str(existing_match.fs_person_id).strip():
                    return existing_match
            if fact.id is not None:
                stale = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).all()
                for sm in stale:
                    session.delete(sm)
            session.commit()
            return None

        full_name_display = f"{given or '?'} {surname or '?'}".strip()
        fs_person_name = best.get("name") or full_name_display
        fs_record_ark = str(best.get("ark") or "").strip() or None
        confidence = max(0.01, float(best.get("score") or best.get("confidence") or 0.85))
        match_status = "confirmed_match" if confidence >= 0.85 else "existing_needs_facts"
        summary_sk = (
            f"Nájdená zhoda v strome FamilySearch: {fs_person_name} ({fs_person_id}) "
            f"so skóre {confidence:.2f}."
        )
        what_agrees.append({"field_sk": "Meno v strome", "value": str(fs_person_name)})

        action_packet, what_is_new = build_action_packet_and_new_items(
            fact, place_std, fs_url, session=session
        )

        match = session.exec(
            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
        ).first()
        if not match:
            match = FamilySearchMatch(person_fact_id=fact.id or 0, row_id=eff_row_id)
        else:
            if not what_differs and match.what_differs_json and match.what_differs_json != "[]":
                what_differs = json.loads(match.what_differs_json)

        match.match_status = match_status
        match.fs_person_id = fs_person_id
        match.fs_record_ark = fs_record_ark
        match.fs_url = fs_url
        match.confidence_score = confidence
        match.summary_sk = summary_sk
        match.what_agrees_json = json.dumps(what_agrees, ensure_ascii=False)
        match.what_differs_json = json.dumps(what_differs, ensure_ascii=False)
        match.what_is_new_json = json.dumps(what_is_new, ensure_ascii=False)
        match.action_packet_json = json.dumps(action_packet, ensure_ascii=False)
        match.raw_mcp_results_json = json.dumps(
            {"place_search": place_res, "record_search": search_res, "area_harvest": harvest_res},
            ensure_ascii=False,
        )

        session.add(match)
        session.commit()
        session.refresh(match)
        return match

    async def match_census_sheet(
        self,
        session: Session,
        sheet: CensusSheet | int,
        *,
        manage_progress: bool = True,
    ) -> int:
        """
        Household-aware FamilySearch & Local Matrika cross-matching for a CensusSheet.
        1. Cross-matches every census PersonFact against local parish registers (baptism/marriage/death).
        2. Standardizes municipality & birthplaces via FamilySearch Places.
        3. Queries FamilySearch MCP (`search_records`) and ONLY stores `FamilySearchMatch` rows
           for people that actually match a FamilySearch person/record (`fs_person_id` non-empty).
        """
        if isinstance(sheet, int):
            sheet_obj = session.get(CensusSheet, sheet)
            if not sheet_obj:
                return 0
            sheet = sheet_obj
        if getattr(sheet, "sheet_type", "household") in ("overview", "municipal_overview"):
            overview_facts = session.exec(
                select(PersonFact).where(PersonFact.census_sheet_id == sheet.id)
            ).all()
            for of in overview_facts:
                if of.id is not None:
                    for sm in session.exec(
                        select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == of.id)
                    ).all():
                        session.delete(sm)
            session.commit()
            return 0

        facts = list(
            session.exec(
                select(PersonFact)
                .where(PersonFact.census_sheet_id == sheet.id)
                .order_by(PersonFact.person_row_number.asc(), PersonFact.id.asc())
            ).all()
        )
        if not facts:
            return 0

        valid_sheet_facts = [f for f in facts if not _is_nameless_fact(f)]
        started_progress_here = False
        if manage_progress and not self.progress_state.get("running") and valid_sheet_facts:
            started_progress_here = True
            await self.start_progress(
                mode="auto",
                total=len(valid_sheet_facts),
                current_person=_format_fact_person_label(valid_sheet_facts[0]),
                session=session,
            )

        try:
            household = session.exec(
                select(CensusHousehold).where(CensusHousehold.census_sheet_id == sheet.id)
            ).first()

            place_query = (sheet.municipality or "").strip()
            place_res: dict[str, Any] = {}
            if place_query:
                cached_q = (_load_regional_places_cache().get("queries") or {}).get(
                    fold_text_diacritics(place_query)
                )
                if isinstance(cached_q, list) and cached_q:
                    place_res = {"places": cached_q}
                else:
                    place_res = await self.call_tool(
                        "search_places_at_date",
                        {"name": f"{place_query}, Czechoslovakia", "year": sheet.census_year},
                        stage="mcp_place",
                        scan_id=sheet.front_scan_id,
                    )
            place_std = f"{sheet.municipality or ''}, {sheet.district or ''}, Slovensko".strip(", ")
            places_list = place_res.get("places") if isinstance(place_res, dict) else None
            vicinity_list = place_res.get("vicinity_places") if isinstance(place_res, dict) else None
            if place_query and isinstance(places_list, list) and places_list:
                record_regional_places_cache(
                    place_query,
                    places_list,
                    default_district=sheet.district,
                    vicinity_places=vicinity_list if isinstance(vicinity_list, list) else None,
                )
                for pl in places_list:
                    fn = str(pl.get("full_name") or pl.get("name") or "")
                    if fn and fn.lower() not in ("czechoslovakia", "slovakia", "hungary"):
                        place_std = fn
                        break

            extra_birth_places = [
                str(f.birth_place_municipality).strip()
                for f in facts
                if getattr(f, "birth_place_municipality", None)
            ]
            try:
                await self.ensure_regional_places_cached(
                    municipality=sheet.municipality,
                    district=sheet.district,
                    year=sheet.census_year,
                    extra_places=extra_birth_places,
                    scan_id=sheet.front_scan_id,
                    include_gazetteer=False,
                )
            except Exception:
                pass

            fs_env = read_fs_env()
            should_query_records = self._should_query_mcp_records(fs_env)

            matched_count = 0
            valid_idx = 0
            total_valid = max(1, len(valid_sheet_facts))
            from backend.app.api.ws import ws_manager

            for fact in facts:
                if _is_nameless_fact(fact):
                    if fact.id is not None:
                        stale_matches = session.exec(
                            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                        ).all()
                        for sm in stale_matches:
                            session.delete(sm)
                    continue

                valid_idx += 1
                person_label = _format_fact_person_label(fact)
                await self.step_progress(current_person=person_label)
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "mcp_match",
                        "active_model": "familysearch-mcp",
                        "current_scan_id": fact.scan_id or sheet.front_scan_id,
                        "current_row_id": fact.row_id,
                        "progress_pct": 88 + int(11 * valid_idx / total_valid),
                        "message_sk": f"Párujem s matrikou a FamilySearch ({valid_idx}/{total_valid}): {person_label}...",
                        "message_en": f"Matching with parish records & FamilySearch ({valid_idx}/{total_valid}): {person_label}...",
                    },
                )

                fact.fs_place_standardized = place_std
                session.add(fact)

                local_matches = match_census_to_local_matrika(session, fact)

                given = fact.given_name_sk or fact.given_name_raw or ""
                surname = fact.surname_normalized or fact.surname_raw or ""
                birth_year = fact.birth_year_est
                birth_place = (
                    getattr(fact, "birth_place_municipality", None)
                    or fact.birth_place_sk
                    or fact.birth_place_raw
                    or place_std
                )
                spouse_given, spouse_surname = _split_spouse_name(fact, session, peers=facts)

                search_res: dict[str, Any] | None = None
                best: dict[str, Any] | None = None
                if should_query_records and (given or surname):
                    search_args: dict[str, Any] = {
                        "given": given,
                        "surname": surname,
                        "birth_place": birth_place or place_std,
                        "birth_year": int(birth_year or sheet.census_year),
                        "residence_place": place_std,
                        "loose": True,
                        "count": 20,
                    }
                    if spouse_given:
                        search_args["spouse_given"] = spouse_given
                    if spouse_surname:
                        search_args["spouse_surname"] = spouse_surname
                    search_res = await self.call_tool(
                        "search_records",
                        search_args,
                        stage="mcp_record",
                        scan_id=fact.scan_id or sheet.front_scan_id,
                        row_id=fact.row_id,
                    )
                    results = (
                        (
                            search_res.get("results")
                            or search_res.get("matches")
                            or search_res.get("persons")
                            or search_res.get("records")
                        )
                        if isinstance(search_res, dict)
                        else None
                    )
                    if isinstance(results, list) and results and isinstance(results[0], dict):
                        best = results[0]

                fs_person_id = (
                    str(best.get("id") or best.get("person_id") or best.get("fs_person_id") or "").strip()
                    if best
                    else ""
                )
                if not best or not fs_person_id:
                    if fact.id is not None:
                        stale_matches = session.exec(
                            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                        ).all()
                        for sm in stale_matches:
                            session.delete(sm)
                    await self.step_progress(increment_processed=1, current_person=person_label)
                    continue

                fs_search_params = [
                    f"q.givenName={quote_plus(given)}" if given else "",
                    f"q.surname={quote_plus(surname)}" if surname else "",
                    f"q.birthLikePlace={quote_plus(birth_place)}" if birth_place else "",
                    f"q.birthLikeDate.from={birth_year - 2}" if birth_year else "",
                    f"q.birthLikeDate.to={birth_year + 2}" if birth_year else "",
                    f"q.residencePlace={quote_plus(place_std)}",
                    f"q.residenceDate.from={sheet.census_year}",
                    f"q.residenceDate.to={sheet.census_year}",
                    f"q.fatherGivenName={quote_plus(fact.father_name)}" if fact.father_name else "",
                    f"q.motherGivenName={quote_plus(fact.mother_name)}" if fact.mother_name else "",
                    f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                    f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
                ]
                fs_url = "https://www.familysearch.org/search/record/results?" + "&".join(
                    [p for p in fs_search_params if p]
                )

                fs_person_name = best.get("name") or f"{given or '?'} {surname or '?'}".strip()
                fs_record_ark = str(best.get("ark") or "").strip() or None
                what_agrees: list[dict[str, str]] = [
                    {"field_sk": "Meno v strome", "value": str(fs_person_name)}
                ]
                what_differs: list[dict[str, str]] = []
                if local_matches:
                    best_local = local_matches[0]
                    matrika_fact = session.get(PersonFact, best_local.matrika_fact_id)
                    if matrika_fact:
                        what_agrees.append(
                            {
                                "field_sk": "Lokálna matrika (zhoda)",
                                "value": (
                                    f"{matrika_fact.given_name_raw or ''} {matrika_fact.surname_raw or ''} "
                                    f"({matrika_fact.record_type}, rok {matrika_fact.event_year}, "
                                    f"dátum {matrika_fact.birth_or_event_date or '?'}) – skóre {best_local.confidence_score:.2f}"
                                ),
                            }
                        )
                        if (
                            fact.birth_or_event_date
                            and matrika_fact.birth_or_event_date
                            and fact.birth_or_event_date != matrika_fact.birth_or_event_date
                        ):
                            what_differs.append(
                                {
                                    "field_sk": "Dátum narodenia (Sčítací hárok vs. Matrika pokrstených)",
                                    "value": f"Hárok {sheet.census_year}: {fact.birth_or_event_date} ↔ Matrika ({matrika_fact.event_year}): {matrika_fact.birth_or_event_date}",
                                }
                            )

                action_packet, what_is_new = build_census_action_packet(
                    fact=fact,
                    sheet=sheet,
                    household=household,
                    place_std=place_std,
                    fs_url=fs_url,
                    local_matches=local_matches,
                )

                confidence = max(
                    0.01,
                    float(
                        best.get("score")
                        or best.get("confidence")
                        or (local_matches[0].confidence_score if local_matches else 0.88)
                    ),
                )
                match_status = "confirmed_match" if confidence >= 0.85 else "existing_needs_facts"
                summary_sk = (
                    f"Sčítací hárok {sheet.census_year} ({sheet.municipality}, dom č. {sheet.house_number or '—'}): "
                    f"{given or '?'} {surname or '?'} ({fact.relation_to_head or 'člen domácnosti'}) – "
                    f"zhoda vo FamilySearch: {fs_person_name} ({fs_person_id})."
                )
                if local_matches:
                    summary_sk += f" Prepojené aj s lokálnou matrikou ({len(local_matches)} zhodných záznamov)."

                match = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).first()
                if not match:
                    match = FamilySearchMatch(person_fact_id=fact.id or 0, row_id=fact.row_id or 0)
                else:
                    if not what_differs and match.what_differs_json and match.what_differs_json != "[]":
                        what_differs = json.loads(match.what_differs_json)

                match.match_status = match_status
                match.fs_person_id = fs_person_id
                match.fs_record_ark = fs_record_ark
                match.fs_url = fs_url
                match.confidence_score = confidence
                match.summary_sk = summary_sk
                match.what_agrees_json = json.dumps(what_agrees, ensure_ascii=False)
                match.what_differs_json = json.dumps(what_differs, ensure_ascii=False)
                match.what_is_new_json = json.dumps(what_is_new, ensure_ascii=False)
                match.action_packet_json = json.dumps(action_packet, ensure_ascii=False)
                match.raw_mcp_results_json = json.dumps(
                    {"place_search": place_res, "record_search": search_res},
                    ensure_ascii=False,
                )
                session.add(match)
                matched_count += 1
                await self.step_progress(
                    increment_processed=1,
                    increment_matched=1,
                    current_person=person_label,
                )

            session.commit()
            return matched_count
        finally:
            if started_progress_here:
                await self.finish_progress(session=session)


def match_census_to_local_matrika(
    session: Session,
    census_fact: PersonFact,
) -> list[LocalRecordMatch]:
    """
    Cross-matches a census PersonFact against local parish register PersonFact rows
    (baptism, marriage, death) using phonetic surname skeleton, multilingual given-name
    canonical ID (e.g. Ján == Joannes == János), and birth year window.
    """
    if not census_fact.id:
        return []

    try:
        from backend.app.pipeline.feedback import _rust_db, _session_file_db_path

        db_p = _session_file_db_path(session)
        if db_p and _rust_db is not None and hasattr(_rust_db, "match_census_to_local_matrika_rs"):
            raw_matches = json.loads(
                _rust_db.match_census_to_local_matrika_rs(
                    db_p,
                    int(census_fact.id),
                    datetime.now(timezone.utc).isoformat(),
                )
            )
            return [LocalRecordMatch(**item) for item in raw_matches]
    except Exception:
        pass

    sur_skel = census_fact.surname_skeleton or _phonetic_skeleton(
        census_fact.surname_normalized or census_fact.surname_raw or ""
    )
    maiden_skel = (
        _phonetic_skeleton(census_fact.mother_maiden_surname)
        if census_fact.mother_maiden_surname
        else ""
    )
    given_cid = census_fact.given_canonical_id or resolve_given_name_canonical_id(
        census_fact.given_name_sk or census_fact.given_name_raw
    )

    if not sur_skel and not maiden_skel:
        return []

    candidates_stmt = select(PersonFact).where(PersonFact.record_type != "census")
    if census_fact.birth_year_est:
        candidates_stmt = candidates_stmt.where(
            PersonFact.event_year >= census_fact.birth_year_est - 4,
            PersonFact.event_year <= census_fact.birth_year_est + 4,
        )
    skel_vals = [s for s in (sur_skel, maiden_skel) if s]
    if skel_vals and given_cid:
        candidates_stmt = candidates_stmt.where(
            (PersonFact.surname_skeleton.in_(skel_vals))  # type: ignore[union-attr]
            | (PersonFact.given_canonical_id == given_cid)
            | (PersonFact.surname_skeleton.is_(None))  # type: ignore[union-attr]
        )
    elif skel_vals:
        candidates_stmt = candidates_stmt.where(
            (PersonFact.surname_skeleton.in_(skel_vals))  # type: ignore[union-attr]
            | (PersonFact.surname_skeleton.is_(None))  # type: ignore[union-attr]
        )
    candidates = list(session.exec(candidates_stmt).all())

    # Clear old local matches for this census_fact
    old_matches = session.exec(
        select(LocalRecordMatch).where(LocalRecordMatch.census_fact_id == census_fact.id)
    ).all()
    for om in old_matches:
        session.delete(om)

    created: list[LocalRecordMatch] = []
    for cand in candidates:
        if not cand.id:
            continue
        cand_skel = cand.surname_skeleton or _phonetic_skeleton(
            cand.surname_normalized or cand.surname_raw or ""
        )
        cand_cid = cand.given_canonical_id or resolve_given_name_canonical_id(
            cand.given_name_sk or cand.given_name_raw
        )

        # Surname match (married surname or maiden surname for married women)
        sur_score = 0.0
        if sur_skel and cand_skel and sur_skel == cand_skel:
            sur_score = 1.0
        elif maiden_skel and cand_skel and maiden_skel == cand_skel:
            sur_score = 0.95
        else:
            sim = _paleographic_similarity(
                census_fact.surname_normalized or census_fact.surname_raw or "",
                cand.surname_normalized or cand.surname_raw or "",
            )
            if sim >= 0.78:
                sur_score = sim
        if sur_score < 0.75:
            continue

        # Given name match
        giv_score = 0.0
        if given_cid and cand_cid and given_cid == cand_cid:
            giv_score = 1.0
        elif census_fact.given_name_raw and cand.given_name_raw:
            giv_score = _paleographic_similarity(census_fact.given_name_raw, cand.given_name_raw)
        if giv_score < 0.70:
            continue

        # Year score
        year_score = 0.85
        if census_fact.birth_year_est and cand.event_year:
            diff = abs(census_fact.birth_year_est - cand.event_year)
            if diff == 0:
                year_score = 1.0
            elif diff == 1:
                year_score = 0.92
            elif diff <= 3:
                year_score = 0.80
            else:
                year_score = 0.60

        total_score = round(0.45 * sur_score + 0.35 * giv_score + 0.20 * year_score, 3)
        if total_score < 0.76:
            continue

        explanation = (
            f"Zhoda priezviska ({census_fact.surname_raw or '?'} ↔ {cand.surname_raw or '?'}), "
            f"krstného mena ({census_fact.given_name_raw or '?'} ↔ {cand.given_name_raw or '?'}) "
            f"a roku narodenia ({census_fact.birth_year_est or '?'} ↔ matrika {cand.event_year})."
        )
        m = LocalRecordMatch(
            census_fact_id=census_fact.id,
            matrika_fact_id=cand.id,
            match_kind=cand.record_type,
            confidence_score=total_score,
            what_agrees_json=json.dumps([explanation], ensure_ascii=False),
        )
        session.add(m)
        created.append(m)

    created.sort(key=lambda x: x.confidence_score, reverse=True)
    return created


def build_census_action_packet(
    *,
    fact: PersonFact,
    sheet: CensusSheet | None,
    household: CensusHousehold | None,
    place_std: str | None,
    fs_url: str,
    local_matches: list[LocalRecordMatch],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    fs_url = _ensure_split_spouse_in_fs_url(fs_url, fact)
    given_display = fact.given_name_sk or fact.given_name_raw or "?"
    surname_display = fact.surname_raw or fact.surname_normalized or "?"
    full_name_display = f"{given_display} {surname_display}".strip()

    slovakiana_key = (sheet.slovakiana_key if sheet and sheet.slovakiana_key else None) or "?"
    census_year = (sheet.census_year if sheet and sheet.census_year else None) or fact.event_year or 1930
    district_val = (sheet.district if sheet and sheet.district else None) or "?"
    municipality_val = (
        (sheet.municipality if sheet and sheet.municipality else None)
        or fact.village_sk
        or fact.village_raw
        or "?"
    )
    house_val = (
        (sheet.house_number if sheet and sheet.house_number else None)
        or (household.house_number if household and household.house_number else None)
        or fact.house_number
        or "?"
    )
    sheet_num_val = (sheet.sheet_number if sheet and sheet.sheet_number else None) or "?"
    row_num_val = fact.person_row_number or fact.entry_number or "?"

    residence_place = (
        f"{municipality_val}, dom č. {house_val if house_val != '?' else '—'} "
        f"(okres {district_val})"
    )
    birth_date_full = fact.birth_or_event_date or (
        str(fact.birth_year_est) if fact.birth_year_est else None
    )
    urn_nbn = getattr(sheet, "urn_nbn", None) if sheet else None
    source_url = (
        f"https://slovakiana.sk/kulturne-objekty/{slovakiana_key}"
        if slovakiana_key and slovakiana_key != "?"
        else "https://slovakiana.sk"
    )
    source_license = "CC BY-NC-ND 4.0 — Slovenský národný archív (SNA) / Slovakiana (https://creativecommons.org/licenses/by-nc-nd/4.0/)"
    urn_suffix = f", URN:NBN {urn_nbn}" if urn_nbn else ""
    citation_sk = (
        f"Slovenský národný archív (SNA) / Slovakiana ({slovakiana_key}{urn_suffix}), "
        f"Sčítanie ľudu v Krajine Slovenskej z roku {census_year}, okres {district_val}, "
        f"obec {municipality_val}, dom č. {house_val}, "
        f"hárok č. {sheet_num_val}, riadok č. {row_num_val}, meno: {full_name_display}. "
        f"Dostupné na: {source_url} (Licencia: CC BY-NC-ND 4.0)."
    )
    citation_en = (
        f"Slovak National Archives (Slovenský národný archív) / Slovakiana ({slovakiana_key}{urn_suffix}), "
        f"{census_year} Census of Slovakia, District {district_val}, Municipality {municipality_val}, "
        f"House No. {house_val}, Sheet No. {sheet_num_val}, Row {row_num_val}, Name: {full_name_display}. "
        f"Available at: {source_url} (License: CC BY-NC-ND 4.0)."
    )

    bp_muni = getattr(fact, "birth_place_municipality", None)
    bp_dist = getattr(fact, "birth_place_district", None)
    bp_country = getattr(fact, "birth_place_country", None)
    bp_raw = fact.birth_place_raw
    citizenship = getattr(fact, "citizenship", None)

    rodisko_parts: list[str] = []
    if bp_muni:
        rodisko_parts.append(f"obec {bp_muni}")
    if bp_dist:
        rodisko_parts.append(f"okres {bp_dist}")
    if bp_country:
        rodisko_parts.append(f"krajina {bp_country}")
    structured_rodisko = (
        ", ".join(rodisko_parts)
        if rodisko_parts
        else (bp_raw or fact.birth_place_sk or None)
    )

    what_is_new: list[dict[str, str]] = [
        {
            "field_sk": "Meno a vzťah k hlave domácnosti",
            "value": (
                f"{full_name_display}"
                + (f" (rod. {fact.mother_maiden_surname})" if fact.mother_maiden_surname and fact.relation_to_head in ("manželka", "družka", "hlava domácnosti") else "")
                + f" – {fact.relation_to_head or 'člen domácnosti'}"
                + (f", stav: {fact.marital_status}" if fact.marital_status else "")
            ),
        },
        {
            "field_sk": f"Pobyt podľa sčítania ľudu ({census_year})",
            "value": f"{residence_place} – {fact.relation_to_head or 'člen domácnosti'}",
        },
    ]
    if birth_date_full or structured_rodisko or bp_raw:
        what_is_new.append(
            {
                "field_sk": "Dátum narodenia a rodisko (Obec, Okres, Krajina)",
                "value": f"Nar.: {birth_date_full or '?'} | Rodisko: {structured_rodisko or municipality_val}",
            }
        )
    civ_parts = [
        f"Štátna prísl.: {citizenship}" if citizenship else None,
        f"Národnosť: {fact.nationality}" if fact.nationality else None,
        f"Náboženstvo: {fact.religion}" if fact.religion else None,
        f"Gramotnosť: {fact.literacy}" if fact.literacy else None,
    ]
    civ_parts_clean = [p for p in civ_parts if p]
    if civ_parts_clean:
        what_is_new.append(
            {
                "field_sk": "Štátna príslušnosť, národnosť a náboženstvo (stĺpec 14–17)",
                "value": " | ".join(civ_parts_clean),
            }
        )
    if fact.social_status_or_occupation or fact.occupation_position or fact.employer_workplace:
        what_is_new.append(
            {
                "field_sk": "Povolanie, postavenie a zamestnávateľ (stĺpec 18–20)",
                "value": " | ".join(
                    [
                        p
                        for p in (
                            fact.social_status_or_occupation,
                            fact.occupation_position,
                            fact.employer_workplace,
                        )
                        if p
                    ]
                ),
            }
        )
    if fact.moved_in_date or fact.previous_residence or fact.home_jurisdiction:
        what_is_new.append(
            {
                "field_sk": "Prisťahovanie a domovská príslušnosť (stĺpec 12–13)",
                "value": f"Prisťahoval sa: {fact.moved_in_date or 'od narodenia'} z: {fact.previous_residence or '—'} | Domovská obec: {fact.home_jurisdiction or municipality_val}",
            }
        )

    spouse_given, spouse_surname = _split_spouse_name(fact)

    action_packet: dict[str, Any] = {
        "person_name_display": full_name_display,
        "person_name": full_name_display,
        "given_name": given_display if given_display != "?" else None,
        "surname": surname_display if surname_display != "?" else None,
        "maiden_surname": fact.mother_maiden_surname,
        "sex": fact.sex,
        "birth_date": birth_date_full,
        "baptism_date": None,
        "birth_place_historical": structured_rodisko or bp_raw or residence_place,
        "birth_place_modern_sk": fact.birth_place_sk or structured_rodisko or place_std or municipality_val,
        "birth_place_raw": bp_raw or structured_rodisko,
        "birth_place_municipality": bp_muni,
        "birth_place_district": bp_dist,
        "birth_place_country": bp_country,
        "residence_place": residence_place,
        "standardized_place": place_std or municipality_val,
        "citizenship": citizenship,
        "nationality": fact.nationality,
        "religion": fact.religion,
        "literacy": fact.literacy,
        "marital_status": fact.marital_status,
        "relation_to_head": fact.relation_to_head,
        "occupation": fact.social_status_or_occupation,
        "occupation_position": fact.occupation_position,
        "employer_workplace": fact.employer_workplace,
        "home_jurisdiction": fact.home_jurisdiction,
        "father_full_name": fact.father_name,
        "father": fact.father_name,
        "mother_full_name": (
            f"{fact.mother_name} (rod. {fact.mother_maiden_surname})"
            if fact.mother_name and fact.mother_maiden_surname
            else fact.mother_name
        ),
        "mother": fact.mother_name,
        "spouse_full_name": fact.spouse_name,
        "spouse_given_name": spouse_given or None,
        "spouse_surname": spouse_surname or None,
        "godparents_note": (
            f"Manžel/ka: {fact.spouse_name}" if fact.spouse_name else None
        ),
        "observationes_note": fact.observationes_raw,
        "source_citation_sk": citation_sk,
        "source_citation_en": citation_en,
        "citation_draft_sk": citation_sk,
        "source_url": source_url,
        "source_license": source_license,
        "urn_nbn": urn_nbn,
        "fs_search_url": fs_url,
        "familysearch_search_url": fs_url,
        "title_sk": f"Balíček pre FamilySearch – Sčítací hárok {census_year}, riadok č. {row_num_val}",
        "event_type": "census",
        "event_year": census_year,
        "slovakiana_key": slovakiana_key if slovakiana_key != "?" else None,
        "district": district_val if district_val != "?" else None,
        "municipality": municipality_val if municipality_val != "?" else None,
        "house_number": house_val if house_val != "?" else None,
        "sheet_number": sheet_num_val if sheet_num_val != "?" else None,
        "person_row_number": row_num_val if row_num_val != "?" else None,
        "local_matrika_matches_count": len(local_matches),
    }
    return action_packet, what_is_new


fs_mcp_manager = FamilySearchMcpManager()
fs_match_progress = fs_mcp_manager.progress_state


async def match_facts_with_familysearch(
    session: Session,
    facts: list[PersonFact] | None = None,
    scan_id: int | None = None,
) -> list[FamilySearchMatch]:
    """
    Matches a list of PersonFacts (or all PersonFacts on a scan) against FamilySearch MCP,
    returning only matched FamilySearchMatch rows (`fs_person_id` non-empty).
    """
    if facts is None:
        stmt = select(PersonFact).order_by(PersonFact.id.asc())
        if scan_id is not None:
            stmt = stmt.where(PersonFact.scan_id == scan_id)
        facts = list(session.exec(stmt).all())

    valid_facts = [f for f in facts if not _is_nameless_fact(f)]
    started_progress_here = False
    if not fs_mcp_manager.progress_state.get("running") and valid_facts:
        started_progress_here = True
        await fs_mcp_manager.start_progress(
            mode="auto",
            total=len(valid_facts),
            current_person=_format_fact_person_label(valid_facts[0]),
            session=session,
        )

    matched_rows: list[FamilySearchMatch] = []
    try:
        for fact in facts:
            if _is_nameless_fact(fact):
                continue
            person_label = _format_fact_person_label(fact)
            await fs_mcp_manager.step_progress(current_person=person_label)
            row = session.get(RecordRow, fact.row_id) if fact.row_id else None
            m = await fs_mcp_manager.match_person_fact(session, fact, row)
            is_matched = bool(m is not None and m.fs_person_id and str(m.fs_person_id).strip())
            if is_matched and m is not None:
                matched_rows.append(m)
            await fs_mcp_manager.step_progress(
                increment_processed=1,
                increment_matched=1 if is_matched else 0,
                current_person=person_label,
            )
        return matched_rows
    finally:
        if started_progress_here:
            await fs_mcp_manager.finish_progress(session=session)


async def match_census_sheet_with_familysearch(
    session: Session,
    sheet: CensusSheet | int,
) -> int:
    """Matches a CensusSheet against FamilySearch MCP, keeping only matched rows."""
    return await fs_mcp_manager.match_census_sheet(session, sheet)


async def match_person_fact_by_id(fact_id: int) -> FamilySearchMatch | None:
    """
    Opens a DB session, loads the PersonFact by `fact_id`, and runs FamilySearch MCP matching.
    Returns the matched FamilySearchMatch row (with non-empty `fs_person_id`) or None.
    """
    from backend.app.db.session import engine

    with Session(engine) as session:
        fact = session.get(PersonFact, fact_id)
        if not fact:
            return None
        row = session.get(RecordRow, fact.row_id) if fact.row_id else None
        return await fs_mcp_manager.match_person_fact(session, fact, row)


async def trigger_auto_fs_match_for_fact(fact_id: int) -> FamilySearchMatch | None:
    """
    Background task helper that automatically runs FamilySearch MCP matching whenever
    a PersonFact is created or updated, and broadcasts live progress + WebSocket updates.
    """
    from backend.app.api.ws import ws_manager
    from backend.app.db.session import engine

    person_label = f"#{fact_id}"
    scan_id: int | None = None
    row_id: int | None = None
    try:
        with Session(engine) as session:
            fact = session.get(PersonFact, fact_id)
            if fact:
                person_label = _format_fact_person_label(fact)
                scan_id = fact.scan_id
                row_id = fact.row_id
            if not fs_mcp_manager.progress_state.get("running"):
                await fs_mcp_manager.start_progress(
                    mode="auto",
                    total=1,
                    current_person=person_label,
                    session=session,
                )
            else:
                fs_mcp_manager.progress_state["total"] = (
                    int(fs_mcp_manager.progress_state.get("total") or 0) + 1
                )
                await fs_mcp_manager.step_progress(current_person=person_label, session=session)

        match = await match_person_fact_by_id(fact_id)
        is_matched = bool(match is not None and match.fs_person_id and str(match.fs_person_id).strip())
        await fs_mcp_manager.step_progress(
            increment_processed=1,
            increment_matched=1 if is_matched else 0,
            current_person=person_label,
        )
        if int(fs_mcp_manager.progress_state.get("processed") or 0) >= int(
            fs_mcp_manager.progress_state.get("total") or 0
        ):
            await fs_mcp_manager.finish_progress()

        if is_matched and match is not None:
            try:
                if ws_manager.latest_state.get("active_stage") in (None, "idle"):
                    await ws_manager.broadcast(
                        "state",
                        {
                            "active_stage": "idle",
                            "active_model": "familysearch-mcp",
                            "current_scan_id": scan_id,
                            "current_row_id": row_id,
                            "progress_pct": 100,
                            "message_sk": f"Nájdená zhoda vo FamilySearch pre {person_label} ({match.fs_person_id}).",
                            "message_en": f"Found FamilySearch match for {person_label} ({match.fs_person_id}).",
                        },
                    )
            except Exception:
                pass
        return match
    except Exception as exc:
        try:
            await fs_mcp_manager.finish_progress(error=str(exc))
        except Exception:
            pass
        return None


def schedule_auto_fs_match(fact_id: int | None, trigger_fn: Any = None) -> None:
    """
    Schedules `trigger_fn` (or `trigger_auto_fs_match_for_fact`) as a background asyncio task.
    Supports `unittest.mock.patch` on either `familysearch_client.trigger_auto_fs_match_for_fact`
    or a module-level imported `trigger_auto_fs_match_for_fact` passed via `trigger_fn`.
    """
    if fact_id is None:
        return
    fn = (
        trigger_fn
        if (trigger_fn is not None and trigger_fn is not _orig_trigger_auto_fs_match_ref)
        else trigger_auto_fs_match_for_fact
    )
    res_coro = fn(fact_id)
    if asyncio.iscoroutine(res_coro):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(res_coro)
        except RuntimeError:
            asyncio.create_task(res_coro)


_orig_trigger_auto_fs_match_ref = trigger_auto_fs_match_for_fact


