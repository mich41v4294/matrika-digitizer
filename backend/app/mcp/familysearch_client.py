from __future__ import annotations

import asyncio
import hashlib
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
    normalize_village,
    record_observed_fs_names,
    record_regional_places_cache,
    resolve_area_key,
    resolve_given_name_canonical_id,
)


ALLOWED_FS_ENV_KEYS: frozenset[str] = frozenset(
    {"FS_ACCESS_TOKEN", "FS_ENVIRONMENT", "FS_CLIENT_ID", "FS_TIMEOUT"}
)

FS_IDENTIFIER_RE: re.Pattern[str] = re.compile(r"\b([A-Za-z0-9]{4}-[A-Za-z0-9]{3,4})\b")
_RECORD_ARK_RE: re.Pattern[str] = re.compile(r"(?:ark:/61903/)?[123]:\d+:", re.IGNORECASE)
_TREE_PERSON_URL_RE: re.Pattern[str] = re.compile(
    r"(?:ark:/61903/4:1:|/platform/tree/persons/|/(?:[a-z]{2}/)?tree/person/(?:details/|about/|vitals/|sources/|memories/|timeline/)?)([A-Za-z0-9]{4}-[A-Za-z0-9]{3,4})\b",
    re.IGNORECASE,
)

_FOREIGN_COUNTRIES_REJECT: tuple[str, ...] = (
    "united states",
    "u.s.a",
    "brazil",
    "argentina",
    "mexico",
    "canada",
    "england",
    "scotland",
    "wales",
    "ireland",
    "australia",
    "new zealand",
    "philippines",
    "indonesia",
    "india",
    "south africa",
    "colombia",
    "peru",
    "chile",
    "venezuela",
    "puerto rico",
)


_HISTORICAL_VILLAGE_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "horny cepen": ("horny cepen", "felso csopony", "felsocsopony", "csopony", "cepen", "cegsen"),
    "dolny cepen": ("dolny cepen", "also csopony", "alsocsopony", "csopony", "cepen"),
    "stredny cepen": ("stredny cepen", "csopony", "cepen"),
    "cegsen": ("horny cepen", "felso csopony", "felsocsopony", "csopony", "cepen"),
    "surovce": ("surovce", "velke surovce", "male surovce", "nagy sur", "nagysur", "kis sur", "kissur", "sur"),
    "velke surovce": ("velke surovce", "surovce", "nagy sur", "nagysur"),
    "male surovce": ("male surovce", "surovce", "kis sur", "kissur"),
    "valtasur": ("valtasur", "valtov sur", "valta sur"),
    "valtov sur": ("valtov sur", "valtasur", "valta sur"),
    "varov sur": ("varov sur", "varra sur", "varrasur"),
    "zemiansky sur": ("zemiansky sur", "zemianske sady", "nemes sur", "nemessur"),
    "zemianske sady": ("zemianske sady", "zemiansky sur", "nemes sur", "nemessur"),
    "sered": ("sered", "szered"),
    "trnava": ("trnava", "nagyszombat"),
    "hlohovec": ("hlohovec", "galgocz", "galgoc"),
    "galanta": ("galanta", "galanthai"),
    "pavlice": ("pavlice", "pald"),
    "dvorniky": ("dvorniky", "udvarnok"),
    "klacany": ("klacany", "kelecseny"),
    "trakovice": ("trakovice", "karkocz", "karkoc"),
}

_REGIONAL_COUNTY_TOKENS: tuple[str, ...] = (
    "trnava",
    "nagyszombat",
    "sered",
    "szered",
    "galanta",
    "hlohovec",
    "galgoc",
    "galgocz",
    "piestany",
    "nitra",
    "nyitra",
    "pozsony",
    "bratislava",
    "presporok",
    "pressburg",
    "trencin",
    "trencsen",
    "topolcany",
    "pezinok",
    "senec",
    "komarom",
    "komarno",
    "bars",
    "tekov",
)


def is_valid_fs_identifier(val: Any) -> bool:
    """Return True if `val` is a canonical FamilySearch identifier in `XXXX-YYY` format."""
    if not val or not isinstance(val, str):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9]{4}-[A-Za-z0-9]{3,4}", val.strip()))


def normalize_fs_identifier(raw_val: Any, fallback_seed: str | None = None) -> str:
    """
    Extract and normalize a genuine FamilySearch Tree Person identifier (`4:1:XXXX-YYY`)
    into `XXXX-YYY` format (e.g. `LD9Q-GJ7`, `KWCZ-001`, `KWCJ-4N2`).
    Strictly rejects FamilySearch Historical Record / Persona / Artifact ARKs (`1:1:`, `1:2:`, `2:1:`, `2:2:`, `3:1:`)
    and never fabricates fake IDs — returns `""` when no valid Tree Person `XXXX-YYY` identifier is present.
    """
    _ = fallback_seed
    explicit_tree_candidates: list[str] = []
    generic_candidates: list[str] = []

    if isinstance(raw_val, dict):
        is_record_entry = str(raw_val.get("id_kind") or "").lower() == "record"
        for ark_key in ("ark", "fs_record_ark", "record_ark", "match_id"):
            ark_v = str(raw_val.get(ark_key) or "").strip()
            if ark_v and _RECORD_ARK_RE.search(ark_v):
                is_record_entry = True

        idents = raw_val.get("identifiers")
        if isinstance(idents, dict):
            for vals in idents.values():
                if isinstance(vals, list):
                    for item in vals:
                        if item and str(item).strip():
                            explicit_tree_candidates.append(str(item).strip())
                elif vals and str(vals).strip():
                    explicit_tree_candidates.append(str(vals).strip())

        links_obj = raw_val.get("links")
        if isinstance(links_obj, dict):
            person_link = links_obj.get("person")
            if isinstance(person_link, dict) and person_link.get("href"):
                explicit_tree_candidates.append(str(person_link["href"]).strip())
            elif isinstance(person_link, str) and person_link.strip():
                explicit_tree_candidates.append(person_link.strip())

        for url_key in ("fs_url", "url", "ark", "fs_record_ark"):
            uv = str(raw_val.get(url_key) or "").strip()
            if uv and not _RECORD_ARK_RE.search(uv):
                m_tree = _TREE_PERSON_URL_RE.search(uv)
                if m_tree:
                    explicit_tree_candidates.append(m_tree.group(1).upper())

        if not is_record_entry:
            for key in ("fs_person_id", "person_id", "fs_id", "id"):
                v = raw_val.get(key)
                if v is not None and str(v).strip():
                    s_val = str(v).strip()
                    if s_val == "primaryPerson" or re.match(r"^[pr]_\d+$", s_val, flags=re.IGNORECASE):
                        continue
                    if _RECORD_ARK_RE.search(s_val):
                        continue
                    generic_candidates.append(s_val)
    elif raw_val is not None:
        s_val = str(raw_val).strip()
        if (
            s_val
            and s_val != "primaryPerson"
            and not re.match(r"^[pr]_\d+$", s_val, flags=re.IGNORECASE)
            and not _RECORD_ARK_RE.search(s_val)
        ):
            generic_candidates.append(s_val)

    for cand in explicit_tree_candidates + generic_candidates:
        if _RECORD_ARK_RE.search(cand):
            continue
        m_url = _TREE_PERSON_URL_RE.search(cand)
        if m_url:
            return m_url.group(1).upper()
        cleaned = cand.split("?", 1)[0].split("#", 1)[0].strip("/")
        if cleaned.upper().startswith("4:1:"):
            cleaned = cleaned[4:]
        if is_valid_fs_identifier(cleaned):
            return cleaned.upper()

    return ""


def _sanitize_env_val(val: Any) -> str:
    return re.sub(r"[\r\n\x00]+", "", str(val or "")).strip()


def _sanitize_fs_token(val: Any) -> str:
    cleaned = _sanitize_env_val(val)
    cleaned = re.sub(r"^Bearer\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


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
                env_map[key] = _sanitize_fs_token(v) if key == "FS_ACCESS_TOKEN" else _sanitize_env_val(v)
    return env_map


def write_fs_env(updates: dict[str, str]) -> dict[str, str]:
    current = read_fs_env()
    for k, v in updates.items():
        if k in ALLOWED_FS_ENV_KEYS and v is not None:
            current[k] = _sanitize_fs_token(v) if k == "FS_ACCESS_TOKEN" else _sanitize_env_val(v)
    lines = [f"{k}={_sanitize_env_val(v)}" for k, v in current.items() if k in ALLOWED_FS_ENV_KEYS]
    settings.fs_env_file.parent.mkdir(parents=True, exist_ok=True)
    settings.fs_env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "fs_mcp_manager" in globals():
        fs_mcp_manager._cached_auth_status = None
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
    given_clean = (fact.given_name_raw or fact.given_name_sk or "").strip(" ?—–-")
    surname_clean = (fact.surname_raw or fact.surname_normalized or "").strip(" ?—–-")
    if not given_clean and not surname_clean:
        return True
    return not _is_valid_name_token(given_clean) and not _is_valid_name_token(surname_clean)


def _has_sufficient_identity_for_fs_match(fact: PersonFact | None) -> bool:
    """
    Tightened identity requirement: a PersonFact must have BOTH a valid given name
    AND a valid surname (or maiden surname) before querying or matching in FamilySearch.
    """
    if _is_nameless_fact(fact) or fact is None:
        return False
    has_given = any(
        _is_valid_name_token(v)
        for v in (fact.given_name_sk, fact.given_name_raw, fact.given_name_hu)
    )
    has_surname = any(
        _is_valid_name_token(v)
        for v in (fact.surname_normalized, fact.surname_raw, fact.mother_maiden_surname)
    )
    return bool(has_given and has_surname)


def _split_parent_name(
    full_name: str | None,
    fallback_surname: str | None = None,
) -> tuple[str, str]:
    """Split a parent's full name into `(given, surname)` for tightened FamilySearch queries and scoring."""
    raw = str(full_name or "").strip()
    fb_sur = str(fallback_surname or "").strip(" ?—–-.,;/")
    if not raw:
        return "", (fb_sur if _is_valid_name_token(fb_sur) else "")
    cleaned = re.sub(r"\([^)]*\)", "", raw)
    cleaned = re.sub(
        r"\b(?:rod\.|rodená|rodený|née|colonus|inquilinus|zeller|r\.?\s*cath\.?|ev\.?\s*a\.?\s*c\.?)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    tokens = [t.strip(" ,.;/-—–?") for t in cleaned.split() if _is_valid_name_token(t)]
    if not tokens:
        return "", (fb_sur if _is_valid_name_token(fb_sur) else "")
    if len(tokens) == 1:
        return tokens[0], (fb_sur if _is_valid_name_token(fb_sur) else "")
    return " ".join(tokens[:-1]), (fb_sur if _is_valid_name_token(fb_sur) else tokens[-1])


def _strip_feminine_surname_suffix(surname: str | None) -> str:
    folded = fold_text_diacritics(str(surname or "").strip(" ?—–-.,;/\"'()[]")).lower()
    if not folded:
        return ""
    for suffix in ("ova", "owna", "ovna", "na", "ne"):
        if len(folded) > len(suffix) + 2 and folded.endswith(suffix):
            return folded[: -len(suffix)]
    if len(folded) > 4 and folded.endswith("a"):
        return folded[:-1]
    return folded


def _has_non_latin_script(text: str | None) -> bool:
    """Return True if `text` contains Cyrillic, Greek, Arabic, Hebrew, or CJK characters."""
    if not text:
        return False
    return bool(re.search(r"[\u0370-\u03FF\u0400-\u052F\u0590-\u06FF\u4E00-\u9FFF]", str(text)))


def _extract_candidate_name_parts(full_name: str | None) -> tuple[list[str], list[str]]:
    """
    Extract candidate `(given_names, surnames)` from a FamilySearch display name,
    handling both Western (`Given Surname`) and Hungarian/Eastern (`Surname Given`) orders.
    """
    if not full_name:
        return [], []
    parts = [p.strip(" ,.;/()[]\"'") for p in str(full_name).split() if p and p not in ("-", "—")]
    parts = [p for p in parts if _is_valid_name_token(p)]
    if not parts:
        return [], []
    if len(parts) == 1:
        if is_known_given_name(parts[0]) or resolve_given_name_canonical_id(parts[0]):
            return [parts[0]], []
        return [], [parts[0]]

    givens: list[str] = [" ".join(parts[:-1])]
    surnames: list[str] = [parts[-1]]

    # Check Hungarian / Eastern order: "Surname GivenName" (e.g. "Kovács János", "Toman Mária")
    first_is_given = bool(is_known_given_name(parts[0]) or resolve_given_name_canonical_id(parts[0]))
    last_is_given = bool(is_known_given_name(parts[-1]) or resolve_given_name_canonical_id(parts[-1]))
    if last_is_given and not first_is_given:
        east_giv = " ".join(parts[1:])
        east_sur = parts[0]
        if east_giv not in givens:
            givens.append(east_giv)
        if east_sur not in surnames:
            surnames.append(east_sur)
    return givens, surnames


def _expand_place_tokens(raw_places: list[Any]) -> tuple[set[str], set[str]]:
    """
    Returns `(village_tokens, county_tokens)` from a list of place strings, expanding
    historical Slovak/Hungarian village aliases (e.g. `Horný Čepeň` <-> `Felső-Csöpöny`).
    """
    village_tokens: set[str] = set()
    county_tokens: set[str] = set()
    ignored = {
        "slovensko",
        "slovakia",
        "czechoslovakia",
        "ceskoslovensko",
        "hungary",
        "uhorsko",
        "madarsko",
        "austria",
        "europe",
        "obec",
        "okres",
        "krajina",
    }

    for rp in raw_places:
        if not rp or not str(rp).strip():
            continue
        raw_str = str(rp).strip()
        v_sk, v_hu, _ = normalize_village(raw_str)
        segments = [seg.strip() for seg in re.split(r"[,/|()]+", raw_str) if seg.strip()]
        first_seg = segments[0] if segments else raw_str
        first_clean = re.sub(r"^(?:obec|okres|krajina|okr\.)\s+", "", first_clean := first_seg, flags=re.IGNORECASE).strip()
        first_clean = re.sub(r"\b(?:dom\s*)?č\.?\s*\d+.*$", "", first_clean, flags=re.IGNORECASE).strip()

        for cand_v in (v_sk, v_hu, first_clean):
            if not cand_v:
                continue
            folded = fold_text_diacritics(cand_v).lower().strip(" .-")
            folded = re.sub(r"^h\.\s*", "horny ", folded)
            folded = re.sub(r"^d\.\s*", "dolny ", folded)
            if len(folded) < 3 or folded in ignored:
                continue
            village_tokens.add(folded)
            for eq in _HISTORICAL_VILLAGE_EQUIVALENTS.get(folded, ()):
                village_tokens.add(eq)
            # Also add core stem without directional prefix (e.g. "horny cepen" -> "cepen")
            core = re.sub(
                r"^(?:horny|dolny|stredny|velky|velke|maly|male|felso|also|nagy|kis)\s+",
                "",
                folded,
            ).strip()
            if len(core) >= 4 and core not in ignored:
                village_tokens.add(core)
                for eq in _HISTORICAL_VILLAGE_EQUIVALENTS.get(core, ()):
                    village_tokens.add(eq)

        full_folded = fold_text_diacritics(raw_str).lower()
        for c_tok in _REGIONAL_COUNTY_TOKENS:
            if re.search(rf"\b{re.escape(c_tok)}\b", full_folded):
                county_tokens.add(c_tok)

    return village_tokens, county_tokens


def _score_surname_match(fact: PersonFact, cand_surnames: list[str]) -> float:
    """
    Strictly score candidate surname(s) against `fact` surnames (married, normalized, maiden, candidates).
    Returns `0.0` if no surname reaches the tightened threshold (`>= 0.85`).
    """
    fact_primary = [
        s.strip()
        for s in (fact.surname_normalized, fact.surname_raw, *(fact.surname_candidates or []))
        if _is_valid_name_token(s)
    ]
    fact_maiden = [fact.mother_maiden_surname.strip()] if _is_valid_name_token(fact.mother_maiden_surname) else []
    valid_cand_surnames = [s.strip() for s in cand_surnames if _is_valid_name_token(s)]
    if not valid_cand_surnames or (not fact_primary and not fact_maiden):
        return 0.0

    best_score = 0.0
    for c_sur in valid_cand_surnames:
        if _has_non_latin_script(c_sur):
            continue
        c_fold = fold_text_diacritics(c_sur).lower()
        c_stem = _strip_feminine_surname_suffix(c_sur)
        c_skel = _phonetic_skeleton(c_stem or c_sur)

        for f_sur in fact_primary:
            f_fold = fold_text_diacritics(f_sur).lower()
            f_stem = _strip_feminine_surname_suffix(f_sur)
            f_skel = _phonetic_skeleton(f_stem or f_sur)
            if c_fold == f_fold or (c_stem and f_stem and c_stem == f_stem):
                return 1.0
            sim = max(
                _paleographic_similarity(f_sur, c_sur),
                _paleographic_similarity(f_stem, c_stem) if (f_stem and c_stem) else 0.0,
            )
            if c_skel and f_skel and c_skel == f_skel and sim >= 0.85:
                best_score = max(best_score, max(0.90, sim))
            elif sim >= 0.88:
                best_score = max(best_score, sim)

        for m_sur in fact_maiden:
            m_fold = fold_text_diacritics(m_sur).lower()
            m_stem = _strip_feminine_surname_suffix(m_sur)
            m_skel = _phonetic_skeleton(m_stem or m_sur)
            if c_fold == m_fold or (c_stem and m_stem and c_stem == m_stem):
                best_score = max(best_score, 0.96)
            else:
                sim = max(
                    _paleographic_similarity(m_sur, c_sur),
                    _paleographic_similarity(m_stem, c_stem) if (m_stem and c_stem) else 0.0,
                )
                if c_skel and m_skel and c_skel == m_skel and sim >= 0.85:
                    best_score = max(best_score, max(0.86, sim * 0.96))
                elif sim >= 0.88:
                    best_score = max(best_score, sim * 0.96)

    return round(best_score, 3) if best_score >= 0.85 else 0.0


def _score_given_name_match(fact: PersonFact, cand_givens: list[str]) -> float:
    """
    Strictly score candidate given name(s) against `fact` given names (Slovak, Latin, Hungarian).
    Returns `0.0` if no given name reaches the tightened threshold (`>= 0.85`).
    """
    fact_givens = [
        g.strip()
        for g in (fact.given_name_sk, fact.given_name_raw, fact.given_name_hu)
        if _is_valid_name_token(g)
    ]
    valid_cand_givens = [g.strip() for g in cand_givens if _is_valid_name_token(g)]
    if not fact_givens or not valid_cand_givens:
        return 0.0

    fact_cids: set[str] = set()
    fact_equivs: set[str] = set()
    fact_tokens: list[str] = []
    for fg in fact_givens:
        for tok in re.split(r"[\s,/()-]+", fg):
            tok_clean = tok.strip(" .?")
            if not _is_valid_name_token(tok_clean):
                continue
            fact_tokens.append(tok_clean)
            cid = resolve_given_name_canonical_id(tok_clean)
            if cid:
                fact_cids.add(cid)
            for eq in get_given_name_equivalents(tok_clean):
                fact_equivs.add(fold_text_diacritics(eq).lower())
            fact_equivs.add(fold_text_diacritics(tok_clean).lower())

    best_score = 0.0
    for cg in valid_cand_givens:
        if _has_non_latin_script(cg):
            continue
        for c_tok in re.split(r"[\s,/()-]+", cg):
            c_clean = c_tok.strip(" .?")
            if not _is_valid_name_token(c_clean):
                continue
            c_fold = fold_text_diacritics(c_clean).lower()
            if c_fold in fact_equivs:
                return 1.0
            c_cid = resolve_given_name_canonical_id(c_clean)
            if c_cid and c_cid in fact_cids:
                return 1.0
            for f_tok in fact_tokens:
                sim = _paleographic_similarity(f_tok, c_clean)
                if sim >= 0.86:
                    best_score = max(best_score, sim)

    return round(best_score, 3) if best_score >= 0.85 else 0.0


def _extract_year_from_text(val: Any) -> int | None:
    if val is None:
        return None
    if isinstance(val, int) and 1500 <= val <= 2025:
        return val
    m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", str(val))
    return int(m.group(1)) if m else None


def _normalize_sex_code(val: Any) -> str | None:
    if not val:
        return None
    s = str(val).strip().lower()
    if s in ("m", "male", "muž", "muz", "chlapec", "masculinum"):
        return "M"
    if s in ("f", "female", "žena", "zena", "dievča", "femininum", "ž", "z"):
        return "F"
    return None


def _build_alternate_surnames_for_tree_search(fact: PersonFact, primary_surname: str) -> list[str]:
    """
    Build up to 3 alternate surnames for FamilySearch Tree Person Search (`q.surname.1`, `q.surname.2`),
    including the masculine/root stem of Slovak feminine surnames (e.g. `Tomanová` -> `Toman`)
    and `mother_maiden_surname` (plus its stem).
    """
    alts: list[str] = []
    seen = {primary_surname.strip().lower()} if primary_surname else set()

    def _add(val: str | None) -> None:
        if not val or not _is_valid_name_token(val):
            return
        c = str(val).strip(" ?—–-.,;/\"'()[]")
        if len(c) >= 2 and c.lower() not in seen:
            seen.add(c.lower())
            alts.append(c)

    def _strip_fem_preserve_case(s: str | None) -> str:
        if not s:
            return ""
        clean = str(s).strip(" ?—–-.,;/\"'()[]")
        for suf in ("ová", "ova", "ówna", "owna", "ovna"):
            if len(clean) > len(suf) + 2 and clean.lower().endswith(suf):
                return clean[: -len(suf)]
        return ""

    _add(_strip_fem_preserve_case(primary_surname))
    if fact.mother_maiden_surname:
        _add(fact.mother_maiden_surname)
        _add(_strip_fem_preserve_case(fact.mother_maiden_surname))
    for sc in fact.surname_candidates or []:
        _add(sc)
        _add(_strip_fem_preserve_case(sc))
    return alts[:3]


def _normalize_record_ark(raw_ark: Any, fs_id: str | None = None, *, is_record_hit: bool = False) -> str | None:
    """
    Normalizes a FamilySearch Tree Person persistent identifier (`4:1:XXXX-YYY`).
    Rejects `1:1:` / `2:2:` / `3:1:` record ARKs so record IDs are never stored as person ARKs.
    """
    _ = is_record_hit
    cleaned = str(raw_ark or "").strip()
    if cleaned and _RECORD_ARK_RE.search(cleaned):
        return None
    if fs_id and is_valid_fs_identifier(fs_id):
        return f"4:1:{fs_id.strip().upper()}"
    return None


def _build_candidate_fs_url(fs_id: str, ark: str | None = None, fallback_search_url: str = "") -> str:
    """Return the direct FamilySearch Family Tree Person URL (`https://www.familysearch.org/tree/person/XXXX-YYY`)."""
    _ = ark
    if is_valid_fs_identifier(fs_id):
        return f"https://www.familysearch.org/tree/person/{fs_id.strip().upper()}"
    return fallback_search_url


def evaluate_fs_match_candidates(
    fact: PersonFact,
    hits: list[Any] | None,
    *,
    place_std: str | None = None,
    is_census: bool = False,
    local_matches: list[LocalRecordMatch] | None = None,
    session: Session | None = None,
    default_fs_url: str = "",
) -> list[dict[str, Any]]:
    """
    Tightly evaluates, filters, and ranks FamilySearch search/match hits against `fact`.
    Every returned candidate has:
    - a genuine `XXXX-YYY` FamilySearch identifier (`fs_person_id` / `fs_id`),
    - an accurate `fs_record_ark` (`1:1:XXXX-YYY` for record personas) and direct `fs_url`,
    - verified Latin-script given name & surname (`>= 0.85`),
    - verified birth/christening year within `±2`/`±3` years when `target_birth_year` is known,
    - verified village or regional county match (rejecting country-only or foreign matches),
    - overall confidence score `>= 0.86`.
    """
    if not hits or not _has_sufficient_identity_for_fs_match(fact):
        return []

    fact_sex = _normalize_sex_code(fact.sex)
    target_birth_year = (
        fact.birth_year_est
        if is_census
        else (fact.event_year if fact.record_type == "baptism" else (fact.birth_year_est or fact.event_year))
    )
    has_exact_birth_date = bool(
        fact.birth_or_event_date and re.search(r"\b\d{1,2}\s*[\.\s]\s*[A-Za-zÁ-ž0-9]+", str(fact.birth_or_event_date))
    )
    max_year_diff = 2 if (not is_census or has_exact_birth_date) else 3

    fact_places_raw = [
        p
        for p in (
            getattr(fact, "birth_place_municipality", None),
            fact.birth_place_sk,
            fact.birth_place_raw,
            fact.village_sk,
            fact.village_hu,
            fact.village_raw,
            place_std,
        )
        if p and str(p).strip()
    ]
    fact_village_tokens, fact_county_tokens = _expand_place_tokens(fact_places_raw)
    fact_places_joined_lower = " ".join(str(p).lower() for p in fact_places_raw)

    father_giv, father_sur = _split_parent_name(fact.father_name, fact.surname_normalized or fact.surname_raw)
    mother_giv, mother_sur = _split_parent_name(fact.mother_name, fact.mother_maiden_surname)
    spouse_giv, spouse_sur = _split_spouse_name(fact, session)

    local_agrees: list[dict[str, str]] = []
    local_differs: list[dict[str, str]] = []
    if is_census and local_matches and session is not None:
        best_local = local_matches[0]
        matrika_fact = session.get(PersonFact, best_local.matrika_fact_id)
        if matrika_fact:
            local_agrees.append(
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
                local_differs.append(
                    {
                        "field_sk": "Dátum narodenia (Sčítací hárok vs. Matrika pokrstených)",
                        "value": f"Hárok {fact.event_year or 1930}: {fact.birth_or_event_date} ↔ Matrika ({matrika_fact.event_year}): {matrika_fact.birth_or_event_date}",
                    }
                )

    candidates_by_id: dict[str, dict[str, Any]] = {}

    for hit in hits:
        if not isinstance(hit, dict):
            continue

        # 1. Extract genuine XXXX-YYY identifier (never fabricate hashes)
        fs_id = normalize_fs_identifier(hit)
        if not fs_id:
            continue

        # 2. Extract candidate names (primary + alternate name forms) and reject non-Latin scripts (e.g. Cyrillic)
        cand_name_forms: list[str] = []
        if hit.get("name") and str(hit.get("name")).strip():
            cand_name_forms.append(str(hit["name"]).strip())
        for nf in hit.get("names") or []:
            if isinstance(nf, dict) and nf.get("text"):
                cand_name_forms.append(str(nf["text"]).strip())
            elif isinstance(nf, str) and nf.strip():
                cand_name_forms.append(nf.strip())

        cand_name_forms = [nf for nf in cand_name_forms if not _has_non_latin_script(nf)]
        if not cand_name_forms and not (hit.get("given_name") or hit.get("given") or hit.get("surname")):
            continue

        cand_givens: list[str] = []
        cand_surnames: list[str] = []
        for nf_str in cand_name_forms:
            g_parts, s_parts = _extract_candidate_name_parts(nf_str)
            for gp in g_parts:
                if gp not in cand_givens:
                    cand_givens.append(gp)
            for sp in s_parts:
                if sp not in cand_surnames:
                    cand_surnames.append(sp)
        if hit.get("given_name") or hit.get("given"):
            gv = str(hit.get("given_name") or hit.get("given")).strip()
            if not _has_non_latin_script(gv) and gv not in cand_givens:
                cand_givens.append(gv)
        if hit.get("surname"):
            sv = str(hit["surname"]).strip()
            if not _has_non_latin_script(sv) and sv not in cand_surnames:
                cand_surnames.append(sv)

        # 3. Gate: Sex compatibility
        cand_sex = _normalize_sex_code(hit.get("sex") or hit.get("gender"))
        if fact_sex and cand_sex and fact_sex != cand_sex:
            continue

        # 4. Gate: Strict Surname & Given Name scoring (>= 0.85 each)
        sur_score = _score_surname_match(fact, cand_surnames)
        if sur_score < 0.85:
            continue
        giv_score = _score_given_name_match(fact, cand_givens)
        if giv_score < 0.85:
            continue

        # 5. Extract candidate facts (Birth, Christening, Marriage, Death, Residence, etc.)
        cand_facts = hit.get("facts") if isinstance(hit.get("facts"), list) else []
        cand_birth_date: str | None = str(hit.get("birth_date") or "").strip() or None
        cand_birth_year: int | None = _extract_year_from_text(hit.get("birth_year") or cand_birth_date)
        cand_birth_place: str | None = str(hit.get("birth_place") or hit.get("place") or "").strip() or None
        cand_death_year: int | None = _extract_year_from_text(hit.get("death_year") or hit.get("death_date"))
        cand_marriage_year: int | None = _extract_year_from_text(hit.get("marriage_year") or hit.get("marriage_date"))
        cand_residence_place: str | None = str(hit.get("residence_place") or "").strip() or None
        cand_lifespan: str | None = str(hit.get("lifespan") or "").strip() or None

        if cand_lifespan and ("-" in cand_lifespan or "–" in cand_lifespan):
            parts_ls = re.split(r"[-–]", cand_lifespan, maxsplit=1)
            if not cand_birth_year:
                cand_birth_year = _extract_year_from_text(parts_ls[0])
            if len(parts_ls) > 1 and not cand_death_year:
                cand_death_year = _extract_year_from_text(parts_ls[1])

        for cf in cand_facts:
            if not isinstance(cf, dict):
                continue
            cf_type = str(cf.get("type") or "").strip().lower()
            cf_date = str(cf.get("date") or "").strip() or None
            cf_place = str(cf.get("place") or "").strip() or None
            if cf_type in ("birth", "christening", "baptism"):
                if cf_date and (not cand_birth_date or cf_type == "birth"):
                    cand_birth_date = cf_date
                if not cand_birth_year:
                    cand_birth_year = _extract_year_from_text(cf_date)
                if cf_place and not cand_birth_place:
                    cand_birth_place = cf_place
            elif cf_type in ("death", "burial"):
                if not cand_death_year:
                    cand_death_year = _extract_year_from_text(cf_date)
            elif cf_type == "marriage":
                if not cand_marriage_year:
                    cand_marriage_year = _extract_year_from_text(cf_date)
            elif cf_type in ("residence", "census"):
                if cf_place and not cand_residence_place:
                    cand_residence_place = cf_place

        # 6. Gate: Strict Birth Year & Lifespan Consistency
        if is_census and cand_death_year and cand_death_year < (fact.event_year or 1930) - 1:
            continue
        if cand_birth_year and cand_death_year and cand_death_year < cand_birth_year:
            continue
        if target_birth_year and cand_marriage_year and cand_marriage_year < target_birth_year + 14:
            continue

        year_score = 0.85
        year_diff: int | None = None
        if target_birth_year:
            # Strict requirement: when target_birth_year is known, candidate MUST have a matching birth/christening year
            if cand_birth_year is None:
                continue
            year_diff = abs(target_birth_year - cand_birth_year)
            if year_diff > max_year_diff:
                continue
            if year_diff == 0:
                year_score = 1.0
            elif year_diff == 1:
                year_score = 0.94
            elif year_diff == 2:
                year_score = 0.86
            else:
                year_score = 0.78

        # 7. Gate: Relatives (Parents / Spouse) agreement & contradiction check
        rel_texts: list[str] = []
        for extra in hit.get("also_on_this_record") or []:
            if isinstance(extra, str) and extra.strip():
                rel_texts.append(extra.strip())
            elif isinstance(extra, dict):
                nm = extra.get("name") or extra.get("text")
                if nm:
                    rel_texts.append(str(nm).strip())
        for r_key in ("father_name", "mother_name", "spouse_name"):
            if hit.get(r_key):
                rel_texts.append(str(hit[r_key]).strip())

        rel_bonus = 0.0
        matched_relatives: list[str] = []
        if rel_texts:
            rel_joined_fold = fold_text_diacritics(" | ".join(rel_texts)).lower()
            rel_cids = {
                resolve_given_name_canonical_id(tok)
                for rt in rel_texts
                for tok in re.split(r"[\s,/()-]+", rt)
                if _is_valid_name_token(tok)
            } - {None}

            if father_giv:
                f_cid = resolve_given_name_canonical_id(father_giv)
                if (f_cid and f_cid in rel_cids) or fold_text_diacritics(father_giv).lower() in rel_joined_fold:
                    rel_bonus += 0.04
                    matched_relatives.append(f"Otec: {fact.father_name}")
            if mother_giv:
                m_cid = resolve_given_name_canonical_id(mother_giv)
                if (m_cid and m_cid in rel_cids) or fold_text_diacritics(mother_giv).lower() in rel_joined_fold:
                    rel_bonus += 0.04
                    matched_relatives.append(f"Matka: {fact.mother_name}")
            if spouse_giv:
                sp_cid = resolve_given_name_canonical_id(spouse_giv)
                if (sp_cid and sp_cid in rel_cids) or fold_text_diacritics(spouse_giv).lower() in rel_joined_fold:
                    rel_bonus += 0.04
                    matched_relatives.append(f"Manžel/ka: {fact.spouse_name}")

            # Explicit parent contradiction check on baptism records
            if not is_census and hit.get("father_name") and hit.get("mother_name") and father_giv and mother_giv:
                cand_f_giv, _ = _split_parent_name(str(hit.get("father_name")))
                cand_m_giv, _ = _split_parent_name(str(hit.get("mother_name")))
                f_cid_fact = resolve_given_name_canonical_id(father_giv)
                f_cid_cand = resolve_given_name_canonical_id(cand_f_giv)
                m_cid_fact = resolve_given_name_canonical_id(mother_giv)
                m_cid_cand = resolve_given_name_canonical_id(cand_m_giv)
                if (
                    f_cid_fact
                    and f_cid_cand
                    and f_cid_fact != f_cid_cand
                    and m_cid_fact
                    and m_cid_cand
                    and m_cid_fact != m_cid_cand
                ):
                    continue

        # 8. Gate: Strict Place & Foreign-country check
        cand_places_all = [p for p in (cand_birth_place, cand_residence_place) if p and not _has_non_latin_script(p)]
        cand_places_lower = " ".join(cand_places_all).lower()
        if cand_places_lower:
            if any(
                fc in cand_places_lower and fc not in fact_places_joined_lower
                for fc in _FOREIGN_COUNTRIES_REJECT
            ):
                continue

        place_score = 0.0
        matched_place_str: str | None = None
        if fact_village_tokens:
            if not cand_places_all:
                # Candidate has no place at all: only allow if at least 2 relatives match
                if len(matched_relatives) < 2:
                    continue
                place_score = 0.78
            else:
                cand_village_tokens, cand_county_tokens = _expand_place_tokens(cand_places_all)
                cand_folded = fold_text_diacritics(" ".join(cand_places_all)).lower()
                if (fact_village_tokens & cand_village_tokens) or any(
                    tok in cand_folded for tok in fact_village_tokens if len(tok) >= 4
                ):
                    place_score = 1.0
                    matched_place_str = cand_places_all[0]
                elif cand_village_tokens and any(
                    max(_paleographic_similarity(ft, ct) for ct in cand_village_tokens) >= 0.84
                    for ft in fact_village_tokens
                ):
                    place_score = 0.94
                    matched_place_str = cand_places_all[0]
                elif cand_county_tokens and (
                    (fact_county_tokens and (fact_county_tokens & cand_county_tokens))
                    or (not fact_county_tokens and cand_county_tokens)
                ):
                    # Same county/district, different village: require exact year + strong name or matching relative
                    if (year_diff == 0 and sur_score >= 0.95 and giv_score >= 0.95) or rel_bonus > 0:
                        place_score = 0.82
                    else:
                        continue
                else:
                    # Place is in a completely different county or only says "Hungary"/"Slovakia"
                    continue
        else:
            place_score = 0.85

        rel_bonus = min(0.08, rel_bonus)
        computed_score = 0.40 * sur_score + 0.30 * giv_score + 0.20 * year_score + 0.10 * place_score + rel_bonus
        raw_hit_score = hit.get("score") or hit.get("confidence")
        if isinstance(raw_hit_score, (int, float)) and 0.0 < float(raw_hit_score) <= 1.0:
            computed_score = 0.80 * computed_score + 0.20 * float(raw_hit_score)
        confidence = round(min(0.99, max(0.01, computed_score)), 2)
        if confidence < 0.86:
            continue

        cand_display_name = (
            cand_name_forms[0]
            if cand_name_forms
            else f"{fact.given_name_sk or fact.given_name_raw or '?'} {fact.surname_raw or '?'}"
        )
        raw_ark_val = hit.get("ark") or hit.get("fs_record_ark")
        is_record_hit = bool(
            raw_ark_val
            or hit.get("id_kind") == "record"
            or hit.get("collection")
            or hit.get("collection_id")
            or hit.get("also_on_this_record")
        )
        cand_ark = _normalize_record_ark(raw_ark_val, fs_id, is_record_hit=is_record_hit)
        cand_url = _build_candidate_fs_url(
            fs_id,
            cand_ark,
            str(hit.get("fs_url") or hit.get("url") or default_fs_url),
        )
        if not cand_lifespan and cand_birth_year:
            cand_lifespan = (
                f"{cand_birth_year}–{cand_death_year}"
                if cand_death_year
                else f"*{cand_birth_year}"
            )

        cand_agrees: list[dict[str, str]] = [
            {
                "field_sk": "Meno v strome / zázname FS",
                "value": f"{cand_display_name} ({fs_id})",
            }
        ]
        cand_differs: list[dict[str, str]] = []

        if cand_birth_date or cand_birth_year:
            fact_b_str = fact.birth_or_event_date or (str(target_birth_year) if target_birth_year else None)
            cand_b_str = cand_birth_date or str(cand_birth_year)
            if year_diff == 0 and (not fact.birth_or_event_date or not cand_birth_date or fact.birth_or_event_date == cand_birth_date):
                cand_agrees.append({"field_sk": "Rok / dátum narodenia", "value": cand_b_str})
            elif fact_b_str and cand_b_str and fact_b_str != cand_b_str:
                cand_differs.append(
                    {
                        "field_sk": "Dátum / rok narodenia",
                        "matrika_val": str(fact_b_str),
                        "fs_val": f"{cand_b_str} ({fs_id})",
                    }
                )
        if matched_place_str:
            cand_agrees.append({"field_sk": "Miesto vo FS", "value": matched_place_str})
        elif cand_places_all:
            cand_differs.append(
                {
                    "field_sk": "Miesto vo FS",
                    "matrika_val": str(fact.village_sk or fact.village_raw or place_std or "?"),
                    "fs_val": f"{cand_places_all[0]} ({fs_id})",
                }
            )
        for m_rel in matched_relatives:
            cand_agrees.append({"field_sk": "Príbuzní na zázname", "value": m_rel})

        cand_agrees.extend(local_agrees)
        cand_differs.extend(local_differs)

        candidate_entry: dict[str, Any] = {
            "fs_person_id": fs_id,
            "fs_id": fs_id,
            "fs_record_ark": cand_ark,
            "fs_url": cand_url,
            "name": cand_display_name,
            "sex": cand_sex,
            "lifespan": cand_lifespan,
            "birth_date": cand_birth_date or (str(cand_birth_year) if cand_birth_year else None),
            "birth_year": cand_birth_year,
            "birth_place": cand_birth_place or cand_residence_place or place_std,
            "collection": hit.get("collection"),
            "relatives_summary": ", ".join(rel_texts[:4]) if rel_texts else None,
            "confidence_score": confidence,
            "what_agrees": cand_agrees,
            "what_differs": cand_differs,
            "selected": False,
        }

        existing = candidates_by_id.get(fs_id)
        if existing is None or confidence > float(existing.get("confidence_score") or 0.0):
            candidates_by_id[fs_id] = candidate_entry

    ranked = sorted(
        candidates_by_id.values(),
        key=lambda c: float(c.get("confidence_score") or 0.0),
        reverse=True,
    )[:6]
    return ranked


def apply_candidate_selection_to_match(
    match: FamilySearchMatch,
    candidates: list[dict[str, Any]],
    selected_ids: list[str] | None = None,
    *,
    fact: PersonFact | None = None,
    sheet: CensusSheet | None = None,
    local_matches: list[LocalRecordMatch] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Applies multi-candidate selection (`selected_ids`, each in `XXXX-YYY` format) to `match`,
    updating `candidates_json`, `selected_fs_person_ids_json`, `fs_person_id`, `what_agrees_json`,
    `what_differs_json`, and `summary_sk`.
    """
    normalized_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx_c, c in enumerate(candidates or []):
        if not isinstance(c, dict):
            continue
        cid = normalize_fs_identifier(c, fallback_seed=f"{c.get('name') or idx_c}")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        c_copy = dict(c)
        c_copy["fs_person_id"] = cid
        c_copy["fs_id"] = cid
        normalized_candidates.append(c_copy)

    if not normalized_candidates and match.fs_person_id and str(match.fs_person_id).strip():
        primary_id = normalize_fs_identifier(match.fs_person_id)
        if primary_id:
            normalized_candidates.append(
                {
                    "fs_person_id": primary_id,
                    "fs_id": primary_id,
                    "fs_record_ark": match.fs_record_ark,
                    "fs_url": _build_candidate_fs_url(primary_id, match.fs_record_ark, match.fs_url or ""),
                    "name": (
                        f"{(fact.given_name_sk or fact.given_name_raw or '').strip()} {(fact.surname_raw or fact.surname_normalized or '').strip()}".strip()
                        if fact
                        else primary_id
                    ),
                    "confidence_score": round(float(match.confidence_score or 0.88), 2),
                    "what_agrees": json.loads(match.what_agrees_json or "[]"),
                    "what_differs": json.loads(match.what_differs_json or "[]"),
                    "selected": True,
                }
            )
            seen_ids.add(primary_id)

    clean_selected: list[str] = []
    for raw_sid in selected_ids or []:
        sid = normalize_fs_identifier(raw_sid)
        if sid and sid in seen_ids and sid not in clean_selected:
            clean_selected.append(sid)

    if not clean_selected and normalized_candidates:
        clean_selected = [normalized_candidates[0]["fs_person_id"]]

    for c in normalized_candidates:
        c["selected"] = c["fs_person_id"] in clean_selected

    selected_cands = [c for c in normalized_candidates if c.get("selected")] or normalized_candidates[:1]
    if selected_cands:
        primary_cand = selected_cands[0]
        match.fs_person_id = primary_cand["fs_person_id"]
        if primary_cand.get("fs_record_ark"):
            match.fs_record_ark = primary_cand.get("fs_record_ark")
        match.confidence_score = max(
            float(c.get("confidence_score") or 0.85) for c in selected_cands
        )
        if len(normalized_candidates) > 1 and len(clean_selected) > 1:
            match.match_status = "multiple_candidates"
        elif (
            len(normalized_candidates) > 1
            and float(normalized_candidates[0].get("confidence_score") or 0.0)
            - float(normalized_candidates[1].get("confidence_score") or 0.0)
            <= 0.05
            and not match.reviewed_by_user
        ):
            match.match_status = "multiple_candidates"
        elif match.confidence_score >= 0.88:
            match.match_status = "confirmed_match"
        else:
            match.match_status = "existing_needs_facts"

        merged_agrees: list[Any] = []
        merged_differs: list[Any] = []
        for sc in selected_cands:
            for item in sc.get("what_agrees") or []:
                if item not in merged_agrees:
                    merged_agrees.append(item)
            for item in sc.get("what_differs") or []:
                if item not in merged_differs:
                    merged_differs.append(item)
        if merged_agrees:
            match.what_agrees_json = json.dumps(merged_agrees, ensure_ascii=False)
        if merged_differs or match.what_differs_json in (None, "", "[]"):
            match.what_differs_json = json.dumps(merged_differs, ensure_ascii=False)

        selected_labels = ", ".join(
            f"{sc.get('name') or sc['fs_person_id']} ({sc['fs_person_id']})"
            for sc in selected_cands
        )
        if fact and fact.record_type == "census" and sheet:
            given_d = fact.given_name_sk or fact.given_name_raw or "?"
            surname_d = fact.surname_normalized or fact.surname_raw or "?"
            summary_sk = (
                f"Sčítací hárok {sheet.census_year} ({sheet.municipality}, dom č. {sheet.house_number or '—'}): "
                f"{given_d} {surname_d} ({fact.relation_to_head or 'člen domácnosti'}) – "
                f"zhoda vo FamilySearch: {selected_labels}."
            )
            if local_matches:
                summary_sk += f" Prepojené aj s lokálnou matrikou ({len(local_matches)} zhodných záznamov)."
            match.summary_sk = summary_sk
        elif fact:
            match.summary_sk = (
                f"Nájdená zhoda vo FamilySearch: {selected_labels} "
                f"(skóre {match.confidence_score:.2f}, kandidátov: {len(normalized_candidates)})."
            )

    match.candidates_json = json.dumps(normalized_candidates, ensure_ascii=False)
    match.selected_fs_person_ids_json = json.dumps(clean_selected, ensure_ascii=False)
    try:
        ap = json.loads(match.action_packet_json or "{}")
        if isinstance(ap, dict):
            ap["candidates"] = normalized_candidates
            ap["selected_fs_person_ids"] = clean_selected
            match.action_packet_json = json.dumps(ap, ensure_ascii=False)
    except Exception:
        pass
    return normalized_candidates, clean_selected


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
    Removes any FamilySearchMatch rows where:
    - no valid `XXXX-YYY` FamilySearch ID (`fs_person_id`) is present or `confidence_score <= 0`,
    - the matched name contains non-Latin script characters (e.g. Cyrillic `Стефан Лайкин`),
    - the stored MCP `record_search` results fail the tightened `evaluate_fs_match_candidates` rules, or
    - the row is a legacy unverified match with only `Meno v strome` and no verified birth year/place/candidates.
    Also normalizes `fs_record_ark` (`1:1:XXXX-YYY`) and `candidates_json` on surviving rows.
    """
    all_matches = list(session.exec(select(FamilySearchMatch)).all())
    if not all_matches:
        return 0

    fact_ids = list({m.person_fact_id for m in all_matches if m.person_fact_id is not None})
    facts_by_id = (
        {f.id: f for f in session.exec(select(PersonFact).where(PersonFact.id.in_(fact_ids))).all()}
        if fact_ids
        else {}
    )

    deleted_count = 0
    dirty_updated = False

    for m in all_matches:
        if m.fs_record_ark and _RECORD_ARK_RE.search(str(m.fs_record_ark)):
            session.delete(m)
            deleted_count += 1
            continue

        norm_pid = normalize_fs_identifier(m.fs_person_id)
        if not norm_pid or not is_valid_fs_identifier(norm_pid) or (m.confidence_score or 0.0) <= 0:
            session.delete(m)
            deleted_count += 1
            continue

        if _has_non_latin_script(m.what_agrees_json) or _has_non_latin_script(m.summary_sk):
            session.delete(m)
            deleted_count += 1
            continue

        fact = facts_by_id.get(m.person_fact_id)
        if fact is not None and not _has_sufficient_identity_for_fs_match(fact):
            session.delete(m)
            deleted_count += 1
            continue

        # Re-validate against stored MCP tree_search / record_search results when present
        if fact is not None and m.raw_mcp_results_json and m.raw_mcp_results_json.strip() not in ("", "{}"):
            try:
                raw_mcp = json.loads(m.raw_mcp_results_json)
            except Exception:
                raw_mcp = None
            if isinstance(raw_mcp, dict):
                # Any match created from legacy `record_search` (Historical Records `1:1:`/`2:2:`) without `tree_search` is a Record ID, not a Tree Person ID!
                if "record_search" in raw_mcp and "tree_search" not in raw_mcp:
                    session.delete(m)
                    deleted_count += 1
                    continue
                tree_search = raw_mcp.get("tree_search")
                if isinstance(tree_search, dict):
                    hits = (
                        tree_search.get("results")
                        or tree_search.get("matches")
                        or tree_search.get("persons")
                    )
                    valid_cands = evaluate_fs_match_candidates(
                        fact,
                        hits if isinstance(hits, list) else None,
                        place_std=fact.fs_place_standardized,
                        is_census=(fact.record_type == "census"),
                        session=session,
                        default_fs_url=m.fs_url or "",
                    )
                    if not valid_cands:
                        session.delete(m)
                        deleted_count += 1
                        continue
                    prior_sel: list[str] = []
                    try:
                        prior_sel = json.loads(m.selected_fs_person_ids_json or "[]")
                    except Exception:
                        prior_sel = []
                    apply_candidate_selection_to_match(
                        m,
                        valid_cands,
                        selected_ids=prior_sel,
                        fact=fact,
                    )
                    session.add(m)
                    dirty_updated = True
                    continue

        # Check legacy unverified rows that only had `[{"field_sk": "Meno v strome", ...}]`
        try:
            agrees_list = json.loads(m.what_agrees_json or "[]")
        except Exception:
            agrees_list = []
        if (
            isinstance(agrees_list, list)
            and len(agrees_list) == 1
            and isinstance(agrees_list[0], dict)
            and agrees_list[0].get("field_sk") == "Meno v strome"
            and (not m.candidates_json or m.candidates_json.strip() in ("", "[]"))
        ):
            session.delete(m)
            deleted_count += 1
            continue

        if m.fs_person_id != norm_pid:
            m.fs_person_id = norm_pid
            session.add(m)
            dirty_updated = True
        if m.fs_record_ark:
            norm_ark = _normalize_record_ark(m.fs_record_ark, norm_pid, is_record_hit=True)
            if norm_ark and norm_ark != m.fs_record_ark:
                m.fs_record_ark = norm_ark
                session.add(m)
                dirty_updated = True

    if deleted_count > 0 or dirty_updated:
        session.commit()
    return deleted_count


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
        try:
            from backend.app.pipeline.task_queue import task_queue

            q_fs = task_queue.get_familysearch_queue_info()
            state["queued"] = q_fs["queued"]
            state["queued_count"] = q_fs["queued_count"]
            state["queue_position"] = q_fs["queue_position"]
        except Exception:
            state["queued"] = False
            state["queued_count"] = 0
            state["queue_position"] = None
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
            from backend.app.db.session import engine, safe_session_commit

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
                safe_session_commit(db)
        except Exception:
            pass

        return response_data if isinstance(response_data, dict) else {"result": response_data}

    async def get_auth_status(self, *, force: bool = False) -> dict[str, Any]:
        fs_env = read_fs_env()
        raw_token = _sanitize_fs_token(fs_env.get("FS_ACCESS_TOKEN") or os.environ.get("FS_ACCESS_TOKEN"))
        has_token = bool(raw_token)
        is_mocked = self._is_call_tool_mocked()
        now = time.time()

        if not has_token and not is_mocked:
            connected = self.mcp_bin.exists()
            res = {
                "connected": connected,
                "authenticated": False,
                "token_valid": False,
                "token_accepted": False,
                "token_problem": None,
                "sync_enabled": False,
                "mode": "public_places_and_action_packets",
                "tools_count": 29,
                "has_token_configured": False,
                "env_file": str(settings.fs_env_file),
                "environment": fs_env.get("FS_ENVIRONMENT", "production"),
                "mcp_response": {"authenticated": False, "token_accepted": False},
                "status": "ok" if connected else "error",
            }
            self._cached_auth_status = dict(res)
            self._cached_auth_ts = now
            return res

        if not force and not is_mocked:
            if self._cached_auth_status is not None and (
                now - self._cached_auth_ts < 60.0
                or self._lock.locked()
                or bool(self.progress_state.get("running"))
            ):
                cached = dict(self._cached_auth_status)
                token_valid = bool(has_token and cached.get("token_valid"))
                cached["authenticated"] = token_valid
                cached["token_valid"] = token_valid
                cached["sync_enabled"] = token_valid
                cached["has_token_configured"] = has_token
                cached["mode"] = "live_tree" if token_valid else "public_places_and_action_packets"
                cached["environment"] = fs_env.get("FS_ENVIRONMENT", "production")
                return cached
            if self._lock.locked() or bool(self.progress_state.get("running")):
                connected = self.mcp_bin.exists()
                return {
                    "connected": connected,
                    "authenticated": False,
                    "token_valid": False,
                    "token_accepted": None,
                    "token_problem": None,
                    "sync_enabled": False,
                    "mode": "public_places_and_action_packets",
                    "tools_count": 29,
                    "has_token_configured": has_token,
                    "env_file": str(settings.fs_env_file),
                    "environment": fs_env.get("FS_ENVIRONMENT", "production"),
                    "mcp_response": {"status": "busy"},
                    "status": "ok" if connected else "error",
                }

        mcp_res = await self.call_tool("auth_status", {}, stage="mcp_auth")
        connected = isinstance(mcp_res, dict) and "error" not in mcp_res
        if isinstance(mcp_res, dict) and "token_accepted" in mcp_res:
            token_accepted = mcp_res.get("token_accepted")
            token_valid = bool(token_accepted is True and (has_token or is_mocked))
        elif is_mocked and isinstance(mcp_res, dict):
            token_valid = bool(mcp_res.get("authenticated"))
            token_accepted = True if token_valid else False
        else:
            token_accepted = False
            token_valid = False

        eff_has_token = bool(has_token or (is_mocked and isinstance(mcp_res, dict) and mcp_res.get("authenticated")))
        token_problem = (
            None
            if token_valid
            else (
                (mcp_res.get("token_problem") or mcp_res.get("message") or mcp_res.get("error"))
                if isinstance(mcp_res, dict)
                else None
            )
            or ("FamilySearch token could not be validated." if eff_has_token else None)
        )
        res = {
            "connected": connected,
            "authenticated": token_valid,
            "token_valid": token_valid,
            "token_accepted": token_accepted,
            "token_problem": token_problem,
            "sync_enabled": token_valid,
            "mode": "live_tree" if token_valid else "public_places_and_action_packets",
            "tools_count": 29,
            "has_token_configured": eff_has_token,
            "env_file": str(settings.fs_env_file),
            "environment": fs_env.get("FS_ENVIRONMENT", "production"),
            "mcp_response": mcp_res,
            "status": "ok" if connected else "error",
        }
        self._cached_auth_status = dict(res)
        self._cached_auth_ts = now
        return res

    async def ensure_valid_token(self, *, force: bool = False) -> tuple[bool, dict[str, Any]]:
        """Return `(token_valid, auth_status)` verifying that a valid FamilySearch token is configured."""
        if self._is_call_tool_mocked():
            if self._cached_auth_status is not None and "token_valid" in self._cached_auth_status:
                return bool(self._cached_auth_status["token_valid"]), dict(self._cached_auth_status)
            return True, {
                "connected": True,
                "authenticated": True,
                "token_valid": True,
                "token_accepted": True,
                "token_problem": None,
                "sync_enabled": True,
                "has_token_configured": True,
            }
        status = await self.get_auth_status(force=force)
        return bool(status.get("token_valid")), status

    def _mark_token_rejected(self, message: str | None = None) -> None:
        if self._cached_auth_status is not None:
            self._cached_auth_status["authenticated"] = False
            self._cached_auth_status["token_valid"] = False
            self._cached_auth_status["token_accepted"] = False
            self._cached_auth_status["sync_enabled"] = False
            self._cached_auth_status["mode"] = "public_places_and_action_packets"
            if message:
                self._cached_auth_status["token_problem"] = message

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
        has_token = bool(
            _sanitize_fs_token(env_map.get("FS_ACCESS_TOKEN") or os.environ.get("FS_ACCESS_TOKEN"))
        )
        is_mocked = self._is_call_tool_mocked()
        if is_mocked:
            if self._cached_auth_status is not None and not self._cached_auth_status.get("token_valid", True):
                return False
            return True
        if not has_token:
            return False
        if self._cached_auth_status is not None and "token_valid" in self._cached_auth_status:
            return bool(self._cached_auth_status.get("token_valid"))
        return has_token

    async def match_person_fact(
        self,
        session: Session,
        fact: PersonFact,
        row: RecordRow | None = None,
    ) -> FamilySearchMatch | None:
        """
        Runs FamilySearch MCP tools (place standardization + record/person search)
        and stores a FamilySearchMatch row ONLY when an actual tightened match (`fs_person_id` in XXXX-YYY format)
        is found in FamilySearch MCP.
        """
        if not _has_sufficient_identity_for_fs_match(fact):
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

        sheet: CensusSheet | None = None
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

        from backend.app.db.session import ensure_session_write_unlocked, safe_session_commit

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
                ensure_session_write_unlocked(session)
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
        father_given, father_surname = _split_parent_name(fact.father_name, surname)
        mother_given, mother_surname = _split_parent_name(fact.mother_name, fact.mother_maiden_surname)

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
                f"q.fatherGivenName={quote_plus(father_given)}" if father_given else "",
                f"q.fatherSurname={quote_plus(father_surname)}" if father_surname else "",
                f"q.motherGivenName={quote_plus(mother_given)}" if mother_given else "",
                f"q.motherSurname={quote_plus(mother_surname)}" if mother_surname else "",
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
                f"q.fatherGivenName={quote_plus(father_given)}" if father_given else "",
                f"q.fatherSurname={quote_plus(father_surname)}" if father_surname else "",
                f"q.motherGivenName={quote_plus(mother_given)}" if mother_given else "",
                f"q.motherSurname={quote_plus(mother_surname)}" if mother_surname else "",
                f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
            ]
        fs_url = "https://www.familysearch.org/search/record/results?" + "&".join(
            [p for p in fs_search_params if p]
        )

        local_matches: list[LocalRecordMatch] = []
        if is_census:
            local_matches = match_census_to_local_matrika(session, fact)

        search_res: dict[str, Any] | None = None
        harvest_res: dict[str, Any] | None = None
        candidates: list[dict[str, Any]] = []
        ensure_session_write_unlocked(session)
        token_valid, _ = await self.ensure_valid_token()
        should_query_records = token_valid and self._should_query_mcp_records()
        search_had_auth_error = False
        if should_query_records and (given and surname):
            core_birth_place = (
                (
                    getattr(fact, "birth_place_municipality", None)
                    or fact.birth_place_sk
                    or fact.birth_place_raw
                    or place_std
                )
                if is_census
                else place_std
            )
            core_birth_year = int(fact.birth_year_est or year_int) if is_census else year_int
            alt_surnames = _build_alternate_surnames_for_tree_search(fact, surname)
            fact_sex_code = _normalize_sex_code(fact.sex)
            search_args: dict[str, Any] = {
                "given": given,
                "surname": surname,
                "birth_place": core_birth_place,
                "birth_year": core_birth_year,
                "birth_year_span": 3 if is_census else 2,
                "count": 20,
            }
            if alt_surnames:
                search_args["alternate_surnames"] = alt_surnames
            if fact_sex_code == "M":
                search_args["sex"] = "Male"
            elif fact_sex_code == "F":
                search_args["sex"] = "Female"
            has_optional_relatives = False
            if is_census and place_std:
                search_args["residence_place"] = place_std
                search_args["residence_year"] = year_int
                has_optional_relatives = True
            if spouse_given:
                search_args["spouse_given"] = spouse_given
                has_optional_relatives = True
            if spouse_surname:
                search_args["spouse_surname"] = spouse_surname
                has_optional_relatives = True
            if father_given:
                search_args["father_given"] = father_given
                has_optional_relatives = True
            if father_surname:
                search_args["father_surname"] = father_surname
                has_optional_relatives = True
            if mother_given:
                search_args["mother_given"] = mother_given
                has_optional_relatives = True
            if mother_surname:
                search_args["mother_surname"] = mother_surname
                has_optional_relatives = True

            ensure_session_write_unlocked(session)
            search_res = await self.call_tool(
                "search_tree_persons",
                search_args,
                stage="mcp_record",
                scan_id=fact.scan_id,
                row_id=eff_row_id or None,
            )
            if isinstance(search_res, dict) and search_res.get("error") in (
                "token_rejected",
                "auth_required",
                "not_configured",
            ):
                search_had_auth_error = True
                self._mark_token_rejected(search_res.get("message") or str(search_res.get("error")))
            results = (
                (
                    search_res.get("results")
                    or search_res.get("matches")
                    or search_res.get("persons")
                )
                if isinstance(search_res, dict) and not search_had_auth_error
                else None
            )
            if (
                not results
                and has_optional_relatives
                and not search_had_auth_error
                and not self._is_call_tool_mocked()
            ):
                fallback_args: dict[str, Any] = {
                    "given": given,
                    "surname": surname,
                    "birth_place": core_birth_place,
                    "birth_year": core_birth_year,
                    "birth_year_span": 3,
                    "count": 20,
                }
                if alt_surnames:
                    fallback_args["alternate_surnames"] = alt_surnames
                if fact_sex_code == "M":
                    fallback_args["sex"] = "Male"
                elif fact_sex_code == "F":
                    fallback_args["sex"] = "Female"
                ensure_session_write_unlocked(session)
                fallback_res = await self.call_tool(
                    "search_tree_persons",
                    fallback_args,
                    stage="mcp_record",
                    scan_id=fact.scan_id,
                    row_id=eff_row_id or None,
                )
                if isinstance(fallback_res, dict) and not fallback_res.get("error"):
                    search_res = fallback_res
                    results = (
                        fallback_res.get("results")
                        or fallback_res.get("matches")
                        or fallback_res.get("persons")
                    )

            harvested_surnames, harvested_given = extract_names_from_fs_hits(
                results if isinstance(results, list) else None
            )
            if not is_census and row is not None and (harvested_surnames or harvested_given):
                record_observed_fs_names(area_key, year_int, harvested_surnames, harvested_given)
                _apply_fs_name_matches_to_fact(fact, row, area_key, year_int)
                session.add(fact)
                session.add(row)
                given = fact.given_name_raw or given
                surname = fact.surname_raw or surname

            candidates = evaluate_fs_match_candidates(
                fact,
                results if isinstance(results, list) else None,
                place_std=place_std,
                is_census=is_census,
                local_matches=local_matches,
                session=session,
                default_fs_url=fs_url,
            )

        if not candidates:
            if (not should_query_records or search_had_auth_error) and fact.id is not None:
                session.commit()
                existing_match = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).first()
                if existing_match and existing_match.fs_person_id and str(existing_match.fs_person_id).strip():
                    return existing_match
                return None
            if fact.id is not None:
                stale = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).all()
                for sm in stale:
                    session.delete(sm)
            session.commit()
            return None

        action_packet, what_is_new = build_action_packet_and_new_items(
            fact, place_std, fs_url, session=session
        )

        match = session.exec(
            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
        ).first()
        prior_selected: list[str] = []
        if not match:
            match = FamilySearchMatch(person_fact_id=fact.id or 0, row_id=eff_row_id)
        else:
            try:
                prior_selected = json.loads(match.selected_fs_person_ids_json or "[]")
            except Exception:
                prior_selected = []

        match.fs_url = fs_url
        match.what_is_new_json = json.dumps(what_is_new, ensure_ascii=False)
        match.action_packet_json = json.dumps(action_packet, ensure_ascii=False)
        match.raw_mcp_results_json = json.dumps(
            {"place_search": place_res, "tree_search": search_res, "area_harvest": harvest_res},
            ensure_ascii=False,
        )
        apply_candidate_selection_to_match(
            match,
            candidates,
            selected_ids=prior_selected,
            fact=fact,
            sheet=sheet,
            local_matches=local_matches,
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
        3. Queries FamilySearch MCP (`search_records`) with tightened criteria (`loose=False`)
           and ONLY stores `FamilySearchMatch` rows for people that pass strict candidate verification
           with a canonical `XXXX-YYY` FamilySearch identifier.
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

        valid_sheet_facts = [f for f in facts if _has_sufficient_identity_for_fs_match(f)]
        token_valid, _ = await self.ensure_valid_token()
        should_query_records = token_valid and self._should_query_mcp_records()

        started_progress_here = False
        if (
            should_query_records
            and manage_progress
            and not self.progress_state.get("running")
            and valid_sheet_facts
        ):
            started_progress_here = True
            await self.start_progress(
                mode="auto",
                total=len(valid_sheet_facts),
                current_person=_format_fact_person_label(valid_sheet_facts[0]),
                session=session,
            )

        from backend.app.db.session import ensure_session_write_unlocked, safe_session_commit

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
                    ensure_session_write_unlocked(session)
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

            matched_count = 0
            valid_idx = 0
            total_valid = max(1, len(valid_sheet_facts))
            from backend.app.api.ws import ws_manager

            for fact in facts:
                if not _has_sufficient_identity_for_fs_match(fact):
                    if fact.id is not None:
                        stale_matches = session.exec(
                            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                        ).all()
                        for sm in stale_matches:
                            session.delete(sm)
                    continue

                valid_idx += 1
                person_label = _format_fact_person_label(fact)
                if should_query_records:
                    await self.step_progress(current_person=person_label)
                await ws_manager.broadcast(
                    "state",
                    {
                        "active_stage": "mcp_match",
                        "active_model": "familysearch-mcp" if should_query_records else None,
                        "current_scan_id": fact.scan_id or sheet.front_scan_id,
                        "current_row_id": fact.row_id,
                        "progress_pct": 88 + int(11 * valid_idx / total_valid),
                        "message_sk": (
                            f"Párujem s matrikou a FamilySearch ({valid_idx}/{total_valid}): {person_label}..."
                            if should_query_records
                            else f"Párujem s lokálnou matrikou ({valid_idx}/{total_valid}): {person_label}..."
                        ),
                        "message_en": (
                            f"Matching with parish records & FamilySearch ({valid_idx}/{total_valid}): {person_label}..."
                            if should_query_records
                            else f"Matching with local parish records ({valid_idx}/{total_valid}): {person_label}..."
                        ),
                    },
                )

                fact.fs_place_standardized = place_std
                session.add(fact)

                local_matches = match_census_to_local_matrika(session, fact)
                if not should_query_records:
                    continue

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
                father_given, father_surname = _split_parent_name(fact.father_name, surname)
                mother_given, mother_surname = _split_parent_name(fact.mother_name, fact.mother_maiden_surname)

                fs_search_params = [
                    f"q.givenName={quote_plus(given)}" if given else "",
                    f"q.surname={quote_plus(surname)}" if surname else "",
                    f"q.birthLikePlace={quote_plus(birth_place)}" if birth_place else "",
                    f"q.birthLikeDate.from={birth_year - 2}" if birth_year else "",
                    f"q.birthLikeDate.to={birth_year + 2}" if birth_year else "",
                    f"q.residencePlace={quote_plus(place_std)}",
                    f"q.residenceDate.from={sheet.census_year}",
                    f"q.residenceDate.to={sheet.census_year}",
                    f"q.fatherGivenName={quote_plus(father_given)}" if father_given else "",
                    f"q.fatherSurname={quote_plus(father_surname)}" if father_surname else "",
                    f"q.motherGivenName={quote_plus(mother_given)}" if mother_given else "",
                    f"q.motherSurname={quote_plus(mother_surname)}" if mother_surname else "",
                    f"q.spouseGivenName={quote_plus(spouse_given)}" if spouse_given else "",
                    f"q.spouseSurname={quote_plus(spouse_surname)}" if spouse_surname else "",
                ]
                fs_url = "https://www.familysearch.org/search/record/results?" + "&".join(
                    [p for p in fs_search_params if p]
                )

                search_res: dict[str, Any] | None = None
                candidates: list[dict[str, Any]] = []
                search_had_auth_error = False
                if should_query_records and (given and surname):
                    alt_surnames = _build_alternate_surnames_for_tree_search(fact, surname)
                    fact_sex_code = _normalize_sex_code(fact.sex)
                    search_args: dict[str, Any] = {
                        "given": given,
                        "surname": surname,
                        "birth_place": birth_place or place_std,
                        "birth_year": int(birth_year or sheet.census_year),
                        "birth_year_span": 3,
                        "residence_place": place_std,
                        "residence_year": int(sheet.census_year),
                        "count": 20,
                    }
                    if alt_surnames:
                        search_args["alternate_surnames"] = alt_surnames
                    if fact_sex_code == "M":
                        search_args["sex"] = "Male"
                    elif fact_sex_code == "F":
                        search_args["sex"] = "Female"
                    if spouse_given:
                        search_args["spouse_given"] = spouse_given
                    if spouse_surname:
                        search_args["spouse_surname"] = spouse_surname
                    if father_given:
                        search_args["father_given"] = father_given
                    if father_surname:
                        search_args["father_surname"] = father_surname
                    if mother_given:
                        search_args["mother_given"] = mother_given
                    if mother_surname:
                        search_args["mother_surname"] = mother_surname
                    ensure_session_write_unlocked(session)
                    search_res = await self.call_tool(
                        "search_tree_persons",
                        search_args,
                        stage="mcp_record",
                        scan_id=fact.scan_id or sheet.front_scan_id,
                        row_id=fact.row_id,
                    )
                    if isinstance(search_res, dict) and search_res.get("error") in (
                        "token_rejected",
                        "auth_required",
                        "not_configured",
                    ):
                        search_had_auth_error = True
                        should_query_records = False
                        self._mark_token_rejected(search_res.get("message") or str(search_res.get("error")))
                    results = (
                        (
                            search_res.get("results")
                            or search_res.get("matches")
                            or search_res.get("persons")
                        )
                        if isinstance(search_res, dict) and not search_had_auth_error
                        else None
                    )
                    if (
                        not results
                        and not search_had_auth_error
                        and not self._is_call_tool_mocked()
                    ):
                        fallback_args: dict[str, Any] = {
                            "given": given,
                            "surname": surname,
                            "birth_place": birth_place or place_std,
                            "birth_year": int(birth_year or sheet.census_year),
                            "birth_year_span": 3,
                            "count": 20,
                        }
                        if alt_surnames:
                            fallback_args["alternate_surnames"] = alt_surnames
                        if fact_sex_code == "M":
                            fallback_args["sex"] = "Male"
                        elif fact_sex_code == "F":
                            fallback_args["sex"] = "Female"
                        ensure_session_write_unlocked(session)
                        fallback_res = await self.call_tool(
                            "search_tree_persons",
                            fallback_args,
                            stage="mcp_record",
                            scan_id=fact.scan_id or sheet.front_scan_id,
                            row_id=fact.row_id,
                        )
                        if isinstance(fallback_res, dict) and not fallback_res.get("error"):
                            search_res = fallback_res
                            results = (
                                fallback_res.get("results")
                                or fallback_res.get("matches")
                                or fallback_res.get("persons")
                            )

                    candidates = evaluate_fs_match_candidates(
                        fact,
                        results if isinstance(results, list) else None,
                        place_std=place_std,
                        is_census=True,
                        local_matches=local_matches,
                        session=session,
                        default_fs_url=fs_url,
                    )

                if not candidates:
                    if not search_had_auth_error and fact.id is not None:
                        stale_matches = session.exec(
                            select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                        ).all()
                        for sm in stale_matches:
                            session.delete(sm)
                    await self.step_progress(increment_processed=1, current_person=person_label)
                    continue

                action_packet, what_is_new = build_census_action_packet(
                    fact=fact,
                    sheet=sheet,
                    household=household,
                    place_std=place_std,
                    fs_url=fs_url,
                    local_matches=local_matches,
                )

                match = session.exec(
                    select(FamilySearchMatch).where(FamilySearchMatch.person_fact_id == fact.id)
                ).first()
                prior_selected: list[str] = []
                if not match:
                    match = FamilySearchMatch(person_fact_id=fact.id or 0, row_id=fact.row_id or 0)
                else:
                    try:
                        prior_selected = json.loads(match.selected_fs_person_ids_json or "[]")
                    except Exception:
                        prior_selected = []

                match.fs_url = fs_url
                match.what_is_new_json = json.dumps(what_is_new, ensure_ascii=False)
                match.action_packet_json = json.dumps(action_packet, ensure_ascii=False)
                match.raw_mcp_results_json = json.dumps(
                    {"place_search": place_res, "tree_search": search_res},
                    ensure_ascii=False,
                )
                apply_candidate_selection_to_match(
                    match,
                    candidates,
                    selected_ids=prior_selected,
                    fact=fact,
                    sheet=sheet,
                    local_matches=local_matches,
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
    Skipped when FamilySearch OAuth token is not configured or not valid.
    """
    token_valid, _ = await fs_mcp_manager.ensure_valid_token()
    if not token_valid:
        return None

    from backend.app.db.session import engine
    from backend.app.pipeline.task_queue import task_queue

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
    except Exception:
        pass

    return await task_queue.run_or_enqueue(
        lambda: _trigger_auto_fs_match_for_fact_inner(fact_id, person_label, scan_id, row_id),
        task_type="fs_auto_match",
        category="familysearch",
        title_sk=f"Automatické FamilySearch párovanie: {person_label}",
        title_en=f"Auto FamilySearch pairing: {person_label}",
        initial_stage="mcp_match",
        active_model="familysearch-mcp",
        target_url="/familysearch",
        metadata={"fact_id": fact_id, "scan_id": scan_id, "row_id": row_id},
    )


async def _trigger_auto_fs_match_for_fact_inner(
    fact_id: int,
    person_label: str,
    scan_id: int | None,
    row_id: int | None,
) -> FamilySearchMatch | None:
    from backend.app.api.ws import ws_manager
    from backend.app.db.session import engine

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

        await ws_manager.broadcast(
            "state",
            {
                "active_stage": "mcp_match",
                "active_model": "familysearch-mcp",
                "current_scan_id": scan_id,
                "current_row_id": row_id,
                "progress_pct": 50,
                "message_sk": f"Párujem osobu s FamilySearch ({person_label})…",
                "message_en": f"Matching person with FamilySearch ({person_label})…",
            },
        )

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

        done_sk = (
            f"Nájdená zhoda vo FamilySearch pre {person_label} ({match.fs_person_id})."
            if (is_matched and match is not None)
            else f"FamilySearch párovanie dokončené pre {person_label}."
        )
        done_en = (
            f"Found FamilySearch match for {person_label} ({match.fs_person_id})."
            if (is_matched and match is not None)
            else f"FamilySearch pairing completed for {person_label}."
        )
        await ws_manager.broadcast(
            "state",
            {
                "active_stage": "idle",
                "active_model": None,
                "current_scan_id": scan_id,
                "current_row_id": row_id,
                "progress_pct": 100,
                "message_sk": done_sk,
                "message_en": done_en,
            },
        )
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


