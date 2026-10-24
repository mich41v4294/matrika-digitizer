from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable

from backend.app.db.models import CensusSheet, FamilySearchMatch, PersonFact

_GEDCOM_MONTHS: dict[str, str] = {
    "1": "JAN",
    "01": "JAN",
    "jan": "JAN",
    "januar": "JAN",
    "január": "JAN",
    "january": "JAN",
    "januarii": "JAN",
    "2": "FEB",
    "02": "FEB",
    "feb": "FEB",
    "februar": "FEB",
    "február": "FEB",
    "february": "FEB",
    "februarii": "FEB",
    "3": "MAR",
    "03": "MAR",
    "mar": "MAR",
    "marec": "MAR",
    "march": "MAR",
    "martii": "MAR",
    "4": "APR",
    "04": "APR",
    "apr": "APR",
    "april": "APR",
    "apríl": "APR",
    "aprilis": "APR",
    "5": "MAY",
    "05": "MAY",
    "maj": "MAY",
    "máj": "MAY",
    "may": "MAY",
    "maii": "MAY",
    "maji": "MAY",
    "6": "JUN",
    "06": "JUN",
    "jun": "JUN",
    "jún": "JUN",
    "june": "JUN",
    "junii": "JUN",
    "7": "JUL",
    "07": "JUL",
    "jul": "JUL",
    "júl": "JUL",
    "july": "JUL",
    "julii": "JUL",
    "julius": "JUL",
    "8": "AUG",
    "08": "AUG",
    "aug": "AUG",
    "august": "AUG",
    "augusti": "AUG",
    "augustus": "AUG",
    "9": "SEP",
    "09": "SEP",
    "sep": "SEP",
    "sept": "SEP",
    "september": "SEP",
    "septembris": "SEP",
    "10": "OCT",
    "oct": "OCT",
    "okt": "OCT",
    "oktober": "OCT",
    "október": "OCT",
    "october": "OCT",
    "octobris": "OCT",
    "11": "NOV",
    "nov": "NOV",
    "november": "NOV",
    "novembris": "NOV",
    "12": "DEC",
    "dec": "DEC",
    "december": "DEC",
    "decembris": "DEC",
}


_GEDCOM_CLEAN_RE = re.compile(r"[\r\n\x00]+")
_GEDCOM_YEAR4_RE = re.compile(r"\d{4}")
_GEDCOM_DATE_NUM_RE = re.compile(r"\b(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})\b")
_GEDCOM_DATE_TXT_RE = re.compile(r"\b(\d{1,2})\.?\s*([A-Za-záéíóúýčďľňšťžÁÉÍÓÚÝČĎĽŇŠŤŽ]+)(?:\s+(\d{4}))?")
_GEDCOM_SOUR_KEY_RE = re.compile(r"[^A-Za-z0-9_]")


def _clean_gedcom_val(val: Any) -> str:
    """Strips CR/LF/NUL characters to prevent GEDCOM line/tag injection."""
    if val is None:
        return ""
    return _GEDCOM_CLEAN_RE.sub(" ", str(val)).strip()


def _sanitize_csv_cell(val: Any) -> Any:
    """Prevents CSV formula injection when cells start with =, +, -, or @."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return val
    s = str(val)
    if s and s[0] in ("=", "+", "-", "@") and len(s) > 1:
        return f"'{s}"
    return s


def _format_gedcom_date(raw_date: str | int | None, fallback_year: int | None = None) -> str:
    """Normalizes Slovak/Latin/numeric dates into GEDCOM 5.5.1 `DD MMM YYYY` or `YYYY`."""
    if raw_date is None and fallback_year is None:
        return ""
    s = _clean_gedcom_val(raw_date)
    if not s and fallback_year:
        return str(fallback_year)
    if _GEDCOM_YEAR4_RE.fullmatch(s):
        return s

    # Match DD.MM.YYYY or DD. M. YYYY
    m_num = _GEDCOM_DATE_NUM_RE.search(s)
    if m_num:
        day = int(m_num.group(1))
        mon = _GEDCOM_MONTHS.get(str(int(m_num.group(2))))
        yr = m_num.group(3)
        if mon and 1 <= day <= 31:
            return f"{day} {mon} {yr}"

    # Match DD. MonthName [YYYY]
    m_txt = _GEDCOM_DATE_TXT_RE.search(s)
    if m_txt:
        day = int(m_txt.group(1))
        mon_key = m_txt.group(2).lower().rstrip(".")
        mon = _GEDCOM_MONTHS.get(mon_key)
        yr = m_txt.group(3) or (str(fallback_year) if fallback_year else "")
        if mon and 1 <= day <= 31:
            return f"{day} {mon} {yr}".strip()

    if fallback_year and str(fallback_year) not in s:
        return f"{s} {fallback_year}".strip()
    return s


def _format_gedcom_parent_name(full_name: str, surname: str) -> str:
    cleaned = _clean_gedcom_val(full_name)
    surn = _clean_gedcom_val(surname)
    if not surn:
        return cleaned
    if cleaned.lower().startswith(surn.lower() + " "):
        given_part = cleaned[len(surn) :].strip()
        return f"{given_part} /{surn}/".strip()
    if cleaned.lower().endswith(" " + surn.lower()):
        given_part = cleaned[: -len(surn)].strip()
        return f"{given_part} /{surn}/".strip()
    return f"{cleaned} /{surn}/"


def _resolve_sheets_map(
    facts_list: list[PersonFact],
    sheets_by_id: dict[int, CensusSheet] | None = None,
) -> dict[int, CensusSheet]:
    if sheets_by_id is not None:
        return sheets_by_id
    sheet_ids = list({f.census_sheet_id for f in facts_list if f.census_sheet_id is not None})
    if not sheet_ids:
        return {}
    try:
        from sqlmodel import Session, select
        from backend.app.db.session import engine

        with Session(engine) as session:
            return {
                cs.id: cs
                for cs in session.exec(select(CensusSheet).where(CensusSheet.id.in_(sheet_ids))).all()
                if cs.id is not None
            }
    except Exception:
        return {}


def export_facts_to_csv(
    facts: Iterable[PersonFact],
    *,
    sheets_by_id: dict[int, CensusSheet] | None = None,
) -> str:
    facts_list = list(facts)
    resolved_sheets = _resolve_sheets_map(facts_list, sheets_by_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "ID",
        "Typ matriky",
        "Číslo zápisu",
        "Rok",
        "Dátum narodenia/udalosti",
        "Dátum krstu/pohrebu",
        "Meno (pôvodné)",
        "Meno (slovenské)",
        "Meno (maďarské)",
        "Priezvisko",
        "Pohlavie",
        "Vzťah k hlave domácnosti",
        "Rodinný stav",
        "Otec",
        "Matka",
        "Rodné priezvisko matky",
        "Manžel / Manželka",
        "Povolanie / Stav",
        "Postavenie v povolaní",
        "Zamestnávateľ / Pracovisko",
        "Náboženstvo",
        "Národnosť",
        "Štátna príslušnosť",
        "Gramotnosť",
        "Rodisko",
        "Domovská príslušnosť",
        "Obec (pôvodná)",
        "Obec (slovenská)",
        "Číslo domu",
        "FamilySearch štandardizované miesto",
        "Krstní rodičia / Svedkovia",
        "Kňaz",
        "Poznámky (Observationes)",
        "Zdroj (URL)",
        "Licencia zdroja",
    ])
    for f in facts_list:
        sheet = resolved_sheets.get(f.census_sheet_id) if f.census_sheet_id else None
        if f.record_type == "census":
            s_key = (sheet.slovakiana_key if sheet and sheet.slovakiana_key else None)
            source_url = (
                f"https://slovakiana.sk/kulturne-objekty/{s_key}"
                if s_key
                else "https://slovakiana.sk/scitacie-harky"
            )
            source_license = (
                "CC BY-NC-ND 4.0 — Slovenský národný archív (SNA) / Slovakiana "
                "(https://creativecommons.org/licenses/by-nc-nd/4.0/)"
            )
        else:
            source_url = ""
            source_license = ""

        row_cells = [
            f.id,
            f.record_type,
            f.entry_number or "",
            f.event_year or "",
            f.birth_or_event_date or "",
            f.baptism_or_burial_date or "",
            f.given_name_raw or "",
            f.given_name_sk or "",
            f.given_name_hu or "",
            f.surname_normalized or f.surname_raw or "",
            f.sex or "",
            f.relation_to_head or "",
            f.marital_status or "",
            f.father_name or "",
            f.mother_name or "",
            f.mother_maiden_surname or "",
            f.spouse_name or "",
            f.social_status_or_occupation or "",
            f.occupation_position or "",
            f.employer_workplace or "",
            f.religion or "",
            f.nationality or "",
            f.citizenship or "",
            f.literacy or "",
            f.birth_place_raw or f.birth_place_sk or "",
            f.home_jurisdiction or "",
            f.village_raw or "",
            f.village_sk or "",
            f.house_number or "",
            f.fs_place_standardized or "",
            f.godparents_or_witnesses or "",
            f.officiant or "",
            f.observationes_raw or "",
            source_url,
            source_license,
        ]
        writer.writerow([_sanitize_csv_cell(c) for c in row_cells])
    return buf.getvalue()


def export_facts_to_gedcom(
    facts: Iterable[PersonFact],
    *,
    sheets_by_id: dict[int, CensusSheet] | None = None,
) -> str:
    facts_list = list(facts)
    resolved_sheets = _resolve_sheets_map(facts_list, sheets_by_id)
    has_census = any(f.record_type == "census" for f in facts_list)

    lines: list[str] = [
        "0 HEAD",
        "1 SOUR MATRIKA_DIGITIZER",
        "2 NAME Matrika OCR & FamilySearch Digitizer",
        "2 VERS 1.0",
        f"1 DATE {datetime.now(timezone.utc).strftime('%d %b %Y').upper()}",
        "1 GEDC",
        "2 VERS 5.5.1",
        "2 FORM LINEAGE-LINKED",
        "1 CHAR UTF-8",
    ]
    if has_census:
        lines.append(
            "1 COPR Sčítacie hárky (1930/1940): Slovenský národný archív (SNA) / Slovakiana — "
            "Licencia Creative Commons CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)"
        )

    emitted_families: dict[tuple[Any, ...], tuple[str, str | None, str | None, list[str]]] = {}
    referenced_sources: dict[str, CensusSheet | None] = {}

    for idx, f in enumerate(facts_list, start=1):
        indi_id = f"@I{f.id or idx}@"
        given = _clean_gedcom_val(f.given_name_raw or f.given_name_sk or "Neznáme")
        surname = _clean_gedcom_val(f.surname_raw or f.surname_normalized or "Neznáme")
        place = _clean_gedcom_val(f.fs_place_standardized or f.village_sk or f.village_raw or "Slovakia")
        if f.house_number:
            place = f"{place} (dom č. {_clean_gedcom_val(f.house_number)})"

        sheet = resolved_sheets.get(f.census_sheet_id) if f.census_sheet_id else None
        sour_id: str | None = None
        if f.record_type == "census":
            if sheet and sheet.slovakiana_key:
                clean_key = _GEDCOM_SOUR_KEY_RE.sub("_", sheet.slovakiana_key).upper()
                sour_id = f"@S_{clean_key}@"
                referenced_sources[sour_id] = sheet
            else:
                sour_id = "@S_SLOVAKIANA_SNA@"
                referenced_sources.setdefault(sour_id, None)

        lines.append(f"0 {indi_id} INDI")
        lines.append(f"1 NAME {given} /{surname}/")
        lines.append(f"2 GIVN {given}")
        lines.append(f"2 SURN {surname}")
        if f.sex in ("M", "F"):
            lines.append(f"1 SEX {f.sex}")

        if f.record_type == "baptism":
            lines.append("1 BIRT")
            b_date = _format_gedcom_date(f.birth_or_event_date, f.event_year)
            if b_date:
                lines.append(f"2 DATE {b_date}")
            lines.append(f"2 PLAC {place}")
            lines.append("1 BAPM")
            bp_date = _format_gedcom_date(f.baptism_or_burial_date, f.event_year)
            if bp_date:
                lines.append(f"2 DATE {bp_date}")
            lines.append(f"2 PLAC {place}")
        elif f.record_type == "death":
            lines.append("1 DEAT")
            d_date = _format_gedcom_date(f.birth_or_event_date, f.event_year)
            if d_date:
                lines.append(f"2 DATE {d_date}")
            lines.append(f"2 PLAC {place}")
            if f.cause_of_death:
                lines.append(f"2 CAUS {_clean_gedcom_val(f.cause_of_death)}")
        elif f.record_type == "marriage":
            lines.append("1 EVEN")
            lines.append("2 TYPE Marriage")
            m_date = _format_gedcom_date(f.birth_or_event_date, f.event_year)
            if m_date:
                lines.append(f"2 DATE {m_date}")
            lines.append(f"2 PLAC {place}")
        elif f.record_type == "census":
            if f.birth_or_event_date or f.birth_year_est:
                lines.append("1 BIRT")
                cb_date = _format_gedcom_date(f.birth_or_event_date, f.birth_year_est)
                if cb_date:
                    lines.append(f"2 DATE {cb_date}")
                bp_plac = _clean_gedcom_val(f.birth_place_sk or f.birth_place_raw or place)
                lines.append(f"2 PLAC {bp_plac}")
            lines.append("1 CENS")
            if f.event_year:
                lines.append(f"2 DATE {f.event_year}")
            lines.append(f"2 PLAC {place}")
            if sour_id:
                lines.append(f"2 SOUR {sour_id}")
                page_parts = []
                if sheet and sheet.sheet_number:
                    page_parts.append(f"hárok č. {_clean_gedcom_val(sheet.sheet_number)}")
                if f.house_number:
                    page_parts.append(f"dom č. {_clean_gedcom_val(f.house_number)}")
                if f.person_row_number or f.entry_number:
                    page_parts.append(f"riadok č. {_clean_gedcom_val(f.person_row_number or f.entry_number)}")
                if page_parts:
                    lines.append(f"3 PAGE {', '.join(page_parts)}")
            lines.append("1 RESI")
            if f.event_year:
                lines.append(f"2 DATE {f.event_year}")
            lines.append(f"2 PLAC {place}")
            if f.social_status_or_occupation:
                lines.append(f"1 OCCU {_clean_gedcom_val(f.social_status_or_occupation)}")
            if sour_id:
                lines.append(f"1 SOUR {sour_id}")

        # Include linked Observationes events (e.g. death date recorded in baptism register)
        obs_events = json.loads(f.observationes_events_json or "[]")
        for ev in obs_events:
            if not isinstance(ev, dict):
                continue
            ev_type = ev.get("event_type") or ev.get("type")
            if ev_type == "death" and f.record_type != "death":
                lines.append("1 DEAT")
                if ev.get("year") or ev.get("date"):
                    lines.append(f"2 DATE {_clean_gedcom_val(ev.get('year') or ev.get('date'))}")
                if ev.get("raw_note"):
                    lines.append(f"2 NOTE {_clean_gedcom_val(ev.get('raw_note'))}")
            elif ev_type == "marriage":
                lines.append("1 EVEN")
                lines.append("2 TYPE Marriage Note")
                if ev.get("year") or ev.get("date"):
                    lines.append(f"2 DATE {_clean_gedcom_val(ev.get('year') or ev.get('date'))}")
                if ev.get("raw_note"):
                    lines.append(f"2 NOTE {_clean_gedcom_val(ev.get('raw_note'))}")

        if f.observationes_raw:
            lines.append(f"1 NOTE Poznámka z matriky: {_clean_gedcom_val(f.observationes_raw)}")
        if f.godparents_or_witnesses:
            lines.append(f"1 NOTE Krstní rodičia / svedkovia: {_clean_gedcom_val(f.godparents_or_witnesses)}")

        if f.father_name or f.mother_name:
            if f.record_type == "census" and f.census_sheet_id:
                fam_key: tuple[Any, ...] = (
                    "census",
                    f.census_sheet_id,
                    f.census_household_id or 0,
                    _clean_gedcom_val(f.father_name).lower(),
                    _clean_gedcom_val(f.mother_name).lower(),
                    _clean_gedcom_val(f.mother_maiden_surname).lower(),
                )
            else:
                fam_key = ("row", f.id or idx)

            if fam_key not in emitted_families:
                fam_id = f"@F{f.id or idx}@"
                fath_id = f"@IF{f.id or idx}@" if f.father_name else None
                moth_id = f"@IM{f.id or idx}@" if f.mother_name else None
                emitted_families[fam_key] = (fam_id, fath_id, moth_id, [indi_id])
                lines.append(f"1 FAMC {fam_id}")
                if fath_id and f.father_name:
                    lines.append(f"0 {fath_id} INDI")
                    lines.append(f"1 NAME {_format_gedcom_parent_name(f.father_name, surname)}")
                    lines.append("1 SEX M")
                    if f.social_status_or_occupation and f.record_type != "census":
                        lines.append(f"1 OCCU {_clean_gedcom_val(f.social_status_or_occupation)}")
                if moth_id and f.mother_name:
                    m_surn = f.mother_maiden_surname or ""
                    lines.append(f"0 {moth_id} INDI")
                    lines.append(f"1 NAME {_format_gedcom_parent_name(f.mother_name, m_surn)}")
                    lines.append("1 SEX F")
            else:
                fam_id, fath_id, moth_id, child_ids = emitted_families[fam_key]
                child_ids.append(indi_id)
                lines.append(f"1 FAMC {fam_id}")

    for fam_id, fath_id, moth_id, child_ids in emitted_families.values():
        lines.append(f"0 {fam_id} FAM")
        if fath_id:
            lines.append(f"1 HUSB {fath_id}")
        if moth_id:
            lines.append(f"1 WIFE {moth_id}")
        for cid in child_ids:
            lines.append(f"1 CHIL {cid}")

    for sour_id, sheet in referenced_sources.items():
        lines.append(f"0 {sour_id} SOUR")
        if sheet and sheet.slovakiana_key:
            s_key = _clean_gedcom_val(sheet.slovakiana_key)
            c_year = sheet.census_year or 1930
            muni = _clean_gedcom_val(sheet.municipality or "")
            dist = _clean_gedcom_val(sheet.district or "")
            urn = _clean_gedcom_val(sheet.urn_nbn or "")
            lines.append(
                f"1 TITL Sčítanie ľudu {c_year} – {muni} (okres {dist}), Slovakiana ({s_key})"
            )
            lines.append("1 AUTH Slovenský národný archív (SNA) / Slovakiana")
            pub_parts = [f"https://slovakiana.sk/kulturne-objekty/{s_key}"]
            if urn:
                pub_parts.append(f"URN:NBN {urn}")
            pub_parts.append("Licencia: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)")
            lines.append(f"1 PUBL {' | '.join(pub_parts)}")
            lines.append(
                f"1 NOTE Zdroj: Slovenský národný archív (SNA) / Slovakiana ({s_key}). "
                "Použité pod licenciou Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)."
            )
        else:
            lines.append("1 TITL Sčítanie ľudu v Krajine Slovenskej (1930/1940) – Slovakiana")
            lines.append("1 AUTH Slovenský národný archív (SNA) / Slovakiana")
            lines.append(
                "1 PUBL https://slovakiana.sk/scitacie-harky | "
                "Licencia: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)"
            )

    lines.append("0 TRLR")
    return "\n".join(lines) + "\n"
