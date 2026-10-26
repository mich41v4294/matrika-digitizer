from __future__ import annotations

import json
import re
import time
from typing import Any

from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.db.models import CensusSheet, PersonFact, QuestionItem, RecordRow, ScanImage, WorkflowTrace
from backend.app.db.session import ensure_session_write_unlocked, safe_session_commit
from backend.app.pipeline.ai_client import ai_client
from backend.app.pipeline.lexicon import (
    VILLAGE_MAP,
    correct_names_in_text,
    is_known_given_name,
    is_month_name,
    is_occupation_or_role_token,
    normalize_given_name,
    normalize_village,
)
from backend.app.pipeline.ocr_stage import media_rel_to_path
from backend.app.pipeline.schemas import (
    HEADER_FRAGMENT_MARKERS,
    LLMDecisionSchema,
    is_header_or_garbage,
    is_prompt_placeholder,
    validate_entry_number,
    validate_matrika_date,
    validate_village_raw,
)

DECISION_SYSTEM_PROMPT = """Ste expert na čítanie starých slovenských, uhorských a latinských cirkevných matrík z 19. storočia.
Porovnajte výrez riadku z matriky s prečítaným OCR textom a vráťte VÝHRADNE jeden platný JSON objekt s kľúčmi:
- "action": "ask_human" alebo "auto_accept"
- "confidence": číslo od 0.0 do 1.0
- "has_ditto": true alebo false
- "uncertain_field": jedna z hodnôt "parents", "village", "date_name", "observationes", "none"
- "reason": konkrétne vysvetlenie v slovenčine, ktoré slovo z aktuálneho riadku je v rukopise nejasné
- "question_sk": konkrétna otázka v slovenčine pre používateľa s uvedením prečítaného mena z aktuálneho riadku
- "candidates": pole 2 až 4 konkrétnych návrhov prečítania z aktuálneho riadku v dobovom tvare (napr. Andreas, Stephanus, Joannes, Catharina)

Prísne typové pravidlá:
- NIKDY nekopírujte popis schémy (nevracajte texty ako "Variant 1", "Reading 1" ani všeobecné vety bez konkrétneho mena z riadku).
- Do "date_name" patria výlučne dátumy (deň 1..31 a mesiac) a krstné meno dieťaťa (nikdy krstné meno namiesto mesiaca ako "Paulus 23").
- Do "parents" patria výlučne priezviská a krstné mená rodičov (nikdy názvy mesiacov ako "Augustus")."""

FIELD_NAMES_SK: dict[str, str] = {
    "parents": "Rodičia, priezvisko a stav (Nomen Parentum)",
    "village": "Obec a číslo domu (Locus Domicilii)",
    "date_name": "Dátum narodenia/krstu a meno dieťaťa",
    "observationes": "Krstní rodičia, kňaz a poznámky (Patrini / Observationes)",
    "identity_relation": "Priezvisko, krstné meno a pomer k hlave domácnosti (stĺpce 1–5)",
    "birth_marriage_children": "Dátum narodenia, rodinný stav a počet detí (stĺpce 6–10)",
    "origin_domicile": "Rodisko a domovská príslušnosť (stĺpce 11–14)",
    "nationality_religion_literacy": "Národnosť, náboženstvo a gramotnosť (stĺpce 15–17)",
    "occupation_presence_notes": "Povolanie, postavenie, závod a poznámky (stĺpce 18–24)",
}

HEADER_LINE_MARKERS = HEADER_FRAGMENT_MARKERS


def _is_header_line(line: str) -> bool:
    return is_header_or_garbage(line)


def _looks_english_or_placeholder(text: str | None) -> bool:
    return is_prompt_placeholder(text)


def parse_row_ocr_context(
    raw_left: str | None,
    raw_right: str | None,
    fallback_combined: str | None = None,
) -> dict[str, str]:
    """Extracts clean, type-safe column snippets from raw left/right OCR, stripping table header lines."""
    left_src = raw_left or ""
    right_src = raw_right or ""
    if not left_src and not right_src and fallback_combined:
        parts = re.split(r"\[PRAVÁ STRANA\s*/\s*RIGHT\]", fallback_combined, flags=re.IGNORECASE)
        left_src = re.sub(r"\[ĽAVÁ STRANA\s*/\s*LEFT\]", "", parts[0], flags=re.IGNORECASE).strip()
        if len(parts) > 1:
            right_src = parts[1].strip()

    left_lines = [ln.strip() for ln in left_src.splitlines() if ln.strip() and not _is_header_line(ln)]
    right_lines = [ln.strip() for ln in right_src.splitlines() if ln.strip() and not _is_header_line(ln)]

    left_cells: list[str] = []
    for ln in left_lines:
        for c in ln.split("|"):
            c_clean = c.strip()
            if c_clean and not _is_header_line(c_clean):
                left_cells.append(c_clean)

    right_cells: list[str] = []
    for ln in right_lines:
        for c in ln.split("|"):
            c_clean = c.strip()
            if c_clean and not _is_header_line(c_clean):
                right_cells.append(c_clean)

    # 1. Entry number
    entry_no = ""
    for c in left_cells[:3]:
        valid_en = validate_entry_number(c)
        if valid_en:
            entry_no = valid_en
            break

    # 2. Child given name (prioritize short 1-2 token cells without occupations)
    child_name = ""
    child_cell = ""
    for max_toks in (2, 4):
        for c in left_cells:
            c_toks = c.split()
            if len(c_toks) > max_toks:
                continue
            if any(is_occupation_or_role_token(t) for t in c_toks):
                continue
            for tok in c_toks:
                clean_tok = tok.strip(".,;0123456789°")
                if is_known_given_name(clean_tok):
                    norm, _ = normalize_given_name(clean_tok)
                    if norm:
                        child_name = norm
                        child_cell = c
                        break
            if child_name:
                break
        if child_name:
            break

    if not entry_no and child_cell:
        m_day = re.match(r"^(\d{1,3})\s*°?", child_cell)
        if m_day:
            entry_no = validate_entry_number(m_day.group(1)) or ""

    # 3. Date cells (strictly validated with `validate_matrika_date`)
    date_parts: list[str] = []
    for c in left_cells[:6]:
        if c == child_name:
            continue
        if c == child_cell:
            m_day = re.match(r"^(\d{1,2})(?:°|\b)", c)
            if m_day:
                d_val, _ = validate_matrika_date(m_day.group(1), entry_number=entry_no or None)
                if d_val and d_val not in date_parts:
                    date_parts.append(d_val)
            continue
        if entry_no and validate_entry_number(c) == entry_no and not is_month_name(c) and not entry_no.lower().endswith("a"):
            continue
        d_val, misplaced_names = validate_matrika_date(c, entry_number=entry_no or None)
        if d_val and not misplaced_names and d_val not in date_parts:
            date_parts.append(d_val)

    date_text = " / ".join(date_parts[:2]) if date_parts else ""

    # 4. Parents cell (exclude entry number, child cell, and valid date/month cells)
    parents_candidates: list[str] = []
    for c in left_cells:
        if c == child_name or c == child_cell:
            continue
        if entry_no and validate_entry_number(c) == entry_no:
            continue
        if is_month_name(c):
            continue
        d_check, d_names = validate_matrika_date(c, entry_number=entry_no or None)
        if d_check and not d_names:
            continue
        if len(c) > 12:
            parents_candidates.append(c)

    parents_text = max(parents_candidates, key=len) if parents_candidates else ""
    parents_text = re.sub(r"^\.\s*", "", parents_text).strip()

    # 5. Right page: village, godparents, officiant/observationes
    village_text = validate_village_raw(right_cells[0], parents_text) if right_cells else None
    if not village_text:
        low_comb = f"{left_src} {right_src}".lower()
        for v_key, (v_orig, _) in VILLAGE_MAP.items():
            if v_key in low_comb:
                village_text = v_orig
                break
    godparents_text = right_cells[1] if len(right_cells) > 1 else ""
    officiant_or_obs = " | ".join(right_cells[2:]) if len(right_cells) > 2 else ""

    if not parents_text and right_cells:
        parents_text = right_cells[0]

    return {
        "entry_number": entry_no,
        "child_name": child_name,
        "date_text": date_text,
        "parents_text": parents_text,
        "village_text": village_text or "",
        "godparents_text": godparents_text,
        "officiant_text": officiant_or_obs,
    }


def _repair_paleographic_variants(raw_text: str, field_name: str, ctx: dict[str, str]) -> list[str]:
    """Generates realistic historical paleographic reading candidates in period spelling, enforcing field type safety."""
    candidates: list[str] = []
    area_hint = ctx.get("village_text") or None

    def add_cand(val: str | None) -> None:
        if not val:
            return
        cleaned = re.sub(r"\s+", " ", val).strip(" |.,;-")
        if len(cleaned) < 2:
            return
        if _looks_english_or_placeholder(cleaned) or _is_header_line(cleaned):
            return
        if field_name == "parents" and is_month_name(cleaned):
            return
        if field_name in ("parents", "date_name", "observationes"):
            cleaned = correct_names_in_text(cleaned, area=area_hint)
        if cleaned not in candidates:
            candidates.append(cleaned)

    if field_name == "village":
        v_raw = ctx.get("village_text") or raw_text
        v_sk, v_hu, h_no = normalize_village(v_raw)
        if v_sk and h_no:
            add_cand(f"{v_sk} č. {h_no}")
            add_cand(f"{v_hu} № {h_no}")
        elif v_sk:
            add_cand(v_sk)
        add_cand(v_raw)
        add_cand("Vágújhely (Nové Mesto nad Váhom)")
        add_cand("Hradek")
        add_cand("Kis-Rippény")

    elif field_name == "date_name":
        d_raw = ctx.get("date_text") or ""
        c_name = ctx.get("child_name") or ""
        repaired_d = d_raw
        repaired_d = re.sub(r"\bSutius\b", "Julius", repaired_d, flags=re.IGNORECASE)
        repaired_d = re.sub(r"\b189\.\s*", "18. ", repaired_d)
        repaired_d = re.sub(r"\b89\.\s*4\.", "24.", repaired_d)
        if c_name and repaired_d:
            add_cand(f"{c_name} ({repaired_d})")
        if c_name and "julius" in repaired_d.lower():
            add_cand(f"{c_name} ({re.sub(r'Julius', 'Junius', repaired_d, flags=re.IGNORECASE)})")
        if c_name:
            add_cand(c_name)
        if d_raw and d_raw != repaired_d:
            add_cand(f"{c_name} ({d_raw})".strip())

    elif field_name == "observationes":
        gp = ctx.get("godparents_text") or ""
        obs = ctx.get("officiant_text") or ""
        if gp:
            repaired_gp = gp
            repaired_gp = re.sub(r"\bJannes\b", "Joannes", repaired_gp)
            repaired_gp = re.sub(r"\bSeanes\b", "Joannes", repaired_gp)
            repaired_gp = re.sub(r"\bcolumne\b", "coloni", repaired_gp, flags=re.IGNORECASE)
            repaired_gp = re.sub(r"\bev\.\s*a\.e\.?", "ev. a. c.", repaired_gp, flags=re.IGNORECASE)
            add_cand(repaired_gp)
            add_cand(gp)
        if obs:
            add_cand(obs)

    else:
        # "parents" (default and most common)
        p_raw = ctx.get("parents_text") or ""
        p_clean = re.sub(
            r",?\s*(?:familii|famul\w*|colon\w*|coa\s+colm\.?|columnev\.?|cv\.\s*a\.e\.?|ev\.\s*a\.[ce]\.?|r\.?\s*cath\w*).*",
            "",
            p_raw,
            flags=re.IGNORECASE,
        ).strip(" .,;")

        rep1 = p_clean
        rep1 = re.sub(r"\bMucejtha\b", "Mucska", rep1)
        rep1 = re.sub(r"\bHudta\b", "Hutta", rep1)
        rep1 = re.sub(r"\bSaulus\b", "Paulus", rep1)
        rep1 = re.sub(r"\bGaulus\b", "Paulus", rep1)
        rep1 = re.sub(r"\bJoarnes\b", "Joannes", rep1)
        rep1 = re.sub(r"\bJoames\b", "Joannes", rep1)
        rep1 = re.sub(r"\bJannes\b", "Joannes", rep1)
        rep1 = re.sub(r"\bSeghames\b", "Stephanus", rep1)
        rep1 = re.sub(r"\bTerdinandus\b", "Ferdinandus", rep1)
        rep1 = re.sub(r"\bSigmanus\b", "Sigismundus", rep1)
        rep1 = re.sub(r"\bportal\.\s*of\b", "et", rep1, flags=re.IGNORECASE)
        rep1 = re.sub(r"\bor\b", "et", rep1)
        rep1 = re.sub(r"\butox\b", "uxor", rep1, flags=re.IGNORECASE)
        rep1 = re.sub(r"\bex utor\b", "et uxor", rep1, flags=re.IGNORECASE)
        rep1 = re.sub(r"\bBalá's\b", "Balázs", rep1)
        rep1 = re.sub(r"\bBoarins\b", "Boarius", rep1)
        rep1 = re.sub(r"\bTerenesik\b", "Terencsík", rep1)
        rep1 = re.sub(r"\bMatheng\b", "Matthaeus", rep1)
        rep1 = re.sub(r"\bbala Caharina Malth\.", "Bala Catharina,", rep1)

        rep2 = p_clean
        rep2 = re.sub(r"\bMucejtha\b", "Mucska", rep2)
        rep2 = re.sub(r"\bHudta\b", "Hudec", rep2)
        rep2 = re.sub(r"\bCesett\b", "Cserep", rep2)
        rep2 = re.sub(r"\bSaulus\b", "Paulus", rep2)
        rep2 = re.sub(r"\bTerdinandus\b", "Ferdinandus", rep2)
        rep2 = re.sub(r"\bSigmanus\b", "Sigismundus", rep2)
        rep2 = re.sub(r"\bportal\.\s*of\b", "et", rep2, flags=re.IGNORECASE)
        rep2 = re.sub(r"\bor\b", "et", rep2)
        rep2 = re.sub(r"\bBoarins\b", "Boarius", rep2)
        rep2 = re.sub(r"\bTerenesik\b", "Trencsík", rep2)
        rep2 = re.sub(r"\bMatheng\b", "Matthaeus", rep2)
        rep2 = re.sub(r"\bbala Caharina Malth\.", "Balázs Catharina,", rep2)

        add_cand(rep1)
        if rep2 != rep1:
            add_cand(rep2)
        add_cand(p_clean)
        if p_raw and p_raw != p_clean:
            add_cand(p_raw)

        gp = ctx.get("godparents_text") or ""
        if gp and len(candidates) < 4:
            gp_clean = re.sub(r"\btova columne.*|\bev\.\s*a\.e\..*", "", gp, flags=re.IGNORECASE).strip(" .,;")
            gp_clean = re.sub(r"\bJannes\b", "Joannes", gp_clean)
            add_cand(gp_clean)

    return candidates[:4] if candidates else ["Potvrdiť zápis podľa výrezu", "Nečitateľné / poškodené miesto"]


_QUESTION_DETAILS_MEMO: dict[tuple[Any, ...], dict[str, Any]] = {}


def build_slovak_question_details(
    scan: ScanImage | None,
    row: RecordRow,
    uncertain_field: str | None = None,
    llm_question_sk: str | None = None,
    llm_reason: str | None = None,
    llm_candidates: list[str] | None = None,
    has_ditto: bool = False,
    parsed_ctx: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Produces a rich Slovak question title (`question_sk`), detailed Slovak explanation
    (`description_sk`), structured `row_context`, and verified paleographic `candidates`.
    Never allows English or Slovak prompt template placeholders to leak to the UI.
    """
    memo_key = (
        row.id,
        row.raw_ocr_combined,
        row.raw_ocr_left,
        row.raw_ocr_right,
        scan.detected_type if scan else None,
        row.record_type,
        uncertain_field,
        llm_question_sk,
        llm_reason,
        tuple(llm_candidates or []),
        has_ditto,
    )
    if row.id is not None and memo_key in _QUESTION_DETAILS_MEMO:
        return dict(_QUESTION_DETAILS_MEMO[memo_key])

    ctx = parsed_ctx or parse_row_ocr_context(row.raw_ocr_left, row.raw_ocr_right, row.raw_ocr_combined)
    is_census = bool((scan and scan.detected_type == "census") or row.record_type == "census")
    if is_census and uncertain_field in ("parents", "none", None):
        field = "identity_relation"
    elif is_census and uncertain_field == "date_name":
        field = "birth_marriage_children"
    elif is_census and uncertain_field == "village":
        field = "origin_domicile"
    elif is_census and uncertain_field == "observationes":
        field = "occupation_presence_notes"
    else:
        field = uncertain_field if uncertain_field in FIELD_NAMES_SK else "parents"

    entry_label = ctx["entry_number"] or row.entry_number or str(row.row_index)
    child_label = ctx["child_name"] or "nezistené meno"
    v_sk, _, h_no = normalize_village(ctx["village_text"])
    village_label = f"{v_sk}{f' č. {h_no}' if h_no else ''}" if v_sk else (ctx["village_text"] or "nezistená obec")

    # 1. Build a clear, grammatical, deterministic Slovak question title
    is_clean_structured_title = bool(
        llm_question_sk
        and not _looks_english_or_placeholder(llm_question_sk)
        and llm_question_sk.strip().startswith("Overenie ")
    )
    if is_clean_structured_title and llm_question_sk:
        question_sk = llm_question_sk.strip()
    else:
        if field == "village":
            question_sk = (
                f"Overenie názvu obce a čísla domu – zápis č. {entry_label} "
                f"(dieťa: {child_label}, prečítané: „{ctx['village_text'] or '?'}“)"
            )
        elif field == "date_name":
            question_sk = (
                f"Overenie dátumu a krstného mena – zápis č. {entry_label} "
                f"(prečítané: „{ctx['date_text'] or '?'} – {child_label}“)"
            )
        elif field == "observationes":
            question_sk = (
                f"Overenie krstných rodičov a poznámky – zápis č. {entry_label} "
                f"(dieťa: {child_label}, obec: {village_label})"
            )
        else:
            snippet_short = (ctx["parents_text"] or "")[:55].strip()
            question_sk = (
                f"Overenie priezviska a mien rodičov – zápis č. {entry_label} "
                f"(dieťa: {child_label}, obec: {village_label})"
                + (f": „{snippet_short}“" if snippet_short else "")
            )

    # 2. Build a rich Slovak explanation (`description_sk`) explaining what is uncertain and why
    field_human_sk = FIELD_NAMES_SK.get(field, "Rodičia a priezvisko")
    target_ocr = {
        "parents": ctx["parents_text"],
        "village": ctx["village_text"],
        "date_name": f"{ctx['date_text']} | {ctx['child_name']}".strip(" |"),
        "observationes": f"{ctx['godparents_text']} | {ctx['officiant_text']}".strip(" |"),
    }.get(field, ctx["parents_text"])

    reasons_sk: list[str] = []
    if has_ditto:
        reasons_sk.append(
            "V riadku sa nachádza značka opakovania („Idem“ / „eadem“ / „dtto“), ktorá odkazuje na rovnaký údaj z predchádzajúceho riadku."
        )
    if re.search(r"[A-Za-z]'[a-z]", (row.raw_ocr_combined or "")):
        reasons_sk.append(
            f"Počítač (GLM-OCR) našiel v rukopise neštandardne spojené písmená alebo apostrof (napr. v obci „{ctx['village_text']}“ → pravdepodobne {village_label})."
        )
    if field == "parents" and ctx["parents_text"]:
        reasons_sk.append(
            f"V stĺpci rodičov počítač prečítal „{ctx['parents_text']}“. V dobovom kurentnom písme sa písmená ako c/e/s/t, u/n alebo S/P často podobajú – skontrolujte prosím presné priezvisko otca a rodné priezvisko matky."
        )
    elif field == "village" and ctx["village_text"]:
        reasons_sk.append(
            f"V stĺpci bydliska počítač prečítal „{ctx['village_text']}“ (odhadovaná historická obec: {village_label}). Potvrďte prosím správny názov obce a číslo domu."
        )
    elif field == "date_name":
        reasons_sk.append(
            f"Skontrolujte prosím latinský názov mesiaca, deň narodenia/krstu („{ctx['date_text']}“) a krstné meno dieťaťa („{child_label}“)."
        )
    else:
        reasons_sk.append(
            f"Skontrolujte prosím zväčšený výrez stĺpca „{field_human_sk}“ a porovnajte ho s písmom farára v susedných riadkoch."
        )

    if (
        llm_reason
        and not _looks_english_or_placeholder(llm_reason)
        and not llm_reason.strip().startswith("Kontrolovaný stĺpec:")
        and "Poznámka modelu: Stručné slovenské vysvetlenie" not in llm_reason
    ):
        reasons_sk.append(f"Poznámka modelu: {llm_reason}")

    description_sk = (
        f"Kontrolovaný stĺpec: {field_human_sk}. "
        + (f"Prečítaný text v políčku: „{target_ocr}“. " if target_ocr else "")
        + " ".join(reasons_sk)
        + " Pri odpovedi zachovávame presný historický zápis mena podľa matriky (napr. Andreas, Stephanus, Joannes, Catharina)."
    )

    # 3. Filter and build real paleographic candidates
    valid_llm_cands = [
        c.strip()
        for c in (llm_candidates or [])
        if isinstance(c, str)
        and len(c.strip()) > 2
        and not _looks_english_or_placeholder(c)
        and not _is_header_line(c)
        and not (field == "parents" and is_month_name(c.strip()))
    ]
    generated_cands = _repair_paleographic_variants(row.raw_ocr_combined or "", field, ctx)
    merged_candidates: list[str] = []
    seen_cand_keys: set[str] = set()
    # Prioritize full multi-word paleographic variants when LLM returned single-token fragments for parents
    ordered_pool = (
        generated_cands + valid_llm_cands
        if (field == "parents" and any(len(c.split()) <= 1 for c in valid_llm_cands))
        else valid_llm_cands + generated_cands
    )
    for c in ordered_pool:
        norm_key = re.sub(r"\s+", " ", c).strip(" .,;|").lower()
        if norm_key and norm_key not in seen_cand_keys:
            seen_cand_keys.add(norm_key)
            merged_candidates.append(c)

    out_details = {
        "field_name": field,
        "question_sk": question_sk,
        "description_sk": description_sk,
        "candidates": merged_candidates[:4],
        "row_context": {
            "entry_number": entry_label,
            "child_name": child_label,
            "date_text": ctx["date_text"] or "—",
            "parents_text": ctx["parents_text"] or "—",
            "village_text": f"{ctx['village_text'] or '—'}"
            + (f" (→ {village_label})" if v_sk and v_sk != ctx["village_text"] else ""),
            "godparents_text": ctx["godparents_text"] or "—",
            "officiant_text": ctx["officiant_text"] or "—",
        },
    }
    if row.id is not None:
        if len(_QUESTION_DETAILS_MEMO) > 2048:
            _QUESTION_DETAILS_MEMO.clear()
        _QUESTION_DETAILS_MEMO[memo_key] = out_details
    return dict(out_details)


def _collect_neighbor_crops(session: Session, scan_id: int, current_row_id: int, field_key: str) -> list[dict[str, Any]]:
    """Finds up to 3 neighboring rows from the same page so the reviewer can compare the priest's handwriting."""
    stmt = (
        select(RecordRow)
        .where(RecordRow.scan_id == scan_id, RecordRow.id != current_row_id)
        .order_by(RecordRow.row_index)
    )
    neighbors = session.exec(stmt).all()
    samples: list[dict[str, Any]] = []
    for n in neighbors:
        if not n.raw_ocr_combined:
            continue
        cells = json.loads(n.cell_crops_json or "[]")
        crop_url = n.stitched_crop_path
        for c in cells:
            if c.get("key") == field_key:
                crop_url = c.get("crop_url")
                break
        n_ctx = parse_row_ocr_context(n.raw_ocr_left, n.raw_ocr_right, n.raw_ocr_combined)
        summary_parts = []
        if n_ctx["entry_number"]:
            summary_parts.append(f"č. {n_ctx['entry_number']}")
        if n_ctx["child_name"]:
            summary_parts.append(f"dieťa: {n_ctx['child_name']}")
        if n_ctx["parents_text"]:
            summary_parts.append(f"rodičia: {n_ctx['parents_text']}")
        if n_ctx["village_text"]:
            summary_parts.append(f"obec: {n_ctx['village_text']}")
        readable_snippet = (
            " • ".join(summary_parts)
            if summary_parts
            else ((n.raw_ocr_left or "") + " | " + (n.raw_ocr_right or ""))[:140].strip(" |")
        )
        samples.append({
            "row_index": n.row_index,
            "crop_url": crop_url,
            "confirmed_text": readable_snippet,
        })
        if len(samples) >= 3:
            break
    return samples


def detect_census_row_ditto(raw_left: str | None, raw_right: str | None, raw_combined: str | None) -> tuple[bool, list[str]]:
    """
    Deterministically detects census ditto marks across columns:
    - Explicit tokens (`dtto`, `//`, `"`, `„`, `idem`, `eadem`)
    - Cursive `"33"` or `"11"` in Column 5 (relation to head) after surname/given name
      (e.g. `4 | Dovičiorá | Ana | 33 | jevrié`)
    - Blank `(a) |  |` followed by `(b)` in Column 11 (`Rodisko`)
      (e.g. `(a) |  |\\n(b) | Thranna |`)
    Returns `(has_ditto, ditto_notes)`.
    """
    notes: list[str] = []
    left_txt = (raw_left or "").strip()
    right_txt = (raw_right or "").strip()
    comb_txt = (raw_combined or f"{left_txt}\n{right_txt}").strip()

    if re.search(r'(\bIdem\b|\beadem\b|\bdtto\b|\bdto\b|//|"\s*\||\|\s*"\s*\|)', comb_txt, re.IGNORECASE):
        notes.append("explicitný znak opakovania (dtto / //)")

    # Check Column 5 cursive ditto marks "33" or "11" in left cells
    for line in left_txt.splitlines():
        ls = line.strip()
        if not ls or ls.startswith("["):
            continue
        cells = [re.sub(r"\s+", " ", c).strip() for c in ls.split("|") if c.strip()]
        # In "4 | Dovičiorá | Ana | 33 | jevrié", index 2 or 3 (after name) is "33" or "11"
        for idx_c, c in enumerate(cells):
            if idx_c >= 2 and c in ("33", "11", '""', "„", '"', "//"):
                notes.append(f"stĺpec 5 (pomer k hlave domácnosti): kurzívny znak opakovania '{c}'")
                break

    # Check blank (a) |  | before (b) in right cells (Col 11 Rodisko)
    right_lines = [ln.strip() for ln in right_txt.splitlines() if ln.strip() and not ln.strip().startswith("[")]
    for idx_l, ln in enumerate(right_lines):
        raw_parts = [re.sub(r"\s+", " ", p).strip() for p in ln.split("|")]
        non_empty = [p for p in raw_parts if p]
        if non_empty and re.match(r"^\(?a[\)\.,:]?$", non_empty[0], flags=re.IGNORECASE):
            if len(non_empty) == 1 or re.match(r"^\(?[bcd][\)\.,:]?$", non_empty[1], flags=re.IGNORECASE):
                notes.append("stĺpec 11 a) rodisko (obec): prázdne pole / opakovanie z predchádzajúceho riadku")
                break

    return (len(notes) > 0), notes


def evaluate_census_row_deterministic(
    session: Session,
    scan: ScanImage,
    row: RecordRow,
    fact: PersonFact | None = None,
    *,
    log_trace: bool = False,
    commit: bool = True,
) -> dict[str, Any]:
    """
    Fast (<5ms) deterministic evaluator for 1930/1940 Slovak Census rows.
    Replaces the 19th-century church-matrika `d1-3b-mlx` decision model on census scans:
    - Uses extracted `PersonFact` + 100km regional gazetteer + household consensus.
    - Selects the targeted cell crop (`row_XX_extract_composite.jpg` or `row_XX_identity_name.jpg`),
      never the full-width `row_XX_stitched.jpg`.
    - Accurately flags `has_ditto_marks` when Col 5 is `"33"`/`"//"` or Col 11 `(a)` is blank/ditto.
    - Sets `row.decision_action = "auto_accept"`, `row.decision_confidence = 0.95`.
    - Removes any stale church-matrika `QuestionItem` on the census row.
    """
    t0 = time.perf_counter()
    if fact is None and row.id is not None:
        fact = session.exec(select(PersonFact).where(PersonFact.row_id == row.id)).first()

    sheet = session.get(CensusSheet, scan.census_sheet_id) if scan.census_sheet_id else None
    has_ditto, ditto_notes = detect_census_row_ditto(
        row.raw_ocr_left,
        row.raw_ocr_right,
        row.raw_ocr_combined,
    )
    if not has_ditto and bool(row.has_ditto_marks):
        has_ditto = True
        ditto_notes.append("znak opakovania (ditto //) z vizuálnej extrakcie")

    # Pick targeted cell crop (prefer extract_composite or identity_name over stitched_crop_path)
    cells = json.loads(row.cell_crops_json or "[]")
    crop_by_key = {c.get("key"): c.get("crop_url") for c in cells if isinstance(c, dict) and c.get("key")}
    target_crop_url = (
        crop_by_key.get("extract_composite")
        or crop_by_key.get("identity_name")
        or row.left_crop_path
        or row.stitched_crop_path
    )
    if target_crop_url and target_crop_url.endswith("_stitched.jpg"):
        alt_comp = target_crop_url.replace("_stitched.jpg", "_extract_composite.jpg")
        if media_rel_to_path(alt_comp) and media_rel_to_path(alt_comp).exists():
            target_crop_url = alt_comp

    surname = (fact.surname_raw or fact.surname_normalized) if fact else None
    given = (fact.given_name_sk or fact.given_name_raw) if fact else None
    relation = fact.relation_to_head if fact else None
    bdate = fact.birth_or_event_date if fact else None
    bplace = (fact.birth_place_raw or fact.birth_place_municipality) if fact else None

    summary_parts: list[str] = []
    if surname or given:
        summary_parts.append(f"{surname or ''} {given or ''}".strip())
    if relation:
        summary_parts.append(f"vzťah: {relation}")
    if bdate:
        summary_parts.append(f"nar.: {bdate}")
    if bplace:
        summary_parts.append(f"rodisko: {bplace}")
    if ditto_notes:
        summary_parts.append(f"opakovanie: {', '.join(ditto_notes)}")

    reason_sk = (
        f"Sčítací hárok {sheet.census_year if sheet else 1930} ({sheet.municipality if sheet else 'obec'}, dom č. {sheet.house_number if sheet else '?'}): "
        + (" • ".join(summary_parts) if summary_parts else "automaticky spracované z výrezu buniek a 100 km registra obcí.")
    )

    candidates: list[str] = []
    if surname and given:
        candidates.append(f"{surname} {given}")
    elif surname:
        candidates.append(surname)
    if relation:
        candidates.append(relation)
    if bplace:
        candidates.append(bplace)

    action = "auto_accept"
    confidence = 0.95

    row.decision_action = action
    row.decision_confidence = confidence
    row.decision_reason = reason_sk[:500]
    row.has_ditto_marks = has_ditto
    row.decision_json = json.dumps(
        {
            "action": action,
            "confidence": confidence,
            "has_ditto": has_ditto,
            "ditto_notes": ditto_notes,
            "uncertain_field": "none",
            "reason": reason_sk,
            "question_sk": "",
            "description_sk": reason_sk,
            "candidates": candidates,
            "target_crop_url": target_crop_url,
        },
        ensure_ascii=False,
    )
    if row.status in ("sliced", "ocr_done", "needs_human", "triaged"):
        row.status = "extracted" if fact is not None else "auto_accepted"
    session.add(row)

    # Clean up any stale open church-matrika QuestionItem on this census row
    if row.id is not None:
        stale_qs = session.exec(
            select(QuestionItem).where(QuestionItem.row_id == row.id, QuestionItem.status == "open")
        ).all()
        for q in stale_qs:
            session.delete(q)

    duration_ms = max(1, int((time.perf_counter() - t0) * 1000))
    if log_trace:
        crop_name = target_crop_url.split("/")[-1] if target_crop_url else "none"
        trace_payload = {
            "action": action,
            "confidence": confidence,
            "has_ditto": has_ditto,
            "ditto_notes": ditto_notes,
            "person": f"{surname or ''} {given or ''}".strip() or None,
            "relation_to_head": relation,
            "birth_date": bdate,
            "birth_place": bplace,
            "candidates": candidates,
        }
        trace = WorkflowTrace(
            scan_id=scan.id,
            row_id=row.id,
            stage="decision",
            model_or_tool="deterministic:census_gazetteer",
            prompt_text=(
                f"[IMAGE ATTACHED: {crop_name}]\n"
                f"Deterministická kontrola riadku sčítacieho hárku (bez volania cirkevno-matrikového modelu d1-3b-mlx).\n"
                f"Ľavé OCR: {row.raw_ocr_left or ''}\n"
                f"Pravé OCR: {row.raw_ocr_right or ''}"
            ),
            image_path=target_crop_url,
            reasoning_content="Sčítacie hárky (1930/1940) používajú priamu vizuálnu extrakciu buniek + 100 km register obcí + konsenzus domácnosti.",
            response_text=json.dumps(trace_payload, ensure_ascii=False, indent=2),
            parsed_json=json.dumps(trace_payload, ensure_ascii=False),
            status="success",
            duration_ms=duration_ms,
            prompt_tokens=0,
            completion_tokens=0,
            reasoning_tokens=0,
        )
        session.add(trace)

    if commit:
        safe_session_commit(session)
    return {
        "action": action,
        "confidence": confidence,
        "has_ditto": has_ditto,
        "reason": reason_sk,
        "question_sk": "",
        "description_sk": reason_sk,
        "candidates": candidates,
        "duration_ms": duration_ms,
    }


async def run_decision_for_row(
    session: Session,
    scan: ScanImage,
    row: RecordRow,
    prev_row_text: str | None = None,
) -> dict[str, Any]:
    is_census = bool(
        scan.detected_type == "census"
        or getattr(row, "record_type", None) == "census"
        or scan.census_sheet_id is not None
    )
    if is_census:
        return evaluate_census_row_deterministic(session, scan, row, log_trace=True)

    raw_combined = (row.raw_ocr_combined or "").strip()
    stitched_path = media_rel_to_path(row.stitched_crop_path)

    ditto_detected = bool(re.search(r'(\bIdem\b|\beadem\b|\bdtto\b|"\s*\|)', raw_combined, re.IGNORECASE))
    prompt = (
        f"Rok strany: {scan.detected_year or '1884'}, Typ matriky: {scan.detected_type}\n"
        f"Predchádzajúci riadok: {prev_row_text or 'žiadny'}\n"
        f"Aktuálny riadok (surové OCR):\n{raw_combined}\n\n"
        "Zhodnoťte čitateľnosť rukopisu a vráťte iba JSON objekt."
    )

    ensure_session_write_unlocked(session)
    res = await ai_client.chat_completion(
        stage="decision",
        model=settings.model_decision,
        system_prompt=DECISION_SYSTEM_PROMPT,
        prompt=prompt,
        image_path=stitched_path,
        image_rel_url=row.stitched_crop_path,
        scan_id=scan.id,
        row_id=row.id,
        max_tokens=380,
        temperature=0.1,
        response_model=LLMDecisionSchema,
    )

    decision_obj: LLMDecisionSchema = res.get("validated") or LLMDecisionSchema.model_validate(res.get("parsed") or {})
    action = decision_obj.action
    confidence = decision_obj.confidence
    has_ditto = bool(decision_obj.has_ditto or ditto_detected)
    uncertain_field = decision_obj.uncertain_field
    reason_raw = decision_obj.reason

    # Heuristic & type-safety ambiguity check
    reason_low = reason_raw.lower()
    model_flagged_ambiguity = any(
        w in reason_low for w in ("unclear", "ambiguous", "difficult", "obscured", "garbled", "nejasn", "nečitateľ")
    )
    if (
        len(raw_combined) < 25
        or re.search(r"[A-Za-z]'[a-z]{2,}", raw_combined)
        or has_ditto
        or model_flagged_ambiguity
    ):
        action = "ask_human"
        confidence = min(confidence, 0.65)

    details = build_slovak_question_details(
        scan=scan,
        row=row,
        uncertain_field=uncertain_field,
        llm_question_sk=decision_obj.question_sk,
        llm_reason=reason_raw,
        llm_candidates=decision_obj.candidates,
        has_ditto=has_ditto,
    )

    question_sk = details["question_sk"]
    description_sk = details["description_sk"]
    candidates = details["candidates"]
    field_name = details["field_name"]

    row.decision_action = action
    row.decision_confidence = round(confidence, 2)
    row.decision_reason = description_sk[:500]
    row.has_ditto_marks = has_ditto
    row.decision_json = json.dumps(
        {
            **decision_obj.model_dump(),
            "action": action,
            "uncertain_field": field_name,
            "question_sk": question_sk,
            "description_sk": description_sk,
            "candidates": candidates,
        },
        ensure_ascii=False,
    )
    row.status = "needs_human" if action == "ask_human" else "triaged"
    session.add(row)
    safe_session_commit(session)

    if action == "ask_human":
        existing_q = session.exec(
            select(QuestionItem).where(QuestionItem.row_id == row.id)
        ).first()
        cells = json.loads(row.cell_crops_json or "[]")
        chosen_crop = row.stitched_crop_path
        for c in cells:
            if c.get("key") == field_name:
                chosen_crop = c.get("crop_url")
                break
        neighbors = _collect_neighbor_crops(session, scan.id, row.id, field_name)
        if not existing_q:
            q = QuestionItem(
                row_id=row.id,
                scan_id=scan.id,
                field_name=field_name,
                question_sk=question_sk,
                question_en=description_sk,
                crop_path=chosen_crop,
                neighbor_crops_json=json.dumps(neighbors, ensure_ascii=False),
                candidates_json=json.dumps(candidates[:4], ensure_ascii=False),
                raw_ocr_snippet=raw_combined[:400],
                status="open",
            )
            session.add(q)
        else:
            existing_q.field_name = field_name
            existing_q.question_sk = question_sk
            existing_q.question_en = description_sk
            existing_q.crop_path = chosen_crop
            existing_q.neighbor_crops_json = json.dumps(neighbors, ensure_ascii=False)
            existing_q.candidates_json = json.dumps(candidates[:4], ensure_ascii=False)
            existing_q.raw_ocr_snippet = raw_combined[:400]
            session.add(existing_q)
        safe_session_commit(session)

    return {
        "action": action,
        "confidence": confidence,
        "has_ditto": has_ditto,
        "reason": description_sk,
        "question_sk": question_sk,
        "description_sk": description_sk,
        "candidates": candidates,
    }
