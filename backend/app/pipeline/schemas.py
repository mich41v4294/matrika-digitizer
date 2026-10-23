"""
Type-Safe Pydantic Schemas, Semantic Field Validators, and Cross-Field Reconciliation
for LLM Tasks (OCR Header/Row Parsing, Decision Triage, and Genealogical Extraction).

Enforces strict domain boundaries so that:
- Personal given names (e.g. 'Paulus', 'Andreas', 'Joannes') or occupations ('sutor', 'boarius')
  can NEVER be accepted as dates ('Paulus 23', 'Kidveny Gaulus sutor').
- Latin/Hungarian month names (e.g. 'Augustus', 'Julius', 'Majus') or punctuation ('.', '/', '1')
  can NEVER be accepted as person/parent names or surnames ('father_name': 'Augustus').
- Cross-field swaps by the LLM (e.g. date='Paulus 23' & father_name='Augustus', or parents cell
  shifted into village_raw / godparents / observationes) are automatically detected and repaired.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.pipeline.lexicon import (
    KNOWN_FEMALE_GIVEN_NAMES,
    KNOWN_HISTORICAL_GIVEN_NAMES,
    KNOWN_MALE_GIVEN_NAMES,
    VILLAGE_MAP,
    correct_names_in_text,
    infer_sex_from_given_name,
    is_known_given_name,
    is_month_name,
    is_occupation_or_role_token,
    name_match_repair_notes,
    normalize_given_name,
    normalize_month_name,
    normalize_occupation,
    normalize_religion,
    normalize_surname,
    normalize_village,
    parse_observationes_events,
    sanitize_corrupted_surname_suffix,
)

SexLiteral = Literal["M", "F"]
LegitimacyLiteral = Literal["legit", "illegit"]
DecisionActionLiteral = Literal["auto_accept", "ask_human"]
UncertainFieldLiteral = Literal[
    "parents",
    "village",
    "date_name",
    "observationes",
    "identity_relation",
    "identity_name",
    "birth_marriage_children",
    "birth_marital",
    "origin_domicile",
    "nationality_religion_literacy",
    "occupation_presence_notes",
    "none",
]

HEADER_FRAGMENT_MARKERS: tuple[str, ...] = (
    "nomen",
    "vitatis",
    "haptis",
    "baptisand",
    "parentum",
    "conditio",
    "theligio",
    "religio",
    "patrinorum",
    "domicilii",
    "from legis",
    "illeg.",
    "mase",
    "situs",
    "siraus",
    "currens",
    "citus",
    "concord",
    "somnatio",
    "poradové číslo",
    "príbuzenský alebo iný pomer",
    "prednostovi domácnosti",
    "domovská príslušnosť",
    "znalosť čítania",
    "postavenie v povolaní",
)

PROMPT_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    # English placeholders
    "clear, polite",
    "question in slovak",
    "for reviewer",
    "hard to read",
    "short explanation",
    "names are ",
    "handwriting is",
    "cursive",
    "ambiguous",
    "difficult to read",
    "partially obscured",
    "reading 1",
    "reading 2",
    "reading 3",
    "evaluate ocr",
    # Slovak prompt template echoes (from d1-3b-mlx copying system prompt examples)
    "stručné slovenské vysvetlenie",
    "krátke vysvetlenie v slovenčine",
    "konkrétna otázka v slovenčine",
    "čo presne treba v tomto riadku skontrolovať",
    " ktoré priezvisko, meno, obec alebo dátum je v rukopise",
    "presný variant 1",
    "variant 1 z riadku",
    "variant 2",
    "variant 3",
)

NOISE_TOKENS: set[str] = {
    "",
    ".",
    "..",
    "...",
    "/",
    "//",
    "-",
    "—",
    "?",
    "1",
    "1.",
    "1 .",
    ". 1",
    "null",
    "none",
    "n/a",
    "undefined",
}

DATE_ALLOWED_WORDS: set[str] = {
    "die",
    "d",
    "eodem",
    "eadem",
    "idem",
    "dtto",
    "anno",
    "a",
    "a°",
    "hó",
    "nap",
}


def is_header_or_garbage(text: str | None) -> bool:
    if not text:
        return True
    low = text.strip().lower()
    if low in NOISE_TOKENS:
        return True
    # Pipe-separated tick marks like "1 | . | 1 | ..."
    if re.match(r"^[1./\s|]+$", low):
        return True
    hits = sum(1 for m in HEADER_FRAGMENT_MARKERS if m in low)
    if hits >= 1 and len(low) < 35:
        return True
    if hits >= 2:
        return True
    if re.match(r"^\s*(?:mase|rens\.|from legis)", low):
        return True
    return False


def is_prompt_placeholder(text: str | None) -> bool:
    if not text or not text.strip():
        return True
    low = text.strip().lower()
    if low in NOISE_TOKENS:
        return True
    return any(marker in low for marker in PROMPT_PLACEHOLDER_MARKERS)


def validate_entry_number(val: Any) -> str | None:
    """Validates sequential entry number (e.g. '51', '52', '10a', '6a'). Rejects years, house numbers, or words."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in NOISE_TOKENS:
        return None
    s_clean = re.sub(r"[.°\s]+", "", s)
    m = re.match(r"^(\d{1,3}[a-zA-Z]?)$", s_clean)
    if not m:
        return None
    digits = int(re.sub(r"\D", "", m.group(1)))
    if digits == 0 or digits > 500:
        return None
    return m.group(1)


def validate_event_year(val: Any, page_year: str | int | None = None) -> int:
    """Validates event year in range 1650..1950, preferring page_year when LLM misreads a day cell (like '1893' for '18.')."""
    page_y: int | None = None
    if page_year is not None:
        try:
            py = int(str(page_year).strip())
            if 1650 <= py <= 1950:
                page_y = py
        except Exception:
            pass

    if val is not None:
        try:
            vy = int(str(val).strip())
            if 1650 <= vy <= 1950:
                # If page_year is known (e.g. 1884) and LLM year differs by > 2 years (e.g. 1893 from OCR '189. Augustus'),
                # trust the scan header's detected page year.
                if page_y is not None and abs(vy - page_y) > 2:
                    return page_y
                return vy
        except Exception:
            pass

    return page_y if page_y is not None else 1884


def validate_matrika_date(
    val: Any,
    *,
    entry_number: str | None = None,
) -> tuple[str | None, list[str]]:
    """
    Strictly validates a historical matrika date (birth/baptism/marriage/burial).
    Returns (valid_date_or_None, misplaced_person_names).

    Type Safety Rules:
    1. Repairs known paleographic date OCR misreads ('Sutius' -> 'Julius', '189. Augustus' -> '18. Augustus', '89. 4.' / '894.' -> '24.').
    2. NEVER allows personal given names (e.g. 'Paulus', 'Saulus', 'Gaulus', 'Andreas', 'Joannes') inside a date!
       If input is 'Paulus 23', 'Paulus' is extracted into `misplaced_person_names` and stripped from the date.
    3. NEVER allows occupations ('sutor', 'boarius', 'opilio'), surnames ('Kidveny'), header fragments ('citus, St Concord S.'),
       or noise ('Iclein.', '.', '/', '1893') as a date.
    4. Requires at least one valid month name OR a valid day of month (1..31) or ditto marker ('eodem', 'idem').
    """
    if val is None:
        return None, []
    s = str(val).strip()
    if not s or s.lower() in NOISE_TOKENS:
        return None, []

    # Pre-repair known OCR date glitches (e.g. '189. 3.' -> '18.', '189. Augustus' -> '18. Augustus', '89. 4.' -> '24.')
    s = re.sub(r"\b189\.\s*3\.?", "18.", s)
    s = re.sub(r"\b189\.\s*", "18. ", s)
    s = re.sub(r"\b89\.\s*4\.?", "24.", s)
    s = re.sub(r"\bSutius\b", "Julius", s, flags=re.IGNORECASE)
    s = re.sub(r"\bSurius\b", "Julius", s, flags=re.IGNORECASE)
    s = re.sub(r"\bJutius\b", "Julius", s, flags=re.IGNORECASE)

    # Reject if identical to entry_number ONLY when there is no degree/ordinal marker and value > 31
    s_compact = re.sub(r"[\s.°]+", "", s).lower()
    m_ord_day = re.match(r"^(\d{1,2})\s*(?:°|a|\.\s*a)$", s, re.IGNORECASE)
    if m_ord_day:
        d_ord = int(m_ord_day.group(1))
        if 1 <= d_ord <= 31:
            return f"{d_ord}.", []

    if entry_number:
        entry_compact = re.sub(r"[\s.°]+", "", str(entry_number)).lower()
        if s_compact == entry_compact and not is_month_name(s_compact):
            # Only reject if it's a plain number matching entry_number (e.g. '51.', '52.', '53.', '59.')
            try:
                if int(re.sub(r"\D", "", s_compact)) > 31 or "." in str(val):
                    return None, []
            except Exception:
                return None, []

    if is_header_or_garbage(s):
        return None, []

    misplaced_names: list[str] = []
    valid_month: str | None = None
    valid_days: list[int] = []
    has_ditto = False
    forbidden_word_found = False

    raw_tokens = re.split(r"[\s,/|]+", s)
    kept_tokens: list[str] = []

    for tok in raw_tokens:
        tok_clean = tok.strip(".,;°()[]{}")
        if not tok_clean:
            continue

        # Check pure number token
        if re.match(r"^\d+$", tok_clean):
            num = int(tok_clean)
            if 1 <= num <= 31:
                valid_days.append(num)
                kept_tokens.append(f"{num}.")
            else:
                # 3-4 digit number (like 1893, 1494, 894) is not a valid day of month
                forbidden_word_found = True
            continue

        # Check number + degree/letter token like '22°'
        m_deg = re.match(r"^(\d{1,2})°$", tok.strip(".,;"))
        if m_deg:
            num = int(m_deg.group(1))
            if 1 <= num <= 31:
                valid_days.append(num)
                kept_tokens.append(f"{num}.")
                continue

        # Alphabetic or alphanumeric token
        alpha_clean = re.sub(r"[^a-zA-Z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ°]", "", tok_clean)
        low = alpha_clean.lower()

        if not low:
            continue

        if is_month_name(low):
            canon_month = normalize_month_name(low) or tok_clean
            valid_month = canon_month
            kept_tokens.append(canon_month)
        elif low in DATE_ALLOWED_WORDS:
            if low in ("eodem", "eadem", "idem", "dtto"):
                has_ditto = True
                kept_tokens.append(tok_clean)
        elif is_known_given_name(low):
            # TYPE SAFETY VIOLATION: Personal given name (e.g. 'Paulus', 'Gaulus', 'Saulus', 'Andreas') inside date!
            norm_g, _ = normalize_given_name(tok_clean)
            misplaced_names.append(norm_g or tok_clean)
        elif is_occupation_or_role_token(low):
            # Occupation ('sutor', 'boarius', 'opilio') inside date -> reject date
            forbidden_word_found = True
        else:
            # Unknown non-month word (e.g. surname 'Kidveny', 'Iclein', 'Concord') -> reject
            forbidden_word_found = True

    if forbidden_word_found and not valid_month:
        return None, misplaced_names

    if not valid_month and not valid_days and not has_ditto:
        return None, misplaced_names

    if valid_month and valid_days:
        # Preserve original order (Month Day vs Day. Month)
        first_month_pos = s.lower().find(valid_month[:3].lower())
        first_day_pos = s.find(str(valid_days[0]))
        if 0 <= first_month_pos < first_day_pos:
            return f"{valid_month} {valid_days[0]}", misplaced_names
        return f"{valid_days[0]}. {valid_month}", misplaced_names

    if valid_month:
        return valid_month, misplaced_names

    if valid_days:
        return f"{valid_days[0]}.", misplaced_names

    if has_ditto:
        return " ".join(kept_tokens), misplaced_names

    return None, misplaced_names


def _strip_occupations_and_religion_from_name(text: str) -> str:
    """Removes Latin/Hungarian occupation words, religion markers, and village suffixes from a person name string."""
    cleaned = text.strip()
    # Remove leading punctuation / numbers / single-letter column artifacts
    cleaned = re.sub(r"^[\s.,;/|0-9°\-]+", "", cleaned)
    # Remove religion suffixes and everything after them
    cleaned = re.split(
        r",?\s*(?:\b(?:r\.?\s*cath|rom\.?\s*cath|k\.?\s*cath|cathol\w*|ev\.?\s*a\.?\s*[ce]\.?|cv\.?\s*a\.?\s*[ce]\.?|e\.?\s*a\.?\s*c\.?|evang\w*|dath\.)\b)",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    # Remove occupation suffixes and phrases
    cleaned = re.sub(
        r",?\s*\b(?:familii|famul\w*|colon\w*|coa\s+colm\.?|colm\w*|column\w*|tova\s+column\w*|inquil\w*|subinquil\w*|opilio|boari\w*|boarins|braius|molitor|faber|ferrarius|sutor|sartor|cives|civis|miles|mercenari\w*|munarius|nobilis|parochus|parchus|garochus|capellan\w*|capellaans|portal\.?|cotton)\b\.?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove known embedded village adjectives/names in parent cells
    cleaned = re.sub(
        r"\b(?:neressensis|neresse|nezette|kis\s+rippeny|kis\s+dippeng|rom\b|his\b)\b\.?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove trailing/isolated connectors
    cleaned = re.sub(r"\b(?:et\s+uxor|ex\s+utor|et\s+utox|uxor|utox|utor|mor)\b\.?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;/|-")
    return cleaned


def validate_person_name(
    val: Any,
    *,
    role: Literal["given", "surname", "father", "mother", "spouse", "godparents", "officiant"] = "father",
    area: str | None = None,
    year: int | str | None = None,
) -> tuple[str | None, list[str]]:
    """
    Validates a person name field according to its semantic role.
    Returns (cleaned_name_or_None, misplaced_months_or_dates).

    Type Safety Rules:
    1. Rejects punctuation/noise ('.', '/', '1', 'Idem.', 'Iclein.') and header fragments ('citus, St Concord S.').
    2. NEVER allows a bare month name (e.g. 'Augustus', 'Julius', 'Majus', 'April') or date ('August 7', 'Sutius 28')
       as a person name or surname! Extracts any misplaced month/date into `misplaced_months_or_dates`.
    3. Strips digits (e.g. 'Paulus 23' -> 'Paulus' + misplaced date '23.').
    4. Strips occupation tokens ('Boarius', 'Boarins', 'opilio', 'sutor', 'colonus') and religion markers.
    """
    if val is None:
        return None, []
    s = str(val).strip()
    if not s or s.lower() in NOISE_TOKENS or is_header_or_garbage(s):
        return None, []

    if re.match(r"^(?:idem|i\s+dem|eadem|dtto|iclein)\.?$", s.strip(), re.IGNORECASE):
        return None, []

    misplaced_dates: list[str] = []

    # Check if the entire string is a month name or a valid date!
    if is_month_name(s):
        canon_m = normalize_month_name(s) or s
        return None, [canon_m]

    as_date, names_in_date = validate_matrika_date(s)
    if as_date and not names_in_date:
        # The string is purely a date (e.g. 'August 7', '18. Augustus', 'Julius 28', '24.')!
        return None, [as_date]

    if role in ("godparents", "officiant"):
        # For godparents/officiant, reject pipe-separated left-page dumps like "1 | . | 1 | ..."
        if "|" in s and re.search(r"\b1\s*\|\s*\.", s):
            return None, []
        cleaned_gp = re.sub(r"^[\s.,;/|0-9°\-]+", "", s).strip(" .,;/|-")
        if not cleaned_gp or len(cleaned_gp) < 2 or is_header_or_garbage(cleaned_gp):
            return None, []
        return cleaned_gp, []

    # Check tokens for misplaced months or numbers (e.g. 'Paulus 23' or 'Augustus')
    kept_word_tokens: list[str] = []
    for tok in s.split():
        tok_strip = tok.strip(".,;°()[]{}")
        if not tok_strip:
            continue
        if re.match(r"^\d+°?$", tok_strip):
            num = int(re.sub(r"\D", "", tok_strip))
            if 1 <= num <= 31:
                misplaced_dates.append(f"{num}.")
            continue
        if is_month_name(tok_strip):
            misplaced_dates.append(normalize_month_name(tok_strip) or tok_strip)
            continue
        kept_word_tokens.append(tok)

    s_without_dates = " ".join(kept_word_tokens)
    cleaned = _strip_occupations_and_religion_from_name(s_without_dates)
    if not cleaned or len(cleaned) < 2 or cleaned.lower() in NOISE_TOKENS:
        return None, misplaced_dates

    if role == "given":
        # Take first valid given name token or clean given name
        norm_g, _ = normalize_given_name(cleaned, area=area, year=year)
        if not norm_g:
            return None, misplaced_dates
        # If multiple tokens and only one is a known given name, extract that given name
        toks = [t.strip(".,;") for t in norm_g.split() if t.strip(".,;")]
        known = [t for t in toks if is_known_given_name(t)]
        if known:
            best_g, _ = normalize_given_name(" ".join(known[:2]), area=area, year=year)
            return best_g, misplaced_dates
        # Reject unknown multi-word garbled strings if none of the tokens is a known given name
        if len(toks) > 1:
            return None, misplaced_dates
        return norm_g, misplaced_dates

    if role == "surname":
        toks = [
            t.strip(".,;")
            for t in cleaned.split()
            if len(t.strip(".,;")) > 1
            and not is_occupation_or_role_token(t)
            and not is_month_name(t)
        ]
        if not toks:
            return None, misplaced_dates
        # Prefer token that is NOT a known given name if multiple tokens exist (e.g. 'Havalda Michaël' -> 'Havalda')
        non_given = [t for t in toks if not is_known_given_name(t)]
        chosen = non_given[0] if non_given else toks[0]
        if len(chosen) < 2 or is_header_or_garbage(chosen):
            return None, misplaced_dates
        norm_s, _ = normalize_surname(chosen, area=area, year=year)
        return (norm_s or chosen), misplaced_dates

    # role in ("father", "mother", "spouse")
    toks = [
        t.strip(".,;")
        for t in cleaned.split()
        if len(t.strip(".,;")) > 1
        and not is_occupation_or_role_token(t)
        and not is_month_name(t)
    ]
    if not toks:
        return None, misplaced_dates

    # Enforce gender-role type safety for father vs. mother
    if role == "mother":
        low_toks = [re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄË]", "", t).lower() for t in toks]
        has_male = any(lt in KNOWN_MALE_GIVEN_NAMES for lt in low_toks)
        has_female = any(lt in KNOWN_FEMALE_GIVEN_NAMES for lt in low_toks)
        if has_male and not has_female:
            # A male name (e.g. 'Kovaasik Joseph') cannot be a mother's name
            return None, misplaced_dates
        if has_male and has_female:
            # Extract the <Surname> <FemaleGivenName> pair (e.g. 'Lalus Josephus David Maria' -> 'David Maria')
            for idx, lt in enumerate(low_toks):
                if lt in KNOWN_FEMALE_GIVEN_NAMES:
                    start_idx = max(0, idx - 1) if (idx > 0 and low_toks[idx - 1] not in KNOWN_MALE_GIVEN_NAMES) else idx
                    toks = toks[start_idx : idx + 1]
                    break
    elif role == "father":
        low_toks = [re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄË]", "", t).lower() for t in toks]
        has_male = any(lt in KNOWN_MALE_GIVEN_NAMES for lt in low_toks)
        has_female = any(lt in KNOWN_FEMALE_GIVEN_NAMES for lt in low_toks)
        if has_female and not has_male:
            # A female given name (e.g. 'Catharina' or 'Ivo Catharina') cannot be a father's name
            return None, misplaced_dates

    corrected_toks: list[str] = []
    for t in toks:
        if is_known_given_name(t):
            g, _ = normalize_given_name(t, area=area, year=year)
            corrected_toks.append(g or t)
        else:
            s, _ = normalize_surname(t, area=area, year=year)
            corrected_toks.append(s or t)
    result = " ".join(corrected_toks)
    if is_header_or_garbage(result):
        return None, misplaced_dates
    return result, misplaced_dates


def parse_parents_structured(
    raw_parents: str | None,
    *,
    area: str | None = None,
    year: int | str | None = None,
) -> dict[str, Any]:
    """
    Type-safe parser for historical 'Nomen Parentum' cells.
    Handles:
    - Explicit separators: 'et uxor', 'ex utor', 'et utox', 'et', '&', 'portal. of', 'or'
    - Implicit 3-4 token Hungarian/Latin order: '<FatherSurname> <FatherGiven> <MotherSurname> [<MotherGiven>]'
      (e.g. 'Mucejtha Andreas Hudta Catharina', 'Novotny Saulus Cesett')
    - Leading/embedded occupation tokens ('Boarins Terenesik Matheng', 'Kidveny Gaulus sutor')
    - Godparent spillover after 'Idem.' (e.g. Row 11)
    """
    out: dict[str, Any] = {
        "father_name": None,
        "mother_name": None,
        "surname_raw": None,
        "mother_maiden_surname": None,
        "occupation": None,
        "religion": None,
        "misplaced_dates": [],
    }
    if not raw_parents or not raw_parents.strip():
        return out

    out["occupation"] = normalize_occupation(raw_parents)
    out["religion"] = normalize_religion(raw_parents)

    # First check if raw_parents is actually a misplaced month/date (like 'Augustus')
    if is_month_name(raw_parents.strip()):
        out["misplaced_dates"].append(normalize_month_name(raw_parents.strip()) or raw_parents.strip())
        return out

    work = re.sub(r"^\s*[.|/1]+\s*", "", raw_parents).strip()
    # If there are multiple lines or pipe cells in a raw parent snippet, check if separated by 'et' or take best parent segment
    parts = re.split(
        r"\bet\s+uxor\b|\bex\s+utor\b|\bet\s+utox\b|\bet\b|&|\b(?:portal\.?\s+)?of\b|\bor\b",
        work,
        maxsplit=1,
        flags=re.IGNORECASE,
    )

    father_clean, f_dates = validate_person_name(parts[0], role="father", area=area, year=year)
    out["misplaced_dates"].extend(f_dates)

    if len(parts) > 1:
        mother_raw_part = parts[1]
        # Handle spillover after 'Idem.' in multi-column merges (e.g. 'Boarius Kovaasik Joseph Idem. Mikpa Catharina Dath.')
        if "idem" in mother_raw_part.lower():
            idem_parts = re.split(r"\bidem\.?\b", mother_raw_part, flags=re.IGNORECASE)
            # Pick the segment that contains a female given name (e.g. 'Mikpa Catharina')
            chosen_seg = idem_parts[0]
            for seg in idem_parts:
                seg_toks = [
                    re.sub(r"[^a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]", "", t).lower()
                    for t in seg.split()
                ]
                if any(t in KNOWN_FEMALE_GIVEN_NAMES for t in seg_toks):
                    chosen_seg = seg
                    break
            mother_raw_part = chosen_seg

        mother_clean, m_dates = validate_person_name(mother_raw_part, role="mother", area=area, year=year)
        out["misplaced_dates"].extend(m_dates)

        if father_clean:
            f_toks = father_clean.split()
            out["father_name"] = " ".join(f_toks[:2]) if len(f_toks) >= 2 else father_clean
            f_surn_tok = (
                f_toks[1]
                if len(f_toks) >= 2 and is_known_given_name(f_toks[0]) and not is_known_given_name(f_toks[1])
                else f_toks[0]
            )
            surn, _ = validate_person_name(f_surn_tok, role="surname", area=area, year=year)
            out["surname_raw"] = surn

        if mother_clean:
            m_toks = mother_clean.split()
            out["mother_name"] = " ".join(m_toks[:2]) if len(m_toks) >= 2 else mother_clean
            m_surn_tok = (
                m_toks[1]
                if len(m_toks) >= 2 and is_known_given_name(m_toks[0]) and not is_known_given_name(m_toks[1])
                else m_toks[0]
            )
            m_surn, _ = validate_person_name(m_surn_tok, role="surname", area=area, year=year)
            if m_surn and m_surn.lower() not in KNOWN_FEMALE_GIVEN_NAMES:
                out["mother_maiden_surname"] = m_surn
        return out

    # Single segment (no explicit 'et'): check for '<FatherSurname> <FatherGiven> <MotherSurname> [<MotherGiven>]'
    if father_clean:
        toks = father_clean.split()
        if len(toks) >= 3 and is_known_given_name(toks[1]):
            out["father_name"] = f"{toks[0]} {toks[1]}"
            surn, _ = validate_person_name(toks[0], role="surname", area=area, year=year)
            out["surname_raw"] = surn
            mother_tokens = toks[2:4]
            out["mother_name"] = " ".join(mother_tokens)
            m_surn, _ = validate_person_name(mother_tokens[0], role="surname", area=area, year=year)
            if m_surn and m_surn.lower() not in KNOWN_FEMALE_GIVEN_NAMES:
                out["mother_maiden_surname"] = m_surn
        elif len(toks) >= 2:
            # Check if token[0] is a female given name (e.g. 'bala Caharina' where 'bala' is surname and 'Caharina' is mother/parent)
            out["father_name"] = " ".join(toks[:2])
            surn, _ = validate_person_name(toks[0], role="surname", area=area, year=year)
            out["surname_raw"] = surn
        else:
            # Single token: only valid as father_name if it's a known given name or surname (never a month/occupation)
            out["father_name"] = toks[0]
            if not is_known_given_name(toks[0]):
                surn, _ = validate_person_name(toks[0], role="surname", area=area, year=year)
                out["surname_raw"] = surn or toks[0]

    return out


def validate_sex(val: Any, given_name: str | None = None) -> SexLiteral | None:
    """Strictly returns 'M', 'F', or None, prioritizing lexicon gender of `given_name`."""
    inferred = infer_sex_from_given_name(given_name)
    if inferred in ("M", "F"):
        return inferred  # type: ignore[return-value]
    if val is not None:
        s = str(val).strip().lower()
        if s in ("m", "male", "masc", "masculinum", "muž", "chlapec"):
            return "M"
        if s in ("f", "female", "fem", "femininum", "žena", "dievča"):
            return "F"
    return None


def validate_legitimacy(val: Any, raw_ocr: str | None = None) -> LegitimacyLiteral:
    if val is not None:
        s = str(val).strip().lower()
        if "illeg" in s or "neleg" in s or "nemanj" in s:
            return "illegit"
        if "leg" in s:
            return "legit"
    if raw_ocr and re.search(r"\billeg", raw_ocr, re.IGNORECASE):
        # Ignore header line 'from legis | illeg.' when checking row legitimacy
        non_header_lines = [
            ln for ln in raw_ocr.splitlines() if not is_header_or_garbage(ln)
        ]
        if any("illeg" in ln.lower() for ln in non_header_lines):
            return "illegit"
    return "legit"


def validate_village_raw(val: Any, parents_text: str | None = None) -> str | None:
    """Rejects noise, parent cells, or sub-row child entries masquerading as village names."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in NOISE_TOKENS or is_header_or_garbage(s):
        return None
    if re.match(r"^(?:idem|eadem|dtto)\.?$", s, re.IGNORECASE):
        return s
    # Reject if LLM put the parents cell into village_raw (e.g. 'Sleep Sigmanus or Jadák Catharina columnev. a.e.')
    low = s.lower()
    if any(
        m in low
        for m in (" et ", "uxor", "utox", "columnev", "ev. a.", "cv. a.", "r. cath", "familii")
    ):
        return None
    if parents_text and len(s) > 10 and s.lower() in parents_text.lower():
        return None
    # Reject sub-row number + given name like '25° Josephus', '10a', 'Ivo. Catharina.'
    if re.match(r"^\d{1,3}\s*[a-zA-Z°]", s) or "catharina" in low:
        return None
    return s


def validate_house_number(val: Any, village_raw: str | None = None) -> str | None:
    """Validates house number, preferring the number explicitly written in `village_raw` and rejecting '1' from sex/legit columns."""
    _, _, village_h_no = normalize_village(village_raw)
    if village_h_no:
        return village_h_no
    if val is None:
        return None
    s = str(val).strip().strip(".,;№#")
    if not s or s.lower() in NOISE_TOKENS:
        return None
    if not re.match(r"^\d{1,3}[a-zA-Z]?$", s):
        return None
    digits = int(re.sub(r"\D", "", s))
    if digits == 0 or digits > 999:
        return None
    # If LLM returned '1' when village_raw has no '1' in it, it came from the '1 | . | 1' sex/legitimacy column
    if s == "1" and (not village_raw or "1" not in village_raw):
        return None
    return s


class LLMDecisionSchema(BaseModel):
    """Type-safe Pydantic schema for the Stage 3 Decision Model (`d1-3b-mlx`)."""

    model_config = ConfigDict(extra="ignore")

    action: DecisionActionLiteral = "auto_accept"
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    has_ditto: bool = False
    uncertain_field: UncertainFieldLiteral = "parents"
    reason: str = ""
    question_sk: str = ""
    candidates: list[str] = Field(default_factory=list)

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, v: Any) -> DecisionActionLiteral:
        s = str(v or "auto_accept").strip().lower()
        if s in ("ask_human", "auto_accept"):
            return s  # type: ignore[return-value]
        if any(k in s for k in ("ask", "review", "human", "uncertain", "check")):
            return "ask_human"
        return "auto_accept"

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> float:
        try:
            f = float(v)
            return max(0.0, min(1.0, f))
        except Exception:
            return 0.65

    @field_validator("uncertain_field", mode="before")
    @classmethod
    def _coerce_uncertain_field(cls, v: Any) -> UncertainFieldLiteral:
        s = str(v or "parents").strip().lower()
        if s in (
            "parents",
            "village",
            "date_name",
            "observationes",
            "identity_relation",
            "identity_name",
            "birth_marriage_children",
            "birth_marital",
            "origin_domicile",
            "nationality_religion_literacy",
            "occupation_presence_notes",
            "none",
        ):
            return s  # type: ignore[return-value]
        if "ident" in s or "surname" in s or "relation" in s or "pomer" in s:
            return "identity_relation"
        if "marr" in s or "child" in s or "stav" in s:
            return "birth_marriage_children"
        if "origin" in s or "rodisk" in s or "prislus" in s:
            return "origin_domicile"
        if "nation" in s or "relig" in s or "liter" in s:
            return "nationality_religion_literacy"
        if "occup" in s or "povolan" in s or "zavod" in s:
            return "occupation_presence_notes"
        if "date" in s or "name" in s or "given" in s or "birth" in s:
            return "date_name"
        if "vill" in s or "place" in s or "domicil" in s or "house" in s:
            return "village"
        if "obs" in s or "godparent" in s or "patrin" in s or "officiant" in s:
            return "observationes"
        return "parents"

    @field_validator("reason", "question_sk", mode="before")
    @classmethod
    def _clean_text_fields(cls, v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if is_prompt_placeholder(s):
            return ""
        return s

    @field_validator("candidates", mode="before")
    @classmethod
    def _clean_candidates(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if len(s) <= 2 or is_prompt_placeholder(s) or is_header_or_garbage(s):
                continue
            if s not in cleaned:
                cleaned.append(s)
        return cleaned


class LLMExtractionSchema(BaseModel):
    """
    Type-safe Pydantic schema for the Stage 4 Extraction Model (`underdog-saluki-27b-1.0`).
    Automatically detects and repairs semantic type violations and cross-field swaps
    (such as `birth_or_event_date='Paulus 23'` and `father_name='Augustus'`).
    """

    model_config = ConfigDict(extra="ignore")

    entry_number: str | None = None
    event_year: int | None = None
    birth_or_event_date: str | None = None
    baptism_or_burial_date: str | None = None
    given_name_raw: str | None = None
    surname_raw: str | None = None
    sex: SexLiteral | None = None
    father_name: str | None = None
    mother_name: str | None = None
    mother_maiden_surname: str | None = None
    village_raw: str | None = None
    house_number: str | None = None
    godparents_or_witnesses: str | None = None
    officiant: str | None = None
    observationes_raw: str | None = None
    type_safety_repairs: list[str] = Field(default_factory=list)

    @field_validator("sex", mode="before")
    @classmethod
    def _coerce_sex_before(cls, v: Any) -> SexLiteral | None:
        return validate_sex(v)

    @field_validator("event_year", mode="before")
    @classmethod
    def _coerce_event_year_before(cls, v: Any) -> int | None:
        if v is None or str(v).strip() == "":
            return None
        try:
            y = int(str(v).strip())
            return y if 1650 <= y <= 1950 else None
        except Exception:
            return None

    @field_validator(
        "entry_number",
        "birth_or_event_date",
        "baptism_or_burial_date",
        "given_name_raw",
        "surname_raw",
        "father_name",
        "mother_name",
        "mother_maiden_surname",
        "village_raw",
        "house_number",
        "godparents_or_witnesses",
        "officiant",
        "observationes_raw",
        mode="before",
    )
    @classmethod
    def _coerce_scalar_or_list(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x is not None)
        s = str(v).strip()
        if not s or s.lower() in NOISE_TOKENS:
            return None
        return s

    @model_validator(mode="after")
    def _enforce_semantic_types_and_cross_field_swaps(self) -> "LLMExtractionSchema":
        repairs: list[str] = list(self.type_safety_repairs)

        # 1. Validate entry_number
        raw_entry = self.entry_number
        self.entry_number = validate_entry_number(raw_entry)
        if raw_entry and not self.entry_number:
            repairs.append(f"Rejected invalid entry_number={raw_entry!r}")

        # 2. Validate dates and capture any misplaced person names (e.g. 'Paulus 23' -> name 'Paulus', day '23.')
        raw_birth = self.birth_or_event_date
        clean_birth, birth_misplaced_names = validate_matrika_date(
            raw_birth, entry_number=self.entry_number
        )
        if raw_birth and clean_birth != raw_birth:
            repairs.append(
                f"Sanitized birth_or_event_date from {raw_birth!r} to {clean_birth!r}"
                + (f" (rescued person name {birth_misplaced_names})" if birth_misplaced_names else "")
            )
        self.birth_or_event_date = clean_birth

        raw_bapt = self.baptism_or_burial_date
        clean_bapt, bapt_misplaced_names = validate_matrika_date(
            raw_bapt, entry_number=self.entry_number
        )
        if raw_bapt and clean_bapt != raw_bapt:
            repairs.append(
                f"Sanitized baptism_or_burial_date from {raw_bapt!r} to {clean_bapt!r}"
            )
        self.baptism_or_burial_date = clean_bapt

        rescued_person_names = birth_misplaced_names + bapt_misplaced_names
        rescued_dates_or_months: list[str] = []
        area_hint = self.village_raw
        year_hint = self.event_year

        # 3. Check if village_raw, godparents, or observationes accidentally received the parents cell
        for wrong_field in ("village_raw", "godparents_or_witnesses", "observationes_raw"):
            f_val = getattr(self, wrong_field)
            if not f_val:
                continue
            low_f = f_val.lower()
            looks_like_parents = any(
                m in low_f
                for m in ("familii", "coa colm", "columnev", "portal. of", "hudta catharina", "saulus cesett", "jadák catharina")
            )
            if looks_like_parents:
                if not self.father_name or is_month_name(self.father_name):
                    parsed_p = parse_parents_structured(f_val, area=area_hint, year=year_hint)
                    if parsed_p["father_name"]:
                        if self.father_name and is_month_name(self.father_name):
                            rescued_dates_or_months.append(
                                normalize_month_name(self.father_name) or self.father_name
                            )
                        self.father_name = parsed_p["father_name"]
                        self.mother_name = self.mother_name or parsed_p["mother_name"]
                        self.surname_raw = self.surname_raw or parsed_p["surname_raw"]
                        self.mother_maiden_surname = (
                            self.mother_maiden_surname or parsed_p["mother_maiden_surname"]
                        )
                        repairs.append(
                            f"Rescued parents ({self.father_name!r}) misplaced in {wrong_field}"
                        )
                setattr(self, wrong_field, None)

        # 4. Validate person name fields and capture any misplaced months/dates (e.g. father_name='Augustus')
        raw_given = self.given_name_raw
        clean_given, given_dates = validate_person_name(
            raw_given, role="given", area=area_hint, year=year_hint
        )
        if raw_given and clean_given != raw_given:
            match_notes = name_match_repair_notes(raw_given, clean_given)
            repairs.extend(match_notes or [f"Sanitized given_name_raw from {raw_given!r} to {clean_given!r}"])
        self.given_name_raw = clean_given
        rescued_dates_or_months.extend(given_dates)

        raw_father = self.father_name
        clean_father, father_dates = validate_person_name(
            raw_father, role="father", area=area_hint, year=year_hint
        )
        if raw_father and clean_father != raw_father:
            match_notes = name_match_repair_notes(raw_father, clean_father)
            if match_notes:
                repairs.extend(match_notes)
            else:
                repairs.append(
                    f"Sanitized father_name from {raw_father!r} to {clean_father!r}"
                    + (f" (rescued month/date {father_dates})" if father_dates else "")
                )
        self.father_name = clean_father
        rescued_dates_or_months.extend(father_dates)

        raw_mother = self.mother_name
        # If father_name was empty/dot and LLM put both father & mother into mother_name (e.g. Row 19)
        if not self.father_name and raw_mother:
            parsed_both = parse_parents_structured(raw_mother, area=area_hint, year=year_hint)
            if parsed_both["father_name"] and parsed_both["mother_name"]:
                self.father_name = parsed_both["father_name"]
                self.mother_name = parsed_both["mother_name"]
                self.surname_raw = self.surname_raw or parsed_both["surname_raw"]
                self.mother_maiden_surname = (
                    self.mother_maiden_surname or parsed_both["mother_maiden_surname"]
                )
                repairs.append(
                    f"Split combined parents in mother_name into father={self.father_name!r}, mother={self.mother_name!r}"
                )
                raw_mother = self.mother_name

        clean_mother, mother_dates = validate_person_name(
            raw_mother, role="mother", area=area_hint, year=year_hint
        )
        if raw_mother and clean_mother != raw_mother:
            repairs.extend(name_match_repair_notes(raw_mother, clean_mother))
        self.mother_name = clean_mother
        rescued_dates_or_months.extend(mother_dates)

        # Check for LLM misalignment: surname_raw='Mucejtha', father_name='Andreas Hudta', mother_name='Catharina'
        if self.surname_raw and self.father_name and self.mother_name:
            f_toks = self.father_name.split()
            m_toks = self.mother_name.split()
            if (
                len(f_toks) == 2
                and is_known_given_name(f_toks[0])
                and not is_known_given_name(f_toks[1])
                and len(m_toks) == 1
                and is_known_given_name(m_toks[0])
            ):
                given_f = f_toks[0]
                maiden_m = f_toks[1]
                given_m = m_toks[0]
                self.father_name = f"{self.surname_raw} {given_f}"
                self.mother_name = f"{maiden_m} {given_m}"
                self.mother_maiden_surname = maiden_m
                repairs.append(
                    f"Realigned <Surname> <FatherGiven> <MotherSurname> <MotherGiven> to father={self.father_name!r}, mother={self.mother_name!r}"
                )

        raw_surname = self.surname_raw
        clean_surname, surname_dates = validate_person_name(
            raw_surname, role="surname", area=area_hint, year=year_hint
        )
        if raw_surname and clean_surname != raw_surname:
            repairs.extend(
                name_match_repair_notes(raw_surname, clean_surname)
                or [f"Sanitized surname_raw from {raw_surname!r} to {clean_surname!r}"]
            )
        self.surname_raw = clean_surname
        rescued_dates_or_months.extend(surname_dates)

        raw_maiden = self.mother_maiden_surname
        clean_maiden, maiden_dates = validate_person_name(
            raw_maiden, role="surname", area=area_hint, year=year_hint
        )
        if raw_maiden and clean_maiden != raw_maiden:
            repairs.extend(name_match_repair_notes(raw_maiden, clean_maiden))
        self.mother_maiden_surname = clean_maiden
        rescued_dates_or_months.extend(maiden_dates)

        # 5. Cross-Field Swap Resolution (e.g. birth_or_event_date='Paulus 23' & father_name='Augustus')
        if rescued_person_names and not self.father_name:
            rescued_name = rescued_person_names[0]
            if self.surname_raw and self.surname_raw.lower() != rescued_name.lower():
                self.father_name = f"{self.surname_raw} {rescued_name}"
            else:
                self.father_name = rescued_name
            repairs.append(
                f"Cross-field swap: assigned rescued person name {rescued_name!r} from date field to father_name={self.father_name!r}"
            )
        elif rescued_person_names and self.father_name:
            # If father_name only had a surname (not a known given name), append the rescued given name
            f_toks = self.father_name.split()
            if len(f_toks) == 1 and not is_known_given_name(f_toks[0]):
                self.father_name = f"{self.father_name} {rescued_person_names[0]}"
                repairs.append(
                    f"Cross-field swap: attached rescued given name {rescued_person_names[0]!r} to father_name={self.father_name!r}"
                )

        if rescued_dates_or_months:
            rescued_m = rescued_dates_or_months[0]
            if not self.birth_or_event_date:
                self.birth_or_event_date = rescued_m
                repairs.append(
                    f"Cross-field swap: moved misplaced month/date {rescued_m!r} from person field to birth_or_event_date"
                )
            elif re.match(r"^\d{1,2}\.?$", self.birth_or_event_date) and is_month_name(rescued_m):
                day_num = self.birth_or_event_date.rstrip(".")
                self.birth_or_event_date = f"{rescued_m} {day_num}"
                repairs.append(
                    f"Cross-field swap: combined misplaced month {rescued_m!r} with day {day_num!r} -> birth_or_event_date={self.birth_or_event_date!r}"
                )

        # 6. Validate sex, village, house_number, godparents, officiant, observationes
        self.sex = validate_sex(self.sex, self.given_name_raw)
        self.village_raw = validate_village_raw(self.village_raw, self.father_name)
        self.house_number = validate_house_number(self.house_number, self.village_raw)

        clean_gp, _ = validate_person_name(self.godparents_or_witnesses, role="godparents")
        if clean_gp:
            clean_gp = correct_names_in_text(clean_gp, area=area_hint, year=year_hint)
        self.godparents_or_witnesses = clean_gp
        clean_off, _ = validate_person_name(self.officiant, role="officiant")
        self.officiant = clean_off
        clean_obs, _ = validate_person_name(self.observationes_raw, role="godparents")
        self.observationes_raw = clean_obs

        self.type_safety_repairs = repairs
        return self


class ValidatedPersonFactPayload(BaseModel):
    """Final strictly-typed payload written to `RecordRow.normalized_json` and `PersonFact`."""

    model_config = ConfigDict(extra="ignore")

    entry_number: str | None = None
    event_year: int = 1884
    birth_or_event_date: str | None = None
    baptism_or_burial_date: str | None = None
    given_name_raw: str | None = None
    given_name_sk: str | None = None
    given_name_hu: str | None = None
    surname_raw: str | None = None
    surname_normalized: str | None = None
    sex: SexLiteral | None = None
    legitimacy: LegitimacyLiteral = "legit"
    religion: str | None = None
    father_name: str | None = None
    mother_name: str | None = None
    mother_maiden_surname: str | None = None
    spouse_name: str | None = None
    social_status_or_occupation: str | None = None
    village_raw: str | None = None
    village_sk: str | None = None
    village_hu: str | None = None
    house_number: str | None = None
    godparents_or_witnesses: str | None = None
    officiant: str | None = None
    age_at_event: str | None = None
    cause_of_death: str | None = None
    observationes_raw: str | None = None
    observationes_events: list[dict[str, Any]] = Field(default_factory=list)
    type_safety_repairs: list[str] = Field(default_factory=list)


def apply_human_answer_to_fields(
    target: dict[str, Any],
    field_name: str,
    answer: str,
    answered_by: str = "Reviewer",
) -> dict[str, Any]:
    """
    Type-safely applies a human answer from QuestionCenter to a fact/extraction dictionary.
    Prevents dates from overwriting given names or combined parent answers from losing mother's maiden name.
    """
    ans = (answer or "").strip()
    if not ans:
        return target

    if field_name == "village":
        target["village_raw"] = ans
        v_sk, v_hu, h_no = normalize_village(ans)
        if v_sk:
            target["village_sk"] = v_sk
        if v_hu:
            target["village_hu"] = v_hu
        if h_no:
            target["house_number"] = h_no

    elif field_name == "parents":
        parsed_p = parse_parents_structured(
            ans,
            area=target.get("village_hu") or target.get("village_sk") or target.get("village_raw"),
            year=target.get("event_year"),
        )
        if parsed_p["father_name"]:
            target["father_name"] = parsed_p["father_name"]
        else:
            target["father_name"] = ans
        if parsed_p["surname_raw"]:
            target["surname_raw"] = parsed_p["surname_raw"]
            target["surname_normalized"] = parsed_p["surname_raw"]
        if parsed_p["mother_name"]:
            target["mother_name"] = parsed_p["mother_name"]
        if parsed_p["mother_maiden_surname"]:
            target["mother_maiden_surname"] = parsed_p["mother_maiden_surname"]
        if parsed_p["occupation"] and not target.get("social_status_or_occupation"):
            target["social_status_or_occupation"] = parsed_p["occupation"]

    elif field_name == "date_name":
        # Format may be "Andreas (18. Augustus)", "28. Julius / 28. Julius", or a single date or given name
        paren_m = re.match(r"^([^(]+)\(([^)]+)\)$", ans.strip())
        if paren_m:
            name_part = paren_m.group(1).strip()
            date_part = paren_m.group(2).strip()
            clean_g, _ = validate_person_name(name_part, role="given")
            clean_d, _ = validate_matrika_date(date_part, entry_number=target.get("entry_number"))
            if clean_g:
                target["given_name_raw"] = clean_g
                target["given_name_sk"] = clean_g
                target["given_name_hu"] = clean_g
                target["sex"] = validate_sex(target.get("sex"), clean_g)
            if clean_d:
                target["birth_or_event_date"] = clean_d
        elif "/" in ans:
            parts = [p.strip() for p in ans.split("/") if p.strip()]
            d1, n1 = validate_matrika_date(parts[0], entry_number=target.get("entry_number")) if parts else (None, [])
            d2, n2 = validate_matrika_date(parts[1], entry_number=target.get("entry_number")) if len(parts) > 1 else (None, [])
            if d1 and not n1:
                target["birth_or_event_date"] = d1
            if d2 and not n2:
                target["baptism_or_burial_date"] = d2
        else:
            clean_d, misplaced_n = validate_matrika_date(ans, entry_number=target.get("entry_number"))
            clean_g, _ = validate_person_name(ans, role="given")
            if clean_d and not misplaced_n:
                target["birth_or_event_date"] = clean_d
            elif clean_g:
                target["given_name_raw"] = clean_g
                target["given_name_sk"] = clean_g
                target["given_name_hu"] = clean_g
                target["sex"] = validate_sex(target.get("sex"), clean_g)

    elif field_name in ("identity_relation", "identity_name"):
        from backend.app.pipeline.lexicon import extract_maiden_surname, normalize_census_relation

        # Can be "Kováč Ján (prednosta)" or "Kováčová rod. Vargová Mária" or "Surname Given"
        rel_m = re.search(r"\(([^)]+)\)", ans)
        if rel_m:
            target["relation_to_head"] = normalize_census_relation(rel_m.group(1))
            ans_names = re.sub(r"\([^)]+\)", "", ans).strip()
        else:
            ans_names = ans
        married, maiden = extract_maiden_surname(ans_names)
        if maiden:
            target["mother_maiden_surname"] = maiden
        toks = [t for t in (married or ans_names).split() if t]
        if len(toks) >= 2:
            target["surname_raw"] = toks[0]
            target["surname_normalized"] = toks[0]
            target["given_name_raw"] = " ".join(toks[1:])
            target["given_name_sk"] = target["given_name_raw"]
            target["sex"] = validate_sex(target.get("sex"), target["given_name_raw"])
        elif len(toks) == 1:
            if is_known_given_name(toks[0]):
                target["given_name_raw"] = toks[0]
                target["given_name_sk"] = toks[0]
            else:
                target["surname_raw"] = toks[0]
                target["surname_normalized"] = toks[0]

    elif field_name in ("birth_marriage_children", "birth_marital"):
        cdate, cyear = validate_census_date(ans)
        if cdate:
            target["birth_or_event_date"] = cdate
        if cyear:
            target["birth_year_est"] = cyear

    elif field_name == "origin_domicile":
        target["birth_place_raw"] = ans
        v_sk, _, _ = normalize_village(ans)
        target["birth_place_sk"] = v_sk or ans

    elif field_name == "nationality_religion_literacy":
        rel_norm = normalize_religion(ans)
        if rel_norm:
            target["religion"] = rel_norm
        else:
            target["nationality"] = ans

    elif field_name == "occupation_presence_notes":
        occ_norm = normalize_occupation(ans)
        target["social_status_or_occupation"] = occ_norm or ans

    else:
        existing_obs = target.get("observationes_raw") or ""
        target["observationes_raw"] = f"{existing_obs} | Overené ({answered_by}): {ans}".strip(" |")

    return target


_ROMAN_MONTHS: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}


def validate_census_date(val: Any) -> tuple[str | None, int | None]:
    """
    Validates a 1930/1940 census birth or marriage date (which INCLUDES a 4-digit year,
    unlike church register day+month cells).
    Examples:
      - '14. V. 1892' -> ('14. 5. 1892', 1892)
      - '12. 3. 1878' -> ('12. 3. 1878', 1878)
      - '5. mája 1904' -> ('5. Máj 1904', 1904)
      - '1889' -> ('1889', 1889)
    """
    if val is None:
        return None, None
    s = str(val).strip()
    if not s or s.lower() in NOISE_TOKENS:
        return None, None

    year_val: int | None = None
    ym = re.search(r"\b(18\d{2}|19[0-4]\d)\b", s)
    if ym:
        year_val = int(ym.group(1))

    # ISO format YYYY-MM-DD
    iso_m = re.match(r"^(18\d{2}|19[0-4]\d)-(\d{1,2})-(\d{1,2})$", s)
    if iso_m:
        y, m, d = int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{d}. {m}. {y}", y

    # Numeric or Roman DD. MM. YYYY
    dm_m = re.search(
        r"\b(\d{1,2})\s*[./\-]\s*(\d{1,2}|[I|V|X]{1,4})\s*[./\-]?\s*(18\d{2}|19[0-4]\d)\b",
        s,
        re.IGNORECASE,
    )
    if dm_m:
        day = int(dm_m.group(1))
        m_raw = dm_m.group(2).lower()
        month = int(m_raw) if m_raw.isdigit() else _ROMAN_MONTHS.get(m_raw, 0)
        yr = int(dm_m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{day}. {month}. {yr}", yr

    # Day + Month Word + Year
    tokens = re.findall(r"[A-Za-z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰäëÄËčďľĺňŕřšťžČĎĽĹŇŔŘŠŤŽýÝôÔ]+", s)
    day_num: int | None = None
    month_str: str | None = None
    for tok in tokens:
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= 31 and day_num is None:
                day_num = n
        elif tok.lower() in _ROMAN_MONTHS and month_str is None:
            month_str = str(_ROMAN_MONTHS[tok.lower()]) + "."
        elif is_month_name(tok) and month_str is None:
            month_str = normalize_month_name(tok)

    if day_num and month_str and year_val:
        return f"{day_num}. {month_str} {year_val}", year_val
    if month_str and year_val:
        return f"{month_str} {year_val}", year_val
    if year_val:
        return str(year_val), year_val
    if day_num and month_str:
        return f"{day_num}. {month_str}", None
    return None, None


class LLMCensusExtractionSchema(BaseModel):
    """
    Type-safe Pydantic schema for 1930 & 1940 Slovak Census Sheet person rows.
    """

    model_config = ConfigDict(extra="ignore")

    apartment_number: str | None = "1"
    person_row_number: str | None = None
    surname_raw: str | None = None
    maiden_surname: str | None = None
    given_name_raw: str | None = None
    relation_to_head: str | None = None
    sex: SexLiteral | None = None
    birth_date_raw: str | None = None
    birth_year: int | None = None
    marital_status: str | None = None
    marriage_date_raw: str | None = None
    children_born_count: str | None = None
    children_dead_count: str | None = None
    birth_place_raw: str | None = None
    birth_place_municipality: str | None = None
    birth_place_district: str | None = None
    birth_place_country: str | None = None
    citizenship: str | None = None
    moved_in_date: str | None = None
    previous_residence: str | None = None
    home_jurisdiction: str | None = None
    nationality: str | None = None
    religion: str | None = None
    literacy: str | None = None
    occupation: str | None = None
    occupation_position: str | None = None
    employer_workplace: str | None = None
    presence_type: str | None = None
    notes_raw: str | None = None
    type_safety_repairs: list[str] = Field(default_factory=list)

    @field_validator(
        "apartment_number",
        "person_row_number",
        "surname_raw",
        "maiden_surname",
        "given_name_raw",
        "relation_to_head",
        "birth_date_raw",
        "marital_status",
        "marriage_date_raw",
        "children_born_count",
        "children_dead_count",
        "birth_place_raw",
        "birth_place_municipality",
        "birth_place_district",
        "birth_place_country",
        "citizenship",
        "moved_in_date",
        "previous_residence",
        "home_jurisdiction",
        "nationality",
        "religion",
        "literacy",
        "occupation",
        "occupation_position",
        "employer_workplace",
        "presence_type",
        "notes_raw",
        mode="before",
    )
    @classmethod
    def _coerce_census_str(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x is not None)
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "undefined"}:
            return None
        return s

    @field_validator("birth_year", mode="before")
    @classmethod
    def _coerce_census_birth_year(cls, v: Any) -> int | None:
        if v is None or str(v).strip() == "":
            return None
        try:
            m = re.search(r"\b(18\d{2}|19[0-4]\d)\b", str(v))
            return int(m.group(1)) if m else int(v)
        except Exception:
            return None

    @field_validator("sex", mode="before")
    @classmethod
    def _coerce_census_sex(cls, v: Any) -> SexLiteral | None:
        return validate_sex(v)

    @model_validator(mode="after")
    def _validate_census_fields(self) -> "LLMCensusExtractionSchema":
        from backend.app.pipeline.lexicon import extract_maiden_surname, normalize_census_relation

        repairs: list[str] = list(self.type_safety_repairs)
        if self.surname_raw:
            married, maiden = extract_maiden_surname(self.surname_raw)
            if maiden and not self.maiden_surname:
                self.maiden_surname = maiden
                self.surname_raw = married
                repairs.append(f"Split maiden surname {maiden!r} from surname_raw")

        clean_bdate, byear = validate_census_date(self.birth_date_raw or self.birth_year)
        if clean_bdate:
            self.birth_date_raw = clean_bdate
        if byear and not self.birth_year:
            self.birth_year = byear

        if self.relation_to_head:
            self.relation_to_head = normalize_census_relation(self.relation_to_head)

        self.sex = validate_sex(self.sex, self.given_name_raw)

        if self.surname_raw:
            s_clean = re.sub(r"^(?:vd\.?|va\.?|ovd\.?)\s+", "", self.surname_raw.strip(), flags=re.IGNORECASE).strip()
            s_clean = sanitize_corrupted_surname_suffix(s_clean) or s_clean
            self.surname_raw = s_clean

        self.type_safety_repairs = repairs
        return self


class LLMCensusOverviewRowSchema(BaseModel):
    """
    Type-safe Pydantic schema for extracting a single house row from a
    Prehľad sčítacieho obvodu (1930/1940 Census District Overview Sheet).
    """

    model_config = ConfigDict(extra="ignore")

    ordinal_number: str | None = None
    settlement_name: str | None = None
    house_number: str | None = None
    is_inhabited: bool = True
    owner_name: str | None = None
    household_count: int | None = None
    present_persons_count: int | None = None
    remarks: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("ordinal_number", "settlement_name", "house_number", "owner_name", "remarks", mode="before")
    @classmethod
    def _coerce_opt_str(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"null", "none", "n/a", "undefined"}:
            return None
        return s

    @field_validator("is_inhabited", mode="before")
    @classmethod
    def _coerce_inhabited(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return True
        s = str(v).strip().lower()
        if s in {"0", "false", "no", "nie", "neobývaný", "neobyvany", "prázdny", "prazdny"}:
            return False
        return True

    @field_validator("household_count", "present_persons_count", mode="before")
    @classmethod
    def _coerce_opt_int(cls, v: Any) -> int | None:
        if v is None or str(v).strip() == "":
            return None
        try:
            m = re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else None
        except Exception:
            return None

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_overview_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.85

