"""
Fine-Tuned OpenCV Layout Slicer for 1930 and 1940 Slovak Census Sheets (Sčítacie a popisné hárky).

Key differences from 19th-century church register spreads (`layout.py`):
1. Single wide landscape sheet (or front `_v` / back `_r` page) with 24 numbered rubric columns
   and 10–15 dense horizontal person rows (`0.028 * h <= row_height <= 0.10 * h`).
2. Continuous horizontal rules across the entire width (`spine_offset_px = 0.0`), split into
   left half (Cols 1–10: Identity, Relation, Sex, Birth Date, Marital Status, Marriage, Children)
   and right half (Cols 11–24: Birthplace, Immigration, Domicile, Nationality, Religion, Literacy,
   Occupation, Position, Workplace, Presence, Notes) at `x ≈ 0.42 * w`.
3. Deep administrative & rubric header banner on the front page (`0.24 * h .. 0.32 * h`).
4. Automatic blank-row suppression so pre-numbered empty lines at the bottom of a small household
   do not waste OCR/LLM calls.
5. Generates 5 census semantic `cell_regions` for QuestionCenter human-in-the-loop review.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.app.pipeline.layout import (
    LayoutResult,
    RowSliceInfo,
    _AngledLineModel,
    _warp_piecewise_angled_strip,
)
from backend.app.pipeline.lexicon import fold_text_diacritics


MUNICIPAL_OVERVIEW_HEADER_MARKERS: tuple[str, ...] = (
    "obecny prehlad",
    "meno scitacieho revizora",
    "vyplni scitaci revizor",
    "vyhotovi revizor",
    "v obciach rozdelenych na dva alebo viac",
    "sucet pre celu obec",
    "podla prehladov scitacich obvodov",
)

OVERVIEW_HEADER_MARKERS: tuple[str, ...] = (
    "prehlad scitacieho obvodu",
    "prehlad obvodu",
    "meno scitacieho komisara",
    "scitaci komisar",
    "pocet bytovych stran",
    "radove cislo osad",
    "meno majitela domu",
    "dom je obyvany",
    "pocet pritomnych osob",
)


def detect_is_municipal_overview_sheet(header_text: str | None, raw_ocr: str | None = "") -> bool:
    """
    Detects whether a census sheet is an 'Obecný prehľad' (Municipal Overview Sheet, Form X)
    which joins and summarizes multiple 'Prehľad sčítacieho obvodu' (District Overview / Prehľad domov) sheets.
    """
    combined = f"{header_text or ''} {raw_ocr or ''}".strip()
    if not combined:
        return False
    folded = fold_text_diacritics(combined)
    if "meno majitela domu" in folded or "popisne cislo domu" in folded:
        return False
    for marker in MUNICIPAL_OVERVIEW_HEADER_MARKERS:
        if marker in folded:
            return True
    return False


def detect_is_overview_sheet(header_text: str | None, raw_ocr: str | None = "") -> bool:
    """
    Detects whether a census sheet is a 'Prehľad sčítacieho obvodu' (District Overview Sheet / Prehľad domov)
    rather than an 'Obecný prehľad' ('municipal_overview') or an individual household sheet ('household').
    """
    if detect_is_municipal_overview_sheet(header_text, raw_ocr):
        return False
    combined = f"{header_text or ''} {raw_ocr or ''}".strip()
    if not combined:
        return False
    folded = fold_text_diacritics(combined)
    for marker in OVERVIEW_HEADER_MARKERS:
        if marker in folded:
            return True
    return False


def extract_municipal_overview_header_metadata(header_text: str | None) -> dict[str, str | None]:
    """Extracts revisor_name from Obecný prehľad header OCR."""
    if not header_text:
        return {"revisor_name": None}
    revisor_name: str | None = None
    m_rev = re.search(
        r"(?:rev[ií]zor\w*)\s*[:\-|]?\s*([A-ZÁČĎÉĚÍĽĹŇÓÔŔŘŠŤÚŮÝŽ][A-Za-záčďéěíľĺňóôŕřšťúůýžÁČĎÉĚÍĽĹŇÓÔŔŘŠŤÚŮÝŽ.\-\s\u0301]{2,40})",
        header_text,
        re.IGNORECASE,
    )
    if m_rev:
        raw_rev = m_rev.group(1).split("|")[0].split("\n")[0].strip(" .,;:-")
        if (
            len(raw_rev) >= 3
            and "meno" not in raw_rev.lower()
            and "obec" not in raw_rev.lower()
            and "prehľad" not in raw_rev.lower()
            and "prehlad" not in raw_rev.lower()
            and "vypln" not in raw_rev.lower()
        ):
            raw_rev = re.sub(r"\bFranti[sš]el\w*\b|\bFranti[sš]ek\w+\b", "František", raw_rev, flags=re.IGNORECASE)
            raw_rev = re.sub(r"\bMally\u0301?\b", "Mallý", raw_rev, flags=re.IGNORECASE)
            raw_rev = re.sub(r"\bChlumsky\b", "Chlumský", raw_rev, flags=re.IGNORECASE)
            raw_rev = re.sub(r"\bEuglius\b|\bJulias\b", "Július", raw_rev, flags=re.IGNORECASE)
            revisor_name = raw_rev
    return {"revisor_name": revisor_name}


def extract_overview_header_metadata(header_text: str | None) -> dict[str, str | None]:
    """Extracts obvod_number and commissioner_name from Prehľad sčítacieho obvodu header OCR."""
    if not header_text:
        return {"obvod_number": None, "commissioner_name": None}
    obvod_number: str | None = None
    commissioner_name: str | None = None

    m_obv = re.search(
        r"(?:obvod\w*\s*(?:č\.?|číslo|c\.?)?\s*[:\-]?\s*)([IVX0-9]{1,5}\b)",
        header_text,
        re.IGNORECASE,
    )
    if m_obv:
        cand = m_obv.group(1).strip()
        if cand.lower() not in {"od", "do", "v", "z"}:
            obvod_number = cand

    m_kom = re.search(
        r"(?:komis[aá]r\w*|rev[ií]zor\w*)\s*[:\-|]?\s*([A-ZÁČĎÉĚÍĽĹŇÓÔŔŘŠŤÚŮÝŽ][A-Za-záčďéěíľĺňóôŕřšťúůýžÁČĎÉĚÍĽĹŇÓÔŔŘŠŤÚŮÝŽ.\-\s]{2,40})",
        header_text,
        re.IGNORECASE,
    )
    if m_kom:
        raw_kom = m_kom.group(1).split("|")[0].split("\n")[0].strip(" .,;:-")
        if len(raw_kom) >= 3 and "meno" not in raw_kom.lower() and "prehľad" not in raw_kom.lower() and "prehlad" not in raw_kom.lower():
            raw_kom = re.sub(r"\bJoref\b|\bZoref\b", "Jozef", raw_kom, flags=re.IGNORECASE)
            commissioner_name = raw_kom

    return {"obvod_number": obvod_number, "commissioner_name": commissioner_name}


def suppress_auditor_crayon(img_bgr: np.ndarray) -> np.ndarray:
    """
    Attenuates high-saturation green and red census auditor crayon markings
    (e.g. '624', '2', '10', '677', vertical green continuation bars) by replacing
    saturated green/red pixels with the local warm paper background tone while
    preserving dark ink handwriting and small '//' ditto marks.
    """
    if img_bgr is None or img_bgr.size == 0 or img_bgr.ndim != 3:
        return img_bgr

    out = img_bgr.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    b_ch = out[:, :, 0].astype(np.int16)
    g_ch = out[:, :, 1].astype(np.int16)
    r_ch = out[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # 1. Green / teal / cyan auditor crayon:
    # Hue ~32..105 in OpenCV (0..180 scale), moderate-to-high saturation, or strong G dominance
    green_hsv = (h_ch >= 32) & (h_ch <= 105) & (s_ch >= 38) & (v_ch >= 50)
    green_rgb = (g_ch - r_ch >= 14) & (g_ch - b_ch >= 10) & (g_ch >= 55)
    green_mask = green_hsv | green_rgb

    # 2. Red / crimson / magenta auditor crayon:
    # Avoid sepia/brown ink (Hue 12..26); target pure red/magenta with strong R dominance
    red_hsv = ((h_ch <= 10) | (h_ch >= 165)) & (s_ch >= 65) & (v_ch >= 65)
    red_rgb = (r_ch - g_ch >= 32) & (r_ch - b_ch >= 28) & (r_ch >= 80)
    red_mask = red_hsv & red_rgb

    crayon_mask = (green_mask | red_mask).astype(np.uint8) * 255
    if not np.any(crayon_mask):
        return out

    # Slightly dilate to catch anti-aliased colour fringes, but protect very dark ink strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crayon_dil = cv2.dilate(crayon_mask, kernel, iterations=1)
    # Dark handwriting ink has low luminance and low green excess
    dark_ink = (gray < 68) & ((g_ch - r_ch) < 18) & ((r_ch - g_ch) < 42)
    final_mask = (crayon_dil > 0) & (~dark_ink)

    if not np.any(final_mask):
        return out

    # Estimate warm paper background tone from non-crayon pixels
    non_crayon = out[~final_mask]
    if non_crayon.size >= 30:
        paper_bgr = np.percentile(non_crayon.reshape(-1, 3), 78, axis=0).astype(np.uint8)
    else:
        paper_bgr = np.percentile(out.reshape(-1, 3), 78, axis=0).astype(np.uint8)

    # Build a smooth local paper background so replacement has natural local shading
    bg_canvas = out.copy()
    bg_canvas[final_mask] = paper_bgr
    local_bg = cv2.GaussianBlur(bg_canvas, (31, 31), 0)
    out[final_mask] = local_bg[final_mask]
    return out


def _estimate_line_angles_hough(
    strip: np.ndarray,
    max_abs_deg: float = 3.5,
) -> tuple[float, list[float]]:
    """Estimates dominant horizontal line skew angle (in degrees) using Probabilistic Hough."""
    edges = cv2.Canny(strip, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=80,
        minLineLength=max(80, strip.shape[1] // 8),
        maxLineGap=25,
    )
    if lines is None:
        return 0.0, []
    angles: list[float] = []
    weights: list[float] = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 30:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if abs(ang) <= max_abs_deg:
            angles.append(ang)
            weights.append(float(np.hypot(dx, dy)))
    if not angles:
        return 0.0, []
    order = np.argsort(angles)
    sorted_a = np.array(angles)[order]
    sorted_w = np.array(weights)[order]
    cum_w = np.cumsum(sorted_w)
    med_idx = int(np.searchsorted(cum_w, 0.5 * cum_w[-1]))
    return float(sorted_a[min(med_idx, len(sorted_a) - 1)]), angles


def _detect_census_split_x(gray: np.ndarray, header_y_end: int, bottom_y_end: int) -> int:
    """
    Finds the vertical column rule near 0.42 * w (between Col. 10 'Children' and Col. 11 'Birthplace')
    so no handwritten cell is bisected when splitting into left/right OCR strips.
    """
    h, w = gray.shape
    search_l = int(w * 0.37)
    search_r = int(w * 0.47)
    y0 = max(0, min(header_y_end, int(h * 0.30)))
    y1 = max(y0 + 40, min(h, bottom_y_end))
    strip = gray[y0:y1, search_l:search_r]
    if strip.size == 0:
        return int(w * 0.42)
    inv = np.clip(float(np.percentile(strip, 75)) - strip.astype(np.float64), 0.0, None)
    col_energy = np.mean(inv, axis=0)
    k = 7
    smooth = np.convolve(col_energy, np.ones(k) / float(k), mode="same")
    # Mild preference toward 0.42 * w
    xs = np.arange(len(smooth), dtype=np.float64) + search_l
    prior = -0.10 * (np.abs(xs - w * 0.42) / max(1.0, w * 0.05)) * float(np.max(smooth) + 1.0)
    best_offset = int(np.argmax(smooth + prior))
    return search_l + best_offset


def _skew_compensated_row_projection(
    horiz_lines: np.ndarray,
    x0: int,
    xc: float,
    tan_a: float,
    n_slices: int = 12,
) -> np.ndarray:
    """Computes a skew-compensated horizontal row projection across `n_slices` vertical strips."""
    h = horiz_lines.shape[0]
    xs = np.linspace(0, horiz_lines.shape[1], n_slices + 1, dtype=int)
    proj = np.zeros(h, dtype=np.float64)
    y_coords = np.arange(h, dtype=np.float64)
    for i in range(n_slices):
        sx0, sx1 = xs[i], xs[i + 1]
        if sx1 <= sx0:
            continue
        x_mid = x0 + 0.5 * (sx0 + sx1)
        dy = int(round((x_mid - xc) * tan_a))
        col_prof = np.mean(horiz_lines[:, sx0:sx1], axis=1)
        shifted = np.interp(
            y_coords + dy,
            y_coords,
            col_prof,
            left=0.0,
            right=0.0,
        )
        proj += shifted
    proj /= float(n_slices)
    return proj


def _detect_census_row_boundaries(
    gray: np.ndarray,
    page_side: str = "front",
) -> tuple[list[_AngledLineModel], dict[str, Any]]:
    """
    Detects continuous horizontal row boundaries on a 1930/1940 census sheet.
    """
    h, w = gray.shape
    x0 = int(w * 0.03)
    x1 = int(w * 0.97)
    strip = gray[:, x0:x1]
    angle_deg, _ = _estimate_line_angles_hough(strip, max_abs_deg=3.5)
    tan_a = float(np.tan(np.deg2rad(angle_deg)))
    xc = 0.5 * w

    bg = float(np.percentile(strip, 78))
    inv_u8 = np.clip(bg - strip.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    # Dilate vertically by 3px before horizontal morphological opening so slightly tilted 1-2px lines survive
    inv_dil = cv2.dilate(inv_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    morph_w = max(50, (x1 - x0) // 45)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, 1))
    horiz_lines = cv2.morphologyEx(inv_dil, cv2.MORPH_OPEN, horiz_kernel).astype(np.float64)

    proj = _skew_compensated_row_projection(horiz_lines, x0, xc, tan_a, n_slices=12)

    smooth = np.convolve(proj, np.ones(9) / 9.0, mode="same")

    # Front page has administrative banner + rubric header down to ~0.24h..0.41h;
    # Back page has a shorter rubric header down to ~0.14h..0.22h.
    y_min_search = int(h * (0.20 if page_side != "back" else 0.13))
    y_max_search = int(h * 0.95)
    min_dist = max(18, int(h * 0.025))

    region = smooth[y_min_search:y_max_search]
    thresh = max(float(np.percentile(region, 72)) if region.size else 4.0, 5.5)

    cands: list[tuple[float, int]] = []
    win = max(10, min_dist // 2)
    for y in range(y_min_search, y_max_search):
        lo = max(y_min_search, y - win)
        hi = min(y_max_search, y + win + 1)
        if smooth[y] >= thresh and smooth[y] == np.max(smooth[lo:hi]):
            cands.append((float(smooth[y]), y))

    cands.sort(reverse=True)
    selected: list[int] = []
    for _, y in cands:
        if all(abs(y - sy) >= min_dist for sy in selected):
            selected.append(y)
    selected.sort()

    # Regularize grid & trim header/rubric lines above the uniform data-row grid
    diffs = [float(d) for d in np.diff(selected) if h * 0.040 <= d <= h * 0.095] if len(selected) >= 2 else []
    med_step = float(np.median(diffs)) if diffs else float(h * 0.060)

    # Trim leading header lines (e.g. top of column titles at ~0.25h or '1..24' strip at ~0.38h)
    # whose gap to the next line is much larger or smaller than the regular person-row height `med_step`
    while len(selected) >= 4:
        first_gap = selected[1] - selected[0]
        if 0.72 * med_step <= first_gap <= 1.28 * med_step:
            break
        selected.pop(0)

    if len(selected) < 5:
        start_y = selected[0] if selected else int(h * (0.26 if page_side != "back" else 0.16))
        n_rows = 11 if page_side != "back" else 12
        selected = [int(round(start_y + i * med_step)) for i in range(n_rows + 1) if start_y + i * med_step < h * 0.96]
    else:
        # Fill any missing interior horizontal line where gap >= 1.75 * med_step
        filled: list[int] = [selected[0]]
        for y_next in selected[1:]:
            gap = y_next - filled[-1]
            if gap >= 1.75 * med_step:
                n_missing = max(1, int(round(gap / med_step)) - 1)
                if n_missing <= 4:
                    step_fill = gap / float(n_missing + 1)
                    for k in range(1, n_missing + 1):
                        filled.append(int(round(filled[-1] + step_fill)))
            filled.append(y_next)
        selected = filled

    models = [_AngledLineModel(y_center=float(y), tan_a=tan_a, x_center=xc) for y in selected]
    meta = {
        "skew_left_deg": round(angle_deg, 2),
        "skew_right_deg": round(angle_deg, 2),
        "col_tilt_left_deg": 0.0,
        "col_tilt_right_deg": 0.0,
        "spine_offset_px": 0.0,
        "slicing_mode": f"census_24col_{page_side}",
    }
    return models, meta


def _row_has_handwriting(row_strip: np.ndarray) -> bool:
    """
    Checks whether a sliced census person row has actual handwritten/typed text
    specifically in the Name columns (x = 0.045w .. 0.255w: Priezvisko, Krstné meno, Pomer),
    suppressing empty pre-printed rows and rejecting rows that only have a dark stain/blot
    in the middle of the sheet.
    """
    rh, rw = row_strip.shape[:2]
    if rh < 12 or rw < 60:
        return False
    # Crop interior of Name columns (x = 0.045w .. 0.255w), avoiding top/bottom horizontal border rules
    y0 = max(2, int(rh * 0.18))
    y1 = max(y0 + 4, int(rh * 0.82))
    nx0 = int(rw * 0.045)
    nx1 = int(rw * 0.255)
    roi_slice = row_strip[y0:y1, nx0:nx1]
    if roi_slice.size == 0:
        return False
    clean_roi = suppress_auditor_crayon(roi_slice) if roi_slice.ndim == 3 else roi_slice
    gray_roi = cv2.cvtColor(clean_roi, cv2.COLOR_BGR2GRAY) if clean_roi.ndim == 3 else clean_roi
    name_roi = gray_roi.astype(np.float32)

    # Subtract column-wise median so pre-printed vertical grid rules do not trigger false positives
    col_bg = np.median(name_roi, axis=0, keepdims=True)
    residual = col_bg - name_roi
    ink_mask = (residual > 24.0).astype(np.uint8)
    name_ink_ratio = float(np.mean(ink_mask > 0))

    # Hard guard: if the Name columns are empty (< 0.007), reject even if there is a stain elsewhere
    if name_ink_ratio < 0.007:
        return False

    # Require horizontal stroke transitions typical of cursive handwriting (not a uniform grey blot)
    horiz_transitions = int(np.sum(np.abs(np.diff(ink_mask, axis=1)) > 0))
    min_transitions = max(10, int((y1 - y0) * 0.35))
    if name_ink_ratio >= 0.008 and horiz_transitions >= min_transitions:
        return True

    # Secondary check using connected components in Name columns
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    valid_stroke_blobs = 0
    for lbl in range(1, num_labels):
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        # Exclude full-height vertical grid line remnants or tiny 1-2px noise
        if area >= 8 and bw >= 2 and bh < int((y1 - y0) * 0.92):
            valid_stroke_blobs += 1
    return bool(name_ink_ratio >= 0.008 and valid_stroke_blobs >= 2)


def _overview_row_has_handwriting(row_strip: np.ndarray) -> bool:
    """
    Checks whether a sliced row on a 'Prehľad sčítacieho obvodu' (District Overview Sheet)
    contains handwritten/typed entries in the House Number / Settlement / Owner Name columns.
    """
    rh, rw = row_strip.shape[:2]
    if rh < 10 or rw < 60:
        return False
    clean_strip = suppress_auditor_crayon(row_strip) if row_strip.ndim == 3 else row_strip
    gray = cv2.cvtColor(clean_strip, cv2.COLOR_BGR2GRAY) if clean_strip.ndim == 3 else clean_strip
    y0 = max(2, int(rh * 0.16))
    y1 = max(y0 + 4, int(rh * 0.84))
    # Check across Cols 1-8 (0.04w .. 0.86w: Radové č., Osada, Číslo domu, Majiteľ domu, Počty)
    roi = gray[y0:y1, int(rw * 0.04) : int(rw * 0.86)].astype(np.float32)
    if roi.size == 0:
        return False
    col_bg = np.median(roi, axis=0, keepdims=True)
    residual = col_bg - roi
    ink_mask = (residual > 24.0).astype(np.uint8)
    ink_ratio = float(np.mean(ink_mask > 0))
    horiz_transitions = int(np.sum(np.abs(np.diff(ink_mask, axis=1)) > 0))
    if ink_ratio < 0.008 or horiz_transitions < max(10, int((y1 - y0) * 0.35)):
        return False

    # Also require handwriting strokes in the Owner Name / middle columns (0.35w .. 0.67w)
    # so blank 'Súhrn' / 'Spolu' summary rows with empty owner columns are suppressed
    c6_roi = gray[y0:y1, int(rw * 0.35) : int(rw * 0.67)].astype(np.float32)
    if c6_roi.size > 0:
        c6_ink = ((np.median(c6_roi, axis=0, keepdims=True) - c6_roi) > 24.0).astype(np.uint8)
        c6_ratio = float(np.mean(c6_ink > 0))
        c6_trans = int(np.sum(np.abs(np.diff(c6_ink, axis=1)) > 0))
        if c6_ratio < 0.016 or c6_trans < max(6, int((y1 - y0) * 0.16)):
            return False
    return True


def _detect_overview_subpage_lines(
    gray: np.ndarray,
    page_x0: int,
    page_x1: int,
    *,
    is_cover_page: bool,
) -> tuple[list[_AngledLineModel], float, np.ndarray, float]:
    """
    Detects horizontal row boundary lines for a single page or half-page of a
    'Prehľad sčítacieho obvodu' scan.
    """
    h, _ = gray.shape
    page_w = page_x1 - page_x0
    x0 = page_x0 + int(page_w * 0.04)
    x1 = page_x0 + int(page_w * 0.96)
    strip = gray[:, x0:x1]
    angle_deg, _ = _estimate_line_angles_hough(strip, max_abs_deg=3.5)
    tan_a = float(np.tan(np.deg2rad(angle_deg)))
    xc = page_x0 + 0.5 * page_w

    bg = float(np.percentile(strip, 78))
    inv_u8 = np.clip(bg - strip.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    inv_dil = cv2.dilate(inv_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    morph_w = max(45, (x1 - x0) // 30)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, 1))
    horiz_lines = cv2.morphologyEx(inv_dil, cv2.MORPH_OPEN, horiz_kernel).astype(np.float64)

    proj = _skew_compensated_row_projection(horiz_lines, x0, xc, tan_a, n_slices=12)
    smooth = np.convolve(proj, np.ones(7) / 7.0, mode="same")

    # Overview cover page (Page 1) has administrative header + column rubric down to ~0.24h..0.42h;
    # Continuation pages (Pages 2, 3, 4) have a shorter rubric header down to ~0.10h..0.22h.
    y_min_search = int(h * (0.24 if is_cover_page else 0.10))
    y_max_search = int(h * 0.94)

    raw_peaks = [
        y
        for y in range(y_min_search, y_max_search)
        if smooth[y] > 8.5 and smooth[y] == np.max(smooth[max(0, y - 20) : min(h, y + 21)])
    ]
    diffs = [float(d) for d in np.diff(raw_peaks) if h * 0.025 <= d <= h * 0.048] if len(raw_peaks) >= 2 else []
    med_step = float(np.median(diffs)) if diffs else float(h * 0.032)

    # Locate the strong bottom border of the column-number rubric band (1 2 3 4 5 6 7 8 9)
    start_y: int | None = None
    max_start_y = int(h * (0.48 if is_cover_page else 0.22))
    for y in raw_peaks:
        if smooth[y] >= 24.0 and y < max_start_y:
            if any(0.80 * med_step <= (y2 - y) <= 1.50 * med_step for y2 in raw_peaks):
                if start_y is None or smooth[y] >= smooth[start_y] * 0.72:
                    start_y = y
    if start_y is None and not is_cover_page:
        # Fallback for synthetic continuation pages with taller headers
        for y in raw_peaks:
            if smooth[y] >= 18.0 and y < int(h * 0.42):
                if any(0.80 * med_step <= (y2 - y) <= 1.50 * med_step for y2 in raw_peaks):
                    start_y = y
                    break
    if start_y is None:
        start_y = raw_peaks[0] if raw_peaks else int(h * (0.36 if is_cover_page else 0.12))

    selected: list[int] = [start_y]
    cur = start_y
    while True:
        lo = int(round(cur + (0.80 if len(selected) > 1 else 0.85) * med_step))
        hi = int(round(cur + (1.35 if len(selected) > 1 else 1.48) * med_step))
        if lo >= y_max_search:
            break
        sub = smooth[lo : min(y_max_search, hi + 1)]
        if sub.size == 0 or float(np.max(sub)) < 6.5:
            break
        best_y = lo + int(np.argmax(sub))
        selected.append(best_y)
        cur = best_y
        # Stop at the thick double border above 'Súhrn' on continuation pages
        if not is_cover_page and len(selected) >= 16 and float(smooth[best_y]) >= 55.0:
            break

    line_models = [_AngledLineModel(y_center=float(y), tan_a=tan_a, x_center=xc) for y in selected]
    return line_models, angle_deg, smooth, med_step


def slice_census_overview_sheet(
    image_path: Path,
    scan_id: int,
    crops_root: Path,
    *,
    page_side: str = "front",
    census_year: int = 1930,
) -> LayoutResult:
    """
    Slices a 'Prehľad sčítacieho obvodu' (front _v, back _r, or extra page_N) into
    header.jpg, per-row OCR strips (with auditor crayon suppressed), natural stitched crop,
    and overlay.jpg.
    Supports both 4-page folded bifolio spreads (w > 1.15 * h, where _v has Page 1 on right +
    Page 4 on left, and _r has Page 2 on left + Page 3 on right) and single-page scans.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read census overview image at {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scan_crop_dir = crops_root / f"scan_{scan_id}"
    scan_crop_dir.mkdir(parents=True, exist_ok=True)

    # Determine sub-pages to slice:
    # A 4-page folded bifolio spread (w > 1.15 * h) has two 9-column portrait pages side-by-side:
    # - On 'front' (_v): Right half is Booklet Page 1 (cover), Left half is Booklet Page 4 (back cover).
    # - On 'back' (_r): Left half is Booklet Page 2, Right half is Booklet Page 3.
    subpages: list[tuple[int, int, int, str, bool]] = []
    if w > int(h * 1.15):
        if page_side == "front":
            subpages = [
                (int(w * 0.495), w, 1, "right", True),
                (0, int(w * 0.505), 4, "left", False),
            ]
        else:
            p_idx = 2
            if "_" in page_side:
                tail = page_side.split("_", 1)[1]
                if tail.isdigit():
                    p_idx = max(2, int(tail) * 2 - 1)
            subpages = [
                (0, int(w * 0.505), p_idx, "left", False),
                (int(w * 0.495), w, p_idx + 1, "right", False),
            ]
    else:
        p_idx = 1 if page_side == "front" else 2
        if "_" in page_side:
            tail = page_side.split("_", 1)[1]
            if tail.isdigit():
                p_idx = int(tail)
        subpages = [(0, w, p_idx, "full", page_side == "front")]

    overlay = img.copy()
    rows_info: list[RowSliceInfo] = []
    out_row_idx = 0
    pad_y = max(8, int(h * 0.004))
    primary_header_y_end = int(h * 0.36)
    primary_header_rel = f"/media/crops/scan_{scan_id}/header.jpg"
    primary_angle_deg = 0.0

    def _process_subpages(subpage_specs: list[tuple[int, int, int, str, bool]]) -> int:
        nonlocal out_row_idx, primary_header_y_end, primary_angle_deg
        added_total = 0
        for sp_idx, (page_x0, page_x1, booklet_page, page_half, is_cover_page) in enumerate(subpage_specs):
            page_w = page_x1 - page_x0
            line_models, angle_deg, _smooth, med_step = _detect_overview_subpage_lines(
                gray,
                page_x0,
                page_x1,
                is_cover_page=is_cover_page,
            )
            if not line_models:
                continue
            first_line = line_models[0]
            split_x = page_x0 + int(page_w * 0.56)

            header_top = _AngledLineModel(
                y_center=max(16.0, first_line.y_center - max(h * (0.36 if is_cover_page else 0.14), 380.0)),
                tan_a=first_line.tan_a,
                x_center=first_line.x_center,
            )
            hdr_h = max(140, int(round(first_line.y_center - header_top.y_center + 24)))
            header_crop = _warp_piecewise_angled_strip(
                img, page_x0, page_x1, header_top, first_line, pad_y=12, out_h=hdr_h
            )
            if sp_idx == 0:
                header_fname = "header.jpg"
                primary_header_y_end = int(round(first_line.y_center))
                primary_angle_deg = angle_deg
            else:
                header_fname = f"header_{page_half}.jpg"
            header_file = scan_crop_dir / header_fname
            cv2.imwrite(str(header_file), suppress_auditor_crayon(header_crop), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            sub_header_rel = f"/media/crops/scan_{scan_id}/{header_fname}"

            candidate_strips: list[tuple[int, _AngledLineModel, _AngledLineModel, np.ndarray, np.ndarray, np.ndarray, bool]] = []
            consec_empty = 0
            any_kept_on_subpage = False
            for idx in range(1, len(line_models)):
                top_m = line_models[idx - 1]
                bot_m = line_models[idx]
                y0 = max(0, int(round(top_m.y_center)) - pad_y)
                y1 = min(h, max(y0 + 28, int(round(bot_m.y_center)) + pad_y))
                if abs(angle_deg) <= 0.25:
                    stitched_strip = img[y0:y1, page_x0:page_x1].copy()
                else:
                    row_out_h = max(28, int(round((bot_m.y_center - top_m.y_center) + 2 * pad_y)))
                    stitched_strip = _warp_piecewise_angled_strip(
                        img, page_x0, page_x1, top_m, bot_m, pad_y=pad_y, out_h=row_out_h
                    )

                has_ink = _overview_row_has_handwriting(stitched_strip)
                if booklet_page in (3, 4) or page_half == "full":
                    if has_ink:
                        # Stop before the 'Súhrn' settlement summary table or revisor signature block
                        if any_kept_on_subpage and consec_empty >= 2:
                            break
                        consec_empty = 0
                        any_kept_on_subpage = True
                    else:
                        consec_empty += 1
                        if not any_kept_on_subpage and booklet_page == 4 and consec_empty >= 2:
                            break
                        continue
                else:
                    if has_ink:
                        any_kept_on_subpage = True

                # Full active-page width strip for single-pass Table Recognition OCR, plus right half crop
                left_strip = stitched_strip.copy()
                # On continuation pages (Pages 2, 3, 4), Row 1 has a narrow 'Prenos' carry-over band
                # at the top of Columns 7-8 (0.68w..0.94w). Mask that top-right carry-over band on OCR strips
                # so carry-over totals do not overwrite Row 1's household/person counts.
                if not is_cover_page and idx == 1 and (bot_m.y_center - top_m.y_center) >= 1.20 * med_step:
                    rh_s, rw_s = left_strip.shape[:2]
                    paper_bgr = np.percentile(left_strip.reshape(-1, 3), 78, axis=0).astype(np.uint8)
                    left_strip[: int(rh_s * 0.36), int(rw_s * 0.68) : int(rw_s * 0.94)] = paper_bgr

                right_strip = left_strip[:, max(0, int(page_w * 0.34)) : page_w]
                candidate_strips.append((idx, top_m, bot_m, left_strip, right_strip, stitched_strip, has_ink))

            any_ink = any(item[6] for item in candidate_strips)
            if not any_ink and not (is_cover_page and sp_idx == 0 and len(subpage_specs) == 1):
                continue

            hdr_pts = np.array(
                [[kx, int(round(first_line.eval_y(kx)))] for kx in np.linspace(page_x0 + 10, page_x1 - 10, 9)],
                dtype=np.int32,
            )
            cv2.polylines(overlay, [hdr_pts], isClosed=False, color=(255, 90, 30), thickness=4)

            for orig_idx, top_m, bot_m, left_strip, right_strip, stitched_strip, has_ink in candidate_strips:
                if any_ink and not has_ink:
                    continue
                if not any_ink and orig_idx > 10:
                    continue
                out_row_idx += 1
                added_total += 1

                left_file = scan_crop_dir / f"row_{out_row_idx:02d}_left.jpg"
                right_file = scan_crop_dir / f"row_{out_row_idx:02d}_right.jpg"
                stitched_file = scan_crop_dir / f"row_{out_row_idx:02d}_stitched.jpg"

                left_ocr_img = suppress_auditor_crayon(left_strip)
                right_ocr_img = left_ocr_img[:, max(0, int(page_w * 0.34)) : page_w]
                cv2.imwrite(str(left_file), left_ocr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(str(right_file), right_ocr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                cv2.imwrite(str(stitched_file), stitched_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

                xs_full = [int(round(v)) for v in np.linspace(page_x0 + 10, page_x1 - 10, 9)]
                poly = np.array(
                    [[x, int(round(top_m.eval_y(x)))] for x in xs_full]
                    + [[x, int(round(bot_m.eval_y(x)))] for x in reversed(xs_full)],
                    dtype=np.int32,
                )
                cv2.polylines(overlay, [poly], isClosed=True, color=(60, 210, 60), thickness=3)
                label_x = page_x0 + 28
                label_y = int(round(top_m.eval_y(label_x))) + 34
                cv2.putText(
                    overlay,
                    f"H{out_row_idx}",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (20, 160, 20),
                    2,
                )

                left_quad = [
                    [page_x0, int(round(top_m.eval_y(page_x0)))],
                    [split_x, int(round(top_m.eval_y(split_x)))],
                    [split_x, int(round(bot_m.eval_y(split_x)))],
                    [page_x0, int(round(bot_m.eval_y(page_x0)))],
                ]
                right_quad = [
                    [split_x, int(round(top_m.eval_y(split_x)))],
                    [page_x1, int(round(top_m.eval_y(page_x1)))],
                    [page_x1, int(round(bot_m.eval_y(page_x1)))],
                    [split_x, int(round(bot_m.eval_y(split_x)))],
                ]

                rows_info.append(
                    RowSliceInfo(
                        row_index=out_row_idx,
                        y_start=int(round(top_m.y_center)),
                        y_end=int(round(bot_m.y_center)),
                        left_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_left.jpg",
                        right_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_right.jpg",
                        stitched_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_stitched.jpg",
                        cell_crops=[
                            {
                                "key": "overview_page",
                                "booklet_page": booklet_page,
                                "page_half": page_half,
                                "page_x0": page_x0,
                                "page_x1": page_x1,
                                "header_crop_url": sub_header_rel,
                            }
                        ],
                        angle_left_deg=round(top_m.angle_deg, 2),
                        angle_right_deg=round(top_m.angle_deg, 2),
                        left_quad=left_quad,
                        right_quad=right_quad,
                    )
                )
        return added_total

    added = _process_subpages(subpages)
    # Fallback for full-width synthetic overview test images where w > 1.15 * h but a single table spans 0..w
    if added == 0 and len(subpages) > 1:
        fallback_p = 1 if page_side == "front" else 2
        _process_subpages([(0, w, fallback_p, "full", page_side == "front")])

    overlay_scale = min(1.0, 2200.0 / float(w))
    overlay_small = (
        cv2.resize(overlay, (int(w * overlay_scale), int(h * overlay_scale)))
        if overlay_scale < 1.0
        else overlay
    )
    overlay_file = scan_crop_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_file), overlay_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return LayoutResult(
        width=w,
        height=h,
        spine_x=0,
        skew_left_deg=round(primary_angle_deg, 2),
        skew_right_deg=round(primary_angle_deg, 2),
        header_y_end=primary_header_y_end,
        header_crop_rel=primary_header_rel,
        overlay_rel=f"/media/crops/scan_{scan_id}/overlay.jpg",
        vertical_lines_x=[],
        rows=rows_info,
        col_tilt_left_deg=0.0,
        col_tilt_right_deg=0.0,
        spine_offset_px=0.0,
        slicing_mode=f"census_overview_{page_side}",
    )


def slice_census_sheet(
    image_path: Path,
    scan_id: int,
    crops_root: Path,
    *,
    page_side: str = "front",
    census_year: int = 1930,
    is_blackened: bool = False,
    is_overview: bool = False,
) -> LayoutResult:
    """
    Slices a 1930/1940 Slovakiana census sheet scan into:
    - `header.jpg`: Top administrative address boxes + 24-column rubric header
    - Per-row `left` (Cols 1-10), `right` (Cols 11-24), `stitched`, and 5 semantic `cell_crops`
    - `overlay.jpg`: Visual grid inspection overlay
    """
    if is_overview:
        return slice_census_overview_sheet(
            image_path,
            scan_id,
            crops_root,
            page_side=page_side,
            census_year=census_year,
        )

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read census image at {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    line_models, angle_meta = _detect_census_row_boundaries(gray, page_side=page_side)
    first_line = line_models[0]
    last_line = line_models[-1]

    header_y_end = int(round(first_line.y_center))
    # Census sheets are flat loose sheets without a book spine (spine_x = 0).
    # Use split_x at Col 10/11 boundary (0.375 * w) purely to split wide 5000px rows into left/right OCR strips.
    split_x = int(w * 0.375)

    scan_crop_dir = crops_root / f"scan_{scan_id}"
    scan_crop_dir.mkdir(parents=True, exist_ok=True)

    # Save full administrative + rubric header band above Row 1
    header_top = _AngledLineModel(
        y_center=max(8.0, first_line.y_center - max(h * 0.25, 420.0)),
        tan_a=first_line.tan_a,
        x_center=first_line.x_center,
    )
    hdr_h = max(140, int(round(first_line.y_center - header_top.y_center + 24)))
    header_crop = _warp_piecewise_angled_strip(img, 0, w, header_top, first_line, pad_y=12, out_h=hdr_h)
    header_file = scan_crop_dir / "header.jpg"
    cv2.imwrite(str(header_file), suppress_auditor_crayon(header_crop), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    header_rel = f"/media/crops/scan_{scan_id}/header.jpg"

    # Approximate 24-column x positions for subtle visual column guides (no fake spine line!)
    col_fractions = [
        0.025, 0.048, 0.145, 0.215, 0.255, 0.280, 0.330, 0.355,
        0.365, 0.375, 0.465, 0.505, 0.535, 0.565, 0.610, 0.655,
        0.690, 0.765, 0.815, 0.885, 0.915, 0.950,
    ]
    vert_cols_x = [int(round(f * w)) for f in col_fractions]

    overlay = img.copy()
    for vx in vert_cols_x:
        y_t = int(round(first_line.eval_y(vx)))
        y_b = int(round(last_line.eval_y(vx)))
        cv2.line(overlay, (vx, y_t), (vx, y_b), (0, 180, 255), 1)

    hdr_pts = np.array(
        [[kx, int(round(first_line.eval_y(kx)))] for kx in np.linspace(10, w - 10, 9)],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [hdr_pts], isClosed=False, color=(255, 90, 30), thickness=4)

    rows_info: list[RowSliceInfo] = []
    pad_y = max(10, int(h * 0.005))

    candidate_strips: list[tuple[int, _AngledLineModel, _AngledLineModel, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    for idx in range(1, len(line_models)):
        top_m = line_models[idx - 1]
        bot_m = line_models[idx]
        y0 = max(0, int(round(top_m.y_center)) - pad_y)
        y1 = min(h, max(y0 + 32, int(round(bot_m.y_center)) + pad_y))
        if abs(angle_meta["skew_left_deg"]) <= 0.25:
            stitched_strip = img[y0:y1, :].copy()
        else:
            row_out_h = max(32, int(round((bot_m.y_center - top_m.y_center) + 2 * pad_y)))
            stitched_strip = _warp_piecewise_angled_strip(
                img, 0, w, top_m, bot_m, pad_y=pad_y, out_h=row_out_h
            )
        left_strip = stitched_strip[:, :split_x]
        right_strip = stitched_strip[:, split_x:]
        has_ink = _row_has_handwriting(stitched_strip)
        candidate_strips.append((idx, top_m, bot_m, left_strip, right_strip, stitched_strip, has_ink))

    # Keep all rows with handwriting in the Name columns; if none pass, keep at least first 3 rows
    any_ink = any(item[6] for item in candidate_strips)
    out_row_idx = 0

    for orig_idx, top_m, bot_m, left_strip, right_strip, stitched_strip, has_ink in candidate_strips:
        if any_ink and not has_ink:
            continue
        if not any_ink and orig_idx > 3:
            continue
        out_row_idx += 1

        left_file = scan_crop_dir / f"row_{out_row_idx:02d}_left.jpg"
        right_file = scan_crop_dir / f"row_{out_row_idx:02d}_right.jpg"
        stitched_file = scan_crop_dir / f"row_{out_row_idx:02d}_stitched.jpg"

        # Apply auditor crayon suppression once to stitched_strip and slice left/right OCR strips
        stitched_ocr_img = suppress_auditor_crayon(stitched_strip)
        left_ocr_img = stitched_ocr_img[:, :split_x]
        right_ocr_img = stitched_ocr_img[:, split_x:]

        cv2.imwrite(str(left_file), left_ocr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(right_file), right_ocr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(stitched_file), stitched_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # 5 semantic column-group crops tuned to exact 1930/1940 census sheet columns
        cell_regions = [
            ("identity_name", int(w * 0.01), int(w * 0.255), "Priezvisko, meno a pomer k hlave (stĺpce 1–5)"),
            ("birth_marital", int(w * 0.245), int(w * 0.375), "Dátum narodenia, stav a deti (stĺpce 6–10)"),
            ("origin_domicile", int(w * 0.365), int(w * 0.565), "Rodisko, prisťahovanie a príslušnosť (stĺpce 11–14)"),
            ("nationality_religion_literacy", int(w * 0.555), int(w * 0.690), "Národnosť, náboženstvo a gramotnosť (stĺpce 15–17)"),
            ("occupation_presence_notes", int(w * 0.680), int(w * 0.990), "Povolanie, závod a poznámky (stĺpce 18–24)"),
        ]
        cell_crops_list: list[dict[str, Any]] = []
        for cell_key, cx0, cx1, label_sk in cell_regions:
            cell_img = stitched_ocr_img[:, max(0, cx0) : min(w, cx1)]
            c_file = scan_crop_dir / f"row_{out_row_idx:02d}_{cell_key}.jpg"
            cv2.imwrite(str(c_file), cell_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cell_crops_list.append(
                {
                    "key": cell_key,
                    "label_sk": label_sk,
                    "x0": cx0,
                    "x1": cx1,
                    "crop_url": f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_{cell_key}.jpg",
                }
            )

        xs_full = [int(round(v)) for v in np.linspace(10, w - 10, 9)]
        poly = np.array(
            [[x, int(round(top_m.eval_y(x)))] for x in xs_full]
            + [[x, int(round(bot_m.eval_y(x)))] for x in reversed(xs_full)],
            dtype=np.int32,
        )
        cv2.polylines(overlay, [poly], isClosed=True, color=(60, 210, 60), thickness=3)
        label_y = int(round(top_m.eval_y(28))) + 42
        cv2.putText(
            overlay,
            f"R{out_row_idx}",
            (28, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (20, 160, 20),
            3,
        )

        left_quad = [
            [0, int(round(top_m.eval_y(0)))],
            [split_x, int(round(top_m.eval_y(split_x)))],
            [split_x, int(round(bot_m.eval_y(split_x)))],
            [0, int(round(bot_m.eval_y(0)))],
        ]
        right_quad = [
            [split_x, int(round(top_m.eval_y(split_x)))],
            [w, int(round(top_m.eval_y(w)))],
            [w, int(round(bot_m.eval_y(w)))],
            [split_x, int(round(bot_m.eval_y(split_x)))],
        ]

        rows_info.append(
            RowSliceInfo(
                row_index=out_row_idx,
                y_start=int(round(top_m.y_center)),
                y_end=int(round(bot_m.y_center)),
                left_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_left.jpg",
                right_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_right.jpg",
                stitched_crop_rel=f"/media/crops/scan_{scan_id}/row_{out_row_idx:02d}_stitched.jpg",
                cell_crops=cell_crops_list,
                angle_left_deg=round(top_m.angle_deg, 2),
                angle_right_deg=round(top_m.angle_deg, 2),
                left_quad=left_quad,
                right_quad=right_quad,
            )
        )

    overlay_scale = min(1.0, 2200.0 / float(w))
    overlay_small = (
        cv2.resize(overlay, (int(w * overlay_scale), int(h * overlay_scale)))
        if overlay_scale < 1.0
        else overlay
    )
    overlay_file = scan_crop_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_file), overlay_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return LayoutResult(
        width=w,
        height=h,
        spine_x=0,
        skew_left_deg=angle_meta["skew_left_deg"],
        skew_right_deg=angle_meta["skew_right_deg"],
        header_y_end=header_y_end,
        header_crop_rel=header_rel,
        overlay_rel=f"/media/crops/scan_{scan_id}/overlay.jpg",
        vertical_lines_x=vert_cols_x,
        rows=rows_info,
        col_tilt_left_deg=0.0,
        col_tilt_right_deg=0.0,
        spine_offset_px=0.0,
        slicing_mode=angle_meta["slicing_mode"],
    )
