from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class RowSliceInfo:
    row_index: int
    y_start: int
    y_end: int
    left_crop_rel: str
    right_crop_rel: str
    stitched_crop_rel: str
    cell_crops: list[dict[str, Any]]
    angle_left_deg: float = 0.0
    angle_right_deg: float = 0.0
    left_quad: list[list[int]] = field(default_factory=list)
    right_quad: list[list[int]] = field(default_factory=list)


@dataclass
class LayoutResult:
    width: int
    height: int
    spine_x: int
    skew_left_deg: float
    skew_right_deg: float
    header_y_end: int
    header_crop_rel: str
    overlay_rel: str
    vertical_lines_x: list[int]
    rows: list[RowSliceInfo]
    col_tilt_left_deg: float = 0.0
    col_tilt_right_deg: float = 0.0
    spine_offset_px: float = 0.0
    slicing_mode: str = "angled_quad"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class _AngledLineModel:
    """Represents an angled row boundary y(x) = y_center + tan_a * (x - x_center) plus 5 piecewise knots."""

    y_center: float
    tan_a: float
    x_center: float
    knot_xs: list[int] = field(default_factory=list)
    knot_ys: list[float] = field(default_factory=list)

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan(self.tan_a)))

    def eval_y(self, x: float) -> float:
        if len(self.knot_xs) >= 2 and len(self.knot_xs) == len(self.knot_ys):
            return float(np.interp(x, self.knot_xs, self.knot_ys))
        return float(self.y_center + self.tan_a * (x - self.x_center))

    def eval_y_linear(self, x: float) -> float:
        return float(self.y_center + self.tan_a * (x - self.x_center))


def find_spine_x(gray: np.ndarray) -> int:
    """
    Detects the vertical book gutter/spine seam in the middle band (0.35w .. 0.65w).
    If there is no prominent dark vertical gutter/fold shadow compared to the surrounding
    paper (e.g. a flat single-page sheet without a real book spine), returns 0 instead
    of picking an arbitrary table column rule.
    """
    h, w = gray.shape
    search_l = int(w * 0.35)
    search_r = int(w * 0.65)
    strip = gray[int(h * 0.08) : int(h * 0.92), search_l:search_r]
    if strip.size == 0:
        return 0
    col_mean = np.mean(strip, axis=0)

    # Check gutter-width vertical darkness contrast against surrounding paper
    k_gutter = max(31, int(w * 0.012) | 1)
    padded_g = np.pad(col_mean, (k_gutter // 2, k_gutter // 2), mode="edge")
    smooth_g = np.convolve(padded_g, np.ones(k_gutter) / float(k_gutter), mode="valid")
    paper_level = float(np.percentile(smooth_g, 75))
    min_level = float(np.min(smooth_g))
    contrast_drop = paper_level - min_level
    rel_drop = contrast_drop / max(1.0, paper_level)
    if contrast_drop < 18.0 or rel_drop < 0.09:
        return 0

    k = 15
    padded = np.pad(col_mean, (k // 2, k // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(k) / float(k), mode="valid")
    xs = np.arange(len(smooth), dtype=np.float64) + search_l
    centrality_penalty = 0.08 * (np.abs(xs - w * 0.5) / max(1.0, w * 0.08)) * float(np.median(smooth))
    min_offset = int(np.argmin(smooth + centrality_penalty))
    return search_l + min_offset


_detect_spine_x = find_spine_x


def _build_horizontal_feature_map(gray: np.ndarray) -> np.ndarray:
    """
    Extracts thin dark horizontal rule lines while suppressing paper-to-scanner step edges
    and preserving tilted lines up to +-8 degrees.
    """
    bh = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19)),
    )
    _, bw = cv2.threshold(bh, 22, 255, cv2.THRESH_BINARY)
    bright_mask = (cv2.blur(gray, (25, 25)) > 85).astype(np.uint8)
    bw = bw * bright_mask
    # Short horizontal kernel (25px) removes vertical rules & isolated characters while keeping tilted lines intact
    hl = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
    )
    return cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))


def _detect_vertical_column_angle_prior(gray: np.ndarray, x0: int, x1: int) -> float:
    """Estimates page rotation/tilt from vertical column dividers (immune to cross-row harmonic aliasing)."""
    h, _ = gray.shape
    y0, y1 = int(h * 0.12), int(h * 0.90)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < 60 or roi.shape[1] < 60:
        return 0.0
    sh, sw = max(30, roi.shape[0] // 2), max(30, roi.shape[1] // 2)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    bh = cv2.morphologyEx(
        small,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)),
    )
    vert_e = cv2.morphologyEx(
        bh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
    )
    n_bands = 14
    b_edges = np.linspace(0, sh, n_bands + 1, dtype=int)
    b_profs = []
    b_ys = []
    for i in range(n_bands):
        by0, by1 = b_edges[i], b_edges[i + 1]
        if by1 > by0:
            b_profs.append(np.mean(vert_e[by0:by1, :], axis=0))
            b_ys.append(0.5 * (by0 + by1) - sh / 2.0)
    if not b_profs:
        return 0.0
    b_profs_arr = np.array(b_profs, dtype=np.float32)
    b_ys_arr = np.array(b_ys, dtype=np.float32)

    try:
        from backend.app.pipeline.lexicon import _rust_lex

        if _rust_lex is not None and hasattr(_rust_lex, "sheared_radon_best_angle_rs"):
            best_a_rs, _ = _rust_lex.sheared_radon_best_angle_rs(
                b_profs_arr.ravel().tolist(),
                len(b_ys_arr),
                sw,
                (-b_ys_arr).tolist(),
                -7.5,
                7.5,
                151,
                0,
                sw,
            )
            return float(best_a_rs)
    except Exception:
        pass

    best_a = 0.0
    best_sc = -1.0
    for a in np.linspace(-7.5, 7.5, 151):
        tan_a = np.tan(np.radians(a))
        acc = np.zeros(sw, dtype=np.float32)
        for i in range(len(b_ys_arr)):
            shift = int(round(-b_ys_arr[i] * tan_a))
            if shift == 0:
                acc += b_profs_arr[i]
            elif 0 < shift < sw:
                acc[:-shift] += b_profs_arr[i, shift:]
            elif -sw < shift < 0:
                acc[-shift:] += b_profs_arr[i, : sw + shift]
        diff = np.diff(acc)
        sc = float(np.sum(diff * diff))
        if sc > best_sc:
            best_sc = sc
            best_a = float(a)
    return best_a


def _sheared_projection(
    profs: np.ndarray,
    xs: np.ndarray,
    tan_a: float,
    h: int,
) -> np.ndarray:
    """Computes the angle-compensated horizontal projection profile at slope tan_a."""
    n_strips = len(xs)
    acc = np.zeros(h, dtype=np.float32)
    for i in range(n_strips):
        sh_y = int(round(xs[i] * tan_a))
        if sh_y == 0:
            acc += profs[i]
        elif 0 < sh_y < h:
            acc[:-sh_y] += profs[i, sh_y:]
        elif -h < sh_y < 0:
            acc[-sh_y:] += profs[i, : h + sh_y]
    return acc / max(1, n_strips)


def _analyze_half_page_candidates(
    gray: np.ndarray,
    hl: np.ndarray,
    x0: int,
    x1: int,
) -> dict[str, Any]:
    """
    Finds the dominant row tilt angle on one half-page via sheared Radon projection
    and extracts strength-sorted Non-Maximum Suppressed row boundary candidates.
    """
    h, _ = gray.shape
    sub = hl[:, x0:x1]
    sw = x1 - x0
    xc = 0.5 * (x0 + x1)
    n_strips = 16
    edges = np.linspace(0, sw, n_strips + 1, dtype=int)
    profs = np.array(
        [np.mean(sub[:, edges[i] : edges[i + 1]], axis=1) for i in range(n_strips)],
        dtype=np.float32,
    )
    xs = np.array(
        [0.5 * (edges[i] + edges[i + 1]) - sw / 2.0 for i in range(n_strips)],
        dtype=np.float32,
    )

    col_prior = _detect_vertical_column_angle_prior(gray, x0, x1)
    search_min = max(-7.5, col_prior - 4.5)
    search_max = min(7.5, col_prior + 4.5)

    by0, by1 = int(h * 0.12), int(h * 0.96)
    best_a = 0.0
    acc: np.ndarray | None = None
    try:
        from backend.app.pipeline.lexicon import _rust_lex

        if _rust_lex is not None and hasattr(_rust_lex, "sheared_radon_best_angle_rs"):
            best_a_rs, acc_list = _rust_lex.sheared_radon_best_angle_rs(
                profs.ravel().tolist(),
                n_strips,
                h,
                xs.tolist(),
                float(search_min),
                float(search_max),
                141,
                by0,
                by1,
            )
            best_a = float(best_a_rs)
            acc = np.array(acc_list, dtype=np.float32)
    except Exception:
        acc = None

    if acc is None:
        best_sc = -1.0
        for a in np.linspace(search_min, search_max, 141):
            tan_a_cand = float(np.tan(np.radians(a)))
            acc_cand = _sheared_projection(profs, xs, tan_a_cand, h)
            diff = np.diff(acc_cand[by0:by1])
            sc = float(np.sum(diff * diff))
            if sc > best_sc:
                best_sc = sc
                best_a = float(a)
        tan_a = float(np.tan(np.radians(best_a)))
        acc = _sheared_projection(profs, xs, tan_a, h)
    else:
        tan_a = float(np.tan(np.radians(best_a)))

    smooth = np.convolve(acc, np.ones(11) / 11.0, mode="same")
    thresh = max(float(np.percentile(smooth[by0:by1], 80)), 14.0)

    cands: list[tuple[float, int]] = []
    for y in range(int(h * 0.13), int(h * 0.985)):
        if smooth[y] >= thresh and smooth[y] == np.max(smooth[max(0, y - 25) : min(h, y + 26)]):
            cands.append((float(smooth[y]), y))
    cands.sort(reverse=True)
    min_dist = int(h * 0.052)
    selected: list[int] = []
    for _, y in cands:
        if all(abs(y - sy) >= min_dist for sy in selected):
            selected.append(y)
    selected.sort()

    return {
        "angle": best_a,
        "tan_a": tan_a,
        "col_prior": col_prior,
        "xc": xc,
        "x0": x0,
        "x1": x1,
        "edges": edges,
        "profs": profs,
        "xs": xs,
        "smooth": smooth,
        "thresh": thresh,
        "selected": selected,
    }


def _fit_angled_line_model(
    yc: float,
    page_info: dict[str, Any],
    x_bound_0: int,
    x_bound_1: int,
    h: int,
) -> _AngledLineModel:
    """
    Refines the local slope and 5 piecewise knots of a single row boundary line across
    the vertical strips of a half-page, capturing both per-row tilt and gentle spine curvature.
    """
    profs: np.ndarray = page_info["profs"]
    xs: np.ndarray = page_info["xs"]
    edges: np.ndarray = page_info["edges"]
    x0: int = page_info["x0"]
    xc: float = page_info["xc"]
    tan_page: float = page_info["tan_a"]
    thresh: float = page_info["thresh"]
    n_strips = len(xs)

    pts_x: list[float] = []
    pts_y: list[float] = []
    pts_w: list[float] = []

    for i in range(n_strips):
        exp_y = int(round(yc + xs[i] * tan_page))
        w0 = max(0, exp_y - 24)
        w1 = min(h, exp_y + 25)
        if w1 > w0:
            loc = profs[i, w0:w1]
            pk = int(np.argmax(loc))
            pk_val = float(loc[pk])
            if pk_val >= max(8.0, thresh * 0.55):
                pts_x.append(float(x0 + 0.5 * (edges[i] + edges[i + 1])))
                pts_y.append(float(w0 + pk))
                pts_w.append(pk_val)

    if len(pts_x) >= 4:
        px = np.array(pts_x, dtype=np.float64) - xc
        py = np.array(pts_y, dtype=np.float64)
        pw = np.array(pts_w, dtype=np.float64)
        m, b = np.polyfit(px, py, 1, w=pw)
        res_err = np.abs(py - (m * px + b))
        inliers = res_err < 12.0
        if int(np.sum(inliers)) >= 4:
            m, b = np.polyfit(px[inliers], py[inliers], 1, w=pw[inliers])
        max_dm = float(np.tan(np.radians(1.6)))
        m = float(np.clip(m, tan_page - max_dm, tan_page + max_dm))
        b = float(np.clip(b, yc - 16.0, yc + 16.0))
    else:
        m = float(tan_page)
        b = float(yc)

    knot_xs = [int(round(v)) for v in np.linspace(x_bound_0, x_bound_1, 5)]
    knot_ys = [float(b + m * (kx - xc)) for kx in knot_xs]

    # Gentle quadratic refinement for book gutter curvature when strong strip inliers exist
    if len(pts_x) >= 7:
        px = np.array(pts_x, dtype=np.float64) - xc
        py = np.array(pts_y, dtype=np.float64)
        pw = np.array(pts_w, dtype=np.float64)
        lin_pred = b + m * px
        inliers = np.abs(py - lin_pred) < 11.0
        if int(np.sum(inliers)) >= 6:
            try:
                q_coeffs = np.polyfit(px[inliers], py[inliers], 2, w=pw[inliers])
                refined_knots: list[float] = []
                for idx_k, kx in enumerate(knot_xs):
                    lin_y = knot_ys[idx_k]
                    quad_y = float(np.polyval(q_coeffs, kx - xc))
                    blended = 0.65 * lin_y + 0.35 * quad_y
                    refined_knots.append(float(np.clip(blended, lin_y - 10.0, lin_y + 10.0)))
                knot_ys = refined_knots
            except Exception:
                pass

    return _AngledLineModel(
        y_center=b,
        tan_a=m,
        x_center=xc,
        knot_xs=knot_xs,
        knot_ys=knot_ys,
    )


def _detect_angled_row_boundaries(
    gray: np.ndarray,
    spine_x: int,
) -> tuple[list[tuple[_AngledLineModel, _AngledLineModel]], dict[str, Any]]:
    """
    Detects paired (left_line_model, right_line_model) boundaries for all rows across the spread.
    Handles both ruled tables and unruled handwritten registers under arbitrary page tilt angles.
    """
    h, w = gray.shape
    hl = _build_horizontal_feature_map(gray)

    L = _analyze_half_page_candidates(gray, hl, int(spine_x * 0.05), int(spine_x * 0.96))
    R = _analyze_half_page_candidates(
        gray,
        hl,
        int(spine_x + (w - spine_x) * 0.04),
        int(spine_x + (w - spine_x) * 0.95),
    )

    if len(L["selected"]) < 5 and len(R["selected"]) < 5:
        slicing_mode = "angled_valley_quad"
        hdr_l = int(h * 0.08) + int(np.argmax(L["smooth"][int(h * 0.08) : int(h * 0.18)]))
        hdr_r = int(h * 0.08) + int(np.argmax(R["smooth"][int(h * 0.08) : int(h * 0.18)]))
        inv_ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        inv_ink = cv2.morphologyEx(
            inv_ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        # Compute sheared ink profile along left page angle to find angled inter-row valleys
        sub_ink = inv_ink[:, L["x0"] : L["x1"]]
        edges = L["edges"]
        ink_profs = np.array(
            [np.mean(sub_ink[:, edges[i] : edges[i + 1]], axis=1) for i in range(len(L["xs"]))],
            dtype=np.float32,
        )
        sheared_ink = _sheared_projection(ink_profs, L["xs"], L["tan_a"], h)
        smooth_ink = np.convolve(sheared_ink, np.ones(35) / 35.0, mode="same")

        valleys = [hdr_l]
        min_v_step = int(h * 0.06)
        for y in range(hdr_l + min_v_step, int(h * 0.93)):
            if smooth_ink[y] == np.min(smooth_ink[max(hdr_l, y - 85) : min(h, y + 86)]):
                if y - valleys[-1] >= min_v_step:
                    valleys.append(y)
        if len(valleys) < 4:
            valleys = [int(round(v)) for v in np.linspace(hdr_l, int(h * 0.92), 9)]
        med_offset = float(hdr_r - hdr_l)
        final_pairs = [(y, int(round(y + med_offset))) for y in valleys]
    else:
        slicing_mode = "angled_ruled_quad"
        all_diffs: list[float] = []
        for seq in (L["selected"], R["selected"]):
            if len(seq) >= 2:
                for d in np.diff(seq):
                    if h * 0.05 <= d <= h * 0.16:
                        all_diffs.append(float(d))
        med_step = float(np.median(all_diffs)) if all_diffs else float(h * 0.08)

        for D in (L, R):
            while (
                len(D["selected"]) >= 3
                and D["selected"][0] < h * 0.19
                and D["selected"][1] < h * 0.25
            ):
                gap0 = D["selected"][1] - D["selected"][0]
                if D["selected"][0] < h * 0.135 or gap0 < 0.78 * med_step:
                    D["selected"].pop(0)
                else:
                    break

        l_sp = [y + (spine_x - L["xc"]) * L["tan_a"] for y in L["selected"]]
        r_sp = [y + (spine_x - R["xc"]) * R["tan_a"] for y in R["selected"]]

        used_r: set[int] = set()
        pairs: list[tuple[int | None, int | None, float]] = []
        offsets: list[float] = []
        for i, ly_s in enumerate(l_sp):
            best_j = None
            best_d = 0.42 * med_step
            for j, ry_s in enumerate(r_sp):
                if j in used_r:
                    continue
                d = abs(ly_s - ry_s)
                if d <= best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                used_r.add(best_j)
                offsets.append(r_sp[best_j] - ly_s)
                pairs.append((L["selected"][i], R["selected"][best_j], 0.5 * (ly_s + r_sp[best_j])))
            else:
                pairs.append((L["selected"][i], None, ly_s))

        for j, ry_s in enumerate(r_sp):
            if j not in used_r:
                pairs.append((None, R["selected"][j], ry_s))

        med_offset = float(np.clip(np.median(offsets), -45.0, 45.0)) if offsets else 0.0
        pairs.sort(key=lambda x: x[2])

        first_both_idx = next(
            (idx for idx, p in enumerate(pairs) if p[0] is not None and p[1] is not None),
            0,
        )
        cleaned: list[tuple[int | None, int | None, float]] = []
        for idx, (yl, yr, sp_y) in enumerate(pairs):
            if idx < first_both_idx and (yl is None or yr is None) and sp_y < h * 0.22:
                continue
            if cleaned and (sp_y - cleaned[-1][2]) < 0.62 * med_step:
                prev_both = cleaned[-1][0] is not None and cleaned[-1][1] is not None
                curr_both = yl is not None and yr is not None
                if curr_both and not prev_both:
                    cleaned[-1] = (yl, yr, sp_y)
                continue
            cleaned.append((yl, yr, sp_y))

        resolved: list[tuple[int, int]] = []
        for yl, yr, _ in cleaned:
            if yl is None and yr is not None:
                ly_s = (yr + (spine_x - R["xc"]) * R["tan_a"]) - med_offset
                yl = int(round(ly_s - (spine_x - L["xc"]) * L["tan_a"]))
            if yr is None and yl is not None:
                ry_s = (yl + (spine_x - L["xc"]) * L["tan_a"]) + med_offset
                yr = int(round(ry_s - (spine_x - R["xc"]) * R["tan_a"]))
            if yl is not None and yr is not None:
                resolved.append((yl, yr))

        final_pairs = []
        for idx in range(len(resolved)):
            final_pairs.append(resolved[idx])
            if idx < len(resolved) - 1:
                yl0, yr0 = resolved[idx]
                yl1, yr1 = resolved[idx + 1]
                gap = 0.5 * ((yl1 - yl0) + (yr1 - yr0))
                if gap > 1.62 * med_step:
                    n_sp = max(2, int(round(gap / med_step)))
                    for s in range(1, n_sp):
                        frac = s / float(n_sp)
                        iyi_l = int(round(yl0 + frac * (yl1 - yl0)))
                        iyi_r = int(round(yr0 + frac * (yr1 - yr0)))
                        final_pairs.append((iyi_l, iyi_r))

    boundary_models: list[tuple[_AngledLineModel, _AngledLineModel]] = []
    for yl, yr in final_pairs:
        lm = _fit_angled_line_model(float(yl), L, 0, spine_x, h)
        rm = _fit_angled_line_model(float(yr), R, spine_x, w, h)
        boundary_models.append((lm, rm))

    meta = {
        "skew_left_deg": round(float(L["angle"]), 2),
        "skew_right_deg": round(float(R["angle"]), 2),
        "col_tilt_left_deg": round(float(L["col_prior"]), 2),
        "col_tilt_right_deg": round(float(R["col_prior"]), 2),
        "spine_offset_px": round(float(med_offset), 1),
        "slicing_mode": slicing_mode,
    }
    return boundary_models, meta


def _detect_angled_vertical_columns(
    gray: np.ndarray,
    spine_x: int,
    header_y: int,
    bottom_y: int,
) -> list[dict[str, Any]]:
    """Detects vertical column divider lines along with their tilt angle on both pages."""
    h, w = gray.shape
    bh = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1)),
    )
    _, bw = cv2.threshold(bh, 22, 255, cv2.THRESH_BINARY)
    bright_mask = (cv2.blur(gray, (25, 25)) > 85).astype(np.uint8)
    bw = bw * bright_mask
    vl = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)),
    )
    vl = cv2.dilate(vl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

    y0 = max(int(h * 0.10), min(header_y, int(h * 0.30)))
    y1 = min(int(h * 0.95), max(bottom_y, int(h * 0.70)))
    yc = 0.5 * (y0 + y1)
    n_bands = 12
    b_edges = np.linspace(y0, y1, n_bands + 1, dtype=int)
    b_ys = np.array([0.5 * (b_edges[i] + b_edges[i + 1]) - yc for i in range(n_bands)], dtype=np.float32)

    cols_out: list[dict[str, Any]] = []
    for side, x_start, x_end in [
        ("L", int(w * 0.02), max(int(w * 0.05), spine_x - 15)),
        ("R", min(w - 20, spine_x + 15), int(w * 0.98)),
    ]:
        sw = x_end - x_start
        if sw <= 40:
            continue
        b_profs = np.array(
            [np.mean(vl[b_edges[i] : b_edges[i + 1], x_start:x_end], axis=0) for i in range(n_bands)],
            dtype=np.float32,
        )
        best_a = 0.0
        acc: np.ndarray | None = None
        try:
            from backend.app.pipeline.lexicon import _rust_lex

            if _rust_lex is not None and hasattr(_rust_lex, "sheared_radon_best_angle_rs"):
                best_a_rs, acc_list = _rust_lex.sheared_radon_best_angle_rs(
                    b_profs.ravel().tolist(),
                    n_bands,
                    sw,
                    b_ys.tolist(),
                    -6.5,
                    6.5,
                    131,
                    0,
                    sw,
                )
                best_a = float(best_a_rs)
                acc = np.array(acc_list, dtype=np.float32)
        except Exception:
            acc = None

        if acc is None:
            best_sc = -1.0
            for a in np.linspace(-6.5, 6.5, 131):
                tan_a = float(np.tan(np.radians(a)))
                acc_c = np.zeros(sw, dtype=np.float32)
                for i in range(n_bands):
                    sh_x = int(round(b_ys[i] * tan_a))
                    if sh_x == 0:
                        acc_c += b_profs[i]
                    elif 0 < sh_x < sw:
                        acc_c[:-sh_x] += b_profs[i, sh_x:]
                    elif -sw < sh_x < 0:
                        acc_c[-sh_x:] += b_profs[i, : sw + sh_x]
                diff = np.diff(acc_c)
                sc = float(np.sum(diff * diff))
                if sc > best_sc:
                    best_sc = sc
                    best_a = float(a)

            tan_a = float(np.tan(np.radians(best_a)))
            acc = np.zeros(sw, dtype=np.float32)
            for i in range(n_bands):
                sh_x = int(round(b_ys[i] * tan_a))
                if sh_x == 0:
                    acc += b_profs[i]
                elif 0 < sh_x < sw:
                    acc[:-sh_x] += b_profs[i, sh_x:]
                elif -sw < sh_x < 0:
                    acc[-sh_x:] += b_profs[i, : sw + sh_x]
            acc /= n_bands
        else:
            tan_a = float(np.tan(np.radians(best_a)))

        smooth = np.convolve(acc, np.ones(9) / 9.0, mode="same")
        thresh = max(float(np.percentile(smooth, 84)), 8.0)
        min_col_w = int(w * 0.032)

        cands: list[tuple[float, int]] = []
        for x in range(10, sw - 10):
            if smooth[x] >= thresh and smooth[x] == np.max(smooth[max(0, x - 20) : min(sw, x + 21)]):
                cands.append((float(smooth[x]), x_start + x))
        cands.sort(reverse=True)
        sel_x: list[int] = []
        for _, x_abs in cands:
            if all(abs(x_abs - sx) >= min_col_w for sx in sel_x):
                sel_x.append(x_abs)
        sel_x.sort()
        for x_abs in sel_x:
            cols_out.append({
                "x_center": int(x_abs),
                "yc": float(yc),
                "tan_a": float(tan_a),
                "angle_deg": round(best_a, 2),
                "side": side,
            })
    return cols_out


def _warp_piecewise_angled_strip(
    img: np.ndarray,
    x0: int,
    x1: int,
    top_model: _AngledLineModel,
    bot_model: _AngledLineModel,
    pad_y: int,
    out_h: int | None = None,
) -> np.ndarray:
    """
    Slices a row between top_model and bot_model across [x0, x1] under its exact local angle
    and piecewise gutter curve, rectifying it into a flat horizontal image strip.
    """
    if out_h is None:
        h_l = (bot_model.eval_y(x0) - top_model.eval_y(x0)) + 2 * pad_y
        h_r = (bot_model.eval_y(x1) - top_model.eval_y(x1)) + 2 * pad_y
        out_h = max(24, int(round(0.5 * (h_l + h_r))))

    img_h, img_w = img.shape[:2]
    if (
        abs(top_model.tan_a) < 1e-4
        and abs(bot_model.tan_a) < 1e-4
        and not top_model.knot_xs
        and not bot_model.knot_xs
    ):
        y_top = max(0, int(round(top_model.y_center)) - pad_y)
        y_bot = min(img_h, max(y_top + 1, int(round(bot_model.y_center)) + pad_y))
        cx0 = max(0, min(img_w - 1, x0))
        cx1 = max(cx0 + 1, min(img_w, x1))
        direct = img[y_top:y_bot, cx0:cx1].copy()
        if direct.shape[0] != out_h or direct.shape[1] != max(1, x1 - x0):
            return cv2.resize(direct, (max(1, x1 - x0), out_h), interpolation=cv2.INTER_CUBIC)
        return direct

    knots = [int(round(v)) for v in np.linspace(x0, x1, 5)]
    sub_strips: list[np.ndarray] = []
    for k in range(len(knots) - 1):
        kx0, kx1 = knots[k], knots[k + 1]
        sw = max(1, kx1 - kx0)
        y_tl = top_model.eval_y(kx0) - pad_y
        y_tr = top_model.eval_y(kx1) - pad_y
        y_br = bot_model.eval_y(kx1) + pad_y
        y_bl = bot_model.eval_y(kx0) + pad_y

        ry0 = max(0, int(np.floor(min(y_tl, y_tr, y_bl, y_br))) - 4)
        ry1 = min(img_h, int(np.ceil(max(y_tl, y_tr, y_bl, y_br))) + 4)
        rx0 = max(0, kx0 - 2)
        rx1 = min(img_w, kx1 + 2)
        roi = img[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            roi = img
            rx0, ry0 = 0, 0

        src_quad = np.array(
            [
                [kx0 - rx0, y_tl - ry0],
                [kx1 - rx0, y_tr - ry0],
                [kx1 - rx0, y_br - ry0],
                [kx0 - rx0, y_bl - ry0],
            ],
            dtype=np.float32,
        )
        dst_rect = np.array(
            [[0, 0], [sw - 1, 0], [sw - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        mat = cv2.getPerspectiveTransform(src_quad, dst_rect)
        sub_warped = cv2.warpPerspective(
            roi,
            mat,
            (sw, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        sub_strips.append(sub_warped)

    return np.hstack(sub_strips) if len(sub_strips) > 1 else sub_strips[0]


def slice_matrika_spread(
    image_path: Path,
    scan_id: int,
    crops_root: Path,
) -> LayoutResult:
    """
    Splits a two-page matrika spread at the spine, detects the per-page and per-row tilt angles,
    slices each row along its angled quadrilateral/curve on the left and right pages, rectifies
    the extracted strips into level horizontal images, and draws the angled grid overlay.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spine_x = find_spine_x(gray)
    split_x = spine_x if spine_x > 0 else (w // 2)

    boundary_models, angle_meta = _detect_angled_row_boundaries(gray, split_x)
    if spine_x == 0:
        angle_meta["spine_offset_px"] = 0.0

    scan_crop_dir = crops_root / f"scan_{scan_id}"
    scan_crop_dir.mkdir(parents=True, exist_ok=True)

    first_lt, first_rt = boundary_models[0]
    last_lb, last_rb = boundary_models[-1]

    header_y_end = int(round(0.5 * (first_lt.y_center + first_rt.y_center)))
    bottom_y_end = int(round(0.5 * (last_lb.y_center + last_rb.y_center)))

    # Rectify header band using the top of Row 1 as the angled bottom of the header
    header_top_l = _AngledLineModel(
        y_center=max(12.0, first_lt.y_center - max(h * 0.16, 360.0)),
        tan_a=first_lt.tan_a,
        x_center=first_lt.x_center,
    )
    header_top_r = _AngledLineModel(
        y_center=max(12.0, first_rt.y_center - max(h * 0.16, 360.0)),
        tan_a=first_rt.tan_a,
        x_center=first_rt.x_center,
    )
    hdr_h = max(120, int(round(0.5 * ((first_lt.y_center - header_top_l.y_center) + (first_rt.y_center - header_top_r.y_center)) + 28)))
    hdr_left = _warp_piecewise_angled_strip(img, 0, split_x, header_top_l, first_lt, pad_y=14, out_h=hdr_h)
    hdr_right = _warp_piecewise_angled_strip(img, split_x, w, header_top_r, first_rt, pad_y=14, out_h=hdr_h)
    header_crop = np.hstack([hdr_left, hdr_right])
    header_file = scan_crop_dir / "header.jpg"
    cv2.imwrite(str(header_file), header_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    header_rel = f"/media/crops/scan_{scan_id}/header.jpg"

    angled_cols = _detect_angled_vertical_columns(gray, split_x, header_y_end, bottom_y_end)
    vert_cols_x = [c["x_center"] for c in angled_cols]

    overlay = img.copy()
    # Draw spine line in cyan only when a real book spine was detected
    if spine_x > 0:
        cv2.line(overlay, (spine_x, 0), (spine_x, h), (255, 220, 0), 4)

    # Draw angled vertical column lines in amber between the angled header boundary and last row boundary
    for col in angled_cols:
        xc_col = col["x_center"]
        yc_col = col["yc"]
        tan_col = col["tan_a"]
        top_m = first_lt if col["side"] == "L" else first_rt
        bot_m = last_lb if col["side"] == "L" else last_rb
        y_t = int(round(top_m.eval_y(xc_col)))
        y_b = int(round(bot_m.eval_y(xc_col)))
        x_t = int(np.clip(round(xc_col + (y_t - yc_col) * tan_col), 0, w - 1))
        x_b = int(np.clip(round(xc_col + (y_b - yc_col) * tan_col), 0, w - 1))
        cv2.line(overlay, (x_t, y_t), (x_b, y_b), (0, 180, 255), 2)

    # Draw angled header boundary in blue
    hdr_pts_l = np.array([[kx, int(round(first_lt.eval_y(kx)))] for kx in np.linspace(10, split_x, 5)], dtype=np.int32)
    hdr_pts_r = np.array([[kx, int(round(first_rt.eval_y(kx)))] for kx in np.linspace(split_x, w - 10, 5)], dtype=np.int32)
    cv2.polylines(overlay, [hdr_pts_l], isClosed=False, color=(255, 90, 30), thickness=4)
    cv2.polylines(overlay, [hdr_pts_r], isClosed=False, color=(255, 90, 30), thickness=4)

    rows_info: list[RowSliceInfo] = []
    pad_y = max(14, int(h * 0.006))

    for idx in range(1, len(boundary_models)):
        lt, rt = boundary_models[idx - 1]
        lb, rb = boundary_models[idx]

        h_l = (lb.y_center - lt.y_center) + 2 * pad_y
        h_r = (rb.y_center - rt.y_center) + 2 * pad_y
        row_out_h = max(32, int(round(0.5 * (h_l + h_r))))

        left_strip = _warp_piecewise_angled_strip(img, 0, split_x, lt, lb, pad_y=pad_y, out_h=row_out_h)
        right_strip = _warp_piecewise_angled_strip(img, split_x, w, rt, rb, pad_y=pad_y, out_h=row_out_h)
        stitched_strip = np.hstack([left_strip, right_strip])

        left_file = scan_crop_dir / f"row_{idx:02d}_left.jpg"
        right_file = scan_crop_dir / f"row_{idx:02d}_right.jpg"
        stitched_file = scan_crop_dir / f"row_{idx:02d}_stitched.jpg"

        cv2.imwrite(str(left_file), left_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(right_file), right_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(stitched_file), stitched_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        # Save 4 semantic region crops from the angle-rectified & spine-aligned stitched strip
        cell_regions = [
            ("date_name", int(w * 0.05), int(split_x * 0.80), "Dátum a meno"),
            ("parents", int(split_x * 0.65), int(split_x + (w - split_x) * 0.42), "Rodičia / Priezvisko"),
            ("village", int(split_x + (w - split_x) * 0.18), int(split_x + (w - split_x) * 0.65), "Obec a dom č."),
            ("observationes", int(split_x + (w - split_x) * 0.60), w, "Krstní rodičia / Poznámky"),
        ]
        cell_crops_list: list[dict[str, Any]] = []
        for cell_key, cx0, cx1, label_sk in cell_regions:
            cell_img = stitched_strip[:, max(0, cx0) : min(w, cx1)]
            c_file = scan_crop_dir / f"row_{idx:02d}_{cell_key}.jpg"
            cv2.imwrite(str(c_file), cell_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            cell_crops_list.append({
                "key": cell_key,
                "label_sk": label_sk,
                "x0": cx0,
                "x1": cx1,
                "crop_url": f"/media/crops/scan_{scan_id}/row_{idx:02d}_{cell_key}.jpg",
            })

        # Build angled polygon for Left page and Right page on the visual overlay
        xs_l = [int(round(v)) for v in np.linspace(10, split_x, 5)]
        poly_l = np.array(
            [[x, int(round(lt.eval_y(x)))] for x in xs_l]
            + [[x, int(round(lb.eval_y(x)))] for x in reversed(xs_l)],
            dtype=np.int32,
        )
        xs_r = [int(round(v)) for v in np.linspace(split_x, w - 10, 5)]
        poly_r = np.array(
            [[x, int(round(rt.eval_y(x)))] for x in xs_r]
            + [[x, int(round(rb.eval_y(x)))] for x in reversed(xs_r)],
            dtype=np.int32,
        )
        cv2.polylines(overlay, [poly_l], isClosed=True, color=(60, 210, 60), thickness=3)
        cv2.polylines(overlay, [poly_r], isClosed=True, color=(60, 210, 60), thickness=3)

        row_ang_l = round(0.5 * (lt.angle_deg + lb.angle_deg), 2)
        row_ang_r = round(0.5 * (rt.angle_deg + rb.angle_deg), 2)
        label_y = int(round(lt.eval_y(28))) + 55
        cv2.putText(
            overlay,
            f"R{idx}",
            (28, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (20, 160, 20),
            4,
        )

        left_quad = [
            [0, int(round(lt.eval_y(0)))],
            [split_x, int(round(lt.eval_y(split_x)))],
            [split_x, int(round(lb.eval_y(split_x)))],
            [0, int(round(lb.eval_y(0)))],
        ]
        right_quad = [
            [split_x, int(round(rt.eval_y(split_x)))],
            [w, int(round(rt.eval_y(w)))],
            [w, int(round(rb.eval_y(w)))],
            [split_x, int(round(rb.eval_y(split_x)))],
        ]

        y0_mean = int(round(0.5 * (lt.y_center + rt.y_center)))
        y1_mean = int(round(0.5 * (lb.y_center + rb.y_center)))

        rows_info.append(
            RowSliceInfo(
                row_index=idx,
                y_start=y0_mean,
                y_end=y1_mean,
                left_crop_rel=f"/media/crops/scan_{scan_id}/row_{idx:02d}_left.jpg",
                right_crop_rel=f"/media/crops/scan_{scan_id}/row_{idx:02d}_right.jpg",
                stitched_crop_rel=f"/media/crops/scan_{scan_id}/row_{idx:02d}_stitched.jpg",
                cell_crops=cell_crops_list,
                angle_left_deg=row_ang_l,
                angle_right_deg=row_ang_r,
                left_quad=left_quad,
                right_quad=right_quad,
            )
        )

    # Save resized overlay (max width 2200px for fast browser inspection)
    overlay_scale = min(1.0, 2200.0 / float(w))
    if overlay_scale < 1.0:
        overlay_small = cv2.resize(overlay, (int(w * overlay_scale), int(h * overlay_scale)))
    else:
        overlay_small = overlay
    overlay_file = scan_crop_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_file), overlay_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return LayoutResult(
        width=w,
        height=h,
        spine_x=spine_x,
        skew_left_deg=angle_meta["skew_left_deg"],
        skew_right_deg=angle_meta["skew_right_deg"],
        header_y_end=header_y_end,
        header_crop_rel=header_rel,
        overlay_rel=f"/media/crops/scan_{scan_id}/overlay.jpg",
        vertical_lines_x=vert_cols_x,
        rows=rows_info,
        col_tilt_left_deg=angle_meta["col_tilt_left_deg"],
        col_tilt_right_deg=angle_meta["col_tilt_right_deg"],
        spine_offset_px=angle_meta["spine_offset_px"],
        slicing_mode=angle_meta["slicing_mode"],
    )
