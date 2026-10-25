use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

const BASE64_TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64_encode_internal(input: &[u8]) -> String {
    let out_len = input.len().div_ceil(3) * 4;
    let mut out = Vec::with_capacity(out_len);
    let mut chunks = input.chunks_exact(3);
    for chunk in &mut chunks {
        let b0 = chunk[0] as u32;
        let b1 = chunk[1] as u32;
        let b2 = chunk[2] as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(BASE64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(BASE64_TABLE[((n >> 6) & 0x3F) as usize]);
        out.push(BASE64_TABLE[(n & 0x3F) as usize]);
    }
    let rem = chunks.remainder();
    if rem.len() == 1 {
        let n = (rem[0] as u32) << 16;
        out.push(BASE64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(BASE64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(b'=');
        out.push(b'=');
    } else if rem.len() == 2 {
        let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
        out.push(BASE64_TABLE[((n >> 18) & 0x3F) as usize]);
        out.push(BASE64_TABLE[((n >> 12) & 0x3F) as usize]);
        out.push(BASE64_TABLE[((n >> 6) & 0x3F) as usize]);
        out.push(b'=');
    }
    // Safe because BASE64_TABLE and '=' are valid ASCII
    unsafe { String::from_utf8_unchecked(out) }
}

#[pyfunction]
pub fn base64_encode_bytes_rs(data: &[u8]) -> String {
    base64_encode_internal(data)
}

#[inline]
fn bgr_to_hsv_pixel(b: u8, g: u8, r: u8) -> (u8, u8, u8) {
    let bf = b as i32;
    let gf = g as i32;
    let rf = r as i32;
    let max_v = bf.max(gf).max(rf);
    let min_v = bf.min(gf).min(rf);
    let diff = max_v - min_v;
    let v = max_v as u8;
    if max_v == 0 || diff == 0 {
        return (0, 0, v);
    }
    let s = ((255 * diff + (max_v >> 1)) / max_v).clamp(0, 255) as u8;
    let h_deg = if max_v == rf {
        60 * (gf - bf) / diff
    } else if max_v == gf {
        120 + 60 * (bf - rf) / diff
    } else {
        240 + 60 * (rf - gf) / diff
    };
    let h_pos = if h_deg < 0 { h_deg + 360 } else { h_deg };
    let h_opencv = (h_pos / 2).clamp(0, 180) as u8;
    (h_opencv, s, v)
}

pub fn suppress_auditor_crayon_bgr_internal(img_bgr: &[u8], h: usize, w: usize) -> Vec<u8> {
    let n_pix = h * w;
    if img_bgr.len() != n_pix * 3 || h == 0 || w == 0 {
        return img_bgr.to_vec();
    }

    // Step 1: Parallel per-pixel crayon mask & dark-ink mask
    let mut crayon_mask = vec![0_u8; n_pix];
    let mut dark_ink = vec![0_u8; n_pix];

    crayon_mask
        .par_chunks_mut(w)
        .zip(dark_ink.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (c_row, d_row))| {
            let row_bgr = &img_bgr[(y * w * 3)..((y + 1) * w * 3)];
            for x in 0..w {
                let b = row_bgr[x * 3];
                let g = row_bgr[x * 3 + 1];
                let r = row_bgr[x * 3 + 2];
                let bi = b as i16;
                let gi = g as i16;
                let ri = r as i16;

                let (hv, sv, vv) = bgr_to_hsv_pixel(b, g, r);
                let green_hsv = (32..=105).contains(&hv) && sv >= 38 && vv >= 50;
                let green_rgb = (gi - ri >= 14) && (gi - bi >= 10) && (gi >= 55);
                let red_hsv = (hv <= 10 || hv >= 165) && sv >= 65 && vv >= 65;
                let red_rgb = (ri - gi >= 32) && (ri - bi >= 28) && (ri >= 80);

                if green_hsv || green_rgb || red_hsv || red_rgb {
                    c_row[x] = 1;
                }
                let gray = ((29 * (b as u32) + 150 * (g as u32) + 77 * (r as u32) + 128) >> 8) as u8;
                if gray < 68 && (gi - ri) < 18 && (ri - gi) < 42 {
                    d_row[x] = 1;
                }
            }
        });

    if !crayon_mask.iter().any(|&m| m != 0) {
        return img_bgr.to_vec();
    }

    // Step 2: 3x3 cross/ellipse dilation of crayon_mask minus dark_ink
    let mut final_mask = vec![0_u8; n_pix];
    final_mask
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, f_row)| {
            let y_prev = y.saturating_sub(1);
            let y_next = (y + 1).min(h - 1);
            for x in 0..w {
                if dark_ink[y * w + x] != 0 {
                    continue;
                }
                let x_prev = x.saturating_sub(1);
                let x_next = (x + 1).min(w - 1);
                let dil = crayon_mask[y * w + x] != 0
                    || crayon_mask[y_prev * w + x] != 0
                    || crayon_mask[y_next * w + x] != 0
                    || crayon_mask[y * w + x_prev] != 0
                    || crayon_mask[y * w + x_next] != 0;
                if dil {
                    f_row[x] = 1;
                }
            }
        });

    if !final_mask.iter().any(|&m| m != 0) {
        return img_bgr.to_vec();
    }

    // Step 3: Estimate 78th-percentile warm paper background tone via 256-bin histograms
    let mut hist_b = [0_u64; 256];
    let mut hist_g = [0_u64; 256];
    let mut hist_r = [0_u64; 256];
    let mut count_non_crayon = 0_u64;
    let step = (n_pix / 65536).max(1);
    for idx in (0..n_pix).step_by(step) {
        if final_mask[idx] == 0 {
            let off = idx * 3;
            hist_b[img_bgr[off] as usize] += 1;
            hist_g[img_bgr[off + 1] as usize] += 1;
            hist_r[img_bgr[off + 2] as usize] += 1;
            count_non_crayon += 1;
        }
    }
    let percentile_from_hist = |hist: &[u64; 256], total: u64| -> u8 {
        if total == 0 {
            return 238;
        }
        let target = ((total as f64) * 0.78) as u64;
        let mut acc = 0_u64;
        for (val, &cnt) in hist.iter().enumerate() {
            acc += cnt;
            if acc >= target {
                return val as u8;
            }
        }
        238
    };
    let paper_b = percentile_from_hist(&hist_b, count_non_crayon);
    let paper_g = percentile_from_hist(&hist_g, count_non_crayon);
    let paper_r = percentile_from_hist(&hist_r, count_non_crayon);

    // Step 4: Replace masked pixels with local smooth background (fast 15-px radius box average around masked pixels)
    let mut bg_canvas = img_bgr.to_vec();
    for idx in 0..n_pix {
        if final_mask[idx] != 0 {
            let off = idx * 3;
            bg_canvas[off] = paper_b;
            bg_canvas[off + 1] = paper_g;
            bg_canvas[off + 2] = paper_r;
        }
    }

    let mut out = bg_canvas.clone();
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, out_row)| {
            let m_row = &final_mask[(y * w)..((y + 1) * w)];
            if !m_row.iter().any(|&v| v != 0) {
                return;
            }
            let y0 = y.saturating_sub(12);
            let y1 = (y + 12).min(h - 1);
            for x in 0..w {
                if m_row[x] == 0 {
                    continue;
                }
                let x0 = x.saturating_sub(12);
                let x1 = (x + 12).min(w - 1);
                let mut sum_b = 0_u32;
                let mut sum_g = 0_u32;
                let mut sum_r = 0_u32;
                let mut cnt = 0_u32;
                for sy in (y0..=y1).step_by(3) {
                    let s_row = &bg_canvas[(sy * w * 3)..((sy + 1) * w * 3)];
                    for sx in (x0..=x1).step_by(3) {
                        let o = sx * 3;
                        sum_b += s_row[o] as u32;
                        sum_g += s_row[o + 1] as u32;
                        sum_r += s_row[o + 2] as u32;
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    out_row[x * 3] = (sum_b / cnt) as u8;
                    out_row[x * 3 + 1] = (sum_g / cnt) as u8;
                    out_row[x * 3 + 2] = (sum_r / cnt) as u8;
                }
            }
        });

    out
}

#[pyfunction]
pub fn suppress_auditor_crayon_bytes_rs<'py>(
    py: Python<'py>,
    img_bgr: &[u8],
    h: usize,
    w: usize,
) -> Bound<'py, PyBytes> {
    let out = py.allow_threads(|| suppress_auditor_crayon_bgr_internal(img_bgr, h, w));
    PyBytes::new_bound(py, &out)
}

#[pyfunction]
pub fn sheared_radon_best_angle_rs(
    py: Python<'_>,
    profs_flat: Vec<f32>,
    n_strips: usize,
    h: usize,
    xs: Vec<f32>,
    search_min_deg: f64,
    search_max_deg: f64,
    n_steps: usize,
    by0: usize,
    by1: usize,
) -> (f64, Vec<f32>) {
    py.allow_threads(move || {
        if n_strips == 0 || h == 0 || profs_flat.len() < n_strips * h || n_steps == 0 {
            return (0.0, vec![0.0_f32; h]);
        }
        let by0_clamped = by0.min(h.saturating_sub(1));
        let by1_clamped = by1.min(h).max(by0_clamped + 1);

        let compute_acc = |tan_a: f64| -> Vec<f32> {
            let mut acc = vec![0.0_f32; h];
            for i in 0..n_strips {
                let sh_y = ((xs[i] as f64) * tan_a).round() as isize;
                let row = &profs_flat[(i * h)..((i + 1) * h)];
                if sh_y == 0 {
                    for y in 0..h {
                        acc[y] += row[y];
                    }
                } else if sh_y > 0 && (sh_y as usize) < h {
                    let s = sh_y as usize;
                    for y in 0..(h - s) {
                        acc[y] += row[y + s];
                    }
                } else if sh_y < 0 && ((-sh_y) as usize) < h {
                    let s = (-sh_y) as usize;
                    for y in 0..(h - s) {
                        acc[y + s] += row[y];
                    }
                }
            }
            let inv_n = 1.0_f32 / (n_strips as f32);
            for v in &mut acc {
                *v *= inv_n;
            }
            acc
        };

        let best = (0..n_steps)
            .into_par_iter()
            .map(|step_idx| {
                let frac = if n_steps > 1 {
                    (step_idx as f64) / ((n_steps - 1) as f64)
                } else {
                    0.0
                };
                let a = search_min_deg + frac * (search_max_deg - search_min_deg);
                let tan_a = a.to_radians().tan();
                let acc = compute_acc(tan_a);
                let mut sc = 0.0_f64;
                for y in by0_clamped..(by1_clamped - 1) {
                    let d = (acc[y + 1] - acc[y]) as f64;
                    sc += d * d;
                }
                (step_idx, a, sc)
            })
            .reduce(
                || (usize::MAX, 0.0_f64, -1.0_f64),
                |acc_l, acc_r| {
                    if acc_r.2 > acc_l.2 || (acc_r.2 == acc_l.2 && acc_r.0 < acc_l.0) {
                        acc_r
                    } else {
                        acc_l
                    }
                },
            );

        let best_a = best.1;
        let best_acc = compute_acc(best_a.to_radians().tan());
        (best_a, best_acc)
    })
}

pub fn html_table_to_pipe_text_internal(raw_html_or_text: &str) -> String {
    if raw_html_or_text.is_empty() {
        return String::new();
    }
    let decode_entities = |s: &str| -> String {
        if !s.contains('&') {
            return s.to_string();
        }
        s.replace("&nbsp;", " ")
            .replace("&#160;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&#X27;", "'")
            .replace("&apos;", "'")
    };
    let low = raw_html_or_text.to_ascii_lowercase();
    if !low.contains("<table") && !low.contains("<td") && !low.contains("<th") {
        return decode_entities(raw_html_or_text.trim());
    }

    let collapse_ws = |s: &str| -> String {
        let decoded = decode_entities(s);
        let mut out = String::with_capacity(decoded.len());
        let mut in_ws = false;
        for ch in decoded.chars() {
            if ch.is_whitespace() {
                if !in_ws && !out.is_empty() {
                    out.push(' ');
                    in_ws = true;
                }
            } else {
                out.push(ch);
                in_ws = false;
            }
        }
        out.trim().to_string()
    };

    let bytes = raw_html_or_text.as_bytes();
    let len = bytes.len();
    let mut rows_out: Vec<String> = Vec::new();
    let mut cur_row_cells: Vec<String> = Vec::new();
    let mut cur_cell_buf = String::new();
    let mut in_tr = false;
    let mut in_cell = false;

    let mut i = 0;
    while i < len {
        if bytes[i] == b'<' {
            if let Some(close_rel) = bytes[i..].iter().position(|&b| b == b'>') {
                let tag_raw = &raw_html_or_text[(i + 1)..(i + close_rel)];
                let tag_low = tag_raw
                    .trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                match tag_low.as_str() {
                    "tr" => {
                        in_tr = true;
                        cur_row_cells.clear();
                        cur_cell_buf.clear();
                        in_cell = false;
                    }
                    "/tr" => {
                        if in_cell {
                            cur_row_cells.push(collapse_ws(&cur_cell_buf));
                            cur_cell_buf.clear();
                            in_cell = false;
                        }
                        if in_tr {
                            if cur_row_cells.iter().any(|c| !c.is_empty()) {
                                rows_out.push(cur_row_cells.join(" | "));
                            }
                            cur_row_cells.clear();
                            in_tr = false;
                        }
                    }
                    "td" | "th" => {
                        if in_cell {
                            cur_row_cells.push(collapse_ws(&cur_cell_buf));
                            cur_cell_buf.clear();
                        }
                        in_cell = true;
                    }
                    "/td" | "/th" => {
                        if in_cell {
                            cur_row_cells.push(collapse_ws(&cur_cell_buf));
                            cur_cell_buf.clear();
                            in_cell = false;
                        }
                    }
                    "br" | "br/" => {
                        if in_cell {
                            cur_cell_buf.push(' ');
                        }
                    }
                    _ => {
                        if in_cell {
                            cur_cell_buf.push(' ');
                        }
                    }
                }
                i += close_rel + 1;
                continue;
            }
        }
        let ch = raw_html_or_text[i..].chars().next().unwrap_or(' ');
        if in_cell {
            cur_cell_buf.push(ch);
        }
        i += ch.len_utf8();
    }

    if !rows_out.is_empty() {
        return rows_out.join("\n");
    }

    // Fallback: strip all tags and join with " | "
    let mut fallback_parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut j = 0;
    while j < len {
        if bytes[j] == b'<' {
            if let Some(close_rel) = bytes[j..].iter().position(|&b| b == b'>') {
                let cleaned = collapse_ws(&cur);
                if !cleaned.is_empty() {
                    fallback_parts.push(cleaned);
                }
                cur.clear();
                j += close_rel + 1;
                continue;
            }
        }
        let ch = raw_html_or_text[j..].chars().next().unwrap_or(' ');
        cur.push(ch);
        j += ch.len_utf8();
    }
    let cleaned = collapse_ws(&cur);
    if !cleaned.is_empty() {
        fallback_parts.push(cleaned);
    }
    fallback_parts.join(" | ")
}

#[pyfunction]
#[pyo3(signature = (raw_html_or_text=None))]
pub fn html_table_to_pipe_text_rs(raw_html_or_text: Option<&str>) -> String {
    match raw_html_or_text {
        Some(s) => html_table_to_pipe_text_internal(s),
        None => String::new(),
    }
}

#[pyfunction]
pub fn skew_compensated_row_projection_u8_rs(
    py: Python<'_>,
    horiz_u8: &[u8],
    h: usize,
    w_strip: usize,
    x0: usize,
    xc: f64,
    tan_a: f64,
    n_slices: usize,
) -> Vec<f64> {
    py.allow_threads(|| {
        if h == 0 || w_strip == 0 || n_slices == 0 || horiz_u8.len() < h * w_strip {
            return vec![0.0_f64; h];
        }
        let mut xs = Vec::with_capacity(n_slices + 1);
        for i in 0..=n_slices {
            xs.push((i * w_strip) / n_slices);
        }

        let slice_projs: Vec<Vec<f64>> = (0..n_slices)
            .into_par_iter()
            .map(|i| {
                let sx0 = xs[i];
                let sx1 = xs[i + 1];
                if sx1 <= sx0 {
                    return vec![0.0_f64; h];
                }
                let sw = (sx1 - sx0) as f64;
                let mut col_prof = vec![0.0_f64; h];
                for y in 0..h {
                    let row_slice = &horiz_u8[(y * w_strip + sx0)..(y * w_strip + sx1)];
                    let sum: u32 = row_slice.iter().map(|&v| v as u32).sum();
                    col_prof[y] = (sum as f64) / sw;
                }
                let x_mid = (x0 as f64) + 0.5 * ((sx0 + sx1) as f64);
                let dy = ((x_mid - xc) * tan_a).round() as isize;
                if dy == 0 {
                    return col_prof;
                }
                let mut shifted = vec![0.0_f64; h];
                for y in 0..h {
                    let src_y = (y as isize) + dy;
                    if src_y >= 0 && (src_y as usize) < h {
                        shifted[y] = col_prof[src_y as usize];
                    }
                }
                shifted
            })
            .collect();

        let mut proj = vec![0.0_f64; h];
        let inv_n = 1.0 / (n_slices as f64);
        for sp in &slice_projs {
            for y in 0..h {
                proj[y] += sp[y];
            }
        }
        for v in &mut proj {
            *v *= inv_n;
        }
        proj
    })
}

#[pyfunction]
pub fn row_ink_metrics_gray_u8_rs(
    py: Python<'_>,
    gray_u8: &[u8],
    rh: usize,
    rw: usize,
) -> (f64, usize) {
    py.allow_threads(|| {
        if rh == 0 || rw == 0 || gray_u8.len() < rh * rw {
            return (0.0, 0);
        }
        // Compute median per column (exact float median matching np.median)
        let mut col_medians = vec![0.0_f32; rw];
        col_medians
            .par_iter_mut()
            .enumerate()
            .for_each(|(x, med_out)| {
                let mut col_vals = Vec::with_capacity(rh);
                for y in 0..rh {
                    col_vals.push(gray_u8[y * rw + x]);
                }
                col_vals.sort_unstable();
                let mid = rh / 2;
                *med_out = if rh % 2 == 1 {
                    col_vals[mid] as f32
                } else {
                    0.5 * ((col_vals[mid - 1] as f32) + (col_vals[mid] as f32))
                };
            });

        let mut ink_count = 0_usize;
        let mut horiz_transitions = 0_usize;
        for y in 0..rh {
            let row = &gray_u8[(y * rw)..((y + 1) * rw)];
            let mut prev_ink = false;
            for x in 0..rw {
                let is_ink = (col_medians[x] - (row[x] as f32)) > 24.0;
                if is_ink {
                    ink_count += 1;
                }
                if x > 0 && is_ink != prev_ink {
                    horiz_transitions += 1;
                }
                prev_ink = is_ink;
            }
        }
        let ink_ratio = (ink_count as f64) / ((rh * rw) as f64);
        (ink_ratio, horiz_transitions)
    })
}

fn lcs_matches_sum(a: &[u8], b: &[u8]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut best_i = 0;
    let mut best_j = 0;
    let mut best_size = 0;

    // Find longest contiguous matching block in a and b
    let mut j2len = vec![0_usize; b.len() + 1];
    let mut new_j2len = vec![0_usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        new_j2len.fill(0);
        for (j, &cb) in b.iter().enumerate() {
            if ca == cb {
                let k = j2len[j] + 1;
                new_j2len[j + 1] = k;
                if k > best_size {
                    best_size = k;
                    best_i = i + 1 - k;
                    best_j = j + 1 - k;
                }
            }
        }
        std::mem::swap(&mut j2len, &mut new_j2len);
    }

    if best_size == 0 {
        return 0;
    }
    let left = if best_i > 0 && best_j > 0 {
        lcs_matches_sum(&a[..best_i], &b[..best_j])
    } else {
        0
    };
    let right = if (best_i + best_size) < a.len() && (best_j + best_size) < b.len() {
        lcs_matches_sum(&a[(best_i + best_size)..], &b[(best_j + best_size)..])
    } else {
        0
    };
    left + best_size + right
}

pub fn sequence_ratio_internal(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a == b {
        return 1.0;
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let total_len = ab.len() + bb.len();
    if total_len == 0 {
        return 1.0;
    }
    let matches = lcs_matches_sum(ab, bb);
    (2.0 * (matches as f64)) / (total_len as f64)
}

#[pyfunction]
pub fn sequence_ratio_rs(a: &str, b: &str) -> f64 {
    sequence_ratio_internal(a, b)
}
