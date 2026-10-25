use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

static STATS_CACHE: Lazy<Mutex<HashMap<(String, i64, i64, i64, String, String, String, String, String), Value>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static TRACE_BASE_CACHE: Lazy<Mutex<HashMap<(String, i64, i64, String, String, String, String), CachedTraceBase>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn open_sqlite(db_path: &str) -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(db_path)?;
    conn.busy_timeout(Duration::from_secs(30))?;
    let _ = conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA temp_store=MEMORY;
         PRAGMA cache_size=-64000;",
    );
    Ok(conn)
}

pub fn fold_char(c: char) -> char {
    match c {
        'á' | 'Á' | 'ä' | 'Ä' | 'à' | 'À' | 'â' | 'Â' => 'a',
        'é' | 'É' | 'ë' | 'Ë' | 'ě' | 'Ě' | 'è' | 'È' | 'ê' | 'Ê' => 'e',
        'í' | 'Í' | 'ì' | 'Ì' | 'î' | 'Î' => 'i',
        'ó' | 'Ó' | 'ô' | 'Ô' | 'ö' | 'Ö' | 'ő' | 'Ő' | 'ø' | 'Ø' | 'ò' | 'Ò' => 'o',
        'ú' | 'Ú' | 'ü' | 'Ü' | 'ű' | 'Ű' | 'ů' | 'Ů' | 'ù' | 'Ù' | 'û' | 'Û' => 'u',
        'ý' | 'Ý' => 'y',
        'č' | 'Č' | 'ć' | 'Ć' => 'c',
        'ď' | 'Ď' | 'đ' | 'Đ' => 'd',
        'ľ' | 'Ľ' | 'ĺ' | 'Ĺ' | 'ł' | 'Ł' => 'l',
        'ň' | 'Ň' | 'ń' | 'Ń' => 'n',
        'ŕ' | 'Ŕ' | 'ř' | 'Ř' => 'r',
        'š' | 'Š' | 'ś' | 'Ś' => 's',
        'ť' | 'Ť' => 't',
        'ž' | 'Ž' | 'ź' | 'Ź' | 'ż' | 'Ż' => 'z',
        _ => c.to_ascii_lowercase(),
    }
}

pub fn fold_text_diacritics_internal(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        out.push(fold_char(c));
    }
    out.trim().to_string()
}

pub fn apply_digraphs_internal(token: &str) -> String {
    let mut s = String::with_capacity(token.len());
    for c in token.chars() {
        s.push(fold_char(c));
    }
    for (src, dst) in [
        ("cz", "cs"),
        ("tz", "cs"),
        ("sz", "s"),
        ("zs", "z"),
        ("dy", "gy"),
        ("nj", "ny"),
        ("lj", "ly"),
        ("th", "t"),
    ] {
        if s.contains(src) {
            s = s.replace(src, dst);
        }
    }
    if s.contains('w') {
        s = s.replace('w', "v");
    }
    s
}

pub fn orthographic_fold_internal(token: &str) -> String {
    let mut s = String::with_capacity(token.len());
    for c in token.chars() {
        let fc = fold_char(c);
        if fc.is_ascii_lowercase() || fc.is_ascii_digit() {
            s.push(fc);
        }
    }
    if s.is_empty() {
        return s;
    }
    for (src, dst) in [
        ("sch", "s"),
        ("cz", "c"),
        ("cs", "c"),
        ("ts", "c"),
        ("tz", "c"),
        ("sz", "s"),
        ("zs", "z"),
        ("gy", "d"),
        ("dy", "d"),
        ("ny", "n"),
        ("nj", "n"),
        ("ly", "l"),
        ("lj", "l"),
        ("ph", "f"),
        ("th", "t"),
        ("gh", "g"),
        ("w", "v"),
        ("y", "i"),
    ] {
        if s.contains(src) {
            s = s.replace(src, dst);
        }
    }
    let mut dedup = String::with_capacity(s.len());
    let mut prev: Option<char> = None;
    for ch in s.chars() {
        if ch.is_ascii_lowercase() && prev == Some(ch) {
            continue;
        }
        dedup.push(ch);
        prev = Some(ch);
    }
    dedup
}

pub fn phonetic_skeleton_internal(token: &str) -> String {
    let s = apply_digraphs_internal(token);
    let mut out = String::with_capacity(s.len());
    let mut prev: Option<char> = None;
    for ch in s.chars() {
        if matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
            continue;
        }
        if prev == Some(ch) {
            continue;
        }
        out.push(ch);
        prev = Some(ch);
    }
    out
}

#[inline]
fn confusion_cost_paleo(a: u8, b: u8) -> f64 {
    if a == b {
        return 0.0;
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    match (x, y) {
        (b'v', b'w') => 0.08,
        (b'v', b'y') | (b's', b'y') => 0.12,
        (b'i', b'y') | (b'j', b'y') | (b'a', b'o') | (b'n', b'u') | (b'u', b'v')
        | (b'i', b'l') | (b'i', b'j') => 0.15,
        (b'i', b't') | (b'l', b't') => 0.16,
        (b'r', b'u') | (b'n', b'v') | (b'h', b'm') | (b'c', b'e') | (b'e', b'o')
        | (b'c', b'k') | (b'n', b'r') | (b'g', b'q') | (b'r', b'v') => 0.18,
        (b'g', b'y') | (b'j', b'q') | (b'm', b'n') | (b'c', b'o') | (b'f', b'l')
        | (b'j', b'l') | (b'f', b'i') | (b'j', b't') | (b'g', b'j') | (b'b', b'p')
        | (b'r', b'z') => 0.20,
        (b'q', b'y') | (b'f', b'j') | (b'f', b'v') | (b'j', b's') | (b'h', b's')
        | (b's', b'z') => 0.22,
        (b'b', b'k') => 0.24,
        (b'm', b'u') | (b'm', b'w') | (b'p', b's') | (b'f', b's') | (b'f', b't') => 0.25,
        (b'c', b't') | (b'g', b'k') | (b'g', b's') | (b'b', b'h') | (b'c', b'z') => 0.28,
        (b'h', b'k') | (b'd', b'o') => 0.30,
        _ => 1.0,
    }
}

#[inline]
fn confusion_cost_search(a: u8, b: u8) -> Option<f64> {
    if a == b {
        return Some(0.0);
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    match (x, y) {
        (b'v', b'w') => Some(0.08),
        (b'v', b'y') | (b's', b'y') => Some(0.12),
        (b's', b'z') => Some(0.14),
        (b'i', b'y') | (b'j', b'y') | (b'a', b'o') | (b'n', b'u') | (b'u', b'v')
        | (b'i', b'l') | (b'i', b'j') => Some(0.15),
        (b'i', b't') | (b'l', b't') => Some(0.16),
        (b'r', b'u') | (b'n', b'v') | (b'h', b'm') | (b'c', b'e') | (b'e', b'o')
        | (b'c', b'k') | (b'n', b'r') | (b'g', b'q') | (b'r', b'v') | (b'd', b't')
        | (b'f', b't') => Some(0.18),
        (b'g', b'y') | (b'j', b'q') | (b'm', b'n') | (b'c', b'o') | (b'f', b'l')
        | (b'j', b'l') | (b'f', b'i') | (b'j', b't') | (b'g', b'j') | (b'b', b'p')
        | (b'r', b'z') | (b'c', b'z') | (b'h', b'k') | (b'b', b'h') => Some(0.20),
        (b'q', b'y') | (b'f', b'j') | (b'f', b'v') | (b'j', b's') | (b'h', b's') => Some(0.22),
        (b'b', b'k') => Some(0.24),
        (b'm', b'u') | (b'm', b'w') | (b'p', b's') | (b'f', b's') => Some(0.25),
        (b'c', b't') | (b'g', b'k') | (b'g', b's') => Some(0.28),
        (b'd', b'o') => Some(0.30),
        _ => None,
    }
}

#[inline]
fn digraph_to_char_cost(dg: &[u8; 2], ch: u8) -> Option<f64> {
    match (dg, ch) {
        (b"cz", b'c')
        | (b"cs", b'c')
        | (b"ts", b'c')
        | (b"tz", b'c')
        | (b"sz", b's')
        | (b"zs", b'z')
        | (b"th", b't')
        | (b"ph", b'f')
        | (b"gh", b'g') => Some(0.05),
        (b"gy", b'd') | (b"ny", b'n') | (b"ly", b'l') => Some(0.08),
        (b"vv", b'w') | (b"vv", b'v') => Some(0.10),
        (b"rn", b'm') => Some(0.12),
        (b"in", b'm')
        | (b"ni", b'm')
        | (b"nn", b'm')
        | (b"uu", b'w')
        | (b"uu", b'v')
        | (b"cl", b'd') => Some(0.15),
        (b"st", b'h')
        | (b"li", b'h')
        | (b"li", b'u')
        | (b"ri", b'n')
        | (b"ri", b'u') => Some(0.18),
        (b"si", b'h') => Some(0.20),
        _ => None,
    }
}

#[inline]
fn digraph_to_digraph_cost(dg_a: &[u8; 2], dg_b: &[u8; 2]) -> Option<f64> {
    if dg_a == dg_b {
        return None;
    }
    match (dg_a, dg_b) {
        (b"ur", b"nz") | (b"nz", b"ur") | (b"st", b"si") | (b"si", b"st") => Some(0.12),
        (b"st", b"sr") | (b"sr", b"st") => Some(0.15),
        _ => None,
    }
}

pub fn paleographic_distance_internal(left: &str, right: &str) -> f64 {
    let a = apply_digraphs_internal(left);
    let b = apply_digraphs_internal(right);
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let n = ab.len();
    let m = bb.len();
    if n == 0 {
        return m as f64;
    }
    if m == 0 {
        return n as f64;
    }
    let mut prev: Vec<f64> = (0..=m).map(|j| (j as f64) * 0.9).collect();
    let mut curr = vec![0.0; m + 1];
    for (i, &ca) in ab.iter().enumerate() {
        curr[0] = ((i + 1) as f64) * 0.9;
        for (j, &cb) in bb.iter().enumerate() {
            let del = prev[j + 1] + 0.9;
            let ins = curr[j] + 0.9;
            let sub = prev[j] + confusion_cost_paleo(ca, cb);
            curr[j + 1] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

pub fn paleographic_similarity_internal(left: &str, right: &str) -> f64 {
    let a = apply_digraphs_internal(left);
    let b = apply_digraphs_internal(right);
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let max_len = a.chars().count().max(b.chars().count()).max(1) as f64;
    let dist = paleographic_distance_internal(left, right);
    (1.0 - dist / max_len).max(0.0)
}

pub fn ocr_dp_distance_internal(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let n = ab.len();
    let m = bb.len();
    if n == 0 {
        return (m as f64) * 0.78;
    }
    if m == 0 {
        return (n as f64) * 0.78;
    }

    let stride = m + 1;
    let mut dp = vec![0.0_f64; (n + 1) * stride];
    for i in 1..=n {
        dp[i * stride] = (i as f64) * 0.78;
    }
    for j in 1..=m {
        dp[j] = (j as f64) * 0.78;
    }

    for i in 1..=n {
        let ca = ab[i - 1];
        for j in 1..=m {
            let cb = bb[j - 1];
            let sub = if ca == cb {
                0.0
            } else if let Some(conf) = confusion_cost_search(ca, cb) {
                conf
            } else if i == 1 && j == 1 {
                1.25
            } else {
                0.95
            };

            let mut best = (dp[(i - 1) * stride + j] + 0.78)
                .min(dp[i * stride + (j - 1)] + 0.78)
                .min(dp[(i - 1) * stride + (j - 1)] + sub);

            if i >= 2 {
                let dg_a = [ab[i - 2], ab[i - 1]];
                if let Some(c21) = digraph_to_char_cost(&dg_a, cb) {
                    best = best.min(dp[(i - 2) * stride + (j - 1)] + c21);
                }
            }
            if j >= 2 {
                let dg_b = [bb[j - 2], bb[j - 1]];
                if let Some(c12) = digraph_to_char_cost(&dg_b, ca) {
                    best = best.min(dp[(i - 1) * stride + (j - 2)] + c12);
                }
            }
            if i >= 2 && j >= 2 {
                let dg_a = [ab[i - 2], ab[i - 1]];
                let dg_b = [bb[j - 2], bb[j - 1]];
                if let Some(c22) = digraph_to_digraph_cost(&dg_a, &dg_b) {
                    best = best.min(dp[(i - 2) * stride + (j - 2)] + c22);
                }
            }

            dp[i * stride + j] = best;
        }
    }
    dp[n * stride + m]
}

pub fn clean_ascii_alnum(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let fc = fold_char(c);
        if fc.is_ascii_lowercase() || fc.is_ascii_digit() {
            out.push(fc);
        }
    }
    out
}

#[inline]
pub fn round4(val: f64) -> f64 {
    (val * 10000.0).round() / 10000.0
}

pub fn weighted_edit_similarity_internal(left: &str, right: &str) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let a_raw = clean_ascii_alnum(left);
    let b_raw = clean_ascii_alnum(right);
    if a_raw.is_empty() || b_raw.is_empty() {
        return 0.0;
    }
    if a_raw == b_raw {
        return 1.0;
    }
    let a_orth = orthographic_fold_internal(left);
    let b_orth = orthographic_fold_internal(right);
    if !a_orth.is_empty() && a_orth == b_orth {
        return 0.97;
    }

    let max_raw = a_raw.len().max(b_raw.len()).max(1) as f64;
    let dist_raw = ocr_dp_distance_internal(&a_raw, &b_raw);
    let sim_raw = (1.0 - dist_raw / max_raw).max(0.0);

    let max_orth = a_orth.len().max(b_orth.len()).max(1) as f64;
    let dist_orth = ocr_dp_distance_internal(&a_orth, &b_orth);
    let sim_orth = (1.0 - dist_orth / max_orth).max(0.0);

    round4(sim_raw.max(sim_orth))
}

#[pyfunction]
#[pyo3(signature = (text=None))]
fn fold_text_diacritics_rs(text: Option<&str>) -> String {
    match text {
        Some(s) => fold_text_diacritics_internal(s),
        None => String::new(),
    }
}

#[pyfunction]
fn apply_digraphs_rs(token: &str) -> String {
    apply_digraphs_internal(token)
}

#[pyfunction]
#[pyo3(signature = (token=None))]
fn orthographic_fold_rs(token: Option<&str>) -> String {
    match token {
        Some(s) => orthographic_fold_internal(s),
        None => String::new(),
    }
}

#[pyfunction]
fn phonetic_skeleton_rs(token: &str) -> String {
    phonetic_skeleton_internal(token)
}

#[pyfunction]
fn paleographic_distance_rs(left: &str, right: &str) -> f64 {
    paleographic_distance_internal(left, right)
}

#[pyfunction]
fn paleographic_similarity_rs(left: &str, right: &str) -> f64 {
    paleographic_similarity_internal(left, right)
}

#[pyfunction]
fn ocr_dp_distance_rs(a: &str, b: &str) -> f64 {
    ocr_dp_distance_internal(a, b)
}

#[pyfunction]
#[pyo3(signature = (left=None, right=None))]
fn weighted_edit_similarity_rs(left: Option<&str>, right: Option<&str>) -> f64 {
    match (left, right) {
        (Some(l), Some(r)) => weighted_edit_similarity_internal(l, r),
        _ => 0.0,
    }
}

#[pyfunction]
fn batch_score_candidates_against_lexicon_rs(
    raw: &str,
    candidates: Vec<(String, f64)>,
    threshold: f64,
) -> (Option<String>, f64) {
    let raw_folded = apply_digraphs_internal(raw);
    let raw_skel = phonetic_skeleton_internal(raw);
    let raw_low = raw.to_lowercase();

    for (name, _) in &candidates {
        if name.to_lowercase() == raw_low {
            return (Some(name.clone()), 1.0);
        }
    }
    for (name, _) in &candidates {
        if apply_digraphs_internal(name) == raw_folded {
            return (Some(name.clone()), 0.98);
        }
    }

    let max_w = candidates
        .iter()
        .map(|(_, w)| *w)
        .fold(1.0_f64, |a, b| a.max(b));
    let ln_max_w = (1.0 + max_w).ln();

    let mut best_name: Option<String> = None;
    let mut best_score = 0.0_f64;

    for (name, weight) in &candidates {
        let cand_skel = phonetic_skeleton_internal(name);
        if !raw_skel.is_empty() && !cand_skel.is_empty() {
            let min_l = raw_skel.len().min(cand_skel.len()) as f64;
            let max_l = raw_skel.len().max(cand_skel.len()) as f64;
            if (min_l / max_l) < 0.45 && paleographic_similarity_internal(raw, name) < 0.9 {
                continue;
            }
        }
        let sim = paleographic_similarity_internal(raw, name);
        if sim < threshold - 0.08 {
            continue;
        }
        let freq_norm = if max_w > 0.0 && ln_max_w > 0.0 {
            (1.0 + *weight).ln() / ln_max_w
        } else {
            0.0
        };
        let combined = 0.88 * sim + 0.12 * freq_norm;
        if combined > best_score {
            best_score = combined;
            best_name = Some(name.clone());
        }
    }

    if let Some(ref b_name) = best_name {
        let b_sim = paleographic_similarity_internal(raw, b_name);
        if b_sim >= threshold {
            return (best_name, best_score.max(b_sim));
        }
    }
    (None, 0.0)
}

// ============================================================================
// OCR Feedback & WorkflowTrace Statistics / Report Engine in Rust (rusqlite)
// ============================================================================

fn canonical_stage_group_internal(stage: &str) -> &'static str {
    let s = stage.trim().to_ascii_lowercase();
    if s == "layout" || s == "census_layout" || s.contains("layout") {
        return "layout";
    }
    if s == "ocr"
        || s == "ocr_header"
        || s == "ocr_left"
        || s == "ocr_right"
        || s == "census_overview_ocr"
        || s.starts_with("ocr")
    {
        return "ocr";
    }
    if s == "decision" || s.contains("decision") {
        return "decision";
    }
    if s == "extract"
        || s == "census_extract"
        || s == "census_overview_extract"
        || s.contains("extract")
    {
        return "extract";
    }
    if s.starts_with("mcp") {
        return "mcp";
    }
    "other"
}

pub struct ModelDefaults {
    pub default_ocr: String,
    pub default_decision: String,
    pub default_extract: String,
    pub default_layout: String,
    pub known_local_lower: HashSet<String>,
}

impl ModelDefaults {
    fn normalize_model_tag(&self, raw: Option<&str>, stage_hint: &str) -> String {
        let val = raw.unwrap_or("").trim();
        if val.is_empty() {
            return match stage_hint {
                "ocr" => self.default_ocr.clone(),
                "decision" => self.default_decision.clone(),
                "layout" => self.default_layout.clone(),
                _ => self.default_extract.clone(),
            };
        }
        if val.contains(':') {
            return val.to_string();
        }
        let low = val.to_ascii_lowercase();
        if self.known_local_lower.contains(&low)
            || low.contains("mlx")
            || low.contains("saluki")
            || low.contains("glm-ocr")
            || low.contains("d1-3b")
        {
            return format!("local:{}", val);
        }
        format!("freellm:{}", val)
    }
}

fn format_model_combo_internal(
    defaults: &ModelDefaults,
    ocr_model: &str,
    extract_model: &str,
    decision_model: Option<&str>,
    is_census: bool,
) -> String {
    let ocr_m = defaults.normalize_model_tag(Some(ocr_model), "ocr");
    let ext_m = defaults.normalize_model_tag(Some(extract_model), "extract");
    let dec_m = decision_model.unwrap_or("").trim();
    let has_llm_decision = !dec_m.is_empty()
        && !is_census
        && !dec_m.starts_with("deterministic:")
        && !dec_m.starts_with("skipped")
        && !dec_m.starts_with("none");
    if has_llm_decision {
        let dec_norm = defaults.normalize_model_tag(Some(dec_m), "decision");
        format!(
            "OCR: {} + Decision: {} + Extract: {}",
            ocr_m, dec_norm, ext_m
        )
    } else {
        format!("OCR: {} + Extract: {}", ocr_m, ext_m)
    }
}

#[derive(Clone)]
struct ProjectedTrace {
    scan_id: Option<i64>,
    row_id: Option<i64>,
    stage: String,
    model_or_tool: String,
    status: String,
    duration_ms: i64,
    prompt_tokens: i64,
    completion_tokens: i64,
    reasoning_tokens: i64,
    image_path: Option<String>,
}

#[derive(Clone)]
struct ProjectedFeedback {
    id: i64,
    row_id: Option<i64>,
    verdict: String,
    model_combo: String,
    layout_model: Option<String>,
    ocr_model: Option<String>,
    decision_model: Option<String>,
    extract_model: Option<String>,
    total_duration_ms: i64,
    layout_duration_ms: i64,
    ocr_duration_ms: i64,
    decision_duration_ms: i64,
    extract_duration_ms: i64,
    mcp_duration_ms: i64,
    stage_breakdown_json: String,
}

#[derive(Clone, Copy, Default)]
struct TimingSample {
    layout_ms: i64,
    scan_layout_ms: i64,
    ocr_ms: i64,
    ocr_header_ms: i64,
    ocr_left_ms: i64,
    ocr_right_ms: i64,
    decision_ms: i64,
    extract_ms: i64,
    mcp_ms: i64,
    total_ms: i64,
    prompt_tokens: i64,
    completion_tokens: i64,
    reasoning_tokens: i64,
}

#[derive(Clone)]
struct ComboBucket {
    model_combo: String,
    ocr_model: String,
    decision_model: Option<String>,
    extract_model: String,
    layout_model: String,
    correct_count: i64,
    incorrect_count: i64,
    total_rated: i64,
    timing_samples: Vec<TimingSample>,
    seen_row_keys: HashSet<(i64, String)>,
}

struct StepGroup {
    stage: String,
    canonical_stage: String,
    model_or_tool: String,
    durations: Vec<i64>,
    ok_count: i64,
    error_count: i64,
    prompt_tokens: Vec<i64>,
    completion_tokens: Vec<i64>,
    reasoning_tokens: Vec<i64>,
    row_ids: HashSet<i64>,
}

#[derive(Clone)]
struct CachedStepSummary {
    stage: String,
    canonical_stage: String,
    model_or_tool: String,
    calls_count: usize,
    ok_count: i64,
    error_count: i64,
    avg_duration_ms: i64,
    min_duration_ms: i64,
    max_duration_ms: i64,
    total_duration_ms: i64,
    avg_prompt_tokens: i64,
    avg_completion_tokens: i64,
    avg_reasoning_tokens: i64,
    row_ids: HashSet<i64>,
}

#[derive(Clone)]
struct CachedTraceBase {
    total_rows_analyzed: usize,
    combo_buckets: HashMap<String, ComboBucket>,
    combo_order: Vec<String>,
    steps: Vec<CachedStepSummary>,
}

fn avg_i64(vals: &[i64]) -> i64 {
    if vals.is_empty() {
        return 0;
    }
    let sum: i64 = vals.iter().sum();
    ((sum as f64) / (vals.len() as f64)).round() as i64
}

fn stage_order(stage: &str) -> i32 {
    match stage {
        "layout" | "census_layout" => 1,
        "ocr_header" => 2,
        "ocr_left" => 3,
        "ocr_right" => 4,
        "ocr" | "census_overview_ocr" => 5,
        "decision" => 6,
        "extract" | "census_extract" | "census_overview_extract" => 7,
        "mcp_place" => 8,
        "mcp_record" => 9,
        "mcp_compare" => 10,
        _ => 99,
    }
}

pub fn compute_feedback_statistics_from_conn(
    conn: &Connection,
    defaults: &ModelDefaults,
    data_dir: &str,
) -> Result<Value, rusqlite::Error> {
    let (tr_count, tr_max_id): (i64, i64) = conn
        .query_row(
            "SELECT COUNT(id), COALESCE(MAX(id), 0) FROM workflowtrace",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap_or((0, 0));
    let (fb_count, fb_max_upd): (i64, String) = conn
        .query_row(
            "SELECT COUNT(id), COALESCE(MAX(updated_at), '') FROM ocrfeedback",
            [],
            |r| Ok((r.get(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default())),
        )
        .unwrap_or((0, String::new()));

    let cache_key = (
        data_dir.to_string(),
        tr_count,
        tr_max_id,
        fb_count,
        fb_max_upd,
        defaults.default_ocr.clone(),
        defaults.default_decision.clone(),
        defaults.default_extract.clone(),
        defaults.default_layout.clone(),
    );
    if let Ok(guard) = STATS_CACHE.lock() {
        if let Some(cached) = guard.get(&cache_key) {
            return Ok(cached.clone());
        }
    }

    let mut fb_stmt = conn.prepare(
        "SELECT id, row_id, verdict, model_combo, layout_model, ocr_model, decision_model,
                extract_model, total_duration_ms, layout_duration_ms, ocr_duration_ms,
                decision_duration_ms, extract_duration_ms, mcp_duration_ms, stage_breakdown_json
         FROM ocrfeedback
         ORDER BY id DESC",
    )?;
    let all_feedback: Vec<ProjectedFeedback> = fb_stmt
        .query_map([], |row| {
            Ok(ProjectedFeedback {
                id: row.get(0)?,
                row_id: row.get(1)?,
                verdict: row.get::<_, Option<String>>(2)?.unwrap_or_else(|| "correct".to_string()),
                model_combo: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                layout_model: row.get(4)?,
                ocr_model: row.get(5)?,
                decision_model: row.get(6)?,
                extract_model: row.get(7)?,
                total_duration_ms: row.get::<_, Option<i64>>(8)?.unwrap_or(0),
                layout_duration_ms: row.get::<_, Option<i64>>(9)?.unwrap_or(0),
                ocr_duration_ms: row.get::<_, Option<i64>>(10)?.unwrap_or(0),
                decision_duration_ms: row.get::<_, Option<i64>>(11)?.unwrap_or(0),
                extract_duration_ms: row.get::<_, Option<i64>>(12)?.unwrap_or(0),
                mcp_duration_ms: row.get::<_, Option<i64>>(13)?.unwrap_or(0),
                stage_breakdown_json: row
                    .get::<_, Option<String>>(14)?
                    .unwrap_or_else(|| "{}".to_string()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let mut row_verdicts: HashMap<i64, Vec<String>> = HashMap::new();
    for fb in &all_feedback {
        if let Some(rid) = fb.row_id {
            row_verdicts.entry(rid).or_default().push(fb.verdict.clone());
        }
    }

    fn ensure_bucket(
        combo_buckets: &mut HashMap<String, ComboBucket>,
        combo_order: &mut Vec<String>,
        combo_key: &str,
        ocr_m: &str,
        dec_m: Option<&str>,
        ext_m: &str,
        lay_m: Option<&str>,
    ) {
        if !combo_buckets.contains_key(combo_key) {
            combo_order.push(combo_key.to_string());
            combo_buckets.insert(
                combo_key.to_string(),
                ComboBucket {
                    model_combo: combo_key.to_string(),
                    ocr_model: ocr_m.to_string(),
                    decision_model: dec_m.map(|s| s.to_string()),
                    extract_model: ext_m.to_string(),
                    layout_model: lay_m
                        .filter(|s| !s.is_empty())
                        .unwrap_or("opencv:angled_quad_row_slicer")
                        .to_string(),
                    correct_count: 0,
                    incorrect_count: 0,
                    total_rated: 0,
                    timing_samples: Vec::new(),
                    seen_row_keys: HashSet::new(),
                },
            );
        }
    }

    let base_key = (
        data_dir.to_string(),
        tr_count,
        tr_max_id,
        defaults.default_ocr.clone(),
        defaults.default_decision.clone(),
        defaults.default_extract.clone(),
        defaults.default_layout.clone(),
    );

    let trace_base: CachedTraceBase = {
        let mut cached_base: Option<CachedTraceBase> = None;
        if let Ok(guard) = TRACE_BASE_CACHE.lock() {
            if let Some(b) = guard.get(&base_key) {
                cached_base = Some(b.clone());
            }
        }
        if let Some(b) = cached_base {
            b
        } else {
            let mut tr_stmt = conn.prepare(
                "SELECT scan_id, row_id, stage, model_or_tool, status, duration_ms,
                        prompt_tokens, completion_tokens, reasoning_tokens, image_path
                 FROM workflowtrace
                 ORDER BY id ASC",
            )?;
            let all_traces: Vec<ProjectedTrace> = tr_stmt
                .query_map([], |row| {
                    Ok(ProjectedTrace {
                        scan_id: row.get(0)?,
                        row_id: row.get(1)?,
                        stage: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        model_or_tool: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        status: row
                            .get::<_, Option<String>>(4)?
                            .unwrap_or_else(|| "ok".to_string()),
                        duration_ms: row.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        prompt_tokens: row.get::<_, Option<i64>>(6)?.unwrap_or(0),
                        completion_tokens: row.get::<_, Option<i64>>(7)?.unwrap_or(0),
                        reasoning_tokens: row.get::<_, Option<i64>>(8)?.unwrap_or(0),
                        image_path: row.get(9)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;

            let mut scan_layout_map: HashMap<i64, ProjectedTrace> = HashMap::new();
            let mut scan_header_ocr_map: HashMap<i64, ProjectedTrace> = HashMap::new();
            let mut scan_row_ids_map: HashMap<i64, HashSet<i64>> = HashMap::new();
            let mut row_traces_map: HashMap<i64, Vec<ProjectedTrace>> = HashMap::new();
            let mut row_traces_order: Vec<i64> = Vec::new();

            for t in &all_traces {
                if let (Some(sid), Some(rid)) = (t.scan_id, t.row_id) {
                    scan_row_ids_map.entry(sid).or_default().insert(rid);
                }
                if let Some(rid) = t.row_id {
                    let entry = row_traces_map.entry(rid).or_insert_with(|| {
                        row_traces_order.push(rid);
                        Vec::new()
                    });
                    entry.push(t.clone());
                } else if let Some(sid) = t.scan_id {
                    let cg = canonical_stage_group_internal(&t.stage);
                    if cg == "layout" {
                        scan_layout_map.insert(sid, t.clone());
                    } else if t.stage == "ocr_header" {
                        scan_header_ocr_map.insert(sid, t.clone());
                    }
                }
            }

            let mut base_buckets: HashMap<String, ComboBucket> = HashMap::new();
            let mut base_order: Vec<String> = Vec::new();

            for &r_id in &row_traces_order {
                let r_traces = match row_traces_map.get(&r_id) {
                    Some(v) if !v.is_empty() => v,
                    _ => continue,
                };
                let s_id = r_traces.last().and_then(|t| t.scan_id);
                let s_rows_cnt = s_id
                    .and_then(|sid| scan_row_ids_map.get(&sid).map(|set| set.len()))
                    .unwrap_or(1)
                    .max(1) as i64;

                let lay_tr = s_id.and_then(|sid| scan_layout_map.get(&sid));
                let scan_lay_ms = lay_tr.map(|t| t.duration_ms).unwrap_or(0);
                let lay_ms = if scan_lay_ms > 0 {
                    ((scan_lay_ms as f64) / (s_rows_cnt as f64)).round() as i64
                } else {
                    0
                };
                let lay_m = lay_tr
                    .map(|t| t.model_or_tool.as_str())
                    .unwrap_or("opencv:angled_quad_row_slicer");

                let hdr_tr = s_id.and_then(|sid| scan_header_ocr_map.get(&sid));
                let scan_hdr_ms = hdr_tr.map(|t| t.duration_ms).unwrap_or(0);
                let hdr_ms = if scan_hdr_ms > 0 {
                    ((scan_hdr_ms as f64) / (s_rows_cnt as f64)).round() as i64
                } else {
                    0
                };

                let mut dedup_ocr: HashMap<(String, String), &ProjectedTrace> = HashMap::new();
                let mut dedup_ocr_order: Vec<(String, String)> = Vec::new();
                for t in r_traces {
                    if canonical_stage_group_internal(&t.stage) == "ocr" && t.stage != "ocr_header"
                    {
                        let key = (
                            t.stage.clone(),
                            t.image_path.clone().unwrap_or_default(),
                        );
                        if !dedup_ocr.contains_key(&key) {
                            dedup_ocr_order.push(key.clone());
                        }
                        dedup_ocr.insert(key, t);
                    }
                }
                let ocr_list: Vec<&ProjectedTrace> = dedup_ocr_order
                    .iter()
                    .filter_map(|k| dedup_ocr.get(k).copied())
                    .collect();

                let ocr_l_ms: i64 = ocr_list
                    .iter()
                    .filter(|t| {
                        matches!(
                            t.stage.as_str(),
                            "ocr_left" | "ocr" | "census_overview_ocr"
                        )
                    })
                    .map(|t| t.duration_ms)
                    .sum();
                let ocr_r_ms: i64 = ocr_list
                    .iter()
                    .filter(|t| t.stage == "ocr_right")
                    .map(|t| t.duration_ms)
                    .sum();
                let ocr_tot_ms = ocr_l_ms + ocr_r_ms + hdr_ms;
                let ocr_m = defaults.normalize_model_tag(
                    ocr_list
                        .last()
                        .map(|t| t.model_or_tool.as_str())
                        .or_else(|| hdr_tr.map(|h| h.model_or_tool.as_str())),
                    "ocr",
                );

                let dec_tr = r_traces
                    .iter()
                    .rev()
                    .find(|t| canonical_stage_group_internal(&t.stage) == "decision");
                let dec_ms = dec_tr.map(|t| t.duration_ms).unwrap_or(0);
                let dec_m = if let Some(dt) = dec_tr {
                    if dt.model_or_tool.starts_with("deterministic:") {
                        dt.model_or_tool.clone()
                    } else {
                        defaults.normalize_model_tag(Some(&dt.model_or_tool), "decision")
                    }
                } else {
                    "deterministic:census_gazetteer".to_string()
                };
                let is_census_row = dec_tr.is_none() || dec_m.starts_with("deterministic:");

                let ext_tr = r_traces
                    .iter()
                    .rev()
                    .find(|t| canonical_stage_group_internal(&t.stage) == "extract");
                let ext_ms = ext_tr.map(|t| t.duration_ms).unwrap_or(0);
                let ext_m = defaults
                    .normalize_model_tag(ext_tr.map(|t| t.model_or_tool.as_str()), "extract");

                let mcp_ms: i64 = r_traces
                    .iter()
                    .filter(|t| canonical_stage_group_internal(&t.stage) == "mcp")
                    .map(|t| t.duration_ms)
                    .sum();

                let tot_ms = lay_ms + ocr_tot_ms + dec_ms + ext_ms + mcp_ms;
                let combo_key = format_model_combo_internal(
                    defaults,
                    &ocr_m,
                    &ext_m,
                    Some(&dec_m),
                    is_census_row,
                );

                let p_tok: i64 = r_traces.iter().map(|t| t.prompt_tokens).sum();
                let c_tok: i64 = r_traces.iter().map(|t| t.completion_tokens).sum();
                let r_tok: i64 = r_traces.iter().map(|t| t.reasoning_tokens).sum();

                ensure_bucket(
                    &mut base_buckets,
                    &mut base_order,
                    &combo_key,
                    &ocr_m,
                    Some(&dec_m),
                    &ext_m,
                    Some(lay_m),
                );
                if let Some(bucket) = base_buckets.get_mut(&combo_key) {
                    bucket.seen_row_keys.insert((r_id, combo_key.clone()));
                    bucket.timing_samples.push(TimingSample {
                        layout_ms: lay_ms,
                        scan_layout_ms: scan_lay_ms,
                        ocr_ms: ocr_tot_ms,
                        ocr_header_ms: hdr_ms,
                        ocr_left_ms: ocr_l_ms,
                        ocr_right_ms: ocr_r_ms,
                        decision_ms: dec_ms,
                        extract_ms: ext_ms,
                        mcp_ms,
                        total_ms: tot_ms,
                        prompt_tokens: p_tok,
                        completion_tokens: c_tok,
                        reasoning_tokens: r_tok,
                    });
                }
            }

            let mut step_groups: HashMap<(String, String), StepGroup> = HashMap::new();
            let mut step_keys_order: Vec<(String, String)> = Vec::new();

            for t in &all_traces {
                if t.stage == "debug_playground" {
                    continue;
                }
                let cg = canonical_stage_group_internal(&t.stage);
                let norm_mod = if cg == "layout"
                    || cg == "mcp"
                    || t.model_or_tool.starts_with("opencv:")
                    || t.model_or_tool.starts_with("deterministic:")
                {
                    t.model_or_tool.clone()
                } else {
                    let hint = if matches!(cg, "ocr" | "decision" | "extract") {
                        cg
                    } else {
                        "extract"
                    };
                    defaults.normalize_model_tag(Some(&t.model_or_tool), hint)
                };
                let key = (t.stage.clone(), norm_mod.clone());
                if !step_groups.contains_key(&key) {
                    step_keys_order.push(key.clone());
                    step_groups.insert(
                        key.clone(),
                        StepGroup {
                            stage: t.stage.clone(),
                            canonical_stage: cg.to_string(),
                            model_or_tool: norm_mod,
                            durations: Vec::new(),
                            ok_count: 0,
                            error_count: 0,
                            prompt_tokens: Vec::new(),
                            completion_tokens: Vec::new(),
                            reasoning_tokens: Vec::new(),
                            row_ids: HashSet::new(),
                        },
                    );
                }
                if let Some(sg) = step_groups.get_mut(&key) {
                    sg.durations.push(t.duration_ms);
                    if t.status == "ok" || t.status == "success" {
                        sg.ok_count += 1;
                    } else {
                        sg.error_count += 1;
                    }
                    sg.prompt_tokens.push(t.prompt_tokens);
                    sg.completion_tokens.push(t.completion_tokens);
                    sg.reasoning_tokens.push(t.reasoning_tokens);
                    if let Some(rid) = t.row_id {
                        sg.row_ids.insert(rid);
                    } else if let Some(sid) = t.scan_id {
                        if let Some(rids) = scan_row_ids_map.get(&sid) {
                            for &rid in rids {
                                sg.row_ids.insert(rid);
                            }
                        }
                    }
                }
            }

            let mut steps_summary: Vec<CachedStepSummary> = Vec::with_capacity(step_keys_order.len());
            for key in &step_keys_order {
                if let Some(sg) = step_groups.get(key) {
                    let durs = &sg.durations;
                    let min_d = durs.iter().copied().min().unwrap_or(0);
                    let max_d = durs.iter().copied().max().unwrap_or(0);
                    let sum_d: i64 = durs.iter().sum();
                    steps_summary.push(CachedStepSummary {
                        stage: sg.stage.clone(),
                        canonical_stage: sg.canonical_stage.clone(),
                        model_or_tool: sg.model_or_tool.clone(),
                        calls_count: durs.len(),
                        ok_count: sg.ok_count,
                        error_count: sg.error_count,
                        avg_duration_ms: avg_i64(durs),
                        min_duration_ms: min_d,
                        max_duration_ms: max_d,
                        total_duration_ms: sum_d,
                        avg_prompt_tokens: avg_i64(&sg.prompt_tokens),
                        avg_completion_tokens: avg_i64(&sg.completion_tokens),
                        avg_reasoning_tokens: avg_i64(&sg.reasoning_tokens),
                        row_ids: sg.row_ids.clone(),
                    });
                }
            }
            steps_summary.sort_by(|a, b| {
                (stage_order(&a.stage), a.stage.as_str(), -(a.calls_count as i64)).cmp(&(
                    stage_order(&b.stage),
                    b.stage.as_str(),
                    -(b.calls_count as i64),
                ))
            });

            let built = CachedTraceBase {
                total_rows_analyzed: row_traces_map.len(),
                combo_buckets: base_buckets,
                combo_order: base_order,
                steps: steps_summary,
            };
            if let Ok(mut guard) = TRACE_BASE_CACHE.lock() {
                if guard.len() > 8 {
                    guard.clear();
                }
                guard.insert(base_key, built.clone());
            }
            built
        }
    };

    let mut combo_buckets = trace_base.combo_buckets;
    let mut combo_order = trace_base.combo_order;

    for fb in &all_feedback {
        let ocr_m = defaults.normalize_model_tag(fb.ocr_model.as_deref(), "ocr");
        let ext_m = defaults.normalize_model_tag(fb.extract_model.as_deref(), "extract");
        let dec_m = fb
            .decision_model
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "deterministic:census_gazetteer".to_string());
        let combo_key = if !fb.model_combo.is_empty() {
            fb.model_combo.clone()
        } else {
            format_model_combo_internal(
                defaults,
                &ocr_m,
                &ext_m,
                Some(&dec_m),
                dec_m.starts_with("deterministic:"),
            )
        };
        ensure_bucket(
            &mut combo_buckets,
            &mut combo_order,
            &combo_key,
            &ocr_m,
            Some(&dec_m),
            &ext_m,
            fb.layout_model.as_deref(),
        );
        if let Some(bucket) = combo_buckets.get_mut(&combo_key) {
            bucket.total_rated += 1;
            if fb.verdict == "correct" {
                bucket.correct_count += 1;
            } else {
                bucket.incorrect_count += 1;
            }
            let sb: Value = serde_json::from_str(&fb.stage_breakdown_json).unwrap_or(json!({}));
            let sample_row_id = fb.row_id.unwrap_or(-fb.id);
            let sample_key = (sample_row_id, combo_key.clone());
            if !bucket.seen_row_keys.contains(&sample_key)
                && (fb.total_duration_ms > 0 || bucket.timing_samples.is_empty())
            {
                bucket.seen_row_keys.insert(sample_key);
                let get_sb = |k: &str| sb.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
                bucket.timing_samples.push(TimingSample {
                    layout_ms: if fb.layout_duration_ms != 0 {
                        fb.layout_duration_ms
                    } else {
                        get_sb("layout_ms")
                    },
                    scan_layout_ms: {
                        let sl = get_sb("scan_layout_ms");
                        if sl != 0 {
                            sl
                        } else {
                            fb.layout_duration_ms
                        }
                    },
                    ocr_ms: if fb.ocr_duration_ms != 0 {
                        fb.ocr_duration_ms
                    } else {
                        get_sb("ocr_ms")
                    },
                    ocr_header_ms: get_sb("ocr_header_ms"),
                    ocr_left_ms: {
                        let ol = get_sb("ocr_left_ms");
                        if ol != 0 {
                            ol
                        } else {
                            fb.ocr_duration_ms
                        }
                    },
                    ocr_right_ms: get_sb("ocr_right_ms"),
                    decision_ms: if fb.decision_duration_ms != 0 {
                        fb.decision_duration_ms
                    } else {
                        get_sb("decision_ms")
                    },
                    extract_ms: if fb.extract_duration_ms != 0 {
                        fb.extract_duration_ms
                    } else {
                        get_sb("extract_ms")
                    },
                    mcp_ms: if fb.mcp_duration_ms != 0 {
                        fb.mcp_duration_ms
                    } else {
                        get_sb("mcp_ms")
                    },
                    total_ms: if fb.total_duration_ms != 0 {
                        fb.total_duration_ms
                    } else {
                        get_sb("total_ms")
                    },
                    prompt_tokens: get_sb("prompt_tokens"),
                    completion_tokens: get_sb("completion_tokens"),
                    reasoning_tokens: get_sb("reasoning_tokens"),
                });
            }
        }
    }

    let active_ocr_m = defaults.normalize_model_tag(None, "ocr");
    let active_ext_m = defaults.normalize_model_tag(None, "extract");
    let active_dec_m = defaults.normalize_model_tag(None, "decision");
    let active_census_combo =
        format_model_combo_internal(defaults, &active_ocr_m, &active_ext_m, None, true);
    let active_matrika_combo = format_model_combo_internal(
        defaults,
        &active_ocr_m,
        &active_ext_m,
        Some(&active_dec_m),
        false,
    );

    if combo_buckets.is_empty() {
        ensure_bucket(
            &mut combo_buckets,
            &mut combo_order,
            &active_census_combo,
            &active_ocr_m,
            Some("deterministic:census_gazetteer"),
            &active_ext_m,
            Some("opencv:census_sheet_slicer"),
        );
    }

    let mut model_combos_out: Vec<Value> = Vec::new();
    let mut all_samples_flat: Vec<TimingSample> = Vec::new();

    for combo_key in &combo_order {
        let b = match combo_buckets.get(combo_key) {
            Some(b) => b,
            None => continue,
        };
        let non_zero: Vec<TimingSample> = b
            .timing_samples
            .iter()
            .copied()
            .filter(|s| s.total_ms > 0)
            .collect();
        let samples_ref: &[TimingSample] = if non_zero.is_empty() {
            &b.timing_samples
        } else {
            &non_zero
        };
        all_samples_flat.extend_from_slice(samples_ref);

        let tot_rated = b.total_rated;
        let corr = b.correct_count;
        let incorr = b.incorrect_count;
        let rate_pct: Value = if tot_rated > 0 {
            let r = ((1000.0 * (corr as f64) / (tot_rated as f64)).round()) / 10.0;
            json!(r)
        } else {
            Value::Null
        };

        let tot_ms_list: Vec<i64> = samples_ref.iter().map(|s| s.total_ms).collect();
        let min_tot = tot_ms_list.iter().copied().min().unwrap_or(0);
        let max_tot = tot_ms_list.iter().copied().max().unwrap_or(0);
        let is_active =
            b.model_combo == active_census_combo || b.model_combo == active_matrika_combo;
        let rows_proc = (b.timing_samples.len() as i64).max(tot_rated);

        model_combos_out.push(json!({
            "model_combo": b.model_combo,
            "ocr_model": b.ocr_model,
            "decision_model": b.decision_model,
            "extract_model": b.extract_model,
            "layout_model": b.layout_model,
            "is_active_combo": is_active,
            "total_rated": tot_rated,
            "correct_count": corr,
            "incorrect_count": incorr,
            "correctness_rate_pct": rate_pct,
            "rows_processed": rows_proc,
            "avg_total_ms": avg_i64(&tot_ms_list),
            "min_total_ms": min_tot,
            "max_total_ms": max_tot,
            "steps": {
                "layout_ms": avg_i64(&samples_ref.iter().map(|s| s.layout_ms).collect::<Vec<_>>()),
                "scan_layout_ms": avg_i64(&samples_ref.iter().map(|s| s.scan_layout_ms).collect::<Vec<_>>()),
                "ocr_ms": avg_i64(&samples_ref.iter().map(|s| s.ocr_ms).collect::<Vec<_>>()),
                "ocr_header_ms": avg_i64(&samples_ref.iter().map(|s| s.ocr_header_ms).collect::<Vec<_>>()),
                "ocr_left_ms": avg_i64(&samples_ref.iter().map(|s| s.ocr_left_ms).collect::<Vec<_>>()),
                "ocr_right_ms": avg_i64(&samples_ref.iter().map(|s| s.ocr_right_ms).collect::<Vec<_>>()),
                "decision_ms": avg_i64(&samples_ref.iter().map(|s| s.decision_ms).collect::<Vec<_>>()),
                "extract_ms": avg_i64(&samples_ref.iter().map(|s| s.extract_ms).collect::<Vec<_>>()),
                "mcp_ms": avg_i64(&samples_ref.iter().map(|s| s.mcp_ms).collect::<Vec<_>>()),
            },
            "avg_tokens": {
                "prompt_tokens": avg_i64(&samples_ref.iter().map(|s| s.prompt_tokens).collect::<Vec<_>>()),
                "completion_tokens": avg_i64(&samples_ref.iter().map(|s| s.completion_tokens).collect::<Vec<_>>()),
                "reasoning_tokens": avg_i64(&samples_ref.iter().map(|s| s.reasoning_tokens).collect::<Vec<_>>()),
            }
        }));
    }

    model_combos_out.sort_by(|a, b| {
        let a_rated = a.get("total_rated").and_then(|v| v.as_i64()).unwrap_or(0);
        let b_rated = b.get("total_rated").and_then(|v| v.as_i64()).unwrap_or(0);
        let a_has = if a_rated > 0 { 0 } else { 1 };
        let b_has = if b_rated > 0 { 0 } else { 1 };
        let a_act = if a
            .get("is_active_combo")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            0
        } else {
            1
        };
        let b_act = if b
            .get("is_active_combo")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            0
        } else {
            1
        };
        let a_rows = a.get("rows_processed").and_then(|v| v.as_i64()).unwrap_or(0);
        let b_rows = b.get("rows_processed").and_then(|v| v.as_i64()).unwrap_or(0);
        (a_has, a_act, -a_rated, -a_rows).cmp(&(b_has, b_act, -b_rated, -b_rows))
    });

    let mut individual_steps_out: Vec<Value> = Vec::with_capacity(trace_base.steps.len());
    for sg in &trace_base.steps {
        let mut corr_r = 0_i64;
        let mut incorr_r = 0_i64;
        for rid in &sg.row_ids {
            if let Some(verdicts) = row_verdicts.get(rid) {
                for v in verdicts {
                    if v == "correct" {
                        corr_r += 1;
                    } else {
                        incorr_r += 1;
                    }
                }
            }
        }
        let tot_r = corr_r + incorr_r;
        let rate_pct: Value = if tot_r > 0 {
            let r = ((1000.0 * (corr_r as f64) / (tot_r as f64)).round()) / 10.0;
            json!(r)
        } else {
            Value::Null
        };

        individual_steps_out.push(json!({
            "stage": sg.stage,
            "canonical_stage": sg.canonical_stage,
            "model_or_tool": sg.model_or_tool,
            "calls_count": sg.calls_count,
            "ok_count": sg.ok_count,
            "error_count": sg.error_count,
            "avg_duration_ms": sg.avg_duration_ms,
            "min_duration_ms": sg.min_duration_ms,
            "max_duration_ms": sg.max_duration_ms,
            "total_duration_ms": sg.total_duration_ms,
            "avg_prompt_tokens": sg.avg_prompt_tokens,
            "avg_completion_tokens": sg.avg_completion_tokens,
            "avg_reasoning_tokens": sg.avg_reasoning_tokens,
            "rated_rows_count": tot_r,
            "correct_rows_count": corr_r,
            "incorrect_rows_count": incorr_r,
            "correctness_rate_pct": rate_pct,
        }));
    }

    let total_fb = all_feedback.len() as i64;
    let total_corr = all_feedback.iter().filter(|f| f.verdict == "correct").count() as i64;
    let total_incorr = total_fb - total_corr;
    let overall_rate: Value = if total_fb > 0 {
        let r = ((1000.0 * (total_corr as f64) / (total_fb as f64)).round()) / 10.0;
        json!(r)
    } else {
        Value::Null
    };

    let md_path = Path::new(data_dir).join("ocr_feedback_debug_report.md");
    let json_path = Path::new(data_dir).join("ocr_feedback_debug_report.json");

    let out = json!({
        "summary": {
            "total_feedback": total_fb,
            "correct_count": total_corr,
            "incorrect_count": total_incorr,
            "correctness_rate_pct": overall_rate,
            "total_rows_analyzed": trace_base.total_rows_analyzed,
            "avg_total_duration_ms": avg_i64(&all_samples_flat.iter().map(|s| s.total_ms).collect::<Vec<_>>()),
            "avg_steps_ms": {
                "layout_ms": avg_i64(&all_samples_flat.iter().map(|s| s.layout_ms).collect::<Vec<_>>()),
                "ocr_ms": avg_i64(&all_samples_flat.iter().map(|s| s.ocr_ms).collect::<Vec<_>>()),
                "ocr_header_ms": avg_i64(&all_samples_flat.iter().map(|s| s.ocr_header_ms).collect::<Vec<_>>()),
                "ocr_left_ms": avg_i64(&all_samples_flat.iter().map(|s| s.ocr_left_ms).collect::<Vec<_>>()),
                "ocr_right_ms": avg_i64(&all_samples_flat.iter().map(|s| s.ocr_right_ms).collect::<Vec<_>>()),
                "decision_ms": avg_i64(&all_samples_flat.iter().map(|s| s.decision_ms).collect::<Vec<_>>()),
                "extract_ms": avg_i64(&all_samples_flat.iter().map(|s| s.extract_ms).collect::<Vec<_>>()),
                "mcp_ms": avg_i64(&all_samples_flat.iter().map(|s| s.mcp_ms).collect::<Vec<_>>()),
            },
            "report_files": {
                "markdown_path": md_path.to_string_lossy(),
                "json_path": json_path.to_string_lossy(),
                "relative_markdown_path": "data/ocr_feedback_debug_report.md",
                "relative_json_path": "data/ocr_feedback_debug_report.json",
            }
        },
        "model_combos": model_combos_out,
        "individual_steps": individual_steps_out,
    });
    if let Ok(mut guard) = STATS_CACHE.lock() {
        if guard.len() > 16 {
            guard.clear();
        }
        guard.insert(cache_key, out.clone());
    }
    Ok(out)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn get_feedback_statistics_rs(
    py: Python<'_>,
    db_path: String,
    data_dir: String,
    default_ocr: String,
    default_decision: String,
    default_extract: String,
    default_layout: String,
    known_local_models: Vec<String>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let known_local_lower: HashSet<String> = known_local_models
            .into_iter()
            .map(|s| s.to_ascii_lowercase())
            .collect();
        let defaults = ModelDefaults {
            default_ocr,
            default_decision,
            default_extract,
            default_layout,
            known_local_lower,
        };
        let stats = compute_feedback_statistics_from_conn(&conn, &defaults, &data_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string(&stats).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
fn system_status_counts_rs(py: Python<'_>, db_path: String) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let scans: i64 = conn
            .query_row("SELECT COUNT(*) FROM scanimage", [], |r| r.get(0))
            .unwrap_or(0);
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM recordrow", [], |r| r.get(0))
            .unwrap_or(0);
        let facts: i64 = conn
            .query_row("SELECT COUNT(*) FROM personfact", [], |r| r.get(0))
            .unwrap_or(0);
        let open_questions: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM questionitem WHERE status = 'open'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let (new_fs, unreviewed_fs): (i64, i64) = conn
            .query_row(
                "SELECT COUNT(id), COALESCE(SUM(CASE WHEN reviewed_by_user = 0 THEN 1 ELSE 0 END), 0)
                 FROM familysearchmatch
                 WHERE fs_person_id IS NOT NULL AND trim(fs_person_id) != '' AND confidence_score > 0",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap_or((0, 0));
        let census_sheets: i64 = conn
            .query_row("SELECT COUNT(*) FROM censussheet", [], |r| r.get(0))
            .unwrap_or(0);

        let out = json!({
            "scans": scans,
            "rows": rows,
            "facts": facts,
            "open_questions": open_questions,
            "new_to_familysearch": new_fs,
            "unreviewed_matches": unreviewed_fs,
            "fs_pending_matches": unreviewed_fs,
            "fs_matched": new_fs,
            "census_sheets": census_sheets,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
fn get_census_stats_rs(py: Python<'_>, db_path: String) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let (total_sheets, sheets_1930, sheets_1940, downloaded_sheets, processed_sheets): (
            i64,
            i64,
            i64,
            i64,
            i64,
        ) = conn
            .query_row(
                "SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN census_year = 1930 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN census_year = 1940 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN download_status IN ('downloaded', 'sliced', 'ocr_done', 'extracted', 'matched') THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN download_status IN ('extracted', 'matched') THEN 1 ELSE 0 END), 0)
                 FROM censussheet",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap_or((0, 0, 0, 0, 0));

        let census_facts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM personfact WHERE record_type = 'census'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let local_cross_matches: i64 = conn
            .query_row("SELECT COUNT(*) FROM localrecordmatch", [], |r| r.get(0))
            .unwrap_or(0);

        let mut muni_stmt = conn
            .prepare(
                "SELECT municipality, district, COUNT(id) as cnt
             FROM censussheet
             GROUP BY municipality, district
             ORDER BY cnt DESC
             LIMIT 200",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let top_municipalities: Vec<Value> = muni_stmt
            .query_map([], |r| {
                let m: Option<String> = r.get(0)?;
                let d: Option<String> = r.get(1)?;
                let c: i64 = r.get(2)?;
                Ok((m, d, c))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .filter_map(|res| res.ok())
            .filter_map(|(m, d, c)| {
                let m_str = m?;
                if m_str.is_empty() {
                    return None;
                }
                Some(json!({
                    "municipality": m_str,
                    "district": d,
                    "count": c,
                }))
            })
            .collect();

        let mut dist_stmt = conn
            .prepare(
                "SELECT district, COUNT(id)
             FROM censussheet
             WHERE district IS NOT NULL
             GROUP BY district
             ORDER BY district ASC",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let local_districts: Vec<Value> = dist_stmt
            .query_map([], |r| {
                let d: Option<String> = r.get(0)?;
                let c: i64 = r.get(1)?;
                Ok((d, c))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .filter_map(|res| res.ok())
            .filter_map(|(d, c)| {
                let d_str = d?;
                if d_str.is_empty() {
                    return None;
                }
                Some(json!({
                    "district": d_str,
                    "count": c,
                }))
            })
            .collect();

        let out = json!({
            "total_cataloged_sheets": total_sheets,
            "sheets_1930": sheets_1930,
            "sheets_1940": sheets_1940,
            "downloaded_sheets": downloaded_sheets,
            "processed_sheets": processed_sheets,
            "extracted_census_persons": census_facts,
            "local_matrika_cross_matches": local_cross_matches,
            "top_municipalities": top_municipalities,
            "local_districts": local_districts,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

mod feedback;
mod lexicon;

#[pymodule]
fn matrika_db_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fold_text_diacritics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(apply_digraphs_rs, m)?)?;
    m.add_function(wrap_pyfunction!(orthographic_fold_rs, m)?)?;
    m.add_function(wrap_pyfunction!(phonetic_skeleton_rs, m)?)?;
    m.add_function(wrap_pyfunction!(paleographic_distance_rs, m)?)?;
    m.add_function(wrap_pyfunction!(paleographic_similarity_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ocr_dp_distance_rs, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_edit_similarity_rs, m)?)?;
    m.add_function(wrap_pyfunction!(batch_score_candidates_against_lexicon_rs, m)?)?;
    m.add_function(wrap_pyfunction!(lexicon::fuzzy_token_similarity_rs, m)?)?;
    m.add_function(wrap_pyfunction!(lexicon::fuzzy_name_match_score_rs, m)?)?;
    m.add_function(wrap_pyfunction!(lexicon::score_person_facts_search_rs, m)?)?;
    m.add_function(wrap_pyfunction!(get_feedback_statistics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(system_status_counts_rs, m)?)?;
    m.add_function(wrap_pyfunction!(get_census_stats_rs, m)?)?;
    m.add_function(wrap_pyfunction!(feedback::build_feedback_lookup_maps_rs, m)?)?;
    m.add_function(wrap_pyfunction!(
        feedback::build_and_persist_llm_debug_report_rs,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(feedback::upsert_ocr_feedback_rs, m)?)?;
    m.add_function(wrap_pyfunction!(feedback::delete_ocr_feedback_rs, m)?)?;
    m.add_function(wrap_pyfunction!(feedback::reset_ocr_feedback_rs, m)?)?;
    Ok(())
}
