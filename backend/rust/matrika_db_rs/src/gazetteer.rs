use once_cell::sync::Lazy;
use pyo3::prelude::*;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::{
    apply_digraphs_internal, fold_text_diacritics_internal, paleographic_distance_internal,
    paleographic_similarity_internal, phonetic_skeleton_internal,
};
use crate::lexicon::{fuzzy_name_match_score_internal, resolve_given_cid_pub};

static CANONICAL_MUNI_SYNONYMS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (k, v) in [
        ("horny cepen", "horny cepen"),
        ("cepen horny", "horny cepen"),
        ("hor cepen", "horny cepen"),
        ("hr cepen", "horny cepen"),
        ("dolny cepen", "dolny cepen"),
        ("cepen dolny", "dolny cepen"),
        ("dol cepen", "dolny cepen"),
        ("dl cepen", "dolny cepen"),
        ("stredny cepen", "stredny cepen"),
        ("cepen stredny", "stredny cepen"),
        ("str cepen", "stredny cepen"),
        ("velke surovce", "velke surovce"),
        ("surovce velke", "velke surovce"),
        ("vel surovce", "velke surovce"),
        ("velke sulovce", "velke surovce"),
        ("sulovce velke", "velke surovce"),
        ("felke sluovce", "velke surovce"),
        ("male surovce", "male surovce"),
        ("surovce male", "male surovce"),
        ("mal surovce", "male surovce"),
        ("horne zelenice", "horne zelenice"),
        ("zelenice horne", "horne zelenice"),
        ("hornie zelenice", "horne zelenice"),
        ("zelenice hornie", "horne zelenice"),
        ("hor zelenice", "horne zelenice"),
        ("dolne zelenice", "dolne zelenice"),
        ("zelenice dolne", "dolne zelenice"),
        ("dolnie zelenice", "dolne zelenice"),
        ("zelenice dolnie", "dolne zelenice"),
        ("dol zelenice", "dolne zelenice"),
        ("velke ripnany", "velke ripnany"),
        ("ripnany velke", "velke ripnany"),
        ("vel ripnany", "velke ripnany"),
        ("male ripnany", "male ripnany"),
        ("ripnany male", "male ripnany"),
        ("mal ripnany", "male ripnany"),
        ("velke topolcany", "velke topolcany"),
        ("topolcany velke", "velke topolcany"),
        ("vel topolcany", "velke topolcany"),
    ] {
        m.insert(k, v);
    }
    m
});

static CANONICAL_MUNI_DISPLAY: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (k, v) in [
        ("horny cepen", "Horný Čepeň"),
        ("dolny cepen", "Dolný Čepeň"),
        ("stredny cepen", "Stredný Čepeň"),
        ("velke surovce", "Veľké Šúrovce"),
        ("male surovce", "Malé Šúrovce"),
        ("horne zelenice", "Horné Zelenice"),
        ("dolne zelenice", "Dolné Zelenice"),
        ("velke ripnany", "Veľké Ripňany"),
        ("male ripnany", "Malé Ripňany"),
        ("velke topolcany", "Veľké Topoľčany"),
    ] {
        m.insert(k, v);
    }
    m
});

static GENERIC_SETTLEMENT_KEYS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "",
        "dedina",
        "obec",
        "mlyn",
        "skupina domov",
        "zel strazny domcek",
        "zeleznicny strazny domcek",
        "strazny domcek",
        "dvor",
        "majer",
        "samota",
        "osada",
        "ulica",
    ]
    .into_iter()
    .collect()
});

static STREET_STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "ul", "ulica", "nam", "namestie", "osada", "cesta", "trieda", "st", "street",
    ]
    .into_iter()
    .collect()
});

pub static MUNI_HISTORICAL_DISTRICT_ALIASES: Lazy<HashMap<&'static str, HashSet<&'static str>>> =
    Lazy::new(|| {
        let mut m = HashMap::new();
        let trn_gal_ser: HashSet<&'static str> =
            ["trnava", "galanta", "sered", "772", "712"].into_iter().collect();
        for k in [
            "horny cepen",
            "dolny cepen",
            "stredny cepen",
            "velke surovce",
            "male surovce",
            "surovce",
            "valtasur",
            "varov sur",
        ] {
            m.insert(k, trn_gal_ser.clone());
        }
        let trn_gal: HashSet<&'static str> = ["trnava", "galanta", "772", "712"].into_iter().collect();
        for k in ["majcichov", "vlckovce", "opoj", "krizovany nad dudvahom"] {
            m.insert(k, trn_gal.clone());
        }
        let gal_trn_hlo: HashSet<&'static str> =
            ["galanta", "trnava", "hlohovec", "712", "772", "715"]
                .into_iter()
                .collect();
        for k in [
            "zemianske sady",
            "salgocka",
            "horne zelenice",
            "dolne zelenice",
        ] {
            m.insert(k, gal_trn_hlo.clone());
        }
        m.insert(
            "siladice",
            ["hlohovec", "trnava", "715", "772"].into_iter().collect(),
        );
        let nit_top_pie: HashSet<&'static str> =
            ["nitra", "topolcany", "piestany", "740", "770", "748"]
                .into_iter()
                .collect();
        m.insert("velke ripnany", nit_top_pie.clone());
        m.insert("male ripnany", nit_top_pie);
        m
    });

pub fn canonical_municipality_key_internal(municipality: Option<&str>) -> String {
    let raw = match municipality {
        Some(s) if !s.trim().is_empty() => s,
        _ => return String::new(),
    };
    let folded = fold_text_diacritics_internal(raw);
    let mut words: Vec<&str> = Vec::new();
    for tok in folded.split(|c: char| !c.is_ascii_alphanumeric()) {
        if !tok.is_empty() {
            let mapped = match tok {
                "hornie" => "horne",
                "dolnie" => "dolne",
                _ => tok,
            };
            words.push(mapped);
        }
    }
    if words.is_empty() {
        return String::new();
    }
    let cleaned = words.join(" ");
    if let Some(&syn) = CANONICAL_MUNI_SYNONYMS.get(cleaned.as_str()) {
        return syn.to_string();
    }
    cleaned
}

pub fn canonical_municipality_display_internal(municipality: Option<&str>) -> Option<String> {
    let raw = municipality?.trim();
    if raw.is_empty() {
        return None;
    }
    let m_key = canonical_municipality_key_internal(Some(raw));
    if let Some(&disp) = CANONICAL_MUNI_DISPLAY.get(m_key.as_str()) {
        return Some(disp.to_string());
    }
    Some(raw.to_string())
}

pub fn parse_house_number_int_internal(house_no: Option<&str>) -> Option<i64> {
    let raw = house_no?.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.contains('/') {
        for part in raw.rsplit('/') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            let digits: String = p
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(n) = digits.parse::<i64>() {
                return Some(n);
            }
        }
    }
    let digits: String = raw
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<i64>().ok()
}

fn strip_hn_prefix(raw: &str) -> &str {
    let mut s = raw.trim();
    let low = s.to_lowercase();
    if low.starts_with("dom ") {
        s = s[4..].trim_start();
    }
    let low2 = s.to_lowercase();
    for p in ["číslo", "cislo", "nro.", "nro", "no.", "no", "č.", "c.", "č", "c", "№", "#"] {
        if low2.starts_with(p) {
            let rest = &s[p.len()..];
            if rest.is_empty()
                || rest.starts_with(|c: char| c.is_whitespace() || c.is_ascii_digit() || c == '.')
            {
                s = rest.trim_start_matches(['.', ' ']);
                break;
            }
        }
    }
    s.trim()
}

pub fn normalize_house_number_internal(house_no: Option<&str>) -> String {
    let raw_in = match house_no {
        Some(s) => s.trim(),
        None => return String::new(),
    };
    if raw_in.is_empty() {
        return String::new();
    }
    let mut raw = strip_hn_prefix(raw_in);
    if raw.contains('/') {
        for p in raw.rsplit('/') {
            let pt = p.trim_matches([' ', '-', '.']);
            if pt.chars().any(|c| c.is_ascii_digit()) {
                raw = pt;
                break;
            }
        }
    }
    let cleaned = raw
        .trim_matches(|c: char| matches!(c, ' ' | '.' | ',' | ';' | '-' | '/'))
        .to_lowercase();
    if !cleaned.chars().any(|c| c.is_ascii_digit()) {
        return String::new();
    }

    let bytes = cleaned.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() && !bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    if idx >= bytes.len() {
        return String::new();
    }
    let start_digits = idx;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    let digits_slice = &cleaned[start_digits..idx];
    let num_val = digits_slice.parse::<u64>().unwrap_or(0);
    let mut suffix = String::new();
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx < bytes.len() && bytes[idx].is_ascii_lowercase() {
        suffix.push(bytes[idx] as char);
    }
    format!("{}{}", num_val, suffix)
}

pub fn normalize_street_query_internal(raw: Option<&str>) -> (String, Vec<String>) {
    let s = match raw {
        Some(v) if !v.trim().is_empty() => v.trim(),
        _ => return (String::new(), Vec::new()),
    };
    let folded = fold_text_diacritics_internal(s);
    let mut tokens: Vec<String> = Vec::new();
    for tok in folded.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | ';' | '.' | '/' | '(' | ')' | '-' | '–' | '—')
    }) {
        if !tok.is_empty() && !STREET_STOPWORDS.contains(tok) {
            tokens.push(tok.to_string());
        }
    }
    let cleaned = if !tokens.is_empty() {
        tokens.join(" ")
    } else {
        folded.trim().to_string()
    };
    (cleaned, tokens)
}

pub fn text_matches_street_internal(
    candidates: &[Option<String>],
    street_query: Option<&str>,
) -> bool {
    let sq = match street_query {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return true,
    };
    let (q_clean, q_tokens) = normalize_street_query_internal(Some(sq));
    if q_clean.is_empty() {
        return true;
    }
    let q_canon = canonical_municipality_key_internal(Some(&q_clean));
    let mut valid_cands: Vec<Option<String>> = Vec::new();

    for opt_c in candidates {
        let c_str = match opt_c {
            Some(ref s) if !s.trim().is_empty() => s.trim(),
            _ => continue,
        };
        if matches!(c_str, "-" | "—" | "?") {
            continue;
        }
        let c_fold = fold_text_diacritics_internal(c_str);
        let (c_clean, _) = normalize_street_query_internal(Some(c_str));
        if c_clean.is_empty() && c_fold.is_empty() {
            continue;
        }
        let eff_c = if !c_clean.is_empty() {
            c_clean.clone()
        } else {
            c_fold.clone()
        };
        valid_cands.push(Some(eff_c));
        if (!c_clean.is_empty() && c_clean.contains(&q_clean)) || c_fold.contains(&q_clean) {
            return true;
        }
        if !q_tokens.is_empty() && q_tokens.iter().all(|t| c_fold.contains(t.as_str())) {
            return true;
        }
        let c_canon = canonical_municipality_key_internal(Some(if !c_clean.is_empty() {
            &c_clean
        } else {
            c_str
        }));
        if !q_canon.is_empty()
            && !c_canon.is_empty()
            && (q_canon == c_canon || c_canon.contains(&q_canon))
        {
            return true;
        }
    }

    if !valid_cands.is_empty() && q_clean.len() >= 4 {
        if fuzzy_name_match_score_internal(&q_clean, &valid_cands, 0.82) >= 0.82 {
            return true;
        }
    }
    false
}

pub fn is_compatible_settlement_internal(
    overview_settlement: Option<&str>,
    candidate_street: Option<&str>,
    municipality: Option<&str>,
) -> bool {
    let ov_key = canonical_municipality_key_internal(overview_settlement);
    let st_key = canonical_municipality_key_internal(candidate_street);
    let muni_key = canonical_municipality_key_internal(municipality);

    if ov_key.is_empty()
        || GENERIC_SETTLEMENT_KEYS.contains(ov_key.as_str())
        || (!muni_key.is_empty() && ov_key == muni_key)
    {
        return true;
    }
    if st_key.is_empty()
        || GENERIC_SETTLEMENT_KEYS.contains(st_key.as_str())
        || (!muni_key.is_empty() && st_key == muni_key)
    {
        return true;
    }
    ov_key == st_key
}

pub fn can_match_overview_house_to_sheet_internal(
    ov_sheet_id: Option<i64>,
    ov_year: Option<i64>,
    ov_muni_raw: Option<&str>,
    ov_mkey: Option<&str>,
    ov_hn_raw: Option<&str>,
    cand_id: Option<i64>,
    cand_type: Option<&str>,
    cand_year: Option<i64>,
    cand_muni_raw: Option<&str>,
    cand_mkey: Option<&str>,
    cand_hn_raw: Option<&str>,
) -> bool {
    let ct = cand_type.unwrap_or("household");
    if ct == "overview" || ct == "municipal_overview" {
        return false;
    }
    if let (Some(oid), Some(cid)) = (ov_sheet_id, cand_id) {
        if oid == cid {
            return false;
        }
    }
    if let (Some(oy), Some(cy)) = (ov_year, cand_year) {
        if oy != cy {
            return false;
        }
    }
    let ov_muni = canonical_municipality_key_internal(ov_muni_raw);
    let cand_muni = canonical_municipality_key_internal(cand_muni_raw);
    if ov_muni.is_empty() || cand_muni.is_empty() || ov_muni != cand_muni {
        return false;
    }
    if let (Some(omk), Some(cmk)) = (
        ov_mkey.map(|s| s.trim()).filter(|s| !s.is_empty()),
        cand_mkey.map(|s| s.trim()).filter(|s| !s.is_empty()),
    ) {
        if omk != cmk {
            return false;
        }
    }
    let ov_hn = normalize_house_number_internal(ov_hn_raw);
    let cand_hn = normalize_house_number_internal(cand_hn_raw);
    if ov_hn.is_empty() || cand_hn.is_empty() {
        return false;
    }
    if ov_hn != cand_hn {
        let cr = cand_hn_raw.unwrap_or("");
        if cr.contains(',') && ov_hn.bytes().all(|b| b.is_ascii_digit()) {
            let base_cand = cand_hn.trim_end_matches(|c: char| c.is_ascii_lowercase());
            if base_cand == ov_hn {
                return true;
            }
        }
        return false;
    }
    true
}

pub fn sheet_matches_municipality_internal(
    sheet_muni: &str,
    sheet_muni_folded: &str,
    sheet_muni_key: Option<&str>,
    municipality: Option<&str>,
) -> bool {
    let m_val = match municipality {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return true,
    };
    if let Some(mk) = sheet_muni_key {
        if !mk.is_empty() && mk == m_val {
            return true;
        }
    }
    let m_folded = fold_text_diacritics_internal(m_val);
    let m_canon = canonical_municipality_key_internal(Some(m_val));
    let s_canon = canonical_municipality_key_internal(Some(sheet_muni));
    let s_folded = if !sheet_muni.is_empty() {
        fold_text_diacritics_internal(sheet_muni)
    } else {
        fold_text_diacritics_internal(sheet_muni_folded)
    };
    if !m_canon.is_empty() && s_canon == m_canon {
        return true;
    }
    if !m_folded.is_empty() && s_folded.contains(&m_folded) {
        return true;
    }
    if !m_canon.is_empty() && s_canon.contains(&m_canon) {
        return true;
    }
    false
}

pub fn sheet_matches_district_internal(
    sheet_muni: &str,
    sheet_dist: Option<&str>,
    sheet_dist_folded: Option<&str>,
    sheet_dist_key: Option<&str>,
    district: Option<&str>,
    municipality_specified: bool,
) -> bool {
    let d_in = match district {
        Some(s) if !s.trim().is_empty() => s.trim(),
        _ => return true,
    };
    let mut d_raw = d_in;
    let low = d_raw.to_lowercase();
    for p in ["okres ", "okr. ", "okr "] {
        if low.starts_with(p) {
            d_raw = d_raw[p.len()..].trim();
            break;
        }
    }
    let d_folded = fold_text_diacritics_internal(d_raw);
    if d_folded.is_empty() {
        return true;
    }
    if let Some(dk) = sheet_dist_key {
        if !dk.is_empty() && dk == d_raw {
            return true;
        }
    }
    let s_dist_folded = match sheet_dist.filter(|s| !s.is_empty()) {
        Some(sd) => fold_text_diacritics_internal(sd),
        None => fold_text_diacritics_internal(sheet_dist_folded.unwrap_or("")),
    };
    if s_dist_folded.contains(&d_folded) {
        return true;
    }
    let m_canon = canonical_municipality_key_internal(Some(sheet_muni));
    if let Some(aliases) = MUNI_HISTORICAL_DISTRICT_ALIASES.get(m_canon.as_str()) {
        if municipality_specified {
            if aliases.contains(d_folded.as_str()) || aliases.contains(d_raw) {
                return true;
            }
        } else if matches!(
            d_folded.as_str(),
            "galanta" | "sered" | "topolcany" | "piestany" | "712" | "770" | "748"
        ) && (aliases.contains(d_folded.as_str()) || aliases.contains(d_raw))
        {
            return true;
        }
    }
    false
}

pub fn parse_observationes_events_internal(obs_text: Option<&str>) -> Vec<Value> {
    let text = match obs_text {
        Some(s) if s.trim().len() >= 2 => s.trim(),
        _ => return Vec::new(),
    };
    let low = text.to_lowercase();
    let mut events = Vec::new();

    let extract_year = |txt: &str, min_century: u8| -> Option<i64> {
        let bytes = txt.as_bytes();
        if bytes.len() < 4 {
            return None;
        }
        for i in 0..=(bytes.len() - 4) {
            if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                continue;
            }
            if i + 4 < bytes.len() && bytes[i + 4].is_ascii_alphanumeric() {
                continue;
            }
            let b0 = bytes[i];
            let b1 = bytes[i + 1];
            let b2 = bytes[i + 2];
            let b3 = bytes[i + 3];
            if b2.is_ascii_digit() && b3.is_ascii_digit() {
                let valid_prefix = (b0 == b'1' && b1 >= min_century && b1 <= b'9')
                    || (b0 == b'2' && b1 == b'0');
                if valid_prefix {
                    if let Ok(yr_str) = std::str::from_utf8(&bytes[i..i + 4]) {
                        if let Ok(yr) = yr_str.parse::<i64>() {
                            return Some(yr);
                        }
                    }
                }
            }
        }
        None
    };

    let has_death = text.contains('+')
        || text.contains('†')
        || low.contains("mortu")
        || low.contains("obiit")
        || low.contains("meghalt")
        || low.contains("zomrel");
    if has_death {
        let mut yr_val: Option<i64> = None;
        for clause in text.split([';', '|', '\n']) {
            let cl_low = clause.to_lowercase();
            if clause.contains('+')
                || clause.contains('†')
                || cl_low.contains("mortu")
                || cl_low.contains("obiit")
                || cl_low.contains("meghalt")
                || cl_low.contains("zomrel")
            {
                if let Some(y) = extract_year(clause, b'7') {
                    yr_val = Some(y);
                    break;
                }
            }
        }
        if yr_val.is_none() {
            yr_val = extract_year(text, b'7');
        }
        let date_str = yr_val
            .map(|y| y.to_string())
            .unwrap_or_else(|| text.to_string());
        events.push(json!({
            "event_type": "death",
            "type": "death",
            "label_sk": "Úmrtie zapísané v poznámke (Observationes)",
            "raw_note": text,
            "year": yr_val,
            "date": date_str,
        }));
    }

    let has_marr = low.contains("matrimon")
        || low.contains("contraxit")
        || low.contains("copulat")
        || low.contains("házasság")
        || low.contains("esküv")
        || low.contains("sobáš");
    if has_marr {
        let mut yr_val: Option<i64> = None;
        for clause in text.split([';', '|', '\n']) {
            let cl_low = clause.to_lowercase();
            if cl_low.contains("matrimon")
                || cl_low.contains("contraxit")
                || cl_low.contains("copulat")
                || cl_low.contains("házasság")
                || cl_low.contains("esküv")
                || cl_low.contains("sobáš")
            {
                if let Some(y) = extract_year(clause, b'8') {
                    yr_val = Some(y);
                    break;
                }
            }
        }
        if yr_val.is_none() {
            yr_val = extract_year(text, b'8');
        }
        let date_str = yr_val
            .map(|y| y.to_string())
            .unwrap_or_else(|| text.to_string());
        events.push(json!({
            "event_type": "marriage",
            "type": "marriage",
            "label_sk": "Sobáš zapísaný v poznámke (Observationes)",
            "raw_note": text,
            "year": yr_val,
            "date": date_str,
        }));
    }

    events
}

#[allow(clippy::too_many_arguments)]
pub fn compute_person_fact_search_fields_internal(
    surname_raw: Option<&str>,
    surname_normalized: Option<&str>,
    surname_stem: Option<&str>,
    given_name_raw: Option<&str>,
    given_name_sk: Option<&str>,
    given_name_hu: Option<&str>,
    village_raw: Option<&str>,
    village_sk: Option<&str>,
    village_hu: Option<&str>,
    house_number: Option<&str>,
    birth_year_est: Option<i64>,
    birth_or_event_date: Option<&str>,
    record_type: Option<&str>,
    event_year: Option<i64>,
    age_at_event: Option<&str>,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<i64>,
    Option<i64>,
) {
    let sur = surname_stem
        .filter(|s| !s.is_empty())
        .or_else(|| surname_normalized.filter(|s| !s.is_empty()))
        .or_else(|| surname_raw.filter(|s| !s.is_empty()));
    let surname_folded = sur.map(apply_digraphs_internal).filter(|s| !s.is_empty());
    let surname_skeleton = sur
        .map(phonetic_skeleton_internal)
        .filter(|s| !s.is_empty());

    let giv = given_name_sk
        .filter(|s| !s.is_empty())
        .or_else(|| given_name_raw.filter(|s| !s.is_empty()))
        .or_else(|| given_name_hu.filter(|s| !s.is_empty()));
    let given_canonical_id = giv.and_then(resolve_given_cid_pub);

    let vil = village_sk
        .filter(|s| !s.is_empty())
        .or_else(|| village_raw.filter(|s| !s.is_empty()))
        .or_else(|| village_hu.filter(|s| !s.is_empty()));
    let village_folded = vil
        .map(fold_text_diacritics_internal)
        .filter(|s| !s.is_empty());

    let house_number_int = parse_house_number_int_internal(house_number);

    let mut resolved_birth_year = birth_year_est;
    if resolved_birth_year.is_none() {
        if let Some(bdate) = birth_or_event_date {
            let bytes = bdate.as_bytes();
            if bytes.len() >= 4 {
                for i in 0..=(bytes.len() - 4) {
                    if i > 0 && bytes[i - 1].is_ascii_alphanumeric() {
                        continue;
                    }
                    if i + 4 < bytes.len() && bytes[i + 4].is_ascii_alphanumeric() {
                        continue;
                    }
                    let b0 = bytes[i];
                    let b1 = bytes[i + 1];
                    if bytes[i + 2].is_ascii_digit()
                        && bytes[i + 3].is_ascii_digit()
                        && ((b0 == b'1' && (b'7'..=b'9').contains(&b1))
                            || (b0 == b'2' && b1 == b'0' && bytes[i + 2] <= b'2'))
                    {
                        if let Ok(s) = std::str::from_utf8(&bytes[i..i + 4]) {
                            if let Ok(yr) = s.parse::<i64>() {
                                resolved_birth_year = Some(yr);
                                break;
                            }
                        }
                    }
                }
            }
        }
        if resolved_birth_year.is_none() {
            if record_type == Some("baptism") {
                if let Some(ey) = event_year {
                    resolved_birth_year = Some(ey);
                }
            } else if let (Some(ey), Some(age_s)) = (event_year, age_at_event) {
                let digits: String = age_s
                    .chars()
                    .skip_while(|c| !c.is_ascii_digit())
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(age_n) = digits.parse::<i64>() {
                    if age_n <= 120 {
                        resolved_birth_year = Some(ey - age_n);
                    }
                }
            }
        }
    }

    (
        surname_folded,
        surname_skeleton,
        given_canonical_id,
        village_folded,
        house_number_int,
        resolved_birth_year,
    )
}

#[pyfunction]
#[pyo3(signature = (municipality=None))]
pub fn canonical_municipality_key_rs(municipality: Option<&str>) -> String {
    canonical_municipality_key_internal(municipality)
}

#[pyfunction]
#[pyo3(signature = (municipality=None))]
pub fn canonical_municipality_display_rs(municipality: Option<&str>) -> Option<String> {
    canonical_municipality_display_internal(municipality)
}

#[pyfunction]
#[pyo3(signature = (house_no=None))]
pub fn parse_house_number_int_rs(house_no: Option<&str>) -> Option<i64> {
    parse_house_number_int_internal(house_no)
}

#[pyfunction]
#[pyo3(signature = (house_no=None))]
pub fn normalize_house_number_rs(house_no: Option<&str>) -> String {
    normalize_house_number_internal(house_no)
}

#[pyfunction]
#[pyo3(signature = (candidates, street_query=None))]
pub fn text_matches_street_rs(candidates: Vec<Option<String>>, street_query: Option<&str>) -> bool {
    text_matches_street_internal(&candidates, street_query)
}

#[pyfunction]
#[pyo3(signature = (overview_settlement=None, candidate_street=None, municipality=None))]
pub fn is_compatible_settlement_rs(
    overview_settlement: Option<&str>,
    candidate_street: Option<&str>,
    municipality: Option<&str>,
) -> bool {
    is_compatible_settlement_internal(overview_settlement, candidate_street, municipality)
}

#[pyfunction]
#[pyo3(signature = (obs_text=None))]
pub fn parse_observationes_events_rs(obs_text: Option<&str>) -> String {
    let evs = parse_observationes_events_internal(obs_text);
    serde_json::to_string(&evs).unwrap_or_else(|_| "[]".to_string())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (surname_raw=None, surname_normalized=None, surname_stem=None, given_name_raw=None, given_name_sk=None, given_name_hu=None, village_raw=None, village_sk=None, village_hu=None, house_number=None, birth_year_est=None, birth_or_event_date=None, record_type=None, event_year=None, age_at_event=None))]
pub fn compute_person_fact_search_fields_rs(
    surname_raw: Option<&str>,
    surname_normalized: Option<&str>,
    surname_stem: Option<&str>,
    given_name_raw: Option<&str>,
    given_name_sk: Option<&str>,
    given_name_hu: Option<&str>,
    village_raw: Option<&str>,
    village_sk: Option<&str>,
    village_hu: Option<&str>,
    house_number: Option<&str>,
    birth_year_est: Option<i64>,
    birth_or_event_date: Option<&str>,
    record_type: Option<&str>,
    event_year: Option<i64>,
    age_at_event: Option<&str>,
) -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<i64>,
    Option<i64>,
) {
    compute_person_fact_search_fields_internal(
        surname_raw,
        surname_normalized,
        surname_stem,
        given_name_raw,
        given_name_sk,
        given_name_hu,
        village_raw,
        village_sk,
        village_hu,
        house_number,
        birth_year_est,
        birth_or_event_date,
        record_type,
        event_year,
        age_at_event,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (ov_sheet_id=None, ov_year=None, ov_muni_raw=None, ov_mkey=None, ov_hn_raw=None, cand_id=None, cand_type=None, cand_year=None, cand_muni_raw=None, cand_mkey=None, cand_hn_raw=None))]
pub fn can_match_overview_house_to_sheet_rs(
    ov_sheet_id: Option<i64>,
    ov_year: Option<i64>,
    ov_muni_raw: Option<&str>,
    ov_mkey: Option<&str>,
    ov_hn_raw: Option<&str>,
    cand_id: Option<i64>,
    cand_type: Option<&str>,
    cand_year: Option<i64>,
    cand_muni_raw: Option<&str>,
    cand_mkey: Option<&str>,
    cand_hn_raw: Option<&str>,
) -> bool {
    can_match_overview_house_to_sheet_internal(
        ov_sheet_id,
        ov_year,
        ov_muni_raw,
        ov_mkey,
        ov_hn_raw,
        cand_id,
        cand_type,
        cand_year,
        cand_muni_raw,
        cand_mkey,
        cand_hn_raw,
    )
}

#[pyfunction]
#[pyo3(signature = (query_folded, candidates, sheet_muni_folded=None))]
pub fn batch_match_gazetteer_rs(
    query_folded: &str,
    candidates: Vec<String>,
    sheet_muni_folded: Option<&str>,
) -> (Option<String>, f64) {
    let sheet_mf = sheet_muni_folded.unwrap_or("");
    let mut best_cand: Option<String> = None;
    let mut best_score = 0.0_f64;

    for cand in candidates {
        let cf = fold_text_diacritics_internal(&cand);
        if cf == query_folded {
            return (Some(cand), 1.0);
        }
        let len_diff = (query_folded.len() as isize - cf.len() as isize).unsigned_abs();
        if len_diff > 5 {
            continue;
        }
        let dist_cost = paleographic_distance_internal(query_folded, &cf);
        let max_l = query_folded.chars().count().max(cf.chars().count()).max(1) as f64;
        let sim = (1.0 - dist_cost / max_l).max(0.0);
        let mut score = sim.max(paleographic_similarity_internal(query_folded, &cf));
        if !sheet_mf.is_empty() && cf == sheet_mf {
            score += 0.10;
        }
        if score > best_score {
            best_score = score;
            best_cand = Some(cand);
        }
    }
    (best_cand, best_score)
}
