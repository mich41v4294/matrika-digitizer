use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rusqlite::{params, params_from_iter, Connection};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::gazetteer::{
    canonical_municipality_display_internal, canonical_municipality_key_internal,
    is_compatible_settlement_internal, normalize_house_number_internal,
    sheet_matches_district_internal, sheet_matches_municipality_internal,
    text_matches_street_internal, MUNI_HISTORICAL_DISTRICT_ALIASES,
};
use crate::lexicon::fuzzy_name_match_score_internal;
use crate::{fold_text_diacritics_internal, open_sqlite};

fn parse_birth_year_range(birth_year: Option<&str>) -> Option<(i64, i64)> {
    let raw = birth_year?.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.len() == 4 && raw.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(yr) = raw.parse::<i64>() {
            return Some((yr, yr));
        }
    }
    for sep in ["±", "+-", "+/-"] {
        if let Some(pos) = raw.find(sep) {
            let left = raw[..pos].trim();
            let right = raw[(pos + sep.len())..].trim();
            if left.len() == 4 {
                if let (Ok(yr), Ok(tol)) = (left.parse::<i64>(), right.parse::<i64>()) {
                    return Some((yr - tol, yr + tol));
                }
            }
        }
    }
    let low = raw.to_lowercase();
    for pfx in ["~", "cca.", "cca", "okolo"] {
        if low.starts_with(pfx) {
            let rest = raw[pfx.len()..].trim();
            if rest.len() == 4 {
                if let Ok(yr) = rest.parse::<i64>() {
                    return Some((yr - 2, yr + 2));
                }
            }
        }
    }
    let digits_groups: Vec<i64> = raw
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| s.len() == 4)
        .filter_map(|s| s.parse::<i64>().ok())
        .collect();
    if digits_groups.len() == 2 {
        let y1 = digits_groups[0];
        let y2 = digits_groups[1];
        return Some((y1.min(y2), y1.max(y2)));
    }
    if digits_groups.len() == 1 {
        let yr = digits_groups[0];
        let trimmed = raw.trim();
        if trimmed.ends_with(['-', '–', '—']) {
            return Some((yr, 2100));
        }
        if trimmed.starts_with(['-', '–', '—']) {
            return Some((1600, yr));
        }
    }
    None
}

fn extract_birth_years(byear_est: Option<i64>, bdate: Option<&str>) -> Vec<i64> {
    let mut years = Vec::with_capacity(2);
    if let Some(by) = byear_est {
        years.push(by);
    }
    if let Some(s) = bdate {
        let bytes = s.as_bytes();
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
                    && ((b0 == b'1' && (b'6'..=b'9').contains(&b1))
                        || (b0 == b'2' && b1 == b'0'))
                {
                    if let Ok(ys) = std::str::from_utf8(&bytes[i..i + 4]) {
                        if let Ok(yr) = ys.parse::<i64>() {
                            if !years.contains(&yr) {
                                years.push(yr);
                            }
                        }
                    }
                }
            }
        }
    }
    years
}

#[derive(Clone)]
struct FactFilterData {
    id: i64,
    census_sheet_id: i64,
    given_name_raw: Option<String>,
    given_name_sk: Option<String>,
    given_name_hu: Option<String>,
    surname_raw: Option<String>,
    surname_normalized: Option<String>,
    mother_maiden_surname: Option<String>,
    father_name: Option<String>,
    mother_name: Option<String>,
    spouse_name: Option<String>,
    surname_candidates_json: Option<String>,
    relation_to_head: Option<String>,
    sex: Option<String>,
    birth_or_event_date: Option<String>,
    birth_year_est: Option<i64>,
    birth_place_raw: Option<String>,
    birth_place_sk: Option<String>,
    birth_place_municipality: Option<String>,
    birth_place_district: Option<String>,
    birth_place_country: Option<String>,
    home_jurisdiction: Option<String>,
    previous_residence: Option<String>,
    village_raw: Option<String>,
    village_sk: Option<String>,
    observationes_raw: Option<String>,
    ocr_txt: Option<String>,
}

impl FactFilterData {
    fn struct_cands(&self) -> Vec<Option<String>> {
        let mut parts = Vec::with_capacity(12);
        for opt_s in [
            &self.given_name_raw,
            &self.given_name_sk,
            &self.given_name_hu,
            &self.surname_raw,
            &self.surname_normalized,
            &self.mother_maiden_surname,
            &self.father_name,
            &self.mother_name,
            &self.spouse_name,
        ] {
            if let Some(ref s) = opt_s {
                if !s.is_empty() {
                    parts.push(Some(s.clone()));
                }
            }
        }
        if let Some(ref sc_s) = self.surname_candidates_json {
            if !sc_s.is_empty() && sc_s != "[]" {
                if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(sc_s) {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            if !s.is_empty() {
                                parts.push(Some(s.to_string()));
                            }
                        } else if let Some(obj) = item.as_object() {
                            for k in [
                                "raw_ocr",
                                "normalized_ocr",
                                "consensus_stem",
                                "resolved_surname",
                            ] {
                                if let Some(v) = obj.get(k).and_then(|v| v.as_str()) {
                                    if !v.is_empty() {
                                        parts.push(Some(v.to_string()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        parts
    }

    fn matches_person_name(&self, person_name: Option<&str>) -> bool {
        let p_clean = match person_name {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return true,
        };
        let cands = self.struct_cands();
        if !cands.is_empty() && fuzzy_name_match_score_internal(p_clean, &cands, 0.76) >= 0.76 {
            return true;
        }
        let p_fold = fold_text_diacritics_internal(p_clean);
        if let Some(ref obs) = self.observationes_raw {
            if !p_fold.is_empty() && fold_text_diacritics_internal(obs).contains(&p_fold) {
                return true;
            }
        }
        if let Some(ref ot) = self.ocr_txt {
            if !p_fold.is_empty() && fold_text_diacritics_internal(ot).contains(&p_fold) {
                return true;
            }
            if cands.is_empty()
                && fuzzy_name_match_score_internal(p_clean, &[Some(ot.clone())], 0.76) >= 0.76
            {
                return true;
            }
        }
        false
    }

    fn matches_birth_year(&self, birth_year: Option<&str>) -> bool {
        let raw = match birth_year {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return true,
        };
        let years = extract_birth_years(self.birth_year_est, self.birth_or_event_date.as_deref());
        if let Some((y_min, y_max)) = parse_birth_year_range(Some(raw)) {
            return years.iter().any(|&y| y >= y_min && y <= y_max);
        }
        let raw_f = fold_text_diacritics_internal(raw);
        if years.iter().any(|y| y.to_string().contains(&raw_f)) {
            return true;
        }
        if let Some(ref bd) = self.birth_or_event_date {
            if fold_text_diacritics_internal(bd).contains(&raw_f) {
                return true;
            }
        }
        false
    }

    fn matches_birth_place(&self, birth_place: Option<&str>) -> bool {
        let bp_val = match birth_place {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return true,
        };
        let bp_folded = fold_text_diacritics_internal(bp_val);
        let bp_canon = canonical_municipality_key_internal(Some(bp_val));
        let mut combined_parts = Vec::with_capacity(8);
        for opt_p in [
            &self.birth_place_raw,
            &self.birth_place_sk,
            &self.birth_place_municipality,
            &self.birth_place_district,
            &self.birth_place_country,
            &self.home_jurisdiction,
            &self.previous_residence,
            &self.village_raw,
            &self.village_sk,
        ] {
            if let Some(ref p) = opt_p {
                if !p.is_empty() {
                    combined_parts.push(p.as_str());
                }
            }
        }
        let combined_folded = fold_text_diacritics_internal(&combined_parts.join(" "));
        if !bp_folded.is_empty() && combined_folded.contains(&bp_folded) {
            return true;
        }
        let q_toks: Vec<&str> = bp_folded
            .split(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '/'))
            .filter(|t| !t.is_empty() && *t != "okr" && *t != "okres")
            .collect();
        if !q_toks.is_empty() && q_toks.iter().all(|t| combined_folded.contains(*t)) {
            return true;
        }
        for cand_opt in [
            self.birth_place_municipality.as_deref(),
            self.birth_place_sk.as_deref(),
            self.birth_place_raw
                .as_deref()
                .and_then(|s| s.split(',').next()),
            self.home_jurisdiction
                .as_deref()
                .and_then(|s| s.split(',').next()),
        ] {
            if let Some(cm) = cand_opt.filter(|s| !s.is_empty()) {
                let cand_canon = canonical_municipality_key_internal(Some(cm));
                if !bp_canon.is_empty() && cand_canon == bp_canon {
                    return true;
                }
                if let Some(aliases) = MUNI_HISTORICAL_DISTRICT_ALIASES.get(cand_canon.as_str()) {
                    if aliases.contains(bp_folded.as_str()) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn serialize_preview(&self, matched_filter: bool) -> Value {
        let given = self
            .given_name_sk
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.given_name_raw.as_deref().filter(|s| !s.is_empty()))
            .unwrap_or("");
        let surname = self
            .surname_raw
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.surname_normalized.as_deref().filter(|s| !s.is_empty()))
            .unwrap_or("");
        let rel_low = self
            .relation_to_head
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let is_child_or_male = self.sex.as_deref() == Some("M")
            || matches!(
                rel_low.as_str(),
                "syn"
                    | "dcéra"
                    | "vnuk"
                    | "vnučka"
                    | "chovanec"
                    | "chovanka"
                    | "adoptívny syn"
                    | "adoptívna dcéra"
                    | "nevlastný syn"
                    | "nevlastná dcéra"
            );
        let maiden = if is_child_or_male {
            None
        } else {
            self.mother_maiden_surname
                .as_deref()
                .filter(|s| !s.is_empty())
        };
        let mut full_name = format!("{} {}", given, surname).trim().to_string();
        if let Some(md) = maiden {
            full_name = format!("{} (rod. {})", full_name, md).trim().to_string();
        }
        let years = extract_birth_years(self.birth_year_est, self.birth_or_event_date.as_deref());
        let byear = years.first().copied();
        let bplace = self
            .birth_place_raw
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| self.birth_place_sk.as_deref().filter(|s| !s.is_empty()))
            .or_else(|| {
                self.birth_place_municipality
                    .as_deref()
                    .filter(|s| !s.is_empty())
            });

        json!({
            "id": self.id,
            "given_name": if given.is_empty() { None } else { Some(given) },
            "surname": if surname.is_empty() { None } else { Some(surname) },
            "surname_normalized": self.surname_normalized.as_deref().filter(|s| !s.is_empty()),
            "maiden_surname": maiden,
            "full_name": if full_name.is_empty() { None } else { Some(full_name) },
            "relation_to_head": self.relation_to_head,
            "birth_date": self.birth_or_event_date,
            "birth_year": byear,
            "birth_place": bplace,
            "domicile": self.home_jurisdiction,
            "matched_filter": matched_filter,
        })
    }
}

#[pyfunction]
#[pyo3(signature = (db_path, census_year=1930, district=None))]
pub fn get_municipalities_summary_rs(
    py: Python<'_>,
    db_path: String,
    census_year: i64,
    district: Option<String>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let resolved_year = if census_year == 1940 { 1940 } else { 1930 };
        let dist_filter = district.as_deref().filter(|s| !s.trim().is_empty());

        let mut ov_count_by_muni_key: HashMap<String, i64> = HashMap::new();
        let mut ov_persons_by_muni_key: HashMap<String, i64> = HashMap::new();
        {
            let mut seen_ovh: HashSet<(i64, String)> = HashSet::new();
            let mut ov_stmt = conn
                .prepare(
                    "SELECT census_sheet_id, row_index, house_number, municipality, present_persons_count
                     FROM censusoverviewhouse
                     WHERE census_year = ?1
                     ORDER BY census_sheet_id ASC, row_index ASC, id ASC",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let ov_rows = ov_stmt
                .query_map(params![resolved_year], |r| {
                    Ok((
                        r.get::<_, Option<i64>>(0)?.unwrap_or(0),
                        r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for row_res in ov_rows {
                if let Ok((ov_sid, r_idx, hn_val, raw_muni, pp_val)) = row_res {
                    let hn_norm = normalize_house_number_internal(hn_val.as_deref());
                    let dedup_k = (
                        ov_sid,
                        if !hn_norm.is_empty() {
                            hn_norm
                        } else {
                            r_idx.to_string()
                        },
                    );
                    if !seen_ovh.insert(dedup_k) {
                        continue;
                    }
                    let m_key = {
                        let k = canonical_municipality_key_internal(raw_muni.as_deref());
                        if k.is_empty() {
                            "neznama obec".to_string()
                        } else {
                            k
                        }
                    };
                    *ov_count_by_muni_key.entry(m_key.clone()).or_insert(0) += 1;
                    *ov_persons_by_muni_key.entry(m_key).or_insert(0) += pp_val;
                }
            }
        }

        let mut mov_houses_by_muni_key: HashMap<String, i64> = HashMap::new();
        let mut mov_persons_by_muni_key: HashMap<String, i64> = HashMap::new();
        {
            let mut mov_stmt = conn
                .prepare(
                    "SELECT census_sheet_id, municipality, is_summary_row, houses_total,
                            houses_inhabited, houses_uninhabited, present_persons_count
                     FROM censusmunicipaloverviewdistrict
                     WHERE census_year = ?1
                     ORDER BY census_sheet_id ASC, row_index ASC",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut mov_by_sheet: HashMap<
                i64,
                Vec<(String, bool, Option<i64>, Option<i64>, Option<i64>, Option<i64>)>,
            > = HashMap::new();
            let mov_rows = mov_stmt
                .query_map(params![resolved_year], |r| {
                    Ok((
                        r.get::<_, Option<i64>>(0)?.unwrap_or(0),
                        r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        r.get::<_, Option<bool>>(2)?.unwrap_or(false),
                        r.get::<_, Option<i64>>(3)?,
                        r.get::<_, Option<i64>>(4)?,
                        r.get::<_, Option<i64>>(5)?,
                        r.get::<_, Option<i64>>(6)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for r in mov_rows.flatten() {
                if r.0 > 0 {
                    mov_by_sheet
                        .entry(r.0)
                        .or_default()
                        .push((r.1, r.2, r.3, r.4, r.5, r.6));
                }
            }

            for md_list in mov_by_sheet.values() {
                if md_list.is_empty() {
                    continue;
                }
                let non_sum_cnt = md_list.iter().filter(|x| !x.1).count();
                if non_sum_cnt == 0 || non_sum_cnt > 10 {
                    continue;
                }
                let mut seq_steps = 0;
                let mut prev_tot: Option<i64> = None;
                for x in md_list.iter().filter(|x| !x.1) {
                    if let Some(ht) = x.2 {
                        let inh = x.3.unwrap_or(0);
                        let uninh = x.4.unwrap_or(0);
                        if let Some(pt) = prev_tot {
                            if (1..=2).contains(&(ht - pt)) && ht != (inh + uninh) {
                                seq_steps += 1;
                            }
                        }
                        prev_tot = Some(ht);
                    }
                }
                if seq_steps >= 2 {
                    continue;
                }

                let m_key = {
                    let k = canonical_municipality_key_internal(Some(&md_list[0].0));
                    if k.is_empty() {
                        "neznama obec".to_string()
                    } else {
                        k
                    }
                };
                let sum_houses = md_list
                    .iter()
                    .filter(|x| x.1)
                    .find_map(|x| x.2)
                    .unwrap_or_else(|| md_list.iter().filter(|x| !x.1).filter_map(|x| x.2).sum());
                let sum_persons = md_list
                    .iter()
                    .filter(|x| x.1)
                    .find_map(|x| x.5)
                    .unwrap_or_else(|| md_list.iter().filter(|x| !x.1).filter_map(|x| x.5).sum());
                *mov_houses_by_muni_key.entry(m_key.clone()).or_insert(0) += sum_houses;
                *mov_persons_by_muni_key.entry(m_key).or_insert(0) += sum_persons;
            }
        }

        let mut person_counts_by_sheet: HashMap<i64, i64> = HashMap::new();
        {
            let mut pf_stmt = conn
                .prepare(
                    "SELECT census_sheet_id, COUNT(id)
                     FROM personfact
                     WHERE record_type = 'census' AND census_sheet_id IS NOT NULL
                     GROUP BY census_sheet_id",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let pf_rows = pf_stmt
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (sid, cnt) in pf_rows.flatten() {
                person_counts_by_sheet.insert(sid, cnt);
            }
        }

        struct MuniSummaryEntry {
            municipality: String,
            municipality_key: Option<String>,
            district: String,
            district_key: Option<String>,
            total_sheets: i64,
            downloaded_sheets: i64,
            ocr_done_sheets: i64,
            overview_sheets_count: i64,
            overview_houses_count: i64,
            overview_persons_count: i64,
            extracted_persons_count: i64,
            persons_count: i64,
        }

        let mut by_muni: HashMap<String, MuniSummaryEntry> = HashMap::new();
        let mut muni_order: Vec<String> = Vec::new();
        {
            let mut sh_stmt = conn
                .prepare(
                    "SELECT id, census_year, municipality, municipality_folded, municipality_key,
                            district, district_folded, district_key, download_status, sheet_type
                     FROM censussheet
                     ORDER BY id ASC",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let sh_rows = sh_stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<i64>>(1)?.unwrap_or(1930),
                        r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                        r.get::<_, Option<String>>(7)?,
                        r.get::<_, Option<String>>(8)?.unwrap_or_default(),
                        r.get::<_, Option<String>>(9)?
                            .unwrap_or_else(|| "household".to_string()),
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for r in sh_rows.flatten() {
                let (sid, cyear, muni, _muni_f, muni_k, dist, dist_f, dist_k, dl_status, stype) = r;
                if let Some(df) = dist_filter {
                    if !sheet_matches_district_internal(
                        &muni,
                        dist.as_deref(),
                        dist_f.as_deref(),
                        dist_k.as_deref(),
                        Some(df),
                        false,
                    ) {
                        continue;
                    }
                }
                let m_display = canonical_municipality_display_internal(Some(&muni))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Neznáma obec".to_string());
                let m_canon = {
                    let k = canonical_municipality_key_internal(Some(&m_display));
                    if k.is_empty() {
                        "neznama obec".to_string()
                    } else {
                        k
                    }
                };
                let d_name = dist.unwrap_or_default();

                if !by_muni.contains_key(&m_canon) {
                    let ov_houses = ov_count_by_muni_key
                        .get(&m_canon)
                        .copied()
                        .filter(|&v| v > 0)
                        .unwrap_or_else(|| {
                            mov_houses_by_muni_key.get(&m_canon).copied().unwrap_or(0)
                        });
                    let ov_persons = mov_persons_by_muni_key
                        .get(&m_canon)
                        .copied()
                        .unwrap_or(0)
                        .max(ov_persons_by_muni_key.get(&m_canon).copied().unwrap_or(0));
                    muni_order.push(m_canon.clone());
                    by_muni.insert(
                        m_canon.clone(),
                        MuniSummaryEntry {
                            municipality: m_display,
                            municipality_key: muni_k.clone(),
                            district: d_name.clone(),
                            district_key: dist_k.clone(),
                            total_sheets: 0,
                            downloaded_sheets: 0,
                            ocr_done_sheets: 0,
                            overview_sheets_count: 0,
                            overview_houses_count: ov_houses,
                            overview_persons_count: ov_persons,
                            extracted_persons_count: 0,
                            persons_count: ov_persons,
                        },
                    );
                } else if let Some(entry) = by_muni.get_mut(&m_canon) {
                    if cyear == resolved_year {
                        if !d_name.is_empty() {
                            entry.district = d_name;
                        }
                        if dist_k.is_some() {
                            entry.district_key = dist_k;
                        }
                        if muni_k.is_some() {
                            entry.municipality_key = muni_k;
                        }
                    } else {
                        if entry.district.is_empty() && !d_name.is_empty() {
                            entry.district = d_name;
                        }
                        if entry.district_key.is_none() && dist_k.is_some() {
                            entry.district_key = dist_k;
                        }
                        if entry.municipality_key.is_none() && muni_k.is_some() {
                            entry.municipality_key = muni_k;
                        }
                    }
                }

                if cyear != resolved_year {
                    continue;
                }
                if let Some(entry) = by_muni.get_mut(&m_canon) {
                    entry.total_sheets += 1;
                    if matches!(
                        dl_status.as_str(),
                        "downloaded" | "sliced" | "ocr_done" | "extracted" | "matched"
                    ) {
                        entry.downloaded_sheets += 1;
                    }
                    if matches!(dl_status.as_str(), "ocr_done" | "extracted" | "matched") {
                        entry.ocr_done_sheets += 1;
                    }
                    if matches!(stype.as_str(), "overview" | "municipal_overview") {
                        entry.overview_sheets_count += 1;
                    }
                    entry.extracted_persons_count +=
                        person_counts_by_sheet.get(&sid).copied().unwrap_or(0);
                    entry.persons_count = entry
                        .extracted_persons_count
                        .max(entry.overview_persons_count);
                }
            }
        }

        let mut list_out: Vec<MuniSummaryEntry> = muni_order
            .into_iter()
            .filter_map(|k| by_muni.remove(&k))
            .collect();
        list_out.sort_by(|a, b| {
            (-a.ocr_done_sheets, -a.total_sheets, a.municipality.as_str()).cmp(&(
                -b.ocr_done_sheets,
                -b.total_sheets,
                b.municipality.as_str(),
            ))
        });

        let total_munis = list_out.len();
        let munis_json: Vec<Value> = list_out
            .into_iter()
            .map(|e| {
                json!({
                    "municipality": e.municipality,
                    "municipality_key": e.municipality_key,
                    "district": e.district,
                    "district_key": e.district_key,
                    "total_sheets": e.total_sheets,
                    "downloaded_sheets": e.downloaded_sheets,
                    "ocr_done_sheets": e.ocr_done_sheets,
                    "overview_sheets_count": e.overview_sheets_count,
                    "overview_houses_count": e.overview_houses_count,
                    "overview_persons_count": e.overview_persons_count,
                    "extracted_persons_count": e.extracted_persons_count,
                    "persons_count": e.persons_count,
                    "years": [resolved_year],
                })
            })
            .collect();

        let out = json!({
            "census_year": resolved_year,
            "municipalities": munis_json,
            "total_municipalities": total_munis,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (db_path, candidate_sheet_ids, q=None, street=None, person_name=None, birth_year=None, birth_place=None))]
pub fn filter_census_sheets_and_previews_rs(
    py: Python<'_>,
    db_path: String,
    candidate_sheet_ids: Vec<i64>,
    q: Option<String>,
    street: Option<String>,
    person_name: Option<String>,
    birth_year: Option<String>,
    birth_place: Option<String>,
) -> PyResult<String> {
    py.allow_threads(move || {
        if candidate_sheet_ids.is_empty() {
            return Ok(
                serde_json::to_string(&json!({
                    "filtered_sheet_ids": [],
                    "available_streets": [],
                    "overview_counts": {},
                    "municipal_districts_counts": {},
                    "fallback_streets": {},
                    "previews_by_sheet": {},
                }))
                .unwrap(),
            );
        }

        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        #[derive(Clone)]
        struct SheetMeta {
            id: i64,
            slovakiana_key: String,
            municipality: String,
            district: Option<String>,
            street_raw: Option<String>,
            house_number: Option<String>,
            household_owner: Option<String>,
            commissioner_name: Option<String>,
        }

        let mut sheets_meta: Vec<SheetMeta> = Vec::with_capacity(candidate_sheet_ids.len());
        let mut sheet_by_id: HashMap<i64, SheetMeta> =
            HashMap::with_capacity(candidate_sheet_ids.len());
        let mut available_streets: HashSet<String> = HashSet::new();

        for chunk in candidate_sheet_ids.chunks(450) {
            let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT id, slovakiana_key, municipality, district, street_raw,
                        house_number, household_owner, commissioner_name
                 FROM censussheet WHERE id IN ({})",
                ph.join(", ")
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok(SheetMeta {
                        id: r.get(0)?,
                        slovakiana_key: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        municipality: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        district: r.get(3)?,
                        street_raw: r.get(4)?,
                        house_number: r.get(5)?,
                        household_owner: r.get(6)?,
                        commissioner_name: r.get(7)?,
                    })
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for sm in rows.flatten() {
                if let Some(ref st) = sm.street_raw {
                    let st_t = st.trim();
                    if !st_t.is_empty() && !matches!(st_t, "-" | "—" | "?") {
                        available_streets.insert(st_t.to_string());
                    }
                }
                sheet_by_id.insert(sm.id, sm);
            }
        }
        for sid in &candidate_sheet_ids {
            if let Some(sm) = sheet_by_id.get(sid) {
                sheets_meta.push(sm.clone());
            }
        }

        let mut hh_streets_by_sheet: HashMap<i64, Vec<String>> = HashMap::new();
        let mut hh_names_by_sheet: HashMap<i64, Vec<String>> = HashMap::new();
        for chunk in candidate_sheet_ids.chunks(450) {
            let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT census_sheet_id, street, house_owner_name, notes_raw
                 FROM censushousehold
                 WHERE census_sheet_id IN ({})
                 ORDER BY id ASC",
                ph.join(", ")
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (csid, st_opt, owner_opt, notes_opt) in rows.flatten() {
                if let Some(ref st) = st_opt {
                    let st_t = st.trim();
                    if !st_t.is_empty() && !matches!(st_t, "-" | "—" | "?") {
                        available_streets.insert(st_t.to_string());
                        hh_streets_by_sheet
                            .entry(csid)
                            .or_default()
                            .push(st_t.to_string());
                    }
                }
                let names_vec = hh_names_by_sheet.entry(csid).or_default();
                if let Some(own) = owner_opt.filter(|s| !s.is_empty()) {
                    names_vec.push(own);
                }
                if let Some(notes_s) = notes_opt.filter(|s| !s.is_empty()) {
                    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&notes_s) {
                        for k in [
                            "head_of_household_name",
                            "head_given_name",
                            "head_surname",
                            "wife_name",
                            "wife_given_name",
                            "wife_maiden_name",
                            "wife_maiden_surname",
                        ] {
                            if let Some(v) = map.get(k).and_then(|v| v.as_str()) {
                                if !v.is_empty() {
                                    names_vec.push(v.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut facts_by_sheet: HashMap<i64, Vec<FactFilterData>> = HashMap::new();
        for chunk in candidate_sheet_ids.chunks(450) {
            let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT
                    pf.id, pf.row_id, pf.census_sheet_id,
                    pf.given_name_raw, pf.given_name_sk, pf.given_name_hu,
                    pf.surname_raw, pf.surname_normalized, pf.mother_maiden_surname,
                    pf.father_name, pf.mother_name, pf.spouse_name, pf.surname_candidates_json,
                    pf.relation_to_head, pf.sex, pf.birth_or_event_date, pf.birth_year_est,
                    pf.birth_place_raw, pf.birth_place_sk, pf.birth_place_municipality,
                    pf.birth_place_district, pf.birth_place_country, pf.home_jurisdiction,
                    pf.previous_residence, pf.village_raw, pf.village_sk, pf.observationes_raw,
                    rr.raw_ocr_combined, rr.raw_ocr_left, rr.raw_ocr_right
                 FROM personfact pf
                 LEFT JOIN recordrow rr ON rr.id = pf.row_id
                 WHERE pf.census_sheet_id IN ({})
                 ORDER BY pf.id ASC",
                ph.join(", ")
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    let ocr_c: Option<String> = r.get(27)?;
                    let ocr_l: Option<String> = r.get(28)?;
                    let ocr_r: Option<String> = r.get(29)?;
                    let ocr_parts: Vec<&str> = [
                        ocr_c.as_deref().unwrap_or(""),
                        ocr_l.as_deref().unwrap_or(""),
                        ocr_r.as_deref().unwrap_or(""),
                    ]
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .collect();
                    let ocr_txt = if ocr_parts.is_empty() {
                        None
                    } else {
                        Some(ocr_parts.join(" "))
                    };
                    let _row_id_unused: Option<i64> = r.get(1)?;
                    Ok(FactFilterData {
                        id: r.get(0)?,
                        census_sheet_id: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                        given_name_raw: r.get(3)?,
                        given_name_sk: r.get(4)?,
                        given_name_hu: r.get(5)?,
                        surname_raw: r.get(6)?,
                        surname_normalized: r.get(7)?,
                        mother_maiden_surname: r.get(8)?,
                        father_name: r.get(9)?,
                        mother_name: r.get(10)?,
                        spouse_name: r.get(11)?,
                        surname_candidates_json: r.get(12)?,
                        relation_to_head: r.get(13)?,
                        sex: r.get(14)?,
                        birth_or_event_date: r.get(15)?,
                        birth_year_est: r.get(16)?,
                        birth_place_raw: r.get(17)?,
                        birth_place_sk: r.get(18)?,
                        birth_place_municipality: r.get(19)?,
                        birth_place_district: r.get(20)?,
                        birth_place_country: r.get(21)?,
                        home_jurisdiction: r.get(22)?,
                        previous_residence: r.get(23)?,
                        village_raw: r.get(24)?,
                        village_sk: r.get(25)?,
                        observationes_raw: r.get(26)?,
                        ocr_txt,
                    })
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for f in rows.flatten() {
                if f.census_sheet_id > 0 {
                    facts_by_sheet.entry(f.census_sheet_id).or_default().push(f);
                }
            }
        }

        #[derive(Clone)]
        struct OvhFilterData {
            owner_name: Option<String>,
            owner_surname_folded: Option<String>,
            settlement_name: Option<String>,
            remarks: Option<String>,
            raw_ocr_text: Option<String>,
        }

        let mut overview_counts: HashMap<i64, i64> = HashMap::new();
        let mut ovh_by_overview_sheet: HashMap<i64, Vec<OvhFilterData>> = HashMap::new();
        let mut matched_ovh_owners_by_sheet: HashMap<i64, Vec<String>> = HashMap::new();
        let mut matched_ovh_settlements_by_sheet: HashMap<i64, Vec<String>> = HashMap::new();
        {
            let mut seen_ovh: HashSet<(i64, String)> = HashSet::new();
            let cand_set: HashSet<i64> = candidate_sheet_ids.iter().copied().collect();
            let mut stmt = conn
                .prepare(
                    "SELECT census_sheet_id, matched_sheet_id, row_index, house_number,
                            settlement_name, owner_name, owner_surname_folded, remarks, raw_ocr_text
                     FROM censusoverviewhouse",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, Option<i64>>(0)?.unwrap_or(0),
                        r.get::<_, Option<i64>>(1)?,
                        r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                        r.get::<_, Option<String>>(7)?,
                        r.get::<_, Option<String>>(8)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for (csid, msid_opt, r_idx, hn_opt, sett_opt, own_opt, own_f_opt, rem_opt, ocr_opt) in
                rows.flatten()
            {
                let in_cs = cand_set.contains(&csid);
                let in_ms = msid_opt.map(|m| cand_set.contains(&m)).unwrap_or(false);
                if !in_cs && !in_ms {
                    continue;
                }
                let hn_norm = normalize_house_number_internal(hn_opt.as_deref());
                let ov_k = (
                    csid,
                    if !hn_norm.is_empty() {
                        hn_norm
                    } else {
                        r_idx.to_string()
                    },
                );
                if seen_ovh.insert(ov_k) {
                    *overview_counts.entry(csid).or_insert(0) += 1;
                    ovh_by_overview_sheet.entry(csid).or_default().push(OvhFilterData {
                        owner_name: own_opt.clone(),
                        owner_surname_folded: own_f_opt,
                        settlement_name: sett_opt.clone(),
                        remarks: rem_opt.clone(),
                        raw_ocr_text: ocr_opt,
                    });
                }
                if let Some(ref st) = sett_opt {
                    let st_t = st.trim();
                    if !st_t.is_empty() && !matches!(st_t, "-" | "—" | "?") {
                        available_streets.insert(st_t.to_string());
                    }
                }
                if let Some(msid) = msid_opt {
                    if let Some(ref st) = sett_opt {
                        if !st.is_empty() {
                            matched_ovh_settlements_by_sheet
                                .entry(msid)
                                .or_default()
                                .push(st.clone());
                        }
                    }
                    for tx in [own_opt, rem_opt].into_iter().flatten() {
                        if !tx.is_empty() {
                            matched_ovh_owners_by_sheet
                                .entry(msid)
                                .or_default()
                                .push(tx);
                        }
                    }
                }
            }
        }

        let mut municipal_districts_counts: HashMap<i64, i64> = HashMap::new();
        let mut md_names_by_sheet: HashMap<i64, Vec<String>> = HashMap::new();
        for chunk in candidate_sheet_ids.chunks(450) {
            let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT census_sheet_id, is_summary_row, commissioner_name, remarks
                 FROM censusmunicipaloverviewdistrict
                 WHERE census_sheet_id IN ({})",
                ph.join(", ")
            );
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map(params_from_iter(chunk.iter()), |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, Option<bool>>(1)?.unwrap_or(false),
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (csid, is_sum, comm_opt, rem_opt) in rows.flatten() {
                if !is_sum {
                    *municipal_districts_counts.entry(csid).or_insert(0) += 1;
                }
                let vec = md_names_by_sheet.entry(csid).or_default();
                if let Some(c) = comm_opt.filter(|s| !s.is_empty()) {
                    vec.push(c);
                }
                if let Some(rem) = rem_opt.filter(|s| !s.is_empty()) {
                    vec.push(rem);
                }
            }
        }

        let st_q = street.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty());
        let pn_q = person_name
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        let by_q = birth_year
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        let bp_q = birth_place
            .as_deref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        let q_val = q.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty());

        let ovh_matches_person = |ovh: &OvhFilterData, pq: &str| -> bool {
            let cands = vec![
                ovh.owner_name.clone(),
                ovh.owner_surname_folded.clone(),
                ovh.remarks.clone(),
            ];
            if fuzzy_name_match_score_internal(pq, &cands, 0.76) >= 0.76 {
                return true;
            }
            if let Some(ref raw_ocr) = ovh.raw_ocr_text {
                let p_fold = fold_text_diacritics_internal(pq);
                if !p_fold.is_empty() && fold_text_diacritics_internal(raw_ocr).contains(&p_fold) {
                    return true;
                }
                if ovh.owner_name.as_deref().unwrap_or("").is_empty()
                    && fuzzy_name_match_score_internal(pq, &[Some(raw_ocr.clone())], 0.76) >= 0.76
                {
                    return true;
                }
            }
            false
        };

        let filtered_ids: Vec<i64> = sheets_meta
            .par_iter()
            .filter_map(|sm| {
                let sid = sm.id;
                let s_facts = facts_by_sheet.get(&sid).map(|v| v.as_slice()).unwrap_or(&[]);
                let s_ovhs = ovh_by_overview_sheet
                    .get(&sid)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                if let Some(sq) = st_q {
                    let mut st_cands: Vec<Option<String>> = Vec::new();
                    st_cands.push(sm.street_raw.clone());
                    if let Some(hh_sts) = hh_streets_by_sheet.get(&sid) {
                        for hst in hh_sts {
                            st_cands.push(Some(hst.clone()));
                        }
                    }
                    if let Some(ov_sts) = matched_ovh_settlements_by_sheet.get(&sid) {
                        for ost in ov_sts {
                            st_cands.push(Some(ost.clone()));
                        }
                    }
                    for ovh in s_ovhs {
                        st_cands.push(ovh.settlement_name.clone());
                    }
                    if !text_matches_street_internal(&st_cands, Some(sq)) {
                        return None;
                    }
                }

                let mut name_parts: Vec<Option<String>> = Vec::new();
                if let Some(ref ho) = sm.household_owner {
                    name_parts.push(Some(ho.clone()));
                }
                if let Some(ref cn) = sm.commissioner_name {
                    name_parts.push(Some(cn.clone()));
                }
                if let Some(m_owns) = matched_ovh_owners_by_sheet.get(&sid) {
                    for mo in m_owns {
                        name_parts.push(Some(mo.clone()));
                    }
                }
                for ovh in s_ovhs {
                    name_parts.push(ovh.owner_name.clone());
                    name_parts.push(ovh.remarks.clone());
                }
                if let Some(md_ns) = md_names_by_sheet.get(&sid) {
                    for mn in md_ns {
                        name_parts.push(Some(mn.clone()));
                    }
                }
                if let Some(hh_ns) = hh_names_by_sheet.get(&sid) {
                    for hn in hh_ns {
                        name_parts.push(Some(hn.clone()));
                    }
                }

                if let Some(qv) = q_val {
                    let q_folded = fold_text_diacritics_internal(qv);
                    let meta_text = fold_text_diacritics_internal(
                        &[
                            sm.municipality.as_str(),
                            sm.district.as_deref().unwrap_or(""),
                            sm.street_raw.as_deref().unwrap_or(""),
                            sm.house_number.as_deref().unwrap_or(""),
                            sm.slovakiana_key.as_str(),
                            sm.household_owner.as_deref().unwrap_or(""),
                        ]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" "),
                    );
                    let q_match = (!q_folded.is_empty() && meta_text.contains(&q_folded))
                        || fuzzy_name_match_score_internal(qv, &name_parts, 0.76) >= 0.76
                        || s_facts.iter().any(|pf| {
                            pf.matches_person_name(Some(qv))
                                || pf.matches_birth_year(Some(qv))
                                || pf.matches_birth_place(Some(qv))
                        })
                        || s_ovhs.iter().any(|ovh| ovh_matches_person(ovh, qv));
                    if !q_match {
                        return None;
                    }
                }

                if pn_q.is_some() || by_q.is_some() || bp_q.is_some() {
                    let fact_combined = s_facts.iter().any(|pf| {
                        pf.matches_person_name(pn_q)
                            && pf.matches_birth_year(by_q)
                            && pf.matches_birth_place(bp_q)
                    });
                    let header_match = pn_q
                        .map(|pq| {
                            (fuzzy_name_match_score_internal(pq, &name_parts, 0.76) >= 0.76
                                || s_ovhs.iter().any(|ovh| ovh_matches_person(ovh, pq)))
                                && (by_q.is_none()
                                    || s_facts.iter().any(|pf| pf.matches_birth_year(by_q)))
                                && (bp_q.is_none()
                                    || s_facts.iter().any(|pf| pf.matches_birth_place(bp_q)))
                        })
                        .unwrap_or(false);
                    if !(fact_combined || header_match) {
                        return None;
                    }
                }

                Some(sid)
            })
            .collect();

        let mut previews_by_sheet = serde_json::Map::new();
        let mut fallback_streets = serde_json::Map::new();
        for &sid in &filtered_ids {
            if let Some(s_facts) = facts_by_sheet.get(&sid) {
                let mut p_list = Vec::with_capacity(s_facts.len());
                for pf in s_facts {
                    let is_match = if pn_q.is_some() || by_q.is_some() || bp_q.is_some() {
                        pf.matches_person_name(pn_q)
                            && pf.matches_birth_year(by_q)
                            && pf.matches_birth_place(bp_q)
                    } else if let Some(qv) = q_val {
                        pf.matches_person_name(Some(qv))
                            || pf.matches_birth_year(Some(qv))
                            || pf.matches_birth_place(Some(qv))
                    } else {
                        false
                    };
                    p_list.push(pf.serialize_preview(is_match));
                }
                previews_by_sheet.insert(sid.to_string(), Value::Array(p_list));
            }
            let fb_st = hh_streets_by_sheet
                .get(&sid)
                .and_then(|v| v.first())
                .or_else(|| {
                    matched_ovh_settlements_by_sheet
                        .get(&sid)
                        .and_then(|v| v.first())
                });
            if let Some(st_s) = fb_st {
                fallback_streets.insert(sid.to_string(), Value::String(st_s.clone()));
            }
        }

        let mut sorted_streets: Vec<String> = available_streets.into_iter().collect();
        sorted_streets.sort();

        let ov_cnt_map: serde_json::Map<String, Value> = overview_counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();
        let md_cnt_map: serde_json::Map<String, Value> = municipal_districts_counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), json!(v)))
            .collect();

        let out = json!({
            "filtered_sheet_ids": filtered_ids,
            "available_streets": sorted_streets,
            "overview_counts": Value::Object(ov_cnt_map),
            "municipal_districts_counts": Value::Object(md_cnt_map),
            "fallback_streets": Value::Object(fallback_streets),
            "previews_by_sheet": Value::Object(previews_by_sheet),
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

pub fn batch_find_linked_house_sheets_conn(
    conn: &Connection,
    target_sheet_ids: &[i64],
) -> Result<Value, rusqlite::Error> {
    if target_sheet_ids.is_empty() {
        return Ok(json!({}));
    }

    #[derive(Clone)]
    struct SheetRow {
        id: i64,
        slovakiana_key: String,
        census_year: i64,
        municipality: String,
        municipality_key: Option<String>,
        district: Option<String>,
        street_raw: Option<String>,
        house_number: Option<String>,
        house_number_int: Option<i64>,
        obvod_number: Option<String>,
        sheet_number: Option<String>,
        box_number: Option<String>,
        is_blackened: bool,
        download_status: String,
        household_owner: Option<String>,
        sheet_type: String,
    }

    let read_sh = |r: &rusqlite::Row<'_>| -> Result<SheetRow, rusqlite::Error> {
        Ok(SheetRow {
            id: r.get(0)?,
            slovakiana_key: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
            census_year: r.get::<_, Option<i64>>(2)?.unwrap_or(1930),
            municipality: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            municipality_key: r.get(4)?,
            district: r.get(5)?,
            street_raw: r.get(6)?,
            house_number: r.get(7)?,
            house_number_int: r.get(8)?,
            obvod_number: r.get(9)?,
            sheet_number: r.get(10)?,
            box_number: r.get(11)?,
            is_blackened: r.get::<_, Option<bool>>(12)?.unwrap_or(false),
            download_status: r
                .get::<_, Option<String>>(13)?
                .unwrap_or_else(|| "cataloged".to_string()),
            household_owner: r.get(14)?,
            sheet_type: r
                .get::<_, Option<String>>(15)?
                .unwrap_or_else(|| "household".to_string()),
        })
    };

    let base_select = "SELECT id, slovakiana_key, census_year, municipality, municipality_key,
                              district, street_raw, house_number, house_number_int, obvod_number,
                              sheet_number, box_number, is_blackened, download_status,
                              household_owner, sheet_type
                       FROM censussheet";

    let mut targets: Vec<(SheetRow, String, String)> = Vec::new();
    let mut hn_ints: HashSet<i64> = HashSet::new();
    let mut hn_strs: HashSet<String> = HashSet::new();

    for chunk in target_sheet_ids.chunks(450) {
        let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
        let sql = format!("{} WHERE id IN ({})", base_select, ph.join(", "));
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(chunk.iter()), read_sh)?;
        for s in rows.flatten() {
            if matches!(s.sheet_type.as_str(), "overview" | "municipal_overview") {
                continue;
            }
            let t_muni = canonical_municipality_key_internal(Some(&s.municipality));
            let t_hn = normalize_house_number_internal(s.house_number.as_deref());
            if t_muni.is_empty() || t_hn.is_empty() {
                continue;
            }
            if let Some(hi) = s.house_number_int {
                hn_ints.insert(hi);
            }
            if let Some(ref hs) = s.house_number {
                let hst = hs.trim();
                if !hst.is_empty() {
                    hn_strs.insert(hst.to_string());
                }
            }
            targets.push((s, t_muni, t_hn));
        }
    }

    if targets.is_empty() {
        return Ok(json!({}));
    }

    let mut stmt_all = conn.prepare(&format!(
        "{} WHERE house_number IS NOT NULL AND sheet_type NOT IN ('overview', 'municipal_overview')",
        base_select
    ))?;
    let all_cands = stmt_all.query_map([], read_sh)?;

    let mut by_muni_hn: HashMap<(String, String), Vec<SheetRow>> = HashMap::new();
    for c in all_cands.flatten() {
        let has_hn_match = c.house_number_int.map(|i| hn_ints.contains(&i)).unwrap_or(false)
            || c.house_number
                .as_deref()
                .map(|s| hn_strs.contains(s.trim()))
                .unwrap_or(false);
        if !has_hn_match && (!hn_ints.is_empty() || !hn_strs.is_empty()) {
            let c_hn_tmp = normalize_house_number_internal(c.house_number.as_deref());
            if c_hn_tmp.is_empty() {
                continue;
            }
        }
        let c_muni = canonical_municipality_key_internal(Some(&c.municipality));
        let c_hn = normalize_house_number_internal(c.house_number.as_deref());
        if !c_muni.is_empty() && !c_hn.is_empty() {
            by_muni_hn.entry((c_muni, c_hn)).or_default().push(c);
        }
    }

    let mut matched_by_target: HashMap<i64, Vec<SheetRow>> = HashMap::new();
    let mut needed_other_ids: HashSet<i64> = HashSet::new();

    for (s, t_muni, t_hn) in &targets {
        if let Some(cands) = by_muni_hn.get(&(t_muni.clone(), t_hn.clone())) {
            let mut cross: Vec<SheetRow> = Vec::new();
            for other in cands {
                if other.id == s.id || other.census_year == s.census_year {
                    continue;
                }
                if let (Some(ref smk), Some(ref omk)) = (&s.municipality_key, &other.municipality_key)
                {
                    if !smk.is_empty() && !omk.is_empty() && smk != omk {
                        continue;
                    }
                }
                if !is_compatible_settlement_internal(
                    s.street_raw.as_deref(),
                    other.street_raw.as_deref(),
                    Some(&s.municipality),
                ) {
                    continue;
                }
                needed_other_ids.insert(other.id);
                cross.push(other.clone());
            }
            if !cross.is_empty() {
                cross.sort_by(|a, b| {
                    (
                        a.census_year,
                        a.sheet_number.as_deref().unwrap_or(""),
                        a.id,
                    )
                        .cmp(&(
                            b.census_year,
                            b.sheet_number.as_deref().unwrap_or(""),
                            b.id,
                        ))
                });
                matched_by_target.insert(s.id, cross);
            }
        }
    }

    if needed_other_ids.is_empty() {
        return Ok(json!({}));
    }

    let other_id_vec: Vec<i64> = needed_other_ids.into_iter().collect();
    let mut hh_head_by_sheet: HashMap<i64, String> = HashMap::new();
    for chunk in other_id_vec.chunks(450) {
        let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
        let sql = format!(
            "SELECT census_sheet_id, house_owner_name, notes_raw
             FROM censushousehold
             WHERE census_sheet_id IN ({})
             ORDER BY id ASC",
            ph.join(", ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(chunk.iter()), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;
        for (csid, owner_opt, notes_opt) in rows.flatten() {
            if hh_head_by_sheet.contains_key(&csid) {
                continue;
            }
            let mut head_name: Option<String> = None;
            if let Some(ref ns) = notes_opt {
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(ns) {
                    let hg = map
                        .get("head_given_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let hs = map
                        .get("head_surname")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let comb = format!("{} {}", hg, hs).trim().to_string();
                    if !comb.is_empty() {
                        head_name = Some(comb);
                    } else if let Some(hon) = map
                        .get("head_of_household_name")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        head_name = Some(hon.to_string());
                    }
                }
            }
            if head_name.is_none() {
                head_name = owner_opt.filter(|s| !s.is_empty());
            }
            if let Some(hn) = head_name {
                hh_head_by_sheet.insert(csid, hn);
            }
        }
    }

    let mut facts_by_other: HashMap<
        i64,
        Vec<(Option<String>, Option<String>, Option<String>, Option<i64>)>,
    > = HashMap::new();
    for chunk in other_id_vec.chunks(450) {
        let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
        let sql = format!(
            "SELECT census_sheet_id, COALESCE(given_name_sk, given_name_raw),
                    COALESCE(surname_normalized, surname_raw), relation_to_head, birth_year_est
             FROM personfact
             WHERE census_sheet_id IN ({})
             ORDER BY id ASC",
            ph.join(", ")
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(chunk.iter()), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<i64>>(4)?,
            ))
        })?;
        for (csid, g, s, rel, by) in rows.flatten() {
            facts_by_other.entry(csid).or_default().push((g, s, rel, by));
        }
    }

    let mut entry_cache: HashMap<i64, Value> = HashMap::new();
    let mut out_map = serde_json::Map::new();

    for (sid, cross_list) in matched_by_target {
        let mut entries = Vec::with_capacity(cross_list.len());
        for o in cross_list {
            let oid = o.id;
            let val = entry_cache
                .entry(oid)
                .or_insert_with(|| {
                    let f_list = facts_by_other.get(&oid).map(|v| v.as_slice()).unwrap_or(&[]);
                    let mut head_name = hh_head_by_sheet
                        .get(&oid)
                        .cloned()
                        .or_else(|| o.household_owner.clone().filter(|s| !s.is_empty()));
                    if head_name.is_none() {
                        if let Some((g0, s0, _, _)) = f_list.first() {
                            let comb = format!(
                                "{} {}",
                                g0.as_deref().unwrap_or(""),
                                s0.as_deref().unwrap_or("")
                            )
                            .trim()
                            .to_string();
                            if !comb.is_empty() {
                                head_name = Some(comb);
                            }
                        }
                    }
                    let persons_preview: Vec<Value> = f_list
                        .iter()
                        .take(12)
                        .map(|(g, s, rel, by)| {
                            json!({
                                "given_name": g,
                                "surname": s,
                                "relation_to_head": rel,
                                "birth_year": by,
                            })
                        })
                        .collect();
                    let canonical_muni =
                        canonical_municipality_display_internal(Some(&o.municipality))
                            .unwrap_or_else(|| o.municipality.clone());
                    let settlement = o
                        .street_raw
                        .clone()
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| canonical_muni.clone());

                    json!({
                        "id": o.id,
                        "cair_id": o.slovakiana_key,
                        "slovakiana_key": o.slovakiana_key,
                        "slovakiana_id": o.slovakiana_key,
                        "census_year": o.census_year,
                        "municipality": canonical_muni,
                        "district": o.district,
                        "settlement": settlement,
                        "house_number": o.house_number,
                        "enumeration_district": o.obvod_number,
                        "sheet_number": o.sheet_number,
                        "box_number": o.box_number,
                        "is_blackened": o.is_blackened,
                        "download_status": o.download_status,
                        "head_of_household": head_name,
                        "persons_count": f_list.len(),
                        "persons_preview": persons_preview,
                    })
                })
                .clone();
            entries.push(val);
        }
        out_map.insert(sid.to_string(), Value::Array(entries));
    }

    Ok(Value::Object(out_map))
}

#[pyfunction]
pub fn batch_find_linked_house_sheets_rs(
    py: Python<'_>,
    db_path: String,
    target_sheet_ids: Vec<i64>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let val = batch_find_linked_house_sheets_conn(&conn, &target_sheet_ids)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string(&val).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, filters_json, only_undone=true, run_ocr=true, overview_only=false))]
pub fn find_matching_census_sheets_rs(
    py: Python<'_>,
    db_path: String,
    filters_json: String,
    only_undone: bool,
    run_ocr: bool,
    overview_only: bool,
) -> PyResult<Vec<i64>> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let filters: Value = serde_json::from_str(&filters_json).unwrap_or_else(|_| json!({}));

        let years: Vec<i64> = filters
            .get("years")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_i64())
            .map(|y| vec![if y == 1940 { 1940 } else { 1930 }])
            .unwrap_or_else(|| vec![1930]);
        let include_blackened = filters
            .get("include_blackened")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let category_keys: Vec<String> = filters
            .get("category_keys")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let muni_keys: Vec<String> = category_keys
            .iter()
            .filter(|k| k.len() >= 4)
            .cloned()
            .collect();
        let dist_keys: Vec<String> = category_keys
            .iter()
            .filter(|k| k.len() == 3)
            .cloned()
            .collect();
        let municipality = filters
            .get("municipality")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let district = filters
            .get("district")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let search_text = filters
            .get("search_text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        let eff_muni_name = if !municipality.is_empty() {
            municipality.clone()
        } else if !muni_keys.is_empty() && !search_text.is_empty() {
            search_text.clone()
        } else {
            String::new()
        };
        let eff_muni_canon = canonical_municipality_key_internal(Some(&eff_muni_name));
        let search_canon = canonical_municipality_key_internal(Some(&search_text));
        let search_folded = fold_text_diacritics_internal(&search_text);
        let dist_folded = fold_text_diacritics_internal(&district);

        let mut sql = String::from(
            "SELECT id, slovakiana_key, census_year, municipality, municipality_folded,
                    municipality_key, district, district_folded, district_key,
                    street_raw, house_number, house_number_int, title_raw,
                    household_owner, sheet_type, download_status, front_scan_id
             FROM censussheet
             WHERE has_images = 1",
        );
        if !years.is_empty() {
            sql.push_str(&format!(" AND census_year = {}", years[0]));
        }
        if !include_blackened {
            sql.push_str(" AND is_blackened = 0");
        }
        if only_undone && !overview_only {
            if run_ocr {
                sql.push_str(
                    " AND (download_status IS NULL OR download_status NOT IN ('ocr_done', 'extracted', 'matched'))",
                );
            } else {
                sql.push_str(
                    " AND (download_status IS NULL OR download_status NOT IN ('downloaded', 'sliced', 'ocr_done', 'extracted', 'matched'))",
                );
            }
        }

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Option<i64>>(2)?.unwrap_or(1930),
                    r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                    r.get::<_, Option<String>>(7)?,
                    r.get::<_, Option<String>>(8)?,
                    r.get::<_, Option<String>>(9)?,
                    r.get::<_, Option<String>>(10)?,
                    r.get::<_, Option<i64>>(11)?,
                    r.get::<_, Option<String>>(12)?,
                    r.get::<_, Option<String>>(13)?,
                    r.get::<_, Option<String>>(14)?
                        .unwrap_or_else(|| "household".to_string()),
                    r.get::<_, Option<String>>(15)?.unwrap_or_default(),
                    r.get::<_, Option<i64>>(16)?,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut matched: Vec<(i32, i32, i64, i64, i64)> = Vec::new();
        for r in rows.flatten() {
            let (
                sid,
                skey,
                cyear,
                muni,
                muni_f,
                muni_k,
                dist,
                dist_f,
                dist_k,
                street,
                hn,
                hn_int,
                title,
                owner,
                stype,
                dl_status,
                front_sid,
            ) = r;
            let hn_non_empty = hn.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
            let is_potential_ov =
                matches!(stype.as_str(), "municipal_overview" | "overview") || !hn_non_empty;

            if overview_only {
                if !is_potential_ov {
                    continue;
                }
                if only_undone {
                    let is_done = matches!(dl_status.as_str(), "ocr_done" | "extracted" | "matched")
                        || (dl_status == "sliced"
                            && stype == "household"
                            && front_sid.filter(|&x| x > 0).is_some());
                    if is_done {
                        continue;
                    }
                }
            }

            if !muni_keys.is_empty() || !municipality.is_empty() {
                let key_ok = muni_k
                    .as_ref()
                    .map(|k| muni_keys.contains(k))
                    .unwrap_or(false);
                let name_ok = !eff_muni_name.is_empty()
                    && sheet_matches_municipality_internal(
                        &muni,
                        &muni_f,
                        muni_k.as_deref(),
                        Some(&eff_muni_name),
                    );
                if !(key_ok || name_ok) {
                    continue;
                }
            } else if !dist_keys.is_empty() || !district.is_empty() {
                let d_key_ok = dist_k
                    .as_ref()
                    .map(|k| dist_keys.contains(k))
                    .unwrap_or(false);
                let s_dist_f = match dist.as_deref().filter(|s| !s.is_empty()) {
                    Some(d) => fold_text_diacritics_internal(d),
                    None => fold_text_diacritics_internal(dist_f.as_deref().unwrap_or("")),
                };
                let d_name_ok = !dist_folded.is_empty() && s_dist_f.contains(&dist_folded);
                if !(d_key_ok || d_name_ok) {
                    continue;
                }
            }

            if !search_text.is_empty() {
                let s_muni_canon = canonical_municipality_key_internal(Some(&muni));
                let is_same_muni = (!eff_muni_canon.is_empty() && search_canon == eff_muni_canon)
                    || (!search_canon.is_empty()
                        && !s_muni_canon.is_empty()
                        && search_canon == s_muni_canon);
                if !is_same_muni {
                    let haystack = fold_text_diacritics_internal(
                        &[
                            title.as_deref().unwrap_or(""),
                            muni.as_str(),
                            street.as_deref().unwrap_or(""),
                            hn.as_deref().unwrap_or(""),
                            skey.as_str(),
                            owner.as_deref().unwrap_or(""),
                        ]
                        .into_iter()
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(" "),
                    );
                    if !haystack.contains(&search_folded) {
                        continue;
                    }
                }
            }

            let prio_ov = if is_potential_ov { 0 } else { 1 };
            let prio_dl = if matches!(dl_status.as_str(), "downloaded" | "sliced" | "queued") {
                0
            } else if dl_status == "error" {
                2
            } else {
                1
            };
            matched.push((prio_ov, prio_dl, cyear, hn_int.unwrap_or(999999), sid));
        }

        matched.sort();
        Ok(matched.into_iter().map(|x| x.4).collect())
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, census_year=1930, municipality=None, district=None, status_filter=None, order_desc=true))]
pub fn query_census_sheet_ids_by_location_rs(
    py: Python<'_>,
    db_path: String,
    census_year: i64,
    municipality: Option<String>,
    district: Option<String>,
    status_filter: Option<String>,
    order_desc: bool,
) -> PyResult<(Vec<i64>, Vec<String>)> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let resolved_year = if census_year == 1940 { 1940 } else { 1930 };

        let mut sql = format!(
            "SELECT id, municipality, municipality_folded, municipality_key,
                    district, district_folded, district_key, street_raw
             FROM censussheet
             WHERE census_year = {}",
            resolved_year
        );

        if let Some(ref sf) = status_filter {
            let s = sf.trim();
            if !s.is_empty() {
                if s == "idle" {
                    sql.push_str(" AND download_status = 'cataloged'");
                } else if matches!(s, "pending" | "pending_ocr" | "undone" | "not_done") {
                    sql.push_str(
                        " AND (download_status IS NULL OR download_status NOT IN ('ocr_done', 'extracted', 'matched'))",
                    );
                } else {
                    let safe = s.replace('\'', "''");
                    sql.push_str(&format!(" AND download_status = '{}'", safe));
                }
            }
        }

        if order_desc {
            sql.push_str(" ORDER BY id DESC");
        } else {
            sql.push_str(" ORDER BY id ASC");
        }

        #[derive(Clone)]
        struct LocRow {
            id: i64,
            muni: String,
            muni_f: String,
            muni_k: Option<String>,
            dist: Option<String>,
            dist_f: Option<String>,
            dist_k: Option<String>,
            street: Option<String>,
        }

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut all_matching: Vec<LocRow> = stmt
            .query_map([], |r| {
                Ok(LocRow {
                    id: r.get(0)?,
                    muni: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    muni_f: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    muni_k: r.get(3)?,
                    dist: r.get(4)?,
                    dist_f: r.get(5)?,
                    dist_k: r.get(6)?,
                    street: r.get(7)?,
                })
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let m_opt = municipality.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty());
        let has_muni = m_opt.is_some();
        if let Some(m_val) = m_opt {
            all_matching.retain(|s| {
                sheet_matches_municipality_internal(
                    &s.muni,
                    &s.muni_f,
                    s.muni_k.as_deref(),
                    Some(m_val),
                )
            });
            let target_canon = canonical_municipality_key_internal(Some(m_val));
            let exact: Vec<LocRow> = all_matching
                .iter()
                .filter(|s| {
                    canonical_municipality_key_internal(Some(&s.muni)) == target_canon
                        || s.muni_k.as_deref() == Some(m_val)
                })
                .cloned()
                .collect();
            if !exact.is_empty() {
                all_matching = exact;
            }
        }

        if let Some(d_val) = district.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let dist_matching: Vec<LocRow> = all_matching
                .iter()
                .filter(|s| {
                    sheet_matches_district_internal(
                        &s.muni,
                        s.dist.as_deref(),
                        s.dist_f.as_deref(),
                        s.dist_k.as_deref(),
                        Some(d_val),
                        has_muni,
                    )
                })
                .cloned()
                .collect();
            if !dist_matching.is_empty() || !has_muni {
                all_matching = dist_matching;
            }
        }

        let mut streets_set: HashSet<String> = HashSet::new();
        let mut ids_out: Vec<i64> = Vec::with_capacity(all_matching.len());
        for s in all_matching {
            ids_out.push(s.id);
            if let Some(ref st) = s.street {
                let st_t = st.trim();
                if !st_t.is_empty() && !matches!(st_t, "-" | "—" | "?") {
                    streets_set.insert(st_t.to_string());
                }
            }
        }
        let mut streets: Vec<String> = streets_set.into_iter().collect();
        streets.sort();
        Ok((ids_out, streets))
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, census_year=1930, municipality=None, district=None, street=None, query=None))]
pub fn filter_overview_houses_ids_rs(
    py: Python<'_>,
    db_path: String,
    census_year: i64,
    municipality: Option<String>,
    district: Option<String>,
    street: Option<String>,
    query: Option<String>,
) -> PyResult<Vec<i64>> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let resolved_year = if census_year == 1940 { 1940 } else { 1930 };

        struct OvhRow {
            id: i64,
            csid: i64,
            r_idx: i64,
            hn: Option<String>,
            muni: String,
            dist: Option<String>,
            sett: Option<String>,
            owner: Option<String>,
            owner_f: Option<String>,
            rem: Option<String>,
            ocr: Option<String>,
        }

        let mut stmt = conn
            .prepare(
                "SELECT id, census_sheet_id, row_index, house_number, municipality,
                        district, settlement_name, owner_name, owner_surname_folded,
                        remarks, raw_ocr_text
                 FROM censusoverviewhouse
                 WHERE census_year = ?1
                 ORDER BY census_sheet_id ASC, row_index ASC, id ASC",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let raw_rows = stmt
            .query_map(params![resolved_year], |r| {
                Ok(OvhRow {
                    id: r.get(0)?,
                    csid: r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                    r_idx: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
                    hn: r.get(3)?,
                    muni: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    dist: r.get(5)?,
                    sett: r.get(6)?,
                    owner: r.get(7)?,
                    owner_f: r.get(8)?,
                    rem: r.get(9)?,
                    ocr: r.get(10)?,
                })
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut seen_keys: HashSet<(i64, String)> = HashSet::new();
        let mut houses: Vec<OvhRow> = Vec::new();
        for h in raw_rows.flatten() {
            let hn_norm = normalize_house_number_internal(h.hn.as_deref());
            let hk = (
                h.csid,
                if !hn_norm.is_empty() {
                    hn_norm
                } else {
                    h.r_idx.to_string()
                },
            );
            if seen_keys.insert(hk) {
                houses.push(h);
            }
        }

        if let Some(m_val) = municipality.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let target_canon = canonical_municipality_key_internal(Some(m_val));
            let m_fold = fold_text_diacritics_internal(m_val);
            houses.retain(|h| {
                canonical_municipality_key_internal(Some(&h.muni)) == target_canon
                    || (!m_fold.is_empty()
                        && fold_text_diacritics_internal(&h.muni).contains(&m_fold))
            });
        }

        if let Some(d_val) = district.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let d_fold = fold_text_diacritics_internal(d_val);
            houses.retain(|h| {
                !d_fold.is_empty()
                    && fold_text_diacritics_internal(h.dist.as_deref().unwrap_or(""))
                        .contains(&d_fold)
            });
        }

        if let Some(st_val) = street.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            houses.retain(|h| text_matches_street_internal(&[h.sett.clone()], Some(st_val)));
        }

        if let Some(q_val) = query.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let q_fold = fold_text_diacritics_internal(q_val);
            let matched_ids: Vec<i64> = houses
                .par_iter()
                .filter_map(|h| {
                    let cands = vec![h.owner.clone(), h.owner_f.clone(), h.rem.clone()];
                    let mut ok = fuzzy_name_match_score_internal(q_val, &cands, 0.76) >= 0.76;
                    if !ok {
                        if let Some(ref raw_ocr) = h.ocr {
                            if !q_fold.is_empty()
                                && fold_text_diacritics_internal(raw_ocr).contains(&q_fold)
                            {
                                ok = true;
                            } else if h.owner.as_deref().unwrap_or("").is_empty()
                                && fuzzy_name_match_score_internal(
                                    q_val,
                                    &[Some(raw_ocr.clone())],
                                    0.76,
                                ) >= 0.76
                            {
                                ok = true;
                            }
                        }
                    }
                    if !ok && !q_fold.is_empty() {
                        let meta_f = fold_text_diacritics_internal(&format!(
                            "{} {} {}",
                            h.hn.as_deref().unwrap_or(""),
                            h.sett.as_deref().unwrap_or(""),
                            h.rem.as_deref().unwrap_or("")
                        ));
                        if meta_f.contains(&q_fold) {
                            ok = true;
                        }
                    }
                    if ok {
                        Some(h.id)
                    } else {
                        None
                    }
                })
                .collect();
            return Ok(matched_ids);
        }

        Ok(houses.into_iter().map(|h| h.id).collect())
    })
}
