use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rusqlite::{params, params_from_iter};
use serde_json::{json, Value};

use crate::gazetteer::compute_person_fact_search_fields_internal;
use crate::lexicon::{resolve_given_cid_pub, surname_core_stem_internal};
use crate::{
    apply_digraphs_internal, open_sqlite, orthographic_fold_internal,
    paleographic_similarity_internal, phonetic_skeleton_internal, round4,
};

fn cursive_digraph_fold(folded: &str) -> String {
    let mut res = folded.to_lowercase();
    if res.starts_with("tr") && res.len() >= 5 {
        res = format!("t{}", &res[2..]);
    }
    res = res.replace("ur", "nz").replace("ni", "m").replace("in", "m");
    let mut dedup = String::with_capacity(res.len());
    let mut prev: Option<char> = None;
    for ch in res.chars() {
        if ch.is_ascii_lowercase() && prev == Some(ch) {
            continue;
        }
        dedup.push(ch);
        prev = Some(ch);
    }
    if dedup.len() == 6 {
        let b = dedup.as_bytes();
        if (b[0] == b's' || b[0] == b'f')
            && b[1] == b'o'
            && b[2] == b'm'
            && matches!(b[3], b'a' | b'e' | b'o')
            && (b[5] == b'n' || b[5] == b'r')
        {
            let mut m_bytes = b.to_vec();
            m_bytes[0] = b't';
            m_bytes[5] = b'n';
            if let Ok(s) = String::from_utf8(m_bytes) {
                return s;
            }
        }
    }
    dedup
}

#[inline]
fn paleo_pair_cost(a: u8, b: u8) -> f64 {
    if a == b {
        return 0.0;
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    match (x, y) {
        (b'n', b'u') | (b'r', b'z') => 0.08,
        (b'n', b'r') | (b's', b't') => 0.10,
        (b'f', b't')
        | (b'l', b't')
        | (b'c', b'k')
        | (b'i', b'y')
        | (b'i', b'j')
        | (b'u', b'v')
        | (b'r', b'v') => 0.12,
        (b'h', b'k') | (b'b', b'p') | (b's', b'z') => 0.14,
        (b'm', b'n') => 0.15,
        (b'a', b'o') => 0.16,
        (b'e', b'i') | (b'e', b'o') | (b'd', b't') | (b'g', b'q') => 0.18,
        (b'b', b'h') => 0.20,
        (b'h', b's') => 0.22,
        _ => 0.85,
    }
}

pub fn weighted_paleo_distance_internal(a: &str, b: &str) -> f64 {
    if a == b {
        return 0.0;
    }
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let la = ab.len();
    let lb = bb.len();
    if la == 0 {
        return (lb as f64) * 0.75;
    }
    if lb == 0 {
        return (la as f64) * 0.75;
    }
    let mut prev: Vec<f64> = (0..=lb).map(|j| (j as f64) * 0.75).collect();
    let mut curr = vec![0.0_f64; lb + 1];
    for (i, &ca) in ab.iter().enumerate() {
        curr[0] = ((i + 1) as f64) * 0.75;
        for (j, &cb) in bb.iter().enumerate() {
            let del = prev[j + 1] + 0.75;
            let ins = curr[j] + 0.75;
            let sub = prev[j] + paleo_pair_cost(ca, cb);
            curr[j + 1] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[lb]
}

pub fn paleographic_stem_similarity_internal(left: Option<&str>, right: Option<&str>) -> f64 {
    let (l, r) = match (left, right) {
        (Some(l), Some(r)) if !l.trim().is_empty() && !r.trim().is_empty() => (l.trim(), r.trim()),
        _ => return 0.0,
    };
    if l == r {
        return 1.0;
    }
    if l.to_lowercase() == r.to_lowercase() {
        return 0.99;
    }
    let f_l = surname_core_stem_internal(l);
    let f_r = surname_core_stem_internal(r);
    if f_l.is_empty() || f_r.is_empty() {
        return 0.0;
    }
    if f_l == f_r {
        return 0.97;
    }
    let max_len = f_l.len().max(f_r.len()).max(1) as f64;
    let direct_dist = weighted_paleo_distance_internal(&f_l, &f_r);
    let direct_sim = (1.0 - direct_dist / max_len).max(0.0);

    let dg_l = cursive_digraph_fold(&f_l);
    let dg_r = cursive_digraph_fold(&f_r);
    let digraph_sim = if dg_l == dg_r {
        0.95
    } else {
        let dg_max = dg_l.len().max(dg_r.len()).max(1) as f64;
        let dg_dist = weighted_paleo_distance_internal(&dg_l, &dg_r);
        (0.95 - dg_dist / dg_max).max(0.0)
    };

    round4(direct_sim.max(digraph_sim))
}

#[pyfunction]
#[pyo3(signature = (a, b))]
pub fn weighted_paleo_distance_rs(a: &str, b: &str) -> f64 {
    weighted_paleo_distance_internal(a, b)
}

#[pyfunction]
#[pyo3(signature = (left=None, right=None))]
pub fn paleographic_stem_similarity_rs(left: Option<&str>, right: Option<&str>) -> f64 {
    paleographic_stem_similarity_internal(left, right)
}

#[pyfunction]
pub fn match_census_to_local_matrika_rs(
    py: Python<'_>,
    db_path: String,
    census_fact_id: i64,
    now_iso: String,
) -> PyResult<String> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let census_row: Option<(
            i64,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<i64>,
        )> = conn
            .query_row(
                "SELECT id, surname_skeleton, surname_normalized, surname_raw,
                        mother_maiden_surname, given_canonical_id, given_name_sk,
                        given_name_raw, birth_year_est
                 FROM personfact WHERE id = ?1",
                params![census_fact_id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                        r.get(8)?,
                    ))
                },
            )
            .ok();

        let (
            cf_id,
            cf_skel_opt,
            cf_sur_norm,
            cf_sur_raw,
            cf_maiden,
            cf_cid_opt,
            cf_giv_sk,
            cf_giv_raw,
            cf_byear,
        ) = match census_row {
            Some(row) => row,
            None => return Ok("[]".to_string()),
        };

        let cf_sur_str = cf_sur_norm
            .as_deref()
            .filter(|s| !s.is_empty())
            .or_else(|| cf_sur_raw.as_deref().filter(|s| !s.is_empty()))
            .unwrap_or("");
        let sur_skel = cf_skel_opt
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| phonetic_skeleton_internal(cf_sur_str));
        let maiden_skel = cf_maiden
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(phonetic_skeleton_internal)
            .unwrap_or_default();
        if sur_skel.is_empty() && maiden_skel.is_empty() {
            return Ok("[]".to_string());
        }

        let given_cid = cf_cid_opt.filter(|s| !s.is_empty()).or_else(|| {
            cf_giv_sk
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| cf_giv_raw.as_deref().filter(|s| !s.is_empty()))
                .and_then(resolve_given_cid_pub)
        });

        let mut sql = String::from(
            "SELECT id, record_type, surname_skeleton, surname_normalized, surname_raw,
                    given_canonical_id, given_name_sk, given_name_raw, event_year
             FROM personfact
             WHERE record_type != 'census'",
        );
        if let Some(by) = cf_byear {
            sql.push_str(&format!(
                " AND event_year >= {} AND event_year <= {}",
                by - 4,
                by + 4
            ));
        }
        let mut skel_clauses: Vec<String> = Vec::new();
        for sk in [&sur_skel, &maiden_skel] {
            if !sk.is_empty() {
                let safe_sk = sk.replace('\'', "''");
                skel_clauses.push(format!("'{}'", safe_sk));
            }
        }
        if !skel_clauses.is_empty() {
            let in_list = skel_clauses.join(", ");
            if let Some(ref cid) = given_cid {
                let safe_cid = cid.replace('\'', "''");
                sql.push_str(&format!(
                    " AND (surname_skeleton IN ({}) OR given_canonical_id = '{}' OR surname_skeleton IS NULL)",
                    in_list, safe_cid
                ));
            } else {
                sql.push_str(&format!(
                    " AND (surname_skeleton IN ({}) OR surname_skeleton IS NULL)",
                    in_list
                ));
            }
        }

        let mut cands: Vec<(
            i64,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<i64>,
        )> = Vec::new();
        {
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get(0)?,
                        r.get::<_, Option<String>>(1)?
                            .unwrap_or_else(|| "baptism".to_string()),
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                        r.get(8)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for r in rows.flatten() {
                cands.push(r);
            }
        }

        let mut scored_matches: Vec<(i64, String, f64, String)> = cands
            .par_iter()
            .filter_map(
                |(
                    cand_id,
                    cand_rtype,
                    cand_skel_opt,
                    cand_sur_norm,
                    cand_sur_raw,
                    cand_cid_opt,
                    cand_giv_sk,
                    cand_giv_raw,
                    cand_eyear,
                )| {
                    if *cand_id <= 0 {
                        return None;
                    }
                    let cand_sur_str = cand_sur_norm
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .or_else(|| cand_sur_raw.as_deref().filter(|s| !s.is_empty()))
                        .unwrap_or("");
                    let cand_skel = cand_skel_opt
                        .clone()
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| phonetic_skeleton_internal(cand_sur_str));
                    let cand_cid = cand_cid_opt.clone().filter(|s| !s.is_empty()).or_else(|| {
                        cand_giv_sk
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .or_else(|| cand_giv_raw.as_deref().filter(|s| !s.is_empty()))
                            .and_then(resolve_given_cid_pub)
                    });

                    let mut sur_score = 0.0_f64;
                    if !sur_skel.is_empty() && !cand_skel.is_empty() && sur_skel == cand_skel {
                        sur_score = 1.0;
                    } else if !maiden_skel.is_empty()
                        && !cand_skel.is_empty()
                        && maiden_skel == cand_skel
                    {
                        sur_score = 0.95;
                    } else {
                        let sim = paleographic_similarity_internal(cf_sur_str, cand_sur_str);
                        if sim >= 0.78 {
                            sur_score = sim;
                        }
                    }
                    if sur_score < 0.75 {
                        return None;
                    }

                    let mut giv_score = 0.0_f64;
                    if given_cid.is_some() && given_cid == cand_cid {
                        giv_score = 1.0;
                    } else if let (Some(g1), Some(g2)) = (
                        cf_giv_raw.as_deref().filter(|s| !s.is_empty()),
                        cand_giv_raw.as_deref().filter(|s| !s.is_empty()),
                    ) {
                        giv_score = paleographic_similarity_internal(g1, g2);
                    }
                    if giv_score < 0.70 {
                        return None;
                    }

                    let year_score = match (cf_byear, *cand_eyear) {
                        (Some(by), Some(ey)) => {
                            let diff = (by - ey).abs();
                            if diff == 0 {
                                1.0
                            } else if diff == 1 {
                                0.92
                            } else if diff <= 3 {
                                0.80
                            } else {
                                0.60
                            }
                        }
                        _ => 0.85,
                    };

                    let total_score =
                        ((0.45 * sur_score + 0.35 * giv_score + 0.20 * year_score) * 1000.0).round()
                            / 1000.0;
                    if total_score < 0.76 {
                        return None;
                    }

                    let explanation = format!(
                        "Zhoda priezviska ({} ↔ {}), krstného mena ({} ↔ {}) a roku narodenia ({} ↔ matrika {}).",
                        cf_sur_raw.as_deref().unwrap_or("?"),
                        cand_sur_raw.as_deref().unwrap_or("?"),
                        cf_giv_raw.as_deref().unwrap_or("?"),
                        cand_giv_raw.as_deref().unwrap_or("?"),
                        cf_byear.map(|y| y.to_string()).unwrap_or_else(|| "?".to_string()),
                        cand_eyear.map(|y| y.to_string()).unwrap_or_else(|| "?".to_string()),
                    );
                    let agrees_json =
                        serde_json::to_string(&vec![explanation]).unwrap_or_else(|_| "[]".to_string());
                    Some((*cand_id, cand_rtype.clone(), total_score, agrees_json))
                },
            )
            .collect();

        scored_matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let db_ts = now_iso.replace('T', " ");
        let mut out_items: Vec<Value> = Vec::with_capacity(scored_matches.len());
        {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tx.execute(
                "DELETE FROM localrecordmatch WHERE census_fact_id = ?1",
                params![cf_id],
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            for (mat_id, mkind, conf, agrees_j) in &scored_matches {
                tx.execute(
                    "INSERT INTO localrecordmatch (
                        census_fact_id, matrika_fact_id, confidence_score, match_kind,
                        what_agrees_json, reviewed_by_user, created_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
                    params![cf_id, mat_id, conf, mkind, agrees_j, db_ts],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let new_id = tx.last_insert_rowid();
                out_items.push(json!({
                    "id": new_id,
                    "census_fact_id": cf_id,
                    "matrika_fact_id": mat_id,
                    "confidence_score": conf,
                    "match_kind": mkind,
                    "what_agrees_json": agrees_j,
                    "reviewed_by_user": false,
                }));
            }
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        serde_json::to_string(&out_items).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, sheet_ids=None))]
pub fn bulk_index_census_sheets_rs(
    py: Python<'_>,
    db_path: String,
    sheet_ids: Option<Vec<i64>>,
) -> PyResult<usize> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let read_sheet = |r: &rusqlite::Row<'_>| -> Result<(i64, String, String, String, String, String, String), rusqlite::Error> {
            let sid: i64 = r.get(0)?;
            let owner: String = r.get::<_, Option<String>>(1)?.unwrap_or_default();
            let muni: String = r.get::<_, Option<String>>(2)?.unwrap_or_default();
            let muni_f: String = r.get::<_, Option<String>>(3)?.unwrap_or_default();
            let dist: String = r.get::<_, Option<String>>(4)?.unwrap_or_default();
            let dist_f: String = r.get::<_, Option<String>>(5)?.unwrap_or_default();
            let street: String = r.get::<_, Option<String>>(6)?.unwrap_or_default();
            let hn: String = r.get::<_, Option<String>>(7)?.unwrap_or_default();
            let hn_int: Option<i64> = r.get(8)?;
            let box_n: String = r.get::<_, Option<String>>(9)?.unwrap_or_default();
            let sheet_n: String = r.get::<_, Option<String>>(10)?.unwrap_or_default();
            let cyear: i64 = r.get::<_, Option<i64>>(11)?.unwrap_or(1930);
            let skey: String = r.get::<_, Option<String>>(12)?.unwrap_or_default();
            let title: String = r.get::<_, Option<String>>(13)?.unwrap_or_default();

            let surnames_phonetic = if !owner.is_empty() {
                format!(
                    "{} {}",
                    apply_digraphs_internal(&owner),
                    phonetic_skeleton_internal(&owner)
                )
            } else {
                String::new()
            };
            let places_text = [muni.as_str(), muni_f.as_str(), dist.as_str(), dist_f.as_str(), street.as_str()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let hn_int_s = hn_int.map(|n| n.to_string()).unwrap_or_default();
            let cyear_s = cyear.to_string();
            let house_and_sheet = [
                hn.as_str(),
                hn_int_s.as_str(),
                box_n.as_str(),
                sheet_n.as_str(),
                cyear_s.as_str(),
                skey.as_str(),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

            Ok((
                sid,
                owner,
                surnames_phonetic,
                places_text,
                house_and_sheet,
                "scitaci harok census".to_string(),
                title,
            ))
        };

        let base_sql = "SELECT id, household_owner, municipality, municipality_folded,
                               district, district_folded, street_raw, house_number,
                               house_number_int, box_number, sheet_number, census_year,
                               slovakiana_key, title_raw
                        FROM censussheet";

        let mut rows_to_index = Vec::new();
        match sheet_ids {
            Some(ref ids) if ids.is_empty() => return Ok(0),
            Some(ref ids) => {
                for chunk in ids.chunks(450) {
                    let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                    let sql = format!("{} WHERE id IN ({})", base_sql, ph.join(", "));
                    let mut stmt = conn
                        .prepare(&sql)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let chunk_rows = stmt
                        .query_map(params_from_iter(chunk.iter()), read_sheet)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    rows_to_index.extend(chunk_rows);
                }
            }
            None => {
                let mut stmt = conn
                    .prepare(base_sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let all_rows = stmt
                    .query_map([], read_sheet)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                rows_to_index.extend(all_rows);
            }
        }

        let count = rows_to_index.len();
        {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if sheet_ids.is_none() {
                tx.execute(
                    "DELETE FROM search_index_fts WHERE entity_type = 'census_sheet'",
                    [],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            } else {
                let sids_str: Vec<String> =
                    rows_to_index.iter().map(|(sid, _, _, _, _, _, _)| sid.to_string()).collect();
                for chunk in sids_str.chunks(450) {
                    let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                    let del_sql = format!(
                        "DELETE FROM search_index_fts WHERE entity_type = 'census_sheet' AND entity_id IN ({})",
                        ph.join(", ")
                    );
                    tx.execute(&del_sql, params_from_iter(chunk.iter()))
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
            }

            {
                let mut ins_stmt = tx
                    .prepare(
                        "INSERT INTO search_index_fts (
                            entity_type, entity_id, names_text, surnames_phonetic,
                            places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
                        ) VALUES ('census_sheet', ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                for (sid, names_t, phon_t, places_t, hs_t, occ_t, notes_t) in &rows_to_index {
                    ins_stmt
                        .execute(params![
                            sid.to_string(),
                            names_t,
                            phon_t,
                            places_t,
                            hs_t,
                            occ_t,
                            notes_t
                        ])
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
            }
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(count)
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, fact_ids=None))]
pub fn bulk_index_person_facts_rs(
    py: Python<'_>,
    db_path: String,
    fact_ids: Option<Vec<i64>>,
) -> PyResult<usize> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let base_sql = "SELECT
                pf.id, pf.record_type, pf.entry_number, pf.person_row_number,
                pf.event_year, pf.birth_year_est, pf.birth_or_event_date, pf.age_at_event,
                pf.given_name_raw, pf.given_name_sk, pf.given_name_hu,
                pf.surname_raw, pf.surname_normalized, pf.mother_maiden_surname,
                pf.father_name, pf.mother_name, pf.spouse_name, pf.surname_candidates_json,
                pf.village_raw, pf.village_sk, pf.village_hu, pf.fs_place_standardized,
                pf.birth_place_raw, pf.birth_place_sk, pf.home_jurisdiction, pf.previous_residence,
                pf.house_number, pf.social_status_or_occupation, pf.occupation_position,
                pf.employer_workplace, pf.religion, pf.nationality, pf.relation_to_head,
                pf.marital_status, pf.observationes_raw,
                rr.raw_ocr_combined
            FROM personfact pf
            LEFT JOIN recordrow rr ON rr.id = pf.row_id";

        struct FactFtsItem {
            id: i64,
            surname_folded: Option<String>,
            surname_skeleton: Option<String>,
            given_canonical_id: Option<String>,
            village_folded: Option<String>,
            house_number_int: Option<i64>,
            birth_year_est: Option<i64>,
            names_text: String,
            surnames_phonetic: String,
            places_text: String,
            house_and_sheet: String,
            occupation_and_religion: String,
            notes_and_ocr: String,
        }

        let read_fact = |r: &rusqlite::Row<'_>| -> Result<FactFtsItem, rusqlite::Error> {
            let fid: i64 = r.get(0)?;
            let rtype: Option<String> = r.get(1)?;
            let entry_n: Option<String> = r.get(2)?;
            let prow_n: Option<String> = r.get(3)?;
            let eyear: Option<i64> = r.get(4)?;
            let byear: Option<i64> = r.get(5)?;
            let bdate: Option<String> = r.get(6)?;
            let age_ev: Option<String> = r.get(7)?;
            let g_raw: Option<String> = r.get(8)?;
            let g_sk: Option<String> = r.get(9)?;
            let g_hu: Option<String> = r.get(10)?;
            let s_raw: Option<String> = r.get(11)?;
            let s_norm: Option<String> = r.get(12)?;
            let maiden: Option<String> = r.get(13)?;
            let fath: Option<String> = r.get(14)?;
            let moth: Option<String> = r.get(15)?;
            let spouse: Option<String> = r.get(16)?;
            let sc_json: Option<String> = r.get(17)?;
            let v_raw: Option<String> = r.get(18)?;
            let v_sk: Option<String> = r.get(19)?;
            let v_hu: Option<String> = r.get(20)?;
            let fs_pl: Option<String> = r.get(21)?;
            let bp_raw: Option<String> = r.get(22)?;
            let bp_sk: Option<String> = r.get(23)?;
            let home_j: Option<String> = r.get(24)?;
            let prev_res: Option<String> = r.get(25)?;
            let hn: Option<String> = r.get(26)?;
            let occ: Option<String> = r.get(27)?;
            let occ_pos: Option<String> = r.get(28)?;
            let emp: Option<String> = r.get(29)?;
            let rel: Option<String> = r.get(30)?;
            let nat: Option<String> = r.get(31)?;
            let rel_head: Option<String> = r.get(32)?;
            let mar: Option<String> = r.get(33)?;
            let obs: Option<String> = r.get(34)?;
            let ocr_c: Option<String> = r.get(35)?;

            let sur_val = s_norm
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| s_raw.as_deref().filter(|s| !s.is_empty()))
                .unwrap_or("");
            let s_stem = surname_core_stem_internal(sur_val);
            let m_stem = maiden
                .as_deref()
                .map(surname_core_stem_internal)
                .unwrap_or_default();

            let (sf_fold, sf_skel, sf_cid, sf_vfold, sf_hint, sf_byear) =
                compute_person_fact_search_fields_internal(
                    s_raw.as_deref(),
                    s_norm.as_deref(),
                    if !s_stem.is_empty() {
                        Some(s_stem.as_str())
                    } else {
                        None
                    },
                    g_raw.as_deref(),
                    g_sk.as_deref(),
                    g_hu.as_deref(),
                    v_raw.as_deref(),
                    v_sk.as_deref(),
                    v_hu.as_deref(),
                    hn.as_deref(),
                    byear,
                    bdate.as_deref(),
                    rtype.as_deref(),
                    eyear,
                    age_ev.as_deref(),
                );

            let mut cand_names: Vec<String> = Vec::new();
            for opt_s in [
                &g_raw, &g_sk, &g_hu, &s_raw, &s_norm, &maiden, &fath, &moth, &spouse,
            ] {
                if let Some(ref s) = opt_s {
                    if !s.is_empty() {
                        cand_names.push(s.clone());
                    }
                }
            }
            if !s_stem.is_empty() {
                cand_names.push(s_stem.clone());
            }
            if !m_stem.is_empty() {
                cand_names.push(m_stem.clone());
            }
            if let Some(ref sc_s) = sc_json {
                if !sc_s.is_empty() && sc_s != "[]" {
                    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(sc_s) {
                        for item in arr {
                            if let Some(s) = item.as_str() {
                                if !s.is_empty() {
                                    cand_names.push(s.to_string());
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
                                            cand_names.push(v.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let mut phon_parts: Vec<String> = Vec::new();
            for s in &cand_names {
                if !s.is_empty() {
                    phon_parts.push(apply_digraphs_internal(s));
                    phon_parts.push(orthographic_fold_internal(s));
                    phon_parts.push(phonetic_skeleton_internal(s));
                }
            }

            let places_text = [
                v_raw.as_deref().unwrap_or(""),
                v_sk.as_deref().unwrap_or(""),
                v_hu.as_deref().unwrap_or(""),
                fs_pl.as_deref().unwrap_or(""),
                bp_raw.as_deref().unwrap_or(""),
                bp_sk.as_deref().unwrap_or(""),
                home_j.as_deref().unwrap_or(""),
                prev_res.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

            let hint_str = sf_hint.map(|n| n.to_string()).unwrap_or_default();
            let ey_str = eyear.map(|n| n.to_string()).unwrap_or_default();
            let by_str = sf_byear.map(|n| n.to_string()).unwrap_or_default();
            let house_and_sheet = [
                hn.as_deref().unwrap_or(""),
                hint_str.as_str(),
                entry_n.as_deref().unwrap_or(""),
                prow_n.as_deref().unwrap_or(""),
                ey_str.as_str(),
                by_str.as_str(),
                bdate.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

            let occ_text = [
                occ.as_deref().unwrap_or(""),
                occ_pos.as_deref().unwrap_or(""),
                emp.as_deref().unwrap_or(""),
                rel.as_deref().unwrap_or(""),
                nat.as_deref().unwrap_or(""),
                rel_head.as_deref().unwrap_or(""),
                mar.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

            let notes_text = [
                obs.as_deref().unwrap_or(""),
                ocr_c.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

            Ok(FactFtsItem {
                id: fid,
                surname_folded: sf_fold,
                surname_skeleton: sf_skel,
                given_canonical_id: sf_cid,
                village_folded: sf_vfold,
                house_number_int: sf_hint,
                birth_year_est: sf_byear,
                names_text: cand_names.join(" "),
                surnames_phonetic: phon_parts.join(" "),
                places_text,
                house_and_sheet,
                occupation_and_religion: occ_text,
                notes_and_ocr: notes_text,
            })
        };

        let mut items: Vec<FactFtsItem> = Vec::new();
        match fact_ids {
            Some(ref ids) if ids.is_empty() => return Ok(0),
            Some(ref ids) => {
                for chunk in ids.chunks(450) {
                    let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                    let sql = format!("{} WHERE pf.id IN ({})", base_sql, ph.join(", "));
                    let mut stmt = conn
                        .prepare(&sql)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let chunk_rows = stmt
                        .query_map(params_from_iter(chunk.iter()), read_fact)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    items.extend(chunk_rows);
                }
            }
            None => {
                let mut stmt = conn
                    .prepare(base_sql)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let all_rows = stmt
                    .query_map([], read_fact)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                items.extend(all_rows);
            }
        }

        let count = items.len();
        {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if fact_ids.is_none() {
                tx.execute(
                    "DELETE FROM search_index_fts WHERE entity_type = 'person_fact'",
                    [],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            } else {
                let fids_str: Vec<String> = items.iter().map(|it| it.id.to_string()).collect();
                for chunk in fids_str.chunks(450) {
                    let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                    let del_sql = format!(
                        "DELETE FROM search_index_fts WHERE entity_type = 'person_fact' AND entity_id IN ({})",
                        ph.join(", ")
                    );
                    tx.execute(&del_sql, params_from_iter(chunk.iter()))
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
            }

            {
                let mut upd_stmt = tx
                    .prepare(
                        "UPDATE personfact SET
                            surname_folded = ?1, surname_skeleton = ?2, given_canonical_id = ?3,
                            village_folded = ?4, house_number_int = ?5,
                            birth_year_est = COALESCE(birth_year_est, ?6)
                         WHERE id = ?7",
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let mut ins_stmt = tx
                    .prepare(
                        "INSERT INTO search_index_fts (
                            entity_type, entity_id, names_text, surnames_phonetic,
                            places_text, house_and_sheet, occupation_and_religion, notes_and_ocr
                        ) VALUES ('person_fact', ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                for it in &items {
                    upd_stmt
                        .execute(params![
                            it.surname_folded,
                            it.surname_skeleton,
                            it.given_canonical_id,
                            it.village_folded,
                            it.house_number_int,
                            it.birth_year_est,
                            it.id,
                        ])
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    ins_stmt
                        .execute(params![
                            it.id.to_string(),
                            it.names_text,
                            it.surnames_phonetic,
                            it.places_text,
                            it.house_and_sheet,
                            it.occupation_and_religion,
                            it.notes_and_ocr,
                        ])
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }
            }
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(count)
    })
}
