use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::Value;
use std::collections::HashMap;

use crate::open_sqlite;

static GEDCOM_MONTHS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (k, v) in [
        ("1", "JAN"),
        ("01", "JAN"),
        ("jan", "JAN"),
        ("januar", "JAN"),
        ("január", "JAN"),
        ("january", "JAN"),
        ("januarii", "JAN"),
        ("2", "FEB"),
        ("02", "FEB"),
        ("feb", "FEB"),
        ("februar", "FEB"),
        ("február", "FEB"),
        ("february", "FEB"),
        ("februarii", "FEB"),
        ("3", "MAR"),
        ("03", "MAR"),
        ("mar", "MAR"),
        ("marec", "MAR"),
        ("march", "MAR"),
        ("martii", "MAR"),
        ("4", "APR"),
        ("04", "APR"),
        ("apr", "APR"),
        ("april", "APR"),
        ("apríl", "APR"),
        ("aprilis", "APR"),
        ("5", "MAY"),
        ("05", "MAY"),
        ("maj", "MAY"),
        ("máj", "MAY"),
        ("may", "MAY"),
        ("maii", "MAY"),
        ("maji", "MAY"),
        ("6", "JUN"),
        ("06", "JUN"),
        ("jun", "JUN"),
        ("jún", "JUN"),
        ("june", "JUN"),
        ("junii", "JUN"),
        ("7", "JUL"),
        ("07", "JUL"),
        ("jul", "JUL"),
        ("júl", "JUL"),
        ("july", "JUL"),
        ("julii", "JUL"),
        ("julius", "JUL"),
        ("8", "AUG"),
        ("08", "AUG"),
        ("aug", "AUG"),
        ("august", "AUG"),
        ("augusti", "AUG"),
        ("augustus", "AUG"),
        ("9", "SEP"),
        ("09", "SEP"),
        ("sep", "SEP"),
        ("sept", "SEP"),
        ("september", "SEP"),
        ("septembris", "SEP"),
        ("10", "OCT"),
        ("oct", "OCT"),
        ("okt", "OCT"),
        ("oktober", "OCT"),
        ("október", "OCT"),
        ("october", "OCT"),
        ("octobris", "OCT"),
        ("11", "NOV"),
        ("nov", "NOV"),
        ("november", "NOV"),
        ("novembris", "NOV"),
        ("12", "DEC"),
        ("dec", "DEC"),
        ("december", "DEC"),
        ("decembris", "DEC"),
    ] {
        m.insert(k, v);
    }
    m
});

fn clean_gedcom_val(val: Option<&str>) -> String {
    let raw = match val {
        Some(s) => s,
        None => return String::new(),
    };
    let mut out = String::with_capacity(raw.len());
    let mut prev_ws = false;
    for ch in raw.chars() {
        if matches!(ch, '\r' | '\n' | '\0') {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

fn format_gedcom_date_internal(raw_date: Option<&str>, fallback_year: Option<i64>) -> String {
    let s = clean_gedcom_val(raw_date);
    if s.is_empty() {
        return fallback_year.map(|y| y.to_string()).unwrap_or_default();
    }
    if s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit()) {
        return s;
    }

    // Try DD.MM.YYYY or DD. M. YYYY
    let parts_dot: Vec<&str> = s.split('.').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();
    if parts_dot.len() == 3 {
        if let (Ok(day), Ok(mon_num), Ok(yr)) = (
            parts_dot[0].parse::<u32>(),
            parts_dot[1].parse::<u32>(),
            parts_dot[2].parse::<u32>(),
        ) {
            if (1..=31).contains(&day) && (1000..=2999).contains(&yr) {
                let mon_key = mon_num.to_string();
                if let Some(&mon_str) = GEDCOM_MONTHS.get(mon_key.as_str()) {
                    return format!("{} {} {}", day, mon_str, yr);
                }
            }
        }
    }

    // Try DD[.] MonthName [YYYY]
    let toks: Vec<&str> = s.split_whitespace().collect();
    if toks.len() >= 2 {
        let day_tok = toks[0].trim_end_matches('.');
        if let Ok(day) = day_tok.parse::<u32>() {
            if (1..=31).contains(&day) {
                let mon_key = toks[1].trim_end_matches('.').to_lowercase();
                if let Some(&mon_str) = GEDCOM_MONTHS.get(mon_key.as_str()) {
                    let yr_str = if toks.len() >= 3
                        && toks[2].len() == 4
                        && toks[2].bytes().all(|b| b.is_ascii_digit())
                    {
                        toks[2].to_string()
                    } else {
                        fallback_year.map(|y| y.to_string()).unwrap_or_default()
                    };
                    return if yr_str.is_empty() {
                        format!("{} {}", day, mon_str)
                    } else {
                        format!("{} {} {}", day, mon_str, yr_str)
                    };
                }
            }
        }
    }

    if let Some(fy) = fallback_year {
        let fy_s = fy.to_string();
        if !s.contains(&fy_s) {
            return format!("{} {}", s, fy_s).trim().to_string();
        }
    }
    s
}

fn format_gedcom_parent_name(full_name: &str, surname: &str) -> String {
    let cleaned = clean_gedcom_val(Some(full_name));
    let surn = clean_gedcom_val(Some(surname));
    if surn.is_empty() {
        return cleaned;
    }
    let cl_low = cleaned.to_lowercase();
    let sn_low = surn.to_lowercase();
    let prefix = format!("{} ", sn_low);
    if cl_low.starts_with(&prefix) {
        let given_part = cleaned[surn.len()..].trim();
        return format!("{} /{}/", given_part, surn).trim().to_string();
    }
    let suffix = format!(" {}", sn_low);
    if cl_low.ends_with(&suffix) && cleaned.len() > surn.len() {
        let given_part = cleaned[..cleaned.len() - surn.len()].trim();
        return format!("{} /{}/", given_part, surn).trim().to_string();
    }
    format!("{} /{}/", cleaned, surn)
}

fn write_csv_cell(out: &mut String, val: &str, is_first: bool) {
    if !is_first {
        out.push(',');
    }
    let sanitized = if val.len() > 1 && matches!(val.as_bytes()[0], b'=' | b'+' | b'-' | b'@') {
        format!("'{}", val)
    } else {
        val.to_string()
    };
    if sanitized.contains([',', '"', '\r', '\n']) {
        out.push('"');
        for ch in sanitized.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(&sanitized);
    }
}

struct ExportFact {
    id: i64,
    record_type: String,
    entry_number: Option<String>,
    event_year: Option<i64>,
    birth_or_event_date: Option<String>,
    baptism_or_burial_date: Option<String>,
    given_name_raw: Option<String>,
    given_name_sk: Option<String>,
    given_name_hu: Option<String>,
    surname_raw: Option<String>,
    surname_normalized: Option<String>,
    sex: Option<String>,
    relation_to_head: Option<String>,
    marital_status: Option<String>,
    father_name: Option<String>,
    mother_name: Option<String>,
    mother_maiden_surname: Option<String>,
    spouse_name: Option<String>,
    social_status_or_occupation: Option<String>,
    occupation_position: Option<String>,
    employer_workplace: Option<String>,
    religion: Option<String>,
    nationality: Option<String>,
    citizenship: Option<String>,
    literacy: Option<String>,
    birth_place_raw: Option<String>,
    birth_place_sk: Option<String>,
    home_jurisdiction: Option<String>,
    village_raw: Option<String>,
    village_sk: Option<String>,
    house_number: Option<String>,
    fs_place_standardized: Option<String>,
    godparents_or_witnesses: Option<String>,
    officiant: Option<String>,
    observationes_raw: Option<String>,
    observationes_events_json: String,
    cause_of_death: Option<String>,
    person_row_number: Option<String>,
    birth_year_est: Option<i64>,
    census_sheet_id: Option<i64>,
    census_household_id: Option<i64>,
    sheet_slovakiana_key: Option<String>,
    sheet_census_year: Option<i64>,
    sheet_municipality: Option<String>,
    sheet_district: Option<String>,
    sheet_urn_nbn: Option<String>,
    sheet_number: Option<String>,
}

fn load_export_facts(
    db_path: &str,
    only_new_to_fs: bool,
) -> Result<Vec<ExportFact>, rusqlite::Error> {
    let conn = open_sqlite(db_path)?;
    let where_sql = if only_new_to_fs {
        "WHERE pf.id IN (
            SELECT DISTINCT person_fact_id FROM familysearchmatch
            WHERE fs_person_id IS NOT NULL AND trim(fs_person_id) != '' AND confidence_score > 0
         )"
    } else {
        ""
    };
    let sql = format!(
        "SELECT
            pf.id, pf.record_type, pf.entry_number, pf.event_year,
            pf.birth_or_event_date, pf.baptism_or_burial_date,
            pf.given_name_raw, pf.given_name_sk, pf.given_name_hu,
            pf.surname_raw, pf.surname_normalized, pf.sex,
            pf.relation_to_head, pf.marital_status,
            pf.father_name, pf.mother_name, pf.mother_maiden_surname, pf.spouse_name,
            pf.social_status_or_occupation, pf.occupation_position, pf.employer_workplace,
            pf.religion, pf.nationality, pf.citizenship, pf.literacy,
            pf.birth_place_raw, pf.birth_place_sk, pf.home_jurisdiction,
            pf.village_raw, pf.village_sk, pf.house_number, pf.fs_place_standardized,
            pf.godparents_or_witnesses, pf.officiant, pf.observationes_raw,
            pf.observationes_events_json, pf.cause_of_death, pf.person_row_number,
            pf.birth_year_est, pf.census_sheet_id, pf.census_household_id,
            cs.slovakiana_key, cs.census_year, cs.municipality, cs.district, cs.urn_nbn, cs.sheet_number
         FROM personfact pf
         LEFT JOIN censussheet cs ON cs.id = pf.census_sheet_id
         {}
         ORDER BY pf.id ASC",
        where_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map([], |r| {
            Ok(ExportFact {
                id: r.get(0)?,
                record_type: r
                    .get::<_, Option<String>>(1)?
                    .unwrap_or_else(|| "baptism".to_string()),
                entry_number: r.get(2)?,
                event_year: r.get(3)?,
                birth_or_event_date: r.get(4)?,
                baptism_or_burial_date: r.get(5)?,
                given_name_raw: r.get(6)?,
                given_name_sk: r.get(7)?,
                given_name_hu: r.get(8)?,
                surname_raw: r.get(9)?,
                surname_normalized: r.get(10)?,
                sex: r.get(11)?,
                relation_to_head: r.get(12)?,
                marital_status: r.get(13)?,
                father_name: r.get(14)?,
                mother_name: r.get(15)?,
                mother_maiden_surname: r.get(16)?,
                spouse_name: r.get(17)?,
                social_status_or_occupation: r.get(18)?,
                occupation_position: r.get(19)?,
                employer_workplace: r.get(20)?,
                religion: r.get(21)?,
                nationality: r.get(22)?,
                citizenship: r.get(23)?,
                literacy: r.get(24)?,
                birth_place_raw: r.get(25)?,
                birth_place_sk: r.get(26)?,
                home_jurisdiction: r.get(27)?,
                village_raw: r.get(28)?,
                village_sk: r.get(29)?,
                house_number: r.get(30)?,
                fs_place_standardized: r.get(31)?,
                godparents_or_witnesses: r.get(32)?,
                officiant: r.get(33)?,
                observationes_raw: r.get(34)?,
                observationes_events_json: r
                    .get::<_, Option<String>>(35)?
                    .unwrap_or_else(|| "[]".to_string()),
                cause_of_death: r.get(36)?,
                person_row_number: r.get(37)?,
                birth_year_est: r.get(38)?,
                census_sheet_id: r.get(39)?,
                census_household_id: r.get(40)?,
                sheet_slovakiana_key: r.get(41)?,
                sheet_census_year: r.get(42)?,
                sheet_municipality: r.get(43)?,
                sheet_district: r.get(44)?,
                sheet_urn_nbn: r.get(45)?,
                sheet_number: r.get(46)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[pyfunction]
#[pyo3(signature = (db_path, only_new_to_fs=false))]
pub fn export_facts_to_csv_rs(
    py: Python<'_>,
    db_path: String,
    only_new_to_fs: bool,
) -> PyResult<String> {
    py.allow_threads(move || {
        let facts =
            load_export_facts(&db_path, only_new_to_fs).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut out = String::with_capacity(512 + facts.len() * 256);
        let headers = [
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
        ];
        for (i, h) in headers.iter().enumerate() {
            write_csv_cell(&mut out, h, i == 0);
        }
        out.push_str("\r\n");

        for f in &facts {
            let (source_url, source_license) = if f.record_type == "census" {
                let url = match f.sheet_slovakiana_key.as_deref().filter(|s| !s.is_empty()) {
                    Some(k) => format!("https://slovakiana.sk/kulturne-objekty/{}", k),
                    None => "https://slovakiana.sk/scitacie-harky".to_string(),
                };
                let lic = "CC BY-NC-ND 4.0 — Slovenský národný archív (SNA) / Slovakiana (https://creativecommons.org/licenses/by-nc-nd/4.0/)".to_string();
                (url, lic)
            } else {
                (String::new(), String::new())
            };

            let id_str = f.id.to_string();
            let yr_str = f.event_year.map(|y| y.to_string()).unwrap_or_default();
            let sur = f
                .surname_normalized
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| f.surname_raw.as_deref())
                .unwrap_or("");
            let bp = f
                .birth_place_raw
                .as_deref()
                .filter(|s| !s.is_empty())
                .or_else(|| f.birth_place_sk.as_deref())
                .unwrap_or("");

            let cells: [&str; 35] = [
                &id_str,
                &f.record_type,
                f.entry_number.as_deref().unwrap_or(""),
                &yr_str,
                f.birth_or_event_date.as_deref().unwrap_or(""),
                f.baptism_or_burial_date.as_deref().unwrap_or(""),
                f.given_name_raw.as_deref().unwrap_or(""),
                f.given_name_sk.as_deref().unwrap_or(""),
                f.given_name_hu.as_deref().unwrap_or(""),
                sur,
                f.sex.as_deref().unwrap_or(""),
                f.relation_to_head.as_deref().unwrap_or(""),
                f.marital_status.as_deref().unwrap_or(""),
                f.father_name.as_deref().unwrap_or(""),
                f.mother_name.as_deref().unwrap_or(""),
                f.mother_maiden_surname.as_deref().unwrap_or(""),
                f.spouse_name.as_deref().unwrap_or(""),
                f.social_status_or_occupation.as_deref().unwrap_or(""),
                f.occupation_position.as_deref().unwrap_or(""),
                f.employer_workplace.as_deref().unwrap_or(""),
                f.religion.as_deref().unwrap_or(""),
                f.nationality.as_deref().unwrap_or(""),
                f.citizenship.as_deref().unwrap_or(""),
                f.literacy.as_deref().unwrap_or(""),
                bp,
                f.home_jurisdiction.as_deref().unwrap_or(""),
                f.village_raw.as_deref().unwrap_or(""),
                f.village_sk.as_deref().unwrap_or(""),
                f.house_number.as_deref().unwrap_or(""),
                f.fs_place_standardized.as_deref().unwrap_or(""),
                f.godparents_or_witnesses.as_deref().unwrap_or(""),
                f.officiant.as_deref().unwrap_or(""),
                f.observationes_raw.as_deref().unwrap_or(""),
                &source_url,
                &source_license,
            ];
            for (i, c) in cells.iter().enumerate() {
                if i == 0 || (i == 3 && !c.is_empty()) {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(c);
                } else {
                    write_csv_cell(&mut out, c, i == 0);
                }
            }
            out.push_str("\r\n");
        }
        Ok(out)
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, today_gedcom_date, only_new_to_fs=false))]
pub fn export_facts_to_gedcom_rs(
    py: Python<'_>,
    db_path: String,
    today_gedcom_date: String,
    only_new_to_fs: bool,
) -> PyResult<String> {
    py.allow_threads(move || {
        let facts =
            load_export_facts(&db_path, only_new_to_fs).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let has_census = facts.iter().any(|f| f.record_type == "census");

        let mut lines: Vec<String> = Vec::with_capacity(64 + facts.len() * 18);
        lines.push("0 HEAD".to_string());
        lines.push("1 SOUR MATRIKA_DIGITIZER".to_string());
        lines.push("2 NAME Matrika OCR & FamilySearch Digitizer".to_string());
        lines.push("2 VERS 1.0".to_string());
        lines.push(format!("1 DATE {}", today_gedcom_date));
        lines.push("1 GEDC".to_string());
        lines.push("2 VERS 5.5.1".to_string());
        lines.push("2 FORM LINEAGE-LINKED".to_string());
        lines.push("1 CHAR UTF-8".to_string());
        if has_census {
            lines.push(
                "1 COPR Sčítacie hárky (1930/1940): Slovenský národný archív (SNA) / Slovakiana — Licencia Creative Commons CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)".to_string(),
            );
        }

        let mut emitted_families: HashMap<String, (String, Option<String>, Option<String>, Vec<String>)> =
            HashMap::new();
        let mut emitted_family_order: Vec<String> = Vec::new();
        let mut referenced_sources: HashMap<String, Option<(String, i64, String, String, String)>> =
            HashMap::new();
        let mut referenced_source_order: Vec<String> = Vec::new();

        for (idx, f) in facts.iter().enumerate() {
            let fid = if f.id > 0 { f.id } else { (idx + 1) as i64 };
            let indi_id = format!("@I{}@", fid);
            let given = {
                let g = clean_gedcom_val(
                    f.given_name_raw
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .or_else(|| f.given_name_sk.as_deref()),
                );
                if g.is_empty() {
                    "Neznáme".to_string()
                } else {
                    g
                }
            };
            let surname = {
                let s = clean_gedcom_val(
                    f.surname_raw
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .or_else(|| f.surname_normalized.as_deref()),
                );
                if s.is_empty() {
                    "Neznáme".to_string()
                } else {
                    s
                }
            };
            let mut place = {
                let p = clean_gedcom_val(
                    f.fs_place_standardized
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .or_else(|| f.village_sk.as_deref().filter(|s| !s.is_empty()))
                        .or_else(|| f.village_raw.as_deref()),
                );
                if p.is_empty() {
                    "Slovakia".to_string()
                } else {
                    p
                }
            };
            if let Some(hn) = f.house_number.as_deref().filter(|s| !s.is_empty()) {
                place = format!("{} (dom č. {})", place, clean_gedcom_val(Some(hn)));
            }

            let sour_id: Option<String> = if f.record_type == "census" {
                if let Some(sk) = f.sheet_slovakiana_key.as_deref().filter(|s| !s.is_empty()) {
                    let clean_key: String = sk
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
                        .collect();
                    let sid = format!("@S_{}@", clean_key);
                    if !referenced_sources.contains_key(&sid) {
                        referenced_source_order.push(sid.clone());
                    }
                    referenced_sources.insert(
                        sid.clone(),
                        Some((
                            clean_gedcom_val(Some(sk)),
                            f.sheet_census_year.unwrap_or(1930),
                            clean_gedcom_val(f.sheet_municipality.as_deref()),
                            clean_gedcom_val(f.sheet_district.as_deref()),
                            clean_gedcom_val(f.sheet_urn_nbn.as_deref()),
                        )),
                    );
                    Some(sid)
                } else {
                    let sid = "@S_SLOVAKIANA_SNA@".to_string();
                    if !referenced_sources.contains_key(&sid) {
                        referenced_source_order.push(sid.clone());
                        referenced_sources.insert(sid.clone(), None);
                    }
                    Some(sid)
                }
            } else {
                None
            };

            lines.push(format!("0 {} INDI", indi_id));
            lines.push(format!("1 NAME {} /{}/", given, surname));
            lines.push(format!("2 GIVN {}", given));
            lines.push(format!("2 SURN {}", surname));
            if let Some(sx) = f.sex.as_deref() {
                if sx == "M" || sx == "F" {
                    lines.push(format!("1 SEX {}", sx));
                }
            }

            match f.record_type.as_str() {
                "baptism" => {
                    lines.push("1 BIRT".to_string());
                    let b_date =
                        format_gedcom_date_internal(f.birth_or_event_date.as_deref(), f.event_year);
                    if !b_date.is_empty() {
                        lines.push(format!("2 DATE {}", b_date));
                    }
                    lines.push(format!("2 PLAC {}", place));
                    lines.push("1 BAPM".to_string());
                    let bp_date = format_gedcom_date_internal(
                        f.baptism_or_burial_date.as_deref(),
                        f.event_year,
                    );
                    if !bp_date.is_empty() {
                        lines.push(format!("2 DATE {}", bp_date));
                    }
                    lines.push(format!("2 PLAC {}", place));
                }
                "death" => {
                    lines.push("1 DEAT".to_string());
                    let d_date =
                        format_gedcom_date_internal(f.birth_or_event_date.as_deref(), f.event_year);
                    if !d_date.is_empty() {
                        lines.push(format!("2 DATE {}", d_date));
                    }
                    lines.push(format!("2 PLAC {}", place));
                    if let Some(cod) = f.cause_of_death.as_deref().filter(|s| !s.is_empty()) {
                        lines.push(format!("2 CAUS {}", clean_gedcom_val(Some(cod))));
                    }
                }
                "marriage" => {
                    lines.push("1 EVEN".to_string());
                    lines.push("2 TYPE Marriage".to_string());
                    let m_date =
                        format_gedcom_date_internal(f.birth_or_event_date.as_deref(), f.event_year);
                    if !m_date.is_empty() {
                        lines.push(format!("2 DATE {}", m_date));
                    }
                    lines.push(format!("2 PLAC {}", place));
                }
                "census" => {
                    if f.birth_or_event_date.as_deref().filter(|s| !s.is_empty()).is_some()
                        || f.birth_year_est.is_some()
                    {
                        lines.push("1 BIRT".to_string());
                        let cb_date = format_gedcom_date_internal(
                            f.birth_or_event_date.as_deref(),
                            f.birth_year_est,
                        );
                        if !cb_date.is_empty() {
                            lines.push(format!("2 DATE {}", cb_date));
                        }
                        let bp_plac = clean_gedcom_val(
                            f.birth_place_sk
                                .as_deref()
                                .filter(|s| !s.is_empty())
                                .or_else(|| f.birth_place_raw.as_deref().filter(|s| !s.is_empty())),
                        );
                        lines.push(format!(
                            "2 PLAC {}",
                            if bp_plac.is_empty() { &place } else { &bp_plac }
                        ));
                    }
                    lines.push("1 CENS".to_string());
                    if let Some(ey) = f.event_year {
                        lines.push(format!("2 DATE {}", ey));
                    }
                    lines.push(format!("2 PLAC {}", place));
                    if let Some(ref sid) = sour_id {
                        lines.push(format!("2 SOUR {}", sid));
                        let mut page_parts = Vec::new();
                        if let Some(sn) = f.sheet_number.as_deref().filter(|s| !s.is_empty()) {
                            page_parts.push(format!("hárok č. {}", clean_gedcom_val(Some(sn))));
                        }
                        if let Some(hn) = f.house_number.as_deref().filter(|s| !s.is_empty()) {
                            page_parts.push(format!("dom č. {}", clean_gedcom_val(Some(hn))));
                        }
                        if let Some(rn) = f
                            .person_row_number
                            .as_deref()
                            .filter(|s| !s.is_empty())
                            .or_else(|| f.entry_number.as_deref().filter(|s| !s.is_empty()))
                        {
                            page_parts.push(format!("riadok č. {}", clean_gedcom_val(Some(rn))));
                        }
                        if !page_parts.is_empty() {
                            lines.push(format!("3 PAGE {}", page_parts.join(", ")));
                        }
                    }
                    lines.push("1 RESI".to_string());
                    if let Some(ey) = f.event_year {
                        lines.push(format!("2 DATE {}", ey));
                    }
                    lines.push(format!("2 PLAC {}", place));
                    if let Some(occ) = f
                        .social_status_or_occupation
                        .as_deref()
                        .filter(|s| !s.is_empty())
                    {
                        lines.push(format!("1 OCCU {}", clean_gedcom_val(Some(occ))));
                    }
                    if let Some(ref sid) = sour_id {
                        lines.push(format!("1 SOUR {}", sid));
                    }
                }
                _ => {}
            }

            if let Ok(Value::Array(obs_events)) =
                serde_json::from_str::<Value>(&f.observationes_events_json)
            {
                for ev in obs_events {
                    let ev_type = ev
                        .get("event_type")
                        .or_else(|| ev.get("type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let dt_val = ev
                        .get("year")
                        .and_then(|v| {
                            v.as_i64()
                                .map(|n| n.to_string())
                                .or_else(|| v.as_str().map(|s| s.to_string()))
                        })
                        .or_else(|| ev.get("date").and_then(|v| v.as_str()).map(|s| s.to_string()));
                    let note_val = ev.get("raw_note").and_then(|v| v.as_str());
                    if ev_type == "death" && f.record_type != "death" {
                        lines.push("1 DEAT".to_string());
                        if let Some(ref d) = dt_val {
                            let cd = clean_gedcom_val(Some(d));
                            if !cd.is_empty() {
                                lines.push(format!("2 DATE {}", cd));
                            }
                        }
                        if let Some(n) = note_val {
                            let cn = clean_gedcom_val(Some(n));
                            if !cn.is_empty() {
                                lines.push(format!("2 NOTE {}", cn));
                            }
                        }
                    } else if ev_type == "marriage" {
                        lines.push("1 EVEN".to_string());
                        lines.push("2 TYPE Marriage Note".to_string());
                        if let Some(ref d) = dt_val {
                            let cd = clean_gedcom_val(Some(d));
                            if !cd.is_empty() {
                                lines.push(format!("2 DATE {}", cd));
                            }
                        }
                        if let Some(n) = note_val {
                            let cn = clean_gedcom_val(Some(n));
                            if !cn.is_empty() {
                                lines.push(format!("2 NOTE {}", cn));
                            }
                        }
                    }
                }
            }

            if let Some(obs) = f.observationes_raw.as_deref().filter(|s| !s.is_empty()) {
                lines.push(format!(
                    "1 NOTE Poznámka z matriky: {}",
                    clean_gedcom_val(Some(obs))
                ));
            }
            if let Some(gp) = f
                .godparents_or_witnesses
                .as_deref()
                .filter(|s| !s.is_empty())
            {
                lines.push(format!(
                    "1 NOTE Krstní rodičia / svedkovia: {}",
                    clean_gedcom_val(Some(gp))
                ));
            }

            let has_father = f.father_name.as_deref().filter(|s| !s.is_empty()).is_some();
            let has_mother = f.mother_name.as_deref().filter(|s| !s.is_empty()).is_some();
            if has_father || has_mother {
                let fam_key = if f.record_type == "census" && f.census_sheet_id.is_some() {
                    format!(
                        "census|{}|{}|{}|{}|{}",
                        f.census_sheet_id.unwrap_or(0),
                        f.census_household_id.unwrap_or(0),
                        clean_gedcom_val(f.father_name.as_deref()).to_lowercase(),
                        clean_gedcom_val(f.mother_name.as_deref()).to_lowercase(),
                        clean_gedcom_val(f.mother_maiden_surname.as_deref()).to_lowercase(),
                    )
                } else {
                    format!("row|{}", fid)
                };

                if let Some((fam_id, _, _, children)) = emitted_families.get_mut(&fam_key) {
                    children.push(indi_id.clone());
                    lines.push(format!("1 FAMC {}", fam_id));
                } else {
                    let fam_id = format!("@F{}@", fid);
                    let fath_id = if has_father {
                        Some(format!("@IF{}@", fid))
                    } else {
                        None
                    };
                    let moth_id = if has_mother {
                        Some(format!("@IM{}@", fid))
                    } else {
                        None
                    };
                    lines.push(format!("1 FAMC {}", fam_id));
                    if let (Some(ref f_id), Some(f_name)) = (&fath_id, f.father_name.as_deref()) {
                        lines.push(format!("0 {} INDI", f_id));
                        lines.push(format!(
                            "1 NAME {}",
                            format_gedcom_parent_name(f_name, &surname)
                        ));
                        lines.push("1 SEX M".to_string());
                        if f.record_type != "census" {
                            if let Some(occ) = f
                                .social_status_or_occupation
                                .as_deref()
                                .filter(|s| !s.is_empty())
                            {
                                lines.push(format!("1 OCCU {}", clean_gedcom_val(Some(occ))));
                            }
                        }
                    }
                    if let (Some(ref m_id), Some(m_name)) = (&moth_id, f.mother_name.as_deref()) {
                        let m_surn = f.mother_maiden_surname.as_deref().unwrap_or("");
                        lines.push(format!("0 {} INDI", m_id));
                        lines.push(format!(
                            "1 NAME {}",
                            format_gedcom_parent_name(m_name, m_surn)
                        ));
                        lines.push("1 SEX F".to_string());
                    }
                    emitted_family_order.push(fam_key.clone());
                    emitted_families.insert(fam_key, (fam_id, fath_id, moth_id, vec![indi_id]));
                }
            }
        }

        for fk in &emitted_family_order {
            if let Some((fam_id, fath_id, moth_id, child_ids)) = emitted_families.get(fk) {
                lines.push(format!("0 {} FAM", fam_id));
                if let Some(ref fi) = fath_id {
                    lines.push(format!("1 HUSB {}", fi));
                }
                if let Some(ref mi) = moth_id {
                    lines.push(format!("1 WIFE {}", mi));
                }
                for cid in child_ids {
                    lines.push(format!("1 CHIL {}", cid));
                }
            }
        }

        for sid in &referenced_source_order {
            lines.push(format!("0 {} SOUR", sid));
            if let Some(Some((s_key, c_year, muni, dist, urn))) = referenced_sources.get(sid) {
                lines.push(format!(
                    "1 TITL Sčítanie ľudu {} – {} (okres {}), Slovakiana ({})",
                    c_year, muni, dist, s_key
                ));
                lines.push("1 AUTH Slovenský národný archív (SNA) / Slovakiana".to_string());
                let mut pub_parts = vec![format!("https://slovakiana.sk/kulturne-objekty/{}", s_key)];
                if !urn.is_empty() {
                    pub_parts.push(format!("URN:NBN {}", urn));
                }
                pub_parts.push(
                    "Licencia: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)"
                        .to_string(),
                );
                lines.push(format!("1 PUBL {}", pub_parts.join(" | ")));
                lines.push(format!(
                    "1 NOTE Zdroj: Slovenský národný archív (SNA) / Slovakiana ({}). Použité pod licenciou Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).",
                    s_key
                ));
            } else {
                lines.push("1 TITL Sčítanie ľudu v Krajine Slovenskej (1930/1940) – Slovakiana".to_string());
                lines.push("1 AUTH Slovenský národný archív (SNA) / Slovakiana".to_string());
                lines.push(
                    "1 PUBL https://slovakiana.sk/scitacie-harky | Licencia: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/)".to_string(),
                );
            }
        }

        lines.push("0 TRLR".to_string());
        lines.push(String::new());
        Ok(lines.join("\n"))
    })
}
