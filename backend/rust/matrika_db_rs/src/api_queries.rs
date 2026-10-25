use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::params;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

use crate::open_sqlite;

fn resolve_scan_media_url_internal(
    original_path: Option<&str>,
    filename: Option<&str>,
    census_root: Option<&str>,
    matrika_root: Option<&str>,
    upload_root: Option<&str>,
) -> Option<String> {
    let raw = original_path.unwrap_or("").trim();
    if raw.is_empty() {
        return filename
            .filter(|f| !f.is_empty())
            .map(|f| format!("/media/uploads/{}", f));
    }
    if raw.starts_with("/media/") || raw.starts_with("http://") || raw.starts_with("https://") {
        return Some(raw.to_string());
    }
    let p = Path::new(raw);
    let p_canon = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());

    let try_rel = |root_opt: Option<&str>, prefix: &str| -> Option<String> {
        let root_str = root_opt.filter(|s| !s.is_empty())?;
        let r_path = Path::new(root_str);
        let r_canon = r_path.canonicalize().unwrap_or_else(|_| r_path.to_path_buf());
        if let Ok(rel) = p_canon.strip_prefix(&r_canon).or_else(|_| p.strip_prefix(r_path)) {
            let rel_posix = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            if !rel_posix.is_empty() {
                return Some(format!("{}/{}", prefix, rel_posix));
            }
        }
        None
    };

    if let Some(u) = try_rel(census_root, "/media/uploads/slovakiana") {
        return Some(u);
    }
    if let Some(u) = try_rel(matrika_root, "/media/uploads/matrika") {
        return Some(u);
    }
    if let Some(u) = try_rel(upload_root, "/media/uploads") {
        return Some(u);
    }
    if let Some(idx) = raw.find("slovakiana/") {
        return Some(format!("/media/uploads/{}", &raw[idx..]));
    }
    if let Some(idx) = raw.find("matrika/") {
        return Some(format!("/media/uploads/{}", &raw[idx..]));
    }
    let fname = filename
        .filter(|s| !s.is_empty())
        .or_else(|| p.file_name().and_then(|n| n.to_str()))
        .unwrap_or("");
    Some(format!("/media/uploads/{}", fname))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (db_path, scan_type=None, is_census=None, census_root=None, matrika_root=None, upload_root=None))]
pub fn list_scans_rs(
    py: Python<'_>,
    db_path: String,
    scan_type: Option<String>,
    is_census: Option<bool>,
    census_root: Option<String>,
    matrika_root: Option<String>,
    upload_root: Option<String>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut row_counts: HashMap<i64, i64> = HashMap::new();
        {
            let mut st = conn
                .prepare("SELECT scan_id, COUNT(id) FROM recordrow GROUP BY scan_id")
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = st
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (sid, cnt) in rows.flatten() {
                row_counts.insert(sid, cnt);
            }
        }

        let mut open_q_counts: HashMap<i64, i64> = HashMap::new();
        {
            let mut st = conn
                .prepare(
                    "SELECT scan_id, COUNT(id) FROM questionitem WHERE status = 'open' GROUP BY scan_id",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = st
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for (sid, cnt) in rows.flatten() {
                open_q_counts.insert(sid, cnt);
            }
        }

        let mut batches_out: Vec<Value> = Vec::new();
        {
            let mut st = conn
                .prepare(
                    "SELECT id, name, parish_or_book, record_type_hint, total_scans,
                            processed_scans, status, watchlist_filter, created_at
                     FROM batch ORDER BY id DESC",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = st
                .query_map([], |r| {
                    Ok(json!({
                        "id": r.get::<_, i64>(0)?,
                        "name": r.get::<_, Option<String>>(1)?,
                        "parish_or_book": r.get::<_, Option<String>>(2)?,
                        "record_type_hint": r.get::<_, Option<String>>(3)?,
                        "total_scans": r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        "processed_scans": r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        "status": r.get::<_, Option<String>>(6)?,
                        "watchlist_filter": r.get::<_, Option<String>>(7)?,
                        "created_at": r.get::<_, Option<String>>(8)?,
                    }))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for b in rows.flatten() {
                batches_out.push(b);
            }
        }

        let where_sql = if scan_type.as_deref() == Some("matrika") || is_census == Some(false) {
            "WHERE census_sheet_id IS NULL AND detected_type != 'census'"
        } else if scan_type.as_deref() == Some("census") || is_census == Some(true) {
            "WHERE census_sheet_id IS NOT NULL OR detected_type = 'census'"
        } else {
            ""
        };

        let sql = format!(
            "SELECT id, batch_id, census_sheet_id, page_side, filename, original_path,
                    overlay_path, header_crop_path, width, height, spine_x,
                    detected_year, detected_type, header_raw_ocr, status, error_message, created_at
             FROM scanimage {} ORDER BY id DESC",
            where_sql
        );
        let mut st = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let scans_rows = st
            .query_map([], |r| {
                let sid: i64 = r.get(0)?;
                let fname: Option<String> = r.get(4)?;
                let orig_path: Option<String> = r.get(5)?;
                let orig_url = resolve_scan_media_url_internal(
                    orig_path.as_deref(),
                    fname.as_deref(),
                    census_root.as_deref(),
                    matrika_root.as_deref(),
                    upload_root.as_deref(),
                );
                Ok(json!({
                    "id": sid,
                    "batch_id": r.get::<_, Option<i64>>(1)?,
                    "census_sheet_id": r.get::<_, Option<i64>>(2)?,
                    "page_side": r.get::<_, Option<String>>(3)?,
                    "filename": fname,
                    "original_url": orig_url,
                    "overlay_url": r.get::<_, Option<String>>(6)?,
                    "header_crop_url": r.get::<_, Option<String>>(7)?,
                    "width": r.get::<_, Option<i64>>(8)?.unwrap_or(0),
                    "height": r.get::<_, Option<i64>>(9)?.unwrap_or(0),
                    "spine_x": r.get::<_, Option<i64>>(10)?.unwrap_or(0),
                    "detected_year": r.get::<_, Option<String>>(11)?,
                    "detected_type": r.get::<_, Option<String>>(12)?,
                    "header_raw_ocr": r.get::<_, Option<String>>(13)?,
                    "status": r.get::<_, Option<String>>(14)?,
                    "error_message": r.get::<_, Option<String>>(15)?,
                    "rows_count": row_counts.get(&sid).copied().unwrap_or(0),
                    "open_questions": open_q_counts.get(&sid).copied().unwrap_or(0),
                    "created_at": r.get::<_, Option<String>>(16)?,
                }))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let out = json!({
            "batches": batches_out,
            "scans": scans_rows,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, scan_id=None, row_id=None, stage=None, limit=100))]
pub fn list_workflow_traces_rs(
    py: Python<'_>,
    db_path: String,
    scan_id: Option<i64>,
    row_id: Option<i64>,
    stage: Option<String>,
    limit: i64,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut clauses: Vec<String> = Vec::new();
        if let Some(sid) = scan_id {
            clauses.push(format!("scan_id = {}", sid));
        }
        if let Some(rid) = row_id {
            clauses.push(format!("row_id = {}", rid));
        }
        if let Some(ref stg) = stage {
            let s = stg.trim();
            if !s.is_empty() {
                match s {
                    "ocr" => clauses.push(
                        "stage IN ('ocr', 'ocr_header', 'ocr_left', 'ocr_right', 'census_overview_ocr')"
                            .to_string(),
                    ),
                    "extract" => clauses.push(
                        "stage IN ('extract', 'census_extract', 'census_overview_extract')"
                            .to_string(),
                    ),
                    "layout" => clauses.push("stage IN ('layout', 'census_layout')".to_string()),
                    "mcp_search" | "mcp_record" => {
                        clauses.push("stage IN ('mcp_record', 'mcp_search')".to_string())
                    }
                    _ => {
                        let safe = s.replace('\'', "''");
                        clauses.push(format!("stage = '{}'", safe));
                    }
                }
            }
        }

        let where_sql = if clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", clauses.join(" AND "))
        };
        let eff_limit = limit.clamp(1, 2000);
        let sql = format!(
            "SELECT id, scan_id, row_id, stage, model_or_tool, prompt_text, image_path,
                    reasoning_content, response_text, parsed_json, status, duration_ms,
                    prompt_tokens, completion_tokens, reasoning_tokens, created_at
             FROM workflowtrace
             {}
             ORDER BY id DESC
             LIMIT {}",
            where_sql, eff_limit
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let traces: Vec<Value> = stmt
            .query_map([], |r| {
                Ok(json!({
                    "id": r.get::<_, i64>(0)?,
                    "scan_id": r.get::<_, Option<i64>>(1)?,
                    "row_id": r.get::<_, Option<i64>>(2)?,
                    "stage": r.get::<_, Option<String>>(3)?,
                    "model_or_tool": r.get::<_, Option<String>>(4)?,
                    "prompt_text": r.get::<_, Option<String>>(5)?,
                    "image_path": r.get::<_, Option<String>>(6)?,
                    "reasoning_content": r.get::<_, Option<String>>(7)?,
                    "response_text": r.get::<_, Option<String>>(8)?,
                    "parsed_json": r.get::<_, Option<String>>(9)?,
                    "status": r.get::<_, Option<String>>(10)?,
                    "duration_ms": r.get::<_, Option<i64>>(11)?.unwrap_or(0),
                    "prompt_tokens": r.get::<_, Option<i64>>(12)?.unwrap_or(0),
                    "completion_tokens": r.get::<_, Option<i64>>(13)?.unwrap_or(0),
                    "reasoning_tokens": r.get::<_, Option<i64>>(14)?.unwrap_or(0),
                    "created_at": r.get::<_, Option<String>>(15)?,
                }))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let out = json!({ "traces": traces });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
pub fn get_fs_progress_counts_rs(
    py: Python<'_>,
    db_path: String,
) -> PyResult<(i64, i64, i64)> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let total_facts: i64 = conn
            .query_row("SELECT COUNT(*) FROM personfact", [], |r| r.get(0))
            .unwrap_or(0);
        let (total_matched, reviewed): (i64, i64) = conn
            .query_row(
                "SELECT COUNT(id), COALESCE(SUM(CASE WHEN reviewed_by_user != 0 THEN 1 ELSE 0 END), 0)
                 FROM familysearchmatch
                 WHERE fs_person_id IS NOT NULL AND trim(fs_person_id) != '' AND confidence_score > 0",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap_or((0, 0));
        Ok((total_facts, total_matched, reviewed))
    })
}

#[pyfunction]
#[pyo3(signature = (db_path, trace_limit=80))]
pub fn get_debug_overview_stats_and_traces_rs(
    py: Python<'_>,
    db_path: String,
    trace_limit: i64,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let total_scans: i64 = conn
            .query_row("SELECT COUNT(*) FROM scanimage", [], |r| r.get(0))
            .unwrap_or(0);
        let total_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM recordrow", [], |r| r.get(0))
            .unwrap_or(0);
        let total_facts: i64 = conn
            .query_row("SELECT COUNT(*) FROM personfact", [], |r| r.get(0))
            .unwrap_or(0);
        let open_questions: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM questionitem WHERE status = 'open'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let total_matches: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM familysearchmatch
                 WHERE fs_person_id IS NOT NULL AND trim(fs_person_id) != '' AND confidence_score > 0",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let eff_limit = trace_limit.clamp(1, 500);
        let mut stmt = conn
            .prepare(
                "SELECT id, scan_id, row_id, stage, model_or_tool, prompt_text, image_path,
                        reasoning_content, response_text, parsed_json, status, duration_ms,
                        prompt_tokens, completion_tokens, reasoning_tokens, created_at
                 FROM workflowtrace
                 ORDER BY id DESC
                 LIMIT ?1",
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let recent_traces: Vec<Value> = stmt
            .query_map(params![eff_limit], |r| {
                Ok(json!({
                    "id": r.get::<_, i64>(0)?,
                    "scan_id": r.get::<_, Option<i64>>(1)?,
                    "row_id": r.get::<_, Option<i64>>(2)?,
                    "stage": r.get::<_, Option<String>>(3)?,
                    "model_or_tool": r.get::<_, Option<String>>(4)?,
                    "prompt_text": r.get::<_, Option<String>>(5)?,
                    "image_path": r.get::<_, Option<String>>(6)?,
                    "reasoning_content": r.get::<_, Option<String>>(7)?,
                    "response_text": r.get::<_, Option<String>>(8)?,
                    "parsed_json": r.get::<_, Option<String>>(9)?,
                    "status": r.get::<_, Option<String>>(10)?,
                    "duration_ms": r.get::<_, Option<i64>>(11)?.unwrap_or(0),
                    "prompt_tokens": r.get::<_, Option<i64>>(12)?.unwrap_or(0),
                    "completion_tokens": r.get::<_, Option<i64>>(13)?.unwrap_or(0),
                    "reasoning_tokens": r.get::<_, Option<i64>>(14)?.unwrap_or(0),
                    "created_at": r.get::<_, Option<String>>(15)?,
                }))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let out = json!({
            "stats": {
                "total_scans": total_scans,
                "total_rows": total_rows,
                "total_facts": total_facts,
                "open_questions": open_questions,
                "total_matches": total_matches,
                "traces_logged": recent_traces.len(),
            },
            "recent_traces": recent_traces,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}
