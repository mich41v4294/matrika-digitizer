use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::{
    canonical_stage_group_internal, compute_feedback_statistics_from_conn,
    format_model_combo_internal, open_sqlite, ModelDefaults,
};

#[derive(Clone)]
struct FullTrace {
    id: i64,
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
    prompt_text: Option<String>,
    reasoning_content: Option<String>,
    response_text: Option<String>,
    parsed_json: String,
    created_at: Option<String>,
}

fn serialize_trace_for_snapshot(t: &FullTrace) -> Value {
    let parsed_obj: Value = serde_json::from_str(&t.parsed_json)
        .unwrap_or_else(|_| Value::String(t.parsed_json.clone()));
    json!({
        "id": t.id,
        "scan_id": t.scan_id,
        "row_id": t.row_id,
        "stage": t.stage,
        "canonical_stage": canonical_stage_group_internal(&t.stage),
        "model_or_tool": t.model_or_tool,
        "status": t.status,
        "duration_ms": t.duration_ms,
        "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "reasoning_tokens": t.reasoning_tokens,
        "image_path": t.image_path,
        "prompt_text": t.prompt_text,
        "reasoning_content": t.reasoning_content,
        "response_text": t.response_text,
        "parsed_json": parsed_obj,
        "created_at": t.created_at,
    })
}

fn query_traces_where(
    conn: &Connection,
    where_sql: &str,
    param_val: i64,
) -> Result<Vec<FullTrace>, rusqlite::Error> {
    let sql = format!(
        "SELECT id, scan_id, row_id, stage, model_or_tool, status, duration_ms,
                prompt_tokens, completion_tokens, reasoning_tokens, image_path,
                prompt_text, reasoning_content, response_text, parsed_json, created_at
         FROM workflowtrace
         WHERE {} = ?1
         ORDER BY id ASC",
        where_sql
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params![param_val], |r| {
            Ok(FullTrace {
                id: r.get(0)?,
                scan_id: r.get(1)?,
                row_id: r.get(2)?,
                stage: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                model_or_tool: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                status: r.get::<_, Option<String>>(5)?.unwrap_or_else(|| "ok".to_string()),
                duration_ms: r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                prompt_tokens: r.get::<_, Option<i64>>(7)?.unwrap_or(0),
                completion_tokens: r.get::<_, Option<i64>>(8)?.unwrap_or(0),
                reasoning_tokens: r.get::<_, Option<i64>>(9)?.unwrap_or(0),
                image_path: r.get(10)?,
                prompt_text: r.get(11)?,
                reasoning_content: r.get(12)?,
                response_text: r.get(13)?,
                parsed_json: r
                    .get::<_, Option<String>>(14)?
                    .unwrap_or_else(|| "{}".to_string()),
                created_at: r.get(15)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

#[derive(Clone)]
struct FeedbackRow {
    id: i64,
    target_type: String,
    row_id: Option<i64>,
    scan_id: Option<i64>,
    census_sheet_id: Option<i64>,
    person_fact_id: Option<i64>,
    overview_house_id: Option<i64>,
    municipal_district_id: Option<i64>,
    verdict: String,
    comment: Option<String>,
    expected_correction: Option<String>,
    model_combo: String,
    layout_model: Option<String>,
    ocr_model: Option<String>,
    decision_model: Option<String>,
    extract_model: Option<String>,
    mcp_model: Option<String>,
    total_duration_ms: i64,
    layout_duration_ms: i64,
    ocr_duration_ms: i64,
    decision_duration_ms: i64,
    extract_duration_ms: i64,
    mcp_duration_ms: i64,
    raw_ocr_snapshot: Option<String>,
    extracted_snapshot_json: String,
    stage_breakdown_json: String,
    traces_snapshot_json: String,
    crop_url: Option<String>,
    created_at: Option<String>,
    updated_at: Option<String>,
}

fn read_feedback_row(r: &rusqlite::Row<'_>) -> Result<FeedbackRow, rusqlite::Error> {
    Ok(FeedbackRow {
        id: r.get(0)?,
        target_type: r.get::<_, Option<String>>(1)?.unwrap_or_else(|| "row".to_string()),
        row_id: r.get(2)?,
        scan_id: r.get(3)?,
        census_sheet_id: r.get(4)?,
        person_fact_id: r.get(5)?,
        overview_house_id: r.get(6)?,
        municipal_district_id: r.get(7)?,
        verdict: r
            .get::<_, Option<String>>(8)?
            .unwrap_or_else(|| "correct".to_string()),
        comment: r.get(9)?,
        expected_correction: r.get(10)?,
        model_combo: r.get::<_, Option<String>>(11)?.unwrap_or_default(),
        layout_model: r.get(12)?,
        ocr_model: r.get(13)?,
        decision_model: r.get(14)?,
        extract_model: r.get(15)?,
        mcp_model: r.get(16)?,
        total_duration_ms: r.get::<_, Option<i64>>(17)?.unwrap_or(0),
        layout_duration_ms: r.get::<_, Option<i64>>(18)?.unwrap_or(0),
        ocr_duration_ms: r.get::<_, Option<i64>>(19)?.unwrap_or(0),
        decision_duration_ms: r.get::<_, Option<i64>>(20)?.unwrap_or(0),
        extract_duration_ms: r.get::<_, Option<i64>>(21)?.unwrap_or(0),
        mcp_duration_ms: r.get::<_, Option<i64>>(22)?.unwrap_or(0),
        raw_ocr_snapshot: r.get(23)?,
        extracted_snapshot_json: r
            .get::<_, Option<String>>(24)?
            .unwrap_or_else(|| "{}".to_string()),
        stage_breakdown_json: r
            .get::<_, Option<String>>(25)?
            .unwrap_or_else(|| "{}".to_string()),
        traces_snapshot_json: r
            .get::<_, Option<String>>(26)?
            .unwrap_or_else(|| "[]".to_string()),
        crop_url: r.get(27)?,
        created_at: r.get(28)?,
        updated_at: r.get(29)?,
    })
}

const FEEDBACK_SELECT_COLS: &str = "id, target_type, row_id, scan_id, census_sheet_id, person_fact_id, overview_house_id, municipal_district_id, verdict, comment, expected_correction, model_combo, layout_model, ocr_model, decision_model, extract_model, mcp_model, total_duration_ms, layout_duration_ms, ocr_duration_ms, decision_duration_ms, extract_duration_ms, mcp_duration_ms, raw_ocr_snapshot, extracted_snapshot_json, stage_breakdown_json, traces_snapshot_json, crop_url, created_at, updated_at";

fn serialize_feedback_row(fb: &FeedbackRow, include_traces: bool) -> Value {
    let extracted_obj: Value =
        serde_json::from_str(&fb.extracted_snapshot_json).unwrap_or_else(|_| json!({}));
    let stage_breakdown: Value =
        serde_json::from_str(&fb.stage_breakdown_json).unwrap_or_else(|_| json!({}));
    let traces_list: Value = if include_traces {
        serde_json::from_str(&fb.traces_snapshot_json).unwrap_or_else(|_| json!([]))
    } else {
        json!([])
    };

    let mut label_parts: Vec<String> = Vec::new();
    let cs_info = extracted_obj.get("census_sheet").unwrap_or(&Value::Null);
    let sc_info = extracted_obj.get("scan").unwrap_or(&Value::Null);
    let row_info = extracted_obj.get("row").unwrap_or(&Value::Null);
    let pf_info = extracted_obj.get("person_fact").unwrap_or(&Value::Null);
    let oh_info = extracted_obj.get("overview_house").unwrap_or(&Value::Null);
    let md_info = extracted_obj
        .get("municipal_district")
        .unwrap_or(&Value::Null);

    if let Some(sk) = cs_info
        .get("slovakiana_key")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        label_parts.push(sk.to_string());
    } else if let Some(fn_s) = sc_info
        .get("filename")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        label_parts.push(fn_s.to_string());
    }

    if pf_info.is_object() {
        let g = pf_info
            .get("given_name_raw")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let s = pf_info
            .get("surname_raw")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let p_name = format!("{} {}", g, s).trim().to_string();
        let r_num = pf_info
            .get("person_row_number")
            .filter(|v| !v.is_null() && v.as_str() != Some(""))
            .or_else(|| {
                pf_info
                    .get("entry_number")
                    .filter(|v| !v.is_null() && v.as_str() != Some(""))
            })
            .or_else(|| row_info.get("row_index").filter(|v| !v.is_null()));
        if let Some(rn) = r_num {
            let rn_str = match rn {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                _ => String::new(),
            };
            if !rn_str.is_empty() {
                label_parts.push(format!("Row #{}", rn_str));
            }
        }
        if !p_name.is_empty() {
            label_parts.push(p_name);
        }
    } else if oh_info.is_object() {
        let hn = oh_info
            .get("house_number")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("—");
        let own = oh_info
            .get("owner_name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("—");
        label_parts.push(format!("Dom č. {} ({})", hn, own));
    } else if md_info.is_object() {
        let obv = md_info
            .get("obvod_number")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("Spolu");
        let comm = md_info
            .get("commissioner_name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("—");
        label_parts.push(format!("Obvod {} ({})", obv, comm));
    } else if let Some(ri) = row_info.get("row_index").filter(|v| !v.is_null()) {
        let ri_str = match ri {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            _ => String::new(),
        };
        if !ri_str.is_empty() {
            label_parts.push(format!("Row #{}", ri_str));
        }
    }

    let display_label = if !label_parts.is_empty() {
        label_parts.join(" · ")
    } else {
        format!("Row #{}", fb.row_id.unwrap_or(fb.id))
    };

    let norm_ts = |ts: &Option<String>| -> Option<String> {
        ts.as_ref().map(|s| {
            if s.contains(' ') && !s.contains('T') {
                s.replacen(' ', "T", 1)
            } else {
                s.clone()
            }
        })
    };

    let mut out = json!({
        "id": fb.id,
        "target_type": fb.target_type,
        "row_id": fb.row_id,
        "scan_id": fb.scan_id,
        "census_sheet_id": fb.census_sheet_id,
        "person_fact_id": fb.person_fact_id,
        "overview_house_id": fb.overview_house_id,
        "municipal_district_id": fb.municipal_district_id,
        "verdict": fb.verdict,
        "comment": fb.comment,
        "expected_correction": fb.expected_correction,
        "display_label": display_label,
        "model_combo": fb.model_combo,
        "layout_model": fb.layout_model,
        "ocr_model": fb.ocr_model,
        "decision_model": fb.decision_model,
        "extract_model": fb.extract_model,
        "mcp_model": fb.mcp_model,
        "total_duration_ms": fb.total_duration_ms,
        "layout_duration_ms": fb.layout_duration_ms,
        "ocr_duration_ms": fb.ocr_duration_ms,
        "decision_duration_ms": fb.decision_duration_ms,
        "extract_duration_ms": fb.extract_duration_ms,
        "mcp_duration_ms": fb.mcp_duration_ms,
        "raw_ocr_snapshot": fb.raw_ocr_snapshot,
        "extracted_snapshot": extracted_obj,
        "stage_breakdown": stage_breakdown,
        "crop_url": fb.crop_url,
        "created_at": norm_ts(&fb.created_at),
        "updated_at": norm_ts(&fb.updated_at),
    });
    if include_traces {
        if let Some(obj) = out.as_object_mut() {
            obj.insert("traces_snapshot".to_string(), traces_list);
        }
    }
    out
}

pub fn format_markdown_report_internal(payload: &Value) -> String {
    let stats = payload.get("statistics").unwrap_or(&Value::Null);
    let summary = stats.get("summary").unwrap_or(&Value::Null);
    let combos = stats
        .get("model_combos")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let steps = stats
        .get("individual_steps")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let items = payload
        .get("feedback_items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let overall_rate_str = match summary.get("correctness_rate_pct") {
        Some(v) if !v.is_null() => format!("{}%", v),
        _ => "N/A".to_string(),
    };

    let gen_at = payload
        .get("generated_at")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let total_fb = summary
        .get("total_feedback")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let corr_cnt = summary
        .get("correct_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let incorr_cnt = summary
        .get("incorrect_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let avg_tot_ms = summary
        .get("avg_total_duration_ms")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let mut lines: Vec<String> = Vec::with_capacity(64 + items.len() * 32);
    lines.push("# Matrika & Census Digitizer — OCR Feedback & LLM Debugging Report".to_string());
    lines.push(format!("- **Generated At**: `{}`", gen_at));
    lines.push("- **Workspace Report Files**: `data/ocr_feedback_debug_report.md` | `data/ocr_feedback_debug_report.json`".to_string());
    lines.push(format!(
        "- **Total Feedback Items**: `{}` (`{}` CORRECT, `{}` INCORRECT)",
        total_fb, corr_cnt, incorr_cnt
    ));
    lines.push(format!("- **Overall Correctness Rate**: `{}`", overall_rate_str));
    lines.push(format!("- **Average Total Row Time**: `{} ms`", avg_tot_ms));
    lines.push(String::new());
    lines.push("## 1. Model Combination Correctness & Average Step Timing".to_string());
    lines.push(String::new());
    lines.push("| Model Combination | Correctness Rate | Correct / Rated | Rows | Avg Total (ms) | Layout (ms) | OCR Total (ms) | OCR Left / Right / Hdr (ms) | Decision (ms) | Extract (ms) | MCP (ms) |".to_string());
    lines.push("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |".to_string());

    for c in &combos {
        let rate_str = match c.get("correctness_rate_pct") {
            Some(v) if !v.is_null() => format!("**{}%**", v),
            _ => "— (unrated)".to_string(),
        };
        let st = c.get("steps").unwrap_or(&Value::Null);
        let get_i = |obj: &Value, k: &str| obj.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
        let mc = c
            .get("model_combo")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        lines.push(format!(
            "| `{}` | {} | {} / {} ({} err) | {} | **{} ms** | {} ms | {} ms | L:{} / R:{} / H:{} | {} ms | {} ms | {} ms |",
            mc,
            rate_str,
            get_i(c, "correct_count"),
            get_i(c, "total_rated"),
            get_i(c, "incorrect_count"),
            get_i(c, "rows_processed"),
            get_i(c, "avg_total_ms"),
            get_i(st, "layout_ms"),
            get_i(st, "ocr_ms"),
            get_i(st, "ocr_left_ms"),
            get_i(st, "ocr_right_ms"),
            get_i(st, "ocr_header_ms"),
            get_i(st, "decision_ms"),
            get_i(st, "extract_ms"),
            get_i(st, "mcp_ms"),
        ));
    }

    lines.push(String::new());
    lines.push("## 2. Individual Pipeline Steps Breakdown".to_string());
    lines.push(String::new());
    lines.push("| Stage | Canonical Step | Model / Tool | Calls | Avg Time (ms) | Min / Max (ms) | Avg Tokens (In / Out / Reason) | Rated Row Accuracy |".to_string());
    lines.push("| --- | --- | --- | --- | --- | --- | --- | --- |".to_string());

    for s in &steps {
        let get_i = |k: &str| s.get(k).and_then(|v| v.as_i64()).unwrap_or(0);
        let s_rate = match s.get("correctness_rate_pct") {
            Some(v) if !v.is_null() => format!(
                "{}% ({}/{})",
                v,
                get_i("correct_rows_count"),
                get_i("rated_rows_count")
            ),
            _ => "—".to_string(),
        };
        let st_str = s.get("stage").and_then(|v| v.as_str()).unwrap_or("");
        let cs_str = s
            .get("canonical_stage")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let mt_str = s
            .get("model_or_tool")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        lines.push(format!(
            "| `{}` | `{}` | `{}` | {} | **{} ms** | {} / {} ms | {} / {} / {} | {} |",
            st_str,
            cs_str,
            mt_str,
            get_i("calls_count"),
            get_i("avg_duration_ms"),
            get_i("min_duration_ms"),
            get_i("max_duration_ms"),
            get_i("avg_prompt_tokens"),
            get_i("avg_completion_tokens"),
            get_i("avg_reasoning_tokens"),
            s_rate
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "## 3. Detailed User Feedback Entries ({} items)",
        items.len()
    ));
    lines.push(String::new());

    if items.is_empty() {
        lines.push("_No OCR feedback entries recorded yet._".to_string());
    } else {
        for (idx, item) in items.iter().enumerate() {
            let verdict_upper = item
                .get("verdict")
                .and_then(|v| v.as_str())
                .unwrap_or("correct")
                .to_uppercase();
            let ext_snap = item.get("extracted_snapshot").unwrap_or(&Value::Null);
            let sc_info = ext_snap.get("scan").unwrap_or(&Value::Null);
            let cs_info = ext_snap.get("census_sheet").unwrap_or(&Value::Null);
            let row_info = ext_snap.get("row").unwrap_or(&Value::Null);
            let sb = item.get("stage_breakdown").unwrap_or(&Value::Null);
            let traces = item
                .get("traces_snapshot")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            let disp = item
                .get("display_label")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let fb_id = item.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            let comment_str = item
                .get("comment")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("_None provided_");

            lines.push(format!(
                "### {}. [{}] {} (Feedback ID #{})",
                idx + 1,
                verdict_upper,
                disp,
                fb_id
            ));
            lines.push(format!("- **Verdict**: `{}`", verdict_upper));
            lines.push(format!(
                "- **User Comment / Issue Description**: {}",
                comment_str
            ));
            if let Some(exp) = item
                .get("expected_correction")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                lines.push(format!("- **Expected Correction**: `{}`", exp));
            }

            let val_disp = |obj: &Value, k: &str| -> String {
                match obj.get(k) {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    Some(Value::Bool(b)) => b.to_string(),
                    _ => "None".to_string(),
                }
            };
            let get_i = |obj: &Value, k: &str| obj.get(k).and_then(|v| v.as_i64()).unwrap_or(0);

            lines.push(format!("- **Updated At**: `{}`", val_disp(item, "updated_at")));
            lines.push(format!(
                "- **Target**: `target_type={}`, `row_id={}`, `scan_id={}`, `census_sheet_id={}`, `person_fact_id={}`, `overview_house_id={}`, `municipal_district_id={}`",
                val_disp(item, "target_type"),
                val_disp(item, "row_id"),
                val_disp(item, "scan_id"),
                val_disp(item, "census_sheet_id"),
                val_disp(item, "person_fact_id"),
                val_disp(item, "overview_house_id"),
                val_disp(item, "municipal_district_id"),
            ));
            lines.push(format!(
                "- **Model Combo**: `{}`",
                val_disp(item, "model_combo")
            ));
            lines.push(format!(
                "  - **Layout**: `{}` (`{} ms`, scan total: `{} ms`)",
                val_disp(item, "layout_model"),
                get_i(item, "layout_duration_ms"),
                get_i(sb, "scan_layout_ms"),
            ));
            lines.push(format!(
                "  - **OCR**: `{}` (`{} ms` — left: `{} ms`, right: `{} ms`, header: `{} ms`)",
                val_disp(item, "ocr_model"),
                get_i(item, "ocr_duration_ms"),
                get_i(sb, "ocr_left_ms"),
                get_i(sb, "ocr_right_ms"),
                get_i(sb, "ocr_header_ms"),
            ));
            lines.push(format!(
                "  - **Decision**: `{}` (`{} ms`)",
                val_disp(item, "decision_model"),
                get_i(item, "decision_duration_ms"),
            ));
            lines.push(format!(
                "  - **Extract**: `{}` (`{} ms`)",
                val_disp(item, "extract_model"),
                get_i(item, "extract_duration_ms"),
            ));
            let mcp_m = item
                .get("mcp_model")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("none");
            lines.push(format!(
                "  - **MCP**: `{}` (`{} ms`)",
                mcp_m,
                get_i(item, "mcp_duration_ms"),
            ));
            lines.push(format!(
                "  - **Total Row Processing Time**: `{} ms` (tokens: prompt=`{}`, completion=`{}`, reasoning=`{}`)",
                get_i(item, "total_duration_ms"),
                get_i(sb, "prompt_tokens"),
                get_i(sb, "completion_tokens"),
                get_i(sb, "reasoning_tokens"),
            ));

            if cs_info.is_object() {
                lines.push(format!(
                    "- **Census Sheet Context**: key=`{}`, year=`{}`, municipality=`{}`, district=`{}`, house_number=`{}`, sheet_type=`{}`",
                    val_disp(cs_info, "slovakiana_key"),
                    val_disp(cs_info, "census_year"),
                    val_disp(cs_info, "municipality"),
                    val_disp(cs_info, "district"),
                    val_disp(cs_info, "house_number"),
                    val_disp(cs_info, "sheet_type"),
                ));
            }
            if sc_info.is_object() {
                lines.push(format!(
                    "- **Scan Files**: filename=`{}`, original_path=`{}`, overlay_path=`{}`",
                    val_disp(sc_info, "filename"),
                    val_disp(sc_info, "original_path"),
                    val_disp(sc_info, "overlay_path"),
                ));
            }
            if row_info.is_object() || item.get("crop_url").filter(|v| !v.is_null()).is_some() {
                let st_crop = row_info
                    .get("stitched_crop_path")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("crop_url").and_then(|v| v.as_str()))
                    .unwrap_or("None");
                lines.push(format!(
                    "- **Row Crops**: stitched=`{}`, left=`{}`, right=`{}`",
                    st_crop,
                    val_disp(row_info, "left_crop_path"),
                    val_disp(row_info, "right_crop_path"),
                ));
            }

            let raw_ocr = item
                .get("raw_ocr_snapshot")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("(empty)");
            lines.push(String::new());
            lines.push("#### Raw OCR Text".to_string());
            lines.push("```text".to_string());
            lines.push(raw_ocr.to_string());
            lines.push("```".to_string());
            lines.push(String::new());
            lines.push("#### Extracted Structured Data (Snapshot JSON)".to_string());
            lines.push("```json".to_string());

            let mut filtered_snap = serde_json::Map::new();
            if let Some(map) = ext_snap.as_object() {
                for k in ["person_fact", "overview_house", "municipal_district", "row"] {
                    if let Some(v) = map.get(k) {
                        if !v.is_null() {
                            filtered_snap.insert(k.to_string(), v.clone());
                        }
                    }
                }
            }
            lines.push(
                serde_json::to_string_pretty(&Value::Object(filtered_snap))
                    .unwrap_or_else(|_| "{}".to_string()),
            );
            lines.push("```".to_string());

            if !traces.is_empty() {
                lines.push(String::new());
                lines.push(format!(
                    "#### Pipeline Stage Traces ({} calls)",
                    traces.len()
                ));
                for tr in &traces {
                    lines.push(format!(
                        "- **Trace #{}** — `{}` · `{}` · `{} ms` · status=`{}` · image=`{}`",
                        val_disp(tr, "id"),
                        val_disp(tr, "stage"),
                        val_disp(tr, "model_or_tool"),
                        val_disp(tr, "duration_ms"),
                        val_disp(tr, "status"),
                        val_disp(tr, "image_path"),
                    ));
                    let p_txt = tr
                        .get("prompt_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .replace('\n', "\n    ");
                    lines.push("  - **Prompt**:".to_string());
                    lines.push("    ```text".to_string());
                    lines.push(format!("    {}", p_txt));
                    lines.push("    ```".to_string());

                    if let Some(rc) = tr
                        .get("reasoning_content")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        lines.push("  - **Reasoning**:".to_string());
                        lines.push("    ```text".to_string());
                        lines.push(format!("    {}", rc.replace('\n', "\n    ")));
                        lines.push("    ```".to_string());
                    }

                    let r_txt = tr
                        .get("response_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .replace('\n', "\n    ");
                    lines.push("  - **Response**:".to_string());
                    lines.push("    ```text".to_string());
                    lines.push(format!("    {}", r_txt));
                    lines.push("    ```".to_string());
                }
            }
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

#[pyfunction]
#[pyo3(signature = (db_path, census_sheet_id=None, scan_ids=None, row_ids=None, fact_ids=None))]
pub fn build_feedback_lookup_maps_rs(
    py: Python<'_>,
    db_path: String,
    census_sheet_id: Option<i64>,
    scan_ids: Option<Vec<i64>>,
    row_ids: Option<Vec<i64>>,
    fact_ids: Option<Vec<i64>>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let conn = open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut where_clause = String::new();
        let mut params_vec: Vec<i64> = Vec::new();

        let s_ids = scan_ids.unwrap_or_default();
        let r_ids = row_ids.unwrap_or_default();
        let f_ids = fact_ids.unwrap_or_default();

        if let Some(cs_id) = census_sheet_id {
            if !s_ids.is_empty() {
                let placeholders: Vec<String> =
                    (0..s_ids.len()).map(|i| format!("?{}", i + 2)).collect();
                where_clause = format!(
                    "WHERE (census_sheet_id = ?1 OR scan_id IN ({}))",
                    placeholders.join(", ")
                );
                params_vec.push(cs_id);
                params_vec.extend(s_ids);
            } else {
                where_clause = "WHERE census_sheet_id = ?1".to_string();
                params_vec.push(cs_id);
            }
        } else if !s_ids.is_empty() {
            let placeholders: Vec<String> =
                (0..s_ids.len()).map(|i| format!("?{}", i + 1)).collect();
            where_clause = format!("WHERE scan_id IN ({})", placeholders.join(", "));
            params_vec.extend(s_ids);
        } else if !r_ids.is_empty() {
            let placeholders: Vec<String> =
                (0..r_ids.len()).map(|i| format!("?{}", i + 1)).collect();
            where_clause = format!("WHERE row_id IN ({})", placeholders.join(", "));
            params_vec.extend(r_ids);
        } else if !f_ids.is_empty() {
            let placeholders: Vec<String> =
                (0..f_ids.len()).map(|i| format!("?{}", i + 1)).collect();
            where_clause = format!("WHERE person_fact_id IN ({})", placeholders.join(", "));
            params_vec.extend(f_ids);
        }

        let sql = format!(
            "SELECT {} FROM ocrfeedback {} ORDER BY updated_at DESC, id DESC",
            FEEDBACK_SELECT_COLS, where_clause
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let fb_rows: Vec<FeedbackRow> = stmt
            .query_map(params_from_iter(params_vec.iter()), read_feedback_row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut by_row = serde_json::Map::new();
        let mut by_fact = serde_json::Map::new();
        let mut by_house = serde_json::Map::new();
        let mut by_dist = serde_json::Map::new();

        for fb in &fb_rows {
            let ser = serialize_feedback_row(fb, false);
            if let Some(rid) = fb.row_id {
                let k = rid.to_string();
                by_row.entry(k).or_insert_with(|| ser.clone());
            }
            if let Some(fid) = fb.person_fact_id {
                let k = fid.to_string();
                by_fact.entry(k).or_insert_with(|| ser.clone());
            }
            if let Some(hid) = fb.overview_house_id {
                let k = hid.to_string();
                by_house.entry(k).or_insert_with(|| ser.clone());
            }
            if let Some(did) = fb.municipal_district_id {
                let k = did.to_string();
                by_dist.entry(k).or_insert_with(|| ser.clone());
            }
        }

        let out = json!({
            "by_row": Value::Object(by_row),
            "by_fact": Value::Object(by_fact),
            "by_house": Value::Object(by_house),
            "by_dist": Value::Object(by_dist),
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (db_path, data_dir, generated_at, ai_runtime_config_json, default_ocr, default_decision, default_extract, default_layout, known_local_models, verdict_filter=None, feedback_id=None, persist_files=false))]
pub fn build_and_persist_llm_debug_report_rs(
    py: Python<'_>,
    db_path: String,
    data_dir: String,
    generated_at: String,
    ai_runtime_config_json: String,
    default_ocr: String,
    default_decision: String,
    default_extract: String,
    default_layout: String,
    known_local_models: Vec<String>,
    verdict_filter: Option<String>,
    feedback_id: Option<i64>,
    persist_files: bool,
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

        let sql_all = format!(
            "SELECT {} FROM ocrfeedback ORDER BY verdict DESC, updated_at DESC, id DESC",
            FEEDBACK_SELECT_COLS
        );
        let mut stmt = conn
            .prepare(&sql_all)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let all_fb_rows: Vec<FeedbackRow> = stmt
            .query_map([], read_feedback_row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let ai_cfg: Value =
            serde_json::from_str(&ai_runtime_config_json).unwrap_or_else(|_| json!({}));

        let all_serialized: Vec<Value> = all_fb_rows
            .iter()
            .map(|fb| serialize_feedback_row(fb, true))
            .collect();

        if persist_files {
            let full_payload = json!({
                "generated_at": generated_at,
                "ai_runtime_config": ai_cfg,
                "statistics": stats,
                "feedback_count": all_serialized.len(),
                "feedback_items": all_serialized,
            });
            let full_md = format_markdown_report_internal(&full_payload);
            let full_json =
                serde_json::to_string_pretty(&full_payload).unwrap_or_else(|_| "{}".to_string());
            let dir_path = Path::new(&data_dir);
            let _ = fs::create_dir_all(dir_path);
            let _ = fs::write(dir_path.join("ocr_feedback_debug_report.md"), &full_md);
            let _ = fs::write(dir_path.join("ocr_feedback_debug_report.json"), &full_json);

            if feedback_id.is_none()
                && !matches!(
                    verdict_filter.as_deref(),
                    Some("correct") | Some("incorrect")
                )
            {
                let res = json!({
                    "payload": full_payload,
                    "markdown": full_md,
                });
                return serde_json::to_string(&res)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }
        }

        let filtered_serialized: Vec<Value> = all_fb_rows
            .iter()
            .filter(|fb| {
                if let Some(fid) = feedback_id {
                    fb.id == fid
                } else if let Some(ref vf) = verdict_filter {
                    if vf == "correct" || vf == "incorrect" {
                        &fb.verdict == vf
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .map(|fb| serialize_feedback_row(fb, true))
            .collect();

        let filtered_payload = json!({
            "generated_at": generated_at,
            "ai_runtime_config": ai_cfg,
            "statistics": stats,
            "feedback_count": filtered_serialized.len(),
            "feedback_items": filtered_serialized,
        });
        let filtered_md = format_markdown_report_internal(&filtered_payload);

        let res = json!({
            "payload": filtered_payload,
            "markdown": filtered_md,
        });
        serde_json::to_string(&res).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_row_execution_context_conn(
    conn: &Connection,
    defaults: &ModelDefaults,
    mut target_type: String,
    mut row_id: Option<i64>,
    mut scan_id: Option<i64>,
    mut census_sheet_id: Option<i64>,
    mut person_fact_id: Option<i64>,
    mut overview_house_id: Option<i64>,
    mut municipal_district_id: Option<i64>,
) -> Result<Value, rusqlite::Error> {
    let mut fact_val: Option<Value> = None;
    let mut fact_rec_type: Option<String> = None;

    let load_fact_by = |conn: &Connection, where_col: &str, id_val: i64| -> Result<Option<(i64, i64, i64, Option<i64>, String, Value)>, rusqlite::Error> {
        let sql = format!(
            "SELECT id, row_id, scan_id, census_sheet_id, record_type, entry_number, person_row_number,
                    surname_raw, surname_normalized, given_name_raw, given_name_sk, mother_maiden_surname,
                    relation_to_head, sex, birth_or_event_date, baptism_or_burial_date, birth_year_est,
                    event_year, marital_status, birth_place_raw, birth_place_municipality,
                    birth_place_district, birth_place_country, home_jurisdiction, citizenship,
                    nationality, religion, literacy, social_status_or_occupation, occupation_position,
                    employer_workplace, father_name, mother_name, village_raw, house_number, observationes_raw
             FROM personfact WHERE {} = ?1 LIMIT 1",
            where_col
        );
        conn.query_row(&sql, params![id_val], |r| {
            let fid: i64 = r.get(0)?;
            let rid: i64 = r.get::<_, Option<i64>>(1)?.unwrap_or(0);
            let sid: i64 = r.get::<_, Option<i64>>(2)?.unwrap_or(0);
            let csid: Option<i64> = r.get(3)?;
            let rtype: String = r.get::<_, Option<String>>(4)?.unwrap_or_else(|| "baptism".to_string());
            let obj = json!({
                "id": fid,
                "record_type": rtype,
                "entry_number": r.get::<_, Option<String>>(5)?,
                "person_row_number": r.get::<_, Option<String>>(6)?,
                "surname_raw": r.get::<_, Option<String>>(7)?,
                "surname_normalized": r.get::<_, Option<String>>(8)?,
                "given_name_raw": r.get::<_, Option<String>>(9)?,
                "given_name_sk": r.get::<_, Option<String>>(10)?,
                "mother_maiden_surname": r.get::<_, Option<String>>(11)?,
                "relation_to_head": r.get::<_, Option<String>>(12)?,
                "sex": r.get::<_, Option<String>>(13)?,
                "birth_or_event_date": r.get::<_, Option<String>>(14)?,
                "baptism_or_burial_date": r.get::<_, Option<String>>(15)?,
                "birth_year_est": r.get::<_, Option<i64>>(16)?,
                "event_year": r.get::<_, Option<i64>>(17)?,
                "marital_status": r.get::<_, Option<String>>(18)?,
                "birth_place_raw": r.get::<_, Option<String>>(19)?,
                "birth_place_municipality": r.get::<_, Option<String>>(20)?,
                "birth_place_district": r.get::<_, Option<String>>(21)?,
                "birth_place_country": r.get::<_, Option<String>>(22)?,
                "home_jurisdiction": r.get::<_, Option<String>>(23)?,
                "citizenship": r.get::<_, Option<String>>(24)?,
                "nationality": r.get::<_, Option<String>>(25)?,
                "religion": r.get::<_, Option<String>>(26)?,
                "literacy": r.get::<_, Option<String>>(27)?,
                "social_status_or_occupation": r.get::<_, Option<String>>(28)?,
                "occupation_position": r.get::<_, Option<String>>(29)?,
                "employer_workplace": r.get::<_, Option<String>>(30)?,
                "father_name": r.get::<_, Option<String>>(31)?,
                "mother_name": r.get::<_, Option<String>>(32)?,
                "village_raw": r.get::<_, Option<String>>(33)?,
                "house_number": r.get::<_, Option<String>>(34)?,
                "observationes_raw": r.get::<_, Option<String>>(35)?,
            });
            Ok((fid, rid, sid, csid, rtype, obj))
        }).optional()
    };

    if let Some(pf_id) = person_fact_id.filter(|&x| x > 0) {
        if let Some((fid, rid, sid, csid, rtype, obj)) = load_fact_by(conn, "id", pf_id)? {
            person_fact_id = Some(fid);
            if row_id.filter(|&x| x > 0).is_none() && rid > 0 {
                row_id = Some(rid);
            }
            if scan_id.filter(|&x| x > 0).is_none() && sid > 0 {
                scan_id = Some(sid);
            }
            if census_sheet_id.filter(|&x| x > 0).is_none() {
                census_sheet_id = csid;
            }
            if target_type == "row" {
                target_type = "person_fact".to_string();
            }
            fact_rec_type = Some(rtype);
            fact_val = Some(obj);
        }
    }

    let mut house_val: Option<Value> = None;
    let mut house_crop: Option<String> = None;
    let mut house_raw_ocr: Option<String> = None;
    if let Some(oh_id) = overview_house_id.filter(|&x| x > 0) {
        if let Some((hid, sid, csid, r_idx, st_crop, raw_ocr, obj)) = conn
            .query_row(
                "SELECT id, scan_id, census_sheet_id, row_index, ordinal_number, settlement_name,
                        house_number, is_inhabited, owner_name, household_count,
                        present_persons_count, remarks, raw_ocr_text, stitched_crop_path
                 FROM censusoverviewhouse WHERE id = ?1",
                params![oh_id],
                |r| {
                    let hid: i64 = r.get(0)?;
                    let sid: Option<i64> = r.get(1)?;
                    let csid: i64 = r.get(2)?;
                    let r_idx: i64 = r.get::<_, Option<i64>>(3)?.unwrap_or(1);
                    let raw_ocr: Option<String> = r.get(12)?;
                    let st_crop: Option<String> = r.get(13)?;
                    let obj = json!({
                        "id": hid,
                        "row_index": r_idx,
                        "ordinal_number": r.get::<_, Option<String>>(4)?,
                        "settlement_name": r.get::<_, Option<String>>(5)?,
                        "house_number": r.get::<_, Option<String>>(6)?,
                        "is_inhabited": r.get::<_, Option<bool>>(7)?,
                        "owner_name": r.get::<_, Option<String>>(8)?,
                        "household_count": r.get::<_, Option<i64>>(9)?,
                        "present_persons_count": r.get::<_, Option<i64>>(10)?,
                        "remarks": r.get::<_, Option<String>>(11)?,
                        "raw_ocr_text": raw_ocr,
                    });
                    Ok((hid, sid, csid, r_idx, st_crop, raw_ocr, obj))
                },
            )
            .optional()?
        {
            overview_house_id = Some(hid);
            if scan_id.filter(|&x| x > 0).is_none() {
                scan_id = sid;
            }
            if census_sheet_id.filter(|&x| x > 0).is_none() && csid > 0 {
                census_sheet_id = Some(csid);
            }
            if target_type == "row" {
                target_type = "overview_house".to_string();
            }
            if row_id.filter(|&x| x > 0).is_none() {
                if let Some(s_id) = scan_id.filter(|&x| x > 0) {
                    let mut c_stmt = conn.prepare(
                        "SELECT id, row_index, stitched_crop_path FROM recordrow WHERE scan_id = ?1 ORDER BY row_index ASC",
                    )?;
                    let cands: Vec<(i64, i64, Option<String>)> = c_stmt
                        .query_map(params![s_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                        .collect::<Result<Vec<_>, _>>()?;
                    for (cid, c_idx, c_crop) in cands {
                        if st_crop.as_ref().filter(|s| !s.is_empty()).is_some()
                            && c_crop == st_crop
                        {
                            row_id = Some(cid);
                            break;
                        }
                        if c_idx == r_idx {
                            row_id = Some(cid);
                            break;
                        }
                    }
                }
            }
            house_crop = st_crop;
            house_raw_ocr = raw_ocr;
            house_val = Some(obj);
        }
    }

    let mut dist_val: Option<Value> = None;
    let mut dist_crop: Option<String> = None;
    let mut dist_raw_ocr: Option<String> = None;
    if let Some(md_id) = municipal_district_id.filter(|&x| x > 0) {
        if let Some((did, sid, csid, r_idx, st_crop, raw_ocr, obj)) = conn
            .query_row(
                "SELECT id, scan_id, census_sheet_id, row_index, ordinal_number, settlement_name,
                        obvod_number, commissioner_name, houses_total, houses_inhabited,
                        houses_uninhabited, household_count, present_persons_count,
                        is_summary_row, raw_ocr_text, stitched_crop_path
                 FROM censusmunicipaloverviewdistrict WHERE id = ?1",
                params![md_id],
                |r| {
                    let did: i64 = r.get(0)?;
                    let sid: Option<i64> = r.get(1)?;
                    let csid: i64 = r.get(2)?;
                    let r_idx: i64 = r.get::<_, Option<i64>>(3)?.unwrap_or(1);
                    let raw_ocr: Option<String> = r.get(14)?;
                    let st_crop: Option<String> = r.get(15)?;
                    let obj = json!({
                        "id": did,
                        "row_index": r_idx,
                        "ordinal_number": r.get::<_, Option<String>>(4)?,
                        "settlement_name": r.get::<_, Option<String>>(5)?,
                        "obvod_number": r.get::<_, Option<String>>(6)?,
                        "commissioner_name": r.get::<_, Option<String>>(7)?,
                        "houses_total": r.get::<_, Option<i64>>(8)?,
                        "houses_inhabited": r.get::<_, Option<i64>>(9)?,
                        "houses_uninhabited": r.get::<_, Option<i64>>(10)?,
                        "household_count": r.get::<_, Option<i64>>(11)?,
                        "present_persons_count": r.get::<_, Option<i64>>(12)?,
                        "is_summary_row": r.get::<_, Option<bool>>(13)?.unwrap_or(false),
                        "raw_ocr_text": raw_ocr,
                    });
                    Ok((did, sid, csid, r_idx, st_crop, raw_ocr, obj))
                },
            )
            .optional()?
        {
            municipal_district_id = Some(did);
            if scan_id.filter(|&x| x > 0).is_none() {
                scan_id = sid;
            }
            if census_sheet_id.filter(|&x| x > 0).is_none() && csid > 0 {
                census_sheet_id = Some(csid);
            }
            if target_type == "row" {
                target_type = "municipal_district".to_string();
            }
            if row_id.filter(|&x| x > 0).is_none() {
                if let Some(s_id) = scan_id.filter(|&x| x > 0) {
                    let mut c_stmt = conn.prepare(
                        "SELECT id, row_index, stitched_crop_path FROM recordrow WHERE scan_id = ?1 ORDER BY row_index ASC",
                    )?;
                    let cands: Vec<(i64, i64, Option<String>)> = c_stmt
                        .query_map(params![s_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                        .collect::<Result<Vec<_>, _>>()?;
                    for (cid, c_idx, c_crop) in cands {
                        if st_crop.as_ref().filter(|s| !s.is_empty()).is_some()
                            && c_crop == st_crop
                        {
                            row_id = Some(cid);
                            break;
                        }
                        if c_idx == r_idx {
                            row_id = Some(cid);
                            break;
                        }
                    }
                }
            }
            dist_crop = st_crop;
            dist_raw_ocr = raw_ocr;
            dist_val = Some(obj);
        }
    }

    let mut row_val: Option<Value> = None;
    let mut row_stitched_crop: Option<String> = None;
    let mut row_left_crop: Option<String> = None;
    let mut row_right_crop: Option<String> = None;
    let mut row_cell_crops_json = "[]".to_string();
    let mut row_raw_ocr_combined: Option<String> = None;
    let mut row_raw_ocr_left: Option<String> = None;
    let mut row_rec_type: Option<String> = None;
    let mut row_index_val: i64 = 0;

    if let Some(r_id) = row_id.filter(|&x| x > 0) {
        if let Some((
            rid,
            sid,
            r_idx,
            st_crop,
            l_crop,
            r_crop,
            cc_json,
            ocr_l,
            ocr_r,
            ocr_comb,
            rtype,
            obj,
        )) = conn
            .query_row(
                "SELECT id, scan_id, row_index, entry_number, y_start, y_end, stitched_crop_path,
                        left_crop_path, right_crop_path, cell_crops_json, raw_ocr_left, raw_ocr_right,
                        raw_ocr_combined, decision_action, decision_confidence, normalized_json, record_type
                 FROM recordrow WHERE id = ?1",
                params![r_id],
                |r| {
                    let rid: i64 = r.get(0)?;
                    let sid: i64 = r.get(1)?;
                    let r_idx: i64 = r.get(2)?;
                    let st_crop: Option<String> = r.get(6)?;
                    let l_crop: Option<String> = r.get(7)?;
                    let r_crop: Option<String> = r.get(8)?;
                    let cc_json: String = r
                        .get::<_, Option<String>>(9)?
                        .unwrap_or_else(|| "[]".to_string());
                    let ocr_l: Option<String> = r.get(10)?;
                    let ocr_r: Option<String> = r.get(11)?;
                    let ocr_comb: Option<String> = r.get(12)?;
                    let norm_json: String = r
                        .get::<_, Option<String>>(15)?
                        .unwrap_or_else(|| "{}".to_string());
                    let rtype: String = r
                        .get::<_, Option<String>>(16)?
                        .unwrap_or_else(|| "baptism".to_string());
                    let cell_crops_val: Value =
                        serde_json::from_str(&cc_json).unwrap_or_else(|_| json!([]));
                    let norm_val: Value =
                        serde_json::from_str(&norm_json).unwrap_or_else(|_| json!({}));
                    let obj = json!({
                        "id": rid,
                        "row_index": r_idx,
                        "entry_number": r.get::<_, Option<String>>(3)?,
                        "y_start": r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        "y_end": r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                        "stitched_crop_path": st_crop,
                        "left_crop_path": l_crop,
                        "right_crop_path": r_crop,
                        "cell_crops": cell_crops_val,
                        "raw_ocr_left": ocr_l,
                        "raw_ocr_right": ocr_r,
                        "raw_ocr_combined": ocr_comb,
                        "decision_action": r.get::<_, Option<String>>(13)?,
                        "decision_confidence": r.get::<_, Option<f64>>(14)?.unwrap_or(0.0),
                        "normalized": norm_val,
                    });
                    Ok((
                        rid, sid, r_idx, st_crop, l_crop, r_crop, cc_json, ocr_l, ocr_r, ocr_comb,
                        rtype, obj,
                    ))
                },
            )
            .optional()?
        {
            row_id = Some(rid);
            if scan_id.filter(|&x| x > 0).is_none() && sid > 0 {
                scan_id = Some(sid);
            }
            if fact_val.is_none() {
                if let Some((fid, _, _, csid, f_rtype, f_obj)) = load_fact_by(conn, "row_id", rid)?
                {
                    person_fact_id = Some(fid);
                    if census_sheet_id.filter(|&x| x > 0).is_none() {
                        census_sheet_id = csid;
                    }
                    fact_rec_type = Some(f_rtype);
                    fact_val = Some(f_obj);
                }
            }
            row_index_val = r_idx;
            row_stitched_crop = st_crop;
            row_left_crop = l_crop;
            row_right_crop = r_crop;
            row_cell_crops_json = cc_json;
            row_raw_ocr_left = ocr_l;
            let _ = ocr_r;
            row_raw_ocr_combined = ocr_comb;
            row_rec_type = Some(rtype);
            row_val = Some(obj);
        }
    }

    let mut scan_val: Option<Value> = None;
    let mut scan_header_ocr: Option<String> = None;
    let mut scan_header_crop: Option<String> = None;
    let mut scan_det_type: Option<String> = None;

    let load_scan = |conn: &Connection, sid: i64| -> Result<Option<(i64, Option<i64>, Option<String>, Option<String>, Option<String>, Value)>, rusqlite::Error> {
        conn.query_row(
            "SELECT id, census_sheet_id, filename, page_side, original_path, overlay_path,
                    header_crop_path, detected_year, detected_type, header_raw_ocr
             FROM scanimage WHERE id = ?1",
            params![sid],
            |r| {
                let s_id: i64 = r.get(0)?;
                let csid: Option<i64> = r.get(1)?;
                let hdr_crop: Option<String> = r.get(6)?;
                let det_type: Option<String> = r.get(8)?;
                let hdr_ocr: Option<String> = r.get(9)?;
                let obj = json!({
                    "id": s_id,
                    "filename": r.get::<_, Option<String>>(2)?,
                    "page_side": r.get::<_, Option<String>>(3)?,
                    "original_path": r.get::<_, Option<String>>(4)?,
                    "overlay_path": r.get::<_, Option<String>>(5)?,
                    "header_crop_path": hdr_crop,
                    "detected_year": r.get::<_, Option<String>>(7)?,
                    "detected_type": det_type,
                });
                Ok((s_id, csid, hdr_crop, det_type, hdr_ocr, obj))
            },
        ).optional()
    };

    if let Some(s_id) = scan_id.filter(|&x| x > 0) {
        if let Some((sid, csid, hdr_crop, det_type, hdr_ocr, obj)) = load_scan(conn, s_id)? {
            scan_id = Some(sid);
            if census_sheet_id.filter(|&x| x > 0).is_none() {
                census_sheet_id = csid;
            }
            scan_header_crop = hdr_crop;
            scan_det_type = det_type;
            scan_header_ocr = hdr_ocr;
            scan_val = Some(obj);
        }
    }

    let mut sheet_val: Option<Value> = None;
    let mut sheet_type_str: Option<String> = None;
    if let Some(cs_id) = census_sheet_id.filter(|&x| x > 0) {
        if let Some((csid, f_sid, b_sid, stype, obj)) = conn
            .query_row(
                "SELECT id, slovakiana_key, census_year, district, municipality, house_number,
                        sheet_number, sheet_type, obvod_number, commissioner_name,
                        front_scan_id, back_scan_id
                 FROM censussheet WHERE id = ?1",
                params![cs_id],
                |r| {
                    let csid: i64 = r.get(0)?;
                    let stype: String = r
                        .get::<_, Option<String>>(7)?
                        .unwrap_or_else(|| "household".to_string());
                    let f_sid: Option<i64> = r.get(10)?;
                    let b_sid: Option<i64> = r.get(11)?;
                    let obj = json!({
                        "id": csid,
                        "slovakiana_key": r.get::<_, Option<String>>(1)?,
                        "census_year": r.get::<_, Option<i64>>(2)?,
                        "district": r.get::<_, Option<String>>(3)?,
                        "municipality": r.get::<_, Option<String>>(4)?,
                        "house_number": r.get::<_, Option<String>>(5)?,
                        "sheet_number": r.get::<_, Option<String>>(6)?,
                        "sheet_type": stype,
                        "obvod_number": r.get::<_, Option<String>>(8)?,
                        "commissioner_name": r.get::<_, Option<String>>(9)?,
                    });
                    Ok((csid, f_sid, b_sid, stype, obj))
                },
            )
            .optional()?
        {
            census_sheet_id = Some(csid);
            sheet_type_str = Some(stype);
            sheet_val = Some(obj);
            if scan_val.is_none() {
                if let Some(fb_sid) = f_sid.or(b_sid).filter(|&x| x > 0) {
                    if let Some((sid, _, hdr_crop, det_type, hdr_ocr, sc_obj)) =
                        load_scan(conn, fb_sid)?
                    {
                        scan_id = Some(sid);
                        scan_header_crop = hdr_crop;
                        scan_det_type = det_type;
                        scan_header_ocr = hdr_ocr;
                        scan_val = Some(sc_obj);
                    }
                }
            }
        }
    }

    if row_val.is_some()
        && sheet_val.is_some()
        && fact_val.is_none()
        && house_val.is_none()
        && dist_val.is_none()
    {
        if let Some(cs_id) = census_sheet_id {
            if sheet_type_str.as_deref() == Some("overview") {
                if let Some((hid, raw_ocr, st_crop, obj)) = conn
                    .query_row(
                        "SELECT id, row_index, ordinal_number, settlement_name, house_number,
                                is_inhabited, owner_name, household_count, present_persons_count,
                                remarks, raw_ocr_text, stitched_crop_path
                         FROM censusoverviewhouse
                         WHERE census_sheet_id = ?1 AND (stitched_crop_path = ?2 OR row_index = ?3)
                         LIMIT 1",
                        params![cs_id, row_stitched_crop, row_index_val],
                        |r| {
                            let hid: i64 = r.get(0)?;
                            let raw_ocr: Option<String> = r.get(10)?;
                            let st_crop: Option<String> = r.get(11)?;
                            let obj = json!({
                                "id": hid,
                                "row_index": r.get::<_, Option<i64>>(1)?.unwrap_or(1),
                                "ordinal_number": r.get::<_, Option<String>>(2)?,
                                "settlement_name": r.get::<_, Option<String>>(3)?,
                                "house_number": r.get::<_, Option<String>>(4)?,
                                "is_inhabited": r.get::<_, Option<bool>>(5)?,
                                "owner_name": r.get::<_, Option<String>>(6)?,
                                "household_count": r.get::<_, Option<i64>>(7)?,
                                "present_persons_count": r.get::<_, Option<i64>>(8)?,
                                "remarks": r.get::<_, Option<String>>(9)?,
                                "raw_ocr_text": raw_ocr,
                            });
                            Ok((hid, raw_ocr, st_crop, obj))
                        },
                    )
                    .optional()?
                {
                    overview_house_id = Some(hid);
                    house_raw_ocr = raw_ocr;
                    house_crop = st_crop;
                    house_val = Some(obj);
                }
            } else if sheet_type_str.as_deref() == Some("municipal_overview") {
                if let Some((did, raw_ocr, st_crop, obj)) = conn
                    .query_row(
                        "SELECT id, row_index, ordinal_number, settlement_name, obvod_number,
                                commissioner_name, houses_total, houses_inhabited, houses_uninhabited,
                                household_count, present_persons_count, is_summary_row,
                                raw_ocr_text, stitched_crop_path
                         FROM censusmunicipaloverviewdistrict
                         WHERE census_sheet_id = ?1 AND (stitched_crop_path = ?2 OR row_index = ?3)
                         LIMIT 1",
                        params![cs_id, row_stitched_crop, row_index_val],
                        |r| {
                            let did: i64 = r.get(0)?;
                            let raw_ocr: Option<String> = r.get(12)?;
                            let st_crop: Option<String> = r.get(13)?;
                            let obj = json!({
                                "id": did,
                                "row_index": r.get::<_, Option<i64>>(1)?.unwrap_or(1),
                                "ordinal_number": r.get::<_, Option<String>>(2)?,
                                "settlement_name": r.get::<_, Option<String>>(3)?,
                                "obvod_number": r.get::<_, Option<String>>(4)?,
                                "commissioner_name": r.get::<_, Option<String>>(5)?,
                                "houses_total": r.get::<_, Option<i64>>(6)?,
                                "houses_inhabited": r.get::<_, Option<i64>>(7)?,
                                "houses_uninhabited": r.get::<_, Option<i64>>(8)?,
                                "household_count": r.get::<_, Option<i64>>(9)?,
                                "present_persons_count": r.get::<_, Option<i64>>(10)?,
                                "is_summary_row": r.get::<_, Option<bool>>(11)?.unwrap_or(false),
                                "raw_ocr_text": raw_ocr,
                            });
                            Ok((did, raw_ocr, st_crop, obj))
                        },
                    )
                    .optional()?
                {
                    municipal_district_id = Some(did);
                    dist_raw_ocr = raw_ocr;
                    dist_crop = st_crop;
                    dist_val = Some(obj);
                }
            }
        }
    }

    let is_census = census_sheet_id.filter(|&x| x > 0).is_some()
        || scan_det_type.as_deref() == Some("census")
        || row_rec_type.as_deref() == Some("census")
        || fact_rec_type.as_deref() == Some("census");

    let scan_rows_count: i64 = if let Some(s_id) = scan_id.filter(|&x| x > 0) {
        conn.query_row(
            "SELECT COUNT(id) FROM recordrow WHERE scan_id = ?1",
            params![s_id],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(1)
        .max(1)
    } else {
        1
    };

    let mut row_traces: Vec<FullTrace> = if let Some(r_id) = row_id.filter(|&x| x > 0) {
        query_traces_where(conn, "row_id", r_id)?
    } else {
        Vec::new()
    };

    let scan_traces: Vec<FullTrace> = if let Some(s_id) = scan_id.filter(|&x| x > 0) {
        query_traces_where(conn, "scan_id", s_id)?
    } else {
        Vec::new()
    };

    if row_val.is_some() && !scan_traces.is_empty() {
        let mut row_crops: HashSet<String> = HashSet::new();
        for c in [&row_left_crop, &row_right_crop, &row_stitched_crop] {
            if let Some(s) = c.as_ref().filter(|s| !s.is_empty()) {
                row_crops.insert(s.clone());
            }
        }
        if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&row_cell_crops_json) {
            for item in items {
                if let Some(cu) = item.get("crop_url").and_then(|v| v.as_str()) {
                    if !cu.is_empty() {
                        row_crops.insert(cu.to_string());
                    }
                }
            }
        }
        let mut seen_tids: HashSet<i64> = row_traces.iter().map(|t| t.id).collect();
        for st in &scan_traces {
            if !seen_tids.contains(&st.id) {
                if let Some(ref ip) = st.image_path {
                    if !ip.is_empty() && row_crops.contains(ip) {
                        row_traces.push(st.clone());
                        seen_tids.insert(st.id);
                    }
                }
            }
        }
        row_traces.sort_by_key(|t| t.id);
    }

    let latest_layout_trace = scan_traces
        .iter()
        .rev()
        .find(|t| canonical_stage_group_internal(&t.stage) == "layout");
    let scan_layout_ms = latest_layout_trace.map(|t| t.duration_ms).unwrap_or(0);
    let layout_duration_ms = if scan_layout_ms > 0 {
        ((scan_layout_ms as f64) / (scan_rows_count as f64)).round() as i64
    } else {
        0
    };
    let layout_model = latest_layout_trace
        .map(|t| t.model_or_tool.clone())
        .unwrap_or_else(|| {
            if is_census {
                "opencv:census_sheet_slicer".to_string()
            } else {
                "opencv:angled_quad_row_slicer".to_string()
            }
        });

    let latest_header_ocr = scan_traces.iter().rev().find(|t| t.stage == "ocr_header");
    let scan_ocr_header_ms = latest_header_ocr.map(|t| t.duration_ms).unwrap_or(0);
    let ocr_header_ms = if scan_ocr_header_ms > 0 {
        ((scan_ocr_header_ms as f64) / (scan_rows_count as f64)).round() as i64
    } else {
        0
    };

    let mut dedup_ocr_map: HashMap<(String, String), FullTrace> = HashMap::new();
    let mut dedup_ocr_order: Vec<(String, String)> = Vec::new();
    for t in &row_traces {
        if canonical_stage_group_internal(&t.stage) == "ocr" && t.stage != "ocr_header" {
            let key = (
                t.stage.clone(),
                t.image_path.clone().unwrap_or_default(),
            );
            if !dedup_ocr_map.contains_key(&key) {
                dedup_ocr_order.push(key.clone());
            }
            dedup_ocr_map.insert(key, t.clone());
        }
    }
    let latest_ocr_traces: Vec<FullTrace> = dedup_ocr_order
        .iter()
        .filter_map(|k| dedup_ocr_map.get(k).cloned())
        .collect();

    let ocr_left_ms: i64 = latest_ocr_traces
        .iter()
        .filter(|t| {
            matches!(
                t.stage.as_str(),
                "ocr_left" | "ocr" | "census_overview_ocr"
            )
        })
        .map(|t| t.duration_ms)
        .sum();
    let ocr_right_ms: i64 = latest_ocr_traces
        .iter()
        .filter(|t| t.stage == "ocr_right")
        .map(|t| t.duration_ms)
        .sum();
    let ocr_duration_ms = ocr_left_ms + ocr_right_ms + ocr_header_ms;

    let raw_ocr_model = latest_ocr_traces
        .last()
        .map(|t| t.model_or_tool.as_str())
        .or_else(|| latest_header_ocr.map(|t| t.model_or_tool.as_str()))
        .or_else(|| {
            scan_traces
                .iter()
                .rev()
                .find(|t| canonical_stage_group_internal(&t.stage) == "ocr")
                .map(|t| t.model_or_tool.as_str())
        });
    let ocr_model = defaults.normalize_model_tag(raw_ocr_model, "ocr");

    let latest_decision_trace = row_traces
        .iter()
        .rev()
        .find(|t| canonical_stage_group_internal(&t.stage) == "decision");
    let decision_duration_ms = latest_decision_trace.map(|t| t.duration_ms).unwrap_or(0);
    let decision_model = if let Some(dt) = latest_decision_trace {
        if dt.model_or_tool.starts_with("deterministic:") {
            dt.model_or_tool.clone()
        } else {
            defaults.normalize_model_tag(Some(&dt.model_or_tool), "decision")
        }
    } else if is_census {
        "deterministic:census_gazetteer".to_string()
    } else {
        defaults.normalize_model_tag(None, "decision")
    };

    let latest_extract_trace = row_traces
        .iter()
        .rev()
        .find(|t| canonical_stage_group_internal(&t.stage) == "extract");
    let extract_duration_ms = latest_extract_trace.map(|t| t.duration_ms).unwrap_or(0);
    let raw_extract_model = latest_extract_trace
        .map(|t| t.model_or_tool.as_str())
        .or_else(|| {
            scan_traces
                .iter()
                .rev()
                .find(|t| canonical_stage_group_internal(&t.stage) == "extract")
                .map(|t| t.model_or_tool.as_str())
        });
    let extract_model = defaults.normalize_model_tag(raw_extract_model, "extract");

    let mcp_source: Vec<&FullTrace> = {
        let row_mcp: Vec<&FullTrace> = row_traces
            .iter()
            .filter(|t| canonical_stage_group_internal(&t.stage) == "mcp")
            .collect();
        if row_mcp.is_empty() && row_id.filter(|&x| x > 0).is_none() {
            scan_traces
                .iter()
                .filter(|t| canonical_stage_group_internal(&t.stage) == "mcp")
                .collect()
        } else {
            row_mcp
        }
    };
    let mut dedup_mcp_map: HashMap<String, FullTrace> = HashMap::new();
    let mut dedup_mcp_order: Vec<String> = Vec::new();
    for t in mcp_source {
        let key = format!("{}:{}", t.stage, t.model_or_tool);
        if !dedup_mcp_map.contains_key(&key) {
            dedup_mcp_order.push(key.clone());
        }
        dedup_mcp_map.insert(key, t.clone());
    }
    let latest_mcp_traces: Vec<FullTrace> = dedup_mcp_order
        .iter()
        .filter_map(|k| dedup_mcp_map.get(k).cloned())
        .collect();
    let mcp_duration_ms: i64 = latest_mcp_traces.iter().map(|t| t.duration_ms).sum();
    let mcp_model: Option<String> = latest_mcp_traces.last().map(|t| t.model_or_tool.clone());

    let total_duration_ms = layout_duration_ms
        + ocr_duration_ms
        + decision_duration_ms
        + extract_duration_ms
        + mcp_duration_ms;

    let model_combo = format_model_combo_internal(
        defaults,
        &ocr_model,
        &extract_model,
        Some(&decision_model),
        is_census,
    );

    let mut combined_traces: Vec<FullTrace> = Vec::new();
    if let Some(t) = latest_layout_trace {
        combined_traces.push(t.clone());
    }
    if let Some(t) = latest_header_ocr {
        combined_traces.push(t.clone());
    }
    combined_traces.extend(latest_ocr_traces.iter().cloned());
    if let Some(t) = latest_decision_trace {
        combined_traces.push(t.clone());
    }
    if let Some(t) = latest_extract_trace {
        combined_traces.push(t.clone());
    }
    combined_traces.extend(latest_mcp_traces.iter().cloned());

    let prompt_tokens: i64 = combined_traces.iter().map(|t| t.prompt_tokens).sum();
    let completion_tokens: i64 = combined_traces.iter().map(|t| t.completion_tokens).sum();
    let reasoning_tokens: i64 = combined_traces.iter().map(|t| t.reasoning_tokens).sum();

    let ocr_calls_count =
        latest_ocr_traces.len() + if latest_header_ocr.is_some() { 1 } else { 0 };

    let stage_breakdown = json!({
        "layout_ms": layout_duration_ms,
        "scan_layout_ms": scan_layout_ms,
        "ocr_ms": ocr_duration_ms,
        "ocr_header_ms": ocr_header_ms,
        "scan_ocr_header_ms": scan_ocr_header_ms,
        "ocr_left_ms": ocr_left_ms,
        "ocr_right_ms": ocr_right_ms,
        "ocr_calls_count": ocr_calls_count,
        "decision_ms": decision_duration_ms,
        "extract_ms": extract_duration_ms,
        "mcp_ms": mcp_duration_ms,
        "total_ms": total_duration_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "scan_rows_count": scan_rows_count,
    });

    let non_empty = |opt: &Option<String>| opt.as_ref().filter(|s| !s.is_empty()).cloned();
    let raw_ocr_snapshot = non_empty(&row_raw_ocr_combined)
        .or_else(|| non_empty(&row_raw_ocr_left))
        .or_else(|| non_empty(&house_raw_ocr))
        .or_else(|| non_empty(&dist_raw_ocr))
        .or_else(|| non_empty(&scan_header_ocr));

    let crop_url = non_empty(&row_stitched_crop)
        .or_else(|| non_empty(&house_crop))
        .or_else(|| non_empty(&dist_crop))
        .or_else(|| non_empty(&row_left_crop))
        .or_else(|| non_empty(&scan_header_crop));

    let scan_snap = scan_val.unwrap_or_else(|| {
        json!({
            "id": scan_id,
            "filename": null,
            "page_side": null,
            "original_path": null,
            "overlay_path": null,
            "header_crop_path": null,
            "detected_year": null,
            "detected_type": null,
        })
    });

    let mut extracted_snapshot = json!({
        "target_type": target_type,
        "is_census": is_census,
        "scan": scan_snap,
        "census_sheet": sheet_val,
        "row": row_val,
    });
    if let Some(obj) = extracted_snapshot.as_object_mut() {
        if let Some(fv) = fact_val {
            obj.insert("person_fact".to_string(), fv);
        }
        if let Some(hv) = house_val {
            obj.insert("overview_house".to_string(), hv);
        }
        if let Some(dv) = dist_val {
            obj.insert("municipal_district".to_string(), dv);
        }
    }

    let traces_snapshot: Vec<Value> = combined_traces
        .iter()
        .map(serialize_trace_for_snapshot)
        .collect();

    Ok(json!({
        "target_type": target_type,
        "row_id": row_id,
        "scan_id": scan_id,
        "census_sheet_id": census_sheet_id,
        "person_fact_id": person_fact_id,
        "overview_house_id": overview_house_id,
        "municipal_district_id": municipal_district_id,
        "model_combo": model_combo,
        "layout_model": layout_model,
        "ocr_model": ocr_model,
        "decision_model": decision_model,
        "extract_model": extract_model,
        "mcp_model": mcp_model,
        "total_duration_ms": total_duration_ms,
        "layout_duration_ms": layout_duration_ms,
        "ocr_duration_ms": ocr_duration_ms,
        "decision_duration_ms": decision_duration_ms,
        "extract_duration_ms": extract_duration_ms,
        "mcp_duration_ms": mcp_duration_ms,
        "raw_ocr_snapshot": raw_ocr_snapshot,
        "extracted_snapshot": extracted_snapshot,
        "stage_breakdown": stage_breakdown,
        "traces_snapshot": traces_snapshot,
        "crop_url": crop_url,
    }))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (db_path, data_dir, now_iso, ai_runtime_config_json, default_ocr, default_decision, default_extract, default_layout, known_local_models, verdict, comment=None, expected_correction=None, target_type="row".to_string(), row_id=None, scan_id=None, census_sheet_id=None, person_fact_id=None, overview_house_id=None, municipal_district_id=None))]
pub fn upsert_ocr_feedback_rs(
    py: Python<'_>,
    db_path: String,
    data_dir: String,
    now_iso: String,
    ai_runtime_config_json: String,
    default_ocr: String,
    default_decision: String,
    default_extract: String,
    default_layout: String,
    known_local_models: Vec<String>,
    verdict: String,
    comment: Option<String>,
    expected_correction: Option<String>,
    target_type: String,
    row_id: Option<i64>,
    scan_id: Option<i64>,
    census_sheet_id: Option<i64>,
    person_fact_id: Option<i64>,
    overview_house_id: Option<i64>,
    municipal_district_id: Option<i64>,
) -> PyResult<String> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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

        let v_low = verdict.trim().to_ascii_lowercase();
        let norm_verdict = if matches!(
            v_low.as_str(),
            "incorrect" | "wrong" | "bad" | "false" | "0" | "error"
        ) {
            "incorrect".to_string()
        } else {
            "correct".to_string()
        };

        let ctx = collect_row_execution_context_conn(
            &conn,
            &defaults,
            target_type,
            row_id,
            scan_id,
            census_sheet_id,
            person_fact_id,
            overview_house_id,
            municipal_district_id,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let resolved_target = ctx
            .get("target_type")
            .and_then(|v| v.as_str())
            .unwrap_or("row")
            .to_string();
        let resolved_row_id = ctx.get("row_id").and_then(|v| v.as_i64());
        let resolved_scan_id = ctx.get("scan_id").and_then(|v| v.as_i64());
        let resolved_sheet_id = ctx.get("census_sheet_id").and_then(|v| v.as_i64());
        let resolved_fact_id = ctx.get("person_fact_id").and_then(|v| v.as_i64());
        let resolved_house_id = ctx.get("overview_house_id").and_then(|v| v.as_i64());
        let resolved_dist_id = ctx.get("municipal_district_id").and_then(|v| v.as_i64());
        let model_combo = ctx
            .get("model_combo")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let find_existing = |conn: &Connection,
                             col: &str,
                             id_val: i64,
                             combo: &str|
         -> Result<Option<(i64, Option<String>, Option<String>, Option<String>)>, rusqlite::Error> {
            let sql = format!(
                "SELECT id, comment, expected_correction, created_at
                 FROM ocrfeedback
                 WHERE {} = ?1 AND model_combo = ?2
                 LIMIT 1",
                col
            );
            conn.query_row(&sql, params![id_val, combo], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
            .optional()
        };

        let mut existing: Option<(i64, Option<String>, Option<String>, Option<String>)> = None;
        if let Some(fid) = resolved_fact_id.filter(|&x| x > 0) {
            existing = find_existing(&conn, "person_fact_id", fid, &model_combo)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        if existing.is_none() {
            if let Some(hid) = resolved_house_id.filter(|&x| x > 0) {
                existing = find_existing(&conn, "overview_house_id", hid, &model_combo)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        if existing.is_none() {
            if let Some(did) = resolved_dist_id.filter(|&x| x > 0) {
                existing = find_existing(&conn, "municipal_district_id", did, &model_combo)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }
        if existing.is_none() {
            if let Some(rid) = resolved_row_id.filter(|&x| x > 0) {
                existing = find_existing(&conn, "row_id", rid, &model_combo)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }

        let final_comment = match comment {
            Some(c) => {
                let t = c.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
            None => existing.as_ref().and_then(|e| e.1.clone()),
        };

        let final_expected = match expected_correction {
            Some(ec) => {
                let t = ec.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
            None => existing.as_ref().and_then(|e| e.2.clone()),
        };

        let layout_model = ctx
            .get("layout_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let ocr_model = ctx
            .get("ocr_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let decision_model = ctx
            .get("decision_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let extract_model = ctx
            .get("extract_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let mcp_model = ctx
            .get("mcp_model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let total_duration_ms = ctx
            .get("total_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let layout_duration_ms = ctx
            .get("layout_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let ocr_duration_ms = ctx
            .get("ocr_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let decision_duration_ms = ctx
            .get("decision_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let extract_duration_ms = ctx
            .get("extract_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let mcp_duration_ms = ctx
            .get("mcp_duration_ms")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let raw_ocr_snapshot = ctx
            .get("raw_ocr_snapshot")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let crop_url = ctx
            .get("crop_url")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let extracted_snapshot_json = serde_json::to_string(
            ctx.get("extracted_snapshot").unwrap_or(&json!({})),
        )
        .unwrap_or_else(|_| "{}".to_string());
        let stage_breakdown_json =
            serde_json::to_string(ctx.get("stage_breakdown").unwrap_or(&json!({})))
                .unwrap_or_else(|_| "{}".to_string());
        let traces_snapshot_json =
            serde_json::to_string(ctx.get("traces_snapshot").unwrap_or(&json!([])))
                .unwrap_or_else(|_| "[]".to_string());

        let db_ts = now_iso.replace('T', " ");
        let saved_id: i64;
        let created_at_val: Option<String>;

        {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if let Some((ex_id, _, _, ex_created)) = existing {
                tx.execute(
                    "UPDATE ocrfeedback SET
                        target_type = ?1, row_id = ?2, scan_id = ?3, census_sheet_id = ?4,
                        person_fact_id = ?5, overview_house_id = ?6, municipal_district_id = ?7,
                        verdict = ?8, comment = ?9, expected_correction = ?10, model_combo = ?11,
                        layout_model = ?12, ocr_model = ?13, decision_model = ?14, extract_model = ?15,
                        mcp_model = ?16, total_duration_ms = ?17, layout_duration_ms = ?18,
                        ocr_duration_ms = ?19, decision_duration_ms = ?20, extract_duration_ms = ?21,
                        mcp_duration_ms = ?22, raw_ocr_snapshot = ?23, extracted_snapshot_json = ?24,
                        stage_breakdown_json = ?25, traces_snapshot_json = ?26, crop_url = ?27,
                        updated_at = ?28
                     WHERE id = ?29",
                    params![
                        resolved_target,
                        resolved_row_id,
                        resolved_scan_id,
                        resolved_sheet_id,
                        resolved_fact_id,
                        resolved_house_id,
                        resolved_dist_id,
                        norm_verdict,
                        final_comment,
                        final_expected,
                        model_combo,
                        layout_model,
                        ocr_model,
                        decision_model,
                        extract_model,
                        mcp_model,
                        total_duration_ms,
                        layout_duration_ms,
                        ocr_duration_ms,
                        decision_duration_ms,
                        extract_duration_ms,
                        mcp_duration_ms,
                        raw_ocr_snapshot,
                        extracted_snapshot_json,
                        stage_breakdown_json,
                        traces_snapshot_json,
                        crop_url,
                        db_ts,
                        ex_id,
                    ],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                saved_id = ex_id;
                created_at_val = ex_created.or_else(|| Some(now_iso.clone()));
            } else {
                tx.execute(
                    "INSERT INTO ocrfeedback (
                        target_type, row_id, scan_id, census_sheet_id, person_fact_id,
                        overview_house_id, municipal_district_id, verdict, comment,
                        expected_correction, model_combo, layout_model, ocr_model,
                        decision_model, extract_model, mcp_model, total_duration_ms,
                        layout_duration_ms, ocr_duration_ms, decision_duration_ms,
                        extract_duration_ms, mcp_duration_ms, raw_ocr_snapshot,
                        extracted_snapshot_json, stage_breakdown_json, traces_snapshot_json,
                        crop_url, created_at, updated_at
                    ) VALUES (
                        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                        ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29
                    )",
                    params![
                        resolved_target,
                        resolved_row_id,
                        resolved_scan_id,
                        resolved_sheet_id,
                        resolved_fact_id,
                        resolved_house_id,
                        resolved_dist_id,
                        norm_verdict,
                        final_comment,
                        final_expected,
                        model_combo,
                        layout_model,
                        ocr_model,
                        decision_model,
                        extract_model,
                        mcp_model,
                        total_duration_ms,
                        layout_duration_ms,
                        ocr_duration_ms,
                        decision_duration_ms,
                        extract_duration_ms,
                        mcp_duration_ms,
                        raw_ocr_snapshot,
                        extracted_snapshot_json,
                        stage_breakdown_json,
                        traces_snapshot_json,
                        crop_url,
                        db_ts,
                        db_ts,
                    ],
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                saved_id = tx.last_insert_rowid();
                created_at_val = Some(now_iso.clone());
            }
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        let saved_row = FeedbackRow {
            id: saved_id,
            target_type: resolved_target,
            row_id: resolved_row_id,
            scan_id: resolved_scan_id,
            census_sheet_id: resolved_sheet_id,
            person_fact_id: resolved_fact_id,
            overview_house_id: resolved_house_id,
            municipal_district_id: resolved_dist_id,
            verdict: norm_verdict,
            comment: final_comment,
            expected_correction: final_expected,
            model_combo,
            layout_model,
            ocr_model,
            decision_model,
            extract_model,
            mcp_model,
            total_duration_ms,
            layout_duration_ms,
            ocr_duration_ms,
            decision_duration_ms,
            extract_duration_ms,
            mcp_duration_ms,
            raw_ocr_snapshot,
            extracted_snapshot_json,
            stage_breakdown_json,
            traces_snapshot_json,
            crop_url,
            created_at: created_at_val,
            updated_at: Some(now_iso.clone()),
        };

        let serialized_saved = serialize_feedback_row(&saved_row, true);
        let stats = compute_feedback_statistics_from_conn(&conn, &defaults, &data_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let sql_all = format!(
            "SELECT {} FROM ocrfeedback ORDER BY verdict DESC, updated_at DESC, id DESC",
            FEEDBACK_SELECT_COLS
        );
        let mut stmt = conn
            .prepare(&sql_all)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let all_fb_rows: Vec<FeedbackRow> = stmt
            .query_map([], read_feedback_row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let all_serialized: Vec<Value> = all_fb_rows
            .iter()
            .map(|fb| serialize_feedback_row(fb, true))
            .collect();

        let ai_cfg: Value =
            serde_json::from_str(&ai_runtime_config_json).unwrap_or_else(|_| json!({}));
        let full_payload = json!({
            "generated_at": now_iso,
            "ai_runtime_config": ai_cfg,
            "statistics": stats,
            "feedback_count": all_serialized.len(),
            "feedback_items": all_serialized,
        });
        let full_md = format_markdown_report_internal(&full_payload);
        let full_json =
            serde_json::to_string_pretty(&full_payload).unwrap_or_else(|_| "{}".to_string());
        let dir_path = Path::new(&data_dir);
        let _ = fs::create_dir_all(dir_path);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.md"), &full_md);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.json"), &full_json);

        let report_files = full_payload
            .get("statistics")
            .and_then(|s| s.get("summary"))
            .and_then(|s| s.get("report_files"))
            .cloned()
            .unwrap_or_else(|| json!({}));

        let out = json!({
            "status": "ok",
            "feedback": serialized_saved,
            "stats": full_payload.get("statistics"),
            "report_markdown": full_md,
            "report_files": report_files,
        });
        serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn delete_ocr_feedback_rs(
    py: Python<'_>,
    db_path: String,
    data_dir: String,
    now_iso: String,
    ai_runtime_config_json: String,
    default_ocr: String,
    default_decision: String,
    default_extract: String,
    default_layout: String,
    known_local_models: Vec<String>,
    feedback_id: i64,
) -> PyResult<Option<String>> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let deleted = {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let cnt = tx
                .execute("DELETE FROM ocrfeedback WHERE id = ?1", params![feedback_id])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            cnt
        };
        if deleted == 0 {
            return Ok(None);
        }

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

        let sql_all = format!(
            "SELECT {} FROM ocrfeedback ORDER BY verdict DESC, updated_at DESC, id DESC",
            FEEDBACK_SELECT_COLS
        );
        let mut stmt = conn
            .prepare(&sql_all)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let all_fb_rows: Vec<FeedbackRow> = stmt
            .query_map([], read_feedback_row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let all_serialized: Vec<Value> = all_fb_rows
            .iter()
            .map(|fb| serialize_feedback_row(fb, true))
            .collect();

        let ai_cfg: Value =
            serde_json::from_str(&ai_runtime_config_json).unwrap_or_else(|_| json!({}));
        let full_payload = json!({
            "generated_at": now_iso,
            "ai_runtime_config": ai_cfg,
            "statistics": stats,
            "feedback_count": all_serialized.len(),
            "feedback_items": all_serialized,
        });
        let full_md = format_markdown_report_internal(&full_payload);
        let full_json =
            serde_json::to_string_pretty(&full_payload).unwrap_or_else(|_| "{}".to_string());
        let dir_path = Path::new(&data_dir);
        let _ = fs::create_dir_all(dir_path);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.md"), &full_md);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.json"), &full_json);

        let report_files = full_payload
            .get("statistics")
            .and_then(|s| s.get("summary"))
            .and_then(|s| s.get("report_files"))
            .cloned()
            .unwrap_or_else(|| json!({}));

        let out = json!({
            "status": "deleted",
            "deleted_id": feedback_id,
            "stats": full_payload.get("statistics"),
            "report_markdown": full_md,
            "report_files": report_files,
        });
        let s = serde_json::to_string(&out).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Some(s))
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (db_path, data_dir, now_iso, ai_runtime_config_json, default_ocr, default_decision, default_extract, default_layout, known_local_models, census_sheet_id=None, scan_ids=None, row_ids=None, person_fact_id=None, overview_house_id=None, municipal_district_id=None, persist_files=true))]
pub fn reset_ocr_feedback_rs(
    py: Python<'_>,
    db_path: String,
    data_dir: String,
    now_iso: String,
    ai_runtime_config_json: String,
    default_ocr: String,
    default_decision: String,
    default_extract: String,
    default_layout: String,
    known_local_models: Vec<String>,
    census_sheet_id: Option<i64>,
    scan_ids: Option<Vec<i64>>,
    row_ids: Option<Vec<i64>>,
    person_fact_id: Option<i64>,
    overview_house_id: Option<i64>,
    municipal_district_id: Option<i64>,
    persist_files: bool,
) -> PyResult<i64> {
    py.allow_threads(move || {
        let mut conn =
            open_sqlite(&db_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let total_fb_cnt: i64 = conn
            .query_row("SELECT COUNT(id) FROM ocrfeedback", [], |r| r.get(0))
            .unwrap_or(0);
        if total_fb_cnt == 0 {
            return Ok(0);
        }

        let mut target_sheet_ids: HashSet<i64> = HashSet::new();
        if let Some(sid) = census_sheet_id.filter(|&x| x > 0) {
            target_sheet_ids.insert(sid);
        }
        let has_explicit_row = row_ids.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
        let mut target_scan_ids: HashSet<i64> =
            scan_ids.unwrap_or_default().into_iter().filter(|&x| x > 0).collect();
        let mut target_row_ids: HashSet<i64> =
            row_ids.unwrap_or_default().into_iter().filter(|&x| x > 0).collect();
        let mut target_fact_ids: HashSet<i64> = HashSet::new();
        if let Some(fid) = person_fact_id.filter(|&x| x > 0) {
            target_fact_ids.insert(fid);
        }
        let mut target_house_ids: HashSet<i64> = HashSet::new();
        if let Some(hid) = overview_house_id.filter(|&x| x > 0) {
            target_house_ids.insert(hid);
        }
        let mut target_dist_ids: HashSet<i64> = HashSet::new();
        if let Some(did) = municipal_district_id.filter(|&x| x > 0) {
            target_dist_ids.insert(did);
        }

        if !target_sheet_ids.is_empty() {
            for &sid in &target_sheet_ids {
                if let Ok(Some((f_sc, b_sc))) = conn
                    .query_row(
                        "SELECT front_scan_id, back_scan_id FROM censussheet WHERE id = ?1",
                        params![sid],
                        |r| Ok((r.get::<_, Option<i64>>(0)?, r.get::<_, Option<i64>>(1)?)),
                    )
                    .optional()
                {
                    if let Some(f) = f_sc.filter(|&x| x > 0) {
                        target_scan_ids.insert(f);
                    }
                    if let Some(b) = b_sc.filter(|&x| x > 0) {
                        target_scan_ids.insert(b);
                    }
                }
                if let Ok(mut st) =
                    conn.prepare("SELECT id FROM scanimage WHERE census_sheet_id = ?1")
                {
                    if let Ok(rows) = st.query_map(params![sid], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_scan_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) =
                    conn.prepare("SELECT id FROM personfact WHERE census_sheet_id = ?1")
                {
                    if let Ok(rows) = st.query_map(params![sid], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_fact_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) =
                    conn.prepare("SELECT id FROM censusoverviewhouse WHERE census_sheet_id = ?1")
                {
                    if let Ok(rows) = st.query_map(params![sid], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_house_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) = conn.prepare(
                    "SELECT id FROM censusmunicipaloverviewdistrict WHERE census_sheet_id = ?1",
                ) {
                    if let Ok(rows) = st.query_map(params![sid], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_dist_ids.insert(id_res);
                        }
                    }
                }
            }
        }

        if !target_scan_ids.is_empty() && !has_explicit_row {
            for &sc_id in &target_scan_ids {
                if let Ok(mut st) = conn.prepare("SELECT id FROM recordrow WHERE scan_id = ?1") {
                    if let Ok(rows) = st.query_map(params![sc_id], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_row_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) = conn.prepare("SELECT id FROM personfact WHERE scan_id = ?1") {
                    if let Ok(rows) = st.query_map(params![sc_id], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_fact_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) =
                    conn.prepare("SELECT id FROM censusoverviewhouse WHERE scan_id = ?1")
                {
                    if let Ok(rows) = st.query_map(params![sc_id], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_house_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(mut st) = conn
                    .prepare("SELECT id FROM censusmunicipaloverviewdistrict WHERE scan_id = ?1")
                {
                    if let Ok(rows) = st.query_map(params![sc_id], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_dist_ids.insert(id_res);
                        }
                    }
                }
            }
        }

        if !target_row_ids.is_empty() {
            for &r_id in &target_row_ids {
                if let Ok(mut st) = conn.prepare("SELECT id FROM personfact WHERE row_id = ?1") {
                    if let Ok(rows) = st.query_map(params![r_id], |r| r.get::<_, i64>(0)) {
                        for id_res in rows.flatten() {
                            target_fact_ids.insert(id_res);
                        }
                    }
                }
                if let Ok(Some((sc_id, r_idx, st_crop))) = conn
                    .query_row(
                        "SELECT scan_id, row_index, stitched_crop_path FROM recordrow WHERE id = ?1",
                        params![r_id],
                        |r| {
                            Ok((
                                r.get::<_, Option<i64>>(0)?,
                                r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                                r.get::<_, Option<String>>(2)?,
                            ))
                        },
                    )
                    .optional()
                {
                    if let Some(sid) = sc_id.filter(|&x| x > 0) {
                        if let Ok(mut st) = conn.prepare(
                            "SELECT id, row_index, stitched_crop_path FROM censusoverviewhouse WHERE scan_id = ?1",
                        ) {
                            if let Ok(rows) = st.query_map(params![sid], |r| {
                                Ok((
                                    r.get::<_, i64>(0)?,
                                    r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                                    r.get::<_, Option<String>>(2)?,
                                ))
                            }) {
                                for (hid, h_idx, h_crop) in rows.flatten() {
                                    if (st_crop.as_ref().filter(|s| !s.is_empty()).is_some()
                                        && st_crop == h_crop)
                                        || h_idx == r_idx
                                    {
                                        target_house_ids.insert(hid);
                                    }
                                }
                            }
                        }
                        if let Ok(mut st) = conn.prepare(
                            "SELECT id, row_index, stitched_crop_path FROM censusmunicipaloverviewdistrict WHERE scan_id = ?1",
                        ) {
                            if let Ok(rows) = st.query_map(params![sid], |r| {
                                Ok((
                                    r.get::<_, i64>(0)?,
                                    r.get::<_, Option<i64>>(1)?.unwrap_or(0),
                                    r.get::<_, Option<String>>(2)?,
                                ))
                            }) {
                                for (did, d_idx, d_crop) in rows.flatten() {
                                    if (st_crop.as_ref().filter(|s| !s.is_empty()).is_some()
                                        && st_crop == d_crop)
                                        || d_idx == r_idx
                                    {
                                        target_dist_ids.insert(did);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let candidates: Vec<(i64, Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<i64>, Option<i64>)> = {
            let mut fb_stmt = conn
                .prepare(
                    "SELECT id, census_sheet_id, scan_id, row_id, person_fact_id, overview_house_id, municipal_district_id
                     FROM ocrfeedback",
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = fb_stmt
                .query_map([], |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            rows
        };

        let mut ids_to_delete: Vec<i64> = Vec::new();
        for (fb_id, cs_id, sc_id, r_id, pf_id, oh_id, md_id) in candidates {
            let mut del = false;
            if let Some(cid) = cs_id {
                if target_sheet_ids.contains(&cid) {
                    del = true;
                }
            }
            if !del && !has_explicit_row {
                if let Some(sid) = sc_id {
                    if target_scan_ids.contains(&sid) {
                        del = true;
                    }
                }
            }
            if !del {
                if let Some(rid) = r_id {
                    if target_row_ids.contains(&rid) {
                        del = true;
                    }
                }
            }
            if !del {
                if let Some(fid) = pf_id {
                    if target_fact_ids.contains(&fid) {
                        del = true;
                    }
                }
            }
            if !del {
                if let Some(hid) = oh_id {
                    if target_house_ids.contains(&hid) {
                        del = true;
                    }
                }
            }
            if !del {
                if let Some(did) = md_id {
                    if target_dist_ids.contains(&did) {
                        del = true;
                    }
                }
            }
            if del {
                ids_to_delete.push(fb_id);
            }
        }

        if ids_to_delete.is_empty() {
            return Ok(0);
        }

        let deleted_count = ids_to_delete.len() as i64;
        {
            let tx = conn
                .transaction()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            for chunk in ids_to_delete.chunks(450) {
                let ph: Vec<String> = (0..chunk.len()).map(|i| format!("?{}", i + 1)).collect();
                let sql = format!("DELETE FROM ocrfeedback WHERE id IN ({})", ph.join(", "));
                tx.execute(&sql, params_from_iter(chunk.iter()))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
            tx.commit()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        if !persist_files {
            return Ok(deleted_count);
        }

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

        let sql_all = format!(
            "SELECT {} FROM ocrfeedback ORDER BY verdict DESC, updated_at DESC, id DESC",
            FEEDBACK_SELECT_COLS
        );
        let mut stmt = conn
            .prepare(&sql_all)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let all_fb_rows: Vec<FeedbackRow> = stmt
            .query_map([], read_feedback_row)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let all_serialized: Vec<Value> = all_fb_rows
            .iter()
            .map(|fb| serialize_feedback_row(fb, true))
            .collect();
        let ai_cfg: Value =
            serde_json::from_str(&ai_runtime_config_json).unwrap_or_else(|_| json!({}));
        let full_payload = json!({
            "generated_at": now_iso,
            "ai_runtime_config": ai_cfg,
            "statistics": stats,
            "feedback_count": all_serialized.len(),
            "feedback_items": all_serialized,
        });
        let full_md = format_markdown_report_internal(&full_payload);
        let full_json =
            serde_json::to_string_pretty(&full_payload).unwrap_or_else(|_| "{}".to_string());
        let dir_path = Path::new(&data_dir);
        let _ = fs::create_dir_all(dir_path);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.md"), &full_md);
        let _ = fs::write(dir_path.join("ocr_feedback_debug_report.json"), &full_json);

        Ok(deleted_count)
    })
}

