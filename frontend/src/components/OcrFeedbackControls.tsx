import React, { useEffect, useRef, useState } from "react";
import { Check, MessageSquare, ThumbsDown, ThumbsUp, X } from "lucide-react";

export interface OcrFeedbackItem {
  id: number;
  target_type: string;
  row_id: number | null;
  scan_id: number | null;
  census_sheet_id: number | null;
  person_fact_id: number | null;
  overview_house_id: number | null;
  municipal_district_id: number | null;
  verdict: "correct" | "incorrect";
  comment: string | null;
  expected_correction?: string | null;
  display_label?: string;
  model_combo: string;
  layout_model?: string | null;
  ocr_model?: string | null;
  decision_model?: string | null;
  extract_model?: string | null;
  mcp_model?: string | null;
  total_duration_ms: number;
  layout_duration_ms: number;
  ocr_duration_ms: number;
  decision_duration_ms: number;
  extract_duration_ms: number;
  mcp_duration_ms: number;
  raw_ocr_snapshot?: string | null;
  extracted_snapshot?: Record<string, any>;
  stage_breakdown?: Record<string, any>;
  traces_snapshot?: Array<Record<string, any>>;
  crop_url?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

interface Props {
  lang: "sk" | "en";
  targetType?:
    | "row"
    | "person_fact"
    | "overview_house"
    | "municipal_district"
    | "scan"
    | "census_sheet";
  rowId?: number | null;
  scanId?: number | null;
  censusSheetId?: number | null;
  personFactId?: number | null;
  overviewHouseId?: number | null;
  municipalDistrictId?: number | null;
  initialFeedback?: OcrFeedbackItem | null;
  resetVersion?: number;
  compact?: boolean;
  onFeedbackSaved?: (feedback: OcrFeedbackItem | null) => void;
}

function OcrFeedbackControlsInner({
  lang,
  targetType = "row",
  rowId = null,
  scanId = null,
  censusSheetId = null,
  personFactId = null,
  overviewHouseId = null,
  municipalDistrictId = null,
  initialFeedback = null,
  resetVersion = 0,
  compact = true,
  onFeedbackSaved,
}: Props) {
  const [feedback, setFeedback] = useState<OcrFeedbackItem | null>(initialFeedback);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [showNoteInput, setShowNoteInput] = useState<boolean>(false);
  const [noteText, setNoteText] = useState<string>(initialFeedback?.comment || "");
  const onSavedRef = useRef(onFeedbackSaved);
  onSavedRef.current = onFeedbackSaved;

  useEffect(() => {
    setFeedback(initialFeedback || null);
    setNoteText(initialFeedback?.comment || "");
    if (!initialFeedback) {
      setShowNoteInput(false);
    }
  }, [
    initialFeedback,
    initialFeedback?.id,
    initialFeedback?.verdict,
    initialFeedback?.comment,
    targetType,
    rowId,
    scanId,
    censusSheetId,
    personFactId,
    overviewHouseId,
    municipalDistrictId,
  ]);

  useEffect(() => {
    if (resetVersion > 0) {
      setFeedback(initialFeedback || null);
      setNoteText(initialFeedback?.comment || "");
      if (!initialFeedback) {
        setShowNoteInput(false);
      }
    }
  }, [resetVersion]);

  const submitVerdict = async (
    verdict: "correct" | "incorrect",
    commentOverride?: string
  ) => {
    setSubmitting(true);
    try {
      const commentVal =
        commentOverride !== undefined ? commentOverride : noteText;
      const res = await fetch("/api/debug/feedback?light=1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          verdict,
          comment: commentVal.trim() || null,
          target_type: targetType,
          row_id: rowId,
          scan_id: scanId,
          census_sheet_id: censusSheetId,
          person_fact_id: personFactId,
          overview_house_id: overviewHouseId,
          municipal_district_id: municipalDistrictId,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const saved: OcrFeedbackItem = data.feedback;
        setFeedback(saved);
        setNoteText(saved.comment || "");
        if (verdict === "incorrect" && commentOverride === undefined && !saved.comment) {
          setShowNoteInput(true);
        } else if (commentOverride !== undefined) {
          setShowNoteInput(false);
        }
        onSavedRef.current?.(saved);
      }
    } catch {
      // ignore network error
    } finally {
      setSubmitting(false);
    }
  };

  const handleClearFeedback = async () => {
    if (!feedback?.id) return;
    setSubmitting(true);
    try {
      const res = await fetch(`/api/debug/feedback/${feedback.id}?light=1`, {
        method: "DELETE",
      });
      if (res.ok) {
        setFeedback(null);
        setNoteText("");
        setShowNoteInput(false);
        onSavedRef.current?.(null);
      }
    } catch {
      // ignore
    } finally {
      setSubmitting(false);
    }
  };

  const isCorrect = feedback?.verdict === "correct";
  const isIncorrect = feedback?.verdict === "incorrect";

  return (
    <div
      onClick={(e) => e.stopPropagation()}
      className="inline-flex flex-col items-end gap-1"
      data-testid="ocr-feedback-controls"
    >
      <div className="inline-flex items-center gap-1">
        <button
          type="button"
          disabled={submitting}
          data-testid="ocr-feedback-correct-btn"
          onClick={() => void submitVerdict("correct")}
          title={
            lang === "sk"
              ? `Označiť OCR ako správne${
                  feedback?.model_combo ? ` (${feedback.model_combo})` : ""
                }`
              : `Mark OCR as correct${
                  feedback?.model_combo ? ` (${feedback.model_combo})` : ""
                }`
          }
          className={`px-2 py-1 min-h-[28px] rounded-md text-xs font-semibold inline-flex items-center gap-1 transition cursor-pointer border ${
            isCorrect
              ? "bg-emerald-600 text-white border-emerald-700 shadow-2xs"
              : "bg-stone-100 hover:bg-emerald-100 text-stone-700 hover:text-emerald-900 border-stone-300"
          }`}
        >
          <ThumbsUp className="w-3 h-3 shrink-0" />
          <span>
            {compact
              ? lang === "sk"
                ? "OCR ✓"
                : "OCR ✓"
              : lang === "sk"
              ? "Správne OCR"
              : "Correct OCR"}
          </span>
        </button>

        <button
          type="button"
          disabled={submitting}
          data-testid="ocr-feedback-incorrect-btn"
          onClick={() => void submitVerdict("incorrect")}
          title={
            lang === "sk"
              ? `Označiť OCR ako chybné (pre Debug a LLM report)${
                  feedback?.comment ? `: "${feedback.comment}"` : ""
                }`
              : `Mark OCR as incorrect (for Debug & LLM report)${
                  feedback?.comment ? `: "${feedback.comment}"` : ""
                }`
          }
          className={`px-2 py-1 min-h-[28px] rounded-md text-xs font-semibold inline-flex items-center gap-1 transition cursor-pointer border ${
            isIncorrect
              ? "bg-red-600 text-white border-red-700 shadow-2xs"
              : "bg-stone-100 hover:bg-red-100 text-stone-700 hover:text-red-900 border-stone-300"
          }`}
        >
          <ThumbsDown className="w-3 h-3 shrink-0" />
          <span>
            {compact
              ? lang === "sk"
                ? "OCR ✗"
                : "OCR ✗"
              : lang === "sk"
              ? "Chybné OCR"
              : "Incorrect OCR"}
          </span>
        </button>

        {feedback && (
          <>
            <button
              type="button"
              onClick={() => setShowNoteInput((v) => !v)}
              title={
                feedback.comment
                  ? `${lang === "sk" ? "Poznámka:" : "Note:"} ${feedback.comment}`
                  : lang === "sk"
                  ? "Pridať poznámku k OCR pre LLM"
                  : "Add OCR debug note for LLM"
              }
              className={`p-1 min-h-[28px] min-w-[28px] rounded-md border inline-flex items-center justify-center text-xs cursor-pointer ${
                feedback.comment || showNoteInput
                  ? "bg-amber-100 text-amber-900 border-amber-400"
                  : "bg-stone-100 hover:bg-stone-200 text-stone-600 border-stone-300"
              }`}
            >
              <MessageSquare className="w-3 h-3" />
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={() => void handleClearFeedback()}
              title={
                lang === "sk" ? "Zrušiť hodnotenie OCR" : "Clear OCR feedback"
              }
              className="p-1 min-h-[28px] min-w-[24px] rounded-md bg-stone-100 hover:bg-stone-200 text-stone-500 hover:text-stone-800 border border-stone-300 inline-flex items-center justify-center text-xs cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </>
        )}
      </div>

      {showNoteInput && (
        <div className="flex items-center gap-1 bg-white p-1.5 rounded-lg border border-amber-400 shadow-sm text-xs max-w-xs sm:max-w-sm">
          <input
            type="text"
            value={noteText}
            onChange={(e) => setNoteText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void submitVerdict(feedback?.verdict || "incorrect", noteText);
              } else if (e.key === "Escape") {
                setShowNoteInput(false);
              }
            }}
            placeholder={
              lang === "sk"
                ? "Poznámka pre LLM (napr. správny text)…"
                : "Note for LLM (e.g. expected text)…"
            }
            className="flex-1 min-w-[150px] px-2 py-1 rounded border border-stone-300 text-stone-900 text-xs focus:border-amber-500 focus:outline-none"
          />
          <button
            type="button"
            disabled={submitting}
            onClick={() =>
              void submitVerdict(feedback?.verdict || "incorrect", noteText)
            }
            className="px-2 py-1 rounded bg-amber-600 hover:bg-amber-700 text-white font-semibold inline-flex items-center gap-1 cursor-pointer shrink-0"
          >
            <Check className="w-3 h-3" />
            <span>{lang === "sk" ? "Uložiť" : "Save"}</span>
          </button>
        </div>
      )}
    </div>
  );
}

const OcrFeedbackControls = React.memo(
  OcrFeedbackControlsInner,
  (prev, next) =>
    prev.lang === next.lang &&
    prev.targetType === next.targetType &&
    prev.rowId === next.rowId &&
    prev.scanId === next.scanId &&
    prev.censusSheetId === next.censusSheetId &&
    prev.personFactId === next.personFactId &&
    prev.overviewHouseId === next.overviewHouseId &&
    prev.municipalDistrictId === next.municipalDistrictId &&
    prev.resetVersion === next.resetVersion &&
    prev.compact === next.compact &&
    (prev.initialFeedback?.id ?? null) === (next.initialFeedback?.id ?? null) &&
    (prev.initialFeedback?.verdict ?? null) === (next.initialFeedback?.verdict ?? null) &&
    (prev.initialFeedback?.comment ?? null) === (next.initialFeedback?.comment ?? null)
);

export default OcrFeedbackControls;
