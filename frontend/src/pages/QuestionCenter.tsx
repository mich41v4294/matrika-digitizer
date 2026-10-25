import React, { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  SkipForward,
  Sparkles,
  Eye,
  Volume2,
  Edit3,
  ExternalLink,
  FileText,
  Info,
  RefreshCw,
} from "lucide-react";

interface RowContext {
  entry_number?: string | null;
  child_name?: string | null;
  date_text?: string | null;
  parents_text?: string | null;
  village_text?: string | null;
  godparents_text?: string | null;
  officiant_text?: string | null;
  raw_clean?: string | null;
}

type NeighborCropItem =
  | string
  | { row_index?: number; crop_url?: string; crop_path?: string; confirmed_text?: string };

interface QuestionItem {
  id: number;
  row_id: number;
  scan_id: number;
  scan_filename: string;
  detected_year: string | null;
  row_index: number;
  entry_number?: string | null;
  field_name: string;
  question_sk: string;
  description_sk?: string;
  question_en: string;
  row_context?: RowContext | null;
  crop_url: string | null;
  stitched_row_url: string | null;
  neighbor_crops: NeighborCropItem[];
  candidates: string[];
  raw_ocr_snippet: string | null;
  status: string;
  selected_answer: string | null;
}

interface Props {
  lang?: "sk" | "en";
  refreshStatus?: () => void;
}

const FIELD_BADGE_LABELS_SK: Record<string, string> = {
  parents: "Rodičia (Parentes)",
  village: "Bydlisko (Locus & Domus)",
  date_name: "Dátum a meno (Nomen)",
  godparents: "Krstní rodičia (Patrini)",
  observationes: "Poznámky (Observationes)",
  ditto_mark: "Značka opakovania („ / Idem)",
  identity_relation: "Sčítací hárok – Meno, priezvisko a pomer k hlave (stĺpce 1–5)",
  birth_marriage_children: "Sčítací hárok – Dátum narodenia a stav (stĺpce 6–10)",
  origin_domicile: "Sčítací hárok – Rodisko a domovská obec (stĺpce 11–14)",
  nationality_religion_literacy: "Sčítací hárok – Národnosť a náboženstvo (stĺpce 15–17)",
  occupation_presence_notes: "Sčítací hárok – Povolanie a poznámky (stĺpce 18–24)",
};

const FIELD_BADGE_LABELS_EN: Record<string, string> = {
  parents: "Parents (Parentes)",
  village: "Residence (Locus & Domus)",
  date_name: "Date & Name (Nomen)",
  godparents: "Godparents (Patrini)",
  observationes: "Remarks (Observationes)",
  ditto_mark: "Ditto mark („ / Idem)",
  identity_relation: "Census sheet – Name, surname & relation to head (cols 1–5)",
  birth_marriage_children: "Census sheet – Birth date & marital status (cols 6–10)",
  origin_domicile: "Census sheet – Birthplace & domicile (cols 11–14)",
  nationality_religion_literacy: "Census sheet – Nationality & religion (cols 15–17)",
  occupation_presence_notes: "Census sheet – Occupation & notes (cols 18–24)",
};

export default function QuestionCenter({ lang = "sk", refreshStatus }: Props) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [questions, setQuestions] = useState<QuestionItem[]>([]);
  const [counts, setCounts] = useState({ open: 0, answered: 0, skipped_for_admin: 0 });
  const [loading, setLoading] = useState(true);
  const [currentIdx, setCurrentIdx] = useState(() => {
    const idx = Number(searchParams.get("idx"));
    return Number.isFinite(idx) && idx >= 0 ? idx : 0;
  });
  const [customAnswer, setCustomAnswer] = useState("");
  const [showFullRow, setShowFullRow] = useState(false);
  const [cropFilter, setCropFilter] = useState<"normal" | "contrast" | "invert">("normal");
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<"open" | "skipped_for_admin" | "answered">(
    () => {
      const st = searchParams.get("status");
      return st === "skipped_for_admin" || st === "answered" ? st : "open";
    }
  );

  useEffect(() => {
    const next = new URLSearchParams();
    if (filterStatus !== "open") next.set("status", filterStatus);
    if (currentIdx > 0) next.set("idx", String(currentIdx));
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [filterStatus, currentIdx]);

  const loadQuestions = async (statusFilter = filterStatus, silent = false) => {
    if (!silent) setLoading(true);
    try {
      const r = await fetch(`/api/questions?status=${statusFilter}`);
      if (r.ok) {
        const res = (await r.json()) as { counts: typeof counts; questions: QuestionItem[] };
        setCounts(res.counts);
        setQuestions(res.questions);
        setCurrentIdx(0);
        setCustomAnswer("");
      }
    } catch {
      // ignore
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    loadQuestions(filterStatus);
  }, [filterStatus]);

  const activeQ = questions[currentIdx];

  const speakQuestion = (q: QuestionItem) => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const fullText =
        lang === "en" && q.question_en
          ? q.question_en
          : q.description_sk
          ? `${q.question_sk}. ${q.description_sk}`
          : q.question_sk;
      const u = new SpeechSynthesisUtterance(fullText);
      u.lang = lang === "en" ? "en-US" : "sk-SK";
      u.rate = 0.92;
      window.speechSynthesis.speak(u);
    }
  };

  const submitAnswer = async (answerText: string) => {
    if (!activeQ || !answerText.trim()) return;
    const answeredId = activeQ.id;
    const cleanAnswer = answerText.trim();
    setSubmitting(true);
    setErrorMsg(null);
    try {
      const r = await fetch(`/api/questions/${answeredId}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer: cleanAnswer,
          answered_by: filterStatus === "skipped_for_admin" ? "Admin" : "Reviewer",
        }),
      });
      if (!r.ok) {
        const errData = await r.json().catch(() => ({}));
        setErrorMsg(
          errData.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť odpoveď. Skúste to znova."
              : "Failed to save answer. Please try again.")
        );
        return;
      }
      // Confirmed DB commit: update local list and counts immediately, then sync silently
      setCustomAnswer("");
      if (filterStatus === "answered") {
        setQuestions((prev) =>
          prev.map((q) =>
            q.id === answeredId
              ? { ...q, status: "answered", selected_answer: cleanAnswer }
              : q
          )
        );
      } else {
        setQuestions((prev) => {
          const next = prev.filter((q) => q.id !== answeredId);
          setCurrentIdx((idx) => Math.max(0, Math.min(idx, next.length - 1)));
          return next;
        });
        setCounts((prev) => ({
          ...prev,
          open: filterStatus === "open" ? Math.max(0, prev.open - 1) : prev.open,
          skipped_for_admin:
            filterStatus === "skipped_for_admin"
              ? Math.max(0, prev.skipped_for_admin - 1)
              : prev.skipped_for_admin,
          answered: prev.answered + 1,
        }));
      }
      void loadQuestions(filterStatus, true);
      refreshStatus?.();
    } catch {
      setErrorMsg(
        lang === "sk"
          ? "Chyba spojenia so serverom pri ukladaní odpovede."
          : "Network error while saving answer."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const skipQuestion = async () => {
    if (!activeQ) return;
    const skippedId = activeQ.id;
    setSubmitting(true);
    setErrorMsg(null);
    try {
      const r = await fetch(`/api/questions/${skippedId}/skip`, { method: "POST" });
      if (!r.ok) {
        const errData = await r.json().catch(() => ({}));
        setErrorMsg(
          errData.detail ||
            (lang === "sk"
              ? "Nepodarilo sa odložiť otázku."
              : "Failed to defer question.")
        );
        return;
      }
      // Confirmed DB commit: update local list and counts immediately, then sync silently
      setCustomAnswer("");
      if (filterStatus !== "skipped_for_admin") {
        setQuestions((prev) => {
          const next = prev.filter((q) => q.id !== skippedId);
          setCurrentIdx((idx) => Math.max(0, Math.min(idx, next.length - 1)));
          return next;
        });
        setCounts((prev) => ({
          ...prev,
          open: filterStatus === "open" ? Math.max(0, prev.open - 1) : prev.open,
          answered:
            filterStatus === "answered" ? Math.max(0, prev.answered - 1) : prev.answered,
          skipped_for_admin: prev.skipped_for_admin + 1,
        }));
      }
      void loadQuestions(filterStatus, true);
      refreshStatus?.();
    } catch {
      setErrorMsg(
        lang === "sk"
          ? "Chyba spojenia so serverom pri odkladaní otázky."
          : "Network error while deferring question."
      );
    } finally {
      setSubmitting(false);
    }
  };

  const tabClass = (active: boolean) =>
    `px-3 py-1.5 rounded-lg font-semibold text-xs sm:text-sm cursor-pointer transition ${
      active
        ? "bg-amber-600 text-white shadow-2xs"
        : "text-stone-600 hover:bg-stone-100 hover:text-stone-900"
    }`;

  const badgeLabels = lang === "en" ? FIELD_BADGE_LABELS_EN : FIELD_BADGE_LABELS_SK;

  return (
    <div className="space-y-4">
      <div className="bg-white border border-stone-200 shadow-2xs rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <HelpCircle className="w-5 h-5 text-amber-600 shrink-0" />
          <h1 className="text-lg font-bold text-stone-900 truncate">
            {lang === "sk"
              ? "Otázky na kontrolu rukopisu"
              : "Handwriting Review Questions"}
          </h1>
          {activeQ && (
            <span className="px-2 py-0.5 rounded-md bg-stone-100 text-stone-700 text-xs font-semibold shrink-0">
              {currentIdx + 1} / {questions.length}
            </span>
          )}
        </div>
        <div className="inline-flex flex-wrap gap-1 bg-stone-100 p-1 rounded-xl border border-stone-200">
          <button
            onClick={() => setFilterStatus("open")}
            className={tabClass(filterStatus === "open")}
          >
            {lang === "sk" ? `Na vyriešenie (${counts.open})` : `Open (${counts.open})`}
          </button>
          <button
            onClick={() => setFilterStatus("skipped_for_admin")}
            className={tabClass(filterStatus === "skipped_for_admin")}
          >
            {lang === "sk"
              ? `Odložené (${counts.skipped_for_admin})`
              : `Deferred (${counts.skipped_for_admin})`}
          </button>
          <button
            onClick={() => setFilterStatus("answered")}
            className={tabClass(filterStatus === "answered")}
          >
            {lang === "sk"
              ? `Vyriešené (${counts.answered})`
              : `Resolved (${counts.answered})`}
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-red-50 border border-red-400 rounded-xl p-3 flex items-center gap-2 text-red-900 font-medium text-sm">
          <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {loading ? (
        <div className="bg-white border border-stone-200 rounded-xl p-8 text-center space-y-3">
          <div className="w-8 h-8 rounded-full border-2 border-amber-600 border-t-transparent animate-spin mx-auto" />
          <p className="text-sm font-semibold text-stone-700">
            {lang === "sk"
              ? "Načítavam otázky na kontrolu rukopisu…"
              : "Loading handwriting review questions…"}
          </p>
        </div>
      ) : !activeQ ? (
        <div className="bg-white border border-emerald-300 rounded-xl p-8 text-center space-y-3">
          <CheckCircle2 className="w-10 h-10 text-emerald-600 mx-auto" />
          <h2 className="text-lg font-bold text-stone-900">
            {lang === "sk"
              ? "V tejto kategórii nie sú ďalšie otázky."
              : "No more questions in this category."}
          </h2>
          <p className="text-sm text-stone-600">
            {lang === "sk"
              ? "Čitateľné riadky sú uložené v prehľade kníh a FamilySearch balíčkoch."
              : "Confirmed rows are saved in the Books overview and FamilySearch action packets."}
          </p>
          {filterStatus === "open" && counts.skipped_for_admin > 0 && (
            <button
              onClick={() => setFilterStatus("skipped_for_admin")}
              className="mt-1 inline-flex items-center gap-2 px-4 py-2 bg-amber-700 hover:bg-amber-800 text-white font-semibold text-sm rounded-lg cursor-pointer"
            >
              {lang === "sk"
                ? `Zobraziť odložené (${counts.skipped_for_admin})`
                : `View deferred (${counts.skipped_for_admin})`}
            </button>
          )}
        </div>
      ) : (
        <div className="bg-white border border-stone-200 shadow-2xs rounded-xl p-5 space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs sm:text-sm">
            <Link
              to={`/knihy?scan=${activeQ.scan_id}&row=${activeQ.row_id}`}
              className="bg-stone-100 hover:bg-stone-200 text-stone-800 font-semibold px-3 py-1.5 min-h-[34px] rounded-full inline-flex items-center gap-1.5 transition"
              title={
                lang === "sk"
                  ? "Otvoriť zdrojový sken v prehľade kníh"
                  : "Open source scan in Books & Scans"
              }
            >
              <span>
                {activeQ.scan_filename} · {activeQ.detected_year || "?"} ·{" "}
                {activeQ.entry_number
                  ? `č. ${activeQ.entry_number}`
                  : `${lang === "sk" ? "riadok" : "row"} ${activeQ.row_index}`}
              </span>
              <ExternalLink className="w-3.5 h-3.5 text-stone-600" />
            </Link>
            <span className="bg-amber-50 border border-amber-300 text-amber-900 font-semibold px-3 py-1.5 min-h-[34px] inline-flex items-center rounded-full">
              {badgeLabels[activeQ.field_name] || activeQ.field_name}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            <div className="lg:col-span-6 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-semibold text-stone-800">
                  {lang === "sk" ? "Výrez rukopisu" : "Handwriting crop"}
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setCropFilter((prev) =>
                        prev === "normal" ? "contrast" : prev === "contrast" ? "invert" : "normal"
                      )
                    }
                    className="px-3 py-1.5 min-h-[34px] rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-900 font-semibold text-xs sm:text-sm cursor-pointer"
                  >
                    {cropFilter === "normal"
                      ? lang === "sk"
                        ? "Kontrast"
                        : "Contrast"
                      : cropFilter === "contrast"
                      ? lang === "sk"
                        ? "Negatív"
                        : "Invert"
                      : lang === "sk"
                      ? "Pôvodné"
                      : "Original"}
                  </button>
                  {activeQ.stitched_row_url && (
                    <button
                      onClick={() => setShowFullRow(!showFullRow)}
                      className="flex items-center gap-1.5 px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-50 hover:bg-amber-100 border border-amber-300 text-amber-900 font-semibold text-xs sm:text-sm cursor-pointer"
                    >
                      <Eye className="w-4 h-4" />
                      {showFullRow
                        ? lang === "sk"
                          ? "Skryť riadok"
                          : "Hide full row"
                        : lang === "sk"
                        ? "Celý riadok"
                        : "Full row"}
                    </button>
                  )}
                </div>
              </div>

              {activeQ.crop_url && (
                <div className="bg-stone-900 p-3 rounded-xl overflow-x-auto">
                  <img
                    src={activeQ.crop_url}
                    alt={lang === "sk" ? "Výrez rukopisu" : "Handwriting crop"}
                    className={`max-h-48 mx-auto object-contain rounded bg-white p-0.5 ${
                      cropFilter === "contrast"
                        ? "crop-high-contrast"
                        : cropFilter === "invert"
                        ? "crop-invert"
                        : ""
                    }`}
                  />
                </div>
              )}

              {showFullRow && activeQ.stitched_row_url && (
                <div className="bg-stone-800 p-3 rounded-xl overflow-x-auto">
                  <img
                    src={activeQ.stitched_row_url}
                    alt={lang === "sk" ? "Celý spojený riadok" : "Full stitched row"}
                    className="max-h-24 object-contain rounded bg-white p-0.5"
                  />
                </div>
              )}

              {activeQ.neighbor_crops.length > 0 && (
                <div className="bg-stone-50 p-3 rounded-xl border border-stone-200 space-y-2">
                  <span className="text-xs sm:text-sm font-semibold text-stone-700">
                    {lang === "sk"
                      ? "Susedné riadky toho istého pisára"
                      : "Neighboring rows by the same scribe"}
                  </span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {activeQ.neighbor_crops.map((item, idx) => {
                      const imgUrl =
                        typeof item === "string" ? item : item.crop_url || item.crop_path || "";
                      const caption =
                        typeof item === "object" && item !== null ? item.confirmed_text : null;
                      if (!imgUrl) return null;
                      return (
                        <div key={idx} className="rounded-lg border border-stone-200 bg-white p-2">
                          <img
                            src={imgUrl}
                            alt={`Neighbor row ${idx + 1}`}
                            className="max-h-16 w-full object-contain rounded"
                          />
                          {caption && (
                            <div
                              className="text-xs text-stone-700 mt-1.5 px-0.5 leading-snug break-words"
                              title={caption}
                            >
                              {caption}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-6 space-y-3">
              <div className="bg-amber-50 border-l-4 border-amber-600 p-3.5 rounded-r-xl space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <p className="text-base font-bold text-stone-900 leading-snug">
                    {lang === "en" && activeQ.question_en
                      ? activeQ.question_en
                      : activeQ.question_sk}
                  </p>
                  <button
                    onClick={() => speakQuestion(activeQ)}
                    className="shrink-0 p-2 min-h-[36px] min-w-[36px] inline-flex items-center justify-center rounded-lg bg-white hover:bg-stone-100 border border-amber-200 text-stone-800 cursor-pointer"
                    title={lang === "sk" ? "Prečítať otázku nahlas" : "Read question aloud"}
                  >
                    <Volume2 className="w-4 h-4 text-amber-800" />
                  </button>
                </div>
                {(activeQ.description_sk || activeQ.question_en) && (
                  <div className="flex items-start gap-1.5 text-sm text-stone-700">
                    <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                    <span>
                      {lang === "en"
                        ? activeQ.description_sk || activeQ.question_sk
                        : activeQ.description_sk || activeQ.question_en}
                    </span>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <span className="text-sm font-semibold text-stone-800">
                  {lang === "sk"
                    ? "Kliknite na správne čítanie"
                    : "Click the correct reading"}
                </span>
                <div className="grid grid-cols-1 gap-2">
                  {activeQ.candidates.map((cand, idx) => (
                    <div
                      key={idx}
                      className="flex items-stretch rounded-xl border border-amber-500 bg-amber-50 hover:bg-amber-100 overflow-hidden"
                    >
                      <button
                        disabled={submitting}
                        onClick={() => submitAnswer(cand)}
                        className="flex-1 flex items-center justify-between px-4 py-2.5 text-left text-base font-semibold text-stone-900 cursor-pointer"
                      >
                        <span className="flex items-center gap-2.5 min-w-0">
                          <span className="inline-flex w-6 h-6 items-center justify-center rounded-full bg-amber-800 text-white text-xs font-bold shrink-0">
                            {idx + 1}
                          </span>
                          <span className="truncate">{cand}</span>
                        </span>
                        <Sparkles className="w-4 h-4 text-amber-700 shrink-0 ml-2" />
                      </button>
                      <button
                        type="button"
                        onClick={() => setCustomAnswer(cand)}
                        title={lang === "sk" ? "Predvyplniť na úpravu" : "Pre-fill for editing"}
                        className="px-2.5 bg-amber-200/70 hover:bg-amber-300 text-amber-950 text-xs font-semibold flex flex-col items-center justify-center gap-0.5 border-l border-amber-400 cursor-pointer"
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                        {lang === "sk" ? "Upraviť" : "Edit"}
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-semibold text-stone-800">
                  {lang === "sk"
                    ? "Alebo napíšte vlastný prepis"
                    : "Or type a custom transcription"}
                </label>
                <div className="flex flex-col sm:flex-row gap-2">
                  <input
                    type="text"
                    value={customAnswer}
                    onChange={(e) => setCustomAnswer(e.target.value)}
                    placeholder={
                      lang === "sk"
                        ? "Presný prepis podľa rukopisu…"
                        : "Exact transcription from handwriting…"
                    }
                    className="flex-1 px-3 py-2 text-base rounded-lg border border-stone-300 focus:border-amber-600 focus:outline-none"
                  />
                  <button
                    disabled={submitting || !customAnswer.trim()}
                    onClick={() => submitAnswer(customAnswer)}
                    className="px-4 py-2 min-h-[38px] rounded-lg bg-emerald-700 hover:bg-emerald-800 disabled:bg-stone-200 disabled:text-stone-600 text-white font-semibold text-sm inline-flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    {submitting && <RefreshCw className="w-4 h-4 animate-spin" />}
                    <span>
                      {submitting
                        ? lang === "sk"
                          ? "Ukladám…"
                          : "Saving…"
                        : lang === "sk"
                        ? "Potvrdiť"
                        : "Confirm"}
                    </span>
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                <div className="flex gap-1.5">
                  <button
                    disabled={currentIdx === 0 || submitting}
                    onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
                    className="px-3.5 py-2 min-h-[36px] rounded-lg bg-stone-100 hover:bg-stone-200 disabled:opacity-40 font-medium text-sm cursor-pointer"
                  >
                    {lang === "sk" ? "← Predošlá" : "← Previous"}
                  </button>
                  <button
                    disabled={currentIdx >= questions.length - 1 || submitting}
                    onClick={() => setCurrentIdx((i) => Math.min(questions.length - 1, i + 1))}
                    className="px-3.5 py-2 min-h-[36px] rounded-lg bg-stone-100 hover:bg-stone-200 disabled:opacity-40 font-medium text-sm cursor-pointer"
                  >
                    {lang === "sk" ? "Ďalšia →" : "Next →"}
                  </button>
                </div>
                {filterStatus === "open" && (
                  <button
                    disabled={submitting}
                    onClick={skipQuestion}
                    className="flex items-center gap-1.5 px-3.5 py-2 min-h-[36px] rounded-lg bg-stone-200 hover:bg-stone-300 disabled:opacity-50 text-stone-800 font-semibold text-sm cursor-pointer"
                  >
                    {submitting ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <SkipForward className="w-4 h-4" />
                    )}
                    {lang === "sk" ? "Odložiť na neskôr" : "Defer question"}
                  </button>
                )}
              </div>
            </div>
          </div>

          {activeQ.row_context && (
            <details className="bg-stone-50 border border-stone-200 rounded-xl">
              <summary className="cursor-pointer px-3.5 py-2.5 text-sm font-semibold text-stone-800 flex flex-wrap items-center gap-2">
                <FileText className="w-4 h-4 text-amber-700 shrink-0" />
                <span>
                  {lang === "sk" ? "Už prečítaný riadok" : "Already extracted row context"}
                </span>
                {activeQ.row_context.child_name && (
                  <span className="font-normal text-stone-600">
                    — {activeQ.row_context.child_name}
                    {activeQ.row_context.date_text ? ` · ${activeQ.row_context.date_text}` : ""}
                  </span>
                )}
              </summary>
              <div className="px-3.5 pb-3.5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 text-sm">
                <div
                  className={`p-2.5 rounded-lg border ${
                    activeQ.field_name === "date_name"
                      ? "bg-amber-100/80 border-amber-400"
                      : "bg-white border-stone-200"
                  }`}
                >
                  <div className="text-xs font-semibold text-stone-700">
                    {lang === "sk" ? "Osoba a dátum" : "Person & date"}
                  </div>
                  <div className="font-semibold text-stone-900">
                    {activeQ.row_context.child_name || "?"}{" "}
                    <span className="font-normal text-stone-700">
                      ({activeQ.row_context.date_text || "—"})
                    </span>
                  </div>
                </div>
                <div
                  className={`p-2.5 rounded-lg border ${
                    activeQ.field_name === "parents"
                      ? "bg-amber-100/80 border-amber-400"
                      : "bg-white border-stone-200"
                  }`}
                >
                  <div className="text-xs font-semibold text-stone-700">
                    {lang === "sk" ? "Rodičia" : "Parents"}
                  </div>
                  <div className="font-semibold text-stone-900 break-words">
                    {activeQ.row_context.parents_text || "—"}
                  </div>
                </div>
                <div
                  className={`p-2.5 rounded-lg border ${
                    activeQ.field_name === "village"
                      ? "bg-amber-100/80 border-amber-400"
                      : "bg-white border-stone-200"
                  }`}
                >
                  <div className="text-xs font-semibold text-stone-700">
                    {lang === "sk" ? "Obec a číslo domu" : "Village & house number"}
                  </div>
                  <div className="font-semibold text-stone-900 break-words">
                    {activeQ.row_context.village_text || "—"}
                  </div>
                </div>
                {activeQ.row_context.godparents_text && (
                  <div
                    className={`p-2.5 rounded-lg border sm:col-span-2 ${
                      activeQ.field_name === "godparents" ||
                      activeQ.field_name === "observationes"
                        ? "bg-amber-100/80 border-amber-400"
                        : "bg-white border-stone-200"
                    }`}
                  >
                    <div className="text-xs font-semibold text-stone-700">
                      {lang === "sk" ? "Krstní rodičia" : "Godparents"}
                    </div>
                    <div className="font-medium text-stone-800 break-words">
                      {activeQ.row_context.godparents_text}
                    </div>
                  </div>
                )}
                {activeQ.row_context.officiant_text && (
                  <div className="p-2.5 rounded-lg border bg-white border-stone-200">
                    <div className="text-xs font-semibold text-stone-700">
                      {lang === "sk" ? "Krstiaci kňaz" : "Officiant"}
                    </div>
                    <div className="font-medium text-stone-800 break-words">
                      {activeQ.row_context.officiant_text}
                    </div>
                  </div>
                )}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
