import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Bell,
  BookOpen,
  CheckCircle2,
  ExternalLink,
  FileSpreadsheet,
  HelpCircle,
  Home,
  RefreshCw,
  Search,
  Upload,
  X,
} from "lucide-react";
import HomeDashboard from "./pages/HomeDashboard";
import {
  type FeatureFlags,
  syncBackendFeatureFlagsIfIdle,
  useFeatureFlags,
} from "./featureFlags";

const QuestionCenter = lazy(() => import("./pages/QuestionCenter"));
const FamilySearchReview = lazy(() => import("./pages/FamilySearchReview"));
const BulkUpload = lazy(() => import("./pages/BulkUpload"));
const BooksOverview = lazy(() => import("./pages/BooksOverview"));
const CensusHub = lazy(() => import("./pages/CensusHub"));
const CensusSheetView = lazy(() => import("./pages/CensusSheetView"));
const DebugWorkflow = lazy(() => import("./pages/DebugWorkflow"));

export interface OcrCompletionEvent {
  event_id?: number;
  timestamp?: number;
  status?: "completed" | "error" | string;
  kind?: string;
  sheet_id?: number | null;
  slovakiana_key?: string | null;
  sheet_label?: string | null;
  scan_id?: number | null;
  scan_filename?: string | null;
  batch_id?: number | null;
  job_id?: number | null;
  rows_processed?: number;
  persons_count?: number;
  houses_count?: number;
  districts_count?: number;
  matched_count?: number;
  duration_sec?: number;
  title_sk?: string;
  title_en?: string;
  message_sk?: string;
  message_en?: string;
  target_url?: string | null;
}

export interface PipelineState {
  active_stage: string;
  active_model: string | null;
  current_scan_id: number | null;
  current_row_id: number | null;
  current_row_index?: number | null;
  total_rows?: number | null;
  current_sheet_id?: number | null;
  current_slovakiana_key?: string | null;
  current_sheet_label?: string | null;
  job_id?: number | null;
  job_sheet_index?: number | null;
  job_total_sheets?: number | null;
  progress_pct: number;
  started_at?: number | null;
  updated_at?: number | null;
  message_sk: string;
  message_en: string;
  last_completed?: OcrCompletionEvent | null;
}

export interface SystemStatus {
  pangolin_public_url: string;
  build_commit?: string;
  ai_server: {
    online: boolean;
    base_url: string;
    configured: { ocr: string; decision: string; extract: string };
    models: Array<{
      id: string;
      type?: string;
      state?: string;
      quantization?: string;
      arch?: string;
    }>;
  };
  familysearch: {
    has_token: boolean;
    environment: string;
  };
  counts: {
    scans: number;
    rows: number;
    facts: number;
    open_questions: number;
    new_to_familysearch: number;
    unreviewed_matches: number;
    census_sheets?: number;
  };
  pipeline_state: PipelineState;
  feature_flags?: Partial<FeatureFlags>;
}

export function emitClientOcrNotification(event: OcrCompletionEvent): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent<OcrCompletionEvent>("matrika:ocr-completed", {
      detail: {
        timestamp: Date.now() / 1000,
        status: "completed",
        ...event,
      },
    })
  );
}

function playSubtleCompletionTone(isError = false): void {
  if (typeof window === "undefined") return;
  try {
    const AudioCtx =
      window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(isError ? 320 : 587.33, ctx.currentTime);
    if (!isError) {
      osc.frequency.setValueAtTime(880, ctx.currentTime + 0.11);
    }
    gain.gain.setValueAtTime(0.04, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.28);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.29);
    setTimeout(() => {
      void ctx.close().catch(() => {});
    }, 400);
  } catch {
    // ignore audio restrictions
  }
}

function formatStageBadgeLabel(stage: string, lang: "sk" | "en"): string {
  switch (stage) {
    case "census_download":
      return lang === "sk" ? "1. Sťahovanie skenov" : "1. Downloading scans";
    case "census_harvest":
      return lang === "sk" ? "Katalóg Slovakiana" : "Slovakiana Catalog";
    case "layout":
      return lang === "sk" ? "2. Rozdelenie riadkov" : "2. Row slicing";
    case "ocr":
      return lang === "sk" ? "3. AI OCR rukopisu" : "3. Handwriting OCR";
    case "decision":
      return lang === "sk" ? "Kontrola čitateľnosti" : "Readability check";
    case "extract":
      return lang === "sk" ? "4. Extrakcia údajov" : "4. Fact extraction";
    case "mcp_match":
      return lang === "sk" ? "5. Párovanie (FS & Matrika)" : "5. Matching (FS & Matrika)";
    default:
      return stage;
  }
}

interface ToastItem extends OcrCompletionEvent {
  toastId: string;
  createdAtMs: number;
}

export default function App() {
  const location = useLocation();
  const { flags: featureFlags, toggleFlag } = useFeatureFlags();
  const [lang, setLang] = useState<"sk" | "en">(
    () => (localStorage.getItem("matrika_lang") as "sk" | "en") || "sk"
  );
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  const seenEventIdsRef = useRef<Set<number>>(new Set());
  const initializedLastCompletedRef = useRef<boolean>(false);
  const lastDedupeKeyRef = useRef<{ key: string; timeMs: number }>({ key: "", timeMs: 0 });
  const prevStageRef = useRef<string>("idle");

  const toggleLang = () => {
    const next = lang === "sk" ? "en" : "sk";
    setLang(next);
    localStorage.setItem("matrika_lang", next);
  };

  const dismissToast = useCallback((toastId: string) => {
    setToasts((prev) => prev.filter((t) => t.toastId !== toastId));
  }, []);

  const pushToastNotification = useCallback((ev: OcrCompletionEvent) => {
    if (typeof ev.event_id === "number") {
      if (seenEventIdsRef.current.has(ev.event_id)) return;
      seenEventIdsRef.current.add(ev.event_id);
    }
    const nowMs = Date.now();
    const dedupeKey = [
      ev.status || "completed",
      ev.slovakiana_key || ev.sheet_id || ev.scan_id || ev.job_id || "",
      ev.message_sk || ev.title_sk || "",
    ].join("|");
    if (
      lastDedupeKeyRef.current.key === dedupeKey &&
      nowMs - lastDedupeKeyRef.current.timeMs < 4500
    ) {
      return;
    }
    lastDedupeKeyRef.current = { key: dedupeKey, timeMs: nowMs };

    const toastId = `toast-${nowMs}-${Math.random().toString(36).slice(2, 7)}`;
    const newToast: ToastItem = {
      ...ev,
      toastId,
      createdAtMs: nowMs,
    };
    setToasts((prev) => [newToast, ...prev].slice(0, 4));
    playSubtleCompletionTone(ev.status === "error");
  }, []);

  // Auto-dismiss toasts after 12 seconds
  useEffect(() => {
    if (toasts.length === 0) return;
    const timer = window.setInterval(() => {
      const now = Date.now();
      setToasts((prev) => prev.filter((t) => now - t.createdAtMs < 12000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [toasts.length]);

  // Listen for client-dispatched completion notifications (from CensusSheetView, CensusHub, BooksOverview, DebugWorkflow)
  useEffect(() => {
    const handler = (e: Event) => {
      const custom = e as CustomEvent<OcrCompletionEvent>;
      if (custom.detail) {
        pushToastNotification(custom.detail);
      }
    };
    window.addEventListener("matrika:ocr-completed", handler);
    return () => window.removeEventListener("matrika:ocr-completed", handler);
  }, [pushToastNotification]);

  const handlePipelineStateTransition = useCallback(
    (nextPs: PipelineState | undefined | null, isInitialLoad = false) => {
      if (!nextPs) return;

      // Track initial `last_completed` event_id on first load so we don't pop up stale past events on page refresh
      if (!initializedLastCompletedRef.current) {
        initializedLastCompletedRef.current = true;
        if (nextPs.last_completed?.event_id != null) {
          seenEventIdsRef.current.add(nextPs.last_completed.event_id);
        }
      } else if (
        nextPs.last_completed &&
        nextPs.last_completed.event_id != null &&
        !seenEventIdsRef.current.has(nextPs.last_completed.event_id)
      ) {
        pushToastNotification(nextPs.last_completed);
      }

      const prevStage = prevStageRef.current;
      const nextStage = nextPs.active_stage || "idle";
      prevStageRef.current = nextStage;

      if (!isInitialLoad && prevStage !== "idle" && nextStage === "idle") {
        const isErr =
          (nextPs.progress_pct === 0 &&
            (nextPs.message_sk?.toLowerCase().includes("chyba") ||
              nextPs.message_en?.toLowerCase().includes("error"))) ||
          false;
        if (nextPs.progress_pct >= 100 || isErr) {
          const targetUrl =
            nextPs.current_slovakiana_key || nextPs.current_sheet_id
              ? `/scitacie-harky/${nextPs.current_slovakiana_key || nextPs.current_sheet_id}`
              : nextPs.current_scan_id
              ? `/knihy?scan=${nextPs.current_scan_id}`
              : null;
          pushToastNotification({
            status: isErr ? "error" : "completed",
            sheet_id: nextPs.current_sheet_id,
            slovakiana_key: nextPs.current_slovakiana_key,
            sheet_label: nextPs.current_sheet_label,
            scan_id: nextPs.current_scan_id,
            title_sk: isErr ? "Chyba pri spracovaní OCR" : "AI OCR dokončené!",
            title_en: isErr ? "OCR Processing Error" : "AI OCR Completed!",
            message_sk: nextPs.message_sk,
            message_en: nextPs.message_en,
            target_url: targetUrl,
          });
        }
      }

      window.dispatchEvent(
        new CustomEvent<PipelineState>("matrika:pipeline-state", { detail: nextPs })
      );
    },
    [pushToastNotification]
  );

  const fetchStatus = useCallback(
    async (isInitial = false) => {
      try {
        const r = await fetch("/api/status");
        if (r.ok) {
          const data = (await r.json()) as SystemStatus;
          setStatus(data);
          if (data.feature_flags && typeof data.feature_flags === "object") {
            syncBackendFeatureFlagsIfIdle(data.feature_flags);
          }
          handlePipelineStateTransition(data.pipeline_state, isInitial);
        }
      } catch {
        // ignore transient poll errors
      }
    },
    [handlePipelineStateTransition]
  );

  const isBusy = Boolean(
    status?.pipeline_state?.active_stage &&
      status.pipeline_state.active_stage !== "idle"
  );

  // Elapsed time counter while pipeline is busy
  useEffect(() => {
    if (!isBusy) {
      setElapsedSec(0);
      return;
    }
    const startedAtSec = status?.pipeline_state?.started_at;
    const localStartMs = startedAtSec ? startedAtSec * 1000 : Date.now();
    const updateElapsed = () => {
      setElapsedSec(Math.max(1, Math.round((Date.now() - localStartMs) / 1000)));
    };
    updateElapsed();
    const t = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(t);
  }, [isBusy, status?.pipeline_state?.started_at]);

  // Adaptive status polling (2s when busy, 8s when idle)
  useEffect(() => {
    void fetchStatus(true);
  }, [fetchStatus]);

  useEffect(() => {
    const pollMs = isBusy ? 2000 : 8000;
    const interval = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void fetchStatus(false);
    }, pollMs);
    return () => window.clearInterval(interval);
  }, [isBusy, fetchStatus]);

  // Auto-reconnecting WebSocket to /ws/pipeline
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: number | null = null;
    let unmounted = false;

    const connectWs = () => {
      if (unmounted) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
      try {
        ws = new WebSocket(wsUrl);
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "state" && msg.data) {
              const incomingState = msg.data as PipelineState;
              const wasActive = prevStageRef.current !== "idle";
              handlePipelineStateTransition(incomingState, false);
              setStatus((prev) =>
                prev
                  ? {
                      ...prev,
                      pipeline_state: { ...prev.pipeline_state, ...incomingState },
                    }
                  : prev
              );
              if (wasActive && incomingState.active_stage === "idle") {
                void fetchStatus(false);
              }
            } else if (msg.type === "ocr_completed" && msg.data) {
              pushToastNotification(msg.data as OcrCompletionEvent);
              void fetchStatus(false);
            } else if (msg.type === "sheet_updated" && msg.data) {
              window.dispatchEvent(
                new CustomEvent("matrika:sheet-updated", { detail: msg.data })
              );
            } else if (msg.type === "feature_flags" && msg.data) {
              const rawFlags =
                msg.data?.flags && typeof msg.data.flags === "object"
                  ? msg.data.flags
                  : msg.data;
              syncBackendFeatureFlagsIfIdle(rawFlags);
            }
          } catch {
            // ignore
          }
        };
        ws.onclose = () => {
          if (!unmounted) {
            reconnectTimer = window.setTimeout(connectWs, 3000);
          }
        };
      } catch {
        if (!unmounted) {
          reconnectTimer = window.setTimeout(connectWs, 5000);
        }
      }
    };

    connectWs();

    return () => {
      unmounted = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, [fetchStatus, handlePipelineStateTransition, pushToastNotification]);

  const allNavItems = [
    {
      to: "/scitacie-harky",
      label: lang === "sk" ? "Sčítacie hárky" : "Census Sheets",
      icon: FileSpreadsheet,
      badge: status?.counts.census_sheets || null,
      badgeColor: "bg-purple-700 text-white",
      enabled: featureFlags.tab_census,
    },
    {
      to: "/",
      label: lang === "sk" ? "Domov" : "Home",
      icon: Home,
      badge: null,
      enabled: featureFlags.tab_matrika,
    },
    {
      to: "/otazky",
      label: lang === "sk" ? "Otázky" : "Questions",
      icon: HelpCircle,
      badge: status?.counts.open_questions || null,
      badgeColor: "bg-amber-600 text-white",
      enabled: featureFlags.tab_questions,
    },
    {
      to: "/knihy",
      label: lang === "sk" ? "Knihy a strany" : "Books & Scans",
      icon: BookOpen,
      badge: status?.counts.scans || null,
      badgeColor: "bg-stone-700 text-white",
      enabled: featureFlags.tab_records,
    },
    {
      to: "/hladanie",
      label: lang === "sk" ? "Vyhľadávanie" : "Search",
      icon: Search,
      badge: status?.counts.facts || null,
      badgeColor: "bg-stone-700 text-white",
      enabled: featureFlags.tab_search,
    },
    {
      to: "/familysearch",
      label: "FamilySearch",
      icon: CheckCircle2,
      badge: status?.counts.unreviewed_matches || null,
      badgeColor: "bg-emerald-700 text-white",
      enabled: featureFlags.tab_familysearch,
    },
    {
      to: "/nahrat",
      label: lang === "sk" ? "Nahrať" : "Upload",
      icon: Upload,
      badge: null,
      enabled: featureFlags.tab_matrika,
    },
  ];

  const navItems = allNavItems.filter((item) => item.enabled);

  const aiOnline = Boolean(status?.ai_server?.online);
  const debugActive = location.pathname.startsWith("/debug");
  const buildCommitHash =
    (typeof __BUILD_COMMIT_HASH__ !== "undefined" &&
    __BUILD_COMMIT_HASH__ &&
    __BUILD_COMMIT_HASH__ !== "dev"
      ? __BUILD_COMMIT_HASH__
      : status?.build_commit && status.build_commit !== "dev"
      ? status.build_commit
      : typeof __BUILD_COMMIT_HASH__ !== "undefined" && __BUILD_COMMIT_HASH__
      ? __BUILD_COMMIT_HASH__
      : status?.build_commit) || "dev";

  const ps = status?.pipeline_state;
  const activeSheetLink =
    ps?.current_slovakiana_key || ps?.current_sheet_id
      ? `/scitacie-harky/${ps.current_slovakiana_key || ps.current_sheet_id}`
      : ps?.current_scan_id
      ? `/knihy?scan=${ps.current_scan_id}`
      : null;
  const activePct = Math.max(8, Math.min(99, ps?.progress_pct || 15));

  return (
    <div className="min-h-screen flex flex-col bg-stone-100 text-stone-900 overflow-x-hidden">
      <header className="bg-stone-900 text-stone-50 border-b border-stone-800 shadow-2xs">
        <div className="w-full max-w-none mx-auto px-3 lg:px-5 py-2 flex flex-wrap lg:flex-nowrap items-center justify-between gap-2">
          <Link
            to={featureFlags.tab_matrika ? "/" : "/scitacie-harky"}
            className="flex items-center gap-2.5 shrink-0"
          >
            <div className="w-8 h-8 rounded-lg bg-amber-600 flex items-center justify-center font-bold text-base text-white shadow-2xs">
              M
            </div>
            <div className="leading-tight">
              <h1 className="text-sm xl:text-base font-bold tracking-tight whitespace-nowrap text-white">
                {lang === "sk"
                  ? "Čítačka matrík a sčítacích hárkov"
                  : "Matrika & Census Digitizer"}
              </h1>
            </div>
          </Link>

          <nav className="w-full lg:w-auto lg:flex-1 order-3 lg:order-none flex flex-wrap lg:flex-nowrap items-center gap-1 min-w-0 pt-1.5 lg:pt-0 border-t border-stone-800 lg:border-t-0 overflow-x-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active =
                item.to === "/"
                  ? location.pathname === "/"
                  : item.to === "/scitacie-harky"
                  ? location.pathname.startsWith("/scitacie-harky") ||
                    location.pathname.startsWith("/scitanie") ||
                    location.pathname.startsWith("/census")
                  : item.to === "/otazky"
                  ? location.pathname.startsWith("/otazky") ||
                    location.pathname.startsWith("/ulohy")
                  : item.to === "/knihy"
                  ? location.pathname.startsWith("/knihy") ||
                    location.pathname.startsWith("/prehlad")
                  : item.to === "/nahrat"
                  ? location.pathname.startsWith("/nahrat") ||
                    location.pathname.startsWith("/sprava")
                  : location.pathname.startsWith(item.to);
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 min-h-[34px] rounded-lg font-semibold text-xs xl:text-sm whitespace-nowrap transition shrink-0 ${
                    active
                      ? "bg-amber-600 text-white shadow-2xs"
                      : "text-stone-300 hover:bg-stone-800 hover:text-white"
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                  {item.badge !== null && item.badge > 0 && (
                    <span
                      className={`ml-0.5 px-1.5 py-0 rounded-full text-xs font-bold leading-5 ${
                        active ? "bg-stone-950/60 text-white" : item.badgeColor
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-1.5 shrink-0">
            {featureFlags.admin_debug_overlay && (
              <>
                <span
                  className="flex items-center gap-1.5 px-2.5 py-1.5 min-h-[34px] rounded-md bg-stone-800 text-xs font-semibold whitespace-nowrap"
                  title={
                    aiOnline
                      ? `AI: ${status?.ai_server?.base_url || "online"}`
                      : "AI server offline"
                  }
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      aiOnline ? "bg-emerald-400" : "bg-red-400"
                    }`}
                  />
                  <span className={aiOnline ? "text-emerald-300" : "text-red-300"}>
                    {aiOnline ? "AI" : "AI offline"}
                  </span>
                </span>
                <button
                  type="button"
                  onClick={() => toggleFlag("admin_debug_overlay")}
                  data-testid="header-admin-overlay-pill"
                  title={
                    lang === "sk"
                      ? "Vypnúť technické nástroje (Admin Debug Overlay)"
                      : "Turn off inline debug tools (Admin Debug Overlay)"
                  }
                  className="px-2.5 py-1.5 min-h-[34px] rounded-md bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold whitespace-nowrap cursor-pointer transition"
                >
                  {lang === "sk" ? "Admin Overlay: ZAP" : "Admin Overlay: ON"}
                </button>
              </>
            )}
            <button
              onClick={toggleLang}
              className="px-3 py-1.5 min-h-[34px] min-w-[38px] rounded-md bg-stone-800 hover:bg-stone-700 border border-stone-700 font-semibold text-xs cursor-pointer text-stone-200"
              title="Prepnúť jazyk / Switch language"
            >
              {lang === "sk" ? "SK" : "EN"}
            </button>
            <Link
              to="/debug"
              title={lang === "sk" ? "Admin & Debug konzola" : "Admin & Debug Console"}
              className={`px-2.5 py-1.5 min-h-[34px] rounded-md font-semibold text-xs inline-flex items-center gap-1.5 transition ${
                debugActive
                  ? "bg-amber-600 text-white"
                  : "text-stone-300 hover:bg-stone-800 hover:text-white border border-stone-700"
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Debug</span>
            </Link>
          </div>
        </div>
      </header>

      {isBusy && ps && (
        <div
          data-testid="global-pipeline-progress-banner"
          className="bg-amber-50 border-b border-amber-300 px-4 lg:px-6 py-2 shadow-xs"
        >
          <div className="w-full max-w-none mx-auto flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              <RefreshCw className="w-4 h-4 text-amber-700 animate-spin shrink-0" />
              <span className="px-2 py-0.5 rounded-md bg-amber-600 text-white text-xs font-bold shrink-0">
                {formatStageBadgeLabel(ps.active_stage, lang)}
              </span>
              {ps.job_total_sheets != null && ps.job_total_sheets > 0 && (
                <span className="px-2 py-0.5 rounded-md bg-stone-900 text-white text-xs font-bold shrink-0">
                  {lang === "sk"
                    ? `Hárok ${ps.job_sheet_index || 1}/${ps.job_total_sheets}`
                    : `Sheet ${ps.job_sheet_index || 1}/${ps.job_total_sheets}`}
                </span>
              )}
              {(ps.current_slovakiana_key || ps.current_sheet_label) && activeSheetLink && (
                <Link
                  to={activeSheetLink}
                  className="px-2 py-0.5 rounded bg-white hover:bg-amber-100 text-stone-900 border border-amber-300 text-xs font-bold inline-flex items-center gap-1 shrink-0 transition"
                  title={
                    lang === "sk"
                      ? "Otvoriť práve spracovávaný hárok"
                      : "Open currently processing sheet"
                  }
                >
                  <FileSpreadsheet className="w-3.5 h-3.5 text-amber-700" />
                  <span>
                    {ps.current_slovakiana_key
                      ? `${ps.current_slovakiana_key}${
                          ps.current_sheet_label &&
                          !ps.current_sheet_label.includes(ps.current_slovakiana_key)
                            ? ` · ${ps.current_sheet_label}`
                            : ""
                        }`
                      : ps.current_sheet_label}
                  </span>
                </Link>
              )}
              {ps.total_rows != null && ps.total_rows > 0 && (
                <span className="px-2 py-0.5 rounded bg-white border border-stone-300 text-stone-800 text-xs font-bold shrink-0">
                  {lang === "sk"
                    ? `Riadok ${ps.current_row_index || 1}/${ps.total_rows}`
                    : `Row ${ps.current_row_index || 1}/${ps.total_rows}`}
                </span>
              )}
              <span className="font-semibold text-xs sm:text-sm text-stone-900 truncate">
                {lang === "sk" ? ps.message_sk : ps.message_en}
              </span>
              {featureFlags.admin_debug_overlay && ps.active_model && (
                <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-mono shrink-0">
                  {ps.active_model}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2.5 shrink-0">
              {elapsedSec > 0 && (
                <span className="text-xs font-semibold text-stone-700">{elapsedSec} s</span>
              )}
              <span className="px-2 py-0.5 rounded-full bg-amber-600 text-white text-xs font-bold">
                {activePct}%
              </span>
              <div className="w-36 sm:w-48 bg-amber-200/90 rounded-full h-2.5 overflow-hidden shrink-0 border border-amber-300">
                <div
                  className="bg-amber-600 h-full transition-all duration-300 animate-pulse"
                  style={{ width: `${activePct}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Popup Toast Notifications when OCR completes or errors */}
      {toasts.length > 0 && (
        <div
          data-testid="ocr-toast-container"
          role="region"
          aria-label={lang === "sk" ? "Upozornenia OCR" : "OCR Notifications"}
          className="fixed bottom-4 right-4 z-50 max-w-md w-[calc(100vw-2rem)] space-y-2.5 pointer-events-none"
        >
          {toasts.map((t) => {
            const isError = t.status === "error";
            const title =
              lang === "sk"
                ? t.title_sk ||
                  (isError ? "Chyba pri spracovaní OCR" : "AI OCR čítanie dokončené!")
                : t.title_en ||
                  (isError ? "OCR Processing Error" : "AI OCR Reading Completed!");
            const msg =
              lang === "sk"
                ? t.message_sk || "Hárok bol úspešne prečítaný a spracovaný."
                : t.message_en || "Sheet has been successfully read and processed.";
            const badgeText =
              t.slovakiana_key ||
              t.sheet_label ||
              t.scan_filename ||
              (t.sheet_id ? `Hárok #${t.sheet_id}` : null) ||
              (t.scan_id ? `Sken #${t.scan_id}` : null);

            return (
              <div
                key={t.toastId}
                data-testid="ocr-toast-notification"
                role="status"
                aria-live="polite"
                className={`pointer-events-auto rounded-xl p-4 border-2 shadow-xl transition-all ${
                  isError
                    ? "bg-red-50 border-red-500 text-red-950"
                    : "bg-emerald-50 border-emerald-500 text-stone-900"
                }`}
              >
                <div className="flex items-start justify-between gap-2.5">
                  <div className="flex items-start gap-2.5 min-w-0">
                    {isError ? (
                      <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                    ) : (
                      <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
                    )}
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="font-bold text-sm text-stone-900 flex items-center gap-1">
                          <Bell className="w-3.5 h-3.5 text-amber-600" />
                          <span>{title}</span>
                        </span>
                        {badgeText && (
                          <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 font-mono font-bold text-xs">
                            {badgeText}
                          </span>
                        )}
                        {t.duration_sec != null && t.duration_sec > 0 && (
                          <span className="px-1.5 py-0.5 rounded bg-emerald-200/80 text-emerald-950 text-[11px] font-bold">
                            {t.duration_sec} s
                          </span>
                        )}
                      </div>
                      <p className="text-xs sm:text-sm text-stone-800 leading-snug">{msg}</p>
                      <div className="flex flex-wrap items-center gap-2 pt-1">
                        {t.target_url && (
                          <Link
                            to={t.target_url}
                            onClick={() => dismissToast(t.toastId)}
                            className="px-3 py-1 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold inline-flex items-center gap-1 transition"
                          >
                            <span>
                              {lang === "sk" ? "Otvoriť výsledok" : "Open result"}
                            </span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </Link>
                        )}
                        <button
                          type="button"
                          onClick={() => dismissToast(t.toastId)}
                          className="px-2.5 py-1 rounded-lg bg-white/80 hover:bg-white border border-stone-300 text-stone-700 text-xs font-semibold cursor-pointer"
                        >
                          {lang === "sk" ? "Zavrieť" : "Dismiss"}
                        </button>
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => dismissToast(t.toastId)}
                    title={lang === "sk" ? "Zavrieť upozornenie" : "Close notification"}
                    className="p-1 rounded-lg hover:bg-black/10 text-stone-500 hover:text-stone-900 cursor-pointer shrink-0"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <main className="flex-1 w-full max-w-none mx-auto px-4 lg:px-6 py-5 overflow-x-hidden">
        <Suspense
          fallback={
            <div className="bg-white rounded-xl border border-stone-200 p-10 text-center space-y-2.5 shadow-2xs">
              <RefreshCw className="w-6 h-6 text-amber-600 animate-spin mx-auto" />
              <div className="text-sm font-semibold text-stone-700">
                {lang === "sk" ? "Načítavam…" : "Loading…"}
              </div>
            </div>
          }
        >
        <Routes>
          <Route
            path="/"
            element={
              featureFlags.tab_matrika ? (
                <HomeDashboard status={status} lang={lang} refreshStatus={fetchStatus} />
              ) : (
                <Navigate to="/scitacie-harky" replace />
              )
            }
          />
          <Route
            path="/otazky"
            element={<QuestionCenter lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/familysearch"
            element={<FamilySearchReview lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/nahrat"
            element={<BulkUpload lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/knihy"
            element={<BooksOverview lang={lang} refreshStatus={fetchStatus} mode="scans" />}
          />
          <Route
            path="/knihy/:bookId"
            element={<BooksOverview lang={lang} refreshStatus={fetchStatus} mode="scans" />}
          />
          <Route
            path="/knihy/:bookId/strana/:scanId"
            element={<BooksOverview lang={lang} refreshStatus={fetchStatus} mode="scans" />}
          />
          <Route
            path="/hladanie"
            element={<BooksOverview lang={lang} refreshStatus={fetchStatus} mode="search" />}
          />
          <Route
            path="/prehlad"
            element={<BooksOverview lang={lang} refreshStatus={fetchStatus} mode="all" />}
          />
          <Route
            path="/scitacie-harky"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/scitacie-harky/obec/:municipality"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/scitacie-harky/:sheetId"
            element={<CensusSheetView lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/scitanie"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/scitanie/obec/:municipality"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/scitanie/:sheetId"
            element={<CensusSheetView lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/census"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/census/obec/:municipality"
            element={<CensusHub lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/census/sheet/:sheetId"
            element={<CensusSheetView lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/debug"
            element={<DebugWorkflow lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/debug/:scanId"
            element={<DebugWorkflow lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/ulohy"
            element={<QuestionCenter lang={lang} refreshStatus={fetchStatus} />}
          />
          <Route
            path="/sprava"
            element={<BulkUpload lang={lang} refreshStatus={fetchStatus} />}
          />
        </Routes>
        </Suspense>
      </main>

      <footer className="bg-white border-t border-stone-200 py-3 px-4 text-center text-stone-600 text-xs space-y-1">
        <div className="font-medium text-stone-700 flex flex-wrap items-center justify-center gap-x-2 gap-y-1">
          <span>
            {lang === "sk"
              ? "Čítanie slovenských, maďarských a latinských cirkevných matrík a sčítacích hárkov 1930/1940 · FamilySearch · lokálna AI"
              : "Slovak / Hungarian / Latin church register & 1930/1940 census sheet OCR · FamilySearch · local AI"}
          </span>
          <span className="text-stone-400 hidden sm:inline">·</span>
          <span
            data-testid="build-commit-hash"
            title={
              lang === "sk"
                ? `Verzia zostavenia (git commit): ${buildCommitHash}`
                : `Build commit hash: ${buildCommitHash}`
            }
            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-stone-100 border border-stone-200 text-stone-600 font-mono text-[11px]"
          >
            <span className="font-sans text-stone-500">
              {lang === "sk" ? "Zostavenie:" : "Build:"}
            </span>{" "}
            <code className="font-semibold text-stone-800">{buildCommitHash}</code>
          </span>
        </div>
        <div className="text-xs text-stone-600 flex flex-wrap items-center justify-center gap-x-2 gap-y-1">
          <span>
            {lang === "sk"
              ? "Zdroj sčítacích hárkov: Slovenský národný archív (SNA) /"
              : "Census sheets source: Slovak National Archives (SNA) /"}
          </span>
          <a
            href="https://slovakiana.sk/scitacie-harky"
            target="_blank"
            rel="noreferrer"
            className="text-amber-700 hover:underline font-semibold inline-flex items-center gap-1 py-0.5"
          >
            <span>Slovakiana (Národné osvetové centrum)</span>
            <ExternalLink className="w-3 h-3" />
          </a>
          <span>·</span>
          <a
            href="https://creativecommons.org/licenses/by-nc-nd/4.0/"
            target="_blank"
            rel="noreferrer"
            className="text-stone-800 hover:underline font-semibold inline-flex items-center gap-1 py-0.5"
          >
            <span>Licencia CC BY-NC-ND 4.0</span>
            <ExternalLink className="w-3 h-3" />
          </a>
          <span>
            {lang === "sk"
              ? "(len na nekomerčný genealogický výskum, bez šírenia odvodených výrezov)"
              : "(non-commercial genealogical research only, no redistribution of modified crops)"}
          </span>
        </div>
      </footer>
    </div>
  );
}
