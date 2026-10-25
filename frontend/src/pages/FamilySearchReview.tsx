import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Globe2,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  ExternalLink,
  RefreshCw,
  KeyRound,
  BookOpen,
  Sparkles,
  ChevronDown,
  ChevronRight,
  FileSpreadsheet,
} from "lucide-react";
import { HoverZoomRowCrop } from "../components/ZoomableScanViewer";
import { useFeatureFlags } from "../featureFlags";

type SummaryItem =
  | string
  | { field_sk?: string; value?: string; fs_val?: string; matrika_val?: string };

interface ActionPacket {
  person_name_display?: string;
  person_name?: string;
  given_name?: string;
  latin_name?: string;
  surname?: string;
  maiden_surname?: string | null;
  sex?: string;
  birth_date?: string | null;
  baptism_date?: string | null;
  birth_place_historical?: string | null;
  birth_place_raw?: string | null;
  birth_place_municipality?: string | null;
  birth_place_district?: string | null;
  birth_place_country?: string | null;
  residence_place?: string | null;
  standardized_place?: string | null;
  birth_place_modern_sk?: string | null;
  citizenship?: string | null;
  nationality?: string | null;
  religion?: string | null;
  literacy?: string | null;
  marital_status?: string | null;
  relation_to_head?: string | null;
  occupation?: string | null;
  occupation_position?: string | null;
  employer_workplace?: string | null;
  home_jurisdiction?: string | null;
  father_full_name?: string | null;
  father?: string | null;
  mother_full_name?: string | null;
  mother?: string | null;
  spouse_full_name?: string | null;
  spouse_given_name?: string | null;
  spouse_surname?: string | null;
  godparents_note?: string | null;
  godparents?: string | null;
  observationes_note?: string | null;
  source_citation_sk?: string;
  source_citation_en?: string;
  citation_draft_sk?: string;
  fs_search_url?: string;
  familysearch_search_url?: string;
  event_type?: string;
  event_year?: number | null;
  slovakiana_key?: string | null;
  district?: string | null;
  municipality?: string | null;
  house_number?: string | null;
  sheet_number?: string | null;
  person_row_number?: string | number | null;
}

interface FsMatchItem {
  id: number;
  fact_id: number;
  row_id: number;
  scan_id: number;
  scan_filename: string | null;
  scan_url?: string | null;
  overlay_url?: string | null;
  header_crop_url?: string | null;
  header_crop_path?: string | null;
  stitched_crop_url: string | null;
  y_start?: number | null;
  y_end?: number | null;
  scan_width?: number | null;
  scan_height?: number | null;
  status: string;
  match_status?: string;
  reviewed_by_user?: boolean;
  fs_person_id: string | null;
  fs_person_name: string | null;
  fs_url: string | null;
  confidence_score: number | null;
  standardized_place: string | null;
  what_agrees: SummaryItem[];
  what_differs: SummaryItem[];
  what_is_new: SummaryItem[];
  action_packet: ActionPacket;
  explanation_sk: string | null;
  fact: Record<string, any> | null;
}

interface RowCropHoverProps {
  stitchedCropUrl: string;
  headerCropUrl?: string | null;
  scanUrl?: string | null;
  yStart?: number | null;
  yEnd?: number | null;
  scanWidth?: number | null;
  scanHeight?: number | null;
  altText: string;
  hoverZoomEnabled: boolean;
  lang: "sk" | "en";
}

function RowCropWithContextHover({
  stitchedCropUrl,
  headerCropUrl,
  scanUrl,
  yStart,
  yEnd,
  scanWidth,
  scanHeight,
  altText,
  hoverZoomEnabled,
  lang,
}: RowCropHoverProps) {
  return (
    <HoverZoomRowCrop
      cropUrl={stitchedCropUrl}
      headerCropUrl={headerCropUrl}
      scanUrl={hoverZoomEnabled ? scanUrl : null}
      yStart={hoverZoomEnabled ? yStart : null}
      yEnd={hoverZoomEnabled ? yEnd : null}
      scanWidth={scanWidth}
      scanHeight={scanHeight}
      rowLabel={altText}
      lang={lang}
    />
  );
}

interface FsMatchProgress {
  running: boolean;
  mode: "idle" | "rematch" | "auto" | string;
  processed: number;
  total: number;
  matched_count: number;
  current_person: string;
  total_facts_in_db: number;
  total_matched_in_db: number;
  reviewed_count: number;
  last_finished_at: number | null;
  error: string | null;
  progress_pct?: number;
}

interface FsStatus {
  connected: boolean;
  authenticated: boolean;
  mode: string;
  auth_mode?: string;
  tools_count?: number;
  progress?: FsMatchProgress;
}

function extractFsProgress(payload: any): FsMatchProgress | null {
  const src = payload?.progress && typeof payload.progress === "object" ? payload.progress : payload;
  if (!src || typeof src !== "object" || typeof src.running !== "boolean") {
    return null;
  }
  return {
    running: Boolean(src.running),
    mode: String(src.mode || "idle"),
    processed: Number(src.processed ?? 0),
    total: Number(src.total ?? 0),
    matched_count: Number(src.matched_count ?? 0),
    current_person: String(src.current_person ?? ""),
    total_facts_in_db: Number(src.total_facts_in_db ?? 0),
    total_matched_in_db: Number(src.total_matched_in_db ?? 0),
    reviewed_count: Number(src.reviewed_count ?? 0),
    last_finished_at: typeof src.last_finished_at === "number" ? src.last_finished_at : null,
    error: src.error ? String(src.error) : null,
    progress_pct: typeof src.progress_pct === "number" ? src.progress_pct : undefined,
  };
}

interface Props {
  lang?: "sk" | "en";
  refreshStatus?: () => void;
}

function renderSummaryItem(item: SummaryItem): React.ReactNode {
  if (typeof item === "string") {
    return item;
  }
  if (item && typeof item === "object") {
    if (item.field_sk && item.value) {
      return (
        <span>
          <strong className="font-semibold">{item.field_sk}:</strong> {String(item.value)}
        </span>
      );
    }
    if (item.field_sk && (item.fs_val || item.matrika_val)) {
      return (
        <span>
          <strong className="font-semibold">{item.field_sk}:</strong>{" "}
          {String(item.matrika_val || "?")} vs. FS: {String(item.fs_val || "?")}
        </span>
      );
    }
    return JSON.stringify(item);
  }
  return String(item ?? "");
}

export default function FamilySearchReview({ lang = "sk", refreshStatus }: Props) {
  const { flags: featureFlags } = useFeatureFlags();
  const [searchParams, setSearchParams] = useSearchParams();
  const [matches, setMatches] = useState<FsMatchItem[]>([]);
  const [fsStatus, setFsStatus] = useState<FsStatus | null>(null);
  const [progress, setProgress] = useState<FsMatchProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncingAll, setSyncingAll] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [tokenInput, setTokenInput] = useState("");
  const [refreshingId, setRefreshingId] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const prevRunningRef = useRef<boolean>(false);
  const prevMatchedCountRef = useRef<number>(0);

  const [reviewFilter, setReviewFilter] = useState<"all" | "unreviewed" | "reviewed">(
    () => {
      const rf = searchParams.get("review");
      return rf === "unreviewed" || rf === "reviewed" ? rf : "all";
    }
  );
  const [sourceFilter, setSourceFilter] = useState<"all" | "matrika" | "census">(
    () => {
      const sf = searchParams.get("source");
      return sf === "matrika" || sf === "census" ? sf : "all";
    }
  );

  const [expandedIds, setExpandedIds] = useState<Set<number>>(() => {
    const raw = searchParams.get("expanded");
    if (!raw) return new Set();
    return new Set(
      raw
        .split(",")
        .map((x) => Number(x.trim()))
        .filter((n) => Number.isFinite(n) && n > 0)
    );
  });

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (expandedIds.size > 0) {
      next.set("expanded", Array.from(expandedIds).join(","));
    } else {
      next.delete("expanded");
    }
    if (reviewFilter !== "all") next.set("review", reviewFilter);
    else next.delete("review");
    if (sourceFilter !== "all") next.set("source", sourceFilter);
    else next.delete("source");
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [expandedIds, reviewFilter, sourceFilter]);

  const loadMatchesOnly = async () => {
    try {
      const matchesRes = await fetch("/api/familysearch/matches").then((r) =>
        r.ok ? r.json() : null
      );
      if (matchesRes) {
        const rawMatches: FsMatchItem[] = matchesRes.matches || [];
        setMatches(
          rawMatches.filter((m) => Boolean(String(m.fs_person_id ?? "").trim()))
        );
        const prog = extractFsProgress(matchesRes);
        if (prog) setProgress(prog);
      }
    } catch {
      // ignore
    }
  };

  const loadAll = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const [statusRes, matchesRes] = await Promise.all([
        fetch("/api/familysearch/status").then((r) => (r.ok ? r.json() : null)),
        fetch("/api/familysearch/matches").then((r) => (r.ok ? r.json() : { matches: [] })),
      ]);
      if (statusRes) {
        setFsStatus(statusRes);
        const prog = extractFsProgress(statusRes);
        if (prog) {
          setProgress(prog);
          prevRunningRef.current = prog.running;
          prevMatchedCountRef.current = prog.matched_count;
        }
      }
      const rawMatches: FsMatchItem[] = matchesRes?.matches || [];
      setMatches(
        rawMatches.filter((m) => Boolean(String(m.fs_person_id ?? "").trim()))
      );
      const matchesProg = extractFsProgress(matchesRes);
      if (matchesProg && !statusRes) {
        setProgress(matchesProg);
      }
    } catch {
      // ignore
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    loadAll(false);

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "fs_match_progress" || msg.type === "fs_matches_updated") {
            const nextProg = extractFsProgress(msg.data || msg);
            if (nextProg) {
              const wasRunning = prevRunningRef.current;
              const prevMatched = prevMatchedCountRef.current;
              prevRunningRef.current = nextProg.running;
              prevMatchedCountRef.current = nextProg.matched_count;
              setProgress(nextProg);
              if (
                msg.type === "fs_matches_updated" ||
                (wasRunning && !nextProg.running) ||
                nextProg.matched_count !== prevMatched
              ) {
                loadMatchesOnly();
                refreshStatus?.();
              }
            } else if (msg.type === "fs_matches_updated") {
              loadMatchesOnly();
              refreshStatus?.();
            }
          } else if (
            msg.type === "state" &&
            msg.data?.active_model === "familysearch-mcp"
          ) {
            loadMatchesOnly();
            refreshStatus?.();
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore ws errors
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const isActiveMatching = Boolean(
    syncingAll || progress?.running || refreshingId !== null
  );

  useEffect(() => {
    const intervalMs = isActiveMatching ? 1000 : 8000;
    const timer = setInterval(async () => {
      if (document.visibilityState === "hidden") return;
      try {
        const r = await fetch("/api/familysearch/status");
        if (!r.ok) return;
        const data = await r.json();
        setFsStatus(data);
        const nextProg = extractFsProgress(data);
        if (nextProg) {
          const wasRunning = prevRunningRef.current;
          const prevMatched = prevMatchedCountRef.current;
          prevRunningRef.current = nextProg.running;
          prevMatchedCountRef.current = nextProg.matched_count;
          setProgress(nextProg);
          if (
            (wasRunning && !nextProg.running) ||
            nextProg.matched_count !== prevMatched
          ) {
            await loadMatchesOnly();
            refreshStatus?.();
          }
        }
      } catch {
        // ignore poll errors
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isActiveMatching]);

  const copyField = async (key: string, value: string | null | undefined) => {
    const strVal = String(value ?? "").trim();
    if (!strVal) return;
    try {
      await navigator.clipboard.writeText(strVal);
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 1800);
    } catch {
      // fallback
    }
  };

  const handleSaveToken = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!tokenInput.trim()) return;
    setErrorMsg(null);
    try {
      const r = await fetch("/api/familysearch/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ access_token: tokenInput.trim() }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        setErrorMsg(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť FamilySearch token."
              : "Failed to save FamilySearch token.")
        );
        return;
      }
      setShowTokenModal(false);
      setTokenInput("");
      await loadAll(true);
      refreshStatus?.();
    } catch {
      setErrorMsg(
        lang === "sk"
          ? "Chyba spojenia pri ukladaní tokenu."
          : "Network error while saving token."
      );
    }
  };

  const refreshMatch = async (factId: number) => {
    setRefreshingId(factId);
    setErrorMsg(null);
    try {
      const r = await fetch(`/api/familysearch/matches/${factId}/refresh`, {
        method: "POST",
      });
      if (!r.ok) {
        setErrorMsg(
          lang === "sk"
            ? "Nepodarilo sa obnoviť porovnanie."
            : "Failed to refresh comparison."
        );
      } else {
        const resData = await r.json().catch(() => null);
        const prog = extractFsProgress(resData);
        if (prog) setProgress(prog);
      }
      await loadAll(true);
      refreshStatus?.();
    } finally {
      setRefreshingId(null);
    }
  };

  const syncAllMatches = async () => {
    setSyncingAll(true);
    setErrorMsg(null);
    setProgress((prev) =>
      prev
        ? {
            ...prev,
            running: true,
            mode: "rematch",
            processed: 0,
            total: prev.total_facts_in_db || prev.total || 0,
          }
        : {
            running: true,
            mode: "rematch",
            processed: 0,
            total: 0,
            matched_count: matches.length,
            current_person: "",
            total_facts_in_db: 0,
            total_matched_in_db: matches.length,
            reviewed_count: matches.filter((m) => Boolean(m.reviewed_by_user)).length,
            last_finished_at: null,
            error: null,
          }
    );
    try {
      const r = await fetch("/api/familysearch/rematch", { method: "POST" });
      if (!r.ok) {
        setErrorMsg(
          lang === "sk"
            ? "Nepodarilo sa spustiť párovanie s FamilySearch MCP."
            : "Failed to run FamilySearch MCP matching."
        );
      } else {
        const resData = await r.json().catch(() => null);
        const prog = extractFsProgress(resData);
        if (prog) setProgress(prog);
      }
      await loadAll(true);
      refreshStatus?.();
    } finally {
      setSyncingAll(false);
    }
  };

  const toggleReviewed = async (matchId: number, currentReviewed: boolean) => {
    setErrorMsg(null);
    const r = await fetch(`/api/familysearch/matches/${matchId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewed: !currentReviewed }),
    });
    if (!r.ok) {
      setErrorMsg(
        lang === "sk"
          ? "Nepodarilo sa zmeniť stav kontroly."
          : "Failed to update review state."
      );
      return;
    }
    const resData = await r.json().catch(() => null);
    const confirmedReviewed =
      typeof resData?.reviewed_by_user === "boolean"
        ? resData.reviewed_by_user
        : !currentReviewed;
    setMatches((prev) =>
      prev.map((m) =>
        m.id === matchId ? { ...m, reviewed_by_user: confirmedReviewed } : m
      )
    );
    const prog = extractFsProgress(resData);
    if (prog) setProgress(prog);
    refreshStatus?.();
  };

  const toggleExpanded = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const filteredMatches = useMemo(() => {
    return matches.filter((m) => {
      if (!String(m.fs_person_id ?? "").trim()) return false;
      const isReviewed = Boolean(m.reviewed_by_user);
      if (reviewFilter === "unreviewed" && isReviewed) return false;
      if (reviewFilter === "reviewed" && !isReviewed) return false;

      const isCensus =
        m.fact?.record_type === "census" || m.action_packet?.event_type === "census";
      if (sourceFilter === "census" && !isCensus) return false;
      if (sourceFilter === "matrika" && isCensus) return false;

      return true;
    });
  }, [matches, reviewFilter, sourceFilter]);

  const statusBadge = (status: string) => {
    switch (status) {
      case "confirmed_match":
      case "matched":
        return (
          <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 font-semibold text-xs flex items-center gap-1">
            <CheckCircle2 className="w-3.5 h-3.5" />{" "}
            {lang === "sk" ? "Zhoda vo FS" : "FS Match"}
          </span>
        );
      case "new_person":
      case "not_in_tree":
        return (
          <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 font-semibold text-xs flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5" />{" "}
            {lang === "sk" ? "Nová osoba" : "New Person"}
          </span>
        );
      case "existing_needs_facts":
      case "multiple_candidates":
        return (
          <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-900 font-semibold text-xs flex items-center gap-1">
            <AlertCircle className="w-3.5 h-3.5" />{" "}
            {lang === "sk" ? "Doplniť fakty" : "Add Facts"}
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded-full bg-stone-200 text-stone-800 font-semibold text-xs flex items-center gap-1">
            <BookOpen className="w-3.5 h-3.5" />{" "}
            {lang === "sk" ? "Akčný balíček" : "Action Packet"}
          </span>
        );
    }
  };

  const totalMatched =
    matches.length > 0
      ? matches.length
      : progress?.total_matched_in_db ?? progress?.matched_count ?? 0;
  const reviewedCount =
    matches.length > 0
      ? matches.filter((m) => Boolean(m.reviewed_by_user)).length
      : progress?.reviewed_count ?? 0;
  const reviewPct =
    totalMatched > 0
      ? Math.min(100, Math.round((reviewedCount / Math.max(totalMatched, 1)) * 100))
      : 0;
  const totalInRun =
    progress?.total && progress.total > 0
      ? progress.total
      : progress?.total_facts_in_db ?? 0;
  const processedInRun = isActiveMatching
    ? progress?.processed ?? 0
    : progress?.processed && progress.processed > 0
    ? progress.processed
    : progress?.total_facts_in_db ?? 0;
  const matchingPct =
    totalInRun > 0
      ? Math.min(100, Math.round((processedInRun / Math.max(totalInRun, 1)) * 100))
      : isActiveMatching
      ? 15
      : 100;
  const verifiedPeopleCount =
    progress?.processed && progress.processed > 0
      ? progress.processed
      : progress?.total_facts_in_db ?? 0;
  const currentPersonLabel = progress?.current_person || "";
  const liveMatchedCount = isActiveMatching
    ? progress?.matched_count ?? totalMatched
    : totalMatched;

  return (
    <div className="space-y-4">
      <div className="bg-stone-900 text-amber-50 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-lg font-bold flex items-center gap-2">
            <Globe2 className="w-5 h-5 text-amber-400 shrink-0" />
            {lang === "sk"
              ? "FamilySearch MCP – Spárované osoby"
              : "FamilySearch MCP – Matched People"}
          </h1>
          <p className="text-stone-400 text-xs mt-0.5">
            {lang === "sk"
              ? "Zobrazujú sa iba osoby spárované vo FamilySearch MCP. Párovanie sa spúšťa automaticky pri pridaní alebo úprave osoby."
              : "Showing only people matched in FamilySearch MCP. Matching runs automatically whenever a new person is added or updated."}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {fsStatus && (
            <div
              className="px-2.5 py-1 rounded-lg bg-stone-800 text-xs"
              title={
                fsStatus.authenticated
                  ? lang === "sk"
                    ? "OAuth pripojený"
                    : "OAuth connected"
                  : lang === "sk"
                  ? "Verejné miesta"
                  : "Public places mode"
              }
            >
              <span
                className={
                  fsStatus.connected ? "text-emerald-300 font-semibold" : "text-red-300"
                }
              >
                {fsStatus.connected ? "MCP" : "MCP offline"}
              </span>
            </div>
          )}
          <button
            onClick={syncAllMatches}
            disabled={syncingAll || Boolean(progress?.running)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-stone-700 hover:bg-stone-600 text-amber-100 font-semibold text-xs cursor-pointer"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${
                syncingAll || progress?.running ? "animate-spin" : ""
              }`}
            />
            {syncingAll || progress?.running
              ? lang === "sk"
                ? "Hľadám zhody…"
                : "Matching…"
              : lang === "sk"
              ? "Prehľadať zhody"
              : "Re-match FS"}
          </button>
          <button
            onClick={() => setShowTokenModal(!showTokenModal)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-stone-950 font-semibold text-xs cursor-pointer"
          >
            <KeyRound className="w-3.5 h-3.5" />
            Token
          </button>
        </div>
      </div>

      {/* Prominent FamilySearch MCP Matching & Review Progress Card */}
      <div
        data-testid="fs-progress-card"
        className="bg-white rounded-xl p-4 border border-stone-200 shadow-xs space-y-3"
      >
        {isActiveMatching && (
          <div
            data-testid="fs-live-matching-progress"
            className="bg-amber-50 border border-amber-300 rounded-lg p-3 space-y-2"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2 font-bold text-amber-950">
                <RefreshCw className="w-4 h-4 text-amber-700 animate-spin shrink-0" />
                <span>
                  {lang === "sk"
                    ? progress?.mode === "auto"
                      ? "Automatické párovanie s FamilySearch MCP prebieha…"
                      : "Párovanie s FamilySearch MCP prebieha…"
                    : progress?.mode === "auto"
                    ? "Automatic FamilySearch MCP matching in progress…"
                    : "FamilySearch MCP matching in progress…"}
                </span>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-amber-600 text-white font-bold text-xs">
                {matchingPct}%
              </span>
            </div>

            <div
              role="progressbar"
              aria-label={
                lang === "sk"
                  ? "Priebeh párovania vo FamilySearch MCP"
                  : "FamilySearch MCP matching progress"
              }
              aria-valuenow={matchingPct}
              aria-valuemin={0}
              aria-valuemax={100}
              data-testid="fs-matching-progress-bar"
              className="w-full bg-amber-200/80 rounded-full h-3 overflow-hidden"
            >
              <div
                className="bg-amber-600 h-full transition-all duration-300 animate-pulse"
                style={{
                  width: `${Math.max(totalInRun > 0 ? matchingPct : 15, 8)}%`,
                }}
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-stone-800">
              <span className="font-semibold">
                {lang === "sk"
                  ? `Spracované: ${processedInRun} / ${totalInRun} osôb`
                  : `Processed: ${processedInRun} / ${totalInRun} people`}
              </span>
              {currentPersonLabel && (
                <span className="text-amber-950 font-medium truncate max-w-md">
                  {lang === "sk"
                    ? `Práve sa overuje: ${currentPersonLabel}`
                    : `Currently checking: ${currentPersonLabel}`}
                </span>
              )}
              <span className="font-semibold text-emerald-800">
                {lang === "sk"
                  ? `Nájdené zhody: ${liveMatchedCount}`
                  : `Matches found: ${liveMatchedCount}`}
              </span>
            </div>
          </div>
        )}

        {/* Always-visible Review & Coverage Progress Bar */}
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-bold text-stone-900">
                {lang === "sk"
                  ? `Skontrolované zhody: ${reviewedCount} / ${totalMatched} (${reviewPct}%)`
                  : `Reviewed matches: ${reviewedCount} / ${totalMatched} (${reviewPct}%)`}
              </span>
              <span className="text-stone-500">
                {lang === "sk"
                  ? `Overené osoby v databáze: ${verifiedPeopleCount} · Nájdené zhody vo FamilySearch: ${totalMatched}`
                  : `Verified people in database: ${verifiedPeopleCount} · Matches found in FamilySearch: ${totalMatched}`}
              </span>
            </div>

            <button
              type="button"
              onClick={syncAllMatches}
              disabled={syncingAll || Boolean(progress?.running)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-950 font-semibold text-xs cursor-pointer transition"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${
                  syncingAll || progress?.running ? "animate-spin" : ""
                }`}
              />
              <span>
                {lang === "sk"
                  ? "Znova spustiť priraďovanie"
                  : "Re-run FamilySearch matching"}
              </span>
            </button>
          </div>

          <div
            role="progressbar"
            aria-label={
              lang === "sk"
                ? "Priebeh kontroly zhôd vo FamilySearch"
                : "FamilySearch match review progress"
            }
            aria-valuenow={reviewPct}
            aria-valuemin={0}
            aria-valuemax={100}
            data-testid="fs-review-progress-bar"
            className="w-full bg-stone-200 rounded-full h-2.5 overflow-hidden"
          >
            <div
              className="bg-emerald-600 h-full transition-all duration-300"
              style={{
                width: `${
                  totalMatched > 0 ? Math.max(reviewPct, reviewedCount > 0 ? 5 : 0) : 0
                }%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="bg-white rounded-xl p-3 border border-stone-200 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-semibold text-stone-700 mr-1">
            {lang === "sk" ? "Kontrola:" : "Review:"}
          </span>
          {(
            [
              { id: "all", labelSk: "Všetky", labelEn: "All" },
              { id: "unreviewed", labelSk: "Neskontrolované", labelEn: "Unreviewed" },
              { id: "reviewed", labelSk: "Skontrolované", labelEn: "Reviewed" },
            ] as const
          ).map((pill) => (
            <button
              key={pill.id}
              type="button"
              onClick={() => setReviewFilter(pill.id)}
              className={`px-3 py-1.5 min-h-[32px] rounded-full text-xs font-semibold transition cursor-pointer ${
                reviewFilter === pill.id
                  ? "bg-stone-900 text-amber-300"
                  : "bg-stone-100 text-stone-700 hover:bg-stone-200"
              }`}
            >
              {lang === "sk" ? pill.labelSk : pill.labelEn}
            </button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="font-semibold text-stone-700 mr-1">
            {lang === "sk" ? "Zdroj:" : "Source:"}
          </span>
          {(
            [
              { id: "all", labelSk: "Všetky zdroje", labelEn: "All sources" },
              { id: "matrika", labelSk: "Matrika", labelEn: "Matrika" },
              { id: "census", labelSk: "Sčítacie hárky", labelEn: "Census sheets" },
            ] as const
          ).map((pill) => (
            <button
              key={pill.id}
              type="button"
              onClick={() => setSourceFilter(pill.id)}
              className={`px-3 py-1.5 min-h-[32px] rounded-full text-xs font-semibold transition cursor-pointer ${
                sourceFilter === pill.id
                  ? "bg-amber-600 text-white"
                  : "bg-stone-100 text-stone-700 hover:bg-stone-200"
              }`}
            >
              {lang === "sk" ? pill.labelSk : pill.labelEn}
            </button>
          ))}
          <span className="ml-2 text-stone-600 font-medium">
            {filteredMatches.length} / {matches.length}
          </span>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-red-50 border border-red-400 rounded-xl p-3 flex items-center gap-2 text-red-900 font-medium text-sm">
          <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {showTokenModal && (
        <form
          onSubmit={handleSaveToken}
          className="bg-amber-50 border border-amber-300 rounded-xl p-3 flex flex-col sm:flex-row items-center gap-3"
        >
          <div className="flex-1 space-y-1 w-full">
            <label className="font-semibold text-stone-900 text-xs block">
              FamilySearch OAuth access token
            </label>
            <input
              type="password"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              placeholder="Bearer token…"
              className="w-full px-3 py-2 rounded-lg border border-stone-300 bg-white text-sm"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 rounded-lg bg-amber-800 hover:bg-amber-900 text-white font-semibold text-sm shrink-0 cursor-pointer"
          >
            {lang === "sk" ? "Uložiť" : "Save"}
          </button>
        </form>
      )}

      {loading ? (
        <div className="bg-white rounded-xl p-8 text-center text-stone-500 text-sm font-medium">
          {lang === "sk"
            ? "Načítavam FamilySearch záznamy…"
            : "Loading FamilySearch records…"}
        </div>
      ) : filteredMatches.length === 0 ? (
        <div className="bg-white border border-stone-200 rounded-xl p-8 text-center space-y-3">
          <p className="text-sm text-stone-600 max-w-xl mx-auto">
            {matches.length === 0
              ? lang === "sk"
                ? "Zatiaľ sa nenašli žiadne osoby spárované vo FamilySearch MCP. Nespárované osoby sa tu nezobrazujú — párovanie s FamilySearch sa spúšťa automaticky pri pridaní novej osoby."
                : "No people matched in FamilySearch MCP yet. Unmatched people are hidden — FamilySearch matching runs automatically whenever a new person is added."
              : lang === "sk"
              ? "Pre zvolený filter sa nenašli žiadne spárované osoby."
              : "No matched people found for the selected filter."}
          </p>
          {matches.length === 0 && (
            <button
              onClick={syncAllMatches}
              disabled={syncingAll}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-700 hover:bg-amber-800 text-white font-semibold text-sm cursor-pointer"
            >
              <RefreshCw className={`w-4 h-4 ${syncingAll ? "animate-spin" : ""}`} />
              {lang === "sk" ? "Spustiť párovanie s FamilySearch" : "Run FamilySearch matching"}
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredMatches.map((m) => {
            const p: ActionPacket = m.action_packet || {};
            const fact = m.fact || {};
            const expanded = expandedIds.has(m.id);
            const isCensus = fact.record_type === "census" || p.event_type === "census";
            const eventYear = fact.event_year || p.event_year || null;

            const personDisplay =
              p.person_name_display ||
              p.person_name ||
              `${fact.given_name_raw || "?"} ${fact.surname_raw || "?"}`;
            const givenName = p.given_name || p.latin_name || fact.given_name_raw || null;
            const surname = p.surname || fact.surname_raw || fact.surname_normalized || null;
            const sex = p.sex || fact.sex || null;
            const birthDate =
              p.birth_date ||
              (fact.birth_or_event_date
                ? `${fact.birth_or_event_date} ${fact.event_year || ""}`.trim()
                : null);
            const baptismDate = isCensus
              ? null
              : p.baptism_date ||
                (fact.baptism_or_burial_date
                  ? `${fact.baptism_or_burial_date} ${fact.event_year || ""}`.trim()
                  : null);

            const rodiskoParts = [
              p.birth_place_municipality || fact.birth_place_municipality,
              p.birth_place_district || fact.birth_place_district
                ? `okr. ${p.birth_place_district || fact.birth_place_district}`
                : null,
              p.birth_place_country || fact.birth_place_country,
            ].filter(Boolean);
            const rodiskoStructured =
              rodiskoParts.length > 0
                ? rodiskoParts.join(", ")
                : p.birth_place_raw || fact.birth_place_raw || null;
            const residenceDisplay =
              p.residence_place ||
              (fact.village_raw
                ? `${fact.village_raw}${fact.house_number ? `, dom č. ${fact.house_number}` : ""}`
                : null);

            const birthPlaceHist = isCensus
              ? [
                  residenceDisplay ? `Pobyt: ${residenceDisplay}` : null,
                  rodiskoStructured ? `Rodisko: ${rodiskoStructured}` : null,
                ]
                  .filter(Boolean)
                  .join(" | ") ||
                p.birth_place_historical ||
                m.standardized_place ||
                null
              : p.birth_place_historical ||
                p.standardized_place ||
                m.standardized_place ||
                fact.village_hu ||
                fact.village_raw ||
                null;
            const birthPlaceModern =
              p.birth_place_modern_sk ||
              m.standardized_place ||
              fact.village_sk ||
              fact.village_raw ||
              null;
            const fatherFull = p.father_full_name || p.father || fact.father_name || null;
            const motherFull = p.mother_full_name || p.mother || fact.mother_name || null;
            const godparentsNote =
              p.godparents_note || p.godparents || fact.godparents_or_witnesses || null;
            const obsNote = p.observationes_note || fact.observationes_raw || null;
            const citationText =
              (lang === "en" && p.source_citation_en) ||
              p.source_citation_sk ||
              p.citation_draft_sk ||
              null;
            const searchUrl =
              p.fs_search_url || p.familysearch_search_url || m.fs_url || null;

            const copyFields: Array<{ label: string; val: string | null | undefined; key: string }> =
              (
                isCensus
                  ? [
                      {
                        label: lang === "sk" ? "Celé meno" : "Full name",
                        val: personDisplay,
                        key: `${m.id}-name`,
                      },
                      {
                        label: lang === "sk" ? "Krstné meno" : "Given name",
                        val: givenName,
                        key: `${m.id}-given`,
                      },
                      {
                        label: lang === "sk" ? "Priezvisko" : "Surname",
                        val: surname,
                        key: `${m.id}-sur`,
                      },
                      {
                        label: lang === "sk" ? "Rodné priezvisko" : "Maiden surname",
                        val: p.maiden_surname || fact.mother_maiden_surname,
                        key: `${m.id}-maiden`,
                      },
                      {
                        label: lang === "sk" ? "Vzťah k hlave dom." : "Relation to head",
                        val: p.relation_to_head || fact.relation_to_head,
                        key: `${m.id}-rel`,
                      },
                      {
                        label:
                          lang === "sk"
                            ? "Pohlavie / Rodinný stav"
                            : "Sex / Marital status",
                        val: [sex, p.marital_status || fact.marital_status]
                          .filter(Boolean)
                          .join(" · "),
                        key: `${m.id}-sex`,
                      },
                      {
                        label: lang === "sk" ? "Dátum narodenia" : "Birth date",
                        val: birthDate,
                        key: `${m.id}-birth`,
                      },
                      {
                        label:
                          lang === "sk"
                            ? "Miesto pobytu / rodisko"
                            : "Residence / Birthplace",
                        val: birthPlaceHist,
                        key: `${m.id}-place-h`,
                      },
                      {
                        label: lang === "sk" ? "Miesto (FS štandard)" : "Place (FS standard)",
                        val: birthPlaceModern,
                        key: `${m.id}-place-m`,
                      },
                      {
                        label:
                          lang === "sk"
                            ? "Štátna prísl. / Národnosť / Nábož."
                            : "Citizenship / Nationality / Religion",
                        val: [
                          p.citizenship || fact.citizenship,
                          p.nationality || fact.nationality,
                          p.religion || fact.religion,
                        ]
                          .filter(Boolean)
                          .join(" · "),
                        key: `${m.id}-civ`,
                      },
                      {
                        label: lang === "sk" ? "Povolanie" : "Occupation",
                        val: [
                          p.occupation || fact.social_status_or_occupation,
                          p.occupation_position || fact.occupation_position,
                          p.employer_workplace || fact.employer_workplace,
                        ]
                          .filter(Boolean)
                          .join(" | "),
                        key: `${m.id}-occ`,
                      },
                      {
                        label: lang === "sk" ? "Manžel / Manželka" : "Spouse",
                        val: p.spouse_full_name || fact.spouse_name,
                        key: `${m.id}-spouse`,
                      },
                      {
                        label: lang === "sk" ? "Otec" : "Father",
                        val: fatherFull,
                        key: `${m.id}-father`,
                      },
                      {
                        label: lang === "sk" ? "Matka" : "Mother",
                        val: motherFull,
                        key: `${m.id}-mother`,
                      },
                      {
                        label: lang === "sk" ? "Poznámky" : "Notes",
                        val: obsNote,
                        key: `${m.id}-obs`,
                      },
                    ]
                  : [
                      {
                        label: lang === "sk" ? "Celé meno" : "Full name",
                        val: personDisplay,
                        key: `${m.id}-name`,
                      },
                      {
                        label: lang === "sk" ? "Krstné meno" : "Given name",
                        val: givenName,
                        key: `${m.id}-given`,
                      },
                      {
                        label: lang === "sk" ? "Priezvisko" : "Surname",
                        val: surname,
                        key: `${m.id}-sur`,
                      },
                      {
                        label: lang === "sk" ? "Pohlavie" : "Sex",
                        val: sex,
                        key: `${m.id}-sex`,
                      },
                      {
                        label: lang === "sk" ? "Dátum narodenia" : "Birth date",
                        val: birthDate,
                        key: `${m.id}-birth`,
                      },
                      {
                        label: lang === "sk" ? "Dátum krstu" : "Baptism date",
                        val: baptismDate,
                        key: `${m.id}-bapt`,
                      },
                      {
                        label: lang === "sk" ? "Miesto (historické)" : "Place (historical)",
                        val: birthPlaceHist,
                        key: `${m.id}-place-h`,
                      },
                      {
                        label: lang === "sk" ? "Miesto (FS)" : "Place (FS)",
                        val: birthPlaceModern,
                        key: `${m.id}-place-m`,
                      },
                      {
                        label: lang === "sk" ? "Otec" : "Father",
                        val: fatherFull,
                        key: `${m.id}-father`,
                      },
                      {
                        label: lang === "sk" ? "Matka" : "Mother",
                        val: motherFull,
                        key: `${m.id}-mother`,
                      },
                      {
                        label: lang === "sk" ? "Krstní rodičia" : "Godparents",
                        val: godparentsNote,
                        key: `${m.id}-godp`,
                      },
                      {
                        label: lang === "sk" ? "Poznámky" : "Notes",
                        val: obsNote,
                        key: `${m.id}-obs`,
                      },
                    ]
              ).filter((f) => String(f.val ?? "").trim() !== "");

            return (
              <div key={m.id} className="bg-white border border-stone-200 rounded-xl p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0 space-y-0.5">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {isCensus ? (
                        <span className="px-2.5 py-0.5 rounded-full bg-indigo-100 text-indigo-900 font-semibold text-xs inline-flex items-center gap-1">
                          <FileSpreadsheet className="w-3.5 h-3.5" />
                          {lang === "sk" ? "Sčítací hárok" : "Census sheet"}{" "}
                          {eventYear || ""}
                        </span>
                      ) : (
                        <span className="px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-900 font-semibold text-xs inline-flex items-center gap-1">
                          <BookOpen className="w-3.5 h-3.5" />
                          Matrika{eventYear ? ` ${eventYear}` : ""}
                        </span>
                      )}
                      <span className="text-xs font-semibold text-stone-600 uppercase tracking-wide">
                        {m.scan_filename || p.slovakiana_key || `Sken #${m.scan_id}`}
                        {fact.person_row_number || fact.entry_number
                          ? ` · č. ${fact.person_row_number || fact.entry_number}`
                          : ""}
                        {eventYear ? ` · ${eventYear}` : ""}
                      </span>
                    </div>
                    <h2 className="text-lg font-bold text-stone-900 leading-tight">
                      {personDisplay}
                    </h2>
                    {(birthPlaceHist || m.standardized_place) && (
                      <p className="text-xs sm:text-sm text-amber-900 font-medium">
                        {isCensus ? (
                          <span>
                            <span className="text-stone-600 font-normal">
                              {lang === "sk"
                                ? "Miesto pobytu / rodisko: "
                                : "Residence / Birthplace: "}
                            </span>
                            {birthPlaceHist || m.standardized_place}
                          </span>
                        ) : (
                          m.standardized_place || birthPlaceHist
                        )}
                      </p>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    {m.fs_person_id && (
                      <span className="px-2.5 py-1 rounded-full bg-emerald-900 text-emerald-100 font-mono font-bold text-xs">
                        FS ID: {m.fs_person_id}
                      </span>
                    )}
                    {statusBadge(m.match_status || m.status)}
                    <button
                      onClick={() => toggleReviewed(m.id, Boolean(m.reviewed_by_user))}
                      className={`px-3 py-1.5 min-h-[34px] rounded-lg font-semibold text-xs flex items-center gap-1.5 cursor-pointer ${
                        m.reviewed_by_user
                          ? "bg-emerald-700 text-white"
                          : "bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300"
                      }`}
                    >
                      <Check className="w-3.5 h-3.5" />
                      {m.reviewed_by_user
                        ? lang === "sk"
                          ? "Skontrolované"
                          : "Reviewed"
                        : lang === "sk"
                        ? "Skontrolovať"
                        : "Mark reviewed"}
                    </button>
                    <button
                      onClick={() => refreshMatch(m.fact_id)}
                      disabled={refreshingId === m.fact_id}
                      className="p-2 min-h-[34px] min-w-[34px] inline-flex items-center justify-center rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 cursor-pointer"
                      title={lang === "sk" ? "Znova porovnať" : "Re-compare"}
                    >
                      <RefreshCw
                        className={`w-4 h-4 ${refreshingId === m.fact_id ? "animate-spin" : ""}`}
                      />
                    </button>
                    {searchUrl && (
                      <a
                        href={searchUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="flex items-center gap-1.5 px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-800 hover:bg-amber-900 text-white font-semibold text-xs"
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                        {lang === "sk" ? "Vyhľadať vo FS" : "Search in FS"}
                      </a>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                    <h3 className="font-bold text-emerald-950 text-xs sm:text-sm mb-1.5">
                      {lang === "sk" ? "Čo je nové" : "What is new"}
                    </h3>
                    <ul className="text-xs sm:text-sm text-emerald-950 space-y-1 list-disc list-inside leading-snug">
                      {(m.what_is_new || []).map((item, idx) => (
                        <li key={idx}>{renderSummaryItem(item)}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="bg-stone-50 border border-stone-200 rounded-lg p-3">
                    <h3 className="font-bold text-stone-900 text-xs sm:text-sm mb-1.5">
                      {lang === "sk" ? "Čo súhlasí" : "What agrees"}
                    </h3>
                    {m.what_agrees && m.what_agrees.length > 0 ? (
                      <ul className="text-xs sm:text-sm text-stone-800 space-y-1 list-disc list-inside leading-snug">
                        {m.what_agrees.map((item, idx) => (
                          <li key={idx}>{renderSummaryItem(item)}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs sm:text-sm text-stone-600 italic leading-relaxed">
                        {lang === "sk"
                          ? "Zatiaľ neprepojené s konkrétnou osobou vo FS / lokálnej matrike."
                          : "Not yet linked to a specific person in FS / local register."}
                      </p>
                    )}
                  </div>

                  <div className="bg-rose-50/70 border border-rose-200 rounded-lg p-3">
                    <h3 className="font-bold text-rose-950 text-xs sm:text-sm mb-1.5">
                      {lang === "sk" ? "Čo sa líši" : "Discrepancies"}
                    </h3>
                    {m.what_differs && m.what_differs.length > 0 ? (
                      <ul className="text-xs sm:text-sm text-rose-950 space-y-1 list-disc list-inside leading-snug">
                        {m.what_differs.map((item, idx) => (
                          <li key={idx}>{renderSummaryItem(item)}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs sm:text-sm text-stone-600 italic leading-relaxed">
                        {lang === "sk"
                          ? "Žiadne zistené nezrovnalosti."
                          : "No discrepancies detected."}
                      </p>
                    )}
                  </div>

                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                    <h3 className="font-bold text-amber-950 text-xs sm:text-sm mb-1.5">
                      {lang === "sk" ? "Citácia zdroja" : "Source citation"}
                    </h3>
                    <p className="text-xs sm:text-sm text-stone-800 leading-relaxed mb-2">
                      {citationText ||
                        (lang === "sk"
                          ? "Citácia nie je k dispozícii."
                          : "Citation not available.")}
                    </p>
                    {citationText && (
                      <button
                        onClick={() => copyField(`${m.id}-cit`, citationText)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 min-h-[32px] rounded-md bg-amber-800 hover:bg-amber-900 text-white text-xs font-semibold cursor-pointer"
                      >
                        {copiedKey === `${m.id}-cit` ? (
                          <>
                            <Check className="w-3.5 h-3.5" />{" "}
                            {lang === "sk" ? "Skopírované" : "Copied"}
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />{" "}
                            {lang === "sk" ? "Kopírovať" : "Copy"}
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => toggleExpanded(m.id)}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 min-h-[34px] rounded-lg bg-stone-100 hover:bg-stone-200 border border-stone-200 text-sm font-semibold text-stone-800 hover:text-stone-950 cursor-pointer"
                >
                  {expanded ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                  {lang === "sk"
                    ? "Akčný balíček a výrez riadku"
                    : "Action packet & row crop"}
                </button>

                {expanded && (
                  <div className="space-y-3">
                    {m.stitched_crop_url && (
                      <RowCropWithContextHover
                        stitchedCropUrl={m.stitched_crop_url}
                        headerCropUrl={
                          m.header_crop_url ||
                          m.header_crop_path ||
                          (m.scan_id ? `/media/crops/scan_${m.scan_id}/header.jpg` : null)
                        }
                        scanUrl={m.scan_url}
                        yStart={m.y_start}
                        yEnd={m.y_end}
                        scanWidth={m.scan_width}
                        scanHeight={m.scan_height}
                        altText={
                          isCensus
                            ? lang === "sk"
                              ? "Riadok sčítacieho hárku"
                              : "Census sheet row"
                            : lang === "sk"
                            ? "Spojený riadok matriky"
                            : "Stitched register row"
                        }
                        hoverZoomEnabled={featureFlags.hover_row_zoom}
                        lang={lang}
                      />
                    )}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                      {copyFields.map((f) => {
                        const isCopied = copiedKey === f.key;
                        return (
                          <button
                            key={f.key}
                            onClick={() => copyField(f.key, f.val)}
                            className={`flex items-center justify-between p-2.5 rounded-lg border text-left cursor-pointer ${
                              isCopied
                                ? "bg-emerald-50 border-emerald-500 text-emerald-950"
                                : "bg-stone-50 hover:bg-amber-50 border-stone-200 hover:border-amber-400 text-stone-900"
                            }`}
                          >
                            <div className="min-w-0 pr-2">
                              <div className="text-[11px] font-medium text-stone-500">
                                {f.label}
                              </div>
                              <div className="text-sm font-semibold truncate">
                                {String(f.val)}
                              </div>
                            </div>
                            {isCopied ? (
                              <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                            ) : (
                              <Copy className="w-4 h-4 text-stone-400 shrink-0" />
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
