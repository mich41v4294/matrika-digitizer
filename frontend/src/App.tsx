import React, { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  BookOpen,
  CheckCircle2,
  ExternalLink,
  FileSpreadsheet,
  HelpCircle,
  Home,
  Search,
  Upload,
} from "lucide-react";
import HomeDashboard from "./pages/HomeDashboard";
import QuestionCenter from "./pages/QuestionCenter";
import FamilySearchReview from "./pages/FamilySearchReview";
import BulkUpload from "./pages/BulkUpload";
import BooksOverview from "./pages/BooksOverview";
import CensusHub from "./pages/CensusHub";
import CensusSheetView from "./pages/CensusSheetView";
import DebugWorkflow from "./pages/DebugWorkflow";
import { useFeatureFlags } from "./featureFlags";

export interface SystemStatus {
  pangolin_public_url: string;
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
  pipeline_state: {
    active_stage: string;
    active_model: string | null;
    current_scan_id: number | null;
    current_row_id: number | null;
    progress_pct: number;
    message_sk: string;
    message_en: string;
  };
}

export default function App() {
  const location = useLocation();
  const { flags: featureFlags } = useFeatureFlags();
  const [lang, setLang] = useState<"sk" | "en">(
    () => (localStorage.getItem("matrika_lang") as "sk" | "en") || "sk"
  );
  const [status, setStatus] = useState<SystemStatus | null>(null);

  const toggleLang = () => {
    const next = lang === "sk" ? "en" : "sk";
    setLang(next);
    localStorage.setItem("matrika_lang", next);
  };

  const fetchStatus = async () => {
    try {
      const r = await fetch("/api/status");
      if (r.ok) {
        const data = await r.json();
        setStatus(data);
      }
    } catch {
      // ignore transient poll errors
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void fetchStatus();
    }, 4000);

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "state" && msg.data) {
            setStatus((prev) =>
              prev ? { ...prev, pipeline_state: { ...prev.pipeline_state, ...msg.data } } : prev
            );
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore ws fallback to polling
    }

    return () => {
      clearInterval(interval);
      if (ws) ws.close();
    };
  }, []);

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

  const isBusy =
    status?.pipeline_state?.active_stage &&
    status.pipeline_state.active_stage !== "idle";
  const aiOnline = Boolean(status?.ai_server?.online);
  const debugActive = location.pathname.startsWith("/debug");

  return (
    <div className="min-h-screen flex flex-col bg-stone-100 text-stone-900 overflow-x-hidden">
      <header className="bg-stone-900 text-stone-50 border-b-2 border-amber-600">
        <div className="w-full max-w-none mx-auto px-3 lg:px-5 py-2 flex flex-wrap lg:flex-nowrap items-center justify-between gap-2">
          <Link
            to={featureFlags.tab_matrika ? "/" : "/scitacie-harky"}
            className="flex items-center gap-2 shrink-0"
          >
            <div className="w-8 h-8 rounded-lg bg-amber-600 flex items-center justify-center font-bold text-base text-white">
              M
            </div>
            <div className="leading-tight">
              <h1 className="text-sm xl:text-base font-bold tracking-tight whitespace-nowrap">
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
                      ? "bg-amber-600 text-white"
                      : "text-stone-200 hover:bg-stone-800"
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                  {item.badge !== null && item.badge > 0 && (
                    <span
                      className={`ml-0.5 px-1.5 py-0 rounded-full text-xs font-bold leading-5 ${
                        active ? "bg-stone-900 text-amber-300" : item.badgeColor
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
              onClick={toggleLang}
              className="px-3 py-1.5 min-h-[34px] min-w-[38px] rounded-md bg-stone-800 hover:bg-stone-700 border border-stone-600 font-semibold text-xs cursor-pointer"
              title="Prepnúť jazyk / Switch language"
            >
              {lang === "sk" ? "SK" : "EN"}
            </button>
            <Link
              to="/debug"
              title={lang === "sk" ? "Technický Debug & Feature Flags" : "Debug Workflow & Feature Flags"}
              className={`px-2.5 py-1.5 min-h-[34px] rounded-md font-semibold text-xs inline-flex items-center gap-1.5 ${
                debugActive
                  ? "bg-amber-600 text-white"
                  : "text-stone-200 hover:bg-stone-800 hover:text-white border border-stone-700"
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Debug</span>
            </Link>
          </div>
        </div>
      </header>

      {isBusy && status?.pipeline_state && (
        <div className="bg-amber-100 border-b border-amber-500 px-4 lg:px-6 py-2">
          <div className="w-full max-w-none mx-auto flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-2.5 h-2.5 rounded-full bg-amber-600 animate-pulse shrink-0" />
              <span className="font-semibold text-sm text-stone-900 truncate">
                {lang === "sk"
                  ? status.pipeline_state.message_sk
                  : status.pipeline_state.message_en}
              </span>
              {status.pipeline_state.active_model && (
                <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-mono shrink-0">
                  {status.pipeline_state.active_model}
                </span>
              )}
            </div>
            <div className="w-48 bg-stone-300 rounded-full h-2 overflow-hidden shrink-0">
              <div
                className="bg-amber-600 h-full transition-all duration-300"
                style={{ width: `${Math.max(8, status.pipeline_state.progress_pct || 15)}%` }}
              />
            </div>
          </div>
        </div>
      )}

      <main className="flex-1 w-full max-w-none mx-auto px-4 lg:px-6 py-5 overflow-x-hidden">
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
      </main>

      <footer className="bg-stone-200 border-t border-stone-300 py-3 px-4 text-center text-stone-700 text-xs space-y-1.5">
        <div className="font-medium text-stone-800">
          {lang === "sk"
            ? "Čítanie slovenských, maďarských a latinských cirkevných matrík a sčítacích hárkov 1930/1940 · FamilySearch · lokálna AI"
            : "Slovak / Hungarian / Latin church register & 1930/1940 census sheet OCR · FamilySearch · local AI"}
        </div>
        <div className="text-xs text-stone-700 flex flex-wrap items-center justify-center gap-x-2 gap-y-1">
          <span>
            {lang === "sk"
              ? "Zdroj sčítacích hárkov: Slovenský národný archív (SNA) /"
              : "Census sheets source: Slovak National Archives (SNA) /"}
          </span>
          <a
            href="https://slovakiana.sk/scitacie-harky"
            target="_blank"
            rel="noreferrer"
            className="text-amber-900 hover:underline font-semibold inline-flex items-center gap-1 py-1"
          >
            <span>Slovakiana (Národné osvetové centrum)</span>
            <ExternalLink className="w-3 h-3" />
          </a>
          <span>·</span>
          <a
            href="https://creativecommons.org/licenses/by-nc-nd/4.0/"
            target="_blank"
            rel="noreferrer"
            className="text-stone-900 hover:underline font-semibold inline-flex items-center gap-1 py-1"
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
