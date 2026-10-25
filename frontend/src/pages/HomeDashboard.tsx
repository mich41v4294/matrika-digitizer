import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  AlertCircle,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  HelpCircle,
  Search,
  Sparkles,
  Upload,
} from "lucide-react";
import { SystemStatus } from "../App";
import { useFeatureFlags } from "../featureFlags";

interface Props {
  status: SystemStatus | null;
  lang: "sk" | "en";
  refreshStatus: () => void;
}

export default function HomeDashboard({ status, lang, refreshStatus }: Props) {
  const navigate = useNavigate();
  const { flags: featureFlags } = useFeatureFlags();
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [sampleNotice, setSampleNotice] = useState<string | null>(null);
  const [quickQuery, setQuickQuery] = useState("");

  const handleQuickSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = quickQuery.trim();
    if (!q) {
      navigate("/hladanie");
      return;
    }
    navigate(`/hladanie?q=${encodeURIComponent(q)}`);
  };

  const handleQuickStartSamples = async () => {
    setLoadingSamples(true);
    setSampleNotice(null);
    try {
      const r = await fetch("/api/uploads/load-samples", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          limit: 7,
          auto_process_first: true,
          max_rows_first: 3,
        }),
      });
      if (r.ok) {
        const data = await r.json().catch(() => ({}));
        if (data.loaded && data.loaded > 0) {
          refreshStatus();
          navigate("/knihy");
        } else {
          setSampleNotice(
            lang === "sk"
              ? "Priečinok s ukážkami (samples/) je na tomto serveri prázdny. Nahrajte vlastné skeny matrík v sekcii „Nahrať“ alebo vyhľadajte sčítacie hárky zo Slovakiany."
              : "The sample scans directory (samples/) is not populated on this server. Upload your own register scans in 'Upload' or browse 1930/1940 census sheets from Slovakiana."
          );
        }
      } else {
        const errData = await r.json().catch(() => ({}));
        setSampleNotice(
          errData.detail ||
            (lang === "sk"
              ? "Ukážkové skeny nie sú k dispozícii. Nahrajte vlastné fotky matrík v sekcii „Nahrať“ alebo otvorte „Sčítacie hárky“."
              : "Sample scans are unavailable. Upload your own register photos in 'Upload' or open 'Census Sheets'.")
        );
      }
    } catch {
      setSampleNotice(
        lang === "sk"
          ? "Nepodarilo sa načítať ukážkové skeny zo servera. Skúste nahrať vlastné fotky v sekcii „Nahrať“."
          : "Could not load sample scans from the server. Try uploading your own photos in 'Upload'."
      );
    } finally {
      setLoadingSamples(false);
    }
  };

  const counts = status?.counts || {
    scans: 0,
    rows: 0,
    facts: 0,
    open_questions: 0,
    new_to_familysearch: 0,
    unreviewed_matches: 0,
    census_sheets: 0,
  };

  return (
    <div className="space-y-5">
      <div className="bg-white rounded-xl p-5 border border-stone-200 shadow-2xs space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-xl font-bold text-stone-900">
              {lang === "sk"
                ? "Čo ideme dnes robiť?"
                : "What would you like to do?"}
            </h2>
            <p className="text-sm text-stone-600 mt-0.5">
              {lang === "sk"
                ? "Prečítajte cirkevné matriky a sčítacie hárky 1930/1940, skontrolujte nejasný rukopis a doplňte údaje do FamilySearch."
                : "Read church registers and 1930/1940 census sheets, review uncertain handwriting, and sync facts to FamilySearch."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {counts.scans === 0 && (
              <button
                onClick={handleQuickStartSamples}
                disabled={loadingSamples}
                className="px-3.5 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-sm flex items-center gap-2 cursor-pointer disabled:opacity-50"
              >
                <Sparkles className="w-4 h-4" />
                <span>
                  {loadingSamples
                    ? lang === "sk"
                      ? "Načítavam…"
                      : "Loading…"
                    : lang === "sk"
                    ? "Načítať ukážky"
                    : "Load samples"}
                </span>
              </button>
            )}
            <a
              href="/api/records/export/csv"
              className="px-3 py-2 rounded-lg bg-white hover:bg-stone-50 border border-stone-300 font-semibold text-sm flex items-center gap-1.5"
            >
              <Download className="w-4 h-4" />
              CSV
            </a>
            <a
              href="/api/records/export/gedcom"
              className="px-3 py-2 rounded-lg bg-stone-900 hover:bg-stone-800 text-white font-semibold text-sm flex items-center gap-1.5"
            >
              <Download className="w-4 h-4" />
              GEDCOM
            </a>
          </div>
        </div>

        {/* Unified Quick Search Bar across Census & Matriky */}
        <form onSubmit={handleQuickSearchSubmit} className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-stone-400 absolute left-3.5 top-3" />
            <input
              type="text"
              value={quickQuery}
              onChange={(e) => setQuickQuery(e.target.value)}
              placeholder={
                lang === "sk"
                  ? "Rýchle vyhľadávanie priezviska, mena alebo obce naprieč sčítaním a matrikami..."
                  : "Quick search for surname, given name, or village across census and registers..."
              }
              className="w-full pl-10 pr-4 py-2 min-h-[40px] rounded-lg border border-stone-300 bg-stone-50 focus:bg-white text-sm"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 min-h-[40px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-sm inline-flex items-center gap-1.5 cursor-pointer shrink-0"
          >
            <Search className="w-4 h-4" />
            <span>{lang === "sk" ? "Hľadať" : "Search"}</span>
          </button>
        </form>

        {sampleNotice && (
          <div className="bg-amber-50 border border-amber-300 rounded-lg p-3 flex items-start gap-2 text-amber-950 text-xs">
            <AlertCircle className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <div className="flex-1">
              <span>{sampleNotice}</span>{" "}
              <Link to="/nahrat" className="font-bold underline ml-1">
                {lang === "sk" ? "Prejsť na nahrávanie →" : "Go to upload →"}
              </Link>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {featureFlags.tab_census && (
          <Link
            to="/scitacie-harky"
            className="group bg-white hover:bg-stone-50 border border-stone-200 hover:border-amber-500 rounded-xl p-5 shadow-2xs transition flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="w-10 h-10 rounded-lg bg-amber-600 text-white flex items-center justify-center">
                  <FileSpreadsheet className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-stone-100 text-stone-800 border border-stone-200 font-bold text-sm">
                  {counts.census_sheets ?? 0}{" "}
                  {lang === "sk" ? "hárkov" : "sheets"}
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-900">
                {lang === "sk"
                  ? "Sčítacie hárky (1930 / 1940)"
                  : "Census Sheets (1930 / 1940)"}
              </h3>
              <p className="text-sm text-stone-600">
                {lang === "sk"
                  ? "Prehliadajte obce, adresár domov a rodín zo sčítania 1930 a 1940 a overujte údaje pomocou výrezov."
                  : "Browse villages, house directories, and households from the 1930 and 1940 census with field-level crops."}
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-amber-600 font-semibold text-sm group-hover:translate-x-0.5 transition">
              <span>
                {lang === "sk" ? "Otvoriť sčítacie hárky" : "Open census sheets"}
              </span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </Link>
        )}

        {featureFlags.tab_questions && (
          <Link
            to="/otazky"
            className="group bg-white hover:bg-stone-50 border border-stone-200 hover:border-amber-500 rounded-xl p-5 shadow-2xs transition flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="w-10 h-10 rounded-lg bg-stone-800 text-white flex items-center justify-center">
                  <HelpCircle className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-stone-100 text-stone-800 border border-stone-200 font-bold text-sm">
                  {counts.open_questions}
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-900">
                {lang === "sk" ? "Otázky na kontrolu" : "Questions to review"}
              </h3>
              <p className="text-sm text-stone-600">
                {lang === "sk"
                  ? "Zväčšené nejasné slová a ukážka písma toho istého pisára. Kliknite na správnu možnosť."
                  : "Zoomed uncertain handwriting with neighboring rows from the same scribe. Click the right option."}
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-amber-600 font-semibold text-sm group-hover:translate-x-0.5 transition">
              <span>{lang === "sk" ? "Otvoriť otázky" : "Open questions"}</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </Link>
        )}

        {featureFlags.tab_familysearch && (
          <Link
            to="/familysearch"
            className="group bg-white hover:bg-stone-50 border border-stone-200 hover:border-emerald-500 rounded-xl p-5 shadow-2xs transition flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="w-10 h-10 rounded-lg bg-emerald-600 text-white flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 font-bold text-sm">
                  {counts.new_to_familysearch}
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-900">
                {lang === "sk" ? "Zhody vo FamilySearch" : "FamilySearch Matches"}
              </h3>
              <p className="text-sm text-stone-600">
                {lang === "sk"
                  ? "Osoby z matrík a sčítaní so zhodou vo FamilySearch MCP (automatické párovanie pri pridaní novej osoby)."
                  : "People from registers and census sheets matched in FamilySearch MCP (automatically matched when added)."}
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-emerald-700 font-semibold text-sm group-hover:translate-x-0.5 transition">
              <span>{lang === "sk" ? "Porovnať záznamy" : "Review matches"}</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </Link>
        )}

        {(featureFlags.tab_records || featureFlags.tab_search) && (
          <Link
            to="/knihy"
            className="group bg-white hover:bg-stone-50 border border-stone-300 rounded-xl p-5 transition flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="w-10 h-10 rounded-lg bg-stone-800 text-white flex items-center justify-center">
                  <BookOpen className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-stone-200 text-stone-900 font-semibold text-sm">
                  {counts.scans} {lang === "sk" ? "strán" : "scans"} · {counts.facts}{" "}
                  {lang === "sk" ? "osôb" : "people"}
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-900">
                {lang === "sk" ? "Knihy a osoby" : "Books & people"}
              </h3>
              <p className="text-sm text-stone-700">
                {lang === "sk"
                  ? "Prehliadajte naskenované strany matrík a hľadajte podľa priezviska, obce alebo roka."
                  : "Inspect scanned register pages and search by surname, village, or year alongside original row crops."}
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-stone-800 font-semibold text-sm group-hover:translate-x-0.5 transition">
              <span>{lang === "sk" ? "Zobraziť knihy" : "View books"}</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </Link>
        )}

        {featureFlags.tab_matrika && (
          <Link
            to="/nahrat"
            className="group bg-white hover:bg-stone-50 border border-stone-300 rounded-xl p-5 transition flex flex-col justify-between"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="w-10 h-10 rounded-lg bg-blue-700 text-white flex items-center justify-center">
                  <Upload className="w-5 h-5" />
                </div>
                <span className="px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-900 font-semibold text-sm">
                  JPG / PNG / WebP
                </span>
              </div>
              <h3 className="text-lg font-bold text-stone-900">
                {lang === "sk" ? "Nahrať fotky matrík" : "Upload scans"}
              </h3>
              <p className="text-sm text-stone-700">
                {lang === "sk"
                  ? "Vyberte fotky matričných kníh z počítača. Nahrávajú sa po častiach a automaticky sa spracujú."
                  : "Select register photos from your computer. Files upload in chunks and can be processed automatically."}
              </p>
            </div>
            <div className="mt-3 flex items-center gap-1.5 text-blue-800 font-semibold text-sm group-hover:translate-x-0.5 transition">
              <span>{lang === "sk" ? "Nahrať fotky" : "Upload scans"}</span>
              <ArrowRight className="w-4 h-4" />
            </div>
          </Link>
        )}
      </div>
    </div>
  );
}
