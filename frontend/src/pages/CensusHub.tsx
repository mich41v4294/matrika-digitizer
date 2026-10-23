import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { LinkedHouseSheet } from "../api";
import { useFeatureFlags } from "../featureFlags";
import { foldText, matchesPersonNameQuery } from "../utils/fuzzyName";
import {
  Building2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  ExternalLink,
  FileSpreadsheet,
  FolderDown,
  Home,
  Layers,
  MapPin,
  Pause,
  Play,
  RefreshCw,
  Scale,
  Search,
  Sparkles,
  Users,
  X,
} from "lucide-react";

interface FacetItem {
  key?: string;
  tagKey: string;
  name: string;
  count?: number;
}

interface CensusStats {
  total_cataloged_sheets: number;
  sheets_1930: number;
  sheets_1940: number;
  downloaded_sheets: number;
  processed_sheets: number;
  extracted_census_persons: number;
  local_matrika_cross_matches: number;
  local_districts?: Array<{
    district: string;
    count: number;
  }>;
  top_municipalities: Array<{
    municipality: string;
    district: string | null;
    count: number;
  }>;
}

interface MunicipalitySummaryItem {
  municipality: string;
  municipality_key: string | null;
  district: string;
  district_key: string | null;
  total_sheets: number;
  downloaded_sheets: number;
  ocr_done_sheets: number;
  overview_sheets_count: number;
  overview_houses_count: number;
  extracted_persons_count: number;
  years: number[];
  slovakiana_count?: number;
  slovakiana_tag_key?: string;
}

interface CensusJob {
  id: number;
  name: string;
  job_type: string;
  census_year_tag: string | null;
  district_tag: string | null;
  municipality_tag: string | null;
  query_text: string | null;
  status: string;
  total_sheets: number;
  cataloged_sheets: number;
  downloaded_sheets: number;
  processed_sheets: number;
  failed_sheets: number;
  current_page: number;
  error_message: string | null;
  is_paused_runtime?: boolean;
}

interface ExtractedPersonPreview {
  id?: number;
  given_name: string | null;
  surname: string | null;
  surname_normalized?: string | null;
  maiden_surname: string | null;
  full_name?: string | null;
  relation_to_head: string | null;
  birth_date: string | null;
  birth_year: number | null;
  birth_place: string | null;
  domicile?: string | null;
  matched_filter?: boolean;
}

interface SerializedSheet {
  id: number;
  slovakiana_id: string;
  slovakiana_key?: string;
  census_year: number;
  district_name: string | null;
  municipality_name: string;
  street_name: string | null;
  matched_street?: boolean;
  house_number_raw: string | null;
  house_number_conscription: string | null;
  household_owner?: string | null;
  box_number: string | null;
  sheet_number: string | null;
  is_blackened: boolean;
  thumbnail_url: string | null;
  source_web_url: string | null;
  download_status: string;
  pipeline_status: string;
  front_scan_id: number | null;
  sheet_type?: "household" | "overview" | string;
  obvod_number?: string | null;
  commissioner_name?: string | null;
  overview_houses_count?: number;
  linked_house_sheets?: LinkedHouseSheet[];
  extracted_persons_preview?: ExtractedPersonPreview[];
  members_preview?: ExtractedPersonPreview[];
}

interface SerializedHousehold {
  id: number;
  head_of_household_name?: string | null;
  head_surname: string | null;
  head_given_name: string | null;
  wife_name?: string | null;
  wife_given_name: string | null;
  wife_maiden_surname: string | null;
  member_count: number;
}

interface CensusSheetRow {
  sheet: SerializedSheet;
  household: SerializedHousehold | null;
  extracted_members_count: number;
  extracted_persons_preview?: ExtractedPersonPreview[];
  members_preview?: ExtractedPersonPreview[];
  extracted_members?: ExtractedPersonPreview[];
  overview_houses_count?: number;
  linked_house_sheets?: LinkedHouseSheet[];
}

interface HouseDirectoryEntry {
  house_number: string;
  house_number_int: number;
  ordinal_number: number | string | null;
  municipality?: string | null;
  street_name?: string | null;
  settlement_name: string | null;
  is_inhabited: boolean | null;
  owner_name: string | null;
  overview_house_id: number | null;
  overview_sheet_id: number | null;
  overview_household_count: number | null;
  overview_present_persons_count: number | null;
  remarks: string | null;
  stitched_crop_path: string | null;
  sheet_id: number | null;
  slovakiana_key: string | null;
  head_name: string | null;
  wife_name: string | null;
  extracted_members_count: number;
  extracted_members?: ExtractedPersonPreview[];
  extracted_persons_preview?: ExtractedPersonPreview[];
  members_preview?: ExtractedPersonPreview[];
  pipeline_status: string;
  sheet: SerializedSheet | null;
  household: SerializedHousehold | null;
  linked_house_sheets?: LinkedHouseSheet[];
}

interface MunicipalityOverviewResponse {
  municipality: string | null;
  district: string | null;
  census_year: number;
  overview_sheets: SerializedSheet[];
  houses_directory: HouseDirectoryEntry[];
  total_houses: number;
  available_streets?: string[];
}

interface Props {
  lang: "sk" | "en";
  refreshStatus: () => void;
}

function foldMunicipalityKey(name: string | null | undefined): string {
  if (!name) return "";
  return name
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function parseBirthYearRange(
  rawQuery: string
): { min: number; max: number } | null {
  const raw = rawQuery.trim();
  if (!raw) return null;
  const mRange = raw.match(
    /^(\d{4})\s*(?:[-–—]+|\.\.+|do)\s*(\d{4})$/i
  );
  if (mRange) {
    const a = Number(mRange[1]);
    const b = Number(mRange[2]);
    return { min: Math.min(a, b), max: Math.max(a, b) };
  }
  const mTol = raw.match(/^(\d{4})\s*(?:±|\+-|\+\/-)\s*(\d{1,2})$/);
  if (mTol) {
    const yr = Number(mTol[1]);
    const tol = Number(mTol[2]);
    return { min: yr - tol, max: yr + tol };
  }
  const mApprox = raw.match(/^(?:~|cca\.?|okolo)\s*(\d{4})$/i);
  if (mApprox) {
    const yr = Number(mApprox[1]);
    return { min: yr - 2, max: yr + 2 };
  }
  const mOpenStart = raw.match(/^(\d{4})\s*[-–—]+$/);
  if (mOpenStart) {
    return { min: Number(mOpenStart[1]), max: 2100 };
  }
  const mOpenEnd = raw.match(/^[-–—]+\s*(\d{4})$/);
  if (mOpenEnd) {
    return { min: 1600, max: Number(mOpenEnd[1]) };
  }
  const mExact = raw.match(/^(\d{4})$/);
  if (mExact) {
    const yr = Number(mExact[1]);
    return { min: yr, max: yr };
  }
  return null;
}

function extractMemberBirthYears(m: ExtractedPersonPreview): number[] {
  const out: number[] = [];
  if (m.birth_year != null && !Number.isNaN(Number(m.birth_year))) {
    out.push(Number(m.birth_year));
  }
  if (m.birth_date) {
    const matches = m.birth_date.match(/\b(1[6789]\d{2}|20\d{2})\b/g);
    if (matches) {
      for (const yStr of matches) {
        const y = Number(yStr);
        if (!out.includes(y)) out.push(y);
      }
    }
  }
  return out;
}

function memberMatchesBirthYear(
  m: ExtractedPersonPreview,
  birthYearQuery: string
): boolean {
  const q = birthYearQuery.trim();
  if (!q) return true;
  const range = parseBirthYearRange(q);
  const years = extractMemberBirthYears(m);
  if (range) {
    return years.some((y) => y >= range.min && y <= range.max);
  }
  const qf = foldText(q);
  return (
    years.some((y) => String(y).includes(qf)) ||
    foldText(m.birth_date).includes(qf)
  );
}

function memberMatchesBirthPlace(
  m: ExtractedPersonPreview,
  birthPlaceQuery: string
): boolean {
  const q = birthPlaceQuery.trim();
  if (!q) return true;
  const qf = foldText(q);
  const combined = foldText(
    [m.birth_place, m.domicile].filter(Boolean).join(" ")
  );
  if (combined.includes(qf)) return true;
  const tokens = qf
    .split(/[\s,;()/]+/)
    .filter((t) => t && t !== "okr" && t !== "okres");
  if (tokens.length > 0 && tokens.every((t) => combined.includes(t))) {
    return true;
  }
  return false;
}

const STREET_STOPWORDS = new Set([
  "ul",
  "ulica",
  "nam",
  "namestie",
  "osada",
  "cesta",
  "trieda",
  "st",
  "street",
]);

function normalizeStreetTokens(raw: string | null | undefined): string[] {
  const folded = foldText(raw);
  if (!folded) return [];
  return folded
    .split(/[\s,;./()\-–—]+/)
    .filter((t) => t && !STREET_STOPWORDS.has(t));
}

function matchesStreetQuery(
  candidates: Array<string | null | undefined>,
  streetQuery: string
): boolean {
  const qRaw = streetQuery.trim();
  if (!qRaw) return true;
  const qTokens = normalizeStreetTokens(qRaw);
  const qClean = qTokens.length > 0 ? qTokens.join(" ") : foldText(qRaw);
  if (!qClean) return true;

  for (const cand of candidates) {
    if (!cand || !cand.trim()) continue;
    const cTrim = cand.trim();
    if (cTrim === "-" || cTrim === "—" || cTrim === "?") continue;
    const cFold = foldText(cTrim);
    const cTokens = normalizeStreetTokens(cTrim);
    const cClean = cTokens.length > 0 ? cTokens.join(" ") : cFold;
    if (cClean.includes(qClean) || cFold.includes(qClean)) return true;
    if (qTokens.length > 0 && qTokens.every((t) => cFold.includes(t))) return true;
  }
  return false;
}

function memberMatchesPersonFilters(
  m: ExtractedPersonPreview,
  personNameQuery: string,
  birthYearQuery: string,
  birthPlaceQuery: string
): boolean {
  const nameOk = matchesPersonNameQuery(
    [
      m.given_name,
      m.surname,
      m.surname_normalized,
      m.maiden_surname,
      m.full_name,
    ],
    personNameQuery
  );
  if (!nameOk) return false;
  if (!memberMatchesBirthYear(m, birthYearQuery)) return false;
  if (!memberMatchesBirthPlace(m, birthPlaceQuery)) return false;
  return true;
}

function formatShortBirthPlace(rawPlace: string | null | undefined): string {
  if (!rawPlace) return "";
  return rawPlace
    .replace(/,\s*(?:Slovensko|Československo|Czechoslovakia|Slovakia)\s*$/i, "")
    .trim();
}

const KNOWN_DISTRICT_KEYS = new Set([
  "trnava",
  "galanta",
  "hlohovec",
  "nitra",
  "topolcany",
  "piestany",
  "sered",
  "bratislava",
  "senica",
  "skalica",
  "pezinok",
  "modra",
  "malacky",
  "dunajska streda",
  "sala",
  "samorin",
  "komarno",
  "nove zamky",
  "zlate moravce",
  "vrable",
  "levice",
  "trencin",
  "nove mesto nad vahom",
  "myjava",
]);

export default function CensusHub({ lang, refreshStatus }: Props) {
  const { flags: featureFlags } = useFeatureFlags();
  const { municipality: routeMunicipality } = useParams<{ municipality?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const [stats, setStats] = useState<CensusStats | null>(null);
  const [districts, setDistricts] = useState<FacetItem[]>([]);
  const [cities, setCities] = useState<FacetItem[]>([]);
  const [localMunicipalities, setLocalMunicipalities] = useState<MunicipalitySummaryItem[]>([]);
  const [jobs, setJobs] = useState<CensusJob[]>([]);
  const [jobsExpanded, setJobsExpanded] = useState<boolean>(false);
  const [dismissedActiveJobId, setDismissedActiveJobId] = useState<number | null>(null);
  const [splitMenuOpen, setSplitMenuOpen] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const splitMenuRef = useRef<HTMLDivElement | null>(null);
  const jobsPopoverRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (splitMenuRef.current && !splitMenuRef.current.contains(target)) {
        setSplitMenuOpen(false);
      }
      if (jobsPopoverRef.current && !jobsPopoverRef.current.contains(target)) {
        setJobsExpanded(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSplitMenuOpen(false);
        setJobsExpanded(false);
      }
    };
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  // Left Sidebar filters (`Okresy a obce`) initialized from URL
  const [selectedYearFilter, setSelectedYearFilter] = useState<"" | "1930" | "1940">(() => {
    const yr = searchParams.get("rok") || searchParams.get("year") || "";
    return yr === "1930" || yr === "1940" ? yr : "";
  });
  const [selectedDistrictTag, setSelectedDistrictTag] = useState<string>(
    () => searchParams.get("okres_tag") || ""
  );
  const [selectedDistrictName, setSelectedDistrictName] = useState<string>(
    () => searchParams.get("okres") || searchParams.get("district") || ""
  );
  const [muniSearchQuery, setMuniSearchQuery] = useState<string>(
    () => searchParams.get("muni_q") || ""
  );
  const [selectedMunicipality, setSelectedMunicipality] = useState<string>(() => {
    if (routeMunicipality) {
      try {
        return decodeURIComponent(routeMunicipality);
      } catch {
        return routeMunicipality;
      }
    }
    return searchParams.get("obec") || searchParams.get("municipality") || "";
  });
  const [selectedMunicipalityTag, setSelectedMunicipalityTag] = useState<string>(
    () => searchParams.get("obec_tag") || ""
  );

  // Genealogical, Street & Birthplace filters initialized from URL (`ulica`, `meno`, `rok_nar`, `rodisko`)
  const [streetFilter, setStreetFilter] = useState<string>(
    () => searchParams.get("ulica") || searchParams.get("street") || ""
  );
  const [personNameFilter, setPersonNameFilter] = useState<string>(
    () => searchParams.get("meno") || searchParams.get("person_name") || ""
  );
  const [birthYearFilter, setBirthYearFilter] = useState<string>(
    () => searchParams.get("rok_nar") || searchParams.get("birth_year") || ""
  );
  const [birthPlaceFilter, setBirthPlaceFilter] = useState<string>(
    () => searchParams.get("rodisko") || searchParams.get("birth_place") || ""
  );

  // Right Main Workspace state initialized from URL
  const [activeTab, setActiveTab] = useState<"overview" | "sheets">(() => {
    const t = searchParams.get("tab");
    if (t === "sheets" || t === "overview") return t;
    const initMuni =
      routeMunicipality ||
      searchParams.get("obec") ||
      searchParams.get("municipality") ||
      "";
    return initMuni ? "overview" : "sheets";
  });
  const [workspaceSearch, setWorkspaceSearch] = useState<string>(
    () => searchParams.get("q") || ""
  );
  const [cardSubFilter, setCardSubFilter] = useState<
    "all" | "overview" | "household" | "ocr_done" | "pending_ocr"
  >(() => {
    const f = searchParams.get("filter");
    return f === "overview" ||
      f === "household" ||
      f === "ocr_done" ||
      f === "pending_ocr"
      ? f
      : "all";
  });
  const [maxSheetsInput, setMaxSheetsInput] = useState<number>(40);

  // Sync route param `/scitacie-harky/obec/:municipality` if it changes
  useEffect(() => {
    if (routeMunicipality) {
      try {
        const decoded = decodeURIComponent(routeMunicipality);
        setSelectedMunicipality(decoded);
      } catch {
        setSelectedMunicipality(routeMunicipality);
      }
    }
  }, [routeMunicipality]);

  // Sync state to URL search params so every view can be copied & bookmarked directly
  useEffect(() => {
    const next = new URLSearchParams();
    if (selectedMunicipality) next.set("obec", selectedMunicipality);
    if (selectedMunicipalityTag) next.set("obec_tag", selectedMunicipalityTag);
    if (selectedDistrictName) next.set("okres", selectedDistrictName);
    if (selectedDistrictTag) next.set("okres_tag", selectedDistrictTag);
    if (selectedYearFilter) next.set("rok", selectedYearFilter);
    const effectiveTab = selectedMunicipality ? activeTab : "sheets";
    if (effectiveTab !== (selectedMunicipality ? "overview" : "sheets")) {
      next.set("tab", effectiveTab);
    } else if (selectedMunicipality && activeTab === "overview") {
      next.set("tab", "overview");
    }
    if (streetFilter.trim()) next.set("ulica", streetFilter.trim());
    if (personNameFilter.trim()) next.set("meno", personNameFilter.trim());
    if (birthYearFilter.trim()) next.set("rok_nar", birthYearFilter.trim());
    if (birthPlaceFilter.trim()) next.set("rodisko", birthPlaceFilter.trim());
    if (workspaceSearch.trim()) next.set("q", workspaceSearch.trim());
    if (cardSubFilter !== "all") next.set("filter", cardSubFilter);
    if (muniSearchQuery.trim()) next.set("muni_q", muniSearchQuery.trim());

    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [
    selectedMunicipality,
    selectedMunicipalityTag,
    selectedDistrictName,
    selectedDistrictTag,
    selectedYearFilter,
    activeTab,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
    workspaceSearch,
    cardSubFilter,
    muniSearchQuery,
  ]);

  // Data for Right Main Workspace
  const [muniOverview, setMuniOverview] = useState<MunicipalityOverviewResponse | null>(null);
  const [sheets, setSheets] = useState<CensusSheetRow[]>([]);
  const [availableStreets, setAvailableStreets] = useState<string[]>([]);
  const [totalSheets, setTotalSheets] = useState<number>(0);
  const [scannedSheetsTotal, setScannedSheetsTotal] = useState<number>(0);
  const [isSearchingCensus, setIsSearchingCensus] = useState<boolean>(true);
  const [censusSearchProgress, setCensusSearchProgress] = useState<{
    running: boolean;
    processed: number;
    total: number;
    progress_pct: number;
    elapsed_ms: number;
  }>({
    running: true,
    processed: 0,
    total: 0,
    progress_pct: 15,
    elapsed_ms: 0,
  });
  const censusSearchAbortRef = useRef<AbortController | null>(null);
  const isSearchingCensusRef = useRef<boolean>(false);
  const hasLoadedCensusOnceRef = useRef<boolean>(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [processingSheetId, setProcessingSheetId] = useState<string | null>(null);

  const selectedYearTag =
    selectedYearFilter === "1930"
      ? "10001"
      : selectedYearFilter === "1940"
      ? "10003"
      : "";

  const loadFacets = async (districtTag?: string, yearTag?: string) => {
    try {
      const params = new URLSearchParams();
      if (districtTag) params.set("district_tag", districtTag);
      if (yearTag) params.set("year_tag", yearTag);
      const r = await fetch(`/api/census/facets?${params.toString()}`);
      if (r.ok) {
        const data = await r.json();
        if (data.districts?.length) setDistricts(data.districts);
        setCities(data.cities || []);
      }
    } catch {
      // ignore offline
    }
  };

  // Listen for live search_progress broadcasts for census scope
  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "search_progress" && msg.data && msg.data.scope === "census") {
            const d = msg.data;
            if (d.running && isSearchingCensusRef.current) {
              setCensusSearchProgress((prev) => ({
                ...prev,
                running: true,
                processed: Number(d.processed ?? prev.processed),
                total: Number(d.total ?? prev.total),
                progress_pct: Math.max(prev.progress_pct, Number(d.progress_pct ?? 20)),
                elapsed_ms: Number(d.elapsed_ms ?? prev.elapsed_ms),
              }));
            }
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore
    }
    return () => {
      if (ws) ws.close();
    };
  }, []);

  // Smooth client-side progress ticker while a census search request is in flight
  useEffect(() => {
    if (!isSearchingCensus) return;
    const t0 = performance.now();
    const timer = window.setInterval(() => {
      setCensusSearchProgress((prev) => {
        if (!prev.running) return prev;
        const nextPct = Math.min(92, prev.progress_pct + Math.max(2, Math.round((94 - prev.progress_pct) * 0.22)));
        const estTotal = Math.max(
          prev.total,
          scannedSheetsTotal,
          stats?.total_cataloged_sheets ?? 0,
          totalSheets,
          1
        );
        const estProcessed = Math.min(estTotal, Math.max(prev.processed, Math.round((nextPct / 100) * estTotal)));
        return {
          ...prev,
          total: estTotal,
          processed: estProcessed,
          progress_pct: nextPct,
          elapsed_ms: Math.round(performance.now() - t0),
        };
      });
    }, 200);
    return () => window.clearInterval(timer);
  }, [isSearchingCensus, scannedSheetsTotal, stats?.total_cataloged_sheets, totalSheets]);

  const loadWorkspaceData = async (isBackgroundPoll = false) => {
    if (isBackgroundPoll && isSearchingCensusRef.current) {
      return;
    }

    let controller: AbortController | null = null;
    const tStart = performance.now();
    if (!isBackgroundPoll) {
      if (censusSearchAbortRef.current) {
        censusSearchAbortRef.current.abort();
      }
      controller = new AbortController();
      censusSearchAbortRef.current = controller;
      isSearchingCensusRef.current = true;
      setIsSearchingCensus(true);
      setCensusSearchProgress((prev) => ({
        running: true,
        processed: 0,
        total: Math.max(
          scannedSheetsTotal,
          stats?.total_cataloged_sheets ?? 0,
          totalSheets,
          prev.total,
          1
        ),
        progress_pct: 15,
        elapsed_ms: 0,
      }));
    }

    try {
      const sheetParams = new URLSearchParams();
      if (workspaceSearch.trim()) sheetParams.set("q", workspaceSearch.trim());
      if (selectedDistrictName) sheetParams.set("district", selectedDistrictName);
      if (selectedMunicipality) sheetParams.set("municipality", selectedMunicipality);
      if (selectedYearFilter) sheetParams.set("census_year", selectedYearFilter);
      if (streetFilter.trim()) sheetParams.set("street", streetFilter.trim());
      if (personNameFilter.trim()) sheetParams.set("person_name", personNameFilter.trim());
      if (birthYearFilter.trim()) sheetParams.set("birth_year", birthYearFilter.trim());
      if (birthPlaceFilter.trim()) sheetParams.set("birth_place", birthPlaceFilter.trim());
      sheetParams.set("limit", "120");

      const muniSummaryParams = new URLSearchParams();
      if (selectedYearFilter) muniSummaryParams.set("census_year", selectedYearFilter);
      if (selectedDistrictName) muniSummaryParams.set("district", selectedDistrictName);

      const ovParams = new URLSearchParams();
      if (selectedMunicipality) ovParams.set("municipality", selectedMunicipality);
      if (selectedDistrictName) ovParams.set("district", selectedDistrictName);
      if (selectedYearFilter) ovParams.set("census_year", selectedYearFilter);
      if (streetFilter.trim()) ovParams.set("street", streetFilter.trim());
      if (personNameFilter.trim()) ovParams.set("person_name", personNameFilter.trim());
      if (birthYearFilter.trim()) ovParams.set("birth_year", birthYearFilter.trim());
      if (birthPlaceFilter.trim()) ovParams.set("birth_place", birthPlaceFilter.trim());
      if (workspaceSearch.trim()) ovParams.set("q", workspaceSearch.trim());

      const sig = controller ? { signal: controller.signal } : undefined;
      const [statsRes, jobsRes, sheetsRes, muniSumRes, ovRes] = await Promise.all([
        fetch("/api/census/stats", sig),
        fetch("/api/census/jobs", sig),
        fetch(`/api/census/sheets?${sheetParams.toString()}`, sig),
        fetch(`/api/census/municipalities-summary?${muniSummaryParams.toString()}`, sig),
        fetch(`/api/census/municipality-overview?${ovParams.toString()}`, sig),
      ]);
      if (controller?.signal.aborted) return;

      hasLoadedCensusOnceRef.current = true;
      let resolvedTotalCataloged = stats?.total_cataloged_sheets ?? 0;
      if (statsRes.ok) {
        const stJson = await statsRes.json();
        setStats(stJson);
        if (typeof stJson.total_cataloged_sheets === "number") {
          resolvedTotalCataloged = stJson.total_cataloged_sheets;
        }
      }
      if (jobsRes.ok) {
        const jd = await jobsRes.json();
        setJobs(jd.jobs || []);
      }
      if (sheetsRes.ok) {
        const sd = await sheetsRes.json();
        const loadedSheets = sd.sheets || [];
        const resolvedTotal = sd.total || 0;
        const resolvedScanned =
          typeof sd.scanned_total === "number"
            ? sd.scanned_total
            : Math.max(resolvedTotalCataloged, resolvedTotal);
        const elapsedMs =
          typeof sd.search_progress?.elapsed_ms === "number" && sd.search_progress.elapsed_ms > 0
            ? sd.search_progress.elapsed_ms
            : Math.max(1, Math.round(performance.now() - tStart));
        setSheets(loadedSheets);
        setTotalSheets(resolvedTotal);
        setScannedSheetsTotal(resolvedScanned);
        if (Array.isArray(sd.available_streets)) {
          setAvailableStreets(sd.available_streets);
        }
        if (!isBackgroundPoll) {
          setCensusSearchProgress({
            running: false,
            processed: resolvedScanned,
            total: Math.max(resolvedScanned, 1),
            progress_pct: 100,
            elapsed_ms: elapsedMs,
          });
        }
      }
      if (muniSumRes.ok) {
        const md = await muniSumRes.json();
        setLocalMunicipalities(md.municipalities || []);
      }
      if (ovRes.ok) {
        const od = (await ovRes.json()) as MunicipalityOverviewResponse;
        setMuniOverview(od);
      }
    } catch (err: any) {
      if (err?.name === "AbortError") {
        return;
      }
      // ignore poll error
    } finally {
      if (!isBackgroundPoll && (!controller || !controller.signal.aborted)) {
        isSearchingCensusRef.current = false;
        setIsSearchingCensus(false);
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    loadFacets(selectedDistrictTag || undefined, selectedYearTag || undefined);
  }, [selectedDistrictTag, selectedYearTag]);

  useEffect(() => {
    isSearchingCensusRef.current = true;
    setIsSearchingCensus(true);
    setCensusSearchProgress((prev) => ({
      ...prev,
      running: true,
      processed: 0,
      total: Math.max(
        scannedSheetsTotal,
        stats?.total_cataloged_sheets ?? 0,
        totalSheets,
        prev.total,
        1
      ),
      progress_pct: 15,
      elapsed_ms: 0,
    }));
    const hasTextQuery = Boolean(
      workspaceSearch.trim() ||
        streetFilter.trim() ||
        personNameFilter.trim() ||
        birthYearFilter.trim() ||
        birthPlaceFilter.trim()
    );
    const debounceTimer = window.setTimeout(
      () => {
        loadWorkspaceData(false);
      },
      hasLoadedCensusOnceRef.current && hasTextQuery ? 120 : 0
    );
    const timer = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadWorkspaceData(true);
    }, 8000);
    return () => {
      window.clearTimeout(debounceTimer);
      window.clearInterval(timer);
    };
  }, [
    workspaceSearch,
    selectedDistrictName,
    selectedMunicipality,
    selectedYearFilter,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
  ]);

  // Combine local DB municipalities and Slovakiana facet municipalities into a unified sidebar list
  const combinedMunicipalities = useMemo(() => {
    const map = new Map<string, MunicipalitySummaryItem>();

    for (const loc of localMunicipalities) {
      const k = foldMunicipalityKey(loc.municipality);
      map.set(k, { ...loc });
    }

    for (const c of cities) {
      const k = foldMunicipalityKey(c.name);
      const tag = c.tagKey || c.key || "";
      const existing = map.get(k);
      if (existing) {
        existing.slovakiana_count = c.count;
        existing.slovakiana_tag_key = tag;
        if (!existing.municipality_key && tag) {
          existing.municipality_key = tag;
        }
      } else {
        map.set(k, {
          municipality: c.name,
          municipality_key: tag || null,
          district: selectedDistrictName || "",
          district_key: selectedDistrictTag || null,
          total_sheets: 0,
          downloaded_sheets: 0,
          ocr_done_sheets: 0,
          overview_sheets_count: 0,
          overview_houses_count: 0,
          extracted_persons_count: 0,
          years: selectedYearFilter ? [Number(selectedYearFilter)] : [1930],
          slovakiana_count: c.count,
          slovakiana_tag_key: tag,
        });
      }
    }

    let list = Array.from(map.values());
    if (muniSearchQuery.trim()) {
      const q = foldMunicipalityKey(muniSearchQuery);
      list = list.filter(
        (m) =>
          foldMunicipalityKey(m.municipality).includes(q) ||
          (m.district && foldMunicipalityKey(m.district).includes(q))
      );
    }

    list.sort((a, b) => {
      if (b.ocr_done_sheets !== a.ocr_done_sheets) {
        return b.ocr_done_sheets - a.ocr_done_sheets;
      }
      if (b.total_sheets !== a.total_sheets) {
        return b.total_sheets - a.total_sheets;
      }
      return a.municipality.localeCompare(b.municipality, "sk");
    });

    return list;
  }, [
    localMunicipalities,
    cities,
    muniSearchQuery,
    selectedDistrictName,
    selectedDistrictTag,
    selectedYearFilter,
  ]);

  // Filter houses directory by workspaceSearch + streetFilter + personNameFilter + birthYearFilter + birthPlaceFilter
  const filteredHousesDirectory = useMemo(() => {
    const list = muniOverview?.houses_directory || [];
    const hasQ = Boolean(workspaceSearch.trim());
    const hasSt = Boolean(streetFilter.trim());
    const hasPn = Boolean(personNameFilter.trim());
    const hasBy = Boolean(birthYearFilter.trim());
    const hasBp = Boolean(birthPlaceFilter.trim());
    if (!hasQ && !hasSt && !hasPn && !hasBy && !hasBp) return list;

    const qFolded = foldText(workspaceSearch);
    return list.filter((h) => {
      if (
        hasSt &&
        !h.sheet?.matched_street &&
        !matchesStreetQuery(
          [h.street_name, h.sheet?.street_name, h.settlement_name],
          streetFilter
        )
      ) {
        return false;
      }

      const members =
        h.extracted_members || h.extracted_persons_preview || h.members_preview || [];
      if (members.some((m) => Boolean(m.matched_filter))) {
        return true;
      }
      if (hasQ) {
        const metaFolded = foldText(
          [
            h.house_number,
            h.owner_name,
            h.head_name,
            h.wife_name,
            h.slovakiana_key,
            h.street_name,
            h.settlement_name,
            h.remarks,
          ]
            .filter(Boolean)
            .join(" ")
        );
        const qMatchesMeta =
          metaFolded.includes(qFolded) ||
          matchesPersonNameQuery([h.owner_name, h.head_name, h.wife_name, h.remarks], workspaceSearch);
        const qMatchesMembers = members.some(
          (m) =>
            matchesPersonNameQuery(
              [m.given_name, m.surname, m.surname_normalized, m.maiden_surname, m.full_name],
              workspaceSearch
            ) ||
            memberMatchesBirthYear(m, workspaceSearch) ||
            memberMatchesBirthPlace(m, workspaceSearch)
        );
        if (!qMatchesMeta && !qMatchesMembers) return false;
      }

      if (hasPn || hasBy || hasBp) {
        const memberCombinedMatch = members.some((m) =>
          memberMatchesPersonFilters(m, personNameFilter, birthYearFilter, birthPlaceFilter)
        );
        const headerNameMatch =
          hasPn &&
          matchesPersonNameQuery([h.owner_name, h.head_name, h.wife_name, h.remarks], personNameFilter) &&
          (!hasBy || members.some((m) => memberMatchesBirthYear(m, birthYearFilter))) &&
          (!hasBp || members.some((m) => memberMatchesBirthPlace(m, birthPlaceFilter)));
        if (!memberCombinedMatch && !headerNameMatch) return false;
      }

      return true;
    });
  }, [
    muniOverview,
    workspaceSearch,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
  ]);

  // Filter sheets in Tab 2 by cardSubFilter + streetFilter + personNameFilter + birthYearFilter + birthPlaceFilter
  const filteredSheets = useMemo(() => {
    const hasSt = Boolean(streetFilter.trim());
    const hasPn = Boolean(personNameFilter.trim());
    const hasBy = Boolean(birthYearFilter.trim());
    const hasBp = Boolean(birthPlaceFilter.trim());

    return sheets.filter((row) => {
      const { sheet: s, household, extracted_members_count } = row;
      const isOverview = s.sheet_type === "overview";
      if (!featureFlags.census_overview_sheets && isOverview) return false;
      const isOcrDone =
        s.download_status === "ocr_done" ||
        s.download_status === "extracted" ||
        s.download_status === "matched" ||
        extracted_members_count > 0 ||
        (s.overview_houses_count || 0) > 0;

      if (cardSubFilter === "overview" && !isOverview) return false;
      if (cardSubFilter === "household" && isOverview) return false;
      if (cardSubFilter === "ocr_done" && !isOcrDone) return false;
      if (cardSubFilter === "pending_ocr" && isOcrDone) return false;

      if (
        hasSt &&
        !s.matched_street &&
        !isOverview &&
        !matchesStreetQuery([s.street_name], streetFilter)
      ) {
        return false;
      }

      if (hasPn || hasBy || hasBp) {
        const members =
          row.extracted_persons_preview ||
          row.extracted_members ||
          row.members_preview ||
          s.extracted_persons_preview ||
          [];
        if (members.some((m) => Boolean(m.matched_filter))) {
          return true;
        }
        if (isOverview) {
          return true;
        }
        const memberCombinedMatch = members.some((m) =>
          memberMatchesPersonFilters(m, personNameFilter, birthYearFilter, birthPlaceFilter)
        );
        const headerNameMatch =
          hasPn &&
          matchesPersonNameQuery(
            [
              s.household_owner,
              household?.head_of_household_name,
              household?.head_given_name,
              household?.head_surname,
              household?.wife_name,
              household?.wife_given_name,
              household?.wife_maiden_surname,
            ],
            personNameFilter
          ) &&
          (!hasBy || members.some((m) => memberMatchesBirthYear(m, birthYearFilter))) &&
          (!hasBp || members.some((m) => memberMatchesBirthPlace(m, birthPlaceFilter)));
        if (!memberCombinedMatch && !headerNameMatch) return false;
      }

      return true;
    });
  }, [
    sheets,
    cardSubFilter,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
    featureFlags.census_overview_sheets,
  ]);

  const activeMuniSummary = useMemo(() => {
    if (!selectedMunicipality) return null;
    const targetFolded = foldMunicipalityKey(selectedMunicipality);
    return (
      combinedMunicipalities.find(
        (m) => foldMunicipalityKey(m.municipality) === targetFolded
      ) ||
      localMunicipalities.find(
        (m) => foldMunicipalityKey(m.municipality) === targetFolded
      ) ||
      null
    );
  }, [combinedMunicipalities, localMunicipalities, selectedMunicipality]);

  const resolveMunicipalityAndDistrictTags = (
    muniName?: string,
    muniTagKey?: string | null
  ) => {
    const targetName = (muniName ?? selectedMunicipality).trim();
    const targetFolded = targetName ? foldMunicipalityKey(targetName) : "";
    const matchedCity = targetFolded
      ? cities.find((c) => foldMunicipalityKey(c.name) === targetFolded)
      : null;
    const matchedMuniSummary = targetFolded
      ? combinedMunicipalities.find(
          (m) => foldMunicipalityKey(m.municipality) === targetFolded
        ) ||
        localMunicipalities.find(
          (m) => foldMunicipalityKey(m.municipality) === targetFolded
        ) ||
        null
      : activeMuniSummary;

    const resolvedMuniTag = (
      muniTagKey ||
      (targetName === selectedMunicipality ? selectedMunicipalityTag : "") ||
      matchedMuniSummary?.slovakiana_tag_key ||
      matchedMuniSummary?.municipality_key ||
      matchedCity?.tagKey ||
      matchedCity?.key ||
      ""
    ).trim();

    const distName = (
      selectedDistrictName ||
      matchedMuniSummary?.district ||
      muniOverview?.district ||
      ""
    ).trim();
    const matchedDistrict = distName
      ? districts.find(
          (d) => foldMunicipalityKey(d.name) === foldMunicipalityKey(distName)
        )
      : null;
    const resolvedDistTag = (
      selectedDistrictTag ||
      matchedMuniSummary?.district_key ||
      matchedDistrict?.tagKey ||
      matchedDistrict?.key ||
      ""
    ).trim();

    return {
      targetName,
      resolvedMuniTag,
      distName,
      resolvedDistTag,
    };
  };

  const handleHarvestMunicipality = async (
    muniName?: string,
    muniTagKey?: string | null
  ) => {
    const { targetName, resolvedMuniTag, distName, resolvedDistTag } =
      resolveMunicipalityAndDistrictTags(muniName, muniTagKey);
    setBusyAction(`harvest_${targetName || "all"}`);
    try {
      // First index immediate hits via remote-search so user sees sheets right away
      await fetch("/api/census/sheets/remote-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_text: resolvedMuniTag
            ? workspaceSearch.trim() || null
            : targetName || workspaceSearch.trim() || null,
          year_tag: selectedYearTag || null,
          district_tag: resolvedDistTag || null,
          municipality_tag: resolvedMuniTag || null,
          page: 1,
          page_size: Math.min(50, maxSheetsInput || 40),
          save_to_catalog: true,
        }),
      });
      // Also kick off background catalog harvest job if batch > 20
      await fetch("/api/census/jobs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Zber hárkov: ${targetName || distName || "Slovakiana"}`,
          job_type: "catalog_harvest",
          census_year_tag: selectedYearTag || null,
          district_tag: resolvedDistTag || null,
          municipality_tag: resolvedMuniTag || null,
          municipality: targetName || null,
          district: distName || null,
          query_text: resolvedMuniTag
            ? workspaceSearch.trim() || null
            : targetName || workspaceSearch.trim() || null,
          max_sheets: maxSheetsInput,
          run_pipeline: false,
          run_mcp: false,
        }),
      });
      if (targetName) {
        setSelectedMunicipality(targetName);
      }
      if (resolvedMuniTag) {
        setSelectedMunicipalityTag(resolvedMuniTag);
      }
      setDismissedActiveJobId(null);
      await loadWorkspaceData();
      refreshStatus();
    } finally {
      setBusyAction(null);
    }
  };

  const handleBulkOcrWorkspace = async () => {
    setBusyAction("bulk_ocr");
    try {
      const { targetName, resolvedMuniTag, distName, resolvedDistTag } =
        resolveMunicipalityAndDistrictTags();
      await fetch("/api/census/jobs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Hromadné OCR: ${targetName || distName || "Výber"}`,
          job_type: "download_and_pipeline",
          census_year_tag: selectedYearTag || null,
          district_tag: resolvedDistTag || null,
          municipality_tag: resolvedMuniTag || null,
          municipality: targetName || null,
          district: distName || null,
          query_text: resolvedMuniTag
            ? workspaceSearch.trim() || null
            : targetName || workspaceSearch.trim() || null,
          max_sheets: maxSheetsInput,
          run_pipeline: true,
          run_mcp: true,
          only_undone: true,
        }),
      });
      setDismissedActiveJobId(null);
      await loadWorkspaceData();
      refreshStatus();
    } finally {
      setBusyAction(null);
    }
  };

  const handlePauseResumeJob = async (job: CensusJob) => {
    const action = job.status === "paused" ? "resume" : "pause";
    await fetch(`/api/census/jobs/${job.id}/${action}`, { method: "POST" });
    await loadWorkspaceData();
  };

  const handleDeleteJob = async (jobId: number) => {
    await fetch(`/api/census/jobs/${jobId}`, { method: "DELETE" });
    await loadWorkspaceData();
  };

  const formatJobStatusLabel = (status: string) => {
    if (lang === "en") return status;
    switch (status) {
      case "running":
        return "Prebieha";
      case "completed":
        return "Dokončené";
      case "paused":
        return "Pozastavené";
      case "error":
        return "Chyba";
      default:
        return status;
    }
  };

  const handleProcessSingleSheet = async (sheetKey: string) => {
    setProcessingSheetId(sheetKey);
    try {
      await fetch(
        `/api/census/sheets/${sheetKey}/download-and-process?run_pipeline=true&run_mcp=true`,
        { method: "POST" }
      );
      await loadWorkspaceData();
      refreshStatus();
    } finally {
      setProcessingSheetId(null);
    }
  };

  const slovakianaMunicipalityUrl = useMemo(() => {
    if (!selectedMunicipality) return null;
    const targetFolded = foldMunicipalityKey(selectedMunicipality);

    const matchedCity = cities.find(
      (c) => foldMunicipalityKey(c.name) === targetFolded
    );
    const muniKey = (
      selectedMunicipalityTag ||
      activeMuniSummary?.slovakiana_tag_key ||
      activeMuniSummary?.municipality_key ||
      matchedCity?.tagKey ||
      matchedCity?.key ||
      ""
    ).trim();

    const distName =
      selectedDistrictName ||
      activeMuniSummary?.district ||
      muniOverview?.district ||
      "";
    const matchedDistrict = distName
      ? districts.find(
          (d) => foldMunicipalityKey(d.name) === foldMunicipalityKey(distName)
        )
      : null;
    const distKey = (
      selectedDistrictTag ||
      activeMuniSummary?.district_key ||
      matchedDistrict?.tagKey ||
      matchedDistrict?.key ||
      ""
    ).trim();

    const categoryKeys: string[] = [];
    if (distKey) categoryKeys.push(distKey);
    if (muniKey && !categoryKeys.includes(muniKey)) {
      categoryKeys.push(muniKey);
    }

    const filterPayload: Record<string, unknown> = {
      onlyUnBlackened: false,
      categoryKeys,
    };
    if (categoryKeys.length === 0) {
      filterPayload.searchText = selectedMunicipality;
    }

    return `https://slovakiana.sk/scitacie-harky?filter=${encodeURIComponent(
      JSON.stringify(filterPayload)
    )}&page=1&lm=0`;
  }, [
    selectedMunicipality,
    selectedMunicipalityTag,
    selectedDistrictTag,
    selectedDistrictName,
    activeMuniSummary,
    muniOverview,
    cities,
    districts,
  ]);

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === "running"),
    [jobs]
  );
  const runningJobsCount = activeJobs.length;
  const activeRunningJob = useMemo(
    () => activeJobs.find((j) => j.id !== dismissedActiveJobId) || null,
    [activeJobs, dismissedActiveJobId]
  );

  const [locationQuickFilter, setLocationQuickFilter] = useState<string>(() => {
    if (selectedMunicipality) return selectedMunicipality;
    if (selectedDistrictName) return `Okres ${selectedDistrictName}`;
    return "";
  });

  useEffect(() => {
    if (selectedMunicipality) {
      setLocationQuickFilter(selectedMunicipality);
    } else if (selectedDistrictName) {
      setLocationQuickFilter(`Okres ${selectedDistrictName}`);
    } else {
      setLocationQuickFilter("");
    }
  }, [selectedMunicipality, selectedDistrictName]);

  const handleLocationQuickFilterChange = (val: string) => {
    setLocationQuickFilter(val);
    const trimmed = val.trim();
    if (!trimmed) {
      setSelectedMunicipality("");
      setSelectedMunicipalityTag("");
      setSelectedDistrictName("");
      setSelectedDistrictTag("");
      return;
    }
    const okrMatch = trimmed.match(/^okr(?:es)?\.?\s+(.+)$/i);
    const folded = foldMunicipalityKey(okrMatch ? okrMatch[1] : trimmed);
    const matchedFacetDist = districts.find(
      (d) => foldMunicipalityKey(d.name) === folded
    );
    if (okrMatch || matchedFacetDist || KNOWN_DISTRICT_KEYS.has(folded)) {
      const distName =
        matchedFacetDist?.name ||
        (okrMatch ? okrMatch[1].trim() : trimmed);
      setSelectedDistrictName(distName);
      setSelectedDistrictTag(
        matchedFacetDist?.tagKey || matchedFacetDist?.key || ""
      );
      setSelectedMunicipality("");
      setSelectedMunicipalityTag("");
    } else {
      const matchedMuni = combinedMunicipalities.find(
        (m) => foldMunicipalityKey(m.municipality) === folded
      );
      setSelectedMunicipality(matchedMuni ? matchedMuni.municipality : trimmed);
      setSelectedMunicipalityTag(
        matchedMuni?.slovakiana_tag_key || matchedMuni?.municipality_key || ""
      );
    }
  };

  const streetSuggestions = useMemo(() => {
    const map = new Map<string, string>();
    const addCandidate = (val?: string | null) => {
      if (!val || !val.trim()) return;
      const t = val.trim();
      if (t === "-" || t === "—" || t === "?") return;
      const k = foldText(t);
      if (k && !map.has(k)) {
        map.set(k, t);
      }
    };
    for (const st of availableStreets) addCandidate(st);
    for (const st of muniOverview?.available_streets || []) addCandidate(st);
    for (const row of sheets) addCandidate(row.sheet.street_name);
    for (const h of muniOverview?.houses_directory || []) {
      addCandidate(h.street_name);
      addCandidate(h.sheet?.street_name);
      addCandidate(h.settlement_name);
    }
    return Array.from(map.values()).sort((a, b) => a.localeCompare(b, "sk"));
  }, [availableStreets, muniOverview, sheets]);

  const hasAnyActiveFilter = Boolean(
    selectedMunicipality ||
      selectedDistrictName ||
      selectedYearFilter ||
      streetFilter.trim() ||
      personNameFilter.trim() ||
      birthYearFilter.trim() ||
      birthPlaceFilter.trim() ||
      workspaceSearch.trim() ||
      cardSubFilter !== "all"
  );

  const handleClearAllFilters = () => {
    setSelectedMunicipality("");
    setSelectedMunicipalityTag("");
    setSelectedDistrictName("");
    setSelectedDistrictTag("");
    setSelectedYearFilter("");
    setLocationQuickFilter("");
    setStreetFilter("");
    setPersonNameFilter("");
    setBirthYearFilter("");
    setBirthPlaceFilter("");
    setWorkspaceSearch("");
    setCardSubFilter("all");
  };

  const hasActivePersonFilter = Boolean(
    personNameFilter.trim() ||
      birthYearFilter.trim() ||
      birthPlaceFilter.trim() ||
      workspaceSearch.trim()
  );

  const [cardsPage, setCardsPage] = useState<number>(1);
  const CARDS_PER_PAGE = 24;
  useEffect(() => {
    setCardsPage(1);
  }, [
    selectedMunicipality,
    selectedDistrictName,
    selectedYearFilter,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
    workspaceSearch,
    cardSubFilter,
  ]);
  const totalCardPages = Math.max(1, Math.ceil(filteredSheets.length / CARDS_PER_PAGE));
  const safeCardsPage = Math.min(cardsPage, totalCardPages);
  const pagedSheets = filteredSheets.slice(
    (safeCardsPage - 1) * CARDS_PER_PAGE,
    safeCardsPage * CARDS_PER_PAGE
  );

  const activePersonQuery = (
    personNameFilter.trim() || workspaceSearch.trim()
  ).trim();
  const buildSheetDetailLink = (
    sheetRef: string | number,
    opts?: { rowId?: number | null; houseId?: number | null }
  ): string => {
    const sp = new URLSearchParams();
    if (opts?.rowId) sp.set("row", String(opts.rowId));
    if (opts?.houseId) sp.set("house", String(opts.houseId));
    if (activePersonQuery) sp.set("meno", activePersonQuery);
    const qs = sp.toString();
    return `/scitacie-harky/${sheetRef}${qs ? `?${qs}` : ""}`;
  };

  const renderHouseholdMembersBadges = (
    members: ExtractedPersonPreview[] | undefined
  ) => {
    if (!members || members.length === 0) return null;
    const hasSpecificPersonFilter = Boolean(
      personNameFilter.trim() ||
        birthYearFilter.trim() ||
        birthPlaceFilter.trim()
    );
    const hasQuickQ = Boolean(workspaceSearch.trim());

    const annotated = members.map((m, i) => {
      const matched =
        Boolean(m.matched_filter) ||
        (hasSpecificPersonFilter &&
          memberMatchesPersonFilters(
            m,
            personNameFilter,
            birthYearFilter,
            birthPlaceFilter
          )) ||
        (hasQuickQ &&
          !hasSpecificPersonFilter &&
          (matchesPersonNameQuery(
            [
              m.given_name,
              m.surname,
              m.surname_normalized,
              m.maiden_surname,
              m.full_name,
            ],
            workspaceSearch
          ) ||
            memberMatchesBirthYear(m, workspaceSearch) ||
            memberMatchesBirthPlace(m, workspaceSearch)));
      return { m, matched, idx: i };
    });

    if (hasActivePersonFilter) {
      annotated.sort((a, b) => {
        if (a.matched !== b.matched) return a.matched ? -1 : 1;
        return a.idx - b.idx;
      });
    }

    const matchedCount = annotated.filter((a) => a.matched).length;
    const maxVisible = Math.max(5, matchedCount);
    const visible = annotated.slice(0, maxVisible);
    const hiddenCount = Math.max(0, annotated.length - visible.length);

    return (
      <div className="flex flex-wrap items-center gap-1 pt-1">
        {visible.map(({ m, matched, idx }) => {
          const fullName = `${m.given_name || ""} ${m.surname || ""}`.trim();
          const maidenPart =
            m.maiden_surname &&
            foldText(m.maiden_surname) !== foldText(m.surname)
              ? ` (rod. ${m.maiden_surname})`
              : "";
          const yearPart = m.birth_year ? ` *${m.birth_year}` : "";
          const shortPlace = formatShortBirthPlace(m.birth_place);
          const placePart =
            shortPlace && (matched || Boolean(birthPlaceFilter.trim()))
              ? ` · ${shortPlace}`
              : "";
          const tooltip = [
            `${fullName}${maidenPart}`,
            m.relation_to_head ? `Vzťah: ${m.relation_to_head}` : null,
            m.birth_date
              ? `Nar.: ${m.birth_date}`
              : m.birth_year
              ? `Rok nar.: ${m.birth_year}`
              : null,
            m.birth_place ? `Rodisko: ${m.birth_place}` : null,
            m.domicile ? `Domov: ${m.domicile}` : null,
          ]
            .filter(Boolean)
            .join(" | ");

          return (
            <span
              key={`${m.id ?? idx}-${fullName}`}
              title={tooltip}
              className={`px-2 py-0.5 rounded text-xs inline-flex items-center gap-1 border transition ${
                matched
                  ? "bg-emerald-100 text-emerald-950 border-emerald-400 ring-1 ring-emerald-300 font-bold"
                  : "bg-white text-stone-700 border-stone-200 font-medium"
              }`}
            >
              <span>
                {matched ? "★ " : ""}
                {fullName || (lang === "sk" ? "Osoba" : "Person")}
                {maidenPart}
                {yearPart}
                {placePart}
              </span>
            </span>
          );
        })}
        {hiddenCount > 0 && (
          <span className="px-2 py-0.5 rounded bg-stone-100 text-stone-700 border border-stone-200 text-xs font-semibold">
            +{hiddenCount} {lang === "sk" ? "ďalší" : "more"}
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
      {/* LEFT SIDEBAR: Okresy a obce */}
      <aside className="lg:col-span-3 bg-white rounded-xl border border-stone-300 p-3.5 space-y-3 lg:sticky lg:top-4">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-bold text-stone-900 flex items-center gap-1.5">
            <MapPin className="w-4 h-4 text-amber-600" />
            <span>
              {lang === "sk" ? "Okresy a obce" : "Districts & Municipalities"}
            </span>
          </h2>
          <a
            href="https://slovakiana.sk/scitacie-harky/?page=1&lm=0"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-semibold text-amber-800 hover:underline inline-flex items-center gap-1 py-1 px-1.5 min-h-[32px]"
          >
            <span>Slovakiana</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        {/* Year Filter Pills: Všetky | 1930 | 1940 */}
        <div className="grid grid-cols-3 gap-1 bg-stone-100 p-1 rounded-lg border border-stone-200 text-xs">
          {(
            [
              { label: lang === "sk" ? "Všetky" : "All", value: "" },
              { label: "1930", value: "1930" },
              { label: "1940", value: "1940" },
            ] as const
          ).map((yr) => {
            const active = selectedYearFilter === yr.value;
            return (
              <button
                key={yr.value || "all"}
                type="button"
                onClick={() => setSelectedYearFilter(yr.value)}
                className={`py-1.5 min-h-[32px] rounded-md font-bold transition cursor-pointer ${
                  active
                    ? "bg-stone-900 text-amber-300 shadow-xs"
                    : "text-stone-700 hover:bg-stone-200/70"
                }`}
              >
                {yr.label}
              </button>
            );
          })}
        </div>

        {/* District (`Okres`) Selector Dropdown */}
        <div>
          <label className="block text-xs font-semibold text-stone-700 mb-1">
            {lang === "sk" ? "Historický okres" : "Historical District"}
          </label>
          <select
            value={selectedDistrictTag || selectedDistrictName}
            onChange={(e) => {
              const val = e.target.value;
              if (!val) {
                setSelectedDistrictTag("");
                setSelectedDistrictName("");
                return;
              }
              const facetDist = districts.find(
                (d) => (d.tagKey || d.key) === val || d.name === val
              );
              if (facetDist) {
                setSelectedDistrictTag(facetDist.tagKey || facetDist.key || "");
                setSelectedDistrictName(facetDist.name);
              } else {
                setSelectedDistrictTag("");
                setSelectedDistrictName(val);
              }
            }}
            className="w-full px-2.5 py-1.5 min-h-[34px] rounded-lg border border-stone-300 text-xs sm:text-sm bg-white"
          >
            <option value="">
              {lang === "sk"
                ? "Všetky okresy na Slovensku"
                : "All Districts in Slovakia"}
            </option>
            {districts.length > 0
              ? districts.map((d) => {
                  const k = d.tagKey || d.key || d.name;
                  return (
                    <option key={k} value={k}>
                      {d.name} {d.count ? `(${d.count})` : ""}
                    </option>
                  );
                })
              : (stats?.local_districts || []).map((d) => (
                  <option key={d.district} value={d.district}>
                    {d.district} ({d.count})
                  </option>
                ))}
          </select>
        </div>

        {/* Search Input for Filtering Municipalities */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-stone-500 absolute left-2.5 top-2.5" />
          <input
            type="text"
            value={muniSearchQuery}
            onChange={(e) => setMuniSearchQuery(e.target.value)}
            placeholder={
              lang === "sk"
                ? "Filtrovať obec (napr. Trnava, Nitra)..."
                : "Filter municipality (e.g. Trnava, Nitra)..."
            }
            className="w-full pl-8 pr-2.5 py-1.5 min-h-[34px] rounded-lg border border-stone-300 text-xs sm:text-sm"
          />
        </div>

        {/* Scrollable List of Municipalities (`Obce`) */}
        <div className="space-y-2 max-h-[62vh] overflow-y-auto pr-0.5">
          {/* Top Item: Všetky obce */}
          <button
            type="button"
            onClick={() => {
              setSelectedMunicipality("");
              setSelectedMunicipalityTag("");
              setActiveTab("sheets");
            }}
            className={`w-full text-left rounded-lg p-3 border transition cursor-pointer flex items-center justify-between gap-2 ${
              selectedMunicipality === ""
                ? "bg-stone-900 text-white border-stone-900"
                : "bg-stone-50 hover:bg-stone-100 text-stone-800 border-stone-200"
            }`}
          >
            <div className="min-w-0">
              <div className="font-bold text-xs sm:text-sm flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-amber-400 shrink-0" />
                <span>{lang === "sk" ? "Všetky obce" : "All Municipalities"}</span>
              </div>
              <div
                className={`text-xs mt-0.5 ${
                  selectedMunicipality === "" ? "text-stone-300" : "text-stone-600"
                }`}
              >
                {lang === "sk"
                  ? "Celý lokálny katalóg sčítania"
                  : "Entire local census catalog"}
              </div>
            </div>
            <span
              className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                selectedMunicipality === ""
                  ? "bg-amber-500 text-stone-950"
                  : "bg-stone-200 text-stone-800"
              }`}
            >
              {stats?.total_cataloged_sheets ?? totalSheets}
            </span>
          </button>

          {combinedMunicipalities.map((m) => {
            const isSelected =
              Boolean(selectedMunicipality) &&
              foldMunicipalityKey(selectedMunicipality) ===
                foldMunicipalityKey(m.municipality);
            const isHarvesting = busyAction === `harvest_${m.municipality}`;
            return (
              <div
                key={`${m.municipality}-${m.district}`}
                onClick={() => {
                  setSelectedMunicipality(m.municipality);
                  setSelectedMunicipalityTag(
                    m.slovakiana_tag_key || m.municipality_key || ""
                  );
                  setActiveTab("overview");
                }}
                className={`rounded-lg p-3 border transition cursor-pointer space-y-2 ${
                  isSelected
                    ? "bg-amber-50 border-amber-500 ring-1 ring-amber-400"
                    : "bg-white hover:bg-stone-50 border-stone-200"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-bold text-xs sm:text-sm text-stone-900">
                      {m.municipality}
                    </div>
                    {m.district && (
                      <div className="text-xs text-stone-600">
                        {lang === "sk" ? "okr." : "dist."} {m.district}
                      </div>
                    )}
                  </div>

                  {/* 1-click "Stiahnuť z archívu" action button */}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleHarvestMunicipality(
                        m.municipality,
                        m.slovakiana_tag_key || m.municipality_key
                      );
                    }}
                    disabled={busyAction !== null}
                    title={
                      lang === "sk"
                        ? "Stiahnuť hárky obce z archívu Slovakiana"
                        : "Download municipality sheets from Slovakiana archive"
                    }
                    className="px-2.5 py-1.5 min-h-[32px] rounded-md bg-stone-900 hover:bg-stone-700 text-amber-300 text-xs font-semibold inline-flex items-center gap-1 shrink-0 cursor-pointer disabled:opacity-50"
                  >
                    {isHarvesting ? (
                      <RefreshCw className="w-3 h-3 animate-spin" />
                    ) : (
                      <FolderDown className="w-3 h-3" />
                    )}
                    <span>
                      {lang === "sk" ? "Stiahnuť z archívu" : "Fetch from archive"}
                    </span>
                  </button>
                </div>

                {/* Counts pills: local sheets / OCR-done / overview houses / Slovakiana */}
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="px-2 py-0.5 rounded bg-stone-100 text-stone-800 font-semibold">
                    {lang === "sk" ? "V DB:" : "In DB:"} {m.total_sheets}
                  </span>
                  {m.ocr_done_sheets > 0 && (
                    <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 font-semibold">
                      OCR: {m.ocr_done_sheets}
                    </span>
                  )}
                  {m.overview_houses_count > 0 && (
                    <span className="px-2 py-0.5 rounded bg-purple-100 text-purple-900 font-semibold">
                      {lang === "sk" ? "Domy:" : "Houses:"} {m.overview_houses_count}
                    </span>
                  )}
                  {m.slovakiana_count != null && (
                    <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-semibold">
                      {lang === "sk" ? "Archív:" : "Archive:"} {m.slovakiana_count}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      {/* RIGHT MAIN WORKSPACE */}
      <section className="lg:col-span-9 space-y-4">
        {/* Unified Header & Action Bar */}
        <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
          {/* Row 1: Identity & Primary Action Toolbar */}
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-lg font-bold text-stone-900">
                  {selectedMunicipality
                    ? `${selectedMunicipality}${
                        activeMuniSummary?.district || muniOverview?.district
                          ? ` (${lang === "sk" ? "okres" : "district"} ${
                              activeMuniSummary?.district || muniOverview?.district
                            })`
                          : ""
                      }`
                    : lang === "sk"
                    ? "Všetky obce v databáze"
                    : "All Municipalities in Database"}
                </h1>
                {selectedYearFilter && (
                  <span className="px-2.5 py-0.5 rounded-full bg-emerald-100 text-emerald-900 text-xs font-bold">
                    Rok {selectedYearFilter}
                  </span>
                )}
                {Boolean(selectedMunicipality) && slovakianaMunicipalityUrl && (
                  <a
                    href={slovakianaMunicipalityUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs font-semibold text-amber-800 hover:underline inline-flex items-center gap-1 py-1 px-1.5 min-h-[32px]"
                  >
                    <span>Slovakiana</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>

              {/* Inline KPI Strip */}
              <div className="flex flex-wrap items-center gap-1.5 mt-1.5 text-xs">
                <span className="px-2 py-0.5 rounded-md bg-stone-100 border border-stone-200 font-medium text-stone-700">
                  {lang === "sk" ? "Hárky v DB:" : "Sheets in DB:"}{" "}
                  <strong className="font-bold text-stone-900">
                    {activeMuniSummary
                      ? activeMuniSummary.total_sheets
                      : totalSheets}
                  </strong>
                </span>
                {featureFlags.census_overview_sheets && (
                  <>
                    <span className="px-2 py-0.5 rounded-md bg-purple-50 border border-purple-200 font-medium text-purple-900">
                      {lang === "sk" ? "Prehľady obvodov:" : "District overviews:"}{" "}
                      <strong className="font-bold">
                        {muniOverview?.overview_sheets.length ?? 0}
                      </strong>
                    </span>
                    <span className="px-2 py-0.5 rounded-md bg-amber-50 border border-amber-200 font-medium text-amber-900">
                      {lang === "sk" ? "Domy v prehľade:" : "Houses in overview:"}{" "}
                      <strong className="font-bold">
                        {muniOverview?.total_houses ?? 0}
                      </strong>
                    </span>
                  </>
                )}
                <span className="px-2 py-0.5 rounded-md bg-emerald-50 border border-emerald-200 font-medium text-emerald-900">
                  {lang === "sk" ? "Prečítané osoby:" : "Extracted persons:"}{" "}
                  <strong className="font-bold">
                    {activeMuniSummary
                      ? activeMuniSummary.extracted_persons_count
                      : stats?.extracted_census_persons ?? 0}
                  </strong>
                </span>
              </div>
            </div>

            {/* Unified Split-Button Action Group & Jobs History Popover */}
            <div className="flex flex-wrap items-center justify-end gap-2">
              <div ref={splitMenuRef} className="relative inline-flex">
                <div className="inline-flex items-stretch rounded-lg border border-stone-300 bg-white shadow-2xs overflow-hidden">
                  <label className="inline-flex items-center gap-1.5 px-2.5 py-1.5 min-h-[34px] bg-stone-50 border-r border-stone-300 text-xs">
                    <span className="text-stone-600 font-medium">
                      {lang === "sk" ? "Dávka:" : "Batch:"}
                    </span>
                    <input
                      type="number"
                      min={1}
                      max={2000}
                      value={maxSheetsInput}
                      onChange={(e) => setMaxSheetsInput(Number(e.target.value) || 40)}
                      className="w-14 min-h-[28px] bg-white rounded border border-stone-300 px-1 text-xs font-bold text-stone-900 focus:outline-none text-center"
                    />
                  </label>

                  <button
                    type="button"
                    onClick={() => {
                      setSplitMenuOpen(false);
                      handleBulkOcrWorkspace();
                    }}
                    disabled={busyAction !== null}
                    title={
                      lang === "sk"
                        ? "Stiahnuť hárky zo Slovakiany a spustiť hromadné OCR"
                        : "Download sheets from Slovakiana and run batch OCR"
                    }
                    className="px-3.5 py-1.5 min-h-[34px] bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50 transition"
                  >
                    {busyAction === "bulk_ocr" ? (
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {lang === "sk"
                        ? "Stiahnuť a spustiť OCR"
                        : "Download & Run OCR"}
                    </span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setSplitMenuOpen((prev) => !prev)}
                    disabled={busyAction !== null}
                    title={
                      lang === "sk"
                        ? "Ďalšie možnosti: Iba načítať katalóg zo Slovakiany"
                        : "More options: Only fetch catalog from Slovakiana"
                    }
                    aria-label={
                      lang === "sk"
                        ? "Ďalšie možnosti sťahovania"
                        : "More download options"
                    }
                    className="px-2.5 py-1.5 min-h-[34px] min-w-[34px] bg-amber-700 hover:bg-amber-800 text-white border-l border-amber-500/50 inline-flex items-center justify-center cursor-pointer disabled:opacity-50 transition"
                  >
                    {splitMenuOpen ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                </div>

                {splitMenuOpen && (
                  <div className="absolute right-0 top-full mt-1.5 z-30 w-72 bg-white rounded-lg border border-stone-200 shadow-xl p-1">
                    <button
                      type="button"
                      onClick={() => {
                        setSplitMenuOpen(false);
                        handleHarvestMunicipality();
                      }}
                      disabled={busyAction !== null}
                      className="w-full text-left px-3 py-2 rounded-md hover:bg-stone-100 text-stone-800 text-xs font-semibold flex items-start gap-2 cursor-pointer disabled:opacity-50"
                    >
                      <FolderDown className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                      <div>
                        <div>
                          {lang === "sk"
                            ? "Iba načítať katalóg zo Slovakiany"
                            : "Only fetch catalog from Slovakiana"}
                        </div>
                        <div className="text-[11px] font-normal text-stone-500">
                          {lang === "sk"
                            ? "Stiahne zoznam a metadáta hárkov obce bez spustenia OCR"
                            : "Fetches sheet list and metadata without starting OCR"}
                        </div>
                      </div>
                    </button>
                  </div>
                )}
              </div>

              {jobs.length > 0 && (
                <div ref={jobsPopoverRef} className="relative">
                  <button
                    type="button"
                    onClick={() => setJobsExpanded((prev) => !prev)}
                    className={`px-2.5 py-1.5 rounded-lg border text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer transition ${
                      runningJobsCount > 0
                        ? "bg-amber-50 border-amber-300 text-amber-900"
                        : "bg-stone-100 hover:bg-stone-200 border-stone-300 text-stone-700"
                    }`}
                  >
                    {runningJobsCount > 0 && (
                      <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                    )}
                    <span>
                      {runningJobsCount > 0
                        ? `Úlohy (${runningJobsCount}/${jobs.length})`
                        : `História úloh (${jobs.length})`}
                    </span>
                    {jobsExpanded ? (
                      <ChevronUp className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" />
                    )}
                  </button>

                  {jobsExpanded && (
                    <div className="absolute right-0 top-full mt-1.5 z-30 w-96 bg-white rounded-xl border border-stone-300 shadow-xl p-3 space-y-2">
                      <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-stone-500 pb-1 border-b border-stone-200">
                        <span>História úloh ({jobs.length})</span>
                        <button
                          type="button"
                          onClick={() => setJobsExpanded(false)}
                          className="text-stone-500 hover:text-stone-800 inline-flex items-center gap-0.5 cursor-pointer"
                        >
                          <span>Skryť</span>
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="max-h-72 overflow-y-auto space-y-1.5 pr-0.5">
                        {jobs.slice(0, 8).map((job) => (
                          <div
                            key={job.id}
                            className="bg-stone-50 rounded-lg px-2.5 py-2 border border-stone-200 space-y-1 text-xs"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="flex items-center gap-1.5 min-w-0">
                                <span
                                  className={`px-2 py-0.5 rounded-full font-bold text-xs shrink-0 ${
                                    job.status === "completed"
                                      ? "bg-emerald-100 text-emerald-900"
                                      : job.status === "running"
                                      ? "bg-amber-100 text-amber-900"
                                      : job.status === "error"
                                      ? "bg-red-100 text-red-900"
                                      : "bg-stone-200 text-stone-800"
                                  }`}
                                >
                                  {formatJobStatusLabel(job.status)}
                                </span>
                                <span className="font-semibold text-stone-900 truncate">
                                  {job.name}
                                </span>
                              </div>
                              <div className="flex items-center gap-1 shrink-0">
                                {(job.status === "running" || job.status === "paused") && (
                                  <button
                                    type="button"
                                    onClick={() => handlePauseResumeJob(job)}
                                    className="px-2 py-1 rounded bg-stone-800 hover:bg-stone-700 text-white text-xs font-semibold inline-flex items-center gap-1 cursor-pointer shrink-0"
                                  >
                                    {job.status === "paused" ? (
                                      <>
                                        <Play className="w-3 h-3" />
                                        <span>{lang === "sk" ? "Pokračovať" : "Resume"}</span>
                                      </>
                                    ) : (
                                      <>
                                        <Pause className="w-3 h-3" />
                                        <span>{lang === "sk" ? "Pozastaviť" : "Pause"}</span>
                                      </>
                                    )}
                                  </button>
                                )}
                                <button
                                  type="button"
                                  onClick={() => handleDeleteJob(job.id)}
                                  title={lang === "sk" ? "Odstrániť úlohu z histórie" : "Remove job"}
                                  className="p-1 rounded hover:bg-stone-200 text-stone-500 hover:text-red-700 cursor-pointer"
                                >
                                  <X className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </div>
                            <div className="text-xs text-stone-600">
                              Katalóg: {job.cataloged_sheets} · Stiahnuté:{" "}
                              {job.downloaded_sheets} · Prečítané: {job.processed_sheets}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Slim 1-Line Active Job Progress Bar (only when running or paused and not dismissed) */}
          {activeRunningJob && (
            <div className="px-3 py-1.5 rounded-lg bg-amber-50/80 border border-amber-200 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 min-w-0 flex-1">
                {activeRunningJob.status === "running" ? (
                  <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin shrink-0" />
                ) : (
                  <Pause className="w-3.5 h-3.5 text-stone-600 shrink-0" />
                )}
                <span className="font-bold text-stone-900 truncate">
                  {activeRunningJob.name}
                </span>
                <div className="w-24 sm:w-32 h-1.5 bg-amber-200/80 rounded-full overflow-hidden shrink-0">
                  <div
                    className="h-full bg-amber-600 rounded-full transition-all"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.round(
                          ((activeRunningJob.job_type === "catalog_harvest"
                            ? activeRunningJob.cataloged_sheets
                            : Math.max(
                                activeRunningJob.processed_sheets,
                                activeRunningJob.downloaded_sheets
                              )) /
                            Math.max(
                              activeRunningJob.total_sheets || 0,
                              activeRunningJob.cataloged_sheets || 0,
                              1
                            )) *
                            100
                        )
                      )}%`,
                    }}
                  />
                </div>
                <span className="text-stone-600 truncate">
                  Katalóg: <strong>{activeRunningJob.cataloged_sheets}</strong> · Stiahnuté:{" "}
                  <strong>
                    {activeRunningJob.downloaded_sheets}
                    {activeRunningJob.total_sheets
                      ? `/${activeRunningJob.total_sheets}`
                      : ""}
                  </strong>{" "}
                  · Prečítané: <strong>{activeRunningJob.processed_sheets}</strong>
                </span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  type="button"
                  onClick={() => handlePauseResumeJob(activeRunningJob)}
                  className="px-2 py-0.5 rounded bg-stone-800 hover:bg-stone-700 text-white text-[11px] font-semibold inline-flex items-center gap-1 cursor-pointer"
                >
                  {activeRunningJob.status === "paused" ? (
                    <>
                      <Play className="w-3 h-3" />
                      <span>Pokračovať</span>
                    </>
                  ) : (
                    <>
                      <Pause className="w-3 h-3" />
                      <span>Pozastaviť</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => setDismissedActiveJobId(activeRunningJob.id)}
                  title="Skryť panel prebiehajúcej úlohy"
                  className="px-1.5 py-0.5 rounded hover:bg-amber-200/60 text-stone-600 hover:text-stone-900 text-[11px] font-semibold inline-flex items-center gap-0.5 cursor-pointer"
                >
                  <span>Skryť</span>
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Row 2: Unified Search Input, Person/Birth-Year/Village Filters & Workspace Sub-Tabs */}
          <div className="space-y-2.5 pt-2.5 border-t border-stone-200">
            <div className="flex flex-wrap items-center justify-between gap-2.5">
              {/* Sub-Tabs */}
              <div className="inline-flex rounded-lg bg-stone-100 p-1 border border-stone-200">
                {Boolean(selectedMunicipality) &&
                  featureFlags.census_overview_sheets && (
                    <button
                      type="button"
                      onClick={() => setActiveTab("overview")}
                      className={`px-3.5 py-1.5 min-h-[34px] rounded-md text-xs font-bold inline-flex items-center gap-1.5 cursor-pointer transition ${
                        activeTab === "overview"
                          ? "bg-stone-900 text-amber-300 shadow-xs"
                          : "text-stone-700 hover:text-stone-900"
                      }`}
                    >
                      <Building2 className="w-3.5 h-3.5" />
                      <span>
                        {lang === "sk"
                          ? "Prehľad obce, obvodov a domov"
                          : "Municipality, Districts & Houses"}
                      </span>
                    </button>
                  )}
                <button
                  type="button"
                  onClick={() => setActiveTab("sheets")}
                  className={`px-3.5 py-1.5 min-h-[34px] rounded-md text-xs font-bold inline-flex items-center gap-1.5 cursor-pointer transition ${
                    !selectedMunicipality ||
                    !featureFlags.census_overview_sheets ||
                    activeTab === "sheets"
                      ? "bg-stone-900 text-amber-300 shadow-xs"
                      : "text-stone-700 hover:text-stone-900"
                  }`}
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>
                    {lang === "sk"
                      ? `Všetky hárky (Karty) (${totalSheets})`
                      : `All Sheets (Cards) (${totalSheets})`}
                  </span>
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2 flex-1 justify-end min-w-[260px]">
                {/* General Quick Search Input */}
                <div className="relative flex-1 min-w-[220px] max-w-sm">
                  <Search className="w-3.5 h-3.5 text-stone-500 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    value={workspaceSearch}
                    onChange={(e) => setWorkspaceSearch(e.target.value)}
                    placeholder={
                      lang === "sk"
                        ? "Rýchle hľadanie (dom č., majiteľ, cair-...)..."
                        : "Quick search (house #, owner, cair-...)..."
                    }
                    className="w-full pl-8 pr-8 py-1.5 min-h-[34px] rounded-lg border border-stone-300 text-xs sm:text-sm"
                  />
                  {workspaceSearch && (
                    <button
                      type="button"
                      onClick={() => setWorkspaceSearch("")}
                      className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                      title={
                        lang === "sk"
                          ? "Vyčistiť rýchle hľadanie"
                          : "Clear quick search"
                      }
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                {hasAnyActiveFilter && (
                  <button
                    type="button"
                    onClick={handleClearAllFilters}
                    className="px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300 text-xs font-bold inline-flex items-center gap-1 cursor-pointer shrink-0"
                    title={
                      lang === "sk"
                        ? "Zrušiť všetky aktívne filtre (obec, okres, meno, rok narodenia)"
                        : "Clear all active filters"
                    }
                  >
                    <X className="w-3.5 h-3.5" />
                    <span>{lang === "sk" ? "Zrušiť filtre" : "Clear filters"}</span>
                  </button>
                )}
              </div>
            </div>

            {/* Structured Filter Bar: Obec/Okres | Ulica/Osada | Meno/Priezvisko | Rok narodenia | Rodisko */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2 bg-stone-50/90 p-2 rounded-lg border border-stone-200">
              {/* 1. Obec / Okres quick filter */}
              <div className="relative">
                <MapPin className="w-3.5 h-3.5 text-amber-600 absolute left-2.5 top-2.5 pointer-events-none" />
                <input
                  type="text"
                  list="census-muni-district-datalist"
                  value={locationQuickFilter}
                  onChange={(e) => handleLocationQuickFilterChange(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Obec / okres (napr. Trnava)..."
                      : "Municipality / district..."
                  }
                  aria-label="Filtrovať podľa obce alebo okresu"
                  className="w-full pl-8 pr-8 py-1.5 min-h-[34px] rounded-md border border-stone-300 bg-white text-xs sm:text-sm"
                />
                {locationQuickFilter && (
                  <button
                    type="button"
                    onClick={() => handleLocationQuickFilterChange("")}
                    className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                    title="Zrušiť filter obce / okresu"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
                <datalist id="census-muni-district-datalist">
                  {combinedMunicipalities
                    .filter((m) => {
                      if (m.total_sheets > 0) return true;
                      const q = foldMunicipalityKey(locationQuickFilter);
                      return q.length >= 2 && foldMunicipalityKey(m.municipality).includes(q);
                    })
                    .slice(0, 60)
                    .map((m) => (
                      <option
                        key={`muni-${m.municipality}-${m.district}`}
                        value={m.municipality}
                      >
                        {m.district ? `okr. ${m.district}` : "Obec"}
                      </option>
                    ))}
                  {(districts.length > 0
                    ? districts.map((d) => d.name)
                    : (stats?.local_districts || []).map((d) => d.district)
                  ).map((dName) => (
                    <option key={`dist-${dName}`} value={`Okres ${dName}`} />
                  ))}
                </datalist>
              </div>

              {/* 2. Ulica / Osada filter */}
              <div className="relative">
                <Building2 className="w-3.5 h-3.5 text-purple-700 absolute left-2.5 top-2.5 pointer-events-none" />
                <input
                  type="text"
                  list="census-street-datalist"
                  value={streetFilter}
                  onChange={(e) => setStreetFilter(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Ulica / osada (napr. Hlavná)..."
                      : "Street / settlement (e.g. Hlavná)..."
                  }
                  aria-label={
                    lang === "sk"
                      ? "Filtrovať podľa ulice alebo osady"
                      : "Filter by street or settlement"
                  }
                  className="w-full pl-8 pr-8 py-1.5 min-h-[34px] rounded-md border border-stone-300 bg-white text-xs sm:text-sm"
                />
                {streetFilter && (
                  <button
                    type="button"
                    onClick={() => setStreetFilter("")}
                    className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                    title={
                      lang === "sk"
                        ? "Vyčistiť filter ulice / osady"
                        : "Clear street / settlement filter"
                    }
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
                <datalist id="census-street-datalist">
                  {streetSuggestions.slice(0, 80).map((stName) => (
                    <option key={`street-${stName}`} value={stName} />
                  ))}
                </datalist>
              </div>

              {/* 3. Meno / Priezvisko filter */}
              <div className="relative">
                <Users className="w-3.5 h-3.5 text-emerald-700 absolute left-2.5 top-2.5 pointer-events-none" />
                <input
                  type="text"
                  value={personNameFilter}
                  onChange={(e) => setPersonNameFilter(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Meno / priezvisko (napr. Kováč)..."
                      : "Name / surname (e.g. Kováč)..."
                  }
                  aria-label="Filtrovať podľa mena alebo priezviska osoby"
                  className="w-full pl-8 pr-8 py-1.5 min-h-[34px] rounded-md border border-stone-300 bg-white text-xs sm:text-sm"
                />
                {personNameFilter && (
                  <button
                    type="button"
                    onClick={() => setPersonNameFilter("")}
                    className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                    title="Vyčistiť filter mena"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* 4. Rok narodenia filter */}
              <div className="relative">
                <span className="text-xs font-bold text-amber-800 absolute left-2.5 top-2 pointer-events-none">
                  {lang === "sk" ? "*Rok" : "*Yr"}
                </span>
                <input
                  type="text"
                  value={birthYearFilter}
                  onChange={(e) => setBirthYearFilter(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Rok nar. (napr. 1900, 1900±2)..."
                      : "Birth year (e.g. 1900, 1900±2)..."
                  }
                  aria-label="Filtrovať podľa roku narodenia"
                  className="w-full pl-10 pr-8 py-1.5 min-h-[34px] rounded-md border border-stone-300 bg-white text-xs sm:text-sm"
                />
                {birthYearFilter && (
                  <button
                    type="button"
                    onClick={() => setBirthYearFilter("")}
                    className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                    title="Vyčistiť filter roku narodenia"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* 5. Rodisko / Domov (obec/okres) filter */}
              <div className="relative">
                <Home className="w-3.5 h-3.5 text-indigo-700 absolute left-2.5 top-2.5 pointer-events-none" />
                <input
                  type="text"
                  value={birthPlaceFilter}
                  onChange={(e) => setBirthPlaceFilter(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Rodisko / domov (napr. Trnava)..."
                      : "Birthplace / domicile..."
                  }
                  aria-label="Filtrovať podľa rodiska alebo domovskej príslušnosti"
                  className="w-full pl-8 pr-8 py-1.5 min-h-[34px] rounded-md border border-stone-300 bg-white text-xs sm:text-sm"
                />
                {birthPlaceFilter && (
                  <button
                    type="button"
                    onClick={() => setBirthPlaceFilter("")}
                    className="absolute right-1 top-1 p-1.5 text-stone-500 hover:text-stone-800 cursor-pointer"
                    title="Vyčistiť filter rodiska"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Census Search Progress Bar */}
            {(() => {
              const effTotalSheets = Math.max(
                scannedSheetsTotal,
                stats?.total_cataloged_sheets ?? 0,
                totalSheets
              );
              const censusPct = isSearchingCensus
                ? Math.max(10, Math.min(98, censusSearchProgress.progress_pct || 20))
                : 100;
              const censusQuerySummary = [
                selectedMunicipality || (selectedDistrictName ? `okr. ${selectedDistrictName}` : ""),
                selectedYearFilter,
                streetFilter.trim()
                  ? `${lang === "sk" ? "ul." : "st."} ${streetFilter.trim()}`
                  : "",
                personNameFilter.trim(),
                birthYearFilter.trim() ? `*${birthYearFilter.trim()}` : "",
                birthPlaceFilter.trim(),
                workspaceSearch.trim(),
              ]
                .filter(Boolean)
                .join(" · ");

              return (
                <div
                  data-testid="census-search-progress-card"
                  className={`rounded-lg px-3 py-2 border space-y-1.5 transition-colors ${
                    isSearchingCensus
                      ? "bg-amber-50 border-amber-300"
                      : "bg-stone-100/70 border-stone-200"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <div className="flex items-center gap-2 font-bold text-stone-900 min-w-0">
                      {isSearchingCensus ? (
                        <RefreshCw className="w-3.5 h-3.5 text-amber-700 animate-spin shrink-0" />
                      ) : (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
                      )}
                      <span className="truncate">
                        {isSearchingCensus
                          ? lang === "sk"
                            ? `Prehľadávam sčítacie hárky a prehľady obvodov${
                                censusQuerySummary ? ` („${censusQuerySummary}“)` : ""
                              }…`
                            : `Searching census sheets and district overviews${
                                censusQuerySummary ? ` ("${censusQuerySummary}")` : ""
                              }…`
                          : hasAnyActiveFilter
                          ? lang === "sk"
                            ? `Vyhľadávanie dokončené — Nájdených ${filteredSheets.length} hárkov${
                                selectedMunicipality
                                  ? ` · ${filteredHousesDirectory.length} domov v adresári`
                                  : ""
                              }`
                            : `Search complete — Found ${filteredSheets.length} sheets${
                                selectedMunicipality
                                  ? ` · ${filteredHousesDirectory.length} houses in directory`
                                  : ""
                              }`
                          : lang === "sk"
                          ? `Katalóg sčítacích hárkov pripravený — Prehľadaných ${effTotalSheets} hárkov v databáze`
                          : `Census catalog ready — Scanned ${effTotalSheets} sheets in database`}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-stone-600 font-medium">
                        {lang === "sk"
                          ? `Prehľadané: ${
                              isSearchingCensus ? censusSearchProgress.processed : effTotalSheets
                            } / ${effTotalSheets}`
                          : `Scanned: ${
                              isSearchingCensus ? censusSearchProgress.processed : effTotalSheets
                            } / ${effTotalSheets}`}
                        {!isSearchingCensus && censusSearchProgress.elapsed_ms > 0
                          ? ` (${censusSearchProgress.elapsed_ms} ms)`
                          : ""}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                          isSearchingCensus
                            ? "bg-amber-600 text-white"
                            : "bg-emerald-100 text-emerald-900"
                        }`}
                      >
                        {censusPct}%
                      </span>
                    </div>
                  </div>

                  <div
                    role="progressbar"
                    aria-label={
                      lang === "sk"
                        ? "Priebeh vyhľadávania v sčítacích hárkoch"
                        : "Census sheets search progress"
                    }
                    aria-valuenow={censusPct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    data-testid="census-search-progress-bar"
                    className={`w-full rounded-full h-2 overflow-hidden ${
                      isSearchingCensus ? "bg-amber-200/80" : "bg-stone-200"
                    }`}
                  >
                    <div
                      className={`h-full transition-all duration-200 ${
                        isSearchingCensus ? "bg-amber-600 animate-pulse" : "bg-emerald-600"
                      }`}
                      style={{ width: `${censusPct}%` }}
                    />
                  </div>
                </div>
              );
            })()}
          </div>
        </div>

        {/* TAB 1: Prehľad obce, obvodov a domov */}
        {Boolean(selectedMunicipality) &&
        featureFlags.census_overview_sheets &&
        activeTab === "overview" ? (
          <div className="space-y-4">
            {/* Top Section: Prehľady sčítacích obvodov cards */}
            {!muniOverview?.overview_sheets ||
            muniOverview.overview_sheets.length === 0 ? (
              <div className="bg-white rounded-xl px-4 py-2.5 border border-stone-300 flex flex-wrap items-center justify-between gap-2 text-xs sm:text-sm">
                <div className="flex items-center gap-2 font-bold text-stone-800">
                  <Building2 className="w-4 h-4 text-purple-700 shrink-0" />
                  <span>
                    {lang === "sk"
                      ? "Prehľady sčítacích obvodov (0)"
                      : "District Overview Sheets (0)"}
                  </span>
                </div>
                <span className="text-xs text-stone-600">
                  {lang === "sk" ? (
                    <>
                      Pre túto obec zatiaľ nie je klasifikovaný úvodný hárok obvodu (č. 0001).
                    </>
                  ) : (
                    <>
                      No introductory District Overview Sheet (#0001) classified yet for this municipality.
                    </>
                  )}
                </span>
              </div>
            ) : (
            <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                    <Building2 className="w-5 h-5 text-purple-700" />
                    <span>
                      {lang === "sk"
                        ? "Prehľady sčítacích obvodov"
                        : "District Overview Sheets"}{" "}
                      ({muniOverview?.overview_sheets.length ?? 0})
                    </span>
                  </h2>
                  <p className="text-xs sm:text-sm text-stone-600">
                    {lang === "sk"
                      ? "Sumárne hárky sčítacích komisárov so zoznamom všetkých domov, majiteľov a počtom prítomných osôb v obci."
                      : "Summary sheets by census commissioners listing all houses, owners, and present residents in the municipality."}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {muniOverview.overview_sheets.map((ov) => (
                    <div
                      key={ov.slovakiana_id}
                      className="bg-purple-50/50 rounded-xl p-3.5 border border-purple-200 flex flex-col justify-between gap-3"
                    >
                      <div className="flex gap-3">
                        {ov.thumbnail_url ? (
                          <img
                            src={ov.thumbnail_url}
                            alt={ov.slovakiana_id}
                            className="w-20 h-16 object-cover rounded-lg border border-purple-300 bg-stone-200 shrink-0"
                          />
                        ) : (
                          <div className="w-20 h-16 rounded-lg border border-purple-300 bg-purple-100 flex items-center justify-center text-xs font-bold text-purple-900 shrink-0">
                            Obvod
                          </div>
                        )}
                        <div className="min-w-0 flex-1 space-y-1">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-mono font-bold">
                              {ov.slovakiana_id}
                            </span>
                            <span className="px-2 py-0.5 rounded-full bg-purple-200 text-purple-950 text-xs font-bold">
                              {lang === "sk" ? "Prehľad obvodu" : "District overview"}{" "}
                              {ov.obvod_number ? `č. ${ov.obvod_number}` : ""}
                            </span>
                            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 text-xs font-bold">
                              {ov.census_year}
                            </span>
                          </div>
                          <div className="font-bold text-sm text-stone-900">
                            {ov.municipality_name}
                            {ov.district_name ? ` (okr. ${ov.district_name})` : ""}
                          </div>
                          <div className="text-xs text-stone-700 flex flex-wrap items-center gap-2">
                            {ov.commissioner_name && (
                              <span>
                                {lang === "sk" ? "Komisár:" : "Commissioner:"}{" "}
                                <strong>{ov.commissioner_name}</strong>
                              </span>
                            )}
                            <span>
                              {lang === "sk" ? "Zaznamenaných domov:" : "Recorded houses:"}{" "}
                              <strong>{ov.overview_houses_count ?? 0}</strong>
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between gap-2 pt-2 border-t border-purple-200/70">
                        <Link
                          to={buildSheetDetailLink(ov.slovakiana_id)}
                          className="px-3 py-1.5 min-h-[34px] rounded-lg bg-purple-800 hover:bg-purple-900 text-white font-semibold text-xs inline-flex items-center gap-1.5"
                        >
                          <Building2 className="w-3.5 h-3.5" />
                          <span>
                            {lang === "sk"
                              ? "Otvoriť prehľad obvodu"
                              : "Open district overview"}
                          </span>
                        </Link>
                        <button
                          type="button"
                          onClick={() => handleProcessSingleSheet(ov.slovakiana_id)}
                          disabled={processingSheetId === ov.slovakiana_id}
                          className="px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                        >
                          {processingSheetId === ov.slovakiana_id ? (
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Download className="w-3.5 h-3.5" />
                          )}
                          <span>
                            {lang === "sk" ? "Prečítať OCR obvodu" : "Read district OCR"}
                          </span>
                        </button>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
            )}

            {/* Main Table: Adresár domov a rodín v obci (`Dom č. 1 .. N`) */}
            {Boolean(selectedMunicipality) && (
            <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                    <Home className="w-5 h-5 text-amber-600" />
                    <span>
                      {lang === "sk"
                        ? `Adresár domov a rodín v obci (${filteredHousesDirectory.length})`
                        : `Municipality House & Family Directory (${filteredHousesDirectory.length})`}
                    </span>
                  </h2>
                  <p className="text-xs sm:text-sm text-stone-600">
                    {lang === "sk"
                      ? "Prepojenie záznamov z Prehľadu sčítacieho obvodu (majiteľ domu, počet osôb) s konkrétnymi popisnými sčítacími hárkami rodín (`Dom č. 1 .. N`)."
                      : "Cross-reference of District Overview entries (house owner, resident count) with household census sheets (`Dom č. 1 .. N`)."}
                  </p>
                </div>
              </div>

              {filteredHousesDirectory.length === 0 ? (
                <div className="text-center py-8 text-stone-600 text-sm">
                  {lang === "sk"
                    ? "Pre tento filter sa nenašli žiadne domy. Skúste vybrať obec v ľavom paneli alebo stiahnuť hárky zo Slovakiany."
                    : "No houses match the current filter. Select a municipality on the left or fetch sheets from Slovakiana."}
                </div>
              ) : (
                <div className="w-full overflow-x-auto">
                  <table className="w-full table-auto text-left border-collapse text-xs sm:text-sm">
                    <thead>
                      <tr className="border-b border-stone-300 bg-stone-100 text-stone-800 font-bold">
                        <th className="py-2.5 px-2.5">
                          {lang === "sk" ? "Dom č. & Osada" : "House # & Settlement"}
                        </th>
                        <th className="py-2.5 px-2.5">
                          {lang === "sk"
                            ? "Majiteľ domu (podľa prehľadu obvodu)"
                            : "House Owner (per district overview)"}
                        </th>
                        <th className="py-2.5 px-2.5 text-center">
                          {lang === "sk"
                            ? "Bytové strany / Prítomné osoby"
                            : "Dwelling Units / Present Residents"}
                        </th>
                        <th className="py-2.5 px-2.5">
                          {lang === "sk"
                            ? "Sčítací hárok rodiny (Hlava domácnosti & členovia)"
                            : "Family Census Sheet (Head & Members)"}
                        </th>
                        <th className="py-2.5 px-2.5 text-right">
                          {lang === "sk" ? "Akcia" : "Action"}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-200">
                      {filteredHousesDirectory.map((entry, idx) => {
                        const rowKey = `${entry.house_number}-${
                          entry.slovakiana_key || entry.overview_house_id || idx
                        }`;
                        return (
                          <tr key={rowKey} className="hover:bg-stone-50 transition">
                            {/* Dom č. & Osada */}
                            <td className="py-2.5 px-2.5">
                              <div className="flex items-center gap-1.5">
                                <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-bold">
                                  Dom č. {entry.house_number || "—"}
                                </span>
                                {entry.is_inhabited === false && (
                                  <span className="px-1.5 py-0.5 rounded bg-stone-200 text-stone-700 text-xs">
                                    {lang === "sk" ? "neobývaný" : "uninhabited"}
                                  </span>
                                )}
                              </div>
                              {(entry.street_name || entry.settlement_name) && (
                                <div className="text-xs text-stone-600 mt-0.5">
                                  {entry.street_name &&
                                  (!entry.settlement_name ||
                                    foldText(entry.street_name) === foldText(entry.settlement_name))
                                    ? `${lang === "sk" ? "Ulica:" : "Street:"} ${entry.street_name}`
                                    : `${lang === "sk" ? "Osada:" : "Settlement:"} ${entry.settlement_name}${
                                        entry.street_name ? ` · ${entry.street_name}` : ""
                                      }`}
                                </div>
                              )}
                            </td>

                            {/* Majiteľ domu */}
                            <td className="py-2.5 px-2.5">
                              <div className="font-semibold text-stone-900">
                                {entry.owner_name || "—"}
                              </div>
                              {entry.remarks && (
                                <div className="text-xs text-stone-600">
                                  {lang === "sk" ? "Pozn.:" : "Note:"} {entry.remarks}
                                </div>
                              )}
                            </td>

                            {/* Bytové strany / Prítomné osoby */}
                            <td className="py-2.5 px-2.5 text-center">
                              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-stone-100 border border-stone-200 text-xs">
                                <span title="Počet bytových strán">
                                  {lang === "sk" ? "Byty:" : "Units:"}{" "}
                                  <strong>
                                    {entry.overview_household_count ?? "—"}
                                  </strong>
                                </span>
                                <span>·</span>
                                <span
                                  className="text-amber-900"
                                  title="Počet prítomných osôb"
                                >
                                  {lang === "sk" ? "Osôb:" : "People:"}{" "}
                                  <strong>
                                    {entry.overview_present_persons_count ??
                                      (entry.extracted_members_count > 0
                                        ? entry.extracted_members_count
                                        : "—")}
                                  </strong>
                                </span>
                              </div>
                            </td>

                            {/* Sčítací hárok rodiny */}
                            <td className="py-2.5 px-2.5">
                              {entry.slovakiana_key ? (
                                <div className="space-y-1">
                                  <div className="flex flex-wrap items-center gap-1.5">
                                    <Link
                                      to={buildSheetDetailLink(entry.slovakiana_key, {
                                        rowId: (
                                          entry.extracted_members ||
                                          entry.sheet?.extracted_persons_preview ||
                                          []
                                        ).find((m) => Boolean(m.matched_filter))?.id,
                                      })}
                                      className="px-2 py-0.5 rounded bg-amber-100 hover:bg-amber-200 text-amber-950 font-mono font-bold text-xs inline-flex items-center gap-1"
                                    >
                                      <FileSpreadsheet className="w-3.5 h-3.5" />
                                      <span>{entry.slovakiana_key}</span>
                                    </Link>
                                    {entry.extracted_members_count > 0 && (
                                      <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 font-bold text-xs inline-flex items-center gap-1">
                                        <Users className="w-3.5 h-3.5" />
                                        <span>
                                          {entry.extracted_members_count}{" "}
                                          {lang === "sk" ? "prečítaných" : "extracted"}
                                        </span>
                                      </span>
                                    )}
                                    {(entry.linked_house_sheets ||
                                      entry.sheet?.linked_house_sheets ||
                                      []
                                    ).map((linked) => {
                                      const lRef =
                                        linked.cair_id ||
                                        linked.slovakiana_key ||
                                        String(linked.id);
                                      return (
                                        <Link
                                          key={`${linked.id}-${lRef}`}
                                          to={`/scitacie-harky/${lRef}`}
                                          className="px-2 py-0.5 rounded-full bg-indigo-100 hover:bg-indigo-200 text-indigo-950 border border-indigo-300 font-bold text-xs inline-flex items-center gap-1"
                                          title={`Prepojený hárok pre dom č. ${
                                            linked.house_number || entry.house_number
                                          } z roku ${linked.census_year}`}
                                        >
                                          <span>
                                            🔗 {linked.census_year}: {lRef}
                                            {linked.head_of_household
                                              ? ` (${linked.head_of_household})`
                                              : ""}
                                          </span>
                                        </Link>
                                      );
                                    })}
                                  </div>
                                  {(entry.head_name || entry.wife_name) && (
                                    <div className="text-stone-800">
                                      {entry.head_name && (
                                        <span>
                                          <strong>{lang === "sk" ? "Hlava:" : "Head:"}</strong>{" "}
                                          {entry.head_name}
                                        </span>
                                      )}
                                      {entry.wife_name && (
                                        <span className="text-stone-600">
                                          {" "}
                                          + <strong>{lang === "sk" ? "Manž.:" : "Wife:"}</strong>{" "}
                                          {entry.wife_name}
                                        </span>
                                      )}
                                    </div>
                                  )}
                                  {renderHouseholdMembersBadges(
                                    entry.extracted_members ||
                                      entry.sheet?.extracted_persons_preview
                                  )}
                                </div>
                              ) : (
                                <span className="text-stone-600 italic text-xs">
                                  {lang === "sk"
                                    ? "Popisný hárok domu zatiaľ nie je spárovaný v DB"
                                    : "Household sheet not yet paired in DB"}
                                </span>
                              )}
                            </td>

                            {/* Akcia */}
                            <td className="py-2.5 px-2.5 text-right">
                              {entry.slovakiana_key ? (
                                <div className="inline-flex items-center gap-1.5">
                                  <Link
                                    to={buildSheetDetailLink(entry.slovakiana_key, {
                                      rowId: (
                                        entry.extracted_members ||
                                        entry.sheet?.extracted_persons_preview ||
                                        []
                                      ).find((m) => Boolean(m.matched_filter))?.id,
                                    })}
                                    className="px-3 py-1.5 min-h-[32px] rounded-md bg-stone-900 hover:bg-stone-800 text-white text-xs font-semibold inline-flex items-center gap-1"
                                  >
                                    <span>{lang === "sk" ? "Otvoriť" : "Open"}</span>
                                  </Link>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      handleProcessSingleSheet(entry.slovakiana_key!)
                                    }
                                    disabled={
                                      processingSheetId === entry.slovakiana_key
                                    }
                                    title="Spustiť AI OCR pre tento dom"
                                    className="px-2.5 py-1.5 min-h-[32px] rounded-md bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                                  >
                                    {processingSheetId === entry.slovakiana_key ? (
                                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                    ) : (
                                      <Download className="w-3.5 h-3.5" />
                                    )}
                                    <span>OCR</span>
                                  </button>
                                </div>
                              ) : entry.overview_sheet_id ? (
                                <Link
                                  to={buildSheetDetailLink(entry.overview_sheet_id, {
                                    houseId: entry.overview_house_id,
                                  })}
                                  className="px-3 py-1.5 min-h-[32px] rounded-md bg-purple-100 hover:bg-purple-200 text-purple-900 text-xs font-semibold inline-flex items-center gap-1"
                                >
                                  <span>
                                    {lang === "sk" ? "V prehľade obvodu" : "In district overview"}
                                  </span>
                                </Link>
                              ) : null}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            )}
          </div>
        ) : (
          /* TAB 2: Všetky hárky (Karty) */
          <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
            {/* Sub-filter Pills */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5 text-xs">
                {(
                  [
                    { id: "all", label: lang === "sk" ? "Všetky" : "All" },
                    ...(featureFlags.census_overview_sheets
                      ? [
                          {
                            id: "overview" as const,
                            label:
                              lang === "sk"
                                ? "Prehľady obvodov"
                                : "District Overviews",
                          },
                        ]
                      : []),
                    {
                      id: "household",
                      label:
                        lang === "sk"
                          ? "Domácnosti (Popisné hárky)"
                          : "Households",
                    },
                    {
                      id: "ocr_done",
                      label: lang === "sk" ? "Prečítané OCR" : "OCR Completed",
                    },
                    {
                      id: "pending_ocr",
                      label: lang === "sk" ? "Čaká na OCR" : "Pending OCR",
                    },
                  ] as const
                ).map((pill) => {
                  const active = cardSubFilter === pill.id;
                  return (
                    <button
                      key={pill.id}
                      type="button"
                      onClick={() => setCardSubFilter(pill.id)}
                      className={`px-3 py-1.5 min-h-[32px] rounded-full font-semibold transition cursor-pointer ${
                        active
                          ? "bg-stone-900 text-amber-300"
                          : "bg-stone-100 text-stone-700 hover:bg-stone-200"
                      }`}
                    >
                      {pill.label}
                    </button>
                  );
                })}
              </div>
              <span className="text-xs text-stone-600">
                {lang === "sk" ? (
                  <>
                    Zobrazených <strong>{filteredSheets.length}</strong> z {totalSheets} hárkov
                  </>
                ) : (
                  <>
                    Showing <strong>{filteredSheets.length}</strong> of {totalSheets} sheets
                  </>
                )}
              </span>
            </div>

            {loading ? (
              <div className="text-center py-10 text-stone-600 text-sm">
                {lang === "sk"
                  ? "Načítavam sčítacie hárky…"
                  : "Loading census sheets…"}
              </div>
            ) : filteredSheets.length === 0 ? (
              <div className="text-center py-10 text-stone-600 text-sm">
                {lang === "sk"
                  ? "Pre zvolený filter sa nenašli žiadne sčítacie hárky."
                  : "No census sheets match the selected filter."}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredSheets.length > CARDS_PER_PAGE && (
                  <div className="flex flex-wrap items-center justify-between gap-2 bg-stone-50 border border-stone-200 rounded-lg px-3 py-2 text-xs sm:text-sm">
                    <span className="text-stone-700">
                      {lang === "sk" ? (
                        <>
                          Zobrazené karty{" "}
                          <strong>
                            {(safeCardsPage - 1) * CARDS_PER_PAGE + 1}–
                            {Math.min(
                              filteredSheets.length,
                              safeCardsPage * CARDS_PER_PAGE
                            )}
                          </strong>{" "}
                          z <strong>{filteredSheets.length}</strong> hárkov
                        </>
                      ) : (
                        <>
                          Showing cards{" "}
                          <strong>
                            {(safeCardsPage - 1) * CARDS_PER_PAGE + 1}–
                            {Math.min(
                              filteredSheets.length,
                              safeCardsPage * CARDS_PER_PAGE
                            )}
                          </strong>{" "}
                          of <strong>{filteredSheets.length}</strong> sheets
                        </>
                      )}
                    </span>
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        disabled={safeCardsPage <= 1}
                        onClick={() => setCardsPage((p) => Math.max(1, p - 1))}
                        className="px-3 py-1.5 min-h-[32px] rounded-md bg-white hover:bg-stone-100 border border-stone-300 disabled:opacity-40 font-semibold text-xs cursor-pointer"
                      >
                        {lang === "sk" ? "← Predošlá strana" : "← Previous"}
                      </button>
                      <span className="px-2 font-semibold text-stone-800 text-xs">
                        {safeCardsPage} / {totalCardPages}
                      </span>
                      <button
                        type="button"
                        disabled={safeCardsPage >= totalCardPages}
                        onClick={() =>
                          setCardsPage((p) => Math.min(totalCardPages, p + 1))
                        }
                        className="px-3 py-1.5 min-h-[32px] rounded-md bg-white hover:bg-stone-100 border border-stone-300 disabled:opacity-40 font-semibold text-xs cursor-pointer"
                      >
                        {lang === "sk" ? "Ďalšia strana →" : "Next →"}
                      </button>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
                  {pagedSheets.map(
                  ({
                    sheet: s,
                    household,
                    extracted_members_count,
                    extracted_persons_preview,
                    members_preview,
                    extracted_members,
                    overview_houses_count,
                  }) => {
                    const isOverview = s.sheet_type === "overview";
                    const houseCount =
                      overview_houses_count ?? s.overview_houses_count ?? 0;
                    const sheetMembersPreview =
                      extracted_persons_preview ||
                      members_preview ||
                      extracted_members ||
                      s.extracted_persons_preview ||
                      s.members_preview ||
                      [];

                    return (
                      <div
                        key={s.slovakiana_id}
                        className={`rounded-xl p-3.5 border flex flex-col justify-between gap-3 transition ${
                          isOverview
                            ? "bg-purple-50/40 border-purple-200 hover:border-purple-500"
                            : "bg-stone-50 border-stone-200 hover:border-amber-500"
                        }`}
                      >
                        <div className="flex gap-3">
                          {s.thumbnail_url ? (
                            <img
                              src={s.thumbnail_url}
                              alt={s.slovakiana_id}
                              className="w-20 h-16 object-cover rounded-lg border border-stone-300 bg-stone-200 shrink-0"
                            />
                          ) : (
                            <div className="w-20 h-16 rounded-lg border border-stone-300 bg-stone-200 flex items-center justify-center text-xs text-stone-500 shrink-0">
                              {s.census_year}
                            </div>
                          )}

                          <div className="min-w-0 flex-1 space-y-1">
                            <div className="flex flex-wrap items-center gap-1.5">
                              <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-mono font-bold">
                                {s.slovakiana_id}
                              </span>
                              {isOverview ? (
                                <span className="px-2 py-0.5 rounded-full bg-purple-200 text-purple-950 text-xs font-bold">
                                  {lang === "sk" ? "Prehľad obvodu" : "District overview"}
                                </span>
                              ) : (
                                (s.house_number_conscription ||
                                  s.house_number_raw) && (
                                  <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 text-xs font-bold">
                                    Dom č.{" "}
                                    {s.house_number_conscription ||
                                      s.house_number_raw}
                                  </span>
                                )
                              )}
                              <span
                                className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                                  s.census_year === 1930
                                    ? "bg-emerald-100 text-emerald-900"
                                    : "bg-stone-200 text-stone-800"
                                }`}
                              >
                                {s.census_year}
                              </span>
                              {!isOverview &&
                                (() => {
                                  const crossYearLinks = (
                                    s.linked_house_sheets || []
                                  ).filter((l) => l.census_year !== s.census_year);
                                  if (crossYearLinks.length === 0) return null;
                                  const firstLink = crossYearLinks[0];
                                  const lRef =
                                    firstLink.cair_id ||
                                    firstLink.slovakiana_key ||
                                    String(firstLink.id);
                                  return (
                                    <Link
                                      to={`/scitacie-harky/${lRef}`}
                                      className="px-2 py-0.5 rounded-full bg-indigo-100 hover:bg-indigo-200 text-indigo-950 border border-indigo-300 text-xs font-bold inline-flex items-center gap-1"
                                      title={`Prejsť na hárok z roku ${firstLink.census_year} pre dom č. ${
                                        firstLink.house_number || s.house_number_raw || "—"
                                      }`}
                                    >
                                      <span>
                                        🔗 {firstLink.census_year}
                                        {crossYearLinks.length > 1
                                          ? ` (${crossYearLinks.length})`
                                          : ""}{" "}
                                        →
                                      </span>
                                    </Link>
                                  );
                                })()}
                            </div>

                            <div className="font-bold text-stone-900 text-sm">
                              {s.municipality_name}
                            </div>

                            <div className="text-xs text-stone-700 leading-snug">
                              {s.district_name
                                ? `${lang === "sk" ? "Okres" : "District"} ${s.district_name}`
                                : ""}
                              {s.street_name ? ` · ${s.street_name}` : ""}
                              {s.sheet_number
                                ? ` · ${lang === "sk" ? "hárok" : "sheet"} ${s.sheet_number}`
                                : ""}
                            </div>
                          </div>
                        </div>

                        {isOverview ? (
                          <div className="bg-purple-100/70 border border-purple-200 rounded-lg px-2.5 py-1.5 text-xs text-purple-950 flex items-center justify-between">
                            <span className="truncate">
                              {s.commissioner_name
                                ? `${lang === "sk" ? "Komisár:" : "Commissioner:"} ${s.commissioner_name}`
                                : lang === "sk"
                                ? "Zoznam domov a majiteľov v obvode"
                                : "List of houses and owners in district"}
                            </span>
                            <span className="ml-2 px-1.5 py-0.5 rounded bg-purple-200 font-bold shrink-0">
                              {houseCount} {lang === "sk" ? "domov" : "houses"}
                            </span>
                          </div>
                        ) : (
                          <div className="space-y-1.5">
                            {household &&
                              (household.head_surname ||
                                household.head_given_name) && (
                                <div className="bg-amber-50/80 border border-amber-200 rounded-lg px-2.5 py-1.5 text-xs text-stone-800 flex items-center justify-between">
                                  <div className="truncate">
                                    <span className="font-semibold text-amber-950">
                                      {lang === "sk" ? "Hlava:" : "Head:"}{" "}
                                    </span>
                                    {household.head_given_name || ""}{" "}
                                    {household.head_surname || ""}
                                    {household.wife_given_name
                                      ? ` + ${lang === "sk" ? "manž." : "wife"} ${household.wife_given_name}${
                                          household.wife_maiden_surname
                                            ? ` (rod. ${household.wife_maiden_surname})`
                                            : ""
                                        }`
                                      : ""}
                                  </div>
                                  <span className="ml-2 px-1.5 py-0.5 rounded bg-amber-200/80 text-amber-950 font-bold shrink-0 flex items-center gap-1">
                                    <Users className="w-3 h-3" />
                                    {extracted_members_count ||
                                      household.member_count}
                                  </span>
                                </div>
                              )}
                            {renderHouseholdMembersBadges(sheetMembersPreview)}
                          </div>
                        )}

                        <div className="flex items-center justify-between gap-2 pt-1 border-t border-stone-200/80">
                          <Link
                            to={buildSheetDetailLink(s.slovakiana_id, {
                              rowId: sheetMembersPreview.find((m) =>
                                Boolean(m.matched_filter)
                              )?.id,
                            })}
                            className="px-3 py-1.5 min-h-[34px] rounded-lg bg-stone-900 hover:bg-stone-800 text-white font-semibold text-xs flex items-center gap-1.5"
                          >
                            <FileSpreadsheet className="w-3.5 h-3.5" />
                            <span>{lang === "sk" ? "Otvoriť hárok" : "Open sheet"}</span>
                          </Link>

                          <button
                            type="button"
                            onClick={() =>
                              handleProcessSingleSheet(s.slovakiana_id)
                            }
                            disabled={processingSheetId === s.slovakiana_id}
                            className="px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                          >
                            {processingSheetId === s.slovakiana_id ? (
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Download className="w-3.5 h-3.5" />
                            )}
                            <span>
                              {s.download_status === "downloaded" ||
                              s.download_status === "extracted" ||
                              s.download_status === "matched"
                                ? lang === "sk"
                                  ? "Znova prečítať OCR"
                                  : "Re-run OCR"
                                : lang === "sk"
                                ? "Stiahnuť & Prečítať"
                                : "Download & Read"}
                            </span>
                          </button>
                        </div>
                      </div>
                    );
                  }
                )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Slovakiana CC BY-NC-ND 4.0 Attribution Notice */}
        <div className="bg-stone-100 border border-stone-300 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-2 text-xs text-stone-600">
          <div className="flex items-start gap-2">
            <Scale className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <div>
              {lang === "sk" ? (
                <>
                  <strong>Zdroj sčítacích hárkov (1930 / 1940):</strong>{" "}
                  Slovenský národný archív (SNA) /{" "}
                  <a
                    href="https://slovakiana.sk/scitacie-harky"
                    target="_blank"
                    rel="noreferrer"
                    className="font-semibold text-amber-800 hover:underline"
                  >
                    Slovakiana (Národné osvetové centrum)
                  </a>
                  . Sprístupnené pod licenciou{" "}
                  <a
                    href="https://creativecommons.org/licenses/by-nc-nd/4.0/"
                    target="_blank"
                    rel="noreferrer"
                    className="font-bold text-stone-900 hover:underline"
                  >
                    CC BY-NC-ND 4.0
                  </a>{" "}
                  (Uveďte autora – Nekomerčne – Bez odvodených diel). Lokálne výrezy riadkov a
                  OCR mriežky slúžia výhradne ako dočasná pomôcka na čítanie pri nekomerčnom
                  výskume a nesmú byť ďalej šírené.
                </>
              ) : (
                <>
                  <strong>Source of 1930 / 1940 Census Sheets:</strong>{" "}
                  Slovak National Archives (SNA) /{" "}
                  <a
                    href="https://slovakiana.sk/scitacie-harky"
                    target="_blank"
                    rel="noreferrer"
                    className="font-semibold text-amber-800 hover:underline"
                  >
                    Slovakiana (National Enlightenment Centre)
                  </a>
                  . Licensed under{" "}
                  <a
                    href="https://creativecommons.org/licenses/by-nc-nd/4.0/"
                    target="_blank"
                    rel="noreferrer"
                    className="font-bold text-stone-900 hover:underline"
                  >
                    CC BY-NC-ND 4.0
                  </a>{" "}
                  (Attribution – NonCommercial – NoDerivatives). Local row crops and OpenCV grid
                  overlays are transient reading aids for non-commercial research and must not be
                  redistributed.
                </>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
