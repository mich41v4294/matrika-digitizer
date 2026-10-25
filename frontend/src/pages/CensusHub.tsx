import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import type { LinkedHouseSheet } from "../api";
import { emitClientOcrNotification, type PipelineState } from "../App";
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
  PanelLeftClose,
  PanelLeftOpen,
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
  overview_persons_count?: number;
  extracted_persons_count: number;
  persons_count?: number;
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
  current_sheet_id?: number | null;
  current_slovakiana_key?: string | null;
  current_sheet_label?: string | null;
  current_sheet_index?: number | null;
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
  sheet_type?: "household" | "overview" | "municipal_overview" | string;
  obvod_number?: string | null;
  commissioner_name?: string | null;
  overview_houses_count?: number;
  municipal_districts_count?: number;
  municipal_overview_districts?: Array<{
    id: number;
    row_index: number;
    ordinal_number: string | number | null;
    settlement_name: string | null;
    obvod_number: string | null;
    commissioner_name: string | null;
    houses_total: number | null;
    houses_inhabited: number | null;
    houses_uninhabited: number | null;
    household_count: number | null;
    present_persons_count: number | null;
    is_summary_row: boolean;
    matched_overview_sheet_id: number | null;
    matched_overview_slovakiana_key?: string | null;
    matched_overview_houses_count?: number;
  }>;
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
  overview_slovakiana_key?: string | null;
  obvod_number?: string | null;
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
  municipal_overview_sheets?: SerializedSheet[];
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
  const [villageMenuCollapsed, setVillageMenuCollapsed] = useState<boolean>(
    () => localStorage.getItem("matrika_village_menu_collapsed") === "true"
  );
  const [loading, setLoading] = useState<boolean>(true);
  const splitMenuRef = useRef<HTMLDivElement | null>(null);
  const jobsPopoverRef = useRef<HTMLDivElement | null>(null);

  const toggleVillageMenu = () => {
    setVillageMenuCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("matrika_village_menu_collapsed", String(next));
      return next;
    });
  };

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

  // Left Sidebar filters (`Okresy a obce`) initialized from URL — always a single year (1930 by default)
  const [selectedYearFilter, setSelectedYearFilter] = useState<"1930" | "1940">(() => {
    const yr = searchParams.get("rok") || searchParams.get("year") || "";
    return yr === "1940" ? "1940" : "1930";
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
  const loadWorkspaceDataRef = useRef<(isBackgroundPoll?: boolean) => Promise<void>>(
    async () => {}
  );
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [loadingFacets, setLoadingFacets] = useState<boolean>(false);
  const [jobActionId, setJobActionId] = useState<number | null>(null);
  const [livePipelineState, setLivePipelineState] = useState<PipelineState | null>(null);
  const [localSheetProgressMap, setLocalSheetProgressMap] = useState<
    Record<
      string,
      {
        sheetKey: string;
        active_stage: string;
        active_model: string | null;
        progress_pct: number;
        current_row_index: number | null;
        total_rows: number | null;
        message_sk: string;
        message_en: string;
        started_at_ms: number;
        elapsed_sec: number;
      }
    >
  >({});
  const processingSheetId = Object.keys(localSheetProgressMap)[0] || null;
  const [recentlyCompletedSheets, setRecentlyCompletedSheets] = useState<
    Record<
      string,
      {
        message_sk: string;
        message_en: string;
        elapsed_sec: number;
        finished_at_ms: number;
      }
    >
  >({});

  const selectedYearTag = selectedYearFilter === "1940" ? "10003" : "10001";

  const loadFacets = async (districtTag?: string, yearTag?: string) => {
    setLoadingFacets(true);
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
    } finally {
      setLoadingFacets(false);
    }
  };

  // Listen for live search_progress, state, and sheet_updated broadcasts
  useEffect(() => {
    const handleCustomState = (e: Event) => {
      const custom = e as CustomEvent<PipelineState>;
      if (custom.detail) {
        const d = custom.detail;
        setLivePipelineState(d);
        if (d.active_stage && d.active_stage !== "idle" && d.current_slovakiana_key) {
          const skKey = d.current_slovakiana_key;
          setLocalSheetProgressMap((prev) => {
            const cur = prev[skKey];
            if (!cur) return prev;
            return {
              ...prev,
              [skKey]: {
                ...cur,
                active_stage: d.active_stage,
                active_model: d.active_model ?? cur.active_model,
                progress_pct: Math.max(cur.progress_pct, Math.min(99, Number(d.progress_pct || 12))),
                current_row_index: d.current_row_index ?? cur.current_row_index,
                total_rows: d.total_rows ?? cur.total_rows,
                message_sk: d.message_sk || cur.message_sk,
                message_en: d.message_en || cur.message_en,
              },
            };
          });
        }
      }
    };
    const handleSheetUpdated = () => {
      void loadWorkspaceDataRef.current(true);
    };
    window.addEventListener("matrika:pipeline-state", handleCustomState);
    window.addEventListener("matrika:sheet-updated", handleSheetUpdated);

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
          } else if (msg.type === "state" && msg.data) {
            const d = msg.data as PipelineState;
            setLivePipelineState(d);
            if (d.active_stage && d.active_stage !== "idle" && d.current_slovakiana_key) {
              const skKey = d.current_slovakiana_key;
              setLocalSheetProgressMap((prev) => {
                const cur = prev[skKey];
                if (!cur) return prev;
                return {
                  ...prev,
                  [skKey]: {
                    ...cur,
                    active_stage: d.active_stage,
                    active_model: d.active_model ?? cur.active_model,
                    progress_pct: Math.max(cur.progress_pct, Math.min(99, Number(d.progress_pct || 12))),
                    current_row_index: d.current_row_index ?? cur.current_row_index,
                    total_rows: d.total_rows ?? cur.total_rows,
                    message_sk: d.message_sk || cur.message_sk,
                    message_en: d.message_en || cur.message_en,
                  },
                };
              });
            }
          } else if (msg.type === "sheet_updated" && msg.data) {
            const skKey = String(msg.data.slovakiana_key || "");
            if (skKey && msg.data.download_status !== "error") {
              const cnt =
                msg.data.persons_count ??
                msg.data.overview_houses_count ??
                msg.data.municipal_districts_count;
              setRecentlyCompletedSheets((prev) => ({
                ...prev,
                [skKey]: {
                  message_sk:
                    cnt != null
                      ? `AI OCR dokončené (${cnt} záznamov)`
                      : "AI OCR čítanie hárku dokončené",
                  message_en:
                    cnt != null
                      ? `AI OCR completed (${cnt} records)`
                      : "Sheet AI OCR completed",
                  elapsed_sec: 0,
                  finished_at_ms: Date.now(),
                },
              }));
            }
            void loadWorkspaceDataRef.current(true);
          } else if (msg.type === "completion" || msg.type === "ocr_completed") {
            void loadWorkspaceDataRef.current(true);
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore
    }
    return () => {
      window.removeEventListener("matrika:pipeline-state", handleCustomState);
      window.removeEventListener("matrika:sheet-updated", handleSheetUpdated);
      if (ws) ws.close();
    };
  }, []);

  // Smooth client-side ticker while single sheet OCR is running in CensusHub
  useEffect(() => {
    if (!processingSheetId) return;
    const ticker = window.setInterval(() => {
      setLocalSheetProgressMap((prev) => {
        const keys = Object.keys(prev);
        if (keys.length === 0) return prev;
        const next = { ...prev };
        for (const k of keys) {
          const item = next[k];
          const elapsed = Math.max(1, Math.round((Date.now() - item.started_at_ms) / 1000));
          const step = item.progress_pct < 35 ? 2 : item.progress_pct < 75 ? 1 : 0.5;
          const nextPct = Math.min(94, Math.floor(item.progress_pct + step));
          next[k] = {
            ...item,
            elapsed_sec: elapsed,
            progress_pct: Math.max(item.progress_pct, nextPct),
          };
        }
        return next;
      });
    }, 650);
    return () => window.clearInterval(ticker);
  }, [processingSheetId]);

  // Smooth client-side progress ticker while a census search request is in flight
  useEffect(() => {
    if (!isSearchingCensus) return;
    const t0 = performance.now();
    const timer = window.setInterval(() => {
      const elapsed = Math.round(performance.now() - t0);
      if (elapsed < 220) return;
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
          elapsed_ms: elapsed,
        };
      });
    }, 450);
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

      const shouldFetchMuniOverview = Boolean(
        selectedMunicipality.trim() ||
          activeTab === "overview" ||
          streetFilter.trim() ||
          personNameFilter.trim() ||
          birthYearFilter.trim() ||
          birthPlaceFilter.trim() ||
          workspaceSearch.trim()
      );
      const sig = controller ? { signal: controller.signal } : undefined;
      const [statsRes, jobsRes, sheetsRes, muniSumRes, ovRes] = await Promise.all([
        fetch("/api/census/stats", sig),
        fetch("/api/census/jobs", sig),
        fetch(`/api/census/sheets?${sheetParams.toString()}`, sig),
        fetch(`/api/census/municipalities-summary?${muniSummaryParams.toString()}`, sig),
        shouldFetchMuniOverview
          ? fetch(`/api/census/municipality-overview?${ovParams.toString()}`, sig)
          : Promise.resolve(null),
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
      if (ovRes && ovRes.ok) {
        const od = (await ovRes.json()) as MunicipalityOverviewResponse;
        setMuniOverview(od);
      } else if (!shouldFetchMuniOverview) {
        setMuniOverview(null);
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
  loadWorkspaceDataRef.current = loadWorkspaceData;

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
    }, 18000);
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
          overview_persons_count: 0,
          extracted_persons_count: 0,
          persons_count: 0,
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
      const isOverview =
        s.sheet_type === "overview" || s.sheet_type === "municipal_overview";
      if (!featureFlags.census_overview_sheets && isOverview) return false;
      const isOcrDone =
        s.download_status === "ocr_done" ||
        s.download_status === "extracted" ||
        s.download_status === "matched" ||
        extracted_members_count > 0 ||
        (s.overview_houses_count || 0) > 0 ||
        (s.municipal_districts_count || 0) > 0;

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

  // Poll faster (every 3.5s) while a census job is actively running, or every 18s when idle
  const hasRunningJob = useMemo(
    () => jobs.some((j) => j.status === "running"),
    [jobs]
  );
  useEffect(() => {
    if (!hasRunningJob) return;
    const fastTimer = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadWorkspaceData(true);
    }, 3500);
    return () => window.clearInterval(fastTimer);
  }, [
    hasRunningJob,
    workspaceSearch,
    selectedDistrictName,
    selectedMunicipality,
    selectedYearFilter,
    streetFilter,
    personNameFilter,
    birthYearFilter,
    birthPlaceFilter,
  ]);

  const handleHarvestMunicipality = async (
    muniName?: string,
    muniTagKey?: string | null
  ) => {
    const { targetName, resolvedMuniTag, distName, resolvedDistTag } =
      resolveMunicipalityAndDistrictTags(muniName, muniTagKey);
    setBusyAction(`harvest_${targetName || "all"}`);
    try {
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

  const handleOcrOverviewOnlyMunicipality = async (
    muniName?: string,
    muniTagKey?: string | null
  ) => {
    const { targetName, resolvedMuniTag, distName, resolvedDistTag } =
      resolveMunicipalityAndDistrictTags(muniName, muniTagKey);
    const matchedMuni = targetName
      ? combinedMunicipalities.find(
          (m) =>
            foldMunicipalityKey(m.municipality) ===
            foldMunicipalityKey(targetName)
        )
      : activeMuniSummary;
    const targetLimit = Math.max(
      maxSheetsInput || 40,
      matchedMuni?.slovakiana_count || 0,
      matchedMuni?.total_sheets || 0,
      500
    );
    setBusyAction(`overview_ocr_${targetName || "all"}`);
    try {
      await fetch("/api/census/jobs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name:
            lang === "sk"
              ? `Stiahnuť hárky a OCR prehľadov: ${targetName || distName || "Výber"}`
              : `Download & OCR Overviews: ${targetName || distName || "Selection"}`,
          job_type: "overview_only",
          census_year_tag: selectedYearTag || null,
          district_tag: resolvedDistTag || null,
          municipality_tag: resolvedMuniTag || null,
          municipality: targetName || null,
          district: distName || null,
          query_text: resolvedMuniTag
            ? workspaceSearch.trim() || null
            : targetName || workspaceSearch.trim() || null,
          max_sheets: targetLimit,
          run_pipeline: true,
          run_mcp: false,
          only_undone: true,
          overview_only: true,
        }),
      });
      if (targetName) {
        setSelectedMunicipality(targetName);
        setActiveTab("overview");
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
    setJobActionId(job.id);
    try {
      const action = job.status === "paused" ? "resume" : "pause";
      await fetch(`/api/census/jobs/${job.id}/${action}`, { method: "POST" });
      await loadWorkspaceData();
    } finally {
      setJobActionId(null);
    }
  };

  const handleDeleteJob = async (jobId: number) => {
    setJobActionId(jobId);
    try {
      await fetch(`/api/census/jobs/${jobId}`, { method: "DELETE" });
      await loadWorkspaceData();
    } finally {
      setJobActionId(null);
    }
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
    const startMs = Date.now();
    setRecentlyCompletedSheets((prev) => {
      const next = { ...prev };
      delete next[sheetKey];
      return next;
    });
    setLocalSheetProgressMap((prev) => ({
      ...prev,
      [sheetKey]: {
        sheetKey,
        active_stage: "census_download",
        active_model: "glm-ocr",
        progress_pct: 10,
        current_row_index: null,
        total_rows: null,
        message_sk: `Spúšťam sťahovanie a AI OCR čítanie hárku (${sheetKey})…`,
        message_en: `Starting scan download and AI OCR for sheet (${sheetKey})…`,
        started_at_ms: startMs,
        elapsed_sec: 0,
      },
    }));
    try {
      const r = await fetch(
        `/api/census/sheets/${sheetKey}/download-and-process?run_pipeline=true&run_mcp=true`,
        { method: "POST" }
      );
      const resData = r.ok ? await r.json().catch(() => ({})) : {};
      const elapsedSec = Math.max(1, Math.round((Date.now() - startMs) / 1000));
      const pRes = resData?.pipeline || {};
      const rowsProc = Number(pRes.rows_processed ?? resData?.rows_processed ?? 0);
      const personsCnt = pRes.persons_count;
      const housesCnt = pRes.overview_houses_count;
      const distsCnt = pRes.municipal_districts_count;

      const summarySk =
        personsCnt != null
          ? `AI OCR dokončené: ${personsCnt} osôb (${rowsProc} riadkov)`
          : housesCnt != null
          ? `AI OCR dokončené: ${housesCnt} domov v prehľade`
          : distsCnt != null
          ? `AI OCR dokončené: ${distsCnt} obvodov`
          : rowsProc > 0
          ? `AI OCR dokončené (${rowsProc} riadkov)`
          : "AI OCR čítanie hárku bolo úspešne dokončené.";
      const summaryEn =
        personsCnt != null
          ? `AI OCR completed: ${personsCnt} people (${rowsProc} rows)`
          : housesCnt != null
          ? `AI OCR completed: ${housesCnt} houses in overview`
          : distsCnt != null
          ? `AI OCR completed: ${distsCnt} districts`
          : rowsProc > 0
          ? `AI OCR completed (${rowsProc} rows)`
          : "Sheet AI OCR completed successfully.";

      if (r.ok && !resData?.error && pRes?.status !== "error") {
        setRecentlyCompletedSheets((prev) => ({
          ...prev,
          [sheetKey]: {
            message_sk: summarySk,
            message_en: summaryEn,
            elapsed_sec: elapsedSec,
            finished_at_ms: Date.now(),
          },
        }));
        emitClientOcrNotification({
          status: "completed",
          kind: "census_sheet",
          slovakiana_key: sheetKey,
          rows_processed: rowsProc,
          persons_count: personsCnt,
          houses_count: housesCnt,
          districts_count: distsCnt,
          duration_sec: elapsedSec,
          title_sk: "Sčítací hárok prečítaný",
          title_en: "Census sheet OCR completed",
          message_sk: `${summarySk} (${sheetKey})`,
          message_en: `${summaryEn} (${sheetKey})`,
          target_url: `/scitacie-harky/${sheetKey}`,
        });
      } else {
        setLivePipelineState((prev) =>
          prev && prev.current_slovakiana_key === sheetKey
            ? { ...prev, active_stage: "idle", progress_pct: 0 }
            : prev
        );
      }
      await loadWorkspaceData();
      refreshStatus();
    } catch {
      setLivePipelineState((prev) =>
        prev && prev.current_slovakiana_key === sheetKey
          ? { ...prev, active_stage: "idle", progress_pct: 0 }
          : prev
      );
    } finally {
      setLocalSheetProgressMap((prev) => {
        const next = { ...prev };
        delete next[sheetKey];
        return next;
      });
    }
  };

  const getAttachedSheetState = (sheetKey?: string | null, sheetId?: number | null) => {
    if (!sheetKey && !sheetId) return null;
    const localSheetProgress = sheetKey ? localSheetProgressMap[sheetKey] : null;
    const isLocalActive = Boolean(localSheetProgress);
    const isBackendActive = Boolean(
      livePipelineState?.active_stage &&
        livePipelineState.active_stage !== "idle" &&
        ((sheetKey && livePipelineState.current_slovakiana_key === sheetKey) ||
          (sheetId != null && livePipelineState.current_sheet_id === sheetId))
    );
    const queuedTaskForSheet = (livePipelineState?.queued_tasks || []).find(
      (qt) =>
        (sheetKey && qt.metadata?.slovakiana_key === sheetKey) ||
        (sheetId != null && qt.metadata?.sheet_id === sheetId)
    );
    if (queuedTaskForSheet && !isBackendActive) {
      const qPos = queuedTaskForSheet.queue_position ?? 1;
      return {
        active: true,
        completed: false,
        stage: "queued",
        model: null,
        pct: 5,
        rowIdx: null,
        totRows: null,
        msgSk:
          queuedTaskForSheet.message_sk ||
          `Vo fronte (#${qPos}) – čaká na dokončenie prebiehajúcej úlohy…`,
        msgEn:
          queuedTaskForSheet.message_en ||
          `Queued (#${qPos}) – waiting for ongoing task to complete…`,
        elapsedSec: localSheetProgress?.elapsed_sec || 0,
      };
    }
    if (isLocalActive || isBackendActive) {
      const stage =
        (isBackendActive ? livePipelineState?.active_stage : null) ||
        localSheetProgress?.active_stage ||
        "ocr";
      const model =
        (isBackendActive ? livePipelineState?.active_model : null) ||
        localSheetProgress?.active_model ||
        null;
      const pct = Math.max(
        8,
        Math.min(
          99,
          Math.max(
            isBackendActive ? Number(livePipelineState?.progress_pct || 12) : 0,
            isLocalActive ? Number(localSheetProgress?.progress_pct || 10) : 0
          )
        )
      );
      const rowIdx =
        (isBackendActive ? livePipelineState?.current_row_index : null) ??
        localSheetProgress?.current_row_index ??
        null;
      const totRows =
        (isBackendActive ? livePipelineState?.total_rows : null) ??
        localSheetProgress?.total_rows ??
        null;
      const msgSk =
        (isBackendActive ? livePipelineState?.message_sk : null) ||
        localSheetProgress?.message_sk ||
        "Prebieha AI OCR čítanie rukopisu…";
      const msgEn =
        (isBackendActive ? livePipelineState?.message_en : null) ||
        localSheetProgress?.message_en ||
        "Running handwriting AI OCR…";
      const elapsedSec = localSheetProgress?.elapsed_sec || 0;
      return {
        active: true,
        completed: false,
        stage,
        model,
        pct,
        rowIdx,
        totRows,
        msgSk,
        msgEn,
        elapsedSec,
      };
    }
    if (sheetKey && recentlyCompletedSheets[sheetKey]) {
      const comp = recentlyCompletedSheets[sheetKey];
      return {
        active: false,
        completed: true,
        stage: "idle",
        model: null,
        pct: 100,
        rowIdx: null,
        totRows: null,
        msgSk: comp.message_sk,
        msgEn: comp.message_en,
        elapsedSec: comp.elapsed_sec,
      };
    }
    return null;
  };

  const renderAttachedSheetProgress = (
    sheetKey?: string | null,
    sheetId?: number | null
  ) => {
    const st = getAttachedSheetState(sheetKey, sheetId);
    if (!st) return null;
    const stageOrder = ["census_download", "layout", "ocr", "extract", "mcp_match"];
    const currentStageIdx = st.active ? Math.max(0, stageOrder.indexOf(st.stage)) : stageOrder.length;
    const miniStages = [
      { key: "census_download", sk: "1. Skeny", en: "1. Scans" },
      { key: "layout", sk: "2. Riadky", en: "2. Rows" },
      { key: "ocr", sk: "3. OCR", en: "3. OCR" },
      { key: "extract", sk: "4. Extrakcia", en: "4. Extract" },
      { key: "mcp_match", sk: "5. FS", en: "5. FS" },
    ];

    return (
      <div
        data-testid={`attached-sheet-progress-${sheetKey || sheetId}`}
        className={`rounded-lg px-2.5 py-2 border space-y-1.5 text-xs transition-all ${
          st.active
            ? "bg-amber-50/95 border-amber-400 shadow-2xs"
            : "bg-emerald-50/95 border-emerald-300"
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5 font-bold text-stone-900 min-w-0">
            {st.active ? (
              <RefreshCw className="w-3.5 h-3.5 text-amber-700 animate-spin shrink-0" />
            ) : (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
            )}
            <span className="truncate">
              {lang === "sk" ? st.msgSk : st.msgEn}
            </span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {st.totRows != null && st.totRows > 0 && st.active && (
              <span className="px-1.5 py-0.5 rounded bg-white border border-amber-300 text-amber-950 text-[11px] font-bold">
                {lang === "sk"
                  ? `Riadok ${st.rowIdx || 1}/${st.totRows}`
                  : `Row ${st.rowIdx || 1}/${st.totRows}`}
              </span>
            )}
            {st.elapsedSec > 0 && (
              <span className="text-stone-600 font-semibold text-[11px]">
                {st.elapsedSec} s
              </span>
            )}
            <span
              className={`px-2 py-0.5 rounded-full font-bold text-[11px] ${
                st.active ? "bg-amber-600 text-white" : "bg-emerald-700 text-white"
              }`}
            >
              {st.pct}%
            </span>
            {!st.active && sheetKey && (
              <button
                type="button"
                onClick={() =>
                  setRecentlyCompletedSheets((prev) => {
                    const next = { ...prev };
                    delete next[sheetKey];
                    return next;
                  })
                }
                title={lang === "sk" ? "Skryť" : "Dismiss"}
                className="p-0.5 rounded hover:bg-emerald-200/70 text-stone-600 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        <div
          role="progressbar"
          aria-valuenow={st.pct}
          aria-valuemin={0}
          aria-valuemax={100}
          className={`w-full rounded-full h-2 overflow-hidden ${
            st.active ? "bg-amber-200/80" : "bg-emerald-200"
          }`}
        >
          <div
            className={`h-full transition-all duration-300 ${
              st.active ? "bg-amber-600 animate-pulse" : "bg-emerald-600"
            }`}
            style={{ width: `${st.pct}%` }}
          />
        </div>

        {st.active && (
          <div className="flex flex-wrap items-center gap-1 pt-0.5">
            {miniStages.map((ms, idx) => {
              const isDone = idx < currentStageIdx;
              const isCurrent = idx === currentStageIdx;
              return (
                <span
                  key={ms.key}
                  className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                    isCurrent
                      ? "bg-amber-600 text-white"
                      : isDone
                      ? "bg-emerald-100 text-emerald-900 border border-emerald-300"
                      : "bg-white/70 text-stone-500 border border-stone-200"
                  }`}
                >
                  {lang === "sk" ? ms.sk : ms.en}
                </span>
              );
            })}
            {st.model && (
              <span className="ml-auto px-1.5 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-[10px]">
                {st.model}
              </span>
            )}
          </div>
        )}
      </div>
    );
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

  // Fast poll of /api/census/jobs while any job is running so progress updates in real time
  useEffect(() => {
    if (runningJobsCount === 0 && !processingSheetId) return;
    const fastTimer = window.setInterval(async () => {
      if (document.visibilityState === "hidden") return;
      try {
        const jr = await fetch("/api/census/jobs");
        if (jr.ok) {
          const jd = await jr.json();
          const nextJobs: CensusJob[] = jd.jobs || [];
          setJobs(nextJobs);
          if (nextJobs.filter((j) => j.status === "running").length === 0 && runningJobsCount > 0) {
            void loadWorkspaceData(true);
            refreshStatus();
          }
        }
      } catch {
        // ignore
      }
    }, 1800);
    return () => window.clearInterval(fastTimer);
  }, [runningJobsCount, processingSheetId]);

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
      selectedYearFilter !== "1930" ||
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
    setSelectedYearFilter("1930");
    setLocationQuickFilter("");
    setStreetFilter("");
    setPersonNameFilter("");
    setBirthYearFilter("");
    setBirthPlaceFilter("");
    setWorkspaceSearch("");
    setCardSubFilter("all");
  };

  const handleSwitchYear = (nextYear: "1930" | "1940") => {
    if (nextYear === selectedYearFilter) return;
    const targetYrNum = Number(nextYear);
    const trimmedQ = workspaceSearch.trim();
    if (trimmedQ && /^cair-[a-z0-9_-]+$/i.test(trimmedQ)) {
      const qLower = trimmedQ.toLowerCase();
      const matchedSheetRow = sheets.find(
        (r) => r.sheet.slovakiana_id.toLowerCase() === qLower
      );
      const matchedHouseEntry = (muniOverview?.houses_directory || []).find(
        (h) => (h.slovakiana_key || "").toLowerCase() === qLower
      );
      const linked =
        (matchedSheetRow?.sheet.linked_house_sheets || []).find(
          (l) => l.census_year === targetYrNum
        ) ||
        (matchedHouseEntry?.linked_house_sheets || []).find(
          (l) => l.census_year === targetYrNum
        );
      if (linked) {
        const linkedRef = linked.cair_id || linked.slovakiana_key || "";
        if (linkedRef) {
          setWorkspaceSearch(linkedRef);
        }
        if (!selectedMunicipality && linked.municipality) {
          setSelectedMunicipality(linked.municipality);
        }
      } else if (matchedSheetRow || matchedHouseEntry) {
        const hn =
          matchedSheetRow?.sheet.house_number_conscription ||
          matchedSheetRow?.sheet.house_number_raw ||
          matchedHouseEntry?.house_number ||
          "";
        const muni =
          matchedSheetRow?.sheet.municipality_name ||
          matchedHouseEntry?.municipality ||
          "";
        if (muni && !selectedMunicipality) {
          setSelectedMunicipality(muni);
        }
        if (hn) {
          setWorkspaceSearch(hn);
        }
      }
    }
    setSelectedYearFilter(nextYear);
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
      {!villageMenuCollapsed && (
      <aside
        data-testid="village-selection-sidebar"
        className="lg:col-span-3 bg-white rounded-xl border border-stone-300 p-3.5 space-y-3 lg:sticky lg:top-4"
      >
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-sm font-bold text-stone-900 flex items-center gap-1.5 min-w-0">
            <MapPin className="w-4 h-4 text-amber-600 shrink-0" />
            <span className="truncate">
              {lang === "sk" ? "Okresy a obce" : "Districts & Municipalities"}
            </span>
            {(loadingFacets || loading) && (
              <span title={lang === "sk" ? "Načítavam obce…" : "Loading municipalities…"}>
                <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin shrink-0" />
              </span>
            )}
          </h2>
          <div className="flex items-center gap-1 shrink-0">
            <a
              href="https://slovakiana.sk/scitacie-harky/?page=1&lm=0"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-semibold text-amber-800 hover:underline inline-flex items-center gap-1 py-1 px-1.5 min-h-[32px]"
            >
              <span>Slovakiana</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button
              type="button"
              onClick={toggleVillageMenu}
              data-testid="toggle-village-menu-btn"
              aria-expanded={!villageMenuCollapsed}
              aria-label={
                lang === "sk"
                  ? "Zbaliť výber obcí"
                  : "Collapse municipalities menu"
              }
              title={
                lang === "sk"
                  ? "Zbaliť výber obcí (Okresy a obce)"
                  : "Collapse Districts & Municipalities menu"
              }
              className="p-1.5 min-h-[30px] min-w-[30px] rounded-lg text-stone-500 hover:text-stone-900 hover:bg-stone-100 inline-flex items-center justify-center shrink-0 cursor-pointer transition"
            >
              <PanelLeftClose className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Year Switch: 1930 | 1940 (always single year, 1930 by default) */}
        <div
          data-testid="census-year-switch"
          role="group"
          aria-label={lang === "sk" ? "Rok sčítania" : "Census year"}
          className="grid grid-cols-2 gap-1 bg-stone-100 p-1 rounded-lg border border-stone-200 text-xs"
        >
          {(
            [
              { label: "1930", value: "1930" },
              { label: "1940", value: "1940" },
            ] as const
          ).map((yr) => {
            const active = selectedYearFilter === yr.value;
            return (
              <button
                key={yr.value}
                type="button"
                data-testid={`census-year-btn-${yr.value}`}
                aria-pressed={active}
                onClick={() => handleSwitchYear(yr.value)}
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
              {selectedYearFilter === "1940"
                ? stats?.sheets_1940 ?? totalSheets
                : stats?.sheets_1930 ?? totalSheets}
            </span>
          </button>

          {combinedMunicipalities.map((m) => {
            const isSelected =
              Boolean(selectedMunicipality) &&
              foldMunicipalityKey(selectedMunicipality) ===
                foldMunicipalityKey(m.municipality);
            const isHarvesting = busyAction === `harvest_${m.municipality}`;
            const isOcrOverviews = busyAction === `overview_ocr_${m.municipality}`;
            const totalVillageSheets = m.slovakiana_count ?? m.total_sheets;
            const extractedPersons = m.persons_count ?? m.extracted_persons_count;
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
                className={`rounded-lg p-2.5 border transition cursor-pointer space-y-2 ${
                  isSelected
                    ? "bg-amber-50/90 border-amber-500 ring-1 ring-amber-400 shadow-2xs"
                    : "bg-white hover:bg-stone-50/80 border-stone-200"
                }`}
              >
                {/* Row 1: Village Name + District on left, Sheet Count pill on right */}
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="font-bold text-xs sm:text-sm text-stone-900 leading-snug break-words">
                      {m.municipality}
                    </div>
                    {m.district && (
                      <div className="text-[11px] text-stone-500 mt-0.5 truncate">
                        {lang === "sk" ? "okr." : "dist."} {m.district}
                      </div>
                    )}
                  </div>

                  {(m.slovakiana_count != null || m.total_sheets > 0) && (
                    <span
                      title={
                        lang === "sk"
                          ? `Prečítané OCR: ${m.ocr_done_sheets} z ${totalVillageSheets} hárkov`
                          : `OCR done: ${m.ocr_done_sheets} of ${totalVillageSheets} sheets`
                      }
                      className={`px-2 py-0.5 rounded-md text-[11px] font-bold shrink-0 border ${
                        isSelected
                          ? "bg-amber-200/80 text-amber-950 border-amber-300"
                          : "bg-amber-50 text-amber-900 border-amber-200/80"
                      }`}
                    >
                      {lang === "sk" ? "Hárky:" : "Sheets:"}{" "}
                      {m.ocr_done_sheets}/{totalVillageSheets}
                    </span>
                  )}
                </div>

                {/* Optional Row 2: Extracted houses & persons badges when present */}
                {(m.overview_houses_count > 0 || extractedPersons > 0) && (
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    {m.overview_houses_count > 0 && (
                      <span className="px-1.5 py-0.5 rounded bg-purple-100 text-purple-900 font-semibold">
                        {lang === "sk" ? "Domy:" : "Houses:"} {m.overview_houses_count}
                      </span>
                    )}
                    {extractedPersons > 0 && (
                      <span className="px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-900 font-semibold">
                        {lang === "sk" ? "Osoby:" : "Persons:"} {extractedPersons}
                      </span>
                    )}
                  </div>
                )}

                {/* Bottom Row: Full-width 2-column action buttons (Admin Debug Overlay only) */}
                {featureFlags.admin_debug_overlay && (
                  <div
                    className={`grid gap-1.5 pt-1.5 border-t ${
                      isSelected ? "border-amber-200/80" : "border-stone-100"
                    } ${
                      featureFlags.census_overview_sheets
                        ? "grid-cols-2"
                        : "grid-cols-1"
                    }`}
                  >
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
                          ? "Stiahnuť katalóg hárkov obce z archívu Slovakiana"
                          : "Fetch municipality sheet catalog from Slovakiana archive"
                      }
                      className={`px-2 py-1 min-h-[28px] rounded-md text-[11px] font-semibold inline-flex items-center justify-center gap-1 cursor-pointer disabled:opacity-50 transition border ${
                        isSelected
                          ? "bg-stone-900 hover:bg-stone-800 text-amber-300 border-stone-900"
                          : "bg-stone-100 hover:bg-stone-900 text-stone-700 hover:text-amber-300 border-stone-200"
                      }`}
                    >
                      {isHarvesting ? (
                        <RefreshCw className="w-3 h-3 animate-spin shrink-0" />
                      ) : (
                        <FolderDown className="w-3 h-3 shrink-0" />
                      )}
                      <span className="truncate">
                        {lang === "sk" ? "Stiahnuť z archívu" : "Fetch catalog"}
                      </span>
                    </button>

                    {featureFlags.census_overview_sheets && (
                      <button
                        type="button"
                        data-testid={`village-ocr-overviews-btn-${foldMunicipalityKey(m.municipality).replace(/\s+/g, "-")}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleOcrOverviewOnlyMunicipality(
                            m.municipality,
                            m.slovakiana_tag_key || m.municipality_key
                          );
                        }}
                        disabled={busyAction !== null}
                        title={
                          lang === "sk"
                            ? "Stiahnuť všetky skeny obce, detegovať typy hárkov a spustiť AI OCR iba pre prehľady obvodov (bez pomalého OCR všetkých rodinných hárkov)"
                            : "Download all village scans, run sheet type detection, and OCR only the overview sheets"
                        }
                        className={`px-2 py-1 min-h-[28px] rounded-md text-[11px] font-semibold inline-flex items-center justify-center gap-1 cursor-pointer disabled:opacity-50 transition border ${
                          isSelected
                            ? "bg-amber-600 hover:bg-amber-700 text-white border-amber-600"
                            : "bg-amber-50/80 hover:bg-amber-600 text-amber-950 hover:text-white border-amber-200"
                        }`}
                      >
                        {isOcrOverviews ? (
                          <RefreshCw className="w-3 h-3 animate-spin shrink-0" />
                        ) : (
                          <Building2 className="w-3 h-3 shrink-0" />
                        )}
                        <span className="truncate">
                          {lang === "sk" ? "OCR prehľadov" : "OCR overviews"}
                        </span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </aside>
      )}

      {/* RIGHT MAIN WORKSPACE */}
      <section
        className={`${
          villageMenuCollapsed ? "lg:col-span-12" : "lg:col-span-9"
        } space-y-4`}
      >
        {/* Unified Header & Action Bar */}
        <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
          {/* Row 1: Identity & Primary Action Toolbar */}
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                {villageMenuCollapsed && (
                  <button
                    type="button"
                    onClick={toggleVillageMenu}
                    data-testid="toggle-village-menu-btn"
                    aria-expanded={false}
                    aria-label={
                      lang === "sk"
                        ? "Rozbaliť výber obcí"
                        : "Expand municipalities menu"
                    }
                    title={
                      lang === "sk"
                        ? "Rozbaliť výber obcí (Okresy a obce)"
                        : "Expand Districts & Municipalities menu"
                    }
                    className="px-2.5 py-1.5 min-h-[32px] rounded-lg bg-stone-100 hover:bg-amber-50 border border-stone-300 hover:border-amber-400 text-stone-800 hover:text-amber-900 text-xs font-bold inline-flex items-center gap-1.5 cursor-pointer transition shadow-2xs"
                  >
                    <PanelLeftOpen className="w-4 h-4 text-amber-600 shrink-0" />
                    <MapPin className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                    <span>
                      {lang === "sk" ? "Okresy a obce" : "Districts & Municipalities"}
                    </span>
                  </button>
                )}
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
                <div
                  data-testid="workspace-year-switch"
                  role="group"
                  aria-label={
                    lang === "sk" ? "Prepnúť rok sčítania" : "Switch census year"
                  }
                  className="inline-flex items-center rounded-lg bg-stone-100 p-0.5 border border-stone-300 text-xs"
                >
                  {(["1930", "1940"] as const).map((yr) => {
                    const active = selectedYearFilter === yr;
                    return (
                      <button
                        key={yr}
                        type="button"
                        data-testid={`workspace-year-btn-${yr}`}
                        aria-pressed={active}
                        onClick={() => handleSwitchYear(yr)}
                        title={
                          selectedMunicipality
                            ? lang === "sk"
                              ? `Prepnúť obec ${selectedMunicipality} na rok ${yr}`
                              : `Switch ${selectedMunicipality} to year ${yr}`
                            : lang === "sk"
                            ? `Prepnúť na sčítanie ${yr}`
                            : `Switch to ${yr} census`
                        }
                        className={`px-2.5 py-1 min-h-[28px] rounded-md font-bold transition cursor-pointer ${
                          active
                            ? "bg-stone-900 text-amber-300 shadow-2xs"
                            : "text-stone-700 hover:bg-stone-200/80"
                        }`}
                      >
                        {lang === "sk" ? `Rok ${yr}` : `Year ${yr}`}
                      </button>
                    );
                  })}
                </div>
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

            {/* Unified Action Group & Jobs History Popover */}
            <div className="flex flex-wrap items-center justify-end gap-2">
              <div ref={splitMenuRef} className="relative inline-flex">
                <div className="inline-flex items-stretch rounded-lg border border-stone-300 bg-white shadow-2xs overflow-hidden">
                  {featureFlags.admin_debug_overlay && (
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
                  )}

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

                  {featureFlags.admin_debug_overlay && (
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
                  )}
                </div>

                {splitMenuOpen && (
                  <div className="absolute right-0 top-full mt-1.5 z-30 w-80 bg-white rounded-lg border border-stone-200 shadow-xl p-1 space-y-0.5">
                    {featureFlags.census_overview_sheets && (
                      <button
                        type="button"
                        onClick={() => {
                          setSplitMenuOpen(false);
                          handleOcrOverviewOnlyMunicipality();
                        }}
                        disabled={busyAction !== null}
                        className="w-full text-left px-3 py-2 rounded-md hover:bg-purple-50 text-stone-800 text-xs font-semibold flex items-start gap-2 cursor-pointer disabled:opacity-50"
                      >
                        <Building2 className="w-4 h-4 text-purple-700 shrink-0 mt-0.5" />
                        <div>
                          <div>
                            {lang === "sk"
                              ? "Stiahnuť hárky a OCR iba prehľadov"
                              : "Download all & OCR overview sheets only"}
                          </div>
                          <div className="text-[11px] font-normal text-stone-500">
                            {lang === "sk"
                              ? "Stiahne všetky skeny obce, deteguje typ hárku a prečíta len prehľady obvodov a domov"
                              : "Downloads all scans, runs type detection, and OCRs only overview sheets"}
                          </div>
                        </div>
                      </button>
                    )}
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
                                    disabled={jobActionId === job.id}
                                    onClick={() => handlePauseResumeJob(job)}
                                    className="px-2 py-1 rounded bg-stone-800 hover:bg-stone-700 disabled:opacity-50 text-white text-xs font-semibold inline-flex items-center gap-1 cursor-pointer shrink-0"
                                  >
                                    {jobActionId === job.id ? (
                                      <RefreshCw className="w-3 h-3 animate-spin" />
                                    ) : job.status === "paused" ? (
                                      <Play className="w-3 h-3" />
                                    ) : (
                                      <Pause className="w-3 h-3" />
                                    )}
                                    <span>
                                      {job.status === "paused"
                                        ? lang === "sk"
                                          ? "Pokračovať"
                                          : "Resume"
                                        : lang === "sk"
                                        ? "Pozastaviť"
                                        : "Pause"}
                                    </span>
                                  </button>
                                )}
                                <button
                                  type="button"
                                  disabled={jobActionId === job.id}
                                  onClick={() => handleDeleteJob(job.id)}
                                  title={lang === "sk" ? "Odstrániť úlohu z histórie" : "Remove job"}
                                  className="p-1 rounded hover:bg-stone-200 disabled:opacity-50 text-stone-500 hover:text-red-700 cursor-pointer"
                                >
                                  {jobActionId === job.id ? (
                                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                  ) : (
                                    <X className="w-3.5 h-3.5" />
                                  )}
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

          {/* Immediate Starting Banner when busyAction is triggered before activeRunningJob appears */}
          {busyAction && !activeRunningJob && (
            <div className="px-3 py-2 rounded-lg bg-amber-50 border border-amber-300 flex items-center gap-2 text-xs font-semibold text-stone-900">
              <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin shrink-0" />
              <span>
                {lang === "sk"
                  ? "Spúšťam sťahovanie a spracovanie hárkov na serveri…"
                  : "Starting sheet download and processing on the server…"}
              </span>
            </div>
          )}

          {/* Slim 1-Line Active Job Progress Bar (only when running or paused and not dismissed) */}
          {activeRunningJob && (
            <div className="px-3 py-2 rounded-lg bg-amber-50/90 border border-amber-300 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-2 min-w-0 flex-1">
                {activeRunningJob.status === "running" ? (
                  <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin shrink-0" />
                ) : (
                  <Pause className="w-3.5 h-3.5 text-stone-600 shrink-0" />
                )}
                <span className="font-bold text-stone-900 truncate">
                  {activeRunningJob.name}
                </span>
                {(activeRunningJob.current_slovakiana_key ||
                  livePipelineState?.current_slovakiana_key) && (
                  <Link
                    to={`/scitacie-harky/${
                      activeRunningJob.current_slovakiana_key ||
                      livePipelineState?.current_slovakiana_key
                    }`}
                    className="px-2 py-0.5 rounded bg-amber-200/90 hover:bg-amber-300 text-stone-900 border border-amber-400 font-mono font-bold text-[11px] inline-flex items-center gap-1 shrink-0"
                  >
                    <FileSpreadsheet className="w-3 h-3 text-amber-800" />
                    <span>
                      {activeRunningJob.current_slovakiana_key ||
                        livePipelineState?.current_slovakiana_key}
                      {activeRunningJob.current_sheet_label
                        ? ` · ${activeRunningJob.current_sheet_label}`
                        : ""}
                    </span>
                  </Link>
                )}
                <div className="w-28 sm:w-36 h-2 bg-amber-200/80 rounded-full overflow-hidden shrink-0">
                  <div
                    className="h-full bg-amber-600 rounded-full transition-all"
                    style={{
                      width: `${Math.max(
                        8,
                        Math.min(
                          100,
                          Math.round(
                            ((activeRunningJob.job_type === "catalog_harvest"
                              ? activeRunningJob.cataloged_sheets
                              : Math.max(
                                  activeRunningJob.processed_sheets,
                                  activeRunningJob.downloaded_sheets
                                ) +
                                (livePipelineState?.active_stage &&
                                livePipelineState.active_stage !== "idle"
                                  ? (livePipelineState.progress_pct || 10) / 100
                                  : 0)) /
                              Math.max(
                                activeRunningJob.total_sheets || 0,
                                activeRunningJob.cataloged_sheets || 0,
                                1
                              )) *
                              100
                          )
                        )
                      )}%`,
                    }}
                  />
                </div>
                <span className="text-stone-700 truncate">
                  {livePipelineState?.active_stage === "census_harvest" &&
                  livePipelineState?.message_sk ? (
                    <span>
                      {lang === "sk"
                        ? livePipelineState.message_sk
                        : livePipelineState.message_en || livePipelineState.message_sk}
                    </span>
                  ) : (
                    <>
                      Katalóg: <strong>{activeRunningJob.cataloged_sheets}</strong> · Stiahnuté:{" "}
                      <strong>
                        {activeRunningJob.downloaded_sheets}
                        {activeRunningJob.total_sheets
                          ? `/${activeRunningJob.total_sheets}`
                          : ""}
                      </strong>{" "}
                      · Prečítané: <strong>{activeRunningJob.processed_sheets}</strong>
                    </>
                  )}
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

          {/* Row 2: Unified Search Input & Person/Birth-Year/Village Filters */}
          <div className="space-y-2.5 pt-2.5 border-t border-stone-200">
            <div className="flex flex-wrap items-center justify-between gap-2.5">
              {/* General Quick Search Input */}
              <div className="relative flex-1 min-w-[220px] max-w-md">
                <Search className="w-3.5 h-3.5 text-stone-500 absolute left-2.5 top-2.5" />
                <input
                  type="text"
                  value={workspaceSearch}
                  onChange={(e) => setWorkspaceSearch(e.target.value)}
                  placeholder={
                    lang === "sk"
                      ? "Rýchle hľadanie (dom č., majiteľ, osoba, cair-...)..."
                      : "Quick search (house #, owner, person, cair-...)..."
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

              <div className="flex flex-wrap items-center gap-2 justify-end">
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

                {Boolean(selectedMunicipality) && featureFlags.census_overview_sheets && (
                  <button
                    type="button"
                    onClick={() =>
                      setActiveTab((prev) => (prev === "overview" ? "sheets" : "overview"))
                    }
                    className="px-3 py-1.5 min-h-[34px] rounded-lg bg-stone-100 hover:bg-stone-200 border border-stone-300 text-stone-700 text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer transition"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5 text-stone-600" />
                    <span>
                      {activeTab === "overview"
                        ? lang === "sk"
                          ? `Surové karty hárkov (${totalSheets})`
                          : `Raw sheet cards (${totalSheets})`
                        : lang === "sk"
                        ? "← Späť na adresár domov obce"
                        : "← Back to village house directory"}
                    </span>
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

              if (!isSearchingCensus && !hasAnyActiveFilter) {
                return null;
              }
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
            {/* Unified Section: Obecný prehľad (Súhrn obvodov) + Prehľady sčítacích obvodov (Prehľady domov) */}
            {(!muniOverview?.overview_sheets ||
              muniOverview.overview_sheets.length === 0) &&
            (!muniOverview?.municipal_overview_sheets ||
              muniOverview.municipal_overview_sheets.length === 0) ? (
              <div className="bg-white rounded-xl px-4 py-2.5 border border-stone-300 flex flex-wrap items-center justify-between gap-2 text-xs sm:text-sm">
                <div className="flex items-center gap-2 font-bold text-stone-800">
                  <Building2 className="w-4 h-4 text-purple-700 shrink-0" />
                  <span>
                    {lang === "sk"
                      ? "Prehľady sčítacích obvodov (0)"
                      : "District Overview Sheets (0)"}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2.5">
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
                  <button
                    type="button"
                    data-testid="empty-overview-ocr-btn"
                    onClick={() => handleOcrOverviewOnlyMunicipality()}
                    disabled={busyAction !== null}
                    className="px-3 py-1.5 min-h-[32px] rounded-lg bg-stone-900 hover:bg-stone-800 text-amber-300 font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50 transition"
                  >
                    {busyAction?.startsWith("overview_ocr_") ? (
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Building2 className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {lang === "sk"
                        ? "Stiahnuť hárky a OCR iba prehľadov"
                        : "Download all & OCR overviews only"}
                    </span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-xl px-4 py-3 border border-stone-200 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm">
                <div className="flex flex-wrap items-center gap-2 min-w-0">
                  <Building2 className="w-4 h-4 text-amber-600 shrink-0" />
                  <h2 className="font-bold text-stone-900">
                    {lang === "sk"
                      ? "Prehľady sčítacích obvodov"
                      : "District Overview Sheets"}{" "}
                    ({muniOverview?.overview_sheets.length ?? 0})
                  </h2>
                  <span className="text-stone-600">
                    {lang === "sk"
                      ? `· ${muniOverview?.total_houses ?? 0} zaznamenaných domov v obci`
                      : `· ${muniOverview?.total_houses ?? 0} recorded houses in municipality`}
                  </span>
                  {(muniOverview?.municipal_overview_sheets || []).map((mov) => {
                    const dists = mov.municipal_overview_districts || [];
                    const distOnly = dists.filter((d) => !d.is_summary_row);
                    const sumRow = dists.find((d) => d.is_summary_row) || null;
                    const totPersons =
                      sumRow?.present_persons_count ??
                      distOnly.reduce((a, d) => a + (d.present_persons_count || 0), 0);
                    return (
                      <span
                        key={mov.slovakiana_id}
                        className="inline-flex items-center gap-1.5 text-stone-700"
                      >
                        <span>
                          · {lang === "sk" ? "Prítomné osoby:" : "Present residents:"}{" "}
                          <strong>{totPersons}</strong>
                        </span>
                        <Link
                          to={buildSheetDetailLink(mov.slovakiana_id)}
                          className="px-2 py-0.5 rounded bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 font-semibold text-xs inline-flex items-center gap-1"
                        >
                          <span>
                            {lang === "sk" ? "Obecný prehľad" : "Municipal overview"} ({mov.slovakiana_id}) ↗
                          </span>
                        </Link>
                      </span>
                    );
                  })}
                </div>

                {/* Secondary Links to Raw Overview Sheets */}
                <div className="flex flex-wrap items-center gap-1.5">
                  {(muniOverview?.overview_sheets || []).map((ov) => (
                    <Link
                      key={ov.slovakiana_id}
                      to={buildSheetDetailLink(ov.slovakiana_id)}
                      className="px-2.5 py-1 rounded-md bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 text-xs font-semibold inline-flex items-center gap-1 transition"
                      title={
                        lang === "sk"
                          ? `Otvoriť zdrojový hárok prehľadu obvodu (${ov.slovakiana_id})`
                          : `Open raw district overview sheet (${ov.slovakiana_id})`
                      }
                    >
                      <FileSpreadsheet className="w-3.5 h-3.5 text-amber-600" />
                      <span>
                        {lang === "sk" ? "Zdrojový prehľad obvodu" : "Raw district overview"}{" "}
                        {ov.obvod_number ? `č. ${ov.obvod_number}` : `(${ov.slovakiana_id})`} ↗
                      </span>
                    </Link>
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
                    {(isSearchingCensus || loading) && (
                      <span className="px-2 py-0.5 rounded-full bg-amber-50 border border-amber-300 text-amber-800 text-xs font-semibold inline-flex items-center gap-1">
                        <RefreshCw className="w-3 h-3 animate-spin" />
                        <span>{lang === "sk" ? "Načítavam…" : "Loading…"}</span>
                      </span>
                    )}
                  </h2>
                  <p className="text-xs sm:text-sm text-stone-600">
                    {lang === "sk"
                      ? "Prepojenie záznamov z Prehľadu sčítacieho obvodu (majiteľ domu, počet osôb) s konkrétnymi popisnými sčítacími hárkami rodín (`Dom č. 1 .. N`)."
                      : "Cross-reference of District Overview entries (house owner, resident count) with household census sheets (`Dom č. 1 .. N`)."}
                  </p>
                </div>
              </div>

              {filteredHousesDirectory.length === 0 ? (
                loading || isSearchingCensus ? (
                  <div className="py-10 text-center text-stone-600 text-sm flex flex-col items-center justify-center gap-2">
                    <RefreshCw className="w-6 h-6 text-amber-600 animate-spin" />
                    <span className="font-semibold text-stone-800">
                      {lang === "sk"
                        ? `Načítavam adresár domov a rodín pre obec ${selectedMunicipality}…`
                        : `Loading house & family directory for ${selectedMunicipality}…`}
                    </span>
                  </div>
                ) : (
                  <div className="text-center py-8 text-stone-600 text-sm">
                    {lang === "sk"
                      ? "Pre tento filter sa nenašli žiadne domy. Skúste vybrať obec v ľavom paneli alebo stiahnuť hárky zo Slovakiany."
                      : "No houses match the current filter. Select a municipality on the left or fetch sheets from Slovakiana."}
                  </div>
                )
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
                                {entry.obvod_number && (
                                  <span className="px-1.5 py-0.5 rounded-full bg-purple-100 text-purple-900 border border-purple-200 text-[11px] font-bold">
                                    Obvod {entry.obvod_number}
                                  </span>
                                )}
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
                                  {renderAttachedSheetProgress(
                                    entry.slovakiana_key,
                                    entry.sheet_id
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
                                    disabled={Boolean(
                                      getAttachedSheetState(
                                        entry.slovakiana_key,
                                        entry.sheet_id
                                      )?.active
                                    )}
                                    title="Spustiť AI OCR pre tento dom"
                                    className="px-2.5 py-1.5 min-h-[32px] rounded-md bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                                  >
                                    {getAttachedSheetState(
                                      entry.slovakiana_key,
                                      entry.sheet_id
                                    )?.active ? (
                                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                    ) : (
                                      <Download className="w-3.5 h-3.5" />
                                    )}
                                    <span>
                                      {getAttachedSheetState(
                                        entry.slovakiana_key,
                                        entry.sheet_id
                                      )?.active
                                        ? `OCR (${
                                            getAttachedSheetState(
                                              entry.slovakiana_key,
                                              entry.sheet_id
                                            )?.pct
                                          }%)`
                                        : "OCR"}
                                    </span>
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

            {loading || (isSearchingCensus && filteredSheets.length === 0) ? (
              <div className="py-10 text-center text-stone-600 text-sm flex flex-col items-center justify-center gap-2">
                <RefreshCw className="w-6 h-6 text-amber-600 animate-spin" />
                <span className="font-semibold text-stone-800">
                  {lang === "sk"
                    ? "Načítavam sčítacie hárky…"
                    : "Loading census sheets…"}
                </span>
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

                <div
                  className={`grid grid-cols-1 md:grid-cols-2 ${
                    villageMenuCollapsed
                      ? "lg:grid-cols-3 2xl:grid-cols-4"
                      : "xl:grid-cols-3"
                  } gap-3`}
                >
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
                    const isMunicipalOverview = s.sheet_type === "municipal_overview";
                    const isOverview = s.sheet_type === "overview" || isMunicipalOverview;
                    const houseCount =
                      overview_houses_count ?? s.overview_houses_count ?? 0;
                    const distCount = s.municipal_districts_count ?? 0;
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
                          getAttachedSheetState(s.slovakiana_id, s.id)?.active
                            ? "bg-amber-50/70 border-amber-500 ring-2 ring-amber-400/70 shadow-sm"
                            : isMunicipalOverview
                            ? "bg-indigo-50/40 border-indigo-200 hover:border-indigo-500"
                            : isOverview
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
                              {isMunicipalOverview ? (
                                <span className="px-2 py-0.5 rounded-full bg-indigo-200 text-indigo-950 text-xs font-bold">
                                  {lang === "sk" ? "Obecný prehľad" : "Municipal overview"}
                                </span>
                              ) : isOverview ? (
                                <span className="px-2 py-0.5 rounded-full bg-purple-200 text-purple-950 text-xs font-bold">
                                  {lang === "sk" ? "Prehľad obvodu" : "District overview"}{" "}
                                  {s.obvod_number ? `č. ${s.obvod_number}` : ""}
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

                        {isMunicipalOverview ? (
                          <div className="bg-indigo-100/70 border border-indigo-200 rounded-lg px-2.5 py-1.5 text-xs text-indigo-950 flex items-center justify-between">
                            <span className="truncate">
                              {s.commissioner_name
                                ? `${lang === "sk" ? "Revízor:" : "Revisor:"} ${s.commissioner_name}`
                                : lang === "sk"
                                ? "Súhrn sčítacích obvodov obce"
                                : "Summary of census districts"}
                            </span>
                            <span className="ml-2 px-1.5 py-0.5 rounded bg-indigo-200 font-bold shrink-0">
                              {distCount} {lang === "sk" ? "obvody" : "districts"}
                            </span>
                          </div>
                        ) : isOverview ? (
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

                        {renderAttachedSheetProgress(s.slovakiana_id, s.id)}

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
                            disabled={Boolean(
                              getAttachedSheetState(s.slovakiana_id, s.id)?.active
                            )}
                            className="px-3 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                          >
                            {getAttachedSheetState(s.slovakiana_id, s.id)?.active ? (
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Download className="w-3.5 h-3.5" />
                            )}
                            <span>
                              {getAttachedSheetState(s.slovakiana_id, s.id)?.active
                                ? `OCR (${
                                    getAttachedSheetState(s.slovakiana_id, s.id)?.pct
                                  }%)…`
                                : s.download_status === "downloaded" ||
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
