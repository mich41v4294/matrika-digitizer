import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  BookOpen,
  CheckCircle2,
  Edit3,
  Eye,
  FileSpreadsheet,
  Play,
  RefreshCw,
  Save,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import ZoomableScanViewer, { HoverZoomRowCrop } from "../components/ZoomableScanViewer";
import { useFeatureFlags } from "../featureFlags";

interface ScanSummary {
  id: number;
  batch_id?: number | null;
  filename: string;
  status: string;
  error_message?: string | null;
  detected_year: string | null;
  detected_type?: string | null;
  rows_count: number;
  open_questions: number;
  original_url?: string | null;
  overlay_url: string | null;
  header_crop_url?: string | null;
  header_crop_path?: string | null;
  width?: number;
  height?: number;
}

interface BatchSummary {
  id: number;
  name: string;
  parish_or_book?: string | null;
  record_type_hint?: string | null;
}

interface ScanRowDetail {
  id: number;
  row_index: number;
  entry_number: string | null;
  y_start: number;
  y_end: number;
  stitched_crop_url: string | null;
  stitched_crop_path: string | null;
  raw_ocr_combined: string | null;
  status: string;
  fact?: RecordItem["fact"] | null;
}

interface SelectedScanDetail {
  scan: ScanSummary;
  original_url: string | null;
  overlay_url: string | null;
  header_crop_path: string | null;
  width: number;
  height: number;
  rows: ScanRowDetail[];
}

interface RecordItem {
  fact: {
    id: number;
    row_id: number;
    scan_id: number;
    record_type?: string;
    census_sheet_id?: number | null;
    relation_to_head?: string | null;
    social_status_or_occupation?: string | null;
    birth_place_raw?: string | null;
    entry_number: string | null;
    event_year?: number | null;
    birth_or_event_date: string | null;
    baptism_or_burial_date: string | null;
    given_name_raw: string | null;
    given_name_sk?: string | null;
    surname_raw: string | null;
    surname_normalized: string | null;
    father_name: string | null;
    mother_name: string | null;
    mother_maiden_surname: string | null;
    village_raw: string | null;
    village_sk: string | null;
    fs_place_standardized: string | null;
    house_number: string | null;
    godparents_or_witnesses: string | null;
    observationes_raw: string | null;
    overall_confidence: number;
  };
  observationes_events: Array<{
    event_type: string;
    type?: string;
    date?: string;
    year?: number | null;
    raw_note: string;
  }>;
  street_name?: string | null;
  stitched_crop_url: string | null;
  scan_filename: string | null;
  scan_url?: string | null;
}

interface CatalogSheetHit {
  id: number;
  slovakiana_id: string;
  census_year: number;
  district_name: string | null;
  municipality_name: string;
  street_name: string | null;
  house_number_raw: string | null;
  house_number_conscription: string | null;
  is_blackened: boolean;
  pipeline_status: string;
}

interface Props {
  lang: "sk" | "en";
  refreshStatus: () => void;
  mode?: "scans" | "search" | "all";
}

function resolveScanMediaUrl(
  explicitUrl?: string | null,
  filename?: string | null
): string | null {
  if (explicitUrl && explicitUrl.trim()) return explicitUrl.trim();
  if (!filename) return null;
  if (filename.startsWith("cair-")) {
    const cairKey = filename.replace(/_[vr]\.[a-z0-9]+$/i, "");
    return `/media/uploads/slovakiana/${cairKey}/${filename}`;
  }
  return `/media/uploads/${filename}`;
}

export default function BooksOverview({
  lang,
  refreshStatus,
  mode = "all",
}: Props) {
  const { flags: featureFlags } = useFeatureFlags();
  const { bookId: routeBookId, scanId: routeScanId } = useParams<{
    bookId?: string;
    scanId?: string;
  }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [batches, setBatches] = useState<BatchSummary[]>([]);
  const [showCensusScans, setShowCensusScans] = useState<boolean>(false);
  const [records, setRecords] = useState<RecordItem[]>([]);
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [scannedTotal, setScannedTotal] = useState<number>(0);
  const [catalogSheets, setCatalogSheets] = useState<CatalogSheetHit[]>([]);
  const [loadingInitial, setLoadingInitial] = useState<boolean>(true);
  const [isSearching, setIsSearching] = useState<boolean>(true);
  const [searchProgress, setSearchProgress] = useState<{
    running: boolean;
    processed: number;
    total: number;
    matched_records: number;
    matched_sheets: number;
    progress_pct: number;
    stage_sk: string;
    stage_en: string;
    elapsed_ms: number;
  }>({
    running: true,
    processed: 0,
    total: 0,
    matched_records: 0,
    matched_sheets: 0,
    progress_pct: 15,
    stage_sk: "Prehľadávam matriky a sčítacie hárky…",
    stage_en: "Searching registers and census sheets…",
    elapsed_ms: 0,
  });
  const searchAbortRef = useRef<AbortController | null>(null);
  const isSearchingRef = useRef<boolean>(false);
  const hasLoadedScansOnceRef = useRef<boolean>(false);
  const [recordsPage, setRecordsPage] = useState<number>(1);
  const RECORDS_PER_PAGE = 25;
  const [searchQuery, setSearchQuery] = useState(() => searchParams.get("q") || "");
  const [recordTypeFilter, setRecordTypeFilter] = useState(
    () => searchParams.get("type") || ""
  );
  const [villageFilter, setVillageFilter] = useState(
    () => searchParams.get("obec") || searchParams.get("village") || ""
  );
  const [streetFilter, setStreetFilter] = useState(
    () => searchParams.get("ulica") || searchParams.get("street") || ""
  );
  const [houseNumberFilter, setHouseNumberFilter] = useState(
    () => searchParams.get("dom") || searchParams.get("house") || ""
  );
  const [selectedScanId, setSelectedScanId] = useState<number | null>(() => {
    const raw =
      routeScanId || routeBookId || searchParams.get("scan") || searchParams.get("book");
    const num = raw ? Number(raw) : NaN;
    return Number.isFinite(num) && num > 0 ? num : null;
  });
  const [filterRecordsByScan, setFilterRecordsByScan] = useState<boolean>(
    () => searchParams.get("only_scan") === "1"
  );
  const [selectedScanDetail, setSelectedScanDetail] = useState<SelectedScanDetail | null>(
    null
  );
  const [selectedRowId, setSelectedRowId] = useState<number | null>(() => {
    const r = Number(searchParams.get("row"));
    return Number.isFinite(r) && r > 0 ? r : null;
  });
  const [activeScanTab, setActiveScanTab] = useState<"overlay" | "original">(() =>
    searchParams.get("mode") === "original" ? "original" : "overlay"
  );
  const [processingId, setProcessingId] = useState<number | null>(null);

  // Inline Edit & Delete State for Records
  const [editingFactId, setEditingFactId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<Record<string, any>>({});
  const [savingFactId, setSavingFactId] = useState<number | null>(null);
  const [confirmDeleteFactId, setConfirmDeleteFactId] = useState<number | null>(null);
  const [deletingFactId, setDeletingFactId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  // Sync routeBookId and routeScanId URL params
  useEffect(() => {
    const raw = routeScanId || routeBookId;
    if (raw) {
      const num = Number(raw);
      if (Number.isFinite(num) && num > 0) {
        setSelectedScanId(num);
      }
    }
  }, [routeBookId, routeScanId]);

  useEffect(() => {
    const next = new URLSearchParams();
    if (selectedScanId !== null) next.set("scan", String(selectedScanId));
    if (selectedRowId !== null) next.set("row", String(selectedRowId));
    if (activeScanTab === "original") next.set("mode", "original");
    if (filterRecordsByScan) next.set("only_scan", "1");
    if (searchQuery.trim()) next.set("q", searchQuery.trim());
    if (recordTypeFilter) next.set("type", recordTypeFilter);
    if (villageFilter.trim()) next.set("obec", villageFilter.trim());
    if (streetFilter.trim()) next.set("ulica", streetFilter.trim());
    if (houseNumberFilter.trim()) next.set("dom", houseNumberFilter.trim());
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [
    selectedScanId,
    selectedRowId,
    activeScanTab,
    filterRecordsByScan,
    searchQuery,
    recordTypeFilter,
    villageFilter,
    streetFilter,
    houseNumberFilter,
  ]);

  // Listen for live search_progress broadcasts over /ws/pipeline
  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "search_progress" && msg.data && msg.data.scope === "unified") {
            const d = msg.data;
            if (d.running && isSearchingRef.current) {
              setSearchProgress((prev) => ({
                ...prev,
                running: true,
                processed: Number(d.processed ?? prev.processed),
                total: Number(d.total ?? prev.total),
                matched_records: Number(d.matched_records ?? prev.matched_records),
                matched_sheets: Number(d.matched_sheets ?? prev.matched_sheets),
                progress_pct: Math.max(prev.progress_pct, Number(d.progress_pct ?? 20)),
                stage_sk: d.stage_sk || prev.stage_sk,
                stage_en: d.stage_en || prev.stage_en,
                elapsed_ms: Number(d.elapsed_ms ?? prev.elapsed_ms),
              }));
            }
          }
        } catch {
          // ignore ws parse error
        }
      };
    } catch {
      // ignore ws connection error
    }
    return () => {
      if (ws) ws.close();
    };
  }, []);

  // Smooth client-side progress ticker while a search request is in flight
  useEffect(() => {
    if (!isSearching) return;
    const t0 = performance.now();
    const timer = window.setInterval(() => {
      setSearchProgress((prev) => {
        if (!prev.running) return prev;
        const nextPct = Math.min(92, prev.progress_pct + Math.max(2, Math.round((94 - prev.progress_pct) * 0.22)));
        const estTotal = Math.max(prev.total, scannedTotal, totalRecords, 1);
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
  }, [isSearching, scannedTotal, totalRecords]);

  const loadData = async (isBackgroundPoll = false) => {
    if (isBackgroundPoll && isSearchingRef.current) {
      return;
    }

    const recParams = new URLSearchParams();
    if (searchQuery.trim()) recParams.set("q", searchQuery.trim());
    if (recordTypeFilter) recParams.set("record_type", recordTypeFilter);
    if (villageFilter.trim()) recParams.set("village", villageFilter.trim());
    if (streetFilter.trim()) recParams.set("street", streetFilter.trim());
    if (houseNumberFilter.trim()) recParams.set("house_number", houseNumberFilter.trim());
    if (filterRecordsByScan && selectedScanId) {
      recParams.set("scan_id", String(selectedScanId));
    }

    let controller: AbortController | null = null;
    const tStart = performance.now();
    if (!isBackgroundPoll) {
      if (searchAbortRef.current) {
        searchAbortRef.current.abort();
      }
      controller = new AbortController();
      searchAbortRef.current = controller;
      isSearchingRef.current = true;
      setIsSearching(true);
      setSearchProgress((prev) => ({
        running: true,
        processed: 0,
        total: Math.max(scannedTotal, totalRecords, prev.total, 1),
        matched_records: records.length,
        matched_sheets: catalogSheets.length,
        progress_pct: 15,
        stage_sk: "Prehľadávam matriky a sčítacie hárky…",
        stage_en: "Searching registers and census sheets…",
        elapsed_ms: 0,
      }));
    }

    try {
      const shouldFetchScans = isBackgroundPoll || !hasLoadedScansOnceRef.current;
      const [scansRes, recRes] = await Promise.all([
        shouldFetchScans
          ? fetch("/api/scans", controller ? { signal: controller.signal } : undefined)
          : Promise.resolve(null),
        fetch(
          `/api/records?${recParams.toString()}`,
          controller ? { signal: controller.signal } : undefined
        ),
      ]);
      if (controller?.signal.aborted) return;

      if (scansRes && scansRes.ok) {
        hasLoadedScansOnceRef.current = true;
        const sData = await scansRes.json();
        const loadedScans: ScanSummary[] = sData.scans || [];
        setScans(loadedScans);
        setBatches(sData.batches || []);
        setSelectedScanId((prev) => {
          if (prev !== null) return prev;
          if (mode === "search") return null;
          const firstMatrika =
            loadedScans.find(
              (s) => !s.filename.startsWith("cair-") && s.detected_type !== "census"
            ) || loadedScans[0];
          return firstMatrika ? firstMatrika.id : null;
        });
      }
      if (recRes.ok) {
        const rData = await recRes.json();
        if (controller?.signal.aborted) return;
        const loadedRecs = rData.records || [];
        const resolvedTotal =
          typeof rData.total === "number" ? rData.total : loadedRecs.length;
        const resolvedScanned =
          typeof rData.scanned_total === "number"
            ? rData.scanned_total
            : typeof rData.search_progress?.total === "number"
            ? rData.search_progress.total
            : Math.max(resolvedTotal, scannedTotal);
        const loadedCatalogSheets = rData.catalog_sheets || [];
        const elapsedMs =
          typeof rData.search_progress?.elapsed_ms === "number" && rData.search_progress.elapsed_ms > 0
            ? rData.search_progress.elapsed_ms
            : Math.max(1, Math.round(performance.now() - tStart));

        setRecords(loadedRecs);
        setTotalRecords(resolvedTotal);
        setScannedTotal(resolvedScanned);
        setCatalogSheets(loadedCatalogSheets);
        if (!isBackgroundPoll) {
          setSearchProgress({
            running: false,
            processed: resolvedScanned,
            total: Math.max(resolvedScanned, 1),
            matched_records: resolvedTotal,
            matched_sheets: loadedCatalogSheets.length,
            progress_pct: 100,
            stage_sk: "Vyhľadávanie dokončené",
            stage_en: "Search complete",
            elapsed_ms: elapsedMs,
          });
        }
      }
    } catch (err: any) {
      if (err?.name === "AbortError") {
        return;
      }
    } finally {
      if (!isBackgroundPoll && (!controller || !controller.signal.aborted)) {
        isSearchingRef.current = false;
        setIsSearching(false);
        setLoadingInitial(false);
      }
    }
  };

  useEffect(() => {
    setRecordsPage(1);
    isSearchingRef.current = true;
    setIsSearching(true);
    setSearchProgress((prev) => ({
      ...prev,
      running: true,
      processed: 0,
      total: Math.max(scannedTotal, totalRecords, prev.total, 1),
      progress_pct: 15,
      stage_sk: "Prehľadávam matriky a sčítacie hárky…",
      stage_en: "Searching registers and census sheets…",
      elapsed_ms: 0,
    }));
    const hasTextQuery = Boolean(
      searchQuery.trim() ||
        villageFilter.trim() ||
        streetFilter.trim() ||
        houseNumberFilter.trim()
    );
    const debounceTimer = window.setTimeout(
      () => {
        loadData(false);
      },
      hasLoadedScansOnceRef.current && hasTextQuery ? 120 : 0
    );
    const pollInterval = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadData(true);
    }, 8000);
    return () => {
      window.clearTimeout(debounceTimer);
      window.clearInterval(pollInterval);
    };
  }, [
    searchQuery,
    recordTypeFilter,
    villageFilter,
    streetFilter,
    houseNumberFilter,
    selectedScanId,
    filterRecordsByScan,
  ]);

  useEffect(() => {
    if (!selectedScanId) {
      setSelectedScanDetail(null);
      return;
    }
    let cancelled = false;
    fetch(`/api/scans/${selectedScanId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.scan) return;
        const scanObj: ScanSummary = data.scan;
        const mappedRows: ScanRowDetail[] = (data.rows || []).map((r: any) => ({
          ...r,
          stitched_crop_url: r.stitched_crop_url || r.stitched_crop_path || null,
          stitched_crop_path: r.stitched_crop_path || r.stitched_crop_url || null,
        }));
        setSelectedScanDetail({
          scan: scanObj,
          original_url: resolveScanMediaUrl(scanObj.original_url, scanObj.filename),
          overlay_url: scanObj.overlay_url || null,
          header_crop_path:
            scanObj.header_crop_path ||
            scanObj.header_crop_url ||
            `/media/crops/scan_${scanObj.id}/header.jpg`,
          width: scanObj.width || 0,
          height: scanObj.height || 0,
          rows: mappedRows,
        });
        setSelectedRowId((prev) => {
          if (prev && mappedRows.some((r) => r.id === prev)) return prev;
          return mappedRows[0]?.id ?? null;
        });
      });
    return () => {
      cancelled = true;
    };
  }, [selectedScanId]);

  const handleProcessScan = async (scanId: number, maxRows: number | null = null) => {
    setProcessingId(scanId);
    setActionError(null);
    try {
      const r = await fetch(`/api/scans/${scanId}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_rows: maxRows, background: true }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa spustiť čítanie strany."
              : "Failed to start scan processing.")
        );
      }
      refreshStatus();
      setTimeout(loadData, 1500);
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia pri spúšťaní spracovania."
          : "Network error while starting scan processing."
      );
    } finally {
      setProcessingId(null);
    }
  };

  const startEditFact = (r: RecordItem["fact"]) => {
    setActionError(null);
    setConfirmDeleteFactId(null);
    setEditingFactId(r.id);
    setEditForm({
      entry_number: r.entry_number ?? "",
      given_name_raw: r.given_name_raw ?? "",
      surname_raw: r.surname_raw ?? "",
      surname_normalized: r.surname_normalized ?? "",
      birth_or_event_date: r.birth_or_event_date ?? "",
      baptism_or_burial_date: r.baptism_or_burial_date ?? "",
      father_name: r.father_name ?? "",
      mother_name: r.mother_name ?? "",
      mother_maiden_surname: r.mother_maiden_surname ?? "",
      village_raw: r.village_raw ?? "",
      house_number: r.house_number ?? "",
      godparents_or_witnesses: r.godparents_or_witnesses ?? "",
      relation_to_head: r.relation_to_head ?? "",
      social_status_or_occupation: r.social_status_or_occupation ?? "",
      birth_place_raw: r.birth_place_raw ?? "",
      observationes_raw: r.observationes_raw ?? "",
    });
  };

  const handleSaveFact = async (factId: number) => {
    setSavingFactId(factId);
    setActionError(null);
    try {
      const res = await fetch(`/api/records/${factId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(editForm),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť zmeny záznamu."
              : "Failed to save record changes.")
        );
        return;
      }
      setEditingFactId(null);
      await loadData();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia so serverom pri ukladaní záznamu."
          : "Network error while saving record."
      );
    } finally {
      setSavingFactId(null);
    }
  };

  const handleDeleteFact = async (factId: number) => {
    setDeletingFactId(factId);
    setActionError(null);
    try {
      const res = await fetch(`/api/records/${factId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa vymazať záznam."
              : "Failed to delete record.")
        );
        return;
      }
      setConfirmDeleteFactId(null);
      if (editingFactId === factId) setEditingFactId(null);
      await loadData();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia pri mazaní záznamu."
          : "Network error while deleting record."
      );
    } finally {
      setDeletingFactId(null);
    }
  };

  const matrikaScans = useMemo(
    () =>
      scans.filter(
        (s) => !s.filename.startsWith("cair-") && s.detected_type !== "census"
      ),
    [scans]
  );
  const censusScansCount = scans.length - matrikaScans.length;
  const visibleScans = showCensusScans ? scans : matrikaScans;

  const selectedRow =
    selectedScanDetail?.rows.find((r) => r.id === selectedRowId) ||
    selectedScanDetail?.rows[0] ||
    null;

  const activeScanImageUrl =
    activeScanTab === "overlay"
      ? selectedScanDetail?.overlay_url || selectedScanDetail?.original_url || null
      : selectedScanDetail?.original_url || null;

  const rowsById = new Map<number, ScanRowDetail>();
  if (selectedScanDetail?.rows) {
    for (const r of selectedScanDetail.rows) {
      rowsById.set(r.id, r);
    }
  }

  const scansById = new Map<number, ScanSummary>();
  for (const s of scans) {
    scansById.set(s.id, s);
  }

  const batchesById = new Map<number, BatchSummary>();
  for (const b of batches) {
    batchesById.set(b.id, b);
  }

  const totalRecordPages = Math.max(1, Math.ceil(records.length / RECORDS_PER_PAGE));
  const safeRecordsPage = Math.min(recordsPage, totalRecordPages);
  const pagedRecords = records.slice(
    (safeRecordsPage - 1) * RECORDS_PER_PAGE,
    safeRecordsPage * RECORDS_PER_PAGE
  );
  const displayTotalCount = totalRecords > records.length ? totalRecords : records.length;

  const scanInspectorSection = (
    <>
      {/* Matrika Scans Selector */}
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-bold text-stone-900 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-stone-700" />
              <span>
                {lang === "sk" ? "Strany matrík" : "Matrika scans"} (
                {visibleScans.length})
              </span>
            </h2>
            {censusScansCount > 0 && (
              <button
                type="button"
                onClick={() => setShowCensusScans((prev) => !prev)}
                className={`px-2.5 py-1 rounded-full text-xs font-semibold cursor-pointer transition ${
                  showCensusScans
                    ? "bg-purple-100 text-purple-900 border border-purple-300"
                    : "bg-stone-100 text-stone-600 hover:bg-stone-200"
                }`}
              >
                {showCensusScans
                  ? lang === "sk"
                    ? `Skryť sčítacie skeny (${censusScansCount})`
                    : `Hide census scans (${censusScansCount})`
                  : lang === "sk"
                  ? `Zobraziť aj sčítacie skeny (+${censusScansCount})`
                  : `Show census scans (+${censusScansCount})`}
              </button>
            )}
          </div>

          {selectedScanId !== null && (
            <button
              type="button"
              onClick={() => setFilterRecordsByScan((prev) => !prev)}
              className={`px-3 py-1.5 min-h-[34px] rounded-lg text-xs font-semibold cursor-pointer ${
                filterRecordsByScan
                  ? "bg-amber-600 text-white"
                  : "bg-stone-100 hover:bg-stone-200 text-stone-700"
              }`}
            >
              {filterRecordsByScan
                ? lang === "sk"
                  ? "Zobraziť v hľadaní všetky strany a sčítania"
                  : "Show all scans & census in search"
                : lang === "sk"
                ? "Filtrovať zoznam nižšie len na vybranú stranu"
                : "Filter list below to selected scan"}
            </button>
          )}
        </div>

        {loadingInitial ? (
          <div className="bg-stone-50 border border-stone-200 rounded-xl p-6 text-center space-y-2">
            <div className="w-6 h-6 rounded-full border-2 border-amber-600 border-t-transparent animate-spin mx-auto" />
            <p className="text-sm text-stone-700 font-medium">
              {lang === "sk" ? "Načítavam strany matrík…" : "Loading matrika scans…"}
            </p>
          </div>
        ) : visibleScans.length === 0 ? (
          <div className="bg-stone-50 border border-stone-200 rounded-xl p-6 text-center space-y-2">
            <p className="text-sm text-stone-600">
              {lang === "sk"
                ? "Zatiaľ nie sú nahrané žiadne strany matrík."
                : "No church register scans have been uploaded yet."}
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
              <Link
                to="/nahrat"
                className="px-3.5 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold inline-flex items-center gap-1.5"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>{lang === "sk" ? "Nahrať fotky matrík" : "Upload scans"}</span>
              </Link>
              {censusScansCount > 0 && !showCensusScans && (
                <button
                  type="button"
                  onClick={() => setShowCensusScans(true)}
                  className="px-3.5 py-1.5 min-h-[34px] rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 text-xs font-semibold cursor-pointer"
                >
                  {lang === "sk"
                    ? `Zobraziť sčítacie skeny (${censusScansCount})`
                    : `Show census scans (${censusScansCount})`}
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {visibleScans.map((s) => (
              <div
                key={s.id}
                className={`rounded-lg px-3.5 py-2 border text-sm flex items-center gap-3 min-w-0 ${
                  selectedScanId === s.id
                    ? "bg-amber-50 border-amber-500 ring-1 ring-amber-400"
                    : "bg-stone-50 border-stone-200"
                }`}
              >
                <button
                  onClick={() => setSelectedScanId((prev) => (prev === s.id ? null : s.id))}
                  className="text-left min-w-0 cursor-pointer"
                  title={s.error_message || s.filename}
                >
                  <div className="font-semibold text-stone-900 break-all sm:break-normal flex flex-wrap items-center gap-1.5">
                    <span className="text-xs font-mono text-stone-600">#{s.id}</span>
                    <span>{s.filename}</span>
                  </div>
                  <div className="text-xs text-stone-600">
                    {s.batch_id && batchesById.get(s.batch_id)?.parish_or_book
                      ? `${batchesById.get(s.batch_id)?.parish_or_book} · `
                      : ""}
                    {s.detected_year ? `${s.detected_year} · ` : ""}
                    {s.status} · {s.rows_count} {lang === "sk" ? "riadkov" : "rows"}
                  </div>
                  {s.error_message && (
                    <div
                      className="text-xs text-red-700"
                      title={s.error_message}
                    >
                      {s.error_message}
                    </div>
                  )}
                </button>
                <button
                  onClick={() => handleProcessScan(s.id)}
                  disabled={processingId === s.id}
                  className="px-3 py-1.5 min-h-[32px] rounded-md bg-stone-900 hover:bg-stone-800 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer shrink-0"
                >
                  <Play className="w-3.5 h-3.5" />
                  {lang === "sk" ? "Prečítať" : "Read"}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Prominent Matrika Scan Overview with Manual Zoom & Row Crop */}
      {selectedScanDetail && (
        <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
                <Eye className="w-4 h-4 text-amber-600" />
                <span>
                  {lang === "sk" ? "Prehľad strany matriky:" : "Matrika Page Overview:"}{" "}
                  <span className="text-amber-800">{selectedScanDetail.scan.filename}</span>
                </span>
              </h3>
              {selectedScanDetail.scan.detected_year && (
                <span className="px-2 py-0.5 rounded-full bg-amber-100 text-amber-900 text-xs font-bold">
                  {selectedScanDetail.scan.detected_year}
                </span>
              )}
              <span className="text-xs text-stone-500">
                ({selectedScanDetail.rows.length}{" "}
                {lang === "sk" ? "detekovaných riadkov" : "detected rows"})
              </span>
            </div>

            {/* Unified Mode Tabs: Originálny sken vs Detekované riadky (Overlay) */}
            <div className="inline-flex rounded-lg bg-stone-100 p-1 border border-stone-200">
              <button
                type="button"
                onClick={() => setActiveScanTab("original")}
                className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                  activeScanTab === "original"
                    ? "bg-amber-600 text-white shadow-xs"
                    : "text-stone-700 hover:text-stone-900"
                }`}
              >
                {lang === "sk" ? "Originálny sken" : "Original Scan"}
              </button>
              <button
                type="button"
                onClick={() => setActiveScanTab("overlay")}
                className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                  activeScanTab === "overlay"
                    ? "bg-amber-600 text-white shadow-xs"
                    : "text-stone-700 hover:text-stone-900"
                }`}
              >
                {lang === "sk" ? "Detekované riadky (Overlay)" : "Detected Rows (Overlay)"}
              </button>
            </div>
          </div>

          <ZoomableScanViewer
            src={activeScanImageUrl || ""}
            alt={selectedScanDetail.scan.filename}
            title={selectedScanDetail.scan.filename}
            subtitle={
              activeScanTab === "overlay"
                ? lang === "sk"
                  ? "Detekované riadky (Overlay)"
                  : "Detected Rows (Overlay)"
                : lang === "sk"
                ? "Originálny sken strany"
                : "Original Page Scan"
            }
            defaultHeightClass="max-h-[420px]"
            expandedHeightClass="max-h-[85vh]"
            lang={lang}
          />

          {/* Row Selector Pills & HoverZoomRowCrop for Selected Row */}
          {selectedScanDetail.rows.length > 0 && (
            <div className="space-y-2 pt-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs font-semibold text-stone-700 mr-1">
                  {lang === "sk" ? "Vyberte riadok strany:" : "Select row on page:"}
                </span>
                {selectedScanDetail.rows.map((r) => {
                  const isSel = selectedRow?.id === r.id;
                  const personName = r.fact
                    ? `${r.fact.given_name_raw || ""} ${
                        r.fact.surname_normalized || r.fact.surname_raw || ""
                      }`.trim()
                    : "";
                  return (
                    <button
                      key={r.id}
                      type="button"
                      onClick={() => setSelectedRowId(r.id)}
                      className={`px-3 py-1.5 min-h-[32px] rounded-lg text-xs font-semibold transition cursor-pointer ${
                        isSel
                          ? "bg-amber-600 text-white"
                          : "bg-stone-100 text-stone-800 hover:bg-stone-200"
                      }`}
                    >
                      #{r.row_index}
                      {r.entry_number ? ` (č. ${r.entry_number})` : ""}
                      {personName ? ` · ${personName}` : ""}
                    </button>
                  );
                })}
              </div>

              {selectedRow &&
                (selectedRow.stitched_crop_path || selectedRow.stitched_crop_url) && (
                  <HoverZoomRowCrop
                    cropUrl={
                      selectedRow.stitched_crop_path || selectedRow.stitched_crop_url || ""
                    }
                    headerCropUrl={selectedScanDetail?.header_crop_path}
                    scanUrl={
                      featureFlags.hover_row_zoom
                        ? selectedScanDetail?.original_url ||
                          selectedScanDetail?.scan?.original_url
                        : null
                    }
                    yStart={featureFlags.hover_row_zoom ? selectedRow.y_start : null}
                    yEnd={featureFlags.hover_row_zoom ? selectedRow.y_end : null}
                    scanHeight={
                      selectedScanDetail?.height || selectedScanDetail?.scan?.height
                    }
                    rowLabel={
                      lang === "sk"
                        ? `Výrez vybraného riadku #${selectedRow.row_index}${
                            selectedRow.entry_number
                              ? ` (záznam č. ${selectedRow.entry_number})`
                              : ""
                          }`
                        : `Selected row crop #${selectedRow.row_index}${
                            selectedRow.entry_number
                              ? ` (entry #${selectedRow.entry_number})`
                              : ""
                          }`
                    }
                    lang={lang}
                  />
                )}
            </div>
          )}
        </div>
      )}
    </>
  );

  const hasActiveSearchFilters = Boolean(
    searchQuery.trim() ||
      recordTypeFilter ||
      villageFilter.trim() ||
      streetFilter.trim() ||
      houseNumberFilter.trim() ||
      filterRecordsByScan
  );
  const activeQuerySummary = [
    searchQuery.trim(),
    villageFilter.trim(),
    streetFilter.trim() ? `${lang === "sk" ? "ul." : "st."} ${streetFilter.trim()}` : "",
    houseNumberFilter.trim() ? `č. ${houseNumberFilter.trim()}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const effectiveScannedTotal = Math.max(
    scannedTotal,
    searchProgress.total,
    displayTotalCount
  );
  const displaySearchPct = isSearching
    ? Math.max(10, Math.min(98, searchProgress.progress_pct || 20))
    : 100;

  const unifiedSearchSection = (
    <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-bold text-stone-900">
          {lang === "sk"
            ? `Globálne vyhľadávanie – Matriky + Sčítacie hárky (${displayTotalCount} osôb)`
            : `Unified Search – Registers + Census (${displayTotalCount} people)`}
        </h3>
        <div className="flex flex-wrap items-center gap-2 flex-1 justify-end">
          <div className="relative min-w-[220px] flex-1 max-w-sm">
            {isSearching ? (
              <RefreshCw className="w-4 h-4 text-amber-600 animate-spin absolute left-3 top-2.5" />
            ) : (
              <Search className="w-4 h-4 text-stone-500 absolute left-3 top-2.5" />
            )}
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={
                lang === "sk"
                  ? "Priezvisko, krstné meno (napr. Ján = Joannes = János), obec…"
                  : "Surname, given name (e.g. Ján = Joannes), village…"
              }
              className="w-full pl-9 pr-8 py-2 rounded-lg border border-stone-300 text-sm"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery("")}
                title={lang === "sk" ? "Vyčistiť hľadanie" : "Clear search"}
                className="absolute right-2 top-2 p-1 text-stone-500 hover:text-stone-800 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          <select
            value={recordTypeFilter}
            onChange={(e) => setRecordTypeFilter(e.target.value)}
            className="px-3 py-2 rounded-lg border border-stone-300 text-sm bg-white"
          >
            <option value="">{lang === "sk" ? "Všetky zdroje" : "All sources"}</option>
            <option value="baptism">
              {lang === "sk" ? "Matrika – Krsty" : "Baptisms"}
            </option>
            <option value="marriage">
              {lang === "sk" ? "Matrika – Sobáše" : "Marriages"}
            </option>
            <option value="death">
              {lang === "sk" ? "Matrika – Úmrtia" : "Deaths"}
            </option>
            <option value="census">
              {lang === "sk" ? "Sčítacie hárky (1930/1940)" : "Census (1930/1940)"}
            </option>
          </select>

          <input
            type="text"
            value={villageFilter}
            onChange={(e) => setVillageFilter(e.target.value)}
            placeholder={lang === "sk" ? "Obec..." : "Village..."}
            aria-label={lang === "sk" ? "Filtrovať podľa obce" : "Filter by village"}
            className="w-32 px-3 py-2 rounded-lg border border-stone-300 text-sm"
          />

          <input
            type="text"
            value={streetFilter}
            onChange={(e) => setStreetFilter(e.target.value)}
            placeholder={lang === "sk" ? "Ulica..." : "Street..."}
            aria-label={lang === "sk" ? "Filtrovať podľa ulice" : "Filter by street"}
            className="w-32 px-3 py-2 rounded-lg border border-stone-300 text-sm"
          />

          <input
            type="text"
            value={houseNumberFilter}
            onChange={(e) => setHouseNumberFilter(e.target.value)}
            placeholder={lang === "sk" ? "Č. domu..." : "House #..."}
            aria-label={lang === "sk" ? "Filtrovať podľa čísla domu" : "Filter by house number"}
            className="w-24 px-3 py-2 rounded-lg border border-stone-300 text-sm"
          />

          {hasActiveSearchFilters && (
            <button
              type="button"
              onClick={() => {
                setSearchQuery("");
                setRecordTypeFilter("");
                setVillageFilter("");
                setStreetFilter("");
                setHouseNumberFilter("");
                setFilterRecordsByScan(false);
              }}
              className="px-3 py-2 rounded-lg bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300 text-xs font-bold inline-flex items-center gap-1 cursor-pointer shrink-0"
            >
              <X className="w-3.5 h-3.5" />
              <span>{lang === "sk" ? "Zrušiť filtre" : "Clear filters"}</span>
            </button>
          )}
        </div>
      </div>

      {/* Search Progress Bar */}
      <div
        data-testid="search-progress-card"
        className={`rounded-xl p-3 border space-y-2 transition-colors ${
          isSearching
            ? "bg-amber-50 border-amber-300"
            : "bg-stone-100/70 border-stone-200"
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2 font-bold text-stone-900 min-w-0">
            {isSearching ? (
              <RefreshCw className="w-4 h-4 text-amber-700 animate-spin shrink-0" />
            ) : (
              <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
            )}
            <span className="truncate">
              {isSearching
                ? lang === "sk"
                  ? `${searchProgress.stage_sk || "Prehľadávam matriky a sčítacie hárky…"}${
                      activeQuerySummary ? ` („${activeQuerySummary}“)` : ""
                    }`
                  : `${searchProgress.stage_en || "Searching registers and census sheets…"}${
                      activeQuerySummary ? ` ("${activeQuerySummary}")` : ""
                    }`
                : hasActiveSearchFilters
                ? lang === "sk"
                  ? `Vyhľadávanie dokončené — Nájdených ${displayTotalCount} osôb${
                      catalogSheets.length > 0
                        ? ` a ${catalogSheets.length} sčítacích hárkov`
                        : ""
                    }`
                  : `Search complete — Found ${displayTotalCount} people${
                      catalogSheets.length > 0
                        ? ` and ${catalogSheets.length} census sheets`
                        : ""
                    }`
                : lang === "sk"
                ? `Index vyhľadávania pripravený — Prehľadaných ${effectiveScannedTotal} záznamov (${displayTotalCount} osôb)`
                : `Search index ready — Scanned ${effectiveScannedTotal} records (${displayTotalCount} people)`}
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <span className="text-stone-600 font-medium">
              {lang === "sk"
                ? `Prehľadané: ${
                    isSearching ? searchProgress.processed : effectiveScannedTotal
                  } / ${effectiveScannedTotal}`
                : `Scanned: ${
                    isSearching ? searchProgress.processed : effectiveScannedTotal
                  } / ${effectiveScannedTotal}`}
              {!isSearching && searchProgress.elapsed_ms > 0
                ? ` (${searchProgress.elapsed_ms} ms)`
                : ""}
            </span>
            <span
              className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                isSearching
                  ? "bg-amber-600 text-white"
                  : "bg-emerald-100 text-emerald-900"
              }`}
            >
              {displaySearchPct}%
            </span>
          </div>
        </div>

        <div
          role="progressbar"
          aria-label={
            lang === "sk" ? "Priebeh vyhľadávania" : "Search progress"
          }
          aria-valuenow={displaySearchPct}
          aria-valuemin={0}
          aria-valuemax={100}
          data-testid="search-progress-bar"
          className={`w-full rounded-full h-2.5 overflow-hidden ${
            isSearching ? "bg-amber-200/80" : "bg-stone-200"
          }`}
        >
          <div
            className={`h-full transition-all duration-200 ${
              isSearching ? "bg-amber-600 animate-pulse" : "bg-emerald-600"
            }`}
            style={{ width: `${displaySearchPct}%` }}
          />
        </div>
      </div>

      {actionError && (
        <div className="bg-red-50 border border-red-400 rounded-xl p-3 flex items-center gap-2 text-red-900 font-medium text-xs">
          <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {catalogSheets.length > 0 && (
        <div className="bg-amber-50/70 rounded-xl p-3 border border-amber-200 space-y-2">
          <div className="text-xs font-bold text-amber-950 flex items-center gap-1.5">
            <FileSpreadsheet className="w-4 h-4 text-amber-700" />
            <span>
              {lang === "sk"
                ? `Nájdené sčítacie hárky v katalógu Slovakiana (${catalogSheets.length})`
                : `Matching Cataloged Census Sheets (${catalogSheets.length})`}
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {catalogSheets.slice(0, 12).map((cs) => (
              <Link
                key={cs.slovakiana_id || cs.id}
                to={`/scitacie-harky/${cs.slovakiana_id || cs.id}${
                  searchQuery.trim()
                    ? `?meno=${encodeURIComponent(searchQuery.trim())}`
                    : ""
                }`}
                className="px-3 py-1.5 min-h-[34px] rounded-lg bg-white hover:bg-amber-100 border border-amber-300 text-xs font-semibold text-stone-900 flex items-center gap-1.5"
              >
                <span className="px-1.5 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-xs">
                  {cs.slovakiana_id}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-amber-200 text-amber-950 font-bold">
                  {cs.census_year}
                </span>
                <span>
                  {cs.municipality_name}
                  {cs.street_name ? `, ${cs.street_name}` : ""}
                  {cs.house_number_conscription || cs.house_number_raw
                    ? ` č. ${cs.house_number_conscription || cs.house_number_raw}`
                    : ""}
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {loadingInitial ? (
        <div className="bg-stone-50 border border-stone-200 rounded-xl p-8 text-center space-y-2">
          <div className="w-7 h-7 rounded-full border-2 border-amber-600 border-t-transparent animate-spin mx-auto" />
          <p className="text-sm text-stone-700 font-medium">
            {lang === "sk"
              ? "Načítavam extrahované osoby z matrík a sčítaní…"
              : "Loading extracted person records…"}
          </p>
        </div>
      ) : records.length === 0 ? (
        <div className="bg-stone-50 border border-stone-200 rounded-xl p-8 text-center text-sm text-stone-600">
          {lang === "sk"
            ? "Pre zvolený filter sa nenašli žiadne extrahované osoby."
            : "No extracted person records match the current filter."}
        </div>
      ) : (
        <div className="space-y-3">
          {records.length > RECORDS_PER_PAGE && (
            <div className="flex flex-wrap items-center justify-between gap-2 bg-stone-50 border border-stone-200 rounded-lg px-3 py-2 text-xs sm:text-sm">
              <span className="text-stone-700">
                {lang === "sk" ? (
                  <>
                    Zobrazené záznamy{" "}
                    <strong>
                      {(safeRecordsPage - 1) * RECORDS_PER_PAGE + 1}–
                      {Math.min(records.length, safeRecordsPage * RECORDS_PER_PAGE)}
                    </strong>{" "}
                    z <strong>{displayTotalCount}</strong> osôb
                  </>
                ) : (
                  <>
                    Showing{" "}
                    <strong>
                      {(safeRecordsPage - 1) * RECORDS_PER_PAGE + 1}–
                      {Math.min(records.length, safeRecordsPage * RECORDS_PER_PAGE)}
                    </strong>{" "}
                    of <strong>{displayTotalCount}</strong> people
                  </>
                )}
              </span>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  disabled={safeRecordsPage <= 1}
                  onClick={() => setRecordsPage((p) => Math.max(1, p - 1))}
                  className="px-3 py-1.5 min-h-[32px] rounded-md bg-white hover:bg-stone-100 border border-stone-300 disabled:opacity-40 font-semibold text-xs cursor-pointer"
                >
                  {lang === "sk" ? "← Predošlá strana" : "← Previous"}
                </button>
                <span className="px-2 font-semibold text-stone-800 text-xs">
                  {safeRecordsPage} / {totalRecordPages}
                </span>
                <button
                  type="button"
                  disabled={safeRecordsPage >= totalRecordPages}
                  onClick={() =>
                    setRecordsPage((p) => Math.min(totalRecordPages, p + 1))
                  }
                  className="px-3 py-1.5 min-h-[32px] rounded-md bg-white hover:bg-stone-100 border border-stone-300 disabled:opacity-40 font-semibold text-xs cursor-pointer"
                >
                  {lang === "sk" ? "Ďalšia strana →" : "Next →"}
                </button>
              </div>
            </div>
          )}

          {pagedRecords.map((item) => {
            const r = item.fact;
            const isCensus = r.record_type === "census";
            const isEditing = editingFactId === r.id;
            const isConfirmingDelete = confirmDeleteFactId === r.id;
            const matchedRow = rowsById.get(r.row_id);
            const matchedScan = scansById.get(r.scan_id);
            const recordScanUrl = resolveScanMediaUrl(
              item.scan_url ||
                (selectedScanDetail?.scan.id === r.scan_id
                  ? selectedScanDetail.original_url
                  : matchedScan?.original_url),
              item.scan_filename
            );
            const recordScanHeight =
              selectedScanDetail?.scan.id === r.scan_id
                ? selectedScanDetail.height
                : matchedScan?.height || null;

            return (
              <div
                key={r.id}
                onClick={() => {
                  if (matchedRow) {
                    setSelectedRowId(matchedRow.id);
                  }
                }}
                className="bg-stone-50 rounded-xl p-3.5 border border-stone-200 space-y-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-base font-bold text-stone-900 flex flex-wrap items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                        isCensus
                          ? "bg-amber-100 text-amber-900"
                          : "bg-stone-200 text-stone-800"
                      }`}
                    >
                      {isCensus
                        ? `${lang === "sk" ? "Sčítanie" : "Census"} ${r.event_year || 1930}`
                        : `${r.record_type || "matrika"} ${r.event_year || ""}`}
                    </span>
                    <span>
                      {r.entry_number ? `#${r.entry_number} ` : ""}
                      {r.given_name_raw || "?"}{" "}
                      <span className="text-amber-900">
                        {r.surname_normalized || r.surname_raw || ""}
                      </span>
                    </span>
                    {isCensus && r.census_sheet_id && (
                      <Link
                        to={`/scitacie-harky/${r.census_sheet_id}?row=${r.id}${
                          searchQuery.trim()
                            ? `&meno=${encodeURIComponent(searchQuery.trim())}`
                            : ""
                        }`}
                        className="text-xs font-semibold text-amber-700 hover:underline"
                      >
                        {lang === "sk"
                          ? "→ Otvoriť sčítací hárok"
                          : "→ Open census sheet"}
                      </Link>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-1.5">
                    {(r.birth_or_event_date || r.baptism_or_burial_date) && (
                      <span className="px-2 py-0.5 rounded-full bg-stone-200 font-medium text-xs">
                        {r.birth_or_event_date || r.baptism_or_burial_date}
                      </span>
                    )}
                    {(r.village_raw || item.street_name) && (
                      <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-950 font-medium text-xs">
                        {r.village_raw}
                        {item.street_name &&
                        (!r.village_raw ||
                          r.village_raw.toLowerCase() !== item.street_name.toLowerCase())
                          ? `${r.village_raw ? ", " : ""}${item.street_name}`
                          : ""}
                        {r.house_number ? ` č. ${r.house_number}` : ""}
                        {r.fs_place_standardized ? ` → ${r.fs_place_standardized}` : ""}
                      </span>
                    )}
                    {item.observationes_events &&
                      item.observationes_events.map((ev, idx) => {
                        const evType = ev.event_type || ev.type || "death";
                        const evDate = ev.date || (ev.year ? String(ev.year) : ev.raw_note);
                        const isMarriage = evType === "marriage";
                        return (
                          <span
                            key={idx}
                            className={`px-2 py-0.5 rounded-full font-semibold text-xs ${
                              isMarriage
                                ? "bg-purple-100 text-purple-900"
                                : "bg-red-100 text-red-900"
                            }`}
                          >
                            {isMarriage
                              ? `${lang === "sk" ? "Sobáš" : "Marriage"}: ${evDate}`
                              : `† ${evDate}`}
                          </span>
                        );
                      })}

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isEditing) setEditingFactId(null);
                        else startEditFact(r);
                      }}
                      className={`px-3 py-1.5 min-h-[32px] rounded-lg text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer ${
                        isEditing
                          ? "bg-amber-600 text-white"
                          : "bg-stone-200 hover:bg-stone-300 text-stone-800"
                      }`}
                    >
                      <Edit3 className="w-3.5 h-3.5" />
                      <span>
                        {isEditing
                          ? lang === "sk"
                            ? "Zavrieť"
                            : "Close"
                          : lang === "sk"
                          ? "Upraviť"
                          : "Edit"}
                      </span>
                    </button>

                    {isConfirmingDelete ? (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        className="inline-flex items-center gap-1.5 bg-red-50 border border-red-300 rounded-lg px-2.5 py-1 min-h-[32px]"
                      >
                        <button
                          type="button"
                          disabled={deletingFactId === r.id}
                          onClick={() => handleDeleteFact(r.id)}
                          className="text-xs font-bold text-red-700 hover:underline cursor-pointer"
                        >
                          {deletingFactId === r.id
                            ? "…"
                            : lang === "sk"
                            ? "Potvrdiť vymazanie"
                            : "Confirm delete"}
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirmDeleteFactId(null)}
                          className="text-xs text-stone-600 hover:text-stone-900 cursor-pointer ml-1 px-1"
                        >
                          ✕
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmDeleteFactId(r.id);
                        }}
                        title={lang === "sk" ? "Vymazať záznam" : "Delete record"}
                        className="p-2 min-h-[32px] min-w-[32px] inline-flex items-center justify-center rounded-lg bg-stone-200 hover:bg-red-100 text-stone-700 hover:text-red-700 cursor-pointer"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {isCensus ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-x-4 gap-y-0.5 text-sm text-stone-800">
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Pomer k hlave:" : "Relation to head:"}
                      </span>{" "}
                      {r.relation_to_head || "—"}
                    </div>
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Povolanie:" : "Occupation:"}
                      </span>{" "}
                      {r.social_status_or_occupation || "—"}
                    </div>
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Rodisko:" : "Birthplace:"}
                      </span>{" "}
                      {r.birth_place_raw || "—"}
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-x-4 gap-y-0.5 text-sm text-stone-800">
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Otec:" : "Father:"}
                      </span>{" "}
                      {r.father_name || "—"}
                    </div>
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Matka:" : "Mother:"}
                      </span>{" "}
                      {r.mother_name || "—"}
                      {r.mother_maiden_surname ? ` (rod. ${r.mother_maiden_surname})` : ""}
                    </div>
                    <div>
                      <span className="text-stone-500">
                        {lang === "sk" ? "Krstní / Svedkovia:" : "Godparents / Witnesses:"}
                      </span>{" "}
                      {r.godparents_or_witnesses || "—"}
                    </div>
                  </div>
                )}

                {isEditing && (
                  <div
                    onClick={(e) => e.stopPropagation()}
                    className="bg-amber-50/80 border border-amber-300 rounded-xl p-3.5 space-y-3 text-xs mt-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-200 pb-2">
                      <span className="font-bold text-amber-950 uppercase tracking-wider">
                        {lang === "sk"
                          ? `Úprava záznamu #${r.id}`
                          : `Edit Record #${r.id}`}
                      </span>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          disabled={savingFactId === r.id}
                          onClick={() => handleSaveFact(r.id)}
                          className="px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                        >
                          <Save className="w-3.5 h-3.5" />
                          <span>
                            {savingFactId === r.id
                              ? lang === "sk"
                                ? "Ukladám…"
                                : "Saving…"
                              : lang === "sk"
                              ? "Uložiť zmeny"
                              : "Save changes"}
                          </span>
                        </button>
                        <button
                          type="button"
                          onClick={() => setEditingFactId(null)}
                          className="px-2.5 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 font-semibold inline-flex items-center gap-1 cursor-pointer"
                        >
                          <X className="w-3.5 h-3.5" />
                          <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Poradové č." : "Entry #"}
                        </label>
                        <input
                          type="text"
                          value={editForm.entry_number}
                          onChange={(e) =>
                            setEditForm({ ...editForm, entry_number: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Krstné meno" : "Given name"}
                        </label>
                        <input
                          type="text"
                          value={editForm.given_name_raw}
                          onChange={(e) =>
                            setEditForm({ ...editForm, given_name_raw: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Priezvisko (prepísané)" : "Surname (raw)"}
                        </label>
                        <input
                          type="text"
                          value={editForm.surname_raw}
                          onChange={(e) =>
                            setEditForm({ ...editForm, surname_raw: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk"
                            ? "Priezvisko (normalizované)"
                            : "Surname (normalized)"}
                        </label>
                        <input
                          type="text"
                          value={editForm.surname_normalized}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              surname_normalized: e.target.value,
                            })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk"
                            ? "Dátum narodenia / udalosti"
                            : "Birth / event date"}
                        </label>
                        <input
                          type="text"
                          value={editForm.birth_or_event_date}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              birth_or_event_date: e.target.value,
                            })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk"
                            ? "Dátum krstu / pohrebu"
                            : "Baptism / burial date"}
                        </label>
                        <input
                          type="text"
                          value={editForm.baptism_or_burial_date}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              baptism_or_burial_date: e.target.value,
                            })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Obec" : "Village"}
                        </label>
                        <input
                          type="text"
                          value={editForm.village_raw}
                          onChange={(e) =>
                            setEditForm({ ...editForm, village_raw: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Číslo domu" : "House #"}
                        </label>
                        <input
                          type="text"
                          value={editForm.house_number}
                          onChange={(e) =>
                            setEditForm({ ...editForm, house_number: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Meno otca" : "Father's name"}
                        </label>
                        <input
                          type="text"
                          value={editForm.father_name}
                          onChange={(e) =>
                            setEditForm({ ...editForm, father_name: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk" ? "Meno matky" : "Mother's name"}
                        </label>
                        <input
                          type="text"
                          value={editForm.mother_name}
                          onChange={(e) =>
                            setEditForm({ ...editForm, mother_name: e.target.value })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk"
                            ? "Rodné priezvisko matky"
                            : "Mother's maiden surname"}
                        </label>
                        <input
                          type="text"
                          value={editForm.mother_maiden_surname}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              mother_maiden_surname: e.target.value,
                            })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div>
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {isCensus
                            ? lang === "sk"
                              ? "Povolanie"
                              : "Occupation"
                            : lang === "sk"
                            ? "Krstní rodičia / svedkovia"
                            : "Godparents / witnesses"}
                        </label>
                        <input
                          type="text"
                          value={
                            isCensus
                              ? editForm.social_status_or_occupation
                              : editForm.godparents_or_witnesses
                          }
                          onChange={(e) =>
                            setEditForm(
                              isCensus
                                ? {
                                    ...editForm,
                                    social_status_or_occupation: e.target.value,
                                  }
                                : {
                                    ...editForm,
                                    godparents_or_witnesses: e.target.value,
                                  }
                            )
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                      <div className="sm:col-span-2 lg:col-span-4">
                        <label className="block font-semibold text-stone-700 mb-0.5">
                          {lang === "sk"
                            ? "Poznámky (Observationes)"
                            : "Remarks (Observationes)"}
                        </label>
                        <input
                          type="text"
                          value={editForm.observationes_raw}
                          onChange={(e) =>
                            setEditForm({
                              ...editForm,
                              observationes_raw: e.target.value,
                            })
                          }
                          className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {item.stitched_crop_url && (
                  <HoverZoomRowCrop
                    cropUrl={item.stitched_crop_url}
                    headerCropUrl={
                      selectedScanDetail?.scan.id === r.scan_id
                        ? selectedScanDetail.header_crop_path
                        : matchedScan?.header_crop_path ||
                          matchedScan?.header_crop_url ||
                          null
                    }
                    scanUrl={featureFlags.hover_row_zoom ? recordScanUrl : null}
                    yStart={featureFlags.hover_row_zoom ? matchedRow?.y_start : null}
                    yEnd={featureFlags.hover_row_zoom ? matchedRow?.y_end : null}
                    scanHeight={recordScanHeight}
                    rowLabel={
                      lang === "sk"
                        ? `Výrez vybraného riadku: ${r.given_name_raw || "?"} ${
                            r.surname_normalized || r.surname_raw || ""
                          }`.trim()
                        : `Selected row crop: ${r.given_name_raw || "?"} ${
                            r.surname_normalized || r.surname_raw || ""
                          }`.trim()
                    }
                    lang={lang}
                  />
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-5">
      {mode === "search" ? (
        <>{unifiedSearchSection}</>
      ) : (
        <>
          {scanInspectorSection}
          {unifiedSearchSection}
        </>
      )}
    </div>
  );
}
