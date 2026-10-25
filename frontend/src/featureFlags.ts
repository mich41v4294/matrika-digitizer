import { useCallback, useEffect, useState } from "react";

export interface FeatureFlags {
  // Navigation tab flags (all core tabs enabled by default)
  tab_census: boolean;
  tab_matrika: boolean;
  tab_questions: boolean;
  tab_records: boolean;
  tab_search: boolean;
  tab_familysearch: boolean;

  // Functional pipeline / UI flags (default true, except admin_debug_overlay which defaults to false)
  census_multi_byt: boolean;
  census_surname_consensus: boolean;
  census_overview_sheets: boolean;
  skip_row_ocr: boolean;
  hover_row_zoom: boolean;
  manual_scan_zoom: boolean;
  admin_debug_overlay: boolean;
}

export type FeatureFlagKey = keyof FeatureFlags;

export const FEATURE_FLAGS_STORAGE_KEY = "matrika_feature_flags_v1";
export const FEATURE_FLAGS_EVENT = "feature-flags-changed";

export const DEFAULT_FEATURE_FLAGS: FeatureFlags = {
  tab_census: true,
  tab_matrika: true,
  tab_questions: true,
  tab_records: true,
  tab_search: true,
  tab_familysearch: true,
  census_multi_byt: true,
  census_surname_consensus: true,
  census_overview_sheets: true,
  skip_row_ocr: true,
  hover_row_zoom: true,
  manual_scan_zoom: true,
  admin_debug_overlay: false,
};

export interface FeatureFlagDescriptor {
  key: FeatureFlagKey;
  shortLabelSk: string;
  shortLabelEn: string;
  labelSk: string;
  labelEn: string;
  descriptionSk: string;
  descriptionEn: string;
  category: "tab" | "functional";
}

export const NAVIGATION_FLAG_DESCRIPTORS: FeatureFlagDescriptor[] = [
  {
    key: "tab_census",
    shortLabelSk: "Sčítacie hárky",
    shortLabelEn: "Census Sheets",
    labelSk: "Sčítacie hárky",
    labelEn: "Census Sheets",
    descriptionSk: "Hlavná záložka pre sčítanie ľudu 1930/1940 (Slovakiana)",
    descriptionEn: "Main workspace for 1930/1940 Slovakiana census sheets",
    category: "tab",
  },
  {
    key: "tab_matrika",
    shortLabelSk: "Matrika",
    shortLabelEn: "Matrika",
    labelSk: "Domov & Nahrávanie matrík",
    labelEn: "Home & Matrika Upload",
    descriptionSk: "Domovská obrazovka a nahrávanie skenov matričných kníh",
    descriptionEn: "Home dashboard and church register scan upload",
    category: "tab",
  },
  {
    key: "tab_questions",
    shortLabelSk: "Otázky",
    shortLabelEn: "Questions",
    labelSk: "Otázky",
    labelEn: "Questions",
    descriptionSk: "Centrum otázok pre kontrolu nejasného rukopisu",
    descriptionEn: "Question Center for reviewing uncertain handwriting",
    category: "tab",
  },
  {
    key: "tab_records",
    shortLabelSk: "Knihy a strany",
    shortLabelEn: "Books & Scans",
    labelSk: "Knihy a strany matrík",
    labelEn: "Books & Register Scans",
    descriptionSk: "Prehliadač naskenovaných strán matrík, výrezov riadkov a extrahovaných osôb",
    descriptionEn: "Inspector for scanned register pages, row crops, and extracted records",
    category: "tab",
  },
  {
    key: "tab_search",
    shortLabelSk: "Vyhľadávanie",
    shortLabelEn: "Search",
    labelSk: "Vyhľadávanie",
    labelEn: "Unified Search",
    descriptionSk: "Vyhľadávanie osôb, priezvisk a obcí naprieč matrikami a sčítaniami",
    descriptionEn: "Cross-record search for people, surnames, and villages across registers and census",
    category: "tab",
  },
  {
    key: "tab_familysearch",
    shortLabelSk: "FamilySearch",
    shortLabelEn: "FamilySearch",
    labelSk: "FamilySearch",
    labelEn: "FamilySearch",
    descriptionSk: "Porovnávanie a dopĺňanie záznamov do FamilySearch",
    descriptionEn: "Compare and copy extracted genealogical facts to FamilySearch",
    category: "tab",
  },
];

export const FUNCTIONAL_FLAG_DESCRIPTORS: FeatureFlagDescriptor[] = [
  {
    key: "census_multi_byt",
    shortLabelSk: "Viac bytov na hárku",
    shortLabelEn: "Multiple dwelling units",
    labelSk: "Rozdelenie hárku na viac bytov (Byt č. I, II...)",
    labelEn: "Split sheet into multiple dwelling units (Byt I, II...)",
    descriptionSk:
      "Automatická detekcia a zoskupenie viacerých bytov/domácností na jednom sčítacom hárku",
    descriptionEn:
      "Automatic detection and grouping of multiple dwelling units/households on a single census sheet",
    category: "functional",
  },
  {
    key: "census_surname_consensus",
    shortLabelSk: "Konsenzus priezviska v byte",
    shortLabelEn: "Dwelling surname consensus",
    labelSk: "Konsenzus priezviska v rámci bytu (napr. Kováč vs. Kovač)",
    labelEn: "Dwelling surname consensus (e.g. Kováč vs. Kovač)",
    descriptionSk:
      "Zjednotenie skomolených OCR priezvisk podľa prevládajúceho priezviska v byte",
    descriptionEn:
      "Unify noisy OCR surname variants based on the dominant surname in the dwelling unit",
    category: "functional",
  },
  {
    key: "census_overview_sheets",
    shortLabelSk: "Prehľady obvodov",
    shortLabelEn: "District overview sheets",
    labelSk: "Prehľady sčítacích obvodov a adresár domov",
    labelEn: "Census district overview sheets & house directory",
    descriptionSk:
      "Podpora pre hárky prehľadov sčítacích obvodov, zoznam domov a prepínanie typu hárku",
    descriptionEn:
      "Support for enumeration district overview sheets, house directories, and sheet-type switching",
    category: "functional",
  },
  {
    key: "skip_row_ocr",
    shortLabelSk: "Preskočiť textové OCR riadkov",
    shortLabelEn: "Skip row text OCR",
    labelSk:
      "Preskočiť textové OCR riadkov a posielať výrezy buniek priamo do Extract modelu (bez OCR nápovied)",
    labelEn:
      "Skip row text OCR and send column crops directly to Extract model (without OCR hints)",
    descriptionSk:
      "Použije GLM-OCR iba na hlavičky hárkov a tabuľky prehľadov obvodov; pri riadkoch osôb preskočí textové OCR a pošle zložený výrez všetkých 5 skupín stĺpcov (stĺpce 1–24) priamo do vizuálneho Extract modelu bez OCR nápovied",
    descriptionEn:
      "Uses GLM-OCR only for sheet headers and overview tables; skips row text OCR on person rows and sends the 5-band column crop composite (Cols 1–24) straight to the Extract VLM without OCR hints",
    category: "functional",
  },
  {
    key: "hover_row_zoom",
    shortLabelSk: "Hover zoom riadku",
    shortLabelEn: "Hover row zoom",
    labelSk: "Priblíženie kontextu riadku pri nabehnutí myšou",
    labelEn: "Contextual row zoom on hover",
    descriptionSk:
      "Zobrazenie okolitých riadkov (±2 až ±5.5 riadku) pri nabehnutí myšou na výrez riadku",
    descriptionEn:
      "Display neighboring rows (±2 to ±5.5 rows) and column headers when hovering over a row crop",
    category: "functional",
  },
  {
    key: "manual_scan_zoom",
    shortLabelSk: "Manuálny zoom skenu",
    shortLabelEn: "Manual scan zoom",
    labelSk: "Manuálne priblíženie a posúvanie skenov",
    labelEn: "Manual scan zoom and pan",
    descriptionSk:
      "Interaktívne približovanie (50%–400%) a posúvanie celého skenu myšou",
    descriptionEn:
      "Interactive zooming (50%–400%) and mouse panning for full document scans",
    category: "functional",
  },
  {
    key: "admin_debug_overlay",
    shortLabelSk: "Admin Debug Overlay",
    shortLabelEn: "Admin Debug Overlay",
    labelSk: "Zobraziť technické nástroje (OCR ✓/✗, CV Overlay, typ hárku) v aplikácii",
    labelEn: "Show inline debug controls (OCR ✓/✗, CV Overlay, sheet type override) in app",
    descriptionSk:
      "Zobrazí na sčítacích hárkoch a matrikách technické ovládacie prvky pre hodnotenie OCR, CV detekciu riadkov a manuálnu zmenu typu hárku",
    descriptionEn:
      "Displays inline OCR rating buttons, OpenCV row overlays, and manual sheet-type overrides on user pages",
    category: "functional",
  },
];

const FEATURE_FLAGS_PENDING_UPDATES_KEY =
  "matrika_feature_flags_pending_updates_v1";

let memoryFlags: FeatureFlags | null = null;
let localMutationSeq = 0;
let pendingSyncCount = 0;
let inFlightFetch: Promise<FeatureFlags | null> | null = null;

function readPendingUpdates(): Partial<FeatureFlags> | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(FEATURE_FLAGS_PENDING_UPDATES_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as Partial<FeatureFlags>;
  } catch {
    return null;
  }
}

function writePendingUpdates(updates: Partial<FeatureFlags> | null): void {
  if (typeof window === "undefined") return;
  try {
    if (!updates || Object.keys(updates).length === 0) {
      window.sessionStorage.removeItem(FEATURE_FLAGS_PENDING_UPDATES_KEY);
    } else {
      window.sessionStorage.setItem(
        FEATURE_FLAGS_PENDING_UPDATES_KEY,
        JSON.stringify(updates)
      );
    }
  } catch {
    // Ignore storage errors
  }
}

function sanitizeFeatureFlags(raw: unknown): FeatureFlags {
  const result: FeatureFlags = { ...DEFAULT_FEATURE_FLAGS };
  if (!raw || typeof raw !== "object") {
    return result;
  }
  const obj = raw as Record<string, unknown>;
  (Object.keys(DEFAULT_FEATURE_FLAGS) as FeatureFlagKey[]).forEach((key) => {
    if (typeof obj[key] === "boolean") {
      result[key] = obj[key] as boolean;
    }
  });
  return result;
}

function areFlagsEqual(a: FeatureFlags, b: FeatureFlags): boolean {
  return (Object.keys(DEFAULT_FEATURE_FLAGS) as FeatureFlagKey[]).every(
    (key) => a[key] === b[key]
  );
}

export function getFeatureFlags(): FeatureFlags {
  if (typeof window === "undefined") {
    return memoryFlags ? { ...memoryFlags } : { ...DEFAULT_FEATURE_FLAGS };
  }
  try {
    const raw = window.localStorage.getItem(FEATURE_FLAGS_STORAGE_KEY);
    if (!raw) {
      return memoryFlags ? { ...memoryFlags } : { ...DEFAULT_FEATURE_FLAGS };
    }
    const parsed = sanitizeFeatureFlags(JSON.parse(raw));
    memoryFlags = parsed;
    return parsed;
  } catch {
    return memoryFlags ? { ...memoryFlags } : { ...DEFAULT_FEATURE_FLAGS };
  }
}

function persistAndNotifyFlags(next: FeatureFlags): void {
  const prev = memoryFlags ?? getFeatureFlags();
  let hasStored = false;
  if (typeof window !== "undefined") {
    try {
      hasStored = Boolean(window.localStorage.getItem(FEATURE_FLAGS_STORAGE_KEY));
    } catch {
      hasStored = false;
    }
  }
  const changed = !areFlagsEqual(prev, next) || !hasStored;
  memoryFlags = next;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(FEATURE_FLAGS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Ignore storage quota errors
    }
    if (changed) {
      window.dispatchEvent(
        new CustomEvent<FeatureFlags>(FEATURE_FLAGS_EVENT, { detail: next })
      );
    }
  }
}

export function applyBackendFeatureFlags(rawFlags: unknown): FeatureFlags | null {
  if (!rawFlags || typeof rawFlags !== "object") return null;
  const next = sanitizeFeatureFlags(rawFlags);
  persistAndNotifyFlags(next);
  return next;
}

export function syncBackendFeatureFlagsIfIdle(rawFlags: unknown): FeatureFlags | null {
  if (pendingSyncCount > 0 || Boolean(readPendingUpdates())) {
    return getFeatureFlags();
  }
  return applyBackendFeatureFlags(rawFlags);
}

async function syncFlagsToBackend(
  updatesToFlush: Partial<FeatureFlags>,
  seq: number
): Promise<FeatureFlags | null> {
  pendingSyncCount += 1;
  try {
    const res = await fetch("/api/debug/feature-flags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flags: updatesToFlush }),
      keepalive: true,
    });
    if (res.ok) {
      const data = await res.json();
      const rawFlags =
        data?.flags && typeof data.flags === "object" ? data.flags : data;
      if (seq === localMutationSeq) {
        writePendingUpdates(null);
        if (rawFlags && typeof rawFlags === "object") {
          return applyBackendFeatureFlags(rawFlags);
        }
      }
    }
    return null;
  } catch {
    // Optional backend endpoint — ignore network/404 errors silently
    return null;
  } finally {
    pendingSyncCount = Math.max(0, pendingSyncCount - 1);
  }
}

export function setFeatureFlags(
  updates: Partial<FeatureFlags>,
  syncBackend = true
): FeatureFlags {
  const current = getFeatureFlags();
  const next = sanitizeFeatureFlags({ ...current, ...updates });
  persistAndNotifyFlags(next);
  if (typeof window !== "undefined" && syncBackend) {
    const existingPending = readPendingUpdates() || {};
    const mergedPending: Partial<FeatureFlags> = {
      ...existingPending,
      ...updates,
    };
    writePendingUpdates(mergedPending);
    localMutationSeq += 1;
    void syncFlagsToBackend(mergedPending, localMutationSeq);
  }
  return next;
}

export function toggleFeatureFlag(key: FeatureFlagKey): FeatureFlags {
  const current = getFeatureFlags();
  return setFeatureFlags({ [key]: !current[key] });
}

export function resetToCensusOnlyPreset(): FeatureFlags {
  return setFeatureFlags({
    tab_census: true,
    tab_matrika: false,
    tab_questions: false,
    tab_records: false,
    tab_search: false,
    tab_familysearch: false,
  });
}

export function enableAllTabsPreset(): FeatureFlags {
  return setFeatureFlags({
    tab_census: true,
    tab_matrika: true,
    tab_questions: true,
    tab_records: true,
    tab_search: true,
    tab_familysearch: true,
  });
}

export async function fetchBackendFeatureFlags(): Promise<FeatureFlags | null> {
  if (inFlightFetch) {
    return inFlightFetch;
  }
  const seqAtStart = localMutationSeq;
  inFlightFetch = (async () => {
    try {
      const pending = readPendingUpdates();
      if (pending && Object.keys(pending).length > 0 && pendingSyncCount === 0) {
        const flushed = await syncFlagsToBackend(pending, localMutationSeq);
        if (flushed) return flushed;
      }
      const res = await fetch("/api/debug/feature-flags", {
        cache: "no-store",
      });
      if (!res.ok) return null;
      const data = await res.json();
      const rawFlags =
        data?.flags && typeof data.flags === "object" ? data.flags : data;
      if (!rawFlags || typeof rawFlags !== "object") return null;
      if (
        localMutationSeq !== seqAtStart ||
        pendingSyncCount > 0 ||
        Boolean(readPendingUpdates())
      ) {
        return getFeatureFlags();
      }
      return applyBackendFeatureFlags(rawFlags);
    } catch {
      return null;
    } finally {
      inFlightFetch = null;
    }
  })();
  return inFlightFetch;
}

if (typeof window !== "undefined") {
  void fetchBackendFeatureFlags();
}

export function useFeatureFlags(): {
  flags: FeatureFlags;
  updateFlags: (updates: Partial<FeatureFlags>) => FeatureFlags;
  toggleFlag: (key: FeatureFlagKey) => FeatureFlags;
  resetCensusOnly: () => FeatureFlags;
  enableAllTabs: () => FeatureFlags;
} {
  const [flags, setFlagsState] = useState<FeatureFlags>(() => getFeatureFlags());

  useEffect(() => {
    const handleCustomEvent = (e: Event) => {
      const ce = e as CustomEvent<FeatureFlags>;
      if (ce.detail) {
        setFlagsState(sanitizeFeatureFlags(ce.detail));
      } else {
        setFlagsState(getFeatureFlags());
      }
    };

    const handleStorageEvent = (e: StorageEvent) => {
      if (!e.key || e.key === FEATURE_FLAGS_STORAGE_KEY) {
        setFlagsState(getFeatureFlags());
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void fetchBackendFeatureFlags();
      }
    };

    window.addEventListener(FEATURE_FLAGS_EVENT, handleCustomEvent);
    window.addEventListener("storage", handleStorageEvent);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    void fetchBackendFeatureFlags();
    return () => {
      window.removeEventListener(FEATURE_FLAGS_EVENT, handleCustomEvent);
      window.removeEventListener("storage", handleStorageEvent);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  const updateFlags = useCallback((updates: Partial<FeatureFlags>) => {
    const next = setFeatureFlags(updates);
    setFlagsState(next);
    return next;
  }, []);

  const toggleFlag = useCallback((key: FeatureFlagKey) => {
    const next = toggleFeatureFlag(key);
    setFlagsState(next);
    return next;
  }, []);

  const resetCensusOnly = useCallback(() => {
    const next = resetToCensusOnlyPreset();
    setFlagsState(next);
    return next;
  }, []);

  const enableAllTabs = useCallback(() => {
    const next = enableAllTabsPreset();
    setFlagsState(next);
    return next;
  }, []);

  return {
    flags,
    updateFlags,
    toggleFlag,
    resetCensusOnly,
    enableAllTabs,
  };
}
