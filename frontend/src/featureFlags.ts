import { useCallback, useEffect, useState } from "react";

export interface FeatureFlags {
  // Navigation tab flags (all core tabs enabled by default)
  tab_census: boolean;
  tab_matrika: boolean;
  tab_questions: boolean;
  tab_records: boolean;
  tab_search: boolean;
  tab_familysearch: boolean;

  // Functional pipeline / UI flags (default true)
  census_multi_byt: boolean;
  census_surname_consensus: boolean;
  census_overview_sheets: boolean;
  hover_row_zoom: boolean;
  manual_scan_zoom: boolean;
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
  hover_row_zoom: true,
  manual_scan_zoom: true,
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
];

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

export function getFeatureFlags(): FeatureFlags {
  if (typeof window === "undefined") {
    return { ...DEFAULT_FEATURE_FLAGS };
  }
  try {
    const raw = window.localStorage.getItem(FEATURE_FLAGS_STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_FEATURE_FLAGS };
    }
    return sanitizeFeatureFlags(JSON.parse(raw));
  } catch {
    return { ...DEFAULT_FEATURE_FLAGS };
  }
}

async function syncFlagsToBackend(flags: FeatureFlags): Promise<void> {
  try {
    await fetch("/api/debug/feature-flags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flags }),
    });
  } catch {
    // Optional backend endpoint — ignore network/404 errors silently
  }
}

export function setFeatureFlags(
  updates: Partial<FeatureFlags>,
  syncBackend = true
): FeatureFlags {
  const current = getFeatureFlags();
  const next = sanitizeFeatureFlags({ ...current, ...updates });
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(FEATURE_FLAGS_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Ignore storage quota errors
    }
    window.dispatchEvent(
      new CustomEvent<FeatureFlags>(FEATURE_FLAGS_EVENT, { detail: next })
    );
    if (syncBackend) {
      void syncFlagsToBackend(next);
    }
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
  try {
    const res = await fetch("/api/debug/feature-flags");
    if (!res.ok) return null;
    const data = await res.json();
    const rawFlags = data?.flags && typeof data.flags === "object" ? data.flags : data;
    if (!rawFlags || typeof rawFlags !== "object") return null;
    // Only apply backend defaults if user has not yet persisted local flags
    const hasLocal =
      typeof window !== "undefined" &&
      Boolean(window.localStorage.getItem(FEATURE_FLAGS_STORAGE_KEY));
    if (!hasLocal) {
      return setFeatureFlags(sanitizeFeatureFlags(rawFlags), false);
    }
    return getFeatureFlags();
  } catch {
    return null;
  }
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

    window.addEventListener(FEATURE_FLAGS_EVENT, handleCustomEvent);
    window.addEventListener("storage", handleStorageEvent);
    return () => {
      window.removeEventListener(FEATURE_FLAGS_EVENT, handleCustomEvent);
      window.removeEventListener("storage", handleStorageEvent);
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
