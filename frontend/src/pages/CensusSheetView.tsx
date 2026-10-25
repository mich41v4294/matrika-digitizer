import React, { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  Building2,
  CheckCircle2,
  Download,
  Edit3,
  ExternalLink,
  Eye,
  FileSpreadsheet,
  Home,
  Layers,
  MapPin,
  Plus,
  RefreshCw,
  Save,
  Scale,
  Sparkles,
  Trash2,
  UserCheck,
  Users,
  X,
} from "lucide-react";
import ZoomableScanViewer, {
  HoverZoomRowCrop,
  type ActiveFieldFocus,
} from "../components/ZoomableScanViewer";
import OcrFeedbackControls, { type OcrFeedbackItem } from "../components/OcrFeedbackControls";
import { useFeatureFlags } from "../featureFlags";
import type { CensusSheetDetail, LinkedHouseSheet } from "../api";
import { emitClientOcrNotification, type PipelineState } from "../App";
import { matchesPersonNameQuery, stripFemSuffix } from "../utils/fuzzyName";

export interface RelationPresetGroup {
  groupLabel: string;
  options: Array<{ value: string; label: string }>;
}

export const RELATION_TO_HEAD_GROUPS: RelationPresetGroup[] = [
  {
    groupLabel: "Hlava domácnosti a partneri",
    options: [
      { value: "hlava domácnosti", label: "hlava domácnosti (Hlava domácnosti / Majiteľ bytu)" },
      { value: "manželka", label: "manželka" },
      { value: "manžel", label: "manžel" },
      { value: "družka", label: "družka" },
      { value: "druh", label: "druh" },
    ],
  },
  {
    groupLabel: "Deti a chovanci",
    options: [
      { value: "syn", label: "syn" },
      { value: "dcéra", label: "dcéra" },
      { value: "nevlastný syn", label: "nevlastný syn" },
      { value: "nevlastná dcéra", label: "nevlastná dcéra" },
      { value: "adoptívny syn", label: "adoptívny syn" },
      { value: "adoptívna dcéra", label: "adoptívna dcéra" },
      { value: "chovanec", label: "chovanec" },
      { value: "chovanka", label: "chovanka" },
    ],
  },
  {
    groupLabel: "Rodičia, svokrovci, zať a nevesta",
    options: [
      { value: "matka", label: "matka" },
      { value: "otec", label: "otec" },
      { value: "svokra", label: "svokra" },
      { value: "svokor", label: "svokor" },
      { value: "svatka", label: "svatka" },
      { value: "svat", label: "svat" },
      { value: "zať", label: "zať" },
      { value: "nevesta", label: "nevesta" },
    ],
  },
  {
    groupLabel: "Vnuci a starí rodičia",
    options: [
      { value: "vnuk", label: "vnuk" },
      { value: "vnučka", label: "vnučka" },
      { value: "pravnuk", label: "pravnuk" },
      { value: "pravnučka", label: "pravnučka" },
      { value: "stará matka", label: "stará matka" },
      { value: "starý otec", label: "starý otec" },
    ],
  },
  {
    groupLabel: "Súrodenci a širšia rodina",
    options: [
      { value: "brat", label: "brat" },
      { value: "sestra", label: "sestra" },
      { value: "nevlastný brat", label: "nevlastný brat" },
      { value: "nevlastná sestra", label: "nevlastná sestra" },
      { value: "švagor", label: "švagor" },
      { value: "švagriná", label: "švagriná" },
      { value: "strýko", label: "strýko" },
      { value: "stryná", label: "stryná" },
      { value: "ujo", label: "ujo" },
      { value: "teta", label: "teta" },
      { value: "synovec", label: "synovec" },
      { value: "neter", label: "neter" },
      { value: "bratranec", label: "bratranec" },
      { value: "sesternica", label: "sesternica" },
      { value: "príbuzný", label: "príbuzný" },
      { value: "príbuzná", label: "príbuzná" },
    ],
  },
  {
    groupLabel: "Služobníctvo, čeľaď a pomocníci",
    options: [
      { value: "slúžka", label: "slúžka" },
      { value: "sluha", label: "sluha" },
      { value: "čeľadník", label: "čeľadník" },
      { value: "paholok", label: "paholok" },
      { value: "gazdiná", label: "gazdiná" },
      { value: "pestúnka", label: "pestúnka" },
      { value: "učeník", label: "učeník" },
      { value: "tovariš", label: "tovariš" },
      { value: "pomocník", label: "pomocník" },
    ],
  },
  {
    groupLabel: "Podnájomníci a hostia",
    options: [
      { value: "podnájomník", label: "podnájomník" },
      { value: "podnájomníčka", label: "podnájomníčka" },
      { value: "bývajúci", label: "bývajúci" },
      { value: "hosť", label: "hosť" },
    ],
  },
];

export const ALL_RELATION_TO_HEAD_OPTIONS: Array<{ value: string; label: string }> =
  RELATION_TO_HEAD_GROUPS.flatMap((g) => g.options);

export const QUICK_RELATION_PILLS: string[] = [
  "hlava domácnosti",
  "manželka",
  "syn",
  "dcéra",
  "matka",
  "otec",
  "svokra",
  "svokor",
  "zať",
  "nevesta",
  "vnuk",
  "vnučka",
  "brat",
  "sestra",
  "švagor",
  "švagriná",
  "slúžka",
  "sluha",
  "podnájomník",
  "hosť",
];

const ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

function toRomanBytLabel(indexOrRaw: number | string | null | undefined): string {
  if (typeof indexOrRaw === "string" && indexOrRaw.trim()) {
    const cleaned = indexOrRaw
      .trim()
      .replace(/^byt\s*(č\.?|c\.?)?\s*/i, "")
      .replace(/\.$/, "")
      .trim();
    const asNum = Number(cleaned);
    if (!Number.isNaN(asNum) && asNum >= 1 && asNum <= ROMAN_NUMERALS.length) {
      return ROMAN_NUMERALS[asNum - 1];
    }
    if (/^[IVX]+$/i.test(cleaned)) {
      return cleaned.toUpperCase();
    }
    return cleaned;
  }
  if (typeof indexOrRaw === "number" && indexOrRaw >= 1) {
    return ROMAN_NUMERALS[indexOrRaw - 1] || String(indexOrRaw);
  }
  return "I";
}

function isHeadOfHouseholdRelation(rel: string | null | undefined): boolean {
  if (!rel) return false;
  const low = rel.trim().toLowerCase();
  return (
    low === "hlava domácnosti" ||
    low === "hlava domacnosti" ||
    low === "hlava" ||
    low === "majiteľ bytu" ||
    low === "majitel bytu" ||
    low === "majiteľ domu" ||
    low === "majitel domu" ||
    low === "prednosta domácnosti" ||
    low.includes("hlava domácnosti") ||
    low.includes("majiteľ bytu")
  );
}

function stripFeminineSurnameSuffixClient(surname: string | null | undefined): string {
  return surname ? stripFemSuffix(surname) : "";
}

function formatConsensusSurnameForMemberClient(
  consensusSurname: string,
  sex: string | null | undefined,
  currentSurnameRaw: string | null | undefined
): string {
  const c = consensusSurname.trim().replace(/-+$/, "");
  if (!c) return consensusSurname;
  const cur = (currentSurnameRaw || "").trim();
  const curLow = cur.toLowerCase();
  const cLow = c.toLowerCase();
  const isFem = sex === "F" || curLow.endsWith("ová") || curLow.endsWith("ova") || curLow.endsWith("á");
  if (!isFem) return c;

  // Adjectival surnames (-ský, -cký, -ý, -í) -> feminine is -ská, -cká, -á (NEVER -ová!)
  if (cLow.endsWith("ský") || cLow.endsWith("sky")) {
    return c.slice(0, -3) + "ská";
  }
  if (cLow.endsWith("cký") || cLow.endsWith("cky")) {
    return c.slice(0, -3) + "cká";
  }
  if (cLow.endsWith("ý") || cLow.endsWith("í")) {
    return c.slice(0, -1) + "á";
  }
  if (cLow.endsWith("ová") || cLow.endsWith("ova") || cLow.endsWith("á")) {
    return c;
  }
  // Only add -ová if the member's surname already used -ová/-ova
  if (curLow.endsWith("ová") || curLow.endsWith("ova")) {
    if ((cLow.endsWith("a") || cLow.endsWith("o")) && c.length >= 4) {
      return c.slice(0, -1) + "ová";
    }
    return `${c}ová`;
  }
  return c;
}

function levenshteinDistance(a: string, b: string): number {
  const s = a.toLowerCase();
  const t = b.toLowerCase();
  if (s === t) return 0;
  const m = s.length;
  const n = t.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = s[i - 1] === t[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }
  return dp[m][n];
}

interface ScanRowRecord {
  id: number;
  scan_id: number;
  row_index: number;
  entry_number?: string | null;
  y_start: number;
  y_end: number;
  stitched_crop_path: string | null;
  left_crop_path?: string | null;
  right_crop_path?: string | null;
  status?: string;
  page_side?: string | null;
  page_number?: number | null;
  ocr_feedback?: OcrFeedbackItem | null;
}

interface CensusPersonMember {
  fact: {
    id: number;
    row_id: number;
    scan_id: number;
    person_row_number: number | string | null;
    surname_raw: string | null;
    surname_normalized: string | null;
    mother_maiden_surname: string | null;
    given_name_raw: string | null;
    given_name_sk: string | null;
    relation_to_head: string | null;
    sex: string | null;
    birth_or_event_date: string | null;
    birth_year_est: number | null;
    marital_status: string | null;
    marriage_date_raw: string | null;
    marriage_date?: string | null;
    children_born_count: number | string | null;
    children_dead_count: number | string | null;
    children_deceased_count?: number | string | null;
    birth_place_raw: string | null;
    birth_place_sk: string | null;
    birth_place_municipality?: string | null;
    birth_place_district?: string | null;
    birth_place_country?: string | null;
    citizenship?: string | null;
    moved_in_date: string | null;
    previous_residence: string | null;
    home_jurisdiction: string | null;
    nationality: string | null;
    religion: string | null;
    literacy: string | null;
    social_status_or_occupation: string | null;
    occupation_position: string | null;
    employer_workplace: string | null;
    presence_type: string | null;
    presence_status?: string | null;
    observationes_raw: string | null;
    father_name: string | null;
    mother_name: string | null;
    spouse_name: string | null;
    census_household_id?: number | null;
    household_id?: number | null;
    dwelling_unit?: string | null;
    byt_number?: string | null;
    apartment_number?: string | null;
    surname_ocr_raw?: string | null;
    original_ocr_surname?: string | null;
    surname_consensus_from?: string | null;
    surname_candidates_json?: string | null;
  };
  household_id?: number | null;
  dwelling_unit?: string | null;
  row?: ScanRowRecord | null;
  stitched_crop_url: string | null;
  cell_crops: Record<string, string>;
  fs_match: {
    id: number;
    match_status: string;
    fs_url: string | null;
    confidence_score: number;
    summary_sk: string | null;
  } | null;
  local_matches: Array<{
    match: {
      id: number;
      match_type: string;
      confidence_score: number;
      explanation_sk: string;
    };
    matrika_fact: {
      id: number;
      record_type: string;
      event_year: number | null;
      birth_or_event_date: string | null;
      given_name_raw: string | null;
      surname_raw: string | null;
      father_name: string | null;
      mother_name: string | null;
    } | null;
  }>;
  ocr_feedback?: OcrFeedbackItem | null;
}

export interface CensusOverviewHouseRow {
  id: number;
  census_sheet_id: number;
  scan_id?: number | null;
  row_id?: number | null;
  page_side?: string | null;
  page_number?: number | null;
  booklet_page?: number | null;
  page_half?: string | null;
  header_crop_url?: string | null;
  row_index: number;
  census_year: number;
  district: string | null;
  municipality: string;
  obvod_number: string | null;
  ordinal_number: string | number | null;
  settlement_name: string | null;
  house_number: string | null;
  house_number_int?: number | null;
  is_inhabited: boolean | null;
  owner_name: string | null;
  household_count: number | null;
  present_persons_count: number | null;
  remarks: string | null;
  matched_sheet_id: number | null;
  matched_slovakiana_key?: string | null;
  matched_cair_id?: string | null;
  matched_head_name?: string | null;
  matched_members_count?: number;
  matched_sheet?: LinkedHouseSheet | null;
  matched_sheets?: LinkedHouseSheet[];
  overview_slovakiana_key?: string | null;
  stitched_crop_path: string | null;
  raw_ocr_text?: string | null;
  confidence?: number;
  y_start?: number | null;
  y_end?: number | null;
  ocr_feedback?: OcrFeedbackItem | null;
}

export interface CensusMunicipalOverviewDistrictRow {
  id: number;
  census_sheet_id: number;
  scan_id?: number | null;
  row_id?: number | null;
  page_side?: string | null;
  page_number?: number | null;
  header_crop_url?: string | null;
  row_index: number;
  census_year: number;
  district: string | null;
  municipality: string;
  ordinal_number: string | number | null;
  settlement_name: string | null;
  obvod_number: string | null;
  commissioner_name: string | null;
  houses_total: number | null;
  houses_inhabited: number | null;
  houses_uninhabited: number | null;
  household_count: number | null;
  present_persons_count: number | null;
  remarks: string | null;
  is_summary_row: boolean;
  matched_overview_sheet_id: number | null;
  matched_overview_slovakiana_key?: string | null;
  matched_overview_houses_count?: number;
  matched_overview_sheet?: CensusSheetDetail | null;
  stitched_crop_path: string | null;
  raw_ocr_text?: string | null;
  confidence?: number;
  y_start?: number | null;
  y_end?: number | null;
  ocr_feedback?: OcrFeedbackItem | null;
}

interface ScanDetailInfo {
  id: number;
  filename: string;
  image_url?: string | null;
  original_url?: string | null;
  overlay_path: string | null;
  header_crop_path?: string | null;
  header_crop_url?: string | null;
  width?: number;
  height?: number;
  page_side?: string | null;
}

interface SheetPageInfo {
  page_number: number;
  page_side: string;
  label: string;
  scan_id: number | null;
  scan_url: string | null;
  overlay_url: string | null;
  header_crop_url: string | null;
  width?: number;
  height?: number;
}

interface SheetDetailResponse {
  sheet: CensusSheetDetail;
  front_scan: ScanDetailInfo | null;
  back_scan: ScanDetailInfo | null;
  front_scan_url?: string | null;
  front_overlay_url?: string | null;
  back_scan_url?: string | null;
  back_overlay_url?: string | null;
  pages?: SheetPageInfo[];
  household: {
    id: number;
    apartment_number?: string | null;
    dwelling_unit?: string | null;
    byt_label?: string | null;
    head_surname: string | null;
    head_given_name: string | null;
    head_name?: string | null;
    wife_given_name: string | null;
    wife_maiden_surname: string | null;
    wife_name?: string | null;
    consensus_surname?: string | null;
    member_count: number;
  } | null;
  households?: Array<{
    id?: number;
    apartment_number?: string | null;
    dwelling_unit?: string | null;
    byt_label?: string | null;
    head_surname?: string | null;
    head_given_name?: string | null;
    head_name?: string | null;
    wife_given_name?: string | null;
    wife_maiden_surname?: string | null;
    wife_name?: string | null;
    consensus_surname?: string | null;
    member_count?: number;
  }>;
  members: CensusPersonMember[];
  rows?: ScanRowRecord[];
  overview_houses?: CensusOverviewHouseRow[];
  municipal_overview_districts?: CensusMunicipalOverviewDistrictRow[];
  linked_overview_sheets?: CensusSheetDetail[];
  joined_overview_houses?: CensusOverviewHouseRow[];
  linked_municipal_overview?: {
    id: number;
    slovakiana_id: string;
    revisor_name?: string | null;
    commissioner_name?: string | null;
    districts?: CensusMunicipalOverviewDistrictRow[];
    sibling_overview_sheets?: CensusSheetDetail[];
  } | null;
  linked_overview_house?: CensusOverviewHouseRow | null;
  linked_house_sheets?: LinkedHouseSheet[];
  same_year_house_sheets?: LinkedHouseSheet[];
}

interface Props {
  lang: "sk" | "en";
  refreshStatus: () => void;
}

export interface BytSummaryGroup {
  bytIndex: number;
  bytRoman: string;
  bytLabel: string; // e.g. "Byt č. I"
  householdId: number | null;
  headFullName: string;
  headGivenName: string | null;
  headSurname: string | null;
  wifeFullName: string | null;
  wifeGivenName: string | null;
  wifeMaidenSurname: string | null;
  consensusSurname: string | null;
  memberCount: number;
  members: CensusPersonMember[];
}

function parseExplicitBytFromObservationes(obs: string | null | undefined): string | null {
  if (!obs) return null;
  const m = obs.match(/\bByt\s*(?:č\.?|c\.?)?\s*([IVX0-9]+)\b/i);
  if (m && m[1]) {
    return toRomanBytLabel(m[1]);
  }
  return null;
}

function getExplicitMemberBytRoman(m: CensusPersonMember): string | null {
  const f = m.fact;
  const raw =
    m.dwelling_unit ||
    f.dwelling_unit ||
    f.byt_number ||
    f.apartment_number ||
    parseExplicitBytFromObservationes(f.observationes_raw);
  if (raw && String(raw).trim()) {
    return toRomanBytLabel(String(raw));
  }
  return null;
}

function computeBytConsensusSurname(
  groupMembers: CensusPersonMember[],
  fallbackHeadSurname: string | null
): string | null {
  const counts = new Map<string, { count: number; display: string }>();
  for (const m of groupMembers) {
    const raw = m.fact.surname_normalized || m.fact.surname_raw;
    const stem = stripFeminineSurnameSuffixClient(raw);
    if (!stem || stem === "—") continue;
    const key = stem.toLowerCase();
    const prev = counts.get(key);
    const weight = isHeadOfHouseholdRelation(m.fact.relation_to_head) ? 2 : 1;
    if (prev) {
      prev.count += weight;
    } else {
      counts.set(key, { count: weight, display: stem });
    }
  }
  let best: { count: number; display: string } | null = null;
  for (const entry of counts.values()) {
    if (!best || entry.count > best.count) {
      best = entry;
    }
  }
  if (best) return best.display;
  return stripFeminineSurnameSuffixClient(fallbackHeadSurname) || fallbackHeadSurname || null;
}

function detectSurnameConsensusUnification(
  m: CensusPersonMember,
  bytConsensusSurname: string | null,
  enableConsensus: boolean
): {
  wasUnified: boolean;
  fromRaw: string | null;
  toConsensus: string | null;
} {
  if (!enableConsensus) {
    return { wasUnified: false, fromRaw: null, toConsensus: null };
  }
  const f = m.fact;
  let explicitFrom =
    f.surname_consensus_from || f.original_ocr_surname || f.surname_ocr_raw || null;
  if (!explicitFrom && f.surname_candidates_json) {
    try {
      const parsed = JSON.parse(f.surname_candidates_json);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const firstRaw = parsed[0]?.raw_ocr;
        if (typeof firstRaw === "string" && firstRaw.trim()) {
          explicitFrom = firstRaw.trim();
        }
      }
    } catch {
      // ignore invalid JSON
    }
  }
  if (
    explicitFrom &&
    f.surname_raw &&
    explicitFrom.trim().toLowerCase() !== f.surname_raw.trim().toLowerCase()
  ) {
    return {
      wasUnified: true,
      fromRaw: explicitFrom.trim(),
      toConsensus: f.surname_raw.trim(),
    };
  }

  return { wasUnified: false, fromRaw: null, toConsensus: null };
}

function buildBytGroups(
  detail: SheetDetailResponse,
  multiBytEnabled: boolean
): {
  groups: BytSummaryGroup[];
  memberBytMap: Map<number, BytSummaryGroup>;
} {
  const members = detail.members || [];
  if (members.length === 0) {
    return { groups: [], memberBytMap: new Map<number, BytSummaryGroup>() };
  }
  const rawHouseholds =
    detail.households && detail.households.length > 0
      ? detail.households
      : detail.household
      ? [detail.household]
      : [];

  if (!multiBytEnabled) {
    const singleMembers = members;
    const headM =
      singleMembers.find((m) => isHeadOfHouseholdRelation(m.fact.relation_to_head)) ||
      singleMembers[0] ||
      null;
    const wifeM =
      singleMembers.find((m) => {
        const r = (m.fact.relation_to_head || "").toLowerCase();
        return r === "manželka" || r === "družka";
      }) || null;
    const headGiven =
      detail.household?.head_given_name ||
      headM?.fact.given_name_sk ||
      headM?.fact.given_name_raw ||
      null;
    const headSurname =
      detail.household?.head_surname || headM?.fact.surname_raw || null;
    const wifeGiven =
      detail.household?.wife_given_name ||
      wifeM?.fact.given_name_sk ||
      wifeM?.fact.given_name_raw ||
      null;
    const wifeMaiden =
      detail.household?.wife_maiden_surname ||
      wifeM?.fact.mother_maiden_surname ||
      null;
    const consensus =
      detail.household?.consensus_surname ||
      computeBytConsensusSurname(singleMembers, headSurname);
    const singleGroup: BytSummaryGroup = {
      bytIndex: 1,
      bytRoman: "I",
      bytLabel: "Byt č. I",
      householdId: detail.household?.id ?? null,
      headFullName:
        `${headGiven || ""} ${headSurname || ""}`.trim() || "Neznáma hlava domácnosti",
      headGivenName: headGiven,
      headSurname,
      wifeFullName: wifeGiven
        ? `${wifeGiven}${wifeMaiden ? ` (rod. ${wifeMaiden})` : ""}`
        : null,
      wifeGivenName: wifeGiven,
      wifeMaidenSurname: wifeMaiden,
      consensusSurname: consensus,
      memberCount: singleMembers.length,
      members: singleMembers,
    };
    const map = new Map<number, BytSummaryGroup>();
    for (const m of singleMembers) {
      map.set(m.fact.id, singleGroup);
    }
    return { groups: [singleGroup], memberBytMap: map };
  }

  // Partition members into Byty by:
  // 1) explicit dwelling_unit / apartment_number / Byt tag in observationes, OR
  // 2) distinct household_id / census_household_id when multiple exist, OR
  // 3) sequential boundaries starting at each `hlava domácnosti` (or person_row_number === 1 after row 1)
  const buckets = new Map<string, CensusPersonMember[]>();
  const bucketOrder: string[] = [];

  const distinctHhIds = new Set<number>();
  for (const m of members) {
    const hid = m.household_id ?? m.fact.household_id ?? m.fact.census_household_id;
    if (typeof hid === "number" && hid > 0) {
      distinctHhIds.add(hid);
    }
  }
  const hasMultipleHhIds = distinctHhIds.size > 1;
  const hasExplicitDwellingUnits = members.some((m) => Boolean(getExplicitMemberBytRoman(m)));

    let currentSequentialByt = 1;
  members.forEach((m, idx) => {
    let key: string;
    const explicitRoman = getExplicitMemberBytRoman(m);
    if (explicitRoman) {
      key = explicitRoman;
    } else if (hasMultipleHhIds) {
      const hid = m.household_id ?? m.fact.household_id ?? m.fact.census_household_id;
      key = hid ? `HH_${hid}` : "I";
    } else {
      const isHead = isHeadOfHouseholdRelation(m.fact.relation_to_head);
      const rowNum = Number(m.fact.person_row_number);
      const obsTag = parseExplicitBytFromObservationes(m.fact.observationes_raw);
      const prevSurname = idx > 0 ? stripFeminineSurnameSuffixClient(members[idx - 1].fact.surname_raw) : "";
      const curSurname = stripFeminineSurnameSuffixClient(m.fact.surname_raw);
      const sameFamily =
        Boolean(prevSurname && curSurname) &&
        (prevSurname.toLowerCase() === curSurname.toLowerCase() ||
          levenshteinDistance(prevSurname, curSurname) <= 2);
      if (
        idx > 0 &&
        !sameFamily &&
        (Boolean(obsTag && obsTag !== "I") ||
          rowNum === 1 ||
          (isHead && !hasExplicitDwellingUnits && rowNum === 1))
      ) {
        currentSequentialByt += 1;
      }
      key = toRomanBytLabel(currentSequentialByt);
    }
    if (!buckets.has(key)) {
      buckets.set(key, []);
      bucketOrder.push(key);
    }
    buckets.get(key)!.push(m);
  });

  // If backend returned more households in `data.households` than buckets detected, ensure all are represented
  if (rawHouseholds.length > bucketOrder.length && bucketOrder.length <= 1) {
    const groups: BytSummaryGroup[] = rawHouseholds.map((hh, i) => {
      const roman = toRomanBytLabel(hh.dwelling_unit || hh.apartment_number || i + 1);
      const hhMembers = members.filter((m) => {
        const hid = m.household_id ?? m.fact.household_id ?? m.fact.census_household_id;
        return hh.id && hid === hh.id;
      });
      const assignedMembers = hhMembers.length > 0 ? hhMembers : i === 0 ? members : [];
      const headGiven = hh.head_given_name || null;
      const headSurname = hh.head_surname || null;
      const wifeGiven = hh.wife_given_name || null;
      const wifeMaiden = hh.wife_maiden_surname || null;
      const consensus =
        hh.consensus_surname || computeBytConsensusSurname(assignedMembers, headSurname);
      return {
        bytIndex: i + 1,
        bytRoman: roman,
        bytLabel: `Byt č. ${roman}`,
        householdId: hh.id ?? null,
        headFullName:
          hh.head_name ||
          `${headGiven || ""} ${headSurname || ""}`.trim() ||
          "Neznáma hlava domácnosti",
        headGivenName: headGiven,
        headSurname,
        wifeFullName:
          hh.wife_name ||
          (wifeGiven ? `${wifeGiven}${wifeMaiden ? ` (rod. ${wifeMaiden})` : ""}` : null),
        wifeGivenName: wifeGiven,
        wifeMaidenSurname: wifeMaiden,
        consensusSurname: consensus,
        memberCount: hh.member_count ?? assignedMembers.length,
        members: assignedMembers,
      };
    });
    const map = new Map<number, BytSummaryGroup>();
    for (const g of groups) {
      for (const m of g.members) {
        map.set(m.fact.id, g);
      }
    }
    return { groups, memberBytMap: map };
  }

  const groups: BytSummaryGroup[] = bucketOrder.map((key, idx) => {
    const groupMembers = buckets.get(key) || [];
    const matchedHh =
      rawHouseholds.find((hh) => {
        if (key.startsWith("HH_") && hh.id === Number(key.slice(3))) return true;
        const hhRoman = toRomanBytLabel(hh.dwelling_unit || hh.apartment_number);
        return hhRoman === key;
      }) || (idx === 0 ? rawHouseholds[0] : rawHouseholds[idx]);

    const roman = key.startsWith("HH_") ? toRomanBytLabel(idx + 1) : toRomanBytLabel(key);
    const headM =
      groupMembers.find((m) => isHeadOfHouseholdRelation(m.fact.relation_to_head)) ||
      groupMembers[0] ||
      null;
    const wifeM =
      groupMembers.find((m) => {
        const r = (m.fact.relation_to_head || "").toLowerCase();
        return r === "manželka" || r === "družka";
      }) || null;

    const headGiven =
      headM?.fact.given_name_sk ||
      headM?.fact.given_name_raw ||
      matchedHh?.head_given_name ||
      null;
    const headSurname =
      headM?.fact.surname_raw || matchedHh?.head_surname || null;
    const wifeGiven =
      wifeM?.fact.given_name_sk ||
      wifeM?.fact.given_name_raw ||
      matchedHh?.wife_given_name ||
      null;
    const wifeMaiden =
      wifeM?.fact.mother_maiden_surname || matchedHh?.wife_maiden_surname || null;
    const consensus =
      matchedHh?.consensus_surname ||
      computeBytConsensusSurname(groupMembers, headSurname);

    return {
      bytIndex: idx + 1,
      bytRoman: roman,
      bytLabel: `Byt č. ${roman}`,
      householdId: matchedHh?.id ?? null,
      headFullName:
        `${headGiven || ""} ${headSurname || ""}`.trim() ||
        matchedHh?.head_name ||
        "Neznáma hlava domácnosti",
      headGivenName: headGiven,
      headSurname,
      wifeFullName: wifeGiven
        ? `${wifeGiven}${wifeMaiden ? ` (rod. ${wifeMaiden})` : ""}`
        : matchedHh?.wife_name || null,
      wifeGivenName: wifeGiven,
      wifeMaidenSurname: wifeMaiden,
      consensusSurname: consensus,
      memberCount: groupMembers.length,
      members: groupMembers,
    };
  });

  if (groups.length === 0 && detail.household) {
    const hh = detail.household;
    groups.push({
      bytIndex: 1,
      bytRoman: "I",
      bytLabel: "Byt č. I",
      householdId: hh.id,
      headFullName:
        `${hh.head_given_name || ""} ${hh.head_surname || ""}`.trim() ||
        "Neznáma hlava domácnosti",
      headGivenName: hh.head_given_name,
      headSurname: hh.head_surname,
      wifeFullName: hh.wife_given_name
        ? `${hh.wife_given_name}${
            hh.wife_maiden_surname ? ` (rod. ${hh.wife_maiden_surname})` : ""
          }`
        : null,
      wifeGivenName: hh.wife_given_name,
      wifeMaidenSurname: hh.wife_maiden_surname,
      consensusSurname: hh.consensus_surname || hh.head_surname,
      memberCount: hh.member_count || 0,
      members: [],
    });
  }

  const map = new Map<number, BytSummaryGroup>();
  for (const g of groups) {
    for (const m of g.members) {
      map.set(m.fact.id, g);
    }
  }
  return { groups, memberBytMap: map };
}

function normalizePlaceSpelling(text: string | null | undefined): string {
  if (!text) return "";
  return text.trim();
}

function formatHomeJurisdiction(raw: string | null | undefined): string | null {
  if (!raw || !raw.trim()) return null;
  let cleaned = normalizePlaceSpelling(raw.trim());
  // Strip redundant trailing ", Slovensko" when displayed alongside citizenship
  cleaned = cleaned.replace(/,\s*Slovensko$/i, "").trim();
  return cleaned || null;
}

function formatStructuredBirthPlace(f: CensusPersonMember["fact"]): {
  primary: string;
  secondary: string | null;
} {
  const muni = normalizePlaceSpelling(f.birth_place_municipality?.trim() || "");
  const dist = f.birth_place_district?.trim() || "";
  const country = f.birth_place_country?.trim() || "";
  if (muni || dist || country) {
    const primary =
      muni ||
      normalizePlaceSpelling(f.birth_place_sk || f.birth_place_raw || "") ||
      "—";
    const subParts: string[] = [];
    if (dist) subParts.push(`okr. ${dist}`);
    if (country) subParts.push(country);
    return {
      primary,
      secondary: subParts.length > 0 ? subParts.join(" · ") : null,
    };
  }
  return {
    primary:
      normalizePlaceSpelling(f.birth_place_raw || f.birth_place_sk || "") || "—",
    secondary: null,
  };
}

export default function CensusSheetView({ lang, refreshStatus }: Props) {
  const { sheetId } = useParams<{ sheetId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const { flags: featureFlags } = useFeatureFlags();
  const [detail, setDetail] = useState<SheetDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [switchingType, setSwitchingType] = useState(false);
  const [switchingYear, setSwitchingYear] = useState(false);
  const [feedbackResetVersion, setFeedbackResetVersion] = useState<number>(0);

  const clearAllSheetFeedbackInState = () => {
    setFeedbackResetVersion((v) => v + 1);
    setDetail((prev) =>
      prev
        ? {
            ...prev,
            members: (prev.members || []).map((m) => ({
              ...m,
              ocr_feedback: null,
            })),
            overview_houses: (prev.overview_houses || []).map((h) => ({
              ...h,
              ocr_feedback: null,
            })),
            municipal_overview_districts: (
              prev.municipal_overview_districts || []
            ).map((d) => ({
              ...d,
              ocr_feedback: null,
            })),
            rows: (prev.rows || []).map((r) => ({
              ...r,
              ocr_feedback: null,
            })),
          }
        : prev
    );
  };
  const [sheetProgress, setSheetProgress] = useState<{
    active: boolean;
    completedJustNow: boolean;
    active_stage: string;
    active_model: string | null;
    progress_pct: number;
    current_row_index: number | null;
    total_rows: number | null;
    message_sk: string;
    message_en: string;
    started_at_ms: number;
    elapsed_sec: number;
  }>({
    active: false,
    completedJustNow: false,
    active_stage: "idle",
    active_model: null,
    progress_pct: 0,
    current_row_index: null,
    total_rows: null,
    message_sk: "",
    message_en: "",
    started_at_ms: 0,
    elapsed_sec: 0,
  });

  // Scan viewer controls: side (`front` = _v, `back` = _r, or `page_3`..N) & mode (`overlay` vs `original`)
  const [activeSide, setActiveSide] = useState<string>(() => {
    const s = searchParams.get("side");
    return s && s.trim() ? s.trim() : "front";
  });
  const [activeScanMode, setActiveScanMode] = useState<"overlay" | "original">(() =>
    searchParams.get("mode") === "overlay" ? "overlay" : "original"
  );
  // Full-sheet scan is secondary: hidden by default and shown only on user request (or when admin_debug_overlay is enabled)
  const [showFullSheetScan, setShowFullSheetScan] = useState<boolean>(
    () => searchParams.get("scan") === "1"
  );
  const [activeFieldFocus, setActiveFieldFocus] = useState<ActiveFieldFocus | null>(null);
  const splitLayout = false;

  // Household member selection & drawer editing
  const [selectedMemberId, setSelectedMemberId] = useState<number | null>(() => {
    const r = Number(searchParams.get("row"));
    return Number.isFinite(r) && r > 0 ? r : null;
  });
  const [editingMemberId, setEditingMemberId] = useState<number | null>(() => {
    const r = Number(searchParams.get("editRow"));
    return Number.isFinite(r) && r > 0 ? r : null;
  });
  const [editForm, setEditForm] = useState<Record<string, any>>({});
  const [savingId, setSavingId] = useState<number | null>(null);
  const [addingPerson, setAddingPerson] = useState<boolean>(false);

  // Overview house selection & drawer editing
  const [selectedHouseId, setSelectedHouseId] = useState<number | null>(() => {
    const h = Number(searchParams.get("house"));
    return Number.isFinite(h) && h > 0 ? h : null;
  });
  const [editingHouseId, setEditingHouseId] = useState<number | null>(null);
  const [houseEditForm, setHouseEditForm] = useState<Record<string, any>>({});
  const [savingHouseId, setSavingHouseId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmDeleteMemberId, setConfirmDeleteMemberId] = useState<number | null>(null);
  const [deletingMemberId, setDeletingMemberId] = useState<number | null>(null);
  const [confirmDeleteHouseId, setConfirmDeleteHouseId] = useState<number | null>(null);
  const [deletingHouseId, setDeletingHouseId] = useState<number | null>(null);

  // Municipal overview (Obecný prehľad) district row selection & editing
  const [selectedDistrictRowId, setSelectedDistrictRowId] = useState<number | null>(null);
  const [editingDistrictRowId, setEditingDistrictRowId] = useState<number | null>(null);
  const [districtEditForm, setDistrictEditForm] = useState<Record<string, any>>({});
  const [savingDistrictRowId, setSavingDistrictRowId] = useState<number | null>(null);
  const [confirmDeleteDistrictRowId, setConfirmDeleteDistrictRowId] = useState<number | null>(null);
  const [deletingDistrictRowId, setDeletingDistrictRowId] = useState<number | null>(null);

  const highlightPersonQuery = (
    searchParams.get("meno") ||
    searchParams.get("person_name") ||
    searchParams.get("q") ||
    ""
  ).trim();

  // Sync active side, scan mode, and selected row/house to URL query params
  useEffect(() => {
    const next = new URLSearchParams();
    if (activeSide && activeSide !== "front") next.set("side", activeSide);
    if (activeScanMode === "original") next.set("mode", "original");
    if (selectedMemberId !== null) next.set("row", String(selectedMemberId));
    if (editingMemberId !== null) next.set("editRow", String(editingMemberId));
    if (selectedHouseId !== null) next.set("house", String(selectedHouseId));
    if (highlightPersonQuery) next.set("meno", highlightPersonQuery);
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [activeSide, activeScanMode, selectedMemberId, editingMemberId, selectedHouseId, highlightPersonQuery]);

  const loadDetail = async () => {
    if (!sheetId) return;
    try {
      const r = await fetch(`/api/census/sheets/${sheetId}`);
      if (r.ok) {
        const data = (await r.json()) as SheetDetailResponse;
        setDetail(data);
        if (data.sheet?.slovakiana_id) {
          const canonicalPath = `/scitacie-harky/${data.sheet.slovakiana_id}`;
          if (location.pathname !== canonicalPath) {
            navigate(`${canonicalPath}${location.search}`, { replace: true });
          }
        }
        if (data.members?.length > 0) {
          setSelectedMemberId((prev) => {
            if (prev && data.members.some((m) => m.fact.id === prev)) {
              return prev;
            }
            if (highlightPersonQuery) {
              const fuzzyMatched = data.members.find((m) =>
                matchesPersonNameQuery(
                  [
                    m.fact.given_name_raw,
                    m.fact.given_name_sk,
                    m.fact.surname_raw,
                    m.fact.surname_normalized,
                    m.fact.mother_maiden_surname,
                    m.fact.observationes_raw,
                  ],
                  highlightPersonQuery
                )
              );
              if (fuzzyMatched) return fuzzyMatched.fact.id;
            }
            return data.members[0].fact.id;
          });
        } else {
          setSelectedMemberId(null);
        }
        if (data.overview_houses && data.overview_houses.length > 0) {
          setSelectedHouseId((prev) => {
            if (prev && data.overview_houses!.some((h) => h.id === prev)) {
              return prev;
            }
            if (highlightPersonQuery) {
              const fuzzyHouse = data.overview_houses!.find((h) =>
                matchesPersonNameQuery(
                  [h.owner_name, h.matched_head_name, h.remarks],
                  highlightPersonQuery
                )
              );
              if (fuzzyHouse) return fuzzyHouse.id;
            }
            return data.overview_houses![0].id;
          });
        } else {
          setSelectedHouseId(null);
        }
        if (data.municipal_overview_districts && data.municipal_overview_districts.length > 0) {
          setSelectedDistrictRowId((prev) => {
            if (prev && data.municipal_overview_districts!.some((d) => d.id === prev)) {
              return prev;
            }
            return data.municipal_overview_districts![0].id;
          });
        } else {
          setSelectedDistrictRowId(null);
        }
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const urlRow = Number(searchParams.get("row"));
    const urlEditRow = Number(searchParams.get("editRow"));
    const urlHouse = Number(searchParams.get("house"));
    setSelectedMemberId(Number.isFinite(urlRow) && urlRow > 0 ? urlRow : null);
    setEditingMemberId(Number.isFinite(urlEditRow) && urlEditRow > 0 ? urlEditRow : null);
    setSelectedHouseId(Number.isFinite(urlHouse) && urlHouse > 0 ? urlHouse : null);
    setEditingHouseId(null);
    setSelectedDistrictRowId(null);
    setEditingDistrictRowId(null);
    setSheetProgress((prev) => ({
      ...prev,
      active: false,
      completedJustNow: false,
      progress_pct: 0,
    }));
    setLoading(true);
    loadDetail();
  }, [sheetId]);

  // Listen for live pipeline state & FamilySearch matching broadcasts over /ws/pipeline
  useEffect(() => {
    const matchesCurrentSheet = (d: PipelineState) => {
      if (!sheetId && !detail?.sheet) return false;
      const curKey = detail?.sheet?.slovakiana_id || sheetId;
      const curId = detail?.sheet?.id;
      const frontScanId = detail?.front_scan?.id;
      const backScanId = detail?.back_scan?.id;
      if (d.current_slovakiana_key && curKey && d.current_slovakiana_key === curKey) {
        return true;
      }
      if (d.current_sheet_id != null && curId != null && d.current_sheet_id === curId) {
        return true;
      }
      if (
        d.current_scan_id != null &&
        (d.current_scan_id === frontScanId || d.current_scan_id === backScanId)
      ) {
        return true;
      }
      return false;
    };

    const handlePipelineStateEvent = (e: Event) => {
      const custom = e as CustomEvent<PipelineState>;
      const d = custom.detail;
      if (!d) return;
      const st = String(d.active_stage || "idle");
      const isThisSheet = matchesCurrentSheet(d);
      if (st !== "idle" && (processing || switchingType || isThisSheet)) {
        setSheetProgress((prev) => {
          const incomingPct = Number(d.progress_pct ?? 12);
          return {
            ...prev,
            active: true,
            active_stage: st,
            active_model: d.active_model ?? prev.active_model,
            progress_pct: Math.max(prev.progress_pct, Math.min(99, incomingPct)),
            current_row_index: d.current_row_index ?? prev.current_row_index,
            total_rows: d.total_rows ?? prev.total_rows,
            message_sk: String(d.message_sk || prev.message_sk),
            message_en: String(d.message_en || prev.message_en),
            started_at_ms: prev.started_at_ms || (d.started_at ? d.started_at * 1000 : Date.now()),
          };
        });
      }
    };

    const handleSheetUpdatedEvent = (e: Event) => {
      const custom = e as CustomEvent<any>;
      const d = custom.detail;
      const curKey = detail?.sheet?.slovakiana_id || sheetId;
      const curId = detail?.sheet?.id;
      if (
        d &&
        ((d.slovakiana_key && d.slovakiana_key === curKey) ||
          (d.sheet_id != null && d.sheet_id === curId))
      ) {
        void loadDetail();
        setSheetProgress((prev) =>
          prev.active && !processing && !switchingType
            ? {
                ...prev,
                active: false,
                completedJustNow: true,
                active_stage: "idle",
                progress_pct: 100,
                message_sk: "AI OCR čítanie a párovanie hárku bolo úspešne dokončené.",
                message_en: "AI OCR reading and sheet matching completed successfully.",
              }
            : prev
        );
      }
    };

    window.addEventListener("matrika:pipeline-state", handlePipelineStateEvent);
    window.addEventListener("matrika:sheet-updated", handleSheetUpdatedEvent);

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws/pipeline`;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "state" && msg.data) {
            const d = msg.data as PipelineState;
            const st = String(d.active_stage || "idle");
            const isThisSheet = matchesCurrentSheet(d);
            if (st !== "idle") {
              setSheetProgress((prev) => {
                if (!prev.active && !processing && !switchingType && !isThisSheet) return prev;
                const incomingPct = Number(d.progress_pct ?? 12);
                return {
                  ...prev,
                  active: true,
                  active_stage: st,
                  active_model: d.active_model ?? prev.active_model,
                  progress_pct: Math.max(prev.progress_pct, Math.min(99, incomingPct)),
                  current_row_index: d.current_row_index ?? prev.current_row_index,
                  total_rows: d.total_rows ?? prev.total_rows,
                  message_sk: String(d.message_sk || prev.message_sk),
                  message_en: String(d.message_en || prev.message_en),
                  started_at_ms:
                    prev.started_at_ms || (d.started_at ? d.started_at * 1000 : Date.now()),
                };
              });
            }
          } else if (msg.type === "fs_match_progress" && msg.data) {
            const p = msg.data.progress || msg.data;
            if (p && p.running) {
              setSheetProgress((prev) => {
                if (!prev.active && !processing && !switchingType) return prev;
                const tot = Math.max(1, Number(p.total || 1));
                const proc = Number(p.processed || 0);
                const pct = Math.min(99, 88 + Math.round((11 * proc) / tot));
                const person = String(p.current_person || "").trim();
                return {
                  ...prev,
                  active: true,
                  active_stage: "mcp_match",
                  active_model: "familysearch-mcp",
                  progress_pct: Math.max(prev.progress_pct, pct),
                  current_row_index: proc || prev.current_row_index,
                  total_rows: tot || prev.total_rows,
                  message_sk: person
                    ? `Párujem s matrikou a FamilySearch (${proc}/${tot}): ${person}…`
                    : prev.message_sk,
                  message_en: person
                    ? `Matching with parish records & FamilySearch (${proc}/${tot}): ${person}…`
                    : prev.message_en,
                };
              });
            }
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore ws errors; polling + ticker fallback below
    }
    return () => {
      window.removeEventListener("matrika:pipeline-state", handlePipelineStateEvent);
      window.removeEventListener("matrika:sheet-updated", handleSheetUpdatedEvent);
      if (ws) ws.close();
    };
  }, [processing, switchingType, sheetId, detail?.sheet?.id, detail?.sheet?.slovakiana_id]);

  // Smooth client-side progress ticker + status poll while OCR & matching is running on this sheet
  useEffect(() => {
    if (!processing && !switchingType) return;
    const ticker = setInterval(() => {
      setSheetProgress((prev) => {
        const elapsed = Math.max(
          1,
          Math.round((Date.now() - (prev.started_at_ms || Date.now())) / 1000)
        );
        const step = prev.progress_pct < 35 ? 2 : prev.progress_pct < 75 ? 1 : 0.5;
        const nextPct = Math.min(94, Math.round((prev.progress_pct + step) * 10) / 10);
        return {
          ...prev,
          active: true,
          elapsed_sec: elapsed,
          progress_pct: Math.max(prev.progress_pct, Math.floor(nextPct)),
        };
      });
    }, 650);

    const poller = setInterval(async () => {
      try {
        const r = await fetch("/api/status");
        if (!r.ok) return;
        const stData = await r.json();
        const ps = stData?.pipeline_state;
        if (ps && ps.active_stage && ps.active_stage !== "idle") {
          setSheetProgress((prev) => ({
            ...prev,
            active: true,
            active_stage: String(ps.active_stage),
            active_model: ps.active_model ?? prev.active_model,
            progress_pct: Math.max(
              prev.progress_pct,
              Math.min(99, Number(ps.progress_pct || 12))
            ),
            message_sk: String(ps.message_sk || prev.message_sk),
            message_en: String(ps.message_en || prev.message_en),
          }));
        }
      } catch {
        // ignore transient poll error
      }
    }, 1500);

    return () => {
      clearInterval(ticker);
      clearInterval(poller);
    };
  }, [processing, switchingType]);

  const handleSwitchSheetYear = async (targetYear: 1930 | 1940) => {
    if (!detail || detail.sheet.census_year === targetYear || switchingYear) return;
    const currentSheet = detail.sheet;
    const isOv =
      currentSheet.sheet_type === "overview" ||
      currentSheet.sheet_type === "municipal_overview";

    // Fast path for household sheets when linked_house_sheets already has the target year sheet
    if (!isOv) {
      const directLinked = (
        detail.linked_house_sheets ||
        currentSheet.linked_house_sheets ||
        []
      ).find((l) => l.census_year === targetYear);
      if (directLinked) {
        const ref =
          directLinked.cair_id ||
          directLinked.slovakiana_key ||
          String(directLinked.id);
        const sp = new URLSearchParams();
        if (highlightPersonQuery) sp.set("meno", highlightPersonQuery);
        navigate(`/scitacie-harky/${ref}${sp.toString() ? `?${sp.toString()}` : ""}`);
        return;
      }
    }

    setSwitchingYear(true);
    try {
      const sheetRef = currentSheet.slovakiana_id || String(currentSheet.id);
      const params = new URLSearchParams({ target_year: String(targetYear) });
      const activeHouseObj =
        (detail.overview_houses || []).find((h) => h.id === selectedHouseId) ||
        detail.overview_houses?.[0] ||
        null;
      if (isOv && activeHouseObj?.house_number) {
        params.set("house_number", String(activeHouseObj.house_number));
      } else if (
        !isOv &&
        (currentSheet.house_number_conscription || currentSheet.house_number_raw)
      ) {
        params.set(
          "house_number",
          String(
            currentSheet.house_number_conscription || currentSheet.house_number_raw
          )
        );
      }
      const res = await fetch(
        `/api/census/sheets/${encodeURIComponent(sheetRef)}/switch-year?${params.toString()}`
      );
      if (res.ok) {
        const data = await res.json();
        if (
          (data.target_type === "sheet" || data.target_type === "overview_house") &&
          (data.slovakiana_key || data.sheet_id)
        ) {
          const ref = data.slovakiana_key || String(data.sheet_id);
          const sp = new URLSearchParams();
          if (data.house_id) sp.set("house", String(data.house_id));
          if (highlightPersonQuery) sp.set("meno", highlightPersonQuery);
          navigate(`/scitacie-harky/${ref}${sp.toString() ? `?${sp.toString()}` : ""}`);
          return;
        }
      }
    } catch {
      // Fallback below
    } finally {
      setSwitchingYear(false);
    }

    // Fallback: navigate to the same village (and house number if applicable) in targetYear
    const sp = new URLSearchParams();
    if (currentSheet.municipality_name) {
      sp.set("obec", normalizePlaceSpelling(currentSheet.municipality_name));
    }
    if (currentSheet.district_name) {
      sp.set("okres", currentSheet.district_name);
    }
    sp.set("rok", String(targetYear));
    const hn = !isOv
      ? currentSheet.house_number_conscription || currentSheet.house_number_raw
      : null;
    if (hn) {
      sp.set("q", String(hn));
    }
    if (highlightPersonQuery) {
      sp.set("meno", highlightPersonQuery);
    }
    navigate(`/scitacie-harky?${sp.toString()}`);
  };

  const handleDownloadAndProcess = async () => {
    if (!sheetId) return;
    setActionError(null);
    clearAllSheetFeedbackInState();
    setProcessing(true);
    const startMs = Date.now();
    const isReprocess =
      detail?.sheet.download_status === "downloaded" ||
      detail?.sheet.download_status === "extracted" ||
      detail?.sheet.download_status === "matched";
    setSheetProgress({
      active: true,
      completedJustNow: false,
      active_stage: isReprocess ? "ocr" : "census_download",
      active_model: isReprocess ? "glm-ocr" : "slovakiana-iiif",
      progress_pct: 8,
      current_row_index: null,
      total_rows: null,
      message_sk: isReprocess
        ? "Spúšťam opätovné AI OCR čítanie rukopisu, extrakciu a párovanie hárku…"
        : "Sťahujem skeny (_v + _r) zo Slovakiany a spúšťam AI OCR hárku…",
      message_en: isReprocess
        ? "Starting handwriting AI OCR re-run, extraction, and sheet matching…"
        : "Downloading scans (_v + _r) from Slovakiana and running AI OCR…",
      started_at_ms: startMs,
      elapsed_sec: 0,
    });
    try {
      const r = await fetch(
        `/api/census/sheets/${sheetId}/download-and-process?run_pipeline=true&run_mcp=true`,
        { method: "POST" }
      );
      if (!r.ok) {
        const errBody = await r.json().catch(() => ({}));
        throw new Error(
          errBody.detail ||
            errBody.error ||
            (lang === "sk"
              ? `Chyba pri spracovaní hárku (HTTP ${r.status})`
              : `Sheet processing failed (HTTP ${r.status})`)
        );
      }
      const resData = await r.json().catch(() => ({}));
      if (resData?.error || resData?.pipeline?.status === "error") {
        throw new Error(resData?.error || resData?.pipeline?.error || "OCR error");
      }
      await loadDetail();
      setFeedbackResetVersion((v) => v + 1);
      refreshStatus();
      const totalElapsed = Math.max(1, Math.round((Date.now() - startMs) / 1000));
      const pRes = resData?.pipeline || {};
      const rowsProc = Number(pRes.rows_processed ?? resData?.rows_processed ?? 0);
      const personsCnt = pRes.persons_count;
      const doneMsgSk = "AI OCR čítanie a párovanie hárku bolo úspešne dokončené.";
      const doneMsgEn = "AI OCR reading and sheet matching completed successfully.";
      setSheetProgress((prev) => ({
        ...prev,
        active: false,
        completedJustNow: true,
        active_stage: "idle",
        active_model: null,
        progress_pct: 100,
        elapsed_sec: totalElapsed,
        message_sk: doneMsgSk,
        message_en: doneMsgEn,
      }));
      emitClientOcrNotification({
        status: "completed",
        kind: "census_sheet",
        sheet_id: detail?.sheet?.id ?? resData?.sheet_id ?? null,
        slovakiana_key: detail?.sheet?.slovakiana_id || resData?.slovakiana_key || sheetId,
        rows_processed: rowsProc,
        persons_count: personsCnt,
        duration_sec: totalElapsed,
        title_sk: "AI OCR hárku dokončené!",
        title_en: "Sheet AI OCR completed!",
        message_sk: doneMsgSk,
        message_en: doneMsgEn,
      });
    } catch (err: any) {
      setActionError(
        err?.message ||
          (lang === "sk"
            ? "Nepodarilo sa dokončiť AI OCR a párovanie hárku."
            : "Failed to complete AI OCR and sheet matching.")
      );
      setSheetProgress((prev) => ({
        ...prev,
        active: false,
        completedJustNow: false,
        active_stage: "idle",
      }));
    } finally {
      setProcessing(false);
    }
  };

  const handleSetSheetType = async (nextType: "household" | "overview" | "municipal_overview") => {
    if (!detail) return;
    const prevType = detail.sheet.sheet_type || "household";
    const sheetRef = detail.sheet.slovakiana_id || String(detail.sheet.id);
    setActionError(null);
    clearAllSheetFeedbackInState();
    // Optimistically reflect the chosen sheet type immediately in the UI
    setDetail((prev) =>
      prev
        ? {
            ...prev,
            sheet: {
              ...prev.sheet,
              sheet_type: nextType,
            },
          }
        : prev
    );
    setSwitchingType(true);
    const startMs = Date.now();
    setSheetProgress({
      active: true,
      completedJustNow: false,
      active_stage: "layout",
      active_model: "glm-ocr",
      progress_pct: 10,
      current_row_index: null,
      total_rows: null,
      message_sk: "Mením typ hárku a spúšťam AI OCR extrakciu riadkov…",
      message_en: "Switching sheet type and running AI OCR row extraction…",
      started_at_ms: startMs,
      elapsed_sec: 0,
    });
    try {
      const r = await fetch(`/api/census/sheets/${sheetRef}/set-type`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sheet_type: nextType,
          reprocess: true,
          run_extract: true,
        }),
      });
      if (!r.ok) {
        const errBody = await r.json().catch(() => ({}));
        throw new Error(
          errBody.detail ||
            errBody.error ||
            (lang === "sk"
              ? `Nepodarilo sa zmeniť typ hárku (HTTP ${r.status})`
              : `Failed to switch sheet type (HTTP ${r.status})`)
        );
      }
      await loadDetail();
      setFeedbackResetVersion((v) => v + 1);
      refreshStatus();
      const totalElapsed = Math.max(1, Math.round((Date.now() - startMs) / 1000));
      const doneMsgSk = "Typ hárku bol zmenený a AI OCR extrakcia bola dokončená.";
      const doneMsgEn = "Sheet type updated and AI OCR extraction completed.";
      setSheetProgress((prev) => ({
        ...prev,
        active: false,
        completedJustNow: true,
        active_stage: "idle",
        active_model: null,
        progress_pct: 100,
        elapsed_sec: totalElapsed,
        message_sk: doneMsgSk,
        message_en: doneMsgEn,
      }));
      emitClientOcrNotification({
        status: "completed",
        kind: "census_sheet",
        sheet_id: detail?.sheet?.id ?? null,
        slovakiana_key: sheetRef,
        duration_sec: totalElapsed,
        title_sk: "Extrakcia hárku dokončená!",
        title_en: "Sheet extraction completed!",
        message_sk: doneMsgSk,
        message_en: doneMsgEn,
      });
    } catch (err: any) {
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              sheet: {
                ...prev.sheet,
                sheet_type: prevType,
              },
            }
          : prev
      );
      setActionError(
        err?.message ||
          (lang === "sk"
            ? "Nepodarilo sa zmeniť typ hárku."
            : "Failed to switch sheet type.")
      );
      setSheetProgress((prev) => ({
        ...prev,
        active: false,
        completedJustNow: false,
        active_stage: "idle",
      }));
    } finally {
      setSwitchingType(false);
    }
  };

  const handleToggleSheetType = async () => {
    if (!detail) return;
    const currentType = detail.sheet.sheet_type || "household";
    const nextType =
      currentType === "household"
        ? "overview"
        : currentType === "overview"
        ? "municipal_overview"
        : "household";
    await handleSetSheetType(nextType);
  };

  const startEditMember = (m: CensusPersonMember, inferredBytRoman?: string) => {
    const f = m.fact;
    const explicitByt = getExplicitMemberBytRoman(m);
    setSelectedMemberId(f.id);
    setEditingMemberId(f.id);
    setEditForm({
      person_row_number: f.person_row_number ?? "",
      dwelling_unit: explicitByt || inferredBytRoman || "",
      surname_raw: f.surname_raw ?? "",
      mother_maiden_surname: f.mother_maiden_surname ?? "",
      given_name_raw: f.given_name_raw ?? "",
      relation_to_head: f.relation_to_head ?? "",
      sex: f.sex ?? "",
      birth_or_event_date: f.birth_or_event_date ?? "",
      marital_status: f.marital_status ?? "",
      marriage_date: f.marriage_date ?? f.marriage_date_raw ?? "",
      spouse_name: f.spouse_name ?? "",
      father_name: f.father_name ?? "",
      mother_name: f.mother_name ?? "",
      children_born_count: f.children_born_count ?? "",
      children_deceased_count: f.children_deceased_count ?? f.children_dead_count ?? "",
      birth_place_municipality: f.birth_place_municipality ?? "",
      birth_place_district: f.birth_place_district ?? "",
      birth_place_country: f.birth_place_country ?? "",
      birth_place_raw: f.birth_place_raw ?? "",
      moved_in_date: f.moved_in_date ?? "",
      previous_residence: f.previous_residence ?? "",
      home_jurisdiction: f.home_jurisdiction ?? "",
      citizenship: f.citizenship ?? "",
      nationality: f.nationality ?? "",
      religion: f.religion ?? "",
      literacy: f.literacy ?? "",
      social_status_or_occupation: f.social_status_or_occupation ?? "",
      occupation_position: f.occupation_position ?? "",
      employer_workplace: f.employer_workplace ?? "",
      presence_status: f.presence_status ?? f.presence_type ?? "trvale",
      observationes_raw: f.observationes_raw ?? "",
    });
  };

  const handleSaveMember = async (factId: number) => {
    setSavingId(factId);
    try {
      let nextObservationes = String(editForm.observationes_raw || "").trim();
      const chosenByt = String(editForm.dwelling_unit || "").trim();
      if (chosenByt) {
        const roman = toRomanBytLabel(chosenByt);
        if (/\bByt\s*(?:č\.?|c\.?)?\s*[IVX0-9]+\b/i.test(nextObservationes)) {
          nextObservationes = nextObservationes.replace(
            /\bByt\s*(?:č\.?|c\.?)?\s*[IVX0-9]+\b/i,
            `Byt č. ${roman}`
          );
        } else {
          nextObservationes = nextObservationes
            ? `Byt č. ${roman} · ${nextObservationes}`
            : `Byt č. ${roman}`;
        }
      }
      const payload = {
        ...editForm,
        dwelling_unit: chosenByt ? toRomanBytLabel(chosenByt) : undefined,
        observationes_raw: nextObservationes,
        maiden_surname: editForm.mother_maiden_surname,
        marriage_date_raw: editForm.marriage_date,
        children_dead_count: editForm.children_deceased_count,
        presence_type: editForm.presence_status,
        person_row_number: editForm.person_row_number
          ? Number(editForm.person_row_number) || editForm.person_row_number
          : null,
      };
      let res = await fetch(`/api/census/facts/${factId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        res = await fetch(`/api/census/persons/${factId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť zmeny osoby."
              : "Failed to save person changes.")
        );
        return;
      }
      const resData = await res.json().catch(() => ({}));
      const savedFact = resData.fact;
      setActionError(null);
      setEditingMemberId(null);
      if (savedFact) {
        setDetail((prev) =>
          prev
            ? {
                ...prev,
                members: (prev.members || []).map((m) =>
                  m.fact.id === factId
                    ? {
                        ...m,
                        fact: { ...m.fact, ...savedFact },
                      }
                    : m
                ),
              }
            : prev
        );
      }
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia so serverom pri ukladaní osoby."
          : "Network error while saving person."
      );
    } finally {
      setSavingId(null);
    }
  };

  const handleDeleteMember = async (factId: number) => {
    setDeletingMemberId(factId);
    setActionError(null);
    try {
      const res = await fetch(`/api/census/persons/${factId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa vymazať osobu z hárku."
              : "Failed to delete person from sheet.")
        );
        return;
      }
      setConfirmDeleteMemberId(null);
      if (editingMemberId === factId) setEditingMemberId(null);
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              members: (prev.members || []).filter((m) => m.fact.id !== factId),
            }
          : prev
      );
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia pri mazaní osoby."
          : "Network error while deleting person."
      );
    } finally {
      setDeletingMemberId(null);
    }
  };

  const handleAddPerson = async () => {
    if (!sheetId || addingPerson) return;
    setAddingPerson(true);
    setActionError(null);
    try {
      const nextRow = (detail?.members.length || 0) + 1;
      const r = await fetch(`/api/census/sheets/${sheetId}/persons`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_row_number: nextRow,
          surname_raw: detail?.household?.head_surname || "",
          given_name_raw: "",
          relation_to_head: nextRow === 1 ? "hlava domácnosti" : "syn",
        }),
      });
      if (r.ok) {
        const resData = await r.json().catch(() => ({}));
        const createdFact = resData.fact;
        if (createdFact) {
          setDetail((prev) =>
            prev
              ? {
                  ...prev,
                  members: [
                    ...(prev.members || []),
                    {
                      fact: createdFact,
                      row: null,
                      dwelling_unit: "I",
                      household_id: prev.household?.id ?? null,
                      stitched_crop_url: null,
                      cell_crops: {},
                      fs_match: null,
                      local_matches: [],
                      ocr_feedback: null,
                    },
                  ],
                }
              : prev
          );
        }
        void loadDetail();
        refreshStatus();
      }
    } finally {
      setAddingPerson(false);
    }
  };

  const selectHouseAndSyncPage = (h: CensusOverviewHouseRow) => {
    setSelectedHouseId(h.id);
    const targetSide =
      h.page_side ||
      (detail?.back_scan && h.scan_id === detail.back_scan.id ? "back" : "front");
    if (targetSide && targetSide !== activeSide) {
      setActiveSide(targetSide);
    }
  };

  const startEditHouse = (h: CensusOverviewHouseRow) => {
    selectHouseAndSyncPage(h);
    setEditingHouseId(h.id);
    setHouseEditForm({
      ordinal_number: h.ordinal_number ?? "",
      settlement_name: h.settlement_name ?? "",
      house_number: h.house_number ?? "",
      is_inhabited: h.is_inhabited !== false,
      owner_name: h.owner_name ?? "",
      household_count: h.household_count ?? "",
      present_persons_count: h.present_persons_count ?? "",
      remarks: h.remarks ?? "",
    });
  };

  const handleSaveHouse = async (houseId: number) => {
    setSavingHouseId(houseId);
    setActionError(null);
    try {
      const payload = {
        ordinal_number:
          houseEditForm.ordinal_number !== "" ? Number(houseEditForm.ordinal_number) : null,
        settlement_name: houseEditForm.settlement_name || null,
        house_number: houseEditForm.house_number || null,
        is_inhabited: Boolean(houseEditForm.is_inhabited),
        owner_name: houseEditForm.owner_name || null,
        household_count:
          houseEditForm.household_count !== "" ? Number(houseEditForm.household_count) : null,
        present_persons_count:
          houseEditForm.present_persons_count !== ""
            ? Number(houseEditForm.present_persons_count)
            : null,
        remarks: houseEditForm.remarks || null,
      };
      const res = await fetch(`/api/census/overview-houses/${houseId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť zmeny záznamu domu."
              : "Failed to save house row changes.")
        );
        return;
      }
      const resData = await res.json().catch(() => ({}));
      const savedHouse = resData.overview_house;
      setEditingHouseId(null);
      if (savedHouse) {
        setDetail((prev) =>
          prev
            ? {
                ...prev,
                overview_houses: (prev.overview_houses || []).map((h) =>
                  h.id === houseId ? { ...h, ...savedHouse } : h
                ),
              }
            : prev
        );
      }
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia so serverom pri ukladaní domu."
          : "Network error while saving house row."
      );
    } finally {
      setSavingHouseId(null);
    }
  };

  const handleDeleteHouse = async (houseId: number) => {
    setDeletingHouseId(houseId);
    setActionError(null);
    try {
      const res = await fetch(`/api/census/overview-houses/${houseId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa vymazať riadok domu z prehľadu obvodu."
              : "Failed to delete house row from district overview.")
        );
        return;
      }
      setConfirmDeleteHouseId(null);
      if (editingHouseId === houseId) setEditingHouseId(null);
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              overview_houses: (prev.overview_houses || []).filter(
                (h) => h.id !== houseId
              ),
            }
          : prev
      );
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia pri mazaní domu."
          : "Network error while deleting house row."
      );
    } finally {
      setDeletingHouseId(null);
    }
  };

  const startEditDistrictRow = (d: CensusMunicipalOverviewDistrictRow) => {
    setSelectedDistrictRowId(d.id);
    setEditingDistrictRowId(d.id);
    setDistrictEditForm({
      ordinal_number: d.ordinal_number ?? "",
      settlement_name: d.settlement_name ?? "",
      obvod_number: d.obvod_number ?? "",
      commissioner_name: d.commissioner_name ?? "",
      houses_total: d.houses_total ?? "",
      houses_inhabited: d.houses_inhabited ?? "",
      houses_uninhabited: d.houses_uninhabited ?? "",
      household_count: d.household_count ?? "",
      present_persons_count: d.present_persons_count ?? "",
      remarks: d.remarks ?? "",
    });
  };

  const handleSaveDistrictRow = async (districtId: number) => {
    setSavingDistrictRowId(districtId);
    setActionError(null);
    try {
      const payload = {
        ordinal_number:
          districtEditForm.ordinal_number !== "" ? String(districtEditForm.ordinal_number) : null,
        settlement_name: districtEditForm.settlement_name || null,
        obvod_number: districtEditForm.obvod_number || null,
        commissioner_name: districtEditForm.commissioner_name || null,
        houses_total:
          districtEditForm.houses_total !== "" ? Number(districtEditForm.houses_total) : null,
        houses_inhabited:
          districtEditForm.houses_inhabited !== "" ? Number(districtEditForm.houses_inhabited) : null,
        houses_uninhabited:
          districtEditForm.houses_uninhabited !== "" ? Number(districtEditForm.houses_uninhabited) : null,
        household_count:
          districtEditForm.household_count !== "" ? Number(districtEditForm.household_count) : null,
        present_persons_count:
          districtEditForm.present_persons_count !== ""
            ? Number(districtEditForm.present_persons_count)
            : null,
        remarks: districtEditForm.remarks || null,
      };
      const res = await fetch(`/api/census/municipal-overview-districts/${districtId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa uložiť zmeny záznamu obvodu."
              : "Failed to save district row changes.")
        );
        return;
      }
      const resData = await res.json().catch(() => ({}));
      const savedDistrict = resData.municipal_overview_district;
      setEditingDistrictRowId(null);
      if (savedDistrict) {
        setDetail((prev) =>
          prev
            ? {
                ...prev,
                municipal_overview_districts: (
                  prev.municipal_overview_districts || []
                ).map((d) =>
                  d.id === districtId ? { ...d, ...savedDistrict } : d
                ),
              }
            : prev
        );
      }
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia so serverom pri ukladaní obvodu."
          : "Network error while saving district row."
      );
    } finally {
      setSavingDistrictRowId(null);
    }
  };

  const handleDeleteDistrictRow = async (districtId: number) => {
    setDeletingDistrictRowId(districtId);
    setActionError(null);
    try {
      const res = await fetch(`/api/census/municipal-overview-districts/${districtId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setActionError(
          err.detail ||
            (lang === "sk"
              ? "Nepodarilo sa vymazať riadok obvodu."
              : "Failed to delete district row.")
        );
        return;
      }
      setConfirmDeleteDistrictRowId(null);
      if (editingDistrictRowId === districtId) setEditingDistrictRowId(null);
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              municipal_overview_districts: (
                prev.municipal_overview_districts || []
              ).filter((d) => d.id !== districtId),
            }
          : prev
      );
      void loadDetail();
      refreshStatus();
    } catch {
      setActionError(
        lang === "sk"
          ? "Chyba spojenia pri mazaní riadku obvodu."
          : "Network error while deleting district row."
      );
    } finally {
      setDeletingDistrictRowId(null);
    }
  };

  // Build Byt (dwelling unit / household) groups across the sheet (memoized so editForm keystrokes do not recompute)
  const { groups: bytGroups, memberBytMap } = useMemo(
    () =>
      detail
        ? buildBytGroups(detail, featureFlags.census_multi_byt)
        : { groups: [], memberBytMap: new Map<number, BytSummaryGroup>() },
    [detail, featureFlags.census_multi_byt]
  );

  const fuzzyMatchedMemberIds = useMemo(() => {
    const out = new Set<number>();
    if (!highlightPersonQuery || !detail?.members) return out;
    for (const m of detail.members) {
      const f = m.fact;
      if (
        matchesPersonNameQuery(
          [
            f.given_name_raw,
            f.given_name_sk,
            f.surname_raw,
            f.surname_normalized,
            f.mother_maiden_surname,
            f.observationes_raw,
          ],
          highlightPersonQuery
        )
      ) {
        out.add(f.id);
      }
    }
    return out;
  }, [detail?.members, highlightPersonQuery]);

  const fuzzyMatchedHouseIds = useMemo(() => {
    const out = new Set<number>();
    if (!highlightPersonQuery || !detail?.overview_houses) return out;
    for (const h of detail.overview_houses) {
      if (
        matchesPersonNameQuery(
          [h.owner_name, h.matched_head_name, h.remarks],
          highlightPersonQuery
        )
      ) {
        out.add(h.id);
      }
    }
    return out;
  }, [detail?.overview_houses, highlightPersonQuery]);

  const { rowsById, rowsByCrop } = useMemo(() => {
    const byId = new Map<number, ScanRowRecord>();
    const byCrop = new Map<string, ScanRowRecord>();
    for (const r of detail?.rows || []) {
      byId.set(r.id, r);
      if (r.stitched_crop_path) {
        byCrop.set(r.stitched_crop_path, r);
      }
    }
    for (const m of detail?.members || []) {
      if (m.row && m.row.id) {
        byId.set(m.row.id, m.row);
      }
    }
    return { rowsById: byId, rowsByCrop: byCrop };
  }, [detail?.rows, detail?.members]);

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-stone-200 p-10 text-center space-y-3 shadow-2xs">
        <RefreshCw className="w-7 h-7 text-amber-600 animate-spin mx-auto" />
        <div className="font-bold text-stone-900 text-base">
          {lang === "sk"
            ? `Načítavam sčítací hárok${sheetId ? ` (${sheetId})` : ""}…`
            : `Loading census sheet${sheetId ? ` (${sheetId})` : ""}…`}
        </div>
        <div className="w-48 h-1.5 bg-stone-200 rounded-full overflow-hidden mx-auto">
          <div className="w-2/3 h-full bg-amber-600 rounded-full animate-pulse" />
        </div>
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="p-8 text-center text-red-700">
        {lang === "sk" ? "Sčítací hárok sa nenašiel." : "Census sheet not found."}
      </div>
    );
  }

  const {
    sheet,
    front_scan,
    back_scan,
    household,
    members,
    rows = [],
    overview_houses = [],
    municipal_overview_districts = [],
    linked_overview_sheets = [],
    joined_overview_houses = [],
    linked_municipal_overview,
    linked_overview_house,
  } = detail;
  const linkedHouseSheets: LinkedHouseSheet[] = (
    detail.linked_house_sheets ||
    sheet.linked_house_sheets ||
    []
  ).filter((linked) => linked.census_year !== sheet.census_year);
  const sameYearHouseSheets: LinkedHouseSheet[] =
    detail.same_year_house_sheets || sheet.same_year_house_sheets || [];

  const isMunicipalOverviewSheet =
    featureFlags.census_overview_sheets && sheet.sheet_type === "municipal_overview";
  const isOverviewSheet =
    featureFlags.census_overview_sheets && sheet.sheet_type === "overview";
  const hasMultipleByty = bytGroups.length >= 2;

  // Resolve scan URLs for front (_v), back (_r), and any extra pages
  const frontOriginalUrl =
    detail.front_scan_url ||
    front_scan?.original_url ||
    front_scan?.image_url ||
    sheet.front_image_url ||
    (front_scan ? `/media/uploads/${front_scan.filename}` : null);
  const backOriginalUrl =
    detail.back_scan_url ||
    back_scan?.original_url ||
    back_scan?.image_url ||
    sheet.back_image_url ||
    (back_scan ? `/media/uploads/${back_scan.filename}` : null);

  const extraPages = (detail.pages || []).filter(
    (p) => p.page_side !== "front" && p.page_side !== "back"
  );
  const activePageObj = (detail.pages || []).find((p) => p.page_side === activeSide) || null;

  const activeScan =
    activeSide === "front"
      ? front_scan
      : activeSide === "back"
      ? back_scan
      : activePageObj
      ? {
          id: activePageObj.scan_id || 0,
          filename: activePageObj.page_side,
          image_url: activePageObj.scan_url,
          original_url: activePageObj.scan_url,
          overlay_path: activePageObj.overlay_url,
          header_crop_path: activePageObj.header_crop_url,
          width: activePageObj.width,
          height: activePageObj.height,
          page_side: activePageObj.page_side,
        }
      : front_scan;
  const activeOriginalUrl =
    activeSide === "front"
      ? frontOriginalUrl
      : activeSide === "back"
      ? backOriginalUrl
      : activePageObj?.scan_url || frontOriginalUrl;
  const activeOverlayUrl = activeScan?.overlay_path
    ? `${activeScan.overlay_path}${activeScan.overlay_path.includes("?") ? "&" : "?"}v=flat`
    : activeOriginalUrl;
  const activeViewerSrc =
    activeScanMode === "overlay" ? activeOverlayUrl : activeOriginalUrl;

  const resolveOverviewHouseRow = (h: CensusOverviewHouseRow): ScanRowRecord | null => {
    if (h.row_id && rowsById.has(h.row_id)) {
      return rowsById.get(h.row_id)!;
    }
    if (h.stitched_crop_path && rowsByCrop.has(h.stitched_crop_path)) {
      return rowsByCrop.get(h.stitched_crop_path)!;
    }
    return (
      rows.find(
        (r) =>
          (h.scan_id ? r.scan_id === h.scan_id : true) &&
          (r.entry_number === String(h.ordinal_number ?? h.row_index) ||
            r.row_index === h.row_index)
      ) || null
    );
  };

  const selectedMember =
    members.find((m) => m.fact.id === selectedMemberId) || members[0] || null;
  const selectedMemberRow = selectedMember
    ? selectedMember.row || rowsById.get(selectedMember.fact.row_id) || null
    : null;

  const selectedHouse =
    overview_houses.find((h) => h.id === selectedHouseId) || overview_houses[0] || null;
  const selectedDistrictRow =
    municipal_overview_districts.find((d) => d.id === selectedDistrictRowId) ||
    municipal_overview_districts[0] ||
    null;

  // Determine active row crop props for <HoverZoomRowCrop>
  let activeCropUrl: string | null = null;
  let activeHeaderCropUrl: string | null =
    activeScan?.header_crop_path ||
    activeScan?.header_crop_url ||
    front_scan?.header_crop_path ||
    front_scan?.header_crop_url ||
    null;
  let activeRowYStart: number | null = null;
  let activeRowYEnd: number | null = null;
  let activeRowLabel = "";
  let activeRowScanUrl: string | null = frontOriginalUrl;
  let activeRowScanHeight: number | null = front_scan?.height || null;
  let activeRowScanWidth: number | null = front_scan?.width || null;

  if (isMunicipalOverviewSheet && selectedDistrictRow) {
    activeCropUrl = selectedDistrictRow.stitched_crop_path;
    const matchRow =
      (selectedDistrictRow.row_id && rowsById.get(selectedDistrictRow.row_id)) ||
      (selectedDistrictRow.stitched_crop_path &&
        rowsByCrop.get(selectedDistrictRow.stitched_crop_path)) ||
      null;
    activeRowYStart = selectedDistrictRow.y_start ?? matchRow?.y_start ?? null;
    activeRowYEnd = selectedDistrictRow.y_end ?? matchRow?.y_end ?? null;
    activeHeaderCropUrl =
      selectedDistrictRow.header_crop_url ||
      front_scan?.header_crop_path ||
      front_scan?.header_crop_url ||
      activeHeaderCropUrl;
    activeRowLabel = selectedDistrictRow.is_summary_row
      ? lang === "sk"
        ? `Výrez súhrnného riadku (Spolu za celú obec): ${selectedDistrictRow.houses_total ?? "—"} domov, ${selectedDistrictRow.present_persons_count ?? "—"} osôb`
        : `Summary row crop (Municipality Total): ${selectedDistrictRow.houses_total ?? "—"} houses, ${selectedDistrictRow.present_persons_count ?? "—"} residents`
      : lang === "sk"
      ? `Výrez sčítacieho obvodu č. ${selectedDistrictRow.obvod_number || selectedDistrictRow.row_index}: komisár ${selectedDistrictRow.commissioner_name || "—"} (${selectedDistrictRow.houses_total ?? "—"} domov, ${selectedDistrictRow.present_persons_count ?? "—"} osôb)`
      : `District #${selectedDistrictRow.obvod_number || selectedDistrictRow.row_index} crop: commissioner ${selectedDistrictRow.commissioner_name || "—"} (${selectedDistrictRow.houses_total ?? "—"} houses, ${selectedDistrictRow.present_persons_count ?? "—"} residents)`;
  } else if (isOverviewSheet && selectedHouse) {
    activeCropUrl = selectedHouse.stitched_crop_path;
    const matchRow = resolveOverviewHouseRow(selectedHouse);
    activeRowYStart = selectedHouse.y_start ?? matchRow?.y_start ?? null;
    activeRowYEnd = selectedHouse.y_end ?? matchRow?.y_end ?? null;
    const housePageSide =
      selectedHouse.page_side ||
      (selectedHouse.scan_id && back_scan && selectedHouse.scan_id === back_scan.id
        ? "back"
        : "front");
    const housePageObj = (detail.pages || []).find(
      (p) =>
        (selectedHouse.scan_id && p.scan_id === selectedHouse.scan_id) ||
        p.page_side === housePageSide
    );
    const houseScan =
      housePageSide === "back" && back_scan
        ? back_scan
        : housePageSide === "front" && front_scan
        ? front_scan
        : housePageObj
        ? {
            id: housePageObj.scan_id || 0,
            header_crop_path: housePageObj.header_crop_url,
            header_crop_url: housePageObj.header_crop_url,
            height: housePageObj.height,
            width: housePageObj.width,
          }
        : front_scan;
    activeRowScanUrl =
      housePageSide === "back"
        ? backOriginalUrl
        : housePageSide === "front"
        ? frontOriginalUrl
        : housePageObj?.scan_url || frontOriginalUrl;
    activeHeaderCropUrl =
      selectedHouse.header_crop_url ||
      houseScan?.header_crop_path ||
      houseScan?.header_crop_url ||
      (houseScan?.id ? `/media/crops/scan_${houseScan.id}/header.jpg` : activeHeaderCropUrl);
    activeRowScanHeight = houseScan?.height || null;
    activeRowScanWidth = houseScan?.width || null;
    activeRowLabel =
      lang === "sk"
        ? `Výrez vybraného domu (riadok č. ${
            selectedHouse.ordinal_number || selectedHouse.row_index
          }${selectedHouse.booklet_page ? ` · strana ${selectedHouse.booklet_page}` : ""}): dom č. ${
            selectedHouse.house_number || "—"
          } – ${selectedHouse.owner_name || "neznámy majiteľ"}`
        : `Selected house crop (row #${
            selectedHouse.ordinal_number || selectedHouse.row_index
          }${selectedHouse.booklet_page ? ` · page ${selectedHouse.booklet_page}` : ""}): house #${
            selectedHouse.house_number || "—"
          } – ${selectedHouse.owner_name || "unknown owner"}`;
  } else if (selectedMember) {
    activeCropUrl =
      selectedMemberRow?.stitched_crop_path || selectedMember.stitched_crop_url || null;
    activeRowYStart = selectedMemberRow?.y_start ?? null;
    activeRowYEnd = selectedMemberRow?.y_end ?? null;
    activeHeaderCropUrl =
      activeScan?.header_crop_path ||
      activeScan?.header_crop_url ||
      front_scan?.header_crop_path ||
      front_scan?.header_crop_url ||
      (front_scan?.id ? `/media/crops/scan_${front_scan.id}/header.jpg` : null);
    activeRowLabel =
      lang === "sk"
        ? `Výrez vybraného riadku č. ${
            selectedMember.fact.person_row_number || "—"
          }: ${selectedMember.fact.given_name_raw || ""} ${
            selectedMember.fact.surname_raw || ""
          }`.trim()
        : `Selected row crop #${
            selectedMember.fact.person_row_number || "—"
          }: ${selectedMember.fact.given_name_raw || ""} ${
            selectedMember.fact.surname_raw || ""
          }`.trim();
  }

  return (
    <div className="space-y-4">
      {/* Top Header Bar */}
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <Link
              to={
                sheet.municipality_name
                  ? `/scitacie-harky?obec=${encodeURIComponent(
                      normalizePlaceSpelling(sheet.municipality_name)
                    )}${
                      sheet.district_name
                        ? `&okres=${encodeURIComponent(sheet.district_name)}`
                        : ""
                    }&rok=${sheet.census_year === 1940 ? "1940" : "1930"}`
                  : `/scitacie-harky?rok=${sheet.census_year === 1940 ? "1940" : "1930"}`
              }
              className="p-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-800"
              title={
                sheet.municipality_name
                  ? `Späť na prehľad obce ${normalizePlaceSpelling(
                      sheet.municipality_name
                    )}`
                  : "Späť na zoznam hárkov"
              }
            >
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-mono font-bold">
                  {sheet.slovakiana_id}
                </span>

                {/* Sheet Type Badge */}
                {isMunicipalOverviewSheet ? (
                  <span className="px-2.5 py-0.5 rounded-full bg-indigo-100 text-indigo-950 border border-indigo-300 text-xs font-bold inline-flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-indigo-800" />
                    <span>
                      {lang === "sk"
                        ? "Obecný prehľad (Súhrn obvodov)"
                        : "Municipal Overview (Districts Summary)"}
                    </span>
                  </span>
                ) : isOverviewSheet ? (
                  <span className="px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 border border-purple-300 text-xs font-bold inline-flex items-center gap-1">
                    <Building2 className="w-3.5 h-3.5" />
                    <span>
                      {lang === "sk"
                        ? "Prehľad sčítacieho obvodu (Prehľad domov)"
                        : "District Overview Sheet (Houses)"}
                    </span>
                  </span>
                ) : (
                  <span className="px-2.5 py-0.5 rounded-full bg-stone-100 text-stone-800 border border-stone-300 text-xs font-bold inline-flex items-center gap-1">
                    <Home className="w-3.5 h-3.5 text-amber-700" />
                    <span>
                      {lang === "sk"
                        ? "Sčítací hárok domácnosti"
                        : "Household Census Sheet"}
                    </span>
                  </span>
                )}

                <h2 className="text-lg font-bold text-stone-900">
                  {normalizePlaceSpelling(sheet.municipality_name)}
                  {!isOverviewSheet &&
                  !isMunicipalOverviewSheet &&
                  (sheet.house_number_conscription || sheet.house_number_raw)
                    ? ` – ${lang === "sk" ? "dom č." : "house #"}${
                        sheet.house_number_conscription || sheet.house_number_raw
                      }`
                    : ""}
                  {isOverviewSheet && sheet.obvod_number
                    ? ` – ${lang === "sk" ? "obvod č." : "district #"}${
                        sheet.obvod_number
                      }`
                    : ""}
                </h2>

                {/* Interactive Year Switch (1930 | 1940) to seamlessly transfer to same house/village in new year */}
                <div
                  data-testid="sheet-year-switch"
                  role="group"
                  aria-label={
                    lang === "sk" ? "Prepnúť rok sčítania" : "Switch census year"
                  }
                  className="inline-flex items-center rounded-lg bg-stone-100 p-0.5 border border-stone-300 text-xs"
                >
                  {([1930, 1940] as const).map((yr) => {
                    const active = (sheet.census_year === 1940 ? 1940 : 1930) === yr;
                    return (
                      <button
                        key={yr}
                        type="button"
                        data-testid={`sheet-year-btn-${yr}`}
                        aria-pressed={active}
                        disabled={switchingYear}
                        onClick={() => handleSwitchSheetYear(yr)}
                        title={
                          active
                            ? `${lang === "sk" ? "Aktuálny rok sčítania" : "Current census year"}: ${yr}`
                            : lang === "sk"
                            ? `Prepnúť na rok ${yr} (${normalizePlaceSpelling(
                                sheet.municipality_name
                              )}${
                                !isOverviewSheet &&
                                !isMunicipalOverviewSheet &&
                                (sheet.house_number_conscription || sheet.house_number_raw)
                                  ? `, dom č. ${
                                      sheet.house_number_conscription || sheet.house_number_raw
                                    }`
                                  : ""
                              })`
                            : `Switch to year ${yr} (${normalizePlaceSpelling(
                                sheet.municipality_name
                              )}${
                                !isOverviewSheet &&
                                !isMunicipalOverviewSheet &&
                                (sheet.house_number_conscription || sheet.house_number_raw)
                                  ? `, house #${
                                      sheet.house_number_conscription || sheet.house_number_raw
                                    }`
                                  : ""
                              })`
                        }
                        className={`px-2.5 py-1 min-h-[28px] rounded-md font-bold transition cursor-pointer disabled:opacity-50 ${
                          active
                            ? "bg-stone-900 text-amber-300 shadow-2xs"
                            : "text-stone-700 hover:bg-stone-200/80"
                        }`}
                      >
                        {yr}
                      </button>
                    );
                  })}
                </div>
                {switchingYear && (
                  <span className="px-2 py-0.5 rounded-full bg-amber-50 border border-amber-300 text-amber-800 text-xs font-semibold inline-flex items-center gap-1">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    <span>{lang === "sk" ? "Prepínam rok…" : "Switching year…"}</span>
                  </span>
                )}

                <span
                  className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                    sheet.census_year === 1930
                      ? "bg-emerald-100 text-emerald-900"
                      : "bg-amber-100 text-amber-900"
                  }`}
                >
                  {lang === "sk" ? "Sčítanie" : "Census"} {sheet.census_year}
                  {sheet.is_blackened
                    ? lang === "sk"
                      ? " (čiastočne začiernený)"
                      : " (partially redacted)"
                    : lang === "sk"
                    ? " (plný hárok)"
                    : " (full sheet)"}
                </span>

                {/* Cross-Year Connected House Sheet Badges (1930 <-> 1940) */}
                {!isOverviewSheet &&
                  !isMunicipalOverviewSheet &&
                  linkedHouseSheets.map((linked) => {
                    const targetRef =
                      linked.cair_id || linked.slovakiana_key || String(linked.id);
                    const summaryParts: string[] = [];
                    if (linked.head_of_household) {
                      summaryParts.push(linked.head_of_household);
                    }
                    if (linked.persons_count > 0) {
                      summaryParts.push(
                        `${linked.persons_count} ${
                          linked.persons_count === 1
                            ? "osoba"
                            : linked.persons_count >= 2 && linked.persons_count <= 4
                            ? "osoby"
                            : "osôb"
                        }`
                      );
                    }
                    return (
                      <Link
                        key={`${linked.id}-${linked.cair_id}`}
                        to={`/scitacie-harky/${targetRef}`}
                        className="px-2.5 py-0.5 rounded-full bg-indigo-100 hover:bg-indigo-200 text-indigo-950 border border-indigo-300 text-xs font-bold inline-flex items-center gap-1 transition"
                        title={`Prejsť na prepojený sčítací hárok pre dom č. ${
                          linked.house_number || sheet.house_number_raw || "—"
                        } z roku ${linked.census_year} (${targetRef})`}
                      >
                        <span>
                          🔗 Sčítanie {linked.census_year} · Dom č.{" "}
                          {linked.house_number || sheet.house_number_raw || "—"}
                          {linkedHouseSheets.length > 1 && linked.sheet_number
                            ? ` · hárok č. ${linked.sheet_number}`
                            : ""}
                          {summaryParts.length > 0
                            ? ` (${summaryParts.join(", ")})`
                            : ""}{" "}
                          →
                        </span>
                      </Link>
                    );
                  })}
              </div>

              <div className="text-xs sm:text-sm text-stone-700 mt-1 flex flex-wrap items-center gap-2">
                <span>
                  {lang === "sk" ? "Okres:" : "District:"} {sheet.district_name || "—"}
                </span>
                {sheet.street_name && (
                  <span>
                    · {lang === "sk" ? "Ulica/osada:" : "Street/settlement:"} {sheet.street_name}
                  </span>
                )}
                {sheet.box_number && (
                  <span>
                    · {lang === "sk" ? "Škatuľa:" : "Box:"} {sheet.box_number}
                  </span>
                )}
                {sheet.sheet_number && (
                  <span>
                    · {lang === "sk" ? "Hárok č.:" : "Sheet #:"} {sheet.sheet_number}
                  </span>
                )}
                {sheet.obvod_number && (
                  <span>
                    · {lang === "sk" ? "Obvod č.:" : "District #:"} {sheet.obvod_number}
                  </span>
                )}
                {sheet.commissioner_name && (
                  <span>
                    ·{" "}
                    {isMunicipalOverviewSheet
                      ? lang === "sk"
                        ? "Revízor:"
                        : "Revisor:"
                      : lang === "sk"
                      ? "Komisár:"
                      : "Commissioner:"}{" "}
                    {sheet.commissioner_name}
                  </span>
                )}
                {!isOverviewSheet &&
                  !isMunicipalOverviewSheet &&
                  linked_overview_house && (
                    <span>
                      · {lang === "sk" ? "Majiteľ domu:" : "House owner:"}{" "}
                      <strong className="text-stone-900">
                        {linked_overview_house.owner_name || "—"}
                      </strong>
                      {(linked_overview_house.overview_slovakiana_key ||
                        linked_overview_house.census_sheet_id) && (
                        <>
                          {" "}
                          (
                          <Link
                            to={`/scitacie-harky/${
                              linked_overview_house.overview_slovakiana_key ||
                              linked_overview_house.census_sheet_id
                            }`}
                            className="text-amber-700 hover:underline font-semibold"
                          >
                            {lang === "sk" ? "Prehľad obvodu ↗" : "District overview ↗"}
                          </Link>
                          )
                        </>
                      )}
                    </span>
                  )}
                {isOverviewSheet && linked_municipal_overview && (
                  <span>
                    ·{" "}
                    <Link
                      to={`/scitacie-harky/${linked_municipal_overview.slovakiana_id}`}
                      className="text-amber-700 hover:underline font-semibold"
                    >
                      {lang === "sk"
                        ? `Obecný prehľad (${linked_municipal_overview.slovakiana_id}) ↗`
                        : `Municipal overview (${linked_municipal_overview.slovakiana_id}) ↗`}
                    </Link>
                  </span>
                )}
                {!isOverviewSheet &&
                  !isMunicipalOverviewSheet &&
                  sameYearHouseSheets.length > 0 && (
                    <span className="inline-flex flex-wrap items-center gap-1">
                      · {lang === "sk" ? "Ďalšie byty v dome:" : "Other units in house:"}
                      {sameYearHouseSheets.map((sib) => {
                        const sibRef = sib.cair_id || sib.slovakiana_key || String(sib.id);
                        return (
                          <Link
                            key={`${sib.id}-${sibRef}`}
                            to={`/scitacie-harky/${sibRef}`}
                            className="px-2 py-0.5 rounded bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-300 text-xs font-semibold"
                          >
                            {sib.head_of_household || sibRef}
                          </Link>
                        );
                      })}
                    </span>
                  )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Manual Type Switch Selector (Admin Debug Overlay only) */}
            {featureFlags.admin_debug_overlay && featureFlags.census_overview_sheets && (
              <div className="inline-flex items-center gap-1.5">
                <label className="text-xs font-semibold text-stone-700">
                  {lang === "sk" ? "Typ hárku:" : "Sheet type:"}
                </label>
                <select
                  data-testid="sheet-type-select"
                  value={
                    isMunicipalOverviewSheet
                      ? "municipal_overview"
                      : isOverviewSheet
                      ? "overview"
                      : "household"
                  }
                  disabled={switchingType || processing}
                  onChange={(e) =>
                    handleSetSheetType(
                      e.target.value as "household" | "overview" | "municipal_overview"
                    )
                  }
                  className="px-2.5 py-1.5 min-h-[34px] rounded-lg border border-stone-300 bg-stone-50 hover:bg-stone-100 text-stone-900 text-xs font-semibold cursor-pointer disabled:opacity-50"
                >
                  <option value="household">
                    {lang === "sk"
                      ? "Sčítací hárok domácnosti"
                      : "Household Census Sheet"}
                  </option>
                  <option value="overview">
                    {lang === "sk"
                      ? "Prehľad obvodu (Prehľad domov)"
                      : "District Overview (Houses)"}
                  </option>
                  <option value="municipal_overview">
                    {lang === "sk"
                      ? "Obecný prehľad (Súhrn obvodov)"
                      : "Municipal Overview (Districts Summary)"}
                  </option>
                </select>
              </div>
            )}

            <button
              type="button"
              data-testid="toggle-full-sheet-scan-btn"
              onClick={() => setShowFullSheetScan((prev) => !prev)}
              className={`px-3 py-1.5 min-h-[34px] rounded-lg border text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer transition ${
                showFullSheetScan
                  ? "bg-stone-900 text-white border-stone-900"
                  : "bg-white hover:bg-stone-100 text-stone-800 border-stone-300"
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>
                {showFullSheetScan
                  ? lang === "sk"
                    ? "Skryť celý sken hárku"
                    : "Hide Full Sheet Scan"
                  : lang === "sk"
                  ? "Zobraziť celý sken hárku"
                  : "Show Full Sheet Scan"}
              </span>
            </button>

            <a
              href={
                sheet.source_web_url ||
                `https://slovakiana.sk/kulturne-objekty/${sheet.slovakiana_id}`
              }
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1.5 min-h-[34px] rounded-lg border border-stone-300 hover:bg-stone-100 text-xs font-semibold flex items-center gap-1.5"
            >
              <span>
                {lang === "sk" ? "Slovakiana originál" : "Slovakiana original"}
              </span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button
              onClick={handleDownloadAndProcess}
              disabled={processing || switchingType}
              data-testid="sheet-rerun-ocr-btn"
              className="px-3.5 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            >
              {processing ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Download className="w-4 h-4" />
              )}
              <span>
                {processing
                  ? lang === "sk"
                    ? `Prebieha AI OCR & párovanie (${Math.max(8, sheetProgress.progress_pct || 8)}%)…`
                    : `Running AI OCR & matching (${Math.max(8, sheetProgress.progress_pct || 8)}%)…`
                  : sheet.download_status === "downloaded" ||
                    sheet.download_status === "extracted" ||
                    sheet.download_status === "matched"
                  ? lang === "sk"
                    ? "Znova spustiť AI OCR & párovanie"
                    : "Re-run AI OCR & matching"
                  : lang === "sk"
                  ? "Stiahnuť skeny & Spustiť AI OCR"
                  : "Download scans & Run AI OCR"}
              </span>
            </button>
          </div>
        </div>

        {/* Live AI OCR & Matching Progress Bar Card */}
        {(processing ||
          switchingType ||
          sheetProgress.active ||
          sheetProgress.completedJustNow) &&
          (() => {
            const isRunning = processing || switchingType || sheetProgress.active;
            const pct = isRunning
              ? Math.max(8, Math.min(99, sheetProgress.progress_pct || 8))
              : 100;
            const stageOrder = [
              "census_download",
              "layout",
              "ocr",
              "extract",
              "mcp_match",
            ];
            const currentStageIdx = isRunning
              ? Math.max(0, stageOrder.indexOf(sheetProgress.active_stage))
              : stageOrder.length;
            const stages = [
              {
                key: "census_download",
                labelSk: "1. Skeny",
                labelEn: "1. Scans",
              },
              {
                key: "layout",
                labelSk: "2. Rozdelenie riadkov",
                labelEn: "2. Row slicing",
              },
              {
                key: "ocr",
                labelSk: "3. AI OCR rukopisu",
                labelEn: "3. AI OCR",
              },
              {
                key: "extract",
                labelSk: "4. Extrakcia údajov",
                labelEn: "4. Extraction",
              },
              {
                key: "mcp_match",
                labelSk: "5. Párovanie (Matrika & FS)",
                labelEn: "5. Matching",
              },
            ];

            return (
              <div
                data-testid="sheet-ocr-progress-card"
                className={`rounded-lg px-3.5 py-2.5 border space-y-2 transition-colors ${
                  isRunning
                    ? "bg-amber-50 border-amber-400 shadow-xs"
                    : "bg-emerald-50/90 border-emerald-300"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-2 font-bold text-stone-900 min-w-0">
                    {isRunning ? (
                      <RefreshCw className="w-4 h-4 text-amber-700 animate-spin shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                    )}
                    <span className="truncate">
                      {lang === "sk"
                        ? sheetProgress.message_sk ||
                          "Prebieha AI OCR čítanie rukopisu a párovanie hárku…"
                        : sheetProgress.message_en ||
                          "Running handwriting AI OCR and sheet matching…"}
                    </span>
                    {featureFlags.admin_debug_overlay &&
                      isRunning &&
                      sheetProgress.active_model && (
                        <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-[11px] shrink-0">
                          {sheetProgress.active_model}
                        </span>
                      )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {isRunning &&
                      sheetProgress.total_rows != null &&
                      sheetProgress.total_rows > 0 && (
                        <span className="px-2 py-0.5 rounded bg-white border border-amber-300 text-amber-950 font-bold text-[11px]">
                          {lang === "sk"
                            ? `Riadok ${sheetProgress.current_row_index || 1}/${sheetProgress.total_rows}`
                            : `Row ${sheetProgress.current_row_index || 1}/${sheetProgress.total_rows}`}
                        </span>
                      )}
                    {sheetProgress.elapsed_sec > 0 && (
                      <span className="text-stone-600 font-medium">
                        {sheetProgress.elapsed_sec} s
                      </span>
                    )}
                    <span
                      className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                        isRunning
                          ? "bg-amber-600 text-white"
                          : "bg-emerald-700 text-white"
                      }`}
                    >
                      {pct}%
                    </span>
                    {!isRunning && (
                      <button
                        type="button"
                        onClick={() =>
                          setSheetProgress((prev) => ({
                            ...prev,
                            completedJustNow: false,
                          }))
                        }
                        className="p-0.5 rounded hover:bg-emerald-200/70 text-stone-600 cursor-pointer"
                        title={lang === "sk" ? "Skryť" : "Dismiss"}
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                <div
                  role="progressbar"
                  aria-label={
                    lang === "sk"
                      ? "Priebeh AI OCR a párovania hárku"
                      : "Sheet AI OCR and matching progress"
                  }
                  aria-valuenow={pct}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  data-testid="sheet-ocr-progress-bar"
                  className={`w-full rounded-full h-2.5 overflow-hidden ${
                    isRunning ? "bg-amber-200/80" : "bg-emerald-200"
                  }`}
                >
                  <div
                    className={`h-full transition-all duration-300 ${
                      isRunning ? "bg-amber-600 animate-pulse" : "bg-emerald-600"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>

                <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                  {stages.map((st, idx) => {
                    const isDone = !isRunning || idx < currentStageIdx;
                    const isCurrent = isRunning && idx === currentStageIdx;
                    return (
                      <span
                        key={st.key}
                        className={`px-2 py-0.5 rounded-md text-[11px] font-semibold inline-flex items-center gap-1 ${
                          isCurrent
                            ? "bg-amber-600 text-white shadow-2xs"
                            : isDone
                            ? "bg-emerald-100 text-emerald-900 border border-emerald-300"
                            : "bg-stone-100 text-stone-500 border border-stone-200"
                        }`}
                      >
                        <span>{lang === "sk" ? st.labelSk : st.labelEn}</span>
                      </span>
                    );
                  })}
                </div>
              </div>
            );
          })()}

        {actionError && (
          <div className="bg-red-50 border border-red-400 rounded-lg p-3 flex items-center gap-2 text-red-900 font-medium text-xs">
            <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
            <span>{actionError}</span>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {/* SECONDARY: Full-Sheet Scan Viewer (Shown only on user request or when Admin Debug Overlay is enabled) */}
        {(showFullSheetScan || featureFlags.admin_debug_overlay) && (
          <div
            data-testid="full-sheet-scan-card"
            className="bg-white rounded-xl p-4 border border-stone-300 space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2.5">
                <h3 className="text-base font-bold text-stone-900">
                  {lang === "sk" ? "Zobrazenie hárku" : "Sheet Scan Viewer"}
                </h3>

                {/* Side Tabs: Predná strana vs Zadná strana + any extra pages */}
                <div className="inline-flex rounded-lg bg-stone-100 p-1 border border-stone-200">
                  <button
                    type="button"
                    onClick={() => setActiveSide("front")}
                    className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                      activeSide === "front"
                        ? "bg-stone-900 text-white shadow-xs"
                        : "text-stone-700 hover:text-stone-900"
                    }`}
                  >
                    {featureFlags.admin_debug_overlay
                      ? lang === "sk"
                        ? "Predná strana (_v)"
                        : "Front Side (_v)"
                      : lang === "sk"
                      ? "Predná strana"
                      : "Front Side"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setActiveSide("back")}
                    className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                      activeSide === "back"
                        ? "bg-stone-900 text-white shadow-xs"
                        : "text-stone-700 hover:text-stone-900"
                    }`}
                  >
                    {featureFlags.admin_debug_overlay
                      ? lang === "sk"
                        ? "Zadná strana (_r)"
                        : "Back Side (_r)"
                      : lang === "sk"
                      ? "Zadná strana"
                      : "Back Side"}
                  </button>
                  {extraPages.map((ep) => (
                    <button
                      key={ep.page_side}
                      type="button"
                      onClick={() => setActiveSide(ep.page_side)}
                      className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                        activeSide === ep.page_side
                          ? "bg-stone-900 text-white shadow-xs"
                          : "text-stone-700 hover:text-stone-900"
                      }`}
                    >
                      {ep.label || `${lang === "sk" ? "Strana" : "Page"} ${ep.page_number}`}
                    </button>
                  ))}
                </div>

                {/* Mode Tabs: Originálny sken vs Detekované riadky (Overlay) - Admin Overlay only */}
                {featureFlags.admin_debug_overlay && (
                  <div className="inline-flex rounded-lg bg-stone-100 p-1 border border-stone-200">
                    <button
                      type="button"
                      onClick={() => setActiveScanMode("original")}
                      className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                        activeScanMode === "original"
                          ? "bg-amber-600 text-white shadow-xs"
                          : "text-stone-700 hover:text-stone-900"
                      }`}
                    >
                      {lang === "sk" ? "Originálny sken" : "Original Scan"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveScanMode("overlay")}
                      className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold cursor-pointer transition ${
                        activeScanMode === "overlay"
                          ? "bg-amber-600 text-white shadow-xs"
                          : "text-stone-700 hover:text-stone-900"
                      }`}
                    >
                      {lang === "sk" ? "Detekované riadky (Overlay)" : "Detected Rows (Overlay)"}
                    </button>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={() => setShowFullSheetScan(false)}
                className="px-2.5 py-1 min-h-[30px] rounded-lg bg-stone-100 hover:bg-stone-200 border border-stone-300 text-xs font-semibold text-stone-700 inline-flex items-center gap-1 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
                <span>{lang === "sk" ? "Zavrieť celý sken" : "Close full scan"}</span>
              </button>
            </div>

            <ZoomableScanViewer
              src={activeViewerSrc}
              alt={`${sheet.slovakiana_id} (${activeSide})`}
              title={`${sheet.slovakiana_id} · ${
                activeSide === "front"
                  ? lang === "sk"
                    ? "Predná strana"
                    : "Front Side"
                  : lang === "sk"
                  ? "Zadná strana"
                  : "Back Side"
              }`}
              subtitle={
                activeScanMode === "overlay"
                  ? lang === "sk"
                    ? "Detekované riadky (Overlay)"
                    : "Detected Rows (Overlay)"
                  : lang === "sk"
                  ? "Originálny sken (kliknite na riadok pre výber)"
                  : "Original Scan (click a row to select)"
              }
              defaultHeightClass="max-h-[420px]"
              expandedHeightClass="max-h-[85vh]"
              emptyMessage={
                lang === "sk"
                  ? "Sken zatiaľ nebol stiahnutý do lokálneho úložiska. Kliknite na „Stiahnuť skeny & Spustiť AI OCR“."
                  : "Scan has not yet been downloaded locally. Click 'Download scans & Run AI OCR'."
              }
              lang={lang}
            >
              {/* Interactive row bounding-box overlay when in original scan mode */}
              {activeScanMode === "original" &&
                activeSide === "front" &&
                front_scan?.height &&
                front_scan.height > 0 &&
                !isOverviewSheet &&
                !isMunicipalOverviewSheet &&
                members.map((m) => {
                  const r = m.row || rowsById.get(m.fact.row_id);
                  if (!r || r.y_end <= r.y_start) return null;
                  const topPct = (r.y_start / front_scan.height!) * 100;
                  const heightPct = ((r.y_end - r.y_start) / front_scan.height!) * 100;
                  const isSel = selectedMember?.fact.id === m.fact.id;
                  return (
                    <div
                      key={m.fact.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedMemberId(m.fact.id);
                      }}
                      title={`Riadok č. ${m.fact.person_row_number || r.row_index}: ${
                        m.fact.given_name_raw || ""
                      } ${m.fact.surname_raw || ""}`}
                      style={{
                        top: `${topPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className={`absolute left-1 right-1 rounded transition pointer-events-auto cursor-pointer ${
                        isSel
                          ? "border-2 border-amber-500 bg-amber-500/15"
                          : "border border-transparent hover:border-amber-400/80 hover:bg-amber-400/10"
                      }`}
                    />
                  );
                })}
            </ZoomableScanViewer>
          </div>
        )}

        {/* PRIMARY CROP-FIRST VERIFICATION BAR: Pinned Field/Row Crop right above the data table */}
        {activeCropUrl && (
          <HoverZoomRowCrop
            cropUrl={activeCropUrl}
            headerCropUrl={activeHeaderCropUrl}
            scanUrl={featureFlags.hover_row_zoom ? activeRowScanUrl : null}
            yStart={featureFlags.hover_row_zoom ? activeRowYStart : null}
            yEnd={featureFlags.hover_row_zoom ? activeRowYEnd : null}
            scanHeight={activeRowScanHeight}
            scanWidth={activeRowScanWidth}
            rowLabel={activeRowLabel}
            activeFieldFocus={activeFieldFocus}
            onSelectFieldFocus={setActiveFieldFocus}
            showFieldPills={!isOverviewSheet && !isMunicipalOverviewSheet}
            lang={lang}
          />
        )}

      <div className="space-y-4 min-w-0">
      {/* Global datalist for Pomer k hlave domácnosti */}
      <datalist id="relation-to-head-options">
        {ALL_RELATION_TO_HEAD_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </datalist>

      {/* SECTION 6: Municipal Overview ('Obecný prehľad') OR District Overview ('Prehľad domov') OR Household Table */}
      {isMunicipalOverviewSheet ? (
        <div className="space-y-4">
          <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-3">
              <div>
                <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
                  <Layers className="w-5 h-5 text-indigo-700" />
                  <span>
                    {lang === "sk"
                      ? "Obecný prehľad – Súhrn sčítacích obvodov (Prehľadov domov)"
                      : "Municipal Overview – Summary of Census Districts (House Overviews)"}
                  </span>
                </h3>
                <p className="text-xs text-stone-600 mt-0.5">
                  {lang === "sk"
                    ? "Obecný prehľad (Tlačivo X) spája a sumarizuje jednotlivé Prehľady sčítacích obvodov (prehľady domov) v obciach rozdelených na viacero obvodov."
                    : "The Municipal Overview (Form X) joins and summarizes individual District Overview (house directory) sheets across all census districts of the municipality."}
                </p>
              </div>

              {(() => {
                const districtOnly = municipal_overview_districts.filter((d) => !d.is_summary_row);
                const summaryRow =
                  municipal_overview_districts.find((d) => d.is_summary_row) || null;
                const totHouses =
                  summaryRow?.houses_total ??
                  districtOnly.reduce((acc, d) => acc + (d.houses_total || 0), 0);
                const totInhab =
                  summaryRow?.houses_inhabited ??
                  districtOnly.reduce((acc, d) => acc + (d.houses_inhabited || 0), 0);
                const totUninhab =
                  summaryRow?.houses_uninhabited ??
                  districtOnly.reduce((acc, d) => acc + (d.houses_uninhabited || 0), 0);
                const totUnits =
                  summaryRow?.household_count ??
                  districtOnly.reduce((acc, d) => acc + (d.household_count || 0), 0);
                const totPersons =
                  summaryRow?.present_persons_count ??
                  districtOnly.reduce((acc, d) => acc + (d.present_persons_count || 0), 0);
                return (
                  <div className="flex flex-wrap items-center gap-2 text-xs sm:text-sm">
                    <div className="px-3 py-1.5 rounded-lg bg-stone-100 border border-stone-200">
                      <span className="text-stone-600">
                        {lang === "sk" ? "Obec / Okres: " : "Municipality / District: "}
                      </span>
                      <strong className="text-stone-900">
                        {sheet.municipality_name} ({sheet.district_name || "—"})
                      </strong>
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-indigo-50 border border-indigo-200">
                      <span className="text-indigo-800">
                        {lang === "sk" ? "Sčítací revízor: " : "Census Revisor: "}
                      </span>
                      <strong className="text-indigo-950">
                        {sheet.commissioner_name || "—"}
                      </strong>
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-purple-50 border border-purple-200">
                      <span className="text-purple-800">
                        {lang === "sk" ? "Sčítacie obvody: " : "Census Districts: "}
                      </span>
                      <strong className="text-purple-950">{districtOnly.length}</strong>
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200">
                      <span className="text-emerald-900">
                        {lang === "sk" ? "Domov spolu: " : "Total Houses: "}
                      </span>
                      <strong className="text-emerald-950">
                        {totHouses} ({totInhab} obýv. / {totUninhab} neobýv.)
                      </strong>
                    </div>
                    <div className="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200">
                      <span className="text-amber-900">
                        {lang === "sk"
                          ? "Bytové strany / Osôb spolu: "
                          : "Units / Total Residents: "}
                      </span>
                      <strong className="text-amber-950">
                        {totUnits} / {totPersons}
                      </strong>
                    </div>
                  </div>
                );
              })()}
            </div>

            {municipal_overview_districts.length === 0 ? (
              <div className="text-center py-8 text-stone-600 text-sm">
                {lang === "sk" ? (
                  <>
                    Zatiaľ neboli z tohto obecného prehľadu extrahované žiadne sčítacie obvody. Kliknite vyššie na{" "}
                    <strong>„Znova spustiť AI OCR & párovanie“</strong>.
                  </>
                ) : (
                  <>
                    No census districts have been extracted from this municipal overview yet. Click{" "}
                    <strong>"Re-run AI OCR & matching"</strong> above.
                  </>
                )}
              </div>
            ) : (
              <div className="w-full overflow-x-auto">
                <table className="w-full table-auto text-left border-collapse text-xs sm:text-sm">
                  <thead>
                    <tr className="border-b border-stone-300 bg-stone-100 text-stone-800 font-bold">
                      <th className="py-2.5 px-2">
                        {lang === "sk" ? "Osada" : "Settlement"}
                      </th>
                      <th className="py-2.5 px-2">
                        {lang === "sk" ? "Obvod č." : "District #"}
                      </th>
                      <th className="py-2.5 px-2">
                        {lang === "sk" ? "Meno sčítacieho komisára" : "Census Commissioner"}
                      </th>
                      <th className="py-2.5 px-2 text-center">
                        {lang === "sk" ? "Domov úhrnom" : "Houses Total"}
                      </th>
                      <th className="py-2.5 px-2 text-center">
                        {lang === "sk" ? "Obývané / Neobývané" : "Inhab. / Uninhab."}
                      </th>
                      <th className="py-2.5 px-2 text-center">
                        {lang === "sk" ? "Bytové strany" : "Dwelling Units"}
                      </th>
                      <th className="py-2.5 px-2 text-center">
                        {lang === "sk" ? "Prítomné osoby" : "Present Persons"}
                      </th>
                      <th className="py-2.5 px-2">
                        {lang === "sk"
                          ? "Spojený prehľad domov (obvodu)"
                          : "Joined House Overview (District)"}
                      </th>
                      <th className="py-2.5 px-2 text-right">
                        {lang === "sk" ? "Akcia" : "Action"}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-stone-200">
                    {municipal_overview_districts.map((d) => {
                      const isSelected = selectedDistrictRow?.id === d.id;
                      const isEditing = editingDistrictRowId === d.id;
                      const linkedOvRef =
                        d.matched_overview_slovakiana_key ||
                        d.matched_overview_sheet?.slovakiana_id ||
                        (d.matched_overview_sheet_id
                          ? String(d.matched_overview_sheet_id)
                          : null);
                      return (
                        <React.Fragment key={d.id}>
                          <tr
                            onClick={() => setSelectedDistrictRowId(d.id)}
                            className={`cursor-pointer transition ${
                              d.is_summary_row
                                ? isSelected
                                  ? "bg-indigo-100/90 font-bold"
                                  : "bg-stone-100/90 font-bold hover:bg-indigo-50/70"
                                : isSelected
                                ? "bg-amber-50/80"
                                : "hover:bg-stone-50"
                            }`}
                          >
                            <td className="py-2.5 px-2 align-middle text-stone-800 whitespace-nowrap">
                              {d.is_summary_row ? (
                                <span className="px-2 py-0.5 rounded bg-indigo-900 text-white text-xs font-sans font-bold">
                                  {lang === "sk" ? "Spolu" : "Total"}
                                </span>
                              ) : (
                                <span>
                                  <span className="font-mono font-bold text-stone-600 mr-1">
                                    {d.ordinal_number ?? 1}.
                                  </span>
                                  <span>
                                    {normalizePlaceSpelling(d.settlement_name) ||
                                      normalizePlaceSpelling(sheet.municipality_name) ||
                                      "—"}
                                  </span>
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 px-2 align-middle whitespace-nowrap">
                              {d.is_summary_row ? (
                                <span className="text-xs text-stone-600">
                                  {lang === "sk" ? "Celá obec" : "All districts"}
                                </span>
                              ) : (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-950 border border-purple-300 text-xs font-bold">
                                  {lang === "sk" ? "Obvod č." : "District #"}{" "}
                                  {d.obvod_number || d.row_index}
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 px-2 align-middle font-semibold text-stone-900">
                              {d.is_summary_row
                                ? lang === "sk"
                                  ? "Súčet za celú obec"
                                  : "Municipality Total"
                                : d.commissioner_name || "—"}
                            </td>
                            <td className="py-2.5 px-2 align-middle text-center font-bold text-stone-900">
                              {d.houses_total ?? "—"}
                            </td>
                            <td className="py-2.5 px-2 align-middle text-center whitespace-nowrap">
                              <span className="text-emerald-800 font-semibold">
                                {d.houses_inhabited ?? "—"}
                              </span>
                              <span className="text-stone-400 mx-1">/</span>
                              <span className="text-stone-600">
                                {d.houses_uninhabited ?? 0}
                              </span>
                            </td>
                            <td className="py-2.5 px-2 align-middle text-center font-semibold text-stone-800">
                              {d.household_count ?? "—"}
                            </td>
                            <td className="py-2.5 px-2 align-middle text-center font-bold text-amber-900">
                              {d.present_persons_count ?? "—"}
                            </td>
                            <td className="py-2.5 px-2 align-middle">
                              {d.is_summary_row ? (
                                <span className="text-xs text-indigo-900 font-semibold">
                                  {linked_overview_sheets.length}{" "}
                                  {lang === "sk"
                                    ? "spojené prehľady domov"
                                    : "joined house overviews"}
                                </span>
                              ) : linkedOvRef ? (
                                <Link
                                  to={`/scitacie-harky/${linkedOvRef}`}
                                  onClick={(e) => e.stopPropagation()}
                                  className="px-2.5 py-1 min-h-[32px] rounded-md bg-purple-100 hover:bg-purple-200 text-purple-950 border border-purple-300 text-xs font-semibold inline-flex items-center gap-1.5 whitespace-nowrap"
                                >
                                  <Building2 className="w-3.5 h-3.5 text-purple-700 shrink-0" />
                                  <span>
                                    {d.matched_overview_slovakiana_key || linkedOvRef}
                                  </span>
                                  {typeof d.matched_overview_houses_count === "number" &&
                                    d.matched_overview_houses_count > 0 && (
                                      <span className="px-1.5 py-0.5 rounded bg-purple-200 text-purple-950 font-bold">
                                        {d.matched_overview_houses_count}{" "}
                                        {lang === "sk" ? "domov" : "houses"}
                                      </span>
                                    )}
                                </Link>
                              ) : (
                                <span className="text-stone-500 text-xs">
                                  {lang === "sk" ? "Nespárované" : "Unpaired"}
                                </span>
                              )}
                            </td>
                            <td className="py-2.5 px-2 align-middle text-right whitespace-nowrap">
                              <div className="inline-flex items-center gap-1.5">
                                {featureFlags.admin_debug_overlay && (
                                  <OcrFeedbackControls
                                    lang={lang}
                                    targetType="municipal_district"
                                    municipalDistrictId={d.id}
                                    rowId={d.row_id ?? null}
                                    scanId={d.scan_id ?? front_scan?.id ?? null}
                                    censusSheetId={sheet.id}
                                    initialFeedback={d.ocr_feedback}
                                    resetVersion={feedbackResetVersion}
                                    onFeedbackSaved={(fb) => {
                                      setDetail((prev) =>
                                        prev
                                          ? {
                                              ...prev,
                                              municipal_overview_districts: (
                                                prev.municipal_overview_districts || []
                                              ).map((item) =>
                                                item.id === d.id
                                                  ? { ...item, ocr_feedback: fb }
                                                  : item
                                              ),
                                            }
                                          : prev
                                      );
                                    }}
                                  />
                                )}
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    if (isEditing) {
                                      setEditingDistrictRowId(null);
                                    } else {
                                      startEditDistrictRow(d);
                                    }
                                  }}
                                  className="px-3 py-1.5 min-h-[32px] rounded-md bg-stone-200 hover:bg-stone-300 text-stone-800 text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap cursor-pointer"
                                >
                                  <Edit3 className="w-3.5 h-3.5 shrink-0" />
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
                                {confirmDeleteDistrictRowId === d.id ? (
                                  <div
                                    onClick={(e) => e.stopPropagation()}
                                    className="inline-flex items-center gap-1.5 bg-red-50 border border-red-300 rounded-md px-2.5 py-1 min-h-[32px]"
                                  >
                                    <button
                                      type="button"
                                      disabled={deletingDistrictRowId === d.id}
                                      onClick={() => handleDeleteDistrictRow(d.id)}
                                      className="text-xs font-bold text-red-700 hover:underline cursor-pointer"
                                    >
                                      {deletingDistrictRowId === d.id
                                        ? "…"
                                        : lang === "sk"
                                        ? "Vymazať"
                                        : "Delete"}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => setConfirmDeleteDistrictRowId(null)}
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
                                      setConfirmDeleteDistrictRowId(d.id);
                                    }}
                                    className="p-2 min-h-[32px] min-w-[32px] inline-flex items-center justify-center rounded-md bg-stone-200 hover:bg-red-100 text-stone-700 hover:text-red-700 cursor-pointer"
                                  >
                                    <Trash2 className="w-3.5 h-3.5" />
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                          {isEditing && (
                            <tr>
                              <td
                                colSpan={10}
                                className="bg-amber-50/70 border-b border-amber-200 p-4"
                              >
                                <div className="space-y-3">
                                  <div className="flex flex-wrap items-center justify-between gap-2">
                                    <h4 className="text-xs font-bold uppercase tracking-wider text-amber-950">
                                      {lang === "sk"
                                        ? `Úprava záznamu obvodu v obecnom prehľade (riadok #${d.row_index})`
                                        : `Edit Municipal Overview District Row (#${d.row_index})`}
                                    </h4>
                                    <div className="flex items-center gap-2">
                                      <button
                                        type="button"
                                        onClick={() => handleSaveDistrictRow(d.id)}
                                        disabled={savingDistrictRowId === d.id}
                                        className="px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 text-white font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                      >
                                        {savingDistrictRowId === d.id ? (
                                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                        ) : (
                                          <Save className="w-3.5 h-3.5" />
                                        )}
                                        <span>
                                          {savingDistrictRowId === d.id
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
                                        onClick={() => setEditingDistrictRowId(null)}
                                        className="px-3 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                      >
                                        <X className="w-3.5 h-3.5" />
                                        <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                                      </button>
                                    </div>
                                  </div>
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Číslo obvodu (stĺpec 3)
                                      </label>
                                      <input
                                        type="text"
                                        value={districtEditForm.obvod_number}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            obvod_number: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Meno komisára (stĺpec 4)
                                      </label>
                                      <input
                                        type="text"
                                        value={districtEditForm.commissioner_name}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            commissioner_name: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Domov úhrnom (stĺpec 5)
                                      </label>
                                      <input
                                        type="number"
                                        value={districtEditForm.houses_total}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            houses_total: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Obývané domy (stĺpec 6)
                                      </label>
                                      <input
                                        type="number"
                                        value={districtEditForm.houses_inhabited}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            houses_inhabited: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Neobývané domy (stĺpec 7)
                                      </label>
                                      <input
                                        type="number"
                                        value={districtEditForm.houses_uninhabited}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            houses_uninhabited: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Bytové strany (stĺpec 8)
                                      </label>
                                      <input
                                        type="number"
                                        value={districtEditForm.household_count}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            household_count: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Prítomné osoby (stĺpec 9)
                                      </label>
                                      <input
                                        type="number"
                                        value={districtEditForm.present_persons_count}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            present_persons_count: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-1">
                                        Poznámka (stĺpec 10)
                                      </label>
                                      <input
                                        type="text"
                                        value={districtEditForm.remarks}
                                        onChange={(e) =>
                                          setDistrictEditForm({
                                            ...districtEditForm,
                                            remarks: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                      />
                                    </div>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Compact Link Bar to District Overview Sheets & Primary Village Directory (no duplicate house table) */}
          {linked_overview_sheets.length > 0 && (
            <div className="bg-white rounded-xl p-3.5 border border-stone-200 flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-bold text-stone-800">
                  {lang === "sk"
                    ? `Spojené prehľady obvodov (${linked_overview_sheets.length}):`
                    : `Joined district overviews (${linked_overview_sheets.length}):`}
                </span>
                {linked_overview_sheets.map((ov) => (
                  <Link
                    key={ov.slovakiana_id}
                    to={`/scitacie-harky/${ov.slovakiana_id}`}
                    className="px-2.5 py-1 rounded-md bg-stone-100 hover:bg-stone-200 text-stone-900 border border-stone-300 font-semibold inline-flex items-center gap-1.5"
                  >
                    <Building2 className="w-3.5 h-3.5 text-amber-600" />
                    <span>
                      {lang === "sk" ? "Obvod" : "District"}{" "}
                      {ov.obvod_number ? `č. ${ov.obvod_number}` : ov.slovakiana_id} (
                      {ov.overview_houses_count ?? 0} {lang === "sk" ? "domov" : "houses"})
                    </span>
                  </Link>
                ))}
              </div>
              {sheet.municipality_name && (
                <Link
                  to={`/scitacie-harky?obec=${encodeURIComponent(
                    normalizePlaceSpelling(sheet.municipality_name)
                  )}&rok=${sheet.census_year === 1940 ? "1940" : "1930"}`}
                  className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold inline-flex items-center gap-1.5"
                >
                  <Home className="w-3.5 h-3.5" />
                  <span>
                    {lang === "sk"
                      ? `Adresár domov obce (${joined_overview_houses.length}) →`
                      : `Municipality House Directory (${joined_overview_houses.length}) →`}
                  </span>
                </Link>
              )}
            </div>
          )}
        </div>
      ) : isOverviewSheet ? (
        <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
          {/* Obvod Metadata Summary Cards */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-3">
            <div>
              <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
                <Building2 className="w-5 h-5 text-purple-700" />
                <span>
                  {lang === "sk"
                    ? "Prehľad sčítacieho obvodu – Zoznam domov a majiteľov"
                    : "District Overview – Directory of Houses & Owners"}
                </span>
              </h3>
              <p className="text-xs text-stone-600 mt-0.5">
                {lang === "sk"
                  ? "Kliknite na riadok domu pre zobrazenie výrezu s kontextom okolitých riadkov alebo úpravu údajov."
                  : "Click a house row to view the crop with neighboring rows context or edit house fields."}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs sm:text-sm">
              <div className="px-3 py-1.5 rounded-lg bg-stone-100 border border-stone-200">
                <span className="text-stone-600">
                  {lang === "sk" ? "Obec / Okres: " : "Municipality / District: "}
                </span>
                <strong className="text-stone-900">
                  {sheet.municipality_name} ({sheet.district_name || "—"})
                </strong>
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-purple-50 border border-purple-200">
                <span className="text-purple-800">
                  {lang === "sk" ? "Číslo obvodu: " : "District #: "}
                </span>
                <strong className="text-purple-950">{sheet.obvod_number || "—"}</strong>
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200">
                <span className="text-amber-900">
                  {lang === "sk" ? "Sčítací komisár: " : "Commissioner: "}
                </span>
                <strong className="text-amber-950">{sheet.commissioner_name || "—"}</strong>
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200">
                <span className="text-emerald-900">
                  {lang === "sk" ? "Počet zaznamenaných domov: " : "Recorded houses: "}
                </span>
                <strong className="text-emerald-950">{overview_houses.length}</strong>
              </div>
            </div>
          </div>

          {overview_houses.length === 0 ? (
            <div className="text-center py-8 text-stone-600 text-sm">
              {lang === "sk" ? (
                <>
                  Zatiaľ neboli z tohto prehľadu obvodu extrahované žiadne domy. Kliknite vyššie na{" "}
                  <strong>„Znova spustiť AI OCR & párovanie“</strong>.
                </>
              ) : (
                <>
                  No houses have been extracted from this district overview yet. Click{" "}
                  <strong>"Re-run AI OCR & matching"</strong> above.
                </>
              )}
            </div>
          ) : (
            <div className="w-full overflow-x-auto">
              <table className="w-full table-auto text-left border-collapse text-xs sm:text-sm">
                <thead>
                  <tr className="border-b border-stone-300 bg-stone-100 text-stone-800 font-bold">
                    <th className="py-2.5 px-1.5">
                      {lang === "sk" ? "Radové č." : "Ord. #"}
                    </th>
                    {!splitLayout && (
                      <th className="py-2.5 px-1.5">
                        {lang === "sk" ? "Meno osady" : "Settlement"}
                      </th>
                    )}
                    <th className="py-2.5 px-1.5">
                      {lang === "sk" ? "Popisné č. domu" : "House #"}
                    </th>
                    {!splitLayout && (
                      <th className="py-2.5 px-1.5">
                        {lang === "sk" ? "Obývaný" : "Inhabited"}
                      </th>
                    )}
                    <th className="py-2.5 px-1.5">
                      {lang === "sk" ? "Meno majiteľa domu" : "House owner"}
                    </th>
                    {!splitLayout && (
                      <th className="py-2.5 px-1.5 text-center">
                        {lang === "sk" ? "Bytové strany" : "Units"}
                      </th>
                    )}
                    <th className="py-2.5 px-1.5 text-center">
                      {lang === "sk" ? "Prítomné osoby" : "Residents"}
                    </th>
                    <th className="py-2.5 px-1.5">
                      {lang === "sk" ? "Poznámka" : "Remarks"}
                    </th>
                    <th className="py-2.5 px-1.5">
                      {lang === "sk" ? "Prepojený sčítací hárok" : "Linked census sheet"}
                    </th>
                    <th className="py-2.5 px-1.5 text-right">
                      {lang === "sk" ? "Akcia" : "Action"}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200">
                  {overview_houses.map((h) => {
                    const isSelected = selectedHouse?.id === h.id;
                    const isEditing = editingHouseId === h.id;
                    const isFuzzyQueryMatch = fuzzyMatchedHouseIds.has(h.id);
                    const linkedTarget = h.matched_slovakiana_key || h.matched_sheet_id;
                    return (
                      <React.Fragment key={h.id}>
                        <tr
                          onClick={() => selectHouseAndSyncPage(h)}
                          className={`cursor-pointer transition ${
                            isSelected
                              ? "bg-amber-50/80"
                              : isFuzzyQueryMatch
                              ? "bg-emerald-50/80 ring-1 ring-inset ring-emerald-300"
                              : "hover:bg-stone-50"
                          }`}
                        >
                          <td
                            onMouseEnter={() => {
                              selectHouseAndSyncPage(h);
                              setActiveFieldFocus({
                                key: "ov_ordinal_settlement",
                                valuePreview: String(h.ordinal_number ?? h.row_index),
                              });
                            }}
                            className="py-2.5 px-2 align-middle font-mono font-bold text-stone-800 whitespace-nowrap"
                          >
                            <div className="inline-flex items-center gap-1.5">
                              <span>{h.ordinal_number ?? h.row_index}</span>
                              {(h.booklet_page || h.page_side === "back") && (
                                <span
                                  className="px-1.5 py-0.5 rounded bg-stone-200 text-stone-800 text-xs font-sans font-semibold"
                                  title={
                                    h.page_side === "back"
                                      ? "Zadná strana skenu (_r)"
                                      : "Predná strana skenu (_v)"
                                  }
                                >
                                  str. {h.booklet_page || (h.page_side === "back" ? 2 : 1)}
                                </span>
                              )}
                            </div>
                          </td>
                          {!splitLayout && (
                            <td
                              onMouseEnter={() => {
                                selectHouseAndSyncPage(h);
                                setActiveFieldFocus({
                                  key: "ov_ordinal_settlement",
                                  valuePreview: normalizePlaceSpelling(h.settlement_name) || null,
                                });
                              }}
                              className="py-2.5 px-2 align-middle text-stone-800"
                            >
                              {normalizePlaceSpelling(h.settlement_name) || "—"}
                            </td>
                          )}
                          <td
                            onMouseEnter={() => {
                              selectHouseAndSyncPage(h);
                              setActiveFieldFocus({
                                key: "ov_house_num",
                                valuePreview: h.house_number ? `Dom č. ${h.house_number}` : null,
                              });
                            }}
                            className="py-2.5 px-2 align-middle whitespace-nowrap"
                          >
                            <span className="inline-flex items-center whitespace-nowrap px-2 py-0.5 rounded bg-stone-900 text-amber-300 text-xs font-bold">
                              Dom č. {h.house_number || "—"}
                            </span>
                          </td>
                          {!splitLayout && (
                            <td className="py-2.5 px-2 align-middle whitespace-nowrap">
                              {h.is_inhabited === false ? (
                                <span className="inline-flex items-center whitespace-nowrap px-2 py-0.5 rounded bg-stone-200 text-stone-800 text-xs">
                                  {lang === "sk" ? "Neobývaný" : "Uninhabited"}
                                </span>
                              ) : (
                                <span className="inline-flex items-center whitespace-nowrap px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 text-xs font-semibold">
                                  {lang === "sk" ? "Áno" : "Yes"}
                                </span>
                              )}
                            </td>
                          )}
                          <td
                            onMouseEnter={() => {
                              selectHouseAndSyncPage(h);
                              setActiveFieldFocus({
                                key: "ov_owner_name",
                                valuePreview: h.owner_name || null,
                              });
                            }}
                            className="py-2.5 px-2 align-middle font-semibold text-stone-900"
                          >
                            {h.owner_name || "—"}
                          </td>
                          {!splitLayout && (
                            <td
                              onMouseEnter={() => {
                                selectHouseAndSyncPage(h);
                                setActiveFieldFocus({
                                  key: "ov_counts",
                                  valuePreview: `${h.household_count ?? "—"} byt. / ${h.present_persons_count ?? "—"} os.`,
                                });
                              }}
                              className="py-2.5 px-2 align-middle text-center font-semibold text-stone-800"
                            >
                              {h.household_count ?? "—"}
                            </td>
                          )}
                          <td
                            onMouseEnter={() => {
                              selectHouseAndSyncPage(h);
                              setActiveFieldFocus({
                                key: "ov_counts",
                                valuePreview: `${h.present_persons_count ?? "—"} os.`,
                              });
                            }}
                            className="py-2.5 px-2 align-middle text-center font-bold text-amber-900"
                          >
                            {h.present_persons_count ?? "—"}
                          </td>
                          <td
                            onMouseEnter={() => {
                              selectHouseAndSyncPage(h);
                              setActiveFieldFocus({
                                key: "ov_remarks",
                                valuePreview: h.remarks || null,
                              });
                            }}
                            className="py-2.5 px-2 align-middle text-stone-700 break-words max-w-[200px]"
                          >
                            {h.remarks || "—"}
                          </td>
                          <td className="py-2.5 px-2 align-middle">
                            {h.matched_sheets && h.matched_sheets.length > 0 ? (
                              <div className="flex flex-wrap items-center gap-1.5">
                                {h.matched_sheets.map((ms) => {
                                  const msRef =
                                    ms.cair_id || ms.slovakiana_key || String(ms.id);
                                  const isSameYear = ms.census_year === h.census_year;
                                  return (
                                    <Link
                                      key={`${ms.id}-${msRef}`}
                                      to={`/scitacie-harky/${msRef}`}
                                      onClick={(e) => e.stopPropagation()}
                                      className={`px-2.5 py-1 min-h-[32px] rounded-md text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap ${
                                        isSameYear
                                          ? "bg-emerald-100 hover:bg-emerald-200 text-emerald-950"
                                          : "bg-indigo-100 hover:bg-indigo-200 text-indigo-950 border border-indigo-300"
                                      }`}
                                      title={`Sčítanie ${ms.census_year} · ${msRef}`}
                                    >
                                      <FileSpreadsheet
                                        className={`w-3.5 h-3.5 shrink-0 ${
                                          isSameYear ? "text-emerald-700" : "text-indigo-700"
                                        }`}
                                      />
                                      <span>
                                        {ms.census_year}: {msRef}
                                      </span>
                                      {(ms.head_of_household ||
                                        (isSameYear && h.matched_head_name)) && (
                                        <span className="opacity-85 font-normal">
                                          ({ms.head_of_household || h.matched_head_name})
                                        </span>
                                      )}
                                    </Link>
                                  );
                                })}
                              </div>
                            ) : linkedTarget ? (
                              <Link
                                to={`/scitacie-harky/${linkedTarget}`}
                                onClick={(e) => e.stopPropagation()}
                                className="px-2.5 py-1 min-h-[32px] rounded-md bg-emerald-100 hover:bg-emerald-200 text-emerald-950 text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap"
                              >
                                <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
                                <span>{h.matched_slovakiana_key || `Hárok #${linkedTarget}`}</span>
                                {h.matched_head_name && (
                                  <span className="text-emerald-800 font-normal">
                                    ({h.matched_head_name})
                                  </span>
                                )}
                              </Link>
                            ) : (
                              <span className="text-stone-600 text-xs">
                                {lang === "sk" ? "Nespárované" : "Unpaired"}
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-2 align-middle text-right whitespace-nowrap">
                            <div className="inline-flex items-center gap-1.5">
                              {featureFlags.admin_debug_overlay && (
                                <OcrFeedbackControls
                                  lang={lang}
                                  targetType="overview_house"
                                  overviewHouseId={h.id}
                                  rowId={h.row_id ?? null}
                                  scanId={h.scan_id ?? front_scan?.id ?? null}
                                  censusSheetId={sheet.id}
                                  initialFeedback={h.ocr_feedback}
                                  resetVersion={feedbackResetVersion}
                                  onFeedbackSaved={(fb) => {
                                    setDetail((prev) =>
                                      prev
                                        ? {
                                            ...prev,
                                            overview_houses: (
                                              prev.overview_houses || []
                                            ).map((item) =>
                                              item.id === h.id
                                                ? { ...item, ocr_feedback: fb }
                                                : item
                                            ),
                                          }
                                        : prev
                                    );
                                  }}
                                />
                              )}
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (isEditing) {
                                    setEditingHouseId(null);
                                  } else {
                                    startEditHouse(h);
                                  }
                                }}
                                className="px-3 py-1.5 min-h-[32px] rounded-md bg-stone-200 hover:bg-stone-300 text-stone-800 text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap cursor-pointer"
                              >
                                <Edit3 className="w-3.5 h-3.5 shrink-0" />
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
                              {confirmDeleteHouseId === h.id ? (
                                <div
                                  onClick={(e) => e.stopPropagation()}
                                  className="inline-flex items-center gap-1.5 bg-red-50 border border-red-300 rounded-md px-2.5 py-1 min-h-[32px]"
                                >
                                  <button
                                    type="button"
                                    disabled={deletingHouseId === h.id}
                                    onClick={() => handleDeleteHouse(h.id)}
                                    className="text-xs font-bold text-red-700 hover:underline cursor-pointer"
                                  >
                                    {deletingHouseId === h.id
                                      ? "…"
                                      : lang === "sk"
                                      ? "Vymazať"
                                      : "Delete"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setConfirmDeleteHouseId(null)}
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
                                    setConfirmDeleteHouseId(h.id);
                                  }}
                                  title={
                                    lang === "sk"
                                      ? "Vymazať riadok domu"
                                      : "Delete house row"
                                  }
                                  className="p-2 min-h-[32px] min-w-[32px] inline-flex items-center justify-center rounded-md bg-stone-200 hover:bg-red-100 text-stone-700 hover:text-red-700 cursor-pointer"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>

                        {isEditing && (
                          <tr>
                            <td
                              colSpan={10}
                              className="bg-amber-50/70 border-b border-amber-200 p-4"
                            >
                              <div className="space-y-3">
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <h4 className="text-xs font-bold uppercase tracking-wider text-amber-950">
                                    {lang === "sk"
                                      ? `Úprava záznamu domu v prehľade obvodu (riadok #${h.row_index})`
                                      : `Edit District Overview House Row (#${h.row_index})`}
                                  </h4>
                                  <div className="flex items-center gap-2">
                                    <button
                                      type="button"
                                      onClick={() => handleSaveHouse(h.id)}
                                      disabled={savingHouseId === h.id}
                                      className="px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 disabled:opacity-50 text-white font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                    >
                                      {savingHouseId === h.id ? (
                                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                                      ) : (
                                        <Save className="w-3.5 h-3.5" />
                                      )}
                                      <span>
                                        {savingHouseId === h.id
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
                                      onClick={() => setEditingHouseId(null)}
                                      className="px-3 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                    >
                                      <X className="w-3.5 h-3.5" />
                                      <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                                    </button>
                                  </div>
                                </div>

                                {h.stitched_crop_path && (
                                  <HoverZoomRowCrop
                                    cropUrl={h.stitched_crop_path}
                                    headerCropUrl={h.header_crop_url || activeHeaderCropUrl}
                                    scanUrl={featureFlags.hover_row_zoom ? activeRowScanUrl : null}
                                    yStart={
                                      featureFlags.hover_row_zoom
                                        ? h.y_start ??
                                          resolveOverviewHouseRow(h)?.y_start ??
                                          null
                                        : null
                                    }
                                    yEnd={
                                      featureFlags.hover_row_zoom
                                        ? h.y_end ??
                                          resolveOverviewHouseRow(h)?.y_end ??
                                          null
                                        : null
                                    }
                                    scanHeight={activeRowScanHeight}
                                    scanWidth={activeRowScanWidth}
                                    rowLabel={`Výrez domu (riadok č. ${
                                      h.ordinal_number || h.row_index
                                    }${h.booklet_page ? ` · strana ${h.booklet_page}` : ""}): dom č. ${
                                      h.house_number || "—"
                                    } – ${h.owner_name || ""}`.trim()}
                                  />
                                )}

                                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Radové č. (stĺpec 1)
                                    </label>
                                    <input
                                      type="number"
                                      value={houseEditForm.ordinal_number}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          ordinal_number: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Meno osady (stĺpec 2)
                                    </label>
                                    <input
                                      type="text"
                                      value={houseEditForm.settlement_name}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          settlement_name: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Popisné č. domu (stĺpec 3)
                                    </label>
                                    <input
                                      type="text"
                                      value={houseEditForm.house_number}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          house_number: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Meno majiteľa domu (stĺpec 6)
                                    </label>
                                    <input
                                      type="text"
                                      value={houseEditForm.owner_name}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          owner_name: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Počet bytových strán (stĺpec 7)
                                    </label>
                                    <input
                                      type="number"
                                      value={houseEditForm.household_count}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          household_count: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Počet prítomných osôb (stĺpec 8)
                                    </label>
                                    <input
                                      type="number"
                                      value={houseEditForm.present_persons_count}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          present_persons_count: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Obývaný dom (stĺpce 4–5)
                                    </label>
                                    <select
                                      value={houseEditForm.is_inhabited ? "1" : "0"}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          is_inhabited: e.target.value === "1",
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    >
                                      <option value="1">Obývaný</option>
                                      <option value="0">Neobývaný</option>
                                    </select>
                                  </div>
                                  <div>
                                    <label className="block font-semibold text-stone-700 mb-1">
                                      Poznámka (stĺpec 9)
                                    </label>
                                    <input
                                      type="text"
                                      value={houseEditForm.remarks}
                                      onChange={(e) =>
                                        setHouseEditForm({
                                          ...houseEditForm,
                                          remarks: e.target.value,
                                        })
                                      }
                                      className="w-full px-2.5 py-1.5 rounded border border-stone-300 bg-white"
                                    />
                                  </div>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        /* SECTION 3: Household Table (`Členovia domácnosti`) with Expandable Inline Edit Drawer */
        <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-bold text-stone-900">
                {lang === "sk"
                  ? `Členovia domácnosti na sčítacom hárku (${members.length})`
                  : `Household Members on Census Sheet (${members.length})`}
              </h3>
              <p className="text-xs text-stone-500">
                {lang === "sk"
                  ? "Kliknite na riadok pre náhľad výrezu alebo na „Upraviť“ pre rozbalenie všetkých stĺpcov sčítacieho hárku (1–24)."
                  : "Click a row to preview its crop or click 'Edit' to expand all census columns (1–24)."}
              </p>
            </div>
            <button
              onClick={handleAddPerson}
              disabled={addingPerson}
              className="px-3.5 py-2 min-h-[36px] rounded-lg bg-stone-900 hover:bg-stone-800 disabled:opacity-50 text-white font-semibold text-xs sm:text-sm flex items-center gap-1.5 cursor-pointer"
            >
              {addingPerson ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Plus className="w-4 h-4" />
              )}
              <span>
                {addingPerson
                  ? lang === "sk"
                    ? "Pridávam osobu…"
                    : "Adding person…"
                  : lang === "sk"
                  ? "Pridať osobu do domácnosti"
                  : "Add Person to Household"}
              </span>
            </button>
          </div>

          <div className="w-full overflow-x-auto">
            <table className="w-full table-auto text-left border-collapse text-xs sm:text-sm">
              <thead>
                <tr className="border-b border-stone-300 bg-stone-100 text-stone-800 font-bold">
                  <th className="py-2.5 px-1.5 whitespace-nowrap">#</th>
                  <th className="py-2.5 px-1.5">
                    {lang === "sk" ? "Priezvisko (a rodné)" : "Surname (& maiden)"}
                  </th>
                  <th className="py-2.5 px-1.5">
                    {lang === "sk" ? "Krstné meno" : "Given name"}
                  </th>
                  <th className="py-2.5 px-1.5">
                    {lang === "sk" ? "Pomer k hlave" : "Relation to head"}
                  </th>
                  {!splitLayout && (
                    <th className="py-2.5 px-1.5 text-center">
                      {lang === "sk" ? "Pohl." : "Sex"}
                    </th>
                  )}
                  <th className="py-2.5 px-1.5">
                    {lang === "sk" ? "Dátum nar. / Stav" : "Birth date / Status"}
                  </th>
                  <th className="py-2.5 px-1.5">
                    {lang === "sk"
                      ? splitLayout
                        ? "Rodisko"
                        : "Rodisko (Obec, Okres, Krajina)"
                      : "Birthplace"}
                  </th>
                  {!splitLayout && (
                    <>
                      <th className="py-2.5 px-1.5">
                        {lang === "sk" ? "Domovská a štátna prísl." : "Domicile & Citizenship"}
                      </th>
                      <th className="py-2.5 px-1.5">
                        {lang === "sk" ? "Národnosť / Nábož." : "Nationality / Religion"}
                      </th>
                    </>
                  )}
                  <th className="py-2.5 px-1.5">
                    {lang === "sk" ? "Povolanie / Prepojenia" : "Occupation / Links"}
                  </th>
                  <th className="py-2.5 px-1.5 text-right whitespace-nowrap">
                    {lang === "sk" ? "Akcia" : "Action"}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-200">
                {members.length === 0 && (
                  <tr>
                    <td colSpan={11} className="py-6 px-4 text-center text-xs text-stone-600">
                      {linked_overview_house?.is_inhabited === false ||
                      sheet.household_owner?.toLowerCase().includes("neobývaný") ? (
                        <div className="inline-flex flex-col items-center gap-1 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-amber-950">
                          <span className="font-bold text-sm">
                            Neobývaný dom (0 zapísaných obyvateľov na hárku)
                          </span>
                          <span>
                            Majiteľ domu:{" "}
                            <strong>
                              {linked_overview_house?.owner_name ||
                                sheet.household_owner ||
                                "neznámy"}
                            </strong>
                            {linked_overview_house?.settlement_name
                              ? ` · Osada: ${linked_overview_house.settlement_name}`
                              : ""}
                          </span>
                        </div>
                      ) : (
                        <div className="inline-flex flex-col items-center gap-2 bg-stone-50 border border-stone-200 rounded-xl px-5 py-4 text-stone-700 max-w-lg mx-auto">
                          <span className="font-semibold text-sm text-stone-900">
                            {lang === "sk"
                              ? "Na tomto sčítacom hárku zatiaľ nie sú zapísané žiadne osoby."
                              : "No persons have been recorded on this census sheet yet."}
                          </span>
                          {sheet.is_blackened && (
                            <span className="text-xs text-stone-600 leading-relaxed">
                              {lang === "sk"
                                ? "Poznámka: Hárok zo sčítania 1940 je čiastočne začiernený archívom (ochrana osobných údajov). Môžete spustiť AI OCR na viditeľné riadky alebo pridať čitateľné osoby ručne."
                                : "Note: This 1940 census sheet is partially redacted by the archive. You can run AI OCR on visible rows or add readable persons manually."}
                            </span>
                          )}
                          <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
                            <button
                              type="button"
                              onClick={handleDownloadAndProcess}
                              disabled={processing || switchingType}
                              className="px-3.5 py-1.5 min-h-[34px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                            >
                              {processing ? (
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Download className="w-3.5 h-3.5" />
                              )}
                              <span>
                                {processing
                                  ? lang === "sk"
                                    ? `Prebieha AI OCR (${Math.max(8, sheetProgress.progress_pct || 8)}%)…`
                                    : `Running AI OCR (${Math.max(8, sheetProgress.progress_pct || 8)}%)…`
                                  : lang === "sk"
                                  ? "Spustiť AI OCR hárku"
                                  : "Run AI OCR on sheet"}
                              </span>
                            </button>
                            <button
                              type="button"
                              onClick={handleAddPerson}
                              disabled={addingPerson}
                              className="px-3.5 py-1.5 min-h-[34px] rounded-lg bg-stone-900 hover:bg-stone-800 disabled:opacity-50 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer"
                            >
                              {addingPerson ? (
                                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Plus className="w-3.5 h-3.5" />
                              )}
                              <span>
                                {addingPerson
                                  ? lang === "sk"
                                    ? "Pridávam osobu…"
                                    : "Adding person…"
                                  : lang === "sk"
                                  ? "Pridať osobu ručne"
                                  : "Add person manually"}
                              </span>
                            </button>
                          </div>
                        </div>
                      )}
                    </td>
                  </tr>
                )}
                {members.map((m, memberIdx) => {
                  const f = m.fact;
                  const isEditing = editingMemberId === f.id;
                  const isSelected = selectedMemberId === f.id;
                  const isFuzzyQueryMatch = fuzzyMatchedMemberIds.has(f.id);
                  const bp = formatStructuredBirthPlace(f);
                  const homeJur = formatHomeJurisdiction(f.home_jurisdiction);
                  const hasOccupation = Boolean(
                    (f.social_status_or_occupation &&
                      f.social_status_or_occupation !== "—") ||
                      f.occupation_position ||
                      f.employer_workplace
                  );
                  const bytGroup = memberBytMap.get(f.id) || bytGroups[0] || null;
                  const prevMember = memberIdx > 0 ? members[memberIdx - 1] : null;
                  const prevBytGroup = prevMember
                    ? memberBytMap.get(prevMember.fact.id) || bytGroups[0] || null
                    : null;
                  const showBytSubheader =
                    hasMultipleByty &&
                    bytGroup &&
                    (memberIdx === 0 || prevBytGroup?.bytLabel !== bytGroup.bytLabel);

                  const consensusInfo = detectSurnameConsensusUnification(
                    m,
                    bytGroup?.consensusSurname || null,
                    featureFlags.census_surname_consensus
                  );

                  return (
                    <React.Fragment key={f.id}>
                      {showBytSubheader && bytGroup && (
                        <tr className="bg-amber-100/80 border-y border-amber-300">
                          <td colSpan={11} className="py-1.5 px-3">
                            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="inline-flex items-center whitespace-nowrap px-2 py-0.5 rounded bg-stone-900 text-amber-300 font-bold">
                                  {bytGroup.bytLabel}
                                </span>
                                <span className="font-bold text-stone-900">
                                  {bytGroup.bytLabel} — Hlava domácnosti: {bytGroup.headFullName}
                                </span>
                                {bytGroup.wifeFullName && (
                                  <span className="text-stone-700">
                                    · Manželka: <strong>{bytGroup.wifeFullName}</strong>
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-2 text-[11px]">
                                {bytGroup.consensusSurname && (
                                  <span className="inline-flex items-center whitespace-nowrap px-2 py-0.5 rounded bg-white border border-amber-300 text-amber-950 font-semibold">
                                    Priezvisko bytu (konsenzus): {bytGroup.consensusSurname}
                                  </span>
                                )}
                                <span className="font-semibold text-stone-700 whitespace-nowrap">
                                  {bytGroup.memberCount} osôb
                                </span>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                      <tr
                        onClick={() => setSelectedMemberId(f.id)}
                        className={`cursor-pointer transition ${
                          isSelected
                            ? "bg-amber-50/70"
                            : isFuzzyQueryMatch
                            ? "bg-emerald-50/80 ring-1 ring-inset ring-emerald-300"
                            : "hover:bg-stone-50"
                        }`}
                      >
                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "row_num",
                              valuePreview: String(f.person_row_number || "—"),
                            });
                          }}
                          className="py-2.5 px-2 align-middle font-bold text-stone-800 whitespace-nowrap"
                        >
                          <div className="inline-flex items-center gap-1.5 whitespace-nowrap">
                            <span className="min-w-[1ch]">{f.person_row_number || "—"}</span>
                            {bytGroup && (
                              <span
                                className="inline-flex items-center whitespace-nowrap shrink-0 rounded px-1.5 py-0.5 bg-stone-200 text-stone-800 text-xs font-semibold leading-none"
                                title={`${bytGroup.bytLabel} — Hlava domácnosti: ${bytGroup.headFullName}`}
                              >
                                {bytGroup.bytLabel}
                              </span>
                            )}
                          </div>
                        </td>

                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "surname",
                              valuePreview: f.surname_raw || null,
                            });
                          }}
                          className="py-2.5 px-2 align-middle font-semibold text-stone-900"
                        >
                          <div className="flex flex-wrap items-center gap-1">
                            <span>{f.surname_raw || "—"}</span>
                            {consensusInfo.wasUnified &&
                              consensusInfo.fromRaw &&
                              consensusInfo.toConsensus && (
                                <span
                                  title={`Priezvisko zjednotené podľa konsenzu bytu (${bytGroup?.bytLabel || "Byt"}): pôvodné OCR „${consensusInfo.fromRaw}“ → „${consensusInfo.toConsensus}“`}
                                  className="inline-flex items-center whitespace-nowrap shrink-0 px-1.5 py-0.5 rounded bg-amber-100 border border-amber-300 text-amber-900 text-xs font-semibold leading-none"
                                >
                                  OCR: {consensusInfo.fromRaw} → {consensusInfo.toConsensus}
                                </span>
                              )}
                          </div>
                          {f.mother_maiden_surname &&
                            f.sex !== "M" &&
                            ![
                              "syn",
                              "dcéra",
                              "vnuk",
                              "vnučka",
                              "chovanec",
                              "chovanka",
                              "adoptívny syn",
                              "adoptívna dcéra",
                              "nevlastný syn",
                              "nevlastná dcéra",
                            ].includes((f.relation_to_head || "").trim().toLowerCase()) && (
                            <div className="text-xs font-medium text-amber-900">
                              rod. {f.mother_maiden_surname}
                            </div>
                          )}
                        </td>

                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "given_name",
                              valuePreview: f.given_name_raw || null,
                            });
                          }}
                          className="py-2.5 px-2 align-middle font-semibold text-stone-900"
                        >
                          {f.given_name_raw || "—"}
                        </td>

                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "relation",
                              valuePreview: f.relation_to_head || null,
                            });
                          }}
                          className="py-2.5 px-2 align-middle"
                        >
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                              isHeadOfHouseholdRelation(f.relation_to_head)
                                ? "bg-amber-200/80 text-amber-950 font-bold"
                                : "bg-stone-200 text-stone-800"
                            }`}
                          >
                            {f.relation_to_head || "—"}
                          </span>
                        </td>

                        {!splitLayout && (
                          <td
                            onMouseEnter={() => {
                              setSelectedMemberId(f.id);
                              setActiveFieldFocus({
                                key: "sex",
                                valuePreview: f.sex || null,
                              });
                            }}
                            className="py-2.5 px-2 align-middle text-center font-semibold text-stone-800"
                          >
                            {f.sex || "—"}
                          </td>
                        )}

                        <td
                          data-testid={`cell-dob-${f.id}`}
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "dob_marital",
                              valuePreview:
                                f.birth_or_event_date ||
                                (f.birth_year_est ? String(f.birth_year_est) : null),
                            });
                          }}
                          className="py-2.5 px-2 align-middle"
                        >
                          <div className="font-medium text-stone-900">
                            {f.birth_or_event_date ||
                              (f.birth_year_est ? String(f.birth_year_est) : "—")}
                          </div>
                          {f.marital_status && (
                            <div className="text-xs text-stone-600">
                              {f.marital_status}
                            </div>
                          )}
                        </td>

                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "birthplace",
                              valuePreview: bp.primary !== "—" ? bp.primary : null,
                            });
                          }}
                          className="py-2.5 px-2 align-middle"
                        >
                          <div className="font-medium text-stone-900">
                            {bp.primary}
                          </div>
                          {bp.secondary && (
                            <div className="text-xs text-stone-600">
                              {bp.secondary}
                            </div>
                          )}
                        </td>

                        {!splitLayout && (
                          <>
                            <td
                              onMouseEnter={() => {
                                setSelectedMemberId(f.id);
                                setActiveFieldFocus({
                                  key: "domicile_citizenship",
                                  valuePreview: homeJur || f.citizenship || null,
                                });
                              }}
                              className="py-2.5 px-2 align-middle"
                            >
                              <div className="text-stone-800">
                                {homeJur ? `Dom.: ${homeJur}` : "—"}
                              </div>
                              {f.citizenship && (
                                <div className="text-xs text-stone-600">
                                  Štát.: {f.citizenship}
                                </div>
                              )}
                            </td>

                            <td
                              onMouseEnter={() => {
                                setSelectedMemberId(f.id);
                                setActiveFieldFocus({
                                  key: "nationality_religion",
                                  valuePreview:
                                    [f.nationality, f.religion].filter(Boolean).join(" · ") ||
                                    null,
                                });
                              }}
                              className="py-2.5 px-2 align-middle"
                            >
                              <div className="font-medium text-stone-800">
                                {f.nationality || "—"}
                              </div>
                              {f.religion && (
                                <div className="text-xs text-stone-600">
                                  {f.religion}
                                </div>
                              )}
                            </td>
                          </>
                        )}

                        <td
                          onMouseEnter={() => {
                            setSelectedMemberId(f.id);
                            setActiveFieldFocus({
                              key: "occupation_notes",
                              valuePreview:
                                f.social_status_or_occupation &&
                                f.social_status_or_occupation !== "—"
                                  ? f.social_status_or_occupation
                                  : null,
                            });
                          }}
                          className="py-2.5 px-2 align-middle"
                        >
                          {hasOccupation ? (
                            <div className="space-y-1">
                              <div className="leading-snug">
                                <span className="font-medium text-stone-900">
                                  {f.social_status_or_occupation || "—"}
                                </span>
                                {f.occupation_position && (
                                  <span className="text-stone-600">
                                    {" "}
                                    ({f.occupation_position})
                                  </span>
                                )}
                              </div>
                              {f.employer_workplace && (
                                <div className="text-stone-600 text-xs leading-snug">
                                  {normalizePlaceSpelling(f.employer_workplace)}
                                </div>
                              )}
                              {(Boolean(m.local_matches?.length) ||
                                Boolean(m.fs_match?.fs_url)) && (
                                <div className="flex flex-wrap items-center gap-1 pt-0.5">
                                  {m.local_matches?.map((lm) => (
                                    <div
                                      key={lm.match.id}
                                      className="px-2 py-0.5 rounded bg-blue-100 text-blue-900 font-semibold text-xs inline-flex items-center gap-1"
                                      title={lm.match.explanation_sk}
                                    >
                                      <CheckCircle2 className="w-3 h-3 shrink-0" />
                                      <span>
                                        Matrika ({lm.matrika_fact?.event_year}):{" "}
                                        {lm.matrika_fact?.given_name_raw}{" "}
                                        {lm.matrika_fact?.surname_raw}
                                      </span>
                                    </div>
                                  ))}
                                  {m.fs_match?.fs_url && (
                                    <a
                                      href={m.fs_match.fs_url}
                                      target="_blank"
                                      rel="noreferrer"
                                      onClick={(e) => e.stopPropagation()}
                                      className="px-2 py-1 min-h-[28px] rounded bg-emerald-100 hover:bg-emerald-200 text-emerald-900 font-semibold text-xs inline-flex items-center gap-1"
                                    >
                                      <Sparkles className="w-3 h-3 shrink-0" />
                                      <span>FamilySearch</span>
                                      <ExternalLink className="w-3 h-3 shrink-0" />
                                    </a>
                                  )}
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="flex flex-wrap items-center gap-1.5">
                              {!m.local_matches?.length && !m.fs_match?.fs_url && (
                                <span className="text-stone-600">—</span>
                              )}
                              {m.local_matches?.map((lm) => (
                                <div
                                  key={lm.match.id}
                                  className="px-2 py-0.5 rounded bg-blue-100 text-blue-900 font-semibold text-xs inline-flex items-center gap-1"
                                  title={lm.match.explanation_sk}
                                >
                                  <CheckCircle2 className="w-3 h-3 shrink-0" />
                                  <span>
                                    Matrika ({lm.matrika_fact?.event_year}):{" "}
                                    {lm.matrika_fact?.given_name_raw}{" "}
                                    {lm.matrika_fact?.surname_raw}
                                  </span>
                                </div>
                              ))}
                              {m.fs_match?.fs_url && (
                                <a
                                  href={m.fs_match.fs_url}
                                  target="_blank"
                                  rel="noreferrer"
                                  onClick={(e) => e.stopPropagation()}
                                  className="px-2 py-1 min-h-[28px] rounded bg-emerald-100 hover:bg-emerald-200 text-emerald-900 font-semibold text-xs inline-flex items-center gap-1"
                                >
                                  <Sparkles className="w-3 h-3 shrink-0" />
                                  <span>FamilySearch</span>
                                  <ExternalLink className="w-3 h-3 shrink-0" />
                                </a>
                              )}
                            </div>
                          )}
                        </td>

                        <td className="py-2.5 px-2 align-middle text-right whitespace-nowrap">
                          <div className="inline-flex items-center gap-1.5">
                            {featureFlags.admin_debug_overlay && (
                              <OcrFeedbackControls
                                lang={lang}
                                targetType="person_fact"
                                personFactId={f.id}
                                rowId={f.row_id}
                                scanId={f.scan_id}
                                censusSheetId={sheet.id}
                                initialFeedback={m.ocr_feedback}
                                resetVersion={feedbackResetVersion}
                                onFeedbackSaved={(fb) => {
                                  setDetail((prev) =>
                                    prev
                                      ? {
                                          ...prev,
                                          members: (prev.members || []).map((item) =>
                                            item.fact.id === f.id
                                              ? { ...item, ocr_feedback: fb }
                                              : item
                                          ),
                                        }
                                      : prev
                                  );
                                }}
                              />
                            )}
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isEditing) {
                                  setEditingMemberId(null);
                                } else {
                                  startEditMember(m, bytGroup?.bytRoman);
                                }
                              }}
                              className={`px-3 py-1.5 min-h-[32px] rounded-md text-xs font-semibold inline-flex items-center gap-1 whitespace-nowrap cursor-pointer ${
                                isEditing
                                  ? "bg-amber-600 text-white"
                                  : "bg-stone-200 hover:bg-stone-300 text-stone-800"
                              }`}
                            >
                              <Edit3 className="w-3.5 h-3.5 shrink-0" />
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

                            {confirmDeleteMemberId === f.id ? (
                              <div
                                onClick={(e) => e.stopPropagation()}
                                className="inline-flex items-center gap-1.5 bg-red-50 border border-red-300 rounded-md px-2.5 py-1 min-h-[32px]"
                              >
                                <button
                                  type="button"
                                  disabled={deletingMemberId === f.id}
                                  onClick={() => handleDeleteMember(f.id)}
                                  className="text-xs font-bold text-red-700 hover:underline cursor-pointer"
                                >
                                  {deletingMemberId === f.id
                                    ? "…"
                                    : lang === "sk"
                                    ? "Vymazať"
                                    : "Delete"}
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setConfirmDeleteMemberId(null)}
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
                                  setConfirmDeleteMemberId(f.id);
                                }}
                                title={
                                  lang === "sk"
                                    ? "Vymazať osobu z hárku"
                                    : "Delete person from sheet"
                                }
                                className="p-2 min-h-[32px] min-w-[32px] inline-flex items-center justify-center rounded-md bg-stone-200 hover:bg-red-100 text-stone-700 hover:text-red-700 cursor-pointer"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>

                      {/* Expandable Detail Edit Drawer */}
                    { isEditing && (
                      <tr>
                        <td
                          colSpan={11}
                          className="bg-amber-50/70 border-b border-amber-200 p-4"
                        >
                          <div className="space-y-4">
                            {/* Drawer Header */}
                            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-amber-200/80 pb-2.5">
                              <div className="flex items-center gap-2">
                                <UserCheck className="w-4 h-4 text-amber-800" />
                                <span className="font-bold text-stone-900 text-sm">
                                  {lang === "sk"
                                    ? `Úprava osoby (riadok č. ${f.person_row_number || "—"}${
                                        bytGroup ? ` · ${bytGroup.bytLabel}` : ""
                                      }): `
                                    : `Edit Person (row #${f.person_row_number || "—"}${
                                        bytGroup ? ` · ${bytGroup.bytLabel}` : ""
                                      }): `}
                                  {f.given_name_raw || ""} {f.surname_raw || ""}
                                </span>
                              </div>
                              <div className="flex items-center gap-2">
                                <button
                                  type="button"
                                  onClick={() => handleSaveMember(f.id)}
                                  disabled={savingId === f.id}
                                  className="px-3.5 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                                >
                                  <Save className="w-3.5 h-3.5" />
                                  <span>
                                    {savingId === f.id
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
                                  onClick={() => setEditingMemberId(null)}
                                  className="px-3 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                >
                                  <X className="w-3.5 h-3.5" />
                                  <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                                </button>
                              </div>
                            </div>

                            {/* Inline Row Crop with Pinned Column Header inside Drawer */}
                            {(m.row?.stitched_crop_path || m.stitched_crop_url) && (
                              <HoverZoomRowCrop
                                cropUrl={m.row?.stitched_crop_path || m.stitched_crop_url}
                                headerCropUrl={activeHeaderCropUrl}
                                scanUrl={featureFlags.hover_row_zoom ? activeRowScanUrl : null}
                                yStart={
                                  featureFlags.hover_row_zoom
                                    ? m.row?.y_start ?? rowsById.get(f.row_id)?.y_start ?? null
                                    : null
                                }
                                yEnd={
                                  featureFlags.hover_row_zoom
                                    ? m.row?.y_end ?? rowsById.get(f.row_id)?.y_end ?? null
                                    : null
                                }
                                scanHeight={activeRowScanHeight}
                                scanWidth={activeRowScanWidth}
                                activeFieldFocus={activeFieldFocus}
                                onSelectFieldFocus={setActiveFieldFocus}
                                showFieldPills={true}
                                rowLabel={`Výrez upravovaného riadku č. ${
                                  f.person_row_number || "—"
                                }: ${f.given_name_raw || ""} ${f.surname_raw || ""}`.trim()}
                              />
                            )}

                              {/* 6 Grouped Form Sections */}
                              <div
                                className={`grid grid-cols-1 md:grid-cols-2 ${
                                  splitLayout ? "" : "lg:grid-cols-3"
                                } gap-4 text-xs sm:text-sm`}
                              >
                                {/* 1. Identita a vzťah */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk" ? "1. Identita a vzťah" : "1. Identity & Relation"}
                                  </h4>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Poradové č." : "Row #"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.person_row_number}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            person_row_number: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Byt č. (I, II, III…)" : "Unit # (I, II, III…)"}
                                      </label>
                                      <select
                                        value={editForm.dwelling_unit || ""}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            dwelling_unit: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300 bg-white"
                                      >
                                        <option value="">
                                          {lang === "sk" ? "Automaticky podľa hlavy" : "Auto by head"}
                                        </option>
                                        {ROMAN_NUMERALS.slice(0, 6).map((rom) => (
                                          <option key={rom} value={rom}>
                                            Byt č. {rom}
                                          </option>
                                        ))}
                                      </select>
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Priezvisko" : "Surname"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.surname_raw}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, surname_raw: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                      {bytGroup?.consensusSurname &&
                                        editForm.surname_raw &&
                                        stripFeminineSurnameSuffixClient(
                                          editForm.surname_raw
                                        ).toLowerCase() !==
                                          stripFeminineSurnameSuffixClient(
                                            bytGroup.consensusSurname
                                          ).toLowerCase() && (
                                          <button
                                            type="button"
                                            onClick={() => {
                                              const target = formatConsensusSurnameForMemberClient(
                                                bytGroup.consensusSurname!,
                                                editForm.sex,
                                                String(editForm.surname_raw || "")
                                              );
                                              setEditForm({
                                                ...editForm,
                                                surname_raw: target,
                                              });
                                            }}
                                            className="mt-1 text-xs text-amber-800 hover:underline font-semibold cursor-pointer"
                                          >
                                            Použiť konsenzus bytu ({bytGroup.consensusSurname})
                                          </button>
                                        )}
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Rodné priezvisko" : "Maiden surname"}
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
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
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
                                          setEditForm({
                                            ...editForm,
                                            given_name_raw: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Pohlavie (M/F)" : "Sex (M/F)"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.sex}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, sex: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div className="col-span-2 space-y-1.5 pt-1 border-t border-stone-100">
                                      <label className="block font-semibold text-stone-700">
                                        {lang === "sk"
                                          ? "Pomer k hlave domácnosti (stĺpec 4)"
                                          : "Relation to head of household (col 4)"}
                                      </label>
                                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                                        <input
                                          type="text"
                                          list="relation-to-head-options"
                                          placeholder="Napíšte alebo vyberte vzťah…"
                                          value={editForm.relation_to_head}
                                          onChange={(e) =>
                                            setEditForm({
                                              ...editForm,
                                              relation_to_head: e.target.value,
                                            })
                                          }
                                          className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                        />
                                        <select
                                          value={
                                            ALL_RELATION_TO_HEAD_OPTIONS.some(
                                              (o) => o.value === editForm.relation_to_head
                                            )
                                              ? editForm.relation_to_head
                                              : ""
                                          }
                                          onChange={(e) => {
                                            if (e.target.value) {
                                              setEditForm({
                                                ...editForm,
                                                relation_to_head: e.target.value,
                                              });
                                            }
                                          }}
                                          className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300 bg-stone-50"
                                        >
                                          <option value="">— Rýchly výber zo zoznamu —</option>
                                          {RELATION_TO_HEAD_GROUPS.map((grp) => (
                                            <optgroup key={grp.groupLabel} label={grp.groupLabel}>
                                              {grp.options.map((opt) => (
                                                <option key={opt.value} value={opt.value}>
                                                  {opt.label}
                                                </option>
                                              ))}
                                            </optgroup>
                                          ))}
                                        </select>
                                      </div>
                                      <div className="flex flex-wrap gap-1.5 pt-1">
                                        {QUICK_RELATION_PILLS.map((rel) => {
                                          const active = editForm.relation_to_head === rel;
                                          return (
                                            <button
                                              key={rel}
                                              type="button"
                                              onClick={() =>
                                                setEditForm({
                                                  ...editForm,
                                                  relation_to_head: rel,
                                                })
                                              }
                                              className={`px-2.5 py-1 min-h-[28px] rounded-md text-xs font-semibold transition cursor-pointer ${
                                                active
                                                  ? "bg-amber-600 text-white"
                                                  : "bg-stone-100 hover:bg-stone-200 text-stone-800"
                                              }`}
                                            >
                                              {rel}
                                            </button>
                                          );
                                        })}
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                {/* 2. Narodenie a rodinný stav */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk"
                                      ? "2. Narodenie a rodinný stav"
                                      : "2. Birth & Marital Status"}
                                  </h4>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Dátum narodenia" : "Birth date"}
                                      </label>
                                      <input
                                        type="text"
                                        placeholder="DD.MM.RRRR"
                                        value={editForm.birth_or_event_date}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            birth_or_event_date: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Rodinný stav" : "Marital status"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.marital_status}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            marital_status: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Dátum sobáša" : "Marriage date"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.marriage_date}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            marriage_date: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Manžel / ka" : "Spouse"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.spouse_name}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, spouse_name: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Otec" : "Father"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.father_name}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, father_name: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Matka" : "Mother"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.mother_name}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, mother_name: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk"
                                          ? "Počet živonarodených detí"
                                          : "Children born alive"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.children_born_count}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            children_born_count: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Z nich zomrelých" : "Children deceased"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.children_deceased_count}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            children_deceased_count: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                  </div>
                                </div>

                                {/* 3. Rodisko (stĺpec 11) */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk" ? "3. Rodisko (stĺpec 11)" : "3. Birthplace (col 11)"}
                                  </h4>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "a) Obec" : "a) Municipality"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.birth_place_municipality}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            birth_place_municipality: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "b) Okres" : "b) District"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.birth_place_district}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            birth_place_district: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "c) Krajina" : "c) Country"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.birth_place_country}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            birth_place_country: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Celý zápis rodiska" : "Full birthplace raw"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.birth_place_raw}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            birth_place_raw: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                  </div>
                                </div>

                                {/* 4. Prisťahovanie a príslušnosť (stĺpce 12–14) */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk"
                                      ? "4. Prisťahovanie a príslušnosť (stĺpce 12–14)"
                                      : "4. Residency & Citizenship (cols 12–14)"}
                                  </h4>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Kedy sa prisťahoval" : "Moved in date"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.moved_in_date}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            moved_in_date: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Odkiaľ" : "Previous residence"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.previous_residence}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            previous_residence: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Domovská príslušnosť" : "Domicile"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.home_jurisdiction}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            home_jurisdiction: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Štátna príslušnosť" : "Citizenship"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.citizenship}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            citizenship: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                  </div>
                                </div>

                                {/* 5. Národnosť, Náboženstvo a Gramotnosť (stĺpce 15–17) */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk"
                                      ? "5. Národnosť, Náboženstvo a Gramotnosť (stĺpce 15–17)"
                                      : "5. Nationality, Religion & Literacy (cols 15–17)"}
                                  </h4>
                                  <div className="grid grid-cols-1 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk"
                                          ? "Národnosť (materský jazyk)"
                                          : "Nationality (mother tongue)"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.nationality}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            nationality: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Náboženstvo (vyznanie)" : "Religion"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.religion}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, religion: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk"
                                          ? "Gramotnosť (znalosť čítania a písania)"
                                          : "Literacy (read & write)"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.literacy}
                                        onChange={(e) =>
                                          setEditForm({ ...editForm, literacy: e.target.value })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                  </div>
                                </div>

                                {/* 6. Povolanie a poznámky (stĺpce 18–24) */}
                                <div className="bg-white rounded-lg p-3 border border-amber-200/80 space-y-2.5">
                                  <h4 className="font-bold text-amber-950 uppercase tracking-wider text-xs border-b border-stone-100 pb-1">
                                    {lang === "sk"
                                      ? "6. Povolanie a poznámky (stĺpce 18–24)"
                                      : "6. Occupation & Remarks (cols 18–24)"}
                                  </h4>
                                  <div className="grid grid-cols-2 gap-2">
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        Povolanie
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.social_status_or_occupation}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            social_status_or_occupation: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk"
                                          ? "Postavenie v povolaní"
                                          : "Occupation position"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.occupation_position}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            occupation_position: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk"
                                          ? "Závod / zamestnávateľ"
                                          : "Employer / workplace"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.employer_workplace}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            employer_workplace: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div>
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        {lang === "sk" ? "Prítomnosť" : "Presence status"}
                                      </label>
                                      <input
                                        type="text"
                                        value={editForm.presence_status}
                                        onChange={(e) =>
                                          setEditForm({
                                            ...editForm,
                                            presence_status: e.target.value,
                                          })
                                        }
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                    <div className="col-span-2">
                                      <label className="block font-semibold text-stone-700 mb-0.5">
                                        Poznámka (stĺpec 24)
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
                                        className="w-full px-2.5 py-1.5 min-h-[34px] rounded border border-stone-300"
                                      />
                                    </div>
                                  </div>
                                </div>
                              </div>

                              {/* Drawer Footer */}
                              <div className="flex items-center justify-end gap-2 pt-1">
                                <button
                                  type="button"
                                  onClick={() => setEditingMemberId(null)}
                                  className="px-3 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-800 font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                                >
                                  <X className="w-3.5 h-3.5" />
                                  <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleSaveMember(f.id)}
                                  disabled={savingId === f.id}
                                  className="px-4 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                                >
                                  <Save className="w-3.5 h-3.5" />
                                  <span>
                                    {savingId === f.id
                                      ? lang === "sk"
                                        ? "Ukladám…"
                                        : "Saving…"
                                      : lang === "sk"
                                      ? "Uložiť zmeny"
                                      : "Save changes"}
                                  </span>
                                </button>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      </div>

      {/* Compact 1-Line Source & CC BY-NC-ND 4.0 Attribution Strip */}
      <div className="bg-white border border-stone-200 rounded-lg px-3.5 py-2 flex flex-wrap items-center justify-between gap-2 text-xs text-stone-600">
        <div className="flex flex-wrap items-center gap-2">
          <Scale className="w-3.5 h-3.5 text-stone-500 shrink-0" />
          <span>
            {lang === "sk" ? "Zdroj:" : "Source:"}{" "}
            <strong className="text-stone-800">
              Slovenský národný archív (SNA) / Slovakiana
            </strong>
          </span>
          <span>·</span>
          <a
            href={`https://slovakiana.sk/kulturne-objekty/${sheet.slovakiana_id}`}
            target="_blank"
            rel="noreferrer"
            className="text-amber-700 hover:underline font-semibold inline-flex items-center gap-1"
          >
            <span>
              {lang === "sk" ? "Objekt" : "Object"} {sheet.slovakiana_id}
            </span>
            <ExternalLink className="w-3 h-3" />
          </a>
          {sheet.urn_nbn && (
            <>
              <span>·</span>
              <span className="font-mono text-stone-500">
                URN:NBN: {sheet.urn_nbn}
              </span>
            </>
          )}
        </div>
        <a
          href="https://creativecommons.org/licenses/by-nc-nd/4.0/"
          target="_blank"
          rel="noreferrer"
          className="px-2 py-0.5 rounded bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-200 font-semibold inline-flex items-center gap-1"
        >
          <span>
            {lang === "sk" ? "Licencia: CC BY-NC-ND 4.0" : "License: CC BY-NC-ND 4.0"}
          </span>
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>
      </div>
    </div>
  );
}
