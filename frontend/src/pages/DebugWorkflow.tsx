import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertCircle,
  ArrowUp,
  BarChart3,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  Eye,
  FileText,
  ListOrdered,
  Play,
  RefreshCw,
  Send,
  Server,
  Sliders,
  Sparkles,
  Terminal,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
} from "lucide-react";
import type { QueuedTaskItem, TaskQueueSnapshot } from "../App";
import OcrFeedbackControls, {
  type OcrFeedbackItem,
} from "../components/OcrFeedbackControls";
import {
  FUNCTIONAL_FLAG_DESCRIPTORS,
  NAVIGATION_FLAG_DESCRIPTORS,
  fetchBackendFeatureFlags,
  useFeatureFlags,
} from "../featureFlags";

interface TraceItem {
  id: number;
  scan_id: number | null;
  row_id: number | null;
  stage: string;
  model_or_tool?: string;
  model_name?: string;
  prompt_text: string;
  image_path?: string | null;
  response_text?: string | null;
  raw_response?: string | null;
  reasoning_content?: string | null;
  parsed_json: string | null;
  duration_ms: number;
  status: string;
  error_message: string | null;
  created_at: string;
}

interface ScanDetail {
  scan: {
    id: number;
    filename: string;
    status: string;
    detected_year: string | null;
    detected_type?: string | null;
    spine_x?: number;
    overlay_url: string | null;
    header_crop_url: string | null;
  };
  grid_metadata?: {
    spine_x?: number;
    skew_left_deg?: number;
    skew_right_deg?: number;
    col_tilt_left_deg?: number;
    col_tilt_right_deg?: number;
    spine_offset_px?: number;
    slicing_mode?: string;
  };
  rows: Array<{
    id: number;
    row_index: number;
    entry_number: string | null;
    status: string;
    angle_left_deg?: number;
    angle_right_deg?: number;
    raw_ocr_combined: string | null;
    stitched_crop_url: string | null;
    fact?: { id: number } | null;
    ocr_feedback?: OcrFeedbackItem | null;
  }>;
}

interface ModelComboStat {
  model_combo: string;
  ocr_model: string;
  decision_model: string | null;
  extract_model: string;
  layout_model: string | null;
  is_active_combo: boolean;
  total_rated: number;
  correct_count: number;
  incorrect_count: number;
  correctness_rate_pct: number | null;
  rows_processed: number;
  avg_total_ms: number;
  min_total_ms: number;
  max_total_ms: number;
  steps: {
    layout_ms: number;
    scan_layout_ms: number;
    ocr_ms: number;
    ocr_header_ms: number;
    ocr_left_ms: number;
    ocr_right_ms: number;
    decision_ms: number;
    extract_ms: number;
    mcp_ms: number;
  };
  avg_tokens: {
    prompt_tokens: number;
    completion_tokens: number;
    reasoning_tokens: number;
  };
}

interface IndividualStepStat {
  stage: string;
  canonical_stage: string;
  model_or_tool: string;
  calls_count: number;
  ok_count: number;
  error_count: number;
  avg_duration_ms: number;
  min_duration_ms: number;
  max_duration_ms: number;
  total_duration_ms: number;
  avg_prompt_tokens: number;
  avg_completion_tokens: number;
  avg_reasoning_tokens: number;
  rated_rows_count: number;
  correct_rows_count: number;
  incorrect_rows_count: number;
  correctness_rate_pct: number | null;
}

interface FeedbackStatsState {
  summary: {
    total_feedback: number;
    correct_count: number;
    incorrect_count: number;
    correctness_rate_pct: number | null;
    total_rows_analyzed: number;
    avg_total_duration_ms: number;
    avg_steps_ms: {
      layout_ms: number;
      ocr_ms: number;
      ocr_header_ms: number;
      ocr_left_ms: number;
      ocr_right_ms: number;
      decision_ms: number;
      extract_ms: number;
      mcp_ms: number;
    };
    report_files?: {
      markdown_path: string;
      json_path: string;
      relative_markdown_path: string;
      relative_json_path: string;
    };
  };
  model_combos: ModelComboStat[];
  individual_steps: IndividualStepStat[];
}

export type PipelineStageKey = "ocr" | "decision" | "extract" | "layout";

export interface StageAIConfig {
  provider: "local" | "freellm";
  model: string;
  local_model?: string;
  freellm_model?: string;
}

interface AIConfigState {
  provider: "local" | "freellm";
  freellm_model: string;
  freellm_base_url?: string;
  local_base_url?: string;
  base_url?: string;
  local_max_concurrent?: number;
  freellm_concurrent_unlimited?: boolean;
  online?: boolean;
  local_online?: boolean;
  freellm_online?: boolean;
  stages?: Partial<Record<PipelineStageKey, StageAIConfig>>;
  configured?: {
    ocr: string;
    decision: string;
    extract: string;
    layout?: string;
  };
  local_defaults?: Partial<Record<PipelineStageKey, string>>;
  local_models?:
    | Array<{ id?: string } | string>
    | Partial<Record<PipelineStageKey, string>>;
  models?: Array<{ id?: string } | string>;
  freellm_models?: Array<{ id?: string } | string>;
}

interface DebugOverviewState {
  pangolin_public_url?: string;
  ai_server?: {
    online?: boolean;
    local_online?: boolean;
    freellm_online?: boolean;
    provider?: "local" | "freellm";
    base_url?: string;
    freellm_model?: string;
    stages?: Partial<Record<PipelineStageKey, StageAIConfig>>;
    configured?: {
      ocr: string;
      decision: string;
      extract: string;
      layout?: string;
    };
    models?: Array<{ id?: string } | string>;
    local_models?: Array<{ id?: string } | string>;
    freellm_models?: Array<{ id?: string } | string>;
  };
  task_queue?: TaskQueueSnapshot;
  familysearch_mcp?: {
    connected?: boolean;
    authenticated?: boolean;
    token_valid?: boolean;
    has_token_configured?: boolean;
    token_problem?: string | null;
  };
}

interface Props {
  lang: "sk" | "en";
  refreshStatus: () => void;
}

const DEFAULT_FREELLM_MODEL_SUGGESTIONS = [
  "gpt-4o",
  "gpt-4o-mini",
  "claude-3-5-sonnet",
  "gemini-2.5-pro",
  "gemini-2.5-flash",
  "deepseek-chat",
];

const DEFAULT_LOCAL_MODEL_SUGGESTIONS = [
  "glm-ocr-g32-mixed_4_8-mlx",
  "d1-3b-mlx",
  "underdog-saluki-27b-1.0",
];

const STAGE_DESCRIPTORS: Array<{
  key: PipelineStageKey;
  titleSk: string;
  titleEn: string;
  subtitleSk: string;
  subtitleEn: string;
  defaultLocalModel: string;
  defaultFreellmModel: string;
}> = [
  {
    key: "ocr",
    titleSk: "ocr — Hlavičky hárkov, tabuľky prehľadov & voliteľné OCR riadkov",
    titleEn: "ocr — Sheet Headers, Overview Tables & Optional Row OCR",
    subtitleSk:
      "GLM-OCR vždy číta hlavičku hárku a tabuľky prehľadov obvodov (textové OCR riadkov sa dá preskočiť nižšie)",
    subtitleEn:
      "GLM-OCR always reads sheet headers and overview tables (per-row text OCR can be skipped below)",
    defaultLocalModel: "glm-ocr-g32-mixed_4_8-mlx",
    defaultFreellmModel: "gpt-4o-mini",
  },
  {
    key: "decision",
    titleSk:
      "decision — Kontrola čitateľnosti (iba cirkevné matriky; pri sčítaniach sa preskakuje)",
    titleEn:
      "decision — Readability Confidence Check (church registers only; bypassed for census)",
    subtitleSk:
      "Rozhodovanie o istote prečítania v cirkevných matrikách (sčítacie hárky 1930/1940 idú priamo do Extract)",
    subtitleEn:
      "Confidence triage for church register rows (1930/1940 census sheets go directly to Extract)",
    defaultLocalModel: "d1-3b-mlx",
    defaultFreellmModel: "gpt-4o-mini",
  },
  {
    key: "extract",
    titleSk:
      "extract — Priama vizuálna extrakcia z 5 výrezov stĺpcov (1–24) + 100 km register obcí",
    titleEn:
      "extract — Direct Visual Extraction from 5 Column Crops (1–24) + 100 km Gazetteer",
    subtitleSk:
      "Priama extrakcia zo zloženého výrezu 5 skupín stĺpcov (row_XX_extract_composite.jpg, stĺpce 1–24) bez OCR nápovied + konsenzus domácnosti",
    subtitleEn:
      "Direct extraction from 5-band stacked column crops (row_XX_extract_composite.jpg, Cols 1–24) without OCR hints + household consensus",
    defaultLocalModel: "underdog-saluki-27b-1.0",
    defaultFreellmModel: "gpt-4o-mini",
  },
  {
    key: "layout",
    titleSk: "layout — Rozloženie hárku / Prehľad obvodu (Layout)",
    titleEn: "layout — Sheet Layout & District Overview Detection (Layout)",
    subtitleSk:
      "Detekcia typu hárku a hlavičiek prehľadu obvodu (predvolene: Local AI · glm-ocr-g32-mixed_4_8-mlx)",
    subtitleEn:
      "Sheet type classification and district overview header detection (default: Local AI · glm-ocr-g32-mixed_4_8-mlx)",
    defaultLocalModel: "glm-ocr-g32-mixed_4_8-mlx",
    defaultFreellmModel: "gpt-4o-mini",
  },
];

const DEFAULT_STAGE_CONFIGS: Record<PipelineStageKey, StageAIConfig> = {
  ocr: {
    provider: "local",
    model: "glm-ocr-g32-mixed_4_8-mlx",
    local_model: "glm-ocr-g32-mixed_4_8-mlx",
    freellm_model: "gpt-4o-mini",
  },
  decision: {
    provider: "local",
    model: "d1-3b-mlx",
    local_model: "d1-3b-mlx",
    freellm_model: "gpt-4o-mini",
  },
  extract: {
    provider: "local",
    model: "underdog-saluki-27b-1.0",
    local_model: "underdog-saluki-27b-1.0",
    freellm_model: "gpt-4o-mini",
  },
  layout: {
    provider: "local",
    model: "glm-ocr-g32-mixed_4_8-mlx",
    local_model: "glm-ocr-g32-mixed_4_8-mlx",
    freellm_model: "gpt-4o-mini",
  },
};

function extractModelIds(
  fallbacks: string[],
  ...lists: Array<
    Array<{ id?: string } | string> | Record<string, unknown> | undefined
  >
): string[] {
  const seen = new Set<string>();
  for (const list of lists) {
    if (!list) continue;
    if (Array.isArray(list)) {
      for (const item of list) {
        const id = typeof item === "string" ? item : item?.id;
        if (id && id.trim()) {
          seen.add(id.trim());
        }
      }
    } else if (typeof list === "object") {
      for (const val of Object.values(list)) {
        if (typeof val === "string" && val.trim()) {
          seen.add(val.trim());
        }
      }
    }
  }
  for (const fallback of fallbacks) {
    seen.add(fallback);
  }
  return Array.from(seen);
}

function normalizeStagesFromConfig(
  cfg?: AIConfigState | null
): Record<PipelineStageKey, StageAIConfig> {
  const result: Record<PipelineStageKey, StageAIConfig> = {
    ocr: { ...DEFAULT_STAGE_CONFIGS.ocr },
    decision: { ...DEFAULT_STAGE_CONFIGS.decision },
    extract: { ...DEFAULT_STAGE_CONFIGS.extract },
    layout: { ...DEFAULT_STAGE_CONFIGS.layout },
  };
  if (!cfg) return result;

  const fallbackFl = cfg.freellm_model || "gpt-4o-mini";
  for (const desc of STAGE_DESCRIPTORS) {
    const sk = desc.key;
    const rawStage = cfg.stages?.[sk];
    const locDefault = cfg.local_defaults?.[sk] || desc.defaultLocalModel;
    if (rawStage) {
      const prov = rawStage.provider === "freellm" ? "freellm" : "local";
      const localM = rawStage.local_model || locDefault;
      const freeM = rawStage.freellm_model || fallbackFl;
      const activeM =
        rawStage.model || (prov === "freellm" ? freeM : localM);
      result[sk] = {
        provider: prov,
        model: activeM,
        local_model: localM,
        freellm_model: freeM,
      };
    } else {
      const prov = cfg.provider === "freellm" ? "freellm" : "local";
      const confM = cfg.configured?.[sk];
      result[sk] = {
        provider: prov,
        model: confM || (prov === "freellm" ? fallbackFl : locDefault),
        local_model: locDefault,
        freellm_model: fallbackFl,
      };
    }
  }
  return result;
}

export default function DebugWorkflow({ lang, refreshStatus }: Props) {
  const { scanId: routeScanId } = useParams<{ scanId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    flags: featureFlags,
    toggleFlag,
    resetCensusOnly,
    enableAllTabs,
  } = useFeatureFlags();
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [scans, setScans] = useState<Array<{ id: number; filename: string }>>([]);
  const [selectedScanId, setSelectedScanId] = useState<number | null>(() => {
    const raw = routeScanId || searchParams.get("scan") || "";
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  });
  const [scanDetail, setScanDetail] = useState<ScanDetail | null>(null);
  const [selectedTrace, setSelectedTrace] = useState<TraceItem | null>(null);
  const [stageFilter, setStageFilter] = useState<string>(
    () => searchParams.get("stage") || ""
  );
  const [rerunningRowId, setRerunningRowId] = useState<number | null>(null);
  const [rerunningStageName, setRerunningStageName] = useState<string | null>(null);
  const [reslicingScan, setReslicingScan] = useState<boolean>(false);
  const [cacheRev, setCacheRev] = useState<number>(() => Date.now());

  // AI Provider & Model Config State
  const [overview, setOverview] = useState<DebugOverviewState | null>(null);
  const [aiConfig, setAiConfig] = useState<AIConfigState | null>(null);
  const [, setAiProvider] = useState<"local" | "freellm">("local");
  const [freellmModelInput, setFreellmModelInput] = useState<string>("gpt-4o-mini");
  const [stageConfigs, setStageConfigs] = useState<
    Record<PipelineStageKey, StageAIConfig>
  >(() => ({ ...DEFAULT_STAGE_CONFIGS }));
  const [savingAiConfig, setSavingAiConfig] = useState<boolean>(false);
  const [savingStageKey, setSavingStageKey] = useState<string | null>(null);
  const [aiSavedNotice, setAiSavedNotice] = useState<string | null>(null);
  const [aiSavedError, setAiSavedError] = useState<boolean>(false);
  const noticeTimerRef = useRef<number | null>(null);

  const showAiNotice = (msg: string, isError = false) => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
    }
    setAiSavedError(isError);
    setAiSavedNotice(msg);
    noticeTimerRef.current = window.setTimeout(() => {
      setAiSavedNotice(null);
      setAiSavedError(false);
      noticeTimerRef.current = null;
    }, 4500);
  };

  // Playground State
  const [customModel, setCustomModel] = useState<string>("underdog-saluki-27b-1.0");
  const [customPrompt, setCustomPrompt] = useState<string>("");
  const [customRowId, setCustomRowId] = useState<string>(
    () => searchParams.get("row") || ""
  );
  const [customUseImage, setCustomUseImage] = useState<boolean>(false);
  const [runningCustomPrompt, setRunningCustomPrompt] = useState<boolean>(false);
  const [customResult, setCustomResult] = useState<string | null>(null);
  const [customError, setCustomError] = useState<string | null>(null);

  // OCR Feedback & Model Combo Statistics State
  const [feedbackList, setFeedbackList] = useState<OcrFeedbackItem[]>([]);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStatsState | null>(
    null
  );
  const [feedbackVerdictFilter, setFeedbackVerdictFilter] = useState<
    "" | "incorrect" | "correct"
  >("");
  const [selectedFeedbackItem, setSelectedFeedbackItem] =
    useState<OcrFeedbackItem | null>(null);
  const [reportMarkdown, setReportMarkdown] = useState<string>("");
  const [showReportPreview, setShowReportPreview] = useState<boolean>(false);
  const [copiedReport, setCopiedReport] = useState<boolean>(false);
  const [editingFeedbackId, setEditingFeedbackId] = useState<number | null>(
    null
  );
  const [editingCommentText, setEditingCommentText] = useState<string>("");
  const [feedbackResetVersion, setFeedbackResetVersion] = useState<number>(0);
  const [refreshingDebug, setRefreshingDebug] = useState<boolean>(false);
  const [loadingScanDetail, setLoadingScanDetail] = useState<boolean>(false);

  // Task Queue State
  const [taskQueue, setTaskQueue] = useState<TaskQueueSnapshot>({
    active_task: null,
    running_tasks: [],
    queued_tasks: [],
    history: [],
    counts: {
      running: 0,
      queued: 0,
      completed: 0,
      cancelled: 0,
      error: 0,
      total_active_and_queued: 0,
      local_max_concurrent: 5,
      active_local_requests: 0,
      waiting_local_requests: 0,
      active_freellm_requests: 0,
    },
  });
  const [localMaxConcurrentInput, setLocalMaxConcurrentInput] = useState<number>(5);
  const [busyTaskActionId, setBusyTaskActionId] = useState<string | null>(null);

  const loadTaskQueue = async () => {
    try {
      const res = await fetch("/api/debug/tasks");
      if (res.ok) {
        const data: TaskQueueSnapshot = await res.json();
        setTaskQueue(data);
        const lim =
          data.concurrency?.local_max_concurrent ??
          data.counts?.local_max_concurrent;
        if (typeof lim === "number" && lim >= 1) {
          setLocalMaxConcurrentInput(lim);
        }
      }
    } catch {
      // ignore transient errors
    }
  };

  const handleUpdateLocalMaxConcurrent = async (nextLimit: number) => {
    const clamped = Math.max(1, Math.min(64, Math.round(nextLimit) || 5));
    setLocalMaxConcurrentInput(clamped);
    await postAiConfigUpdate(
      { local_max_concurrent: clamped },
      lang === "sk"
        ? `Limit súbežných požiadaviek pre Local AI nastavený na ${clamped}`
        : `Local AI concurrent request limit set to ${clamped}`,
      "local_max_concurrent"
    );
    await loadTaskQueue();
  };

  const handleCancelQueuedOrActiveTask = async (taskId: string) => {
    setBusyTaskActionId(taskId);
    try {
      const res = await fetch(`/api/debug/tasks/${encodeURIComponent(taskId)}/cancel`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.task_queue) {
          setTaskQueue(data.task_queue);
        } else {
          await loadTaskQueue();
        }
        refreshStatus();
      }
    } finally {
      setBusyTaskActionId(null);
    }
  };

  const handlePrioritizeQueuedTask = async (taskId: string) => {
    setBusyTaskActionId(taskId);
    try {
      const res = await fetch(`/api/debug/tasks/${encodeURIComponent(taskId)}/prioritize`, {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.task_queue) {
          setTaskQueue(data.task_queue);
        } else {
          await loadTaskQueue();
        }
        refreshStatus();
      }
    } finally {
      setBusyTaskActionId(null);
    }
  };

  const handleClearQueuedTasks = async () => {
    setBusyTaskActionId("clear_queued");
    try {
      const res = await fetch("/api/debug/tasks/clear-queued", {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.task_queue) {
          setTaskQueue(data.task_queue);
        } else {
          await loadTaskQueue();
        }
        refreshStatus();
      }
    } finally {
      setBusyTaskActionId(null);
    }
  };

  const handleClearTaskHistory = async () => {
    setBusyTaskActionId("clear_history");
    try {
      const res = await fetch("/api/debug/tasks/clear-history", {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.task_queue) {
          setTaskQueue(data.task_queue);
        } else {
          await loadTaskQueue();
        }
      }
    } finally {
      setBusyTaskActionId(null);
    }
  };

  const loadFeedbackAndStats = async (
    filterOverride?: "" | "incorrect" | "correct"
  ) => {
    const activeFilter =
      filterOverride !== undefined ? filterOverride : feedbackVerdictFilter;
    try {
      const url = `/api/debug/feedback?limit=200${
        activeFilter ? `&verdict=${encodeURIComponent(activeFilter)}` : ""
      }`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        const items: OcrFeedbackItem[] = data.feedback || [];
        setFeedbackList(items);
        if (data.stats) {
          setFeedbackStats(data.stats);
        }
        if (typeof data.report_markdown === "string") {
          setReportMarkdown(data.report_markdown);
        }
        setSelectedFeedbackItem((prev) => {
          if (!prev) return items[0] ?? null;
          return items.find((i) => i.id === prev.id) ?? items[0] ?? null;
        });
        setEditingFeedbackId((prev) => {
          if (prev && !items.some((i) => i.id === prev)) {
            setEditingCommentText("");
            return null;
          }
          return prev;
        });
      }
    } catch {
      // Ignore transient poll errors
    }
  };

  const handleUpdateFeedbackItem = async (
    item: OcrFeedbackItem,
    nextVerdict: "correct" | "incorrect",
    nextComment?: string
  ) => {
    try {
      const res = await fetch("/api/debug/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          verdict: nextVerdict,
          comment:
            nextComment !== undefined ? nextComment.trim() || null : item.comment,
          target_type: item.target_type,
          row_id: item.row_id,
          scan_id: item.scan_id,
          census_sheet_id: item.census_sheet_id,
          person_fact_id: item.person_fact_id,
          overview_house_id: item.overview_house_id,
          municipal_district_id: item.municipal_district_id,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const saved: OcrFeedbackItem | undefined = data.feedback;
        setEditingFeedbackId(null);
        if (data.stats) setFeedbackStats(data.stats);
        if (typeof data.report_markdown === "string") {
          setReportMarkdown(data.report_markdown);
        }
        if (saved) {
          setFeedbackList((prev) => {
            const updated = prev.some((i) => i.id === saved.id)
              ? prev.map((i) => (i.id === saved.id ? saved : i))
              : [saved, ...prev];
            return feedbackVerdictFilter
              ? updated.filter((i) => i.verdict === feedbackVerdictFilter)
              : updated;
          });
          setSelectedFeedbackItem((prev) =>
            prev?.id === saved.id ? saved : prev
          );
          setScanDetail((prev) =>
            prev && saved.row_id
              ? {
                  ...prev,
                  rows: prev.rows.map((r) =>
                    r.id === saved.row_id ? { ...r, ocr_feedback: saved } : r
                  ),
                }
              : prev
          );
        }
      }
    } catch {
      // ignore
    }
  };

  const handleDeleteFeedbackItem = async (feedbackId: number) => {
    try {
      const res = await fetch(`/api/debug/feedback/${feedbackId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        const data = await res.json();
        if (data.stats) setFeedbackStats(data.stats);
        if (typeof data.report_markdown === "string") {
          setReportMarkdown(data.report_markdown);
        }
        setFeedbackList((prev) => {
          const remaining = prev.filter((i) => i.id !== feedbackId);
          setSelectedFeedbackItem((sel) =>
            sel?.id === feedbackId ? remaining[0] ?? null : sel
          );
          return remaining;
        });
        setScanDetail((prev) =>
          prev
            ? {
                ...prev,
                rows: prev.rows.map((r) =>
                  r.ocr_feedback?.id === feedbackId
                    ? { ...r, ocr_feedback: null }
                    : r
                ),
              }
            : prev
        );
      }
    } catch {
      // ignore
    }
  };

  const handleCopyReportForLlm = async (textToCopy?: string) => {
    const content = textToCopy || reportMarkdown;
    if (!content) return;
    try {
      await navigator.clipboard.writeText(content);
      setCopiedReport(true);
      window.setTimeout(() => setCopiedReport(false), 3000);
    } catch {
      // ignore clipboard error
    }
  };

  useEffect(() => {
    if (routeScanId) {
      const parsed = Number(routeScanId);
      if (Number.isFinite(parsed) && parsed > 0) {
        setSelectedScanId(parsed);
      }
    }
  }, [routeScanId]);

  useEffect(() => {
    const next = new URLSearchParams();
    if (selectedScanId !== null) next.set("scan", String(selectedScanId));
    if (selectedTrace?.id) next.set("trace", String(selectedTrace.id));
    if (stageFilter) next.set("stage", stageFilter);
    if (customRowId.trim()) next.set("row", customRowId.trim());
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [selectedScanId, selectedTrace?.id, stageFilter, customRowId]);

  const applyLoadedAiConfig = (cfgData: AIConfigState) => {
    setAiConfig(cfgData);
    if (typeof cfgData.local_max_concurrent === "number" && cfgData.local_max_concurrent >= 1) {
      setLocalMaxConcurrentInput(cfgData.local_max_concurrent);
    }
    const prov = cfgData.provider === "freellm" ? "freellm" : "local";
    setAiProvider(prov);
    const normalizedStages = normalizeStagesFromConfig(cfgData);
    setStageConfigs(normalizedStages);
    if (cfgData.freellm_model) {
      setFreellmModelInput(cfgData.freellm_model);
    } else if (normalizedStages.extract?.freellm_model) {
      setFreellmModelInput(normalizedStages.extract.freellm_model);
    }
    const activeExtractModel =
      normalizedStages.extract?.model ||
      (prov === "freellm"
        ? cfgData.freellm_model
        : cfgData.configured?.extract || "underdog-saluki-27b-1.0");
    if (activeExtractModel) {
      setCustomModel(activeExtractModel);
    }
  };

  const loadAiConfig = async () => {
    try {
      const [cfgRes, ovRes] = await Promise.all([
        fetch("/api/debug/ai-config"),
        fetch("/api/debug/overview"),
      ]);
      if (ovRes.ok) {
        const ovData = await ovRes.json();
        setOverview(ovData);
        if (ovData.task_queue) {
          setTaskQueue(ovData.task_queue);
        }
      }
      if (cfgRes.ok) {
        const cfgData: AIConfigState = await cfgRes.json();
        applyLoadedAiConfig(cfgData);
      }
    } catch {
      // Ignore transient network errors
    }
  };

  const loadDebug = async () => {
    const [trRes, scRes] = await Promise.all([
      fetch(
        `/api/debug/traces?limit=80${
          stageFilter ? `&stage=${encodeURIComponent(stageFilter)}` : ""
        }`
      ),
      fetch("/api/scans"),
    ]);
    if (trRes.ok) {
      const data = await trRes.json();
      const loadedTraces: TraceItem[] = data.traces || [];
      setTraces(loadedTraces);
      setSelectedTrace((prev) => {
        const urlTraceId = Number(searchParams.get("trace"));
        if (!prev && Number.isFinite(urlTraceId) && urlTraceId > 0) {
          const matchedUrlTrace = loadedTraces.find((t) => t.id === urlTraceId);
          if (matchedUrlTrace) return matchedUrlTrace;
        }
        if (!prev) return loadedTraces[0] ?? null;
        return loadedTraces.find((t: TraceItem) => t.id === prev.id) ?? prev;
      });
    }
    if (scRes.ok) {
      const sData = await scRes.json();
      const list = sData.scans || [];
      setScans(list);
      setSelectedScanId((prev) => prev ?? list[0]?.id ?? null);
    }
  };

  useEffect(() => {
    loadAiConfig();
    void loadTaskQueue();
    void loadFeedbackAndStats();
    void fetchBackendFeatureFlags();
  }, []);

  useEffect(() => {
    const onQueueEvent = (e: Event) => {
      const custom = e as CustomEvent<TaskQueueSnapshot>;
      if (custom.detail) {
        setTaskQueue(custom.detail);
      }
    };
    const onPipelineEvent = () => {
      void loadTaskQueue();
    };
    window.addEventListener("matrika:task-queue", onQueueEvent);
    window.addEventListener("matrika:pipeline-state", onPipelineEvent);
    return () => {
      window.removeEventListener("matrika:task-queue", onQueueEvent);
      window.removeEventListener("matrika:pipeline-state", onPipelineEvent);
    };
  }, []);

  const hasActiveOrQueuedTasks = (taskQueue.counts?.total_active_and_queued || 0) > 0;
  useEffect(() => {
    const pollMs = hasActiveOrQueuedTasks ? 1500 : 6000;
    const t = window.setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadTaskQueue();
    }, pollMs);
    return () => window.clearInterval(t);
  }, [hasActiveOrQueuedTasks]);

  useEffect(() => {
    loadDebug();
    void loadFeedbackAndStats();
    const interval = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadDebug();
      void loadFeedbackAndStats();
    }, 15000);
    return () => clearInterval(interval);
  }, [stageFilter, feedbackVerdictFilter]);

  useEffect(() => {
    if (!selectedScanId) return;
    let cancelled = false;
    setLoadingScanDetail(true);
    fetch(`/api/scans/${selectedScanId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled && d) setScanDetail(d);
      })
      .finally(() => {
        if (!cancelled) setLoadingScanDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedScanId]);

  const applyOptimisticStages = (
    nextStages: Record<PipelineStageKey, StageAIConfig>,
    flModelFallback?: string
  ) => {
    setStageConfigs(nextStages);
    const allProvs = new Set(
      STAGE_DESCRIPTORS.map((d) => nextStages[d.key]?.provider || "local")
    );
    const nextGlobalProvider: "local" | "freellm" =
      allProvs.size === 1 && allProvs.has("freellm")
        ? "freellm"
        : allProvs.size === 1 && allProvs.has("local")
        ? "local"
        : nextStages.extract?.provider || "local";
    setAiProvider(nextGlobalProvider);
    const nextFlModel =
      flModelFallback ||
      nextStages.extract?.freellm_model ||
      freellmModelInput ||
      "gpt-4o-mini";
    setFreellmModelInput(nextFlModel);
    if (nextStages.extract?.model) {
      setCustomModel(nextStages.extract.model);
    }
    setAiConfig((prev) => ({
      ...(prev || {
        provider: nextGlobalProvider,
        freellm_model: nextFlModel,
      }),
      provider: nextGlobalProvider,
      freellm_model: nextFlModel,
      stages: nextStages,
      configured: {
        ocr: nextStages.ocr.model,
        decision: nextStages.decision.model,
        extract: nextStages.extract.model,
        layout: nextStages.layout.model,
      },
    }));
  };

  const postAiConfigUpdate = async (
    bodyPayload: Record<string, unknown>,
    noticeLabel: string,
    stageKeyTag?: string
  ) => {
    setSavingAiConfig(true);
    if (stageKeyTag) setSavingStageKey(stageKeyTag);
    showAiNotice(`✓ ${noticeLabel}`, false);
    try {
      const res = await fetch("/api/debug/ai-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyPayload),
      });
      if (res.ok) {
        const updated: AIConfigState = await res.json();
        applyLoadedAiConfig(updated);
        showAiNotice(`✓ ${noticeLabel}`, false);
        const ovRes = await fetch("/api/debug/overview");
        if (ovRes.ok) {
          setOverview(await ovRes.json());
        }
        refreshStatus();
      } else {
        showAiNotice(
          lang === "sk"
            ? `Chyba uloženia AI konfigurácie (HTTP ${res.status})`
            : `Failed to save AI configuration (HTTP ${res.status})`,
          true
        );
      }
    } catch {
      showAiNotice(
        lang === "sk"
          ? "Nepodarilo sa spojiť so serverom /api/debug/ai-config"
          : "Could not connect to /api/debug/ai-config",
        true
      );
    } finally {
      setSavingAiConfig(false);
      setSavingStageKey(null);
    }
  };

  const handleApplyPreset = async (
    preset: "hybrid" | "all_local" | "all_freellm"
  ) => {
    const flModel =
      stageConfigs.extract?.freellm_model ||
      freellmModelInput.trim() ||
      "gpt-4o-mini";

    const nextStages: Record<PipelineStageKey, StageAIConfig> = {
      ocr: { ...stageConfigs.ocr },
      decision: { ...stageConfigs.decision },
      extract: { ...stageConfigs.extract },
      layout: { ...stageConfigs.layout },
    };

    for (const desc of STAGE_DESCRIPTORS) {
      const sk = desc.key;
      const cur = nextStages[sk] || DEFAULT_STAGE_CONFIGS[sk];
      const locM =
        cur.local_model ||
        DEFAULT_STAGE_CONFIGS[sk].local_model ||
        desc.defaultLocalModel;
      const freeM = cur.freellm_model || flModel || desc.defaultFreellmModel;
      const targetProv: "local" | "freellm" =
        preset === "all_local"
          ? "local"
          : preset === "all_freellm"
          ? "freellm"
          : sk === "extract"
          ? "freellm"
          : "local";
      nextStages[sk] = {
        provider: targetProv,
        model: targetProv === "freellm" ? freeM : locM,
        local_model: locM,
        freellm_model: freeM,
      };
    }

    applyOptimisticStages(nextStages, flModel);

    const presetLabel =
      preset === "hybrid"
        ? `Hybrid režim (OCR + Decision lokálne, Extrakcia cez OpenAI/Cloud: ${flModel})`
        : preset === "all_local"
        ? "Všetky fázy prepnuté na Local AI (http://localhost:1234/v1)"
        : `Všetky fázy prepnuté na Cloud API (${flModel})`;
    await postAiConfigUpdate(
      {
        preset,
        freellm_model: flModel,
      },
      presetLabel,
      `preset_${preset}`
    );
  };

  const handleStageProviderToggle = async (
    stageKey: PipelineStageKey,
    nextProvider: "local" | "freellm"
  ) => {
    const current = stageConfigs[stageKey] || DEFAULT_STAGE_CONFIGS[stageKey];
    const nextModel =
      nextProvider === "freellm"
        ? current.freellm_model || freellmModelInput || "gpt-4o-mini"
        : current.local_model || DEFAULT_STAGE_CONFIGS[stageKey].local_model || "";

    const updatedStage: StageAIConfig = {
      ...current,
      provider: nextProvider,
      model: nextModel,
      local_model:
        nextProvider === "local"
          ? nextModel
          : current.local_model || DEFAULT_STAGE_CONFIGS[stageKey].local_model,
      freellm_model:
        nextProvider === "freellm"
          ? nextModel
          : current.freellm_model || freellmModelInput || "gpt-4o-mini",
    };

    const nextStages: Record<PipelineStageKey, StageAIConfig> = {
      ...stageConfigs,
      [stageKey]: updatedStage,
    };
    applyOptimisticStages(nextStages);

    await postAiConfigUpdate(
      {
        stage: stageKey,
        provider: nextProvider,
        model: nextModel,
        local_model: updatedStage.local_model,
        freellm_model: updatedStage.freellm_model,
        stages: nextStages,
      },
      `Fáza ${stageKey}: ${
        nextProvider === "freellm" ? "Cloud API" : "Local AI"
      } (${nextModel})`,
      stageKey
    );
  };

  const handleStageModelChange = (
    stageKey: PipelineStageKey,
    nextModelValue: string
  ) => {
    setStageConfigs((prev) => {
      const cur = prev[stageKey] || DEFAULT_STAGE_CONFIGS[stageKey];
      const isFl = cur.provider === "freellm";
      return {
        ...prev,
        [stageKey]: {
          ...cur,
          model: nextModelValue,
          local_model: isFl ? cur.local_model : nextModelValue,
          freellm_model: isFl ? nextModelValue : cur.freellm_model,
        },
      };
    });
    if (stageKey === "extract" && stageConfigs.extract.provider === "freellm") {
      setFreellmModelInput(nextModelValue);
    }
  };

  const handleSaveSingleStage = async (stageKey: PipelineStageKey) => {
    const cur = stageConfigs[stageKey] || DEFAULT_STAGE_CONFIGS[stageKey];
    const trimmedModel =
      cur.model.trim() ||
      (cur.provider === "freellm"
        ? cur.freellm_model || "gpt-4o-mini"
        : cur.local_model || DEFAULT_STAGE_CONFIGS[stageKey].local_model || "");
    await postAiConfigUpdate(
      {
        stage: stageKey,
        provider: cur.provider,
        model: trimmedModel,
        local_model: cur.provider === "local" ? trimmedModel : cur.local_model,
        freellm_model:
          cur.provider === "freellm" ? trimmedModel : cur.freellm_model,
        stages: {
          ...stageConfigs,
          [stageKey]: {
            ...cur,
            model: trimmedModel,
          },
        },
      },
      `Uložené (${stageKey}: ${
        cur.provider === "freellm" ? "Cloud API" : "Local AI"
      } · ${trimmedModel})`,
      stageKey
    );
  };

  const handleSaveAllStages = async () => {
    await postAiConfigUpdate(
      {
        stages: stageConfigs,
        freellm_model:
          stageConfigs.extract?.freellm_model || freellmModelInput || "gpt-4o-mini",
      },
      lang === "sk"
        ? "Konfigurácia všetkých 4 AI fáz bola uložená"
        : "Configuration of all 4 AI stages saved",
      "all"
    );
  };

  const handleRunCustomPrompt = async () => {
    if (!customPrompt.trim()) return;
    setRunningCustomPrompt(true);
    setCustomResult(null);
    setCustomError(null);
    try {
      const res = await fetch("/api/debug/custom-prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: customModel.trim() || freellmModelInput.trim() || "gpt-4o-mini",
          prompt: customPrompt,
          row_id: customRowId.trim() ? Number(customRowId) : null,
          use_row_image: customUseImage,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.error) {
          setCustomError(String(data.error));
        }
        setCustomResult(data.content || JSON.stringify(data, null, 2));
        await loadDebug();
      } else {
        const errData = await res.json().catch(() => ({}));
        setCustomError(
          errData.detail ||
            errData.error ||
            (lang === "sk"
              ? `Chyba volania modelu (HTTP ${res.status})`
              : `Model request failed (HTTP ${res.status})`)
        );
      }
    } catch {
      setCustomError(
        lang === "sk"
          ? "Chyba spojenia so serverom pri odosielaní promptu."
          : "Network error while sending custom prompt."
      );
    } finally {
      setRunningCustomPrompt(false);
    }
  };

  const handleRerunRow = async (rowId: number, stage: string) => {
    setRerunningRowId(rowId);
    setRerunningStageName(stage);
    setFeedbackResetVersion((v) => v + 1);
    setScanDetail((prev) =>
      prev
        ? {
            ...prev,
            rows: prev.rows.map((r) =>
              r.id === rowId ? { ...r, ocr_feedback: null } : r
            ),
          }
        : prev
    );
    try {
      await fetch(`/api/debug/rerun-stage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row_id: rowId, stage }),
      });
      if (selectedScanId) {
        const res = await fetch(`/api/scans/${selectedScanId}`);
        if (res.ok) {
          setScanDetail(await res.json());
        }
      }
      setFeedbackResetVersion((v) => v + 1);
      await loadDebug();
      await loadFeedbackAndStats();
      refreshStatus();
    } finally {
      setRerunningRowId(null);
      setRerunningStageName(null);
    }
  };

  const handleResliceScan = async () => {
    if (!selectedScanId) return;
    setReslicingScan(true);
    setFeedbackResetVersion((v) => v + 1);
    setScanDetail((prev) =>
      prev
        ? {
            ...prev,
            rows: prev.rows.map((r) => ({ ...r, ocr_feedback: null })),
          }
        : prev
    );
    try {
      await fetch(`/api/scans/${selectedScanId}/slice`, { method: "POST" });
      const res = await fetch(`/api/scans/${selectedScanId}`);
      if (res.ok) {
        setScanDetail(await res.json());
      }
      setCacheRev(Date.now());
      setFeedbackResetVersion((v) => v + 1);
      await loadDebug();
      await loadFeedbackAndStats();
      refreshStatus();
    } finally {
      setReslicingScan(false);
    }
  };

  const localModelSuggestions = extractModelIds(
    DEFAULT_LOCAL_MODEL_SUGGESTIONS,
    aiConfig?.local_models,
    aiConfig?.local_defaults,
    overview?.ai_server?.local_models,
    aiConfig?.models,
    overview?.ai_server?.models
  );

  const freellmModelSuggestions = extractModelIds(
    DEFAULT_FREELLM_MODEL_SUGGESTIONS,
    aiConfig?.freellm_models,
    overview?.ai_server?.freellm_models,
    aiConfig?.models,
    overview?.ai_server?.models
  );

  const modelSuggestions = extractModelIds(
    [...DEFAULT_FREELLM_MODEL_SUGGESTIONS, ...DEFAULT_LOCAL_MODEL_SUGGESTIONS],
    aiConfig?.freellm_models,
    aiConfig?.local_models,
    aiConfig?.models,
    overview?.ai_server?.models
  );

  const activeConfigured =
    aiConfig?.configured || overview?.ai_server?.configured || null;
  const isServerOnline =
    aiConfig?.online ?? overview?.ai_server?.online ?? false;
  const isLocalOnline =
    aiConfig?.local_online ?? overview?.ai_server?.local_online ?? isServerOnline;
  const isFreellmOnline =
    aiConfig?.freellm_online ?? overview?.ai_server?.freellm_online ?? true;

  const isHybridPresetActive =
    stageConfigs.ocr.provider === "local" &&
    stageConfigs.decision.provider === "local" &&
    stageConfigs.layout.provider === "local" &&
    stageConfigs.extract.provider === "freellm";
  const isAllLocalPresetActive =
    stageConfigs.ocr.provider === "local" &&
    stageConfigs.decision.provider === "local" &&
    stageConfigs.extract.provider === "local" &&
    stageConfigs.layout.provider === "local";
  const isAllFreellmPresetActive =
    stageConfigs.ocr.provider === "freellm" &&
    stageConfigs.decision.provider === "freellm" &&
    stageConfigs.extract.provider === "freellm" &&
    stageConfigs.layout.provider === "freellm";

  const [activeConsoleTab, setActiveConsoleTab] = useState<
    "all" | "queue" | "ai" | "benchmark" | "flags" | "inspector"
  >("all");

  const formatTaskCategoryBadge = (cat: string) => {
    switch (cat) {
      case "ocr":
        return {
          labelSk: "AI OCR",
          labelEn: "AI OCR",
          cls: "bg-amber-100 text-amber-900 border-amber-300",
        };
      case "familysearch":
        return {
          labelSk: "FamilySearch MCP",
          labelEn: "FamilySearch MCP",
          cls: "bg-emerald-100 text-emerald-900 border-emerald-300",
        };
      case "harvest":
        return {
          labelSk: "Slovakiana",
          labelEn: "Slovakiana",
          cls: "bg-blue-100 text-blue-900 border-blue-300",
        };
      default:
        return {
          labelSk: "Debug",
          labelEn: "Debug",
          cls: "bg-purple-100 text-purple-900 border-purple-300",
        };
    }
  };

  const formatTaskStatusBadge = (st: string) => {
    switch (st) {
      case "running":
        return {
          labelSk: "Prebieha",
          labelEn: "Running",
          cls: "bg-amber-600 text-white",
        };
      case "queued":
        return {
          labelSk: "Vo fronte",
          labelEn: "Queued",
          cls: "bg-stone-800 text-amber-300",
        };
      case "completed":
        return {
          labelSk: "Dokončené",
          labelEn: "Completed",
          cls: "bg-emerald-100 text-emerald-900 border border-emerald-300",
        };
      case "cancelled":
        return {
          labelSk: "Zrušené",
          labelEn: "Cancelled",
          cls: "bg-stone-200 text-stone-700 border border-stone-300",
        };
      case "paused":
        return {
          labelSk: "Pozastavené",
          labelEn: "Paused",
          cls: "bg-amber-100 text-amber-900 border border-amber-300",
        };
      case "error":
        return {
          labelSk: "Chyba",
          labelEn: "Error",
          cls: "bg-red-100 text-red-900 border border-red-300",
        };
      default:
        return {
          labelSk: st,
          labelEn: st,
          cls: "bg-stone-100 text-stone-700",
        };
    }
  };

  return (
    <div className="space-y-5 text-sm">
      {/* Global Model Datalists for Local AI and Cloud API */}
      <datalist id="local-models-datalist">
        {localModelSuggestions.map((m) => (
          <option key={m} value={m} />
        ))}
      </datalist>
      <datalist id="freellm-models-datalist">
        {freellmModelSuggestions.map((m) => (
          <option key={m} value={m} />
        ))}
      </datalist>
      <datalist id="all-models-datalist">
        {modelSuggestions.map((m) => (
          <option key={m} value={m} />
        ))}
      </datalist>

      {/* Mockup D: Dedicated Admin & Debug Console Header + Admin Overlay Toggle + Tab Bar */}
      <div className="bg-white rounded-xl p-4 border border-stone-200 shadow-2xs space-y-3.5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-lg font-bold text-stone-900 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-amber-600" />
                <span>
                  {lang === "sk"
                    ? "Admin & Debug Konzola"
                    : "Admin & Debug Console"}
                </span>
              </h1>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                  isLocalOnline
                    ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                    : "bg-red-50 text-red-800 border-red-200"
                }`}
              >
                Local AI: {isLocalOnline ? "Online" : "Offline"}
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                  isFreellmOnline
                    ? "bg-amber-50 text-amber-800 border-amber-200"
                    : "bg-stone-100 text-stone-600 border-stone-200"
                }`}
              >
                Cloud API:{" "}
                {isFreellmOnline
                  ? lang === "sk"
                    ? "Pripravené"
                    : "Ready"
                  : lang === "sk"
                  ? "Nedostupné"
                  : "Unavailable"}
              </span>
            </div>
            <p className="text-xs text-stone-600">
              {lang === "sk"
                ? "Centrálna správa AI modelov, kvality OCR, príznakov funkcií (Feature Flags) a nízkoúrovňového inšpektora."
                : "Central administration for AI models, OCR quality benchmarks, Feature Flags, and low-level scan inspection."}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              data-testid="debug-admin-overlay-toggle"
              onClick={() => toggleFlag("admin_debug_overlay")}
              className={`px-3.5 py-2 rounded-lg text-xs font-bold inline-flex items-center gap-2 cursor-pointer transition border ${
                featureFlags.admin_debug_overlay
                  ? "bg-emerald-600 hover:bg-emerald-700 text-white border-emerald-700 shadow-2xs"
                  : "bg-stone-100 hover:bg-stone-200 text-stone-800 border-stone-300"
              }`}
            >
              <Eye className="w-4 h-4" />
              <span>
                {featureFlags.admin_debug_overlay
                  ? lang === "sk"
                    ? "Režim Admin Overlay v aplikácii: ZAPNUTÝ"
                    : "In-App Admin Overlay Mode: ON"
                  : lang === "sk"
                  ? "Režim Admin Overlay v aplikácii: VYPNUTÝ (Zapnúť)"
                  : "In-App Admin Overlay Mode: OFF (Turn ON)"}
              </span>
            </button>
          </div>
        </div>

        {/* 5-Tab Bar (with All Sections default so all tools and E2E selectors are accessible) */}
        <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-stone-200">
          {(
            [
              {
                id: "all",
                labelSk: "Všetky sekcie",
                labelEn: "All Sections",
                badge: null,
              },
              {
                id: "queue",
                labelSk: "Fronta úloh (Queue)",
                labelEn: "Task Queue",
                badge: taskQueue.counts?.total_active_and_queued || null,
              },
              {
                id: "ai",
                labelSk: "1. AI Modely a Fázy",
                labelEn: "1. AI Models & Stages",
                badge: null,
              },
              {
                id: "benchmark",
                labelSk: "2. Kvalita OCR & Benchmark",
                labelEn: "2. OCR Quality & Benchmark",
                badge: null,
              },
              {
                id: "flags",
                labelSk: "3. Feature Flags",
                labelEn: "3. Feature Flags",
                badge: null,
              },
              {
                id: "inspector",
                labelSk: "4. Inšpektor skenov a stôp",
                labelEn: "4. Scan & Trace Inspector",
                badge: null,
              },
            ] as const
          ).map((tab) => {
            const active = activeConsoleTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                data-testid={`debug-tab-${tab.id}`}
                onClick={() => setActiveConsoleTab(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer inline-flex items-center gap-1.5 transition ${
                  active
                    ? "bg-amber-600 text-white shadow-2xs"
                    : "bg-stone-100 hover:bg-stone-200 text-stone-700"
                }`}
              >
                <span>{lang === "sk" ? tab.labelSk : tab.labelEn}</span>
                {tab.badge !== null && tab.badge > 0 && (
                  <span
                    className={`px-1.5 py-0.5 rounded-full text-[11px] font-bold leading-none ${
                      active
                        ? "bg-white text-amber-800"
                        : "bg-amber-600 text-white"
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Task Queue & Background Pipeline Card */}
      {(activeConsoleTab === "all" || activeConsoleTab === "queue") && (
        <div
          data-testid="debug-task-queue-card"
          className="bg-white rounded-xl p-4 border border-stone-300 shadow-2xs space-y-4"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-base font-bold text-stone-900 flex items-center gap-2">
                  <ListOrdered className="w-5 h-5 text-amber-600" />
                  <span>
                    {lang === "sk"
                      ? "Fronta úloh a spracovania (Task Queue)"
                      : "Task Queue & Background Pipeline"}
                  </span>
                </h2>
                <span
                  data-testid="debug-queue-running-badge"
                  className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                    taskQueue.counts.running > 0
                      ? "bg-amber-100 text-amber-900 border-amber-300"
                      : "bg-stone-100 text-stone-700 border-stone-200"
                  }`}
                >
                  {lang === "sk" ? "Práve beží:" : "Running:"}{" "}
                  <strong data-testid="debug-queue-running-count">
                    {taskQueue.counts.running}
                  </strong>
                </span>
                <span
                  data-testid="debug-queue-waiting-badge"
                  className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${
                    taskQueue.counts.queued > 0
                      ? "bg-stone-900 text-amber-300 border-stone-800"
                      : "bg-stone-100 text-stone-700 border-stone-200"
                  }`}
                >
                  {lang === "sk" ? "Čaká vo fronte:" : "Waiting in queue:"}{" "}
                  <strong data-testid="debug-queue-waiting-count">
                    {taskQueue.counts.queued}
                  </strong>
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-900 border border-emerald-200">
                  {lang === "sk" ? "Dokončené:" : "Completed:"}{" "}
                  <strong data-testid="debug-queue-completed-count">
                    {taskQueue.counts.completed}
                  </strong>
                </span>
                {(taskQueue.counts.cancelled > 0 || taskQueue.counts.error > 0) && (
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-stone-100 text-stone-700 border border-stone-300">
                    {lang === "sk" ? "Zrušené / Chyby:" : "Cancelled / Errors:"}{" "}
                    <strong>
                      {taskQueue.counts.cancelled + taskQueue.counts.error}
                    </strong>
                  </span>
                )}
                <span
                  data-testid="debug-queue-local-concurrency-badge"
                  className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-900 border border-blue-200 inline-flex items-center gap-1"
                >
                  <span>
                    {lang === "sk" ? "Local AI požiadavky:" : "Local AI requests:"}{" "}
                    <strong>
                      {taskQueue.concurrency?.active_local_requests ??
                        taskQueue.counts.active_local_requests ??
                        0}
                      /
                      {taskQueue.concurrency?.local_max_concurrent ??
                        taskQueue.counts.local_max_concurrent ??
                        localMaxConcurrentInput}
                    </strong>
                  </span>
                </span>
                <span
                  data-testid="debug-queue-freellm-concurrency-badge"
                  className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-50 text-amber-900 border border-amber-200"
                >
                  {lang === "sk"
                    ? `Cloud API (freellm): súbežne bez fronty (${
                        taskQueue.concurrency?.active_freellm_requests ??
                        taskQueue.counts.active_freellm_requests ??
                        0
                      } aktívnych)`
                    : `Cloud API (freellm): concurrent, no queue (${
                        taskQueue.concurrency?.active_freellm_requests ??
                        taskQueue.counts.active_freellm_requests ??
                        0
                      } active)`}
                </span>
              </div>
              <p className="text-xs text-stone-600">
                {lang === "sk"
                  ? "Požiadavky na Cloud API (freellm) sa neukladajú do fronty a odosielajú sa okamžite súbežne. Pre lokálny model (Local AI) platí limit súbežných požiadaviek (predvolene 5), pričom ďalšie čakajú vo fronte."
                  : "Requests to Cloud API (freellm) are never queued and execute concurrently as they arrive. Local AI requests enforce a concurrent request limit (default 5), queuing additional requests."}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <label
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-stone-100 border border-stone-300 text-xs font-semibold text-stone-800"
                title={
                  lang === "sk"
                    ? "Maximálny počet súbežných požiadaviek na lokálny model (predvolene 5)"
                    : "Maximum concurrent requests to the local model (default 5)"
                }
              >
                <span>
                  {lang === "sk" ? "Limit Local AI:" : "Local AI limit:"}
                </span>
                <input
                  type="number"
                  min={1}
                  max={64}
                  data-testid="debug-local-max-concurrent-input"
                  value={localMaxConcurrentInput}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v) && v >= 1) {
                      setLocalMaxConcurrentInput(v);
                    }
                  }}
                  onBlur={() =>
                    void handleUpdateLocalMaxConcurrent(localMaxConcurrentInput)
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void handleUpdateLocalMaxConcurrent(localMaxConcurrentInput);
                    }
                  }}
                  className="w-12 px-1.5 py-0.5 rounded bg-white border border-stone-300 text-center font-bold text-stone-900 text-xs"
                />
              </label>
              {taskQueue.counts.queued > 0 && (
                <button
                  type="button"
                  data-testid="debug-queue-clear-waiting-btn"
                  disabled={busyTaskActionId !== null}
                  onClick={handleClearQueuedTasks}
                  className="px-3 py-1.5 rounded-lg bg-red-50 hover:bg-red-100 text-red-900 border border-red-300 font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50 transition"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>
                    {lang === "sk"
                      ? `Zrušiť čakajúce (${taskQueue.counts.queued})`
                      : `Clear queued (${taskQueue.counts.queued})`}
                  </span>
                </button>
              )}
              {taskQueue.history.length > 0 && (
                <button
                  type="button"
                  data-testid="debug-queue-clear-history-btn"
                  disabled={busyTaskActionId !== null}
                  onClick={handleClearTaskHistory}
                  className="px-3 py-1.5 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 border border-stone-300 font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer disabled:opacity-50 transition"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>
                    {lang === "sk" ? "Vyčistiť históriu" : "Clear history"}
                  </span>
                </button>
              )}
              <button
                type="button"
                data-testid="debug-queue-refresh-btn"
                onClick={() => void loadTaskQueue()}
                className="px-3 py-1.5 rounded-lg bg-stone-900 hover:bg-stone-800 text-white font-semibold text-xs inline-flex items-center gap-1.5 cursor-pointer transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>{lang === "sk" ? "Obnoviť frontu" : "Refresh queue"}</span>
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Currently Running Task */}
            <div
              data-testid="debug-queue-active-section"
              className="bg-stone-50 rounded-xl p-3.5 border border-stone-200 space-y-2.5"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-stone-700 flex items-center gap-1.5">
                  <Activity className="w-4 h-4 text-amber-600" />
                  <span>
                    {lang === "sk"
                      ? "1. Práve prebiehajúca úloha"
                      : "1. Currently Running Task"}
                  </span>
                </span>
                {taskQueue.active_task && (
                  <span className="px-2 py-0.5 rounded-full bg-amber-600 text-white text-[11px] font-bold animate-pulse">
                    {lang === "sk" ? "AKTÍVNA" : "ACTIVE"}
                  </span>
                )}
              </div>

              {(() => {
                const runningItems: QueuedTaskItem[] =
                  taskQueue.running_tasks && taskQueue.running_tasks.length > 0
                    ? taskQueue.running_tasks
                    : taskQueue.active_task
                    ? [taskQueue.active_task]
                    : [];
                if (runningItems.length === 0) {
                  return (
                    <div
                      data-testid="debug-queue-idle-placeholder"
                      className="bg-white border border-stone-200 rounded-lg p-4 text-center text-xs text-stone-600"
                    >
                      {lang === "sk"
                        ? "Žiadna úloha momentálne neprebieha — fronta je voľná."
                        : "No task is currently running — queue is idle."}
                    </div>
                  );
                }
                return (
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {runningItems.map((at, idx) => {
                      const catBadge = formatTaskCategoryBadge(at.category);
                      const pct = Math.max(5, Math.min(99, at.progress_pct || 10));
                      const durSec = at.duration_ms
                        ? Math.max(1, Math.round(at.duration_ms / 1000))
                        : 1;
                      return (
                        <div
                          key={at.id}
                          data-testid={
                            idx === 0
                              ? "debug-active-task-item"
                              : `debug-running-task-${at.id}`
                          }
                          className="bg-amber-50/90 border border-amber-300 rounded-lg p-3 space-y-2"
                        >
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="space-y-1 min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-1.5">
                                <RefreshCw className="w-3.5 h-3.5 text-amber-700 animate-spin shrink-0" />
                                <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-[11px] font-bold">
                                  #{at.id}
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded border text-[11px] font-bold ${catBadge.cls}`}
                                >
                                  {lang === "sk" ? catBadge.labelSk : catBadge.labelEn}
                                </span>
                                {at.provider && (
                                  <span className="px-2 py-0.5 rounded bg-stone-200 text-stone-800 font-mono text-[11px] font-semibold">
                                    {at.provider === "freellm" ? "freellm (concurrent)" : "local"}
                                  </span>
                                )}
                                {at.active_stage && (
                                  <span className="px-2 py-0.5 rounded bg-amber-600 text-white font-mono text-[11px] font-bold">
                                    {at.active_stage}
                                  </span>
                                )}
                                {at.active_model && (
                                  <span className="px-2 py-0.5 rounded bg-white border border-stone-300 text-stone-800 font-mono text-[11px]">
                                    {at.active_model}
                                  </span>
                                )}
                              </div>
                              <div className="font-bold text-sm text-stone-900">
                                {lang === "sk" ? at.title_sk : at.title_en}
                              </div>
                              <div className="text-xs text-stone-700">
                                {lang === "sk" ? at.message_sk : at.message_en}
                              </div>
                            </div>

                            <div className="flex items-center gap-1.5 shrink-0">
                              {at.target_url && (
                                <Link
                                  to={at.target_url}
                                  className="px-2.5 py-1 rounded-md bg-white hover:bg-stone-100 border border-stone-300 text-stone-800 text-xs font-semibold"
                                >
                                  {lang === "sk" ? "Otvoriť" : "Open"}
                                </Link>
                              )}
                              <button
                                type="button"
                                data-testid={`debug-task-cancel-${at.id}`}
                                disabled={busyTaskActionId === at.task_id}
                                onClick={() => handleCancelQueuedOrActiveTask(at.task_id)}
                                className="px-2.5 py-1 rounded-md bg-red-600 hover:bg-red-700 text-white text-xs font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                              >
                                <X className="w-3.5 h-3.5" />
                                <span>{lang === "sk" ? "Zastaviť" : "Stop"}</span>
                              </button>
                            </div>
                          </div>

                          <div className="flex items-center gap-2 pt-1">
                            <div className="flex-1 bg-amber-200/80 rounded-full h-2 overflow-hidden border border-amber-300">
                              <div
                                className="bg-amber-600 h-full transition-all duration-300"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="text-xs font-bold text-amber-950 shrink-0">
                              {pct}% · {durSec} s
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                );
              })()}
            </div>

            {/* Queued Waiting Tasks */}
            <div
              data-testid="debug-queue-waiting-section"
              className="bg-stone-50 rounded-xl p-3.5 border border-stone-200 space-y-2.5"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-bold uppercase tracking-wider text-stone-700 flex items-center gap-1.5">
                  <Clock className="w-4 h-4 text-stone-600" />
                  <span>
                    {lang === "sk"
                      ? `2. Čakajúce úlohy vo fronte (${taskQueue.queued_tasks.length})`
                      : `2. Queued Tasks Waiting in Line (${taskQueue.queued_tasks.length})`}
                  </span>
                </span>
              </div>

              {taskQueue.queued_tasks.length > 0 ? (
                <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                  {taskQueue.queued_tasks.map((qt, idx) => {
                    const catBadge = formatTaskCategoryBadge(qt.category);
                    const pos = qt.queue_position ?? idx + 1;
                    return (
                      <div
                        key={qt.id}
                        data-testid={`debug-queued-task-${qt.id}`}
                        className="bg-white border border-stone-300 rounded-lg p-2.5 flex flex-wrap items-center justify-between gap-2"
                      >
                        <div className="space-y-1 min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="px-2 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-[11px] font-bold">
                              {lang === "sk" ? `Poradie #${pos}` : `Queue #${pos}`}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-stone-200 text-stone-800 font-mono text-[11px] font-semibold">
                              #{qt.id}
                            </span>
                            <span
                              className={`px-2 py-0.5 rounded border text-[11px] font-bold ${catBadge.cls}`}
                            >
                              {lang === "sk" ? catBadge.labelSk : catBadge.labelEn}
                            </span>
                          </div>
                          <div className="font-bold text-xs text-stone-900 truncate">
                            {lang === "sk" ? qt.title_sk : qt.title_en}
                          </div>
                          <div className="text-[11px] text-stone-600 truncate">
                            {lang === "sk" ? qt.message_sk : qt.message_en}
                          </div>
                        </div>

                        <div className="flex items-center gap-1.5 shrink-0">
                          {pos > 1 && (
                            <button
                              type="button"
                              data-testid={`debug-task-prioritize-${qt.id}`}
                              disabled={busyTaskActionId === qt.task_id}
                              onClick={() => handlePrioritizeQueuedTask(qt.task_id)}
                              title={
                                lang === "sk"
                                  ? "Posunúť na 1. miesto vo fronte"
                                  : "Move to #1 in queue"
                              }
                              className="px-2 py-1 rounded bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300 text-xs font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                            >
                              <ArrowUp className="w-3.5 h-3.5" />
                              <span>{lang === "sk" ? "Prednostne" : "Prioritize"}</span>
                            </button>
                          )}
                          {qt.target_url && (
                            <Link
                              to={qt.target_url}
                              className="px-2 py-1 rounded bg-stone-100 hover:bg-stone-200 border border-stone-300 text-stone-800 text-xs font-semibold"
                            >
                              {lang === "sk" ? "Otvoriť" : "Open"}
                            </Link>
                          )}
                          <button
                            type="button"
                            data-testid={`debug-task-cancel-${qt.id}`}
                            disabled={busyTaskActionId === qt.task_id}
                            onClick={() => handleCancelQueuedOrActiveTask(qt.task_id)}
                            title={
                              lang === "sk"
                                ? "Odstrániť úlohu z fronty"
                                : "Remove task from queue"
                            }
                            className="px-2 py-1 rounded bg-red-50 hover:bg-red-100 text-red-800 border border-red-300 text-xs font-semibold inline-flex items-center gap-1 cursor-pointer disabled:opacity-50"
                          >
                            <X className="w-3.5 h-3.5" />
                            <span>{lang === "sk" ? "Zrušiť" : "Cancel"}</span>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div
                  data-testid="debug-queue-empty-placeholder"
                  className="bg-white border border-stone-200 rounded-lg p-4 text-center text-xs text-stone-600"
                >
                  {lang === "sk"
                    ? "Žiadne ďalšie úlohy nečakajú vo fronte."
                    : "No additional tasks waiting in queue."}
                </div>
              )}
            </div>
          </div>

          {/* Recent Task History */}
          {taskQueue.history.length > 0 && (
            <div
              data-testid="debug-queue-history-section"
              className="bg-stone-50 rounded-xl p-3.5 border border-stone-200 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                  {lang === "sk"
                    ? `3. História nedávnych úloh (${taskQueue.history.length})`
                    : `3. Recent Task History (${taskQueue.history.length})`}
                </span>
              </div>
              <div className="max-h-52 overflow-y-auto space-y-1.5 pr-1">
                {taskQueue.history.slice(0, 15).map((ht) => {
                  const catBadge = formatTaskCategoryBadge(ht.category);
                  const stBadge = formatTaskStatusBadge(ht.status);
                  const durSec =
                    ht.duration_ms != null
                      ? `${Math.max(0.1, Math.round(ht.duration_ms / 100) / 10)} s`
                      : null;
                  return (
                    <div
                      key={ht.id}
                      data-testid={`debug-history-task-${ht.id}`}
                      className="bg-white border border-stone-200 rounded-lg px-3 py-2 flex flex-wrap items-center justify-between gap-2 text-xs"
                    >
                      <div className="flex flex-wrap items-center gap-2 min-w-0 flex-1">
                        <span className="px-1.5 py-0.5 rounded bg-stone-100 border border-stone-200 font-mono text-[11px] font-semibold text-stone-700">
                          #{ht.id}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded-full text-[11px] font-bold ${stBadge.cls}`}
                        >
                          {lang === "sk" ? stBadge.labelSk : stBadge.labelEn}
                        </span>
                        <span
                          className={`px-2 py-0.5 rounded border text-[11px] font-semibold ${catBadge.cls}`}
                        >
                          {lang === "sk" ? catBadge.labelSk : catBadge.labelEn}
                        </span>
                        <span className="font-semibold text-stone-900 truncate">
                          {lang === "sk" ? ht.title_sk : ht.title_en}
                        </span>
                        <span className="text-stone-500 truncate hidden sm:inline">
                          — {lang === "sk" ? ht.message_sk : ht.message_en}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {durSec && (
                          <span className="font-mono text-[11px] text-stone-600">
                            {durSec}
                          </span>
                        )}
                        {ht.target_url && (
                          <Link
                            to={ht.target_url}
                            className="px-2 py-0.5 rounded bg-stone-100 hover:bg-stone-200 border border-stone-300 text-stone-800 text-[11px] font-semibold"
                          >
                            {lang === "sk" ? "Otvoriť" : "Open"}
                          </Link>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Top AI Status & Per-Pipeline-Stage Provider / Model Configuration Card */}
      {(activeConsoleTab === "all" || activeConsoleTab === "ai") && (
      <div className="bg-stone-900 text-stone-100 rounded-xl p-4 border border-amber-500 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold flex flex-wrap items-center gap-2 text-amber-400">
              <Activity className="w-5 h-5" />
              <span>
                {lang === "sk"
                  ? "Debug workflow · Konfigurácia AI fáz (Local AI vs Cloud API)"
                  : "Debug Workflow · AI Stage Configuration (Local AI vs Cloud API)"}
              </span>
              <span
                className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                  isLocalOnline
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "bg-red-500/20 text-red-300 border border-red-500/40"
                }`}
              >
                Local AI: {isLocalOnline ? "Online" : "Offline"}
              </span>
              <span
                className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                  isFreellmOnline
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                    : "bg-stone-700 text-stone-300 border border-stone-600"
                }`}
              >
                Cloud API:{" "}
                {isFreellmOnline
                  ? lang === "sk"
                    ? "Pripravené"
                    : "Ready"
                  : lang === "sk"
                  ? "Nedostupné"
                  : "Unavailable"}
              </span>
            </h2>
            <p className="text-stone-400 text-xs mt-0.5">
              {lang === "sk"
                ? "Nezávislý výber poskytovateľa (Local AI vs Cloud API) a modelu pre každú fázu spracovania (`ocr`, `decision`, `extract`, `layout`)."
                : "Independent provider (Local AI vs Cloud API) and model selection for each pipeline stage (`ocr`, `decision`, `extract`, `layout`)."}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={savingAiConfig}
              onClick={handleSaveAllStages}
              className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-1.5 cursor-pointer"
            >
              {savingAiConfig ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Check className="w-3.5 h-3.5" />
              )}
              <span>
                {savingAiConfig
                  ? lang === "sk"
                    ? "Ukladám…"
                    : "Saving…"
                  : lang === "sk"
                  ? "Uložiť všetky fázy"
                  : "Save all stages"}
              </span>
            </button>
            <button
              disabled={refreshingDebug}
              onClick={async () => {
                setRefreshingDebug(true);
                try {
                  await Promise.all([
                    loadDebug(),
                    loadAiConfig(),
                    loadFeedbackAndStats(),
                  ]);
                } finally {
                  setRefreshingDebug(false);
                }
              }}
              className="px-3 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-700 disabled:opacity-50 border border-stone-600 font-semibold text-xs flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${refreshingDebug ? "animate-spin" : ""}`}
              />{" "}
              {refreshingDebug
                ? lang === "sk"
                  ? "Obnovujem…"
                  : "Refreshing…"
                : "Refresh"}
            </button>
          </div>
        </div>

        {/* 3 Quick Preset Buttons */}
        <div className="bg-stone-950/90 rounded-xl p-3 border border-stone-800 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-stone-400 flex items-center gap-1 mr-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              <span>{lang === "sk" ? "Rýchle predvoľby:" : "Quick presets:"}</span>
            </span>

            <button
              type="button"
              onClick={() => handleApplyPreset("hybrid")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border inline-flex items-center gap-1.5 ${
                isHybridPresetActive
                  ? "bg-amber-500 text-stone-950 border-amber-400 shadow"
                  : "bg-stone-900 hover:bg-stone-800 text-amber-300 border-amber-500/40"
              }`}
            >
              {isHybridPresetActive && <Check className="w-3.5 h-3.5" />}
              <span>
                {lang === "sk"
                  ? "Odporúčané: OCR + Decision lokálne, Extrakcia cez Cloud API"
                  : "Recommended: OCR + Decision Local, Extraction via Cloud API"}
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleApplyPreset("all_local")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border inline-flex items-center gap-1.5 ${
                isAllLocalPresetActive
                  ? "bg-amber-500 text-stone-950 border-amber-400 shadow"
                  : "bg-stone-900 hover:bg-stone-800 text-stone-200 border-stone-700"
              }`}
            >
              {isAllLocalPresetActive && <Check className="w-3.5 h-3.5" />}
              <span>
                {lang === "sk" ? "Všetko lokálne (Local AI)" : "All Local (Local AI)"}
              </span>
            </button>

            <button
              type="button"
              onClick={() => handleApplyPreset("all_freellm")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border inline-flex items-center gap-1.5 ${
                isAllFreellmPresetActive
                  ? "bg-amber-500 text-stone-950 border-amber-400 shadow"
                  : "bg-stone-900 hover:bg-stone-800 text-stone-200 border-stone-700"
              }`}
            >
              {isAllFreellmPresetActive && <Check className="w-3.5 h-3.5" />}
              <span>
                {lang === "sk" ? "Všetko cez Cloud API" : "All via Cloud API"}
              </span>
            </button>

            <button
              type="button"
              data-testid="debug-skip-row-ocr-toggle"
              onClick={() => toggleFlag("skip_row_ocr")}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border inline-flex items-center gap-1.5 ${
                featureFlags.skip_row_ocr
                  ? "bg-emerald-500 text-stone-950 border-emerald-400 shadow"
                  : "bg-stone-900 hover:bg-stone-800 text-stone-200 border-stone-700"
              }`}
              title={
                lang === "sk"
                  ? "Použiť GLM-OCR na hlavičku a tabuľky prehľadov, preskočiť textové OCR riadkov a poslať 5-pásmový výrez stĺpcov priamo do Extract modelu bez OCR nápovied"
                  : "Use GLM-OCR for headers and overview tables, skip row text OCR, and send 5-band column crops directly to Extract model without OCR hints"
              }
            >
              {featureFlags.skip_row_ocr && <Check className="w-3.5 h-3.5" />}
              <span>
                {featureFlags.skip_row_ocr
                  ? lang === "sk"
                    ? "Priamy Extract bez OCR riadkov: ZAP (GLM-OCR iba pre hlavičku a tabuľky)"
                    : "Direct Extract (Skip Row OCR): ON (GLM-OCR for header & tables only)"
                  : lang === "sk"
                  ? "Priamy Extract bez OCR riadkov: VYP (Spúšťať aj OCR riadkov)"
                  : "Direct Extract (Skip Row OCR): OFF (Run row OCR too)"}
              </span>
            </button>
          </div>

          {aiSavedNotice && (
            <span
              className={`px-2.5 py-1 rounded font-sans text-xs font-semibold border ${
                aiSavedError
                  ? "bg-red-500/20 text-red-300 border-red-500/40"
                  : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
              }`}
            >
              {aiSavedNotice}
            </span>
          )}
        </div>

        {/* 4-Stage Table / Grid (ocr, decision, extract, layout) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {STAGE_DESCRIPTORS.map((desc) => {
            const st = stageConfigs[desc.key] || DEFAULT_STAGE_CONFIGS[desc.key];
            const isLocal = st.provider === "local";
            const isSavingThis = savingAiConfig && savingStageKey === desc.key;
            return (
              <div
                key={desc.key}
                className="bg-stone-950/80 rounded-xl p-3.5 border border-stone-800 space-y-2.5"
              >
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-bold text-xs text-amber-300 font-mono">
                      {lang === "sk" ? desc.titleSk : desc.titleEn}
                    </div>
                    <div className="text-xs text-stone-300 mt-0.5">
                      {lang === "sk" ? desc.subtitleSk : desc.subtitleEn}
                    </div>
                  </div>

                  {/* Provider Toggle: Local AI vs Cloud API */}
                  <div className="inline-flex rounded-lg bg-stone-900 p-0.5 border border-stone-700 shrink-0">
                    <button
                      type="button"
                      onClick={() => handleStageProviderToggle(desc.key, "local")}
                      className={`px-2.5 py-1 min-h-[30px] rounded-md text-xs font-semibold transition cursor-pointer ${
                        isLocal
                          ? "bg-emerald-500 text-stone-950 shadow"
                          : "text-stone-300 hover:text-white"
                      }`}
                    >
                      Local AI
                    </button>
                    <button
                      type="button"
                      onClick={() => handleStageProviderToggle(desc.key, "freellm")}
                      className={`px-2.5 py-1 min-h-[30px] rounded-md text-xs font-semibold transition cursor-pointer ${
                        !isLocal
                          ? "bg-amber-500 text-stone-950 shadow"
                          : "text-stone-300 hover:text-white"
                      }`}
                    >
                      Cloud API
                    </button>
                  </div>
                </div>

                {/* Model Input + Datalist + 1-click Save */}
                <div className="flex items-center gap-2">
                  <Server className="w-3.5 h-3.5 text-stone-400 shrink-0" />
                  <input
                    type="text"
                    list={isLocal ? "local-models-datalist" : "freellm-models-datalist"}
                    value={st.model}
                    onChange={(e) => handleStageModelChange(desc.key, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void handleSaveSingleStage(desc.key);
                      }
                    }}
                    placeholder={
                      isLocal ? desc.defaultLocalModel : desc.defaultFreellmModel
                    }
                    className="flex-1 px-2.5 py-1.5 rounded-lg bg-stone-900 border border-stone-700 text-stone-100 text-xs font-mono focus:border-amber-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    disabled={savingAiConfig}
                    onClick={() => handleSaveSingleStage(desc.key)}
                    className="px-2.5 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-700 border border-stone-600 text-stone-100 text-xs font-semibold inline-flex items-center gap-1 cursor-pointer shrink-0"
                  >
                    <Check className="w-3.5 h-3.5 text-amber-400" />
                    <span>
                      {isSavingThis
                        ? lang === "sk"
                          ? "Ukladám…"
                          : "Saving…"
                        : lang === "sk"
                        ? "Uložiť"
                        : "Save"}
                    </span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Active Pipeline Models Summary Footer */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-stone-400 font-mono pt-1 border-t border-stone-800">
          <div className="flex flex-wrap items-center gap-3">
            <span>
              Local:{" "}
              <strong className="text-stone-200">
                {aiConfig?.local_base_url || "http://localhost:1234/v1"}
              </strong>
            </span>
            <span>
              Cloud API:{" "}
              <strong className="text-stone-200">
                {aiConfig?.freellm_base_url || "https://api.openai.com/v1"}
              </strong>
            </span>
            {activeConfigured && (
              <>
                <span>
                  OCR:{" "}
                  <strong className="text-amber-300">
                    {stageConfigs.ocr.provider}:{activeConfigured.ocr}
                  </strong>
                </span>
                <span>
                  Decision:{" "}
                  <strong className="text-amber-300">
                    {stageConfigs.decision.provider}:{activeConfigured.decision}
                  </strong>
                </span>
                <span>
                  Extract:{" "}
                  <strong className="text-amber-300">
                    {stageConfigs.extract.provider}:{activeConfigured.extract}
                  </strong>
                </span>
                <span>
                  Layout:{" "}
                  <strong className="text-amber-300">
                    {stageConfigs.layout.provider}:
                    {activeConfigured.layout || stageConfigs.layout.model}
                  </strong>
                </span>
              </>
            )}
          </div>
        </div>
      </div>
      )}

      {/* Model Combo Statistics & Individual Step Timing Breakdown View */}
      {(activeConsoleTab === "all" || activeConsoleTab === "benchmark") && (
      <>
      <div
        data-testid="debug-model-combo-stats-card"
        className="bg-white rounded-xl p-4 border-2 border-emerald-600/80 shadow-xs space-y-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-3">
          <div>
            <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-700" />
              <span>
                {lang === "sk"
                  ? "Štatistiky úspešnosti kombinácií modelov a trvania krokov (Model Combo Statistics)"
                  : "Model Combo Correctness Rate & Step Timing Breakdown"}
              </span>
            </h3>
            <p className="text-xs text-stone-600 mt-0.5">
              {lang === "sk"
                ? "Porovnanie presnosti OCR podľa hodnotení používateľa (OCR ✓ / OCR ✗) a priemerného času spracovania riadku s rozpadom na jednotlivé kroky (Layout, OCR, Decision, Extract, MCP)."
                : "Comparison of OCR accuracy based on user ratings (OCR ✓ / OCR ✗) and average row processing time broken down by individual pipeline steps (Layout, OCR, Decision, Extract, MCP)."}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <a
              href={`/api/debug/feedback/export?format=md${
                feedbackVerdictFilter ? `&verdict=${feedbackVerdictFilter}` : ""
              }`}
              download
              data-testid="download-llm-report-md"
              className="px-3 py-1.5 rounded-lg bg-stone-900 hover:bg-stone-800 text-white text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-amber-400" />
              <span>
                {lang === "sk"
                  ? "Stiahnuť súbor pre LLM (.md)"
                  : "Download LLM Debug File (.md)"}
              </span>
            </a>
            <a
              href={`/api/debug/feedback/export?format=json${
                feedbackVerdictFilter ? `&verdict=${feedbackVerdictFilter}` : ""
              }`}
              download
              data-testid="download-llm-report-json"
              className="px-3 py-1.5 rounded-lg bg-stone-200 hover:bg-stone-300 text-stone-900 text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-stone-700" />
              <span>JSON (.json)</span>
            </a>
          </div>
        </div>

        {/* Summary KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-stone-50 rounded-xl p-3 border border-stone-200">
            <div className="text-xs text-stone-600 font-semibold">
              {lang === "sk"
                ? "Celková úspešnosť (Correctness Rate)"
                : "Overall Correctness Rate"}
            </div>
            <div
              data-testid="stats-overall-correctness-rate"
              className="text-xl font-bold text-stone-900 mt-1"
            >
              {feedbackStats?.summary.correctness_rate_pct !== null &&
              feedbackStats?.summary.correctness_rate_pct !== undefined
                ? `${feedbackStats.summary.correctness_rate_pct}%`
                : "—"}
            </div>
            <div className="text-xs text-stone-500 mt-0.5">
              {lang === "sk"
                ? `Hodnotených riadkov: ${feedbackStats?.summary.total_feedback ?? 0}`
                : `Rated rows: ${feedbackStats?.summary.total_feedback ?? 0}`}
            </div>
          </div>

          <div className="bg-emerald-50/70 rounded-xl p-3 border border-emerald-200">
            <div className="text-xs text-emerald-900 font-semibold flex items-center gap-1">
              <ThumbsUp className="w-3.5 h-3.5 text-emerald-700" />
              <span>{lang === "sk" ? "Správne OCR (✓)" : "Correct OCR (✓)"}</span>
            </div>
            <div
              data-testid="stats-correct-count"
              className="text-xl font-bold text-emerald-900 mt-1"
            >
              {feedbackStats?.summary.correct_count ?? 0}
            </div>
            <div className="text-xs text-emerald-800 mt-0.5">
              {lang === "sk" ? "Potvrdené používateľom" : "Verified by user"}
            </div>
          </div>

          <div className="bg-red-50/70 rounded-xl p-3 border border-red-200">
            <div className="text-xs text-red-900 font-semibold flex items-center gap-1">
              <ThumbsDown className="w-3.5 h-3.5 text-red-700" />
              <span>{lang === "sk" ? "Chybné OCR (✗)" : "Incorrect OCR (✗)"}</span>
            </div>
            <div
              data-testid="stats-incorrect-count"
              className="text-xl font-bold text-red-900 mt-1"
            >
              {feedbackStats?.summary.incorrect_count ?? 0}
            </div>
            <div className="text-xs text-red-800 mt-0.5">
              {lang === "sk" ? "Zahrnuté v súbore pre LLM" : "Included in LLM debug file"}
            </div>
          </div>

          <div className="bg-amber-50/70 rounded-xl p-3 border border-amber-200">
            <div className="text-xs text-amber-950 font-semibold flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 text-amber-700" />
              <span>
                {lang === "sk"
                  ? "Priemerný čas na riadok"
                  : "Avg Time per Row"}
              </span>
            </div>
            <div
              data-testid="stats-avg-total-ms"
              className="text-xl font-bold text-amber-950 mt-1 font-mono"
            >
              {feedbackStats?.summary.avg_total_duration_ms ?? 0} ms
            </div>
            <div className="text-xs text-amber-900 mt-0.5 font-mono">
              OCR: {feedbackStats?.summary.avg_steps_ms.ocr_ms ?? 0}ms · Ext:{" "}
              {feedbackStats?.summary.avg_steps_ms.extract_ms ?? 0}ms
            </div>
          </div>
        </div>

        {/* Model Combo Correctness & Step Timing Table */}
        <div className="space-y-2">
          <div className="font-bold text-xs uppercase tracking-wider text-stone-700">
            {lang === "sk"
              ? "1. Úspešnosť a priemerný čas podľa kombinácie modelov (Model Combos)"
              : "1. Correctness Rate & Average Time by Model Combination"}
          </div>
          <div className="overflow-x-auto border border-stone-200 rounded-xl">
            <table
              data-testid="model-combos-stats-table"
              className="w-full text-left border-collapse text-xs"
            >
              <thead>
                <tr className="bg-stone-100 border-b border-stone-200 text-stone-800 font-bold">
                  <th className="py-2.5 px-3">
                    {lang === "sk" ? "Kombinácia modelov (OCR + Extract)" : "Model Combo"}
                  </th>
                  <th className="py-2.5 px-3">
                    {lang === "sk" ? "Úspešnosť (Correctness)" : "Correctness Rate"}
                  </th>
                  <th className="py-2.5 px-3 text-right">
                    {lang === "sk" ? "Priem. čas spolu" : "Avg Total Time"}
                  </th>
                  <th className="py-2.5 px-3 text-right">Layout</th>
                  <th className="py-2.5 px-3 text-right">
                    {lang === "sk" ? "OCR (Ľavá / Pravá / Hlav.)" : "OCR (L / R / Hdr)"}
                  </th>
                  <th className="py-2.5 px-3 text-right">Decision</th>
                  <th className="py-2.5 px-3 text-right">Extract</th>
                  <th className="py-2.5 px-3 text-right">MCP</th>
                  <th className="py-2.5 px-3 min-w-[160px]">
                    {lang === "sk" ? "Pomer krokov" : "Step Breakdown"}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-200">
                {(feedbackStats?.model_combos || []).map((c) => {
                  const totMs = Math.max(1, c.avg_total_ms || 1);
                  const layPct = Math.round((100 * c.steps.layout_ms) / totMs);
                  const ocrPct = Math.round((100 * c.steps.ocr_ms) / totMs);
                  const decPct = Math.round((100 * c.steps.decision_ms) / totMs);
                  const extPct = Math.round((100 * c.steps.extract_ms) / totMs);
                  const mcpPct = Math.max(
                    0,
                    100 - layPct - ocrPct - decPct - extPct
                  );
                  return (
                    <tr
                      key={c.model_combo}
                      className={
                        c.is_active_combo ? "bg-amber-50/50" : "hover:bg-stone-50"
                      }
                    >
                      <td className="py-2.5 px-3 align-middle">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="font-mono font-bold text-stone-900">
                            {c.model_combo}
                          </span>
                          {c.is_active_combo && (
                            <span className="px-2 py-0.5 rounded-full bg-amber-200 text-amber-950 font-bold text-[11px]">
                              {lang === "sk" ? "Aktívna" : "Active"}
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-stone-500 font-mono mt-0.5">
                          Layout: {c.layout_model || "opencv"} · Decision:{" "}
                          {c.decision_model || "deterministic"}
                        </div>
                      </td>

                      <td className="py-2.5 px-3 align-middle">
                        {c.correctness_rate_pct !== null ? (
                          <div className="space-y-1">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`px-2 py-0.5 rounded-full font-bold text-xs ${
                                  c.correctness_rate_pct >= 80
                                    ? "bg-emerald-100 text-emerald-900"
                                    : c.correctness_rate_pct >= 50
                                    ? "bg-amber-100 text-amber-900"
                                    : "bg-red-100 text-red-900"
                                }`}
                              >
                                {c.correctness_rate_pct}%
                              </span>
                              <span className="text-stone-700 font-medium">
                                <span className="text-emerald-700 font-bold">
                                  {c.correct_count}✓
                                </span>{" "}
                                /{" "}
                                <span className="text-red-700 font-bold">
                                  {c.incorrect_count}✗
                                </span>{" "}
                                ({c.total_rated}{" "}
                                {lang === "sk" ? "hodnotených" : "rated"})
                              </span>
                            </div>
                            <div className="w-32 bg-red-200 rounded-full h-1.5 overflow-hidden">
                              <div
                                className="bg-emerald-600 h-full"
                                style={{ width: `${c.correctness_rate_pct}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <span className="text-stone-500 italic">
                            {lang === "sk"
                              ? `Zatiaľ nehodnotené (${c.rows_processed} riadkov)`
                              : `Unrated (${c.rows_processed} rows)`}
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono font-bold text-stone-900">
                        <div>{c.avg_total_ms} ms</div>
                        <div className="text-[11px] font-normal text-stone-500">
                          {(c.avg_total_ms / 1000).toFixed(2)} s
                        </div>
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono text-stone-700">
                        <div>{c.steps.layout_ms} ms</div>
                        {c.steps.scan_layout_ms > 0 && (
                          <div className="text-[11px] text-stone-500">
                            scan: {c.steps.scan_layout_ms} ms
                          </div>
                        )}
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono text-stone-800">
                        <div className="font-semibold">{c.steps.ocr_ms} ms</div>
                        <div className="text-[11px] text-stone-500">
                          L:{c.steps.ocr_left_ms} · R:{c.steps.ocr_right_ms} · H:
                          {c.steps.ocr_header_ms}
                        </div>
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono text-stone-700">
                        {c.steps.decision_ms} ms
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono font-semibold text-stone-900">
                        {c.steps.extract_ms} ms
                      </td>

                      <td className="py-2.5 px-3 align-middle text-right font-mono text-stone-700">
                        {c.steps.mcp_ms} ms
                      </td>

                      <td className="py-2.5 px-3 align-middle">
                        {c.avg_total_ms > 0 ? (
                          <div className="space-y-1">
                            <div className="w-full h-2.5 rounded-full bg-stone-200 overflow-hidden flex">
                              {c.steps.layout_ms > 0 && (
                                <div
                                  className="bg-sky-500 h-full"
                                  style={{ width: `${layPct}%` }}
                                  title={`Layout: ${c.steps.layout_ms} ms (${layPct}%)`}
                                />
                              )}
                              {c.steps.ocr_ms > 0 && (
                                <div
                                  className="bg-amber-500 h-full"
                                  style={{ width: `${ocrPct}%` }}
                                  title={`OCR: ${c.steps.ocr_ms} ms (${ocrPct}%)`}
                                />
                              )}
                              {c.steps.decision_ms > 0 && (
                                <div
                                  className="bg-purple-500 h-full"
                                  style={{ width: `${decPct}%` }}
                                  title={`Decision: ${c.steps.decision_ms} ms (${decPct}%)`}
                                />
                              )}
                              {c.steps.extract_ms > 0 && (
                                <div
                                  className="bg-emerald-600 h-full"
                                  style={{ width: `${extPct}%` }}
                                  title={`Extract: ${c.steps.extract_ms} ms (${extPct}%)`}
                                />
                              )}
                              {c.steps.mcp_ms > 0 && (
                                <div
                                  className="bg-indigo-500 h-full"
                                  style={{ width: `${mcpPct}%` }}
                                  title={`MCP: ${c.steps.mcp_ms} ms (${mcpPct}%)`}
                                />
                              )}
                            </div>
                            <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-stone-600">
                              <span className="text-sky-800">Lay {layPct}%</span>
                              <span className="text-amber-800">OCR {ocrPct}%</span>
                              {c.steps.decision_ms > 0 && (
                                <span className="text-purple-800">
                                  Dec {decPct}%
                                </span>
                              )}
                              <span className="text-emerald-800">
                                Ext {extPct}%
                              </span>
                            </div>
                          </div>
                        ) : (
                          <span className="text-stone-400">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Individual Pipeline Steps Breakdown Table */}
        {(feedbackStats?.individual_steps?.length ?? 0) > 0 && (
          <div className="space-y-2 pt-1">
            <div className="font-bold text-xs uppercase tracking-wider text-stone-700">
              {lang === "sk"
                ? "2. Rozpad jednotlivých krokov spracovania (Individual Pipeline Steps)"
                : "2. Individual Pipeline Steps Breakdown"}
            </div>
            <div className="overflow-x-auto border border-stone-200 rounded-xl">
              <table
                data-testid="individual-steps-stats-table"
                className="w-full text-left border-collapse text-xs"
              >
                <thead>
                  <tr className="bg-stone-100 border-b border-stone-200 text-stone-800 font-bold">
                    <th className="py-2 px-3">
                      {lang === "sk" ? "Krok / Fáza" : "Step / Stage"}
                    </th>
                    <th className="py-2 px-3">
                      {lang === "sk" ? "Model / Nástroj" : "Model / Tool"}
                    </th>
                    <th className="py-2 px-3 text-right">
                      {lang === "sk" ? "Volania" : "Calls"}
                    </th>
                    <th className="py-2 px-3 text-right">
                      {lang === "sk" ? "Priemerný čas" : "Avg Time"}
                    </th>
                    <th className="py-2 px-3 text-right">Min / Max</th>
                    <th className="py-2 px-3 text-right">
                      {lang === "sk" ? "Priem. tokeny (In / Out)" : "Avg Tokens (In / Out)"}
                    </th>
                    <th className="py-2 px-3 text-right">
                      {lang === "sk" ? "Úspešnosť riadkov" : "Rated Row Accuracy"}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200">
                  {feedbackStats?.individual_steps.map((s) => (
                    <tr
                      key={`${s.stage}:${s.model_or_tool}`}
                      className="hover:bg-stone-50"
                    >
                      <td className="py-2 px-3 font-mono font-bold text-stone-900">
                        <span className="px-1.5 py-0.5 rounded bg-stone-900 text-amber-300 uppercase text-[11px]">
                          {s.stage}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-mono text-stone-800">
                        {s.model_or_tool}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-stone-700">
                        {s.calls_count}
                        {s.error_count > 0 && (
                          <span className="text-red-600 ml-1">
                            ({s.error_count} err)
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-right font-mono font-bold text-stone-900">
                        {s.avg_duration_ms} ms
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-stone-600">
                        {s.min_duration_ms} / {s.max_duration_ms} ms
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-stone-600">
                        {s.avg_prompt_tokens} / {s.avg_completion_tokens}
                      </td>
                      <td className="py-2 px-3 text-right font-mono">
                        {s.correctness_rate_pct !== null ? (
                          <span className="font-bold text-stone-900">
                            {s.correctness_rate_pct}% ({s.correct_rows_count}/
                            {s.rated_rows_count})
                          </span>
                        ) : (
                          <span className="text-stone-400">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* OCR Feedback Log & LLM Debug File Export */}
      <div
        data-testid="debug-ocr-feedback-card"
        className="bg-white rounded-xl p-4 border border-stone-300 space-y-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-3">
          <div>
            <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
              <FileText className="w-5 h-5 text-amber-600" />
              <span>
                {lang === "sk"
                  ? `Spätná väzba OCR & Súbor pre LLM Debugging (${feedbackList.length})`
                  : `OCR Feedback Log & LLM Debug File (${feedbackList.length})`}
              </span>
            </h3>
            <p className="text-xs text-stone-600 mt-0.5">
              {lang === "sk" ? (
                <>
                  Každé označenie riadku ako{" "}
                  <strong className="text-emerald-800">OCR ✓ (Správne)</strong> alebo{" "}
                  <strong className="text-red-800">OCR ✗ (Chybné)</strong> sa automaticky
                  zapisuje do súboru{" "}
                  <code className="font-mono bg-stone-100 px-1.5 py-0.5 rounded text-stone-900">
                    data/ocr_feedback_debug_report.md
                  </code>{" "}
                  (aj <code className="font-mono">.json</code>) so všetkými detailmi o
                  modeloch, trvaní krokov, výrezoch a promptoch pre LLM.
                </>
              ) : (
                <>
                  Every row marked as{" "}
                  <strong className="text-emerald-800">OCR ✓ (Correct)</strong> or{" "}
                  <strong className="text-red-800">OCR ✗ (Incorrect)</strong> is
                  automatically written to{" "}
                  <code className="font-mono bg-stone-100 px-1.5 py-0.5 rounded text-stone-900">
                    data/ocr_feedback_debug_report.md
                  </code>{" "}
                  (and <code className="font-mono">.json</code>) with full model, timing,
                  crop, and prompt details for LLM debugging.
                </>
              )}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {(["", "incorrect", "correct"] as const).map((vf) => (
              <button
                key={vf}
                type="button"
                onClick={() => {
                  setFeedbackVerdictFilter(vf);
                  void loadFeedbackAndStats(vf);
                }}
                className={`px-2.5 py-1.5 min-h-[32px] rounded-lg text-xs font-semibold cursor-pointer ${
                  feedbackVerdictFilter === vf
                    ? "bg-stone-900 text-white"
                    : "bg-stone-200 text-stone-800 hover:bg-stone-300"
                }`}
              >
                {vf === ""
                  ? lang === "sk"
                    ? "Všetky"
                    : "All"
                  : vf === "incorrect"
                  ? lang === "sk"
                    ? "Iba chybné (✗)"
                    : "Incorrect Only (✗)"
                  : lang === "sk"
                  ? "Iba správne (✓)"
                  : "Correct Only (✓)"}
              </button>
            ))}

            <button
              type="button"
              data-testid="copy-llm-report-btn"
              onClick={() => void handleCopyReportForLlm()}
              className="px-3 py-1.5 min-h-[32px] rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
            >
              {copiedReport ? (
                <Check className="w-3.5 h-3.5" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
              <span>
                {copiedReport
                  ? lang === "sk"
                    ? "Skopírované pre LLM!"
                    : "Copied for LLM!"
                  : lang === "sk"
                  ? "Skopírovať report pre LLM"
                  : "Copy Report for LLM"}
              </span>
            </button>

            <button
              type="button"
              data-testid="toggle-llm-report-preview-btn"
              onClick={() => setShowReportPreview((v) => !v)}
              className="px-3 py-1.5 min-h-[32px] rounded-lg bg-stone-800 hover:bg-stone-700 text-stone-100 text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
            >
              <FileText className="w-3.5 h-3.5 text-amber-400" />
              <span>
                {showReportPreview
                  ? lang === "sk"
                    ? "Skryť súbor pre LLM"
                    : "Hide LLM File"
                  : lang === "sk"
                  ? "Zobraziť súbor pre LLM"
                  : "Preview LLM File"}
              </span>
            </button>
          </div>
        </div>

        {/* Live Preview of the LLM Debug File */}
        {showReportPreview && (
          <div
            data-testid="llm-report-preview-box"
            className="bg-stone-950 text-stone-100 rounded-xl p-4 border border-amber-500/50 space-y-2"
          >
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-800 pb-2 text-xs font-mono">
              <span className="text-amber-400 font-bold">
                data/ocr_feedback_debug_report.md
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleCopyReportForLlm()}
                  className="px-2.5 py-1 rounded bg-stone-800 hover:bg-stone-700 text-amber-300 font-sans font-semibold text-xs inline-flex items-center gap-1 cursor-pointer"
                >
                  <Copy className="w-3 h-3" />
                  <span>{lang === "sk" ? "Kopírovať" : "Copy"}</span>
                </button>
                <a
                  href="/api/debug/feedback/export?format=md"
                  download
                  className="px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white font-sans font-semibold text-xs inline-flex items-center gap-1"
                >
                  <Download className="w-3 h-3" />
                  <span>.md</span>
                </a>
              </div>
            </div>
            <pre className="text-xs font-mono text-stone-200 overflow-x-auto whitespace-pre-wrap max-h-96">
              {reportMarkdown || "—"}
            </pre>
          </div>
        )}

        {/* Feedback Items List & Detail Inspector */}
        {feedbackList.length === 0 ? (
          <div className="p-6 text-center bg-stone-50 rounded-xl border border-stone-200 text-stone-600 text-xs sm:text-sm">
            {lang === "sk"
              ? "Zatiaľ neboli označené žiadne riadky. Kliknite na tlačidlá „OCR ✓“ (Správne) alebo „OCR ✗“ (Chybné) pri ľubovoľnej osobe alebo dome v Sčítacích hárkoch, Knihách a stranách alebo v Grid inspectore nižšie."
              : "No rows have been rated yet. Click 'OCR ✓' (Correct) or 'OCR ✗' (Incorrect) on any row in Census Sheets, Books & Scans, or in the Grid Inspector below."}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <div className="lg:col-span-5 max-h-[520px] overflow-y-auto space-y-2 pr-1">
              {feedbackList.map((item) => {
                const isSel = selectedFeedbackItem?.id === item.id;
                const isCorrect = item.verdict === "correct";
                return (
                  <div
                    key={item.id}
                    onClick={() => setSelectedFeedbackItem(item)}
                    className={`p-3 rounded-xl border transition cursor-pointer space-y-1.5 ${
                      isSel
                        ? "bg-amber-50 border-amber-500 shadow-2xs"
                        : "bg-stone-50 border-stone-200 hover:border-stone-400"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-bold inline-flex items-center gap-1 ${
                          isCorrect
                            ? "bg-emerald-600 text-white"
                            : "bg-red-600 text-white"
                        }`}
                      >
                        {isCorrect ? (
                          <ThumbsUp className="w-3 h-3" />
                        ) : (
                          <ThumbsDown className="w-3 h-3" />
                        )}
                        <span>{isCorrect ? "CORRECT" : "INCORRECT"}</span>
                      </span>
                      <span className="text-xs font-mono text-stone-600">
                        {item.total_duration_ms} ms · #{item.row_id ?? item.id}
                      </span>
                    </div>

                    <div className="font-bold text-stone-900 text-xs truncate">
                      {item.display_label || `Row #${item.row_id ?? item.id}`}
                    </div>

                    <div className="text-[11px] font-mono text-stone-700 truncate">
                      {item.model_combo}
                    </div>

                    {item.comment && (
                      <div className="text-xs bg-amber-100/80 border border-amber-300 text-amber-950 px-2 py-1 rounded">
                        “{item.comment}”
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="lg:col-span-7">
              {selectedFeedbackItem ? (
                <div className="bg-stone-900 text-stone-100 rounded-xl p-4 space-y-3 text-xs">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-700 pb-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded-full font-bold uppercase ${
                          selectedFeedbackItem.verdict === "correct"
                            ? "bg-emerald-600 text-white"
                            : "bg-red-600 text-white"
                        }`}
                      >
                        {selectedFeedbackItem.verdict}
                      </span>
                      <span className="font-bold text-amber-300 text-sm">
                        {selectedFeedbackItem.display_label}
                      </span>
                      {selectedFeedbackItem.census_sheet_id && (
                        <Link
                          to={`/scitacie-harky/${selectedFeedbackItem.census_sheet_id}`}
                          className="px-2 py-0.5 rounded bg-stone-800 hover:bg-stone-700 text-amber-300 border border-stone-600 font-semibold"
                        >
                          {lang === "sk" ? "→ Otvoriť hárok" : "→ Open Sheet"}
                        </Link>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() =>
                          void handleUpdateFeedbackItem(
                            selectedFeedbackItem,
                            selectedFeedbackItem.verdict === "correct"
                              ? "incorrect"
                              : "correct"
                          )
                        }
                        className="px-2.5 py-1 rounded bg-stone-800 hover:bg-stone-700 border border-stone-600 text-stone-200 font-semibold cursor-pointer"
                      >
                        {selectedFeedbackItem.verdict === "correct"
                          ? lang === "sk"
                            ? "Prepnúť na Chybné (✗)"
                            : "Switch to Incorrect (✗)"
                          : lang === "sk"
                          ? "Prepnúť na Správne (✓)"
                          : "Switch to Correct (✓)"}
                      </button>
                      <a
                        href={`/api/debug/feedback/export?format=md&feedback_id=${selectedFeedbackItem.id}`}
                        download
                        className="px-2.5 py-1 rounded bg-amber-600 hover:bg-amber-500 text-white font-semibold inline-flex items-center gap-1"
                      >
                        <Download className="w-3 h-3" />
                        <span>.md</span>
                      </a>
                      <button
                        type="button"
                        onClick={() =>
                          void handleDeleteFeedbackItem(selectedFeedbackItem.id)
                        }
                        className="p-1.5 rounded bg-stone-800 hover:bg-red-900 text-stone-300 hover:text-white border border-stone-700 cursor-pointer"
                        title={lang === "sk" ? "Vymazať" : "Delete"}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* User Note / Comment Editor */}
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800 space-y-1.5">
                    <div className="flex items-center justify-between">
                      <span className="text-amber-300 font-bold uppercase">
                        {lang === "sk"
                          ? "Poznámka / Oprava pre LLM"
                          : "User Note / Correction for LLM"}
                      </span>
                      {editingFeedbackId !== selectedFeedbackItem.id && (
                        <button
                          type="button"
                          onClick={() => {
                            setEditingFeedbackId(selectedFeedbackItem.id);
                            setEditingCommentText(
                              selectedFeedbackItem.comment || ""
                            );
                          }}
                          className="text-amber-400 hover:underline font-semibold cursor-pointer"
                        >
                          {lang === "sk" ? "Upraviť poznámku" : "Edit note"}
                        </button>
                      )}
                    </div>
                    {editingFeedbackId === selectedFeedbackItem.id ? (
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={editingCommentText}
                          onChange={(e) => setEditingCommentText(e.target.value)}
                          placeholder={
                            lang === "sk"
                              ? "Popíšte chybu OCR alebo správnu hodnotu…"
                              : "Describe the OCR error or expected value…"
                          }
                          className="flex-1 px-2.5 py-1 rounded bg-stone-900 border border-stone-700 text-stone-100"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            void handleUpdateFeedbackItem(
                              selectedFeedbackItem,
                              selectedFeedbackItem.verdict,
                              editingCommentText
                            )
                          }
                          className="px-2.5 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-semibold cursor-pointer"
                        >
                          {lang === "sk" ? "Uložiť" : "Save"}
                        </button>
                      </div>
                    ) : (
                      <div className="text-stone-200">
                        {selectedFeedbackItem.comment || (
                          <span className="text-stone-500 italic">
                            {lang === "sk"
                              ? "Bez poznámky"
                              : "No note provided"}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Models & Timing Breakdown */}
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 font-mono">
                    <div className="bg-stone-950 p-2 rounded border border-stone-800">
                      <div className="text-stone-400 text-[10px] uppercase">
                        Layout
                      </div>
                      <div className="text-sky-300 font-bold">
                        {selectedFeedbackItem.layout_duration_ms} ms
                      </div>
                      <div className="text-stone-400 text-[10px] truncate">
                        {selectedFeedbackItem.layout_model || "opencv"}
                      </div>
                    </div>
                    <div className="bg-stone-950 p-2 rounded border border-stone-800">
                      <div className="text-stone-400 text-[10px] uppercase">
                        OCR
                      </div>
                      <div className="text-amber-300 font-bold">
                        {selectedFeedbackItem.ocr_duration_ms} ms
                      </div>
                      <div className="text-stone-400 text-[10px] truncate">
                        {selectedFeedbackItem.ocr_model}
                      </div>
                    </div>
                    <div className="bg-stone-950 p-2 rounded border border-stone-800">
                      <div className="text-stone-400 text-[10px] uppercase">
                        Decision
                      </div>
                      <div className="text-purple-300 font-bold">
                        {selectedFeedbackItem.decision_duration_ms} ms
                      </div>
                      <div className="text-stone-400 text-[10px] truncate">
                        {selectedFeedbackItem.decision_model}
                      </div>
                    </div>
                    <div className="bg-stone-950 p-2 rounded border border-stone-800">
                      <div className="text-stone-400 text-[10px] uppercase">
                        Extract
                      </div>
                      <div className="text-emerald-300 font-bold">
                        {selectedFeedbackItem.extract_duration_ms} ms
                      </div>
                      <div className="text-stone-400 text-[10px] truncate">
                        {selectedFeedbackItem.extract_model}
                      </div>
                    </div>
                    <div className="bg-stone-950 p-2 rounded border border-stone-800">
                      <div className="text-stone-400 text-[10px] uppercase">
                        Total Time
                      </div>
                      <div className="text-white font-bold">
                        {selectedFeedbackItem.total_duration_ms} ms
                      </div>
                      <div className="text-stone-400 text-[10px]">
                        MCP: {selectedFeedbackItem.mcp_duration_ms} ms
                      </div>
                    </div>
                  </div>

                  {selectedFeedbackItem.crop_url && (
                    <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800 space-y-1">
                      <div className="text-stone-400 font-mono text-[11px]">
                        Crop: {selectedFeedbackItem.crop_url}
                      </div>
                      <img
                        src={`${selectedFeedbackItem.crop_url}?v=${cacheRev}`}
                        alt="Feedback row crop"
                        className="max-h-20 rounded border border-stone-700"
                      />
                    </div>
                  )}

                  {selectedFeedbackItem.raw_ocr_snapshot && (
                    <div>
                      <div className="text-stone-300 mb-1 uppercase font-bold font-mono text-[11px]">
                        Raw OCR Snapshot
                      </div>
                      <pre className="bg-stone-950 p-2.5 rounded-lg overflow-x-auto whitespace-pre-wrap text-emerald-300 font-mono max-h-32">
                        {selectedFeedbackItem.raw_ocr_snapshot}
                      </pre>
                    </div>
                  )}

                  {selectedFeedbackItem.extracted_snapshot && (
                    <div>
                      <div className="text-stone-300 mb-1 uppercase font-bold font-mono text-[11px]">
                        Extracted Data Snapshot (JSON)
                      </div>
                      <pre className="bg-stone-950 p-2.5 rounded-lg overflow-x-auto whitespace-pre-wrap text-amber-300 font-mono max-h-40">
                        {JSON.stringify(
                          selectedFeedbackItem.extracted_snapshot.person_fact ||
                            selectedFeedbackItem.extracted_snapshot
                              .overview_house ||
                            selectedFeedbackItem.extracted_snapshot
                              .municipal_district ||
                            selectedFeedbackItem.extracted_snapshot.row ||
                            selectedFeedbackItem.extracted_snapshot,
                          null,
                          2
                        )}
                      </pre>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
      </>
      )}

      {/* Feature Flags */}
      {(activeConsoleTab === "all" || activeConsoleTab === "flags") && (
      <div className="bg-white rounded-xl p-4 border-2 border-amber-500/80 shadow-xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 pb-3">
          <div>
            <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-amber-600" />
              <span>
                {lang === "sk"
                  ? "Feature Flags (Príznaky funkcií a viditeľnosť záložiek)"
                  : "Feature Flags (Navigation Tabs & Pipeline Features)"}
              </span>
            </h3>
            <p className="text-xs text-stone-600 mt-0.5">
              {lang === "sk" ? (
                <>
                  Okamžité zapínanie/vypínanie záložiek v hornej navigácii a funkcií sčítacích
                  hárkov (uložené na serveri a platné pre všetkých používateľov).
                </>
              ) : (
                <>
                  Instant toggling of top navigation tabs and census pipeline features (persisted
                  on the server and applied globally to all users).
                </>
              )}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => resetCensusOnly()}
              className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold cursor-pointer transition"
            >
              {lang === "sk" ? "Len Sčítacie hárky" : "Census Sheets Only"}
            </button>
            <button
              type="button"
              onClick={() => enableAllTabs()}
              className="px-3 py-1.5 rounded-lg bg-stone-900 hover:bg-stone-800 text-white text-xs font-semibold cursor-pointer transition"
            >
              {lang === "sk"
                ? "Zapnúť všetky záložky (Predvolené)"
                : "Enable All Tabs (Default)"}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Navigation Tabs Visibility */}
          <div className="bg-stone-50 rounded-xl p-3 border border-stone-200 space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                {lang === "sk"
                  ? "Záložky v hornej navigácii"
                  : "Top Navigation Tabs"}
              </span>
              <span className="text-xs text-stone-600">
                {lang === "sk" ? "Aktívne:" : "Active:"}{" "}
                {
                  NAVIGATION_FLAG_DESCRIPTORS.filter((d) => featureFlags[d.key]).length
                }{" "}
                / {NAVIGATION_FLAG_DESCRIPTORS.length}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {NAVIGATION_FLAG_DESCRIPTORS.map((desc) => {
                const enabled = featureFlags[desc.key];
                return (
                  <button
                    key={desc.key}
                    type="button"
                    onClick={() => toggleFlag(desc.key)}
                    title={lang === "sk" ? desc.descriptionSk : desc.descriptionEn}
                    className={`flex items-center justify-between gap-2 px-3 py-2 min-h-[42px] rounded-lg border text-left transition cursor-pointer ${
                      enabled
                        ? "bg-amber-50 border-amber-400 text-stone-900 shadow-2xs"
                        : "bg-white border-stone-200 text-stone-600 hover:border-stone-300"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="font-bold text-xs truncate">
                        {lang === "sk" ? desc.labelSk : desc.labelEn}
                      </div>
                      <div className="text-xs text-stone-600 truncate">
                        {lang === "sk" ? desc.shortLabelSk : desc.shortLabelEn}
                      </div>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold shrink-0 ${
                        enabled
                          ? "bg-emerald-700 text-white"
                          : "bg-stone-200 text-stone-700"
                      }`}
                    >
                      {enabled
                        ? lang === "sk"
                          ? "ZAP"
                          : "ON"
                        : lang === "sk"
                        ? "VYP"
                        : "OFF"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Pipeline & UI Feature Flags */}
          <div className="bg-stone-50 rounded-xl p-3 border border-stone-200 space-y-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-stone-700">
                {lang === "sk"
                  ? "Funkcie spracovania a zobrazenia (Pipeline & UI)"
                  : "Pipeline & UI Features"}
              </span>
              <span className="text-xs text-stone-600">
                {lang === "sk" ? "Aktívne:" : "Active:"}{" "}
                {
                  FUNCTIONAL_FLAG_DESCRIPTORS.filter((d) => featureFlags[d.key]).length
                }{" "}
                / {FUNCTIONAL_FLAG_DESCRIPTORS.length}
              </span>
            </div>

            <div className="space-y-1.5">
              {FUNCTIONAL_FLAG_DESCRIPTORS.map((desc) => {
                const enabled = featureFlags[desc.key];
                return (
                  <button
                    key={desc.key}
                    type="button"
                    onClick={() => toggleFlag(desc.key)}
                    title={lang === "sk" ? desc.descriptionSk : desc.descriptionEn}
                    className={`w-full flex items-center justify-between gap-3 px-3 py-2 min-h-[38px] rounded-lg border text-left transition cursor-pointer ${
                      enabled
                        ? "bg-emerald-50/70 border-emerald-400 text-stone-900"
                        : "bg-white border-stone-200 text-stone-600 hover:border-stone-300"
                    }`}
                  >
                    <div className="min-w-0">
                      <div className="font-bold text-xs">
                        {lang === "sk" ? desc.shortLabelSk : desc.shortLabelEn} —{" "}
                        <span className="font-medium text-stone-700">
                          {lang === "sk" ? desc.labelSk : desc.labelEn}
                        </span>
                      </div>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-bold shrink-0 ${
                        enabled
                          ? "bg-emerald-700 text-white"
                          : "bg-stone-200 text-stone-700"
                      }`}
                    >
                      {enabled
                        ? lang === "sk"
                          ? "ZAP"
                          : "ON"
                        : lang === "sk"
                        ? "VYP"
                        : "OFF"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      )}

      {/* Grid Inspector */}
      {(activeConsoleTab === "all" || activeConsoleTab === "inspector") && (
      <>
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
            <Eye className="w-4 h-4 text-amber-600" />
            <span>Grid inspector</span>
            {loadingScanDetail && (
              <RefreshCw className="w-3.5 h-3.5 text-amber-600 animate-spin" />
            )}
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {scans.map((s) => (
              <button
                key={s.id}
                onClick={() => setSelectedScanId(s.id)}
                className={`px-2.5 py-1 rounded-lg font-semibold text-xs cursor-pointer ${
                  selectedScanId === s.id
                    ? "bg-amber-600 text-white"
                    : "bg-stone-200 text-stone-800 hover:bg-stone-300"
                }`}
              >
                #{s.id} {s.filename}
              </button>
            ))}
          </div>
        </div>

        {scanDetail && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-semibold text-stone-800 text-sm">
                  Overlay (year:{" "}
                  <span className="text-amber-700">
                    {scanDetail.scan.detected_year || "pending"}
                  </span>
                  )
                </div>
                <button
                  onClick={handleResliceScan}
                  disabled={reslicingScan}
                  className="px-2.5 py-1 rounded-lg bg-amber-600 hover:bg-amber-700 text-white text-xs font-semibold flex items-center gap-1 cursor-pointer"
                >
                  <RefreshCw className={`w-3 h-3 ${reslicingScan ? "animate-spin" : ""}`} />
                  {reslicingScan ? "Slicing…" : "Re-slice"}
                </button>
              </div>
              {scanDetail.grid_metadata && (
                <div className="flex flex-wrap gap-2 text-xs font-mono bg-stone-100 border border-stone-200 rounded-lg px-2.5 py-1.5 text-stone-800">
                  <span>
                    {scanDetail.grid_metadata.slicing_mode || "angled_ruled_quad"}
                  </span>
                  <span>
                    L {(scanDetail.grid_metadata.skew_left_deg ?? 0).toFixed(2)}°
                  </span>
                  <span>
                    R {(scanDetail.grid_metadata.skew_right_deg ?? 0).toFixed(2)}°
                  </span>
                  {scanDetail.scan.detected_type !== "census" &&
                    (scanDetail.grid_metadata.spine_x ?? scanDetail.scan.spine_x ?? 0) > 0 && (
                      <span>
                        spine {(scanDetail.grid_metadata.spine_offset_px ?? 0).toFixed(1)}px
                      </span>
                    )}
                </div>
              )}
              {scanDetail.scan.overlay_url ? (
                <img
                  src={`${scanDetail.scan.overlay_url}?v=${cacheRev}`}
                  alt="OpenCV Grid Overlay"
                  className="w-full rounded-xl border border-stone-300"
                />
              ) : (
                <div className="p-6 rounded-xl bg-stone-100 text-stone-600 text-center text-sm">
                  {lang === "sk"
                    ? "Sken ešte nebol rozrezaný. Kliknite na „Prečítať“ v sekcii Knihy a strany."
                    : "Scan not yet sliced. Click 'Read' in Books & Scans to generate the grid."}
                </div>
              )}
            </div>

            <div className="space-y-2 max-h-[520px] overflow-y-auto pr-1">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-semibold text-stone-800 text-sm">
                  Stitched rows ({scanDetail.rows.length})
                </div>
                {scanDetail.scan.detected_type === "census" && (
                  <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 border border-emerald-300 text-xs font-semibold">
                    {lang === "sk"
                      ? "Sčítací hárok: Cell OCR → Extract (výrez buniek + 100 km register obcí)"
                      : "Census sheet: Cell OCR → Extract (cell composite + 100 km gazetteer)"}
                  </span>
                )}
              </div>
              {scanDetail.rows.map((row) => (
                <div
                  key={row.id}
                  className="bg-stone-50 rounded-xl p-3 border border-stone-200 space-y-1.5"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-stone-900 text-sm">
                      Row #{row.row_index} · {row.status}
                      {(row.angle_left_deg !== undefined ||
                        row.angle_right_deg !== undefined) && (
                        <span className="ml-1.5 text-xs font-mono text-stone-600">
                          L {(row.angle_left_deg ?? 0).toFixed(2)}° · R{" "}
                          {(row.angle_right_deg ?? 0).toFixed(2)}°
                        </span>
                      )}
                    </span>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <OcrFeedbackControls
                        lang={lang}
                        targetType={row.fact?.id ? "person_fact" : "row"}
                        rowId={row.id}
                        scanId={scanDetail.scan.id}
                        personFactId={row.fact?.id ?? null}
                        initialFeedback={row.ocr_feedback}
                        resetVersion={feedbackResetVersion}
                        onFeedbackSaved={(fb) => {
                          setScanDetail((prev) =>
                            prev
                              ? {
                                  ...prev,
                                  rows: prev.rows.map((rw) =>
                                    rw.id === row.id
                                      ? { ...rw, ocr_feedback: fb }
                                      : rw
                                  ),
                                }
                              : prev
                          );
                          void loadFeedbackAndStats();
                        }}
                      />
                      {(scanDetail.scan.detected_type === "census"
                        ? (["all", "ocr", "extract", "mcp"] as const)
                        : (["all", "ocr", "decision", "extract", "mcp"] as const)
                      ).map((st) => {
                        const isRunningThis =
                          rerunningRowId === row.id && rerunningStageName === st;
                        const fsTokenValid = Boolean(
                          overview?.familysearch_mcp &&
                            (typeof overview.familysearch_mcp.token_valid === "boolean"
                              ? overview.familysearch_mcp.token_valid
                              : overview.familysearch_mcp.authenticated)
                        );
                        const isMcpDisabled = st === "mcp" && !fsTokenValid;
                        return (
                          <button
                            key={st}
                            disabled={rerunningRowId === row.id || isMcpDisabled}
                            onClick={() => handleRerunRow(row.id, st)}
                            title={
                              isMcpDisabled
                                ? lang === "sk"
                                  ? "Vyhľadávanie osôb cez FamilySearch MCP vyžaduje platný OAuth token"
                                  : "Searching people via FamilySearch MCP requires a valid OAuth token"
                                : undefined
                            }
                            className={`px-2.5 py-1 min-h-[28px] rounded-md text-white text-xs font-mono flex items-center gap-1 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
                              st === "all"
                                ? "bg-amber-600 hover:bg-amber-700"
                                : "bg-stone-800 hover:bg-stone-700"
                            }`}
                          >
                            {isRunningThis ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : (
                              <Play className="w-3 h-3" />
                            )}{" "}
                            {st === "all" && scanDetail.scan.detected_type === "census"
                              ? "ocr→extract"
                              : st}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  {rerunningRowId === row.id && (
                    <div className="bg-amber-50 border border-amber-300 rounded-lg px-2.5 py-1.5 flex items-center justify-between gap-2 text-xs text-amber-950 font-semibold">
                      <span className="flex items-center gap-1.5">
                        <RefreshCw className="w-3.5 h-3.5 text-amber-700 animate-spin" />
                        <span>
                          {lang === "sk"
                            ? `Prebieha opakovanie fázy '${rerunningStageName}' pre riadok #${row.row_index}…`
                            : `Re-running stage '${rerunningStageName}' for row #${row.row_index}…`}
                        </span>
                      </span>
                    </div>
                  )}
                  {row.stitched_crop_url && (
                    <img
                      src={`${row.stitched_crop_url}?v=${cacheRev}`}
                      alt={`Row ${row.row_index}`}
                      className="max-h-14 rounded border border-stone-200"
                    />
                  )}
                  {row.raw_ocr_combined && (
                    <pre className="text-xs font-mono bg-stone-900 text-emerald-300 p-2 rounded-lg overflow-x-auto whitespace-pre-wrap">
                      {row.raw_ocr_combined}
                    </pre>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Trace Log */}
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
            <Terminal className="w-4 h-4 text-emerald-700" />
            Trace log ({traces.length})
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {["", "layout", "ocr", "decision", "extract", "mcp_place", "mcp_search"].map(
              (st) => (
                <button
                  key={st}
                  onClick={() => setStageFilter(st)}
                  className={`px-3 py-1.5 min-h-[32px] rounded-lg font-semibold text-xs cursor-pointer ${
                    stageFilter === st
                      ? "bg-stone-900 text-white"
                      : "bg-stone-200 text-stone-800 hover:bg-stone-300"
                  }`}
                >
                  {st === "" ? "All" : st.toUpperCase()}
                </button>
              )
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          <div className="lg:col-span-5 max-h-[520px] overflow-y-auto space-y-1.5 pr-1">
            {traces.map((t) => (
              <button
                key={t.id}
                onClick={() => setSelectedTrace(t)}
                className={`w-full text-left p-2.5 rounded-xl border transition cursor-pointer ${
                  selectedTrace?.id === t.id
                    ? "bg-amber-50 border-amber-500"
                    : "bg-stone-50 border-stone-200 hover:border-stone-400"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="px-1.5 py-0.5 rounded bg-stone-900 text-amber-300 font-mono text-xs font-bold uppercase">
                    {t.stage}
                  </span>
                  <span className="text-xs font-mono text-stone-600">
                    {Math.round(t.duration_ms || 0)} ms · #{t.row_id ?? "—"}
                  </span>
                </div>
                <div className="font-mono text-xs font-semibold text-stone-900 mt-1 truncate flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5 text-stone-600 shrink-0" />
                  <span>{t.model_or_tool || t.model_name || "—"}</span>
                </div>
              </button>
            ))}
          </div>

          <div className="lg:col-span-7">
            {selectedTrace ? (
              <div className="bg-stone-900 text-stone-100 rounded-xl p-4 space-y-3 font-mono text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-700 pb-2">
                  <div>
                    <span className="text-amber-400 font-bold">#{selectedTrace.id}</span>{" "}
                    · {selectedTrace.stage} ·{" "}
                    {selectedTrace.model_or_tool || selectedTrace.model_name || "—"}
                  </div>
                  <div className="text-emerald-400">
                    {Math.round(selectedTrace.duration_ms || 0)} ms ({selectedTrace.status})
                  </div>
                </div>
                {selectedTrace.image_path && (
                  <div className="bg-stone-950 p-2.5 rounded-lg border border-stone-800 space-y-1.5">
                    <div className="text-amber-300 uppercase font-bold text-xs flex items-center justify-between">
                      <span>
                        {lang === "sk"
                          ? "Priložený výrez (Image attached to model)"
                          : "Image attached to model"}
                      </span>
                      <span className="text-stone-400 font-normal">
                        {selectedTrace.image_path}
                      </span>
                    </div>
                    <img
                      src={`${selectedTrace.image_path}?v=${cacheRev}`}
                      alt="Attached crop"
                      className="max-h-24 rounded border border-stone-700"
                    />
                  </div>
                )}
                <div>
                  <div className="text-stone-300 mb-1 uppercase font-bold text-xs">
                    Prompt
                  </div>
                  <pre className="bg-stone-950 p-2.5 rounded-lg overflow-x-auto whitespace-pre-wrap text-stone-200 max-h-36">
                    {selectedTrace.prompt_text}
                  </pre>
                </div>
                <div>
                  <div className="text-stone-300 mb-1 uppercase font-bold text-xs">
                    Output
                  </div>
                  <pre className="bg-stone-950 p-2.5 rounded-lg overflow-x-auto whitespace-pre-wrap text-emerald-300 max-h-44">
                    {selectedTrace.response_text || selectedTrace.raw_response || "—"}
                  </pre>
                </div>
                {selectedTrace.parsed_json && (
                  <div>
                    <div className="text-stone-300 mb-1 uppercase font-bold text-xs">
                      JSON
                    </div>
                    <pre className="bg-stone-950 p-2.5 rounded-lg overflow-x-auto whitespace-pre-wrap text-amber-300 max-h-44">
                      {selectedTrace.parsed_json}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-8 text-center text-stone-500 bg-stone-50 rounded-xl border border-stone-200 text-sm">
                {lang === "sk"
                  ? "Vyberte záznam vľavo pre zobrazenie detailu."
                  : "Select a trace on the left."}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Prompt Playground */}
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-600" />
            <span>AI Prompt Playground</span>
          </h3>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-stone-600 font-semibold">Model:</span>
            <input
              type="text"
              list="all-models-datalist"
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              className="px-2.5 py-1 rounded-lg border border-stone-300 font-mono text-xs min-w-[200px]"
            />
            <input
              type="number"
              value={customRowId}
              onChange={(e) => setCustomRowId(e.target.value)}
              placeholder={lang === "sk" ? "Row ID (voliteľné)" : "Row ID (optional)"}
              className="w-32 px-2.5 py-1 rounded-lg border border-stone-300 text-xs"
            />
            <label className="inline-flex items-center gap-1 text-stone-700 cursor-pointer">
              <input
                type="checkbox"
                checked={customUseImage}
                onChange={(e) => setCustomUseImage(e.target.checked)}
              />
              <span>
                {lang === "sk" ? "Priložiť výrez riadku" : "Attach row image"}
              </span>
            </label>
          </div>
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleRunCustomPrompt();
              }
            }}
            placeholder={
              lang === "sk"
                ? "Zadajte testovací prompt pre vybraný model…"
                : "Enter a test prompt for the selected model…"
            }
            className="flex-1 px-3 py-2 rounded-lg border border-stone-300 text-xs font-mono"
          />
          <button
            type="button"
            disabled={runningCustomPrompt || !customPrompt.trim()}
            onClick={handleRunCustomPrompt}
            className="px-3.5 py-2 rounded-lg bg-stone-900 hover:bg-stone-800 disabled:opacity-50 text-white text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
            <span>
              {runningCustomPrompt
                ? lang === "sk"
                  ? "Odosielam…"
                  : "Sending…"
                : lang === "sk"
                ? "Odoslať"
                : "Send"}
            </span>
          </button>
        </div>

        {customError && (
          <div className="bg-red-50 border border-red-400 rounded-xl p-3 flex items-center gap-2 text-red-900 font-medium text-xs">
            <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
            <span>{customError}</span>
          </div>
        )}

        {customResult && (
          <pre className="bg-stone-950 text-emerald-300 p-3 rounded-xl text-xs font-mono overflow-x-auto whitespace-pre-wrap max-h-48">
            {customResult}
          </pre>
        )}
      </div>
      </>
      )}
    </div>
  );
}
