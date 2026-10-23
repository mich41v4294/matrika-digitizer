import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertCircle,
  Check,
  Cpu,
  Eye,
  Play,
  RefreshCw,
  Send,
  Server,
  Sliders,
  Sparkles,
  Terminal,
} from "lucide-react";
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
  }>;
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
    titleSk: "ocr — Čítanie rukopisu (OCR)",
    titleEn: "ocr — Handwriting Recognition (OCR)",
    subtitleSk:
      "Vizuálne čítanie riadkov a buniek (predvolene: Local AI · glm-ocr-g32-mixed_4_8-mlx)",
    subtitleEn:
      "Visual line and cell transcription (default: Local AI · glm-ocr-g32-mixed_4_8-mlx)",
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
      "Rozhodovanie o istote prečítania v cirkevných matrikách (sčítacie hárky 1930/1940 idú priamo z Cell OCR → Extract)",
    subtitleEn:
      "Confidence triage for church register rows (1930/1940 census sheets go directly from Cell OCR → Extract)",
    defaultLocalModel: "d1-3b-mlx",
    defaultFreellmModel: "gpt-4o-mini",
  },
  {
    key: "extract",
    titleSk:
      "extract — Vizuálna extrakcia údajov + 100 km register obcí + konsenzus (LLM Extract)",
    titleEn:
      "extract — Structured Fact Extraction + 100 km Gazetteer + Surname Consensus (LLM Extract)",
    subtitleSk:
      "Priama extrakcia z výrezu buniek (row_XX_extract_composite.jpg) + 100 km register obcí + konsenzus priezviska v domácnosti",
    subtitleEn:
      "Direct extraction from composite cell crops + 100 km municipality gazetteer + household surname consensus",
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
    void fetchBackendFeatureFlags();
  }, []);

  useEffect(() => {
    loadDebug();
    const interval = setInterval(() => {
      if (document.visibilityState === "hidden") return;
      void loadDebug();
    }, 6000);
    return () => clearInterval(interval);
  }, [stageFilter]);

  useEffect(() => {
    if (!selectedScanId) return;
    fetch(`/api/scans/${selectedScanId}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d) setScanDetail(d);
      });
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
      await loadDebug();
      refreshStatus();
    } finally {
      setRerunningRowId(null);
    }
  };

  const handleResliceScan = async () => {
    if (!selectedScanId) return;
    setReslicingScan(true);
    try {
      await fetch(`/api/scans/${selectedScanId}/slice`, { method: "POST" });
      const res = await fetch(`/api/scans/${selectedScanId}`);
      if (res.ok) {
        setScanDetail(await res.json());
      }
      setCacheRev(Date.now());
      await loadDebug();
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

      {/* Top AI Status & Per-Pipeline-Stage Provider / Model Configuration Card */}
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
              <Check className="w-3.5 h-3.5" />
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
              onClick={() => {
                loadDebug();
                loadAiConfig();
              }}
              className="px-3 py-1.5 rounded-lg bg-stone-800 hover:bg-stone-700 border border-stone-600 font-semibold text-xs flex items-center gap-1.5 cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Refresh
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

      {/* Feature Flags */}
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
                  hárkov (uložené v <code className="font-mono text-stone-800">localStorage</code> a
                  synchronizované so serverom).
                </>
              ) : (
                <>
                  Instant toggling of top navigation tabs and census pipeline features (persisted
                  in <code className="font-mono text-stone-800">localStorage</code> and synced with
                  backend).
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

      {/* Grid Inspector */}
      <div className="bg-white rounded-xl p-4 border border-stone-300 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="text-base font-bold text-stone-900 flex items-center gap-2">
            <Eye className="w-4 h-4 text-amber-600" />
            Grid inspector
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
                    <div className="flex flex-wrap gap-1">
                      {(scanDetail.scan.detected_type === "census"
                        ? (["all", "ocr", "extract", "mcp"] as const)
                        : (["all", "ocr", "decision", "extract", "mcp"] as const)
                      ).map((st) => (
                        <button
                          key={st}
                          disabled={rerunningRowId === row.id}
                          onClick={() => handleRerunRow(row.id, st)}
                          className={`px-2.5 py-1 min-h-[28px] rounded-md text-white text-xs font-mono flex items-center gap-1 cursor-pointer ${
                            st === "all"
                              ? "bg-amber-600 hover:bg-amber-700"
                              : "bg-stone-800 hover:bg-stone-700"
                          }`}
                        >
                          <Play className="w-3 h-3" />{" "}
                          {st === "all" && scanDetail.scan.detected_type === "census"
                            ? "ocr→extract"
                            : st}
                        </button>
                      ))}
                    </div>
                  </div>
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
    </div>
  );
}
