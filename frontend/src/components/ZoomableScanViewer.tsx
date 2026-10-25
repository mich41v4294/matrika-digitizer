import React, { useEffect, useRef, useState } from "react";
import {
  Eye,
  ImageOff,
  Maximize2,
  Minimize2,
  Move,
  Pin,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useFeatureFlags } from "../featureFlags";

export interface ZoomableScanViewerProps {
  src?: string | null;
  alt?: string;
  title?: string;
  subtitle?: string;
  defaultHeightClass?: string;
  expandedHeightClass?: string;
  children?: React.ReactNode;
  headerRight?: React.ReactNode;
  className?: string;
  emptyMessage?: string;
  lang?: "sk" | "en";
}

const ZOOM_PRESETS = [
  { key: "fit", labelSk: "Prispôsobiť", labelEn: "Fit", value: 100, isFit: true },
  { key: "100", labelSk: "100%", labelEn: "100%", value: 100, isFit: false },
  { key: "150", labelSk: "150%", labelEn: "150%", value: 150, isFit: false },
  { key: "200", labelSk: "200%", labelEn: "200%", value: 200, isFit: false },
  { key: "300", labelSk: "300%", labelEn: "300%", value: 300, isFit: false },
] as const;

function resolveLang(explicitLang?: "sk" | "en"): "sk" | "en" {
  if (explicitLang === "sk" || explicitLang === "en") return explicitLang;
  if (typeof window !== "undefined") {
    const stored = window.localStorage.getItem("matrika_lang");
    if (stored === "en") return "en";
  }
  return "sk";
}

export function ZoomableScanViewer({
  src,
  alt = "Sken dokumentu",
  title,
  subtitle,
  defaultHeightClass = "max-h-[560px]",
  expandedHeightClass = "max-h-[85vh]",
  children,
  headerRight,
  className = "",
  emptyMessage,
  lang: explicitLang,
}: ZoomableScanViewerProps) {
  const lang = resolveLang(explicitLang);
  const { flags: featureFlags } = useFeatureFlags();
  const manualZoomEnabled = featureFlags.manual_scan_zoom;

  const [zoomPct, setZoomPct] = useState<number>(100);
  const [fitMode, setFitMode] = useState<boolean>(true);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [imgError, setImgError] = useState<boolean>(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const dragStartRef = useRef<{
    x: number;
    y: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);

  useEffect(() => {
    setImgError(false);
  }, [src]);

  const clampZoom = (val: number) => Math.max(50, Math.min(400, Math.round(val)));

  const handleZoomChange = (nextZoom: number, isFit = false) => {
    if (!manualZoomEnabled) return;
    const clamped = clampZoom(nextZoom);
    setZoomPct(clamped);
    setFitMode(isFit);
  };

  const handleReset = () => {
    setZoomPct(100);
    setFitMode(true);
    if (viewportRef.current) {
      viewportRef.current.scrollTo({ left: 0, top: 0, behavior: "smooth" });
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (!manualZoomEnabled) return;
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 25 : -25;
      setFitMode(false);
      setZoomPct((prev) => clampZoom(prev + delta));
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!manualZoomEnabled || e.button !== 0 || !viewportRef.current) return;
    setIsDragging(true);
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      scrollLeft: viewportRef.current.scrollLeft,
      scrollTop: viewportRef.current.scrollTop,
    };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || !dragStartRef.current || !viewportRef.current) return;
    e.preventDefault();
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    viewportRef.current.scrollLeft = dragStartRef.current.scrollLeft - dx;
    viewportRef.current.scrollTop = dragStartRef.current.scrollTop - dy;
  };

  const stopDragging = () => {
    setIsDragging(false);
    dragStartRef.current = null;
  };

  const heightClass = expanded ? expandedHeightClass : defaultHeightClass;
  const effectiveFitMode = !manualZoomEnabled ? true : fitMode;
  const resolvedEmptyMessage =
    emptyMessage ||
    (lang === "sk"
      ? "Sken zatiaľ nie je k dispozícii."
      : "Scan image is not available yet.");

  return (
    <div className={`bg-stone-900 rounded-xl border border-stone-700 overflow-hidden ${className}`}>
      {/* Control Toolbar */}
      <div className="px-3 py-2 bg-stone-950/90 border-b border-stone-800 flex flex-wrap items-center justify-between gap-2 text-xs text-stone-200">
        <div className="flex flex-wrap items-center gap-2 min-w-0">
          {(title || subtitle) && (
            <div className="mr-2 min-w-0 text-xs sm:text-sm">
              {title && <span className="font-bold text-amber-400">{title}</span>}
              {subtitle && <span className="text-stone-300 ml-1.5">{subtitle}</span>}
            </div>
          )}

          {manualZoomEnabled && (
            <>
              {/* Zoom Out */}
              <button
                type="button"
                onClick={() => handleZoomChange(zoomPct - 25, false)}
                disabled={!effectiveFitMode && zoomPct <= 50}
                title={lang === "sk" ? "Oddialiť (-25%)" : "Zoom out (-25%)"}
                className="px-2.5 py-1.5 min-h-[32px] min-w-[36px] rounded-md bg-stone-800 hover:bg-stone-700 disabled:opacity-40 text-stone-100 font-semibold inline-flex items-center justify-center gap-1 cursor-pointer"
              >
                <ZoomOut className="w-3.5 h-3.5" />
                <span>-</span>
              </button>

              {/* Preset Pills */}
              <div className="flex flex-wrap items-center gap-1">
                {ZOOM_PRESETS.map((preset) => {
                  const active = preset.isFit
                    ? effectiveFitMode
                    : !effectiveFitMode && zoomPct === preset.value;
                  const label = lang === "sk" ? preset.labelSk : preset.labelEn;
                  return (
                    <button
                      key={preset.key}
                      type="button"
                      onClick={() => handleZoomChange(preset.value, preset.isFit)}
                      className={`px-2.5 py-1.5 min-h-[32px] rounded-md font-semibold transition cursor-pointer ${
                        active
                          ? "bg-amber-500 text-stone-950"
                          : "bg-stone-800 text-stone-200 hover:bg-stone-700"
                      }`}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              {/* Zoom In */}
              <button
                type="button"
                onClick={() => handleZoomChange(zoomPct + 25, false)}
                disabled={!effectiveFitMode && zoomPct >= 400}
                title={lang === "sk" ? "Priblížiť (+25%)" : "Zoom in (+25%)"}
                className="px-2.5 py-1.5 min-h-[32px] min-w-[36px] rounded-md bg-stone-800 hover:bg-stone-700 disabled:opacity-40 text-stone-100 font-semibold inline-flex items-center justify-center gap-1 cursor-pointer"
              >
                <ZoomIn className="w-3.5 h-3.5" />
                <span>+</span>
              </button>

              <span className="px-2.5 py-1 rounded-md bg-stone-900 border border-stone-700 font-mono text-xs text-amber-300">
                {effectiveFitMode
                  ? lang === "sk"
                    ? "Fit"
                    : "Fit"
                  : `${zoomPct}%`}
              </span>

              {/* Reset */}
              <button
                type="button"
                onClick={handleReset}
                title={
                  lang === "sk"
                    ? "Resetovať priblíženie a pozíciu"
                    : "Reset zoom and pan position"
                }
                className="px-2.5 py-1.5 min-h-[32px] rounded-md bg-stone-800 hover:bg-stone-700 text-stone-200 font-semibold inline-flex items-center gap-1 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {manualZoomEnabled && (
            <span className="hidden sm:inline-flex items-center gap-1 text-xs text-stone-300">
              <Move className="w-3.5 h-3.5" />
              <span>
                {lang === "sk"
                  ? "Ťahajte myšou pre posun · Ctrl+koliesko pre zoom"
                  : "Drag to pan · Ctrl+wheel to zoom"}
              </span>
            </span>
          )}

          {headerRight}

          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className={`px-3 py-1.5 min-h-[32px] rounded-md font-semibold inline-flex items-center gap-1.5 cursor-pointer ${
              expanded
                ? "bg-amber-600 text-white hover:bg-amber-500"
                : "bg-stone-800 text-stone-200 hover:bg-stone-700"
            }`}
          >
            {expanded ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span>{lang === "sk" ? "Menšie okno" : "Compact view"}</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5" />
                <span>{lang === "sk" ? "Väčšie okno" : "Expand view"}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Scrollable & Pannable Viewport */}
      <div
        ref={viewportRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={stopDragging}
        onMouseLeave={stopDragging}
        className={`overflow-auto p-2 select-none ${heightClass} ${
          manualZoomEnabled
            ? isDragging
              ? "cursor-grabbing"
              : "cursor-grab"
            : "cursor-default"
        }`}
      >
        {src && !imgError ? (
          <div
            style={effectiveFitMode ? undefined : { width: `${zoomPct}%` }}
            className={`relative mx-auto transition-[width] duration-150 ease-out ${
              effectiveFitMode ? "w-fit max-w-full" : ""
            }`}
          >
            <img
              src={src}
              alt={alt}
              draggable={false}
              onError={() => setImgError(true)}
              className={`block rounded select-none pointer-events-none ${
                effectiveFitMode
                  ? expanded
                    ? "max-h-[78vh] w-auto max-w-full mx-auto"
                    : "max-h-[480px] w-auto max-w-full mx-auto"
                  : "w-full h-auto"
              }`}
            />
            {children && (
              <div className="absolute inset-0 pointer-events-none">{children}</div>
            )}
          </div>
        ) : (
          <div className="text-stone-400 text-sm py-12 text-center flex flex-col items-center justify-center gap-2">
            <ImageOff className="w-6 h-6 text-stone-500" />
            <span>
              {imgError
                ? lang === "sk"
                  ? "Obrázok skenu sa nepodarilo načítať (súbor nebol nájdený v lokálnom úložisku)."
                  : "Scan image could not be loaded (file not found in local storage)."
                : resolvedEmptyMessage}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export type CensusFieldCropKey =
  | "row_num"
  | "surname"
  | "given_name"
  | "relation"
  | "sex"
  | "dob_marital"
  | "birthplace"
  | "domicile_citizenship"
  | "nationality_religion"
  | "occupation_notes"
  | "ov_ordinal_settlement"
  | "ov_house_num"
  | "ov_owner_name"
  | "ov_counts"
  | "ov_remarks";

export interface FieldCropRangeSpec {
  key: CensusFieldCropKey;
  xStartPct: number;
  xEndPct: number;
  labelSk: string;
  labelEn: string;
  shortSk: string;
  shortEn: string;
  group: "household" | "overview";
}

export interface ActiveFieldFocus {
  key: CensusFieldCropKey;
  valuePreview?: string | null;
}

export const CENSUS_FIELD_CROP_RANGES: Record<CensusFieldCropKey, FieldCropRangeSpec> = {
  row_num: {
    key: "row_num",
    xStartPct: 0.0,
    xEndPct: 0.075,
    labelSk: "Stĺpce 1–2: Číslo bytu a radové č.",
    labelEn: "Cols 1–2: Dwelling & row #",
    shortSk: "Č. riadku",
    shortEn: "Row #",
    group: "household",
  },
  surname: {
    key: "surname",
    xStartPct: 0.032,
    xEndPct: 0.168,
    labelSk: "Stĺpec 3: Priezvisko (a rodné priezvisko)",
    labelEn: "Col 3: Surname (& maiden name)",
    shortSk: "Priezvisko",
    shortEn: "Surname",
    group: "household",
  },
  given_name: {
    key: "given_name",
    xStartPct: 0.128,
    xEndPct: 0.228,
    labelSk: "Stĺpec 4: Krstné meno",
    labelEn: "Col 4: Given name",
    shortSk: "Meno",
    shortEn: "Given name",
    group: "household",
  },
  relation: {
    key: "relation",
    xStartPct: 0.198,
    xEndPct: 0.268,
    labelSk: "Stĺpec 5: Pomer k hlave domácnosti",
    labelEn: "Col 5: Relation to head",
    shortSk: "Vzťah",
    shortEn: "Relation",
    group: "household",
  },
  sex: {
    key: "sex",
    xStartPct: 0.242,
    xEndPct: 0.292,
    labelSk: "Stĺpec 6: Pohlavie",
    labelEn: "Col 6: Sex",
    shortSk: "Pohlavie",
    shortEn: "Sex",
    group: "household",
  },
  dob_marital: {
    key: "dob_marital",
    xStartPct: 0.262,
    xEndPct: 0.388,
    labelSk: "Stĺpce 7–10: Dátum narodenia (deň, mesiac, rok) a stav",
    labelEn: "Cols 7–10: Date of birth & marital status",
    shortSk: "Dátum nar.",
    shortEn: "Date of birth",
    group: "household",
  },
  birthplace: {
    key: "birthplace",
    xStartPct: 0.355,
    xEndPct: 0.518,
    labelSk: "Stĺpce 11–12: Rodisko (obec, okres, krajina)",
    labelEn: "Cols 11–12: Birthplace (municipality, district)",
    shortSk: "Rodisko",
    shortEn: "Birthplace",
    group: "household",
  },
  domicile_citizenship: {
    key: "domicile_citizenship",
    xStartPct: 0.492,
    xEndPct: 0.578,
    labelSk: "Stĺpce 13–14: Domovská a štátna príslušnosť",
    labelEn: "Cols 13–14: Domicile & citizenship",
    shortSk: "Príslušnosť",
    shortEn: "Domicile",
    group: "household",
  },
  nationality_religion: {
    key: "nationality_religion",
    xStartPct: 0.552,
    xEndPct: 0.698,
    labelSk: "Stĺpce 15–17: Národnosť, náboženstvo a gramotnosť",
    labelEn: "Cols 15–17: Nationality, religion & literacy",
    shortSk: "Národ./Nábož.",
    shortEn: "Nat./Relig.",
    group: "household",
  },
  occupation_notes: {
    key: "occupation_notes",
    xStartPct: 0.675,
    xEndPct: 1.0,
    labelSk: "Stĺpce 18–24: Povolanie, postavenie, závod a poznámky",
    labelEn: "Cols 18–24: Occupation, workplace & remarks",
    shortSk: "Povolanie",
    shortEn: "Occupation",
    group: "household",
  },
  ov_ordinal_settlement: {
    key: "ov_ordinal_settlement",
    xStartPct: 0.0,
    xEndPct: 0.28,
    labelSk: "Stĺpce 1–2: Radové č. a meno osady/ulice",
    labelEn: "Cols 1–2: Ordinal # & settlement/street",
    shortSk: "Osada",
    shortEn: "Settlement",
    group: "overview",
  },
  ov_house_num: {
    key: "ov_house_num",
    xStartPct: 0.22,
    xEndPct: 0.39,
    labelSk: "Stĺpec 3: Popisné číslo domu",
    labelEn: "Col 3: House conscription #",
    shortSk: "Č. domu",
    shortEn: "House #",
    group: "overview",
  },
  ov_owner_name: {
    key: "ov_owner_name",
    xStartPct: 0.34,
    xEndPct: 0.68,
    labelSk: "Stĺpec 6: Meno majiteľa domu",
    labelEn: "Col 6: House owner name",
    shortSk: "Majiteľ domu",
    shortEn: "Owner",
    group: "overview",
  },
  ov_counts: {
    key: "ov_counts",
    xStartPct: 0.63,
    xEndPct: 0.89,
    labelSk: "Stĺpce 7–8: Počet bytov a prítomných osôb",
    labelEn: "Cols 7–8: Dwelling units & present residents",
    shortSk: "Počty osôb",
    shortEn: "Counts",
    group: "overview",
  },
  ov_remarks: {
    key: "ov_remarks",
    xStartPct: 0.82,
    xEndPct: 1.0,
    labelSk: "Stĺpec 9: Poznámka",
    labelEn: "Col 9: Remarks",
    shortSk: "Poznámka",
    shortEn: "Remarks",
    group: "overview",
  },
};

export const HOUSEHOLD_FIELD_PILLS: CensusFieldCropKey[] = [
  "surname",
  "given_name",
  "relation",
  "dob_marital",
  "birthplace",
  "domicile_citizenship",
  "nationality_religion",
  "occupation_notes",
];

export interface HoverZoomRowCropProps {
  cropUrl?: string | null;
  headerCropUrl?: string | null;
  headerYEnd?: number | null;
  scanUrl?: string | null;
  yStart?: number | null;
  yEnd?: number | null;
  scanHeight?: number | null;
  scanWidth?: number | null;
  rowLabel?: string;
  activeFieldFocus?: ActiveFieldFocus | null;
  onSelectFieldFocus?: (focus: ActiveFieldFocus | null) => void;
  showFieldPills?: boolean;
  className?: string;
  lang?: "sk" | "en";
}

function deriveHeaderCropUrlFromRowCrop(
  cropUrl?: string | null,
  explicitHeaderUrl?: string | null
): string | null {
  if (explicitHeaderUrl && explicitHeaderUrl.trim()) {
    return explicitHeaderUrl.trim();
  }
  if (!cropUrl) return null;
  const match = cropUrl.match(/(\/media\/crops\/scan_\d+)\/row_[^/?#]+/i);
  if (match && match[1]) {
    return `${match[1]}/header.jpg`;
  }
  return null;
}

export function HoverZoomRowCrop({
  cropUrl,
  headerCropUrl,
  headerYEnd,
  scanUrl,
  yStart,
  yEnd,
  scanHeight,
  scanWidth,
  rowLabel,
  activeFieldFocus = null,
  onSelectFieldFocus,
  showFieldPills = false,
  className = "",
  lang: explicitLang,
}: HoverZoomRowCropProps) {
  const lang = resolveLang(explicitLang);
  const { flags: featureFlags } = useFeatureFlags();
  const hoverZoomEnabled = featureFlags.hover_row_zoom;

  const [isHovered, setIsHovered] = useState<boolean>(false);
  const [isPinned, setIsPinned] = useState<boolean>(false);
  const [forceFullRow, setForceFullRow] = useState<boolean>(false);
  const [contextPaddingMult, setContextPaddingMult] = useState<number>(3.5);
  const [cropImgFailed, setCropImgFailed] = useState<boolean>(false);
  const [headerImgFailed, setHeaderImgFailed] = useState<boolean>(false);
  const [headerNaturalDim, setHeaderNaturalDim] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [naturalScanDim, setNaturalScanDim] = useState<{
    width: number;
    height: number;
  } | null>(
    scanWidth && scanHeight && scanWidth > 0 && scanHeight > 0
      ? { width: scanWidth, height: scanHeight }
      : null
  );
  const [panRatio, setPanRatio] = useState<{ x: number; y: number }>({ x: 0.5, y: 0.5 });

  // Whenever the hovered cell changes to a new field, automatically exit forceFullRow so the field crop shows immediately
  useEffect(() => {
    if (activeFieldFocus?.key) {
      setForceFullRow(false);
    }
  }, [activeFieldFocus?.key, activeFieldFocus?.valuePreview]);

  useEffect(() => {
    setCropImgFailed(false);
  }, [cropUrl]);

  const resolvedHeaderCropUrl = deriveHeaderCropUrlFromRowCrop(cropUrl, headerCropUrl);

  useEffect(() => {
    setHeaderImgFailed(false);
    setHeaderNaturalDim(null);
  }, [resolvedHeaderCropUrl]);

  useEffect(() => {
    if (!resolvedHeaderCropUrl || headerNaturalDim || headerImgFailed) {
      return;
    }
    let cancelled = false;
    const hImg = new Image();
    hImg.onload = () => {
      if (!cancelled && hImg.naturalWidth > 0 && hImg.naturalHeight > 0) {
        setHeaderNaturalDim({
          width: hImg.naturalWidth,
          height: hImg.naturalHeight,
        });
      }
    };
    hImg.onerror = () => {
      if (!cancelled) {
        setHeaderImgFailed(true);
      }
    };
    hImg.src = resolvedHeaderCropUrl;
    return () => {
      cancelled = true;
    };
  }, [resolvedHeaderCropUrl, headerNaturalDim, headerImgFailed]);

  useEffect(() => {
    if (!scanUrl || !(isHovered || isPinned)) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled && img.naturalWidth > 0 && img.naturalHeight > 0) {
        setNaturalScanDim({
          width: img.naturalWidth,
          height: img.naturalHeight,
        });
      }
    };
    img.src = scanUrl;
    return () => {
      cancelled = true;
    };
  }, [scanUrl, isHovered, isPinned]);

  if (!cropUrl) {
    return null;
  }

  const effectiveScanHeight =
    naturalScanDim?.height || (scanHeight && scanHeight > 0 ? scanHeight : 3536);
  const effectiveScanWidth =
    naturalScanDim?.width || (scanWidth && scanWidth > 0 ? scanWidth : 5000);

  const hasValidRowBand =
    hoverZoomEnabled &&
    Boolean(scanUrl) &&
    yStart != null &&
    yEnd != null &&
    Number(yEnd) > Number(yStart) &&
    Number(yEnd) > 0;

  const showContext = hoverZoomEnabled && (isHovered || isPinned);

  const y0 = hasValidRowBand ? Math.max(0, Number(yStart)) : 0;
  const y1 = hasValidRowBand
    ? Math.min(effectiveScanHeight, Math.max(y0 + 24, Number(yEnd)))
    : 0;
  const rowH = Math.max(24, y1 - y0);
  const padPx = Math.round(rowH * contextPaddingMult);
  const winY0 = Math.max(0, y0 - padPx);
  const winY1 = Math.min(effectiveScanHeight, y1 + padPx);
  const winH = Math.max(rowH * 2, winY1 - winY0);

  const canUseHeaderCropImg = Boolean(resolvedHeaderCropUrl && !headerImgFailed);
  const hdrW = headerNaturalDim?.width || effectiveScanWidth;
  const hdrH = headerNaturalDim?.height || Math.round(effectiveScanHeight * 0.25);
  const hdrSliceY0 = Math.round(hdrH * 0.42);
  const hdrSliceH = Math.max(48, hdrH - hdrSliceY0);

  const fallbackScanHeaderYEnd =
    headerYEnd && headerYEnd > 40
      ? Number(headerYEnd)
      : y0 > 120
      ? Math.min(y0, Math.round(effectiveScanHeight * 0.38))
      : Math.round(effectiveScanHeight * 0.30);
  const fallbackScanHeaderY0 = Math.max(
    0,
    fallbackScanHeaderYEnd - Math.round(effectiveScanHeight * 0.14)
  );
  const fallbackScanHeaderH = Math.max(
    60,
    fallbackScanHeaderYEnd - fallbackScanHeaderY0
  );

  const fieldSpec =
    !forceFullRow && activeFieldFocus?.key
      ? CENSUS_FIELD_CROP_RANGES[activeFieldFocus.key] || null
      : null;

  const sliceX0 = fieldSpec ? fieldSpec.xStartPct : 0;
  const sliceX1 = fieldSpec ? fieldSpec.xEndPct : 1;
  const sliceWidthFraction = Math.max(0.05, sliceX1 - sliceX0);

  const fieldSliceImgStyle: React.CSSProperties | undefined = fieldSpec
    ? {
        width: `${(1 / sliceWidthFraction) * 100}%`,
        maxWidth: "none",
        marginLeft: `${(-sliceX0 / sliceWidthFraction) * 100}%`,
      }
    : undefined;

  const zoomScale = !fieldSpec && showContext && !hasValidRowBand ? 1.65 : 1;
  const sharedHorizTransformStyle: React.CSSProperties | undefined = fieldSliceImgStyle
    ? fieldSliceImgStyle
    : zoomScale > 1
    ? {
        transform: `scale(${zoomScale})`,
        transformOrigin: `${(panRatio.x * 100).toFixed(1)}% 50%`,
      }
    : undefined;

  const renderPinnedColumnHeader = (syncZoomWithRowStrip: boolean) => {
    const transformStyle = syncZoomWithRowStrip ? sharedHorizTransformStyle : undefined;
    if (canUseHeaderCropImg && resolvedHeaderCropUrl) {
      const hdrViewX0 = fieldSpec ? Math.round(hdrW * sliceX0) : 0;
      const hdrViewW = fieldSpec ? Math.max(20, Math.round(hdrW * sliceWidthFraction)) : hdrW;
      return (
        <div className="relative overflow-hidden bg-stone-950 border-b border-stone-700">
          <span className="absolute top-1 left-1.5 z-10 px-2 py-0.5 rounded bg-stone-950/90 text-amber-300 border border-stone-700 text-[11px] font-bold uppercase tracking-wider pointer-events-none shadow">
            {fieldSpec
              ? lang === "sk"
                ? fieldSpec.labelSk
                : fieldSpec.labelEn
              : lang === "sk"
              ? "Hlavička stĺpcov"
              : "Column header"}
          </span>
          {headerNaturalDim ? (
            <svg
              viewBox={`${hdrViewX0} ${hdrSliceY0} ${hdrViewW} ${hdrSliceH}`}
              style={fieldSpec ? undefined : transformStyle}
              className="w-full h-auto max-h-24 block transition-transform duration-150 ease-out"
              preserveAspectRatio="none"
            >
              <image
                href={resolvedHeaderCropUrl}
                x={0}
                y={0}
                width={hdrW}
                height={hdrH}
                preserveAspectRatio="none"
              />
            </svg>
          ) : (
            <img
              src={resolvedHeaderCropUrl}
              alt={lang === "sk" ? "Hlavička stĺpcov" : "Column header"}
              style={transformStyle}
              onError={() => setHeaderImgFailed(true)}
              className="w-full h-auto max-h-24 object-cover object-bottom block transition-transform duration-150 ease-out"
            />
          )}
        </div>
      );
    }

    if (scanUrl) {
      const scanViewX0 = fieldSpec ? Math.round(effectiveScanWidth * sliceX0) : 0;
      const scanViewW = fieldSpec
        ? Math.max(20, Math.round(effectiveScanWidth * sliceWidthFraction))
        : effectiveScanWidth;
      return (
        <div className="relative overflow-hidden bg-stone-950 border-b border-stone-700">
          <span className="absolute top-1 left-1.5 z-10 px-2 py-0.5 rounded bg-stone-950/90 text-amber-300 border border-stone-700 text-[11px] font-bold uppercase tracking-wider pointer-events-none shadow">
            {fieldSpec
              ? lang === "sk"
                ? fieldSpec.labelSk
                : fieldSpec.labelEn
              : lang === "sk"
              ? "Hlavička stĺpcov"
              : "Column header"}
          </span>
          <svg
            viewBox={`${scanViewX0} ${fallbackScanHeaderY0} ${scanViewW} ${fallbackScanHeaderH}`}
            style={fieldSpec ? undefined : transformStyle}
            className="w-full h-auto max-h-24 block transition-transform duration-150 ease-out"
            preserveAspectRatio="none"
          >
            <image
              href={scanUrl}
              x={0}
              y={0}
              width={effectiveScanWidth}
              height={effectiveScanHeight}
              preserveAspectRatio="none"
            />
          </svg>
        </div>
      );
    }

    return null;
  };

  const handleMouseMoveStrip = (e: React.MouseEvent<HTMLDivElement>) => {
    if (zoomScale <= 1) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const rx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ry = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setPanRatio((prev) =>
      Math.abs(prev.x - rx) < 0.005 && Math.abs(prev.y - ry) < 0.005
        ? prev
        : { x: rx, y: ry }
    );
  };

  return (
    <div
      data-testid="hover-zoom-row-crop"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onMouseMove={handleMouseMoveStrip}
      className={`bg-stone-900 rounded-xl p-2.5 border transition-all duration-200 ${
        showContext || fieldSpec
          ? "border-amber-500 shadow-md ring-1 ring-amber-500/30"
          : "border-stone-300"
      } ${className}`}
    >
      {/* Top Status & Field Crop Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2 px-1 text-xs">
        <div className="flex flex-wrap items-center gap-2 text-stone-200 min-w-0">
          <Eye className="w-4 h-4 text-amber-400 shrink-0" />
          {rowLabel ? (
            <span className="font-semibold text-stone-100 text-xs sm:text-sm truncate">
              {rowLabel}
            </span>
          ) : (
            <span className="font-semibold text-stone-200 text-xs sm:text-sm">
              {lang === "sk" ? "Výrez vybraného riadku" : "Selected row crop"}
            </span>
          )}
          {fieldSpec ? (
            <span
              data-testid="active-field-crop-badge"
              className="px-2.5 py-0.5 rounded-full bg-amber-500 text-white font-bold text-xs inline-flex items-center gap-1.5"
            >
              <span>{lang === "sk" ? fieldSpec.labelSk : fieldSpec.labelEn}</span>
              {activeFieldFocus?.valuePreview && (
                <span className="px-1.5 py-0 rounded bg-stone-950/60 text-amber-200 font-mono text-[11px]">
                  {activeFieldFocus.valuePreview}
                </span>
              )}
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded-full bg-stone-800 text-stone-300 text-xs">
              {lang === "sk"
                ? "Najeďte myšou na bunku v tabuľke pre detailný výrez daného údaja"
                : "Hover any cell in the table below to view its targeted column crop"}
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {activeFieldFocus?.key && (
            <div className="inline-flex rounded-lg bg-stone-800 p-0.5 border border-stone-700 text-xs">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setForceFullRow(false);
                }}
                className={`px-2.5 py-1 rounded-md font-semibold cursor-pointer transition ${
                  !forceFullRow
                    ? "bg-amber-500 text-white"
                    : "text-stone-300 hover:text-white"
                }`}
              >
                {lang === "sk" ? "Výrez poľa" : "Field crop"}
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setForceFullRow(true);
                }}
                className={`px-2.5 py-1 rounded-md font-semibold cursor-pointer transition ${
                  forceFullRow
                    ? "bg-amber-500 text-white"
                    : "text-stone-300 hover:text-white"
                }`}
              >
                {lang === "sk" ? "Celý riadok" : "Full row"}
              </button>
            </div>
          )}

          {hoverZoomEnabled && (
            <>
              {showContext && hasValidRowBand && !fieldSpec && (
                <div className="flex items-center gap-1 text-xs">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setContextPaddingMult(2.2);
                    }}
                    className={`px-2 py-1 min-h-[28px] rounded-md cursor-pointer ${
                      contextPaddingMult === 2.2
                        ? "bg-amber-500 text-white font-bold"
                        : "bg-stone-800 text-stone-200 hover:bg-stone-700"
                    }`}
                  >
                    {lang === "sk" ? "±2 riadky" : "±2 rows"}
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setContextPaddingMult(3.5);
                    }}
                    className={`px-2 py-1 min-h-[28px] rounded-md cursor-pointer ${
                      contextPaddingMult === 3.5
                        ? "bg-amber-500 text-white font-bold"
                        : "bg-stone-800 text-stone-200 hover:bg-stone-700"
                    }`}
                  >
                    {lang === "sk" ? "±3.5 riadku" : "±3.5 rows"}
                  </button>
                </div>
              )}

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsPinned((prev) => !prev);
                }}
                title={
                  isPinned
                    ? lang === "sk"
                      ? "Zrušiť pripnutie kontextu"
                      : "Unpin context"
                    : lang === "sk"
                    ? "Pripnúť kontext okolitých riadkov"
                    : "Pin neighboring rows context"
                }
                className={`px-2.5 py-1 min-h-[28px] rounded-md text-xs font-semibold inline-flex items-center gap-1.5 cursor-pointer ${
                  isPinned
                    ? "bg-amber-500 text-white"
                    : "bg-stone-800 text-stone-200 hover:bg-stone-700"
                }`}
              >
                <Pin className="w-3.5 h-3.5" />
                <span>
                  {isPinned
                    ? lang === "sk"
                      ? "Pripnuté"
                      : "Pinned"
                    : lang === "sk"
                    ? "Kontext riadkov"
                    : "Row context"}
                </span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Optional Quick Field Selector Pills */}
      {showFieldPills && onSelectFieldFocus && (
        <div className="flex flex-wrap items-center gap-1 mb-2 px-1">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSelectFieldFocus(null);
              setForceFullRow(true);
            }}
            className={`px-2 py-0.5 rounded text-[11px] font-semibold cursor-pointer transition ${
              !fieldSpec
                ? "bg-amber-500 text-white"
                : "bg-stone-800 text-stone-300 hover:bg-stone-700"
            }`}
          >
            {lang === "sk" ? "Celý riadok (1–24)" : "Full row (1–24)"}
          </button>
          {HOUSEHOLD_FIELD_PILLS.map((fKey) => {
            const spec = CENSUS_FIELD_CROP_RANGES[fKey];
            const isAct = fieldSpec?.key === fKey;
            return (
              <button
                key={fKey}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setForceFullRow(false);
                  onSelectFieldFocus({ key: fKey });
                }}
                className={`px-2 py-0.5 rounded text-[11px] font-semibold cursor-pointer transition ${
                  isAct
                    ? "bg-amber-500 text-white"
                    : "bg-stone-800 text-stone-300 hover:bg-stone-700"
                }`}
              >
                {lang === "sk" ? spec.shortSk : spec.shortEn}
              </button>
            );
          })}
        </div>
      )}

      {/* Main Crop / Contextual View Area with Pinned Aligned Column Header */}
      {!fieldSpec && showContext && hasValidRowBand && scanUrl ? (
        <div className="space-y-1.5">
          {/* Pinned Column Header + Zoomed-out contextual window from full scan */}
          <div className="relative rounded-lg overflow-hidden border border-amber-500/60 bg-stone-950">
            {renderPinnedColumnHeader(false)}
            <svg
              viewBox={`0 ${winY0} ${effectiveScanWidth} ${winH}`}
              className="w-full h-auto max-h-[440px] block"
              preserveAspectRatio="none"
            >
              <image
                href={scanUrl}
                x={0}
                y={0}
                width={effectiveScanWidth}
                height={effectiveScanHeight}
                preserveAspectRatio="none"
              />
              {y0 > winY0 && (
                <rect
                  x={0}
                  y={winY0}
                  width={effectiveScanWidth}
                  height={y0 - winY0}
                  fill="rgba(12, 10, 9, 0.22)"
                />
              )}
              {winY1 > y1 && (
                <rect
                  x={0}
                  y={y1}
                  width={effectiveScanWidth}
                  height={winY1 - y1}
                  fill="rgba(12, 10, 9, 0.22)"
                />
              )}
              <rect
                x={4}
                y={y0}
                width={Math.max(10, effectiveScanWidth - 8)}
                height={rowH}
                fill="rgba(79, 70, 229, 0.14)"
                stroke="#4f46e5"
                strokeWidth={Math.max(4, Math.round(effectiveScanHeight * 0.002))}
                rx={6}
              />
            </svg>
          </div>
          {/* Keep the tight stitched strip underneath for direct comparison */}
          {!cropImgFailed && (
            <div className="rounded bg-stone-950/90 p-1 border border-stone-800">
              <img
                src={cropUrl}
                alt={rowLabel || (lang === "sk" ? "Výrez riadku" : "Row crop")}
                loading="eager"
                decoding="async"
                onError={() => setCropImgFailed(true)}
                className="w-full h-auto block rounded opacity-95"
              />
            </div>
          )}
        </div>
      ) : (
        <div
          className={`relative overflow-hidden rounded-lg bg-stone-950 border border-stone-800 ${
            fieldSpec ? "max-w-3xl mx-auto shadow-inner" : ""
          }`}
        >
          {renderPinnedColumnHeader(true)}
          <div className="relative overflow-hidden bg-stone-950">
            {!cropImgFailed ? (
              <img
                src={cropUrl}
                alt={rowLabel || (lang === "sk" ? "Výrez riadku" : "Row crop")}
                loading="eager"
                decoding="async"
                style={sharedHorizTransformStyle}
                onError={() => setCropImgFailed(true)}
                className="w-full h-auto block transition-all duration-150 ease-out"
              />
            ) : (
              <div className="py-4 px-3 text-center text-xs text-stone-400 flex items-center justify-center gap-1.5">
                <ImageOff className="w-4 h-4 text-stone-500" />
                <span>
                  {lang === "sk"
                    ? "Výrez riadku nie je v lokálnom úložisku k dispozícii."
                    : "Row crop image is not available in local storage."}
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default ZoomableScanViewer;
