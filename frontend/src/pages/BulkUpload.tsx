import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, CheckCircle2, FolderUp, Sparkles, Upload } from "lucide-react";

interface Props {
  lang: "sk" | "en";
  refreshStatus: () => void;
}

export default function BulkUpload({ lang, refreshStatus }: Props) {
  const navigate = useNavigate();
  const [batchName, setBatchName] = useState("Matrika – Nová dávka");
  const [parishOrBook, setParishOrBook] = useState("");
  const [recordTypeHint, setRecordTypeHint] = useState<
    "auto" | "baptism" | "marriage" | "death"
  >("auto");
  const [autoProcess, setAutoProcess] = useState(true);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadedCount, setUploadedCount] = useState(0);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
      setErrorMsg(null);
    }
  };

  const handleStartChunkedUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setUploadedCount(0);
    setErrorMsg(null);

    try {
      const batchRes = await fetch("/api/uploads/batches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: batchName || "Dávka matrík",
          parish_or_book: parishOrBook.trim() || null,
          record_type_hint: recordTypeHint,
        }),
      });
      if (!batchRes.ok) {
        const errData = await batchRes.json().catch(() => ({}));
        setErrorMsg(
          errData.detail ||
            (lang === "sk"
              ? "Nepodarilo sa vytvoriť dávku matrík."
              : "Failed to create upload batch.")
        );
        return;
      }
      const batchData = await batchRes.json();
      const batchId = batchData.batch_id;

      const chunkSize = 5;
      for (let i = 0; i < files.length; i += chunkSize) {
        const slice = files.slice(i, i + chunkSize);
        const form = new FormData();
        form.append("batch_id", String(batchId));
        for (const f of slice) {
          form.append("files", f);
        }
        const chunkRes = await fetch("/api/uploads/chunk", {
          method: "POST",
          body: form,
        });
        if (!chunkRes.ok) {
          const errData = await chunkRes.json().catch(() => ({}));
          setErrorMsg(
            errData.detail ||
              (lang === "sk"
                ? "Chyba pri nahrávaní súborov."
                : "Error uploading scan chunk.")
          );
          return;
        }
        setUploadedCount(Math.min(files.length, i + slice.length));
      }

      if (autoProcess && batchId) {
        await fetch(`/api/scans/batches/${batchId}/process`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ background: true }),
        }).catch(() => {});
      }

      refreshStatus();
      navigate("/knihy");
    } catch {
      setErrorMsg(
        lang === "sk"
          ? "Chyba spojenia so serverom pri nahrávaní."
          : "Network error while uploading files."
      );
    } finally {
      setUploading(false);
    }
  };

  const handleLoadSamples = async () => {
    setLoadingSamples(true);
    setErrorMsg(null);
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
          setErrorMsg(
            lang === "sk"
              ? "Priečinok s ukážkami (samples/) je na tomto serveri prázdny. Vyberte a nahrajte vlastné fotky matrík vyššie."
              : "The sample scans directory (samples/) is empty on this server. Please select and upload your own scan files above."
          );
        }
      } else {
        const errData = await r.json().catch(() => ({}));
        setErrorMsg(
          errData.detail ||
            (lang === "sk"
              ? "Nepodarilo sa načítať ukážkové matriky zo servera."
              : "Failed to load sample scans from the server.")
        );
      }
    } catch {
      setErrorMsg(
        lang === "sk"
          ? "Chyba spojenia so serverom pri načítavaní ukážok."
          : "Network error while loading sample scans."
      );
    } finally {
      setLoadingSamples(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="bg-white rounded-xl p-5 border border-stone-300 space-y-4">
        <h2 className="text-lg font-bold text-stone-900 flex items-center gap-2">
          <Upload className="w-5 h-5 text-blue-700" />
          <span>
            {lang === "sk" ? "Nahrať fotky matrík" : "Upload matrika scans"}
          </span>
        </h2>

        <p className="text-sm text-stone-700">
          {lang === "sk"
            ? "Vyberte fotky (.jpg / .png / .webp). Posielajú sa po 5, takže aj veľká dávka prejde spoľahlivo."
            : "Select scan images (.jpg / .png / .webp). Files upload in chunks of 5 to avoid proxy timeouts."}
        </p>

        <div className="space-y-3 max-w-2xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-semibold text-stone-900 mb-1">
                {lang === "sk" ? "Názov dávky" : "Batch name"}
              </label>
              <input
                type="text"
                value={batchName}
                onChange={(e) => setBatchName(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-stone-300 text-sm bg-white"
              />
            </div>

            <div>
              <label className="block text-sm font-semibold text-stone-900 mb-1">
                {lang === "sk"
                  ? "Farnosť / Kniha (voliteľné)"
                  : "Parish / Book (optional)"}
              </label>
              <input
                type="text"
                value={parishOrBook}
                onChange={(e) => setParishOrBook(e.target.value)}
                placeholder={
                  lang === "sk"
                    ? "napr. Rím. kat. farnosť Trnava"
                    : "e.g. Roman Catholic Parish Trnava"
                }
                className="w-full px-3 py-2 rounded-lg border border-stone-300 text-sm bg-white"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-center">
            <div>
              <label className="block text-sm font-semibold text-stone-900 mb-1">
                {lang === "sk" ? "Typ matričnej knihy" : "Register record type"}
              </label>
              <select
                value={recordTypeHint}
                onChange={(e) =>
                  setRecordTypeHint(
                    e.target.value as "auto" | "baptism" | "marriage" | "death"
                  )
                }
                className="w-full px-3 py-2 rounded-lg border border-stone-300 text-sm bg-white"
              >
                <option value="auto">
                  {lang === "sk"
                    ? "Automatická detekcia (Auto)"
                    : "Automatic detection (Auto)"}
                </option>
                <option value="baptism">
                  {lang === "sk"
                    ? "Matrika pokrstených (Krsty)"
                    : "Baptismal register (Baptisms)"}
                </option>
                <option value="marriage">
                  {lang === "sk"
                    ? "Matrika sobášených (Sobáše)"
                    : "Marriage register (Marriages)"}
                </option>
                <option value="death">
                  {lang === "sk"
                    ? "Matrika zomrelých (Úmrtia)"
                    : "Burial register (Deaths)"}
                </option>
              </select>
            </div>

            <div className="pt-1 sm:pt-5">
              <label className="inline-flex items-center gap-2.5 py-1.5 text-sm font-medium text-stone-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoProcess}
                  onChange={(e) => setAutoProcess(e.target.checked)}
                  className="w-5 h-5 rounded border-stone-300 text-amber-600 focus:ring-amber-500 cursor-pointer"
                />
                <span>
                  {lang === "sk"
                    ? "Automaticky spustiť AI OCR po nahraní"
                    : "Automatically start AI OCR after upload"}
                </span>
              </label>
            </div>
          </div>

          <label className="block border-2 border-dashed border-blue-400 bg-blue-50/60 hover:bg-blue-100/60 rounded-xl p-6 text-center cursor-pointer transition">
            <FolderUp className="w-8 h-8 text-blue-700 mx-auto mb-2" />
            <div className="text-base font-bold text-stone-900">
              {lang === "sk"
                ? "Vyberte fotky z počítača"
                : "Choose photos from your computer"}
            </div>
            <div className="text-sm text-stone-600 mt-0.5">
              {lang === "sk"
                ? "Jednu fotku alebo aj stovky naraz"
                : "One photo or hundreds at once"}
            </div>
            <input
              type="file"
              multiple
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {files.length > 0 && (
            <div className="bg-emerald-50 border border-emerald-500 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-700" />
                <span className="text-sm font-semibold text-stone-900">
                  {files.length}{" "}
                  {lang === "sk" ? "fotiek pripravených" : "files ready"}
                </span>
              </div>
              <button
                onClick={handleStartChunkedUpload}
                disabled={uploading}
                className="px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white font-semibold text-sm cursor-pointer disabled:opacity-50"
              >
                {uploading
                  ? `${uploadedCount} / ${files.length}`
                  : lang === "sk"
                  ? "Spustiť nahrávanie"
                  : "Start upload"}
              </button>
            </div>
          )}

          {errorMsg && (
            <div className="bg-red-50 border border-red-400 rounded-xl p-3 flex items-center gap-2 text-red-900 font-medium text-sm">
              <AlertCircle className="w-4 h-4 text-red-700 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>
      </div>

      <div className="bg-amber-50 rounded-xl p-4 border border-amber-400 flex flex-wrap items-center justify-between gap-3">
        <div className="max-w-xl">
          <h3 className="text-base font-bold text-stone-900">
            {lang === "sk"
              ? "Ukážkové matriky na serveri"
              : "Sample scans on the server"}
          </h3>
          <p className="text-sm text-stone-700">
            {lang === "sk"
              ? "Ak sú v priečinku samples/ umiestnené ukážkové dvojstrany matrík, môžete ich načítať jedným kliknutím."
              : "If sample register spreads are present in the samples/ directory, load them with one click."}
          </p>
        </div>
        <button
          onClick={handleLoadSamples}
          disabled={loadingSamples}
          className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-semibold text-sm flex items-center gap-2 cursor-pointer disabled:opacity-50"
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
      </div>
    </div>
  );
}
