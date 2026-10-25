# Matrika & Slovakiana Census Digitizer

Open-source web application for digitizing historical Central European genealogical records—specifically **Church Registers (*Matriky*: baptisms/births, marriages, deaths)** in Slovak, Hungarian, Latin, and German, alongside **1930 / 1939 / 1940 Slovak Census Sheets (*Sčítacie hárky*)** from **Slovakiana / Slovenský národný archív (SNA)**.

It combines OpenCV + Rust (`matrika_db_rs`) page layout analysis, a stage-batched Vision-Language Model (VLM) & LLM extraction pipeline, a human-in-the-loop Question Center for uncertain handwriting, full-text search (SQLite FTS5 + Rayon parallel paleographic/phonetic scoring) with historical Slovak/Hungarian/Latin/German name and village normalization, native Rust GEDCOM 5.5.1 / CSV export, and read-only **FamilySearch MCP** tree comparison.

---

## Key Capabilities

- **Church Register (*Matrika*) Digitization (`/knihy`, `/nahrat`)**
  - Upload single or two-page church register scans (baptism, marriage, death).
  - OpenCV + Rust table and row segmentation (parallel sheared Radon skew detection) with interactive bounding-box overlays and row-level crops.
  - Stage-batched AI pipeline (OCR → Decision → Structured Extraction) producing normalized `PersonFact` records with parents, spouses, godparents/witnesses, vital dates, and historical places.
  - Inline record editing, manual row creation, and cascade deletion directly inside the Scan Inspector.
- **1930 / 1940 Slovakiana Census Sheet (*Sčítacie hárky*) Digitization (`/scitacie-harky`)**
  - Browse and search digitized census sheets by single census year (`1930` default or `1940`), district (*okres*), municipality (*obec*), street/settlement (*ulica / osada*), and house number via the Slovakiana cultural portal facets, with seamless cross-year switching (`1930` ↔ `1940`) for the same house or village.
  - Supports **Household Census Sheets** (front & back `_r` scans, multi-dwelling/`byt` households, surname consensus, literacy, occupation, religion, nationality, birth/residency rights), **District Overview Sheets (*Prehľad sčítacieho obvodu / Prehľad domov*)**, and **Municipal Overview Sheets (*Obecný prehľad*)** joining district summaries with village-wide house directories.
- **Native Rust Backend Engine (`backend/rust/matrika_db_rs`)**
  - Compiled PyO3 + `rusqlite` + `rayon` extension (`matrika_db_rs`) with a persistent SQLite connection pool (`db_pool.rs`, 64 MB page cache, 256 MB `mmap`, prepared statement cache) and composite indexes.
  - Multi-threaded Rust query, filtering, and serialization engines (`census_queries.rs`, `api_queries.rs`, `feedback.rs`) for Census Hub, Municipality Summary, Scan Inspector, Workflow Traces, and OCR Feedback statistics.
  - Parallel Rust paleographic/phonetic search, 100 km regional gazetteer & surname stemmer (`lexicon.rs`, `gazetteer.rs`, `reconcile.rs`), bulk FTS5 indexer, local Matrika ↔ Census cross-matcher, and native GEDCOM 5.5.1 / CSV exporter (`export.rs`).
  - Parallel Rust image & OCR helpers (`image_layout.rs`) for HSV/RGB auditor crayon suppression, 141-angle sheared Radon projection, base64 encoding, and zero-DOM HTML table parsing.
- **Human-in-the-Loop Question Center (`/otazky`)**
  - Low-confidence transcriptions and ambiguous handwriting are queued as targeted verification cards with high-contrast row crops and clickable links back to the source scan or census sheet.
  - Answering a question automatically updates the extracted record and re-indexes full-text search.
- **Unified Cross-Record Search & Exports (`/hladanie`)**
  - Diacritic-insensitive and phonetic search across Matrika entries and Census households, backed by historical Slovak ↔ Hungarian ↔ German village gazetteers and given-name synonym tables.
  - Export filtered or complete datasets to **GEDCOM 5.5.1** (with deduplicated household family units and formal `SOUR` citations) and **CSV**.
- **FamilySearch MCP Integration (`/familysearch`)**
  - Connects to the bundled `familysearch-mcp` server over stdio to resolve historical jurisdictions, search indexed records, and compare local Matrika/Census facts against FamilySearch Shared Tree profiles (`what_agrees`, `what_differs`, `what_is_missing`, and step-by-step manual citation action packets).
- **Bilingual Interface (`SK` / `EN`), OCR Feedback & Pipeline Inspector (`/debug`)**
  - Accessible high-contrast interface with instant Slovak (`SK`) and English (`EN`) switching, row-level OCR correctness feedback (`Správne` / `Nesprávne` with automatic `data/ocr_feedback_debug_report.md` and `.json` generation), and a developer workflow inspector (`/debug`) for configuring per-stage AI providers (local LM Studio vs. remote OpenAI-compatible endpoints), model combination accuracy/timing benchmarks, feature flags, and prompt traces.

---

## Archival Data Licensing & Attribution (`CC BY-NC-ND 4.0` — Slovakiana / Slovenský národný archív)

This project separates **software source code licensing** from **third-party archival content licensing**:

1. **Software Source Code (`MIT License`)**: All application code in this repository is licensed under the [MIT License](LICENSE) (including the vendored [`vendor/familysearch-mcp`](vendor/familysearch-mcp/LICENSE) package).
2. **Slovakiana Census Sheets (`CC BY-NC-ND 4.0`)**: Historical 1930, 1939, and 1940 census sheets (*Sčítacie hárky*) fetched at runtime from [**Slovakiana (`https://slovakiana.sk/scitacie-harky`)**](https://slovakiana.sk/scitacie-harky) originate from the **Slovenský národný archív (SNA)**, are published by **Národné osvetové centrum (NOC) / Slovakiana**, and are licensed under [**Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (`CC BY-NC-ND 4.0`)**](https://creativecommons.org/licenses/by-nc-nd/4.0/).

When using this application with Slovakiana census sheets, you must comply with the three core conditions of **CC BY-NC-ND 4.0**:

- **`BY` — Attribution**: Always credit **Slovenský národný archív (SNA)** as the custodial archive and **Slovakiana (Národné osvetové centrum)** as the digital source, alongside the canonical object URL (`https://slovakiana.sk/kulturne-objekty/{cair_id}`), persistent identifier (`urn:nbn` when assigned), and license link (`https://creativecommons.org/licenses/by-nc-nd/4.0/`). The UI, GEDCOM `SOUR` records, CSV exports, and FamilySearch action packets include this attribution automatically.
- **`NC` — Non-Commercial Use Only**: Downloading, indexing, and analyzing Slovakiana census scans is permitted strictly for **non-commercial** genealogical, historical, and academic research.
- **`ND` — No Redistribution of Derivative Crops or Overlays**:
  - **Zero archival scans in this repository**: No Slovakiana images, thumbnails, facet caches, or derivative row crops are bundled in this Git repository (`.gitignore` excludes `data/` and `samples/`).
  - **Local transient processing only**: OpenCV row slicing, contrast adjustments, and grid overlays are generated strictly locally on your machine at runtime to assist automated OCR and human reading. Do **not** redistribute cropped or modified derivative images—always cite and link to the unmodified original scan on Slovakiana (`https://slovakiana.sk/kulturne-objekty/{cair_id}`).

---

## Quickstart with Docker Compose

```bash
cp .env.example .env
# Optional: edit .env to configure AI_BASE_URL, FREELLM_API_KEY, or APP_API_KEY
docker compose up --build
```

The application listens on [http://localhost:8080](http://localhost:8080).

- Persistent SQLite database (`matrika.db`), uploaded Matrika scans (`uploads/`), downloaded Slovakiana census scans (`uploads/slovakiana/`), row crops (`crops/`), and `familysearch.env` are stored in `./data` (mounted at `/app/data` in the container).
- By default, the container connects to a local OpenAI-compatible server (such as LM Studio, Ollama, or vLLM) on the host machine via `http://host.docker.internal:1234/v1`. You can also configure a remote OpenAI-compatible endpoint via `FREELLM_BASE_URL` and `FREELLM_API_KEY` in `.env` or live inside `/debug`.

---

## Local Development (without Docker)

### Prerequisites

- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/)
- **Rust toolchain (`cargo` / `rustc`)** (for compiling the native `matrika_db_rs` PyO3 extension)
- **Node.js 22+** and `npm`

### 1. Backend & Rust Extension Setup

```bash
cp .env.example .env
uv venv .venv
uv pip install --python .venv -e ./vendor/familysearch-mcp ./backend/rust/matrika_db_rs -e "./backend[dev]"
PYTHONPATH=. .venv/bin/uvicorn backend.app.main:app --reload --port 8080
```

When iterating on the Rust crate under `backend/rust/matrika_db_rs`, rebuild it in release mode with `maturin`:

```bash
cd backend/rust/matrika_db_rs
PYO3_PYTHON="../../.venv/bin/python" ../../.venv/bin/maturin develop --release
```

### 2. Frontend Setup

```bash
cd frontend
npm ci
npm run dev
```

The Vite development server proxies `/api`, `/media`, and `/ws` requests to `http://localhost:8080`.

---

## Environment Variables

See [`.env.example`](.env.example) for a complete template.

| Variable | Default (Local / Docker) | Purpose |
| --- | --- | --- |
| `PANGOLIN_PUBLIC_URL` | `http://localhost:8080` | Public base URL used for CORS, WebSockets, and deep links |
| `APP_PORT` | `8080` | HTTP port for the FastAPI server |
| `APP_API_KEY` | *(empty)* | Optional bearer / `X-API-Key` token required for mutating (`POST`/`PUT`/`PATCH`/`DELETE`) and `/api/debug/*` routes when set |
| `AI_PROVIDER` | `local` | Default AI provider (`local` or `freellm`) |
| `AI_BASE_URL` | `http://localhost:1234/v1` (`http://host.docker.internal:1234/v1` in Docker) | Local OpenAI-compatible API base URL (e.g. LM Studio) |
| `AI_MODELS_URL` | `http://localhost:1234/api/v0/models` | Local model server health/discovery endpoint |
| `MODEL_OCR` | `glm-ocr-g32-mixed_4_8-mlx` | Local VLM model identifier for the OCR stage |
| `MODEL_DECISION` | `d1-3b-mlx` | Local LLM model identifier for the Decision stage |
| `MODEL_EXTRACT` | `underdog-saluki-27b-1.0` | Local LLM model identifier for the Extraction stage |
| `MODEL_LAYOUT` | `glm-ocr-g32-mixed_4_8-mlx` | Local VLM model identifier for layout analysis |
| `FREELLM_BASE_URL` | `https://api.openai.com/v1` | Remote OpenAI-compatible API base URL for `freellm` stages |
| `FREELLM_API_KEY` | *(empty)* | Bearer API key for `FREELLM_BASE_URL` (never exposed in API responses) |
| `FREELLM_MODEL` | `gpt-4o-mini` | Default model name for stages routed to `freellm` |
| `FS_ENV_FILE` | `./data/familysearch.env` (`/app/data/familysearch.env` in Docker) | Path to the FamilySearch MCP environment file (`FS_ACCESS_TOKEN` is hot-reloaded) |
| `FS_ENVIRONMENT` | `production` | FamilySearch API environment (`production`, `beta`, or `integration`) |
| `FS_TIMEOUT` | `60` | FamilySearch HTTP request timeout in seconds |
| `DATA_DIR` | `./data` (`/app/data` in Docker) | Root directory for SQLite DB, uploads, crops, and runtime config |
| `UPLOAD_DIR` | `<DATA_DIR>/uploads` | Base directory for uploaded scans |
| `MATRIKA_DATA_DIR` | `<DATA_DIR>/uploads/matrika` | Directory for organized Matrika scan batches (`<YYYY-MM-DD>_batch_<id>_<slug>/<seq>_<filename>` + `batch_manifest.json`) |
| `CENSUS_DATA_DIR` | `<DATA_DIR>/uploads/slovakiana` | Directory for downloaded Slovakiana census scans, thumbnails, and facet caches |
| `CROPS_DIR` | `<DATA_DIR>/crops` | Directory for generated OpenCV row and cell crops |
| `DB_PATH` | `<DATA_DIR>/matrika.db` | Path to the SQLite database file |
| `SAMPLE_DIR` | `./samples` (`/app/samples` in Docker) | Optional directory for local sample Matrika scans |
| `FRONTEND_DIST_DIR` | `./frontend/dist` (`/app/frontend/dist` in Docker) | Path to built frontend static assets served by FastAPI |
| `HOST_DATA_DIR` | `./data` | Host path mounted to `/app/data` in `docker-compose.yml` |
| `HOST_MATRIKA_DATA_DIR` | `./data/uploads/matrika` | Host/NFS path mounted for organized Matrika scan batches in `docker-compose.yml` |
| `HOST_CENSUS_DATA_DIR` | `./data/uploads/slovakiana` | Host/NFS path mounted for Slovakiana census scans in `docker-compose.yml` |

---

## Testing & Continuous Integration

Run the backend test suite (exercises both the compiled `matrika_db_rs` Rust extension and Python pipeline with isolated SQLite databases and synthetic fixtures):

```bash
PYTHONPATH=. .venv/bin/pytest backend/tests
```

Verify the Rust extension directly with Cargo:

```bash
cd backend/rust/matrika_db_rs
PYO3_PYTHON="../../.venv/bin/python" cargo check
```

Run the frontend TypeScript check and production build:

```bash
cd frontend
npm run build
```

GitHub Actions CI (`.github/workflows/ci.yml`) automatically runs both the Python 3.12 backend test suite and the Node 22 frontend build on every push and pull request.
