import json
from pathlib import Path
from typing import Any
from pydantic import AliasChoices, Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_REPO_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = BASE_REPO_DIR / "data"

FREELLM_DEFAULT_BASE_URL = "https://api.openai.com/v1"
FREELLM_DEFAULT_API_KEY = ""

STAGE_KEYS: tuple[str, ...] = ("ocr", "decision", "extract", "layout")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_REPO_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pangolin_public_url: str = "http://localhost:8080"
    app_port: int = 8080
    app_api_key: str = ""

    ai_base_url: str = "http://localhost:1234/v1"
    ai_models_url: str = "http://localhost:1234/api/v0/models"
    model_ocr: str = "glm-ocr-g32-mixed_4_8-mlx"
    model_decision: str = "d1-3b-mlx"
    model_extract: str = "underdog-saluki-27b-1.0"
    model_layout: str = "glm-ocr-g32-mixed_4_8-mlx"

    local_model_ocr: str = "glm-ocr-g32-mixed_4_8-mlx"
    local_model_decision: str = "d1-3b-mlx"
    local_model_extract: str = "underdog-saluki-27b-1.0"
    local_model_layout: str = "glm-ocr-g32-mixed_4_8-mlx"

    freellm_base_url: str = FREELLM_DEFAULT_BASE_URL
    freellm_api_key: str = FREELLM_DEFAULT_API_KEY
    ai_provider: str = "local"
    freellm_model: str = "gpt-4o-mini"
    local_max_concurrent: int = Field(
        default=5,
        validation_alias=AliasChoices(
            "LOCAL_MAX_CONCURRENT",
            "LOCAL_AI_MAX_CONCURRENT",
            "local_max_concurrent",
            "local_ai_max_concurrent",
        ),
    )
    ai_stages: dict[str, dict[str, str]] = Field(default_factory=dict)

    data_dir: Path = _DEFAULT_DATA_DIR
    upload_dir: Path | None = None
    matrika_data_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MATRIKA_DATA_DIR",
            "MATRIKA_UPLOAD_DIR",
            "matrika_data_dir",
            "matrika_upload_dir",
        ),
    )
    census_data_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "CENSUS_DATA_DIR",
            "SCITANIE_DATA_DIR",
            "census_data_dir",
            "scitanie_data_dir",
        ),
    )
    crops_dir: Path | None = None
    db_path: Path | None = None
    sample_dir: Path = Field(
        default=BASE_REPO_DIR / "samples",
        validation_alias=AliasChoices("SAMPLE_DIR", "SAMPLES_DIR", "sample_dir", "samples_dir"),
    )
    frontend_dist_dir: Path = Field(
        default=BASE_REPO_DIR / "frontend" / "dist",
        validation_alias=AliasChoices(
            "FRONTEND_DIST_DIR", "FRONTEND_DIST", "frontend_dist_dir", "frontend_dist"
        ),
    )

    fs_env_file: Path | None = None
    fs_environment: str = "production"
    fs_timeout: int = 60

    _feature_flags_cache: dict[str, bool] | None = PrivateAttr(default=None)
    _feature_flags_mtime: tuple[str, int | None] | None = PrivateAttr(default=None)
    _ai_runtime_mtime: tuple[str, int | None] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _derive_data_subpaths(self) -> "Settings":
        self.local_model_ocr = self.model_ocr
        self.local_model_decision = self.model_decision
        self.local_model_extract = self.model_extract
        self.local_model_layout = self.model_layout

        custom_data_dir = self.data_dir.resolve() != _DEFAULT_DATA_DIR.resolve()
        if self.upload_dir is None or (
            custom_data_dir
            and self.upload_dir.resolve() == (_DEFAULT_DATA_DIR / "uploads").resolve()
        ):
            self.upload_dir = self.data_dir / "uploads"
        if self.matrika_data_dir is None or (
            custom_data_dir
            and self.matrika_data_dir.resolve() == (_DEFAULT_DATA_DIR / "uploads" / "matrika").resolve()
        ):
            self.matrika_data_dir = self.upload_dir / "matrika"
        if self.census_data_dir is None or (
            custom_data_dir
            and self.census_data_dir.resolve() == (_DEFAULT_DATA_DIR / "uploads" / "slovakiana").resolve()
        ):
            self.census_data_dir = self.upload_dir / "slovakiana"
        if self.crops_dir is None or (
            custom_data_dir
            and self.crops_dir.resolve() == (_DEFAULT_DATA_DIR / "crops").resolve()
        ):
            self.crops_dir = self.data_dir / "crops"
        if self.db_path is None or (
            custom_data_dir
            and self.db_path.resolve() == (_DEFAULT_DATA_DIR / "matrika.db").resolve()
        ):
            self.db_path = self.data_dir / "matrika.db"
        if self.fs_env_file is None or (
            custom_data_dir
            and self.fs_env_file.resolve() == (_DEFAULT_DATA_DIR / "familysearch.env").resolve()
        ):
            self.fs_env_file = self.data_dir / "familysearch.env"

        self._load_ai_runtime_config()
        return self

    @property
    def ai_runtime_config_path(self) -> Path:
        return self.data_dir / "ai_runtime_config.json"

    @property
    def feature_flags_path(self) -> Path:
        return self.data_dir / "feature_flags.json"

    def get_feature_flags(self) -> dict[str, bool]:
        path = self.feature_flags_path
        try:
            mtime_ns: int | None = path.stat().st_mtime_ns if path.exists() else None
        except OSError:
            mtime_ns = None
        cache_key = (str(path), mtime_ns)
        if self._feature_flags_cache is not None and self._feature_flags_mtime == cache_key:
            return dict(self._feature_flags_cache)

        defaults: dict[str, bool] = {
            "tab_census": True,
            "tab_matrika": True,
            "tab_questions": True,
            "tab_records": True,
            "tab_search": True,
            "tab_familysearch": True,
            "census_multi_byt": True,
            "census_surname_consensus": True,
            "census_overview_sheets": True,
            "skip_row_ocr": True,
            "manual_scan_zoom": True,
            "hover_row_zoom": True,
            "admin_debug_overlay": False,
        }
        if mtime_ns is not None:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        if isinstance(k, str):
                            defaults[k] = bool(v)
            except Exception:
                pass
        self._feature_flags_cache = dict(defaults)
        self._feature_flags_mtime = cache_key
        return defaults

    def set_feature_flags(self, updates: dict[str, Any]) -> dict[str, bool]:
        flags = self.get_feature_flags()
        if isinstance(updates, dict):
            for k, v in updates.items():
                if isinstance(k, str):
                    flags[k] = bool(v)
        path = self.feature_flags_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
        try:
            mtime_ns: int | None = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        self._feature_flags_cache = dict(flags)
        self._feature_flags_mtime = (str(path), mtime_ns)
        return dict(flags)

    def _default_local_model_for_stage(self, stage_key: str) -> str:
        if stage_key == "ocr":
            return self.local_model_ocr
        if stage_key == "decision":
            return self.local_model_decision
        if stage_key == "extract":
            return self.local_model_extract
        if stage_key == "layout":
            return self.local_model_layout
        return self.local_model_extract

    def _default_stages(self, default_provider: str = "local") -> dict[str, dict[str, str]]:
        prov = "freellm" if str(default_provider or "").strip().lower() == "freellm" else "local"
        fl_model = (self.freellm_model or "gpt-4o-mini").strip() or "gpt-4o-mini"
        out: dict[str, dict[str, str]] = {}
        for sk in STAGE_KEYS:
            loc_m = self._default_local_model_for_stage(sk)
            active_m = fl_model if prov == "freellm" else loc_m
            out[sk] = {
                "provider": prov,
                "model": active_m,
                "local_model": loc_m,
                "freellm_model": fl_model,
            }
        return out

    @staticmethod
    def map_stage_key(stage: str | None) -> str:
        norm = str(stage or "").strip().lower()
        if norm in ("ocr", "ocr_header", "ocr_left", "ocr_right", "census_overview_ocr") or norm.startswith("ocr"):
            return "ocr"
        if norm == "decision" or "decision" in norm:
            return "decision"
        if norm in ("extract", "census_extract", "census_overview_extract") or "extract" in norm:
            return "extract"
        if norm in ("layout", "census_layout") or "layout" in norm:
            return "layout"
        if norm == "debug_playground":
            return "debug_playground"
        return "extract"

    def _normalize_stage_entry(
        self,
        stage_key: str,
        raw_entry: dict[str, Any] | None,
        prev_entry: dict[str, str] | None = None,
    ) -> dict[str, str]:
        defaults = self._default_stages("local")[stage_key]
        base = dict(prev_entry or defaults)
        base.setdefault("provider", defaults["provider"])
        base.setdefault("local_model", defaults["local_model"])
        base.setdefault("freellm_model", defaults["freellm_model"])

        if isinstance(raw_entry, dict):
            if raw_entry.get("provider") is not None:
                p = str(raw_entry["provider"]).strip().lower()
                base["provider"] = "freellm" if p == "freellm" else "local"

            in_local = str(raw_entry.get("local_model") or "").strip()
            in_freellm = str(raw_entry.get("freellm_model") or "").strip()
            in_model = str(raw_entry.get("model") or "").strip()

            if in_local:
                base["local_model"] = in_local
            if in_freellm:
                base["freellm_model"] = in_freellm

            if in_model:
                prev_fl = (prev_entry or {}).get("freellm_model")
                prev_loc = (prev_entry or {}).get("local_model")
                prev_mod = (prev_entry or {}).get("model")
                if base["provider"] == "freellm":
                    if not in_freellm or (
                        in_freellm == prev_fl
                        and in_model != prev_mod
                        and in_model != base["local_model"]
                    ):
                        base["freellm_model"] = in_model
                else:
                    if not in_local or (
                        in_local == prev_loc
                        and in_model != prev_mod
                        and in_model != base["freellm_model"]
                    ):
                        base["local_model"] = in_model

        base["provider"] = "freellm" if base["provider"] == "freellm" else "local"
        if not base.get("local_model"):
            base["local_model"] = self._default_local_model_for_stage(stage_key)
        if not base.get("freellm_model"):
            base["freellm_model"] = (self.freellm_model or "gpt-4o-mini").strip() or "gpt-4o-mini"
        base["model"] = (
            base["freellm_model"] if base["provider"] == "freellm" else base["local_model"]
        )
        return base

    def _apply_ai_provider_models(self) -> None:
        if not self.ai_stages:
            self.ai_stages = self._default_stages(self.ai_provider)
        else:
            for sk in STAGE_KEYS:
                self.ai_stages[sk] = self._normalize_stage_entry(sk, {}, self.ai_stages.get(sk))

        self.model_ocr = self.ai_stages["ocr"]["model"]
        self.model_decision = self.ai_stages["decision"]["model"]
        self.model_extract = self.ai_stages["extract"]["model"]
        self.model_layout = self.ai_stages["layout"]["model"]

        all_provs = {self.ai_stages[sk]["provider"] for sk in STAGE_KEYS}
        if all_provs == {"freellm"}:
            self.ai_provider = "freellm"
        elif all_provs == {"local"}:
            self.ai_provider = "local"
        else:
            self.ai_provider = self.ai_stages["extract"]["provider"]

        if self.ai_stages["extract"].get("freellm_model"):
            self.freellm_model = self.ai_stages["extract"]["freellm_model"]

    def _load_ai_runtime_config(self) -> None:
        cfg_path = self.ai_runtime_config_path
        try:
            mtime_ns: int | None = cfg_path.stat().st_mtime_ns if cfg_path.exists() else None
        except OSError:
            mtime_ns = None
        cache_key = (str(cfg_path), mtime_ns)
        if self.ai_stages and self._ai_runtime_mtime == cache_key:
            return

        stages_loaded = False
        legacy_file_loaded = False
        if mtime_ns is not None:
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    prov = str(raw.get("provider") or self.ai_provider).strip().lower()
                    self.ai_provider = "freellm" if prov == "freellm" else "local"
                    model_val = str(raw.get("freellm_model") or "").strip()
                    if model_val:
                        self.freellm_model = model_val

                    raw_conc = raw.get("local_max_concurrent")
                    if raw_conc is not None:
                        try:
                            self.local_max_concurrent = max(1, min(64, int(raw_conc)))
                        except (TypeError, ValueError):
                            pass

                    raw_stages = raw.get("stages")
                    if isinstance(raw_stages, dict) and raw_stages:
                        next_stages = self._default_stages("local")
                        for sk in STAGE_KEYS:
                            if isinstance(raw_stages.get(sk), dict):
                                next_stages[sk] = self._normalize_stage_entry(
                                    sk, raw_stages[sk], next_stages[sk]
                                )
                        self.ai_stages = next_stages
                        stages_loaded = True
                    else:
                        legacy_file_loaded = True
            except Exception:
                pass

        if not stages_loaded and (legacy_file_loaded or not self.ai_stages):
            self.ai_stages = self._default_stages(self.ai_provider)

        self._apply_ai_provider_models()
        self._ai_runtime_mtime = cache_key

    def resolve_stage_ai(
        self,
        stage: str,
        requested_model: str | None = None,
    ) -> dict[str, str]:
        self._load_ai_runtime_config()
        stage_key = self.map_stage_key(stage)
        req_m = str(requested_model or "").strip()

        explicit_provider: str | None = None
        if req_m.lower().startswith("local:"):
            explicit_provider = "local"
            req_m = req_m.split(":", 1)[1].strip()
        elif req_m.lower().startswith("freellm:"):
            explicit_provider = "freellm"
            req_m = req_m.split(":", 1)[1].strip()

        if stage_key == "debug_playground":
            known_local = {
                self.local_model_ocr,
                self.local_model_decision,
                self.local_model_extract,
                self.local_model_layout,
                *(s.get("local_model", "") for s in self.ai_stages.values()),
            } - {""}
            if explicit_provider:
                provider = explicit_provider
                active_model = req_m or (
                    self.freellm_model if provider == "freellm" else self.local_model_extract
                )
            elif req_m:
                low_m = req_m.lower()
                is_local = req_m in known_local or any(
                    tok in low_m for tok in ("mlx", "saluki", "glm-ocr", "d1-3b")
                )
                provider = "local" if is_local else "freellm"
                active_model = req_m
            else:
                ext_cfg = self.ai_stages.get("extract") or self._default_stages()[
                    "extract"
                ]
                provider = ext_cfg["provider"]
                active_model = ext_cfg["model"]
        else:
            stage_cfg = self.ai_stages.get(stage_key) or self._default_stages()[stage_key]
            provider = explicit_provider or stage_cfg["provider"]
            if explicit_provider and req_m:
                active_model = req_m
            else:
                active_model = (
                    stage_cfg["freellm_model"]
                    if provider == "freellm"
                    else stage_cfg["local_model"]
                )
                if not active_model and req_m:
                    active_model = req_m

        if provider == "freellm":
            base_url = (self.freellm_base_url or FREELLM_DEFAULT_BASE_URL).rstrip("/")
            api_key = self.freellm_api_key or FREELLM_DEFAULT_API_KEY
        else:
            provider = "local"
            base_url = self.ai_base_url.rstrip("/")
            api_key = ""

        return {
            "stage_key": stage_key,
            "provider": provider,
            "base_url": base_url,
            "api_key": api_key,
            "model": active_model,
        }

    def get_ai_runtime_config(self) -> dict[str, Any]:
        self._load_ai_runtime_config()
        active_base = (
            self.freellm_base_url.rstrip("/")
            if self.ai_provider == "freellm"
            else self.ai_base_url.rstrip("/")
        )
        stages_copy = {sk: dict(self.ai_stages[sk]) for sk in STAGE_KEYS}
        return {
            "provider": self.ai_provider,
            "freellm_model": self.freellm_model,
            "freellm_base_url": self.freellm_base_url.rstrip("/"),
            "freellm_api_key_configured": bool(self.freellm_api_key),
            "local_base_url": self.ai_base_url.rstrip("/"),
            "base_url": active_base,
            "local_max_concurrent": max(1, int(self.local_max_concurrent or 5)),
            "freellm_concurrent_unlimited": True,
            "stages": stages_copy,
            "configured": {
                "ocr": self.model_ocr,
                "decision": self.model_decision,
                "extract": self.model_extract,
                "layout": self.model_layout,
            },
            "local_defaults": {
                "ocr": self.local_model_ocr,
                "decision": self.local_model_decision,
                "extract": self.local_model_extract,
                "layout": self.local_model_layout,
            },
            "local_models": {
                "ocr": self.local_model_ocr,
                "decision": self.local_model_decision,
                "extract": self.local_model_extract,
                "layout": self.local_model_layout,
            },
        }

    def set_ai_runtime_config(
        self,
        provider: str | None = None,
        freellm_model: str | None = None,
        *,
        stage: str | None = None,
        model: str | None = None,
        local_model: str | None = None,
        stages: dict[str, Any] | None = None,
        preset: str | None = None,
        local_max_concurrent: int | None = None,
    ) -> dict[str, Any]:
        self._load_ai_runtime_config()

        if local_max_concurrent is not None:
            try:
                self.local_max_concurrent = max(1, min(64, int(local_max_concurrent)))
            except (TypeError, ValueError):
                pass

        if freellm_model is not None and str(freellm_model).strip():
            self.freellm_model = str(freellm_model).strip()

        if preset is not None and str(preset).strip():
            p_norm = str(preset).strip().lower()
            chosen_fl = str(freellm_model or model or "").strip()
            if chosen_fl:
                self.freellm_model = chosen_fl

            is_hybrid = (
                ("local" in p_norm and "freellm" in p_norm)
                or p_norm
                in (
                    "hybrid",
                    "mixed",
                    "recommended",
                    "extract_freellm",
                    "local_ocr_freellm_extract",
                    "local_ocr_decision_freellm_extract",
                )
            )
            if is_hybrid:
                for sk in ("ocr", "decision", "layout"):
                    self.ai_stages[sk]["provider"] = "local"
                    if chosen_fl:
                        self.ai_stages[sk]["freellm_model"] = chosen_fl
                    self.ai_stages[sk] = self._normalize_stage_entry(
                        sk, {}, self.ai_stages[sk]
                    )
                self.ai_stages["extract"]["provider"] = "freellm"
                if chosen_fl:
                    self.ai_stages["extract"]["freellm_model"] = chosen_fl
                self.ai_stages["extract"] = self._normalize_stage_entry(
                    "extract", {}, self.ai_stages["extract"]
                )
            elif "freellm" in p_norm:
                for sk in STAGE_KEYS:
                    self.ai_stages[sk]["provider"] = "freellm"
                    if chosen_fl:
                        self.ai_stages[sk]["freellm_model"] = chosen_fl
                    self.ai_stages[sk] = self._normalize_stage_entry(
                        sk, {}, self.ai_stages[sk]
                    )
            else:
                for sk in STAGE_KEYS:
                    self.ai_stages[sk]["provider"] = "local"
                    if chosen_fl:
                        self.ai_stages[sk]["freellm_model"] = chosen_fl
                    self.ai_stages[sk] = self._normalize_stage_entry(
                        sk, {}, self.ai_stages[sk]
                    )

        elif isinstance(stages, dict) and stages and not (stage is not None and str(stage).strip()):
            for raw_k, raw_v in stages.items():
                sk = self.map_stage_key(raw_k)
                if sk in STAGE_KEYS and isinstance(raw_v, dict):
                    self.ai_stages[sk] = self._normalize_stage_entry(
                        sk, raw_v, self.ai_stages.get(sk)
                    )

        elif stage is not None and str(stage).strip():
            if isinstance(stages, dict) and stages:
                for raw_k, raw_v in stages.items():
                    s_key = self.map_stage_key(raw_k)
                    if s_key in STAGE_KEYS and isinstance(raw_v, dict):
                        self.ai_stages[s_key] = self._normalize_stage_entry(
                            s_key, raw_v, self.ai_stages.get(s_key)
                        )
            sk = self.map_stage_key(stage)
            if sk not in STAGE_KEYS:
                sk = "extract"
            upd: dict[str, Any] = {}
            if provider is not None:
                upd["provider"] = provider
            if model is not None:
                upd["model"] = model
            if local_model is not None:
                upd["local_model"] = local_model
            if freellm_model is not None:
                upd["freellm_model"] = freellm_model
            self.ai_stages[sk] = self._normalize_stage_entry(
                sk, upd, self.ai_stages.get(sk)
            )
            if self.ai_stages[sk]["provider"] == "freellm" and self.ai_stages[sk].get(
                "freellm_model"
            ):
                self.freellm_model = self.ai_stages[sk]["freellm_model"]
        elif provider is not None or freellm_model is not None or model is not None:
            norm_provider = (
                "freellm"
                if str(provider or self.ai_provider).strip().lower() == "freellm"
                else "local"
            )
            self.ai_provider = norm_provider
            chosen_fl = str(freellm_model or model or "").strip()
            if chosen_fl:
                self.freellm_model = chosen_fl
            for sk in STAGE_KEYS:
                self.ai_stages[sk]["provider"] = norm_provider
                if chosen_fl:
                    self.ai_stages[sk]["freellm_model"] = chosen_fl
                self.ai_stages[sk] = self._normalize_stage_entry(
                    sk, {}, self.ai_stages[sk]
                )

        self._apply_ai_provider_models()

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = self.ai_runtime_config_path
            cfg_path.write_text(
                json.dumps(
                    {
                        "provider": self.ai_provider,
                        "freellm_model": self.freellm_model,
                        "local_max_concurrent": max(1, int(self.local_max_concurrent or 5)),
                        "stages": self.ai_stages,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._ai_runtime_mtime = (str(cfg_path), cfg_path.stat().st_mtime_ns)
        except OSError:
            self._ai_runtime_mtime = None
        return self.get_ai_runtime_config()

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.matrika_data_dir.mkdir(parents=True, exist_ok=True)
        self.census_data_dir.mkdir(parents=True, exist_ok=True)
        (self.census_data_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fs_env_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.fs_env_file.exists():
            self.fs_env_file.write_text(
                f"FS_ENVIRONMENT={self.fs_environment}\nFS_TIMEOUT={self.fs_timeout}\nFS_ACCESS_TOKEN=\n",
                encoding="utf-8",
            )


settings = Settings()


def get_ai_runtime_config() -> dict[str, Any]:
    return settings.get_ai_runtime_config()


def set_ai_runtime_config(
    provider: str | None = None,
    freellm_model: str | None = None,
    *,
    stage: str | None = None,
    model: str | None = None,
    local_model: str | None = None,
    stages: dict[str, Any] | None = None,
    preset: str | None = None,
    local_max_concurrent: int | None = None,
) -> dict[str, Any]:
    return settings.set_ai_runtime_config(
        provider=provider,
        freellm_model=freellm_model,
        stage=stage,
        model=model,
        local_model=local_model,
        stages=stages,
        preset=preset,
        local_max_concurrent=local_max_concurrent,
    )
