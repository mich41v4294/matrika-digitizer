from __future__ import annotations

import asyncio
import base64
from functools import lru_cache
import io
import json
import re
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from PIL import Image
from pydantic import BaseModel
from sqlmodel import Session

from backend.app.config import settings
from backend.app.db.models import WorkflowTrace
from backend.app.db.session import engine

ModelT = TypeVar("ModelT", bound=BaseModel)


@lru_cache(maxsize=256)
def _encode_image_base64_cached(path_str: str, mtime_ns: int, max_width: int) -> str:
    try:
        import cv2
        from backend.app.pipeline.lexicon import _rust_lex

        img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            if w > max_width:
                ratio = max_width / float(w)
                img_bgr = cv2.resize(
                    img_bgr,
                    (max_width, max(1, int(h * ratio))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                raw_bytes = enc.tobytes()
                if _rust_lex is not None and hasattr(_rust_lex, "base64_encode_bytes_rs"):
                    return str(_rust_lex.base64_encode_bytes_rs(raw_bytes))
                return base64.b64encode(raw_bytes).decode("utf-8")
    except Exception:
        pass
    with Image.open(path_str) as img:
        rgb = img.convert("RGB")
        if rgb.width > max_width:
            ratio = max_width / float(rgb.width)
            rgb = rgb.resize((max_width, int(rgb.height * ratio)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=86)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_base64(image_path: Path, max_width: int = 2000) -> str:
    resolved = Path(image_path).resolve()
    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _encode_image_base64_cached(str(resolved), mtime_ns, max_width)


_JSON_KEY_ALIASES: dict[str, str] = {
    "byt_c": "apartment_number",
    "byt_cislo": "apartment_number",
    "byt_č": "apartment_number",
    "dwelling_unit_number": "apartment_number",
    "flat_number": "apartment_number",
    "poradove_cislo": "person_row_number",
    "poradove_cislo_osoby": "person_row_number",
    "row_number": "person_row_number",
    "priezvisko": "surname_raw",
    "surname": "surname_raw",
    "rodne_priezvisko": "maiden_surname",
    "maiden_name": "maiden_surname",
    "krstne_meno": "given_name_raw",
    "meno": "given_name_raw",
    "given_name": "given_name_raw",
    "pomer_k_hlave": "relation_to_head",
    "pomer_k_hlave_domacnosti": "relation_to_head",
    "relation": "relation_to_head",
    "pohlavie": "sex",
    "gender": "sex",
    "datum_narodenia": "birth_date_raw",
    "birth_date": "birth_date_raw",
    "rok_narodenia": "birth_year",
    "birth_year_est": "birth_year",
    "rodinny_stav": "marital_status",
    "datum_sobasa": "marriage_date_raw",
    "marriage_date": "marriage_date_raw",
    "pocet_deti": "children_born_count",
    "pocet_deti_zivych": "children_born_count",
    "pocet_detí_zivych": "children_born_count",
    "children_born": "children_born_count",
    "zomreli": "children_dead_count",
    "pocet_deti_zomrelych": "children_dead_count",
    "pocet_detí_zomrelych": "children_dead_count",
    "children_dead": "children_dead_count",
    "obec_narodenia": "birth_place_municipality",
    "okres_narodenia": "birth_place_district",
    "krajina_narodenia": "birth_place_country",
    "stat_narodenia": "birth_place_country",
    "pristahoval_sa_kedy": "moved_in_date",
    "prisťahoval_sa_kedy": "moved_in_date",
    "pristahoval_sa_odkial": "previous_residence",
    "prisťahoval_sa_odkiaľ": "previous_residence",
    "domovska_prislusnost": "home_jurisdiction",
    "domovska_prisluhnost": "home_jurisdiction",
    "domovská_príslušnosť": "home_jurisdiction",
    "statna_prislusnost": "citizenship",
    "statna_prisluhnost": "citizenship",
    "štátna_príslušnosť": "citizenship",
    "narodnost": "nationality",
    "národnosť": "nationality",
    "religion_raw": "religion",
    "nabozenstvo": "religion",
    "náboženstvo": "religion",
    "gramotnost": "literacy",
    "znalost_citania_a_pisania": "literacy",
    "occupation_raw": "occupation",
    "povolanie_raw": "occupation",
    "povolanie": "occupation",
    "postavenie_v_povolani": "occupation_position",
    "zamestnavatel": "employer_workplace",
    "zavod": "employer_workplace",
    "presence": "presence_type",
    "pritomnost": "presence_type",
    "prítomnosť": "presence_type",
    "observationes_raw": "notes_raw",
    "poznamka": "notes_raw",
    "poznámka": "notes_raw",
    "notes": "notes_raw",
}


def normalize_json_keys(obj: dict[str, Any], response_model: type[BaseModel] | None = None) -> dict[str, Any]:
    if not isinstance(obj, dict) or not obj:
        return {}
    flattened: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_k not in flattened or flattened[sub_k] is None:
                    flattened[sub_k] = sub_v
        else:
            flattened[k] = v

    allowed_fields = set(response_model.model_fields.keys()) if response_model is not None else None
    normalized: dict[str, Any] = dict(flattened)
    for k, v in list(flattened.items()):
        if v is None:
            continue
        target_k = _JSON_KEY_ALIASES.get(k) or _JSON_KEY_ALIASES.get(k.lower())
        if target_k:
            if allowed_fields is None or target_k in allowed_fields:
                if normalized.get(target_k) is None:
                    normalized[target_k] = v
        # Also support reverse mapping when schema expects observationes_raw instead of notes_raw
        if k == "notes_raw" and (allowed_fields is None or "observationes_raw" in allowed_fields):
            if normalized.get("observationes_raw") is None:
                normalized["observationes_raw"] = v
    return normalized


def extract_json_object(raw_text: str, response_model: type[BaseModel] | None = None) -> dict[str, Any]:
    if not raw_text:
        return {}
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return normalize_json_keys(obj, response_model=response_model)
    except Exception:
        pass
    # Find fenced json block if embedded in prose
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return normalize_json_keys(obj, response_model=response_model)
        except Exception:
            pass
    # Scan for a JSON object starting with `{\s*"` (quoted JSON key) so bare `{` in prose (e.g. "starting { ending }") is ignored
    decoder = json.JSONDecoder()
    brace_positions = [m.start() for m in re.finditer(r'\{\s*"', cleaned)]
    for start_idx in brace_positions:
        try:
            obj, _ = decoder.raw_decode(cleaned, start_idx)
            if isinstance(obj, dict):
                return normalize_json_keys(obj, response_model=response_model)
        except Exception:
            pass
        end_idx = cleaned.rfind("}")
        tries = 0
        while end_idx > start_idx and tries < 4:
            tries += 1
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return normalize_json_keys(obj, response_model=response_model)
            except Exception:
                pass
            end_idx = cleaned.rfind("}", start_idx, end_idx)
    # Handle truncated JSON object ONLY if it actually starts with a JSON key `{\s*"[key]"\s*:`
    trunc_match = re.search(r'\{\s*"[a-zA-Z0-9_Á-ž]+"\s*:', cleaned)
    if trunc_match:
        json_tail = cleaned[trunc_match.start() :]
        partial: dict[str, Any] = {}
        for m in re.finditer(
            r'"([a-zA-Z0-9_Á-ž]+)"\s*:\s*("([^"\\]|\\.)*"|null|true|false|-?\d+(?:\.\d+)?)',
            json_tail,
        ):
            k = m.group(1)
            v_raw = m.group(2)
            try:
                partial[k] = json.loads(v_raw)
            except Exception:
                partial[k] = v_raw.strip('"')
        if partial:
            return normalize_json_keys(partial, response_model=response_model)
    return {}


def split_thinking_and_content(
    raw_content: str,
    api_reasoning: str = "",
    *,
    expect_json: bool = False,
) -> tuple[str, str]:
    """
    Separates <think>...</think> blocks and pre-JSON chain-of-thought prose from `raw_content`
    into `reasoning`, returning `(clean_content, combined_reasoning)`.
    Never treats bare `{` in prose (such as `"starting { ending }"`) as a JSON object.
    """
    reasoning_parts: list[str] = []
    if api_reasoning and api_reasoning.strip():
        reasoning_parts.append(api_reasoning.strip())

    text = raw_content or ""
    # 1. Extract closed <think>...</think> tags
    for m in re.finditer(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE):
        t_chunk = m.group(1).strip()
        if t_chunk and t_chunk not in reasoning_parts:
            reasoning_parts.append(t_chunk)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # 2. Extract unclosed <think> tag (e.g. truncated mid-thought)
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        parts = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)
        text = parts[0].strip()
        if len(parts) > 1 and parts[1].strip():
            reasoning_parts.append(parts[1].strip())

    if expect_json and text:
        # Check for fenced JSON block preceded/followed by prose (requiring `{\s*"` inside fence)
        fence_m = re.search(r'```(?:json)?\s*(\{\s*"[\s\S]*?\})\s*```', text, flags=re.IGNORECASE)
        if fence_m:
            pre = text[: fence_m.start()].strip()
            post = text[fence_m.end() :].strip()
            if pre:
                reasoning_parts.append(pre)
            if post:
                reasoning_parts.append(post)
            text = fence_m.group(1).strip()
        else:
            # Strip outer markdown fence if the whole response is fenced
            stripped_fence = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            stripped_fence = re.sub(r"\s*```$", "", stripped_fence).strip()
            # Find a real JSON object starting with `{\s*"` inside text
            decoder = json.JSONDecoder()
            found_json_str: str | None = None
            found_span: tuple[int, int] | None = None
            for m in re.finditer(r'\{\s*"', stripped_fence):
                s_idx = m.start()
                try:
                    parsed_cand, end_pos = decoder.raw_decode(stripped_fence, s_idx)
                    if isinstance(parsed_cand, dict):
                        found_json_str = stripped_fence[s_idx:end_pos]
                        found_span = (s_idx, end_pos)
                        break
                except Exception:
                    pass
                e_idx = stripped_fence.rfind("}")
                tries_e = 0
                while e_idx > s_idx and tries_e < 4:
                    tries_e += 1
                    cand = stripped_fence[s_idx : e_idx + 1]
                    try:
                        parsed_cand = json.loads(cand)
                        if isinstance(parsed_cand, dict):
                            found_json_str = cand
                            found_span = (s_idx, e_idx + 1)
                            break
                    except Exception:
                        pass
                    e_idx = stripped_fence.rfind("}", s_idx, e_idx)
                if found_json_str is not None:
                    break

            if found_json_str is not None and found_span is not None:
                pre = stripped_fence[: found_span[0]].strip()
                post = stripped_fence[found_span[1] :].strip()
                if pre:
                    reasoning_parts.append(pre)
                if post:
                    reasoning_parts.append(post)
                text = found_json_str.strip()
            else:
                # Check if there is a truncated JSON object starting with `{\s*"[key]"\s*:`
                trunc_m = re.search(r'\{\s*"[a-zA-Z0-9_Á-ž]+"\s*:', stripped_fence)
                if trunc_m:
                    pre = stripped_fence[: trunc_m.start()].strip()
                    if pre:
                        reasoning_parts.append(pre)
                    text = stripped_fence[trunc_m.start() :].strip()
                else:
                    # Entire content is chain-of-thought prose (including phrases like "starting { ending }")
                    reasoning_parts.append(stripped_fence)
                    text = ""

    combined_reasoning = "\n\n".join(p for p in reasoning_parts if p).strip()
    return text, combined_reasoning


def build_schema_keys_prompt_suffix(response_model: type[BaseModel]) -> str:
    keys = [k for k in response_model.model_fields.keys() if k != "type_safety_repairs"]
    keys_csv = ", ".join(f'"{k}"' for k in keys)
    return (
        "\n\nREQUIRED JSON SCHEMA KEYS (return ONLY a single flat raw JSON object using these exact keys; "
        "do NOT invent Slovak or nested key names, and do NOT write thinking/prose outside the JSON): "
        f"[{keys_csv}]"
    )


class LocalAIClient:
    def __init__(self) -> None:
        self._local_base_url = settings.ai_base_url.rstrip("/")
        self._local_models_url = settings.ai_models_url
        self._freellm_base_url = (settings.freellm_base_url or "").rstrip("/")
        self._freellm_models_url = f"{self._freellm_base_url}/models" if self._freellm_base_url else ""
        self.base_url = self._local_base_url
        self.models_url = self._local_models_url
        self._probe_cache_ts: float = 0.0
        self._cached_local_probe: tuple[bool, list[dict[str, Any]], str | None] | None = None
        self._cached_freellm_probe: tuple[bool, list[dict[str, Any]], str | None] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_loop: asyncio.AbstractEventLoop | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if (
            self._http_client is None
            or self._http_client.is_closed
            or self._http_client_loop is not loop
        ):
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
            )
            self._http_client_loop = loop
        return self._http_client

    def _resolve_endpoint(self) -> tuple[str, str, str, dict[str, str], str]:
        cfg = settings.get_ai_runtime_config()
        provider = str(cfg.get("provider") or "local")
        freellm_model = str(cfg.get("freellm_model") or settings.freellm_model).strip()
        if provider == "freellm":
            base_url = (settings.freellm_base_url or "").rstrip("/")
            models_url = f"{base_url}/models" if base_url else ""
            api_key = settings.freellm_api_key or ""
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            return provider, base_url, models_url, headers, freellm_model
        base_url = settings.ai_base_url.rstrip("/")
        models_url = settings.ai_models_url
        return "local", base_url, models_url, {}, freellm_model

    async def _probe_local_models(self) -> tuple[bool, list[dict[str, Any]], str | None]:
        self._local_base_url = settings.ai_base_url.rstrip("/")
        self._local_models_url = settings.ai_models_url
        try:
            client = self._get_http_client()
            resp = await client.get(self._local_models_url, timeout=0.8)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return True, data.get("data", []) or [], None
                if isinstance(data, list):
                    return True, data, None
                return True, [], None
            return False, [], f"HTTP {resp.status_code}"
        except Exception as exc:
            return False, [], str(exc)

    async def _probe_freellm_models(self) -> tuple[bool, list[dict[str, Any]], str | None]:
        base_url = (settings.freellm_base_url or "").rstrip("/")
        if not base_url:
            return False, [], "freellm_base_url not configured"
        self._freellm_base_url = base_url
        self._freellm_models_url = f"{base_url}/models"
        api_key = settings.freellm_api_key or ""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            client = self._get_http_client()
            resp = await client.get(self._freellm_models_url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return True, data.get("data", []) or [], None
                if isinstance(data, list):
                    return True, data, None
                return True, [], None
            return False, [], f"HTTP {resp.status_code}"
        except Exception as exc:
            return False, [], str(exc)

    async def fetch_freellm_models(self) -> list[dict[str, Any]]:
        _, models, _ = await self._probe_freellm_models()
        return models

    async def get_models_status(self, *, force_refresh: bool = False) -> dict[str, Any]:
        cfg = settings.get_ai_runtime_config()
        provider, base_url, models_url, _, freellm_model = self._resolve_endpoint()
        self.base_url = base_url
        self.models_url = models_url

        now = time.monotonic()
        if (
            not force_refresh
            and self._cached_local_probe is not None
            and self._cached_freellm_probe is not None
            and (now - self._probe_cache_ts) < 30.0
        ):
            local_online, local_models, local_err = self._cached_local_probe
            freellm_online, freellm_models, freellm_err = self._cached_freellm_probe
        else:
            (local_online, local_models, local_err), (
                freellm_online,
                freellm_models,
                freellm_err,
            ) = await asyncio.gather(
                self._probe_local_models(),
                self._probe_freellm_models(),
            )
            self._cached_local_probe = (local_online, local_models, local_err)
            self._cached_freellm_probe = (freellm_online, freellm_models, freellm_err)
            self._probe_cache_ts = now

        stages = cfg.get("stages", {})
        configured = {
            "ocr": settings.model_ocr,
            "decision": settings.model_decision,
            "extract": settings.model_extract,
            "layout": settings.model_layout,
        }

        used_providers = {
            str(s.get("provider") or "local")
            for s in stages.values()
            if isinstance(s, dict)
        } or {provider}

        if used_providers == {"local"}:
            online = local_online
            primary_err = local_err
        elif used_providers == {"freellm"}:
            online = freellm_online
            primary_err = freellm_err
        else:
            online = local_online or freellm_online
            primary_err = None if online else (local_err or freellm_err)

        active_models = freellm_models if provider == "freellm" else local_models

        result: dict[str, Any] = {
            "online": online,
            "provider": provider,
            "freellm_model": freellm_model,
            "base_url": base_url,
            "local_base_url": self._local_base_url,
            "freellm_base_url": self._freellm_base_url,
            "local_online": local_online,
            "local_models": local_models,
            "freellm_online": freellm_online,
            "freellm_models": freellm_models,
            "stages": stages,
            "configured": configured,
            "models": active_models,
        }
        if primary_err and not online:
            result["error"] = primary_err
        if local_err:
            result["local_error"] = local_err
        if freellm_err:
            result["freellm_error"] = freellm_err
        return result

    async def chat_completion(
        self,
        *,
        stage: str,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        image_path: Path | None = None,
        image_rel_url: str | None = None,
        scan_id: int | None = None,
        row_id: int | None = None,
        max_tokens: int = 900,
        temperature: float = 0.1,
        timeout_sec: float = 120.0,
        skip_thinking: bool = False,
        response_model: type[ModelT] | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        resolved = settings.resolve_stage_ai(stage, requested_model=model)
        provider = resolved["provider"]
        base_url = resolved["base_url"].rstrip("/")
        api_key = resolved["api_key"]
        active_model = resolved["model"]
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.base_url = base_url

        effective_system_prompt = system_prompt or ""
        if response_model is not None and "REQUIRED JSON SCHEMA KEYS" not in effective_system_prompt:
            effective_system_prompt = (effective_system_prompt + build_schema_keys_prompt_suffix(response_model)).strip()

        messages: list[dict[str, Any]] = []
        if effective_system_prompt:
            messages.append({"role": "system", "content": effective_system_prompt})

        has_image = bool(image_path and image_path.exists())
        b64_img: str | None = None
        if has_image and image_path:
            b64_img = await asyncio.to_thread(encode_image_base64, image_path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        is_saluki = "saluki" in active_model.lower()
        is_glm_ocr = "glm-ocr" in active_model.lower()
        if is_saluki:
            messages.append({"role": "assistant", "content": "<think>\n</think>\n"})
        elif provider == "local" and skip_thinking and not is_glm_ocr and not has_image:
            messages.append({"role": "assistant", "content": "<think>\n</think>\n"})

        effective_max_tokens = max_tokens
        if provider == "freellm" and not is_saluki and not is_glm_ocr:
            effective_max_tokens = max(max_tokens, 2500)
        elif not is_saluki and not is_glm_ocr and not skip_thinking:
            effective_max_tokens = max(max_tokens, 2500)

        payload: dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
        }
        if provider == "freellm" and not is_saluki:
            if skip_thinking or response_model is not None:
                payload["reasoning_effort"] = "low"
            if response_model is not None:
                payload["response_format"] = {"type": "json_object"}

        status = "ok"
        content = ""
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        parsed: dict[str, Any] = {}
        validated: ModelT | None = None
        validation_errors: list[str] = []
        image_rejected_by_endpoint = False

        try:
            client = self._get_http_client()
            resp = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout_sec,
            )
            if resp.status_code >= 400 and (
                "reasoning_effort" in payload
                or "response_format" in payload
                or (messages and messages[-1].get("role") == "assistant")
            ):
                fallback_messages = [m for m in messages if m.get("role") != "assistant"]
                fallback_payload = {
                    k: v for k, v in payload.items() if k not in ("reasoning_effort", "response_format")
                }
                fallback_payload["messages"] = fallback_messages
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    json=fallback_payload,
                    headers=headers,
                    timeout=timeout_sec,
                )
            if resp.status_code >= 400 and has_image:
                # Graceful fallback to text-only prompt if a text-only LLM endpoint rejects image_url
                image_rejected_by_endpoint = True
                text_only_messages = [
                    {"role": "user", "content": prompt}
                    if m.get("role") == "user" and isinstance(m.get("content"), list)
                    else m
                    for m in messages
                    if m.get("role") != "assistant"
                ]
                text_only_payload = {
                    k: v
                    for k, v in payload.items()
                    if k not in ("reasoning_effort", "response_format")
                }
                text_only_payload["messages"] = text_only_messages
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    json=text_only_payload,
                    headers=headers,
                    timeout=timeout_sec,
                )
            resp.raise_for_status()
            data = resp.json()

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw_content = msg.get("content") or ""
            api_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            content, reasoning = split_thinking_and_content(
                raw_content,
                api_reasoning,
                expect_json=(response_model is not None),
            )
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = int(details.get("reasoning_tokens") or 0)

            parsed = extract_json_object(content, response_model=response_model)

            # Retry once with higher max_tokens & explicit JSON-only instruction if response_model is required
            # and the model returned only thinking prose without a JSON object
            if response_model is not None and not parsed:
                retry_prompt_text = (
                    f"{prompt}\n\n"
                    "CRITICAL: Do NOT write any reasoning or explanation. "
                    "Output ONLY the raw JSON object immediately."
                )
                retry_user_content: Any = (
                    [
                        {"type": "text", "text": retry_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ]
                    if (has_image and b64_img and not image_rejected_by_endpoint)
                    else retry_prompt_text
                )
                retry_messages = [
                    {
                        "role": "user",
                        "content": retry_user_content,
                    }
                    if m.get("role") == "user"
                    else m
                    for m in messages
                    if m.get("role") != "assistant"
                ]
                retry_payload = {
                    k: v
                    for k, v in payload.items()
                    if k not in ("reasoning_effort", "response_format")
                }
                retry_payload["messages"] = retry_messages
                retry_payload["max_tokens"] = max(effective_max_tokens, 3500)
                r_resp = await client.post(
                    f"{base_url}/chat/completions",
                    json=retry_payload,
                    headers=headers,
                    timeout=timeout_sec,
                )
                if r_resp.status_code == 200:
                    r_data = r_resp.json()
                    r_choice = (r_data.get("choices") or [{}])[0]
                    r_msg = r_choice.get("message") or {}
                    r_raw_content = r_msg.get("content") or ""
                    r_api_reasoning = r_msg.get("reasoning_content") or r_msg.get("reasoning") or ""
                    r_content, r_reasoning = split_thinking_and_content(
                        r_raw_content,
                        r_api_reasoning,
                        expect_json=True,
                    )
                    r_parsed = extract_json_object(r_content, response_model=response_model)
                    if r_parsed:
                        content = r_content
                        reasoning = "\n\n".join(p for p in (reasoning, r_reasoning) if p).strip()
                        parsed = r_parsed
                        r_usage = r_data.get("usage") or {}
                        prompt_tokens += int(r_usage.get("prompt_tokens") or 0)
                        completion_tokens += int(r_usage.get("completion_tokens") or 0)
                        r_details = r_usage.get("completion_tokens_details") or {}
                        reasoning_tokens += int(r_details.get("reasoning_tokens") or 0)

            if response_model is not None:
                if parsed:
                    try:
                        validated = response_model.model_validate(parsed)
                        parsed = validated.model_dump()
                    except Exception as val_exc:
                        validation_errors.append(str(val_exc))
                        try:
                            validated = response_model()
                        except Exception:
                            pass
                    if not content or not content.strip().startswith("{"):
                        content = json.dumps(
                            {k: v for k, v in parsed.items() if k != "type_safety_repairs"},
                            ensure_ascii=False,
                            indent=2,
                        )
                else:
                    validation_errors.append("Model returned only thinking tokens without JSON output")
                    try:
                        validated = response_model()
                    except Exception:
                        pass
        except Exception as exc:
            status = "error"
            content = f"ERROR: {exc}"
            if response_model is not None:
                try:
                    validated = response_model()
                except Exception:
                    pass

        duration_ms = int((time.perf_counter() - t0) * 1000)

        attached_img_name = (
            image_path.name
            if (image_path and image_path.exists())
            else (Path(image_rel_url).name if image_rel_url else None)
        )
        user_prompt_for_trace = (
            f"[IMAGE ATTACHED: {attached_img_name}]\n{prompt}"
            if attached_img_name
            else prompt
        )

        prompt_text_final = (
            user_prompt_for_trace
            if not effective_system_prompt
            else f"[SYSTEM]\n{effective_system_prompt}\n\n[USER]\n{user_prompt_for_trace}"
        )
        parsed_json_str = json.dumps(parsed, ensure_ascii=False)

        def _persist_trace() -> int | None:
            with Session(engine) as session:
                trace = WorkflowTrace(
                    scan_id=scan_id,
                    row_id=row_id,
                    stage=stage,
                    model_or_tool=f"{provider}:{active_model}",
                    prompt_text=prompt_text_final,
                    image_path=image_rel_url,
                    reasoning_content=reasoning,
                    response_text=content,
                    parsed_json=parsed_json_str,
                    status=status,
                    duration_ms=duration_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
                session.add(trace)
                session.commit()
                session.refresh(trace)
                return trace.id

        trace_id = await asyncio.to_thread(_persist_trace)

        return {
            "trace_id": trace_id,
            "status": status,
            "provider": provider,
            "model": active_model,
            "content": content,
            "reasoning": reasoning,
            "parsed": parsed,
            "validated": validated,
            "validation_errors": validation_errors,
            "image_rejected_by_endpoint": image_rejected_by_endpoint,
            "duration_ms": duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }


ai_client = LocalAIClient()
