"""
Slovakiana Census Portal (https://slovakiana.sk/scitacie-harky) REST & IIIF Client.
Communicates with https://wcm.slovakiana.sk and https://content.slovakiana.sk to:
1. Fetch facet hierarchies (Years 1930/1939/1940, 81 Districts, 3,712 Municipalities).
2. Paginate through all 1.25M census sheets via /censussheet/search.
3. Fetch individual sheet metadata (/culturalobject/{key}) and scan manifests (/digitalobject/{doid}).
4. Download high-resolution front (_v) and back (_r) scan images via IIIF JPEG or JP2 master files.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any
import cv2
import httpx
import numpy as np

from backend.app.config import settings
from backend.app.pipeline.lexicon import (
    canonical_municipality_display,
    canonical_municipality_key,
    fold_text_diacritics,
    parse_house_number_int,
)

WCM_BASE_URL = "https://wcm.slovakiana.sk"
CONTENT_BASE_URL = "https://content.slovakiana.sk"
SLOVAKIANA_KEY_RE = re.compile(r"^cair-[a-zA-Z0-9_-]{3,32}$")


def validate_slovakiana_key(key: str | None) -> str:
    cleaned = str(key or "").strip()
    if not SLOVAKIANA_KEY_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid slovakiana_key: {key!r}")
    return cleaned

YEAR_TO_TAG_KEY: dict[int, str] = {
    1930: "10001",
    1939: "10002",
    1940: "10003",
}
TAG_KEY_TO_YEAR: dict[str, int] = {v: k for k, v in YEAR_TO_TAG_KEY.items()}


def normalize_category_keys_for_slovakiana(category_keys: list[str] | None) -> list[str]:
    """
    Slovakiana filter quirk: District keys are 3 digits ("701".."783"), while
    municipality keys are 4-5 digits ("7236"). When a municipality key is present,
    Slovakiana strips the 3-digit district key so the query filters by the specific
    municipality rather than OR-ing the entire district.
    """
    if not category_keys:
        return []
    cleaned = [str(k).strip() for k in category_keys if str(k).strip()]
    has_municipality_key = any(len(k) >= 4 for k in cleaned)
    if has_municipality_key:
        return [k for k in cleaned if len(k) >= 4]
    return cleaned


def parse_street_and_house_from_raw(
    street_entry: str | None,
    title_raw: str | None = None,
) -> tuple[str | None, str | None, int | None]:
    """
    Parses Slovakiana censusSheetStreet strings like:
      - "Kutlíkova, 17/278" -> street="Kutlíkova", house_number="17/278", house_number_int=278
      - "Škovránková, -/121" -> street="Škovránková", house_number="121", house_number_int=121
      - "-, -/3" -> street=None, house_number="3", house_number_int=3
    """
    street: str | None = None
    house_no: str | None = None
    if street_entry and street_entry.strip():
        raw = street_entry.strip()
        if "," in raw:
            st_part, hn_part = raw.split(",", 1)
            st_clean = st_part.strip(" -")
            street = st_clean if st_clean else None
            hn_clean = hn_part.strip()
            if hn_clean.startswith("-/"):
                hn_clean = hn_clean[2:].strip()
            elif hn_clean.endswith("/-"):
                hn_clean = hn_clean[:-2].strip()
            hn_clean = hn_clean.strip(" -/")
            house_no = hn_clean if hn_clean else None
        else:
            clean = raw.strip(" -")
            if re.match(r"^[\d/\-a-zA-Z]+$", clean) and any(c.isdigit() for c in clean):
                house_no = clean.lstrip("-/")
            elif clean:
                street = clean

    if not house_no and title_raw:
        # Fallback from title: "Sčítací hárok 57/16, Škovránková 121, Bratislava - mesto (Bratislava)"
        parts = [p.strip() for p in title_raw.split(",")]
        if len(parts) >= 3:
            mid = parts[1]
            m = re.match(r"^(.+?)\s+(\d+[\d/\-a-zA-Z]*)$", mid)
            if m:
                if not street and m.group(1).strip(" -"):
                    street = m.group(1).strip(" -")
                house_no = m.group(2).strip(" -/")

    house_int = parse_house_number_int(house_no)
    return street, house_no, house_int


def parse_box_and_sheet_from_title(
    title_raw: str | None,
    house_number: str | None = None,
    census_year: int | None = None,
) -> tuple[str | None, str | None]:
    """Extracts box_number and sheet_number from 'Sčítací hárok 57/16, ...'."""
    if not title_raw:
        return None, None
    m = re.search(r"h[aá]rok\s+(\d+)\s*/\s*(\d+[a-zA-Z]?)", title_raw, re.IGNORECASE)
    if m:
        box = m.group(1).zfill(4)
        raw_sheet = m.group(2)
        hn_clean = (house_number or "").strip()
        # Slovakiana 1940 search title quirk: "Sčítací hárok 462/32092" concatenates
        # sheet_number ("320") + house_number ("92") without a separator.
        if (
            census_year == 1940
            and hn_clean
            and raw_sheet.isdigit()
            and hn_clean.isdigit()
            and len(raw_sheet) > len(hn_clean)
            and raw_sheet.endswith(hn_clean)
        ):
            stripped = raw_sheet[: -len(hn_clean)]
            if stripped:
                raw_sheet = stripped
        sheet = raw_sheet.zfill(4) if raw_sheet.isdigit() else raw_sheet
        return box, sheet
    return None, None


def parse_search_hit(item: dict[str, Any]) -> dict[str, Any]:
    """Normalizes a single item from /censussheet/search into CensusSheet fields."""
    key = str(item.get("key") or "").strip()
    title_raw = str(item.get("name") or "").strip() or None
    municipality = str(item.get("municipality") or "").strip()

    year_val = 1930
    origin_date = str(item.get("displayOriginDate") or "").strip()
    if origin_date.isdigit():
        year_val = int(origin_date)
    else:
        for tag in item.get("tags") or []:
            tname = str(tag.get("name") or "")
            tkey = str(tag.get("tagKey") or "")
            if tname.isdigit():
                year_val = int(tname)
                break
            if tkey in TAG_KEY_TO_YEAR:
                year_val = TAG_KEY_TO_YEAR[tkey]
                break

    district: str | None = None
    district_key: str | None = None
    municipality_key: str | None = None
    for cat in item.get("categories") or []:
        ckey = str(cat.get("categoryKey") or "").strip()
        cname = str(cat.get("name") or "").strip()
        level = cat.get("level")
        if level == 1 or len(ckey) == 3:
            district = cname or district
            district_key = ckey or district_key
        elif level == 2 or len(ckey) >= 4:
            if not municipality and cname:
                municipality = cname
            municipality_key = ckey or municipality_key

    if not district and title_raw:
        dm = re.search(r"\(([^()]+)\)\s*$", title_raw)
        if dm:
            district = dm.group(1).strip()

    if not municipality and title_raw:
        no_paren = re.sub(r"\s*\([^()]+\)\s*$", "", title_raw).strip()
        if "," in no_paren:
            cand_muni = no_paren.rsplit(",", 1)[-1].strip()
            if cand_muni and not cand_muni.lower().startswith("sčítací"):
                municipality = cand_muni
        else:
            m_concat = re.match(r"^Sčítací\s+h[aá]rok\s+\d+/\d+[A-Za-z]?\s*(.+)$", no_paren, re.IGNORECASE)
            if m_concat and m_concat.group(1).strip(" ,-"):
                municipality = m_concat.group(1).strip(" ,-")

    street_list = item.get("censusSheetStreet") or []
    street_entry = street_list[0] if isinstance(street_list, list) and street_list else None
    street_raw, house_number, house_number_int = parse_street_and_house_from_raw(
        street_entry, title_raw
    )
    box_number, sheet_number = parse_box_and_sheet_from_title(
        title_raw, house_number=house_number, census_year=year_val
    )
    canonical_muni = canonical_municipality_display(municipality) or municipality

    return {
        "slovakiana_key": key,
        "title_raw": title_raw,
        "census_year": year_val,
        "district": district,
        "district_key": district_key,
        "district_folded": fold_text_diacritics(district) if district else None,
        "municipality": canonical_muni,
        "municipality_key": municipality_key,
        "municipality_folded": canonical_municipality_key(canonical_muni),
        "street_raw": street_raw,
        "house_number": house_number,
        "house_number_int": house_number_int,
        "box_number": box_number,
        "sheet_number": sheet_number,
        "is_blackened": bool(item.get("blackened", False)),
        "has_images": bool(item.get("hasImages", True)),
        "thumbnail_id": str(item.get("thumbnail")) if item.get("thumbnail") else None,
    }


def iiif_info_to_jpeg_url(iiif_info_url: str | None, max_width: int = 5000) -> str | None:
    """Converts an IIIF info.json URL on content.slovakiana.sk into a direct full-page JPEG URL."""
    if not iiif_info_url:
        return None
    base = re.sub(r"/info\.json$", "", iiif_info_url.strip())
    if max_width >= 5000:
        return f"{base}/full/full/0/default.jpg"
    return f"{base}/full/{max_width},/0/default.jpg"


def parse_detail_and_digital_object(
    co_json: dict[str, Any],
    do_json: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parses /culturalobject/{key} and /digitalobject/{doid} responses into full metadata + image URLs."""
    base = parse_search_hit(co_json)
    attrs: dict[str, list[str]] = {}
    for attr in co_json.get("attributes") or []:
        aname = str(attr.get("name") or "").strip()
        avals = [str(v) for v in (attr.get("values") or []) if v is not None]
        if aname:
            attrs[aname] = avals

    if "Date" in attrs and attrs["Date"]:
        raw_dt = str(attrs["Date"][0]).strip()
        if raw_dt.isdigit():
            base["census_year"] = int(raw_dt)
    if "Municipality" in attrs and attrs["Municipality"] and not base.get("municipality"):
        raw_muni = str(attrs["Municipality"][0]).strip()
        if raw_muni:
            canon_m = canonical_municipality_display(raw_muni) or raw_muni
            base["municipality"] = canon_m
            base["municipality_folded"] = canonical_municipality_key(canon_m)
    if "Box number" in attrs and attrs["Box number"]:
        raw_box = str(attrs["Box number"][0]).strip()
        base["box_number"] = raw_box.zfill(4) if raw_box.isdigit() else raw_box
    if "Number of census sheet" in attrs and attrs["Number of census sheet"]:
        raw_sn = str(attrs["Number of census sheet"][0]).strip()
        base["sheet_number"] = raw_sn.zfill(4) if raw_sn.isdigit() else raw_sn
    if "Identifier" in attrs and attrs["Identifier"]:
        base["urn_nbn"] = attrs["Identifier"][0]
    if "Street" in attrs and attrs["Street"]:
        st, hn, hn_int = parse_street_and_house_from_raw(attrs["Street"][0], base.get("title_raw"))
        if st:
            base["street_raw"] = st
        if hn:
            base["house_number"] = hn
            base["house_number_int"] = hn_int

    doids = co_json.get("digitalObjects") or []
    if doids and isinstance(doids[0], dict):
        base["digital_object_id"] = str(doids[0].get("id") or "") or None

    do_list: list[dict[str, Any]] = []
    if isinstance(do_json, dict):
        do_list = [do_json]
    elif isinstance(do_json, list):
        do_list = [d for d in do_json if isinstance(d, dict)]

    images_out: list[dict[str, Any]] = []
    used_sides: set[str] = set()
    for do_obj in do_list:
        if not base.get("digital_object_id") and do_obj.get("_id"):
            base["digital_object_id"] = str(do_obj["_id"])
        for content_block in do_obj.get("content") or []:
            for img in content_block.get("images") or []:
                idx = len(images_out)
                full_info = img.get("full") or {}
                thumb_info = img.get("thumbnail") or content_block.get("thumbnail") or {}
                file_url = str(full_info.get("fileUrl") or "")
                iiif_url = str(full_info.get("iiifUrl") or "")
                # Identify front (_v), back (_r), and any additional pages (page_3, page_4, ...)
                if idx == 0:
                    page_side = "front"
                    if ("_r." in file_url or "_r/" in iiif_url) and "_v" not in file_url:
                        page_side = "back"
                elif idx == 1:
                    page_side = "back"
                    if ("_v." in file_url or "_v/" in iiif_url) and "front" not in used_sides:
                        page_side = "front"
                else:
                    page_side = f"page_{idx + 1}"

                if page_side in used_sides:
                    if "front" not in used_sides:
                        page_side = "front"
                    elif "back" not in used_sides:
                        page_side = "back"
                    else:
                        page_side = f"page_{idx + 1}"
                used_sides.add(page_side)

                images_out.append(
                    {
                        "index": idx,
                        "page_number": idx + 1,
                        "page_side": page_side,
                        "file_id": img.get("fileId") or full_info.get("fileId"),
                        "width": int(img.get("width") or 0),
                        "height": int(img.get("height") or 0),
                        "jp2_url": file_url or None,
                        "iiif_info_url": iiif_url or None,
                        "jpeg_url": iiif_info_to_jpeg_url(iiif_url),
                        "thumbnail_url": thumb_info.get("fileUrl"),
                    }
                )

    base["attributes"] = attrs
    base["images"] = images_out
    return base


def _extract_facet_items(raw_obj: Any, *, sort_by_name: bool = True) -> list[dict[str, Any]]:
    """
    Normalizes Slovakiana /censussheet/filter/* responses (`{"results": [{"key": "768", "name": "Topoľčany", "count": 26898}, ...]}`)
    into a uniform list of `{"key": ..., "tagKey": ..., "name": ..., "count": ...}` dicts.
    """
    if isinstance(raw_obj, dict):
        items = raw_obj.get("results") or raw_obj.get("items") or raw_obj.get("content") or []
    elif isinstance(raw_obj, list):
        items = raw_obj
    else:
        items = []

    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key_str = str(it.get("key") or it.get("tagKey") or it.get("categoryKey") or "").strip()
        name_str = str(it.get("name") or "").strip()
        if not key_str or not name_str:
            continue
        out.append(
            {
                "key": key_str,
                "tagKey": key_str,
                "name": name_str,
                "count": int(it.get("count") or 0),
            }
        )
    if sort_by_name:
        out.sort(key=lambda x: fold_text_diacritics(x["name"]))
    return out


def _decode_jp2_and_save_jpeg(content: bytes, dest_jpeg_path: Path) -> bool:
    arr = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is not None:
        cv2.imwrite(
            str(dest_jpeg_path),
            decoded,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
        return True
    return False


class SlovakianaClient:
    """Async client for Slovakiana census endpoints with rate limiting and retries."""

    def __init__(self, max_concurrency: int = 4, timeout: float = 45.0) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = timeout
        self._headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "MatrikaDigitizer/1.0 (Genealogical Research & Archival Indexer)",
        }
        self._district_cities_cache: dict[tuple[str, str, bool], list[dict[str, Any]]] = {}
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
                timeout=httpx.Timeout(90.0, connect=10.0),
                headers=self._headers,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
            )
            self._http_client_loop = loop
        return self._http_client

    async def _get_json(self, url: str, params: list[tuple[str, Any]] | dict[str, Any] | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(3):
            async with self._semaphore:
                try:
                    client = self._get_http_client()
                    resp = await client.get(url, params=params, timeout=self._timeout)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as exc:
                    last_err = exc
            await asyncio.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Slovakiana request failed for {url}: {last_err}")

    async def fetch_facets(self, only_unblackened: bool = False, force_refresh: bool = False) -> dict[str, Any]:
        """Fetches years, 81 districts, and 3,712 municipalities from Slovakiana (cached locally)."""
        fname = f"slovakiana_facets_{'unblack' if only_unblackened else 'all'}.json"
        cache_dir = Path(settings.census_data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / fname
        legacy_cache_file = settings.data_dir / fname
        if not force_refresh:
            for candidate in (cache_file, legacy_cache_file):
                if candidate.exists():
                    try:
                        cached = json.loads(candidate.read_text(encoding="utf-8"))
                        if cached.get("districts") and cached.get("municipalities"):
                            if candidate != cache_file:
                                try:
                                    cache_file.write_text(
                                        json.dumps(cached, ensure_ascii=False, indent=2),
                                        encoding="utf-8",
                                    )
                                except OSError:
                                    pass
                            return cached
                    except Exception:
                        pass

        flag_str = "true" if only_unblackened else "false"
        years_raw, districts_raw, cities_raw = await asyncio.gather(
            self._get_json(f"{WCM_BASE_URL}/censussheet/filter/years", {"locale": "skSK", "onlyUnBlackened": flag_str}),
            self._get_json(f"{WCM_BASE_URL}/censussheet/filter/districts", {"locale": "skSK", "onlyUnBlackened": flag_str}),
            self._get_json(f"{WCM_BASE_URL}/censussheet/filter/cities", {"locale": "skSK", "onlyUnBlackened": flag_str}),
        )
        payload = {
            "years": _extract_facet_items(years_raw, sort_by_name=False),
            "districts": _extract_facet_items(districts_raw, sort_by_name=True),
            "municipalities": _extract_facet_items(cities_raw, sort_by_name=True),
        }
        try:
            cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return payload

    async def fetch_cities_for_district(
        self,
        district_tag: str,
        year_tag: str | None = None,
        only_unblackened: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetches the exact municipalities belonging to a Slovakiana 3-digit district key."""
        cache_key = (district_tag, year_tag or "", only_unblackened)
        if cache_key in self._district_cities_cache:
            return self._district_cities_cache[cache_key]

        params: list[tuple[str, Any]] = [
            ("locale", "skSK"),
            ("onlyUnBlackened", "true" if only_unblackened else "false"),
            ("categoryKeys[0]", district_tag),
        ]
        if year_tag:
            params.append(("tagKeys[0]", year_tag))

        cities_raw = await self._get_json(f"{WCM_BASE_URL}/censussheet/filter/cities", params=params)
        cities = _extract_facet_items(cities_raw, sort_by_name=True)
        self._district_cities_cache[cache_key] = cities
        return cities

    async def search_sheets(
        self,
        *,
        page: int = 0,
        size: int = 200,
        only_unblackened: bool = False,
        years: list[int] | None = None,
        category_keys: list[str] | None = None,
        search_text: str = "",
        search_by_keywords: bool = True,
        sort_attribute: str = "created",
        sort_direction: int = 1,
    ) -> dict[str, Any]:
        """Queries /censussheet/search with bracket-indexed array parameters."""
        params: list[tuple[str, Any]] = [
            ("page", page),
            ("size", size),
            ("onlyUnBlackened", "true" if only_unblackened else "false"),
            ("searchText", search_text),
            ("searchByKeywords", "true" if search_by_keywords else "false"),
            ("sortAttribute", sort_attribute),
            ("sortDirection", sort_direction),
        ]
        if years:
            tag_keys = [YEAR_TO_TAG_KEY[y] for y in years if y in YEAR_TO_TAG_KEY]
            for idx, tk in enumerate(tag_keys):
                params.append((f"tagKeys[{idx}]", tk))

        norm_cats = normalize_category_keys_for_slovakiana(category_keys)
        for idx, ck in enumerate(norm_cats):
            params.append((f"categoryKeys[{idx}]", ck))

        raw = await self._get_json(f"{WCM_BASE_URL}/censussheet/search", params=params)
        items_raw = (
            raw.get("content")
            or raw.get("items")
            or raw.get("results")
            or (raw if isinstance(raw, list) else [])
        )
        total = raw.get("totalCount") or raw.get("totalElements") or raw.get("total") or len(items_raw)
        parsed_items = [parse_search_hit(it) for it in items_raw if isinstance(it, dict) and it.get("key")]
        return {
            "total": int(total),
            "page": page,
            "size": size,
            "items": parsed_items,
            "raw_count": len(items_raw),
        }

    async def fetch_sheet_detail(self, slovakiana_key: str) -> dict[str, Any]:
        """Fetches both /culturalobject/{key} and all /digitalobject/{doid} manifests for a census sheet."""
        safe_key = validate_slovakiana_key(slovakiana_key)
        co_json = await self._get_json(f"{WCM_BASE_URL}/culturalobject/{safe_key}")
        do_jsons: list[dict[str, Any]] = []
        doids = co_json.get("digitalObjects") or []
        for do_entry in doids:
            if isinstance(do_entry, dict) and do_entry.get("id"):
                doid = str(do_entry["id"])
                try:
                    fetched_do = await self._get_json(f"{WCM_BASE_URL}/digitalobject/{doid}")
                    if isinstance(fetched_do, dict):
                        do_jsons.append(fetched_do)
                except Exception:
                    pass
        do_arg: dict[str, Any] | list[dict[str, Any]] | None = (
            do_jsons[0] if len(do_jsons) == 1 else (do_jsons if do_jsons else None)
        )
        return parse_detail_and_digital_object(co_json, do_arg)

    async def download_image_to_jpeg(
        self,
        image_desc: dict[str, Any],
        dest_jpeg_path: Path,
    ) -> Path:
        """
        Downloads a single census scan page (front _v or back _r) to dest_jpeg_path.
        Tries the IIIF JPEG endpoint first (5000px wide standard JPEG), falling back
        to the master .jp2 fileUrl with OpenCV/Pillow conversion to .jpg.
        """
        dest_jpeg_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_jpeg_path.exists() and dest_jpeg_path.stat().st_size > 10_000:
            return dest_jpeg_path

        jpeg_url = image_desc.get("jpeg_url")
        jp2_url = image_desc.get("jp2_url")
        thumb_url = image_desc.get("thumbnail_url")

        async with self._semaphore:
            client = self._get_http_client()
            if jpeg_url:
                try:
                    resp = await client.get(jpeg_url, timeout=90.0)
                    if resp.status_code == 200 and len(resp.content) > 10_000:
                        await asyncio.to_thread(dest_jpeg_path.write_bytes, resp.content)
                        return dest_jpeg_path
                except Exception:
                    pass

            if jp2_url:
                try:
                    resp = await client.get(jp2_url, timeout=90.0)
                    if resp.status_code == 200 and len(resp.content) > 10_000:
                        ok = await asyncio.to_thread(
                            _decode_jp2_and_save_jpeg, resp.content, dest_jpeg_path
                        )
                        if ok:
                            return dest_jpeg_path
                except Exception:
                    pass

            if thumb_url:
                resp = await client.get(thumb_url, timeout=30.0)
                if resp.status_code == 200 and len(resp.content) > 1_000:
                    await asyncio.to_thread(dest_jpeg_path.write_bytes, resp.content)
                    return dest_jpeg_path

        raise RuntimeError(f"Could not download census scan image from {jpeg_url or jp2_url}")

    async def download_thumbnail(
        self,
        slovakiana_key: str,
        thumbnail_id: str | None = None,
        fallback_url: str | None = None,
    ) -> Path | None:
        """
        Downloads the 640x640 JPEG thumbnail for a census sheet into
        `data/uploads/slovakiana/thumbnails/{slovakiana_key}.jpg`.
        """
        if not slovakiana_key:
            return None
        try:
            safe_key = validate_slovakiana_key(slovakiana_key)
        except ValueError:
            return None
        thumb_dir = Path(settings.census_data_dir) / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        dest_path = thumb_dir / f"{safe_key}.jpg"
        if dest_path.exists() and dest_path.stat().st_size > 1_000:
            return dest_path

        urls: list[str] = []
        if thumbnail_id:
            urls.append(f"{WCM_BASE_URL}/image/download/{thumbnail_id}?sizeIndex=0")
        if fallback_url:
            urls.append(fallback_url)
        if not urls:
            return None

        async with self._semaphore:
            client = self._get_http_client()
            for u in urls:
                try:
                    resp = await client.get(u, timeout=20.0)
                    if resp.status_code == 200 and len(resp.content) > 1_000:
                        await asyncio.to_thread(dest_path.write_bytes, resp.content)
                        return dest_path
                except Exception:
                    continue
        return None

    async def download_thumbnails_batch(self, items: list[dict[str, Any]]) -> None:
        """Concurrently downloads local thumbnails for a batch of search hits or sheets."""
        tasks = [
            self.download_thumbnail(
                str(it.get("slovakiana_key") or ""),
                str(it["thumbnail_id"]) if it.get("thumbnail_id") else None,
            )
            for it in items
            if it.get("slovakiana_key") and it.get("thumbnail_id")
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


slovakiana_client = SlovakianaClient()
