from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from sqlmodel import Session, select

from backend.app.config import settings
from backend.app.db.models import RecordRow, ScanImage
from backend.app.pipeline.ai_client import ai_client
from backend.app.pipeline.census_layout import (
    detect_is_municipal_overview_sheet,
    detect_is_overview_sheet,
    extract_municipal_overview_header_metadata,
    extract_overview_header_metadata,
    is_valid_commissioner_or_revisor_name,
)


def html_table_to_pipe_text(raw_html_or_text: str) -> str:
    if not raw_html_or_text:
        return ""
    if "<table" not in raw_html_or_text.lower() and "<td" not in raw_html_or_text.lower():
        return raw_html_or_text.strip()
    soup = BeautifulSoup(raw_html_or_text, "html.parser")
    rows_out: list[str] = []
    for tr in soup.find_all("tr"):
        cells = [
            re.sub(r"\s+", " ", cell.get_text(separator=" ", strip=True))
            for cell in tr.find_all(["td", "th"])
        ]
        non_empty = [c for c in cells if c]
        if non_empty:
            rows_out.append(" | ".join(cells))
    return "\n".join(rows_out) if rows_out else soup.get_text(separator=" | ", strip=True)


def detect_header_metadata(raw_header_ocr: str, filename: str = "") -> tuple[str | None, str]:
    """Returns (detected_year, detected_record_type: baptism | marriage | death | census)."""
    combined = f"{raw_header_ocr or ''} {filename}"
    low = combined.lower()

    rec_type = "baptism"
    if any(
        k in low
        for k in [
            "sčítací",
            "scitaci",
            "popisný hárok",
            "popisny harok",
            "sčítanie ľudu",
            "scitanie ludu",
            "census_",
            "cair-ko",
        ]
    ):
        rec_type = "census"
    elif any(k in low for k in ["defunctorum", "halottak", "mortuorum", "obiit", "sepult", "úmrt"]):
        rec_type = "death"
    elif any(k in low for k in ["copulatorum", "matrimon", "házasul", "esküv", "spons", "sobáš"]):
        rec_type = "marriage"
    elif any(k in low for k in ["baptisat", "keresztel", "nativitat", "infantis", "krst"]):
        rec_type = "baptism"

    years = re.findall(r"\b(1[78]\d{2}|19[0-4]\d)\b", combined)
    year = years[0] if years else None
    return year, rec_type


def media_rel_to_path(rel_url: str | None) -> Path | None:
    if not rel_url:
        return None
    census_root = Path(
        getattr(settings, "census_data_dir", None) or (Path(settings.upload_dir) / "slovakiana")
    ).resolve()
    matrika_root = Path(
        getattr(settings, "matrika_data_dir", None) or (Path(settings.upload_dir) / "matrika")
    ).resolve()
    if rel_url.startswith("/media/crops/"):
        sub = rel_url.removeprefix("/media/crops/")
        root = Path(settings.crops_dir).resolve()
        resolved = (root / sub).resolve()
        return resolved if resolved.is_relative_to(root) else None
    if rel_url.startswith("/media/uploads/slovakiana/"):
        sub = rel_url.removeprefix("/media/uploads/slovakiana/")
        resolved = (census_root / sub).resolve()
        if resolved.is_relative_to(census_root) and resolved.exists():
            return resolved
        fallback_root = (Path(settings.upload_dir) / "slovakiana").resolve()
        fallback_res = (fallback_root / sub).resolve()
        if fallback_res.is_relative_to(fallback_root):
            return fallback_res
        return resolved if resolved.is_relative_to(census_root) else None
    if rel_url.startswith("/media/uploads/matrika/"):
        sub = rel_url.removeprefix("/media/uploads/matrika/")
        resolved = (matrika_root / sub).resolve()
        if resolved.is_relative_to(matrika_root) and resolved.exists():
            return resolved
        fallback_root = (Path(settings.upload_dir) / "matrika").resolve()
        fallback_res = (fallback_root / sub).resolve()
        if fallback_res.is_relative_to(fallback_root):
            return fallback_res
        return resolved if resolved.is_relative_to(matrika_root) else None
    if rel_url.startswith("/media/uploads/"):
        sub = rel_url.removeprefix("/media/uploads/")
        root = Path(settings.upload_dir).resolve()
        resolved = (root / sub).resolve()
        return resolved if resolved.is_relative_to(root) else None
    candidate = Path(rel_url).resolve()
    for allowed_root in (
        Path(settings.data_dir).resolve(),
        Path(settings.sample_dir).resolve(),
        census_root,
        matrika_root,
    ):
        if candidate.is_relative_to(allowed_root):
            return candidate
    return None


def resolve_row_extraction_crop(row: RecordRow) -> tuple[Path | None, str | None]:
    """
    Returns `(image_path, image_rel_url)` for the highest-signal row crop to attach
    to multimodal census extraction.
    When `cell_crops_json` has both `identity_name` (Cols 1-5) and `origin_domicile` (Cols 11-14: Rodisko),
    builds and caches a high-resolution stacked composite (`row_XX_extract_composite.jpg`) so the
    extraction model directly sees both the name/relation crop and the Rodisko (Obec, Okres, Krajina) cell crop.
    """
    identity_crop_rel: str | None = None
    origin_crop_rel: str | None = None
    try:
        parsed_crops = json.loads(row.cell_crops_json or "[]")
        if isinstance(parsed_crops, list):
            for item in parsed_crops:
                if not isinstance(item, dict) or not item.get("crop_url"):
                    continue
                k = item.get("key")
                if k in ("identity_name", "identity_relation") and not identity_crop_rel:
                    identity_crop_rel = str(item["crop_url"])
                elif k == "origin_domicile" and not origin_crop_rel:
                    origin_crop_rel = str(item["crop_url"])
        elif isinstance(parsed_crops, dict):
            identity_crop_rel = parsed_crops.get("identity_name") or parsed_crops.get("identity_relation")
            origin_crop_rel = parsed_crops.get("origin_domicile")
    except Exception:
        identity_crop_rel = None
        origin_crop_rel = None

    id_path = media_rel_to_path(identity_crop_rel) if identity_crop_rel else None
    orig_path = media_rel_to_path(origin_crop_rel) if origin_crop_rel else None

    if id_path and id_path.exists() and orig_path and orig_path.exists():
        comp_name = re.sub(r"_identity_name(\.\w+)$", r"_extract_composite\1", id_path.name)
        if comp_name == id_path.name:
            comp_name = f"row_{int(row.row_index or 0):02d}_extract_composite.jpg"
        comp_path = id_path.parent / comp_name
        comp_rel = (
            re.sub(r"_identity_name(\.\w+)$", r"_extract_composite\1", identity_crop_rel)
            if identity_crop_rel and "_identity_name" in identity_crop_rel
            else f"/media/crops/{comp_path.parent.name}/{comp_name}"
        )
        if not comp_path.exists():
            try:
                from PIL import Image, ImageDraw

                im_id = Image.open(id_path).convert("RGB")
                im_orig = Image.open(orig_path).convert("RGB")
                banner_h = 24
                gap_h = 8
                out_w = max(im_id.width, im_orig.width)
                out_h = banner_h + im_id.height + gap_h + banner_h + im_orig.height
                canvas = Image.new("RGB", (out_w, out_h), (250, 248, 242))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle([(0, 0), (out_w, banner_h)], fill=(38, 45, 58))
                draw.text(
                    (8, 5),
                    "COLS 1-5: Priezvisko, Meno, Pomer k hlave domacnosti",
                    fill=(255, 255, 255),
                )
                canvas.paste(im_id, (0, banner_h))
                y2 = banner_h + im_id.height + gap_h
                draw.rectangle([(0, y2), (out_w, y2 + banner_h)], fill=(38, 45, 58))
                draw.text(
                    (8, y2 + 5),
                    "COLS 11-14: Rodisko a) Obec, b) Okres, c) Krajina | Pristahovanie | Domovska prislusnost",
                    fill=(255, 255, 255),
                )
                canvas.paste(im_orig, (0, y2 + banner_h))
                canvas.save(comp_path, format="JPEG", quality=92)
            except Exception:
                return id_path, identity_crop_rel
        if comp_path.exists():
            return comp_path, comp_rel

    if id_path and id_path.exists():
        return id_path, identity_crop_rel

    if orig_path and orig_path.exists():
        return orig_path, origin_crop_rel

    for fallback_rel in (row.stitched_crop_path, row.left_crop_path):
        if fallback_rel:
            fb_path = media_rel_to_path(fallback_rel)
            if fb_path and fb_path.exists():
                return fb_path, fallback_rel

    return None, None


async def run_ocr_for_scan_header(session: Session, scan: ScanImage) -> None:
    header_path = media_rel_to_path(scan.header_crop_path)
    if not header_path or not header_path.exists():
        return
    sheet = None
    if scan.census_sheet_id:
        from backend.app.db.models import CensusSheet

        sheet = session.get(CensusSheet, scan.census_sheet_id)
    hdr_prompt = (
        "Text Recognition:"
        if (sheet and (sheet.sheet_type in ("overview", "municipal_overview") or not sheet.house_number))
        else "Table Recognition:"
    )
    res = await ai_client.chat_completion(
        stage="ocr_header",
        model=settings.model_ocr,
        prompt=hdr_prompt,
        image_path=header_path,
        image_rel_url=scan.header_crop_path,
        scan_id=scan.id,
        max_tokens=600,
    )
    raw_text = html_table_to_pipe_text(res["content"])
    year, rec_type = detect_header_metadata(raw_text, scan.filename)
    scan.header_raw_ocr = raw_text
    if year:
        scan.detected_year = year
    prev_scan_type = scan.detected_type
    if scan.detected_type != "census":
        scan.detected_type = rec_type
    prev_sheet_type = sheet.sheet_type if sheet else None
    if sheet:
        if detect_is_municipal_overview_sheet(raw_text, f"{scan.filename} {sheet.slovakiana_key}"):
            sheet.sheet_type = "municipal_overview"
            meta_rev = extract_municipal_overview_header_metadata(
                raw_text, municipality=sheet.municipality, district=sheet.district
            )
            if meta_rev.get("revisor_name") and (
                not sheet.commissioner_name
                or not is_valid_commissioner_or_revisor_name(
                    sheet.commissioner_name,
                    municipality=sheet.municipality,
                    district=sheet.district,
                )
            ):
                sheet.commissioner_name = meta_rev["revisor_name"]
            elif not is_valid_commissioner_or_revisor_name(
                sheet.commissioner_name,
                municipality=sheet.municipality,
                district=sheet.district,
            ):
                sheet.commissioner_name = None
            session.add(sheet)
        elif detect_is_overview_sheet(raw_text, f"{scan.filename} {sheet.slovakiana_key}"):
            sheet.sheet_type = "overview"
            meta = extract_overview_header_metadata(
                raw_text, municipality=sheet.municipality, district=sheet.district
            )
            if meta.get("obvod_number") and not sheet.obvod_number:
                sheet.obvod_number = meta["obvod_number"]
            if meta.get("commissioner_name") and (
                not sheet.commissioner_name
                or not is_valid_commissioner_or_revisor_name(
                    sheet.commissioner_name,
                    municipality=sheet.municipality,
                    district=sheet.district,
                )
            ):
                sheet.commissioner_name = meta["commissioner_name"]
            elif not is_valid_commissioner_or_revisor_name(
                sheet.commissioner_name,
                municipality=sheet.municipality,
                district=sheet.district,
            ):
                sheet.commissioner_name = None
            session.add(sheet)
    if (
        scan.detected_type != prev_scan_type
        or (sheet and sheet.sheet_type != prev_sheet_type)
    ):
        from backend.app.pipeline.feedback import reset_ocr_feedback

        reset_ocr_feedback(
            session,
            census_sheet_id=sheet.id if sheet else None,
            scan_id=scan.id,
            commit=False,
        )
    session.add(scan)
    session.commit()


async def run_ocr_for_row(session: Session, scan: ScanImage, row: RecordRow) -> dict[str, Any]:
    left_path = media_rel_to_path(row.left_crop_path)
    right_path = media_rel_to_path(row.right_crop_path)

    left_text = ""
    right_text = ""

    is_census = bool(
        scan.detected_type == "census"
        or getattr(scan, "record_type", None) == "census"
        or row.record_type == "census"
    )

    # Check if this scan belongs to a 1940 partially-redacted (blackened) census sheet
    is_blackened_census = False
    is_overview_sheet = False
    if is_census and scan.census_sheet_id:
        from backend.app.db.models import CensusSheet

        sheet = session.get(CensusSheet, scan.census_sheet_id)
        if sheet:
            if sheet.is_blackened:
                is_blackened_census = True
            if sheet.sheet_type in ("overview", "municipal_overview"):
                is_overview_sheet = True

    right_cell_sections: dict[str, str] = {}
    if is_blackened_census:
        if left_path and left_path.exists():
            res_l = await ai_client.chat_completion(
                stage="ocr_left",
                model=settings.model_ocr,
                prompt="Table Recognition:",
                image_path=left_path,
                image_rel_url=row.left_crop_path,
                scan_id=scan.id,
                row_id=row.id,
                max_tokens=700,
            )
            left_text = html_table_to_pipe_text(res_l["content"])
        right_text = "[ZAČIERNENÉ / REDACTED 1940]"
    elif is_census and not is_overview_sheet:
        # Read semantic cell crops for the right half (Cols 11-14, 15-17, 18-24) at optimal aspect ratio
        crop_url_by_key: dict[str, str] = {}
        try:
            parsed_crops = json.loads(row.cell_crops_json or "[]")
            if isinstance(parsed_crops, list):
                for item in parsed_crops:
                    if isinstance(item, dict) and item.get("key") and item.get("crop_url"):
                        crop_url_by_key[str(item["key"])] = str(item["crop_url"])
            elif isinstance(parsed_crops, dict):
                for k, v in parsed_crops.items():
                    if isinstance(v, str):
                        crop_url_by_key[str(k)] = v
        except Exception:
            crop_url_by_key = {}

        semantic_right_keys = (
            "origin_domicile",
            "nationality_religion_literacy",
            "occupation_presence_notes",
        )
        valid_right_crops: list[tuple[str, Path, str]] = []
        for skey in semantic_right_keys:
            c_rel = crop_url_by_key.get(skey)
            c_path = media_rel_to_path(c_rel) if c_rel else None
            if c_path and c_path.exists() and c_rel:
                valid_right_crops.append((skey, c_path, c_rel))

        tasks: list[Any] = []
        has_left_task = bool(left_path and left_path.exists())
        if has_left_task and left_path:
            tasks.append(
                ai_client.chat_completion(
                    stage="ocr_left",
                    model=settings.model_ocr,
                    prompt="Table Recognition:",
                    image_path=left_path,
                    image_rel_url=row.left_crop_path,
                    scan_id=scan.id,
                    row_id=row.id,
                    max_tokens=700,
                )
            )
        for skey, c_path, c_rel in valid_right_crops:
            tasks.append(
                ai_client.chat_completion(
                    stage="ocr_right",
                    model=settings.model_ocr,
                    prompt="Table Recognition:",
                    image_path=c_path,
                    image_rel_url=c_rel,
                    scan_id=scan.id,
                    row_id=row.id,
                    max_tokens=450,
                )
            )

        results = await asyncio.gather(*tasks) if tasks else []
        res_offset = 0
        if has_left_task and results:
            left_text = html_table_to_pipe_text(results[0]["content"])
            res_offset = 1

        cell_parts: list[str] = []
        for idx_c, (skey, _, _) in enumerate(valid_right_crops):
            res_c = results[res_offset + idx_c]
            c_txt = html_table_to_pipe_text(res_c["content"]).strip()
            if c_txt:
                right_cell_sections[skey] = c_txt
                cell_parts.append(re.sub(r"\s*\n+\s*", " | ", c_txt))

        if cell_parts:
            right_text = " | ".join(cell_parts)
        elif right_path and right_path.exists():
            res_r = await ai_client.chat_completion(
                stage="ocr_right",
                model=settings.model_ocr,
                prompt="Table Recognition:",
                image_path=right_path,
                image_rel_url=row.right_crop_path,
                scan_id=scan.id,
                row_id=row.id,
                max_tokens=700,
            )
            right_text = html_table_to_pipe_text(res_r["content"])
    else:
        has_l = bool(left_path and left_path.exists())
        has_r = bool(right_path and right_path.exists())
        if has_l and has_r and left_path and right_path:
            res_l, res_r = await asyncio.gather(
                ai_client.chat_completion(
                    stage="ocr_left",
                    model=settings.model_ocr,
                    prompt="Table Recognition:",
                    image_path=left_path,
                    image_rel_url=row.left_crop_path,
                    scan_id=scan.id,
                    row_id=row.id,
                    max_tokens=700,
                ),
                ai_client.chat_completion(
                    stage="ocr_right",
                    model=settings.model_ocr,
                    prompt="Table Recognition:",
                    image_path=right_path,
                    image_rel_url=row.right_crop_path,
                    scan_id=scan.id,
                    row_id=row.id,
                    max_tokens=700,
                ),
            )
            left_text = html_table_to_pipe_text(res_l["content"])
            right_text = html_table_to_pipe_text(res_r["content"])
        elif has_l and left_path:
            res_l = await ai_client.chat_completion(
                stage="ocr_left",
                model=settings.model_ocr,
                prompt="Table Recognition:",
                image_path=left_path,
                image_rel_url=row.left_crop_path,
                scan_id=scan.id,
                row_id=row.id,
                max_tokens=700,
            )
            left_text = html_table_to_pipe_text(res_l["content"])
        elif has_r and right_path:
            res_r = await ai_client.chat_completion(
                stage="ocr_right",
                model=settings.model_ocr,
                prompt="Table Recognition:",
                image_path=right_path,
                image_rel_url=row.right_crop_path,
                scan_id=scan.id,
                row_id=row.id,
                max_tokens=700,
            )
            right_text = html_table_to_pipe_text(res_r["content"])

    row.raw_ocr_left = left_text
    row.raw_ocr_right = right_text
    if right_cell_sections:
        sec_lines = [f"[ĽAVÁ STRANA / LEFT]\n{left_text}"]
        if right_cell_sections.get("origin_domicile"):
            sec_lines.append(f"[STĹPCE 11-14 RODISKO A PRÍSLUŠNOSŤ]\n{right_cell_sections['origin_domicile']}")
        if right_cell_sections.get("nationality_religion_literacy"):
            sec_lines.append(f"[STĹPCE 15-17 NÁRODNOSŤ, NÁBOŽENSTVO, GRAMOTNOSŤ]\n{right_cell_sections['nationality_religion_literacy']}")
        if right_cell_sections.get("occupation_presence_notes"):
            sec_lines.append(f"[STĹPCE 18-24 POVOLANIE A POZNÁMKY]\n{right_cell_sections['occupation_presence_notes']}")
        row.raw_ocr_combined = "\n".join(sec_lines).strip()
    else:
        row.raw_ocr_combined = f"[ĽAVÁ STRANA / LEFT]\n{left_text}\n[PRAVÁ STRANA / RIGHT]\n{right_text}".strip()
    row.record_type = scan.detected_type

    # Extract sequential entry number if present
    num_match = re.search(r"\b(\d{1,3})\s*\.", left_text) or re.search(r"^\s*(\d{1,3})\b", left_text)
    if num_match:
        row.entry_number = num_match.group(1)

    row.status = "ocr_done"
    session.add(row)
    from backend.app.pipeline.feedback import reset_ocr_feedback

    reset_ocr_feedback(session, row_id=row.id, commit=False)
    session.commit()
    return {"left": left_text, "right": right_text}
