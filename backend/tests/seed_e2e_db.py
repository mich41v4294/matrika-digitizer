"""
Repeatable synthetic database & image seed script for isolated Playwright E2E testing.

Usage:
    DATA_DIR=/tmp/matrika_playwright_audit \
    DB_PATH=/tmp/matrika_playwright_audit/matrika.db \
    python3 -m backend.tests.seed_e2e_db
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

BASE_REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = Path("/tmp/matrika_playwright_audit")

data_dir_env = Path(os.environ.get("DATA_DIR", str(DEFAULT_AUDIT_DIR))).resolve()
db_path_env = Path(os.environ.get("DB_PATH", str(data_dir_env / "matrika.db"))).resolve()

# Guard: never touch the repository's live data/ directory
repo_data_dir = (BASE_REPO_DIR / "data").resolve()
if data_dir_env == repo_data_dir or db_path_env.is_relative_to(repo_data_dir):
    raise RuntimeError(
        f"Refusing to seed live repository data directory ({repo_data_dir})! "
        "Point DATA_DIR and DB_PATH to an isolated /tmp directory."
    )

os.environ["DATA_DIR"] = str(data_dir_env)
os.environ["DB_PATH"] = str(db_path_env)

import cv2
import numpy as np
from sqlmodel import Session

from backend.app.config import settings
from backend.app.db.models import (
    Batch,
    CensusHousehold,
    CensusOverviewHouse,
    CensusSheet,
    FamilySearchMatch,
    LocalRecordMatch,
    PersonFact,
    QuestionItem,
    RecordRow,
    ScanImage,
    WorkflowTrace,
)
from backend.app.db.session import engine, init_db
from backend.app.pipeline.lexicon import compute_person_fact_search_fields
from backend.app.search.fts import rebuild_all_fts


def _write_synthetic_jpeg(
    path: Path,
    width: int,
    height: int,
    title: str,
    subtitle: str = "Synteticky testovaci sken",
    draw_grid: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((height, width, 3), (232, 241, 246), dtype=np.uint8)

    # Outer border
    cv2.rectangle(canvas, (8, 8), (width - 9, height - 9), (90, 75, 60), 2)

    if draw_grid and height >= 200:
        header_y = min(140, height // 4)
        cv2.line(canvas, (8, header_y), (width - 9, header_y), (110, 90, 70), 2)
        step_y = max(45, (height - header_y - 20) // 5)
        for y in range(header_y + step_y, height - 15, step_y):
            cv2.line(canvas, (8, y), (width - 9, y), (180, 170, 155), 1)
        for x_frac in (0.12, 0.38, 0.65, 0.85):
            x = int(width * x_frac)
            cv2.line(canvas, (x, 8), (x, height - 9), (170, 160, 145), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        title[:65],
        (24, min(48, height // 2)),
        font,
        0.75 if width >= 600 else 0.45,
        (40, 35, 30),
        2,
        cv2.LINE_AA,
    )
    if height >= 90 and subtitle:
        cv2.putText(
            canvas,
            subtitle[:80],
            (24, min(88, height - 16)),
            font,
            0.55 if width >= 600 else 0.38,
            (80, 70, 60),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def seed_isolated_database() -> None:
    if data_dir_env.exists():
        shutil.rmtree(data_dir_env)
    settings.ensure_dirs()
    init_db()

    upload_dir = Path(settings.upload_dir)
    census_dir = Path(settings.census_data_dir)
    crops_dir = Path(settings.crops_dir)

    # 1. Pre-populate Slovakiana facets cache & FamilySearch regional places cache
    facets_payload = {
        "years": [
            {"key": "10001", "tagKey": "10001", "name": "1930", "count": 2},
            {"key": "10003", "tagKey": "10003", "name": "1940", "count": 0},
        ],
        "districts": [
            {"key": "769", "tagKey": "769", "name": "Trnava", "count": 2},
            {"key": "742", "tagKey": "742", "name": "Nitra", "count": 0},
        ],
        "municipalities": [
            {"key": "76901", "tagKey": "76901", "name": "Obec Testová", "count": 2},
            {"key": "76902", "tagKey": "76902", "name": "Trnava", "count": 0},
        ],
    }
    for fname in ("slovakiana_facets_all.json", "slovakiana_facets_unblack.json"):
        (data_dir_env / fname).write_text(
            json.dumps(facets_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (census_dir / fname).write_text(
            json.dumps(facets_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    regional_cache = {
        "places": {
            "obec testova": {
                "id": "99001",
                "name": "Obec Testová",
                "district": "Trnava",
                "country": "Slovensko",
                "full_name": "Obec Testová, Trnava, Slovensko",
                "latitude": 48.3774,
                "longitude": 17.5872,
            },
            "trnava": {
                "id": "99002",
                "name": "Trnava",
                "district": "Trnava",
                "country": "Slovensko",
                "full_name": "Trnava, Trnava, Slovensko",
                "latitude": 48.3774,
                "longitude": 17.5872,
            },
        },
        "queries": {
            "obec testova": [
                {
                    "id": "99001",
                    "name": "Obec Testová",
                    "type": "Town",
                    "full_name": "Obec Testová, Trnava, Czechoslovakia",
                    "latitude": 48.3774,
                    "longitude": 17.5872,
                }
            ],
            "trnava": [
                {
                    "id": "99002",
                    "name": "Trnava",
                    "type": "City",
                    "full_name": "Trnava, Trnava, Czechoslovakia",
                    "latitude": 48.3774,
                    "longitude": 17.5872,
                }
            ],
        },
    }
    (data_dir_env / "regional_places_cache.json").write_text(
        json.dumps(regional_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2. Generate synthetic JPEG images for Matrika Scan #1 and Census Scans #2, #3, #4
    matrika_scan_file = upload_dir / "matrika_obec_testova_1898_p01.jpg"
    _write_synthetic_jpeg(
        matrika_scan_file,
        1600,
        1100,
        "Matrika Pokrstenych - Obec Testova 1898",
        "Farnost Obec Testova, strana 1 (zapisy c. 12 - 14)",
    )
    _write_synthetic_jpeg(
        crops_dir / "scan_1" / "overlay.jpg",
        1600,
        1100,
        "Matrika Overlay - Obec Testova 1898",
        "Detekovane riadky #1, #2, #3",
    )
    _write_synthetic_jpeg(
        crops_dir / "scan_1" / "header.jpg",
        1600,
        220,
        "Hlavicka stlpce: Nomen | Parentes | Locus & Domus | Patrini | Observationes",
        draw_grid=False,
    )
    for r_idx, label in (
        (1, "Zapis 12: 14.03.1898 | Joannes Kovac | Jozef Kovac & Maria Vargova | Obec Testova 15"),
        (2, "Zapis 13: 22.07.1898 | Anna Horvathova | Martin Horvath & Katarina Kralova | Obec Testova 24"),
        (3, "Zapis 14: 10.11.1898 | Martin Balaz | Andrej Balaz & Helena Sykorova | Obec Testova 8"),
    ):
        _write_synthetic_jpeg(
            crops_dir / "scan_1" / f"row_0{r_idx}_stitched.jpg",
            1500,
            140,
            label,
            draw_grid=False,
        )
        _write_synthetic_jpeg(
            crops_dir / "scan_1" / f"row_0{r_idx}_left.jpg",
            750,
            140,
            label[:45],
            draw_grid=False,
        )
        _write_synthetic_jpeg(
            crops_dir / "scan_1" / f"row_0{r_idx}_right.jpg",
            750,
            140,
            label[45:],
            draw_grid=False,
        )

    # Census Sheet cair-test001 (Household sheet)
    c1_front = census_dir / "cair-test001" / "cair-test001_v.jpg"
    c1_back = census_dir / "cair-test001" / "cair-test001_r.jpg"
    _write_synthetic_jpeg(
        c1_front,
        1800,
        1250,
        "Scitaci harok cair-test001 (1930) - Predna strana (_v)",
        "Obec Testova, dom c. 15, okres Trnava - Domacnost Jan Kovac",
    )
    _write_synthetic_jpeg(
        c1_back,
        1800,
        1250,
        "Scitaci harok cair-test001 (1930) - Zadna strana (_r)",
        "Obec Testova, dom c. 15 - Zapis osob (riadky 1 - 3)",
    )
    _write_synthetic_jpeg(
        census_dir / "thumbnails" / "cair-test001.jpg",
        320,
        240,
        "cair-test001",
        "Dom c. 15",
        draw_grid=False,
    )
    for scan_num in (2, 3):
        _write_synthetic_jpeg(
            crops_dir / f"scan_{scan_num}" / "overlay.jpg",
            1800,
            1250,
            f"Overlay skenu #{scan_num} (cair-test001)",
            "Detekovane riadky domacnosti",
        )
        _write_synthetic_jpeg(
            crops_dir / f"scan_{scan_num}" / "header.jpg",
            1800,
            260,
            "Stlpce 1-24: Por.c. | Priezvisko | Meno | Pomer k hlave | Nar. | Rodisko | Povolanie",
            draw_grid=False,
        )
    for r_idx, label in (
        (1, "Riadok 1: Kovac Jan | hlava domacnosti | *14.03.1898 Obec Testova | rolnik"),
        (2, "Riadok 2: Kovacova rod. Novakova Maria | manzelka | *05.05.1901 Trnava"),
        (3, "Riadok 3: Kovac Jozef | syn | *19.08.1926 Obec Testova"),
    ):
        _write_synthetic_jpeg(
            crops_dir / "scan_2" / f"row_0{r_idx}_stitched.jpg",
            1700,
            120,
            label,
            draw_grid=False,
        )

    # Census Sheet cair-test002 (Overview sheet)
    c2_front = census_dir / "cair-test002" / "cair-test002_v.jpg"
    _write_synthetic_jpeg(
        c2_front,
        1800,
        1250,
        "Prehlad scitacieho obvodu cair-test002 (1930) - Obvod I",
        "Obec Testova, okres Trnava - Komisar Jozef Kral",
    )
    _write_synthetic_jpeg(
        census_dir / "thumbnails" / "cair-test002.jpg",
        320,
        240,
        "cair-test002",
        "Prehlad obvodu I",
        draw_grid=False,
    )
    _write_synthetic_jpeg(
        crops_dir / "scan_4" / "overlay.jpg",
        1800,
        1250,
        "Overlay prehladu obvodu cair-test002",
        "Detekovane riadky domov #1 (dom c. 15), #2 (dom c. 24)",
    )
    _write_synthetic_jpeg(
        crops_dir / "scan_4" / "header.jpg",
        1800,
        260,
        "Prehlad obvodu: Rad.c. | Osada | Cislo domu | Obyvany | Majitel domu | Bytov | Osob",
        draw_grid=False,
    )
    for r_idx, label in (
        (1, "Dom 1: Obec Testova | dom c. 15 | obyvany | majitel: Jan Kovac | 1 byt | 3 osoby"),
        (2, "Dom 2: Obec Testova | dom c. 24 | obyvany | majitel: Martin Horvath | 1 byt | 4 osoby"),
    ):
        _write_synthetic_jpeg(
            crops_dir / "scan_4" / f"row_0{r_idx}_stitched.jpg",
            1700,
            110,
            label,
            draw_grid=False,
        )

    # Census Sheet cair-test003 (1940 Household sheet for Obec Testová, dom č. 15)
    c3_front = census_dir / "cair-test003" / "cair-test003_v.jpg"
    _write_synthetic_jpeg(
        c3_front,
        1800,
        1250,
        "Scitaci harok cair-test003 (1940) - Predna strana (_v)",
        "Obec Testova, dom c. 15, okres Trnava - Domacnost Jan Kovac (1940)",
    )
    _write_synthetic_jpeg(
        census_dir / "thumbnails" / "cair-test003.jpg",
        320,
        240,
        "cair-test003",
        "Dom c. 15 (1940)",
        draw_grid=False,
    )

    with Session(engine) as session:
        # 3. Seed Matrika Batch, ScanImage, RecordRows, PersonFacts
        batch = Batch(
            name="Matrika Krstov – Obec Testová 1898",
            parish_or_book="Rím. kat. farnosť Obec Testová",
            record_type_hint="baptism",
            total_scans=1,
            processed_scans=1,
            status="completed",
        )
        session.add(batch)
        session.flush()

        scan1 = ScanImage(
            batch_id=batch.id,
            filename="matrika_obec_testova_1898_p01.jpg",
            original_path=str(matrika_scan_file),
            overlay_path="/media/crops/scan_1/overlay.jpg",
            header_crop_path="/media/crops/scan_1/header.jpg",
            width=1600,
            height=1100,
            spine_x=800,
            detected_year="1898",
            detected_type="baptism",
            header_raw_ocr="Anno 1898 | Nomen | Parentes | Locus & Domus | Patrini | Observationes",
            grid_metadata_json=json.dumps(
                {
                    "spine_x": 800,
                    "skew_left_deg": 0.2,
                    "skew_right_deg": -0.1,
                    "slicing_mode": "spread",
                    "rows": [
                        {"row_index": 1, "y_start": 240, "y_end": 420},
                        {"row_index": 2, "y_start": 425, "y_end": 610},
                        {"row_index": 3, "y_start": 615, "y_end": 800},
                    ],
                }
            ),
            status="matched",
        )
        session.add(scan1)
        session.flush()

        row1 = RecordRow(
            scan_id=scan1.id or 1,
            row_index=1,
            entry_number="12",
            y_start=240,
            y_end=420,
            stitched_crop_path="/media/crops/scan_1/row_01_stitched.jpg",
            left_crop_path="/media/crops/scan_1/row_01_left.jpg",
            right_crop_path="/media/crops/scan_1/row_01_right.jpg",
            raw_ocr_left="12 | 14/15 Martius 1898 | Joannes | Jozef Kováč, Mária Vargová",
            raw_ocr_right="Obec Testová č. 15 | Štefan Novák, Anna Nováková | Sobáš 12.05.1924 v Trnave",
            raw_ocr_combined="12 | 14/15 Martius 1898 | Joannes | Jozef Kováč, Mária Vargová | Obec Testová č. 15 | Štefan Novák, Anna Nováková | Sobáš 12.05.1924 v Trnave",
            decision_action="ask_human",
            decision_confidence=0.78,
            decision_reason="Overenie rukopisu v stĺpci rodičov",
            record_type="baptism",
            status="matched",
        )
        row2 = RecordRow(
            scan_id=scan1.id or 1,
            row_index=2,
            entry_number="13",
            y_start=425,
            y_end=610,
            stitched_crop_path="/media/crops/scan_1/row_02_stitched.jpg",
            left_crop_path="/media/crops/scan_1/row_02_left.jpg",
            right_crop_path="/media/crops/scan_1/row_02_right.jpg",
            raw_ocr_left="13 | 22/23 Julius 1898 | Anna | Martin Horváth, Katarína Kráľová",
            raw_ocr_right="Obec Testová č. 24 | Ján Baláž, Alžbeta Balážová | —",
            raw_ocr_combined="13 | 22/23 Julius 1898 | Anna | Martin Horváth, Katarína Kráľová | Obec Testová č. 24 | Ján Baláž, Alžbeta Balážová",
            decision_action="auto_accept",
            decision_confidence=0.94,
            record_type="baptism",
            status="extracted",
        )
        row3 = RecordRow(
            scan_id=scan1.id or 1,
            row_index=3,
            entry_number="14",
            y_start=615,
            y_end=800,
            stitched_crop_path="/media/crops/scan_1/row_03_stitched.jpg",
            left_crop_path="/media/crops/scan_1/row_03_left.jpg",
            right_crop_path="/media/crops/scan_1/row_03_right.jpg",
            raw_ocr_left="14 | 10 Nov 1898 | Martin Baláž | Andrej Baláž, Helena Sýkorová",
            raw_ocr_right="Obec Testová č. 8 | Jozef Kováč, Peter Kráľ",
            raw_ocr_combined="14 | 10 Nov 1898 | Martin Baláž | Andrej Baláž, Helena Sýkorová | Obec Testová č. 8 | Jozef Kováč, Peter Kráľ",
            decision_action="auto_accept",
            decision_confidence=0.91,
            record_type="marriage",
            status="extracted",
        )
        session.add_all([row1, row2, row3])
        session.flush()

        fact1 = PersonFact(
            row_id=row1.id or 1,
            scan_id=scan1.id or 1,
            record_type="baptism",
            entry_number="12",
            event_year=1898,
            birth_or_event_date="14.03.1898",
            baptism_or_burial_date="15.03.1898",
            given_name_raw="Joannes",
            given_name_sk="Ján",
            given_name_hu="János",
            surname_raw="Kováč",
            surname_normalized="Kováč",
            sex="M",
            legitimacy="legitimus",
            religion="rím. kat.",
            father_name="Jozef Kováč",
            mother_name="Mária Vargová",
            mother_maiden_surname="Vargová",
            social_status_or_occupation="agricola",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="15",
            fs_place_standardized="Obec Testová, Trnava, Slovakia",
            godparents_or_witnesses="Štefan Novák, Anna Nováková",
            officiant="Juraj Kráľ, parochus",
            observationes_raw="Sobáš 12.05.1924 v Trnave s Máriou Novákovou",
            observationes_events_json=json.dumps(
                [
                    {
                        "event_type": "marriage",
                        "date": "12.05.1924",
                        "year": 1924,
                        "raw_note": "Sobáš 12.05.1924 v Trnave s Máriou Novákovou",
                    }
                ],
                ensure_ascii=False,
            ),
        )
        fact2 = PersonFact(
            row_id=row2.id or 2,
            scan_id=scan1.id or 1,
            record_type="baptism",
            entry_number="13",
            event_year=1898,
            birth_or_event_date="22.07.1898",
            baptism_or_burial_date="23.07.1898",
            given_name_raw="Anna",
            given_name_sk="Anna",
            surname_raw="Horváthová",
            surname_normalized="Horváth",
            sex="F",
            legitimacy="legitima",
            religion="rím. kat.",
            father_name="Martin Horváth",
            mother_name="Katarína Kráľová",
            mother_maiden_surname="Kráľová",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="24",
            fs_place_standardized="Obec Testová, Trnava, Slovakia",
            godparents_or_witnesses="Ján Baláž, Alžbeta Balážová",
        )
        fact3 = PersonFact(
            row_id=row3.id or 3,
            scan_id=scan1.id or 1,
            record_type="marriage",
            entry_number="14",
            event_year=1898,
            birth_or_event_date="10.11.1898",
            given_name_raw="Martin",
            given_name_sk="Martin",
            surname_raw="Baláž",
            surname_normalized="Baláž",
            sex="M",
            religion="rím. kat.",
            father_name="Andrej Baláž",
            mother_name="Helena Sýkorová",
            mother_maiden_surname="Sýkorová",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="8",
            fs_place_standardized="Obec Testová, Trnava, Slovakia",
            godparents_or_witnesses="Jozef Kováč, Peter Kráľ",
        )
        for mf in (fact1, fact2, fact3):
            sf = compute_person_fact_search_fields(mf)
            mf.surname_folded = sf["surname_folded"]
            mf.surname_skeleton = sf["surname_skeleton"]
            mf.given_canonical_id = sf["given_canonical_id"]
            mf.village_folded = sf["village_folded"]
            mf.house_number_int = sf["house_number_int"]
            session.add(mf)
        session.flush()

        # 4. Seed QuestionItems (1 open, 1 skipped_for_admin)
        q1 = QuestionItem(
            row_id=row1.id or 1,
            scan_id=scan1.id or 1,
            field_name="parents",
            question_sk="Overenie priezviska a mien rodičov – zápis č. 12 (dieťa: Joannes Kováč)",
            question_en="Verify parents' names and surname in entry #12 (child: Joannes Kováč)",
            crop_path="/media/crops/scan_1/row_01_stitched.jpg",
            candidates_json=json.dumps(
                ["Jozef Kováč a Mária Vargová", "Jozef Kovač a Mária Vargová"],
                ensure_ascii=False,
            ),
            raw_ocr_snippet="Jozef Kováč, Mária Vargová",
            status="open",
        )
        q2 = QuestionItem(
            row_id=row2.id or 2,
            scan_id=scan1.id or 1,
            field_name="village",
            question_sk="Overenie názvu obce a čísla domu – zápis č. 13 (Obec Testová č. 24)",
            question_en="Verify village name and house number in entry #13 (Obec Testová #24)",
            crop_path="/media/crops/scan_1/row_02_stitched.jpg",
            candidates_json=json.dumps(
                ["Obec Testová č. 24", "Trnava č. 24"],
                ensure_ascii=False,
            ),
            raw_ocr_snippet="Obec Testová č. 24",
            status="skipped_for_admin",
        )
        session.add_all([q1, q2])
        session.flush()

        # 5. Seed Census Scans (#2, #3 for cair-test001; #4 for cair-test002)
        scan2 = ScanImage(
            filename="cair-test001_v.jpg",
            original_path=str(c1_front),
            overlay_path="/media/crops/scan_2/overlay.jpg",
            header_crop_path="/media/crops/scan_2/header.jpg",
            page_side="front",
            width=1800,
            height=1250,
            detected_year="1930",
            detected_type="census",
            status="matched",
        )
        scan3 = ScanImage(
            filename="cair-test001_r.jpg",
            original_path=str(c1_back),
            overlay_path="/media/crops/scan_3/overlay.jpg",
            header_crop_path="/media/crops/scan_3/header.jpg",
            page_side="back",
            width=1800,
            height=1250,
            detected_year="1930",
            detected_type="census",
            status="matched",
        )
        scan4 = ScanImage(
            filename="cair-test002_v.jpg",
            original_path=str(c2_front),
            overlay_path="/media/crops/scan_4/overlay.jpg",
            header_crop_path="/media/crops/scan_4/header.jpg",
            page_side="front",
            width=1800,
            height=1250,
            detected_year="1930",
            detected_type="census",
            status="extracted",
        )
        session.add_all([scan2, scan3, scan4])
        session.flush()

        # 6. Seed CensusSheet #1 (cair-test001 - household) and CensusSheet #2 (cair-test002 - overview)
        sheet1 = CensusSheet(
            slovakiana_key="cair-test001",
            digital_object_id="doid-test001",
            urn_nbn="urn:nbn:sk:sna-cair-test001",
            census_year=1930,
            district="Trnava",
            district_key="769",
            district_folded="trnava",
            municipality="Obec Testová",
            municipality_key="76901",
            municipality_folded="obec testova",
            street_raw="Hlavná",
            house_number="15",
            house_number_int=15,
            box_number="0101",
            sheet_number="0015",
            title_raw="Sčítací hárok 101/15, Hlavná 15, Obec Testová (Trnava)",
            is_blackened=False,
            has_images=True,
            front_scan_id=scan2.id,
            back_scan_id=scan3.id,
            image_urls_json=json.dumps(
                [
                    {
                        "index": 0,
                        "page_side": "front",
                        "width": 1800,
                        "height": 1250,
                        "jpeg_url": "/media/uploads/slovakiana/cair-test001/cair-test001_v.jpg",
                        "thumbnail_url": "/media/uploads/slovakiana/thumbnails/cair-test001.jpg",
                    },
                    {
                        "index": 1,
                        "page_side": "back",
                        "width": 1800,
                        "height": 1250,
                        "jpeg_url": "/media/uploads/slovakiana/cair-test001/cair-test001_r.jpg",
                    },
                ]
            ),
            household_owner="Ján Kováč",
            sheet_type="household",
            obvod_number="I",
            download_status="matched",
        )
        sheet2 = CensusSheet(
            slovakiana_key="cair-test002",
            digital_object_id="doid-test002",
            urn_nbn="urn:nbn:sk:sna-cair-test002",
            census_year=1930,
            district="Trnava",
            district_key="769",
            district_folded="trnava",
            municipality="Obec Testová",
            municipality_key="76901",
            municipality_folded="obec testova",
            box_number="0101",
            sheet_number="0001",
            title_raw="Prehľad sčítacieho obvodu I, Obec Testová (Trnava)",
            is_blackened=False,
            has_images=True,
            front_scan_id=scan4.id,
            image_urls_json=json.dumps(
                [
                    {
                        "index": 0,
                        "page_side": "front",
                        "width": 1800,
                        "height": 1250,
                        "jpeg_url": "/media/uploads/slovakiana/cair-test002/cair-test002_v.jpg",
                        "thumbnail_url": "/media/uploads/slovakiana/thumbnails/cair-test002.jpg",
                    }
                ]
            ),
            sheet_type="overview",
            obvod_number="I",
            commissioner_name="Jozef Kráľ",
            download_status="extracted",
        )
        sheet3 = CensusSheet(
            slovakiana_key="cair-test003",
            digital_object_id="doid-test003",
            urn_nbn="urn:nbn:sk:sna-cair-test003",
            census_year=1940,
            district="Trnava",
            district_key="769",
            district_folded="trnava",
            municipality="Obec Testová",
            municipality_key="76901",
            municipality_folded="obec testova",
            street_raw="Hlavná",
            house_number="15",
            house_number_int=15,
            box_number="0201",
            sheet_number="0015",
            title_raw="Sčítací hárok 201/15, Hlavná 15, Obec Testová (Trnava)",
            is_blackened=True,
            has_images=True,
            image_urls_json=json.dumps(
                [
                    {
                        "index": 0,
                        "page_side": "front",
                        "width": 1800,
                        "height": 1250,
                        "jpeg_url": "/media/uploads/slovakiana/cair-test003/cair-test003_v.jpg",
                        "thumbnail_url": "/media/uploads/slovakiana/thumbnails/cair-test003.jpg",
                    }
                ]
            ),
            household_owner="Ján Kováč",
            sheet_type="household",
            obvod_number="I",
            download_status="cataloged",
        )
        session.add_all([sheet1, sheet2, sheet3])
        session.flush()

        scan2.census_sheet_id = sheet1.id
        scan3.census_sheet_id = sheet1.id
        scan4.census_sheet_id = sheet2.id
        session.add_all([scan2, scan3, scan4])
        session.flush()

        hh1 = CensusHousehold(
            census_sheet_id=sheet1.id or 1,
            dwelling_unit="I",
            apartment_number="1",
            house_owner_name="Ján Kováč",
            street="Hlavná",
            house_number="15",
            fs_household_status="confirmed_household",
            fs_confidence=0.95,
        )
        session.add(hh1)
        session.flush()

        # Census rows & members on sheet1 (cair-test001)
        crow1 = RecordRow(
            scan_id=scan2.id or 2,
            row_index=1,
            entry_number="1",
            y_start=300,
            y_end=430,
            stitched_crop_path="/media/crops/scan_2/row_01_stitched.jpg",
            raw_ocr_combined="1 | Kováč | Ján | hlava domácnosti | M | 14.03.1898 | ženatý | Obec Testová | roľník",
            decision_action="auto_accept",
            decision_confidence=0.96,
            record_type="census",
            status="matched",
        )
        crow2 = RecordRow(
            scan_id=scan2.id or 2,
            row_index=2,
            entry_number="2",
            y_start=435,
            y_end=565,
            stitched_crop_path="/media/crops/scan_2/row_02_stitched.jpg",
            raw_ocr_combined="2 | Kováčová rod. Nováková | Mária | manželka | F | 05.05.1901 | vydatá | Trnava | v domácnosti",
            decision_action="auto_accept",
            decision_confidence=0.95,
            record_type="census",
            status="extracted",
        )
        crow3 = RecordRow(
            scan_id=scan2.id or 2,
            row_index=3,
            entry_number="3",
            y_start=570,
            y_end=700,
            stitched_crop_path="/media/crops/scan_2/row_03_stitched.jpg",
            raw_ocr_combined="3 | Kováč | Jozef | syn | M | 19.08.1926 | slobodný | Obec Testová | dieťa",
            decision_action="auto_accept",
            decision_confidence=0.95,
            record_type="census",
            status="extracted",
        )
        session.add_all([crow1, crow2, crow3])
        session.flush()

        cfact1 = PersonFact(
            row_id=crow1.id or 4,
            scan_id=scan2.id or 2,
            record_type="census",
            census_sheet_id=sheet1.id,
            census_household_id=hh1.id,
            person_row_number="1",
            entry_number="1",
            event_year=1930,
            given_name_raw="Ján",
            given_name_sk="Ján",
            surname_raw="Kováč",
            surname_normalized="Kováč",
            relation_to_head="hlava domácnosti",
            sex="M",
            marital_status="ženatý",
            marriage_date_raw="12.05.1924",
            birth_or_event_date="14.03.1898",
            birth_year_est=1898,
            birth_place_municipality="Obec Testová",
            birth_place_district="Trnava",
            birth_place_country="Slovensko",
            birth_place_raw="Obec Testová, okr. Trnava, Slovensko",
            birth_place_sk="Obec Testová",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="15",
            home_jurisdiction="Obec Testová",
            citizenship="ČSR",
            nationality="slovenská",
            religion="rím. kat.",
            literacy="čítať a písať",
            social_status_or_occupation="roľník",
            occupation_position="samostatný",
            employer_workplace="vlastné hospodárstvo",
            presence_type="trvale",
            spouse_name="Mária Kováčová rod. Nováková",
            father_name="Jozef Kováč",
            mother_name="Mária Vargová",
        )
        cfact2 = PersonFact(
            row_id=crow2.id or 5,
            scan_id=scan2.id or 2,
            record_type="census",
            census_sheet_id=sheet1.id,
            census_household_id=hh1.id,
            person_row_number="2",
            entry_number="2",
            event_year=1930,
            given_name_raw="Mária",
            given_name_sk="Mária",
            surname_raw="Kováčová",
            surname_normalized="Kováč",
            mother_maiden_surname="Nováková",
            relation_to_head="manželka",
            sex="F",
            marital_status="vydatá",
            marriage_date_raw="12.05.1924",
            children_born_count="2",
            children_dead_count="0",
            birth_or_event_date="05.05.1901",
            birth_year_est=1901,
            birth_place_municipality="Trnava",
            birth_place_district="Trnava",
            birth_place_country="Slovensko",
            birth_place_raw="Trnava, okr. Trnava, Slovensko",
            birth_place_sk="Trnava",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="15",
            home_jurisdiction="Obec Testová",
            citizenship="ČSR",
            nationality="slovenská",
            religion="rím. kat.",
            literacy="čítať a písať",
            social_status_or_occupation="v domácnosti",
            presence_type="trvale",
            spouse_name="Ján Kováč",
        )
        cfact3 = PersonFact(
            row_id=crow3.id or 6,
            scan_id=scan2.id or 2,
            record_type="census",
            census_sheet_id=sheet1.id,
            census_household_id=hh1.id,
            person_row_number="3",
            entry_number="3",
            event_year=1930,
            given_name_raw="Jozef",
            given_name_sk="Jozef",
            surname_raw="Kováč",
            surname_normalized="Kováč",
            relation_to_head="syn",
            sex="M",
            marital_status="slobodný",
            birth_or_event_date="19.08.1926",
            birth_year_est=1926,
            birth_place_municipality="Obec Testová",
            birth_place_district="Trnava",
            birth_place_country="Slovensko",
            birth_place_raw="Obec Testová, okr. Trnava, Slovensko",
            birth_place_sk="Obec Testová",
            village_raw="Obec Testová",
            village_sk="Obec Testová",
            house_number="15",
            home_jurisdiction="Obec Testová",
            citizenship="ČSR",
            nationality="slovenská",
            religion="rím. kat.",
            literacy="—",
            social_status_or_occupation="dieťa",
            presence_type="trvale",
            father_name="Ján Kováč",
            mother_name="Mária Kováčová rod. Nováková",
            mother_maiden_surname="Nováková",
        )
        for cf in (cfact1, cfact2, cfact3):
            sf = compute_person_fact_search_fields(cf)
            cf.surname_folded = sf["surname_folded"]
            cf.surname_skeleton = sf["surname_skeleton"]
            cf.given_canonical_id = sf["given_canonical_id"]
            cf.village_folded = sf["village_folded"]
            cf.house_number_int = sf["house_number_int"]
            session.add(cf)
        session.flush()

        hh1.head_person_fact_id = cfact1.id
        session.add(hh1)

        # Cross-match between Census Ján Kováč and Matrika Joannes Kováč
        lrm = LocalRecordMatch(
            census_fact_id=cfact1.id or 4,
            matrika_fact_id=fact1.id or 1,
            confidence_score=0.96,
            match_kind="baptism",
            what_agrees_json=json.dumps(
                [
                    "Zhoda mena (Ján = Joannes), priezviska (Kováč), obce (Obec Testová č. 15) a dátumu narodenia (14.03.1898)"
                ],
                ensure_ascii=False,
            ),
            reviewed_by_user=False,
        )
        session.add(lrm)

        # 7. Seed Overview Sheet rows & CensusOverviewHouse records on cair-test002
        ov_row1 = RecordRow(
            scan_id=scan4.id or 4,
            row_index=1,
            entry_number="1",
            y_start=310,
            y_end=430,
            stitched_crop_path="/media/crops/scan_4/row_01_stitched.jpg",
            raw_ocr_combined="1 | Obec Testová | 15 | obývaný | Ján Kováč | 1 | 3 | roľnícka usadlosť",
            decision_action="auto_accept",
            decision_confidence=0.93,
            record_type="census",
            status="extracted",
        )
        ov_row2 = RecordRow(
            scan_id=scan4.id or 4,
            row_index=2,
            entry_number="2",
            y_start=435,
            y_end=555,
            stitched_crop_path="/media/crops/scan_4/row_02_stitched.jpg",
            raw_ocr_combined="2 | Obec Testová | 24 | obývaný | Martin Horváth | 1 | 4 | dom",
            decision_action="auto_accept",
            decision_confidence=0.91,
            record_type="census",
            status="extracted",
        )
        session.add_all([ov_row1, ov_row2])
        session.flush()

        ov_house1 = CensusOverviewHouse(
            census_sheet_id=sheet2.id or 2,
            scan_id=scan4.id,
            row_index=1,
            census_year=1930,
            district="Trnava",
            municipality="Obec Testová",
            obvod_number="I",
            ordinal_number="1",
            settlement_name="Obec Testová",
            house_number="15",
            house_number_int=15,
            is_inhabited=True,
            owner_name="Ján Kováč",
            owner_surname_folded="kovac",
            household_count=1,
            present_persons_count=3,
            remarks="roľnícka usadlosť",
            matched_sheet_id=sheet1.id,
            stitched_crop_path="/media/crops/scan_4/row_01_stitched.jpg",
            raw_ocr_text="1 | Obec Testová | 15 | obývaný | Ján Kováč | 1 | 3 | roľnícka usadlosť",
            confidence=0.95,
        )
        ov_house2 = CensusOverviewHouse(
            census_sheet_id=sheet2.id or 2,
            scan_id=scan4.id,
            row_index=2,
            census_year=1930,
            district="Trnava",
            municipality="Obec Testová",
            obvod_number="I",
            ordinal_number="2",
            settlement_name="Obec Testová",
            house_number="24",
            house_number_int=24,
            is_inhabited=True,
            owner_name="Martin Horváth",
            owner_surname_folded="horvath",
            household_count=1,
            present_persons_count=4,
            remarks="dom",
            matched_sheet_id=None,
            stitched_crop_path="/media/crops/scan_4/row_02_stitched.jpg",
            raw_ocr_text="2 | Obec Testová | 24 | obývaný | Martin Horváth | 1 | 4 | dom",
            confidence=0.91,
        )
        session.add_all([ov_house1, ov_house2])
        session.flush()

        # 8. Seed FamilySearchMatches (1 Matrika, 1 Census) with what_agrees, what_differs, what_is_new, and multiple candidates (XXXX-YYY)
        fsm1 = FamilySearchMatch(
            person_fact_id=fact1.id or 1,
            row_id=row1.id or 1,
            match_status="existing_needs_facts",
            fs_person_id="KWCZ-001",
            fs_url="https://www.familysearch.org/search/record/results?q.givenName=J%C3%A1n&q.surname=Kov%C3%A1%C4%8D",
            confidence_score=0.91,
            summary_sk="Matrika pokrstených 1898 (Obec Testová č. 15): Joannes (Ján) Kováč",
            selected_fs_person_ids_json=json.dumps(["KWCZ-001"], ensure_ascii=False),
            candidates_json=json.dumps(
                [
                    {
                        "fs_person_id": "KWCZ-001",
                        "fs_id": "KWCZ-001",
                        "name": "Joannes Kováč",
                        "sex": "M",
                        "lifespan": "1898–1964",
                        "birth_date": "15.03.1898",
                        "birth_year": 1898,
                        "birth_place": "Obec Testová, Trnava, Slovakia",
                        "relatives_summary": "Otec: Jozef Kováč, Matka: Mária Vargová",
                        "confidence_score": 0.91,
                        "what_agrees": [
                            {"field_sk": "Meno a priezvisko", "value": "Ján Kováč (*1898)"},
                            {"field_sk": "Rodičia", "value": "Jozef Kováč & Mária Vargová"},
                        ],
                        "what_differs": [
                            {
                                "field_sk": "Dátum narodenia",
                                "matrika_val": "14.03.1898",
                                "fs_val": "15.03.1898 (uvedený dátum krstu)",
                            }
                        ],
                        "selected": True,
                    },
                    {
                        "fs_person_id": "KWCZ-019",
                        "fs_id": "KWCZ-019",
                        "name": "Ján Kováč",
                        "sex": "M",
                        "lifespan": "*1898",
                        "birth_date": "14.03.1898",
                        "birth_year": 1898,
                        "birth_place": "Obec Testová, Nyitra, Hungary",
                        "relatives_summary": "Otec: Josephus Kováč",
                        "confidence_score": 0.87,
                        "what_agrees": [
                            {"field_sk": "Meno a priezvisko", "value": "Ján Kováč (*1898) (KWCZ-019)"},
                        ],
                        "what_differs": [],
                        "selected": False,
                    },
                ],
                ensure_ascii=False,
            ),
            what_agrees_json=json.dumps(
                [
                    {"field_sk": "Meno a priezvisko", "value": "Ján Kováč (*1898)"},
                    {"field_sk": "Rodičia", "value": "Jozef Kováč & Mária Vargová"},
                ],
                ensure_ascii=False,
            ),
            what_differs_json=json.dumps(
                [
                    {
                        "field_sk": "Dátum narodenia",
                        "matrika_val": "14.03.1898",
                        "fs_val": "15.03.1898 (uvedený dátum krstu)",
                    }
                ],
                ensure_ascii=False,
            ),
            what_is_new_json=json.dumps(
                [
                    {"field_sk": "Krstní rodičia", "value": "Štefan Novák, Anna Nováková"},
                    {"field_sk": "Poznámka o sobáši", "value": "Sobáš 12.05.1924 v Trnave"},
                ],
                ensure_ascii=False,
            ),
            action_packet_json=json.dumps(
                {
                    "person_name_display": "Ján (Joannes) Kováč",
                    "given_name": "Ján",
                    "surname": "Kováč",
                    "event_type": "baptism",
                    "event_year": 1898,
                    "birth_date": "14.03.1898",
                    "baptism_date": "15.03.1898",
                    "standardized_place": "Obec Testová, Trnava, Slovakia",
                },
                ensure_ascii=False,
            ),
            reviewed_by_user=False,
        )
        fsm2 = FamilySearchMatch(
            person_fact_id=cfact1.id or 4,
            row_id=crow1.id or 4,
            match_status="confirmed_match",
            fs_person_id="KWCZ-001",
            fs_url="https://www.familysearch.org/search/record/results?q.givenName=J%C3%A1n&q.surname=Kov%C3%A1%C4%8D&q.residenceDate.from=1930",
            confidence_score=0.96,
            summary_sk="Sčítací hárok 1930 (Obec Testová, dom č. 15): Ján Kováč (hlava domácnosti)",
            selected_fs_person_ids_json=json.dumps(["KWCZ-001"], ensure_ascii=False),
            candidates_json=json.dumps(
                [
                    {
                        "fs_person_id": "KWCZ-001",
                        "fs_id": "KWCZ-001",
                        "name": "Ján Kováč",
                        "sex": "M",
                        "lifespan": "1898–1964",
                        "birth_date": "14.03.1898",
                        "birth_year": 1898,
                        "birth_place": "Obec Testová, Trnava, Slovensko",
                        "relatives_summary": "Manželka: Mária Kováčová",
                        "confidence_score": 0.96,
                        "what_agrees": [
                            {
                                "field_sk": "Lokálna matrika (zhoda)",
                                "value": "Joannes Kováč (baptism, rok 1898, dátum 14.03.1898) – skóre 0.96",
                            }
                        ],
                        "what_differs": [
                            {
                                "field_sk": "Číslo domu v FS",
                                "matrika_val": "Obec Testová č. 15",
                                "fs_val": "Obec Testová (bez čísla domu)",
                            }
                        ],
                        "selected": True,
                    },
                    {
                        "fs_person_id": "L4TR-882",
                        "fs_id": "L4TR-882",
                        "name": "Ján Kováč",
                        "sex": "M",
                        "lifespan": "*1898",
                        "birth_date": "1898",
                        "birth_year": 1898,
                        "birth_place": "Obec Testová, Trnava",
                        "relatives_summary": "Otec: Jozef Kováč",
                        "confidence_score": 0.89,
                        "what_agrees": [
                            {
                                "field_sk": "Meno v strome / zázname FS",
                                "value": "Ján Kováč (L4TR-882)",
                            }
                        ],
                        "what_differs": [],
                        "selected": False,
                    },
                ],
                ensure_ascii=False,
            ),
            what_agrees_json=json.dumps(
                [
                    {
                        "field_sk": "Lokálna matrika (zhoda)",
                        "value": "Joannes Kováč (baptism, rok 1898, dátum 14.03.1898) – skóre 0.96",
                    }
                ],
                ensure_ascii=False,
            ),
            what_differs_json=json.dumps(
                [
                    {
                        "field_sk": "Číslo domu v FS",
                        "matrika_val": "Obec Testová č. 15",
                        "fs_val": "Obec Testová (bez čísla domu)",
                    }
                ],
                ensure_ascii=False,
            ),
            what_is_new_json=json.dumps(
                [
                    {"field_sk": "Povolanie (1930)", "value": "roľník (samostatný)"},
                    {"field_sk": "Sčítací hárok", "value": "cair-test001 (Obec Testová č. 15)"},
                ],
                ensure_ascii=False,
            ),
            action_packet_json=json.dumps(
                {
                    "person_name_display": "Ján Kováč",
                    "given_name": "Ján",
                    "surname": "Kováč",
                    "event_type": "census",
                    "event_year": 1930,
                    "slovakiana_key": "cair-test001",
                    "standardized_place": "Obec Testová, Trnava, Slovensko",
                },
                ensure_ascii=False,
            ),
            reviewed_by_user=False,
        )
        session.add_all([fsm1, fsm2])

        # 9. Seed WorkflowTrace entries for /debug & model combo statistics
        traces_to_seed = [
            WorkflowTrace(
                scan_id=scan1.id,
                row_id=None,
                stage="layout",
                model_or_tool="opencv:angled_quad_row_slicer",
                prompt_text="Image matrika_obec_testova_1898_p01.jpg (1600x1100)",
                image_path="/media/crops/scan_1/overlay.jpg",
                response_text="Sliced 3 angled rows",
                parsed_json='{"rows_count": 3}',
                status="ok",
                duration_ms=180,
            ),
            WorkflowTrace(
                scan_id=scan1.id,
                row_id=row1.id,
                stage="ocr_left",
                model_or_tool="local:glm-ocr-g32-mixed_4_8-mlx",
                prompt_text="Table Recognition:",
                image_path="/media/crops/scan_1/row_01_stitched.jpg",
                response_text="12 | 14/15 Martius 1898 | Joannes | Jozef Kováč, Mária Vargová",
                parsed_json="{}",
                status="ok",
                duration_ms=640,
                prompt_tokens=190,
                completion_tokens=45,
            ),
            WorkflowTrace(
                scan_id=scan1.id,
                row_id=row1.id,
                stage="decision",
                model_or_tool="local:d1-3b-mlx",
                prompt_text="Zhodnoťte čitateľnosť rukopisu riadku č. 12",
                image_path="/media/crops/scan_1/row_01_stitched.jpg",
                response_text='{"action": "ask_human", "confidence": 0.78}',
                parsed_json='{"action": "ask_human", "confidence": 0.78}',
                status="ok",
                duration_ms=310,
                prompt_tokens=210,
                completion_tokens=40,
            ),
            WorkflowTrace(
                scan_id=scan1.id,
                row_id=row1.id,
                stage="extract",
                model_or_tool="local:underdog-saluki-27b-1.0",
                prompt_text="Extrahuj údaje z matričného riadku č. 12 (Obec Testová 1898)",
                image_path="/media/crops/scan_1/row_01_stitched.jpg",
                response_text='{"given_name_raw": "Joannes", "surname_raw": "Kováč", "village_raw": "Obec Testová", "house_number": "15"}',
                parsed_json='{"given_name_raw": "Joannes", "surname_raw": "Kováč", "village_raw": "Obec Testová", "house_number": "15"}',
                status="ok",
                duration_ms=420,
                prompt_tokens=310,
                completion_tokens=95,
            ),
            WorkflowTrace(
                scan_id=scan2.id,
                row_id=None,
                stage="layout",
                model_or_tool="opencv:census_sheet_slicer",
                prompt_text="Image cair-test001_v.jpg (1800x1250)",
                image_path="/media/crops/scan_2/overlay.jpg",
                response_text="Sliced 3 census rows",
                parsed_json='{"rows_count": 3}',
                status="ok",
                duration_ms=210,
            ),
            WorkflowTrace(
                scan_id=scan2.id,
                row_id=crow1.id,
                stage="ocr_left",
                model_or_tool="local:glm-ocr-g32-mixed_4_8-mlx",
                prompt_text="Table Recognition:",
                image_path="/media/crops/scan_2/row_01_stitched.jpg",
                response_text="1 | Kováč | Ján | hlava domácnosti | M | 14.03.1898",
                parsed_json="{}",
                status="ok",
                duration_ms=520,
                prompt_tokens=180,
                completion_tokens=40,
            ),
            WorkflowTrace(
                scan_id=scan2.id,
                row_id=crow1.id,
                stage="ocr_right",
                model_or_tool="local:glm-ocr-g32-mixed_4_8-mlx",
                prompt_text="Table Recognition:",
                image_path="/media/crops/scan_2/row_01_stitched.jpg",
                response_text="Obec Testová | Trnava | Slovensko | roľník",
                parsed_json="{}",
                status="ok",
                duration_ms=480,
                prompt_tokens=175,
                completion_tokens=38,
            ),
            WorkflowTrace(
                scan_id=scan2.id,
                row_id=crow1.id,
                stage="extract",
                model_or_tool="local:underdog-saluki-27b-1.0",
                prompt_text="Extract census row #1 (Obec Testová č. 15)",
                image_path="/media/crops/scan_2/row_01_stitched.jpg",
                response_text='{"surname_raw": "Kováč", "given_name_raw": "Ján", "relation_to_head": "hlava domácnosti"}',
                parsed_json='{"surname_raw": "Kováč", "given_name_raw": "Ján", "relation_to_head": "hlava domácnosti"}',
                status="ok",
                duration_ms=690,
                prompt_tokens=410,
                completion_tokens=110,
            ),
        ]
        session.add_all(traces_to_seed)
        session.commit()

        # 10. Rebuild FTS5 index
        rebuild_all_fts(session)
        session.commit()


if __name__ == "__main__":
    seed_isolated_database()
    print(f"Successfully seeded synthetic E2E database at {db_path_env}")
