from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

from backend.app.census.slovakiana_client import (
    normalize_category_keys_for_slovakiana,
    parse_box_and_sheet_from_title,
    parse_search_hit,
    parse_street_and_house_from_raw,
)
from backend.app.db.models import CensusSheet, PersonFact
from backend.app.export.gedcom import export_facts_to_gedcom
from backend.app.mcp.familysearch_client import match_census_to_local_matrika
from backend.app.pipeline.census_layout import slice_census_sheet
from backend.app.pipeline.decision_stage import detect_census_row_ditto
from backend.app.pipeline.extraction_stage import (
    _extract_birthplace_from_right_ocr,
    _heuristic_fallback_census_extract,
    _paleographic_stem_similarity,
    reconcile_census_household_relationships,
)
from backend.app.pipeline.lexicon import (
    compute_person_fact_search_fields,
    extract_maiden_surname,
    resolve_given_name_canonical_id,
    strip_slovak_feminine_suffix,
)
from backend.app.pipeline.schemas import apply_human_answer_to_fields, validate_census_date
from backend.app.search.fts import (
    ensure_fts_schema,
    index_census_sheet,
    index_person_fact,
    search_unified,
)


class TestSlovakianaClientParsing(unittest.TestCase):
    def test_category_key_normalization_strips_district_when_city_selected(self) -> None:
        # Slovakiana quirk: 3-digit district key must be omitted when 4/5-digit city key is present
        keys = normalize_category_keys_for_slovakiana(["269", "2742"])
        self.assertEqual(keys, ["2742"])

        keys_district_only = normalize_category_keys_for_slovakiana(["269"])
        self.assertEqual(keys_district_only, ["269"])

    def test_parse_street_and_house_and_box_sheet(self) -> None:
        street, raw_h, h_int = parse_street_and_house_from_raw("Kutlíkova, 17/278")
        self.assertEqual(street, "Kutlíkova")
        self.assertEqual(raw_h, "17/278")
        self.assertEqual(h_int, 278)

        street2, raw_h2, h_int2 = parse_street_and_house_from_raw("Škovránková, -/121")
        self.assertEqual(street2, "Škovránková")
        self.assertEqual(raw_h2, "121")
        self.assertEqual(h_int2, 121)

        box, sheet = parse_box_and_sheet_from_title(
            "Sčítací hárok 283/54, Veľké Ripňany (Topoľčany)"
        )
        self.assertEqual(box, "0283")
        self.assertEqual(sheet, "0054")

    def test_parse_search_hit_1930_and_1940(self) -> None:
        hit_1930 = {
            "key": "cair-test001",
            "name": "Sčítací hárok 283/54, - 42, Veľké Ripňany (Topoľčany)",
            "displayOriginDate": "1930",
            "municipality": "Veľké Ripňany",
            "censusSheetStreet": ["-, -/42"],
            "blackened": False,
        }
        parsed_30 = parse_search_hit(hit_1930)
        self.assertEqual(parsed_30["slovakiana_key"], "cair-test001")
        self.assertEqual(parsed_30["census_year"], 1930)
        self.assertEqual(parsed_30["municipality"], "Veľké Ripňany")
        self.assertEqual(parsed_30["house_number"], "42")
        self.assertFalse(parsed_30["is_blackened"])

        hit_1940 = {
            "key": "cair-test002",
            "name": "Sčítací hárok 410/12, Hlavná 105, Veľké Ripňany (Topoľčany)",
            "displayOriginDate": "1940",
            "municipality": "Veľké Ripňany",
            "censusSheetStreet": ["Hlavná, -/105"],
            "blackened": True,
        }
        parsed_40 = parse_search_hit(hit_1940)
        self.assertEqual(parsed_40["census_year"], 1940)
        self.assertTrue(parsed_40["is_blackened"])
        self.assertEqual(parsed_40["street_raw"], "Hlavná")
        self.assertEqual(parsed_40["house_number"], "105")


class TestCensusLexiconAndFTS5(unittest.TestCase):
    def test_feminine_surname_and_maiden_extraction(self) -> None:
        self.assertEqual(strip_slovak_feminine_suffix("Kováčová"), "Kováč")
        self.assertEqual(strip_slovak_feminine_suffix("Novotná"), "Novotný")
        married, maiden = extract_maiden_surname("Kováčová rod. Vargová")
        self.assertEqual(married, "Kováčová")
        self.assertEqual(maiden, "Vargová")

    def test_multilingual_given_name_canonical_id(self) -> None:
        cid_lat = resolve_given_name_canonical_id("Joannes")
        cid_sk = resolve_given_name_canonical_id("Ján")
        cid_hu = resolve_given_name_canonical_id("János")
        self.assertIsNotNone(cid_lat)
        self.assertEqual(cid_lat, cid_sk)
        self.assertEqual(cid_sk, cid_hu)

    def test_fts5_unified_search_and_local_matrika_cross_match(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        ensure_fts_schema(engine)

        with Session(engine) as session:
            # 1. Insert an 1892 baptism PersonFact ("Joannes Kovács")
            baptism_fact = PersonFact(
                row_id=1,
                scan_id=1,
                record_type="baptism",
                entry_number="14",
                event_year=1892,
                birth_or_event_date="15. Martius",
                given_name_raw="Joannes",
                given_name_sk="Ján",
                given_name_hu="János",
                surname_raw="Kovács",
                surname_normalized="Kováč",
                sex="M",
                father_name="Paulus",
                mother_name="Anna",
                village_raw="Nyitra",
                village_sk="Veľké Ripňany",
                house_number="42",
            )
            sf1 = compute_person_fact_search_fields(
                surname_raw=baptism_fact.surname_raw,
                surname_normalized=baptism_fact.surname_normalized,
                given_name_raw=baptism_fact.given_name_raw,
                given_name_sk=baptism_fact.given_name_sk,
                given_name_hu=baptism_fact.given_name_hu,
                village_raw=baptism_fact.village_raw,
                village_sk=baptism_fact.village_sk,
                house_number=baptism_fact.house_number,
            )
            for k, v in sf1.items():
                setattr(baptism_fact, k, v)
            session.add(baptism_fact)
            session.commit()
            session.refresh(baptism_fact)
            index_person_fact(session, baptism_fact)

            # 2. Insert a 1930 CensusSheet and household rows (Head: Ján Kováč, Wife: Mária Kováčová rod. Vargová, Son: Štefan Kováč)
            sheet = CensusSheet(
                slovakiana_key="cair-test001",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_folded="velke ripnany",
                house_number="42",
                house_number_int=42,
                sheet_number="0054",
                is_blackened=False,
            )
            session.add(sheet)
            session.commit()
            session.refresh(sheet)
            index_census_sheet(session, sheet)

            head = PersonFact(
                row_id=2,
                scan_id=2,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="1",
                event_year=1930,
                birth_or_event_date="15.03.1892",
                birth_year_est=1892,
                given_name_raw="Ján",
                given_name_sk="Ján",
                surname_raw="Kováč",
                surname_normalized="Kováč",
                relation_to_head="hlava domácnosti",
                sex="M",
                village_raw="Veľké Ripňany",
                village_sk="Veľké Ripňany",
                house_number="42",
                social_status_or_occupation="roľník",
            )
            wife = PersonFact(
                row_id=3,
                scan_id=2,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="2",
                event_year=1930,
                birth_or_event_date="04.08.1896",
                birth_year_est=1896,
                given_name_raw="Mária",
                given_name_sk="Mária",
                surname_raw="Kováčová",
                surname_normalized="Kováč",
                mother_maiden_surname="Vargová",
                relation_to_head="manželka",
                sex="F",
                village_raw="Veľké Ripňany",
                village_sk="Veľké Ripňany",
                house_number="42",
            )
            son = PersonFact(
                row_id=4,
                scan_id=2,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="3",
                event_year=1930,
                birth_or_event_date="12.01.1922",
                birth_year_est=1922,
                given_name_raw="Štefan",
                given_name_sk="Štefan",
                surname_raw="Kováč",
                surname_normalized="Kováč",
                relation_to_head="syn",
                sex="M",
                village_raw="Veľké Ripňany",
                village_sk="Veľké Ripňany",
                house_number="42",
            )
            for p in (head, wife, son):
                sf = compute_person_fact_search_fields(
                    surname_raw=p.surname_raw,
                    surname_normalized=p.surname_normalized,
                    given_name_raw=p.given_name_raw,
                    given_name_sk=p.given_name_sk,
                    given_name_hu=p.given_name_hu,
                    village_raw=p.village_raw,
                    village_sk=p.village_sk,
                    house_number=p.house_number,
                )
                for k, v in sf.items():
                    setattr(p, k, v)
                session.add(p)
            session.commit()

            household = reconcile_census_household_relationships(session, sheet.id or 0)
            self.assertIsNotNone(household)
            session.refresh(son)
            session.refresh(head)
            self.assertEqual(son.father_name, "Ján")
            self.assertEqual(son.mother_name, "Mária")
            self.assertEqual(son.mother_maiden_surname, "Vargová")
            self.assertIn("Mária", head.spouse_name or "")

            # Cross-match census head ("Ján Kováč", b. 1892) against local baptism ("Joannes Kovács", 1892)
            local_matches = match_census_to_local_matrika(session, head)
            self.assertEqual(len(local_matches), 1)
            self.assertEqual(local_matches[0].matrika_fact_id, baptism_fact.id)
            self.assertGreaterEqual(local_matches[0].confidence_score, 0.90)

            # Unified FTS5 search for "Kovac Jan" (without diacritics) must find both the 1892 baptism and 1930 census head
            res = search_unified(session, q="Kovac Jan")
            found_ids = {item["fact"].id for item in res["records"]}
            self.assertIn(baptism_fact.id, found_ids)
            self.assertIn(head.id, found_ids)

            # GEDCOM export includes CENS, RESI, OCCU
            gedcom = export_facts_to_gedcom([head])
            self.assertIn("1 CENS", gedcom)
            self.assertIn("1 RESI", gedcom)
            self.assertIn("1 OCCU roľník", gedcom)


class TestCensusLayoutAndHeuristics(unittest.TestCase):
    def test_slice_synthetic_census_sheet_suppresses_blank_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a synthetic 2400x1600 landscape census form with 8 table rows:
            # first 3 rows have simulated handwriting ink, bottom 5 rows are blank grid lines
            w, h = 2400, 1600
            img = np.full((h, w, 3), 245, dtype=np.uint8)

            # Draw header & table horizontal lines
            y_positions = [340, 460, 580, 700, 820, 940, 1060, 1180, 1300]
            for y in y_positions:
                cv2.line(img, (40, y), (w - 40, y), (80, 80, 80), 2)

            # Draw vertical grid lines
            for x in (40, 220, 620, 900, 1320, 1680, 2040, w - 40):
                cv2.line(img, (x, 340), (x, 1300), (90, 90, 90), 2)

            # Add simulated handwriting in rows 1, 2, 3 (y=340..700)
            for row_idx in range(3):
                ry = y_positions[row_idx] + 55
                cv2.putText(
                    img,
                    f"{row_idx + 1}  Kovac Jan   hlava domacnosti   15.III.1892   rolnik",
                    (120, ry),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (15, 15, 15),
                    3,
                )

            img_path = tmp_path / "synthetic_census_1930.jpg"
            cv2.imwrite(str(img_path), img)

            res = slice_census_sheet(
                img_path,
                scan_id=99,
                crops_root=tmp_path / "crops",
                page_side="front",
                census_year=1930,
                is_blackened=False,
            )
            self.assertEqual(res.slicing_mode, "census_24col_front")
            # Only the 3 populated rows should be emitted; the 5 empty pre-printed rows are suppressed
            self.assertEqual(len(res.rows), 3)
            for r in res.rows:
                keys = {c["key"] for c in r.cell_crops}
                self.assertIn("identity_name", keys)
                self.assertIn("birth_marital", keys)
                self.assertIn("occupation_presence_notes", keys)

    def test_heuristic_fallback_census_extract_and_question_answer(self) -> None:
        cdate, cyear = validate_census_date("15. III. 1892")
        self.assertEqual(cdate, "15. 3. 1892")
        self.assertEqual(cyear, 1892)

        parsed = _heuristic_fallback_census_extract(
            raw_left="1 | 1 | Kováčová rod. Vargová | Mária | manželka | žen. | 15. III. 1896 | vydatá",
            raw_right="Veľké Ripňany | slovenská | rím. kat. | číta a píše | v domácnosti",
            census_year=1930,
            row_index=2,
        )
        self.assertEqual(parsed["surname_raw"], "Kováčová")
        self.assertEqual(parsed["maiden_surname"], "Vargová")
        self.assertEqual(parsed["given_name_raw"], "Mária")
        self.assertEqual(parsed["relation_to_head"], "manželka")
        self.assertEqual(parsed["sex"], "F")
        self.assertEqual(parsed["birth_year_est"], 1896)
        self.assertEqual(parsed["religion_raw"], "R. K. (rímskokatolícka)")

        updated = apply_human_answer_to_fields(
            dict(parsed),
            "identity_relation",
            "Novotný Štefan (hlava domácnosti)",
        )
        self.assertEqual(updated["surname_raw"], "Novotný")
        self.assertEqual(updated["given_name_raw"], "Štefan")
        self.assertEqual(updated["relation_to_head"], "hlava domácnosti")

    def test_byt_majority_surname_consensus_and_cursive_repairs(self) -> None:
        self.assertGreaterEqual(_paleographic_stem_similarity("Kovač", "Kováč"), 0.85)
        self.assertGreaterEqual(_paleographic_stem_similarity("Horvath", "Horváth"), 0.85)

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        ensure_fts_schema(engine)

        with Session(engine) as session:
            sheet = CensusSheet(
                slovakiana_key="cair-test001",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_folded="velke ripnany",
                house_number="7",
                house_number_int=7,
                sheet_number="0039",
                household_owner="Heter Kovač",
            )
            session.add(sheet)
            session.commit()
            session.refresh(sheet)

            raw_rows = [
                ("1", "Kovač", "Heter", "hlava domácnosti", None, 1896),
                ("2", "Kovačova'", "Mária", "manželka", "F", 1901),
                ("3", "Kováčová", "Havia", "dcéra", "F", 1923),
                ("4", "Kováč", "Ján", "syn", "M", 1925),
                ("5", "Kováčová", "Helena", "dcéra", "F", 1927),
                ("6", "Kováčová", "Steena", "dcéra", "F", 1928),
                ("7", "Vargová", "Alžbeta", "svokra", "F", 1880),
                ("8", "Bystrická", "Vivencia", "švagriná", "F", 1900),
            ]
            facts: list[PersonFact] = []
            for idx, (p_row, sur, giv, rel, sex, byear) in enumerate(raw_rows, start=1):
                f = PersonFact(
                    row_id=idx,
                    scan_id=37,
                    record_type="census",
                    census_sheet_id=sheet.id,
                    person_row_number=p_row,
                    event_year=1930,
                    birth_year_est=byear,
                    surname_raw=sur,
                    surname_normalized=sur,
                    given_name_raw=giv,
                    given_name_sk=giv,
                    relation_to_head=rel,
                    sex=sex,
                    village_raw="Veľké Ripňany",
                    village_sk="Veľké Ripňany",
                    house_number="7",
                )
                session.add(f)
                facts.append(f)
            session.commit()

            hh = reconcile_census_household_relationships(session, sheet.id or 0)
            self.assertIsNotNone(hh)
            session.refresh(sheet)
            for f in facts:
                session.refresh(f)

            self.assertEqual(sheet.household_owner, "Štefan Kováč")
            self.assertEqual(hh.house_owner_name, "Štefan Kováč")
            self.assertEqual(hh.dwelling_unit, "I")
            self.assertEqual(facts[0].given_name_sk, "Štefan")
            self.assertEqual(facts[0].surname_raw, "Kováč")
            self.assertEqual(facts[1].surname_raw, "Kováčová")
            self.assertEqual(facts[2].given_name_sk, "Mária")
            self.assertEqual(facts[2].surname_raw, "Kováčová")
            self.assertEqual(facts[5].given_name_sk, "Irena")
            self.assertEqual(facts[5].surname_raw, "Kováčová")
            self.assertEqual(facts[6].surname_raw, "Vargová")
            self.assertEqual(facts[7].given_name_sk, "Vincencia")
            self.assertEqual(facts[7].surname_raw, "Bystrická")

    def test_blank_birthplace_a_before_b_and_cursive_33_relation_ditto(self) -> None:
        sheet = CensusSheet(
            slovakiana_key="cair-test002",
            census_year=1930,
            district="Trnava",
            municipality="Hrnčiarovce nad Parnou",
            municipality_folded="hrnciarovce nad parnou",
            house_number="38",
            house_number_int=38,
        )
        raw_right_row4 = (
            "(a) |  | \n"
            "(b) | Thranna | \n"
            "(c) | Novenha |\n"
            "[STĹPEC 15-17 NÁRODNOSŤ, NÁBOŽENSTVO, GRAMOTNOSŤ]\n"
            "slovenská | rím. kat. | -\n"
        )
        muni, dist, country = _extract_birthplace_from_right_ocr(raw_right_row4, sheet=sheet)
        self.assertEqual(muni, "//")
        self.assertEqual(dist, "Trnava")
        self.assertEqual(country, "Slovensko")

        prev_row3 = PersonFact(
            row_id=70,
            scan_id=21,
            record_type="census",
            person_row_number="3",
            event_year=1930,
            birth_year_est=1923,
            surname_raw="Nováková",
            surname_normalized="Novák",
            given_name_raw="Alžbeta",
            given_name_sk="Alžbeta",
            relation_to_head="dcéra",
            sex="F",
            birth_place_municipality="Hrnčiarovce nad Parnou",
            birth_place_district="Trnava",
            birth_place_country="Slovensko",
        )
        raw_left_row4 = (
            "[STĹPCE 1-5 PORADOVÉ ČÍSLO, PRIEZVISKO, MENO, POMER K HLAVE DOMÁCNOSTI]\n"
            "4 | Novákorá | Ana | 33 | jevrié | \n"
            "[STĹPCE 6-10 POHLAVIE, DÁTUM NARODENIA, RODINNÝ STAV]\n"
            "20. apr. 1927 | slb.\n"
        )
        fb4 = _heuristic_fallback_census_extract(
            raw_left_row4,
            raw_right_row4,
            census_year=1930,
            sheet=sheet,
            row_index=4,
            prev_fact=prev_row3,
        )
        self.assertEqual(fb4["surname_raw"], "Nováková")
        self.assertEqual(fb4["given_name_raw"], "Anna")
        self.assertEqual(fb4["relation_to_head"], "dcéra")
        self.assertEqual(fb4["birth_date_raw"], "20. Apr. 1927")
        self.assertEqual(fb4["birth_place_municipality"], "Hrnčiarovce nad Parnou")
        self.assertEqual(fb4["birth_place_district"], "Trnava")
        self.assertEqual(fb4["birth_place_country"], "Slovensko")

        has_ditto, notes = detect_census_row_ditto(raw_left_row4, raw_right_row4, None)
        self.assertTrue(has_ditto)
        self.assertGreaterEqual(len(notes), 2)

    def test_no_forced_ova_on_womens_surnames_and_adjectival_stastna_sanitization(self) -> None:
        from backend.app.pipeline.extraction_stage import _inflect_census_surname, _surname_stem
        from backend.app.pipeline.lexicon import sanitize_corrupted_surname_suffix

        # 1. Corrupted double/adjectival -ová suffixes are repaired
        for bad in (
            "Šťastnaová",
            "Šťastnáová",
            "Stastnaova",
            "Stasntnaova",
            "Stasntna",
            "Šťastnýová",
            "Stastnyova",
            "Siastnová",
            "Haštuová",
            "Hastuová",
        ):
            self.assertEqual(sanitize_corrupted_surname_suffix(bad), "Šťastná", f"Failed for {bad}")
            self.assertEqual(_surname_stem(bad), "Šťastný", f"Failed stem for {bad}")

        self.assertEqual(sanitize_corrupted_surname_suffix("Bystrickáová"), "Bystrická")
        self.assertEqual(sanitize_corrupted_surname_suffix("Majerskáová"), "Majerská")
        self.assertEqual(sanitize_corrupted_surname_suffix("Buchta-ová"), "Buchtová")
        self.assertEqual(sanitize_corrupted_surname_suffix("Vančoová"), "Vančová")
        self.assertEqual(sanitize_corrupted_surname_suffix("Parabaová"), "Parabová")

        # 2. Adjectival surnames always use -á / -ská / -cká for females, NEVER -ová
        self.assertEqual(_inflect_census_surname("Šťastný", "F", "manželka", raw_ocr_surname="Stasntna"), "Šťastná")
        self.assertEqual(_inflect_census_surname("Šťastný", "F", "dcéra", raw_ocr_surname=None, force_ova_on_ditto=True), "Šťastná")
        self.assertEqual(_inflect_census_surname("Bystrický", "F", "švagriná", raw_ocr_surname="Bystrická"), "Bystrická")
        self.assertEqual(_inflect_census_surname("Majerský", "F", "manželka", raw_ocr_surname="Majerská"), "Majerská")

        # 3. Noun-stem surnames written WITHOUT -ová on the scan are NOT forcefully given -ová
        self.assertEqual(_inflect_census_surname("Kráľ", "F", "svokra", raw_ocr_surname="Kráľ"), "Kráľ")
        self.assertEqual(_inflect_census_surname("Horváth", "F", "dcéra", raw_ocr_surname="Horvath"), "Horváth")
        self.assertEqual(_inflect_census_surname("Bódi", "F", "manželka", raw_ocr_surname="Bodi"), "Bódi")
        self.assertEqual(_inflect_census_surname("Vinter", "F", "manželka", raw_ocr_surname="Vinter"), "Vinter")
        self.assertEqual(_inflect_census_surname("Harsányi", "F", "manželka", raw_ocr_surname="Harsányi"), "Harsányi")

        # 4. Noun-stem surnames written WITH -ová (or OCR variant -ova/-orá/-ov') DO keep -ová properly
        self.assertEqual(_inflect_census_surname("Horváth", "F", "manželka", raw_ocr_surname="Horvathová"), "Horváthová")
        self.assertEqual(_inflect_census_surname("Paraba", "F", "manželka", raw_ocr_surname="Parabová"), "Parabová")
        self.assertEqual(_inflect_census_surname("Buchta", "F", "dcéra", raw_ocr_surname="Buchta-ov'"), "Buchtová")
        self.assertEqual(_inflect_census_surname("Vančo", "F", "manželka", raw_ocr_surname="Vancora"), "Vančová")

    def test_fuzzy_ocr_name_search_unified_and_census_hub(self) -> None:
        from backend.app.api.census import (
            filter_census_sheets,
            get_municipality_overview,
        )
        from backend.app.db.models import CensusOverviewHouse
        from backend.app.pipeline.lexicon import (
            fuzzy_name_match_score,
            fuzzy_token_similarity,
        )
        from backend.app.search.fts import search_unified_records

        # 1. Verify token-level and multi-token OCR fuzzy similarity scores
        self.assertGreaterEqual(fuzzy_token_similarity("Kováč", "Kowacz"), 0.85)
        self.assertGreaterEqual(fuzzy_token_similarity("Kováč", "Kovacs"), 0.85)
        self.assertGreaterEqual(fuzzy_token_similarity("Kováč", "Kouáč"), 0.80)
        self.assertGreaterEqual(fuzzy_token_similarity("Varga", "Yargo"), 0.80)
        self.assertGreaterEqual(fuzzy_token_similarity("Varga", "Warga"), 0.85)
        self.assertGreaterEqual(fuzzy_token_similarity("Šťastný", "Siastny"), 0.80)
        self.assertGreaterEqual(fuzzy_token_similarity("Šťastný", "Hastua"), 0.80)
        self.assertGreaterEqual(fuzzy_token_similarity("Horváth", "Horwath"), 0.85)
        self.assertGreaterEqual(fuzzy_token_similarity("Horváth", "Morvath"), 0.78)
        self.assertGreaterEqual(fuzzy_token_similarity("Novák", "Nowak"), 0.85)
        self.assertGreaterEqual(fuzzy_token_similarity("Krajčovič", "Krajcsovics"), 0.90)
        self.assertGreaterEqual(
            fuzzy_name_match_score("Ján Kováč", ["Joannes Kowacz"]),
            0.85,
        )

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        ensure_fts_schema(engine)

        with Session(engine) as session:
            # Create Matrika and Census records with cursive/OCR misreads
            f_kowacz = PersonFact(
                row_id=101,
                scan_id=10,
                record_type="baptism",
                event_year=1888,
                given_name_raw="Joannes",
                surname_raw="Kowacz",
                surname_normalized="Kowacz",
                village_raw="Veľké Ripňany",
            )
            f_kouac = PersonFact(
                row_id=102,
                scan_id=10,
                record_type="census",
                event_year=1930,
                given_name_raw="Jano",
                surname_raw="Kouáč",
                surname_normalized="Kouáč",
                village_raw="Veľké Ripňany",
            )
            f_yargo = PersonFact(
                row_id=103,
                scan_id=10,
                record_type="baptism",
                event_year=1890,
                given_name_raw="Stephanus",
                surname_raw="Yargo",
                surname_normalized="Yargo",
                mother_maiden_surname="Warga",
                village_raw="Veľké Ripňany",
            )
            f_siastny = PersonFact(
                row_id=104,
                scan_id=11,
                record_type="census",
                event_year=1930,
                given_name_raw="Mária",
                surname_raw="Siastny",
                surname_normalized="Siastny",
                surname_candidates=["Hastua"],
                village_raw="Horný Čepeň",
            )
            f_morvath = PersonFact(
                row_id=105,
                scan_id=11,
                record_type="census",
                event_year=1940,
                given_name_raw="Jozef",
                surname_raw="Morvath",
                surname_normalized="Horwath",
                village_raw="Horný Čepeň",
            )
            for pf in (f_kowacz, f_kouac, f_yargo, f_siastny, f_morvath):
                sf = compute_person_fact_search_fields(pf)
                for k, v in sf.items():
                    setattr(pf, k, v)
                session.add(pf)
            session.commit()
            for pf in (f_kowacz, f_kouac, f_yargo, f_siastny, f_morvath):
                session.refresh(pf)
                index_person_fact(session, pf)
            session.commit()

            # 2. Verify search_unified_records finds all mis-OCR'd variants
            res_kovac = search_unified_records(session, q="Kováč")
            kovac_ids = {r["fact"].id for r in res_kovac["records"]}
            self.assertIn(f_kowacz.id, kovac_ids)
            self.assertIn(f_kouac.id, kovac_ids)

            res_jan_kovac = search_unified_records(session, q="Ján Kováč")
            jan_kovac_ids = {r["fact"].id for r in res_jan_kovac["records"]}
            self.assertIn(f_kowacz.id, jan_kovac_ids)
            self.assertIn(f_kouac.id, jan_kovac_ids)

            res_varga = search_unified_records(session, q="Varga")
            varga_ids = {r["fact"].id for r in res_varga["records"]}
            self.assertIn(f_yargo.id, varga_ids)

            res_stastny = search_unified_records(session, q="Šťastný")
            stastny_ids = {r["fact"].id for r in res_stastny["records"]}
            self.assertIn(f_siastny.id, stastny_ids)

            res_horvath = search_unified_records(session, q="Horváth")
            horvath_ids = {r["fact"].id for r in res_horvath["records"]}
            self.assertIn(f_morvath.id, horvath_ids)

            # 3. Verify Census Hub sheet & municipality overview filtering with OCR errors
            ov_sheet = CensusSheet(
                slovakiana_key="cair-ovw901",
                census_year=1930,
                district="Trnava",
                municipality="Horný Čepeň",
                municipality_folded="horny cepen",
                sheet_type="overview",
                download_status="extracted",
            )
            hh_sheet = CensusSheet(
                slovakiana_key="cair-hh902",
                census_year=1930,
                district="Trnava",
                municipality="Horný Čepeň",
                municipality_folded="horny cepen",
                house_number="14",
                house_number_int=14,
                household_owner="Joannes Kowacz",
                sheet_type="household",
                download_status="extracted",
            )
            session.add(ov_sheet)
            session.add(hh_sheet)
            session.commit()
            session.refresh(ov_sheet)
            session.refresh(hh_sheet)

            f_siastny.census_sheet_id = hh_sheet.id
            session.add(f_siastny)
            ov_house = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 0,
                row_index=1,
                census_year=1930,
                municipality="Horný Čepeň",
                house_number="19",
                house_number_int=19,
                owner_surname_raw="Morvath",
                owner_given_name_raw="Doref",
                raw_row_text="19 | Morvath Doref | 1 | 4",
            )
            session.add(ov_house)
            session.commit()

            sheets_stastny = filter_census_sheets(
                session=session,
                municipality="Horný Čepeň",
                person_query="Šťastný",
            )
            self.assertEqual(sheets_stastny["total"], 1)
            self.assertEqual(sheets_stastny["sheets"][0]["sheet"]["slovakiana_id"], "cair-hh902")
            self.assertTrue(
                any(m["matched_filter"] for m in sheets_stastny["sheets"][0]["extracted_persons_preview"])
            )

            ov_horvath = get_municipality_overview(
                session=session,
                municipality="Horný Čepeň",
                person_query="Jozef Horváth",
            )
            self.assertEqual(ov_horvath["total_houses"], 1)
            self.assertEqual(ov_horvath["houses_directory"][0]["house_number"], "19")

    def test_familysearch_only_shows_matched_and_auto_triggers_on_person_add(self) -> None:
        from unittest.mock import AsyncMock, patch
        from fastapi.testclient import TestClient
        from backend.app.db.models import FamilySearchMatch, RecordRow
        from backend.app.db.session import engine as app_engine, init_db
        from backend.app.main import app

        init_db()
        client = TestClient(app)

        created_sheet_id: int | None = None
        created_row_ids: list[int] = []
        created_fact_ids: list[int] = []
        created_match_ids: list[int] = []

        try:
            with Session(app_engine) as session:
                sheet = CensusSheet(
                    slovakiana_key="cair-fstest01",
                    census_year=1930,
                    district="Trnava",
                    municipality="Horný Čepeň",
                    municipality_folded="horny cepen",
                    house_number="21",
                    house_number_int=21,
                    sheet_type="household",
                    download_status="extracted",
                )
                session.add(sheet)
                session.commit()
                session.refresh(sheet)
                created_sheet_id = sheet.id

                row1 = RecordRow(scan_id=0, row_index=1, y_start=0, y_end=50, status="extracted", record_type="census")
                row2 = RecordRow(scan_id=0, row_index=2, y_start=50, y_end=100, status="extracted", record_type="census")
                row3 = RecordRow(scan_id=0, row_index=3, y_start=100, y_end=150, status="extracted", record_type="census")
                session.add_all([row1, row2, row3])
                session.commit()
                for r in (row1, row2, row3):
                    session.refresh(r)
                    if r.id:
                        created_row_ids.append(r.id)

                f_matched = PersonFact(
                    row_id=row1.id or 0,
                    scan_id=0,
                    record_type="census",
                    census_sheet_id=sheet.id,
                    person_row_number="1",
                    event_year=1930,
                    birth_year_est=1890,
                    given_name_raw="Ján",
                    given_name_sk="Ján",
                    surname_raw="Kováč",
                    surname_normalized="Kováč",
                    village_raw="Horný Čepeň",
                    village_sk="Horný Čepeň",
                )
                f_unmatched_none = PersonFact(
                    row_id=row2.id or 0,
                    scan_id=0,
                    record_type="census",
                    census_sheet_id=sheet.id,
                    person_row_number="2",
                    event_year=1930,
                    birth_year_est=1895,
                    given_name_raw="Mária",
                    given_name_sk="Mária",
                    surname_raw="Kováčová",
                    surname_normalized="Kováč",
                    village_raw="Horný Čepeň",
                    village_sk="Horný Čepeň",
                )
                f_unmatched_empty = PersonFact(
                    row_id=row3.id or 0,
                    scan_id=0,
                    record_type="census",
                    census_sheet_id=sheet.id,
                    person_row_number="3",
                    event_year=1930,
                    birth_year_est=1920,
                    given_name_raw="Štefan",
                    given_name_sk="Štefan",
                    surname_raw="Kováč",
                    surname_normalized="Kováč",
                    village_raw="Horný Čepeň",
                    village_sk="Horný Čepeň",
                )
                session.add_all([f_matched, f_unmatched_none, f_unmatched_empty])
                session.commit()
                for f in (f_matched, f_unmatched_none, f_unmatched_empty):
                    session.refresh(f)
                    if f.id:
                        created_fact_ids.append(f.id)
                f_matched_id = f_matched.id or 0
                f_unmatched_none_id = f_unmatched_none.id or 0
                f_unmatched_empty_id = f_unmatched_empty.id or 0

                m_valid = FamilySearchMatch(
                    person_fact_id=f_matched_id,
                    row_id=row1.id or 0,
                    match_status="confirmed_match",
                    fs_person_id="KWCJ-4N2",
                    confidence_score=0.93,
                    summary_sk="Zhoda vo FamilySearch: Ján Kováč (KWCJ-4N2)",
                    reviewed_by_user=False,
                )
                m_null = FamilySearchMatch(
                    person_fact_id=f_unmatched_none_id,
                    row_id=row2.id or 0,
                    match_status="new_person",
                    fs_person_id=None,
                    confidence_score=0.90,
                    summary_sk="Nová osoba – bez zhody",
                    reviewed_by_user=False,
                )
                m_empty = FamilySearchMatch(
                    person_fact_id=f_unmatched_empty_id,
                    row_id=row3.id or 0,
                    match_status="existing_needs_facts",
                    fs_person_id="   ",
                    confidence_score=0.88,
                    summary_sk="Prázdne ID",
                    reviewed_by_user=False,
                )
                session.add_all([m_valid, m_null, m_empty])
                session.commit()
                for m in (m_valid, m_null, m_empty):
                    session.refresh(m)
                    if m.id:
                        created_match_ids.append(m.id)

            # 1. Verify /api/familysearch/matches only returns matched rows (fs_person_id="KWCJ-4N2")
            r_matches = client.get("/api/familysearch/matches")
            self.assertEqual(r_matches.status_code, 200)
            matches_data = r_matches.json()["matches"]
            returned_fact_ids = {item["fact_id"] for item in matches_data}
            self.assertIn(f_matched_id, returned_fact_ids)
            self.assertNotIn(f_unmatched_none_id, returned_fact_ids)
            self.assertNotIn(f_unmatched_empty_id, returned_fact_ids)
            for item in matches_data:
                self.assertTrue(bool((item.get("fs_person_id") or "").strip()))

            # 2. Verify /api/status counts only matched rows
            with patch("backend.app.main.ai_client.get_models_status", new=AsyncMock(return_value={"online": False})):
                r_status = client.get("/api/status")
            self.assertEqual(r_status.status_code, 200)
            counts = r_status.json()["counts"]
            self.assertEqual(counts["new_to_familysearch"], len(matches_data))
            self.assertEqual(counts["fs_pending_matches"], sum(1 for x in matches_data if not x.get("reviewed_by_user")))

            # 3. Verify POST /api/census/sheets/{sheet_ref}/persons automatically triggers trigger_auto_fs_match_for_fact
            with patch(
                "backend.app.mcp.familysearch_client.trigger_auto_fs_match_for_fact",
                new_callable=AsyncMock,
            ) as mock_trigger:
                r_add = client.post(
                    f"/api/census/sheets/{created_sheet_id}/persons",
                    json={
                        "given_name_raw": "Anna",
                        "surname_raw": "Kováčová",
                        "relation_to_head": "dcéra",
                        "sex": "F",
                        "birth_or_event_date": "10. 5. 1924",
                        "birth_year_est": 1924,
                    },
                )
                self.assertEqual(r_add.status_code, 200)
                added_fact = r_add.json()["fact"]
                new_fact_id = added_fact["id"]
                created_fact_ids.append(new_fact_id)
                if added_fact.get("row_id"):
                    created_row_ids.append(added_fact["row_id"])
                mock_trigger.assert_called_once_with(new_fact_id)

            # 4. Verify /api/familysearch/progress and /api/familysearch/status expose progress fields
            with patch(
                "backend.app.mcp.familysearch_client.fs_mcp_manager.call_tool",
                new=AsyncMock(return_value={"authenticated": False, "places": []}),
            ):
                r_fs_status = client.get("/api/familysearch/status")
                self.assertEqual(r_fs_status.status_code, 200)
                fs_status_json = r_fs_status.json()
                self.assertIn("progress", fs_status_json)
                for key in (
                    "running",
                    "mode",
                    "processed",
                    "total",
                    "matched_count",
                    "current_person",
                    "total_facts_in_db",
                    "total_matched_in_db",
                    "reviewed_count",
                    "last_finished_at",
                    "error",
                ):
                    self.assertIn(key, fs_status_json["progress"])

                r_fs_prog = client.get("/api/familysearch/progress")
                self.assertEqual(r_fs_prog.status_code, 200)
                self.assertGreaterEqual(r_fs_prog.json()["total_facts_in_db"], 4)
                self.assertGreaterEqual(r_fs_prog.json()["total_matched_in_db"], 1)

            # 5. Verify FamilySearch sync endpoints are disabled when token is missing or invalid
            with patch(
                "backend.app.mcp.familysearch_client.read_fs_env",
                return_value={"FS_ACCESS_TOKEN": "", "FS_ENVIRONMENT": "production"},
            ):
                from backend.app.mcp.familysearch_client import fs_mcp_manager

                fs_mcp_manager._cached_auth_status = None
                fs_mcp_manager._cached_auth_ts = 0.0
                r_no_tok = client.get("/api/familysearch/status")
                self.assertEqual(r_no_tok.status_code, 200)
                no_tok_json = r_no_tok.json()
                self.assertFalse(no_tok_json["authenticated"])
                self.assertFalse(no_tok_json["token_valid"])
                self.assertFalse(no_tok_json["sync_enabled"])
                self.assertFalse(no_tok_json["has_token_configured"])

                r_rematch_no_tok = client.post("/api/familysearch/rematch")
                self.assertEqual(r_rematch_no_tok.status_code, 400)
                self.assertIn("token", r_rematch_no_tok.json()["detail"].lower())

                r_refresh_no_tok = client.post(f"/api/familysearch/matches/{f_matched_id}/refresh")
                self.assertEqual(r_refresh_no_tok.status_code, 400)
                self.assertIn("token", r_refresh_no_tok.json()["detail"].lower())

            # 6. Verify FamilySearch sync endpoints are disabled when token is inserted but rejected (invalid/expired)
            with (
                patch(
                    "backend.app.mcp.familysearch_client.read_fs_env",
                    return_value={"FS_ACCESS_TOKEN": "expired-token-123", "FS_ENVIRONMENT": "production"},
                ),
                patch(
                    "backend.app.mcp.familysearch_client.fs_mcp_manager.call_tool",
                    new=AsyncMock(
                        return_value={
                            "authenticated": True,
                            "token_accepted": False,
                            "token_problem": "FamilySearch rejected the access token (401).",
                        }
                    ),
                ),
            ):
                from backend.app.mcp.familysearch_client import fs_mcp_manager

                fs_mcp_manager._cached_auth_status = None
                fs_mcp_manager._cached_auth_ts = 0.0
                r_bad_tok = client.get("/api/familysearch/status")
                self.assertEqual(r_bad_tok.status_code, 200)
                bad_tok_json = r_bad_tok.json()
                self.assertTrue(bad_tok_json["has_token_configured"])
                self.assertFalse(bad_tok_json["authenticated"])
                self.assertFalse(bad_tok_json["token_valid"])
                self.assertFalse(bad_tok_json["sync_enabled"])
                self.assertIn("rejected", (bad_tok_json.get("token_problem") or "").lower())

                r_rematch_bad_tok = client.post("/api/familysearch/rematch")
                self.assertEqual(r_rematch_bad_tok.status_code, 400)
                self.assertIn("neplatn", r_rematch_bad_tok.json()["detail"].lower())

            # 7. Verify FamilySearch status reports sync_enabled=True when token is accepted
            with (
                patch(
                    "backend.app.mcp.familysearch_client.read_fs_env",
                    return_value={"FS_ACCESS_TOKEN": "valid-token-xyz", "FS_ENVIRONMENT": "production"},
                ),
                patch(
                    "backend.app.mcp.familysearch_client.fs_mcp_manager.call_tool",
                    new=AsyncMock(
                        return_value={
                            "authenticated": True,
                            "token_accepted": True,
                        }
                    ),
                ),
            ):
                from backend.app.mcp.familysearch_client import fs_mcp_manager

                fs_mcp_manager._cached_auth_status = None
                fs_mcp_manager._cached_auth_ts = 0.0
                r_ok_tok = client.get("/api/familysearch/status")
                self.assertEqual(r_ok_tok.status_code, 200)
                ok_tok_json = r_ok_tok.json()
                self.assertTrue(ok_tok_json["has_token_configured"])
                self.assertTrue(ok_tok_json["authenticated"])
                self.assertTrue(ok_tok_json["token_valid"])
                self.assertTrue(ok_tok_json["sync_enabled"])
                fs_mcp_manager._cached_auth_status = None
                fs_mcp_manager._cached_auth_ts = 0.0
        finally:
            from backend.app.db.session import delete_person_fact_cascade
            from sqlmodel import select as _sel

            with Session(app_engine) as cleanup_session:
                for mid in created_match_ids:
                    m_obj = cleanup_session.get(FamilySearchMatch, mid)
                    if m_obj:
                        cleanup_session.delete(m_obj)
                cleanup_session.commit()
                for fid in created_fact_ids:
                    delete_person_fact_cascade(cleanup_session, fid)
                for rid in created_row_ids:
                    r_obj = cleanup_session.get(RecordRow, rid)
                    if r_obj:
                        cleanup_session.delete(r_obj)
                if created_sheet_id:
                    from backend.app.db.models import CensusHousehold

                    for hh_obj in cleanup_session.exec(
                        _sel(CensusHousehold).where(CensusHousehold.census_sheet_id == created_sheet_id)
                    ).all():
                        cleanup_session.delete(hh_obj)
                    s_obj = cleanup_session.get(CensusSheet, created_sheet_id)
                    if s_obj:
                        cleanup_session.delete(s_obj)
                cleanup_session.commit()

    def test_multipage_overview_sheet_extracts_all_pages_without_overwriting(self) -> None:
        import asyncio
        from unittest.mock import AsyncMock, patch
        from sqlmodel import select
        from backend.app.api.census import get_census_sheet_detail
        from backend.app.db.models import CensusOverviewHouse, RecordRow, ScanImage
        from backend.app.pipeline.extraction_stage import extract_census_overview_sheet

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            crops_dir = tmp_path / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            # Build two synthetic two-page bifolio spread images (2400x1600):
            # Front spread (_v): Page 1 on right half (houses 1..4), Page 4 on left half (houses 11..12)
            # Back spread (_r): Page 2 on left half (houses 5..7), Page 3 on right half (houses 8..10)
            w, h = 2400, 1600
            front_img = np.full((h, w, 3), 245, dtype=np.uint8)
            back_img = np.full((h, w, 3), 245, dtype=np.uint8)

            def draw_subpage_table(
                canvas: np.ndarray,
                x0: int,
                x1: int,
                start_y: int,
                n_rows: int,
                populated_rows: int,
                house_start: int,
            ) -> None:
                step = 56
                # Draw thick rubric bottom border at start_y and lighter 2px row lines below
                cv2.line(canvas, (x0 + 30, start_y), (x1 - 30, start_y), (15, 15, 15), 7)
                for r_i in range(1, n_rows + 1):
                    y_line = start_y + r_i * step
                    cv2.line(canvas, (x0 + 30, y_line), (x1 - 30, y_line), (85, 85, 85), 2)
                for r_i in range(populated_rows):
                    ry = start_y + r_i * step + 36
                    hn = house_start + r_i
                    # Write owner name specifically inside Column 6 (0.42..0.66 of page width)
                    page_w = x1 - x0
                    cv2.putText(
                        canvas,
                        f"{hn}.  TestObec   {hn}",
                        (x0 + int(page_w * 0.06), ry),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (10, 10, 10),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        f"Jan Majitel {hn}",
                        (x0 + int(page_w * 0.42), ry),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (10, 10, 10),
                        3,
                    )
                    cv2.putText(
                        canvas,
                        "1   5",
                        (x0 + int(page_w * 0.74), ry),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (10, 10, 10),
                        2,
                    )

            # Page 1 (front right half: y=560, 6 grid rows, 4 populated houses 1..4)
            draw_subpage_table(front_img, int(w * 0.50), w, 560, 6, 4, 1)
            # Page 4 (front left half: y=220, 6 grid rows, 2 populated houses 11..12)
            draw_subpage_table(front_img, 0, int(w * 0.50), 220, 6, 2, 11)
            # Page 2 (back left half: y=220, 5 grid rows, 3 populated houses 5..7)
            draw_subpage_table(back_img, 0, int(w * 0.50), 220, 5, 3, 5)
            # Page 3 (back right half: y=220, 5 grid rows, 3 populated houses 8..10)
            draw_subpage_table(back_img, int(w * 0.50), w, 220, 5, 3, 8)

            front_path = tmp_path / "ov_front.jpg"
            back_path = tmp_path / "ov_back.jpg"
            cv2.imwrite(str(front_path), front_img)
            cv2.imwrite(str(back_path), back_img)

            engine = create_engine("sqlite:///:memory:")
            SQLModel.metadata.create_all(engine)
            ensure_fts_schema(engine)

            with Session(engine) as session:
                sheet = CensusSheet(
                    slovakiana_key="cair-ovmulti01",
                    census_year=1930,
                    district="Topoľčany",
                    municipality="TestObec",
                    municipality_folded="testobec",
                    sheet_type="overview",
                    download_status="downloaded",
                )
                session.add(sheet)
                session.commit()
                session.refresh(sheet)

                front_scan = ScanImage(
                    census_sheet_id=sheet.id,
                    page_side="front",
                    filename="ov_front.jpg",
                    original_path=str(front_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                back_scan = ScanImage(
                    census_sheet_id=sheet.id,
                    page_side="back",
                    filename="ov_back.jpg",
                    original_path=str(back_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                session.add_all([front_scan, back_scan])
                session.commit()
                session.refresh(front_scan)
                session.refresh(back_scan)

                sheet.front_scan_id = front_scan.id
                sheet.back_scan_id = back_scan.id
                session.add(sheet)
                session.commit()

                async def fake_chat_completion(**kwargs: object) -> dict[str, object]:
                    stage = kwargs.get("stage")
                    scan_id = kwargs.get("scan_id")
                    img_rel = str(kwargs.get("image_rel_url") or "")
                    if stage == "ocr_header":
                        return {
                            "content": "Prehľad sčítacieho obvodu | Obvod č. I | Meno sčítacieho komisára: Jozef Testovací",
                            "parsed": {},
                        }
                    m_row = np.frompyfunc(lambda s: s, 1, 1)(img_rel)
                    import re as _re

                    rm = _re.search(r"row_(\d+)_left", str(m_row))
                    r_idx = int(rm.group(1)) if rm else 1
                    if scan_id == front_scan.id:
                        # Front scan has 4 rows on Page 1 (1..4) and 2 rows on Page 4 (11..12)
                        hn = r_idx if r_idx <= 4 else (10 + (r_idx - 4))
                    else:
                        # Back scan has 3 rows on Page 2 (5..7) and 3 rows on Page 3 (8..10)
                        hn = 4 + r_idx
                    return {
                        "content": f"<table><tr><td>{hn}.</td><td>TestObec</td><td>{hn}</td><td>1</td><td>Ján Majiteľ {hn}</td><td>1</td><td>{3 + (hn % 5)}</td></tr></table>",
                        "parsed": {},
                    }

                with (
                    patch.object(
                        extract_census_overview_sheet.__globals__["settings"],
                        "crops_dir",
                        crops_dir,
                    ),
                    patch(
                        "backend.app.pipeline.ocr_stage.settings.crops_dir",
                        crops_dir,
                    ),
                    patch.object(
                        extract_census_overview_sheet.__globals__["ai_client"],
                        "chat_completion",
                        side_effect=fake_chat_completion,
                    ),
                ):
                    houses = asyncio.run(
                        extract_census_overview_sheet(session, sheet, front_scan, back_scan)
                    )
                    detail = asyncio.run(get_census_sheet_detail("cair-ovmulti01", session))

                self.assertEqual(len(houses), 12)
                house_numbers = [int(h.house_number or 0) for h in houses]
                self.assertEqual(house_numbers, list(range(1, 13)))
                self.assertEqual(sheet.commissioner_name, "Jozef Testovací")

                front_houses = [h for h in houses if h.scan_id == front_scan.id]
                back_houses = [h for h in houses if h.scan_id == back_scan.id]
                self.assertEqual(len(front_houses), 6)  # 4 on Page 1 + 2 on Page 4
                self.assertEqual(len(back_houses), 6)   # 3 on Page 2 + 3 on Page 3

                front_rows = session.exec(select(RecordRow).where(RecordRow.scan_id == front_scan.id)).all()
                back_rows = session.exec(select(RecordRow).where(RecordRow.scan_id == back_scan.id)).all()
                self.assertEqual(len(front_rows), 6)
                self.assertEqual(len(back_rows), 6)

                self.assertEqual(len(detail["pages"]), 2)
                self.assertEqual(len(detail["overview_houses"]), 12)
                booklet_pages = [oh["booklet_page"] for oh in detail["overview_houses"]]
                self.assertEqual(booklet_pages, [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])

    def test_batch_worker_processes_only_not_done_sheets_and_harvests_when_needed(self) -> None:
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch
        from backend.app.census.worker import (
            _find_matching_census_sheets,
            run_download_and_pipeline_job,
        )
        from backend.app.db.models import Batch, CensusDownloadJob, RecordRow, ScanImage
        from backend.app.pipeline.orchestrator import process_batch_stage_queued

        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            # Seed 3 already-done sheets and 2 undone sheets in local DB
            s_done_1 = CensusSheet(
                slovakiana_key="cair-b001",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key="2742",
                municipality_folded="velke ripnany",
                house_number="1",
                house_number_int=1,
                download_status="matched",
            )
            s_done_2 = CensusSheet(
                slovakiana_key="cair-b002",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key="2742",
                municipality_folded="velke ripnany",
                house_number="2",
                house_number_int=2,
                download_status="extracted",
            )
            s_done_3 = CensusSheet(
                slovakiana_key="cair-b003",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key="2742",
                municipality_folded="velke ripnany",
                house_number="3",
                house_number_int=3,
                download_status="ocr_done",
            )
            s_todo_4 = CensusSheet(
                slovakiana_key="cair-b004",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key=None,  # Even with None municipality_key, name matching must find it
                municipality_folded="velke ripnany",
                house_number="4",
                house_number_int=4,
                download_status="cataloged",
            )
            s_todo_5 = CensusSheet(
                slovakiana_key="cair-b005",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key="2742",
                municipality_folded="velke ripnany",
                house_number="5",
                house_number_int=5,
                download_status="downloaded",
            )
            session.add_all([s_done_1, s_done_2, s_done_3, s_todo_4, s_todo_5])
            session.commit()

            filters = {
                "years": [1930],
                "category_keys": ["2742"],
                "municipality": "Veľké Ripňany",
                "search_text": "Veľké Ripňany",
                "include_blackened": True,
                "limit_sheets": 3,
                "auto_ocr": True,
                "run_mcp": True,
                "only_undone": True,
            }
            undone = _find_matching_census_sheets(session, filters, only_undone=True, run_ocr=True)
            self.assertEqual([s.slovakiana_key for s in undone], ["cair-b005", "cair-b004"])

            job = CensusDownloadJob(
                job_type="full_pipeline",
                filter_json=json.dumps(filters, ensure_ascii=False),
                status="running",
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id or 1

        processed_sheet_ids: list[int] = []

        async def fake_download_and_prepare(
            sheet_id: int,
            *,
            run_pipeline: bool = True,
            run_mcp: bool = True,
            max_rows: int | None = None,
        ) -> dict[str, object]:
            processed_sheet_ids.append(sheet_id)
            with Session(test_engine) as s_inner:
                sh = s_inner.get(CensusSheet, sheet_id)
                if sh:
                    sh.download_status = "matched"
                    s_inner.add(sh)
                    s_inner.commit()
            return {"sheet_id": sheet_id, "status": "matched"}

        remote_items = [
            {
                "slovakiana_key": f"cair-b00{i}",
                "title_raw": f"Sčítací hárok 283/{i}, Veľké Ripňany (Topoľčany)",
                "census_year": 1930,
                "district": "Topoľčany",
                "municipality": "Veľké Ripňany",
                "municipality_folded": "velke ripnany",
                "house_number": str(i),
                "house_number_int": i,
                "is_blackened": False,
                "has_images": True,
            }
            for i in range(1, 8)
        ]

        with (
            patch("backend.app.census.worker.engine", test_engine),
            patch(
                "backend.app.census.worker.slovakiana_client.search_sheets",
                new=AsyncMock(
                    return_value={
                        "total": 7,
                        "page": 0,
                        "size": 200,
                        "items": remote_items,
                        "raw_count": 7,
                    }
                ),
            ) as mock_search,
            patch(
                "backend.app.census.worker.slovakiana_client.download_thumbnails_batch",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "backend.app.census.worker.download_and_prepare_or_process",
                side_effect=fake_download_and_prepare,
            ),
        ):
            # Batch 1 (limit_sheets=3): DB only had 2 undone (b004, b005), so worker harvests
            # remote catalog (discovering b006, b007 while keeping b001..b003 as done) and processes 3 undone sheets!
            res1 = asyncio.run(run_download_and_pipeline_job(job_id))
            self.assertEqual(res1["processed"], 3)
            self.assertEqual(res1["failed"], 0)
            self.assertGreaterEqual(mock_search.call_count, 1)

            with Session(test_engine) as session:
                processed_keys_1 = [
                    session.get(CensusSheet, sid).slovakiana_key  # type: ignore[union-attr]
                    for sid in processed_sheet_ids
                ]
                self.assertEqual(processed_keys_1, ["cair-b005", "cair-b004", "cair-b006"])
                remaining_undone = _find_matching_census_sheets(session, filters, only_undone=True, run_ocr=True)
                self.assertEqual([s.slovakiana_key for s in remaining_undone], ["cair-b007"])

                # Batch 2 (limit_sheets=3): Only cair-b007 is left undone
                job2 = CensusDownloadJob(
                    job_type="full_pipeline",
                    filter_json=json.dumps(filters, ensure_ascii=False),
                    status="running",
                )
                session.add(job2)
                session.commit()
                session.refresh(job2)
                job2_id = job2.id or 2

            processed_sheet_ids.clear()
            res2 = asyncio.run(run_download_and_pipeline_job(job2_id))
            self.assertEqual(res2["processed"], 1)
            with Session(test_engine) as session:
                processed_keys_2 = [
                    session.get(CensusSheet, sid).slovakiana_key  # type: ignore[union-attr]
                    for sid in processed_sheet_ids
                ]
                self.assertEqual(processed_keys_2, ["cair-b007"])

                # Batch 3: All 7 sheets are now done -> 0 processed, 7 skipped_done
                job3 = CensusDownloadJob(
                    job_type="full_pipeline",
                    filter_json=json.dumps(filters, ensure_ascii=False),
                    status="running",
                )
                session.add(job3)
                session.commit()
                session.refresh(job3)
                job3_id = job3.id or 3

            processed_sheet_ids.clear()
            res3 = asyncio.run(run_download_and_pipeline_job(job3_id))
            self.assertEqual(res3["processed"], 0)
            self.assertEqual(res3["skipped_done"], 7)
            self.assertEqual(processed_sheet_ids, [])

        # Also verify process_batch_stage_queued skips already-extracted/matched scans in a Batch
        with Session(test_engine) as session:
            batch = Batch(name="Test Undone Batch", status="uploaded", total_scans=2)
            session.add(batch)
            session.commit()
            session.refresh(batch)

            sc_done = ScanImage(
                batch_id=batch.id,
                filename="done_scan.jpg",
                original_path="/tmp/done_scan.jpg",
                detected_type="census",
                status="matched",
            )
            sc_todo = ScanImage(
                batch_id=batch.id,
                filename="todo_scan.jpg",
                original_path="/tmp/todo_scan.jpg",
                detected_type="census",
                status="sliced",
                header_raw_ocr="Header OCR",
            )
            session.add_all([sc_done, sc_todo])
            session.commit()
            session.refresh(sc_done)
            session.refresh(sc_todo)
            sc_todo_id = sc_todo.id

            r_done = RecordRow(scan_id=sc_done.id or 0, row_index=1, status="extracted", raw_ocr_combined="done")
            r_todo = RecordRow(scan_id=sc_todo_id or 0, row_index=1, status="sliced")
            session.add_all([r_done, r_todo])
            session.commit()
            batch_id = batch.id or 1

        ocr_called_scan_ids: list[int] = []

        async def fake_run_ocr_for_row(session_arg: Session, scan_arg: ScanImage, row_arg: RecordRow) -> None:
            ocr_called_scan_ids.append(scan_arg.id or 0)
            row_arg.raw_ocr_combined = "1 | Kováč | Ján"
            row_arg.status = "ocr_done"

        async def fake_census_extract(
            session_arg: Session,
            scan_arg: ScanImage,
            row_arg: RecordRow,
            prev_fact: PersonFact | None = None,
            **kwargs: object,
        ) -> PersonFact:
            pf = PersonFact(
                row_id=row_arg.id or 0,
                scan_id=scan_arg.id or 0,
                record_type="census",
                given_name_raw="Ján",
                surname_raw="Kováč",
            )
            session_arg.add(pf)
            session_arg.commit()
            session_arg.refresh(pf)
            return pf

        with (
            patch("backend.app.pipeline.orchestrator.engine", test_engine),
            patch("backend.app.pipeline.orchestrator.run_ocr_for_row", side_effect=fake_run_ocr_for_row),
            patch("backend.app.pipeline.orchestrator.run_census_extraction_for_row", side_effect=fake_census_extract),
        ):
            batch_res = asyncio.run(
                process_batch_stage_queued(batch_id, run_mcp=False, skip_row_ocr=False)
            )
            self.assertEqual(batch_res["scans_processed"], 1)
            self.assertEqual(batch_res["scans_skipped"], 1)
            self.assertEqual(ocr_called_scan_ids, [sc_todo_id])

    def test_skip_row_ocr_direct_visual_extract_with_5_crop_composite_and_header_ocr(self) -> None:
        import asyncio
        from unittest.mock import patch
        from PIL import Image
        from sqlmodel import select
        from backend.app.config import settings
        from backend.app.db.models import RecordRow, ScanImage
        from backend.app.pipeline.ai_client import ai_client
        from backend.app.pipeline.ocr_stage import resolve_row_extraction_crop
        from backend.app.pipeline.orchestrator import process_single_scan_pipeline

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            settings, "crops_dir", Path(tmp_dir) / "crops"
        ), patch(
            "backend.app.pipeline.ocr_stage.settings.crops_dir", Path(tmp_dir) / "crops"
        ):
            tmp_path = Path(tmp_dir)
            img = np.full((1600, 2400, 3), 240, dtype=np.uint8)
            for i in range(5):
                y = 440 + i * 95
                cv2.line(img, (60, y), (2340, y), (25, 25, 25), 3)
                if 1 <= i <= 2:
                    cv2.putText(
                        img,
                        f"{i} Kovac Jan hlava 1892",
                        (140, y - 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (15, 15, 15),
                        2,
                    )
            scan_path = tmp_path / "cair-direct01_front.jpg"
            cv2.imwrite(str(scan_path), img)

            test_engine = create_engine("sqlite:///:memory:")
            SQLModel.metadata.create_all(test_engine)
            ensure_fts_schema(test_engine)

            with Session(test_engine) as session:
                sheet = CensusSheet(
                    slovakiana_key="cair-direct01",
                    census_year=1930,
                    district="Trnava",
                    municipality="Horné Zelenice",
                    house_number="12",
                    sheet_type="household",
                )
                session.add(sheet)
                session.commit()
                session.refresh(sheet)

                scan = ScanImage(
                    census_sheet_id=sheet.id,
                    page_side="front",
                    filename="cair-direct01_front.jpg",
                    original_path=str(scan_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                session.add(scan)
                session.commit()
                session.refresh(scan)
                scan_id = scan.id or 1

            recorded_calls: list[dict[str, object]] = []

            async def fake_chat_completion(**kwargs: object) -> dict[str, object]:
                recorded_calls.append(dict(kwargs))
                stage = str(kwargs.get("stage") or "")
                if stage == "ocr_header":
                    return {
                        "content": "Sčítací hárok | Obec Horné Zelenice | Dom č. 12",
                        "parsed": {},
                    }
                row_id_val = int(kwargs.get("row_id") or 1)
                if row_id_val == 1:
                    return {
                        "content": "{}",
                        "parsed": {
                            "apartment_number": "1",
                            "person_row_number": "1",
                            "surname_raw": "Kováč",
                            "given_name_raw": "Ján",
                            "relation_to_head": "hlava domácnosti",
                            "sex": "M",
                            "birth_date_raw": "14. III. 1892",
                            "birth_year": 1892,
                            "marital_status": "ženatý",
                            "birth_place_municipality": "Horné Zelenice",
                            "birth_place_district": "Hlohovec",
                            "birth_place_country": "Slovensko",
                            "religion": "rím. kat.",
                            "occupation": "roľník",
                        },
                    }
                return {
                    "content": "{}",
                    "parsed": {
                        "apartment_number": "1",
                        "person_row_number": "2",
                        "surname_raw": "Kováčová",
                        "maiden_surname": "Vargová",
                        "given_name_raw": "Mária",
                        "relation_to_head": "manželka",
                        "sex": "F",
                        "birth_date_raw": "5. V. 1896",
                        "birth_year": 1896,
                        "marital_status": "vydatá",
                        "birth_place_municipality": "//",
                        "birth_place_district": "//",
                        "birth_place_country": "//",
                        "religion": "//",
                        "occupation": "v domácnosti",
                    },
                }

            with (
                patch("backend.app.pipeline.orchestrator.engine", test_engine),
                patch.object(ai_client, "chat_completion", side_effect=fake_chat_completion),
            ):
                res = asyncio.run(
                    process_single_scan_pipeline(
                        scan_id,
                        run_mcp=False,
                        finalize_idle=False,
                        skip_row_ocr=True,
                    )
                )

            self.assertEqual(res["persons_count"], 2)
            stages_called = [str(c.get("stage")) for c in recorded_calls]
            # Header OCR runs with glm-ocr, while row OCR (ocr_left / ocr_right) is completely skipped!
            self.assertIn("ocr_header", stages_called)
            self.assertNotIn("ocr_left", stages_called)
            self.assertNotIn("ocr_right", stages_called)

            extract_calls = [c for c in recorded_calls if c.get("stage") == "extract"]
            self.assertEqual(len(extract_calls), 2)
            for ec in extract_calls:
                prompt_str = str(ec.get("prompt") or "")
                self.assertNotIn("Pre-parsed hints:", prompt_str)
                self.assertNotIn("OCR text:", prompt_str)
                img_p = ec.get("image_path")
                self.assertIsNotNone(img_p)
                self.assertTrue(str(img_p).endswith("_extract_composite.jpg"))
                self.assertTrue(Path(str(img_p)).exists())

            with Session(test_engine) as session:
                rows = list(
                    session.exec(
                        select(RecordRow)
                        .where(RecordRow.scan_id == scan_id)
                        .order_by(RecordRow.row_index.asc())
                    ).all()
                )
                self.assertEqual(len(rows), 2)
                comp_p, _ = resolve_row_extraction_crop(rows[0])
                self.assertIsNotNone(comp_p)
                with Image.open(str(comp_p)) as comp_im:
                    # 5 stacked bands + 5 banners -> height > 250px
                    self.assertGreater(comp_im.height, 250)
                facts = list(
                    session.exec(
                        select(PersonFact)
                        .where(PersonFact.scan_id == scan_id)
                        .order_by(PersonFact.id.asc())
                    ).all()
                )
                self.assertEqual(len(facts), 2)
                self.assertEqual(facts[0].surname_raw, "Kováč")
                self.assertEqual(facts[1].surname_raw, "Kováčová")
                self.assertEqual(facts[1].mother_maiden_surname, "Vargová")
                self.assertEqual(facts[1].birth_place_municipality, "Horné Zelenice")

    def test_municipal_overview_sheet_detection_extraction_and_district_joining(self) -> None:
        import asyncio
        import re as _re
        from unittest.mock import patch
        from sqlmodel import select
        from backend.app.config import settings
        from backend.app.db.models import CensusMunicipalOverviewDistrict, ScanImage
        from backend.app.pipeline.ai_client import ai_client
        from backend.app.pipeline.census_layout import (
            detect_is_municipal_overview_sheet,
            detect_is_overview_sheet,
        )
        from backend.app.pipeline.extraction_stage import extract_census_overview_sheet

        obecny_header = (
            "Sčítanie ľudu 1930\nPolitický okres Hlohovec\nObec Hornie Zelenice\n"
            "Meno sčítacieho revízora Františel Mallý\nObecný prehľad\n"
            "vyplní sčítací revízor, ale len v obciach rozdelených na dva alebo viacej sčítacích obvodov.\n"
            "Na koniec vypočíta súčet pre celú obec."
        )
        district_header = (
            "Sčítanie ľudu 1930\nPolitický okres Hlohovec\nČíslo sčítacieho obvodu I.\n"
            "Obec Horné Zelenice\nMeno sčítacieho komisára Eduard Zoltický\n"
            "Prehľad sčítacieho obvodu\nRadové číslo osád | Meno osady | Popisné číslo domu | Meno majiteľa domu"
        )
        self.assertTrue(detect_is_municipal_overview_sheet(obecny_header))
        self.assertFalse(detect_is_overview_sheet(obecny_header))
        self.assertFalse(detect_is_municipal_overview_sheet(district_header))
        self.assertTrue(detect_is_overview_sheet(district_header))

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            settings, "crops_dir", Path(tmp_dir) / "crops"
        ), patch(
            "backend.app.pipeline.ocr_stage.settings.crops_dir", Path(tmp_dir) / "crops"
        ):
            tmp_path = Path(tmp_dir)
            h, w = 1400, 1000
            front_img = np.full((h, w, 3), 240, dtype=np.uint8)
            for i in range(6):
                y = 480 + i * 50
                cv2.line(front_img, (40, y), (w - 40, y), (20, 20, 20), 6 if i == 0 else 2)
                if 1 <= i <= 4:
                    for col_x in (60, 180, 320, 440, 520, 620, 740):
                        cv2.putText(
                            front_img,
                            f"Obvod {i}",
                            (col_x, y - 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (15, 15, 15),
                            2,
                        )
            front_path = tmp_path / "cair-ko2a0od_front.jpg"
            cv2.imwrite(str(front_path), front_img)

            test_engine = create_engine("sqlite:///:memory:")
            SQLModel.metadata.create_all(test_engine)
            ensure_fts_schema(test_engine)

            with Session(test_engine) as session:
                sheet = CensusSheet(
                    slovakiana_key="cair-ko2a0od",
                    census_year=1930,
                    district="Hlohovec",
                    district_key="715",
                    municipality="Horné Zelenice",
                    municipality_key="7647",
                    municipality_folded="horne zelenice",
                    box_number="1196",
                    sheet_number="0108",
                    sheet_type="overview",  # Even if misclassified as overview initially!
                    download_status="downloaded",
                )
                ov1 = CensusSheet(
                    slovakiana_key="cair-ko270vr",
                    census_year=1930,
                    district="Hlohovec",
                    municipality="Horné Zelenice",
                    municipality_folded="horne zelenice",
                    box_number="1196",
                    sheet_number="0074",
                    sheet_type="overview",
                )
                ov2 = CensusSheet(
                    slovakiana_key="cair-ko270vs",
                    census_year=1930,
                    district="Hlohovec",
                    municipality="Horné Zelenice",
                    municipality_folded="horne zelenice",
                    box_number="1196",
                    sheet_number="0075",
                    sheet_type="overview",
                )
                session.add_all([sheet, ov1, ov2])
                session.commit()
                session.refresh(sheet)
                session.refresh(ov1)
                session.refresh(ov2)

                front_scan = ScanImage(
                    census_sheet_id=sheet.id,
                    page_side="front",
                    filename="cair-ko2a0od_front.jpg",
                    original_path=str(front_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                session.add(front_scan)
                session.commit()
                session.refresh(front_scan)
                sheet.front_scan_id = front_scan.id
                session.add(sheet)
                session.commit()

                row_ocrs = {
                    1: "1 | Homie Zeleuca | I. | Edward Lollicky | 55 | 52 | 3 | 81 | 420",
                    2: "11 | Auchal Rafajidus | 53 | 50 | 23 | 72 | 344",
                    3: "Spolu | 105 | 102 |  | 153 | 769",
                    4: "v Horných Zeleniciach, dňa 10. dec. 1930 | 761",
                }

                async def fake_chat_completion(**kwargs: object) -> dict[str, object]:
                    stage = str(kwargs.get("stage") or "")
                    if stage == "ocr_header":
                        return {"content": obecny_header}
                    img_rel = str(kwargs.get("image_rel_url") or "")
                    m_r = _re.search(r"row_0*(\d+)_", img_rel)
                    r_idx = int(m_r.group(1)) if m_r else 1
                    return {"content": row_ocrs.get(r_idx, "")}

                with patch.object(ai_client, "chat_completion", side_effect=fake_chat_completion):
                    houses = asyncio.run(extract_census_overview_sheet(session, sheet, front_scan, None))

                self.assertEqual(houses, [])
                session.refresh(sheet)
                self.assertEqual(sheet.sheet_type, "municipal_overview")
                self.assertEqual(sheet.commissioner_name, "František Mallý")

                districts = list(
                    session.exec(
                        select(CensusMunicipalOverviewDistrict)
                        .where(CensusMunicipalOverviewDistrict.census_sheet_id == sheet.id)
                        .order_by(CensusMunicipalOverviewDistrict.row_index.asc())
                    ).all()
                )
                self.assertEqual(len(districts), 3)
                d1, d2, d_sum = districts[0], districts[1], districts[2]
                self.assertEqual(d1.obvod_number, "I")
                self.assertEqual(d1.commissioner_name, "Eduard Zoltický")
                self.assertEqual((d1.houses_total, d1.houses_inhabited, d1.houses_uninhabited), (53, 52, 1))
                self.assertEqual((d1.household_count, d1.present_persons_count), (81, 420))
                self.assertEqual(d1.matched_overview_sheet_id, ov1.id)

                self.assertEqual(d2.obvod_number, "II")
                self.assertEqual(d2.commissioner_name, "Michal Rafajdus")
                self.assertEqual((d2.houses_total, d2.houses_inhabited, d2.houses_uninhabited), (52, 50, 2))
                self.assertEqual((d2.household_count, d2.present_persons_count), (72, 341))
                self.assertEqual(d2.matched_overview_sheet_id, ov2.id)

                self.assertTrue(d_sum.is_summary_row)
                self.assertEqual((d_sum.houses_total, d_sum.houses_inhabited, d_sum.houses_uninhabited), (105, 102, 3))
                self.assertEqual((d_sum.household_count, d_sum.present_persons_count), (153, 761))

                # Simulate duplicate rows from a concurrent run, a bogus revisor name 'Obecný prehl' on cair-ko2a0od,
                # ov2 ('cair-ko270vs') misclassified as 'municipal_overview' WITH bogus house rows ('Objvany', 'Obfvany', ...),
                # and ov1 ('cair-ko270vr') having raw OCR typo 'Eduard Lotticky', and verify get_municipality_overview &
                # get_census_sheet_detail self-heal all of them and reconstruct CensusOverviewHouse rows 55..58 on the fly.
                from backend.app.api.census import get_census_sheet_detail, get_municipality_overview
                from backend.app.pipeline.census_layout import (
                    classify_overview_rows_structure,
                    is_occupancy_status_token,
                    is_valid_commissioner_or_revisor_name,
                )

                for occ_tok in (
                    "Objvany",
                    "Obfvany",
                    "Objvány",
                    "Objrany",
                    "Objvarij",
                    "Objvary",
                    "Obijrany",
                    "Obfryany",
                    "Objrani",
                    "Objvranž",
                    "obývaný",
                ):
                    self.assertEqual(is_occupancy_status_token(occ_tok), (True, True), occ_tok)
                    self.assertFalse(is_valid_commissioner_or_revisor_name(occ_tok), occ_tok)
                self.assertEqual(is_occupancy_status_token("neobývaný"), (True, False))
                self.assertEqual(is_occupancy_status_token("Martin Vido"), (False, False))

                house_raw_rows = [
                    "55 | objvany | + | Martin Vido | 3 | 3",
                    "56 | obfvany | + | Ján Kováč | 16 | 16",
                    "57 | objvány | + | Jozef Varga | 8 | 8",
                    "58 | objrany | + | Štefan Horváth | 7 | 7",
                ]
                self.assertEqual(classify_overview_rows_structure(house_raw_rows), "overview")

                sheet.commissioner_name = "Obecný prehl"
                ov1.commissioner_name = "Eduard Lotticky"
                ov2.sheet_type = "municipal_overview"
                session.add_all([sheet, ov1, ov2])
                for dup_src in (d1, d2, d_sum):
                    dup_copy = CensusMunicipalOverviewDistrict(
                        census_sheet_id=dup_src.census_sheet_id,
                        scan_id=dup_src.scan_id,
                        row_index=dup_src.row_index,
                        census_year=dup_src.census_year,
                        district=dup_src.district,
                        municipality=dup_src.municipality,
                        ordinal_number=dup_src.ordinal_number,
                        settlement_name=dup_src.settlement_name,
                        obvod_number=dup_src.obvod_number,
                        commissioner_name=dup_src.commissioner_name,
                        houses_total=dup_src.houses_total,
                        houses_inhabited=dup_src.houses_inhabited,
                        houses_uninhabited=dup_src.houses_uninhabited,
                        household_count=dup_src.household_count,
                        present_persons_count=dup_src.present_persons_count,
                        is_summary_row=dup_src.is_summary_row,
                        matched_overview_sheet_id=dup_src.matched_overview_sheet_id,
                    )
                    session.add(dup_copy)

                # Seed bogus CensusMunicipalOverviewDistrict rows on ov2 ('cair-ko270vs') matching the user's screenshot
                bogus_ov2_rows = [
                    (1, "I", "Objvany", 55, 3, 3, "Martin Vido", house_raw_rows[0]),
                    (2, "II", "Obfvany", 56, 16, 16, "Ján Kováč", house_raw_rows[1]),
                    (3, "III", "Objvány", 57, 8, 8, "Jozef Varga", house_raw_rows[2]),
                    (4, "IV", "Objrany", 58, 7, 7, "Štefan Horváth", house_raw_rows[3]),
                ]
                for r_i, obv_n, bad_comm, h_tot, h_inh, h_uninh, rem_owner, raw_t in bogus_ov2_rows:
                    session.add(
                        CensusMunicipalOverviewDistrict(
                            census_sheet_id=ov2.id or 0,
                            row_index=r_i,
                            census_year=1930,
                            district="Hlohovec",
                            municipality="Horné Zelenice",
                            ordinal_number="1",
                            settlement_name="Horné Zelenice",
                            obvod_number=obv_n,
                            commissioner_name=bad_comm,
                            houses_total=h_tot,
                            houses_inhabited=h_inh,
                            houses_uninhabited=h_uninh,
                            remarks=rem_owner,
                            raw_ocr_text=raw_t,
                        )
                    )
                session.commit()

                muni_ov_res = get_municipality_overview(session, municipality="Horné Zelenice", census_year=1930)
                self.assertEqual(
                    [ms["slovakiana_id"] for ms in muni_ov_res["municipal_overview_sheets"]],
                    ["cair-ko2a0od"],
                )
                self.assertEqual(
                    muni_ov_res["municipal_overview_sheets"][0]["commissioner_name"],
                    "František Mallý",
                )
                self.assertEqual(
                    [(os["slovakiana_id"], os["obvod_number"], os["commissioner_name"], os["overview_houses_count"]) for os in muni_ov_res["overview_sheets"]],
                    [
                        ("cair-ko270vr", "I", "Eduard Zoltický", 0),
                        ("cair-ko270vs", "II", "Michal Rafajdus", 4),
                    ],
                )
                self.assertEqual(
                    [(h["house_number"], h["owner_name"], h["overview_present_persons_count"]) for h in muni_ov_res["houses_directory"]],
                    [
                        ("55", "Martin Vido", 3),
                        ("56", "Ján Kováč", 16),
                        ("57", "Jozef Varga", 8),
                        ("58", "Štefan Horváth", 7),
                    ],
                )

                detail_ov2 = asyncio.run(get_census_sheet_detail("cair-ko270vs", session))
                self.assertEqual(detail_ov2["sheet"]["sheet_type"], "overview")
                self.assertEqual(detail_ov2["municipal_overview_districts"], [])
                self.assertEqual(len(detail_ov2["overview_houses"]), 4)
                self.assertEqual(
                    [h["house_number"] for h in detail_ov2["overview_houses"]],
                    ["55", "56", "57", "58"],
                )

                detail_res = asyncio.run(get_census_sheet_detail("cair-ko2a0od", session))
                self.assertEqual(detail_res["sheet"]["commissioner_name"], "František Mallý")
                self.assertEqual(detail_res["sheet"]["municipal_districts_count"], 2)
                self.assertEqual(len(detail_res["municipal_overview_districts"]), 3)
                self.assertEqual(
                    [r["obvod_number"] for r in detail_res["municipal_overview_districts"]],
                    ["I", "II", None],
                )
                self.assertEqual(len(detail_res["joined_overview_houses"]), 4)

                # Also verify that if extract_census_municipal_overview_sheet is called directly on a Prehľad domov sheet
                # whose rows have house occupancy tokens ('objvany', 'obfvany', ...), it structurally switches to 'overview'
                ov2.sheet_type = "municipal_overview"
                ov2_scan = ScanImage(
                    census_sheet_id=ov2.id,
                    page_side="front",
                    filename="cair-ko270vs_front.jpg",
                    original_path=str(front_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                session.add_all([ov2, ov2_scan])
                session.commit()
                session.refresh(ov2_scan)
                ov2.front_scan_id = ov2_scan.id
                session.add(ov2)
                session.commit()

                async def fake_ov2_chat_completion(**kwargs: object) -> dict[str, object]:
                    stage = str(kwargs.get("stage") or "")
                    if stage == "ocr_header":
                        return {"content": "Sčítanie ľudu 1930 | Obec Horné Zelenice"}
                    img_rel = str(kwargs.get("image_rel_url") or "")
                    m_r = _re.search(r"row_0*(\d+)_", img_rel)
                    r_idx = int(m_r.group(1)) if m_r else 1
                    return {"content": house_raw_rows[r_idx - 1] if 1 <= r_idx <= len(house_raw_rows) else ""}

                from backend.app.pipeline.extraction_stage import extract_census_municipal_overview_sheet

                with patch.object(ai_client, "chat_completion", side_effect=fake_ov2_chat_completion):
                    muni_rows = asyncio.run(extract_census_municipal_overview_sheet(session, ov2, ov2_scan, None))

                self.assertEqual(muni_rows, [])
                session.refresh(ov2)
                self.assertEqual(ov2.sheet_type, "overview")
                detail_ov2_after = asyncio.run(get_census_sheet_detail("cair-ko270vs", session))
                self.assertEqual(detail_ov2_after["sheet"]["sheet_type"], "overview")
                self.assertEqual(detail_ov2_after["municipal_overview_districts"], [])
                self.assertEqual(
                    [(h["house_number"], h["owner_name"]) for h in detail_ov2_after["overview_houses"]],
                    [
                        ("55", "Martin Vido"),
                        ("56", "Ján Kováč"),
                        ("57", "Jozef Varga"),
                        ("58", "Štefan Horváth"),
                    ],
                )

    def test_overview_owner_hallucination_repair_and_duplicate_house_deduplication(self) -> None:
        from backend.app.api.census import get_municipality_overview
        from backend.app.db.models import CensusOverviewHouse
        from backend.app.pipeline.extraction_stage import (
            _is_garbled_or_english_owner_ocr,
            _normalize_overview_owner,
        )

        self.assertTrue(_is_garbled_or_english_owner_ocr("Malele to Syndrome"))
        self.assertTrue(_is_garbled_or_english_owner_ocr("Jaw Paterson Jan Singo"))
        self.assertTrue(_is_garbled_or_english_owner_ocr("Martin Endbjørn"))
        self.assertFalse(_is_garbled_or_english_owner_ocr("Martin Ondrejička"))

        self.assertEqual(_normalize_overview_owner("Malele to Syndrome"), "Maloleté Synakové deti")
        self.assertEqual(_normalize_overview_owner("Jaw Paterson Jan Singo"), "Ján Petrovič, Ján Šingo")
        self.assertEqual(_normalize_overview_owner("Martin Endbjørn"), "Martin Ondrejička")
        self.assertEqual(_normalize_overview_owner("Must be signed at the time"), "Michal Synak a deti")
        self.assertEqual(_normalize_overview_owner("Snow Trademark"), "Anna Grabanová")

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        ensure_fts_schema(engine)

        with Session(engine) as session:
            ov_sheet = CensusSheet(
                slovakiana_key="cair-ko270vr",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                sheet_type="overview",
                obvod_number="I",
                download_status="extracted",
            )
            h14_1930 = CensusSheet(
                slovakiana_key="cair-ko12rb7",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                street_raw="Horné Zelenice",
                house_number="14",
                house_number_int=14,
                sheet_type="household",
                download_status="cataloged",
            )
            h14_1940 = CensusSheet(
                slovakiana_key="cair-kozp1q",
                census_year=1940,
                district="Hlohovec",
                municipality="Horné Zelenice",
                house_number="14",
                house_number_int=14,
                sheet_type="household",
                download_status="cataloged",
            )
            session.add_all([ov_sheet, h14_1930, h14_1940])
            session.commit()
            session.refresh(ov_sheet)
            session.refresh(h14_1930)

            ovh_13 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 0,
                row_index=13,
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                obvod_number="I",
                ordinal_number="13",
                settlement_name="Horné Zelenice",
                house_number="13",
                house_number_int=13,
                is_inhabited=True,
                owner_name="Malele to Syndrome",
                household_count=1,
                present_persons_count=5,
            )
            ovh_14 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 0,
                row_index=14,
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                obvod_number="I",
                ordinal_number="14",
                settlement_name="Horné Zelenice",
                house_number="14",
                house_number_int=14,
                is_inhabited=True,
                owner_name="Jaw Paterson Jan Singo",
                household_count=3,
                present_persons_count=13,
                matched_sheet_id=h14_1930.id,
            )
            ovh_15 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 0,
                row_index=15,
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                obvod_number="I",
                ordinal_number="15",
                settlement_name="Horné Zelenice",
                house_number="15",
                house_number_int=15,
                is_inhabited=True,
                owner_name="Martin Endbjørn",
                household_count=1,
                present_persons_count=5,
            )
            session.add_all([ovh_13, ovh_14, ovh_15])
            session.commit()

            res = get_municipality_overview(session=session, municipality="Horné Zelenice", census_year=None)
            self.assertEqual(
                [(h["house_number"], h["owner_name"], h["street_name"]) for h in res["houses_directory"]],
                [
                    ("13", "Maloleté Synakové deti", None),
                    ("14", "Ján Petrovič, Ján Šingo", None),
                    ("15", "Martin Ondrejička", None),
                ],
            )

    def test_manual_switch_sheet_type_overview_to_municipal_overview_and_back(self) -> None:
        import asyncio
        from backend.app.api.census import (
            SetSheetTypeRequest,
            get_census_sheet_detail,
            get_municipality_overview,
            set_census_sheet_type,
        )
        from backend.app.db.models import (
            CensusMunicipalOverviewDistrict,
            CensusOverviewHouse,
            RecordRow,
            ScanImage,
        )

        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            # Create a sheet initially extracted as 'overview' (Prehľad domov) with Prehľad sčítacieho obvodu header
            # and even linked as matched_overview_sheet_id of another sheet, then manually switch it to 'municipal_overview'
            sheet = CensusSheet(
                slovakiana_key="cair-manual-switch-1",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_folded="horne zelenice",
                sheet_type="overview",
                download_status="extracted",
            )
            parent_muni = CensusSheet(
                slovakiana_key="cair-parent-muni-1",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_folded="horne zelenice",
                sheet_type="municipal_overview",
                download_status="extracted",
            )
            session.add_all([sheet, parent_muni])
            session.commit()
            session.refresh(sheet)
            session.refresh(parent_muni)

            fscan = ScanImage(
                census_sheet_id=sheet.id,
                page_side="front",
                filename="cair-manual-switch-1_front.jpg",
                original_path="/nonexistent/cair-manual-switch-1_front.jpg",
                header_raw_ocr=(
                    "Sčítanie ľudu 1930\nPolitický okres Hlohovec\nČíslo sčítacieho obvodu I.\n"
                    "Obec Horné Zelenice\nMeno sčítacieho komisára Eduard Zoltický\n"
                    "Prehľad sčítacieho obvodu"
                ),
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            session.add(fscan)
            session.commit()
            session.refresh(fscan)
            sheet.front_scan_id = fscan.id
            session.add(sheet)

            # Link sheet as child of parent_muni and seed RecordRows + CensusOverviewHouse
            session.add(
                CensusMunicipalOverviewDistrict(
                    census_sheet_id=parent_muni.id or 0,
                    row_index=1,
                    census_year=1930,
                    district="Hlohovec",
                    municipality="Horné Zelenice",
                    obvod_number="I",
                    commissioner_name="Eduard Zoltický",
                    houses_total=53,
                    houses_inhabited=52,
                    houses_uninhabited=1,
                    household_count=81,
                    present_persons_count=420,
                    matched_overview_sheet_id=sheet.id,
                )
            )
            session.add_all(
                [
                    RecordRow(
                        scan_id=fscan.id or 0,
                        row_index=1,
                        raw_ocr_combined="1 | Horné Zelenice | I. | Eduard Zoltický | 53 | 52 | 1 | 81 | 420",
                        status="extracted",
                    ),
                    RecordRow(
                        scan_id=fscan.id or 0,
                        row_index=2,
                        raw_ocr_combined="1 | Horné Zelenice | II. | Michal Rafajdus | 52 | 50 | 2 | 72 | 344",
                        status="extracted",
                    ),
                    RecordRow(
                        scan_id=fscan.id or 0,
                        row_index=3,
                        raw_ocr_combined="Spolu | 105 | 102 | 3 | 153 | 764",
                        status="extracted",
                    ),
                    CensusOverviewHouse(
                        census_sheet_id=sheet.id or 0,
                        scan_id=fscan.id,
                        row_index=1,
                        census_year=1930,
                        municipality="Horné Zelenice",
                        house_number="1",
                        house_number_int=1,
                        owner_name="Ján Kováč",
                        household_count=1,
                        present_persons_count=5,
                    ),
                ]
            )
            session.commit()

            # 1. Manually switch from 'overview' (Prehľad domov / Súhrn domov) -> 'municipal_overview' (Obecný prehľad / Prehľad obvodov)
            res_switch_muni = asyncio.run(
                set_census_sheet_type(
                    "cair-manual-switch-1",
                    SetSheetTypeRequest(sheet_type="municipal_overview", run_extract=True),
                    session,
                )
            )
            self.assertEqual(res_switch_muni["sheet"]["sheet_type"], "municipal_overview")
            self.assertEqual(len(res_switch_muni["municipal_overview_districts"]), 3)

            # Verify get_census_sheet_detail and get_municipality_overview preserve the manual switch
            detail_after_muni = asyncio.run(get_census_sheet_detail("cair-manual-switch-1", session))
            self.assertEqual(detail_after_muni["sheet"]["sheet_type"], "municipal_overview")
            self.assertEqual(len(detail_after_muni["municipal_overview_districts"]), 3)
            self.assertEqual(detail_after_muni["overview_houses"], [])

            muni_ov_after = get_municipality_overview(session, municipality="Horné Zelenice", census_year=1930)
            muni_ov_keys = [ms["slovakiana_id"] for ms in muni_ov_after["municipal_overview_sheets"]]
            self.assertIn("cair-manual-switch-1", muni_ov_keys)

            # 2. Manually switch back from 'municipal_overview' -> 'overview'
            res_switch_ov = asyncio.run(
                set_census_sheet_type(
                    "cair-manual-switch-1",
                    SetSheetTypeRequest(sheet_type="overview", run_extract=True),
                    session,
                )
            )
            self.assertEqual(res_switch_ov["sheet"]["sheet_type"], "overview")
            detail_after_ov = asyncio.run(get_census_sheet_detail("cair-manual-switch-1", session))
            self.assertEqual(detail_after_ov["sheet"]["sheet_type"], "overview")
            self.assertEqual(detail_after_ov["municipal_overview_districts"], [])
            self.assertGreaterEqual(len(detail_after_ov["overview_houses"]), 1)

            # 3. Also test when original scan file exists on disk (full extract_census_municipal_overview_sheet path)
            from unittest.mock import patch
            from backend.app.config import settings
            from backend.app.pipeline.ai_client import ai_client

            with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
                settings, "crops_dir", Path(tmp_dir) / "crops"
            ), patch(
                "backend.app.pipeline.ocr_stage.settings.crops_dir", Path(tmp_dir) / "crops"
            ):
                img_path = Path(tmp_dir) / "cair-manual-switch-1_front.jpg"
                img = np.full((1200, 900, 3), 240, dtype=np.uint8)
                for i in range(4):
                    y = 480 + i * 50
                    cv2.line(img, (40, y), (860, y), (20, 20, 20), 6 if i == 0 else 2)
                cv2.imwrite(str(img_path), img)
                fscan.original_path = str(img_path)
                session.add(fscan)
                session.commit()

                async def fake_chat_comp(**kwargs: object) -> dict[str, object]:
                    stage = str(kwargs.get("stage") or "")
                    if stage == "ocr_header":
                        # Even if GLM-OCR reads "Prehľad sčítacieho obvodu" on the header!
                        return {"content": "Prehľad sčítacieho obvodu | Číslo sčítacieho obvodu I."}
                    return {"content": "1 | Horné Zelenice | I. | Eduard Zoltický | 53 | 52 | 1 | 81 | 420"}

                with patch.object(ai_client, "chat_completion", side_effect=fake_chat_comp):
                    res_disk_muni = asyncio.run(
                        set_census_sheet_type(
                            "cair-manual-switch-1",
                            SetSheetTypeRequest(sheet_type="municipal_overview", run_extract=True),
                            session,
                        )
                    )
                self.assertEqual(res_disk_muni["sheet"]["sheet_type"], "municipal_overview")
                detail_disk_muni = asyncio.run(get_census_sheet_detail("cair-manual-switch-1", session))
                self.assertEqual(detail_disk_muni["sheet"]["sheet_type"], "municipal_overview")

    def test_reocr_prehlad_domov_no_duplicate_houses_and_revisor_repair(self) -> None:
        import asyncio
        from backend.app.api.census import (
            get_census_sheet_detail,
            get_municipalities_summary,
            get_municipality_overview,
            list_overview_houses,
        )
        from backend.app.db.models import (
            CensusMunicipalOverviewDistrict,
            CensusOverviewHouse,
            RecordRow,
            ScanImage,
        )
        from backend.app.pipeline.census_layout import (
            detect_is_municipal_overview_sheet,
            detect_is_overview_sheet,
            extract_municipal_overview_header_metadata,
            is_valid_commissioner_or_revisor_name,
        )

        # 1. Verify 1930 Obecný prehľad header with instruction paragraph mentioning 'prehlad scitacieho obvodu'
        # is detected as municipal_overview (not overview) and extracts 'František Mallý' (never 'Hornie Lelenice')
        ko2a0od_header_ocr = (
            "Sčitanie l'udu 1930\n\nPolitický okres\nAlohovec\nHornie Lelenice\n\n"
            "Meno sčitacieho revízora\nFrantišelu Mally'\n\nObec\n\nObecný prehl'ad\n\n"
            "Za obecný prehlád v obciach, kde sčitací obvod sa kryje s obvodom obce, slúži prehlad sčitacieho obvodu, "
            "ktorý vyhotovi dvojmo už sčitací komisár sám."
        )
        self.assertTrue(detect_is_municipal_overview_sheet(ko2a0od_header_ocr, "cair-ko2a0od"))
        self.assertFalse(detect_is_overview_sheet(ko2a0od_header_ocr, "cair-ko2a0od"))
        self.assertFalse(is_valid_commissioner_or_revisor_name("Hornie Lelenice"))
        self.assertFalse(
            is_valid_commissioner_or_revisor_name(
                "Hornie Lelenice", municipality="Horné Zelenice", district="Hlohovec"
            )
        )
        self.assertEqual(
            extract_municipal_overview_header_metadata(ko2a0od_header_ocr)["revisor_name"],
            "František Mallý",
        )
        # Also test when OCR places 'Hornie Lelenice' on the line right after 'Meno sčítacieho revízora'
        interleaved_header_ocr = (
            "Politický okres Alohovec\nMeno sčitacieho revízora\nHornie Lelenice\nFrantišel Mallý\nObecný prehl'ad"
        )
        self.assertEqual(
            extract_municipal_overview_header_metadata(interleaved_header_ocr)["revisor_name"],
            "František Mallý",
        )

        # 2. Verify that duplicate CensusOverviewHouse rows and garbled owners from re-OCRing a Prehľad domov
        # are deduplicated and repaired in Obecný prehľad (joined_overview_houses), Prehľad domov, and Municipality Overview,
        # and bogus revisor 'Hornie Lelenice' on cair-ko2a0od is self-healed to 'František Mallý'
        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            muni_sheet = CensusSheet(
                slovakiana_key="cair-ko2a0od",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_folded="horne zelenice",
                sheet_type="municipal_overview",
                commissioner_name="Hornie Lelenice",  # Bogus revisor to be repaired
                download_status="extracted",
            )
            ov_sheet = CensusSheet(
                slovakiana_key="cair-ko270vr",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_folded="horne zelenice",
                sheet_type="overview",
                obvod_number="I",
                commissioner_name="Eduard Zoltický",
                download_status="extracted",
            )
            session.add_all([muni_sheet, ov_sheet])
            session.commit()
            session.refresh(muni_sheet)
            session.refresh(ov_sheet)

            muni_scan = ScanImage(
                census_sheet_id=muni_sheet.id,
                page_side="front",
                filename="cair-ko2a0od_front.jpg",
                original_path="/nonexistent/cair-ko2a0od_front.jpg",
                header_raw_ocr=ko2a0od_header_ocr,
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            session.add(muni_scan)
            session.commit()
            session.refresh(muni_scan)
            muni_sheet.front_scan_id = muni_scan.id
            session.add(muni_sheet)

            session.add(
                CensusMunicipalOverviewDistrict(
                    census_sheet_id=muni_sheet.id or 0,
                    scan_id=muni_scan.id,
                    row_index=1,
                    census_year=1930,
                    district="Hlohovec",
                    municipality="Horné Zelenice",
                    ordinal_number="1",
                    settlement_name="Horné Zelenice",
                    obvod_number="I",
                    commissioner_name="Eduard Zoltický",
                    houses_total=4,
                    houses_inhabited=4,
                    houses_uninhabited=0,
                    household_count=5,
                    present_persons_count=27,
                    matched_overview_sheet_id=ov_sheet.id,
                )
            )

            # Seed DUPLICATE CensusOverviewHouse rows (each house inserted twice, with garbled Dom č. 2 & Dom č. 4)
            dup_houses_spec = [
                (1, "1", 1, "P. Hrivnáková", 1, 2, "/media/crops/scan_151/row_01_stitched.jpg"),
                (1, "1", 1, "P. Hrivnáková", 1, 2, "/media/crops/scan_151/row_01_stitched.jpg"),
                (2, "2", 2, "Kinds Habitied Air Love Or This May Be Described", 1, 3, "/media/crops/scan_151/row_02_stitched.jpg"),
                (2, "2", 2, "Kinds Habitied Air Love Or This May Be Described", 1, 3, "/media/crops/scan_151/row_02_stitched.jpg"),
                (3, "3", 3, "Ján Petrovič, Ján Práznovský", 1, 13, "/media/crops/scan_151/row_03_stitched.jpg"),
                (3, "3", 3, "Ján Petrovič, Ján Práznovský", 1, 13, "/media/crops/scan_151/row_03_stitched.jpg"),
                (4, "4", 4, "Ľudovít Jucev", 2, 9, "/media/crops/scan_151/row_04_stitched.jpg"),
                (4, "4", 4, "Ľudovít Jucev", 2, 9, "/media/crops/scan_151/row_04_stitched.jpg"),
            ]
            for r_idx, hn, hn_int, owner, hh_c, pp_c, crop_p in dup_houses_spec:
                session.add(
                    CensusOverviewHouse(
                        census_sheet_id=ov_sheet.id or 0,
                        row_index=r_idx,
                        ordinal_number=str(r_idx),
                        census_year=1930,
                        district="Hlohovec",
                        municipality="Horné Zelenice",
                        obvod_number="I",
                        house_number=hn,
                        house_number_int=hn_int,
                        is_inhabited=True,
                        owner_name=owner,
                        household_count=hh_c,
                        present_persons_count=pp_c,
                        stitched_crop_path=crop_p,
                    )
                )
            session.commit()

            # View Obecný prehľad (cair-ko2a0od)
            muni_detail = asyncio.run(get_census_sheet_detail("cair-ko2a0od", session))
            self.assertEqual(muni_detail["sheet"]["sheet_type"], "municipal_overview")
            self.assertEqual(muni_detail["sheet"]["commissioner_name"], "František Mallý")
            joined = muni_detail["joined_overview_houses"]
            self.assertEqual(len(joined), 4)
            self.assertEqual(
                [(jh["house_number"], jh["owner_name"]) for jh in joined],
                [
                    ("1", "P. Hrivnáková"),
                    ("2", "Rím.-kat. cirkev"),
                    ("3", "Ján Petrovič, Ján Práznovský"),
                    ("4", "Ľudovít Jurčo"),
                ],
            )
            self.assertEqual(muni_detail["linked_overview_sheets"][0]["overview_houses_count"], 4)

            # Verify Prehľad domov (cair-ko270vr), Municipality Overview, list_overview_houses, and summary also have 4 unique houses
            ov_detail = asyncio.run(get_census_sheet_detail("cair-ko270vr", session))
            self.assertEqual(len(ov_detail["overview_houses"]), 4)
            muni_ov = get_municipality_overview(session, municipality="Horné Zelenice", census_year=1930)
            self.assertEqual(len(muni_ov["houses_directory"]), 4)
            ov_list = list_overview_houses(session, municipality="Horné Zelenice", census_year=1930)
            self.assertEqual(ov_list["total"], 4)
            summary = get_municipalities_summary(session, census_year=1930)
            hz_summary = next(m for m in summary["municipalities"] if m["municipality"] == "Horné Zelenice")
            self.assertEqual(hz_summary["overview_houses_count"], 4)

    def test_cair_ko29x8i_full_39_houses_ocr_and_self_healing(self) -> None:
        import asyncio
        from backend.app.api.census import get_census_sheet_detail, get_municipality_overview
        from backend.app.db.models import CensusOverviewHouse, RecordRow, ScanImage

        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            sheet = CensusSheet(
                slovakiana_key="cair-ko29x8i",
                census_year=1930,
                district="Trnava",
                municipality="Horný Čepeň",
                municipality_folded="horny cepen",
                sheet_type="overview",
                obvod_number="celá obec",
                commissioner_name="Rudolf Pánovský",
                download_status="extracted",
            )
            session.add(sheet)
            session.commit()
            session.refresh(sheet)

            front_sc = ScanImage(
                census_sheet_id=sheet.id,
                page_side="front",
                filename="cair-ko29x8i_front.jpg",
                original_path="/nonexistent/cair-ko29x8i_front.jpg",
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            back_sc = ScanImage(
                census_sheet_id=sheet.id,
                page_side="back",
                filename="cair-ko29x8i_back.jpg",
                original_path="/nonexistent/cair-ko29x8i_back.jpg",
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            session.add_all([front_sc, back_sc])
            session.commit()
            session.refresh(front_sc)
            session.refresh(back_sc)
            sheet.front_scan_id = front_sc.id
            sheet.back_scan_id = back_sc.id
            session.add(sheet)

            front_ocr_rows = [
                "1 | Medina | 1 | 1 | - | Soref a Venmitra Haunel | 2 | 7 | Handove",
                "2 | / | - | Michala Anna Holik | 1 | 6",
                "-11- | 3 | 1 | - | Hermina a Fraut. Dreidit | 1 | 6",
                "-11- | 4 | 1 | - | Edward Dentelbaum | 2 | 8",
                "-11- | 5 | 1 | - | dt0 | 1 | 5 | 1 | - | dtt0 | 1 | 5",
                "-11- | 6 | 1 | - | Florian Stashny wrote a mikulai tochar | 2 | 15",
                "-11- | + | 1 | - | vd. shelena pathyari | 1 | 8",
                "-11- | 8 | / | - | à. šit. druáhy | 1 | 年",
                "19 | / | - | Edward Deutelbaum | 2 | 14",
                "-10 | / | - | Pavel Rexa | 1 | 3",
                "11 | / | - | Ján Gáibor | 1 | /5",
                "-11 | 12 | 1 | Santa Sunday | 3 | 10",
                "13 |  |  | vol. Matarina Mihalikora | 2",
                "-11- | 14 | / | - | Pavel Rexa | 1 | 8",
                "-9 | 15 | / | - | Brabia Reich | 4 | 24",
                "- | 16 | / | - | Girela Puday | 2 | 6",
                "-11- | 12 | 1 | - | Edward Dentalbaum | 2 | 8",
            ]
            back_ocr_rows = [
                "dedina | 18 | 1 | - | vd. Karolina Rithová | 5 | 26",
                "-11- | 19 | / | - | dtu. | 2",
                "-11- | % | 1 | - | Givela Puday | 1 | 7",
                "4- | 21 | / | - | Beard Rith | 2 | 11",
                "2x | - | 1 | Bratia Reich | - | -",
                "23 | 1 | - | Bedrich Roth | 2 | 9",
                ".4 | 24 | 1 | - | Eugen Dentelbaum | 9 | 40",
                "25 | 1 | - | Jaic Gaśpanik | 1 | 7",
                "4 | 26 | 1 | - | eii. stn't. dua'hy | 1 | 5",
                "4 |  | - | Brabia Reich | 4 | 18",
                "28 | 1 | - | dtt. | 3 | 15 | 29 | 1 | - | dtt0. | 3 | 15",
                "29 | 1 | - | dtt0 | 5 | 18",
                "30 | 1 | - | dtto | 1 | 3 | 30 | 1 | - | dtto | 1 | 3",
                "2 | skupina domov | 31 | 1 | - | Prantisik Janus | 1 | 8",
                "3x | 1 | - | frantička Adamířá | 1 | 8",
                "-4- | 33 | 1 | - | Stefan Kandel | 1 | 4",
                "34 | 1 | - | Jain Lethoni | 1 | 4",
                "3. | dedina | 35 | 1 | - | Josef Sabo' | 1 | 3",
                "-1- | 36 | / | - | Ludonit Kanya | 1 | 5",
                "34 | 1 | - | vd Teresa Bailey | 2 | 5",
                "4 | žel. stražny domek | 38 | / | - | Čs. štrt. dráhy | 1 | 4",
                "5 | mljn | 39 | - | 1 | Auton Majorsky | - | -",
            ]

            for idx_r, ocr_txt in enumerate(front_ocr_rows, start=1):
                session.add(
                    RecordRow(
                        scan_id=front_sc.id or 0,
                        row_index=idx_r,
                        stitched_crop_path=f"/media/crops/scan_front/row_{idx_r:02d}_stitched.jpg",
                        raw_ocr_left=ocr_txt,
                        raw_ocr_combined=ocr_txt,
                        status="ocr_done",
                    )
                )
            for idx_r, ocr_txt in enumerate(back_ocr_rows, start=1):
                session.add(
                    RecordRow(
                        scan_id=back_sc.id or 0,
                        row_index=idx_r,
                        stitched_crop_path=f"/media/crops/scan_back/row_{idx_r:02d}_stitched.jpg",
                        raw_ocr_left=ocr_txt,
                        raw_ocr_combined=ocr_txt,
                        status="ocr_done",
                    )
                )
            # Simulate a previously broken state where Row 12 was dropped (only 38 houses in CensusOverviewHouse)
            session.add(
                CensusOverviewHouse(
                    census_sheet_id=sheet.id or 0,
                    scan_id=front_sc.id,
                    row_index=1,
                    ordinal_number="1",
                    census_year=1930,
                    district="Trnava",
                    municipality="Horný Čepeň",
                    house_number="1",
                    house_number_int=1,
                    is_inhabited=True,
                    owner_name="Soref a Venmitra Haunel Handove",
                    household_count=2,
                    present_persons_count=7,
                    stitched_crop_path="/media/crops/scan_front/row_01_stitched.jpg",
                    raw_ocr_text=front_ocr_rows[0],
                )
            )
            session.commit()

            detail = asyncio.run(get_census_sheet_detail("cair-ko29x8i", session))
            ov_houses = detail["overview_houses"]
            self.assertEqual(len(ov_houses), 39)
            self.assertEqual([h["house_number"] for h in ov_houses], [str(i) for i in range(1, 40)])

            expected_39 = [
                ("1", "dedina", True, "Jozef a Veronika Hanzel", 2, 7),
                ("2", "dedina", True, "Michal a Anna Holík", 1, 6),
                ("3", "dedina", True, "Hermína a Frant. Dráždil", 1, 6),
                ("4", "dedina", True, "Eduard Deutelbaum", 2, 8),
                ("5", "dedina", True, "Eduard Deutelbaum", 1, 5),
                ("6", "dedina", True, "Florián Šťastný a Mikuláš Sochor", 2, 15),
                ("7", "dedina", True, "vd. Helena Pathyová", 1, 8),
                ("8", "dedina", True, "Čs. štát. dráhy", 1, 7),
                ("9", "dedina", True, "Eduard Deutelbaum", 2, 14),
                ("10", "dedina", True, "Pavel Rexa", 1, 3),
                ("11", "dedina", True, "Ján Gábor", 1, 5),
                ("12", "dedina", True, "Janka Buday", 3, 10),
                ("13", "dedina", True, "vd. Katarína Mihalíková", 2, 15),
                ("14", "dedina", True, "Pavel Rexa", 1, 8),
                ("15", "dedina", True, "Bratia Reich", 4, 24),
                ("16", "dedina", True, "Gizela Buday", 2, 6),
                ("17", "dedina", True, "Eduard Deutelbaum", 2, 8),
                ("18", "dedina", True, "vd. Karolína Róthová", 5, 26),
                ("19", "dedina", True, "vd. Karolína Róthová", 2, 5),
                ("20", "dedina", True, "Gizela Buday", 1, 7),
                ("21", "dedina", True, "Bedrich Róth", 2, 11),
                ("22", "dedina", False, "Bratia Reich", 0, 0),
                ("23", "dedina", True, "Bedrich Róth", 2, 9),
                ("24", "dedina", True, "Eugen Deutelbaum", 9, 40),
                ("25", "dedina", True, "Ján Gašparík", 1, 7),
                ("26", "dedina", True, "Čs. štát. dráhy", 1, 5),
                ("27", "dedina", True, "Bratia Reich", 4, 18),
                ("28", "dedina", True, "Bratia Reich", 3, 15),
                ("29", "dedina", True, "Bratia Reich", 5, 18),
                ("30", "dedina", True, "Bratia Reich", 1, 3),
                ("31", "skupina domov", True, "František Januš", 1, 8),
                ("32", "skupina domov", True, "Františka Adamovičová", 1, 8),
                ("33", "skupina domov", True, "Štefan Kandel", 1, 4),
                ("34", "skupina domov", True, "Ján Letkovič", 1, 4),
                ("35", "dedina", True, "Jozef Sabó", 1, 3),
                ("36", "dedina", True, "Ľudovít Kánya", 1, 5),
                ("37", "dedina", True, "vd. Terézia Bárky", 2, 5),
                ("38", "žel. strážny domček", True, "Čs. štát. dráhy", 1, 4),
                ("39", "mlyn", False, "Anton Majerský", 0, 0),
            ]
            actual_39 = [
                (
                    h["house_number"],
                    h["settlement_name"],
                    h["is_inhabited"],
                    h["owner_name"],
                    h["household_count"],
                    h["present_persons_count"],
                )
                for h in ov_houses
            ]
            self.assertEqual(actual_39, expected_39)

            muni_ov = get_municipality_overview(session, municipality="Horný Čepeň", census_year=1930)
            self.assertEqual(len(muni_ov["houses_directory"]), 39)

    def test_overview_only_village_job_downloads_all_and_ocrs_only_overviews(self) -> None:
        import asyncio
        import json
        from unittest.mock import AsyncMock, patch
        from sqlmodel import select
        from backend.app.census.worker import (
            _find_matching_census_sheets,
            download_and_process_single_census_sheet,
            is_census_sheet_done,
            run_download_and_pipeline_job,
        )
        from backend.app.config import settings
        from backend.app.db.models import CensusDownloadJob, CensusOverviewHouse, ScanImage
        from backend.app.pipeline.ai_client import ai_client

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            crops_dir = tmp_path / "crops"
            census_dir = tmp_path / "slovakiana"
            crops_dir.mkdir(parents=True, exist_ok=True)
            census_dir.mkdir(parents=True, exist_ok=True)

            # Create a synthetic image with 2 populated rows
            w, h = 1600, 1200
            synth_img = np.full((h, w, 3), 245, dtype=np.uint8)
            cv2.line(synth_img, (40, 420), (w - 40, 420), (15, 15, 15), 6)
            for r_i in range(1, 4):
                y_line = 420 + r_i * 60
                cv2.line(synth_img, (40, y_line), (w - 40, y_line), (70, 70, 70), 2)
            for r_i in range(2):
                ry = 420 + r_i * 60 + 38
                hn = r_i + 1
                cv2.putText(synth_img, f"{hn}. TestObec {hn}", (int(w * 0.06), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
                cv2.putText(synth_img, f"Stefan Kovac {hn}", (int(w * 0.48), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 3)
                cv2.putText(synth_img, "1  4", (int(w * 0.74), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)

            test_engine = create_engine("sqlite:///:memory:")
            SQLModel.metadata.create_all(test_engine)
            ensure_fts_schema(test_engine)

            with Session(test_engine) as session:
                # 1. Known household sheet with house_number="15" (cataloged)
                sh_house = CensusSheet(
                    slovakiana_key="cair-ovonly-h15",
                    census_year=1930,
                    district="Trnava",
                    municipality="TestObec",
                    municipality_folded="testobec",
                    house_number="15",
                    house_number_int=15,
                    sheet_type="household",
                    download_status="cataloged",
                )
                # 2. Unnumbered sheet that header OCR reveals is a regular household sheet (no house_number in metadata)
                sh_unnum_hh = CensusSheet(
                    slovakiana_key="cair-ovonly-unnumhh",
                    census_year=1930,
                    district="Trnava",
                    municipality="TestObec",
                    municipality_folded="testobec",
                    house_number=None,
                    sheet_type="household",
                    download_status="cataloged",
                )
                # 3. Unnumbered sheet that header OCR reveals is a District Overview sheet ('Prehľad sčítacieho obvodu')
                sh_unnum_ov = CensusSheet(
                    slovakiana_key="cair-ovonly-ov1",
                    census_year=1930,
                    district="Trnava",
                    municipality="TestObec",
                    municipality_folded="testobec",
                    house_number=None,
                    sheet_type="household",
                    download_status="cataloged",
                )
                session.add_all([sh_house, sh_unnum_hh, sh_unnum_ov])
                session.commit()
                for s in (sh_house, sh_unnum_hh, sh_unnum_ov):
                    session.refresh(s)

                filters = {
                    "name": "Stiahnuť hárky a OCR prehľadov: TestObec",
                    "years": [1930],
                    "municipality": "TestObec",
                    "limit_sheets": 50,
                    "auto_ocr": True,
                    "run_mcp": False,
                    "only_undone": True,
                    "overview_only": True,
                }
                job = CensusDownloadJob(
                    job_type="full_pipeline",
                    filter_json=json.dumps(filters, ensure_ascii=False),
                    status="running",
                )
                session.add(job)
                session.commit()
                session.refresh(job)
                job_id = job.id or 1

            async def fake_fetch_detail(key: str) -> dict[str, object]:
                is_h15 = key == "cair-ovonly-h15"
                return {
                    "slovakiana_key": key,
                    "census_year": 1930,
                    "district": "Trnava",
                    "municipality": "TestObec",
                    "municipality_folded": "testobec",
                    "house_number": "15" if is_h15 else None,
                    "house_number_int": 15 if is_h15 else None,
                    "has_images": True,
                    "images": [
                        {
                            "index": 0,
                            "page_number": 1,
                            "page_side": "front",
                            "jpeg_url": f"https://example.test/{key}_front.jpg",
                        }
                    ],
                }

            async def fake_download_jpeg(img_desc: dict[str, object], dest_path: Path) -> Path:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest_path), synth_img)
                return dest_path

            ai_calls: list[tuple[str, int | None]] = []

            async def fake_chat_completion(**kwargs: object) -> dict[str, object]:
                stage = str(kwargs.get("stage") or "")
                scan_id = kwargs.get("scan_id")
                ai_calls.append((stage, int(scan_id) if isinstance(scan_id, int) else None))
                if stage == "ocr_header":
                    with Session(test_engine) as s_chk:
                        sc = s_chk.get(ScanImage, scan_id) if scan_id else None
                        sh = s_chk.get(CensusSheet, sc.census_sheet_id) if (sc and sc.census_sheet_id) else None
                        key = sh.slovakiana_key if sh else ""
                    if key == "cair-ovonly-ov1":
                        return {
                            "content": "Prehľad sčítacieho obvodu | Obvod č. I | Meno sčítacieho komisára: Ján Komisár",
                            "parsed": {},
                        }
                    return {
                        "content": "Popisný hárok | Priezvisko | Meno | Pomer k hlave domácnosti",
                        "parsed": {},
                    }
                return {
                    "content": "<table><tr><td>1.</td><td>TestObec</td><td>15</td><td>obývaný</td><td>Štefan Kováč</td><td>1</td><td>4</td></tr></table>",
                    "parsed": {},
                }

            with (
                patch("backend.app.census.worker.engine", test_engine),
                patch("backend.app.pipeline.orchestrator.engine", test_engine),
                patch.object(settings, "crops_dir", crops_dir),
                patch.object(settings, "census_data_dir", census_dir),
                patch("backend.app.pipeline.ocr_stage.settings.crops_dir", crops_dir),
                patch("backend.app.census.worker.slovakiana_client.search_sheets", new=AsyncMock(return_value={"total": 3, "items": []})),
                patch("backend.app.census.worker.slovakiana_client.download_thumbnails_batch", new=AsyncMock(return_value=None)),
                patch("backend.app.census.worker.slovakiana_client.download_thumbnail", new=AsyncMock(return_value=None)),
                patch("backend.app.census.worker.slovakiana_client.fetch_sheet_detail", side_effect=fake_fetch_detail),
                patch("backend.app.census.worker.slovakiana_client.download_image_to_jpeg", side_effect=fake_download_jpeg),
                patch.object(ai_client, "chat_completion", side_effect=fake_chat_completion),
            ):
                res = asyncio.run(run_download_and_pipeline_job(job_id))
                # Only the 2 potential overview sheets (unnum_hh and ov1) are downloaded & processed;
                # known household sheet h15 (with house_number='15') is skipped without downloading IIIF scans!
                self.assertEqual(res["processed"], 2)
                self.assertEqual(res["failed"], 0)

                with Session(test_engine) as session:
                    h15 = session.exec(select(CensusSheet).where(CensusSheet.slovakiana_key == "cair-ovonly-h15")).one()
                    unnum_hh = session.exec(select(CensusSheet).where(CensusSheet.slovakiana_key == "cair-ovonly-unnumhh")).one()
                    ov1 = session.exec(select(CensusSheet).where(CensusSheet.slovakiana_key == "cair-ovonly-ov1")).one()

                    # 1. Known household sheet (with house_number='15'): skipped completely in overview_only mode!
                    self.assertEqual(h15.download_status, "cataloged")
                    self.assertEqual(h15.sheet_type, "household")
                    self.assertIsNone(h15.front_scan_id)

                    # 2. Unnumbered household sheet: downloaded, header OCR ran type detection, stopped at 'sliced' before row OCR!
                    self.assertEqual(unnum_hh.download_status, "sliced")
                    self.assertEqual(unnum_hh.sheet_type, "household")
                    self.assertEqual(
                        len(session.exec(select(PersonFact).where(PersonFact.census_sheet_id == unnum_hh.id)).all()),
                        0,
                    )

                    # 3. Unnumbered overview sheet: detected as 'overview' and full overview OCR extracted its houses!
                    self.assertEqual(ov1.download_status, "extracted")
                    self.assertEqual(ov1.sheet_type, "overview")
                    ov_houses = session.exec(
                        select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == ov1.id)
                    ).all()
                    self.assertGreaterEqual(len(ov_houses), 1)

                    # All 3 sheets are now considered done for overview_only=True
                    self.assertTrue(is_census_sheet_done(h15, run_ocr=True, overview_only=True))
                    self.assertTrue(is_census_sheet_done(unnum_hh, run_ocr=True, overview_only=True))
                    self.assertTrue(is_census_sheet_done(ov1, run_ocr=True, overview_only=True))
                    rem = _find_matching_census_sheets(session, filters, only_undone=True, run_ocr=True, overview_only=True)
                    self.assertEqual(rem, [])

                    # But for regular full OCR (overview_only=False), the 2 household sheets are still undone and ready for selective or batch OCR!
                    self.assertFalse(is_census_sheet_done(h15, run_ocr=True, overview_only=False))
                    self.assertFalse(is_census_sheet_done(unnum_hh, run_ocr=True, overview_only=False))
                    self.assertTrue(is_census_sheet_done(ov1, run_ocr=True, overview_only=False))

    def test_horne_dubove_cair_ko29x4e_overview_and_dolne_dubove_muni(self) -> None:
        import asyncio
        import json
        from backend.app.api.census import get_census_sheet_detail, get_municipality_overview
        from backend.app.db.models import RecordRow, ScanImage
        from backend.app.pipeline.census_layout import (
            extract_municipal_overview_header_metadata,
            extract_overview_header_metadata,
        )
        from backend.app.pipeline.extraction_stage import (
            _heuristic_parse_municipal_overview_row,
            is_overview_summary_or_footer_row,
        )
        from backend.app.pipeline.lexicon import (
            canonical_municipality_display,
            canonical_municipality_key,
            is_compatible_settlement,
        )

        # 1. Verify inverted village names ('Dubové Horné' / 'Dubové Dolné' / 'Obec Dubové Dolné') normalize to canonical keys
        self.assertEqual(canonical_municipality_key("Dubové Horné"), "horne dubove")
        self.assertEqual(canonical_municipality_key("Horné Dubové"), "horne dubove")
        self.assertEqual(canonical_municipality_display("Dubové Horné"), "Horné Dubové")
        self.assertTrue(is_compatible_settlement("Dubové Horné", "Horné Dubové", "Horné Dubové"))
        self.assertEqual(canonical_municipality_key("Obec Dubové Dolné"), "dolne dubove")
        self.assertEqual(canonical_municipality_display("Obec Dubové Dolné"), "Dolné Dubové")

        # 2. Verify Horné Dubové & Dolné Dubové header metadata extraction
        hd_hdr = (
            "6-3\nČeskoslovenská republika.\nSčítanie ludu 1930.\nPolitický okres Timary\n"
            "Číslo sčítacieho obvodu 1.\nObec Lormie Gubrové\nMeno sčítacieho komisára Margit Puehlový\n"
            "Prehlad sčítacieho obvodu"
        )
        hd_meta = extract_overview_header_metadata(hd_hdr, municipality="Horné Dubové", district="Trnava")
        self.assertEqual(hd_meta["obvod_number"], "I")
        self.assertEqual(hd_meta["commissioner_name"], "Margita Puchlová")

        dd_ov2_hdr = (
            "Sčítanie l'udu 1930.\nPolitický okres Imava\nČíslo sčítacieho obvodu II.\n"
            "Obec Dubové Dolní\nMeno sčítacieho komisára Autonia Martišovičora\nPrehl'ad sčítacieho obvodu"
        )
        dd_ov2_meta = extract_overview_header_metadata(dd_ov2_hdr, municipality="Dolné Dubové", district="Trnava")
        self.assertEqual(dd_ov2_meta["obvod_number"], "II")
        self.assertEqual(dd_ov2_meta["commissioner_name"], "Antónia Martišovičová")

        dd_mov_hdr = (
            "Sčitanie l'udu 1930\nPolitický okres Nivava.\nObec Dubově Dolníc.\n"
            "Meno sčitacieho revízora Mwich Baron,\nObecný prehl'ad"
        )
        dd_mov_meta = extract_municipal_overview_header_metadata(dd_mov_hdr, municipality="Dolné Dubové", district="Trnava")
        self.assertEqual(dd_mov_meta["revisor_name"], "Imrich Baron")

        # 3. Verify Dolné Dubové Obecný prehľad row parsing (including '-16' dash-ditto artifact on Spolu row)
        dd_r1 = _heuristic_parse_municipal_overview_row(
            "Vicee | I. | 2dn Habara | 72 | 71 | i | 74 | 381",
            seq_district_idx=1,
            default_settlement="Dolné Dubové",
            revisor_name="Imrich Baron",
        )
        self.assertEqual(dd_r1["obvod_number"], "I")
        self.assertEqual(dd_r1["commissioner_name"], "Ján Habara")
        self.assertEqual((dd_r1["houses_total"], dd_r1["houses_inhabited"], dd_r1["houses_uninhabited"], dd_r1["household_count"], dd_r1["present_persons_count"]), (72, 71, 1, 74, 381))

        dd_r2 = _heuristic_parse_municipal_overview_row(
            "Dubové Dolnie. | II | Autoría Martisovića | 75 | 71 | 4 | 76 | 350.",
            seq_district_idx=2,
            default_settlement="Dolné Dubové",
            revisor_name="Imrich Baron",
        )
        self.assertEqual(dd_r2["obvod_number"], "II")
        self.assertEqual(dd_r2["commissioner_name"], "Antónia Martišovičová")
        self.assertEqual((dd_r2["houses_total"], dd_r2["houses_inhabited"], dd_r2["houses_uninhabited"], dd_r2["household_count"], dd_r2["present_persons_count"]), (75, 71, 4, 76, 350))

        dd_sum = _heuristic_parse_municipal_overview_row(
            "Spolu: | - | -16 | 147 | 142 | 5 | 150 | 731",
            seq_district_idx=3,
            default_settlement="Dolné Dubové",
            revisor_name="Imrich Baron",
        )
        self.assertTrue(dd_sum["is_summary_row"])
        self.assertEqual((dd_sum["houses_total"], dd_sum["houses_inhabited"], dd_sum["houses_uninhabited"], dd_sum["household_count"], dd_sum["present_persons_count"]), (147, 142, 5, 150, 731))

        # 4. Verify Horné Dubové (cair-ko29x4e) 4-page bifolio reconstruction, uninhabited Dom č. 7,
        # parallel table_ocr + right_ocr reconciliation, 2-row sandwiched house number repair (20, 23, 20, 23 -> 20, 21, 22, 23),
        # and summary/footer suppression ('Sport: | 12 | 81...', 'v. Trmare di mag. | 10. rw | 1930.')
        self.assertTrue(is_overview_summary_or_footer_row("Sport: | 12 | 81 | 1 |  | 32 | 512 | 92 512"))
        self.assertTrue(is_overview_summary_or_footer_row("v. Trmare di mag. | 10. rw | 1930."))
        self.assertTrue(is_overview_summary_or_footer_row("scitaci komisari."))

        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            sheet = CensusSheet(
                slovakiana_key="cair-ko29x4e",
                census_year=1930,
                district="Trnava",
                municipality="Horné Dubové",
                municipality_folded="horne dubove",
                sheet_type="overview",
                obvod_number="I",
                commissioner_name="Margita Puchlová",
                download_status="extracted",
            )
            h1_sheet = CensusSheet(
                slovakiana_key="cair-ko26r7v",
                census_year=1930,
                district="Trnava",
                municipality="Horné Dubové",
                municipality_folded="horne dubove",
                house_number="1",
                house_number_int=1,
                sheet_type="household",
                download_status="cataloged",
            )
            h7_sheet = CensusSheet(
                slovakiana_key="cair-ko1g2g0",
                census_year=1930,
                district="Trnava",
                municipality="Horné Dubové",
                municipality_folded="horne dubove",
                house_number="7",
                house_number_int=7,
                sheet_type="household",
                download_status="cataloged",
            )
            session.add_all([sheet, h1_sheet, h7_sheet])
            session.commit()
            session.refresh(sheet)

            front_sc = ScanImage(
                census_sheet_id=sheet.id,
                page_side="front",
                filename="cair-ko29x4e_front.jpg",
                original_path="/nonexistent/cair-ko29x4e_front.jpg",
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            back_sc = ScanImage(
                census_sheet_id=sheet.id,
                page_side="back",
                filename="cair-ko29x4e_back.jpg",
                original_path="/nonexistent/cair-ko29x4e_back.jpg",
                detected_year="1930",
                detected_type="census",
                status="extracted",
            )
            session.add_all([front_sc, back_sc])
            session.commit()
            session.refresh(front_sc)
            session.refresh(back_sc)
            sheet.front_scan_id = front_sc.id
            sheet.back_scan_id = back_sc.id
            session.add(sheet)

            # Seed representative rows across Booklet Pages 1, 2, 3, and 4 (plus trailing summary/signature rows on Page 4)
            front_rows_spec = [
                (1, 1, False, "Dubore Harne | 1 | 1 |  | Rosalig Lihutov&#x27; | 1 | 6 | Roxaliq Léhutovq 1 6"),
                (2, 1, False, "20 | 1 |  | yerolim Whitk | 1 | 6 | Yecolim Ouhlih 1 6"),
                (3, 1, False, "3 | 1 |  | Ludovitch Siring April. | 3 | 17 | Ludovit Siring spol. 3 17"),
                (4, 1, False, "4 | 1 | Youref White | 1 | 6 | You of White 1 6"),
                (5, 1, False, "5 | 1 |  | Stefan Uhlhig | 1 | 4 | Stefan Uhlik 1 4"),
                (6, 1, False, "6 | 1 |  | Pavel White | 1 | 4 | Pavel Uhlik 1 4"),
                (7, 1, True, "7 | 1 | Cláring Boledorrió | Marius Boledovich"),
                (8, 1, False, "8 | 1 |  | 1 | 4 | Máring Boledo viétró 1 4"),
                (9, 4, False, "68 | 661 |  | Fishing Station | 1/4 | 5 | Filomeing Stalyvoyi 1 5"),
                (10, 4, False, "82 | 1 | Frontier Athlete | 1 | 6 | Frontier Militia 1 6"),
                (11, 4, False, "Sport: | 12 | 81 | 1 |  | 32 | 512 | 92 512"),
                (12, 4, False, "v. Trmare di mag. | 10. rw | 1930. | 1930."),
                (13, 4, False, "scitaci komisari."),
            ]
            back_rows_spec = [
                (1, 2, False, "20 | 1 |  | Dommik levcőrak | 1 | 9. | Domnik leveomak i 9."),
                (2, 2, False, "23 | 1 |  | A. mary Wristart | 4 | 8 | Amuch Whiting 1 8"),
                (3, 2, False, "20 |  |  | Stefan Writh | 1 | 4 | Stefan Whrik 1 4"),
                (4, 2, False, "23 |  |  | Chaing Hrehmarsv | 1 | 4 | Charles Murchison 1 4"),
                (5, 3, False, "51 |  | Anton Brochdinkey | 2 | 3 | obsc. Shorty | ton Prochidzky 2 3 obc."),
                (6, 3, False, "58 | Ludrit Ahlike Q 14"),
            ]
            for r_idx, bp, col5_u, txt in front_rows_spec:
                session.add(
                    RecordRow(
                        scan_id=front_sc.id or 0,
                        row_index=r_idx,
                        stitched_crop_path=f"/media/crops/scan_hd_f/row_{r_idx:02d}_stitched.jpg",
                        cell_crops_json=json.dumps([{"key": "overview_page", "booklet_page": bp, "col5_uninhabited": col5_u}]),
                        raw_ocr_left=txt,
                        raw_ocr_combined=txt,
                        status="ocr_done",
                    )
                )
            for r_idx, bp, col5_u, txt in back_rows_spec:
                session.add(
                    RecordRow(
                        scan_id=back_sc.id or 0,
                        row_index=r_idx,
                        stitched_crop_path=f"/media/crops/scan_hd_b/row_{r_idx:02d}_stitched.jpg",
                        cell_crops_json=json.dumps([{"key": "overview_page", "booklet_page": bp, "col5_uninhabited": col5_u}]),
                        raw_ocr_left=txt,
                        raw_ocr_combined=txt,
                        status="ocr_done",
                    )
                )
            session.commit()

            det = asyncio.run(get_census_sheet_detail("cair-ko29x4e", session))
            ov_houses = det["overview_houses"]
            # Trailing summary/footer rows (11, 12, 13 on front) must be suppressed -> 16 valid house rows
            self.assertEqual(len(ov_houses), 16)
            by_hn = {h["house_number"]: h for h in ov_houses}

            # Verify Dom č. 1 (Rozália Lehutová, settlement 'Horné Dubové', no leaked remarks, matched to cair-ko26r7v)
            self.assertEqual(by_hn["1"]["owner_name"], "Rozália Lehutová")
            self.assertEqual(by_hn["1"]["settlement_name"], "Horné Dubové")
            self.assertEqual(by_hn["1"]["household_count"], 1)
            self.assertEqual(by_hn["1"]["present_persons_count"], 6)
            self.assertIsNone(by_hn["1"]["remarks"])
            self.assertEqual(by_hn["1"]["matched_slovakiana_key"], "cair-ko26r7v")

            # Verify Dom č. 2 (house number 2 preserved despite '20 | 1' in table_ocr)
            self.assertEqual(by_hn["2"]["owner_name"], "Jerolím Uhlík")
            self.assertEqual((by_hn["2"]["household_count"], by_hn["2"]["present_persons_count"]), (1, 6))

            # Verify Dom č. 6 ('Pavel Uhlik' from right_ocr wins over 'Pavel White' without leaking into remarks)
            self.assertEqual(by_hn["6"]["owner_name"], "Pavel Uhlík")
            self.assertIsNone(by_hn["6"]["remarks"])

            # Verify Dom č. 7 (uninhabited: is_inhabited=False, hh=0, pp=0, matched to cair-ko1g2g0)
            self.assertEqual(by_hn["7"]["owner_name"], "Mária Boledovičová")
            self.assertFalse(by_hn["7"]["is_inhabited"])
            self.assertEqual((by_hn["7"]["household_count"], by_hn["7"]["present_persons_count"]), (0, 0))
            self.assertEqual(by_hn["7"]["matched_slovakiana_key"], "cair-ko1g2g0")

            # Verify Dom č. 8 (owner recovered from right_ocr with exact counts 1, 4)
            self.assertEqual(by_hn["8"]["owner_name"], "Mária Boledovičová")
            self.assertEqual((by_hn["8"]["household_count"], by_hn["8"]["present_persons_count"]), (1, 4))

            # Verify Booklet Page 3 remarks & counts (Anton Prochádzka with 'obec. chudobinec' and Ľudovít Uhlík with 2, 14)
            proch = next(h for h in ov_houses if h["owner_name"] == "Anton Prochádzka")
            self.assertEqual((proch["household_count"], proch["present_persons_count"], proch["remarks"]), (2, 3, "obec. chudobinec"))
            lud_u = next(h for h in ov_houses if h["owner_name"] == "Ľudovít Uhlík")
            self.assertEqual((lud_u["household_count"], lud_u["present_persons_count"]), (2, 14))

            # Verify Booklet Page 4 houses (Filoména Slabáková 1, 5 and František Mihálik 1, 6)
            fil = next(h for h in ov_houses if h["owner_name"] == "Filoména Slabáková")
            self.assertEqual((fil["household_count"], fil["present_persons_count"]), (1, 5))
            mih = next(h for h in ov_houses if h["owner_name"] == "František Mihálik")
            self.assertEqual((mih["household_count"], mih["present_persons_count"]), (1, 6))

            muni_ov = get_municipality_overview(session, municipality="Horné Dubové", census_year=1930)
            self.assertEqual(len(muni_ov["overview_sheets"]), 1)
            self.assertEqual(muni_ov["overview_sheets"][0]["commissioner_name"], "Margita Puchlová")

            # 5. Simulate legacy state where RecordRow.raw_ocr_combined only had table_ocr (without right_ocr),
            # causing rows like Dom č. 8 ('8 | 1 | | 1 | 4') to be dropped from CensusOverviewHouse, while
            # WorkflowTrace had both ocr_left and ocr_right traces. Verify get_census_sheet_detail self-heals all 16 houses!
            from sqlmodel import select
            from backend.app.db.models import CensusOverviewHouse, WorkflowTrace

            for r_obj in session.exec(select(RecordRow).where(RecordRow.scan_id == front_sc.id)).all():
                if r_obj.row_index == 8:
                    r_obj.raw_ocr_left = "8 | 1 |  | 1 | 4"
                    r_obj.raw_ocr_combined = "8 | 1 |  | 1 | 4"
                    session.add(r_obj)
                    session.add(
                        WorkflowTrace(
                            scan_id=front_sc.id,
                            row_id=r_obj.id,
                            stage="ocr_left",
                            model_or_tool="glm-ocr",
                            status="ok",
                            image_path=f"/media/crops/scan_{front_sc.id}/row_08_left.jpg",
                            response_text="8 | 1 |  | 1 | 4",
                        )
                    )
                    session.add(
                        WorkflowTrace(
                            scan_id=front_sc.id,
                            row_id=r_obj.id,
                            stage="ocr_right",
                            model_or_tool="glm-ocr",
                            status="ok",
                            image_path=f"/media/crops/scan_{front_sc.id}/row_08_right.jpg",
                            response_text="Máring Boledo viétró 1 4",
                        )
                    )
                else:
                    session.add(
                        WorkflowTrace(
                            scan_id=front_sc.id,
                            row_id=r_obj.id,
                            stage="ocr_left",
                            model_or_tool="glm-ocr",
                            status="ok",
                            image_path=f"/media/crops/scan_{front_sc.id}/row_{r_obj.row_index:02d}_left.jpg",
                            response_text=r_obj.raw_ocr_combined,
                        )
                    )
            for r_obj in session.exec(select(RecordRow).where(RecordRow.scan_id == back_sc.id)).all():
                session.add(
                    WorkflowTrace(
                        scan_id=back_sc.id,
                        row_id=r_obj.id,
                        stage="ocr_left",
                        model_or_tool="glm-ocr",
                        status="ok",
                        image_path=f"/media/crops/scan_{back_sc.id}/row_{r_obj.row_index:02d}_left.jpg",
                        response_text=r_obj.raw_ocr_combined,
                    )
                )
            for ovh_del in session.exec(
                select(CensusOverviewHouse).where(
                    CensusOverviewHouse.census_sheet_id == sheet.id,
                    CensusOverviewHouse.house_number == "8",
                )
            ).all():
                session.delete(ovh_del)
            session.commit()

            det_healed = asyncio.run(get_census_sheet_detail("cair-ko29x4e", session))
            healed_by_hn = {h["house_number"]: h for h in det_healed["overview_houses"]}
            self.assertEqual(len(det_healed["overview_houses"]), 16)
            self.assertIn("8", healed_by_hn)
            self.assertEqual(healed_by_hn["8"]["owner_name"], "Mária Boledovičová")
            self.assertEqual((healed_by_hn["8"]["household_count"], healed_by_hn["8"]["present_persons_count"]), (1, 4))

    def test_search_and_ocr_linked_sheets_by_house_owner_name(self) -> None:
        from backend.app.api.census import (
            _resolve_owner_linked_household_sheets,
            list_census_sheets,
        )
        from backend.app.census.worker import _find_matching_census_sheets
        from backend.app.db.models import CensusOverviewHouse
        from backend.app.search.fts import search_unified

        test_engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(test_engine)
        ensure_fts_schema(test_engine)

        with Session(test_engine) as session:
            ov_sheet = CensusSheet(
                slovakiana_key="cair-owner-ov1",
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                municipality_folded="testobec",
                sheet_type="overview",
                download_status="extracted",
                has_images=True,
            )
            hh_h7 = CensusSheet(
                slovakiana_key="cair-owner-h7",
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                municipality_folded="testobec",
                house_number="7",
                house_number_int=7,
                sheet_type="household",
                download_status="cataloged",
                has_images=True,
            )
            hh_h8 = CensusSheet(
                slovakiana_key="cair-owner-h8",
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                municipality_folded="testobec",
                house_number="8",
                house_number_int=8,
                sheet_type="household",
                download_status="cataloged",
                has_images=True,
            )
            hh_h9_other = CensusSheet(
                slovakiana_key="cair-owner-h9",
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                municipality_folded="testobec",
                house_number="9",
                house_number_int=9,
                sheet_type="household",
                download_status="cataloged",
                has_images=True,
            )
            session.add_all([ov_sheet, hh_h7, hh_h8, hh_h9_other])
            session.commit()
            for s in (ov_sheet, hh_h7, hh_h8, hh_h9_other):
                session.refresh(s)

            # Note: matched_sheet_id is intentionally left None on h8 to verify dynamic house-number linking!
            ovh7 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 1,
                row_index=1,
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                house_number="7",
                house_number_int=7,
                owner_name="Mária Boledovičová",
                owner_surname_folded="boledovic",
                matched_sheet_id=hh_h7.id,
            )
            ovh8 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 1,
                row_index=2,
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                house_number="8",
                house_number_int=8,
                owner_name="Mária Boledovičová",
                owner_surname_folded="boledovic",
                matched_sheet_id=None,
            )
            ovh9 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 1,
                row_index=3,
                census_year=1930,
                district="Trnava",
                municipality="TestObec",
                house_number="9",
                house_number_int=9,
                owner_name="Štefan Kováč",
                owner_surname_folded="kovac",
                matched_sheet_id=None,
            )
            session.add_all([ovh7, ovh8, ovh9])
            session.commit()

            # 1. Searching census sheets by owner name returns both linked household sheets (even when matched_sheet_id is None)
            sheets_res = list_census_sheets(
                session,
                census_year=1930,
                municipality="TestObec",
                person_name="Boledovičová",
            )
            returned_keys = {item["sheet"]["slovakiana_id"] for item in sheets_res["sheets"]}
            self.assertIn("cair-owner-h7", returned_keys)
            self.assertIn("cair-owner-h8", returned_keys)
            self.assertNotIn("cair-owner-h9", returned_keys)

            # And populates household_owner fallback from the overview sheet onto the un-OCRed household sheet payload
            h8_payload = next(
                item["sheet"]
                for item in sheets_res["sheets"]
                if item["sheet"]["slovakiana_id"] == "cair-owner-h8"
            )
            self.assertEqual(h8_payload["household_owner"], "Mária Boledovičová")

            # 2. Unified search catalog_sheets also includes both linked household sheets
            uni_res = search_unified(session, q="Boledovič", municipality="TestObec")
            uni_cat_keys = {cs.slovakiana_key for cs in uni_res["catalog_sheets"]}
            self.assertIn("cair-owner-h7", uni_cat_keys)
            self.assertIn("cair-owner-h8", uni_cat_keys)
            self.assertNotIn("cair-owner-h9", uni_cat_keys)

            # 3. _resolve_owner_linked_household_sheets resolves only the household sheets where Boledovičová is house owner
            owner_sheets = _resolve_owner_linked_household_sheets(
                session,
                owner_query="Boledovič",
                census_year=1930,
                municipality="TestObec",
            )
            self.assertEqual(
                [s.slovakiana_key for s in owner_sheets],
                ["cair-owner-h7", "cair-owner-h8"],
            )

            # 4. Worker _find_matching_census_sheets respects explicit sheet_ids for selective owner-linked OCR jobs
            job_filters = {
                "years": [1930],
                "municipality": "TestObec",
                "owner_query": "Boledovič",
                "sheet_ids": [s.id for s in owner_sheets],
            }
            worker_targets = _find_matching_census_sheets(
                session, job_filters, only_undone=True, run_ocr=True
            )
            self.assertEqual(
                [s.slovakiana_key for s in worker_targets],
                ["cair-owner-h7", "cair-owner-h8"],
            )

    def test_overview_extraction_does_not_deadlock_sqlite_with_trace_persistence(self) -> None:
        """
        Regression test: When extracting an overview sheet (or municipal overview sheet) on a
        file-backed SQLite database, `session` must never hold an uncommitted autoflushed write
        transaction while `ai_client.chat_completion` persists `WorkflowTrace` in a separate
        thread/session (`sqlite3.OperationalError: database is locked`).
        """
        import asyncio
        import tempfile
        from unittest.mock import patch
        import cv2
        import numpy as np
        from sqlalchemy import event
        from backend.app.config import settings
        from backend.app.db.models import CensusOverviewHouse, ScanImage, WorkflowTrace
        from backend.app.db.session import _set_sqlite_pragma
        from backend.app.pipeline.ai_client import ai_client
        from backend.app.pipeline.extraction_stage import extract_census_overview_sheet

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            db_file = tmp_path / "lock_regression.db"
            crops_dir = tmp_path / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            scan_path = tmp_path / "ov_front.jpg"

            w, h = 1600, 1200
            synth_img = np.full((h, w, 3), 245, dtype=np.uint8)
            cv2.line(synth_img, (40, 420), (w - 40, 420), (15, 15, 15), 6)
            for r_i in range(1, 4):
                y_line = 420 + r_i * 60
                cv2.line(synth_img, (40, y_line), (w - 40, y_line), (70, 70, 70), 2)
            for r_i in range(2):
                ry = 420 + r_i * 60 + 38
                hn = r_i + 1
                cv2.putText(synth_img, f"{hn}. Topolcany {hn}", (int(w * 0.06), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
                cv2.putText(synth_img, f"Jan Kovac {hn}", (int(w * 0.48), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 10, 10), 3)
                cv2.putText(synth_img, "1  4", (int(w * 0.74), ry), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 10, 10), 2)
            cv2.imwrite(str(scan_path), synth_img)

            # Use a short 1-second timeout so any self-deadlock between `session` and `_persist_trace` fails immediately
            file_engine = create_engine(
                f"sqlite:///{db_file}",
                connect_args={"check_same_thread": False, "timeout": 1},
            )
            event.listen(file_engine, "connect", _set_sqlite_pragma)
            SQLModel.metadata.create_all(file_engine)
            ensure_fts_schema(file_engine)

            with Session(file_engine) as session:
                sheet = CensusSheet(
                    id=408,
                    slovakiana_key="cair-lock-408",
                    census_year=1930,
                    district="Topoľčany",
                    municipality="topoľčany",  # Lowercase triggers canonical_municipality_display update on `sheet`!
                    sheet_type="overview",
                    download_status="downloaded",
                    has_images=True,
                )
                session.add(sheet)
                session.commit()

                front_scan = ScanImage(
                    batch_id=1,
                    census_sheet_id=408,
                    page_side="front",
                    filename="cair-lock-408_front.jpg",
                    original_path=str(scan_path),
                    detected_year="1930",
                    detected_type="census",
                    status="uploaded",
                )
                session.add(front_scan)
                session.commit()
                session.refresh(front_scan)
                sheet.front_scan_id = front_scan.id
                session.add(sheet)
                session.commit()

            class _FakeResp:
                status_code = 200

                def raise_for_status(self) -> None:
                    pass

                def json(self) -> dict[str, object]:
                    # Return a garbled owner on table recognition so the `extract` LLM stage is also invoked!
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"ordinal_number": "1", "settlement_name": "Topoľčany", "house_number": "1", "is_inhabited": true, "owner_name": "Ján Kováč", "household_count": 1, "present_persons_count": 4}'
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }

            class _FakeHttpClient:
                def __init__(self) -> None:
                    self.calls = 0

                async def post(self, *args: object, **kwargs: object) -> _FakeResp:
                    self.calls += 1
                    payload = kwargs.get("json") or {}
                    msgs = payload.get("messages") if isinstance(payload, dict) else []
                    is_extract = any(
                        isinstance(m, dict) and m.get("role") == "system" and "Prehľad sčítacieho obvodu" in str(m.get("content") or "")
                        for m in (msgs or [])
                    )
                    if not is_extract:
                        class _OcrResp(_FakeResp):
                            def json(self) -> dict[str, object]:
                                return {
                                    "choices": [
                                        {
                                            "message": {
                                                "content": "<table><tr><td>1.</td><td>Topoľčany</td><td>1</td><td>obývaný</td><td>—</td><td>1</td><td>4</td></tr></table>"
                                            }
                                        }
                                    ],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                                }
                        return _OcrResp()
                    return _FakeResp()

            with (
                patch("backend.app.pipeline.ai_client.engine", file_engine),
                patch.object(ai_client, "_get_http_client", return_value=_FakeHttpClient()),
                patch.object(settings, "crops_dir", crops_dir),
                patch("backend.app.pipeline.ocr_stage.settings.crops_dir", crops_dir),
            ):
                with Session(file_engine) as session:
                    sh = session.get(CensusSheet, 408)
                    sc = session.get(ScanImage, sh.front_scan_id) if sh else None
                    assert sh is not None and sc is not None
                    houses = asyncio.run(
                        extract_census_overview_sheet(
                            session,
                            sh,
                            sc,
                            None,
                            force_district_overview=True,
                        )
                    )
                    self.assertGreaterEqual(len(houses), 1)

                with Session(file_engine) as session:
                    sh_after = session.get(CensusSheet, 408)
                    self.assertIsNotNone(sh_after)
                    self.assertEqual(sh_after.download_status, "extracted")  # type: ignore[union-attr]
                    traces = session.exec(select(WorkflowTrace)).all()
                    self.assertGreater(len(traces), 0)
                    ov_rows = session.exec(
                        select(CensusOverviewHouse).where(CensusOverviewHouse.census_sheet_id == 408)
                    ).all()
                    self.assertGreaterEqual(len(ov_rows), 1)
            file_engine.dispose()


if __name__ == "__main__":
    unittest.main()
