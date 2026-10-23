from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from sqlmodel import Session, SQLModel, create_engine

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
            batch_res = asyncio.run(process_batch_stage_queued(batch_id, run_mcp=False))
            self.assertEqual(batch_res["scans_processed"], 1)
            self.assertEqual(batch_res["scans_skipped"], 1)
            self.assertEqual(ocr_called_scan_ids, [sc_todo_id])



if __name__ == "__main__":
    unittest.main()
