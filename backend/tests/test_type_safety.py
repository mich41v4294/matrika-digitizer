import tempfile
from pathlib import Path
import unittest

from backend.app.config import Settings
from backend.app.db.models import PersonFact
from backend.app.export.gedcom import export_facts_to_gedcom
from backend.app.pipeline.extraction_stage import (
    _heuristic_fallback_extract,
    reconcile_extraction_with_type_safety,
)
from backend.app.pipeline.lexicon import (
    match_given_name_to_area_period,
    match_surname_to_area_period,
    normalize_given_name,
    normalize_surname,
    parse_observationes_events,
    record_observed_fs_names,
    reset_fs_name_cache_for_tests,
)
from backend.app.pipeline.ocr_stage import media_rel_to_path
from backend.app.pipeline.schemas import (
    LLMDecisionSchema,
    LLMExtractionSchema,
    apply_human_answer_to_fields,
    parse_parents_structured,
    validate_house_number,
    validate_matrika_date,
    validate_person_name,
    validate_sex,
)


class TestLLMTypeSafety(unittest.TestCase):
    def test_swapped_date_and_father_name_paulus_augustus(self) -> None:
        """
        Regression test for user-reported case:
        birth_or_event_date='Paulus 23' and father_name='Augustus'
        must be repaired so Augustus goes to date ('Augustus 23') and Paulus goes to father_name.
        """
        raw_llm = {
            "record_type": "baptism",
            "entry_number": "52",
            "event_year": 1884,
            "birth_or_event_date": "Paulus 23",
            "baptism_or_burial_date": "28.",
            "given_name_raw": "Stephanus",
            "surname_raw": "Novotny",
            "sex": "male",
            "father_name": "Augustus",
            "mother_name": "Cesett",
        }
        schema = LLMExtractionSchema.model_validate(raw_llm)
        self.assertEqual(schema.birth_or_event_date, "Augustus 23")
        self.assertEqual(schema.father_name, "Novotny Paulus")
        self.assertEqual(schema.sex, "M")
        self.assertTrue(
            any("Cross-field swap:" in r for r in schema.type_safety_repairs),
            f"Expected cross-field swap repair note in {schema.type_safety_repairs}",
        )

    def test_month_and_occupation_rejected_as_person_names(self) -> None:
        self.assertEqual(validate_person_name("Augustus", role="father"), (None, ["Augustus"]))
        self.assertEqual(validate_person_name("Septembris", role="given"), (None, ["September"]))
        self.assertEqual(validate_person_name("Sutius 28", role="father"), (None, ["Julius 28"]))
        self.assertEqual(validate_person_name("sutor", role="surname"), (None, []))
        self.assertEqual(validate_person_name("Boarins", role="surname"), (None, []))
        self.assertEqual(validate_person_name("Boarius", role="surname"), (None, []))
        self.assertEqual(validate_person_name(".", role="father"), (None, []))
        self.assertEqual(validate_person_name("/", role="mother"), (None, []))

    def test_gender_constraints_on_parents(self) -> None:
        # Male given name alone cannot be mother_name
        self.assertEqual(validate_person_name("Paulus", role="mother"), (None, []))
        self.assertEqual(validate_person_name("Stephanus", role="mother"), (None, []))
        # Female given name alone cannot be father_name
        self.assertEqual(validate_person_name("Catharina", role="father"), (None, []))
        self.assertEqual(validate_person_name("Maria", role="father"), (None, []))
        # Valid combinations pass
        self.assertEqual(validate_person_name("Novotny Paulus", role="father"), ("Novotny Paulus", []))
        self.assertEqual(validate_person_name("Hudta Catharina", role="mother"), ("Hudta Catharina", []))

    def test_date_validator_rejects_person_names_and_extracts_valid_date(self) -> None:
        # Pure person/occupation string -> None date, misplaced names returned
        d_val, names = validate_matrika_date("Kidveny Gaulus sutor")
        self.assertIsNone(d_val)
        self.assertIn("Gaulus", names)

        # Mixed person + day number -> day extracted, person name captured
        d_val2, names2 = validate_matrika_date("Paulus 23")
        self.assertEqual(d_val2, "23.")
        self.assertIn("Paulus", names2)

        # OCR misread month 'Sutius 28' -> normalized to 'Julius 28'
        d_val3, names3 = validate_matrika_date("Sutius 28")
        self.assertEqual(d_val3, "Julius 28")
        self.assertEqual(names3, [])

        # Year or 3-digit noise rejected as date
        self.assertIsNone(validate_matrika_date("1893")[0])
        self.assertIsNone(validate_matrika_date("894.")[0])

    def test_parse_parents_structured_pairs(self) -> None:
        p1 = parse_parents_structured("Mucejtha Andreas Hudta Catharina, familii")
        self.assertEqual(p1["father_name"], "Mucejtha Andreas")
        self.assertEqual(p1["mother_name"], "Hudta Catharina")
        self.assertEqual(p1["mother_maiden_surname"], "Hudta")
        self.assertEqual(p1["surname_raw"], "Mucejtha")

        p2 = parse_parents_structured("Savlik Terdinandus. Schmitta Hermina, cv. a.e.")
        self.assertEqual(p2["father_name"], "Savlik Terdinandus")
        self.assertEqual(p2["mother_name"], "Schmitta Hermina")
        self.assertEqual(p2["mother_maiden_surname"], "Schmitta")
        self.assertEqual(p2["surname_raw"], "Savlik")

    def test_rescue_parents_misplaced_into_village_or_godparents(self) -> None:
        fb = _heuristic_fallback_extract(
            "59. | 12. | 12. | Anna. | . | / | / | Sleep Sigmanus, or Jadák Catharina, cv. a.e",
            "Hisellodro H1. | Stravárit Catharin colm'ev. a.e. O Lehot | Indelp. etinna c. par ochus 6 Lehot | Moséer No. 28",
            "1884",
            "baptism",
        )
        llm_raw = {
            "entry_number": "59",
            "birth_or_event_date": "12.",
            "baptism_or_burial_date": "12.",
            "given_name_raw": "Anna",
            "sex": "F",
            "father_name": "/",
            "mother_name": "/",
            "mother_maiden_surname": ".",
            "village_raw": "Sleep Sigmanus, or Jadák Catharina, cv. a.e",
            "house_number": "1",
        }
        merged = reconcile_extraction_with_type_safety(llm_raw, fb, page_year="1884")
        self.assertEqual(merged["father_name"], "Sleep Sigmanus")
        self.assertEqual(merged["mother_name"], "Jadák Catharina")
        self.assertEqual(merged["mother_maiden_surname"], "Jadák")
        self.assertEqual(merged["village_raw"], "Hisellodro H1.")
        self.assertEqual(merged["house_number"], "1")

    def test_house_number_and_sex_validation(self) -> None:
        self.assertEqual(validate_house_number("1", "Vaé Níhhely No 295."), "295")
        self.assertIsNone(validate_house_number("1749", "Hradett 1749."))
        self.assertEqual(validate_sex("F", "Joannes"), "M")
        self.assertEqual(validate_sex("male", "Andreas"), "M")

    def test_decision_schema_strips_slovak_prompt_echoes(self) -> None:
        raw_dec = {
            "action": "ask_human",
            "confidence": 0.65,
            "uncertain_field": "parents",
            "reason": "Krátke vysvetlenie v slovenčine prečo si model nie je istý",
            "question_sk": "Konkrétna otázka v slovenčine pre používateľa (napr. Rodičia v zápise č. 52 – je priezvisko matky Cserep alebo Cseret?)",
            "candidates": ["Presný variant 1 z riadku", "Variant 2", "Variant 3"],
        }
        dec = LLMDecisionSchema.model_validate(raw_dec)
        self.assertEqual(dec.reason, "")
        self.assertEqual(dec.question_sk, "")
        self.assertEqual(dec.candidates, [])

    def test_apply_human_answer_to_fields_type_safe(self) -> None:
        updates_date = apply_human_answer_to_fields({}, "date_name", "28. Julius / 28. Julius")
        self.assertEqual(updates_date.get("birth_or_event_date"), "28. Julius")
        self.assertEqual(updates_date.get("baptism_or_burial_date"), "28. Julius")
        self.assertNotIn("given_name_raw", updates_date)

        updates_parents = apply_human_answer_to_fields({}, "parents", "Novotny Paulus et Cesett")
        self.assertEqual(updates_parents.get("father_name"), "Novotny Paulus")
        self.assertEqual(updates_parents.get("mother_name"), "Cesett")

    def test_observationes_events_schema_and_gedcom_export(self) -> None:
        obs_events = parse_observationes_events("† 1924 v Trnave; matrimon. contraxit 1908")
        self.assertEqual(len(obs_events), 2)
        self.assertEqual(obs_events[0]["event_type"], "death")
        self.assertEqual(obs_events[0]["type"], "death")
        self.assertEqual(obs_events[0]["date"], "1924")
        self.assertEqual(obs_events[1]["event_type"], "marriage")
        self.assertEqual(obs_events[1]["date"], "1908")

        fact_bapt = PersonFact(
            id=1,
            row_id=1,
            scan_id=1,
            record_type="baptism",
            event_year=1884,
            given_name_raw="Stephanus",
            surname_raw="Novotny",
            sex="M",
            father_name="Novotny Paulus",
            mother_name="Hudta Catharina",
            mother_maiden_surname="Hudta",
        )
        fact_marr = PersonFact(
            id=2,
            row_id=2,
            scan_id=1,
            record_type="marriage",
            event_year=1890,
            birth_or_event_date="12. Majus",
            given_name_raw="Andreas",
            surname_raw="Mucska",
            sex="M",
        )
        ged = export_facts_to_gedcom([fact_bapt, fact_marr])
        self.assertIn("1 NAME Paulus /Novotny/", ged)
        self.assertIn("1 NAME Catharina /Hudta/", ged)
        self.assertIn("2 TYPE Marriage", ged)
        self.assertIn("2 DATE 12. Majus", ged)

    def test_settings_data_subpaths_and_media_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            s = Settings(data_dir=tmp_path, db_path=None)
            self.assertEqual(s.upload_dir, tmp_path / "uploads")
            self.assertEqual(s.matrika_data_dir, tmp_path / "uploads" / "matrika")
            self.assertEqual(s.census_data_dir, tmp_path / "uploads" / "slovakiana")
            self.assertEqual(s.crops_dir, tmp_path / "crops")
            self.assertEqual(s.db_path, tmp_path / "matrika.db")
            self.assertEqual(s.fs_env_file, tmp_path / "familysearch.env")
            s.ensure_dirs()
            self.assertTrue(s.fs_env_file.exists())
            self.assertTrue(s.matrika_data_dir.exists())

        self.assertIsNone(media_rel_to_path("/media/crops/../../../etc/passwd"))
        self.assertIsNone(media_rel_to_path("/media/uploads/slovakiana/../../../../etc/passwd"))
        self.assertIsNone(media_rel_to_path("/media/uploads/matrika/../../../../etc/passwd"))
        self.assertIsNone(media_rel_to_path("/etc/passwd"))

    def test_api_endpoints_and_spa_route_protection(self) -> None:
        import shutil
        from fastapi.testclient import TestClient
        from backend.app.api.scans import resolve_scan_media_url
        from backend.app.config import settings
        from backend.app.db.session import init_db
        from backend.app.main import app

        init_db()
        client = TestClient(app)

        # 1. Health check
        r_health = client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.json()["status"], "ok")
        self.assertTrue(bool(r_health.json().get("build_commit")))
        self.assertNotEqual(r_health.json().get("build_commit"), "dev")

        # 2. Unknown /api/* route must return 404, not SPA index.html
        r_missing_api = client.get("/api/does-not-exist")
        self.assertEqual(r_missing_api.status_code, 404)

        # 3. Create upload batch, reject disallowed extension (.html), and verify organized Matrika folder + manifest
        r_batch = client.post(
            "/api/uploads/batches",
            json={
                "name": "Krsty 1880-1890",
                "parish_or_book": "Farnosť Testová",
                "record_type_hint": "baptism",
            },
        )
        self.assertEqual(r_batch.status_code, 200)
        batch_id = r_batch.json()["batch_id"]
        batch_folder_path: Path | None = None
        try:
            r_bad_ext = client.post(
                "/api/uploads/chunk",
                data={"batch_id": str(batch_id)},
                files=[("files", ("evil.html", b"<script>alert(1)</script>", "text/html"))],
            )
            self.assertEqual(r_bad_ext.status_code, 400)

            minimal_png = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
                b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            r_ok = client.post(
                "/api/uploads/chunk",
                data={"batch_id": str(batch_id)},
                files=[("files", ("page_01.png", minimal_png, "image/png"))],
            )
            self.assertEqual(r_ok.status_code, 200)
            ok_json = r_ok.json()
            self.assertEqual(ok_json["uploaded_count"], 1)
            self.assertIn(f"_batch_{batch_id:04d}_farnost_testova_krsty_1880_1890_baptism", ok_json["batch_folder"])

            batch_folder_path = Path(settings.matrika_data_dir) / ok_json["batch_folder"]
            self.assertTrue((batch_folder_path / "001_page_01.png").exists())
            self.assertTrue((batch_folder_path / "batch_manifest.json").exists())
            media_url = resolve_scan_media_url(str(batch_folder_path / "001_page_01.png"))
            self.assertEqual(media_url, f"/media/uploads/matrika/{ok_json['batch_folder']}/001_page_01.png")
        finally:
            from sqlmodel import Session, select
            from backend.app.db.models import Batch, ScanImage
            from backend.app.db.session import engine

            if batch_folder_path and batch_folder_path.exists():
                shutil.rmtree(batch_folder_path, ignore_errors=True)
            with Session(engine) as session:
                for sc in session.exec(select(ScanImage).where(ScanImage.batch_id == batch_id)).all():
                    session.delete(sc)
                b = session.get(Batch, batch_id)
                if b:
                    session.delete(b)
                session.commit()

        # 4. Trace stage filter aliases (ocr, mcp_search)
        r_tr_ocr = client.get("/api/debug/traces?stage=ocr")
        self.assertEqual(r_tr_ocr.status_code, 200)
        r_tr_mcp = client.get("/api/debug/traces?stage=mcp_search")
        self.assertEqual(r_tr_mcp.status_code, 200)


class TestAreaPeriodNameMatching(unittest.TestCase):
    def test_yargo_and_kowacs_surname_matches(self) -> None:
        varga, v_score = match_surname_to_area_period("Yargo", area="Nyitra", year=1884)
        self.assertEqual(varga, "Varga")
        self.assertGreaterEqual(v_score, 0.82)

        kovacs, s_score = match_surname_to_area_period("Kowacs", area="Nyitra", year=1884)
        self.assertEqual(kovacs, "Kovács")
        self.assertGreaterEqual(s_score, 0.82)

        self.assertEqual(normalize_surname("Yargo")[0], "Varga")
        self.assertEqual(normalize_surname("Kowacs")[0], "Kovács")
        self.assertEqual(normalize_surname("Varga")[0], "Varga")

    def test_given_name_ocr_match_preserves_latin_forms(self) -> None:
        andreas, a_score = match_given_name_to_area_period("Andreas", area="Nyitra", year=1884)
        self.assertEqual(andreas, "Andreas")
        self.assertGreaterEqual(a_score, 0.99)
        self.assertEqual(normalize_given_name("Andreas")[0], "Andreas")
        self.assertEqual(normalize_given_name("Joarnes")[0], "Joannes")
        yoannes, y_score = match_given_name_to_area_period("Yoannes", year=1884)
        self.assertEqual(yoannes, "Joannes")
        self.assertGreaterEqual(y_score, 0.82)

    def test_parse_parents_auto_corrects_ocr_surnames(self) -> None:
        parsed = parse_parents_structured(
            "Yargo Andreas et Kowacs Catharina",
            area="Nyitra",
            year=1884,
        )
        self.assertEqual(parsed["father_name"], "Varga Andreas")
        self.assertEqual(parsed["surname_raw"], "Varga")
        self.assertEqual(parsed["mother_name"], "Kovács Catharina")
        self.assertEqual(parsed["mother_maiden_surname"], "Kovács")

    def test_llm_schema_and_reconcile_log_repairs(self) -> None:
        raw_llm = {
            "entry_number": "51",
            "event_year": 1884,
            "birth_or_event_date": "18. Augustus",
            "given_name_raw": "Stephanus",
            "surname_raw": "Yargo",
            "father_name": "Yargo Andreas",
            "mother_name": "Kowacs Catharina",
            "mother_maiden_surname": "Kowacs",
            "village_raw": "Nyitra",
        }
        schema = LLMExtractionSchema.model_validate(raw_llm)
        self.assertEqual(schema.surname_raw, "Varga")
        self.assertEqual(schema.father_name, "Varga Andreas")
        self.assertEqual(schema.mother_name, "Kovács Catharina")
        self.assertEqual(schema.mother_maiden_surname, "Kovács")
        self.assertTrue(
            any("Yargo" in r and "Varga" in r for r in schema.type_safety_repairs),
            schema.type_safety_repairs,
        )
        self.assertTrue(
            any("Kowacs" in r and "Kovács" in r for r in schema.type_safety_repairs),
            schema.type_safety_repairs,
        )

        fallback = _heuristic_fallback_extract(
            "51. | 18. Augustus | Stephanus | Yargo Andreas et Kowacs Catharina",
            "Nyitra No. 12.",
            "1884",
            "baptism",
        )
        merged = reconcile_extraction_with_type_safety(raw_llm, fallback, page_year="1884")
        self.assertEqual(merged["surname_raw"], "Varga")
        self.assertEqual(merged["father_name"], "Varga Andreas")
        self.assertEqual(merged["mother_maiden_surname"], "Kovács")
        self.assertTrue(
            any("FamilySearch area/period match" in r for r in merged["type_safety_repairs"]),
            merged["type_safety_repairs"],
        )

    def test_dynamic_frequency_cache_feeds_matcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fs_area_names_cache.json"
            reset_fs_name_cache_for_tests(cache_path)
            try:
                record_observed_fs_names(
                    "nyitra",
                    1884,
                    surnames=["Paczolajsky", "Paczolajsky", "Paczolajsky"],
                    given_names=["Andreas"],
                )
                matched, score = match_surname_to_area_period(
                    "Paczolajskv", area="Nyitra", year=1884
                )
                self.assertEqual(matched, "Paczolajsky")
                self.assertGreaterEqual(score, 0.82)
                self.assertTrue(cache_path.exists())
            finally:
                reset_fs_name_cache_for_tests(None)


if __name__ == "__main__":
    unittest.main()
