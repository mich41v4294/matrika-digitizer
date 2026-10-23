from __future__ import annotations

import unittest

from fastapi import BackgroundTasks
from sqlmodel import Session, SQLModel, create_engine

import backend.app.api  # noqa: F401
from backend.app.api.census import (
    _find_linked_house_sheets,
    _find_same_year_house_sheets,
    _serialize_overview_house,
    get_municipality_overview,
    list_census_sheets,
)
from backend.app.census.slovakiana_client import parse_box_and_sheet_from_title
from backend.app.db.models import CensusOverviewHouse, CensusSheet, PersonFact
from backend.app.pipeline.extraction_stage import reconcile_census_household_relationships
from backend.app.pipeline.lexicon import (
    can_match_overview_house_to_sheet,
    canonical_municipality_display,
    canonical_municipality_key,
    compute_person_fact_search_fields,
    normalize_house_number,
)
from backend.app.search.fts import ensure_fts_schema


class TestCensusVillageHouseMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        ensure_fts_schema(self.engine)

    def test_canonical_municipality_and_house_normalization(self) -> None:
        self.assertEqual(canonical_municipality_key("Horný Čepen"), "horny cepen")
        self.assertEqual(canonical_municipality_key("Horný Čepeň"), "horny cepen")
        self.assertEqual(canonical_municipality_display("Horný Čepen"), "Horný Čepeň")
        self.assertEqual(canonical_municipality_key("Horné Zelenice"), "horne zelenice")
        self.assertNotEqual(
            canonical_municipality_key("Horný Čepeň"),
            canonical_municipality_key("Horné Zelenice"),
        )
        self.assertEqual(normalize_house_number("026"), "26")
        self.assertEqual(normalize_house_number("č. d. 26"), "26")
        self.assertEqual(
            parse_box_and_sheet_from_title(
                "Sčítací hárok 462/32092, Horné Zelenice (Hlohovec)",
                house_number="92",
                census_year=1940,
            ),
            ("0462", "0320"),
        )

    def test_cross_village_overview_matching_rejected(self) -> None:
        ov_sheet = CensusSheet(
            id=10,
            slovakiana_key="cair-ovw0001",
            census_year=1930,
            municipality="Veľké Ripňany",
            municipality_key="2742",
            sheet_type="overview",
        )
        ov_house_26 = CensusOverviewHouse(
            id=26,
            census_sheet_id=10,
            row_index=26,
            census_year=1930,
            municipality="Veľké Ripňany",
            settlement_name="Veľké Ripňany",
            house_number="26",
            house_number_int=26,
        )
        other_village_26 = CensusSheet(
            id=20,
            slovakiana_key="cair-oth0026",
            census_year=1940,
            municipality="Horné Zelenice",
            municipality_key="7647",
            house_number="26",
            house_number_int=26,
            sheet_type="household",
        )
        same_village_26 = CensusSheet(
            id=30,
            slovakiana_key="cair-rip0026",
            census_year=1930,
            municipality="Veľké Ripňany",
            municipality_key="2742",
            house_number="26",
            house_number_int=26,
            sheet_type="household",
        )

        self.assertFalse(
            can_match_overview_house_to_sheet(
                ov_house_26, other_village_26, overview_sheet=ov_sheet
            )
        )
        self.assertTrue(
            can_match_overview_house_to_sheet(
                ov_house_26, same_village_26, overview_sheet=ov_sheet
            )
        )

    def test_overview_house_linking_and_cross_year_and_same_year_linking(self) -> None:
        with Session(self.engine) as session:
            ov_sheet = CensusSheet(
                slovakiana_key="cair-ovw1930",
                census_year=1930,
                district="Topoľčany",
                municipality="Veľké Ripňany",
                municipality_key="2742",
                sheet_type="overview",
            )
            hz_1930_26 = CensusSheet(
                slovakiana_key="cair-hz30h26",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_key="7647",
                house_number="26",
                house_number_int=26,
                sheet_number="0110",
                sheet_type="household",
            )
            hz_1940_26 = CensusSheet(
                slovakiana_key="cair-hz40h26",
                census_year=1940,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_key="7647",
                house_number="26",
                house_number_int=26,
                sheet_number="0210",
                sheet_type="household",
            )
            session.add(ov_sheet)
            session.add(hz_1930_26)
            session.add(hz_1940_26)
            session.commit()
            session.refresh(ov_sheet)
            session.refresh(hz_1930_26)
            session.refresh(hz_1940_26)

            # Overview house 26 in Veľké Ripňany mistakenly pointing to Horné Zelenice sheet
            ov_dom_26 = CensusOverviewHouse(
                census_sheet_id=ov_sheet.id or 0,
                row_index=26,
                census_year=1930,
                municipality="Veľké Ripňany",
                settlement_name="Veľké Ripňany",
                house_number="26",
                house_number_int=26,
                matched_sheet_id=hz_1940_26.id,
            )
            session.add(ov_dom_26)
            session.commit()
            session.refresh(ov_dom_26)

            serialized_ov_26 = _serialize_overview_house(
                session, ov_dom_26, overview_sheet=ov_sheet
            )
            self.assertNotEqual(serialized_ov_26["matched_slovakiana_key"], "cair-hz40h26")
            self.assertIsNone(serialized_ov_26["matched_sheet_id"])

            linked = _find_linked_house_sheets(session, hz_1940_26)
            self.assertEqual(len(linked), 1)
            self.assertEqual(linked[0]["cair_id"], "cair-hz30h26")
            self.assertEqual(linked[0]["census_year"], 1930)
            self.assertEqual(linked[0]["municipality"], "Horné Zelenice")

            # Multiple sibling sheets for house 92 in 1940 + 1 sheet in 1930
            hz_1930_92 = CensusSheet(
                slovakiana_key="cair-hz30h92",
                census_year=1930,
                district="Hlohovec",
                municipality="Horné Zelenice",
                municipality_key="7647",
                house_number="92",
                house_number_int=92,
                sheet_number="0192",
                sheet_type="household",
            )
            session.add(hz_1930_92)
            sheets_1940_92: list[CensusSheet] = []
            for s_num in ("0317", "0318", "0319", "0320", "0321", "0322"):
                cs = CensusSheet(
                    slovakiana_key=f"cair-hz40s{s_num}",
                    census_year=1940,
                    district="Hlohovec",
                    municipality="Horné Zelenice",
                    municipality_key="7647",
                    house_number="92",
                    house_number_int=92,
                    sheet_number=s_num,
                    sheet_type="household",
                )
                session.add(cs)
                sheets_1940_92.append(cs)
            session.commit()
            target_320 = next(s for s in sheets_1940_92 if s.sheet_number == "0320")
            session.refresh(target_320)

            linked_92 = _find_linked_house_sheets(session, target_320)
            self.assertEqual(len(linked_92), 1)
            self.assertEqual(linked_92[0]["cair_id"], "cair-hz30h92")
            self.assertEqual(linked_92[0]["census_year"], 1930)

            same_year_92 = _find_same_year_house_sheets(session, target_320)
            self.assertEqual(len(same_year_92), 5)
            self.assertEqual(
                [s["sheet_number"] for s in same_year_92],
                ["0317", "0318", "0319", "0321", "0322"],
            )

    def test_filter_census_sheets_and_municipality_overview_by_person_and_birth_year(self) -> None:
        with Session(self.engine) as session:
            sheet = CensusSheet(
                slovakiana_key="cair-test007",
                census_year=1930,
                district="Trnava",
                municipality="Horný Čepeň",
                municipality_folded="horny cepen",
                municipality_key="10232",
                house_number="7",
                house_number_int=7,
                sheet_number="0007",
                household_owner="Ján Kováč",
                sheet_type="household",
                download_status="extracted",
            )
            session.add(sheet)
            session.commit()
            session.refresh(sheet)

            head = PersonFact(
                row_id=1,
                scan_id=1,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="1",
                event_year=1930,
                birth_year_est=1896,
                birth_or_event_date="15. 3. 1896",
                given_name_raw="Ján",
                given_name_sk="Ján",
                surname_raw="Kováč",
                surname_normalized="Kováč",
                relation_to_head="hlava domácnosti",
                sex="M",
                birth_place_raw="Šúrovce, okr. Trnava",
                birth_place_municipality="Šúrovce",
                birth_place_district="Trnava",
                village_raw="Horný Čepeň",
                village_sk="Horný Čepeň",
                house_number="7",
            )
            wife = PersonFact(
                row_id=2,
                scan_id=1,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="2",
                event_year=1930,
                birth_year_est=1901,
                given_name_raw="Mária",
                given_name_sk="Mária",
                surname_raw="Kováčová",
                surname_normalized="Kováč",
                mother_maiden_surname="Vargová",
                relation_to_head="manželka",
                sex="F",
                village_raw="Horný Čepeň",
                village_sk="Horný Čepeň",
                house_number="7",
            )
            mother_in_law = PersonFact(
                row_id=3,
                scan_id=1,
                record_type="census",
                census_sheet_id=sheet.id,
                person_row_number="3",
                event_year=1930,
                birth_year_est=1875,
                given_name_raw="Alžbeta",
                given_name_sk="Alžbeta",
                surname_raw="Vargová",
                surname_normalized="Varga",
                relation_to_head="svokra",
                sex="F",
                village_raw="Horný Čepeň",
                village_sk="Horný Čepeň",
                house_number="7",
            )
            for p in (head, wife, mother_in_law):
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
            reconcile_census_household_relationships(session, sheet.id or 0)

            res_kovac = list_census_sheets(
                background_tasks=BackgroundTasks(),
                person_name="Kováč",
                birth_year="1896",
                session=session,
            )
            self.assertGreaterEqual(res_kovac["total"], 1)
            self.assertTrue(
                any(
                    row["sheet"]["slovakiana_id"] == "cair-test007"
                    and any(
                        m["surname"] == "Kováč" and m["birth_year"] == 1896 and m["matched_filter"]
                        for m in row["extracted_persons_preview"]
                    )
                    for row in res_kovac["sheets"]
                )
            )

            # Historical district alias (`Galanta` for Horný Čepeň) + feminine surname matching (`Vargová` / `Varga`)
            res_varga = list_census_sheets(
                background_tasks=BackgroundTasks(),
                municipality="Horny Cepen",
                district="Galanta",
                person_name="Vargová",
                session=session,
            )
            self.assertGreaterEqual(res_varga["total"], 1)
            self.assertTrue(
                any(
                    row["sheet"]["slovakiana_id"] == "cair-test007"
                    for row in res_varga["sheets"]
                )
            )

            # Birth year range and birth place filter
            res_range = list_census_sheets(
                background_tasks=BackgroundTasks(),
                birth_year="1880-1905",
                birth_place="Šúrovce",
                session=session,
            )
            self.assertGreaterEqual(res_range["total"], 1)
            self.assertTrue(
                any(
                    row["sheet"]["slovakiana_id"] == "cair-test007"
                    for row in res_range["sheets"]
                )
            )

            # Municipality overview filtered by person_name + birth_year
            ov_filtered = get_municipality_overview(
                municipality="Horný Čepeň",
                district="Galanta",
                person_name="Ján Kováč",
                birth_year="1896±1",
                session=session,
            )
            self.assertEqual(ov_filtered["total_houses"], 1)
            self.assertEqual(ov_filtered["houses_directory"][0]["house_number"], "7")
            self.assertTrue(
                any(
                    m["surname"] == "Kováč" and m["matched_filter"]
                    for m in ov_filtered["houses_directory"][0]["extracted_members"]
                )
            )


if __name__ == "__main__":
    unittest.main()
