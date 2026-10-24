from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Batch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    parish_or_book: Optional[str] = Field(default=None)
    record_type_hint: str = Field(default="auto")  # auto, baptism, marriage, death
    total_scans: int = Field(default=0)
    processed_scans: int = Field(default=0)
    status: str = Field(default="idle")  # idle, slicing, ocr_stage, decision_stage, extract_stage, matching, completed
    watchlist_filter: Optional[str] = Field(default=None)  # comma-separated surnames/villages
    created_at: datetime = Field(default_factory=utcnow)


class ScanImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: Optional[int] = Field(default=None, foreign_key="batch.id", index=True)
    census_sheet_id: Optional[int] = Field(default=None, index=True)
    page_side: Optional[str] = Field(default=None)  # front (_v) or back (_r)
    filename: str = Field(index=True)
    original_path: str
    overlay_path: Optional[str] = Field(default=None)
    header_crop_path: Optional[str] = Field(default=None)
    width: int = Field(default=0)
    height: int = Field(default=0)
    spine_x: int = Field(default=0)
    detected_year: Optional[str] = Field(default=None)
    detected_type: str = Field(default="baptism")  # baptism, marriage, death, census
    header_raw_ocr: Optional[str] = Field(default=None)
    grid_metadata_json: str = Field(default="{}")
    status: str = Field(default="uploaded")  # uploaded, sliced, ocr_done, triaged, extracted, matched, error
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class RecordRow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    scan_id: int = Field(foreign_key="scanimage.id", index=True)
    row_index: int = Field(index=True)
    entry_number: Optional[str] = Field(default=None)
    y_start: int = Field(default=0)
    y_end: int = Field(default=0)
    stitched_crop_path: Optional[str] = Field(default=None)
    left_crop_path: Optional[str] = Field(default=None)
    right_crop_path: Optional[str] = Field(default=None)
    cell_crops_json: str = Field(default="[]")

    raw_ocr_left: Optional[str] = Field(default=None)
    raw_ocr_right: Optional[str] = Field(default=None)
    raw_ocr_combined: Optional[str] = Field(default=None)

    decision_action: str = Field(default="pending")  # pending, auto_accept, ask_human, retry_cell_crop
    decision_confidence: float = Field(default=0.0)
    decision_reason: Optional[str] = Field(default=None)
    decision_json: str = Field(default="{}")
    has_ditto_marks: bool = Field(default=False)

    normalized_json: str = Field(default="{}")
    record_type: str = Field(default="baptism")  # baptism, marriage, death
    status: str = Field(default="sliced")  # sliced, ocr_done, needs_human, triaged, extracted, matched
    created_at: datetime = Field(default_factory=utcnow)


class PersonFact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    row_id: int = Field(foreign_key="recordrow.id", index=True)
    scan_id: int = Field(foreign_key="scanimage.id", index=True)

    record_type: str = Field(default="baptism", index=True)
    entry_number: Optional[str] = Field(default=None)
    event_year: Optional[int] = Field(default=None, index=True)
    birth_or_event_date: Optional[str] = Field(default=None)
    baptism_or_burial_date: Optional[str] = Field(default=None)

    given_name_raw: Optional[str] = Field(default=None)
    given_name_sk: Optional[str] = Field(default=None, index=True)
    given_name_hu: Optional[str] = Field(default=None)
    surname_raw: Optional[str] = Field(default=None, index=True)
    surname_normalized: Optional[str] = Field(default=None, index=True)
    sex: Optional[str] = Field(default=None)
    legitimacy: Optional[str] = Field(default=None)
    religion: Optional[str] = Field(default=None)

    father_name: Optional[str] = Field(default=None)
    mother_name: Optional[str] = Field(default=None)
    mother_maiden_surname: Optional[str] = Field(default=None, index=True)
    spouse_name: Optional[str] = Field(default=None)
    social_status_or_occupation: Optional[str] = Field(default=None)

    village_raw: Optional[str] = Field(default=None)
    village_sk: Optional[str] = Field(default=None, index=True)
    village_hu: Optional[str] = Field(default=None)
    house_number: Optional[str] = Field(default=None)
    fs_place_standardized: Optional[str] = Field(default=None)
    fs_place_id: Optional[str] = Field(default=None)

    godparents_or_witnesses: Optional[str] = Field(default=None)
    officiant: Optional[str] = Field(default=None)
    age_at_event: Optional[str] = Field(default=None)
    cause_of_death: Optional[str] = Field(default=None)
    observationes_raw: Optional[str] = Field(default=None)
    observationes_events_json: str = Field(default="[]")  # linked later death (+) or marriage notes

    # Census-specific & unified search fields
    census_sheet_id: Optional[int] = Field(default=None, foreign_key="censussheet.id", index=True)
    census_household_id: Optional[int] = Field(default=None, foreign_key="censushousehold.id", index=True)
    person_row_number: Optional[str] = Field(default=None)
    relation_to_head: Optional[str] = Field(default=None, index=True)
    marital_status: Optional[str] = Field(default=None, index=True)
    birth_year_est: Optional[int] = Field(default=None, index=True)
    birth_place_raw: Optional[str] = Field(default=None)
    birth_place_sk: Optional[str] = Field(default=None, index=True)
    birth_place_municipality: Optional[str] = Field(default=None)
    birth_place_district: Optional[str] = Field(default=None)
    birth_place_country: Optional[str] = Field(default=None)
    citizenship: Optional[str] = Field(default=None)
    moved_in_date: Optional[str] = Field(default=None)
    previous_residence: Optional[str] = Field(default=None)
    home_jurisdiction: Optional[str] = Field(default=None)
    nationality: Optional[str] = Field(default=None, index=True)
    literacy: Optional[str] = Field(default=None)
    occupation_position: Optional[str] = Field(default=None)
    employer_workplace: Optional[str] = Field(default=None)
    marriage_date_raw: Optional[str] = Field(default=None)
    children_born_count: Optional[str] = Field(default=None)
    children_dead_count: Optional[str] = Field(default=None)
    presence_type: Optional[str] = Field(default=None)

    # Pre-computed search index columns
    surname_candidates_json: str = Field(default="[]")
    surname_folded: Optional[str] = Field(default=None, index=True)
    surname_skeleton: Optional[str] = Field(default=None, index=True)
    given_canonical_id: Optional[str] = Field(default=None, index=True)
    village_folded: Optional[str] = Field(default=None, index=True)
    house_number_int: Optional[int] = Field(default=None, index=True)

    created_at: datetime = Field(default_factory=utcnow)

    def __init__(self, **data: object) -> None:
        import json as _json

        if "surname_candidates" in data and "surname_candidates_json" not in data:
            sc_val = data.pop("surname_candidates")
            if isinstance(sc_val, str):
                data["surname_candidates_json"] = sc_val
            elif sc_val is not None:
                data["surname_candidates_json"] = _json.dumps(list(sc_val), ensure_ascii=False)
        else:
            data.pop("surname_candidates", None)
        if "raw_row_text" in data:
            raw_rt = data.pop("raw_row_text")
            if raw_rt and not data.get("observationes_raw"):
                data["observationes_raw"] = str(raw_rt)
        super().__init__(**data)

    @property
    def household_id(self) -> Optional[int]:
        return self.census_household_id

    @household_id.setter
    def household_id(self, value: Optional[int]) -> None:
        self.census_household_id = value

    @property
    def surname_Candidiates_json(self) -> str:
        return self.surname_candidates_json

    @surname_Candidiates_json.setter
    def surname_Candidiates_json(self, value: str) -> None:
        self.surname_candidates_json = value

    @property
    def surname_candidates(self) -> list[str]:
        import json as _json

        try:
            parsed = _json.loads(self.surname_candidates_json or "[]")
            if isinstance(parsed, list):
                out: list[str] = []
                for item in parsed:
                    if isinstance(item, str) and item:
                        out.append(item)
                    elif isinstance(item, dict):
                        for k in ("raw_ocr", "normalized_ocr", "consensus_stem", "resolved_surname"):
                            v = item.get(k)
                            if isinstance(v, str) and v and v not in out:
                                out.append(v)
                return out
        except Exception:
            pass
        return []

    @surname_candidates.setter
    def surname_candidates(self, value: list[str] | str | None) -> None:
        import json as _json

        if isinstance(value, str):
            self.surname_candidates_json = value
        elif value is None:
            self.surname_candidates_json = "[]"
        else:
            self.surname_candidates_json = _json.dumps(list(value), ensure_ascii=False)

    @property
    def raw_row_text(self) -> Optional[str]:
        return self.observationes_raw

    @raw_row_text.setter
    def raw_row_text(self, value: Optional[str]) -> None:
        self.observationes_raw = value


class CensusSheet(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    slovakiana_key: str = Field(unique=True, index=True)  # e.g. "cair-test001"
    digital_object_id: Optional[str] = Field(default=None, index=True)  # e.g. "doid-1bwsb"
    urn_nbn: Optional[str] = Field(default=None)

    census_year: int = Field(default=1930, index=True)  # 1930, 1939, 1940
    district: Optional[str] = Field(default=None, index=True)  # Okres
    district_key: Optional[str] = Field(default=None, index=True)  # 3-digit key "701".."783"
    district_folded: Optional[str] = Field(default=None, index=True)
    municipality: str = Field(default="", index=True)  # Obec
    municipality_key: Optional[str] = Field(default=None, index=True)  # 4-5 digit key
    municipality_folded: str = Field(default="", index=True)
    street_raw: Optional[str] = Field(default=None)
    house_number: Optional[str] = Field(default=None, index=True)
    house_number_int: Optional[int] = Field(default=None, index=True)
    box_number: Optional[str] = Field(default=None, index=True)
    sheet_number: Optional[str] = Field(default=None, index=True)
    title_raw: Optional[str] = Field(default=None)

    is_blackened: bool = Field(default=False, index=True)
    has_images: bool = Field(default=True)
    thumbnail_id: Optional[str] = Field(default=None)
    front_scan_id: Optional[int] = Field(default=None, foreign_key="scanimage.id", index=True)
    back_scan_id: Optional[int] = Field(default=None, foreign_key="scanimage.id", index=True)
    image_urls_json: str = Field(default="[]")
    attributes_json: str = Field(default="{}")
    header_json: str = Field(default="{}")
    household_owner: Optional[str] = Field(default=None)
    sheet_type: str = Field(default="household", index=True)  # "household" | "overview" | "municipal_overview"
    obvod_number: Optional[str] = Field(default=None, index=True)
    commissioner_name: Optional[str] = Field(default=None)

    download_status: str = Field(default="cataloged", index=True)
    # cataloged, queued, downloaded, sliced, ocr_done, extracted, matched, error
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class CensusOverviewHouse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    census_sheet_id: int = Field(foreign_key="censussheet.id", index=True)
    scan_id: Optional[int] = Field(default=None, foreign_key="scanimage.id", index=True)
    row_index: int = Field(default=1, index=True)
    census_year: int = Field(default=1930, index=True)
    district: Optional[str] = Field(default=None, index=True)
    municipality: str = Field(default="", index=True)
    obvod_number: Optional[str] = Field(default=None, index=True)
    ordinal_number: Optional[str] = Field(default=None)
    settlement_name: Optional[str] = Field(default=None)
    house_number: Optional[str] = Field(default=None, index=True)
    house_number_int: Optional[int] = Field(default=None, index=True)
    is_inhabited: Optional[bool] = Field(default=True)
    owner_name: Optional[str] = Field(default=None, index=True)
    owner_surname_folded: Optional[str] = Field(default=None, index=True)
    household_count: Optional[int] = Field(default=None)
    present_persons_count: Optional[int] = Field(default=None)
    remarks: Optional[str] = Field(default=None)
    matched_sheet_id: Optional[int] = Field(default=None, foreign_key="censussheet.id", index=True)
    stitched_crop_path: Optional[str] = Field(default=None)
    raw_ocr_text: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.85)
    created_at: datetime = Field(default_factory=utcnow)

    def __init__(self, **data: object) -> None:
        full_raw = data.pop("owner_full_name_raw", None)
        sur_raw = data.pop("owner_surname_raw", None)
        giv_raw = data.pop("owner_given_name_raw", None)
        if not data.get("owner_name"):
            if full_raw:
                data["owner_name"] = str(full_raw)
            elif sur_raw or giv_raw:
                data["owner_name"] = " ".join(str(x) for x in (sur_raw, giv_raw) if x)
        raw_rt = data.pop("raw_row_text", None)
        if raw_rt and not data.get("raw_ocr_text"):
            data["raw_ocr_text"] = str(raw_rt)
        super().__init__(**data)

    @property
    def owner_full_name_raw(self) -> Optional[str]:
        return self.owner_name

    @owner_full_name_raw.setter
    def owner_full_name_raw(self, value: Optional[str]) -> None:
        self.owner_name = value

    @property
    def owner_surname_raw(self) -> Optional[str]:
        if not self.owner_name:
            return None
        parts = [p.strip(".,;:()") for p in self.owner_name.split() if p.strip(".,;:()")]
        return parts[0] if parts else None

    @property
    def owner_given_name_raw(self) -> Optional[str]:
        if not self.owner_name:
            return None
        parts = [p.strip(".,;:()") for p in self.owner_name.split() if p.strip(".,;:()")]
        return " ".join(parts[1:]) if len(parts) > 1 else None

    @property
    def raw_row_text(self) -> Optional[str]:
        return self.raw_ocr_text

    @raw_row_text.setter
    def raw_row_text(self, value: Optional[str]) -> None:
        self.raw_ocr_text = value


class CensusMunicipalOverviewDistrict(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    census_sheet_id: int = Field(foreign_key="censussheet.id", index=True)
    scan_id: Optional[int] = Field(default=None, foreign_key="scanimage.id", index=True)
    row_index: int = Field(default=1, index=True)
    census_year: int = Field(default=1930, index=True)
    district: Optional[str] = Field(default=None, index=True)
    municipality: str = Field(default="", index=True)
    ordinal_number: Optional[str] = Field(default=None)
    settlement_name: Optional[str] = Field(default=None)
    obvod_number: Optional[str] = Field(default=None, index=True)
    commissioner_name: Optional[str] = Field(default=None, index=True)
    houses_total: Optional[int] = Field(default=None)
    houses_inhabited: Optional[int] = Field(default=None)
    houses_uninhabited: Optional[int] = Field(default=None)
    household_count: Optional[int] = Field(default=None)
    present_persons_count: Optional[int] = Field(default=None)
    remarks: Optional[str] = Field(default=None)
    is_summary_row: bool = Field(default=False, index=True)
    matched_overview_sheet_id: Optional[int] = Field(default=None, foreign_key="censussheet.id", index=True)
    stitched_crop_path: Optional[str] = Field(default=None)
    raw_ocr_text: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.90)
    created_at: datetime = Field(default_factory=utcnow)


class CensusHousehold(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    census_sheet_id: int = Field(foreign_key="censussheet.id", index=True)
    dwelling_unit: Optional[str] = Field(default="I", index=True)
    apartment_number: Optional[str] = Field(default="1")
    house_owner_name: Optional[str] = Field(default=None)
    street: Optional[str] = Field(default=None)
    house_number: Optional[str] = Field(default=None)
    head_person_fact_id: Optional[int] = Field(default=None, index=True)

    fs_household_status: str = Field(default="unmatched", index=True)
    # unmatched, confirmed_household, partial_household, multiple_candidates, new_household
    fs_record_ark: Optional[str] = Field(default=None)
    fs_confidence: float = Field(default=0.0)
    fs_candidates_json: str = Field(default="[]")
    notes_raw: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class CensusDownloadJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_type: str = Field(default="catalog_harvest", index=True)
    # catalog_harvest, image_download, full_pipeline
    filter_json: str = Field(default="{}")
    cursor_page: int = Field(default=0)
    total_items: int = Field(default=0)
    completed_items: int = Field(default=0)
    failed_items: int = Field(default=0)
    status: str = Field(default="running", index=True)  # running, paused, completed, error
    concurrency: int = Field(default=4)
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class LocalRecordMatch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    census_fact_id: int = Field(foreign_key="personfact.id", index=True)
    matrika_fact_id: int = Field(foreign_key="personfact.id", index=True)
    confidence_score: float = Field(default=0.0)
    match_kind: str = Field(default="baptism")  # baptism, marriage, death
    what_agrees_json: str = Field(default="[]")
    reviewed_by_user: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=utcnow)


class QuestionItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    row_id: int = Field(foreign_key="recordrow.id", index=True)
    scan_id: int = Field(foreign_key="scanimage.id", index=True)
    field_name: str = Field(default="transcription")
    question_sk: str
    question_en: Optional[str] = Field(default=None)
    crop_path: Optional[str] = Field(default=None)
    neighbor_crops_json: str = Field(default="[]")  # [{row_id, crop_path, confirmed_text}]
    candidates_json: str = Field(default="[]")  # list of candidate readings
    raw_ocr_snippet: Optional[str] = Field(default=None)
    status: str = Field(default="open", index=True)  # open, answered, skipped_for_admin
    selected_answer: Optional[str] = Field(default=None)
    answered_by: Optional[str] = Field(default=None)
    answered_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class FamilySearchMatch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    person_fact_id: int = Field(foreign_key="personfact.id", index=True)
    row_id: int = Field(foreign_key="recordrow.id", index=True)
    match_status: str = Field(default="unmatched", index=True)
    # new_person, existing_needs_facts, confirmed_match, place_only
    fs_person_id: Optional[str] = Field(default=None, index=True)
    fs_record_ark: Optional[str] = Field(default=None)
    fs_url: Optional[str] = Field(default=None)
    confidence_score: float = Field(default=0.0)
    summary_sk: Optional[str] = Field(default=None)
    what_is_new_json: str = Field(default="[]")  # list of new facts to add to FS
    what_agrees_json: str = Field(default="[]")
    what_differs_json: str = Field(default="[]")
    action_packet_json: str = Field(default="{}")  # compare_person or structured packet
    raw_mcp_results_json: str = Field(default="{}")
    reviewed_by_user: bool = Field(default=False, index=True)
    user_notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)


class WorkflowTrace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    scan_id: Optional[int] = Field(default=None, index=True)
    row_id: Optional[int] = Field(default=None, index=True)
    stage: str = Field(index=True)  # layout, ocr, decision, extract, mcp_place, mcp_record, mcp_compare
    model_or_tool: str
    prompt_text: Optional[str] = Field(default=None)
    image_path: Optional[str] = Field(default=None)
    reasoning_content: Optional[str] = Field(default=None)
    response_text: Optional[str] = Field(default=None)
    parsed_json: str = Field(default="{}")
    status: str = Field(default="ok")  # ok, error
    duration_ms: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    reasoning_tokens: int = Field(default=0)
    created_at: datetime = Field(default_factory=utcnow)


class OCRFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    target_type: str = Field(default="row", index=True)
    # row, person_fact, overview_house, municipal_district, scan, census_sheet
    row_id: Optional[int] = Field(default=None, index=True)
    scan_id: Optional[int] = Field(default=None, index=True)
    census_sheet_id: Optional[int] = Field(default=None, index=True)
    person_fact_id: Optional[int] = Field(default=None, index=True)
    overview_house_id: Optional[int] = Field(default=None, index=True)
    municipal_district_id: Optional[int] = Field(default=None, index=True)

    verdict: str = Field(default="correct", index=True)  # "correct" | "incorrect"
    comment: Optional[str] = Field(default=None)
    expected_correction: Optional[str] = Field(default=None)

    model_combo: str = Field(default="", index=True)
    layout_model: Optional[str] = Field(default=None)
    ocr_model: Optional[str] = Field(default=None, index=True)
    decision_model: Optional[str] = Field(default=None, index=True)
    extract_model: Optional[str] = Field(default=None, index=True)
    mcp_model: Optional[str] = Field(default=None)

    total_duration_ms: int = Field(default=0)
    layout_duration_ms: int = Field(default=0)
    ocr_duration_ms: int = Field(default=0)
    decision_duration_ms: int = Field(default=0)
    extract_duration_ms: int = Field(default=0)
    mcp_duration_ms: int = Field(default=0)

    raw_ocr_snapshot: Optional[str] = Field(default=None)
    extracted_snapshot_json: str = Field(default="{}")
    stage_breakdown_json: str = Field(default="{}")
    traces_snapshot_json: str = Field(default="[]")
    crop_url: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

