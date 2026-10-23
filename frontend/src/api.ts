export interface LinkedHousePersonPreview {
  given_name: string | null;
  surname: string | null;
  relation_to_head: string | null;
  birth_year: number | null;
}

export interface LinkedHouseSheet {
  id: number;
  cair_id: string;
  slovakiana_key?: string;
  slovakiana_id?: string;
  census_year: number;
  municipality: string | null;
  district: string | null;
  settlement: string | null;
  house_number: string | null;
  enumeration_district: string | null;
  sheet_number?: string | null;
  box_number?: string | null;
  is_blackened?: boolean;
  download_status?: string;
  head_of_household: string | null;
  persons_count: number;
  persons_preview: LinkedHousePersonPreview[];
}

export interface CensusSheetDetail {
  id: number;
  cair_id?: string;
  slovakiana_id: string;
  slovakiana_key?: string;
  urn_nbn?: string | null;
  census_year: number;
  district_name: string | null;
  municipality_name: string;
  street_name: string | null;
  house_number_raw: string | null;
  house_number_conscription: string | null;
  box_number: string | null;
  sheet_number: string | null;
  is_blackened: boolean;
  front_image_url: string | null;
  back_image_url: string | null;
  source_web_url: string | null;
  download_status: string;
  pipeline_status: string;
  sheet_type?: "household" | "overview" | string;
  household_owner?: string | null;
  obvod_number?: string | null;
  commissioner_name?: string | null;
  overview_houses_count?: number;
  linked_house_sheets?: LinkedHouseSheet[];
  same_year_house_sheets?: LinkedHouseSheet[];
}
