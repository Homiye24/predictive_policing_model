from pathlib import Path

import pandas as pd
import pytest

from predictive_policing.data.pipeline import (
    DataIngestionError,
    prepare_incident_data,
    read_incident_csv,
    read_incident_file,
    split_chronologically,
    transform_incidents,
)

COLUMNS = {
    "incident_id": "Incident ID",
    "entry_number": "Entry Number",
    "occurrence_time": "Occurrence",
    "reporting_time": "Reported",
    "division_outpost": "Division",
    "crime_category": "Offence",
    "incident_summary": "Summary",
    "latitude": "Latitude",
    "longitude": "Longitude",
}


def _sample_incidents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Incident ID": "A",
                "Entry Number": "1",
                "Occurrence": "2025-01-01 12:00:00",
                "Reported": "2025-01-01 13:00:00",
                "Division": "  Akure North  ",
                "Offence": " Theft ",
                "Summary": "  Property   taken ",
                "Latitude": "7.25",
                "Longitude": "5.20",
            },
            {
                "Incident ID": "A",
                "Entry Number": "1",
                "Occurrence": "2025-01-01 12:00:00",
                "Reported": "2025-01-01 13:00:00",
                "Division": "  Akure North  ",
                "Offence": " Theft ",
                "Summary": "  Property   taken ",
                "Latitude": "7.25",
                "Longitude": "5.20",
            },
            {
                "Incident ID": "A",
                "Entry Number": "2",
                "Occurrence": "2025-01-02 12:00:00",
                "Reported": "2025-01-02 11:00:00",
                "Division": "Owo",
                "Offence": "Burglary",
                "Summary": "Forced entry",
                "Latitude": "91",
                "Longitude": "5.20",
            },
            {
                "Incident ID": "C",
                "Entry Number": "3",
                "Occurrence": "not a date",
                "Reported": "",
                "Division": "  ",
                "Offence": "Theft",
                "Summary": "",
                "Latitude": "",
                "Longitude": "",
            },
            {
                "Incident ID": "D",
                "Entry Number": "4",
                "Occurrence": "2025-01-04 12:00:00",
                "Reported": "2025-01-04 12:30:00",
                "Division": "Ondo",
                "Offence": "  ",
                "Summary": "Incident details",
                "Latitude": "7.20",
                "Longitude": "",
            },
        ]
    )


def test_transformation_preserves_source_and_reports_quality_issues() -> None:
    source = _sample_incidents()
    source_copy = source.copy(deep=True)
    transformed, report = transform_incidents(
        source,
        columns=COLUMNS,
        timezone="Africa/Lagos",
        source_file="private-records.csv",
    )

    assert len(transformed) == 4
    assert transformed.loc[0, "Division"] == "  Akure North  "
    assert transformed.loc[0, "model_division_outpost"] == "akure north"
    pd.testing.assert_frame_equal(source, source_copy)
    assert transformed.loc[0, "model_incident_summary"] == "Property taken"
    assert transformed.loc[0, "model_occurred_at"] == pd.Timestamp(
        "2025-01-01 11:00:00+00:00"
    )
    assert transformed.loc[0, "model_reporting_delay_hours"] == 1
    assert pd.isna(transformed.loc[2, "model_occurred_at"])
    assert pd.isna(transformed.loc[1, "model_reporting_delay_hours"])
    assert pd.isna(transformed.loc[1, "model_latitude"])
    assert transformed.loc[3, "Offence"] == "  "
    assert transformed.loc[3, "model_crime_category"] == "unknown"
    assert transformed.loc[0, "model_is_weekend"] == False
    assert report.source_file == "private-records.csv"
    assert report.rows_read == 5
    assert report.rows_after_exact_deduplication == 4
    assert report.exact_duplicate_rows_removed == 1
    assert report.invalid_occurrence_timestamps == 1
    assert report.missing_division_outpost == 1
    assert report.missing_crime_category == 1
    assert report.unknown_category_rows == 1
    assert report.missing_coordinate_pairs == 1
    assert report.invalid_coordinate_pairs == 2
    assert report.duplicate_incident_id_rows == 2
    assert report.duplicate_entry_number_rows == 0
    assert report.invalid_reporting_delays == 1


def test_missing_required_source_column_fails_explicitly() -> None:
    incidents = _sample_incidents().drop(columns="Offence")

    with pytest.raises(DataIngestionError, match="Required source columns are absent"):
        transform_incidents(incidents, columns=COLUMNS)


def test_timezone_validation_fails_explicitly() -> None:
    with pytest.raises(DataIngestionError, match="Unknown IANA timezone"):
        transform_incidents(
            _sample_incidents(),
            columns=COLUMNS,
            timezone="Not/A_Timezone",
        )


def test_missing_category_label_must_be_non_empty() -> None:
    with pytest.raises(DataIngestionError, match="non-empty string"):
        transform_incidents(
            _sample_incidents(),
            columns=COLUMNS,
            missing_category_label=" ",
        )


def test_chronological_splits_keep_timestamps_together() -> None:
    frame = pd.DataFrame(
        {
            "model_occurred_at": list(pd.date_range("2025-01-01", periods=10, tz="UTC"))
            + [pd.Timestamp("2025-01-01", tz="UTC")],
            "row": list(range(10)) + [10],
        }
    )

    splits = split_chronologically(frame)

    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(frame)
    assert splits.train["model_occurred_at"].max() <= splits.train_end
    assert splits.validation["model_occurred_at"].min() > splits.train_end
    assert splits.validation["model_occurred_at"].max() <= splits.validation_end
    assert splits.test["model_occurred_at"].min() > splits.validation_end
    assert splits.train["row"].tolist().count(10) == 1


def test_chronological_split_rejects_missing_timestamps() -> None:
    frame = pd.DataFrame({"model_occurred_at": [pd.NaT]})

    with pytest.raises(ValueError, match="cannot proceed"):
        split_chronologically(frame)


def test_prepare_writes_data_and_counts_only_report_atomically(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    prepared_path = tmp_path / "processed" / "cleaned.csv"
    report_path = tmp_path / "processed" / "quality.json"
    _sample_incidents().to_csv(input_path, index=False)

    report = prepare_incident_data(
        input_path=input_path,
        prepared_path=prepared_path,
        report_path=report_path,
        columns=COLUMNS,
    )

    prepared = pd.read_csv(prepared_path)
    report_text = report_path.read_text(encoding="utf-8")
    assert len(prepared) == 4
    assert report.rows_after_exact_deduplication == 4
    assert "Property taken" not in report_text
    assert list((tmp_path / "processed").glob("*.tmp")) == []


def test_csv_reader_rejects_unsupported_file_format(tmp_path: Path) -> None:
    source_path = tmp_path / "input.xlsx"
    source_path.touch()

    with pytest.raises(DataIngestionError, match="Unsupported source format"):
        read_incident_csv(source_path)


def test_excel_reader_loads_selected_sheet(tmp_path: Path) -> None:
    source_path = tmp_path / "incidents.xlsx"
    expected = pd.DataFrame({"Occurrence": ["2025-01-01"], "Division": ["A"]})
    with pd.ExcelWriter(source_path, engine="openpyxl") as writer:
        expected.to_excel(writer, sheet_name="Crime_Data", index=False)
        pd.DataFrame({"ignore": [1]}).to_excel(writer, sheet_name="Other", index=False)

    actual = read_incident_file(source_path, sheet_name="Crime_Data")

    pd.testing.assert_frame_equal(actual, expected)
