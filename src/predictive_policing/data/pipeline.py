"""Auditable local ingestion and transformation for incident-level CSV data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

REQUIRED_FIELDS = ("occurrence_time", "division_outpost", "crime_category")
class DataIngestionError(ValueError):
    """Raised when a source file cannot be safely loaded or transformed."""


@dataclass(frozen=True)
class DataQualityReport:
    """Counts-only quality summary; deliberately excludes incident-level values."""

    source_file: str
    rows_read: int
    rows_after_exact_deduplication: int
    exact_duplicate_rows_removed: int
    invalid_occurrence_timestamps: int
    missing_division_outpost: int
    missing_crime_category: int
    unknown_category_rows: int
    missing_coordinate_pairs: int
    invalid_coordinate_pairs: int
    duplicate_incident_id_rows: int
    duplicate_entry_number_rows: int
    invalid_reporting_delays: int
    timezone: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


@dataclass(frozen=True)
class TemporalSplits:
    """Chronological data partitions with timestamps kept wholly in one split."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def read_incident_csv(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
) -> pd.DataFrame:
    """Read a CSV without mutating the source or logging record contents."""
    source_path = Path(path)
    if not source_path.is_file():
        raise DataIngestionError(f"Incident source file does not exist: {source_path}")
    if source_path.suffix.casefold() != ".csv":
        raise DataIngestionError(
            f"Unsupported source format '{source_path.suffix}'; configure a CSV file."
        )
    if len(delimiter) != 1:
        raise DataIngestionError("CSV delimiter must be exactly one character.")

    try:
        frame = pd.read_csv(
            source_path,
            encoding=encoding,
            sep=delimiter,
            low_memory=False,
            on_bad_lines="error",
        )
    except (OSError, UnicodeError, EmptyDataError, ParserError, ValueError) as error:
        raise DataIngestionError(f"Could not read incident CSV '{source_path}': {error}") from error

    if frame.empty and len(frame.columns) == 0:
        raise DataIngestionError(f"Incident CSV has no header or columns: {source_path}")
    return frame


def read_incident_file(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Read a supported local CSV or Excel workbook."""
    source_path = Path(path)
    suffix = source_path.suffix.casefold()
    if suffix == ".csv":
        return read_incident_csv(source_path, encoding=encoding, delimiter=delimiter)
    if suffix not in {".xlsx", ".xlsm"}:
        raise DataIngestionError(
            f"Unsupported source format '{source_path.suffix}'; use CSV or XLSX."
        )
    if not source_path.is_file():
        raise DataIngestionError(f"Incident source file does not exist: {source_path}")

    try:
        frame = pd.read_excel(
            source_path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    except (OSError, ValueError, ImportError, KeyError) as error:
        raise DataIngestionError(
            f"Could not read incident workbook '{source_path}' (sheet {sheet_name!r}): {error}"
        ) from error
    if not isinstance(frame, pd.DataFrame):
        raise DataIngestionError(
            f"Workbook selection {sheet_name!r} did not produce a single worksheet."
        )
    if frame.empty and len(frame.columns) == 0:
        raise DataIngestionError(f"Incident worksheet has no header or columns: {source_path}")
    return frame


def transform_incidents(
    incidents: pd.DataFrame,
    *,
    columns: Mapping[str, str],
    timezone: str = "Africa/Lagos",
    source_file: str = "<dataframe>",
    missing_category_label: str = "unknown",
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate, normalize, and add model-facing fields while retaining source columns.

    Exact duplicate rows are removed. Conflicting rows sharing an incident ID
    are preserved and reported for explicit review rather than silently dropped.
    """
    if not isinstance(incidents, pd.DataFrame):
        raise DataIngestionError("Incident input must be a pandas DataFrame.")
    if incidents.empty:
        raise DataIngestionError("Incident input contains no records.")
    if not isinstance(missing_category_label, str) or not missing_category_label.strip():
        raise DataIngestionError("Missing-category label must be a non-empty string.")
    missing_category_label = missing_category_label.strip().casefold()

    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise DataIngestionError(f"Unknown IANA timezone '{timezone}'.") from error

    source = incidents.copy(deep=True)
    source.columns = [str(column).strip() for column in source.columns]
    if source.columns.has_duplicates:
        duplicates = source.columns[source.columns.duplicated()].tolist()
        raise DataIngestionError(f"Source has duplicate column names after trimming: {duplicates}")
    if "_source_row_number" in source.columns:
        raise DataIngestionError("Source uses reserved internal column '_source_row_number'.")

    normalized_columns = {
        field: name.strip()
        for field, name in columns.items()
        if isinstance(name, str)
    }

    missing_mappings = [
        field
        for field in REQUIRED_FIELDS
        if not normalized_columns.get(field)
    ]
    if missing_mappings:
        raise DataIngestionError(f"Missing required source column mappings: {missing_mappings}")

    invalid_mappings = [
        field
        for field, name in columns.items()
        if not isinstance(name, str) or not name.strip()
    ]
    if invalid_mappings:
        raise DataIngestionError(f"Invalid source column mappings: {invalid_mappings}")

    reserved_columns = {
        "model_occurred_at",
        "model_reported_at",
        "model_reporting_delay_hours",
        "model_division_outpost",
        "model_crime_category",
        "model_crime_subcategory",
        "model_incident_id",
        "model_entry_number",
        "model_incident_summary",
        "model_latitude",
        "model_longitude",
        "model_hour_sin",
        "model_hour_cos",
        "model_day_of_week_sin",
        "model_day_of_week_cos",
        "model_month_sin",
        "model_month_cos",
        "model_day_of_year_sin",
        "model_day_of_year_cos",
        "model_is_weekend",
    }
    collisions = sorted(reserved_columns.intersection(source.columns))
    if collisions:
        raise DataIngestionError(f"Source uses reserved model columns: {collisions}")

    missing_source_columns = [
        normalized_columns[field]
        for field in REQUIRED_FIELDS
        if normalized_columns[field] not in source.columns
    ]
    if missing_source_columns:
        raise DataIngestionError(
            "Required source columns are absent: "
            f"{missing_source_columns}. Available columns: {source.columns.tolist()}"
        )

    rows_read = len(source)
    source["_source_row_number"] = range(2, rows_read + 2)
    source_columns = [column for column in source.columns if column != "_source_row_number"]
    source = source.drop_duplicates(subset=source_columns, keep="first").copy()
    duplicates_removed = rows_read - len(source)
    source = source.reset_index(drop=True)

    result = source
    occurrence = _parse_timestamps(result[normalized_columns["occurrence_time"]], timezone)
    result["model_occurred_at"] = occurrence
    invalid_occurrence_count = int(occurrence.isna().sum())

    non_text_fields = {
        "occurrence_time",
        "reporting_time",
        "latitude",
        "longitude",
        "incident_summary",
    }
    for field, source_column in normalized_columns.items():
        if field in non_text_fields or source_column not in result.columns:
            continue
        normalized = _normalize_text(result[source_column])
        result[f"model_{field}"] = normalized

    division_column = "model_division_outpost"
    category_column = "model_crime_category"
    missing_division = int(result[division_column].isna().sum())
    missing_category = int(result[category_column].isna().sum())
    result[category_column] = result[category_column].fillna(missing_category_label)
    unknown_category_rows = int(result[category_column].eq(missing_category_label).sum())

    if "incident_summary" in normalized_columns and normalized_columns[
        "incident_summary"
    ] in result.columns:
        result["model_incident_summary"] = _normalize_text(
            result[normalized_columns["incident_summary"]], casefold=False
        )

    if (
        "reporting_time" in normalized_columns
        and normalized_columns["reporting_time"] in result.columns
    ):
        reported = _parse_timestamps(result[normalized_columns["reporting_time"]], timezone)
        result["model_reported_at"] = reported
        delay_hours = (reported - occurrence).dt.total_seconds() / 3600.0
        invalid_delay = delay_hours.lt(0) & delay_hours.notna()
        invalid_reporting_delays = int(invalid_delay.sum())
        result["model_reporting_delay_hours"] = delay_hours.mask(invalid_delay)
    else:
        invalid_reporting_delays = 0

    missing_coordinates, invalid_coordinates = _add_coordinates(result, normalized_columns)
    _add_time_features(result, timezone)
    duplicate_incident_ids = _count_duplicate_values(result, "incident_id")
    duplicate_entry_numbers = _count_duplicate_values(result, "entry_number")

    report = DataQualityReport(
        source_file=Path(source_file).name,
        rows_read=rows_read,
        rows_after_exact_deduplication=len(result),
        exact_duplicate_rows_removed=duplicates_removed,
        invalid_occurrence_timestamps=invalid_occurrence_count,
        missing_division_outpost=missing_division,
        missing_crime_category=missing_category,
        unknown_category_rows=unknown_category_rows,
        missing_coordinate_pairs=missing_coordinates,
        invalid_coordinate_pairs=invalid_coordinates,
        duplicate_incident_id_rows=duplicate_incident_ids,
        duplicate_entry_number_rows=duplicate_entry_numbers,
        invalid_reporting_delays=invalid_reporting_delays,
        timezone=timezone,
    )
    return result, report


def split_chronologically(
    frame: pd.DataFrame,
    *,
    time_column: str = "model_occurred_at",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> TemporalSplits:
    """Split ordered time points, keeping all rows at a timestamp together.

    For final experiments, call this on the aggregated interval table so
    simultaneous incidents and cells from the same interval cannot cross splits.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(not isinstance(value, (int, float)) or value <= 0 or value >= 1 for value in fractions):
        raise ValueError("Split fractions must each be greater than 0 and less than 1.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1.")
    if time_column not in frame.columns:
        raise ValueError(f"Chronological split requires column '{time_column}'.")
    if frame.empty:
        raise ValueError("Cannot split an empty data frame.")

    timestamps = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(
            f"Chronological split cannot proceed with {int(timestamps.isna().sum())} "
            f"missing or invalid values in '{time_column}'."
        )
    unique_times = pd.DatetimeIndex(timestamps.unique()).sort_values()
    if len(unique_times) < 3:
        raise ValueError("Chronological splitting requires at least three distinct time points.")

    train_end_index = max(1, int(len(unique_times) * train_fraction))
    validation_end_index = max(
        train_end_index + 1,
        int(len(unique_times) * (train_fraction + validation_fraction)),
    )
    if validation_end_index >= len(unique_times):
        validation_end_index = len(unique_times) - 1
    if train_end_index >= validation_end_index:
        raise ValueError("Not enough distinct time points for three non-empty chronological splits.")

    train_end = unique_times[train_end_index - 1]
    validation_end = unique_times[validation_end_index - 1]
    return TemporalSplits(
        train=frame.loc[timestamps <= train_end].copy(),
        validation=frame.loc[(timestamps > train_end) & (timestamps <= validation_end)].copy(),
        test=frame.loc[timestamps > validation_end].copy(),
        train_end=train_end,
        validation_end=validation_end,
    )


def prepare_incident_data(
    *,
    input_path: str | Path,
    prepared_path: str | Path,
    report_path: str | Path,
    columns: Mapping[str, str],
    timezone: str = "Africa/Lagos",
    input_sheet: str | int = 0,
    missing_category_label: str = "unknown",
) -> DataQualityReport:
    """Run local file intake and transformation, then atomically write artifacts."""
    input_resolved = Path(input_path).resolve()
    prepared_resolved = Path(prepared_path).resolve()
    report_resolved = Path(report_path).resolve()
    if len({input_resolved, prepared_resolved, report_resolved}) != 3:
        raise DataIngestionError(
            "Input, prepared-data, and quality-report paths must be distinct."
        )
    source = read_incident_file(input_path, sheet_name=input_sheet)
    transformed, report = transform_incidents(
        source,
        columns=columns,
        timezone=timezone,
        source_file=str(input_path),
        missing_category_label=missing_category_label,
    )
    _atomic_write_csv(transformed, Path(prepared_path))
    _atomic_write_json(report.to_dict(), Path(report_path))
    return report


def _parse_timestamps(values: pd.Series, timezone: str) -> pd.Series:
    def parse_one(value: Any) -> pd.Timestamp | pd.NaT:
        if pd.isna(value) or isinstance(value, (int, float)):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                return pd.NaT
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    timezone, ambiguous="NaT", nonexistent="NaT"
                )
            else:
                timestamp = timestamp.tz_convert(timezone)
            return timestamp.tz_convert("UTC")
        except (TypeError, ValueError, OverflowError):
            return pd.NaT

    parsed = values.map(parse_one)
    return pd.to_datetime(parsed, errors="coerce", utc=True)


def _normalize_text(values: pd.Series, *, casefold: bool = True) -> pd.Series:
    normalized = values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    normalized = normalized.mask(normalized.eq(""))
    if casefold:
        normalized = normalized.str.casefold()
    return normalized


def _add_coordinates(
    frame: pd.DataFrame, columns: Mapping[str, str]
) -> tuple[int, int]:
    latitude_source = columns.get("latitude")
    longitude_source = columns.get("longitude")
    has_latitude = bool(latitude_source and latitude_source in frame.columns)
    has_longitude = bool(longitude_source and longitude_source in frame.columns)
    if not has_latitude and not has_longitude:
        frame["model_latitude"] = float("nan")
        frame["model_longitude"] = float("nan")
        return len(frame), 0

    latitude = (
        pd.to_numeric(frame[latitude_source], errors="coerce")
        if has_latitude
        else pd.Series(float("nan"), index=frame.index)
    )
    longitude = (
        pd.to_numeric(frame[longitude_source], errors="coerce")
        if has_longitude
        else pd.Series(float("nan"), index=frame.index)
    )
    both_missing = latitude.isna() & longitude.isna()
    valid = latitude.between(-90, 90) & longitude.between(-180, 180)
    incomplete_or_invalid = ~both_missing & ~valid
    frame["model_latitude"] = latitude.where(valid)
    frame["model_longitude"] = longitude.where(valid)
    return int(both_missing.sum()), int(incomplete_or_invalid.sum())


def _add_time_features(frame: pd.DataFrame, timezone: str) -> None:
    local = frame["model_occurred_at"].dt.tz_convert(timezone)
    hour = local.dt.hour + local.dt.minute / 60.0
    day_of_week = local.dt.day_of_week
    month = local.dt.month - 1
    day_of_year = local.dt.day_of_year - 1
    year_length = local.dt.is_leap_year.map({True: 366, False: 365})

    frame["model_hour_sin"] = _cycle_sin(hour, 24)
    frame["model_hour_cos"] = _cycle_cos(hour, 24)
    frame["model_day_of_week_sin"] = _cycle_sin(day_of_week, 7)
    frame["model_day_of_week_cos"] = _cycle_cos(day_of_week, 7)
    frame["model_month_sin"] = _cycle_sin(month, 12)
    frame["model_month_cos"] = _cycle_cos(month, 12)
    frame["model_day_of_year_sin"] = _cycle_sin(day_of_year, year_length)
    frame["model_day_of_year_cos"] = _cycle_cos(day_of_year, year_length)
    frame["model_is_weekend"] = local.dt.day_of_week.ge(5).astype("boolean")


def _cycle_sin(values: pd.Series, period: Any) -> pd.Series:
    return pd.Series(np.sin(2.0 * np.pi * values / period), index=values.index)


def _cycle_cos(values: pd.Series, period: Any) -> pd.Series:
    return pd.Series(np.cos(2.0 * np.pi * values / period), index=values.index)


def _count_duplicate_values(frame: pd.DataFrame, field: str) -> int:
    column = f"model_{field}"
    if column not in frame:
        return 0
    values = frame[column].dropna()
    return int(values.duplicated(keep=False).sum())


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            frame.to_csv(temporary_file, index=False)
        os.replace(temporary_path, path)
    except (OSError, ValueError) as error:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise DataIngestionError(f"Could not write prepared data to '{path}': {error}") from error


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        raise DataIngestionError(f"Could not write quality report to '{path}': {error}") from error
