"""Loading and validation for experiment configuration."""

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a project configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its required experiment settings."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must contain a YAML mapping.")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check required sections and ensure split and target settings are coherent."""
    data = config.get("data")
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration must define a 'data' mapping.")
    for path_key in ("input_path", "prepared_path", "quality_report_path"):
        if not isinstance(data.get(path_key), str) or not data[path_key].strip():
            raise ConfigurationError(f"'data.{path_key}' must be a non-empty path.")
    if not isinstance(data.get("timezone"), str) or not data["timezone"].strip():
        raise ConfigurationError("'data.timezone' must be a non-empty timezone name.")
    input_sheet = data.get("input_sheet", 0)
    if isinstance(input_sheet, bool) or not isinstance(input_sheet, (str, int)):
        raise ConfigurationError("'data.input_sheet' must be a worksheet name or zero-based index.")
    if isinstance(input_sheet, str) and not input_sheet.strip():
        raise ConfigurationError("'data.input_sheet' worksheet name cannot be empty.")
    if isinstance(input_sheet, int) and input_sheet < 0:
        raise ConfigurationError("'data.input_sheet' index cannot be negative.")
    if (
        not isinstance(data.get("missing_category_label"), str)
        or not data["missing_category_label"].strip()
    ):
        raise ConfigurationError("'data.missing_category_label' must be a non-empty string.")

    columns = data.get("columns")
    required_columns = {"occurrence_time", "division_outpost", "crime_category"}
    if not isinstance(columns, dict) or not required_columns.issubset(columns):
        raise ConfigurationError(
            "'data.columns' must map at least occurrence_time, division_outpost, "
            "and crime_category to source column names."
        )
    if any(not isinstance(name, str) or not name.strip() for name in columns.values()):
        raise ConfigurationError("Every 'data.columns' value must be a non-empty source column name.")

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigurationError("Configuration must define an 'experiment' mapping.")

    if experiment.get("forecast_target") != "incident_count_by_location_time_category":
        raise ConfigurationError(
            "'experiment.forecast_target' must be "
            "'incident_count_by_location_time_category'."
        )

    aggregation_windows = experiment.get("aggregation_windows")
    allowed_windows = {"24h", "3d", "1w"}
    if not isinstance(aggregation_windows, list) or not aggregation_windows:
        raise ConfigurationError("'experiment.aggregation_windows' must be a non-empty list.")
    if len(set(aggregation_windows)) != len(aggregation_windows) or not set(
        aggregation_windows
    ).issubset(allowed_windows):
        raise ConfigurationError(
            f"'experiment.aggregation_windows' entries must be unique values from "
            f"{sorted(allowed_windows)}."
        )
    primary_window = experiment.get("primary_aggregation_window")
    if primary_window not in aggregation_windows:
        raise ConfigurationError(
            "'experiment.primary_aggregation_window' must be included in "
            "'experiment.aggregation_windows'."
        )

    ranking_dimensions = experiment.get("ranking_dimensions")
    expected_dimensions = {"division_outpost", "time_interval", "crime_category"}
    if not isinstance(ranking_dimensions, list) or set(ranking_dimensions) != expected_dimensions:
        raise ConfigurationError(
            "'experiment.ranking_dimensions' must include division_outpost, "
            "time_interval, and crime_category."
        )

    fractions = [
        experiment.get("train_fraction"),
        experiment.get("validation_fraction"),
        experiment.get("test_fraction"),
    ]
    if any(not isinstance(value, (int, float)) or value <= 0 or value >= 1 for value in fractions):
        raise ConfigurationError("Train, validation, and test fractions must each be between 0 and 1.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigurationError("Train, validation, and test fractions must sum to 1.")

    horizon = experiment.get("forecast_horizon_intervals")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ConfigurationError("'experiment.forecast_horizon_intervals' must be a positive integer.")
