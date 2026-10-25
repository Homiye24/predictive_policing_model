from pathlib import Path

import pytest

from predictive_policing.config import ConfigurationError, load_config, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_uses_location_time_category_count_target() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    experiment = config["experiment"]
    data = config["data"]

    assert experiment["forecast_target"] == "incident_count_by_location_time_category"
    assert experiment["aggregation_windows"] == ["24h", "3d", "1w"]
    assert experiment["primary_aggregation_window"] == "24h"
    assert experiment["ranking_dimensions"] == [
        "division_outpost",
        "time_interval",
        "crime_category",
    ]
    assert experiment["forecast_horizon_intervals"] == 1
    assert data["input_path"].endswith("crimedata.xlsx")
    assert data["input_sheet"] == "Crime_Data"
    assert data["columns"]["occurrence_time"] == "Occurrence_Date_Time"
    assert data["columns"]["crime_category"] == "Crime_Category"


def test_invalid_split_proportions_are_rejected() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["test_fraction"] = 0.20

    with pytest.raises(ConfigurationError, match="sum to 1"):
        validate_config(config)


def test_unlisted_aggregation_window_is_rejected() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["aggregation_windows"] = ["24h", "2w"]

    with pytest.raises(ConfigurationError, match="unique values"):
        validate_config(config)


def test_primary_aggregation_window_must_be_enabled() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["primary_aggregation_window"] = "3d"
    config["experiment"]["aggregation_windows"] = ["24h", "1w"]

    with pytest.raises(ConfigurationError, match="must be included"):
        validate_config(config)


def test_forecast_horizon_must_be_a_positive_interval_count() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["forecast_horizon_intervals"] = None

    with pytest.raises(ConfigurationError, match="positive integer"):
        validate_config(config)


def test_binary_hotspot_target_is_rejected() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["forecast_target"] = "hotspot"

    with pytest.raises(ConfigurationError, match="incident_count_by_location_time_category"):
        validate_config(config)
