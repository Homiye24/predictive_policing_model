from pathlib import Path

import pandas as pd
import pytest
import yaml

from predictive_policing.cli import main


def _setup_test_project(tmp_path: Path) -> Path:
    """Create a minimal project structure with config and prepared incident data."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    figures_dir = tmp_path / "outputs" / "figures"
    figures_dir.mkdir(parents=True)

    # 1. Create realistic sample incidents
    divisions = ["Akure", "Owo", "Ikole"]
    categories = ["Theft", "Burglary"]
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    records = []
    for d_idx, date in enumerate(dates):
        for div in divisions:
            records.append(
                {
                    "model_occurred_at": date.isoformat(),
                    "model_division_outpost": div,
                    "model_crime_category": categories[d_idx % len(categories)],
                    "model_latitude": 7.25 if div == "Akure" else (7.20 if div == "Owo" else 7.80),
                    "model_longitude": 5.20 if div == "Akure" else (5.58 if div == "Owo" else 5.51),
                }
            )

    prepared_csv = processed_dir / "cleaned_incidents.csv"
    pd.DataFrame(records).to_csv(prepared_csv, index=False)

    # 2. Create minimal valid config
    config_data = {
        "project": {"name": "test-project", "random_seed": 42},
        "data": {
            "input_path": "data/raw/crimedata.xlsx",
            "prepared_path": "data/processed/cleaned_incidents.csv",
            "quality_report_path": "data/processed/report.json",
            "timezone": "Africa/Lagos",
            "missing_category_label": "unknown",
            "columns": {
                "division_outpost": "Division",
                "latitude": "Latitude",
                "longitude": "Longitude",
                "occurrence_time": "Occurrence",
                "crime_category": "Category",
            },
        },
        "experiment": {
            "aggregation_windows": ["24h"],
            "primary_aggregation_window": "24h",
            "forecast_target": "incident_count_by_location_time_category",
            "ranking_dimensions": ["division_outpost", "time_interval", "crime_category"],
            "forecast_horizon_intervals": 1,
            "lookback_intervals": 5,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "spatial_graph": {
            "bandwidth_sigma_km": 25.0,
            "distance_cutoff_km": 50.0,
        },
    }

    config_path = config_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


def test_cli_spatial_temporal_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _setup_test_project(tmp_path)

    test_args = [
        "predictive-policing",
        "spatial-temporal",
        "--config",
        str(config_path),
        "--dpi",
        "80",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main()
    assert exit_code == 0

    # Verify all expected Stage 2 data files were created on disk
    processed_dir = tmp_path / "data" / "processed"
    assert (processed_dir / "spatial_graph.npz").exists()
    assert (processed_dir / "division_centroids.csv").exists()
    assert (processed_dir / "spatiotemporal_cube_24h.npz").exists()
    assert (processed_dir / "spatiotemporal_cube_24h_flat.csv").exists()
    assert (processed_dir / "sequences_24h.npz").exists()
    assert (processed_dir / "tabular_lag_features_24h.csv").exists()

    # Verify all 4 visualization figures were created on disk
    figures_dir = tmp_path / "outputs" / "figures"
    assert (figures_dir / "spatial_graph_network.png").exists()
    assert (figures_dir / "spatiotemporal_cube_heatmap.png").exists()
    assert (figures_dir / "temporal_crime_distribution.png").exists()
    assert (figures_dir / "lookback_window_diagram.png").exists()


def test_cli_spatial_temporal_missing_prepared_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "bad_config.yaml"

    config_data = {
        "project": {"name": "test", "random_seed": 42},
        "data": {
            "input_path": "nonexistent.xlsx",
            "prepared_path": "nonexistent_prepared.csv",
            "quality_report_path": "report.json",
            "timezone": "Africa/Lagos",
            "missing_category_label": "unknown",
            "columns": {},
        },
        "experiment": {
            "aggregation_windows": ["24h"],
            "primary_aggregation_window": "24h",
            "forecast_horizon_intervals": 1,
            "lookback_intervals": 12,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    test_args = ["predictive-policing", "spatial-temporal", "--config", str(config_path)]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main()
    assert exit_code == 2
