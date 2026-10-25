"""Command-line utilities for the project."""

import argparse
import sys
from pathlib import Path

from predictive_policing.config import ConfigurationError, load_config
from predictive_policing.data.pipeline import prepare_incident_data


def main() -> int:
    parser = argparse.ArgumentParser(prog="predictive-policing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    config_parser = subparsers.add_parser("config-check", help="Validate an experiment YAML file.")
    config_parser.add_argument("--config", type=Path, required=True)
    prepare_parser = subparsers.add_parser(
        "data-prepare", help="Validate and prepare the configured incident CSV."
    )
    prepare_parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        config = load_config(arguments.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if arguments.command == "data-prepare":
        project_root = arguments.config.resolve().parent.parent
        data_config = config["data"]
        input_path = project_root / data_config["input_path"]
        prepared_path = project_root / data_config["prepared_path"]
        report_path = project_root / data_config["quality_report_path"]
        try:
            report = prepare_incident_data(
                input_path=input_path,
                prepared_path=prepared_path,
                report_path=report_path,
                columns=data_config["columns"],
                timezone=data_config["timezone"],
                input_sheet=data_config.get("input_sheet", 0),
                missing_category_label=data_config["missing_category_label"],
            )
        except (OSError, ValueError) as error:
            print(f"Data preparation failed: {error}", file=sys.stderr)
            return 2
        print(f"Prepared {report.rows_after_exact_deduplication} incidents.")
        print(f"Prepared data: {prepared_path}")
        print(f"Quality report: {report_path}")
        print(f"Invalid occurrence timestamps: {report.invalid_occurrence_timestamps}")
        print(f"Invalid coordinate pairs: {report.invalid_coordinate_pairs}")
        print(f"Crime categories assigned to unknown: {report.missing_crime_category}")
        return 0

    experiment = config["experiment"]
    print(f"Configuration is valid: {arguments.config}")
    print(
        "Aggregation windows: "
        + ", ".join(experiment["aggregation_windows"])
        + " (run and compare each resolution separately)."
    )
    primary_window = experiment["primary_aggregation_window"]
    horizon = experiment["forecast_horizon_intervals"]
    print(
        f"Primary forecast: {horizon} interval(s) ahead at {primary_window} "
        f"resolution ({horizon * 24} hours)."
    )
    exploratory_windows = [
        window for window in experiment["aggregation_windows"] if window != primary_window
    ]
    if exploratory_windows:
        print(
            "Exploratory resolutions use the same interval-ahead count, "
            "so their forecast durations follow their aggregation windows: "
            + ", ".join(exploratory_windows)
            + "."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
