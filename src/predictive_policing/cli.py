"""Command-line utilities for the predictive policing pipeline."""

import argparse
import sys
from pathlib import Path

from predictive_policing.config import ConfigurationError, load_config
from predictive_policing.data.pipeline import prepare_incident_data


def main() -> int:
    parser = argparse.ArgumentParser(prog="predictive-policing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Config Check
    config_parser = subparsers.add_parser("config-check", help="Validate an experiment YAML file.")
    config_parser.add_argument("--config", type=Path, required=True)

    # 2. Stage 1: Data Prepare
    prepare_parser = subparsers.add_parser(
        "data-prepare", help="Validate and prepare the configured incident CSV."
    )
    prepare_parser.add_argument("--config", type=Path, required=True)

    # 3. Stage 2: Spatial-Temporal Pipeline & File Handoff
    st_parser = subparsers.add_parser(
        "spatial-temporal",
        help="Execute Stage 2: build spatial graph, space-time cube, sequences, and export files & figures.",
    )
    st_parser.add_argument("--config", type=Path, required=True)
    st_parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip generating figure plots in outputs/figures/.",
    )
    st_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for exported figures (default 300).",
    )

    # 4. Interactive Dashboard
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the interactive NiceGUI decision-support dashboard.",
    )
    dash_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path (default configs/default.yaml).",
    )
    dash_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve dashboard on (default 8080).",
    )
    dash_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface (default 0.0.0.0).",
    )

    arguments = parser.parse_args()

    try:
        config = load_config(arguments.config)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    project_root = arguments.config.resolve().parent.parent

    # Execute Dashboard
    if arguments.command == "dashboard":
        try:
            from predictive_policing.visualization.dashboard import run_dashboard
        except ModuleNotFoundError as exc:
            print(
                f"Error: Dashboard dependencies are missing ({exc}).\n"
                "Please install the visualization extra with:\n"
                "  uv sync --extra visualization",
                file=sys.stderr,
            )
            return 1

        processed_dir = project_root / "data" / "processed"
        print(
            f"Launching Zone 17 Predictive Policing Dashboard on http://{arguments.host}:{arguments.port}"
        )

        run_dashboard(
            host=arguments.host,
            port=arguments.port,
            data_dir=processed_dir,
        )
        return 0

    # Execute Stage 1
    if arguments.command == "data-prepare":
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

    # Execute Stage 2: Spatial-Temporal Engine & Data Handoff
    if arguments.command == "spatial-temporal":
        data_config = config["data"]
        exp_config = config["experiment"]
        spatial_cfg = config.get("spatial_graph", {})

        prepared_path = project_root / data_config["prepared_path"]
        if not prepared_path.exists():
            print(
                f"Prepared data not found at '{prepared_path}'. Run `data-prepare` first.",
                file=sys.stderr,
            )
            return 2

        import pandas as pd

        from predictive_policing.spatial_temporal import (
            build_spatial_graph,
            build_spatiotemporal_cube,
            build_tabular_lag_features,
            compute_division_centroids,
            generate_lookback_sequences,
            save_sequence_split,
            save_spatial_graph,
            save_spatiotemporal_cube,
            save_tabular_lag_features,
            split_sequences_chronological,
        )
        from predictive_policing.visualization import generate_all_stage2_figures

        df = pd.read_csv(prepared_path)
        print(f"Loaded prepared incidents: {len(df):,} records from {prepared_path}")

        processed_dir = prepared_path.parent
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Resolve columns from model prepared names or fallback to config names
        div_col = (
            "model_division_outpost"
            if "model_division_outpost" in df.columns
            else data_config["columns"]["division_outpost"]
        )
        lat_col = (
            "model_latitude"
            if "model_latitude" in df.columns
            else data_config["columns"]["latitude"]
        )
        lon_col = (
            "model_longitude"
            if "model_longitude" in df.columns
            else data_config["columns"]["longitude"]
        )
        time_col = (
            "model_occurred_at"
            if "model_occurred_at" in df.columns
            else data_config["columns"]["occurrence_time"]
        )
        cat_col = (
            "model_crime_category"
            if "model_crime_category" in df.columns
            else data_config["columns"]["crime_category"]
        )

        # 1. Compute Centroids strictly from real incident GPS
        centroids_df = compute_division_centroids(
            df,
            division_col=div_col,
            lat_col=lat_col,
            lon_col=lon_col,
        )
        node_names = centroids_df[div_col].tolist()
        coords = centroids_df[[lat_col, lon_col]].to_numpy(dtype=float)

        sigma = float(spatial_cfg.get("bandwidth_sigma_km", 25.0))
        cutoff = float(spatial_cfg.get("distance_cutoff_km", 50.0))

        # 2. Build and save SpatialGraph
        graph = build_spatial_graph(
            node_names=node_names,
            coordinates=coords,
            bandwidth_sigma=sigma,
            distance_cutoff=cutoff,
        )
        graph_files = save_spatial_graph(graph, processed_dir, centroids_df=centroids_df)
        print(
            f"[Stage 2] Saved SpatialGraph: {graph_files['graph_npz']} (N={graph.num_nodes}, E={graph.num_edges})"
        )
        print(f"[Stage 2] Saved Centroids:    {graph_files['centroids_csv']}")

        # 3. Build and save SpatiotemporalCube (primary window)
        window = exp_config.get("primary_aggregation_window", "24h")
        tz = data_config.get("timezone", "Africa/Lagos")
        cube = build_spatiotemporal_cube(
            df,
            window=window,
            divisions=graph.node_names,
            time_col=time_col,
            division_col=div_col,
            category_col=cat_col,
            timezone=tz,
        )
        cube_files = save_spatiotemporal_cube(
            cube, processed_dir, prefix=f"spatiotemporal_cube_{window}"
        )
        print(
            f"[Stage 2] Saved Cube Tensor:  {cube_files['cube_npz']} "
            f"(Shape: {cube.count_tensor.shape}, Sparsity: {cube.sparsity:.1%})"
        )
        print(f"[Stage 2] Saved Flat Cube:    {cube_files['flat_csv']}")

        # 4. Generate and save SequenceSplit
        lookback = int(exp_config.get("lookback_intervals", 12))
        horizon = int(exp_config.get("forecast_horizon_intervals", 1))
        train_frac = float(exp_config.get("train_fraction", 0.70))
        val_frac = float(exp_config.get("validation_fraction", 0.15))
        test_frac = float(exp_config.get("test_fraction", 0.15))

        x_tensor, y_tensor, target_times, feat_names = generate_lookback_sequences(
            cube=cube,
            lookback_intervals=lookback,
            forecast_horizon=horizon,
            transform_counts="log1p",
        )
        split = split_sequences_chronological(
            X=x_tensor,
            Y=y_tensor,
            target_intervals=target_times,
            feature_names=feat_names,
            divisions=cube.divisions,
            categories=cube.categories,
            train_fraction=train_frac,
            val_fraction=val_frac,
            test_fraction=test_frac,
            lookback_intervals=lookback,
            forecast_horizon=horizon,
        )
        seq_files = save_sequence_split(split, processed_dir, prefix=f"sequences_{window}")
        print(
            f"[Stage 2] Saved Sequences:    {seq_files['sequences_npz']} "
            f"(Train: {split.n_train}, Val: {split.n_val}, Test: {split.n_test})"
        )

        # 5. Build and save Tabular Lag Features
        tab_df = build_tabular_lag_features(cube, spatial_graph=graph)
        tab_path = save_tabular_lag_features(
            tab_df, processed_dir, filename=f"tabular_lag_features_{window}.csv"
        )
        print(
            f"[Stage 2] Saved Tabular Lags: {tab_path} "
            f"({tab_df.shape[0]:,} rows x {tab_df.shape[1]} cols)"
        )

        # 6. Generate Figures
        if not arguments.no_visuals:
            figures_dir = project_root / "outputs" / "figures"
            fig_results = generate_all_stage2_figures(
                graph, cube, output_dir=figures_dir, dpi=arguments.dpi
            )
            print(f"[Stage 2] Generated {len(fig_results)} Figures in {figures_dir}:")
            for k, p in fig_results.items():
                print(f"          - {k}: {p.name}")

        print(
            "\nStage 2 execution completed successfully. Data handoff files are ready for Stage 3 & 4."
        )
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
