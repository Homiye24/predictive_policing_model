"""Serialization and file handoff routines for spatial-temporal pipeline artifacts.

Saves and restores:
1. SpatialGraph -> spatial_graph.npz and division_centroids.csv
2. SpatiotemporalCube -> spatiotemporal_cube_{window}.npz and spatiotemporal_flat_{window}.csv
3. SequenceSplit -> sequences_{window}.npz
4. Tabular lag features -> tabular_lag_features_{window}.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictive_policing.spatial_temporal.aggregation import SpatiotemporalCube
from predictive_policing.spatial_temporal.graph import SpatialGraph
from predictive_policing.spatial_temporal.sequences import SequenceSplit


def save_spatial_graph(
    graph: SpatialGraph,
    output_dir: Path,
    centroids_df: pd.DataFrame | None = None,
    filename_prefix: str = "spatial_graph",
) -> dict[str, Path]:
    """Serialize a SpatialGraph and its centroids table to disk.

    Args:
        graph: SpatialGraph object to save.
        output_dir: Target directory (e.g. data/processed/).
        centroids_df: Optional dataframe of division centroids to save as CSV.
                      If not provided, constructed automatically from graph.centroids.
        filename_prefix: Base filename for the NPZ archive.

    Returns:
        Dictionary mapping artifact keys ('graph_npz', 'centroids_csv') to saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{filename_prefix}.npz"
    np.savez_compressed(
        npz_path,
        node_names=np.array(graph.node_names, dtype=object),
        centroids=graph.centroids.astype(np.float32),
        distance_matrix=graph.distance_matrix.astype(np.float32),
        adjacency_matrix=graph.adjacency_matrix.astype(np.float32),
        normalized_adjacency=graph.normalized_adjacency.astype(np.float32),
        edge_index=graph.edge_index.astype(np.int64),
        edge_weights=graph.edge_weights.astype(np.float32),
        bandwidth_sigma=np.float32(graph.bandwidth_sigma),
        distance_cutoff=np.float32(graph.distance_cutoff),
    )

    csv_path = out_dir / "division_centroids.csv"
    if centroids_df is not None:
        centroids_df.to_csv(csv_path, index=False)
    else:
        # Build DataFrame directly from graph attributes
        auto_df = pd.DataFrame(
            {
                "division_outpost": graph.node_names,
                "latitude": graph.centroids[:, 0],
                "longitude": graph.centroids[:, 1],
                "provenance": ["incident_median"] * len(graph.node_names),
            }
        )
        auto_df.to_csv(csv_path, index=False)

    return {"graph_npz": npz_path, "centroids_csv": csv_path}


def load_spatial_graph(input_path_or_dir: Path, filename_prefix: str = "spatial_graph") -> SpatialGraph:
    """Load and reconstruct a SpatialGraph from disk.

    Args:
        input_path_or_dir: Path to the NPZ file or directory containing it.
        filename_prefix: Name of the file prefix if directory provided.

    Returns:
        Reconstructed SpatialGraph dataclass.
    """
    path = Path(input_path_or_dir)
    if path.is_dir():
        npz_path = path / f"{filename_prefix}.npz"
    else:
        npz_path = path

    if not npz_path.exists():
        raise FileNotFoundError(f"SpatialGraph file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        return SpatialGraph(
            node_names=[str(name) for name in data["node_names"]],
            centroids=data["centroids"].astype(np.float32),
            distance_matrix=data["distance_matrix"].astype(np.float32),
            adjacency_matrix=data["adjacency_matrix"].astype(np.float32),
            normalized_adjacency=data["normalized_adjacency"].astype(np.float32),
            edge_index=data["edge_index"].astype(np.int64),
            edge_weights=data["edge_weights"].astype(np.float32),
            bandwidth_sigma=float(data["bandwidth_sigma"]),
            distance_cutoff=float(data["distance_cutoff"]),
        )


def save_spatiotemporal_cube(
    cube: SpatiotemporalCube,
    output_dir: Path,
    prefix: str = "spatiotemporal_cube_24h",
    save_flat_csv: bool = True,
) -> dict[str, Path]:
    """Serialize a SpatiotemporalCube and optional flat CSV to disk.

    Args:
        cube: SpatiotemporalCube object to save.
        output_dir: Target directory.
        prefix: Filename prefix (e.g. spatiotemporal_cube_24h).
        save_flat_csv: If True, also flattens and saves a readable CSV.

    Returns:
        Dictionary mapping artifact keys ('cube_npz', 'flat_csv') to saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{prefix}.npz"
    # Store intervals as ISO string timestamps
    interval_strs = np.array([ts.isoformat() for ts in cube.intervals], dtype=object)

    np.savez_compressed(
        npz_path,
        count_tensor=cube.count_tensor.astype(np.float32),
        temporal_features=cube.temporal_features.astype(np.float32),
        intervals=interval_strs,
        divisions=np.array(cube.divisions, dtype=object),
        categories=np.array(cube.categories, dtype=object),
        window=np.array(cube.window),
        timezone=np.array(cube.timezone),
    )

    results = {"cube_npz": npz_path}

    if save_flat_csv:
        flat_csv_path = out_dir / f"{prefix}_flat.csv"
        flat_df = cube.to_flat_dataframe()
        flat_df.to_csv(flat_csv_path, index=False)
        results["flat_csv"] = flat_csv_path

    return results


def load_spatiotemporal_cube(
    input_path_or_dir: Path,
    prefix: str = "spatiotemporal_cube_24h",
) -> SpatiotemporalCube:
    """Load and reconstruct a SpatiotemporalCube from disk.

    Args:
        input_path_or_dir: Path to the NPZ file or directory containing it.
        prefix: Prefix name if directory provided.

    Returns:
        Reconstructed SpatiotemporalCube dataclass.
    """
    path = Path(input_path_or_dir)
    if path.is_dir():
        npz_path = path / f"{prefix}.npz"
    else:
        npz_path = path

    if not npz_path.exists():
        raise FileNotFoundError(f"SpatiotemporalCube file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        tz_str = str(data["timezone"])
        # Reconstruct list of timezone-aware pd.Timestamp objects
        intervals = [pd.Timestamp(ts) for ts in data["intervals"]]

        return SpatiotemporalCube(
            intervals=intervals,
            divisions=[str(d) for d in data["divisions"]],
            categories=[str(c) for c in data["categories"]],
            count_tensor=data["count_tensor"].astype(np.float32),
            temporal_features=data["temporal_features"].astype(np.float32),
            window=str(data["window"]),
            timezone=tz_str,
        )


def save_sequence_split(
    split: SequenceSplit,
    output_dir: Path,
    prefix: str = "sequences_24h",
) -> dict[str, Path]:
    """Serialize train, validation, and test SequenceSplit tensors to disk.

    Args:
        split: SequenceSplit object to save.
        output_dir: Target directory.
        prefix: Filename prefix.

    Returns:
        Dictionary with 'sequences_npz' path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{prefix}.npz"

    train_ts = np.array([ts.isoformat() for ts in split.train_target_intervals], dtype=object)
    val_ts = np.array([ts.isoformat() for ts in split.val_target_intervals], dtype=object)
    test_ts = np.array([ts.isoformat() for ts in split.test_target_intervals], dtype=object)

    np.savez_compressed(
        npz_path,
        X_train=split.X_train.astype(np.float32),
        Y_train=split.Y_train.astype(np.float32),
        X_val=split.X_val.astype(np.float32),
        Y_val=split.Y_val.astype(np.float32),
        X_test=split.X_test.astype(np.float32),
        Y_test=split.Y_test.astype(np.float32),
        train_target_intervals=train_ts,
        val_target_intervals=val_ts,
        test_target_intervals=test_ts,
        feature_names=np.array(split.feature_names, dtype=object),
        divisions=np.array(split.divisions, dtype=object),
        categories=np.array(split.categories, dtype=object),
        lookback_intervals=np.int32(split.lookback_intervals),
        forecast_horizon=np.int32(split.forecast_horizon),
        transform_counts=np.array(split.transform_counts if split.transform_counts is not None else ""),
    )

    return {"sequences_npz": npz_path}


def load_sequence_split(
    input_path_or_dir: Path,
    prefix: str = "sequences_24h",
) -> SequenceSplit:
    """Load and reconstruct a SequenceSplit from disk.

    Args:
        input_path_or_dir: Path to NPZ file or directory.
        prefix: Prefix name if directory provided.

    Returns:
        Reconstructed SequenceSplit dataclass.
    """
    path = Path(input_path_or_dir)
    if path.is_dir():
        npz_path = path / f"{prefix}.npz"
    else:
        npz_path = path

    if not npz_path.exists():
        raise FileNotFoundError(f"SequenceSplit file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        raw_transform = str(data["transform_counts"])
        transform_counts = raw_transform if raw_transform else None

        return SequenceSplit(
            X_train=data["X_train"].astype(np.float32),
            Y_train=data["Y_train"].astype(np.float32),
            X_val=data["X_val"].astype(np.float32),
            Y_val=data["Y_val"].astype(np.float32),
            X_test=data["X_test"].astype(np.float32),
            Y_test=data["Y_test"].astype(np.float32),
            train_target_intervals=[pd.Timestamp(ts) for ts in data["train_target_intervals"]],
            val_target_intervals=[pd.Timestamp(ts) for ts in data["val_target_intervals"]],
            test_target_intervals=[pd.Timestamp(ts) for ts in data["test_target_intervals"]],
            feature_names=[str(f) for f in data["feature_names"]],
            divisions=[str(d) for d in data["divisions"]],
            categories=[str(c) for c in data["categories"]],
            lookback_intervals=int(data["lookback_intervals"]),
            forecast_horizon=int(data["forecast_horizon"]),
            transform_counts=transform_counts,
        )


def save_tabular_lag_features(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "tabular_lag_features_24h.csv",
) -> Path:
    """Save tabular lag features dataframe to disk as CSV.

    Args:
        df: Tabular lag features dataframe.
        output_dir: Target directory.
        filename: Output CSV filename.

    Returns:
        Path to saved CSV file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / filename
    df.to_csv(csv_path, index=False)
    return csv_path


def load_tabular_lag_features(file_path: Path) -> pd.DataFrame:
    """Load tabular lag features dataframe from disk and parse intervals as timestamps.

    Args:
        file_path: Path to CSV file.

    Returns:
        Loaded DataFrame with parsed interval column.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Tabular lag features file not found: {path}")
    df = pd.read_csv(path)
    if "interval" in df.columns:
        df["interval"] = pd.to_datetime(df["interval"])
    return df
