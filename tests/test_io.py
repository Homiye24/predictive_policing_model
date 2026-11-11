from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_policing.spatial_temporal.aggregation import (
    SpatiotemporalCube,
    compute_interval_temporal_features,
)
from predictive_policing.spatial_temporal.graph import SpatialGraph
from predictive_policing.spatial_temporal.io import (
    load_sequence_split,
    load_spatial_graph,
    load_spatiotemporal_cube,
    load_tabular_lag_features,
    save_sequence_split,
    save_spatial_graph,
    save_spatiotemporal_cube,
    save_tabular_lag_features,
)
from predictive_policing.spatial_temporal.sequences import SequenceSplit


def _create_dummy_graph() -> SpatialGraph:
    nodes = ["Akure", "Owo", "Ikole"]
    centroids = np.array([[7.25, 5.20], [7.20, 5.58], [7.80, 5.51]], dtype=np.float32)
    dist = np.array([[0.0, 42.0, 71.0], [42.0, 0.0, 68.0], [71.0, 68.0, 0.0]], dtype=np.float32)
    adj = np.array([[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    norm_adj = adj / 1.25
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weights = np.array([0.25, 0.25], dtype=np.float32)

    return SpatialGraph(
        node_names=nodes,
        centroids=centroids,
        distance_matrix=dist,
        adjacency_matrix=adj,
        normalized_adjacency=norm_adj,
        edge_index=edge_index,
        edge_weights=edge_weights,
        bandwidth_sigma=25.0,
        distance_cutoff=50.0,
    )


def _create_dummy_cube() -> SpatiotemporalCube:
    intervals = pd.date_range("2024-01-01", periods=10, freq="D", tz="Africa/Lagos").tolist()
    divisions = ["Akure", "Owo"]
    categories = ["Theft", "Assault"]
    count_tensor = np.arange(40, dtype=np.float32).reshape(10, 2, 2)
    temporal_features = compute_interval_temporal_features(intervals)

    return SpatiotemporalCube(
        intervals=intervals,
        divisions=divisions,
        categories=categories,
        count_tensor=count_tensor,
        temporal_features=temporal_features,
        window="24h",
        timezone="Africa/Lagos",
    )


def _create_dummy_split() -> SequenceSplit:
    x_train = np.ones((5, 12, 2, 9), dtype=np.float32)
    y_train = np.ones((5, 2, 2), dtype=np.float32)
    x_val = np.ones((2, 12, 2, 9), dtype=np.float32) * 2
    y_val = np.ones((2, 2, 2), dtype=np.float32) * 2
    x_test = np.ones((2, 12, 2, 9), dtype=np.float32) * 3
    y_test = np.ones((2, 2, 2), dtype=np.float32) * 3

    train_times = pd.date_range("2024-01-13", periods=5, freq="D", tz="Africa/Lagos").tolist()
    val_times = pd.date_range("2024-01-18", periods=2, freq="D", tz="Africa/Lagos").tolist()
    test_times = pd.date_range("2024-01-20", periods=2, freq="D", tz="Africa/Lagos").tolist()

    return SequenceSplit(
        X_train=x_train,
        Y_train=y_train,
        X_val=x_val,
        Y_val=y_val,
        X_test=x_test,
        Y_test=y_test,
        train_target_intervals=train_times,
        val_target_intervals=val_times,
        test_target_intervals=test_times,
        feature_names=["count_Theft", "count_Assault"] + [f"t_{i}" for i in range(7)],
        divisions=["Akure", "Owo"],
        categories=["Theft", "Assault"],
        lookback_intervals=12,
        forecast_horizon=1,
        transform_counts="log1p",
    )


def test_spatial_graph_roundtrip(tmp_path: Path):
    graph = _create_dummy_graph()
    save_result = save_spatial_graph(graph, tmp_path)

    assert "graph_npz" in save_result
    assert "centroids_csv" in save_result
    assert save_result["graph_npz"].exists()
    assert save_result["centroids_csv"].exists()

    # Verify CSV content
    centroids_df = pd.read_csv(save_result["centroids_csv"])
    assert len(centroids_df) == 3
    assert list(centroids_df["division_outpost"]) == ["Akure", "Owo", "Ikole"]

    # Load and assert exact reconstruction
    loaded_graph = load_spatial_graph(tmp_path)
    assert loaded_graph.node_names == graph.node_names
    np.testing.assert_allclose(loaded_graph.centroids, graph.centroids)
    np.testing.assert_allclose(loaded_graph.distance_matrix, graph.distance_matrix)
    np.testing.assert_allclose(loaded_graph.adjacency_matrix, graph.adjacency_matrix)
    np.testing.assert_allclose(loaded_graph.normalized_adjacency, graph.normalized_adjacency)
    np.testing.assert_array_equal(loaded_graph.edge_index, graph.edge_index)
    np.testing.assert_allclose(loaded_graph.edge_weights, graph.edge_weights)
    assert loaded_graph.bandwidth_sigma == graph.bandwidth_sigma
    assert loaded_graph.distance_cutoff == graph.distance_cutoff


def test_spatiotemporal_cube_roundtrip(tmp_path: Path):
    cube = _create_dummy_cube()
    save_result = save_spatiotemporal_cube(cube, tmp_path, prefix="test_cube")

    assert save_result["cube_npz"].exists()
    assert save_result["flat_csv"].exists()

    # Flat CSV verification (10 intervals * 2 divisions * 2 categories = 40 rows)
    flat_df = pd.read_csv(save_result["flat_csv"])
    assert len(flat_df) == 40
    assert "incident_count" in flat_df.columns
    assert "day_of_week_sin" in flat_df.columns

    # Load and assert exact reconstruction
    loaded_cube = load_spatiotemporal_cube(tmp_path, prefix="test_cube")
    assert loaded_cube.divisions == cube.divisions
    assert loaded_cube.categories == cube.categories
    assert loaded_cube.window == cube.window
    assert loaded_cube.timezone == cube.timezone
    assert len(loaded_cube.intervals) == len(cube.intervals)
    assert loaded_cube.intervals[0] == cube.intervals[0]
    np.testing.assert_allclose(loaded_cube.count_tensor, cube.count_tensor)
    np.testing.assert_allclose(loaded_cube.temporal_features, cube.temporal_features)


def test_sequence_split_roundtrip(tmp_path: Path):
    split = _create_dummy_split()
    save_result = save_sequence_split(split, tmp_path, prefix="test_seq")

    assert save_result["sequences_npz"].exists()

    loaded_split = load_sequence_split(tmp_path, prefix="test_seq")
    assert loaded_split.n_train == split.n_train
    assert loaded_split.n_val == split.n_val
    assert loaded_split.n_test == split.n_test
    np.testing.assert_allclose(loaded_split.X_train, split.X_train)
    np.testing.assert_allclose(loaded_split.Y_train, split.Y_train)
    np.testing.assert_allclose(loaded_split.X_val, split.X_val)
    np.testing.assert_allclose(loaded_split.Y_val, split.Y_val)
    np.testing.assert_allclose(loaded_split.X_test, split.X_test)
    np.testing.assert_allclose(loaded_split.Y_test, split.Y_test)
    assert loaded_split.train_target_intervals == split.train_target_intervals
    assert loaded_split.feature_names == split.feature_names
    assert loaded_split.transform_counts == split.transform_counts


def test_tabular_lag_features_roundtrip(tmp_path: Path):
    df = pd.DataFrame(
        {
            "interval": pd.date_range("2024-01-01", periods=5, freq="D"),
            "division_outpost": ["Station_A"] * 5,
            "target_total_incidents": [1.0, 2.0, 0.0, 3.0, 1.0],
            "lag_1_total": [0.0, 1.0, 2.0, 0.0, 3.0],
        }
    )

    csv_path = save_tabular_lag_features(df, tmp_path, "lags.csv")
    assert csv_path.exists()

    loaded_df = load_tabular_lag_features(csv_path)
    assert len(loaded_df) == 5
    assert list(loaded_df["division_outpost"]) == ["Station_A"] * 5
    np.testing.assert_allclose(loaded_df["target_total_incidents"], df["target_total_incidents"])


def test_missing_files_raise_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="SpatialGraph file not found"):
        load_spatial_graph(tmp_path / "non_existent")

    with pytest.raises(FileNotFoundError, match="SpatiotemporalCube file not found"):
        load_spatiotemporal_cube(tmp_path / "non_existent")

    with pytest.raises(FileNotFoundError, match="SequenceSplit file not found"):
        load_sequence_split(tmp_path / "non_existent")

    with pytest.raises(FileNotFoundError, match="Tabular lag features file not found"):
        load_tabular_lag_features(tmp_path / "non_existent.csv")
