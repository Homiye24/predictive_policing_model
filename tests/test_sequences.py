import numpy as np
import pandas as pd
import pytest

from predictive_policing.spatial_temporal.aggregation import (
    SpatiotemporalCube,
    compute_interval_temporal_features,
)
from predictive_policing.spatial_temporal.graph import SpatialGraph
from predictive_policing.spatial_temporal.sequences import (
    SequenceSplit,
    build_tabular_lag_features,
    generate_lookback_sequences,
    inverse_transform_counts,
    split_sequences_chronological,
    to_torch_dataset,
)


def _make_dummy_cube(
    num_intervals: int = 30,
    num_divisions: int = 3,
    num_categories: int = 2,
) -> SpatiotemporalCube:
    """Helper to create a deterministic dummy SpatiotemporalCube for testing."""
    intervals = pd.date_range(
        start="2024-01-01",
        periods=num_intervals,
        freq="D",
        tz="Africa/Lagos",
    ).tolist()
    divisions = [f"Station_{i}" for i in range(num_divisions)]
    categories = [f"Crime_{c}" for c in range(num_categories)]

    # Deterministic count tensor: interval index + station index + category index
    t = num_intervals
    n = num_divisions
    c = num_categories
    count_tensor = np.zeros((t, n, c), dtype=np.float32)
    for i in range(t):
        for j in range(n):
            for k in range(c):
                count_tensor[i, j, k] = float(i * 10 + j * 2 + k)

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


def test_generate_lookback_sequences_shapes_and_channels():
    cube = _make_dummy_cube(num_intervals=20, num_divisions=4, num_categories=3)
    lookback = 5
    horizon = 1

    x, y, target_times, feature_names = generate_lookback_sequences(
        cube,
        lookback_intervals=lookback,
        forecast_horizon=horizon,
        stride=1,
        transform_counts="log1p",
        include_temporal_features=True,
    )

    # Expected number of samples M = T - (L + H) + 1 = 20 - (5 + 1) + 1 = 15
    expected_m = 15
    assert len(x) == expected_m
    assert len(y) == expected_m
    assert len(target_times) == expected_m

    # X shape: [M, L, N, D]
    # D = 3 crime categories + 7 cyclical temporal features = 10
    assert x.shape == (expected_m, 5, 4, 10)

    # Y shape: [M, N, C] = [15, 4, 3] for horizon = 1
    assert y.shape == (expected_m, 4, 3)

    assert len(feature_names) == 10
    assert feature_names[:3] == ["count_Crime_0", "count_Crime_1", "count_Crime_2"]
    assert "day_of_week_sin" in feature_names

    # First target timestamp should be interval 5
    assert target_times[0] == cube.intervals[5]
    assert target_times[-1] == cube.intervals[19]


def test_generate_lookback_multi_horizon():
    cube = _make_dummy_cube(num_intervals=20, num_divisions=3, num_categories=2)
    lookback = 4
    horizon = 3

    x, y, target_times, _ = generate_lookback_sequences(
        cube,
        lookback_intervals=lookback,
        forecast_horizon=horizon,
        stride=1,
    )

    # M = 20 - (4 + 3) + 1 = 14
    assert len(x) == 14
    assert y.shape == (14, 3, 3, 2)  # [M, H, N, C]
    assert target_times[0] == cube.intervals[4]


def test_generate_lookback_static_node_features():
    cube = _make_dummy_cube(num_intervals=15, num_divisions=3, num_categories=2)
    # 2 static features per node (e.g. lat, lon)
    static_feats = np.array([
        [7.25, 5.20],
        [7.62, 5.22],
        [7.30, 5.15],
    ], dtype=np.float32)

    x, _, _, feature_names = generate_lookback_sequences(
        cube,
        lookback_intervals=3,
        forecast_horizon=1,
        static_node_features=static_feats,
    )

    # D = 2 (categories) + 7 (temporal) + 2 (static) = 11
    assert x.shape == (12, 3, 3, 11)
    assert "static_feat_0" in feature_names
    assert "static_feat_1" in feature_names


def test_generate_lookback_insufficient_length():
    cube = _make_dummy_cube(num_intervals=5, num_divisions=2, num_categories=1)
    with pytest.raises(ValueError, match="at least 13"):
        generate_lookback_sequences(cube, lookback_intervals=12, forecast_horizon=1)


def test_inverse_transform_counts():
    raw = np.array([0.0, 1.0, 4.0, 15.0], dtype=np.float32)
    log1p_vals = np.log1p(raw)

    reconstructed = inverse_transform_counts(log1p_vals, transform="log1p")
    np.testing.assert_allclose(reconstructed, raw, rtol=1e-5, atol=1e-5)

    # Test raw pass-through
    raw_out = inverse_transform_counts(raw, transform="raw")
    np.testing.assert_allclose(raw_out, raw)


def test_split_sequences_chronological_no_leakage():
    cube = _make_dummy_cube(num_intervals=40, num_divisions=2, num_categories=2)
    x, y, target_times, feature_names = generate_lookback_sequences(
        cube,
        lookback_intervals=10,
        forecast_horizon=1,
    )
    # Total sequences M = 40 - 11 + 1 = 30
    assert len(x) == 30

    split = split_sequences_chronological(
        X=x,
        Y=y,
        target_intervals=target_times,
        feature_names=feature_names,
        divisions=cube.divisions,
        categories=cube.categories,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        lookback_intervals=10,
        forecast_horizon=1,
    )

    assert isinstance(split, SequenceSplit)
    assert split.n_train + split.n_val + split.n_test == 30
    assert split.n_train == 21
    assert split.n_val == 4
    assert split.n_test == 5

    # Strict chronological ordering: max(train) < min(val) <= max(val) < min(test)
    assert max(split.train_target_intervals) < min(split.val_target_intervals)
    assert max(split.val_target_intervals) < min(split.test_target_intervals)

    # Verification of zero target leakage: sets of target times are completely disjoint
    train_set = set(split.train_target_intervals)
    val_set = set(split.val_target_intervals)
    test_set = set(split.test_target_intervals)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    # Summary check
    summary = split.summary()
    assert summary["n_train_samples"] == 21
    assert summary["num_divisions"] == 2


def test_build_tabular_lag_features():
    cube = _make_dummy_cube(num_intervals=20, num_divisions=2, num_categories=2)

    # Create a synthetic 2-node graph for spatial neighbor lag testing
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    norm_adj = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    graph = SpatialGraph(
        node_names=["Station_0", "Station_1"],
        centroids=np.array([[7.2, 5.1], [7.3, 5.2]], dtype=np.float32),
        distance_matrix=np.array([[0.0, 15.0], [15.0, 0.0]], dtype=np.float32),
        adjacency_matrix=adj,
        normalized_adjacency=norm_adj,
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_weights=np.array([1.0, 1.0], dtype=np.float32),
        bandwidth_sigma=25.0,
        distance_cutoff=50.0,
    )

    df = build_tabular_lag_features(
        cube,
        spatial_graph=graph,
        lags=(1, 2),
        rolling_windows=(3,),
        include_spatial_lag=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Columns must include interval, division_outpost, target columns, lag columns, spatial lag
    assert "interval" in df.columns
    assert "division_outpost" in df.columns
    assert "target_total_incidents" in df.columns
    assert "target_count_Crime_0" in df.columns
    assert "lag_1_total" in df.columns
    assert "lag_2_total" in df.columns
    assert "rolling_mean_3" in df.columns
    assert "spatial_neighbor_lag_1" in df.columns
    assert "day_of_week_sin" in df.columns

    # Verify spatial lag calculation:
    # At interval index t_idx, spatial_spillover = norm_adj @ total_counts[t_idx - 1]
    # For Station_0, spatial_neighbor_lag_1 should be 0.5 * count_0[t-1] + 0.5 * count_1[t-1]
    row_0 = df[(df["division_outpost"] == "Station_0")].iloc[0]
    expected_spillover = 0.5 * row_0["lag_1_total"] + 0.5 * df[(df["division_outpost"] == "Station_1") & (df["interval"] == row_0["interval"])]["lag_1_total"].values[0]
    np.testing.assert_allclose(row_0["spatial_neighbor_lag_1"], expected_spillover, rtol=1e-5)


def test_to_torch_dataset_import_handling():
    x = np.zeros((5, 3, 2, 4), dtype=np.float32)
    y = np.zeros((5, 2, 1), dtype=np.float32)

    try:
        import torch  # noqa: F401
        dataset = to_torch_dataset(x, y)
        assert len(dataset) == 5
    except ImportError:
        with pytest.raises(ImportError, match="PyTorch is not installed"):
            to_torch_dataset(x, y)
