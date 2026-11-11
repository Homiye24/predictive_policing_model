"""Sliding look-back sequence generation, chronological splitting, and feature fusion.

This module converts a continuous SpatiotemporalCube [T, N, C] into:
1. Deep learning sliding look-back sequence tensors:
   - Input X: [M, L, N, D] where L is lookback (default 12), N is divisions,
     and D is feature channels (category counts + cyclical temporal features).
   - Target Y: [M, N, C] (single-step) or [M, H, N, C] (multi-step horizon).
2. Tabular ML baseline feature matrices with autoregressive lags, rolling statistics,
   and spatial neighbor spillover lag (via normalized graph adjacency).
3. Leakage-free chronological train/val/test splitting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from predictive_policing.spatial_temporal.aggregation import (
    TEMPORAL_FEATURE_NAMES,
    SpatiotemporalCube,
)

if TYPE_CHECKING:
    from predictive_policing.spatial_temporal.graph import SpatialGraph


@dataclass(frozen=True)
class SequenceSplit:
    """Holds strictly chronological train, validation, and test sequence partitions."""

    X_train: np.ndarray  # [M_train, L, N, D]
    Y_train: np.ndarray  # [M_train, N, C] or [M_train, H, N, C]
    X_val: np.ndarray  # [M_val, L, N, D]
    Y_val: np.ndarray  # [M_val, N, C] or [M_val, H, N, C]
    X_test: np.ndarray  # [M_test, L, N, D]
    Y_test: np.ndarray  # [M_test, N, C] or [M_test, H, N, C]

    train_target_intervals: list[pd.Timestamp]
    val_target_intervals: list[pd.Timestamp]
    test_target_intervals: list[pd.Timestamp]

    feature_names: list[str]
    divisions: list[str]
    categories: list[str]
    lookback_intervals: int
    forecast_horizon: int
    transform_counts: str | None

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_nodes(self) -> int:
        return len(self.divisions)

    @property
    def n_categories(self) -> int:
        return len(self.categories)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def summary(self) -> dict[str, int | str | float]:
        """Return structural summary metadata for auditing and logging."""
        return {
            "n_train_samples": self.n_train,
            "n_val_samples": self.n_val,
            "n_test_samples": self.n_test,
            "lookback_intervals": self.lookback_intervals,
            "forecast_horizon": self.forecast_horizon,
            "num_divisions": self.n_nodes,
            "num_categories": self.n_categories,
            "num_feature_channels": self.n_features,
            "transform_counts": str(self.transform_counts),
            "train_start": str(self.train_target_intervals[0])
            if self.train_target_intervals
            else "",
            "train_end": str(self.train_target_intervals[-1])
            if self.train_target_intervals
            else "",
            "val_start": str(self.val_target_intervals[0]) if self.val_target_intervals else "",
            "val_end": str(self.val_target_intervals[-1]) if self.val_target_intervals else "",
            "test_start": str(self.test_target_intervals[0]) if self.test_target_intervals else "",
            "test_end": str(self.test_target_intervals[-1]) if self.test_target_intervals else "",
        }


def inverse_transform_counts(counts: np.ndarray, transform: str | None = "log1p") -> np.ndarray:
    """Invert transformed count predictions back to original incident scale.

    Args:
        counts: Array of transformed predicted values.
        transform: 'log1p' to invert via expm1, or None for identity.

    Returns:
        Non-negative incident count array with same shape as input.
    """
    if transform is None or transform == "raw":
        return np.clip(counts, 0.0, None)
    if transform == "log1p":
        # Invert: exp(x) - 1, clipping negatives at 0.0
        inverted = np.expm1(np.clip(counts, 0.0, None))
        return np.clip(inverted, 0.0, None)
    raise ValueError(f"Unsupported count transformation: '{transform}'")


def generate_lookback_sequences(
    cube: SpatiotemporalCube,
    lookback_intervals: int = 12,
    forecast_horizon: int = 1,
    stride: int = 1,
    transform_counts: Literal["log1p", "raw"] | None = "log1p",
    include_temporal_features: bool = True,
    static_node_features: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp], list[str]]:
    """Construct sliding look-back sequences from a SpatiotemporalCube.

    For each window of length L (lookback_intervals), the model receives
    historical node features [L, N, D] and is evaluated on target [N, C]
    at interval t + L (if forecast_horizon=1), or [H, N, C] (if H > 1).

    Node features per lag step include:
    - Transformed category counts (dimension C)
    - Cyclical calendar features (dimension K=7, broadcast across all nodes)
    - Optional static node features (dimension D_s, broadcast across all time steps)

    Args:
        cube: Continuous SpatiotemporalCube of shape [T, N, C].
        lookback_intervals: Number of historical time intervals to condition on (L).
        forecast_horizon: Forecast horizon step count (H >= 1).
        stride: Step stride between consecutive sliding windows (default 1).
        transform_counts: 'log1p' (default) to apply log(1+x), or 'raw'/None.
        include_temporal_features: If True, appends 7 cyclical calendar features.
        static_node_features: Optional array of shape [N, D_s] with node attributes.

    Returns:
        Tuple of:
        - X: Input array of shape [M, L, N, D], dtype float32.
        - Y: Target array of shape [M, N, C] (or [M, H, N, C] if H > 1), dtype float32.
        - target_intervals: List of M target timestamps (for H=1) or start of targets.
        - feature_names: List of strings naming the D feature channels.
    """
    if lookback_intervals <= 0:
        raise ValueError(f"lookback_intervals must be positive, got {lookback_intervals}")
    if forecast_horizon <= 0:
        raise ValueError(f"forecast_horizon must be positive, got {forecast_horizon}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    t_total = len(cube.intervals)
    min_required = lookback_intervals + forecast_horizon
    if t_total < min_required:
        raise ValueError(
            f"SpatiotemporalCube has {t_total} intervals, but at least {min_required} "
            f"(lookback={lookback_intervals} + horizon={forecast_horizon}) are required."
        )

    # 1. Transform counts if requested
    raw_counts = cube.count_tensor  # [T, N, C]
    if transform_counts == "log1p":
        input_counts = np.log1p(raw_counts).astype(np.float32)
    elif transform_counts in (None, "raw"):
        input_counts = raw_counts.astype(np.float32)
    else:
        raise ValueError(f"Unsupported transform_counts: '{transform_counts}'")

    # 2. Build feature channels for each (t, n)
    feature_names: list[str] = [f"count_{cat}" for cat in cube.categories]
    channels = [input_counts]  # [T, N, C]

    if include_temporal_features:
        # Broadcast temporal features [T, 7] across N nodes -> [T, N, 7]
        temp_feat = cube.temporal_features[:, np.newaxis, :]  # [T, 1, 7]
        temp_feat = np.repeat(temp_feat, len(cube.divisions), axis=1)  # [T, N, 7]
        channels.append(temp_feat.astype(np.float32))
        feature_names.extend(list(TEMPORAL_FEATURE_NAMES))

    if static_node_features is not None:
        if static_node_features.shape[0] != len(cube.divisions):
            raise ValueError(
                f"static_node_features must have {len(cube.divisions)} rows, "
                f"got {static_node_features.shape[0]}"
            )
        # Broadcast [N, D_s] across T intervals -> [T, N, D_s]
        d_s = static_node_features.shape[1]
        static_feat = static_node_features[np.newaxis, :, :]  # [1, N, D_s]
        static_feat = np.repeat(static_feat, t_total, axis=0)  # [T, N, D_s]
        channels.append(static_feat.astype(np.float32))
        feature_names.extend([f"static_feat_{i}" for i in range(d_s)])

    node_features = np.concatenate(channels, axis=2)  # [T, N, D]

    # 3. Slide window across timeline
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    target_intervals: list[pd.Timestamp] = []

    last_valid_start = t_total - (lookback_intervals + forecast_horizon) + 1
    for start_idx in range(0, last_valid_start, stride):
        end_lookback = start_idx + lookback_intervals
        target_start = end_lookback
        target_end = target_start + forecast_horizon

        # X sample: [L, N, D]
        x_sample = node_features[start_idx:end_lookback]
        # Y sample: [H, N, C] or [N, C] if H == 1
        y_slice = raw_counts[target_start:target_end]
        if forecast_horizon == 1:
            y_sample = y_slice[0]  # [N, C]
        else:
            y_sample = y_slice  # [H, N, C]

        x_list.append(x_sample)
        y_list.append(y_sample)
        target_intervals.append(cube.intervals[target_start])

    x_tensor = np.stack(x_list, axis=0).astype(np.float32)
    y_tensor = np.stack(y_list, axis=0).astype(np.float32)

    return x_tensor, y_tensor, target_intervals, feature_names


def split_sequences_chronological(
    X: np.ndarray,
    Y: np.ndarray,
    target_intervals: list[pd.Timestamp],
    feature_names: list[str],
    divisions: list[str],
    categories: list[str],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    lookback_intervals: int = 12,
    forecast_horizon: int = 1,
    transform_counts: str | None = "log1p",
) -> SequenceSplit:
    """Split sliding look-back sequences chronologically without data leakage.

    Splits are determined strictly by sequence target timestamps:
    - Train: First train_fraction of sequence target timestamps.
    - Validation: Next val_fraction of sequence target timestamps.
    - Test: Remaining test_fraction of sequence target timestamps.

    Zero leakage: Not a single validation or test target label is present in
    the training set.

    Args:
        X: Input tensor [M, L, N, D].
        Y: Target tensor [M, N, C] or [M, H, N, C].
        target_intervals: Chronological list of M target timestamps.
        feature_names: List of D feature channel names.
        divisions: List of N division names.
        categories: List of C category names.
        train_fraction: Fraction of sequences for training (default 0.70).
        val_fraction: Fraction of sequences for validation (default 0.15).
        test_fraction: Fraction of sequences for testing (default 0.15).
        lookback_intervals: Historical window length (L).
        forecast_horizon: Target horizon length (H).
        transform_counts: Name of transformation applied to counts.

    Returns:
        SequenceSplit dataclass containing train, val, test arrays and metadata.
    """
    total_samples = len(X)
    if total_samples != len(Y) or total_samples != len(target_intervals):
        raise ValueError(
            f"Length mismatch: X has {len(X)}, Y has {len(Y)}, "
            f"target_intervals has {len(target_intervals)}"
        )

    fractions_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fractions_sum, 1.0, atol=1e-3):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions_sum:.4f}")

    if total_samples < 3:
        raise ValueError(f"Need at least 3 samples to split train/val/test, got {total_samples}")

    train_end = int(np.floor(total_samples * train_fraction))
    val_end = int(np.floor(total_samples * (train_fraction + val_fraction)))

    # Ensure each split has at least 1 sample
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    if val_end >= total_samples:
        val_end = total_samples - 1

    x_train, y_train = X[:train_end], Y[:train_end]
    x_val, y_val = X[train_end:val_end], Y[train_end:val_end]
    x_test, y_test = X[val_end:], Y[val_end:]

    train_times = target_intervals[:train_end]
    val_times = target_intervals[train_end:val_end]
    test_times = target_intervals[val_end:]

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
        feature_names=feature_names,
        divisions=divisions,
        categories=categories,
        lookback_intervals=lookback_intervals,
        forecast_horizon=forecast_horizon,
        transform_counts=transform_counts,
    )


def build_tabular_lag_features(
    cube: SpatiotemporalCube,
    spatial_graph: SpatialGraph | None = None,
    lags: Sequence[int] = (1, 2, 3, 7, 12),
    rolling_windows: Sequence[int] = (3, 7, 12),
    include_spatial_lag: bool = True,
) -> pd.DataFrame:
    r"""Generate tabular feature matrix with autoregressive lags and spatial spillover.

    Constructs a tabular DataFrame suited for baseline machine learning models
    (Random Forest, XGBoost, CatBoost, Poisson Regression).

    For each division i and time interval t:
    - Total incident count at lag k: y_{i, t-k}
    - Rolling mean, rolling max, and rolling sum over past w intervals
    - Spatial neighbor spillover lag: \sum_j A_{ij} y_{j, t-1}
      (measuring whether crime in neighboring divisions yesterday impacts division i today)
    - Cyclical calendar features at interval t
    - Target: total crime count at interval t (and per-category counts)

    Args:
        cube: Continuous SpatiotemporalCube.
        spatial_graph: Optional SpatialGraph providing normalized adjacency.
        lags: Integer sequence of lag steps to include (e.g. 1, 2, 3, 7, 12).
        rolling_windows: Integer sequence of rolling window sizes.
        include_spatial_lag: If True and spatial_graph is provided, calculates
                             spatial neighborhood spillover lag.

    Returns:
        pd.DataFrame sorted chronologically by interval, with complete lag features.
    """
    total_counts = cube.total_counts_by_node  # [T, N]
    t_total, _n_nodes = total_counts.shape
    max_lag = max(max(lags), max(rolling_windows) if rolling_windows else 0)

    if t_total <= max_lag:
        raise ValueError(
            f"Cube has {t_total} intervals, which is insufficient for max lag/window {max_lag}"
        )

    records: list[dict[str, object]] = []

    # Pre-extract temporal features as dictionary for fast lookup
    temp_dict = {ts: cube.temporal_features[i] for i, ts in enumerate(cube.intervals)}

    # Pre-extract normalized adjacency if spatial lag requested
    norm_adj = None
    if include_spatial_lag and spatial_graph is not None:
        norm_adj = spatial_graph.normalized_adjacency  # [N, N]

    for t_idx in range(max_lag, t_total):
        current_ts = cube.intervals[t_idx]
        current_temp = temp_dict[current_ts]

        # Spatial neighbor spillover at lag 1: A @ total_counts[t_idx - 1]
        spatial_spillover = None
        if norm_adj is not None:
            spatial_spillover = norm_adj @ total_counts[t_idx - 1]  # [N]

        for n_idx, div_name in enumerate(cube.divisions):
            row: dict[str, object] = {
                "interval": current_ts,
                "division_outpost": div_name,
                "target_total_incidents": total_counts[t_idx, n_idx],
            }

            # Per-category targets
            for c_idx, cat_name in enumerate(cube.categories):
                row[f"target_count_{cat_name}"] = cube.count_tensor[t_idx, n_idx, c_idx]

            # Autoregressive lag features
            for lag_k in lags:
                row[f"lag_{lag_k}_total"] = total_counts[t_idx - lag_k, n_idx]

            # Rolling statistics over past history
            for window_w in rolling_windows:
                hist_window = total_counts[t_idx - window_w : t_idx, n_idx]
                row[f"rolling_mean_{window_w}"] = float(np.mean(hist_window))
                row[f"rolling_max_{window_w}"] = float(np.max(hist_window))
                row[f"rolling_sum_{window_w}"] = float(np.sum(hist_window))

            # Spatial neighborhood spillover lag
            if spatial_spillover is not None:
                row["spatial_neighbor_lag_1"] = float(spatial_spillover[n_idx])

            # Cyclical temporal calendar features
            for f_idx, feat_name in enumerate(TEMPORAL_FEATURE_NAMES):
                row[feat_name] = float(current_temp[f_idx])

            records.append(row)

    df = pd.DataFrame.from_records(records)
    return df


def to_torch_dataset(X: np.ndarray, Y: np.ndarray):
    """Convert NumPy sequence arrays to a PyTorch TensorDataset.

    Args:
        X: Input tensor [M, L, N, D], float32.
        Y: Target tensor [M, N, C] or [M, H, N, C], float32.

    Returns:
        torch.utils.data.TensorDataset: PyTorch dataset yielding (x, y) tensor tuples.

    Raises:
        ImportError: If PyTorch is not installed in the environment.
    """
    try:
        import torch
        from torch.utils.data import TensorDataset

        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)).float(),
            torch.from_numpy(np.ascontiguousarray(Y)).float(),
        )
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. To use to_torch_dataset, install torch with: "
            "`uv pip install torch` or install the deep-learning extra."
        ) from exc
