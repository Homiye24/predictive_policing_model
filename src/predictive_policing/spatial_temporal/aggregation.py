"""Temporal discretization, multi-resolution bucketing, and space-time cube construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ALLOWED_WINDOWS = ("24h", "3d", "1w")
TEMPORAL_FEATURE_NAMES = (
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
)


@dataclass(frozen=True)
class SpatiotemporalCube:
    """Represents an unbroken, continuous 3D space-time grid of incident counts.

    Axes:
        0: Time intervals (length T)
        1: Police divisions/outposts (length N, strictly ordered)
        2: Crime categories (length C, strictly ordered)
    """

    intervals: list[pd.Timestamp]
    divisions: list[str]
    categories: list[str]
    count_tensor: np.ndarray  # shape: [T, N, C], dtype: float32
    temporal_features: np.ndarray  # shape: [T, 7], dtype: float32
    window: str
    timezone: str

    def __post_init__(self) -> None:
        t, n, c = len(self.intervals), len(self.divisions), len(self.categories)
        if self.count_tensor.shape != (t, n, c):
            raise ValueError(
                f"count_tensor shape {self.count_tensor.shape} does not match expected ({t}, {n}, {c})"
            )
        if self.temporal_features.shape != (t, len(TEMPORAL_FEATURE_NAMES)):
            raise ValueError(
                f"temporal_features shape {self.temporal_features.shape} does not match ({t}, {len(TEMPORAL_FEATURE_NAMES)})"
            )

    @property
    def num_intervals(self) -> int:
        return len(self.intervals)

    @property
    def num_nodes(self) -> int:
        return len(self.divisions)

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def total_incidents(self) -> int:
        """Total number of recorded incidents participating in the cube."""
        return int(np.sum(self.count_tensor))

    @property
    def sparsity(self) -> float:
        """Proportion of (interval, division, category) cells with exactly zero incidents."""
        if self.count_tensor.size == 0:
            return 0.0
        return float(np.mean(self.count_tensor == 0.0))

    @property
    def total_counts_by_node(self) -> np.ndarray:
        """Total incidents per time interval and division summed across categories.

        Returns:
            Array of shape [T, N], dtype float32.
        """
        return np.sum(self.count_tensor, axis=2)

    def to_flat_dataframe(self) -> pd.DataFrame:
        """Flatten the 3D tensor into a tabular DataFrame for tabular ML baselines.

        Columns:
            interval, division_outpost, crime_category, incident_count,
            day_of_week_sin, day_of_week_cos, month_sin, month_cos,
            day_of_year_sin, day_of_year_cos, is_weekend
        """
        # Multi-index product matching tensor layout
        idx = pd.MultiIndex.from_product(
            [self.intervals, self.divisions, self.categories],
            names=["interval", "division_outpost", "crime_category"],
        )

        flat_counts = self.count_tensor.flatten()
        df = pd.DataFrame(index=idx)
        df["incident_count"] = flat_counts

        # Repeat temporal features across (N * C) for each time interval
        temp_df = pd.DataFrame(
            self.temporal_features,
            index=self.intervals,
            columns=list(TEMPORAL_FEATURE_NAMES),
        )
        temp_df.index.name = "interval"

        df = df.reset_index()
        df = df.merge(temp_df, on="interval", how="left")
        return df


def floor_to_interval(
    timestamps: pd.Series,
    window: Literal["24h", "3d", "1w"] | str = "24h",
    timezone: str = "Africa/Lagos",
    anchor_start: pd.Timestamp | None = None,
) -> pd.Series:
    """Discretize timestamps into the starting timestamp of their interval bucket.

    Alignments:
        - '24h': Calendar day 00:00:00 in local timezone.
        - '3d': 3-calendar-day blocks anchored to anchor_start (or earliest date).
        - '1w': Calendar week starting Monday 00:00:00 in local timezone.
    """
    if window not in ALLOWED_WINDOWS:
        raise ValueError(f"Invalid aggregation window '{window}'. Allowed: {ALLOWED_WINDOWS}")

    # Convert to localized pandas datetime
    ts = pd.to_datetime(timestamps, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(timezone)
    else:
        ts = ts.dt.tz_convert(timezone)

    if window == "24h":
        return ts.dt.normalize()

    elif window == "1w":
        # Anchor to Monday 00:00:00: normalized day minus day_of_week (0=Mon, ..., 6=Sun)
        normalized = ts.dt.normalize()
        days_to_monday = pd.to_timedelta(ts.dt.day_of_week, unit="D")
        return normalized - days_to_monday

    elif window == "3d":
        normalized = ts.dt.normalize()
        valid_ts = normalized.dropna()
        if valid_ts.empty:
            return normalized

        if anchor_start is None:
            anchor = valid_ts.min()
        else:
            anchor = (
                anchor_start.tz_convert(timezone).normalize()
                if anchor_start.tz is not None
                else anchor_start.tz_localize(timezone).normalize()
            )

        days_elapsed = (normalized - anchor).dt.days
        block_idx = days_elapsed // 3
        return anchor + pd.to_timedelta(block_idx * 3, unit="D")

    raise ValueError(f"Unhandled window: {window}")


def generate_continuous_interval_range(
    start_interval: pd.Timestamp,
    end_interval: pd.Timestamp,
    window: Literal["24h", "3d", "1w"] | str = "24h",
    timezone: str = "Africa/Lagos",
) -> list[pd.Timestamp]:
    """Generate an unbroken, sequential list of regular intervals from start to end."""
    if window not in ALLOWED_WINDOWS:
        raise ValueError(f"Invalid aggregation window '{window}'. Allowed: {ALLOWED_WINDOWS}")

    start = (
        start_interval.tz_convert(timezone).normalize()
        if start_interval.tz is not None
        else start_interval.tz_localize(timezone).normalize()
    )
    end = (
        end_interval.tz_convert(timezone).normalize()
        if end_interval.tz is not None
        else end_interval.tz_localize(timezone).normalize()
    )

    if start > end:
        raise ValueError(f"start_interval ({start}) cannot be after end_interval ({end})")

    if window == "24h":
        freq = "1D"
    elif window == "3d":
        freq = "3D"
    elif window == "1w":
        freq = "7D"

    date_range = pd.date_range(start=start, end=end, freq=freq, tz=timezone)
    return list(date_range)


def compute_interval_temporal_features(
    intervals: Sequence[pd.Timestamp],
) -> np.ndarray:
    """Compute normalized cyclical sine/cosine calendar features for each interval.

    Returns:
        Array of shape [T, 7] containing:
        [dow_sin, dow_cos, month_sin, month_cos, doy_sin, doy_cos, is_weekend]
    """
    ts_idx = pd.DatetimeIndex(intervals)
    dow = ts_idx.day_of_week
    month = ts_idx.month - 1
    doy = ts_idx.day_of_year - 1
    year_len = np.where(ts_idx.is_leap_year, 366, 365)
    is_weekend = (dow >= 5).astype(float)

    dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0)
    month_sin = np.sin(2.0 * np.pi * month / 12.0)
    month_cos = np.cos(2.0 * np.pi * month / 12.0)
    doy_sin = np.sin(2.0 * np.pi * doy / year_len)
    doy_cos = np.cos(2.0 * np.pi * doy / year_len)

    features = np.column_stack(
        [
            dow_sin,
            dow_cos,
            month_sin,
            month_cos,
            doy_sin,
            doy_cos,
            is_weekend,
        ]
    ).astype(np.float32)
    return features


def build_spatiotemporal_cube(
    incidents_df: pd.DataFrame,
    window: Literal["24h", "3d", "1w"] | str = "24h",
    divisions: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
    time_col: str = "model_occurred_at",
    division_col: str = "model_division_outpost",
    category_col: str = "model_crime_category",
    timezone: str = "Africa/Lagos",
    unknown_category_label: str = "unknown",
    strict_divisions: bool = False,
) -> SpatiotemporalCube:
    """Construct an unbroken, zero-filled 3D space-time cube of incident counts.

    Args:
        incidents_df: Cleaned incident dataframe from Stage 1.
        window: Temporal aggregation window ('24h', '3d', '1w').
        divisions: Canonical ordered list of divisions (from SpatialGraph.node_names).
                   If None, unique divisions from the data are extracted and sorted.
        categories: Canonical ordered list of crime categories.
                    If provided, unexpected categories in data are mapped to unknown_category_label.
        time_col: Column containing timestamp of incident occurrence.
        division_col: Column containing police division/outpost name.
        category_col: Column containing crime category label.
        timezone: Local timezone for interval boundaries.
        unknown_category_label: Label for unclassified or missing categories.
        strict_divisions: If True and divisions is provided, raises an error if incidents
                          contain unknown division names; if False, filters them out.

    Returns:
        SpatiotemporalCube with count_tensor of shape [T, N, C].
    """
    if incidents_df.empty:
        raise ValueError("Cannot build spatiotemporal cube from an empty incident DataFrame.")

    # Validate required columns
    for col in (time_col, division_col, category_col):
        if col not in incidents_df.columns:
            raise ValueError(f"Missing required column in incidents_df: '{col}'")

    working_df = incidents_df.dropna(subset=[time_col, division_col]).copy()
    if working_df.empty:
        raise ValueError("No valid records remaining after dropping null times and divisions.")

    # Standardize division and category text
    working_df[division_col] = working_df[division_col].astype(str).str.strip()
    working_df[category_col] = (
        working_df[category_col].fillna(unknown_category_label).astype(str).str.strip()
    )

    # Reconcile divisions with canonical list
    if divisions is None:
        canonical_divisions = sorted(working_df[division_col].unique())
    else:
        canonical_divisions = list(divisions)
        unrecognized_divs = set(working_df[division_col]) - set(canonical_divisions)
        if unrecognized_divs:
            if strict_divisions:
                raise ValueError(
                    f"Incidents contain unrecognized divisions: {sorted(unrecognized_divs)}"
                )
            working_df = working_df[working_df[division_col].isin(canonical_divisions)]
            if working_df.empty:
                raise ValueError(
                    "All records were filtered out because none matched the provided canonical divisions."
                )

    # Reconcile categories with canonical list
    if categories is None:
        canonical_categories = sorted(working_df[category_col].unique())
        # Ensure unknown category is present if defined
        if (
            unknown_category_label not in canonical_categories
            and unknown_category_label in working_df[category_col].values
        ):
            canonical_categories.append(unknown_category_label)
    else:
        canonical_categories = list(categories)
        # Route unlisted categories to unknown_category_label
        unrecognized_mask = ~working_df[category_col].isin(canonical_categories)
        if unrecognized_mask.any():
            if unknown_category_label not in canonical_categories:
                raise ValueError(
                    f"Incidents contain categories not in canonical list, but '{unknown_category_label}' "
                    f"is not included in canonical categories: {canonical_categories}"
                )
            working_df.loc[unrecognized_mask, category_col] = unknown_category_label

    # Compute interval bucket for each incident
    interval_series = floor_to_interval(working_df[time_col], window=window, timezone=timezone)
    working_df["_interval"] = interval_series

    # Determine full unbroken interval range
    min_interval = interval_series.min()
    max_interval = interval_series.max()
    all_intervals = generate_continuous_interval_range(
        min_interval, max_interval, window=window, timezone=timezone
    )

    # Aggregate counts by (interval, division, category)
    counts = working_df.groupby(["_interval", division_col, category_col]).size()

    # Cartesian reindexing to guarantee all [T x N x C] cells exist with 0 fill
    full_index = pd.MultiIndex.from_product(
        [all_intervals, canonical_divisions, canonical_categories],
        names=["_interval", division_col, category_col],
    )
    reindexed_counts = counts.reindex(full_index, fill_value=0)

    t = len(all_intervals)
    n = len(canonical_divisions)
    c = len(canonical_categories)

    # Reshape strictly into [T, N, C] float32 tensor
    count_tensor = reindexed_counts.to_numpy(dtype=np.float32).reshape(t, n, c)

    # Compute cyclical temporal features
    temporal_features = compute_interval_temporal_features(all_intervals)

    return SpatiotemporalCube(
        intervals=all_intervals,
        divisions=canonical_divisions,
        categories=canonical_categories,
        count_tensor=count_tensor,
        temporal_features=temporal_features,
        window=window,
        timezone=timezone,
    )
