
import numpy as np
import pandas as pd
import pytest

from predictive_policing.spatial_temporal.aggregation import (
    build_spatiotemporal_cube,
    compute_interval_temporal_features,
    floor_to_interval,
    generate_continuous_interval_range,
)


def test_floor_to_interval_24h() -> None:
    # 23:45 WAT on Jan 1 must stay on Jan 1; 00:05 WAT on Jan 2 must land on Jan 2
    ts = pd.Series(
        [
            pd.Timestamp("2025-01-01 23:45:00", tz="Africa/Lagos"),
            pd.Timestamp("2025-01-02 00:05:00", tz="Africa/Lagos"),
        ]
    )
    floored = floor_to_interval(ts, window="24h", timezone="Africa/Lagos")

    assert floored.iloc[0] == pd.Timestamp("2025-01-01 00:00:00", tz="Africa/Lagos")
    assert floored.iloc[1] == pd.Timestamp("2025-01-02 00:00:00", tz="Africa/Lagos")


def test_floor_to_interval_weekly_anchored_to_monday() -> None:
    # Wednesday Jan 1 2025 and Sunday Jan 5 2025 must both floor to Monday Dec 30 2024
    # Monday Jan 6 2025 must floor to Monday Jan 6 2025
    ts = pd.Series(
        [
            pd.Timestamp("2025-01-01 14:00:00", tz="Africa/Lagos"),  # Wed
            pd.Timestamp("2025-01-05 22:00:00", tz="Africa/Lagos"),  # Sun
            pd.Timestamp("2025-01-06 08:00:00", tz="Africa/Lagos"),  # Mon
        ]
    )
    floored = floor_to_interval(ts, window="1w", timezone="Africa/Lagos")

    assert floored.iloc[0] == pd.Timestamp("2024-12-30 00:00:00", tz="Africa/Lagos")
    assert floored.iloc[1] == pd.Timestamp("2024-12-30 00:00:00", tz="Africa/Lagos")
    assert floored.iloc[2] == pd.Timestamp("2025-01-06 00:00:00", tz="Africa/Lagos")


def test_floor_to_interval_3d() -> None:
    anchor = pd.Timestamp("2025-01-01 00:00:00", tz="Africa/Lagos")
    ts = pd.Series(
        [
            pd.Timestamp("2025-01-01 10:00:00", tz="Africa/Lagos"),  # Day 0 -> Block 0 (Jan 1)
            pd.Timestamp("2025-01-03 23:00:00", tz="Africa/Lagos"),  # Day 2 -> Block 0 (Jan 1)
            pd.Timestamp("2025-01-04 01:00:00", tz="Africa/Lagos"),  # Day 3 -> Block 1 (Jan 4)
        ]
    )
    floored = floor_to_interval(ts, window="3d", timezone="Africa/Lagos", anchor_start=anchor)

    assert floored.iloc[0] == anchor
    assert floored.iloc[1] == anchor
    assert floored.iloc[2] == pd.Timestamp("2025-01-04 00:00:00", tz="Africa/Lagos")


def test_generate_continuous_interval_range() -> None:
    start = pd.Timestamp("2025-01-01 00:00:00", tz="Africa/Lagos")
    end = pd.Timestamp("2025-01-05 00:00:00", tz="Africa/Lagos")

    intervals_24h = generate_continuous_interval_range(start, end, window="24h")
    assert len(intervals_24h) == 5
    assert intervals_24h[0] == start
    assert intervals_24h[-1] == end

    intervals_3d = generate_continuous_interval_range(start, end, window="3d")
    assert len(intervals_3d) == 2  # Jan 1 and Jan 4


def test_compute_interval_temporal_features() -> None:
    intervals = [
        pd.Timestamp("2025-01-03 00:00:00", tz="Africa/Lagos"),  # Friday
        pd.Timestamp("2025-01-04 00:00:00", tz="Africa/Lagos"),  # Saturday
        pd.Timestamp("2025-01-05 00:00:00", tz="Africa/Lagos"),  # Sunday
    ]
    features = compute_interval_temporal_features(intervals)

    assert features.shape == (3, 7)
    # Weekend flag is the last column
    assert features[0, 6] == 0.0  # Friday is weekday
    assert features[1, 6] == 1.0  # Saturday is weekend
    assert features[2, 6] == 1.0  # Sunday is weekend
    # Sine and cosine features must stay in [-1, 1]
    assert np.all(features[:, :6] >= -1.0)
    assert np.all(features[:, :6] <= 1.0)


def test_build_spatiotemporal_cube_end_to_end() -> None:
    df = pd.DataFrame(
        [
            # Day 1: 2 thefts at Akure Central
            {
                "model_occurred_at": "2025-01-01 10:00:00",
                "model_division_outpost": "Akure Central",
                "model_crime_category": "Theft",
            },
            {
                "model_occurred_at": "2025-01-01 14:00:00",
                "model_division_outpost": "Akure Central",
                "model_crime_category": "Theft",
            },
            # Day 3: 1 robbery at Owo Division
            {
                "model_occurred_at": "2025-01-03 20:00:00",
                "model_division_outpost": "Owo Division",
                "model_crime_category": "Robbery",
            },
        ]
    )

    divisions = ["Akure Central", "Owo Division", "Quiet Outpost"]
    categories = ["Theft", "Robbery", "Burglary", "unknown"]

    cube = build_spatiotemporal_cube(
        df,
        window="24h",
        divisions=divisions,
        categories=categories,
        timezone="Africa/Lagos",
    )

    # 3 days span: Jan 1, Jan 2, Jan 3 -> T = 3
    # N = 3 divisions, C = 4 categories
    assert cube.num_intervals == 3
    assert cube.num_nodes == 3
    assert cube.num_categories == 4
    assert cube.count_tensor.shape == (3, 3, 4)

    # Total recorded incidents must equal 3
    assert cube.total_incidents == 3

    # Day 0 (Jan 1), Division 0 (Akure Central): 2 thefts, 0 others
    assert cube.count_tensor[0, 0, 0] == 2.0  # Theft = 2
    assert cube.count_tensor[0, 0, 1] == 0.0  # Robbery = 0

    # Day 2 (Jan 3), Division 1 (Owo Division): 1 robbery
    assert cube.count_tensor[2, 1, 1] == 1.0  # Robbery = 1

    # Quiet Outpost (Division 2): must have strictly 0 count across all days and categories
    assert np.all(cube.count_tensor[:, 2, :] == 0.0)

    # Total counts by node
    totals = cube.total_counts_by_node  # shape [3, 3]
    assert totals.shape == (3, 3)
    assert totals[0, 0] == 2.0  # Akure on Day 1
    assert totals[2, 1] == 1.0  # Owo on Day 3
    assert np.all(totals[:, 2] == 0.0)  # Quiet Outpost = 0

    # Sparsity: out of 3 * 3 * 4 = 36 cells, only 2 have non-zero counts
    expected_sparsity = 34.0 / 36.0
    assert pytest.approx(cube.sparsity, rel=1e-4) == expected_sparsity


def test_build_spatiotemporal_cube_unlisted_category_mapped_to_unknown() -> None:
    df = pd.DataFrame(
        [
            {
                "model_occurred_at": "2025-01-01 10:00:00",
                "model_division_outpost": "Akure Central",
                "model_crime_category": "Rare Offence",  # Not in categories list
            }
        ]
    )
    categories = ["Theft", "unknown"]
    cube = build_spatiotemporal_cube(df, window="24h", categories=categories)

    # 'Rare Offence' must be mapped into 'unknown' category (index 1)
    assert cube.categories == ["Theft", "unknown"]
    assert cube.count_tensor[0, 0, 1] == 1.0
    assert cube.count_tensor[0, 0, 0] == 0.0


def test_spatiotemporal_cube_to_flat_dataframe() -> None:
    df = pd.DataFrame(
        [
            {
                "model_occurred_at": "2025-01-01 10:00:00",
                "model_division_outpost": "Akure Central",
                "model_crime_category": "Theft",
            }
        ]
    )
    cube = build_spatiotemporal_cube(df, window="24h")
    flat_df = cube.to_flat_dataframe()

    assert "incident_count" in flat_df.columns
    assert "day_of_week_sin" in flat_df.columns
    assert "is_weekend" in flat_df.columns
    assert len(flat_df) == cube.num_intervals * cube.num_nodes * cube.num_categories
    assert flat_df["incident_count"].sum() == 1.0


def test_build_cube_invalid_window_raises() -> None:
    df = pd.DataFrame(
        [
            {
                "model_occurred_at": "2025-01-01",
                "model_division_outpost": "A",
                "model_crime_category": "B",
            }
        ]
    )
    with pytest.raises(ValueError, match="Invalid aggregation window"):
        build_spatiotemporal_cube(df, window="6h")
