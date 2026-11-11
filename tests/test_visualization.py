import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HAS_DASHBOARD_DEPS = (
    importlib.util.find_spec("plotly") is not None
    and importlib.util.find_spec("nicegui") is not None
)

from predictive_policing.spatial_temporal.aggregation import (
    SpatiotemporalCube,
    compute_interval_temporal_features,
)
from predictive_policing.spatial_temporal.graph import SpatialGraph
from predictive_policing.visualization import (
    generate_all_stage2_figures,
    plot_crime_trends,
    plot_lookback_window_diagram,
    plot_spatial_graph,
    plot_spatiotemporal_heatmap,
    plot_temporal_crime_distribution,
)


def _make_dummy_graph() -> SpatialGraph:
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


def _make_dummy_cube() -> SpatiotemporalCube:
    intervals = pd.date_range("2024-01-01", periods=15, freq="D", tz="Africa/Lagos").tolist()
    divisions = ["Akure", "Owo", "Ikole"]
    categories = ["Theft", "Burglary"]
    count_tensor = np.ones((15, 3, 2), dtype=np.float32) * 2.0
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


def test_plot_spatial_graph(tmp_path: Path):
    graph = _make_dummy_graph()
    out_file = tmp_path / "spatial_graph.png"
    fig = plot_spatial_graph(graph, output_path=out_file, dpi=100)

    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 5000  # Non-trivial image file


def test_plot_spatiotemporal_heatmap(tmp_path: Path):
    cube = _make_dummy_cube()
    out_file = tmp_path / "heatmap.png"
    fig = plot_spatiotemporal_heatmap(cube, output_path=out_file, dpi=100)

    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 5000


def test_plot_temporal_crime_distribution(tmp_path: Path):
    cube = _make_dummy_cube()
    out_file = tmp_path / "timeline.png"
    fig = plot_temporal_crime_distribution(cube, output_path=out_file, dpi=100)

    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 5000

    # Also test the alias
    out_alias = tmp_path / "alias.png"
    fig_alias = plot_crime_trends(cube, output_path=out_alias, dpi=100)
    assert fig_alias is not None
    assert out_alias.exists()


def test_plot_lookback_window_diagram(tmp_path: Path):
    out_file = tmp_path / "diagram.png"
    fig = plot_lookback_window_diagram(
        output_path=out_file, lookback_intervals=12, forecast_horizon=1, dpi=100
    )

    assert fig is not None
    assert out_file.exists()
    assert out_file.stat().st_size > 5000


def test_generate_all_stage2_figures(tmp_path: Path):
    graph = _make_dummy_graph()
    cube = _make_dummy_cube()
    out_dir = tmp_path / "all_figures"

    result = generate_all_stage2_figures(graph, cube, output_dir=out_dir, dpi=100)

    assert len(result) == 4
    for path in result.values():
        assert path.exists()
        assert path.stat().st_size > 5000


@pytest.mark.skipif(
    not HAS_DASHBOARD_DEPS,
    reason="plotly and nicegui are required for dashboard tests",
)
def test_dashboard_plotly_builders():
    from predictive_policing.visualization.dashboard import (
        _create_fallback_demo_data,
        build_plotly_crime_trends,
        build_plotly_cube_heatmap,
        build_plotly_spatial_map,
    )

    graph, cube = _create_fallback_demo_data()
    assert graph.num_nodes == 8
    assert cube.num_intervals == 90
    assert cube.num_categories == 5

    map_fig_osm = build_plotly_spatial_map(graph, map_style="osm")
    assert map_fig_osm is not None
    assert len(map_fig_osm.data) >= 1

    map_fig_schematic = build_plotly_spatial_map(graph, map_style="schematic")
    assert map_fig_schematic is not None
    assert len(map_fig_schematic.data) >= 1

    heatmap_fig = build_plotly_cube_heatmap(cube)
    assert heatmap_fig is not None
    assert len(heatmap_fig.data) == 1

    cat_heatmap_fig = build_plotly_cube_heatmap(cube, category="Theft")
    assert cat_heatmap_fig is not None

    trends_fig = build_plotly_crime_trends(cube)
    assert trends_fig is not None
    assert len(trends_fig.data) == 5
