"""Interactive Web Dashboard for Zone 17 Predictive Policing.

Built with NiceGUI and Plotly for command-level decision support:
- Interactive Geographic Network Map (Police stations & Gaussian highway links)
- Interactive Spatiotemporal Cube Heatmap (Hotspots & natural zero-crime sparsity)
- Multi-Category Temporal Crime Timeline
- Lookback Window Mechanism Simulator (L=12 -> H=1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

from predictive_policing.spatial_temporal.aggregation import (
    SpatiotemporalCube,
    compute_interval_temporal_features,
)
from predictive_policing.spatial_temporal.graph import SpatialGraph
from predictive_policing.spatial_temporal.io import (
    load_spatial_graph,
    load_spatiotemporal_cube,
)


def _create_fallback_demo_data() -> tuple[SpatialGraph, SpatiotemporalCube]:
    """Create a realistic Zone 17 demo graph and cube if processed files are not on disk."""
    divisions = [
        "Okuta Elerinla",
        "Akure North",
        "Owo",
        "Ado Central",
        "Ikole",
        "Idanre",
        "Ondo Town",
        "Ikere",
    ]
    categories = ["Theft", "Burglary", "Assault", "Armed Robbery", "Homicide"]

    coords = np.array(
        [
            [7.25, 5.19],  # Okuta Elerinla
            [7.32, 5.25],  # Akure North
            [7.19, 5.58],  # Owo
            [7.62, 5.22],  # Ado Central
            [7.79, 5.51],  # Ikole
            [7.11, 5.11],  # Idanre
            [7.09, 4.84],  # Ondo Town
            [7.50, 5.23],  # Ikere
        ],
        dtype=np.float32,
    )

    n = len(divisions)
    # Pairwise distances
    from predictive_policing.spatial_temporal.graph import (
        build_gaussian_adjacency,
        compute_pairwise_distances,
        dense_to_edge_list,
        normalize_adjacency_symmetric,
    )

    dist = compute_pairwise_distances(coords)
    raw_adj = build_gaussian_adjacency(dist, bandwidth_sigma=25.0, distance_cutoff=50.0)
    norm_adj = normalize_adjacency_symmetric(raw_adj)
    edge_index, edge_weights = dense_to_edge_list(raw_adj)

    graph = SpatialGraph(
        node_names=divisions,
        centroids=coords,
        distance_matrix=dist,
        adjacency_matrix=raw_adj,
        normalized_adjacency=norm_adj,
        edge_index=edge_index,
        edge_weights=edge_weights,
        bandwidth_sigma=25.0,
        distance_cutoff=50.0,
    )

    t = 90  # 90 days
    intervals = pd.date_range("2024-01-01", periods=t, freq="D", tz="Africa/Lagos").tolist()
    c = len(categories)

    # Realistic Poisson-like crime count tensor with natural 85-90% sparsity
    np.random.seed(42)
    count_tensor = np.zeros((t, n, c), dtype=np.float32)
    # Inject scattered incidents
    for _ in range(350):
        t_i = np.random.randint(0, t)
        n_i = np.random.randint(0, n)
        c_i = np.random.randint(0, c)
        count_tensor[t_i, n_i, c_i] += 1.0

    temp_features = compute_interval_temporal_features(intervals)

    cube = SpatiotemporalCube(
        intervals=intervals,
        divisions=divisions,
        categories=categories,
        count_tensor=count_tensor,
        temporal_features=temp_features,
        window="24h",
        timezone="Africa/Lagos",
    )

    return graph, cube


def build_plotly_spatial_map(
    graph: SpatialGraph,
    map_style: str = "osm",
) -> go.Figure:
    """Build an interactive geographic network map of Police Zone 17 stations.

    Args:
        graph: SpatialGraph instance.
        map_style: "osm" for real OpenStreetMap tiles, or "schematic" for Cartesian coordinate grid.
    """
    lats = graph.centroids[:, 0]
    lons = graph.centroids[:, 1]
    names = graph.node_names

    edge_index = graph.edge_index
    edge_weights = graph.edge_weights

    hover_texts = [
        f"<b>{name} Division</b><br>Lat: {lats[i]:.4f}°N<br>Lon: {lons[i]:.4f}°E"
        for i, name in enumerate(names)
    ]

    fig = go.Figure()

    if map_style == "osm" and hasattr(go, "Scattermap"):
        # Real OpenStreetMap MapLibre rendering
        if edge_index.shape[1] > 0 and edge_weights.size > 0:
            edge_lats: list[float | None] = []
            edge_lons: list[float | None] = []
            for idx in range(edge_index.shape[1]):
                src = int(edge_index[0, idx])
                dst = int(edge_index[1, idx])
                if src < dst:
                    edge_lats.extend([float(lats[src]), float(lats[dst]), None])
                    edge_lons.extend([float(lons[src]), float(lons[dst]), None])

            fig.add_trace(
                go.Scattermap(
                    lat=edge_lats,
                    lon=edge_lons,
                    mode="lines",
                    line={"width": 3, "color": "rgba(37, 99, 235, 0.7)"},
                    hoverinfo="none",
                    name="Spatial Highway Links",
                )
            )

        fig.add_trace(
            go.Scattermap(
                lat=lats,
                lon=lons,
                mode="markers+text",
                text=names,
                textposition="top right",
                hovertext=hover_texts,
                hoverinfo="text",
                marker={
                    "size": 16,
                    "color": "#dc2626",
                },
                name="Police Stations",
            )
        )

        center_lat = float(np.mean(lats)) if len(lats) > 0 else 7.3
        center_lon = float(np.mean(lons)) if len(lons) > 0 else 5.3

        fig.update_layout(
            title="<b>Zone 17 Police Division Spatial Network (OpenStreetMap Overlay)</b>",
            map_style="open-street-map",
            map_zoom=8.0,
            map_center={"lat": center_lat, "lon": center_lon},
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            height=580,
        )
        return fig

    # Fallback / Schematic Cartesian Coordinate View
    if edge_index.shape[1] > 0 and edge_weights.size > 0:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []

        for idx in range(edge_index.shape[1]):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            if src < dst:
                edge_x.extend([float(lons[src]), float(lons[dst]), None])
                edge_y.extend([float(lats[src]), float(lats[dst]), None])

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 2.5, "color": "rgba(43, 108, 176, 0.6)"},
                hoverinfo="none",
                name="Spatial Connectivity (Gaussian)",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=lons,
            y=lats,
            mode="markers+text",
            text=names,
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            marker={
                "size": 18,
                "color": "#e53e3e",
                "line": {"width": 2, "color": "#ffffff"},
                "symbol": "circle",
            },
            name="Police Divisions",
        )
    )

    fig.update_layout(
        title="<b>Zone 17 Police Division Spatial Network (Schematic Grid)</b>",
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        hovermode="closest",
        plot_bgcolor="#f8fafc",
        height=580,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e2e8f0")

    return fig


def build_plotly_cube_heatmap(
    cube: SpatiotemporalCube, category: str = "All Categories", colorscale: str = "YlOrRd"
) -> go.Figure:
    """Build an interactive heatmap for the Space-Time Cube."""
    if category == "All Categories":
        matrix = cube.total_counts_by_node.T  # [N, T]
    else:
        c_idx = cube.categories.index(category) if category in cube.categories else 0
        matrix = cube.count_tensor[:, :, c_idx].T  # [N, T]

    date_strs = [ts.strftime("%Y-%m-%d") for ts in cube.intervals]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=date_strs,
            y=cube.divisions,
            colorscale=colorscale,
            colorbar={"title": "Incidents"},
            hovertemplate="<b>%{y}</b><br>Date: %{x}<br>Count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Spatiotemporal Crime Cube (Space × Time Density)</b><br>Category: {category} | Sparsity: {cube.sparsity:.1%}",
        xaxis_title="Date",
        yaxis_title="Police Division",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        height=500,
    )

    return fig


def build_plotly_crime_trends(cube: SpatiotemporalCube) -> go.Figure:
    """Build multi-category time-series line chart."""
    cat_counts = np.sum(cube.count_tensor, axis=1)  # [T, C]
    date_strs = [ts.strftime("%Y-%m-%d") for ts in cube.intervals]

    fig = go.Figure()

    for c_idx, cat_name in enumerate(cube.categories):
        fig.add_trace(
            go.Scatter(
                x=date_strs,
                y=cat_counts[:, c_idx],
                mode="lines",
                name=cat_name,
                line={"width": 2},
            )
        )

    fig.update_layout(
        title="<b>Zone 17 Crime Category Timeline (Total Counts Across Divisions)</b>",
        xaxis_title="Date",
        yaxis_title="Incident Count",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        hovermode="x unified",
        plot_bgcolor="#f8fafc",
        height=450,
    )

    return fig


def create_dashboard(
    graph: SpatialGraph | None = None,
    cube: SpatiotemporalCube | None = None,
    data_dir: Path | str = "data/processed",
):
    """Render the full NiceGUI dashboard application."""
    proc_dir = Path(data_dir)

    # Attempt to load from processed disk files; fall back gracefully to demo data if absent
    if graph is None:
        try:
            graph = load_spatial_graph(proc_dir)
        except (FileNotFoundError, KeyError, ValueError):
            graph = None

    if cube is None:
        try:
            cube = load_spatiotemporal_cube(proc_dir, prefix="spatiotemporal_cube_24h")
        except (FileNotFoundError, KeyError, ValueError):
            cube = None

    if graph is None or cube is None:
        demo_graph, demo_cube = _create_fallback_demo_data()
        graph = graph or demo_graph
        cube = cube or demo_cube
        using_demo = True
    else:
        using_demo = False

    ui.colors(
        primary="#1a365d",
        secondary="#2b6cb0",
        accent="#d69e2e",
        positive="#38a169",
        negative="#e53e3e",
        info="#3182ce",
    )

    # Header Bar
    with ui.header().classes(
        "bg-blue-900 text-white flex justify-between items-center px-6 py-3 shadow-md"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("shield", size="32px").classes("text-yellow-400")
            with ui.column().classes("gap-0"):
                ui.label("Police Zone 17 Predictive Policing").classes("text-xl font-bold")
                ui.label("Ondo & Ekiti State Commands | Spatial-Temporal Decision Support").classes(
                    "text-xs text-blue-200"
                )
        with ui.row().classes("items-center gap-2"):
            if using_demo:
                ui.badge("DEMO PREVIEW", color="amber").classes("text-black font-bold")
            else:
                ui.badge("REAL PROCESSED DATA", color="green").classes("font-bold")

    # Metrics Summary Row
    with ui.row().classes("w-full px-6 py-4 gap-4"):
        with ui.card().classes("flex-1 bg-slate-50 border-l-4 border-blue-600 shadow-sm p-4"):
            ui.label("Total Incidents").classes("text-xs uppercase text-slate-500 font-semibold")
            ui.label(f"{cube.total_incidents:,}").classes("text-2xl font-bold text-slate-800")
        with ui.card().classes("flex-1 bg-slate-50 border-l-4 border-indigo-600 shadow-sm p-4"):
            ui.label("Active Divisions (Nodes)").classes(
                "text-xs uppercase text-slate-500 font-semibold"
            )
            ui.label(f"{graph.num_nodes}").classes("text-2xl font-bold text-slate-800")
        with ui.card().classes("flex-1 bg-slate-50 border-l-4 border-green-600 shadow-sm p-4"):
            ui.label("Aggregation Window").classes("text-xs uppercase text-slate-500 font-semibold")
            ui.label(f"{cube.window.upper()} (Tactical Daily)").classes(
                "text-2xl font-bold text-slate-800"
            )
        with ui.card().classes("flex-1 bg-slate-50 border-l-4 border-amber-600 shadow-sm p-4"):
            ui.label("Natural Sparsity").classes("text-xs uppercase text-slate-500 font-semibold")
            ui.label(f"{cube.sparsity:.1%}").classes("text-2xl font-bold text-slate-800")

    # Main Content Tabs
    with ui.tabs().classes("w-full px-6 text-blue-900 font-semibold") as tabs:
        tab_map = ui.tab("1. Spatial Graph & Map (WHERE)")
        tab_cube = ui.tab("2. Space-Time Cube Heatmap (WHEN & WHERE)")
        tab_trends = ui.tab("3. Category Trends (WHAT)")
        tab_lookback = ui.tab("4. Lookback Window Simulator (AI MECHANISM)")

    with ui.tab_panels(tabs, value=tab_map).classes("w-full px-6 py-2"):
        # TAB 1: Map
        with ui.tab_panel(tab_map):
            ui.markdown("### Interactive Spatial Division Network (Gaussian Edge Weights)")
            ui.markdown(
                "Nearby divisions are connected by highway links with distance decay "
                "(**σ = 25 km**, **cutoff = 50 km**). Equations 3.8 & 3.9."
            )
            with ui.row().classes("items-center gap-4 mb-3"):
                map_toggle = ui.toggle(
                    {"osm": "OpenStreetMap Real Tiles", "schematic": "Cartesian Coordinate Grid"},
                    value="osm",
                ).classes("text-sm shadow-sm")

            map_container = ui.column().classes("w-full")

            def refresh_map():
                map_container.clear()
                with map_container:
                    m_fig = build_plotly_spatial_map(graph, map_style=map_toggle.value)
                    ui.plotly(m_fig).classes("w-full")

            map_toggle.on("update:model-value", lambda _: refresh_map())
            refresh_map()

        # TAB 2: Cube Heatmap
        with ui.tab_panel(tab_cube):
            ui.markdown("### Spatiotemporal Crime Cube (Division × Time Interval Density)")
            ui.markdown("Notice the hotspots and the natural zero-crime days (zero-filling).")

            with ui.row().classes("items-center gap-4 mb-2"):
                cat_options = ["All Categories"] + list(cube.categories)
                cat_select = ui.select(
                    cat_options, value="All Categories", label="Crime Category"
                ).classes("w-64")
                color_select = ui.select(
                    ["YlOrRd", "Viridis", "Hot", "Blues"], value="YlOrRd", label="Color Scale"
                ).classes("w-44")

            heatmap_container = ui.column().classes("w-full")

            def refresh_heatmap():
                heatmap_container.clear()
                with heatmap_container:
                    h_fig = build_plotly_cube_heatmap(
                        cube, category=cat_select.value, colorscale=color_select.value
                    )
                    ui.plotly(h_fig).classes("w-full")

            cat_select.on("update:model-value", lambda _: refresh_heatmap())
            color_select.on("update:model-value", lambda _: refresh_heatmap())
            refresh_heatmap()

        # TAB 3: Category Trends
        with ui.tab_panel(tab_trends):
            ui.markdown("### Crime Category Timeline Breakdown Across Zone 17")
            trends_fig = build_plotly_crime_trends(cube)
            ui.plotly(trends_fig).classes("w-full")

        # TAB 4: Lookback Window Simulator
        with ui.tab_panel(tab_lookback):
            ui.markdown("### 12-Interval Lookback Conditioning Simulator")
            ui.markdown(
                "In Chapter Three, the hybrid neural model inspects the past **12 intervals** "
                "to predict the **next 1 interval ahead**. Zero future target leakage."
            )

            t_total = len(cube.intervals)
            slider_label = ui.label("Target Day Index: 12").classes("font-bold text-slate-700")

            def on_slider_change(e):
                val = int(e.value)
                slider_label.text = (
                    f"Selected Target Interval (t): Day {val} "
                    f"({cube.intervals[val].strftime('%Y-%m-%d')}) | "
                    f"Conditioned on Past 12 Days (Day {val - 12} to Day {val - 1})"
                )

            ui.slider(
                min=12, max=t_total - 1, value=12, step=1, on_change=on_slider_change
            ).classes("w-full max-w-xl")

            with ui.card().classes("p-4 bg-slate-50 border border-slate-200 mt-2"):
                ui.markdown(
                    "- **Input History (X)**: Shape `[12 × N × D]` (Counts + Cyclical Calendar Features)\n"
                    "- **Target Forecast (Y)**: Shape `[N × C]` (Predicted Incidents for Day `t+1`)\n"
                    "- **Chronological Split**: 70% Train, 15% Validation, 15% Test strictly by target date."
                )


def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8080,
    title: str = "Zone 17 Predictive Policing Dashboard",
    data_dir: Path | str = "data/processed",
):
    """Run the NiceGUI event loop with create_dashboard bound as the root page."""

    def root_page():
        create_dashboard(data_dir=data_dir)

    ui.run(root=root_page, host=host, port=port, title=title, reload=False, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    run_dashboard()
