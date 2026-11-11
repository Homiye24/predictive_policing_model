"""Visualization engine for spatial-temporal crime forecasting representations.

Generates publication-quality figures:
1. Spatial Graph Network Map (division centroids + Gaussian edge connectivity)
2. Spatiotemporal Cube Heatmap (Division x Time interval crime density & sparsity)
3. Temporal Category Distribution (category volume over time)
4. Sliding Lookback Window Mechanism diagram
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # Headless backend: safe for servers and CLI without display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from predictive_policing.spatial_temporal.aggregation import SpatiotemporalCube
    from predictive_policing.spatial_temporal.graph import SpatialGraph


def plot_spatial_graph(
    graph: SpatialGraph,
    output_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the spatial graph network across police divisions.

    Visualizes division centroids on a geographic coordinate plane (lat/lon)
    with edges weighted by the Gaussian distance-decay kernel.

    Args:
        graph: SpatialGraph object.
        output_path: Optional path to save the generated PNG/PDF figure.
        dpi: Resolution for saved raster images.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    lats = graph.centroids[:, 0]
    lons = graph.centroids[:, 1]

    # Draw Gaussian adjacency edges
    edge_index = graph.edge_index
    edge_weights = graph.edge_weights

    if edge_index.shape[1] > 0 and edge_weights.size > 0:
        max_weight = float(np.max(edge_weights)) if np.max(edge_weights) > 0 else 1.0

        for idx in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, idx])
            dst_idx = int(edge_index[1, idx])
            weight = float(edge_weights[idx])

            # Draw only one direction for undirected pairs to avoid overplotting
            if src_idx < dst_idx:
                norm_w = weight / max_weight
                ax.plot(
                    [lons[src_idx], lons[dst_idx]],
                    [lats[src_idx], lats[dst_idx]],
                    color="#2b5c8f",
                    alpha=max(0.15, norm_w * 0.75),
                    linewidth=max(0.8, norm_w * 3.5),
                    zorder=1,
                )

    # Plot station nodes
    ax.scatter(
        lons,
        lats,
        s=120,
        c="#d93829",
        edgecolors="#ffffff",
        linewidth=1.8,
        zorder=3,
        label=f"Police Divisions (N={graph.num_nodes})",
    )

    # Annotate division names
    for i, name in enumerate(graph.node_names):
        ax.annotate(
            name,
            (lons[i], lats[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="semibold",
            color="#1a202c",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": "#cbd5e0", "alpha": 0.85, "lw": 0.6},
            zorder=4,
        )

    ax.set_title(
        f"Zone 17 Spatial Division Graph\n(Gaussian Connectivity: $\\sigma={graph.bandwidth_sigma:.0f}$ km, Cutoff={graph.distance_cutoff:.0f}$ km)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="medium", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()

    if output_path is not None:
        save_file = Path(output_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_spatiotemporal_heatmap(
    cube: SpatiotemporalCube,
    output_path: Path | str | None = None,
    max_divisions_display: int = 25,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the Space x Time crime density heatmap from a SpatiotemporalCube.

    Renders a 2D matrix:
    - Vertical axis: Police Divisions
    - Horizontal axis: Time intervals
    - Values: Total incidents summed across categories

    Directly addresses the reviewer comment for visualizing the space-time cube.

    Args:
        cube: SpatiotemporalCube object.
        output_path: Optional path to save the generated figure.
        max_divisions_display: Limit number of divisions shown to prevent overcrowding.
        dpi: Resolution for saved images.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    total_counts = cube.total_counts_by_node.T  # Shape: [N, T]

    # Select top divisions by volume if total exceeds display limit
    n_total = len(cube.divisions)
    if n_total > max_divisions_display:
        div_sums = np.sum(total_counts, axis=1)
        top_indices = np.argsort(div_sums)[::-1][:max_divisions_display]
        heatmap_data = total_counts[top_indices]
        display_divisions = [cube.divisions[i] for i in top_indices]
    else:
        heatmap_data = total_counts
        display_divisions = cube.divisions

    fig, ax = plt.subplots(figsize=(14, max(6, len(display_divisions) * 0.35)), dpi=dpi)

    # Subsample x-axis ticks if there are many intervals
    t_len = len(cube.intervals)
    step = max(1, t_len // 12)
    xtick_positions = list(range(0, t_len, step))
    xtick_labels = [cube.intervals[i].strftime("%b %d, %Y") for i in xtick_positions]

    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        cbar_kws={"label": "Incident Count per Interval", "shrink": 0.8},
        linewidths=0.05,
        linecolor="#f0f0f0",
    )

    ax.set_yticks(np.arange(len(display_divisions)) + 0.5)
    ax.set_yticklabels(display_divisions, rotation=0, fontsize=9)
    ax.set_xticks(np.array(xtick_positions) + 0.5)
    ax.set_xticklabels(xtick_labels, rotation=35, ha="right", fontsize=8.5)

    ax.set_title(
        f"Zone 17 Spatiotemporal Crime Cube (Space × Time Density)\n"
        f"Resolution: {cube.window} | Sparsity: {cube.sparsity:.1%} zero-cells | Total Incidents: {cube.total_incidents:,}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Time Interval", fontsize=11, fontweight="medium", labelpad=8)
    ax.set_ylabel("Police Division", fontsize=11, fontweight="medium", labelpad=8)

    plt.tight_layout()

    if output_path is not None:
        save_file = Path(output_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")

    return fig


def plot_temporal_crime_distribution(
    cube: SpatiotemporalCube,
    output_path: Path | str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the aggregate crime category timeline and volume breakdown.

    Args:
        cube: SpatiotemporalCube object.
        output_path: Optional path to save the generated figure.
        dpi: Resolution for saved images.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    # Sum counts across divisions: [T, C]
    category_timeline = np.sum(cube.count_tensor, axis=1)  # Shape: [T, C]
    dates = [ts.to_pydatetime() for ts in cube.intervals]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    colors = sns.color_palette("tab10", n_colors=len(cube.categories))

    for c_idx, cat_name in enumerate(cube.categories):
        ax.plot(
            dates,
            category_timeline[:, c_idx],
            label=cat_name,
            color=colors[c_idx],
            linewidth=1.8,
            alpha=0.85,
        )

    ax.set_title(
        f"Zone 17 Crime Category Timeline (Aggregated across {len(cube.divisions)} Divisions)\n"
        f"Window: {cube.window} | Total Tracked Crimes: {cube.total_incidents:,}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Date", fontsize=10, fontweight="medium", labelpad=6)
    ax.set_ylabel("Incident Count", fontsize=10, fontweight="medium", labelpad=6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout()

    if output_path is not None:
        save_file = Path(output_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")

    return fig


plot_crime_trends = plot_temporal_crime_distribution


def plot_lookback_window_diagram(
    output_path: Path | str | None = None,
    lookback_intervals: int = 12,
    forecast_horizon: int = 1,
    dpi: int = 300,
) -> plt.Figure:
    """Render a visual schematic diagram of the sliding lookback window mechanism.

    Illustrates:
    [t-11, t-10, ..., t] -> Predicts -> [t+1]

    Args:
        output_path: Optional path to save figure.
        lookback_intervals: Number of history intervals (L=12).
        forecast_horizon: Number of future intervals (H=1).
        dpi: Figure resolution.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=dpi)

    total_blocks = lookback_intervals + forecast_horizon
    block_width = 0.8
    block_height = 0.6

    for i in range(lookback_intervals):
        # Lookback historical blocks
        rect = plt.Rectangle(
            (i, 0.2),
            block_width,
            block_height,
            facecolor="#2b6cb0",
            edgecolor="#1a365d",
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(rect)
        label = f"t-{lookback_intervals - 1 - i}" if (lookback_intervals - 1 - i) > 0 else "t"
        ax.text(
            i + block_width / 2,
            0.5,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            zorder=3,
        )

    # Forecast target blocks
    for j in range(forecast_horizon):
        target_x = lookback_intervals + 0.5 + j
        rect = plt.Rectangle(
            (target_x, 0.2),
            block_width,
            block_height,
            facecolor="#c53030",
            edgecolor="#742a2a",
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(rect)
        t_label = f"t+{j + 1}"
        ax.text(
            target_x + block_width / 2,
            0.5,
            t_label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    # Arrow connecting lookback to target
    ax.annotate(
        "",
        xy=(lookback_intervals + 0.45, 0.5),
        xytext=(lookback_intervals - 0.15, 0.5),
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#2d3748"},
        zorder=4,
    )
    ax.text(
        lookback_intervals + 0.15,
        0.85,
        "Predicts",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#2d3748",
    )

    # Descriptive sub-labels
    ax.text(
        lookback_intervals / 2 - 0.1,
        -0.05,
        f"Input History Window: X ∈ ℝ^(12 × N × D)\n({lookback_intervals} intervals of historical crime counts + calendar context)",
        ha="center",
        va="top",
        fontsize=9,
        color="#1a365d",
        fontweight="semibold",
    )
    ax.text(
        lookback_intervals + 0.5 + (forecast_horizon * block_width) / 2,
        -0.05,
        "Target Horizon: Y ∈ ℝ^(N × C)\n(Forecast Next Interval)",
        ha="center",
        va="top",
        fontsize=9,
        color="#742a2a",
        fontweight="semibold",
    )

    ax.set_xlim(-0.5, total_blocks + 1.2)
    ax.set_ylim(-0.4, 1.1)
    ax.axis("off")

    ax.set_title(
        f"Sliding Look-Back Window Mechanism (L={lookback_intervals} → H={forecast_horizon})\n"
        "Strictly Chronological Conditioning with Zero Target Leakage",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()

    if output_path is not None:
        save_file = Path(output_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight")

    return fig


def generate_all_stage2_figures(
    graph: SpatialGraph,
    cube: SpatiotemporalCube,
    output_dir: Path | str = "outputs/figures",
    dpi: int = 300,
) -> dict[str, Path]:
    """Generate and save all 4 Stage 2 figures to the specified output directory.

    Args:
        graph: SpatialGraph object.
        cube: SpatiotemporalCube object.
        output_dir: Output directory path.
        dpi: Resolution for saved PNGs.

    Returns:
        Dictionary mapping figure keys to their saved absolute paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig1_path = out / "spatial_graph_network.png"
    fig2_path = out / "spatiotemporal_cube_heatmap.png"
    fig3_path = out / "temporal_crime_distribution.png"
    fig4_path = out / "lookback_window_diagram.png"

    f1 = plot_spatial_graph(graph, output_path=fig1_path, dpi=dpi)
    plt.close(f1)

    f2 = plot_spatiotemporal_heatmap(cube, output_path=fig2_path, dpi=dpi)
    plt.close(f2)

    f3 = plot_temporal_crime_distribution(cube, output_path=fig3_path, dpi=dpi)
    plt.close(f3)

    f4 = plot_lookback_window_diagram(output_path=fig4_path, dpi=dpi)
    plt.close(f4)

    return {
        "spatial_graph": fig1_path,
        "cube_heatmap": fig2_path,
        "crime_timeline": fig3_path,
        "lookback_diagram": fig4_path,
    }
