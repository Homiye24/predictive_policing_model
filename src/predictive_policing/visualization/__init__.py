try:
    from predictive_policing.visualization.dashboard import (
        create_dashboard,
        run_dashboard,
    )
except ImportError:
    create_dashboard = None  # type: ignore[assignment]
    run_dashboard = None  # type: ignore[assignment]

from predictive_policing.visualization.spatial_temporal_plots import (
    generate_all_stage2_figures,
    plot_crime_trends,
    plot_lookback_window_diagram,
    plot_spatial_graph,
    plot_spatiotemporal_heatmap,
    plot_temporal_crime_distribution,
)

__all__ = [
    "create_dashboard",
    "generate_all_stage2_figures",
    "plot_crime_trends",
    "plot_lookback_window_diagram",
    "plot_spatial_graph",
    "plot_spatiotemporal_heatmap",
    "plot_temporal_crime_distribution",
    "run_dashboard",
]
