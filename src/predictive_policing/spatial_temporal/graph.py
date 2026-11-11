"""Spatial graph construction, distance metrics, and adjacency normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class BoundingBox:
    """Geographical bounding box in decimal degrees."""

    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float

    def __post_init__(self) -> None:
        if self.min_latitude >= self.max_latitude:
            raise ValueError("min_latitude must be strictly less than max_latitude.")
        if self.min_longitude >= self.max_longitude:
            raise ValueError("min_longitude must be strictly less than max_longitude.")

    def contains(
        self, lat: float | np.ndarray | pd.Series, lon: float | np.ndarray | pd.Series
    ) -> bool | np.ndarray | pd.Series:
        """Check if coordinates fall inside the bounding box."""
        return (
            (lat >= self.min_latitude)
            & (lat <= self.max_latitude)
            & (lon >= self.min_longitude)
            & (lon <= self.max_longitude)
        )


# Default regional bounding box for Zone 17 (Ondo & Ekiti States, Nigeria)
ZONE_17_BOUNDING_BOX = BoundingBox(
    min_latitude=5.5,
    max_latitude=8.5,
    min_longitude=4.0,
    max_longitude=6.5,
)


@dataclass(frozen=True)
class SpatialGraph:
    """Represents a spatial graph of police divisions/outposts."""

    node_names: list[str]
    centroids: np.ndarray  # shape: [N, 2] -> (lat, lon)
    distance_matrix: np.ndarray  # shape: [N, N] in km
    adjacency_matrix: np.ndarray  # shape: [N, N] weighted with cutoff [0, 1]
    normalized_adjacency: np.ndarray  # shape: [N, N] symmetrically normalized
    edge_index: np.ndarray  # shape: [2, E] integer indices
    edge_weights: np.ndarray  # shape: [E] float weights
    bandwidth_sigma: float
    distance_cutoff: float

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def isolated_nodes(self) -> list[str]:
        """List of divisions with zero connections to other stations in raw adjacency."""
        degrees = np.sum(self.adjacency_matrix > 0, axis=1)
        return [self.node_names[i] for i, deg in enumerate(degrees) if deg == 0]

    @property
    def graph_density(self) -> float:
        """Density of the graph: actual directed edges / maximum possible N*(N-1)."""
        n = self.num_nodes
        if n <= 1:
            return 0.0
        return float(self.num_edges / (n * (n - 1)))


def haversine_distance(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the great-circle distance between coordinates in kilometers.

    Supports both scalar floats and NumPy arrays for vectorized computation.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    # Clip to avoid floating-point errors outside [0, 1]
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def compute_division_centroids(
    incidents_df: pd.DataFrame,
    division_col: str = "division_outpost",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bounding_box: BoundingBox | None = None,
    known_centroids: Mapping[str, tuple[float, float]] | None = None,
    all_divisions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute the geographic centroid (median coordinates) for each division.

    Args:
        incidents_df: Incident dataframe with division and coordinate columns.
        division_col: Name of division column.
        lat_col: Name of latitude column.
        lon_col: Name of longitude column.
        bounding_box: Optional bounding box to reject regional outlier coordinates.
        known_centroids: Optional dictionary of known division coordinates {division: (lat, lon)}
                         used as fallback when incident records lack valid GPS.
        all_divisions: Optional list of all expected divisions. If not supplied, defaults
                       to unique non-null values in incidents_df[division_col].

    Returns:
        DataFrame with columns: [division_col, lat_col, lon_col, 'provenance']
    """
    if division_col not in incidents_df.columns:
        raise ValueError(f"Missing division column: '{division_col}'")

    if all_divisions is None:
        raw_divs = incidents_df[division_col].dropna().unique()
        all_divs = sorted(str(d).strip() for d in raw_divs if str(d).strip())
    else:
        all_divs = sorted(str(d).strip() for d in all_divisions if str(d).strip())

    if not all_divs:
        raise ValueError("No divisions found to compute centroids.")

    # Clean and parse coordinates
    valid_coords = incidents_df.copy()
    has_coords = lat_col in valid_coords.columns and lon_col in valid_coords.columns
    if has_coords:
        valid_coords[lat_col] = pd.to_numeric(valid_coords[lat_col], errors="coerce")
        valid_coords[lon_col] = pd.to_numeric(valid_coords[lon_col], errors="coerce")
        valid_coords = valid_coords.dropna(subset=[lat_col, lon_col])

        # Filter out coordinates outside the regional bounding box
        if bounding_box is not None and not valid_coords.empty:
            in_bounds = bounding_box.contains(valid_coords[lat_col], valid_coords[lon_col])
            valid_coords = valid_coords[in_bounds]
    else:
        valid_coords = pd.DataFrame(columns=[division_col, lat_col, lon_col])

    # Compute median per division from incident data
    computed_records: dict[str, tuple[float, float, str]] = {}
    if not valid_coords.empty and division_col in valid_coords.columns:
        valid_coords[division_col] = valid_coords[division_col].astype(str).str.strip()
        grouped = valid_coords.groupby(division_col)[[lat_col, lon_col]].median()
        for div_name, row in grouped.iterrows():
            computed_records[str(div_name)] = (
                float(row[lat_col]),
                float(row[lon_col]),
                "incident_median",
            )

    # Reconcile across all expected divisions with known_centroids fallback
    rows = []
    missing_coords_divisions = []
    for div in all_divs:
        if div in computed_records:
            lat, lon, prov = computed_records[div]
            rows.append({division_col: div, lat_col: lat, lon_col: lon, "provenance": prov})
        elif known_centroids is not None and div in known_centroids:
            lat, lon = known_centroids[div]
            rows.append(
                {
                    division_col: div,
                    lat_col: float(lat),
                    lon_col: float(lon),
                    "provenance": "known_gazetteer",
                }
            )
        else:
            missing_coords_divisions.append(div)

    if missing_coords_divisions:
        raise ValueError(
            f"The following {len(missing_coords_divisions)} division(s) have no valid "
            f"coordinates from data or gazetteer: {missing_coords_divisions}"
        )

    result_df = pd.DataFrame(rows).sort_values(by=division_col).reset_index(drop=True)
    return result_df


def compute_pairwise_distances(centroids: np.ndarray) -> np.ndarray:
    """Compute an N x N pairwise Haversine distance matrix in kilometers.

    Args:
        centroids: Array of shape [N, 2] where column 0 is Lat, column 1 is Lon.
    Returns:
        N x N symmetric matrix of distances in km.
    """
    lat = centroids[:, 0]
    lon = centroids[:, 1]

    # Vectorized broadcasting: [N, 1] vs [1, N] -> [N, N]
    lat1, lat2 = np.meshgrid(lat, lat, indexing="ij")
    lon1, lon2 = np.meshgrid(lon, lon, indexing="ij")

    distances = haversine_distance(lat1, lon1, lat2, lon2)
    # Ensure exact zero on the diagonal and exact symmetry
    np.fill_diagonal(distances, 0.0)
    return (distances + distances.T) / 2.0


def build_gaussian_adjacency(
    distance_matrix: np.ndarray,
    bandwidth_sigma: float = 25.0,
    distance_cutoff: float = 50.0,
) -> np.ndarray:
    """Transform pairwise distances into connection weights via a Gaussian kernel.

    W_ij = exp(- d(i, j)^2 / (2 * sigma^2))  if d(i, j) <= cutoff, else 0.0
    Diagonal (self-connections) is set to 0.0 in the raw adjacency.
    """
    if bandwidth_sigma <= 0:
        raise ValueError("bandwidth_sigma must be strictly positive.")
    if distance_cutoff <= 0:
        raise ValueError("distance_cutoff must be strictly positive.")

    # Gaussian decay
    kernel = np.exp(-(distance_matrix**2) / (2.0 * (bandwidth_sigma**2)))

    # Apply cutoff: set weights for distances > cutoff to 0.0
    adjacency = np.where(distance_matrix <= distance_cutoff, kernel, 0.0)

    # Set self-loops to zero in raw adjacency (self-connections handled in normalization)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def normalize_adjacency_symmetric(adjacency: np.ndarray) -> np.ndarray:
    """Apply Kipf & Welling symmetric degree normalization: D^{-1/2} (A + I) D^{-1/2}.

    This prevents signal scale explosion in Graph Neural Networks.
    """
    num_nodes = adjacency.shape[0]
    # Add self-loops (identity matrix)
    adj_tilde = adjacency + np.eye(num_nodes, dtype=adjacency.dtype)

    # Degree matrix: sum across rows
    degrees = np.sum(adj_tilde, axis=1)

    # Invert square root: d^(-1/2), handling zero degree if any
    with np.errstate(divide="ignore"):
        degree_inv_sqrt = np.power(degrees, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = np.diag(degree_inv_sqrt)
    normalized = d_mat_inv_sqrt @ adj_tilde @ d_mat_inv_sqrt
    return normalized


def dense_to_edge_list(
    adjacency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an N x N adjacency matrix into sparse PyTorch-Geometric compatible edge tensors.

    Returns:
        edge_index: Shape [2, E] where row 0 is source indices and row 1 is target indices.
        edge_weights: Shape [E] containing non-zero edge weights.
    """
    rows, cols = np.nonzero(adjacency)
    weights = adjacency[rows, cols]
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    return edge_index, weights.astype(np.float32)


def build_spatial_graph(
    node_names: Sequence[str],
    coordinates: np.ndarray,
    bandwidth_sigma: float = 25.0,
    distance_cutoff: float = 50.0,
) -> SpatialGraph:
    """End-to-end builder: takes station names and lat/lon coordinates and builds SpatialGraph."""
    num_nodes = len(node_names)
    if coordinates.shape != (num_nodes, 2):
        raise ValueError(f"Coordinates shape {coordinates.shape} does not match ({num_nodes}, 2)")

    distances = compute_pairwise_distances(coordinates)
    raw_adj = build_gaussian_adjacency(
        distances,
        bandwidth_sigma=bandwidth_sigma,
        distance_cutoff=distance_cutoff,
    )
    norm_adj = normalize_adjacency_symmetric(raw_adj)
    edge_index, edge_weights = dense_to_edge_list(raw_adj)

    return SpatialGraph(
        node_names=list(node_names),
        centroids=coordinates,
        distance_matrix=distances,
        adjacency_matrix=raw_adj,
        normalized_adjacency=norm_adj,
        edge_index=edge_index,
        edge_weights=edge_weights,
        bandwidth_sigma=bandwidth_sigma,
        distance_cutoff=distance_cutoff,
    )
