import numpy as np
import pandas as pd
import pytest

from predictive_policing.spatial_temporal.graph import (
    ZONE_17_BOUNDING_BOX,
    SpatialGraph,
    build_gaussian_adjacency,
    build_spatial_graph,
    compute_division_centroids,
    compute_pairwise_distances,
    dense_to_edge_list,
    haversine_distance,
    normalize_adjacency_symmetric,
)


def test_haversine_distance_known_coordinates() -> None:
    # Akure (Ondo State) ~ 7.2526° N, 5.1931° E
    # Ado-Ekiti (Ekiti State) ~ 7.6211° N, 5.2215° E
    akure_lat, akure_lon = 7.2526, 5.1931
    ado_lat, ado_lon = 7.6211, 5.2215

    distance = haversine_distance(akure_lat, akure_lon, ado_lat, ado_lon)
    # The straight-line distance is approximately 41.0 km +/- 2 km
    assert 39.0 < distance < 43.0

    # Same location must be exactly 0
    zero_dist = haversine_distance(akure_lat, akure_lon, akure_lat, akure_lon)
    assert pytest.approx(zero_dist, abs=1e-6) == 0.0


def test_haversine_vectorized() -> None:
    lats1 = np.array([7.25, 7.62])
    lons1 = np.array([5.19, 5.22])
    lats2 = np.array([7.62, 7.25])
    lons2 = np.array([5.22, 5.19])

    distances = haversine_distance(lats1, lons1, lats2, lons2)
    assert isinstance(distances, np.ndarray)
    assert len(distances) == 2
    assert pytest.approx(distances[0], rel=1e-4) == distances[1]


def test_bounding_box_filtering() -> None:
    bbox = ZONE_17_BOUNDING_BOX
    # Akure is inside
    assert bbox.contains(7.25, 5.19)
    # London (51.5, -0.1) is outside
    assert not bbox.contains(51.5, -0.1)
    # Atlantic ocean (0.0, 0.0) is outside
    assert not bbox.contains(0.0, 0.0)


def test_compute_division_centroids_from_df() -> None:
    df = pd.DataFrame(
        [
            {"division_outpost": "Akure Central", "latitude": 7.25, "longitude": 5.19},
            {"division_outpost": "Akure Central", "latitude": 7.27, "longitude": 5.21},
            # Outlier/typo outside Zone 17 bounding box that should be filtered out
            {"division_outpost": "Akure Central", "latitude": 51.50, "longitude": -0.10},
            {"division_outpost": "Owo Division", "latitude": 7.19, "longitude": 5.58},
            {"division_outpost": "Owo Division", "latitude": 7.20, "longitude": 5.59},
            # Missing row that must be ignored
            {"division_outpost": "Owo Division", "latitude": None, "longitude": None},
        ]
    )

    centroids_df = compute_division_centroids(df, bounding_box=ZONE_17_BOUNDING_BOX)
    assert len(centroids_df) == 2
    assert list(centroids_df["division_outpost"]) == ["Akure Central", "Owo Division"]

    # Median for Akure lat: [7.25, 7.27] (51.50 filtered out) -> median is 7.26
    akure_row = centroids_df[centroids_df["division_outpost"] == "Akure Central"].iloc[0]
    assert pytest.approx(akure_row["latitude"]) == 7.26
    assert pytest.approx(akure_row["longitude"]) == 5.20
    assert akure_row["provenance"] == "incident_median"


def test_compute_division_centroids_gazetteer_fallback() -> None:
    df = pd.DataFrame(
        [
            {"division_outpost": "Akure Central", "latitude": 7.25, "longitude": 5.19},
            # Remote Outpost has records, but zero coordinates
            {"division_outpost": "Remote Outpost", "latitude": None, "longitude": None},
        ]
    )

    known = {"Remote Outpost": (7.50, 5.40)}
    centroids_df = compute_division_centroids(
        df,
        bounding_box=ZONE_17_BOUNDING_BOX,
        known_centroids=known,
    )

    assert len(centroids_df) == 2
    remote_row = centroids_df[centroids_df["division_outpost"] == "Remote Outpost"].iloc[0]
    assert remote_row["latitude"] == 7.50
    assert remote_row["longitude"] == 5.40
    assert remote_row["provenance"] == "known_gazetteer"


def test_compute_division_centroids_missing_raises_error() -> None:
    df = pd.DataFrame(
        [
            {"division_outpost": "Akure Central", "latitude": 7.25, "longitude": 5.19},
            {"division_outpost": "Unknown Outpost", "latitude": None, "longitude": None},
        ]
    )
    with pytest.raises(ValueError, match="no valid coordinates"):
        compute_division_centroids(df, bounding_box=ZONE_17_BOUNDING_BOX)


def test_compute_pairwise_distances() -> None:
    centroids = np.array(
        [
            [7.25, 5.19],  # Station 0 (Akure)
            [7.62, 5.22],  # Station 1 (Ado-Ekiti)
            [7.19, 5.58],  # Station 2 (Owo)
        ]
    )
    dist_mat = compute_pairwise_distances(centroids)

    assert dist_mat.shape == (3, 3)
    # Diagonal must be 0
    assert np.allclose(np.diag(dist_mat), 0.0)
    # Must be symmetric
    assert np.allclose(dist_mat, dist_mat.T)
    # Akure to Ado-Ekiti distance should match
    expected_akure_ado = haversine_distance(7.25, 5.19, 7.62, 5.22)
    assert pytest.approx(dist_mat[0, 1], rel=1e-5) == expected_akure_ado


def test_gaussian_adjacency_and_cutoff() -> None:
    # 3 stations: dist(0, 1) = 20 km, dist(0, 2) = 80 km, dist(1, 2) = 90 km
    dist_mat = np.array(
        [
            [0.0, 20.0, 80.0],
            [20.0, 0.0, 90.0],
            [80.0, 90.0, 0.0],
        ]
    )

    # Cutoff at 50 km: connection between 0 and 1 remains; 0-2 and 1-2 become 0
    adj = build_gaussian_adjacency(dist_mat, bandwidth_sigma=20.0, distance_cutoff=50.0)

    # Diagonal must be 0
    assert np.allclose(np.diag(adj), 0.0)
    # Distances > 50km must be strictly zero
    assert adj[0, 2] == 0.0
    assert adj[1, 2] == 0.0
    # Connection between 0 and 1: exp(- (20^2) / (2 * 20^2)) = exp(-0.5) ~ 0.6065
    expected_w = np.exp(-0.5)
    assert pytest.approx(adj[0, 1], rel=1e-4) == expected_w
    assert pytest.approx(adj[1, 0], rel=1e-4) == expected_w


def test_normalize_adjacency_symmetric() -> None:
    adj = np.array(
        [
            [0.0, 0.8],
            [0.8, 0.0],
        ]
    )
    norm_adj = normalize_adjacency_symmetric(adj)

    # With self-loop: A~ = [[1.0, 0.8], [0.8, 1.0]], degrees = [1.8, 1.8]
    # D^{-1/2} A~ D^{-1/2}: diagonal is 1.0 / 1.8 ~ 0.5555, off-diagonal 0.8 / 1.8 ~ 0.4444
    assert np.allclose(norm_adj, norm_adj.T)
    assert pytest.approx(norm_adj[0, 0], rel=1e-4) == 1.0 / 1.8
    assert pytest.approx(norm_adj[0, 1], rel=1e-4) == 0.8 / 1.8


def test_dense_to_edge_list() -> None:
    adj = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.9],
            [0.0, 0.9, 0.0],
        ]
    )
    edge_index, edge_weights = dense_to_edge_list(adj)

    # 4 directed edges: (0, 1), (1, 0), (1, 2), (2, 1)
    assert edge_index.shape == (2, 4)
    assert len(edge_weights) == 4
    assert list(edge_weights) == [0.5, 0.5, 0.9, 0.9]


def test_build_spatial_graph_end_to_end_and_isolated_nodes() -> None:
    names = ["Station A", "Station B", "Isolated Station"]
    coords = np.array(
        [
            [7.25, 5.19],  # Station A (Akure)
            [7.30, 5.22],  # Station B (~6.5 km from A)
            [8.80, 7.80],  # Isolated Station (~330 km away, well past 50km cutoff)
        ]
    )
    graph = build_spatial_graph(names, coords, bandwidth_sigma=25.0, distance_cutoff=50.0)

    assert isinstance(graph, SpatialGraph)
    assert graph.num_nodes == 3
    assert graph.num_edges == 2  # (A -> B) and (B -> A)
    assert graph.isolated_nodes == ["Isolated Station"]
    # Density: 2 edges out of 3*2 = 6 possible directed edges = 2/6 ~ 0.333
    assert pytest.approx(graph.graph_density, rel=1e-3) == 2.0 / 6.0
