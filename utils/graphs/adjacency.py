"""Utilities for weighted undirected adjacency maps."""

import math
from numbers import Real
from typing import Callable, Dict, List, Tuple


def build_undirected_weighted_adjacency(
    node_ids: List[int], edges: List[Tuple[int, int, int]]
) -> Dict[int, Dict[int, int]]:
    # Input validations
    assert isinstance(node_ids, list), f"{type(node_ids)=}"
    assert node_ids, "node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in node_ids), f"{node_ids=}"
    assert isinstance(edges, list), f"{type(edges)=}"
    assert all(isinstance(edge, tuple) and len(edge) == 3 for edge in edges), f"{edges=}"
    assert all(
        isinstance(edge[0], int) and isinstance(edge[1], int) and isinstance(edge[2], int)
        for edge in edges
    ), f"{edges=}"

    adjacency: Dict[int, Dict[int, int]] = {node_id: {} for node_id in node_ids}
    node_id_set = set(node_ids)

    for node_id_a, node_id_b, edge_weight in edges:
        assert node_id_a in node_id_set, f"{node_id_a=} missing from node_ids"
        assert node_id_b in node_id_set, f"{node_id_b=} missing from node_ids"
        assert node_id_a != node_id_b, f"Self-loop is not supported: {node_id_a=}"
        assert edge_weight >= 0, f"{edge_weight=}"
        assert node_id_b not in adjacency[node_id_a], (
            f"Duplicate undirected edge detected: {(node_id_a, node_id_b)=}"
        )
        adjacency[node_id_a][node_id_b] = edge_weight
        adjacency[node_id_b][node_id_a] = edge_weight

    return adjacency


def compute_degrees(adjacency: Dict[int, Dict[int, int]]) -> Dict[int, int]:
    # Input validations
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert adjacency, "adjacency must be non-empty"
    assert all(isinstance(node_id, int) for node_id in adjacency), f"{adjacency=}"
    assert all(
        isinstance(neighbors, dict) for neighbors in adjacency.values()
    ), f"{adjacency=}"

    degrees = {node_id: len(neighbors) for node_id, neighbors in adjacency.items()}
    return degrees


def compute_weighted_degrees_log1p(
    adjacency: Dict[int, Dict[int, int]]
) -> Dict[int, float]:
    # Input validations
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert adjacency, "adjacency must be non-empty"
    assert all(isinstance(node_id, int) for node_id in adjacency), f"{adjacency=}"
    assert all(
        isinstance(neighbors, dict) for neighbors in adjacency.values()
    ), f"{adjacency=}"

    weighted_degrees: Dict[int, float] = {}
    for node_id, neighbors in adjacency.items():
        weighted_degree = 0.0
        for neighbor_id, edge_weight in neighbors.items():
            assert isinstance(neighbor_id, int), f"{neighbor_id=}"
            assert isinstance(edge_weight, int), f"{edge_weight=}"
            assert edge_weight >= 0, f"{edge_weight=}"
            weighted_degree += math.log1p(edge_weight)
        weighted_degrees[node_id] = weighted_degree

    return weighted_degrees


def compute_weighted_degrees(
    adjacency: Dict[int, Dict[int, int]],
    edge_weight_transform: Callable[[int], float],
) -> Dict[int, float]:
    # Input validations
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert adjacency, "adjacency must be non-empty"
    assert all(isinstance(node_id, int) for node_id in adjacency), f"{adjacency=}"
    assert all(
        isinstance(neighbors, dict) for neighbors in adjacency.values()
    ), f"{adjacency=}"
    assert callable(edge_weight_transform), f"{type(edge_weight_transform)=}"

    weighted_degrees: Dict[int, float] = {}
    for node_id, neighbors in adjacency.items():
        weighted_degree = 0.0
        for neighbor_id, edge_weight in neighbors.items():
            assert isinstance(neighbor_id, int), f"{neighbor_id=}"
            assert isinstance(edge_weight, int), f"{edge_weight=}"
            assert edge_weight >= 0, f"{edge_weight=}"
            transformed_weight = edge_weight_transform(edge_weight)
            assert isinstance(transformed_weight, Real), f"{type(transformed_weight)=}"
            weighted_degree += transformed_weight
        weighted_degrees[node_id] = weighted_degree

    return weighted_degrees
