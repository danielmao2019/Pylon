"""Utilities for subset connectivity in undirected graphs."""

import math
from numbers import Real
from typing import Callable, Dict, List, Tuple


def compute_subset_connectivity_log1p(
    candidate_node_ids: List[int],
    subset_node_ids: List[int],
    adjacency: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    # Input validations
    assert isinstance(candidate_node_ids, list), f"{type(candidate_node_ids)=}"
    assert candidate_node_ids, "candidate_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in candidate_node_ids), (
        f"{candidate_node_ids=}"
    )
    assert isinstance(subset_node_ids, list), f"{type(subset_node_ids)=}"
    assert subset_node_ids, "subset_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in subset_node_ids), (
        f"{subset_node_ids=}"
    )
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert all(node_id in adjacency for node_id in candidate_node_ids), (
        "adjacency missing candidate nodes"
    )
    assert all(node_id in adjacency for node_id in subset_node_ids), (
        "adjacency missing subset nodes"
    )

    subset_node_id_set = set(subset_node_ids)
    subset_neighbor_counts: Dict[int, int] = {}
    subset_weight_sums: Dict[int, float] = {}
    for candidate_node_id in candidate_node_ids:
        candidate_subset_neighbor_count = 0
        candidate_subset_weight_sum = 0.0
        for neighbor_id, edge_weight in adjacency[candidate_node_id].items():
            assert isinstance(neighbor_id, int), f"{neighbor_id=}"
            assert isinstance(edge_weight, int), f"{edge_weight=}"
            assert edge_weight >= 0, f"{edge_weight=}"
            if neighbor_id in subset_node_id_set:
                candidate_subset_neighbor_count += 1
                candidate_subset_weight_sum += math.log1p(edge_weight)
        subset_neighbor_counts[candidate_node_id] = candidate_subset_neighbor_count
        subset_weight_sums[candidate_node_id] = candidate_subset_weight_sum

    return subset_neighbor_counts, subset_weight_sums


def compute_subset_connectivity(
    candidate_node_ids: List[int],
    subset_node_ids: List[int],
    adjacency: Dict[int, Dict[int, int]],
    edge_weight_transform: Callable[[int], float],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    # Input validations
    assert isinstance(candidate_node_ids, list), f"{type(candidate_node_ids)=}"
    assert candidate_node_ids, "candidate_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in candidate_node_ids), (
        f"{candidate_node_ids=}"
    )
    assert isinstance(subset_node_ids, list), f"{type(subset_node_ids)=}"
    assert subset_node_ids, "subset_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in subset_node_ids), (
        f"{subset_node_ids=}"
    )
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert callable(edge_weight_transform), f"{type(edge_weight_transform)=}"
    assert all(node_id in adjacency for node_id in candidate_node_ids), (
        "adjacency missing candidate nodes"
    )
    assert all(node_id in adjacency for node_id in subset_node_ids), (
        "adjacency missing subset nodes"
    )

    subset_node_id_set = set(subset_node_ids)
    subset_neighbor_counts: Dict[int, int] = {}
    subset_weight_sums: Dict[int, float] = {}
    for candidate_node_id in candidate_node_ids:
        candidate_subset_neighbor_count = 0
        candidate_subset_weight_sum = 0.0
        for neighbor_id, edge_weight in adjacency[candidate_node_id].items():
            assert isinstance(neighbor_id, int), f"{neighbor_id=}"
            assert isinstance(edge_weight, int), f"{edge_weight=}"
            assert edge_weight >= 0, f"{edge_weight=}"
            if neighbor_id in subset_node_id_set:
                candidate_subset_neighbor_count += 1
                transformed_weight = edge_weight_transform(edge_weight)
                assert isinstance(transformed_weight, Real), f"{type(transformed_weight)=}"
                candidate_subset_weight_sum += transformed_weight
        subset_neighbor_counts[candidate_node_id] = candidate_subset_neighbor_count
        subset_weight_sums[candidate_node_id] = candidate_subset_weight_sum

    return subset_neighbor_counts, subset_weight_sums


def count_non_subset_nodes_with_min_subset_neighbors(
    node_ids: List[int],
    subset_node_ids: List[int],
    adjacency: Dict[int, Dict[int, int]],
    min_neighbors: int,
) -> Tuple[int, int]:
    # Input validations
    assert isinstance(node_ids, list), f"{type(node_ids)=}"
    assert node_ids, "node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in node_ids), f"{node_ids=}"
    assert isinstance(subset_node_ids, list), f"{type(subset_node_ids)=}"
    assert subset_node_ids, "subset_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in subset_node_ids), (
        f"{subset_node_ids=}"
    )
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert isinstance(min_neighbors, int), f"{type(min_neighbors)=}"
    assert min_neighbors >= 0, f"{min_neighbors=}"
    assert all(node_id in adjacency for node_id in node_ids), (
        "adjacency missing node_ids entries"
    )
    assert all(node_id in adjacency for node_id in subset_node_ids), (
        "adjacency missing subset_node_ids entries"
    )

    subset_node_id_set = set(subset_node_ids)
    total_non_subset_nodes = len(node_ids) - len(subset_node_id_set)
    num_non_subset_nodes_with_min_subset_neighbors = 0
    for node_id in node_ids:
        if node_id in subset_node_id_set:
            continue
        subset_neighbor_count = 0
        for neighbor_id in adjacency[node_id]:
            if neighbor_id in subset_node_id_set:
                subset_neighbor_count += 1
                if subset_neighbor_count >= min_neighbors:
                    break
        if subset_neighbor_count >= min_neighbors:
            num_non_subset_nodes_with_min_subset_neighbors += 1
    return num_non_subset_nodes_with_min_subset_neighbors, total_non_subset_nodes
