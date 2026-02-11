"""Utilities for induced-subgraph computations."""

from typing import Dict, List


def collect_induced_edge_weights(
    subset_node_ids: List[int], adjacency: Dict[int, Dict[int, int]]
) -> List[int]:
    # Input validations
    assert isinstance(subset_node_ids, list), f"{type(subset_node_ids)=}"
    assert subset_node_ids, "subset_node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in subset_node_ids), (
        f"{subset_node_ids=}"
    )
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert all(node_id in adjacency for node_id in subset_node_ids), (
        "adjacency missing subset nodes"
    )

    subset_node_id_set = set(subset_node_ids)
    induced_edge_weights: List[int] = []
    for node_id in subset_node_ids:
        for neighbor_id, edge_weight in adjacency[node_id].items():
            if neighbor_id not in subset_node_id_set:
                continue
            if node_id < neighbor_id:
                induced_edge_weights.append(edge_weight)
    return induced_edge_weights
