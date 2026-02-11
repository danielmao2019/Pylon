"""Utilities for k-core computations on undirected graphs."""

from collections import deque
from typing import Dict, List


def compute_k_core_nodes(
    node_ids: List[int], adjacency: Dict[int, Dict[int, int]], k_core: int
) -> List[int]:
    # Input validations
    assert isinstance(node_ids, list), f"{type(node_ids)=}"
    assert node_ids, "node_ids must be non-empty"
    assert all(isinstance(node_id, int) for node_id in node_ids), f"{node_ids=}"
    assert isinstance(adjacency, dict), f"{type(adjacency)=}"
    assert isinstance(k_core, int), f"{type(k_core)=}"
    assert k_core > 0, f"{k_core=}"
    assert all(node_id in adjacency for node_id in node_ids), (
        "adjacency missing node_id entries from node_ids"
    )

    remaining_node_ids = set(node_ids)
    degrees: Dict[int, int] = {}
    for node_id in node_ids:
        node_degree = 0
        neighbors = adjacency[node_id]
        for neighbor_id in neighbors:
            if neighbor_id in remaining_node_ids:
                node_degree += 1
        degrees[node_id] = node_degree

    removal_queue = deque(
        [node_id for node_id in node_ids if degrees[node_id] < k_core]
    )
    while removal_queue:
        node_id = removal_queue.popleft()
        if node_id not in remaining_node_ids:
            continue
        remaining_node_ids.remove(node_id)
        for neighbor_id in adjacency[node_id]:
            if neighbor_id not in remaining_node_ids:
                continue
            degrees[neighbor_id] -= 1
            if degrees[neighbor_id] < k_core:
                removal_queue.append(neighbor_id)

    return sorted(remaining_node_ids)
