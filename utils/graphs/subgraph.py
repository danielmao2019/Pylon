"""Utilities for induced-subgraph computations."""

from typing import Dict, List, Optional, Tuple

import torch


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


def build_induced_edge_subset(
    num_nodes: int,
    edges: torch.Tensor,
    subset_node_indices: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    edge_vectors: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Input validations
    assert isinstance(num_nodes, int), f"{type(num_nodes)=}"
    assert isinstance(edges, torch.Tensor), f"{type(edges)=}"
    assert isinstance(subset_node_indices, torch.Tensor), f"{type(subset_node_indices)=}"
    assert edge_weights is None or isinstance(edge_weights, torch.Tensor), (
        f"{type(edge_weights)=}"
    )
    assert edge_vectors is None or isinstance(edge_vectors, torch.Tensor), (
        f"{type(edge_vectors)=}"
    )
    assert num_nodes > 0, f"{num_nodes=}"
    assert edges.ndim == 2, f"{edges.shape=}"
    assert int(edges.shape[1]) == 2, f"{edges.shape=}"
    assert edges.dtype == torch.long, f"{edges.dtype=}"
    assert subset_node_indices.ndim == 1, f"{subset_node_indices.shape=}"
    assert subset_node_indices.dtype == torch.long, f"{subset_node_indices.dtype=}"
    assert int(subset_node_indices.numel()) > 0, f"{subset_node_indices.shape=}"
    assert int(subset_node_indices.min().item()) >= 0, f"{subset_node_indices.min()=}"
    assert int(subset_node_indices.max().item()) < num_nodes, (
        f"{subset_node_indices.max()=}, {num_nodes=}"
    )
    assert edge_weights is None or (
        edge_weights.ndim == 1 and int(edge_weights.shape[0]) == int(edges.shape[0])
    ), f"{edge_weights.shape=}, {edges.shape=}"
    assert edge_vectors is None or (
        edge_vectors.ndim == 2
        and int(edge_vectors.shape[0]) == int(edges.shape[0])
        and int(edge_vectors.shape[1]) == 3
    ), f"{edge_vectors.shape=}, {edges.shape=}"
    assert int(edges.min().item()) >= 0, f"{edges.min()=}"
    assert int(edges.max().item()) < num_nodes, f"{edges.max()=}, {num_nodes=}"

    # Input normalizations
    subset_node_indices = subset_node_indices.to(device=edges.device, dtype=torch.long)
    edge_weights = (
        None
        if edge_weights is None
        else edge_weights.to(device=edges.device, dtype=edge_weights.dtype)
    )
    edge_vectors = (
        None
        if edge_vectors is None
        else edge_vectors.to(device=edges.device, dtype=edge_vectors.dtype)
    )

    global_to_local = torch.full(
        (num_nodes,),
        fill_value=-1,
        dtype=torch.long,
        device=edges.device,
    )
    global_to_local[subset_node_indices] = torch.arange(
        int(subset_node_indices.numel()),
        dtype=torch.long,
        device=edges.device,
    )

    edge_mask = (global_to_local[edges[:, 0]] >= 0) & (
        global_to_local[edges[:, 1]] >= 0
    )
    edge_ids = torch.nonzero(edge_mask, as_tuple=False).squeeze(1)
    assert int(edge_ids.numel()) > 0, "No induced edges found for subset."
    subset_edges_global = edges[edge_ids]
    subset_edges_local = torch.stack(
        [
            global_to_local[subset_edges_global[:, 0]],
            global_to_local[subset_edges_global[:, 1]],
        ],
        dim=1,
    )
    assert torch.all(subset_edges_local >= 0), f"{subset_edges_local=}"

    subset_edge_weights = None if edge_weights is None else edge_weights[edge_ids]
    subset_edge_vectors = None if edge_vectors is None else edge_vectors[edge_ids]
    return edge_ids, subset_edges_local, subset_edge_weights, subset_edge_vectors
