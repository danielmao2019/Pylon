"""Utilities for connected-component analysis on induced subgraphs."""

from collections import deque
from typing import List

import torch


def build_induced_connected_components(
    num_nodes: int,
    edges: torch.Tensor,
    subset_node_indices: torch.Tensor,
) -> List[torch.Tensor]:
    # Input validations
    assert isinstance(num_nodes, int), f"{type(num_nodes)=}"
    assert isinstance(edges, torch.Tensor), f"{type(edges)=}"
    assert isinstance(subset_node_indices, torch.Tensor), f"{type(subset_node_indices)=}"
    assert num_nodes > 0, f"{num_nodes=}"
    assert edges.ndim == 2, f"{edges.shape=}"
    assert int(edges.shape[1]) == 2, f"{edges.shape=}"
    assert int(edges.shape[0]) > 0, f"{edges.shape=}"
    assert edges.dtype == torch.long, f"{edges.dtype=}"
    assert subset_node_indices.ndim == 1, f"{subset_node_indices.shape=}"
    assert subset_node_indices.dtype == torch.long, f"{subset_node_indices.dtype=}"
    assert int(subset_node_indices.numel()) > 0, f"{subset_node_indices.shape=}"
    assert int(subset_node_indices.min().item()) >= 0, f"{subset_node_indices.min()=}"
    assert int(subset_node_indices.max().item()) < num_nodes, (
        f"{subset_node_indices.max()=}, {num_nodes=}"
    )
    assert int(edges.min().item()) >= 0, f"{edges.min()=}"
    assert int(edges.max().item()) < num_nodes, f"{edges.max()=}, {num_nodes=}"
    assert torch.equal(
        subset_node_indices,
        torch.unique(subset_node_indices, sorted=True),
    ), "subset_node_indices must be unique and sorted."

    # Input normalizations
    subset_node_indices = subset_node_indices.to(device=edges.device, dtype=torch.long)

    subset_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edges.device)
    subset_mask[subset_node_indices] = True
    induced_edge_mask = subset_mask[edges[:, 0]] & subset_mask[edges[:, 1]]
    induced_edges = edges[induced_edge_mask].detach().cpu()
    subset_node_indices_cpu = subset_node_indices.detach().cpu()

    adjacency = {
        int(node_index.item()): [] for node_index in subset_node_indices_cpu
    }
    for edge in induced_edges:
        node_index_a = int(edge[0].item())
        node_index_b = int(edge[1].item())
        adjacency[node_index_a].append(node_index_b)
        adjacency[node_index_b].append(node_index_a)

    visited = set()
    connected_components: List[torch.Tensor] = []
    for node_index in subset_node_indices_cpu:
        start_node_index = int(node_index.item())
        if start_node_index in visited:
            continue
        queue = deque([start_node_index])
        visited.add(start_node_index)
        component_nodes: List[int] = []
        while queue:
            current_node_index = queue.popleft()
            component_nodes.append(current_node_index)
            for neighbor_node_index in adjacency[current_node_index]:
                if neighbor_node_index in visited:
                    continue
                visited.add(neighbor_node_index)
                queue.append(neighbor_node_index)
        connected_components.append(
            torch.tensor(component_nodes, dtype=torch.long, device=edges.device)
        )

    assert len(connected_components) > 0, "At least one component must exist."
    return connected_components


def compute_component_overlap_counts(
    connected_components: List[torch.Tensor],
    query_node_indices: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    # Input validations
    assert isinstance(connected_components, list), f"{type(connected_components)=}"
    assert len(connected_components) > 0, "connected_components must be non-empty."
    assert all(isinstance(component, torch.Tensor) for component in connected_components), (
        f"{connected_components=}"
    )
    assert isinstance(query_node_indices, torch.Tensor), f"{type(query_node_indices)=}"
    assert isinstance(num_nodes, int), f"{type(num_nodes)=}"
    assert num_nodes > 0, f"{num_nodes=}"
    assert query_node_indices.ndim == 1, f"{query_node_indices.shape=}"
    assert query_node_indices.dtype == torch.long, f"{query_node_indices.dtype=}"
    assert int(query_node_indices.numel()) > 0, f"{query_node_indices.shape=}"
    assert int(query_node_indices.min().item()) >= 0, f"{query_node_indices.min()=}"
    assert int(query_node_indices.max().item()) < num_nodes, (
        f"{query_node_indices.max()=}, {num_nodes=}"
    )
    assert all(component.ndim == 1 for component in connected_components), (
        f"{connected_components=}"
    )
    assert all(component.dtype == torch.long for component in connected_components), (
        f"{connected_components=}"
    )
    assert all(int(component.numel()) > 0 for component in connected_components), (
        f"{connected_components=}"
    )

    query_mask = torch.zeros(num_nodes, dtype=torch.bool, device=query_node_indices.device)
    query_mask[query_node_indices] = True

    overlap_counts: List[int] = []
    for component in connected_components:
        overlap_count = int(query_mask[component].sum().item())
        overlap_counts.append(overlap_count)

    return torch.tensor(overlap_counts, dtype=torch.long, device=query_node_indices.device)
