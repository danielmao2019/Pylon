"""
Laplacian utilities for mesh processing.
"""

import heapq
from typing import List, Tuple

import torch


def cotangent(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    ab = b - a
    ac = c - a
    cross = torch.cross(ab, ac, dim=1)
    return torch.sum(ab * ac, dim=1) / torch.linalg.norm(cross, dim=1)


def build_cotangent_laplacian(
    base_vertices: torch.Tensor, faces: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_vertices = base_vertices.shape[0]
    face_verts = base_vertices[faces]
    v0 = face_verts[:, 0]
    v1 = face_verts[:, 1]
    v2 = face_verts[:, 2]
    cot0 = cotangent(v1, v2, v0)
    cot1 = cotangent(v0, v2, v1)
    cot2 = cotangent(v0, v1, v2)

    edges = torch.cat(
        [
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 0], faces[:, 2]], dim=1),
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
        ],
        dim=0,
    )
    weights = torch.cat([cot0, cot1, cot2], dim=0) * 0.5
    edges_sorted, _ = torch.sort(edges, dim=1)
    indices = edges_sorted.t()
    coalesced = torch.sparse_coo_tensor(
        indices, weights, size=(num_vertices, num_vertices)
    ).coalesce()
    edges_coalesced = coalesced.indices().t()
    weights_coalesced = coalesced.values()

    weight_sums = torch.zeros(num_vertices, device=base_vertices.device)
    weight_sums.index_add_(0, edges_coalesced[:, 0], weights_coalesced)
    weight_sums.index_add_(0, edges_coalesced[:, 1], weights_coalesced)
    return edges_coalesced, weights_coalesced, weight_sums


def compute_cotangent_weights_for_edges(
    base_vertices: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    # Input validations
    assert isinstance(base_vertices, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    assert isinstance(edges, torch.Tensor)
    assert base_vertices.ndim == 2
    assert base_vertices.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert faces.dtype == torch.long
    assert edges.dtype == torch.long
    assert int(faces.min().item()) >= 0
    assert int(faces.max().item()) < int(base_vertices.shape[0])
    assert int(edges.min().item()) >= 0
    assert int(edges.max().item()) < int(base_vertices.shape[0])

    # Input normalizations
    faces = faces.to(device=base_vertices.device)
    edges = edges.to(device=base_vertices.device)

    num_vertices = int(base_vertices.shape[0])
    face_verts = base_vertices[faces]
    v0 = face_verts[:, 0]
    v1 = face_verts[:, 1]
    v2 = face_verts[:, 2]
    cot0 = cotangent(v1, v2, v0)
    cot1 = cotangent(v0, v2, v1)
    cot2 = cotangent(v0, v1, v2)
    tri_edges = torch.cat(
        [
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 0], faces[:, 2]], dim=1),
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
        ],
        dim=0,
    )
    tri_weights = torch.cat([cot0, cot1, cot2], dim=0) * 0.5
    tri_edges, _ = torch.sort(tri_edges, dim=1)

    edge_keys = edges[:, 0] * num_vertices + edges[:, 1]
    edge_keys_sorted, sorted_edge_ids = torch.sort(edge_keys)
    tri_keys = tri_edges[:, 0] * num_vertices + tri_edges[:, 1]
    tri_positions = torch.searchsorted(edge_keys_sorted, tri_keys)
    assert torch.all(tri_positions >= 0)
    assert torch.all(tri_positions < int(edge_keys_sorted.shape[0]))
    assert torch.equal(edge_keys_sorted[tri_positions], tri_keys)

    edge_ids = sorted_edge_ids[tri_positions]
    cotangent_weights = torch.zeros(
        int(edges.shape[0]),
        dtype=base_vertices.dtype,
        device=base_vertices.device,
    )
    cotangent_weights.index_add_(0, edge_ids, tri_weights)
    return cotangent_weights


def laplacian_apply(
    vertices: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    vi = vertices[edges[:, 0]]
    vj = vertices[edges[:, 1]]
    weighted = weights[:, None] * (vi - vj)
    result = torch.zeros_like(vertices)
    result.index_add_(0, edges[:, 0], weighted)
    result.index_add_(0, edges[:, 1], -weighted)
    return result


def build_neighbor_data(
    edges: torch.Tensor,
    weights: torch.Tensor,
    base_vertices: torch.Tensor,
    num_vertices: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # Input validations
    assert isinstance(edges, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(base_vertices, torch.Tensor)
    assert isinstance(num_vertices, int)
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert weights.ndim == 1
    assert int(edges.shape[0]) == int(weights.shape[0])
    assert int(edges.shape[0]) > 0
    assert base_vertices.ndim == 2
    assert base_vertices.shape[1] == 3
    assert int(base_vertices.shape[0]) == num_vertices
    assert edges.dtype == torch.long
    assert int(edges.min().item()) >= 0
    assert int(edges.max().item()) < num_vertices

    # Build directed edge incidence tensorized, then group by source vertex.
    directed_sources = torch.stack(
        [edges[:, 0], edges[:, 1]],
        dim=1,
    ).reshape(-1)
    directed_targets = torch.stack(
        [edges[:, 1], edges[:, 0]],
        dim=1,
    ).reshape(-1)
    directed_weights = weights.unsqueeze(1).repeat(1, 2).reshape(-1)

    sort_ids = torch.argsort(directed_sources, stable=True)
    directed_sources = directed_sources[sort_ids]
    directed_targets = directed_targets[sort_ids]
    directed_weights = directed_weights[sort_ids]

    neighbor_counts = torch.bincount(directed_sources, minlength=num_vertices)
    assert int(neighbor_counts.sum().item()) == int(directed_targets.shape[0])
    split_indices = torch.cumsum(neighbor_counts, dim=0)[:-1].detach().cpu().tolist()

    neighbors = list(
        torch.tensor_split(
            directed_targets,
            indices_or_sections=split_indices,
            dim=0,
        )
    )
    neighbor_weights = list(
        torch.tensor_split(
            directed_weights,
            indices_or_sections=split_indices,
            dim=0,
        )
    )
    directed_rest_edges = (
        base_vertices[directed_sources] - base_vertices[directed_targets]
    )
    neighbor_rest_edges = list(
        torch.tensor_split(
            directed_rest_edges,
            indices_or_sections=split_indices,
            dim=0,
        )
    )
    assert len(neighbors) == num_vertices
    assert len(neighbor_weights) == num_vertices
    assert len(neighbor_rest_edges) == num_vertices
    return neighbors, neighbor_weights, neighbor_rest_edges


def build_adjacency(
    num_vertices: int, edges: torch.Tensor, lengths: torch.Tensor
) -> List[List[Tuple[int, float]]]:
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_vertices)]
    for idx in range(edges.shape[0]):
        i = int(edges[idx, 0].item())
        j = int(edges[idx, 1].item())
        length = float(lengths[idx].item())
        adjacency[i].append((j, length))
        adjacency[j].append((i, length))
    return adjacency


def geodesic_distances(
    num_vertices: int, edges: torch.Tensor, lengths: torch.Tensor, source: int
) -> torch.Tensor:
    adjacency = build_adjacency(num_vertices, edges, lengths)
    distances = [float("inf")] * num_vertices
    distances[source] = 0.0
    visited = [False] * num_vertices
    heap: List[Tuple[float, int]] = [(0.0, source)]
    while heap:
        dist_u, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        for v, weight in adjacency[u]:
            new_dist = dist_u + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return torch.tensor(distances, dtype=torch.float32)
