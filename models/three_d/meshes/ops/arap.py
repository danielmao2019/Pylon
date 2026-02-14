"""
ARAP helpers for mesh deformation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from models.three_d.meshes.ops.linear_system import (
    factorize_laplacian_system,
    solve_factorized_sparse_system,
)

DEFAULT_EARLY_STOP_PATIENCE = 5


def estimate_rotations(
    vertices: torch.Tensor,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    reference_edge_vectors: torch.Tensor,
) -> torch.Tensor:
    # Input validations
    assert isinstance(vertices, torch.Tensor)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3

    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert int(edge_vertex_indices.shape[0]) > 0
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < int(vertices.shape[0])

    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(reference_edge_vectors, torch.Tensor)
    assert reference_edge_vectors.ndim == 2
    assert reference_edge_vectors.shape[1] == 3
    assert int(reference_edge_vectors.shape[0]) == int(edge_vertex_indices.shape[0])

    num_vertices = vertices.shape[0]
    edge_vec = vertices[edge_vertex_indices[:, 0]] - vertices[edge_vertex_indices[:, 1]]
    outer = edge_vec.unsqueeze(2) * reference_edge_vectors.unsqueeze(1)
    weighted_outer = weights[:, None, None] * outer
    cov = torch.zeros(
        (num_vertices, 3, 3), device=vertices.device, dtype=vertices.dtype
    )
    cov.index_add_(0, edge_vertex_indices[:, 0], weighted_outer)
    cov.index_add_(0, edge_vertex_indices[:, 1], weighted_outer)
    u, _, v = torch.linalg.svd(cov)
    det = torch.linalg.det(u @ v)
    signs = torch.ones((num_vertices, 3), device=vertices.device, dtype=vertices.dtype)
    signs[:, 2] = torch.sign(det)
    diag = torch.diag_embed(signs)
    rotations = u @ diag @ v
    return rotations


def apply_arap_operator(
    vertices: torch.Tensor,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    constraint_mask: torch.Tensor,
    lambda_c: float,
) -> torch.Tensor:
    # Input validations
    assert isinstance(vertices, torch.Tensor)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3

    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert int(edge_vertex_indices.shape[0]) > 0
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < int(vertices.shape[0])

    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(constraint_mask, torch.Tensor)
    assert constraint_mask.ndim == 1
    assert int(constraint_mask.shape[0]) == int(vertices.shape[0])

    assert isinstance(lambda_c, float)

    vi = vertices[edge_vertex_indices[:, 0]]
    vj = vertices[edge_vertex_indices[:, 1]]
    weighted = weights[:, None] * (vi - vj)
    result = torch.zeros_like(vertices)
    result.index_add_(0, edge_vertex_indices[:, 0], weighted)
    result.index_add_(0, edge_vertex_indices[:, 1], -weighted)
    result = result + lambda_c * constraint_mask[:, None] * vertices
    return result


def build_arap_rhs(
    rotations: torch.Tensor,
    reference_edge_vectors: torch.Tensor,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    constraint_mask: torch.Tensor,
    targets: torch.Tensor,
    lambda_c: float,
) -> torch.Tensor:
    # Input validations
    assert isinstance(rotations, torch.Tensor)
    assert rotations.ndim == 3
    assert rotations.shape[1] == 3
    assert rotations.shape[2] == 3

    assert isinstance(reference_edge_vectors, torch.Tensor)
    assert reference_edge_vectors.ndim == 2
    assert reference_edge_vectors.shape[1] == 3

    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert int(edge_vertex_indices.shape[0]) > 0
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < int(rotations.shape[0])
    assert int(edge_vertex_indices.shape[0]) == int(reference_edge_vectors.shape[0])

    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(constraint_mask, torch.Tensor)
    assert constraint_mask.ndim == 1
    assert int(constraint_mask.shape[0]) == int(rotations.shape[0])

    assert isinstance(targets, torch.Tensor)
    assert targets.ndim == 2
    assert targets.shape[1] == 3
    assert int(targets.shape[0]) == int(rotations.shape[0])

    assert isinstance(lambda_c, float)

    rot_avg = 0.5 * (
        rotations[edge_vertex_indices[:, 0]] + rotations[edge_vertex_indices[:, 1]]
    )
    term = torch.bmm(rot_avg, reference_edge_vectors.unsqueeze(-1)).squeeze(-1)
    weighted_term = weights[:, None] * term
    rhs = torch.zeros_like(targets)
    rhs.index_add_(0, edge_vertex_indices[:, 0], weighted_term)
    rhs.index_add_(0, edge_vertex_indices[:, 1], -weighted_term)
    rhs = rhs + lambda_c * constraint_mask[:, None] * targets
    return rhs


def compute_arap_energy(
    vertices: torch.Tensor,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    reference_edge_vectors: torch.Tensor,
    rotations: torch.Tensor,
    constraint_mask: torch.Tensor,
    targets: torch.Tensor,
    lambda_c: float,
) -> torch.Tensor:
    """Compute ARAP energy (edge + constraint terms)."""
    # Input validations
    assert isinstance(vertices, torch.Tensor)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3

    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert int(edge_vertex_indices.shape[0]) > 0
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < int(vertices.shape[0])

    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(reference_edge_vectors, torch.Tensor)
    assert reference_edge_vectors.ndim == 2
    assert reference_edge_vectors.shape[1] == 3
    assert int(reference_edge_vectors.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(rotations, torch.Tensor)
    assert rotations.ndim == 3
    assert int(rotations.shape[0]) == int(vertices.shape[0])
    assert rotations.shape[1] == 3
    assert rotations.shape[2] == 3

    assert isinstance(constraint_mask, torch.Tensor)
    assert constraint_mask.ndim == 1
    assert int(constraint_mask.shape[0]) == int(vertices.shape[0])

    assert isinstance(targets, torch.Tensor)
    assert targets.ndim == 2
    assert targets.shape[1] == 3
    assert int(targets.shape[0]) == int(vertices.shape[0])

    assert isinstance(lambda_c, float)

    edge_vec = vertices[edge_vertex_indices[:, 0]] - vertices[edge_vertex_indices[:, 1]]
    rot_avg = 0.5 * (
        rotations[edge_vertex_indices[:, 0]] + rotations[edge_vertex_indices[:, 1]]
    )
    rotated = torch.bmm(rot_avg, reference_edge_vectors.unsqueeze(-1)).squeeze(-1)
    residual = edge_vec - rotated
    edge_energy = weights * torch.sum(residual * residual, dim=1)
    constraint_residual = vertices - targets
    constraint_energy = (
        lambda_c
        * constraint_mask
        * torch.sum(constraint_residual * constraint_residual, dim=1)
    )
    return edge_energy.sum() + constraint_energy.sum()


def run_arap(
    vertices: torch.Tensor,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    reference_edge_vectors: torch.Tensor,
    constraint_mask: torch.Tensor,
    targets: torch.Tensor,
    lambda_c: float,
    max_iters: int,
    factorized_system: Optional[Any] = None,
    early_stop_patience: Optional[int] = DEFAULT_EARLY_STOP_PATIENCE,
    report_iters: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], int]:
    # Input validations
    assert isinstance(vertices, torch.Tensor)
    assert vertices.ndim == 2
    assert vertices.shape[1] == 3

    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert int(edge_vertex_indices.shape[0]) > 0
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < int(vertices.shape[0])

    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(reference_edge_vectors, torch.Tensor)
    assert reference_edge_vectors.ndim == 2
    assert reference_edge_vectors.shape[1] == 3
    assert int(reference_edge_vectors.shape[0]) == int(edge_vertex_indices.shape[0])

    assert isinstance(constraint_mask, torch.Tensor)
    assert constraint_mask.ndim == 1
    assert int(constraint_mask.shape[0]) == int(vertices.shape[0])

    assert isinstance(targets, torch.Tensor)
    assert targets.ndim == 2
    assert targets.shape[1] == 3
    assert int(targets.shape[0]) == int(vertices.shape[0])

    assert isinstance(lambda_c, float)

    assert isinstance(max_iters, int)
    assert max_iters > 0

    assert factorized_system is None or hasattr(factorized_system, "solve")
    assert early_stop_patience is None or (
        isinstance(early_stop_patience, int) and early_stop_patience > 0
    )

    assert report_iters is None or (
        isinstance(report_iters, list)
        and len(report_iters) > 0
        and all(isinstance(step, int) for step in report_iters)
        and min(report_iters) > 0
        and max(report_iters) <= max_iters
        and len(set(report_iters)) == len(report_iters)
    )

    # Input normalizations
    if factorized_system is None:
        factorized_system = factorize_laplacian_system(
            num_vertices=int(vertices.shape[0]),
            edge_vertex_indices=edge_vertex_indices,
            weights=weights,
            constraint_mask=constraint_mask,
            lambda_c=lambda_c,
            square_laplacian=False,
        )
    report_set = set() if report_iters is None else set(report_iters)

    progress: Dict[int, torch.Tensor] = {}
    best_energy: Optional[torch.Tensor] = None
    stale_iters = 0
    iterations_run = 0

    for iter_idx in range(max_iters):
        rotations = estimate_rotations(
            vertices=vertices,
            edge_vertex_indices=edge_vertex_indices,
            weights=weights,
            reference_edge_vectors=reference_edge_vectors,
        )
        rhs = build_arap_rhs(
            rotations=rotations,
            reference_edge_vectors=reference_edge_vectors,
            edge_vertex_indices=edge_vertex_indices,
            weights=weights,
            constraint_mask=constraint_mask,
            targets=targets,
            lambda_c=lambda_c,
        )
        vertices = solve_factorized_sparse_system(
            factorized_system=factorized_system,
            rhs=rhs,
            device=vertices.device,
            dtype=vertices.dtype,
        )
        iterations_run = iter_idx + 1
        if report_set and iterations_run in report_set:
            progress[iterations_run] = vertices.detach().clone()
            logging.info("Captured ARAP iteration %d/%d", iterations_run, max_iters)
        if early_stop_patience is not None:
            energy = compute_arap_energy(
                vertices=vertices,
                edge_vertex_indices=edge_vertex_indices,
                weights=weights,
                reference_edge_vectors=reference_edge_vectors,
                rotations=rotations,
                constraint_mask=constraint_mask,
                targets=targets,
                lambda_c=lambda_c,
            )
            if best_energy is None or energy < best_energy:
                best_energy = energy.detach()
                stale_iters = 0
            else:
                stale_iters += 1
                if stale_iters >= early_stop_patience:
                    break
    return vertices, progress, iterations_run
