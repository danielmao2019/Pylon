"""
Sparse linear-system helpers for mesh operators.
"""

from typing import Any

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch


def build_weighted_laplacian_sparse_matrix(
    num_vertices: int,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
) -> sparse.csc_matrix:
    # Input validations
    assert isinstance(num_vertices, int)
    assert num_vertices > 0
    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert edge_vertex_indices.ndim == 2
    assert edge_vertex_indices.shape[1] == 2
    assert edge_vertex_indices.dtype == torch.long
    assert isinstance(weights, torch.Tensor)
    assert weights.ndim == 1
    assert int(weights.shape[0]) == int(edge_vertex_indices.shape[0])
    assert int(edge_vertex_indices.min().item()) >= 0
    assert int(edge_vertex_indices.max().item()) < num_vertices

    edge_vertex_indices_np = (
        edge_vertex_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    weights_np = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    src = edge_vertex_indices_np[:, 0]
    dst = edge_vertex_indices_np[:, 1]

    rows = np.concatenate([src, dst, src, dst], axis=0)
    cols = np.concatenate([src, dst, dst, src], axis=0)
    data = np.concatenate([weights_np, weights_np, -weights_np, -weights_np], axis=0)

    laplacian_matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices),
        dtype=np.float64,
    ).tocsc()
    return laplacian_matrix


def build_constraint_diagonal_sparse_matrix(
    constraint_mask: torch.Tensor,
    lambda_c: float,
) -> sparse.csc_matrix:
    # Input validations
    assert isinstance(constraint_mask, torch.Tensor)
    assert constraint_mask.ndim == 1
    assert isinstance(lambda_c, float)

    constraint_np = lambda_c * constraint_mask.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    num_vertices = int(constraint_mask.shape[0])
    constraint_matrix = sparse.diags(
        diagonals=constraint_np,
        offsets=0,
        shape=(num_vertices, num_vertices),
        format="csc",
        dtype=np.float64,
    )
    return constraint_matrix


def factorize_sparse_system_matrix(
    system_matrix: sparse.csc_matrix,
) -> Any:
    # Input validations
    assert isinstance(system_matrix, sparse.csc_matrix)
    assert system_matrix.ndim == 2
    assert int(system_matrix.shape[0]) == int(system_matrix.shape[1])

    factorized_system = sparse_linalg.splu(system_matrix)
    return factorized_system


def factorize_laplacian_system(
    num_vertices: int,
    edge_vertex_indices: torch.Tensor,
    weights: torch.Tensor,
    constraint_mask: torch.Tensor,
    lambda_c: float,
    square_laplacian: bool,
) -> Any:
    # Input validations
    assert isinstance(num_vertices, int)
    assert num_vertices > 0
    assert isinstance(edge_vertex_indices, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(constraint_mask, torch.Tensor)
    assert isinstance(lambda_c, float)
    assert isinstance(square_laplacian, bool)
    assert int(constraint_mask.shape[0]) == num_vertices

    laplacian_matrix = build_weighted_laplacian_sparse_matrix(
        num_vertices=num_vertices,
        edge_vertex_indices=edge_vertex_indices,
        weights=weights,
    )
    operator_matrix = laplacian_matrix
    if square_laplacian:
        operator_matrix = laplacian_matrix @ laplacian_matrix

    system_matrix = operator_matrix + build_constraint_diagonal_sparse_matrix(
        constraint_mask=constraint_mask,
        lambda_c=lambda_c,
    )
    factorized_system = factorize_sparse_system_matrix(system_matrix=system_matrix)
    return factorized_system


def solve_factorized_sparse_system(
    factorized_system: Any,
    rhs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Input validations
    assert factorized_system is not None
    assert hasattr(factorized_system, "solve")
    assert isinstance(rhs, torch.Tensor)
    assert rhs.ndim == 2
    assert isinstance(device, torch.device)
    assert isinstance(dtype, torch.dtype)

    rhs_np = rhs.detach().cpu().numpy().astype(np.float64, copy=False)
    solution_np = np.stack(
        [
            factorized_system.solve(rhs_np[:, axis])
            for axis in range(int(rhs_np.shape[1]))
        ],
        axis=1,
    )
    solution = torch.from_numpy(solution_np).to(device=device, dtype=dtype)
    return solution
