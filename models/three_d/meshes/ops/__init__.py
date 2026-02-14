"""
MODELS.THREE_D.MESHES.OPS API
"""

from models.three_d.meshes.ops.arap import (
    DEFAULT_EARLY_STOP_PATIENCE,
    apply_arap_operator,
    build_arap_rhs,
    compute_arap_energy,
    estimate_rotations,
    run_arap,
)
from models.three_d.meshes.ops.laplacian import (
    build_adjacency,
    build_cotangent_laplacian,
    build_neighbor_data,
    compute_cotangent_weights_for_edges,
    cotangent,
    geodesic_distances,
    laplacian_apply,
)
from models.three_d.meshes.ops.linear_system import (
    build_constraint_diagonal_sparse_matrix,
    build_weighted_laplacian_sparse_matrix,
    factorize_laplacian_system,
    factorize_sparse_system_matrix,
    solve_factorized_sparse_system,
)
from models.three_d.meshes.ops.normals import compute_vertex_normals
from models.three_d.meshes.ops.topology import build_topology_edges_from_faces

__all__ = (
    "DEFAULT_EARLY_STOP_PATIENCE",
    "apply_arap_operator",
    "build_arap_rhs",
    "compute_arap_energy",
    "estimate_rotations",
    "run_arap",
    "build_constraint_diagonal_sparse_matrix",
    "build_weighted_laplacian_sparse_matrix",
    "factorize_laplacian_system",
    "factorize_sparse_system_matrix",
    "solve_factorized_sparse_system",
    "build_adjacency",
    "build_cotangent_laplacian",
    "build_neighbor_data",
    "compute_cotangent_weights_for_edges",
    "cotangent",
    "geodesic_distances",
    "laplacian_apply",
    "compute_vertex_normals",
    "build_topology_edges_from_faces",
)
