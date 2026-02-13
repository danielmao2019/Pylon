"""
MODELS.THREE_D.MESHES.OPS API
"""

from models.three_d.meshes.ops.arap import (
    DEFAULT_EARLY_STOP_PATIENCE,
    apply_arap_operator,
    build_arap_rhs,
    compute_arap_energy,
    conjugate_gradient,
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
from models.three_d.meshes.ops.normals import compute_vertex_normals
from models.three_d.meshes.ops.topology import build_topology_edges_from_faces

__all__ = (
    "DEFAULT_EARLY_STOP_PATIENCE",
    "apply_arap_operator",
    "build_arap_rhs",
    "compute_arap_energy",
    "conjugate_gradient",
    "estimate_rotations",
    "run_arap",
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
