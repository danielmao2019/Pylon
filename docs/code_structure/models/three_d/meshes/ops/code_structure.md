# `models/three_d/meshes/ops/` code skeleton

## Code implementation structure

`models/three_d/meshes/ops/__init__.py`

```text
__init__.py
├── from models.three_d.meshes.ops.apply_transform import apply_transform
├── from models.three_d.meshes.ops.arap import DEFAULT_EARLY_STOP_PATIENCE, apply_arap_operator, build_arap_rhs, compute_arap_energy, estimate_rotations, run_arap
├── from models.three_d.meshes.ops.laplacian import build_adjacency, build_cotangent_laplacian, build_neighbor_data, compute_cotangent_weights_for_edges, cotangent, geodesic_distances, laplacian_apply
├── from models.three_d.meshes.ops.linear_system import build_constraint_diagonal_sparse_matrix, build_weighted_laplacian_sparse_matrix, factorize_laplacian_system, factorize_sparse_system_matrix, solve_factorized_sparse_system
├── from models.three_d.meshes.ops.normals import compute_vertex_normals
├── from models.three_d.meshes.ops.topology import build_topology_edges_from_faces
└── from models.three_d.meshes.ops.world_to_camera_transform import world_to_camera_transform
```

`models/three_d/meshes/ops/apply_transform.py`

```text
apply_transform.py
├── from typing import Optional
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from utils.ops.chunked_matmul import chunked_matmul
└── def apply_transform(mesh: Mesh, transform: torch.Tensor, max_divide: int = 0, num_divide: Optional[int] = None) -> Mesh
    ├── # Returns a copy of mesh whose verts are mapped through a 4x4 transform in homogeneous coordinates, leaving faces and texture unchanged.
    ├── impls build the homogeneous [V, 4] verts by appending a ones column to mesh.verts
    ├── calls chunked_matmul  # homogeneous verts by transform.T, passing max_divide and num_divide, chunked over the V rows
    ├── impls drop the homogeneous coordinate to get the [V, 3] transformed verts
    ├── calls Mesh  # rebuild with the transformed verts, original faces, original texture
    └── return      # the transformed Mesh
```

`models/three_d/meshes/ops/arap.py`

```text
arap.py
├── from models.three_d.meshes.ops.linear_system import factorize_laplacian_system, solve_factorized_sparse_system
├── DEFAULT_EARLY_STOP_PATIENCE  # int = 5; default early-stop patience for run_arap.
├── def run_arap(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, reference_edge_vectors: torch.Tensor, constraint_mask: torch.Tensor, targets: torch.Tensor, lambda_c: float, max_iters: int, factorized_system: Optional[Any] = None, early_stop_patience: Optional[int] = DEFAULT_EARLY_STOP_PATIENCE, report_iters: Optional[List[int]] = None) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], int]
│   ├── # Runs the local/global ARAP iteration to deform verts toward targets under soft positional constraints.
│   ├── if factorized_system is None
│   │   └── calls factorize_laplacian_system(num_verts=int(verts.shape[0]), edge_vertex_indices=edge_vertex_indices, weights=weights, constraint_mask=constraint_mask, lambda_c=lambda_c, square_laplacian=False)
│   └── for iter_idx in range(max_iters)
│       ├── calls estimate_rotations(verts=verts, edge_vertex_indices=edge_vertex_indices, weights=weights, reference_edge_vectors=reference_edge_vectors)
│       ├── calls build_arap_rhs(rotations=rotations, reference_edge_vectors=reference_edge_vectors, edge_vertex_indices=edge_vertex_indices, weights=weights, constraint_mask=constraint_mask, targets=targets, lambda_c=lambda_c)
│       ├── calls solve_factorized_sparse_system(factorized_system=factorized_system, rhs=rhs, device=verts.device, dtype=verts.dtype)
│       ├── if report_set and iterations_run in report_set
│       │   └── impls progress[iterations_run] = verts.detach().clone()  # capture this iteration's verts for reporting
│       └── if early_stop_patience is not None
│           ├── calls compute_arap_energy(verts=verts, edge_vertex_indices=edge_vertex_indices, weights=weights, reference_edge_vectors=reference_edge_vectors, rotations=rotations, constraint_mask=constraint_mask, targets=targets, lambda_c=lambda_c)
│           ├── if best_energy is None or energy < best_energy
│           │   └── impls best_energy = energy.detach(); stale_iters = 0
│           └── else
│               ├── impls stale_iters += 1
│               └── if stale_iters >= early_stop_patience
│                   └── break
├── def estimate_rotations(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, reference_edge_vectors: torch.Tensor) -> torch.Tensor
│   ├── # Solves the per-vertex best-fit rotation (local step) via weighted edge-covariance SVD with a reflection fix.
│   ├── impls edge_vec = verts at each edge's first endpoint less verts at its second
│   ├── impls outer = the outer product of edge_vec with reference_edge_vectors
│   ├── impls weighted_outer = weights times outer
│   ├── impls cov = a zeroed [V, 3, 3] tensor
│   ├── impls index-add weighted_outer into cov at each edge's first endpoint
│   ├── impls index-add weighted_outer into cov at each edge's second endpoint  # the same term at both ends, the edge being undirected
│   ├── impls u, _, v = the batched SVD of cov
│   ├── impls signs = ones per vertex, its last entry set to the sign of det(u @ v)  # the reflection fix
│   ├── impls rotations = u @ diag_embed(signs) @ v
│   └── return rotations  # [V, 3, 3], one proper rotation per vertex
├── def build_arap_rhs(rotations: torch.Tensor, reference_edge_vectors: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, constraint_mask: torch.Tensor, targets: torch.Tensor, lambda_c: float) -> torch.Tensor
│   ├── # Assembles the global-step right-hand side from averaged per-edge rotations plus the soft-constraint target term.
│   ├── impls edge_rotations = the mean of each edge's two endpoint rotations
│   ├── impls edge_terms = weights * edge_rotations applied to reference_edge_vectors
│   ├── impls rhs = each vertex's accumulation of edge_terms over its incident edges, signed by edge direction
│   ├── impls add lambda_c * targets into rhs at the vertices constraint_mask selects
│   └── return rhs  # [V, 3], the global step's right-hand side
├── def apply_arap_operator(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, constraint_mask: torch.Tensor, lambda_c: float) -> torch.Tensor
│   ├── # Applies the weighted-Laplacian-plus-constraint operator to verts without forming the sparse matrix.
│   ├── impls weighted = weights times verts at each edge's first endpoint less verts at its second
│   ├── impls result = a zeroed tensor of verts' shape
│   ├── impls index-add weighted into result at each edge's first endpoint
│   ├── impls index-add its negation into result at each edge's second endpoint
│   ├── impls result = result plus lambda_c times constraint_mask times verts
│   └── return result  # [V, 3]
└── def compute_arap_energy(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, reference_edge_vectors: torch.Tensor, rotations: torch.Tensor, constraint_mask: torch.Tensor, targets: torch.Tensor, lambda_c: float) -> torch.Tensor
    ├── # Computes total ARAP energy as the sum of the weighted edge-residual term and the constraint term.
    ├── impls current_edge_vectors = verts differenced across edge_vertex_indices
    ├── impls rotated_reference = each edge's mean endpoint rotation applied to reference_edge_vectors
    ├── impls edge_residual = the weighted squared norm of current_edge_vectors - rotated_reference, summed over edges
    ├── impls constraint_residual = lambda_c * the summed squared norm of verts - targets at the vertices constraint_mask selects
    └── return edge_residual + constraint_residual  # the scalar total ARAP energy
```

`models/three_d/meshes/ops/laplacian.py`

```text
laplacian.py
├── def build_cotangent_laplacian(base_verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Builds the coalesced cotangent-weighted edge graph and per-vertex weight sums for base_verts/faces.
│   ├── calls cotangent(v1, v2, v0)
│   ├── calls cotangent(v0, v2, v1)
│   └── calls cotangent(v0, v1, v2)
├── def compute_cotangent_weights_for_edges(base_verts: torch.Tensor, faces: torch.Tensor, edges: torch.Tensor) -> torch.Tensor
│   ├── # Computes the cotangent weight for each given edge by matching it against the triangles' half-cotangents.
│   ├── calls cotangent(v1, v2, v0)
│   ├── calls cotangent(v0, v2, v1)
│   └── calls cotangent(v0, v1, v2)
├── def cotangent(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor
│   ├── # Returns the cotangent of the angle at vertex a of triangle (a, b, c).
│   ├── impls u = b - a
│   ├── impls v = c - a
│   ├── impls sine_term = the norm of the cross product of u with v
│   └── return the dot product of u with v over sine_term  # cos over sin at a
├── def laplacian_apply(verts: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor
│   ├── # Applies the weighted graph Laplacian to verts as the per-vertex sum of weighted incident edge differences.
│   ├── impls edge_differences = verts at each edge's first endpoint minus verts at its second
│   ├── impls contributions = weights broadcast over edge_differences
│   ├── impls result = a zeros tensor of verts' shape
│   ├── impls scatter-add contributions into result at each edge's first endpoint
│   ├── impls scatter-subtract contributions from result at each edge's second endpoint
│   └── return result  # [V, 3]
├── def build_neighbor_data(edges: torch.Tensor, weights: torch.Tensor, base_verts: torch.Tensor, num_verts: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
│   ├── # Groups undirected edges into per-vertex neighbor index, weight, and reference-edge-vector lists.
│   ├── impls directed_sources = each edge's two endpoints, each taken in turn as the source
│   ├── impls directed_targets = the matching other endpoint of each directed edge
│   ├── impls directed_weights = each edge's weight, repeated for both directions
│   ├── impls sort the three directed arrays by directed_sources, stably
│   ├── impls neighbor_counts = the directed-edge count per vertex
│   ├── impls assert neighbor_counts sums to the directed-edge count
│   ├── impls split_indices = the cumulative neighbor_counts, without its last entry
│   ├── impls neighbors = directed_targets split at split_indices
│   ├── impls neighbor_weights = directed_weights split at split_indices
│   ├── impls directed_reference_edge_vectors = base_verts at directed_sources less base_verts at directed_targets
│   ├── impls neighbor_reference_edge_vectors = directed_reference_edge_vectors split at split_indices
│   ├── impls assert each of the three lists carries one tensor per vertex
│   └── return neighbors, neighbor_weights, neighbor_reference_edge_vectors  # one tensor per vertex in each list
├── def geodesic_distances(num_verts: int, edges: torch.Tensor, lengths: torch.Tensor, source: int) -> torch.Tensor
│   ├── # Computes single-source shortest-path (Dijkstra) distances over the weighted edge graph from source.
│   ├── calls build_adjacency(num_verts, edges, lengths)
│   └── while heap
│       ├── if visited[u]
│       │   └── continue
│       └── for each (v, weight) in adjacency[u]
│           └── if new_dist < distances[v]
│               └── impls distances[v] = new_dist; push (new_dist, v) onto heap
└── def build_adjacency(num_verts: int, edges: torch.Tensor, lengths: torch.Tensor) -> List[List[Tuple[int, float]]]
    ├── # Builds an undirected adjacency list mapping each vertex to its (neighbor, edge-length) pairs.
    └── for each edge index
        ├── impls append (j, length) to adjacency[i]
        └── impls append (i, length) to adjacency[j]
```

`models/three_d/meshes/ops/linear_system.py`

```text
linear_system.py
├── def factorize_laplacian_system(num_verts: int, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, constraint_mask: torch.Tensor, lambda_c: float, square_laplacian: bool) -> Any
│   ├── # Assembles and LU-factorizes the (optionally squared) weighted-Laplacian-plus-constraint system matrix.
│   ├── calls build_weighted_laplacian_sparse_matrix(num_verts=num_verts, edge_vertex_indices=edge_vertex_indices, weights=weights)
│   ├── if square_laplacian
│   │   └── impls operator_matrix = laplacian_matrix @ laplacian_matrix
│   ├── calls build_constraint_diagonal_sparse_matrix(constraint_mask=constraint_mask, lambda_c=lambda_c)
│   └── calls factorize_sparse_system_matrix(system_matrix=system_matrix)
├── def build_weighted_laplacian_sparse_matrix(num_verts: int, edge_vertex_indices: torch.Tensor, weights: torch.Tensor) -> sparse.csc_matrix
│   ├── # Builds the symmetric weighted graph-Laplacian as a scipy CSC matrix from edges and edge weights.
│   ├── impls off_diagonal_entries = -weights at each edge's endpoint pair, in both directions  # the symmetric off-diagonal
│   ├── impls diagonal_entries = each vertex's summed incident weights, on the diagonal
│   ├── impls matrix = a scipy COO matrix of num_verts square over both entry sets
│   └── return matrix converted to CSC  # [num_verts, num_verts]
├── def build_constraint_diagonal_sparse_matrix(constraint_mask: torch.Tensor, lambda_c: float) -> sparse.csc_matrix
│   ├── # Builds the lambda_c-scaled diagonal soft-constraint matrix as a scipy CSC matrix.
│   ├── impls diagonal = constraint_mask as a float vector scaled by lambda_c
│   └── return a scipy CSC diagonal matrix over diagonal  # [V, V]
├── def factorize_sparse_system_matrix(system_matrix: sparse.csc_matrix) -> Any
│   ├── # LU-factorizes a square sparse system matrix via scipy splu.
│   └── return the scipy splu factorization of system_matrix
└── def solve_factorized_sparse_system(factorized_system: Any, rhs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor
    ├── # Solves the factorized system column-by-column for a multi-column torch rhs and returns a torch tensor.
    ├── impls rhs_array = rhs detached onto the cpu as a float64 array
    ├── impls columns = an empty list
    ├── for each column of rhs_array
    │   └── impls append the factorized system's solve of that column to columns
    ├── impls solution = columns stacked back into rhs's column layout
    └── return solution as a tensor of dtype on device  # rhs's own shape
```

`models/three_d/meshes/ops/normals.py`

```text
normals.py
├── def compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor, weights: str) -> torch.Tensor
│   ├── # Computes per-vertex normals from incident face normals under the requested face-normal weighting scheme.
│   ├── if weights == "area"
│   │   ├── calls _compute_vertex_normals_area_weighted(verts=verts, faces=faces)
│   │   └── return
│   ├── if weights == "unit"
│   │   ├── calls _compute_vertex_normals_unit_weighted(verts=verts, faces=faces)
│   │   └── return
│   └── assert False  # unreachable: weights is "area" or "unit"
├── def _compute_vertex_normals_area_weighted(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
│   ├── # Single-mesh per-vertex normals as the L2-normalized sum of UN-normalized (area-weighted) incident face normals.
│   ├── impls face_normals = the cross product of each face's two edge vectors, left un-normalized  # its magnitude is twice the face area, which is the weighting
│   ├── impls normals = a zeros tensor of verts' shape
│   ├── impls index-add face_normals into normals at each of the face's three corners
│   └── return normals L2-normalized per row  # [V, 3]
└── def _compute_vertex_normals_unit_weighted(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
    ├── # Per-vertex normals from UNIT-weighted (face-uniform) face normals, batched or single, value-identical to Deep3DFaceRecon compute_norm.
    ├── if not is_batched
    │   └── impls verts = verts.unsqueeze(0)
    ├── for b in range(num_batch)
    │   └── impls normals[b].index_add_ each of faces[:, 0/1/2] with face_normals[b]
    └── if not is_batched
        └── impls normals = normals.squeeze(0)
```

`models/three_d/meshes/ops/topology.py`

```text
topology.py
└── def build_topology_edges_from_faces(faces: torch.Tensor) -> torch.Tensor
    ├── # Extracts the sorted, unique set of undirected edges from triangle faces.
    ├── impls edges = each face's three corner pairs, stacked into one [3F, 2] index tensor
    ├── impls sort each row so the smaller vertex index comes first  # one canonical direction per undirected edge
    └── return the unique rows of edges, in sorted order             # [E, 2]
```

`models/three_d/meshes/ops/world_to_camera_transform.py`

```text
world_to_camera_transform.py
├── from typing import Optional
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from models.three_d.meshes.ops.apply_transform import apply_transform
└── def world_to_camera_transform(mesh: Mesh, extrinsics: torch.Tensor, max_divide: int = 0, num_divide: Optional[int] = None) -> Mesh
    ├── # High-level API mapping a mesh's verts from world into the camera frame: builds the world-to-camera 4x4 matrix from the inverse camera-to-world extrinsic and applies it via apply_transform.
    ├── impls invert the camera-to-world extrinsics into the world-to-camera 4x4 matrix
    ├── calls apply_transform  # the mesh by the world-to-camera matrix, passing max_divide and num_divide
    └── return  # the camera-frame Mesh
```
