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
    └── return  # the transformed Mesh
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
│   └── # Solves the per-vertex best-fit rotation (local step) via weighted edge-covariance SVD with a reflection fix.
├── def build_arap_rhs(rotations: torch.Tensor, reference_edge_vectors: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, constraint_mask: torch.Tensor, targets: torch.Tensor, lambda_c: float) -> torch.Tensor
│   └── # Assembles the global-step right-hand side from averaged per-edge rotations plus the soft-constraint target term.
├── def apply_arap_operator(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, constraint_mask: torch.Tensor, lambda_c: float) -> torch.Tensor
│   └── # Applies the weighted-Laplacian-plus-constraint operator to verts without forming the sparse matrix.
└── def compute_arap_energy(verts: torch.Tensor, edge_vertex_indices: torch.Tensor, weights: torch.Tensor, reference_edge_vectors: torch.Tensor, rotations: torch.Tensor, constraint_mask: torch.Tensor, targets: torch.Tensor, lambda_c: float) -> torch.Tensor
    └── # Computes total ARAP energy as the sum of the weighted edge-residual term and the constraint term.
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
│   └── # Returns the cotangent of the angle at vertex a of triangle (a, b, c).
├── def laplacian_apply(verts: torch.Tensor, edges: torch.Tensor, weights: torch.Tensor) -> torch.Tensor
│   └── # Applies the weighted graph Laplacian to verts as the per-vertex sum of weighted incident edge differences.
├── def build_neighbor_data(edges: torch.Tensor, weights: torch.Tensor, base_verts: torch.Tensor, num_verts: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]
│   └── # Groups undirected edges into per-vertex neighbor index, weight, and reference-edge-vector lists.
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
│   └── # Builds the symmetric weighted graph-Laplacian as a scipy CSC matrix from edges and edge weights.
├── def build_constraint_diagonal_sparse_matrix(constraint_mask: torch.Tensor, lambda_c: float) -> sparse.csc_matrix
│   └── # Builds the lambda_c-scaled diagonal soft-constraint matrix as a scipy CSC matrix.
├── def factorize_sparse_system_matrix(system_matrix: sparse.csc_matrix) -> Any
│   └── # LU-factorizes a square sparse system matrix via scipy splu.
└── def solve_factorized_sparse_system(factorized_system: Any, rhs: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor
    └── # Solves the factorized system column-by-column for a multi-column torch rhs and returns a torch tensor.
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
│   └── # Single-mesh per-vertex normals as the L2-normalized sum of UN-normalized (area-weighted) incident face normals.
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
    └── # Extracts the sorted, unique set of undirected edges from triangle faces.
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
