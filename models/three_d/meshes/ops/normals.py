"""
Mesh normal computation utilities.
"""

import torch
import torch.nn.functional as F


def compute_vertex_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
    weights: str,
) -> torch.Tensor:
    """
    Compute per-vertex normals from incident face normals under the requested
    face-normal weighting scheme.

    Args:
        verts: Vertex positions. For `weights="area"`, shape `[V, 3]` (single
            mesh); for `weights="unit"`, shape `[N, V, 3]` (batched) or `[V, 3]`
            (single). Float dtype.
        faces: Triangle vertex indices into `verts`, shape `[F, 3]`, integer
            dtype, with values in `[0, V)`.
        weights: Face-normal weighting scheme, one of `"area"` (area-weighted)
            or `"unit"` (unit/face-uniform).

    Returns:
        Per-vertex unit normals, dtype equal to `verts.dtype`. Layout matches the
        chosen weighting helper: `[V, 3]` for `weights="area"`, and the input
        layout (`[N, V, 3]` or `[V, 3]`) for `weights="unit"`.
    """
    if weights == "area":
        return _compute_vertex_normals_area_weighted(verts=verts, faces=faces)
    if weights == "unit":
        return _compute_vertex_normals_unit_weighted(verts=verts, faces=faces)
    assert False, f"Unrecognized weights, expected 'area' or 'unit': {weights=}"


def _compute_vertex_normals_area_weighted(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Compute single-mesh per-vertex normals as the L2-normalized sum of
    UN-normalized (area-weighted) incident face normals.

    Args:
        verts: Vertex positions, shape `[V, 3]`, float dtype.
        faces: Triangle vertex indices into `verts`, shape `[F, 3]`, integer
            dtype, with values in `[0, V)`.

    Returns:
        Per-vertex unit normals, shape `[V, 3]`, dtype equal to `verts.dtype`.
    """
    # Input validations
    assert isinstance(verts, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    assert verts.ndim == 2
    assert verts.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert int(faces.min().item()) >= 0
    assert int(faces.max().item()) < verts.shape[0]

    # Input normalizations
    faces = faces.to(device=verts.device)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.zeros_like(verts)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    assert torch.all(normal_norm > 0)
    normals = normals / normal_norm
    return normals


def _compute_vertex_normals_unit_weighted(
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-vertex normals using UNIT-weighted (face-uniform) face normals.

    Each face contributes its unit-length normal (NOT scaled by face area) to its
    three vertices; per-vertex normals are the L2-normalized sum of the unit
    normals of all incident faces. This reproduces the Deep3DFaceRecon
    `ParametricFaceModel.compute_norm` value-identically: summing unit face
    normals over the `point_buf` adjacency equals accumulating each face's unit
    normal onto its three vertices via `index_add` over `faces`, and
    `cross(v1 - v2, v2 - v3)` has the same orientation as `cross(v1 - v0, v2 - v0)`.

    Args:
        verts: Vertex positions, shape `[N, V, 3]` (batched) or `[V, 3]` (single),
            dtype float. The leading batch dimension, when present, is preserved.
        faces: Triangle vertex indices into `verts`, shape `[F, 3]`, integer dtype,
            with values in `[0, V)`.

    Returns:
        Per-vertex unit normals matching the input layout: shape `[N, V, 3]` when
        `verts` is batched, or `[V, 3]` when `verts` is a single mesh; dtype equal
        to `verts.dtype`.
    """
    # Input validations
    assert isinstance(verts, torch.Tensor), f"{type(verts)=}"
    assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
    assert verts.ndim in (2, 3), f"{verts.shape=}"
    assert verts.shape[-1] == 3, f"{verts.shape=}"
    assert faces.ndim == 2, f"{faces.shape=}"
    assert faces.shape[1] == 3, f"{faces.shape=}"
    assert verts.shape[-2] > 0, f"{verts.shape=}"
    assert faces.shape[0] > 0, f"{faces.shape=}"
    assert int(faces.min().item()) >= 0, f"{int(faces.min().item())=}"
    assert (
        int(faces.max().item()) < verts.shape[-2]
    ), f"{int(faces.max().item())=}, {verts.shape=}"

    # Input normalizations
    is_batched = verts.ndim == 3
    if not is_batched:
        verts = verts.unsqueeze(0)
    faces = faces.to(device=verts.device)

    num_batch = verts.shape[0]
    num_verts = verts.shape[1]
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]
    face_normals = torch.cross(v0 - v1, v1 - v2, dim=-1)
    face_normals = F.normalize(face_normals, dim=-1, p=2)

    normals = torch.zeros(
        size=(num_batch, num_verts, 3),
        dtype=verts.dtype,
        device=verts.device,
    )
    for b in range(num_batch):
        normals[b].index_add_(0, faces[:, 0], face_normals[b])
        normals[b].index_add_(0, faces[:, 1], face_normals[b])
        normals[b].index_add_(0, faces[:, 2], face_normals[b])
    normals = F.normalize(normals, dim=-1, p=2)

    if not is_batched:
        normals = normals.squeeze(0)
    return normals
