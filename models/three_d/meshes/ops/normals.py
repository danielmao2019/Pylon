"""
Mesh normal computation utilities.
"""

import torch


def compute_vertex_normals(
    base_vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    # Input validations
    assert isinstance(base_vertices, torch.Tensor)
    assert isinstance(faces, torch.Tensor)
    assert base_vertices.ndim == 2
    assert base_vertices.shape[1] == 3
    assert faces.ndim == 2
    assert faces.shape[1] == 3
    assert base_vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert int(faces.min().item()) >= 0
    assert int(faces.max().item()) < base_vertices.shape[0]

    # Input normalizations
    faces = faces.to(device=base_vertices.device)

    v0 = base_vertices[faces[:, 0]]
    v1 = base_vertices[faces[:, 1]]
    v2 = base_vertices[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = torch.zeros_like(base_vertices)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
    assert torch.all(normal_norm > 0)
    normals = normals / normal_norm
    return normals
