from typing import Optional

import torch

from data.structures.three_d.mesh.mesh import Mesh
from utils.ops.chunked_matmul import chunked_matmul


def apply_transform(
    mesh: Mesh,
    transform: torch.Tensor,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> Mesh:
    """Map a mesh's verts through a 4x4 transform in homogeneous coordinates.

    The verts are lifted to homogeneous `[V, 4]` coordinates, multiplied by
    `transform.T` on the right via the chunked large-by-small matmul, then the
    homogeneous coordinate is dropped to recover `[V, 3]`. Faces and texture are
    carried over unchanged.

    Args:
        mesh: Source `Mesh`; its `verts` is a float `[V, 3]` tensor.
        transform: Transform tensor of shape `[4, 4]` with the same dtype and
            device as `mesh.verts`.
        max_divide: Maximum number of times the chunked matmul halves the vert
            rows on CUDA OOM.
        num_divide: When not `None`, fixes the chunked matmul to a single pass
            split into `2 ** num_divide` batches of the vert rows.

    Returns:
        A new `Mesh` whose verts are the transformed `[V, 3]` positions, with
        the original faces and original texture.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(transform, torch.Tensor), (
            "Expected `transform` to be a `torch.Tensor`. " f"{type(transform)=}"
        )
        assert transform.shape == (4, 4), (
            "Expected `transform` to be a `[4, 4]` matrix. " f"{transform.shape=}"
        )

    _validate_inputs()

    ones_column = torch.ones(
        (mesh.verts.shape[0], 1),
        dtype=mesh.verts.dtype,
        device=mesh.verts.device,
    )
    homogeneous_verts = torch.cat([mesh.verts, ones_column], dim=1)

    transformed = chunked_matmul(
        large=homogeneous_verts,
        small=transform.T,
        max_divide=max_divide,
        num_divide=num_divide,
    )

    transformed_verts = transformed[:, :3]

    return Mesh(
        verts=transformed_verts,
        faces=mesh.faces,
        texture=mesh.texture,
    )
