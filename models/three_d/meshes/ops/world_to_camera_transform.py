from typing import Optional

import torch

from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.ops.apply_transform import apply_transform


def world_to_camera_transform(
    mesh: Mesh,
    extrinsics: torch.Tensor,
    max_divide: int = 0,
    num_divide: Optional[int] = None,
) -> Mesh:
    """Map a mesh's verts from world coordinates into the camera frame.

    The camera-to-world `extrinsics` is inverted to the world-to-camera 4x4
    matrix, which is then applied to the mesh via `apply_transform`.

    Args:
        mesh: Source `Mesh` whose verts live in world coordinates; its `verts`
            is a float `[V, 3]` tensor.
        extrinsics: Camera-to-world (pose) extrinsic matrix of shape `[4, 4]` in
            the OpenCV convention, on the same dtype and device as `mesh.verts`.
        max_divide: Maximum number of times the chunked matmul halves the vert
            rows on CUDA OOM.
        num_divide: When not `None`, fixes the chunked matmul to a single pass
            split into `2 ** num_divide` batches of the vert rows.

    Returns:
        A new `Mesh` whose verts are expressed in the camera frame, with the
        original faces and original texture.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(extrinsics, torch.Tensor), (
            "Expected `extrinsics` to be a `torch.Tensor`. " f"{type(extrinsics)=}"
        )
        assert extrinsics.shape == (4, 4), (
            "Expected `extrinsics` to be a `[4, 4]` matrix. " f"{extrinsics.shape=}"
        )

    _validate_inputs()

    world_to_camera = torch.inverse(extrinsics)

    return apply_transform(
        mesh=mesh,
        transform=world_to_camera,
        max_divide=max_divide,
        num_divide=num_divide,
    )
