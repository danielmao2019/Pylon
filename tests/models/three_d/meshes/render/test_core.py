"""Test cases for soft-silhouette rendering from triangle meshes."""

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.render.core import render_soft_silhouette_from_mesh


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_soft_silhouette_shape_and_range() -> None:
    """render_soft_silhouette_from_mesh returns an [H, W] float tensor with values in [0, 1].

    Returns:
        None.
    """
    camera = Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0},
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
    mesh = Mesh(
        verts=torch.tensor(
            [[-0.5, -0.5, -2.0], [0.5, -0.5, -2.0], [0.0, 0.5, -2.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )

    silhouette = render_soft_silhouette_from_mesh(
        mesh=mesh,
        camera=camera,
        resolution=(64, 64),
        blur_sigma=1e-4,
    )

    assert silhouette.shape == (64, 64), f"{silhouette.shape=}"
    assert silhouette.dtype == torch.float32, f"{silhouette.dtype=}"
    assert silhouette.min() >= 0.0, f"{silhouette.min()=}"
    assert silhouette.max() <= 1.0, f"{silhouette.max()=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_soft_silhouette_is_differentiable() -> None:
    """The soft silhouette backpropagates non-zero gradients to the mesh vertices.

    Returns:
        None.
    """
    camera = Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0},
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
    verts = torch.tensor(
        [[-0.5, -0.5, -2.0], [0.5, -0.5, -2.0], [0.0, 0.5, -2.0]],
        dtype=torch.float32,
        device=torch.device("cuda"),
        requires_grad=True,
    )
    mesh = Mesh(
        verts=verts,
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64, device=torch.device("cuda")),
    )

    silhouette = render_soft_silhouette_from_mesh(
        mesh=mesh,
        camera=camera,
        resolution=(64, 64),
        blur_sigma=1e-4,
    )
    silhouette.sum().backward()

    assert verts.grad is not None, f"{verts.grad=}"
    assert verts.grad.abs().sum() > 0.0, f"{verts.grad=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA.")
def test_blur_sigma_changes_the_silhouette() -> None:
    """The same mesh rendered under two blur sigmas yields different soft silhouettes.

    Returns:
        None.
    """
    camera = Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0},
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            convention="opengl",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
    mesh = Mesh(
        verts=torch.tensor(
            [[-0.5, -0.5, -2.0], [0.5, -0.5, -2.0], [0.0, 0.5, -2.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )

    silhouette_sharp = render_soft_silhouette_from_mesh(
        mesh=mesh,
        camera=camera,
        resolution=(64, 64),
        blur_sigma=1e-5,
    )
    silhouette_blurry = render_soft_silhouette_from_mesh(
        mesh=mesh,
        camera=camera,
        resolution=(64, 64),
        blur_sigma=1e-3,
    )

    assert not torch.allclose(silhouette_sharp, silhouette_blurry), (
        "Expected different blur sigmas to change the soft silhouette. "
        f"{(silhouette_sharp - silhouette_blurry).abs().max()=}"
    )
