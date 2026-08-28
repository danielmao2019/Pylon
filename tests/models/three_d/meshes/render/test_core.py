"""Tests for triangle-mesh rendering."""

from math import log
from typing import Any, List, Tuple

import pytest
import torch
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras

import models.three_d.meshes.render.core as render_core
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import (
    CameraExtrinsics,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from models.three_d.meshes.render.core import (
    _prepare_cameras,
    render_rgb_from_mesh,
    render_soft_mask_from_mesh,
)

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch3D mesh rendering requires CUDA.",
)


def _build_camera(
    model: str = "pinhole",
    resolution: Tuple[int, int] = (32, 32),
    device: torch.device = torch.device("cpu"),
    focal_requires_grad: bool = False,
) -> Tuple[Camera, torch.Tensor]:
    """Build a standard-frame test camera.

    Args:
        model: Camera model, either `pinhole` or `ortho`.
        resolution: Camera intrinsics resolution `(height, width)`.
        device: Device for the camera tensors.
        focal_requires_grad: Whether the horizontal focal parameter tracks gradients.

    Returns:
        The camera and its horizontal focal tensor.
    """
    height, width = resolution
    focal = torch.tensor(
        data=float(min(height, width)) * 0.8,
        dtype=torch.float32,
        device=device,
        requires_grad=focal_requires_grad,
    )
    params = {
        "fx": focal,
        "fy": torch.tensor(
            float(min(height, width)) * 0.8,
            dtype=torch.float32,
            device=device,
        ),
        "cx": torch.tensor(float(width) / 2.0, dtype=torch.float32, device=device),
        "cy": torch.tensor(float(height) / 2.0, dtype=torch.float32, device=device),
        "h": height,
        "w": width,
    }
    return (
        Camera(
            intrinsics=build_camera_intrinsics(
                model=model,
                params=params,
                intr_convention="standard",
                device=device,
            ),
            extrinsics=CameraExtrinsics(
                extrinsics=torch.eye(4, dtype=torch.float32, device=device),
                extr_convention="standard",
                device=device,
            ),
            device=device,
        ),
        focal,
    )


def _build_mesh(
    view_depth: float = 2.0,
    device: torch.device = torch.device("cpu"),
    reverse_winding: bool = False,
    requires_grad: bool = False,
) -> Mesh:
    """Build a small textured triangle in the standard camera frame.

    Args:
        view_depth: Standard-frame forward coordinate shared by the triangle vertices.
        device: Device for mesh tensors.
        reverse_winding: Whether to reverse the only face's winding.
        requires_grad: Whether mesh vertices track gradients.

    Returns:
        A one-face Mesh with black vertex colors.
    """
    verts = torch.tensor(
        [
            [-0.35, view_depth, -0.35],
            [0.35, view_depth, -0.35],
            [0.0, view_depth, 0.35],
        ],
        dtype=torch.float32,
        device=device,
    )
    if requires_grad:
        verts.requires_grad_()
    face = [0, 2, 1] if reverse_winding else [0, 1, 2]
    faces = torch.tensor([face], dtype=torch.int64, device=device)
    texture = MeshTextureVertexColor(
        vertex_color=torch.zeros((3, 3), dtype=torch.float32, device=device)
    )
    return Mesh(verts=verts, faces=faces, texture=texture)


@_REQUIRES_CUDA
def test_the_mask_is_a_continuous_coverage_the_silhouette_carries_a_gradient_through() -> (
    None
):
    """The silhouette mask carries fractional coverage on triangle boundaries.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="ortho")
    mesh = _build_mesh(view_depth=2.0)
    resolution = (32, 32)

    mask = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=resolution,
    )

    assert torch.all((0.0 <= mask) & (mask <= 1.0)), (
        "Expected all silhouette coverage values to lie in [0, 1]. "
        f"{mask.min()=} {mask.max()=}"
    )
    assert mask.shape == resolution, (
        "Expected the mask shape to match the requested resolution. "
        f"{mask.shape=} {resolution=}"
    )
    assert torch.any((0.0 < mask) & (mask < 1.0)), (
        "Expected fractional soft coverage along the mesh outline. " f"{mask=}"
    )


@_REQUIRES_CUDA
def test_the_mask_is_differentiable_in_the_mesh_and_in_the_camera() -> None:
    """The mask sum backpropagates through mesh vertices and camera parameters.

    Args:
        None.

    Returns:
        None.
    """
    device = torch.device("cuda:0")
    camera, focal = _build_camera(
        model="ortho",
        resolution=(32, 32),
        device=device,
        focal_requires_grad=True,
    )
    mesh = _build_mesh(view_depth=2.0, device=device, requires_grad=True)

    mask = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )
    mask.sum().backward()

    assert mesh.verts.grad is not None and float(mesh.verts.grad.abs().sum()) > 0.0, (
        "Expected the silhouette mask to backpropagate into mesh vertices. "
        f"{mesh.verts.grad=}"
    )
    assert focal.grad is not None and float(focal.grad.abs().sum()) > 0.0, (
        "Expected the silhouette mask to backpropagate into camera intrinsics. "
        f"{focal.grad=}"
    )


@_REQUIRES_CUDA
def test_the_blur_ends_exactly_at_the_configured_coverage_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The render derives its blur radius from sigma and coverage threshold.

    Args:
        monkeypatch: Pytest monkeypatch fixture for recording rasterizer arguments.

    Returns:
        None.
    """
    recorded_blur_radii: List[float] = []
    original_build_rasterizer = render_core._build_rasterizer

    def recording_build_rasterizer(**kwargs: Any) -> object:
        """Record each blur radius before building the real rasterizer.

        Args:
            kwargs: Keyword arguments forwarded to `_build_rasterizer`.

        Returns:
            The real PyTorch3D rasterizer.
        """
        recorded_blur_radii.append(kwargs["blur_radius"])
        return original_build_rasterizer(**kwargs)

    monkeypatch.setattr(render_core, "_build_rasterizer", recording_build_rasterizer)

    camera, _ = _build_camera(model="ortho")
    mesh = _build_mesh(view_depth=2.0)
    for coverage_threshold in (0.25, 1.0e-04):
        render_core.render_soft_mask_from_mesh(
            mesh=mesh,
            camera=camera,
            blend_sigma=1.0e-02,
            blend_gamma=1.0e-04,
            faces_per_pixel=20,
            coverage_threshold=coverage_threshold,
            resolution=(32, 32),
        )

    assert recorded_blur_radii[0] == pytest.approx(log(1 / 0.25 - 1) * 1.0e-02), (
        "Expected the blur radius to match the configured threshold formula. "
        f"{recorded_blur_radii=}"
    )
    assert recorded_blur_radii[1] == pytest.approx(log(1 / 1.0e-04 - 1) * 1.0e-02), (
        "Expected the blur radius to match the configured threshold formula. "
        f"{recorded_blur_radii=}"
    )
    assert recorded_blur_radii[1] > recorded_blur_radii[0], (
        "Expected a smaller coverage threshold to push the blur edge outward. "
        f"{recorded_blur_radii=}"
    )


@_REQUIRES_CUDA
def test_the_render_reproduces_itself_between_runs() -> None:
    """Repeated renders of the same inputs are bitwise stable.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="ortho")
    mesh = _build_mesh(view_depth=2.0)

    first = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )
    second = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )

    assert torch.equal(first, second), (
        "Expected the naive rasterizer settings to reproduce the same mask. "
        f"{first=} {second=}"
    )


@_REQUIRES_CUDA
def test_what_the_mesh_occupies_is_what_the_render_covers() -> None:
    """Both face windings cover the same pixels when backface culling is disabled.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="ortho")
    front_mesh = _build_mesh(view_depth=2.0, reverse_winding=False)
    back_mesh = _build_mesh(view_depth=2.0, reverse_winding=True)

    front_mask = render_soft_mask_from_mesh(
        mesh=front_mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )
    back_mask = render_soft_mask_from_mesh(
        mesh=back_mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )

    assert torch.allclose(
        input=front_mask,
        other=back_mask,
        atol=1.0e-06,
        rtol=0.0,
    ), (
        "Expected winding reversal to preserve coverage when culling is disabled. "
        f"{front_mask=} {back_mask=}"
    )


@_REQUIRES_CUDA
def test_the_rgb_renders_mask_uses_rasterization_face_indices() -> None:
    """The returned RGB mask is driven by face indices, not shaded color.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="pinhole")
    mesh = _build_mesh(view_depth=2.0)

    rgb, valid_mask = render_rgb_from_mesh(
        mesh=mesh,
        camera=camera,
        background=(0, 0, 0),
        return_mask=True,
        resolution=(32, 32),
    )

    assert rgb.shape == (3, 32, 32), (
        "Expected RGB render output to have channel-first render resolution. "
        f"{rgb.shape=}"
    )
    assert valid_mask.dtype == torch.bool, (
        "Expected the returned render mask to be boolean. " f"{valid_mask.dtype=}"
    )
    assert int(valid_mask.sum().item()) > 0, (
        "Expected covered black pixels to stay covered in the validity mask. "
        f"{valid_mask=}"
    )
    assert int(valid_mask.sum().item()) < valid_mask.numel(), (
        "Expected background pixels to stay uncovered. " f"{valid_mask=}"
    )


def test_both_camera_models_build_their_own_pytorch3d_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinhole and orthographic repo cameras select distinct PyTorch3D cameras.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to simulate an unsupported
            already-constructed camera model.

    Returns:
        None.
    """
    pinhole_camera, _ = _build_camera(model="pinhole")
    ortho_camera, _ = _build_camera(model="ortho")

    pinhole = _prepare_cameras(
        camera=pinhole_camera,
        resolution=(32, 32),
        device=torch.device("cpu"),
    )
    ortho = _prepare_cameras(
        camera=ortho_camera,
        resolution=(32, 32),
        device=torch.device("cpu"),
    )

    assert isinstance(pinhole, PerspectiveCameras), (
        "Expected pinhole repo cameras to build PyTorch3D perspective cameras. "
        f"{type(pinhole)=}"
    )
    assert isinstance(ortho, OrthographicCameras), (
        "Expected orthographic repo cameras to build PyTorch3D orthographic cameras. "
        f"{type(ortho)=}"
    )

    unsupported_camera, _ = _build_camera(model="pinhole")
    unsupported_camera = unsupported_camera.to(
        device=torch.device("cpu"),
        dtype=torch.float32,
        intr_convention="pytorch3d",
        extr_convention="pytorch3d",
    ).scale_intrinsics(resolution=(32, 32))
    monkeypatch.setattr(type(unsupported_camera.intrinsics), "MODEL", "unsupported")
    monkeypatch.setattr(unsupported_camera, "to", lambda **_: unsupported_camera)
    monkeypatch.setattr(
        unsupported_camera,
        "scale_intrinsics",
        lambda **_: unsupported_camera,
    )

    with pytest.raises(AssertionError, match="Should not reach here"):
        _prepare_cameras(
            camera=unsupported_camera,
            resolution=(32, 32),
            device=torch.device("cpu"),
        )


def test_the_camera_reaches_pytorch3d_in_pytorch3ds_own_frames() -> None:
    """_prepare_cameras restates both camera frames for PyTorch3D.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="pinhole", resolution=(100, 150))
    device = torch.device("cpu")
    resolution = (200, 300)

    prepared = _prepare_cameras(
        camera=camera,
        resolution=resolution,
        device=device,
    )
    expected_camera = camera.to(
        device=device,
        dtype=torch.float32,
        intr_convention="pytorch3d",
        extr_convention="pytorch3d",
    ).scale_intrinsics(resolution=resolution)
    expected_intrinsics = expected_camera.intrinsics
    expected_extrinsics = expected_camera.extrinsics

    assert prepared.in_ndc(), (
        "Expected PyTorch3D camera params to be in NDC. " f"{prepared.in_ndc()=}"
    )
    assert torch.allclose(
        prepared.focal_length,
        torch.stack((expected_intrinsics.fx, expected_intrinsics.fy)).reshape(1, 2),
    ), (
        "Expected PyTorch3D focal lengths to match the converted intrinsics. "
        f"{prepared.focal_length=} {expected_intrinsics.fx=} {expected_intrinsics.fy=}"
    )
    assert torch.allclose(
        prepared.principal_point,
        torch.stack((expected_intrinsics.cx, expected_intrinsics.cy)).reshape(1, 2),
    ), (
        "Expected PyTorch3D principal point to match the converted intrinsics. "
        f"{prepared.principal_point=} {expected_intrinsics.cx=} "
        f"{expected_intrinsics.cy=}"
    )
    assert torch.allclose(
        prepared.R,
        expected_extrinsics.w2c[:3, :3].transpose(0, 1).unsqueeze(0),
    ), (
        "Expected PyTorch3D rotation to match the converted world-to-camera pose. "
        f"{prepared.R=} {expected_extrinsics.w2c=}"
    )
    assert torch.allclose(prepared.T, expected_extrinsics.w2c[:3, 3].unsqueeze(0)), (
        "Expected PyTorch3D translation to match the converted world-to-camera pose. "
        f"{prepared.T=} {expected_extrinsics.w2c=}"
    )


@_REQUIRES_CUDA
def test_a_render_uses_the_cameras_own_resolution_by_default() -> None:
    """The camera's intrinsic resolution is used when no raster is named.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="ortho", resolution=(24, 28))
    mesh = _build_mesh(view_depth=2.0)

    mask = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
    )
    rgb = render_rgb_from_mesh(mesh=mesh, camera=camera)

    assert mask.shape == (24, 28), (
        "Expected the soft mask to use the camera intrinsic resolution by default. "
        f"{mask.shape=}"
    )
    assert rgb.shape[-2:] == (24, 28), (
        "Expected the RGB render to use the camera intrinsic resolution by default. "
        f"{rgb.shape=}"
    )


@_REQUIRES_CUDA
def test_a_named_resolution_overrides_the_cameras_own() -> None:
    """A named render resolution overrides the camera's intrinsic resolution.

    Args:
        None.

    Returns:
        None.
    """
    camera, _ = _build_camera(model="ortho", resolution=(24, 28))
    mesh = _build_mesh(view_depth=2.0)

    mask = render_soft_mask_from_mesh(
        mesh=mesh,
        camera=camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(30, 34),
    )

    assert mask.shape == (30, 34), (
        "Expected the named render resolution to override the camera resolution. "
        f"{mask.shape=}"
    )


@_REQUIRES_CUDA
def test_the_camera_a_render_is_given_is_the_camera_it_renders_through() -> None:
    """The render uses the given camera's pose under both camera models.

    Args:
        None.

    Returns:
        None.
    """
    pinhole_camera, _ = _build_camera(model="pinhole")
    expected_pinhole = pinhole_camera.to(
        device=torch.device("cpu"),
        dtype=torch.float32,
        intr_convention="pytorch3d",
        extr_convention="pytorch3d",
    ).scale_intrinsics(resolution=(32, 32))
    prepared_pinhole = _prepare_cameras(
        camera=pinhole_camera,
        resolution=(32, 32),
        device=torch.device("cpu"),
    )
    assert torch.allclose(
        prepared_pinhole.T,
        expected_pinhole.extrinsics.w2c[:3, 3].unsqueeze(0),
    ), (
        "Expected pinhole translation to come from the camera's own pose. "
        f"{prepared_pinhole.T=} {expected_pinhole.extrinsics.w2c=}"
    )

    ortho_camera, _ = _build_camera(model="ortho")
    expected_ortho = ortho_camera.to(
        device=torch.device("cpu"),
        dtype=torch.float32,
        intr_convention="pytorch3d",
        extr_convention="pytorch3d",
    ).scale_intrinsics(resolution=(32, 32))
    prepared_ortho = _prepare_cameras(
        camera=ortho_camera,
        resolution=(32, 32),
        device=torch.device("cpu"),
    )
    assert torch.allclose(
        prepared_ortho.T,
        expected_ortho.extrinsics.w2c[:3, 3].unsqueeze(0),
    ), (
        "Expected orthographic translation to come from the camera's own pose. "
        f"{prepared_ortho.T=} {expected_ortho.extrinsics.w2c=}"
    )

    behind_mesh = _build_mesh(view_depth=-2.0)
    behind_mask = render_soft_mask_from_mesh(
        mesh=behind_mesh,
        camera=pinhole_camera,
        blend_sigma=1.0e-02,
        blend_gamma=1.0e-04,
        faces_per_pixel=20,
        coverage_threshold=1.0e-04,
        resolution=(32, 32),
    )
    assert float(behind_mask.max().item()) == 0.0, (
        "Expected a mesh behind the perspective camera to render empty coverage. "
        f"{behind_mask.max()=}"
    )
