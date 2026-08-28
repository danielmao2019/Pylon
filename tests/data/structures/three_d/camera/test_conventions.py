import inspect
from itertools import product
from typing import Dict, List, Union

import numpy as np
import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics import conventions
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_extrinsics,
    validate_extr_convention,
)
from data.structures.three_d.camera.intrinsics import conventions as intr_conventions
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.camera.intrinsics.conventions import (
    transform_intr_convention,
)
from data.structures.three_d.camera.intrinsics.scaling import rescale_intr_params
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_invariants,
    validate_intr_convention,
)

EXTR_CONVENTIONS: List[str] = [
    "standard",
    "opengl",
    "opencv",
    "pytorch3d",
    "arkit",
]


def _build_extrinsics_matrix() -> torch.Tensor:
    """Build a valid 4x4 cam2world matrix with a proper rotation.

    Args:
        None.

    Returns:
        A 4x4 float32 camera-to-world matrix whose 3x3 block is a proper rotation.
    """
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 1.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _build_extrinsics_matrices() -> List[torch.Tensor]:
    """Build distinct valid 4x4 cam2world matrices with proper rotations.

    Args:
        None.

    Returns:
        A list of 4x4 float32 camera-to-world matrices with distinct proper
        rotations and centers.
    """
    rotation_about_z = _build_extrinsics_matrix()
    identity_rotation = torch.eye(4, dtype=torch.float32)
    identity_rotation[:3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    rotation_about_x = torch.tensor(
        [
            [1.0, 0.0, 0.0, -0.7],
            [0.0, 0.0, -1.0, 0.4],
            [0.0, 1.0, 0.0, 2.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return [rotation_about_z, identity_rotation, rotation_about_x]


def _build_extrinsics(extr_convention: str) -> CameraExtrinsics:
    """Build a CameraExtrinsics fixture in the given pose frame.

    Args:
        extr_convention: Pose-frame convention string.

    Returns:
        A CameraExtrinsics on the CPU in the given pose frame.
    """
    return CameraExtrinsics(
        extrinsics=_build_extrinsics_matrix(),
        extr_convention=extr_convention,
        device="cpu",
    )


def _build_cameras(extr_convention: str) -> Cameras:
    """Build a multi-camera Cameras fixture in the given pose frame.

    Args:
        extr_convention: Pose-frame convention string.

    Returns:
        A Cameras of three CPU cameras with distinct poses in the given pose
        frame.
    """
    pose_matrices = _build_extrinsics_matrices()
    intrinsics = [
        build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            },
            intr_convention="standard",
            device="cpu",
        )
        for _ in pose_matrices
    ]
    extrinsics = [
        CameraExtrinsics(
            extrinsics=pose_matrix, extr_convention=extr_convention, device="cpu"
        )
        for pose_matrix in pose_matrices
    ]
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        device="cpu",
    )


def _build_pinhole_params(
    height: int = 240,
    width: int = 320,
) -> Dict[str, Union[int, float]]:
    """Build a pinhole params dict stated in the standard pixel frame.

    Args:
        height: Image height the params are stated against.
        width: Image width the params are stated against.

    Returns:
        A pinhole params dict carrying fx / fy / cx / cy plus h / w.
    """
    return {
        "fx": 400.0,
        "fy": 410.0,
        "cx": 150.0,
        "cy": 110.0,
        "h": height,
        "w": width,
    }


@pytest.mark.parametrize("extr_convention", EXTR_CONVENTIONS)
def test_validate_extr_convention_accepts_all_supported(extr_convention: str) -> None:
    """validate_extr_convention accepts every supported pose-frame convention string.

    Args:
        extr_convention: The pose-frame convention string under test.

    Returns:
        None.
    """
    assert (
        validate_extr_convention(extr_convention) == extr_convention
    ), f"{extr_convention=}"


def test_extr_convention_module_has_one_main_api_and_eight_helpers() -> None:
    """The relocated extrinsics conventions module has one main API and eight helpers.

    Args:
        None.

    Returns:
        None.
    """
    defined_names = {
        name
        for name, function in inspect.getmembers(conventions, inspect.isfunction)
        if function.__module__ == conventions.__name__
    }
    public_names = {name for name in defined_names if not name.startswith("_")}
    assert public_names == {"transform_extr_convention"}, f"{public_names=}"
    expected_helpers = {
        "_opengl_to_standard",
        "_standard_to_opengl",
        "_opencv_to_standard",
        "_standard_to_opencv",
        "_pytorch3d_to_standard",
        "_standard_to_pytorch3d",
        "_arkit_to_standard",
        "_standard_to_arkit",
    }
    assert defined_names - public_names == expected_helpers, f"{defined_names=}"


@pytest.mark.parametrize(
    "source_extr_convention,target_extr_convention",
    list(product(EXTR_CONVENTIONS, EXTR_CONVENTIONS)),
)
def test_extrinsics_conversion_preserves_physical_axes_and_center(
    source_extr_convention: str,
    target_extr_convention: str,
) -> None:
    """Converting a CameraExtrinsics preserves its physical axes and center.

    Args:
        source_extr_convention: Source pose-frame convention.
        target_extr_convention: Target pose-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=source_extr_convention)
    converted = extrinsics.to(extr_convention=target_extr_convention)
    assert torch.allclose(
        converted.center, extrinsics.center, atol=1.0e-06, rtol=0.0
    ), f"{converted.center=} {extrinsics.center=}"
    assert torch.allclose(
        converted.right, extrinsics.right, atol=1.0e-06, rtol=0.0
    ), f"{converted.right=} {extrinsics.right=}"
    assert torch.allclose(
        converted.forward, extrinsics.forward, atol=1.0e-06, rtol=0.0
    ), f"{converted.forward=} {extrinsics.forward=}"
    assert torch.allclose(
        converted.up, extrinsics.up, atol=1.0e-06, rtol=0.0
    ), f"{converted.up=} {extrinsics.up=}"


@pytest.mark.parametrize(
    "source_extr_convention,target_extr_convention",
    list(product(EXTR_CONVENTIONS, EXTR_CONVENTIONS)),
)
def test_extrinsics_direct_and_via_standard_conversion_match(
    source_extr_convention: str,
    target_extr_convention: str,
) -> None:
    """Converting directly between two extr_conventions matches converting via the standard one.

    Args:
        source_extr_convention: Source pose-frame convention.
        target_extr_convention: Target pose-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=source_extr_convention)
    converted_direct = extrinsics.to(extr_convention=target_extr_convention)
    converted_via_standard = extrinsics.to(extr_convention="standard").to(
        extr_convention=target_extr_convention
    )
    assert torch.allclose(
        converted_direct.extrinsics,
        converted_via_standard.extrinsics,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{converted_direct.extrinsics=} {converted_via_standard.extrinsics=}"


@pytest.mark.parametrize(
    "source_extr_convention,target_extr_convention",
    list(product(EXTR_CONVENTIONS, EXTR_CONVENTIONS)),
)
def test_extrinsics_round_trip_returns_original_matrix(
    source_extr_convention: str,
    target_extr_convention: str,
) -> None:
    """Converting a CameraExtrinsics to another extr_convention and back returns the original matrix.

    Args:
        source_extr_convention: Source pose-frame convention.
        target_extr_convention: Target pose-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=source_extr_convention)
    round_trip = extrinsics.to(extr_convention=target_extr_convention).to(
        extr_convention=source_extr_convention
    )
    assert torch.allclose(
        round_trip.extrinsics, extrinsics.extrinsics, atol=1.0e-06, rtol=0.0
    ), f"{round_trip.extrinsics=} {extrinsics.extrinsics=}"


@pytest.mark.parametrize("extr_convention", EXTR_CONVENTIONS)
def test_extrinsics_w2c_is_inverse_of_extrinsics(extr_convention: str) -> None:
    """CameraExtrinsics.w2c is the inverse of the 4x4 cam2world matrix.

    Args:
        extr_convention: The pose-frame convention string under test.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=extr_convention)
    product_matrix = extrinsics.w2c @ extrinsics.extrinsics
    identity = torch.eye(4, dtype=extrinsics.extrinsics.dtype)
    assert torch.allclose(
        product_matrix, identity, atol=1.0e-05, rtol=0.0
    ), f"{product_matrix=}"


def test_transform_extrinsics_applies_the_similarity_and_restabilizes() -> None:
    """A similarity carries a pose the way it carries the world that pose sits in.

    Args:
        None.

    Returns:
        None.
    """
    matrix = _build_extrinsics_matrix()
    extrinsics = CameraExtrinsics(
        extrinsics=matrix,
        extr_convention="standard",
        device="cpu",
    )
    scale = 2.0
    rotation = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    transformed = extrinsics.transform_extrinsics(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    rotation_torch = torch.from_numpy(rotation)
    expected_rotation = rotation_torch @ matrix[:3, :3]
    expected_center = scale * (rotation_torch @ matrix[:3, 3]) + torch.from_numpy(
        translation
    )
    assert torch.allclose(
        transformed.extrinsics[:3, :3], expected_rotation, atol=1e-6
    ), f"{transformed.extrinsics=}"
    assert torch.allclose(
        transformed.extrinsics[:3, 3], expected_center, atol=1e-6
    ), f"{transformed.extrinsics=}"
    validate_camera_extrinsics(transformed.extrinsics)

    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=_build_pinhole_params(),
        intr_convention="standard",
        device="cpu",
    )
    camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics, device="cpu")
    transformed_camera = camera.transform_extrinsics(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    assert torch.allclose(
        transformed_camera.extrinsics.extrinsics, transformed.extrinsics, atol=1e-6
    ), f"{transformed_camera.extrinsics.extrinsics=}"

    cameras = Cameras(intrinsics=[intrinsics], extrinsics=[extrinsics], device="cpu")
    transformed_cameras = cameras.transform_extrinsics(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    for one_camera in transformed_cameras:
        assert torch.allclose(
            one_camera.extrinsics.extrinsics, transformed.extrinsics, atol=1e-6
        ), f"{one_camera.extrinsics.extrinsics=}"


def test_extrinsics_constructor_and_to_apply_dtype_and_copy() -> None:
    """CameraExtrinsics constructor and to apply dtype and copy semantics.

    Args:
        None.

    Returns:
        None.
    """
    extrinsics = CameraExtrinsics(
        extrinsics=_build_extrinsics_matrix(),
        extr_convention="standard",
        device="cpu",
        dtype=torch.float64,
    )
    assert extrinsics.dtype == torch.float64, f"{extrinsics.dtype=}"
    assert extrinsics.extrinsics.dtype == torch.float64, f"{extrinsics.extrinsics=}"

    copied = extrinsics.to(device="cpu", dtype=torch.float64, copy=True)
    assert copied.extrinsics.data_ptr() != extrinsics.extrinsics.data_ptr(), (
        "Expected copy=True to allocate distinct extrinsics storage. "
        f"{copied.extrinsics.data_ptr()=} {extrinsics.extrinsics.data_ptr()=}"
    )
    moved = extrinsics.to(dtype=torch.float32)
    assert moved.dtype == torch.float32, f"{moved.dtype=}"
    assert moved.extrinsics.dtype == torch.float32, f"{moved.extrinsics.dtype=}"


def test_transform_extrinsics_accepts_array_like_inputs_and_keeps_gradients() -> None:
    """Array-like similarity inputs normalize to tensors without leaving autograd.

    Args:
        None.

    Returns:
        None.
    """
    matrix = _build_extrinsics_matrix().to(dtype=torch.float64)
    matrix.requires_grad_()
    scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    rotation = torch.eye(3, dtype=torch.float64, requires_grad=True)
    translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    extrinsics = CameraExtrinsics(
        extrinsics=matrix,
        extr_convention="standard",
        device="cpu",
        dtype=torch.float64,
    )

    transformed = extrinsics.transform_extrinsics(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    loss = transformed.center.sum()
    loss.backward()

    assert transformed.dtype == torch.float64, f"{transformed.dtype=}"
    assert matrix.grad is not None, f"{matrix.grad=}"
    assert scale.grad is not None, f"{scale.grad=}"
    assert rotation.grad is not None, f"{rotation.grad=}"
    assert translation.grad is not None, f"{translation.grad=}"

    list_transformed = extrinsics.transform_extrinsics(
        scale=np.array(2.0, dtype=np.float64),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=(1.0, 2.0, 3.0),
    )
    assert list_transformed.dtype == torch.float64, f"{list_transformed.dtype=}"


def test_camera_and_cameras_to_keep_tensor_state_on_the_autograd_path() -> None:
    """Camera.to and Cameras.to preserve tensor intrinsics and extrinsics gradients.

    Args:
        None.

    Returns:
        None.
    """
    params = {
        "fx": torch.tensor(400.0, dtype=torch.float32, requires_grad=True),
        "fy": torch.tensor(410.0, dtype=torch.float32, requires_grad=True),
        "cx": torch.tensor(160.0, dtype=torch.float32, requires_grad=True),
        "cy": torch.tensor(120.0, dtype=torch.float32, requires_grad=True),
        "h": torch.tensor(240.0, dtype=torch.float32),
        "w": torch.tensor(320.0, dtype=torch.float32),
    }
    matrix = _build_extrinsics_matrix()
    matrix.requires_grad_()
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=params,
        intr_convention="standard",
        device="cpu",
    )
    extrinsics = CameraExtrinsics(
        extrinsics=matrix,
        extr_convention="standard",
        device="cpu",
    )
    camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics, device="cpu")
    moved_camera = camera.to(dtype=torch.float64, extr_convention="pytorch3d")
    camera_loss = moved_camera.intrinsics.fx + moved_camera.extrinsics.center.sum()
    camera_loss.backward(retain_graph=True)
    assert params["fx"].grad is not None, f"{params['fx'].grad=}"
    assert matrix.grad is not None, f"{matrix.grad=}"

    cameras = Cameras(intrinsics=[intrinsics], extrinsics=[extrinsics], device="cpu")
    moved_cameras = cameras.to(dtype=torch.float64, extr_convention="pytorch3d")
    cameras_loss = (
        moved_cameras.intrinsics[0].fx + moved_cameras.extrinsics[0].center.sum()
    )
    cameras_loss.backward()
    assert params["fx"].grad is not None, f"{params['fx'].grad=}"
    assert matrix.grad is not None, f"{matrix.grad=}"


@pytest.mark.parametrize(
    "source_extr_convention,target_extr_convention",
    list(product(EXTR_CONVENTIONS, EXTR_CONVENTIONS)),
)
def test_cameras_conversion_preserves_physical_axes_and_center(
    source_extr_convention: str,
    target_extr_convention: str,
) -> None:
    """Converting a Cameras collection preserves each camera's axes and center.

    Args:
        source_extr_convention: Source pose-frame convention.
        target_extr_convention: Target pose-frame convention.

    Returns:
        None.
    """
    cameras = _build_cameras(extr_convention=source_extr_convention)
    converted = cameras.to(extr_convention=target_extr_convention)
    assert converted.center.shape == (len(cameras), 3), f"{converted.center.shape=}"
    assert torch.allclose(
        converted.center, cameras.center, atol=1.0e-06, rtol=0.0
    ), f"{converted.center=} {cameras.center=}"
    assert torch.allclose(
        converted.right, cameras.right, atol=1.0e-06, rtol=0.0
    ), f"{converted.right=} {cameras.right=}"
    assert torch.allclose(
        converted.forward, cameras.forward, atol=1.0e-06, rtol=0.0
    ), f"{converted.forward=} {cameras.forward=}"
    assert torch.allclose(
        converted.up, cameras.up, atol=1.0e-06, rtol=0.0
    ), f"{converted.up=} {cameras.up=}"


@pytest.mark.parametrize("target_extr_convention", EXTR_CONVENTIONS)
def test_every_supported_extr_convention_is_right_handed(
    target_extr_convention: str,
) -> None:
    """Each supported pose frame's (right, forward, up) triple is positively oriented.

    A camera carries no change of handedness, so converting between two supported
    conventions keeps the rotation determinant at +1.

    Args:
        target_extr_convention: Target pose-frame convention.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention="standard")
    converted = extrinsics.to(extr_convention=target_extr_convention)
    triple_product = torch.dot(
        torch.linalg.cross(converted.right, converted.forward), converted.up
    )
    assert float(triple_product) > 0.0, (
        "Expected the (right, forward, up) triple to be positively oriented. "
        f"{target_extr_convention=} {float(triple_product)=}"
    )
    determinant = torch.linalg.det(converted.extrinsics[:3, :3])
    assert torch.isclose(
        determinant,
        torch.tensor(1.0, dtype=determinant.dtype),
        atol=1.0e-05,
        rtol=0.0,
    ), (
        "Expected the converted rotation block to keep determinant +1. "
        f"{target_extr_convention=} {float(determinant)=}"
    )


def test_validate_intr_convention_accepts_all_supported() -> None:
    """The intrinsics name their own frame from a closed set.

    Args:
        None.

    Returns:
        None.
    """
    for intr_convention in ("standard", "opengl", "pytorch3d", "vulkan"):
        assert (
            validate_intr_convention(intr_convention=intr_convention) == intr_convention
        ), f"{intr_convention=}"
    with pytest.raises(AssertionError):
        validate_intr_convention(intr_convention="ndc")


def test_intr_convention_module_has_one_main_api_and_six_spoke_helpers() -> None:
    """Each frame brings its own inbound and outbound helper against the standard one.

    Args:
        None.

    Returns:
        None.
    """
    functions = [
        name
        for name, obj in inspect.getmembers(intr_conventions, inspect.isfunction)
        if obj.__module__ == intr_conventions.__name__
    ]
    public = [name for name in functions if not name.startswith("_")]
    assert public == ["transform_intr_convention"], f"{public=}"
    assert (
        inspect.getsource(intr_conventions).count("rescale_intr_params") > 0
    ), f"{functions=}"
    spokes = {
        f"_{direction}"
        for direction in (
            "standard_to_opengl",
            "opengl_to_standard",
            "standard_to_pytorch3d",
            "pytorch3d_to_standard",
            "standard_to_vulkan",
            "vulkan_to_standard",
        )
    }
    assert spokes.issubset(set(functions)), f"{functions=}"
    obliques = [name for name in functions if "_to_" in name and "standard" not in name]
    assert obliques == [], f"{obliques=}"


def test_a_frame_change_comes_down_to_the_same_per_axis_rescale() -> None:
    """A frame change's only length step is the per-axis rescale scaling owns.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    rescaled = rescale_intr_params(
        params=params,
        model="pinhole",
        unit_x=2.0,
        unit_y=0.5,
    )
    assert rescaled["fx"] == pytest.approx(800.0), f"{rescaled=}"
    assert rescaled["cx"] == pytest.approx(300.0), f"{rescaled=}"
    assert rescaled["fy"] == pytest.approx(205.0), f"{rescaled=}"
    assert rescaled["cy"] == pytest.approx(55.0), f"{rescaled=}"
    assert rescaled["h"] == params["h"], f"{rescaled=}"
    assert rescaled["w"] == params["w"], f"{rescaled=}"

    simple_params: Dict[str, Union[int, float]] = {
        "f": 400.0,
        "cx": 150.0,
        "cy": 110.0,
        "h": 240,
        "w": 320,
    }
    with pytest.raises(AssertionError):
        transform_intr_convention(
            params=simple_params,
            model="simple_pinhole",
            source_intr_convention="standard",
            target_intr_convention="opengl",
        )
    with pytest.raises(AssertionError):
        rescale_intr_params(
            params=simple_params,
            model="simple_pinhole",
            unit_x=2.0,
            unit_y=0.5,
        )


def test_three_separations_stand_between_standard_and_a_device_frame() -> None:
    """Origin, axis direction, and unit are independent separations.

    Args:
        None.

    Returns:
        None.
    """
    centred = _build_pinhole_params()
    centred["cx"] = 160.0
    centred["cy"] = 120.0
    for frame in ("opengl", "pytorch3d", "vulkan"):
        converted = transform_intr_convention(
            params=centred,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert converted["cx"] == pytest.approx(0.0), f"{frame=} {converted=}"
        assert converted["cy"] == pytest.approx(0.0), f"{frame=} {converted=}"

    below_centre = dict(centred)
    below_centre["cy"] = 121.0
    assert (
        transform_intr_convention(
            params=below_centre,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention="vulkan",
        )["cy"]
        > 0.0
    ), f"{below_centre=}"
    assert (
        transform_intr_convention(
            params=below_centre,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention="opengl",
        )["cy"]
        < 0.0
    ), f"{below_centre=}"

    right_of_centre = dict(centred)
    right_of_centre["cx"] = 161.0
    assert (
        transform_intr_convention(
            params=right_of_centre,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention="pytorch3d",
        )["cx"]
        < 0.0
    ), f"{right_of_centre=}"
    assert (
        transform_intr_convention(
            params=right_of_centre,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention="opengl",
        )["cx"]
        > 0.0
    ), f"{right_of_centre=}"


def test_each_frame_normalizes_by_the_side_its_own_definition_names() -> None:
    """PyTorch3D spans its shorter side alone; opengl and vulkan span each axis by its own.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params(height=240, width=320)
    pytorch3d = transform_intr_convention(
        params=params,
        model="pinhole",
        source_intr_convention="standard",
        target_intr_convention="pytorch3d",
    )
    shorter_unit = 2.0 / float(min(params["h"], params["w"]))
    assert pytorch3d["fx"] == pytest.approx(
        params["fx"] * shorter_unit
    ), f"{pytorch3d=}"
    assert pytorch3d["fy"] == pytest.approx(
        params["fy"] * shorter_unit
    ), f"{pytorch3d=}"
    for frame in ("opengl", "vulkan"):
        converted = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert converted["fx"] == pytest.approx(
            params["fx"] * 2.0 / float(params["w"])
        ), f"{frame=} {converted=}"
        assert converted["fy"] == pytest.approx(
            params["fy"] * 2.0 / float(params["h"])
        ), f"{frame=} {converted=}"


def test_only_the_unit_reaches_the_focal_params() -> None:
    """An axis reversal cancels at the two ends of the linear term, and the origin move touches no coefficient.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    units = {
        "opengl": (2.0 / float(params["w"]), 2.0 / float(params["h"])),
        "vulkan": (2.0 / float(params["w"]), 2.0 / float(params["h"])),
        "pytorch3d": (
            2.0 / float(min(params["h"], params["w"])),
            2.0 / float(min(params["h"], params["w"])),
        ),
    }
    for frame, (unit_x, unit_y) in units.items():
        converted = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert converted["fx"] == pytest.approx(
            params["fx"] * unit_x
        ), f"{frame=} {converted=}"
        assert converted["fy"] == pytest.approx(
            params["fy"] * unit_y
        ), f"{frame=} {converted=}"


def test_the_perspective_and_weak_perspective_models_take_the_same_focal_rule() -> None:
    """A focal is a pixels-per-camera-unit ratio whether or not the projection divides by depth.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
        pinhole = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        ortho = transform_intr_convention(
            params=params,
            model="ortho",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert pinhole == ortho, f"{frame=} {pinhole=} {ortho=}"


def test_one_shared_focal_cannot_carry_two_different_axis_scales() -> None:
    """simple_pinhole states a single f, so a frame normalizing the axes by different sides aborts.

    Args:
        None.

    Returns:
        None.
    """
    params: Dict[str, Union[int, float]] = {
        "f": 400.0,
        "cx": 150.0,
        "cy": 110.0,
        "h": 240,
        "w": 320,
    }
    converted = transform_intr_convention(
        params=params,
        model="simple_pinhole",
        source_intr_convention="standard",
        target_intr_convention="pytorch3d",
    )
    assert converted["f"] == pytest.approx(
        params["f"] * 2.0 / float(min(params["h"], params["w"]))
    ), f"{converted=}"
    for frame in ("opengl", "vulkan"):
        with pytest.raises(AssertionError):
            transform_intr_convention(
                params=params,
                model="simple_pinhole",
                source_intr_convention="standard",
                target_intr_convention=frame,
            )


def test_a_camera_model_with_no_focal_rule_is_refused() -> None:
    """The models this frame change knows are a closed set.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(NotImplementedError):
        transform_intr_convention(
            params=_build_pinhole_params(),
            model="fisheye",
            source_intr_convention="standard",
            target_intr_convention="opengl",
        )


def test_a_direct_conversion_matches_the_one_through_standard() -> None:
    """Every oblique pair is served by composing the two spokes.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    frames = ("standard", "opengl", "pytorch3d", "vulkan")
    for source, target in product(frames, frames):
        direct = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention=source,
            target_intr_convention=target,
        )
        via_standard = transform_intr_convention(
            params=transform_intr_convention(
                params=params,
                model="pinhole",
                source_intr_convention=source,
                target_intr_convention="standard",
            ),
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=target,
        )
        for key in direct:
            assert direct[key] == pytest.approx(
                via_standard[key]
            ), f"{source=} {target=} {key=} {direct=} {via_standard=}"


def test_an_intr_convention_round_trip_returns_the_original_params() -> None:
    """A frame change is a restatement rather than a loss.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=params,
        intr_convention="standard",
        device="cpu",
    )
    for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
        round_tripped = intrinsics.to(intr_convention=frame).to(
            intr_convention="standard"
        )
        for key, value in params.items():
            assert round_tripped.params[key] == pytest.approx(
                value
            ), f"{frame=} {key=} {round_tripped.params=}"


def test_a_converted_intrinsics_still_satisfies_its_own_invariants() -> None:
    """Converted params remain valid for their target frame.

    Args:
        None.

    Returns:
        None.
    """
    frames = ("standard", "opengl", "pytorch3d", "vulkan")
    for model in ("pinhole", "ortho"):
        params = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in _build_pinhole_params().items()
        }
        for target_intr_convention in frames:
            transformed_params = transform_intr_convention(
                params=params,
                model=model,
                source_intr_convention="standard",
                target_intr_convention=target_intr_convention,
            )
            tensor_params = {
                key: torch.as_tensor(value) for key, value in transformed_params.items()
            }
            validate_camera_intrinsics_invariants(
                model=model,
                intr_convention=target_intr_convention,
                params=tensor_params,
            )


def test_a_frame_change_is_measured_against_the_intrinsics_own_resolution() -> None:
    """The resolution fixes where a centred origin sits and what a normalized unit is worth.

    Args:
        None.

    Returns:
        None.
    """
    narrow = build_camera_intrinsics(
        model="pinhole",
        params=_build_pinhole_params(height=240, width=320),
        intr_convention="standard",
        device="cpu",
    ).to(intr_convention="opengl")
    wide = build_camera_intrinsics(
        model="pinhole",
        params=_build_pinhole_params(height=240, width=640),
        intr_convention="standard",
        device="cpu",
    ).to(intr_convention="opengl")
    assert narrow.params["cx"] != pytest.approx(
        wide.params["cx"]
    ), f"{narrow.params=} {wide.params=}"
    assert narrow.params["h"] == 240 and narrow.params["w"] == 320, f"{narrow.params=}"
    assert wide.params["h"] == 240 and wide.params["w"] == 640, f"{wide.params=}"


def test_an_intrinsics_without_a_resolution_is_refused() -> None:
    """A principal point in the standard frame names a location only against a resolution.

    Args:
        None.

    Returns:
        None.
    """
    projection_only: Dict[str, Dict[str, Union[int, float]]] = {
        "simple_pinhole": {"f": 400.0, "cx": 150.0, "cy": 110.0},
        "pinhole": {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0},
        "ortho": {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0},
    }
    for model, params in projection_only.items():
        with pytest.raises(AssertionError):
            build_camera_intrinsics(
                model=model,
                params=params,
                intr_convention="standard",
                device="cpu",
            )


def test_a_camera_names_the_frame_of_each_half_separately() -> None:
    """A pose frame and an image-plane frame are different kinds of thing.

    Args:
        None.

    Returns:
        None.
    """
    camera = Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params=_build_pinhole_params(),
            intr_convention="standard",
            device="cpu",
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=_build_extrinsics_matrix(),
            extr_convention="standard",
            device="cpu",
        ),
        device="cpu",
    )
    both = camera.to(intr_convention="opengl", extr_convention="opencv")
    assert both.extrinsics.extr_convention == "opencv", f"{both.extrinsics=}"
    assert both.intrinsics.intr_convention == "opengl", f"{both.intrinsics=}"

    pose_only = camera.to(extr_convention="opencv")
    assert pose_only.extrinsics.extr_convention == "opencv", f"{pose_only.extrinsics=}"
    assert (
        pose_only.intrinsics.intr_convention == "standard"
    ), f"{pose_only.intrinsics=}"

    plane_only = camera.to(intr_convention="opengl")
    assert (
        plane_only.extrinsics.extr_convention == "standard"
    ), f"{plane_only.extrinsics=}"
    assert (
        plane_only.intrinsics.intr_convention == "opengl"
    ), f"{plane_only.intrinsics=}"
