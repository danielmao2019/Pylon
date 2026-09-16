import inspect
from itertools import product
from typing import Dict, List, Tuple, Union

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
    _rescale_intr_params,
    transform_intr_convention,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_invariants,
    validate_intr_convention,
)


def test_validate_extr_convention_accepts_all_supported() -> None:
    """validate_extr_convention accepts every supported pose-frame convention string.

    Args:
        None.

    Returns:
        None.
    """
    for extr_convention in ("standard", "opengl", "opencv", "pytorch3d", "arkit"):
        returned = validate_extr_convention(extr_convention=extr_convention)
        assert returned == extr_convention, (
            "Expected validate_extr_convention to return the convention it was given. "
            f"{extr_convention=} {returned=}"
        )
    return


def test_extr_convention_module_has_one_main_api_and_eight_helpers() -> None:
    """The relocated extrinsics conventions module has one main API and eight helpers.

    Args:
        None.

    Returns:
        None.
    """
    defined_names: List[str] = []
    for name, function in inspect.getmembers(conventions, inspect.isfunction):
        if function.__module__ == conventions.__name__:
            defined_names.append(name)
    public_names: List[str] = []
    for name in defined_names:
        if not name.startswith("_"):
            public_names.append(name)
    assert public_names == ["transform_extr_convention"], (
        "Expected transform_extr_convention to be the conventions module's only public function. "
        f"{public_names=}"
    )
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
    assert set(defined_names) - set(public_names) == expected_helpers, (
        "Expected the private helpers to be exactly the to-standard and from-standard converters for opengl, opencv, pytorch3d and arkit. "
        f"{defined_names=}"
    )
    return


@pytest.mark.parametrize(
    "source_extr_convention", ["standard", "opengl", "opencv", "pytorch3d", "arkit"]
)
def test_extrinsics_conversion_preserves_physical_axes_and_center(
    source_extr_convention: str,
) -> None:
    """Converting a CameraExtrinsics to every extr_convention preserves its physical axes and center.

    Args:
        source_extr_convention: Source pose-frame convention the extrinsics is built in.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=source_extr_convention)
    for target_extr_convention in (
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ):
        converted = extrinsics.to(extr_convention=target_extr_convention)
        assert (
            torch.allclose(converted.right, extrinsics.right, atol=1.0e-06, rtol=0.0)
            and torch.allclose(
                converted.forward, extrinsics.forward, atol=1.0e-06, rtol=0.0
            )
            and torch.allclose(converted.up, extrinsics.up, atol=1.0e-06, rtol=0.0)
        ), (
            "Expected the converted right / forward / up axes to equal the source ones. "
            f"{source_extr_convention=} {target_extr_convention=} "
            f"{converted.right=} {extrinsics.right=} {converted.forward=} "
            f"{extrinsics.forward=} {converted.up=} {extrinsics.up=}"
        )
        assert torch.allclose(
            converted.center, extrinsics.center, atol=1.0e-06, rtol=0.0
        ), (
            "Expected the converted center to equal the source center. "
            f"{source_extr_convention=} {target_extr_convention=} "
            f"{converted.center=} {extrinsics.center=}"
        )
    return


def test_extrinsics_direct_and_via_standard_conversion_match() -> None:
    """Converting directly between two extr_conventions matches converting via the standard one.

    Args:
        None.

    Returns:
        None.
    """
    for source_extr_convention, target_extr_convention in product(
        ("standard", "opengl", "opencv", "pytorch3d", "arkit"),
        ("standard", "opengl", "opencv", "pytorch3d", "arkit"),
    ):
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
        ), (
            "Expected the direct conversion to match the one through standard. "
            f"{source_extr_convention=} {target_extr_convention=} "
            f"{converted_direct.extrinsics=} {converted_via_standard.extrinsics=}"
        )
    return


@pytest.mark.parametrize(
    "source_extr_convention", ["standard", "opengl", "opencv", "pytorch3d", "arkit"]
)
def test_extrinsics_round_trip_returns_original_matrix(
    source_extr_convention: str,
) -> None:
    """Converting a CameraExtrinsics to every other extr_convention and back returns the original matrix.

    Args:
        source_extr_convention: Source pose-frame convention the extrinsics is built in.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention=source_extr_convention)
    for target_extr_convention in (
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ):
        round_trip = extrinsics.to(extr_convention=target_extr_convention).to(
            extr_convention=source_extr_convention
        )
        assert torch.allclose(
            round_trip.extrinsics, extrinsics.extrinsics, atol=1.0e-06, rtol=0.0
        ), (
            "Expected the round-tripped 4x4 matrix to equal the original. "
            f"{source_extr_convention=} {target_extr_convention=} "
            f"{round_trip.extrinsics=} {extrinsics.extrinsics=}"
        )
    return


@pytest.mark.parametrize(
    "extr_convention", ["standard", "opengl", "opencv", "pytorch3d", "arkit"]
)
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
    assert torch.allclose(product_matrix, identity, atol=1.0e-05, rtol=0.0), (
        "Expected w2c @ extrinsics to equal the 4x4 identity. " f"{product_matrix=}"
    )
    return


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
    ), (
        "Expected the returned rotation block to equal the known rotation composed onto the source's. "
        f"{transformed.extrinsics=}"
    )
    assert torch.allclose(transformed.extrinsics[:3, 3], expected_center, atol=1e-6), (
        "Expected the returned centre to equal the source centre scaled, rotated and translated. "
        f"{transformed.extrinsics=}"
    )
    validate_camera_extrinsics(obj=transformed.extrinsics)

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
    ), (
        "Expected the camera's transformed extrinsics to equal the standalone CameraExtrinsics result. "
        f"{transformed_camera.extrinsics.extrinsics=}"
    )

    cameras = Cameras(intrinsics=intrinsics[None], extrinsics=extrinsics[None])
    transformed_cameras = cameras.transform_extrinsics(
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    for one_camera in transformed_cameras:
        assert torch.allclose(
            one_camera.extrinsics.extrinsics, transformed.extrinsics, atol=1e-6
        ), (
            "Expected every camera in the batch to carry the standalone CameraExtrinsics result. "
            f"{one_camera.extrinsics.extrinsics=}"
        )
    return


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
    assert (
        extrinsics.dtype == torch.float64
        and extrinsics.extrinsics.dtype == torch.float64
    ), (
        "Expected an extrinsics built with float64 to report float64 and hold its matrix in float64. "
        f"{extrinsics.dtype=} {extrinsics.extrinsics.dtype=}"
    )
    copied = extrinsics.to(device="cpu", dtype=torch.float64, copy=True)
    assert copied.extrinsics.data_ptr() != extrinsics.extrinsics.data_ptr(), (
        "Expected copy=True to allocate distinct extrinsics storage. "
        f"{copied.extrinsics.data_ptr()=} {extrinsics.extrinsics.data_ptr()=}"
    )
    moved = extrinsics.to(dtype=torch.float32)
    assert moved.dtype == torch.float32 and moved.extrinsics.dtype == torch.float32, (
        "Expected an extrinsics moved to float32 to report float32 and hold its matrix in float32. "
        f"{moved.dtype=} {moved.extrinsics.dtype=}"
    )
    return


def test_transform_extrinsics_accepts_array_like_inputs_and_keeps_gradients() -> None:
    """Array-like similarity inputs normalize to tensors without leaving autograd.

    Args:
        None.

    Returns:
        None.
    """
    matrix = _build_extrinsics_matrix().to(dtype=torch.float64).requires_grad_()
    scale, rotation, translation = (
        torch.tensor(2.0, dtype=torch.float64, requires_grad=True),
        torch.eye(3, dtype=torch.float64, requires_grad=True),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True),
    )
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
    transformed.center.sum().backward()
    assert (
        transformed.dtype == torch.float64
        and matrix.grad is not None
        and scale.grad is not None
        and rotation.grad is not None
        and translation.grad is not None
    ), (
        "Expected a float64 similarity to return a float64 extrinsics and every tensor input to receive a gradient. "
        f"{transformed.dtype=} {matrix.grad=} {scale.grad=} {rotation.grad=} {translation.grad=}"
    )
    list_transformed = extrinsics.transform_extrinsics(
        scale=np.array(2.0, dtype=np.float64),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation=(1.0, 2.0, 3.0),
    )
    assert list_transformed.dtype == torch.float64, (
        "Expected an array-like similarity to return a float64 extrinsics. "
        f"{list_transformed.dtype=}"
    )
    return


def test_camera_and_cameras_to_keep_tensor_state_on_the_autograd_path() -> None:
    """Camera.to and Cameras.to preserve tensor intrinsics and extrinsics gradients.

    Args:
        None.

    Returns:
        None.
    """
    matrix = _build_extrinsics_matrix().requires_grad_()
    params = {
        "fx": torch.tensor(400.0, dtype=torch.float32, requires_grad=True),
        "fy": torch.tensor(410.0, dtype=torch.float32, requires_grad=True),
        "cx": torch.tensor(160.0, dtype=torch.float32, requires_grad=True),
        "cy": torch.tensor(120.0, dtype=torch.float32, requires_grad=True),
        "h": torch.tensor(240.0, dtype=torch.float32),
        "w": torch.tensor(320.0, dtype=torch.float32),
    }
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
    assert params["fx"].grad is not None, (
        "Expected the source fx param to receive a gradient through the moved camera. "
        f"{params['fx'].grad=}"
    )
    assert matrix.grad is not None, (
        "Expected the source cam2world matrix to receive a gradient through the moved camera. "
        f"{matrix.grad=}"
    )
    params["fx"].grad = None
    matrix.grad = None

    cameras = Cameras(
        intrinsics=intrinsics[None], extrinsics=extrinsics[None], device="cpu"
    )
    moved_cameras = cameras.to(dtype=torch.float64, extr_convention="pytorch3d")
    cameras_loss = (
        moved_cameras.intrinsics[0].fx + moved_cameras.extrinsics[0].center.sum()
    )
    cameras_loss.backward()
    assert params["fx"].grad is not None, (
        "Expected the source fx param to receive a gradient through the moved collection. "
        f"{params['fx'].grad=}"
    )
    assert matrix.grad is not None, (
        "Expected the source cam2world matrix to receive a gradient through the moved collection. "
        f"{matrix.grad=}"
    )
    return


def test_cameras_device_and_dtype_follow_the_given_placement() -> None:
    """A Cameras refuses two components that disagree on device or dtype, brings both to the device and dtype it is handed, and resolves one left unset to the single value both components hold.

    Args:
        None.

    Returns:
        None.
    """
    matrix = _build_extrinsics_matrix()
    pinhole_params = _build_pinhole_params()
    params: Dict[str, torch.Tensor] = {}
    for key, value in pinhole_params.items():
        params[key] = torch.tensor([value], dtype=torch.float32)
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=params,
        intr_convention="standard",
        device="cpu",
    )
    extrinsics = CameraExtrinsics(
        extrinsics=matrix[None],
        extr_convention="standard",
        device="cpu",
    )
    unset = Cameras(intrinsics=intrinsics, extrinsics=extrinsics)
    assert (
        unset.device == intrinsics.device == extrinsics.device == torch.device("cpu")
    ), f"Expected a Cameras handed no device to take the one both components share. {unset.device=} {intrinsics.device=} {extrinsics.device=}"
    assert (
        unset.dtype == intrinsics.dtype == extrinsics.dtype == torch.float32
    ), f"Expected a Cameras handed no dtype to take the one both components share. {unset.dtype=} {intrinsics.dtype=} {extrinsics.dtype=}"
    cast = Cameras(intrinsics=intrinsics, extrinsics=extrinsics, dtype=torch.float64)
    param_dtypes: Dict[str, torch.dtype] = {}
    for key, value in cast.intrinsics.params.items():
        param_dtypes[key] = value.dtype
    assert (
        cast.dtype == torch.float64
    ), f"Expected a Cameras handed a dtype to take it. {cast.dtype=}"
    assert (
        cast.extrinsics.extrinsics.dtype == torch.float64
    ), f"Expected the extrinsics matrix to be cast to the dtype the batch was handed. {cast.extrinsics.extrinsics.dtype=}"
    for dtype in param_dtypes.values():
        assert (
            dtype == torch.float64
        ), f"Expected every intrinsics param to be cast to the dtype the batch was handed. {param_dtypes=}"
    assert cast.device == torch.device(
        "cpu"
    ), f"Expected a Cameras handed only a dtype to keep the device both components share. {cast.device=}"
    float64_extrinsics = CameraExtrinsics(
        extrinsics=matrix[None],
        extr_convention="standard",
        device="cpu",
        dtype=torch.float64,
    )
    with pytest.raises(AssertionError):
        Cameras(intrinsics=intrinsics, extrinsics=float64_extrinsics)
    with pytest.raises(AssertionError):
        Cameras(
            intrinsics=intrinsics, extrinsics=float64_extrinsics, dtype=torch.float32
        )
    if torch.cuda.is_available():
        moved = Cameras(intrinsics=intrinsics, extrinsics=extrinsics, device="cuda")
        current_cuda = torch.device("cuda", torch.cuda.current_device())
        param_devices: Dict[str, torch.device] = {}
        for key, value in moved.intrinsics.params.items():
            param_devices[key] = value.device
        assert (
            moved.device == current_cuda
        ), f"Expected a Cameras handed a bare cuda to spell the device with the current cuda index. {moved.device=}"
        assert (
            moved.extrinsics.extrinsics.device == current_cuda
        ), f"Expected the extrinsics matrix to be brought to the batch's device. {moved.extrinsics.extrinsics.device=}"
        for device in param_devices.values():
            assert (
                device == current_cuda
            ), f"Expected every intrinsics param to be brought to the batch's device. {param_devices=}"
        assert (
            moved.dtype == torch.float32
        ), f"Expected a Cameras handed only a device to keep the dtype both components share. {moved.dtype=}"


def test_transform_extrinsics_normalizes_rotation_input() -> None:
    """transform_extrinsics normalizes each rotation representation to the pose tensor.

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
    rotation_torch = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    expected = extrinsics.transform_extrinsics(
        scale=scale, rotation=rotation_torch, translation=translation
    )
    for rotation in [rotation_torch.numpy(), rotation_torch, rotation_torch.tolist()]:
        transformed = extrinsics.transform_extrinsics(
            scale=scale, rotation=rotation, translation=translation
        )
        assert torch.allclose(transformed.extrinsics, expected.extrinsics, atol=1e-6), (
            "Expected every rotation representation to normalize to the same pose. "
            f"{type(rotation)=} {transformed.extrinsics=} {expected.extrinsics=}"
        )
    return


def test_transform_extrinsics_normalizes_translation_input() -> None:
    """transform_extrinsics normalizes each translation representation to the pose tensor.

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
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    translation_torch = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    expected = extrinsics.transform_extrinsics(
        scale=scale, rotation=rotation, translation=translation_torch
    )
    for translation in [
        translation_torch.numpy(),
        translation_torch,
        tuple(translation_torch.tolist()),
        translation_torch.tolist(),
    ]:
        transformed = extrinsics.transform_extrinsics(
            scale=scale, rotation=rotation, translation=translation
        )
        assert torch.allclose(transformed.extrinsics, expected.extrinsics, atol=1e-6), (
            "Expected every translation representation to normalize to the same pose. "
            f"{type(translation)=} {transformed.extrinsics=} {expected.extrinsics=}"
        )
    return


@pytest.mark.parametrize(
    "source_extr_convention", ["standard", "opengl", "opencv", "pytorch3d", "arkit"]
)
def test_cameras_conversion_preserves_physical_axes_and_center(
    source_extr_convention: str,
) -> None:
    """Converting a Cameras collection preserves each camera's axes and center.

    Args:
        source_extr_convention: Source pose-frame convention.

    Returns:
        None.
    """
    cameras = _build_cameras(extr_convention=source_extr_convention)
    for target_extr_convention in (
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ):
        converted = cameras.to(extr_convention=target_extr_convention)
        assert (
            torch.allclose(converted.right, cameras.right, atol=1.0e-06, rtol=0.0)
            and torch.allclose(
                converted.forward, cameras.forward, atol=1.0e-06, rtol=0.0
            )
            and torch.allclose(converted.up, cameras.up, atol=1.0e-06, rtol=0.0)
        ), (
            "Expected the converted [N, 3] right / forward / up stacks to equal the source ones. "
            f"{source_extr_convention=} {target_extr_convention=} {converted.right=} {cameras.right=} "
            f"{converted.forward=} {cameras.forward=} {converted.up=} {cameras.up=}"
        )
        assert converted.center.shape == (len(cameras), 3) and torch.allclose(
            converted.center, cameras.center, atol=1.0e-06, rtol=0.0
        ), (
            "Expected the converted [N, 3] center stack to equal the source one. "
            f"{source_extr_convention=} {target_extr_convention=} {converted.center=} {cameras.center=}"
        )
    return


def _build_cameras(extr_convention: str) -> Cameras:
    """Build a multi-camera Cameras fixture in the given pose frame.

    Args:
        extr_convention: Pose-frame convention string.

    Returns:
        A Cameras of three CPU cameras with distinct poses in the given pose frame.
    """
    pose_matrices = _build_extrinsics_matrices()
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            "fx": torch.full((len(pose_matrices),), 400.0),
            "fy": torch.full((len(pose_matrices),), 410.0),
            "cx": torch.full((len(pose_matrices),), 160.0),
            "cy": torch.full((len(pose_matrices),), 120.0),
            "h": torch.full((len(pose_matrices),), 240.0),
            "w": torch.full((len(pose_matrices),), 320.0),
        },
        intr_convention="standard",
        device="cpu",
    )
    extrinsics = CameraExtrinsics(
        extrinsics=torch.stack(pose_matrices),
        extr_convention=extr_convention,
        device="cpu",
    )
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        device="cpu",
    )


def test_every_supported_extr_convention_is_right_handed() -> None:
    """Each supported pose frame's (right, forward, up) triple is positively oriented.

    A camera carries no change of handedness, so converting between two supported conventions keeps the rotation determinant at +1.

    Args:
        None.

    Returns:
        None.
    """
    extrinsics = _build_extrinsics(extr_convention="standard")
    for target_extr_convention in (
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ):
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
    return


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


def _build_extrinsics_matrices() -> List[torch.Tensor]:
    """Build distinct valid 4x4 cam2world matrices with proper rotations.

    Args:
        None.

    Returns:
        A list of 4x4 float32 camera-to-world matrices with distinct proper rotations and centers.
    """
    rotation_about_z = _build_extrinsics_matrix()
    identity_rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
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


def _build_extrinsics_matrix() -> torch.Tensor:
    """Build a valid 4x4 cam2world matrix with a proper rotation.

    Args:
        None.

    Returns:
        A 4x4 float32 camera-to-world matrix whose 3x3 block is a proper rotation.
    """
    matrix = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    matrix[:3, 3] = torch.tensor([0.3, -0.2, 1.1], dtype=torch.float32)
    return matrix


def test_validate_intr_convention_accepts_all_supported() -> None:
    """The intrinsics name their own frame from a closed set.

    Args:
        None.

    Returns:
        None.
    """
    for intr_convention in ("standard", "opengl", "pytorch3d", "vulkan"):
        returned = validate_intr_convention(intr_convention=intr_convention)
        assert returned == intr_convention, (
            "Expected validate_intr_convention to return the convention it was given. "
            f"{intr_convention=} {returned=}"
        )
    with pytest.raises(AssertionError):
        validate_intr_convention(intr_convention="ndc")
    return


def test_intr_convention_module_has_one_main_api_and_six_spoke_helpers() -> None:
    """Each frame brings its own inbound and outbound helper against the standard one.

    Args:
        None.

    Returns:
        None.
    """
    functions: List[str] = []
    for name, obj in inspect.getmembers(intr_conventions, inspect.isfunction):
        if obj.__module__ == intr_conventions.__name__:
            functions.append(name)
    public: List[str] = []
    for name in functions:
        if not name.startswith("_"):
            public.append(name)
    assert (
        public == ["transform_intr_convention"] and "_rescale_intr_params" in functions
    ), (
        "Expected transform_intr_convention to be the conventions module's only public function "
        f"and the per-axis rescale each spoke ends in to be its own private helper. {public=} {functions=}"
    )
    spokes = {
        "_standard_to_opengl",
        "_opengl_to_standard",
        "_standard_to_pytorch3d",
        "_pytorch3d_to_standard",
        "_standard_to_vulkan",
        "_vulkan_to_standard",
    }
    obliques: List[str] = []
    for name in functions:
        if "_to_" in name and "standard" not in name:
            obliques.append(name)
    assert spokes.issubset(set(functions)) and obliques == [], (
        "Expected a to-standard and a from-standard helper for opengl, pytorch3d and vulkan, "
        f"and no helper between two non-standard frames. {functions=} {obliques=}"
    )
    return


def test_a_frame_change_comes_down_to_the_same_per_axis_rescale() -> None:
    """A frame change's only length step is the per-axis rescale the conventions module owns.

    Args:
        None.

    Returns:
        None.
    """
    params = _build_pinhole_params()
    rescaled = _rescale_intr_params(
        params=params,
        model="pinhole",
        unit_x=2.0,
        unit_y=0.5,
    )
    assert (
        rescaled["fx"] == pytest.approx(800.0)
        and rescaled["cx"] == pytest.approx(300.0)
        and rescaled["fy"] == pytest.approx(205.0)
        and rescaled["cy"] == pytest.approx(55.0)
        and rescaled["h"] == params["h"]
        and rescaled["w"] == params["w"]
    ), (
        "Expected fx and cx to be scaled by unit_x, fy and cy by unit_y, and h and w to come back untouched. "
        f"{rescaled=} {params=}"
    )
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
        _rescale_intr_params(
            params=simple_params,
            model="simple_pinhole",
            unit_x=2.0,
            unit_y=0.5,
        )
    return


def test_three_separations_stand_between_standard_and_a_device_frame() -> None:
    """Origin, axis direction, and unit are independent separations.

    Args:
        None.

    Returns:
        None.
    """
    centred = {**_build_pinhole_params(), "cx": 160.0, "cy": 120.0}
    for frame in ("opengl", "pytorch3d", "vulkan"):
        converted = transform_intr_convention(
            params=centred,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert converted["cx"] == pytest.approx(0.0) and converted[
            "cy"
        ] == pytest.approx(0.0), (
            "Expected a principal point at half the resolution to land on the device origin. "
            f"{frame=} {converted=}"
        )
    below_centre = {**centred, "cy": 121.0}
    below_vulkan = transform_intr_convention(
        params=below_centre,
        model="pinhole",
        source_intr_convention="standard",
        target_intr_convention="vulkan",
    )
    below_opengl = transform_intr_convention(
        params=below_centre,
        model="pinhole",
        source_intr_convention="standard",
        target_intr_convention="opengl",
    )
    assert below_vulkan["cy"] > 0.0 and below_opengl["cy"] < 0.0, (
        "Expected a point one pixel below centre to come back positive under vulkan and negative under opengl. "
        f"{below_vulkan=} {below_opengl=}"
    )
    right_of_centre = {**centred, "cx": 161.0}
    right_pytorch3d = transform_intr_convention(
        params=right_of_centre,
        model="pinhole",
        source_intr_convention="standard",
        target_intr_convention="pytorch3d",
    )
    right_opengl = transform_intr_convention(
        params=right_of_centre,
        model="pinhole",
        source_intr_convention="standard",
        target_intr_convention="opengl",
    )
    assert right_pytorch3d["cx"] < 0.0 and right_opengl["cx"] > 0.0, (
        "Expected a point one pixel right of centre to come back negative under pytorch3d and positive under opengl. "
        f"{right_pytorch3d=} {right_opengl=}"
    )
    return


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
    assert pytorch3d["fx"] == pytest.approx(params["fx"] * shorter_unit) and pytorch3d[
        "fy"
    ] == pytest.approx(params["fy"] * shorter_unit), (
        "Expected pytorch3d to scale fx and fy by two over the shorter side. "
        f"{pytorch3d=}"
    )
    for frame in ("opengl", "vulkan"):
        converted = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=frame,
        )
        assert converted["fx"] == pytest.approx(
            params["fx"] * 2.0 / float(params["w"])
        ) and converted["fy"] == pytest.approx(
            params["fy"] * 2.0 / float(params["h"])
        ), (
            "Expected fx to be scaled by two over w and fy by two over h. "
            f"{frame=} {converted=}"
        )
    return


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
        assert converted["fx"] == pytest.approx(params["fx"] * unit_x) and converted[
            "fy"
        ] == pytest.approx(params["fy"] * unit_y), (
            "Expected fx and fy to be scaled by their own axis's unit with their signs unchanged. "
            f"{frame=} {converted=}"
        )
    return


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
        assert pinhole == ortho, (
            "Expected a pinhole and an ortho with equal params to convert identically. "
            f"{frame=} {pinhole=} {ortho=}"
        )
    return


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
    ), (
        "Expected pytorch3d to scale the shared f by two over the shorter side. "
        f"{converted=}"
    )
    for frame in ("opengl", "vulkan"):
        with pytest.raises(AssertionError):
            transform_intr_convention(
                params=params,
                model="simple_pinhole",
                source_intr_convention="standard",
                target_intr_convention=frame,
            )
    return


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
    return


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
        standard = transform_intr_convention(
            params=params,
            model="pinhole",
            source_intr_convention=source,
            target_intr_convention="standard",
        )
        via_standard = transform_intr_convention(
            params=standard,
            model="pinhole",
            source_intr_convention="standard",
            target_intr_convention=target,
        )
        for key in direct:
            assert direct[key] == pytest.approx(via_standard[key]), (
                "Expected the direct conversion to match the one through standard. "
                f"{source=} {target=} {key=} {direct=} {via_standard=}"
            )
    return


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
            assert round_tripped.params[key] == pytest.approx(value), (
                "Expected the round-tripped param to equal the original. "
                f"{frame=} {key=} {round_tripped.params=}"
            )
    return


def test_a_converted_intrinsics_still_satisfies_its_own_invariants() -> None:
    """Converted params remain valid for their target frame.

    Args:
        None.

    Returns:
        None.
    """
    pinhole_params = _build_pinhole_params()
    params: Dict[str, torch.Tensor] = {}
    for key, value in pinhole_params.items():
        params[key] = torch.tensor(value, dtype=torch.float32)
    for model in ("pinhole", "ortho"):
        for target_intr_convention in ("standard", "opengl", "pytorch3d", "vulkan"):
            transformed_params = transform_intr_convention(
                params=params,
                model=model,
                source_intr_convention="standard",
                target_intr_convention=target_intr_convention,
            )
            validate_camera_intrinsics_invariants(
                model=model,
                intr_convention=target_intr_convention,
                params=transformed_params,
            )
    return


def test_a_frame_change_is_measured_against_the_intrinsics_own_resolution() -> None:
    """The resolution fixes where a centred origin sits and what a normalized unit is worth.

    Args:
        None.

    Returns:
        None.
    """
    narrow_standard = build_camera_intrinsics(
        model="pinhole",
        params=_build_pinhole_params(height=240, width=320),
        intr_convention="standard",
        device="cpu",
    )
    narrow = narrow_standard.to(intr_convention="opengl")
    wide_standard = build_camera_intrinsics(
        model="pinhole",
        params=_build_pinhole_params(height=240, width=640),
        intr_convention="standard",
        device="cpu",
    )
    wide = wide_standard.to(intr_convention="opengl")
    assert narrow.params["cx"] != pytest.approx(wide.params["cx"]), (
        "Expected two intrinsics whose h and w differ to convert to different results. "
        f"{narrow.params=} {wide.params=}"
    )
    assert (
        narrow.params["h"] == 240
        and narrow.params["w"] == 320
        and wide.params["h"] == 240
        and wide.params["w"] == 640
    ), (
        "Expected each converted intrinsics to carry the h and w it was built with. "
        f"{narrow.params=} {wide.params=}"
    )
    return


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
    params: Dict[str, Union[int, float]] = {
        "fx": 400.0,
        "fy": 410.0,
        "cx": 150.0,
        "cy": 110.0,
        "h": height,
        "w": width,
    }
    return params


def test_an_intrinsics_without_a_resolution_is_refused() -> None:
    """A principal point in the standard frame names a location only against a resolution.

    Args:
        None.

    Returns:
        None.
    """
    projection_only: List[Tuple[str, Dict[str, Union[int, float]]]] = [
        ("simple_pinhole", {"f": 400.0, "cx": 150.0, "cy": 110.0}),
        ("simple_pinhole", {"f": 400.0, "cx": 150.0, "cy": 110.0, "w": 320}),
        ("simple_pinhole", {"f": 400.0, "cx": 150.0, "cy": 110.0, "h": 240}),
        ("pinhole", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0}),
        ("pinhole", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "w": 320}),
        ("pinhole", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "h": 240}),
        ("ortho", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0}),
        ("ortho", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "w": 320}),
        ("ortho", {"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "h": 240}),
    ]
    for model, params in projection_only:
        with pytest.raises(AssertionError):
            build_camera_intrinsics(
                model=model,
                params=params,
                intr_convention="standard",
                device="cpu",
            )
    return


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
    assert both.extrinsics.extr_convention == "opencv", (
        "Expected the returned camera's extr_convention to be the camera-space frame that was named. "
        f"{both.extrinsics=}"
    )
    assert both.intrinsics.intr_convention == "opengl", (
        "Expected the returned camera's intr_convention to be the image-plane frame that was named. "
        f"{both.intrinsics=}"
    )
    pose_only = camera.to(extr_convention="opencv")
    plane_only = camera.to(intr_convention="opengl")
    assert (
        pose_only.extrinsics.extr_convention == "opencv"
        and pose_only.intrinsics.intr_convention == "standard"
        and plane_only.extrinsics.extr_convention == "standard"
        and plane_only.intrinsics.intr_convention == "opengl"
    ), (
        "Expected naming only one of the two frames to set that half and leave the other half's frame unchanged. "
        f"{pose_only.extrinsics=} {pose_only.intrinsics=} {plane_only.extrinsics=} {plane_only.intrinsics=}"
    )
    return


def test_extrinsics_tensor_matrix_stays_differentiable_through_pose_accessors() -> None:
    """Tensor-valued extrinsics stay on the autograd path through the pose accessors.

    Args:
        None.

    Returns:
        None.
    """
    translation = torch.tensor(
        [0.3, -0.2, 1.1], dtype=torch.float32, requires_grad=True
    )
    matrix = torch.cat(
        [
            torch.cat(
                [torch.eye(3, dtype=torch.float32), translation.unsqueeze(1)], dim=1
            ),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        ],
        dim=0,
    )
    extrinsics = CameraExtrinsics(extrinsics=matrix, extr_convention="standard")
    loss = extrinsics.w2c.sum() + extrinsics.center.sum()
    loss.backward()
    assert translation.grad is not None, (
        "Expected the source extrinsics tensor to receive a gradient. "
        f"{translation.grad=}"
    )


def test_extrinsics_constructor_applies_requested_device_dtype_through_to() -> None:
    """CameraExtrinsics.__init__ delegates requested device / dtype movement to to.

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
    assert extrinsics.extrinsics.device == torch.device("cpu"), (
        "Expected the cam2world tensor to carry the requested device. "
        f"{extrinsics.extrinsics.device=}"
    )
    assert extrinsics.extrinsics.dtype == torch.float64, (
        "Expected the cam2world tensor to carry the requested dtype. "
        f"{extrinsics.extrinsics.dtype=}"
    )
    assert extrinsics.device == extrinsics.extrinsics.device, (
        "Expected extrinsics.device to match its tensor state. "
        f"{extrinsics.device=} {extrinsics.extrinsics.device=}"
    )
    assert extrinsics.dtype == extrinsics.extrinsics.dtype, (
        "Expected extrinsics.dtype to match its tensor state. "
        f"{extrinsics.dtype=} {extrinsics.extrinsics.dtype=}"
    )
    return


def test_extrinsics_to_follows_tensor_to_semantics() -> None:
    """CameraExtrinsics.to applies Tensor.to-style device / dtype / copy semantics.

    Args:
        None.

    Returns:
        None.
    """
    extrinsics = CameraExtrinsics(
        extrinsics=_build_extrinsics_matrix(),
        extr_convention="standard",
        device="cpu",
    )
    moved = extrinsics.to(device="cpu", dtype=torch.float64, copy=True)
    assert moved.extrinsics.device == torch.device("cpu"), (
        "Expected the returned cam2world tensor to carry the requested device. "
        f"{moved.extrinsics.device=}"
    )
    assert moved.extrinsics.dtype == torch.float64, (
        "Expected the returned cam2world tensor to carry the requested dtype. "
        f"{moved.extrinsics.dtype=}"
    )
    assert moved.extrinsics.data_ptr() != extrinsics.extrinsics.data_ptr(), (
        "Expected copy=True to allocate storage distinct from the source extrinsics. "
        f"{moved.extrinsics.data_ptr()=} {extrinsics.extrinsics.data_ptr()=}"
    )
    return


def test_camera_and_cameras_to_preserve_tensor_parameter_graphs() -> None:
    """Camera.to and Cameras.to keep tensor intrinsics and extrinsics differentiable.

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
        "h": torch.tensor(240.0, dtype=torch.float32, requires_grad=True),
        "w": torch.tensor(320.0, dtype=torch.float32, requires_grad=True),
    }
    translation = torch.tensor(
        [0.3, -0.2, 1.1], dtype=torch.float32, requires_grad=True
    )
    matrix = torch.cat(
        [
            torch.cat(
                [torch.eye(3, dtype=torch.float32), translation.unsqueeze(1)], dim=1
            ),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        ],
        dim=0,
    )
    intrinsics = build_camera_intrinsics(
        model="ortho", params=params, intr_convention="standard"
    )
    extrinsics = CameraExtrinsics(extrinsics=matrix, extr_convention="standard")
    camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
    moved_camera = camera.to(
        device=intrinsics.device, dtype=torch.float64, extr_convention="pytorch3d"
    )
    cameras = Cameras(intrinsics=intrinsics[None], extrinsics=extrinsics[None])
    moved_cameras = cameras.to(
        device=intrinsics.device, dtype=torch.float64, extr_convention="pytorch3d"
    )
    points_camera = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    loss = (
        moved_camera.intrinsics.project(points_camera=points_camera).sum()
        + moved_cameras.center.sum()
    )
    loss.backward()
    assert (
        params["fx"].grad is not None
        and params["fy"].grad is not None
        and params["cx"].grad is not None
        and params["cy"].grad is not None
    ), (
        "Expected the source fx, fy, cx and cy params to receive gradients. "
        f"{params['fx'].grad=} {params['fy'].grad=} {params['cx'].grad=} {params['cy'].grad=}"
    )
    assert translation.grad is not None, (
        "Expected the source extrinsics tensor to receive a gradient. "
        f"{translation.grad=}"
    )
    return
