import pytest
import torch

from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsicsOrtho,
    CameraIntrinsicsPinhole,
    CameraIntrinsicsSimplePinhole,
    build_camera_intrinsics,
)
from data.structures.three_d.camera.intrinsics.validation import (
    validate_camera_intrinsics_attributes,
    validate_camera_intrinsics_params,
    validate_camera_model,
)


def test_validate_camera_model_accepts_all_supported() -> None:
    """validate_camera_model accepts simple_pinhole, pinhole, and ortho.

    Args:
        None.

    Returns:
        None.
    """
    for model in ("simple_pinhole", "pinhole", "ortho"):
        assert validate_camera_model(model=model) == model, f"{model=}"


def test_validate_camera_model_rejects_unsupported() -> None:
    """validate_camera_model raises on a model string outside the supported set.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(AssertionError):
        validate_camera_model(model="fisheye")


def test_validate_intrinsics_params_dispatches_per_model_keys() -> None:
    """validate_camera_intrinsics_params enforces each model's parameter keys.

    Args:
        None.

    Returns:
        None.
    """
    simple_params = {"f": 400.0, "cx": 160.0, "cy": 120.0}
    pinhole_params = {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}
    assert (
        validate_camera_intrinsics_params(model="simple_pinhole", params=simple_params)
        == simple_params
    ), f"{simple_params=}"
    assert (
        validate_camera_intrinsics_params(model="pinhole", params=pinhole_params)
        == pinhole_params
    ), f"{pinhole_params=}"
    assert (
        validate_camera_intrinsics_params(model="ortho", params=pinhole_params)
        == pinhole_params
    ), f"{pinhole_params=}"

    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(model="simple_pinhole", params=pinhole_params)
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(model="pinhole", params=simple_params)


def test_validate_intrinsics_attributes_checks_model_params_device() -> None:
    """validate_camera_intrinsics_attributes validates model, params, and device.

    Args:
        None.

    Returns:
        None.
    """
    params = {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}
    validate_camera_intrinsics_attributes(model="pinhole", params=params, device="cpu")
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(model="pinhole", params=params, device=0)
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(
            model="pinhole",
            params={"f": 400.0, "cx": 160.0, "cy": 120.0},
            device="cpu",
        )


def test_build_camera_intrinsics_dispatches_to_model_subclass() -> None:
    """build_camera_intrinsics returns the subclass instance for its model string.

    Args:
        None.

    Returns:
        None.
    """
    simple = build_camera_intrinsics(
        model="simple_pinhole",
        params={"f": 400.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    pinhole = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    ortho = build_camera_intrinsics(
        model="ortho",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    assert isinstance(simple, CameraIntrinsicsSimplePinhole), f"{type(simple)=}"
    assert isinstance(pinhole, CameraIntrinsicsPinhole), f"{type(pinhole)=}"
    assert isinstance(ortho, CameraIntrinsicsOrtho), f"{type(ortho)=}"


def test_simple_pinhole_project_applies_perspective_divide() -> None:
    """CameraIntrinsicsSimplePinhole.project applies the perspective divide.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsSimplePinhole(
        params={"f": 400.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    points = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    image = intrinsics.project(points_camera=points)
    expected = torch.tensor([[400.0 * 1.0 / 4.0 + 160.0, 400.0 * 2.0 / 4.0 + 120.0]])
    assert torch.allclose(image, expected, atol=1.0e-05), f"{image=} {expected=}"


def test_pinhole_project_applies_perspective_divide() -> None:
    """CameraIntrinsicsPinhole.project applies the perspective divide with fx / fy.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    points = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    image = intrinsics.project(points_camera=points)
    expected = torch.tensor([[400.0 * 1.0 / 4.0 + 160.0, 410.0 * 2.0 / 4.0 + 120.0]])
    assert torch.allclose(image, expected, atol=1.0e-05), f"{image=} {expected=}"


def test_ortho_project_skips_perspective_divide() -> None:
    """CameraIntrinsicsOrtho.project maps points without the perspective divide.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    near = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    far = torch.tensor([[1.0, 2.0, 40.0]], dtype=torch.float32)
    image_near = intrinsics.project(points_camera=near)
    image_far = intrinsics.project(points_camera=far)
    expected = torch.tensor([[400.0 * 1.0 + 160.0, 410.0 * 2.0 + 120.0]])
    assert torch.allclose(image_near, expected, atol=1.0e-05), f"{image_near=}"
    assert torch.allclose(image_near, image_far, atol=1.0e-05), (
        "Ortho projection must ignore depth (no perspective divide). "
        f"{image_near=} {image_far=}"
    )


def test_fx_fy_cx_cy_derived_from_params() -> None:
    """The fx / fy accessors and the cx / cy accessors are derived from params.

    Args:
        None.

    Returns:
        None.
    """
    simple = CameraIntrinsicsSimplePinhole(
        params={"f": 400.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    assert simple.fx == 400.0 and simple.fy == 400.0, f"{simple.fx=} {simple.fy=}"
    assert simple.cx == 160.0 and simple.cy == 120.0, f"{simple.cx=} {simple.cy=}"

    pinhole = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    assert pinhole.fx == 400.0 and pinhole.fy == 410.0, f"{pinhole.fx=} {pinhole.fy=}"
    assert pinhole.cx == 160.0 and pinhole.cy == 120.0, f"{pinhole.cx=} {pinhole.cy=}"

    ortho = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    assert ortho.fx == 400.0 and ortho.fy == 410.0, f"{ortho.fx=} {ortho.fy=}"
    assert ortho.cx == 160.0 and ortho.cy == 120.0, f"{ortho.cx=} {ortho.cy=}"


def test_fov_defined_for_perspective_subclasses_only() -> None:
    """The perspective subclasses expose fov in degrees while ortho has none.

    Args:
        None.

    Returns:
        None.
    """
    simple = CameraIntrinsicsSimplePinhole(
        params={"f": 400.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    pinhole = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    ortho = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )
    assert isinstance(simple.fov, tuple) and len(simple.fov) == 2, f"{simple.fov=}"
    assert isinstance(pinhole.fov, tuple) and len(pinhole.fov) == 2, f"{pinhole.fov=}"
    assert all(isinstance(value, float) for value in simple.fov), f"{simple.fov=}"
    assert all(isinstance(value, float) for value in pinhole.fov), f"{pinhole.fov=}"
    assert not hasattr(ortho, "fov"), "Ortho intrinsics must not expose fov."


def test_scale_intrinsics_scales_focal_and_principal_point() -> None:
    """CameraIntrinsics.scale_intrinsics scales focal and principal point.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        device="cpu",
    )

    by_factor = intrinsics.scale_intrinsics(scale=2.0)
    assert by_factor.params == {
        "fx": 800.0,
        "fy": 820.0,
        "cx": 320.0,
        "cy": 240.0,
    }, f"{by_factor.params=}"
    assert isinstance(by_factor, CameraIntrinsicsPinhole), f"{type(by_factor)=}"

    by_axes = intrinsics.scale_intrinsics(scale=(2.0, 0.5))
    assert by_axes.params == {
        "fx": 800.0,
        "fy": 205.0,
        "cx": 320.0,
        "cy": 60.0,
    }, f"{by_axes.params=}"

    # Current resolution inferred from the principal point is (W, H) = (320, 240).
    by_resolution = intrinsics.scale_intrinsics(resolution=(480, 640))
    assert by_resolution.params == {
        "fx": 800.0,
        "fy": 820.0,
        "cx": 320.0,
        "cy": 240.0,
    }, f"{by_resolution.params=}"
