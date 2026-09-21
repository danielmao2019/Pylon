import ast
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
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
    validate_camera_intrinsics_invariants,
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
        accepted = validate_camera_model(model=model)
        assert accepted == model, (
            "Expected validate_camera_model to return the model it accepted. "
            f"{model=}"
        )
    return


def test_validate_camera_model_rejects_unsupported() -> None:
    """validate_camera_model raises on a camera-model string outside the supported set.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(AssertionError):
        validate_camera_model(model="fisheye")
    return


def test_validate_intrinsics_params_dispatches_per_model_tensor_keys() -> None:
    """validate_camera_intrinsics_params enforces each model's named scalar tensor keys beside the h and w every model carries.

    Args:
        None.

    Returns:
        None.
    """
    model_params: Dict[str, Dict[str, Union[int, float]]] = {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }
    foreign_model: Dict[str, str] = {
        "simple_pinhole": "pinhole",
        "pinhole": "simple_pinhole",
        "ortho": "simple_pinhole",
    }
    for model, numeric_params in model_params.items():
        params = _tensor_params(params=numeric_params)
        accepted = validate_camera_intrinsics_params(
            model=model, intr_convention="standard", params=params
        )
        assert accepted == params, (
            "Expected the validated params to be the accepted params dict. "
            f"{model=} {set(accepted.keys())=} {set(params.keys())=}"
        )
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_params(
                model=model,
                intr_convention="standard",
                params=_tensor_params(params=model_params[foreign_model[model]]),
            )
    return


def test_intrinsics_params_carry_one_shared_batch_axis() -> None:
    """A camera's params are scalars or a [B] batch, all sharing one leading shape, so a batch of cameras is one intrinsics rather than a list of them.

    Args:
        None.

    Returns:
        None.
    """
    batched_params = _tensor_params(
        params={
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        batch_size=3,
    )
    accepted = validate_camera_intrinsics_params(
        model="pinhole", intr_convention="standard", params=batched_params
    )
    assert accepted == batched_params, (
        "Expected the validated batched params to be the accepted params dict. "
        f"{set(accepted.keys())=} {set(batched_params.keys())=}"
    )

    mismatched_params: Dict[str, torch.Tensor] = {
        **batched_params,
        **_tensor_params(params={"fy": 410.0}, batch_size=2),
    }
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(
            model="pinhole", intr_convention="standard", params=mismatched_params
        )


def test_batched_intrinsics_carry_the_batch_through_its_accessors_and_project() -> None:
    """Every derived quantity a batched intrinsics reports carries the batch axis, so one intrinsics answers for all its cameras in one call.

    Args:
        None.

    Returns:
        None.
    """
    batched_params = _tensor_params(
        params={
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        batch_size=3,
    )
    intrinsics = build_camera_intrinsics(
        model="pinhole", params=batched_params, intr_convention="standard"
    )
    height, width = intrinsics.resolution
    for name, derived in (
        ("fx", intrinsics.fx),
        ("fy", intrinsics.fy),
        ("cx", intrinsics.cx),
        ("cy", intrinsics.cy),
        ("h", height),
        ("w", width),
    ):
        assert derived.shape == (3,), (
            "Expected every derived quantity of a three-camera batch to carry the "
            f"batch axis. {name=} {derived.shape=}"
        )

    points_camera = torch.tensor(
        [
            [[1.0, 2.0, 4.0], [3.0, -1.0, 8.0]],
            [[2.0, 1.0, 5.0], [-1.0, 3.0, 10.0]],
            [[0.5, 0.5, 2.0], [4.0, 4.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    image_points = intrinsics.project(points_camera=points_camera, inplace=False)
    assert image_points.shape == (3, 2, 2), (
        "Expected the image points to carry the batch axis ahead of the point axis. "
        f"{image_points.shape=} {points_camera.shape=}"
    )

    for index in range(3):
        one_params: Dict[str, torch.Tensor] = {}
        for key, value in batched_params.items():
            one_params[key] = value[index]
        one_camera = build_camera_intrinsics(
            model="pinhole", params=one_params, intr_convention="standard"
        )
        one_camera_image_points = one_camera.project(
            points_camera=points_camera[index], inplace=False
        )
        assert torch.equal(image_points[index], one_camera_image_points), (
            "Expected the batched image points slice to equal that camera's own "
            f"projection. {index=} {image_points[index]=} {one_camera_image_points=}"
        )


def test_scale_intrinsics_rescales_a_batch_against_each_cameras_own_resolution() -> (
    None
):
    """A batch states one resolution per camera, so a shared factor lands on each camera's own raster rather than on one resolution the batch does not have.

    Args:
        None.

    Returns:
        None.
    """
    batched_params = _tensor_params(
        params={
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": [240, 300, 360],
            "w": [320, 400, 480],
        },
        batch_size=3,
    )
    intrinsics = build_camera_intrinsics(
        model="pinhole", params=batched_params, intr_convention="standard"
    )
    scaled = intrinsics.scale_intrinsics(scale=2.0)

    height, width = scaled.resolution
    assert height.shape == (3,) and width.shape == (3,), (
        "Expected both scaled resolution sides to carry the batch axis. "
        f"{height.shape=} {width.shape=}"
    )
    assert torch.equal(height, 2.0 * batched_params["h"]) and torch.equal(
        width, 2.0 * batched_params["w"]
    ), (
        "Expected each scaled resolution side to be that camera's own side scaled. "
        f"{height=} {width=} {batched_params['h']=} {batched_params['w']=}"
    )
    assert torch.equal(scaled.fx, 2.0 * batched_params["fx"]) and torch.equal(
        scaled.fy, 2.0 * batched_params["fy"]
    ), (
        "Expected each scaled focal to be that camera's own focal scaled. "
        f"{scaled.fx=} {scaled.fy=} {batched_params['fx']=} {batched_params['fy']=}"
    )


def test_validate_intrinsics_params_rejects_a_params_dict_missing_the_resolution() -> (
    None
):
    """h and w are two of every model's own params rather than a resolution supplied beside them, so a dict carrying the projection keys alone is rejected ahead of the model's own dispatch.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0},
        "pinhole": {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        "ortho": {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
    }.items():
        # The resolution check's own message, since the model's key dispatch refuses this dict too and only the message shows which of the two fired first.
        with pytest.raises(AssertionError, match="resolution keys h and w"):
            validate_camera_intrinsics_params(
                model=model,
                intr_convention="standard",
                params=_tensor_params(params=params),
            )
    return


def test_the_principal_point_must_lie_on_the_image_in_its_own_frames_extent() -> None:
    """A principal point is where the optical axis meets the image, so it lies on the image — and what that bound is depends on the frame, which is why the check reads the two together rather than either alone.

    Args:
        None.

    Returns:
        None.
    """

    # standard: the pixel frame running corner to corner.
    validate_camera_intrinsics_invariants(
        model="pinhole",
        intr_convention="standard",
        params=_tensor_params(
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 320.0,
                "cy": 240.0,
                "h": 240,
                "w": 320,
            }
        ),
    )
    for cx, cy in ((320.5, 240.0), (320.0, 240.5)):
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_invariants(
                model="pinhole",
                intr_convention="standard",
                params=_tensor_params(
                    params={
                        "fx": 400.0,
                        "fy": 410.0,
                        "cx": cx,
                        "cy": cy,
                        "h": 240,
                        "w": 320,
                    }
                ),
            )

    # opengl / vulkan: each axis normalized by its own side, so both bounds are the same.
    for frame in ("opengl", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="pinhole",
            intr_convention=frame,
            params=_tensor_params(
                params={
                    "fx": 400.0,
                    "fy": 410.0,
                    "cx": 1.0,
                    "cy": -1.0,
                    "h": 240,
                    "w": 320,
                }
            ),
        )
        for cx, cy in ((1.5, 0.0), (0.0, -1.5)):
            with pytest.raises(AssertionError):
                validate_camera_intrinsics_invariants(
                    model="pinhole",
                    intr_convention=frame,
                    params=_tensor_params(
                        params={
                            "fx": 400.0,
                            "fy": 410.0,
                            "cx": cx,
                            "cy": cy,
                            "h": 240,
                            "w": 320,
                        }
                    ),
                )

    # pytorch3d: the shorter side alone reaches 1, so the longer axis's bound is the larger.
    validate_camera_intrinsics_invariants(
        model="pinhole",
        intr_convention="pytorch3d",
        params=_tensor_params(
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 1.25,
                "cy": 1.0,
                "h": 240,
                "w": 320,
            }
        ),
    )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_invariants(
            model="pinhole",
            intr_convention="pytorch3d",
            params=_tensor_params(
                params={
                    "fx": 400.0,
                    "fy": 410.0,
                    "cx": 0.0,
                    "cy": 1.25,
                    "h": 240,
                    "w": 320,
                }
            ),
        )

    # ortho: cx / cy name where the world origin lands, so any finite pair is valid.
    for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="ortho",
            intr_convention=frame,
            params=_tensor_params(
                params={
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": -37.5,
                    "cy": 512.0,
                    "h": 240,
                    "w": 240,
                }
            ),
        )
    return


def test_a_centred_principal_point_survives_its_models_own_key_dispatch() -> None:
    """Every frame but standard puts the origin at the image's centre, so half of it carries a negative principal point — which the per-model key dispatch must not read as out of range, that bound belonging to the frame alone.

    Args:
        None.

    Returns:
        None.
    """
    centred: Dict[str, Dict[str, torch.Tensor]] = {
        "simple_pinhole": _tensor_params(
            params={"f": 2.0, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240}
        ),
        "pinhole": _tensor_params(
            params={"fx": 2.0, "fy": 2.5, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240}
        ),
        "ortho": _tensor_params(
            params={"fx": 2.0, "fy": 2.5, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240}
        ),
    }
    for model, params in centred.items():
        accepted = validate_camera_intrinsics_params(
            model=model, intr_convention="opengl", params=params
        )
        assert accepted == params, (
            "Expected a centred principal point to pass its model's key dispatch unchanged. "
            f"{model=} {params=}"
        )
    return


def test_a_frame_that_scales_the_axes_apart_cannot_hold_a_shared_focal() -> None:
    """A model states as many focal params as it has axes to scale independently, so opengl and vulkan, which normalize each axis by its own side, hold a simple_pinhole only on a square image.

    Args:
        None.

    Returns:
        None.
    """

    for frame in ("opengl", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="simple_pinhole",
            intr_convention=frame,
            params=_tensor_params(
                params={"f": 2.0, "cx": 0.0, "cy": 0.0, "h": 240, "w": 240}
            ),
        )
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_invariants(
                model="simple_pinhole",
                intr_convention=frame,
                params=_tensor_params(
                    params={"f": 2.0, "cx": 0.0, "cy": 0.0, "h": 240, "w": 320}
                ),
            )
    for height, width in ((240, 240), (240, 320)):
        validate_camera_intrinsics_invariants(
            model="simple_pinhole",
            intr_convention="pytorch3d",
            params=_tensor_params(
                params={"f": 2.0, "cx": 0.0, "cy": 0.0, "h": height, "w": width}
            ),
        )
        for model in ("pinhole", "ortho"):
            for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
                validate_camera_intrinsics_invariants(
                    model=model,
                    intr_convention=frame,
                    params=_tensor_params(
                        params={
                            "fx": 2.0,
                            "fy": 2.5,
                            "cx": 0.0,
                            "cy": 0.0,
                            "h": height,
                            "w": width,
                        }
                    ),
                )
    return


def test_validate_intrinsics_attributes_checks_model_intr_convention_params_device_dtype() -> (
    None
):
    """validate_camera_intrinsics_attributes validates the camera model, image-plane frame, tensor params, device, and dtype together.

    Args:
        None.

    Returns:
        None.
    """
    params = _tensor_params(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320}
    )
    validate_camera_intrinsics_attributes(
        model="pinhole",
        intr_convention="standard",
        params=params,
        device="cpu",
        dtype=torch.float32,
    )
    # Each row breaks one attribute and keeps the rest valid; the params row's params are replaced by a simple_pinhole key set inside the case.
    for broken, model, intr_convention, case_params, device, dtype in (
        ("model", "fisheye", "standard", params, "cpu", torch.float32),
        ("intr_convention", "pinhole", "ndc", params, "cpu", torch.float32),
        ("params", "pinhole", "standard", params, "cpu", torch.float32),
        ("device", "pinhole", "standard", params, 0, torch.float32),
        ("dtype", "pinhole", "standard", params, "cpu", torch.int64),
    ):
        with pytest.raises(AssertionError):
            if broken == "params":
                case_params = _tensor_params(
                    params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320}
                )
            validate_camera_intrinsics_attributes(
                model=model,
                intr_convention=intr_convention,
                params=case_params,
                device=device,
                dtype=dtype,
            )
    return


def test_intrinsics_constructor_normalizes_scalar_compatible_params_to_tensors() -> (
    None
):
    """build_camera_intrinsics turns Python, numpy 0-d and tensor scalar params into 0-d tensors of the requested dtype, the intrinsics landing on the requested device and a tensor param keeping its autograd path through project.

    Args:
        None.

    Returns:
        None.
    """
    fx = torch.tensor(400.0, dtype=torch.float64, requires_grad=True)
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            "fx": fx,
            "fy": np.array(410.0, dtype=np.float64),
            "cx": 160.0,
            "cy": np.array(120.0, dtype=np.float64),
            "h": 240,
            "w": 320,
        },
        intr_convention="standard",
        device="cpu",
        dtype=torch.float64,
    )
    assert intrinsics.dtype == torch.float64, (
        "Expected the intrinsics to carry the requested dtype. " f"{intrinsics.dtype=}"
    )
    assert intrinsics.device == torch.device("cpu"), (
        "Expected the intrinsics to land on the requested device. "
        f"{intrinsics.device=}"
    )
    for key, value in intrinsics.params.items():
        assert isinstance(value, torch.Tensor), (
            "Expected every scalar-compatible param to become a torch.Tensor. "
            f"{key=} {type(value)=}"
        )
        assert value.shape == (), (
            "Expected every scalar param to become a 0-d tensor. "
            f"{key=} {value.shape=}"
        )
        assert value.dtype == torch.float64, (
            "Expected every param to carry the requested dtype. "
            f"{key=} {value.dtype=}"
        )
    loss = intrinsics.project(
        points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    ).sum()
    loss.backward()
    assert fx.grad is not None, (
        "Expected the source tensor param to receive grad. " f"{fx.grad=}"
    )
    return


def test_build_camera_intrinsics_dispatches_to_model_subclass() -> None:
    """build_camera_intrinsics returns the CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole / CameraIntrinsicsOrtho instance for its model string.

    Args:
        None.

    Returns:
        None.
    """
    for model, subclass, params in (
        (
            "simple_pinhole",
            CameraIntrinsicsSimplePinhole,
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        ),
        (
            "pinhole",
            CameraIntrinsicsPinhole,
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        ),
        (
            "ortho",
            CameraIntrinsicsOrtho,
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        ),
    ):
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        assert type(intrinsics) is subclass, (
            "Expected build_camera_intrinsics to return its model's subclass. "
            f"{model=} {type(intrinsics)=} {subclass=}"
        )
        assert intrinsics.model == model, (
            "Expected the built intrinsics to report the model string it was built for. "
            f"{model=} {intrinsics.model=}"
        )
    return


def test_intrinsics_constructor_applies_requested_device_dtype_through_to() -> None:
    """CameraIntrinsics.__init__ delegates requested device / dtype movement to the object's to method.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=_tensor_params(
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            }
        ),
        intr_convention="standard",
        device="cpu",
        dtype=torch.float64,
    )
    for key, value in intrinsics.params.items():
        assert value.device == torch.device("cpu"), (
            "Expected every param tensor to carry the requested device. "
            f"{key=} {value.device=}"
        )
        assert value.dtype == torch.float64, (
            "Expected every param tensor to carry the requested dtype. "
            f"{key=} {value.dtype=}"
        )
    assert intrinsics.device == torch.device("cpu"), (
        "Expected intrinsics.device to match the placed param tensors. "
        f"{intrinsics.device=}"
    )
    assert intrinsics.dtype == torch.float64, (
        "Expected intrinsics.dtype to match the placed param tensors. "
        f"{intrinsics.dtype=}"
    )
    return


def test_intrinsics_to_follows_tensor_to_semantics() -> None:
    """CameraIntrinsics.to applies Tensor.to-style device / dtype / copy semantics to every scalar param tensor.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params=_tensor_params(
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            }
        ),
        intr_convention="standard",
    )
    # The params' own device and dtype, so only copy=True can give the returned params storage of their own.
    moved = intrinsics.to(device="cpu", dtype=torch.float32, copy=True)
    for key, value in moved.params.items():
        assert value.device == torch.device("cpu"), (
            "Expected every returned param tensor to carry the requested device. "
            f"{key=} {value.device=}"
        )
        assert value.dtype == torch.float32, (
            "Expected every returned param tensor to carry the requested dtype. "
            f"{key=} {value.dtype=}"
        )
        assert value.data_ptr() != intrinsics.params[key].data_ptr(), (
            "Expected copy=True to allocate storage distinct from the source params. "
            f"{key=} {value.data_ptr()=} {intrinsics.params[key].data_ptr()=}"
        )
    return


def test_simple_pinhole_project_applies_perspective_divide() -> None:
    """CameraIntrinsicsSimplePinhole.project applies the perspective divide with a single shared focal length.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsSimplePinhole(
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    image = intrinsics.project(
        points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    )
    expected = torch.tensor([[400.0 * 1.0 / 4.0 + 160.0, 400.0 * 2.0 / 4.0 + 120.0]])
    assert torch.allclose(image, expected, atol=1.0e-05), (
        "Expected the simple_pinhole image points to be f * x / z + cx and f * y / z + cy. "
        f"{image=} {expected=}"
    )
    return


def test_pinhole_project_applies_perspective_divide() -> None:
    """CameraIntrinsicsPinhole.project applies the perspective divide with independent fx / fy.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    image = intrinsics.project(
        points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    )
    expected = torch.tensor([[400.0 * 1.0 / 4.0 + 160.0, 410.0 * 2.0 / 4.0 + 120.0]])
    assert torch.allclose(image, expected, atol=1.0e-05), (
        "Expected the pinhole image points to be fx * x / z + cx and fy * y / z + cy. "
        f"{image=} {expected=}"
    )
    return


def test_ortho_project_skips_perspective_divide() -> None:
    """CameraIntrinsicsOrtho.project maps points without the perspective divide.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    image_near = intrinsics.project(
        points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    )
    image_far = intrinsics.project(
        points_camera=torch.tensor([[1.0, 2.0, 40.0]], dtype=torch.float32)
    )
    expected = torch.tensor([[400.0 * 1.0 + 160.0, 410.0 * 2.0 + 120.0]])
    assert torch.allclose(image_near, expected, atol=1.0e-05), (
        "Expected the ortho image points to be fx * x + cx and fy * y + cy. "
        f"{image_near=} {expected=}"
    )
    assert torch.allclose(image_near, image_far, atol=1.0e-05), (
        "Ortho projection must ignore depth (no perspective divide). "
        f"{image_near=} {image_far=}"
    )
    return


_REPO_ROOT = Path(__file__).resolve().parents[6]
_REPO_SOURCE_ROOTS = (
    "agents",
    "criteria",
    "data",
    "metrics",
    "models",
    "optimizers",
    "pylon",
    "runners",
    "schedulers",
    "utils",
)
_CAMERA_DEPTH_NAMES = {"d", "depth", "depths", "z", "zs", "z_cam", "z_camera"}


def test_every_camera_consumer_projects_through_the_camera() -> None:
    """A camera's image coordinates come from its own project, so a consumer forming them another way carries a second copy of one model's formula.

    Args:
        None.

    Returns:
        None.
    """
    owner = (
        _REPO_ROOT / "data/structures/three_d/camera/intrinsics/camera_intrinsics.py"
    )
    consumers: Dict[str, Tuple[bool, bool]] = {}
    for source_root in _REPO_SOURCE_ROOTS:
        for path in sorted((_REPO_ROOT / source_root).rglob("*.py")):
            if path == owner or "tests" in path.parts or "test" in path.parts:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
            projects, divides = _classify_camera_module(tree=tree)
            if projects or divides:
                consumers[str(path.relative_to(_REPO_ROOT))] = (projects, divides)

    assert consumers, (
        "Expected the scan to find repo-owned camera consumers. " f"{_REPO_ROOT=}"
    )
    hand_rolled: Set[str] = {
        relpath for relpath, (_, divides) in consumers.items() if divides
    }
    assert hand_rolled == set(), (
        "Expected every consumer's image coordinates to come from a project call "
        "rather than from a perspective divide it performs on camera depth. "
        f"{sorted(hand_rolled)=} {sorted(consumers)=}"
    )
    return


def _classify_camera_module(tree: ast.Module) -> Tuple[bool, bool]:
    """Classifies one module by how it reaches image coordinates.

    Args:
        tree: The parsed module.

    Returns:
        A ``(projects_through_the_camera, divides_by_a_camera_depth_itself)`` pair, the second flag set only when the module also reads a camera's focal length and principal point.
    """
    projects = False
    reads_focal = False
    reads_principal_point = False
    divides = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and "intrinsic" in ast.unparse(node.value).lower()
        ):
            reads_focal = reads_focal or node.attr in {"f", "fx", "fy"}
            reads_principal_point = reads_principal_point or node.attr in {"cx", "cy"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "project" and "intrinsic" in ast.unparse(
                node.func.value
            ):
                projects = True
            if (
                node.func.attr in {"div", "div_"}
                and node.args
                and _is_a_camera_depth(node=node.args[0])
            ):
                divides = True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and _is_a_camera_depth(node=node.right)
        ):
            divides = True
    return projects, divides and reads_focal and reads_principal_point


def _is_a_camera_depth(node: ast.expr) -> bool:
    """Decides whether an expression names camera-space depth.

    Args:
        node: The denominator expression of a division as an ast node.

    Returns:
        True when the expression names depth by name or reads the third component off a points-like tensor, else False.
    """
    if isinstance(node, ast.Name):
        return node.id in _CAMERA_DEPTH_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _CAMERA_DEPTH_NAMES
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Attribute) and _is_a_camera_depth(
            node=node.func.value
        )
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    if isinstance(index, ast.Tuple) and index.elts:
        index = index.elts[-1]
    if isinstance(index, ast.Slice):
        index = index.lower
    if not isinstance(index, ast.Constant) or index.value != 2:
        return False
    base = ast.unparse(node.value).lower()
    return any(word in base for word in ("point", "cam", "vert", "xyz"))


def test_project_inplace_overwrites_input_and_matches_not_inplace() -> None:
    """project(inplace=True) overwrites points_camera cols 0,1 with the image points (matching inplace=False), preserves the depth col 2, and returns a tensor aliasing the input, across all three models.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        points_camera = torch.tensor(
            [[1.0, 2.0, 4.0], [3.0, -1.0, 8.0]], dtype=torch.float32
        )
        reference = points_camera.clone()
        expected = intrinsics.project(points_camera=reference.clone(), inplace=False)
        result = intrinsics.project(points_camera=points_camera, inplace=True)
        assert torch.allclose(points_camera[:, :2], expected), (
            "Expected the first two input columns to be overwritten with the "
            "not-inplace image points. "
            f"{model=} {points_camera=} {expected=}"
        )
        assert torch.equal(points_camera[:, 2], reference[:, 2]), (
            "Expected the input depth column to be preserved. "
            f"{model=} {points_camera=} {reference=}"
        )
        assert result.data_ptr() == points_camera.data_ptr(), (
            "Expected the inplace result to alias the input tensor. "
            f"{model=} {result.data_ptr()=} {points_camera.data_ptr()=}"
        )
    return


def test_project_not_inplace_preserves_input_and_returns_new_tensor() -> None:
    """project(inplace=False) returns a fresh [..., 2] and leaves points_camera unchanged, across all three models.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        points_camera = torch.tensor(
            [[1.0, 2.0, 4.0], [3.0, -1.0, 8.0]], dtype=torch.float32
        )
        reference = points_camera.clone()
        result = intrinsics.project(points_camera=points_camera, inplace=False)
        assert result.shape == (*points_camera.shape[:-1], 2), (
            "Expected the not-inplace result to be a [..., 2] tensor over the "
            "input's leading dims. "
            f"{model=} {result.shape=} {points_camera.shape=}"
        )
        assert torch.equal(points_camera, reference), (
            "Expected the input tensor to be left unchanged. "
            f"{model=} {points_camera=} {reference=}"
        )
        assert result.data_ptr() != points_camera.data_ptr(), (
            "Expected the not-inplace result to be a freshly allocated tensor. "
            f"{model=} {result.data_ptr()=} {points_camera.data_ptr()=}"
        )
    return


def test_project_supports_batched_leading_dims() -> None:
    """project handles [..., 3] leading dims: a batched input (inplace and not-inplace) matches projecting its flattened [N, 3] view, across all three models.

    Args:
        None.

    Returns:
        None.
    """
    batched = torch.tensor(
        [
            [[1.0, 2.0, 4.0], [3.0, -1.0, 8.0], [0.5, 0.5, 2.0]],
            [[2.0, 1.0, 5.0], [-1.0, 3.0, 10.0], [4.0, 4.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        for inplace in (False, True):
            result = intrinsics.project(points_camera=batched.clone(), inplace=inplace)
            flat = intrinsics.project(
                points_camera=batched.clone().reshape(-1, 3), inplace=inplace
            )
            assert torch.allclose(result.reshape(-1, 2), flat), (
                "Expected the batched image points to match projecting the "
                "flattened [B * M, 3] view. "
                f"{model=} {inplace=} {result=} {flat=}"
            )
    return


def test_project_rejects_invalid_inputs() -> None:
    """project raises AssertionError on a non-tensor points_camera, a wrong last dim, and a non-bool inplace, across all three models.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        for points_camera, inplace in (
            ([[1.0, 2.0, 4.0]], False),
            (torch.zeros(4, 2, dtype=torch.float32), False),
            (torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32), 1),
        ):
            with pytest.raises(AssertionError):
                intrinsics.project(points_camera=points_camera, inplace=inplace)
    return


def test_fx_fy_cx_cy_derived_from_params() -> None:
    """The per-subclass fx / fy accessors and the base cx / cy accessors are derived from the model params.

    Args:
        None.

    Returns:
        None.
    """
    for model, params, fx_key, fy_key in (
        (
            "simple_pinhole",
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            "f",
            "f",
        ),
        (
            "pinhole",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            "fx",
            "fy",
        ),
        (
            "ortho",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            "fx",
            "fy",
        ),
    ):
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        assert intrinsics.fx == params[fx_key] and intrinsics.fy == params[fy_key], (
            "Expected fx and fy to read that model's focal keys. "
            f"{model=} {intrinsics.fx=} {intrinsics.fy=} {params=}"
        )
        assert intrinsics.cx == params["cx"] and intrinsics.cy == params["cy"], (
            "Expected cx and cy to read that model's cx and cy params. "
            f"{model=} {intrinsics.cx=} {intrinsics.cy=} {params=}"
        )
    return


def test_fov_defined_for_perspective_subclasses_only() -> None:
    """CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole expose fov in degrees while CameraIntrinsicsOrtho has no fov method.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        assert (
            isinstance(intrinsics.fov, tuple)
            and len(intrinsics.fov) == 2
            and isinstance(intrinsics.fov[0], torch.Tensor)
            and isinstance(intrinsics.fov[1], torch.Tensor)
        ), (
            "Expected fov to be a (horizontal, vertical) pair of tensors. "
            f"{model=} {intrinsics.fov=}"
        )
        expected_horizontal = (
            2.0 * torch.atan(intrinsics.cx / intrinsics.fx) * 180.0 / np.pi
        )
        expected_vertical = (
            2.0 * torch.atan(intrinsics.cy / intrinsics.fy) * 180.0 / np.pi
        )
        assert torch.isclose(
            intrinsics.fov[0], expected_horizontal, rtol=1.0e-09
        ) and torch.isclose(intrinsics.fov[1], expected_vertical, rtol=1.0e-09), (
            "Expected each fov angle to match the one its focal length and principal "
            "point imply. "
            f"{model=} {intrinsics.fov=} {expected_horizontal=} {expected_vertical=}"
        )
    ortho_intrinsics = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    assert hasattr(ortho_intrinsics, "fov") is False, (
        "Ortho intrinsics must not expose fov. " f"{type(ortho_intrinsics)=}"
    )
    return


def test_transform_intrinsics_restates_the_camera_onto_the_named_raster() -> None:
    """An affine between two rasters says how coordinates move but not what image they land on, so the raster is named beside it and becomes the h and w the result carries.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    transformed = intrinsics.transform_intrinsics(
        transform=torch.tensor(
            [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        resolution=(480, 640),
    )
    assert transformed.params["h"] == 480 and transformed.params["w"] == 640, (
        "Expected the returned h and w params to be the named raster's. "
        f"{transformed.params=}"
    )
    assert transformed.params["cx"] == pytest.approx(160.0) and transformed.params[
        "cy"
    ] == pytest.approx(130.0), (
        "Expected the returned principal point to be where the affine sends the original's. "
        f"{transformed.params=}"
    )
    return


def test_transform_intrinsics_returns_the_frame_it_was_given() -> None:
    """Applying a transform says nothing about which image-plane frame a caller states its camera in, so the result comes back on the frame it went in on rather than on the pixel frame the composition happens in.

    Args:
        None.

    Returns:
        None.
    """
    for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
        intrinsics = build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": 400.0,
                "fy": 410.0,
                "cx": 150.0,
                "cy": 110.0,
                "h": 240,
                "w": 320,
            },
            intr_convention="standard",
            device="cpu",
        )
        intrinsics = intrinsics.to(intr_convention=frame)
        transformed = intrinsics.transform_intrinsics(
            transform=torch.eye(3, dtype=torch.float32),
            resolution=intrinsics.resolution,
        )
        assert transformed.intr_convention == frame, (
            "Expected the transformed intrinsics to come back on the frame it was given. "
            f"{frame=} {transformed.intr_convention=}"
        )
        for key, value in intrinsics.params.items():
            assert transformed.params[key] == pytest.approx(value), (
                "Expected an identity transform to return every param unchanged. "
                f"{frame=} {key=} {transformed.params[key]=} {value=}"
            )
    return


def test_a_shared_focal_refuses_a_transform_that_scales_the_axes_apart() -> None:
    """simple_pinhole states one f for both axes, so an affine whose two diagonal entries differ has nowhere to put the second and aborts rather than picking one.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="simple_pinhole",
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    with pytest.raises(AssertionError):
        intrinsics.transform_intrinsics(
            transform=torch.tensor(
                [[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            resolution=(120, 640),
        )
    return


def test_transform_intrinsics_refuses_a_sheared_affine() -> None:
    """A skew-free K stays skew-free only under an axis-aligned affine, so an affine carrying an off-diagonal entry aborts rather than having it dropped.

    Args:
        None.

    Returns:
        None.
    """
    for model, params in {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
    }.items():
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        with pytest.raises(AssertionError):
            intrinsics.transform_intrinsics(
                transform=torch.tensor(
                    [[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                resolution=(240, 320),
            )


def test_a_resize_is_the_diagonal_case_of_a_transform() -> None:
    """A resize scales both axes about the pixel frame's own origin, which is a diagonal affine, so the two entries agree rather than each carrying its own rule.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    by_resize = intrinsics.scale_intrinsics(resolution=(120, 640))
    by_transform = intrinsics.transform_intrinsics(
        transform=torch.tensor(
            [[640.0 / 320.0, 0.0, 0.0], [0.0, 120.0 / 240.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        resolution=(120, 640),
    )
    assert by_resize.params == by_transform.params, (
        "Expected a resize to equal its diagonal transform param for param. "
        f"{by_resize.params=} {by_transform.params=}"
    )
    return


def test_scale_intrinsics_scales_focal_and_cx_cy_params() -> None:
    """CameraIntrinsics.scale_intrinsics restates focal length(s) and cx / cy params against a different resolution, the size it is currently stated against being two of its own params rather than a second thing the caller supplies.

    Args:
        None.

    Returns:
        None.
    """
    for model, params, expected_params in [
        (
            "simple_pinhole",
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"f": 800.0, "cx": 320.0, "cy": 240.0, "h": 480, "w": 640},
        ),
        (
            "pinhole",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"fx": 800.0, "fy": 820.0, "cx": 320.0, "cy": 240.0, "h": 480, "w": 640},
        ),
        (
            "ortho",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"fx": 800.0, "fy": 820.0, "cx": 320.0, "cy": 240.0, "h": 480, "w": 640},
        ),
    ]:
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        by_resolution = intrinsics.scale_intrinsics(resolution=(480, 640))
        assert by_resolution.params == expected_params, (
            "Expected a target resolution to scale focal and cx / cy by the target over the own h and w, and to carry the target h and w. "
            f"{model=} {by_resolution.params=} {expected_params=}"
        )

        by_factor = intrinsics.scale_intrinsics(scale=2.0)
        assert by_factor.params == expected_params, (
            "Expected a single factor to scale focal, cx / cy, h and w by that factor. "
            f"{model=} {by_factor.params=} {expected_params=}"
        )
    return


def test_scale_intrinsics_takes_exactly_one_of_a_target_resolution_and_a_factor() -> (
    None
):
    """A target resolution and a factor are two ways to name the same resize, so naming both leaves which one wins unstated and naming neither names no resize at all.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    with pytest.raises(AssertionError):
        intrinsics.scale_intrinsics(resolution=(480, 640), scale=2.0)
    with pytest.raises(AssertionError):
        intrinsics.scale_intrinsics()
    return


def test_only_a_model_carrying_two_focal_params_can_be_scaled_apart() -> None:
    """A model states as many focal params as it has axes to scale independently, so a resize whose two ratios differ is stated axis by axis on pinhole and ortho and has nowhere to go on simple_pinhole's one shared f.

    Args:
        None.

    Returns:
        None.
    """
    for model in ("pinhole", "ortho"):
        intrinsics = build_camera_intrinsics(
            model=model,
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
        scaled = intrinsics.scale_intrinsics(scale=(2.0, 0.5))
        assert (
            scaled.params["fx"] == pytest.approx(800.0)
            and scaled.params["cx"] == pytest.approx(320.0)
            and scaled.params["fy"] == pytest.approx(205.0)
            and scaled.params["cy"] == pytest.approx(60.0)
        ), (
            "Expected fx and cx to scale by sx, and fy and cy by sy. "
            f"{model=} {scaled.params=}"
        )

    intrinsics = build_camera_intrinsics(
        model="simple_pinhole",
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    with pytest.raises(AssertionError):
        intrinsics.scale_intrinsics(scale=(2.0, 0.5))
    return


def test_a_per_axis_normalized_frames_params_do_not_move_with_the_resolution() -> None:
    """opengl and vulkan each measure an axis by its own side, so restating one against a different size — of a different aspect ratio included — moves no param and only the size it reports changes.

    Args:
        None.

    Returns:
        None.
    """
    for frame in ("opengl", "vulkan"):
        intrinsics = build_camera_intrinsics(
            model="pinhole",
            params={"fx": 2.0, "fy": 2.5, "cx": 0.25, "cy": -0.5, "h": 240, "w": 320},
            intr_convention=frame,
            device="cpu",
        )
        scaled = intrinsics.scale_intrinsics(resolution=(120, 640))
        for key in ("fx", "fy", "cx", "cy"):
            assert scaled.params[key] == pytest.approx(intrinsics.params[key]), (
                "Expected a per-axis normalized frame's param not to move with the resolution. "
                f"{frame=} {key=} {scaled.params[key]=} {intrinsics.params[key]=}"
            )
        assert scaled.params["h"] == 120 and scaled.params["w"] == 640, (
            "Expected the restated h and w params to be the target ones. "
            f"{frame=} {scaled.params=}"
        )
    return


def test_the_pytorch3d_frames_params_move_when_the_aspect_ratio_does() -> None:
    """pytorch3d normalizes both axes by the shorter side alone, so its params hold under a resize that keeps the aspect ratio and are restated by one that does not.

    Args:
        None.

    Returns:
        None.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            "fx": torch.tensor(2.0),
            "fy": torch.tensor(2.5),
            "cx": torch.tensor(0.25),
            "cy": torch.tensor(-0.5),
            "h": torch.tensor(240.0),
            "w": torch.tensor(320.0),
        },
        intr_convention="pytorch3d",
        device="cpu",
    )
    uniform = intrinsics.scale_intrinsics(scale=torch.tensor(2.0))
    for key in ("fx", "fy", "cx", "cy"):
        assert uniform.params[key] == pytest.approx(intrinsics.params[key]), (
            "Expected a uniform resize to leave the pytorch3d param unchanged. "
            f"{key=} {uniform.params[key]=} {intrinsics.params[key]=}"
        )

    target_resolution = torch.tensor([120, 640])
    aspect = intrinsics.scale_intrinsics(resolution=target_resolution)
    through_pixels = intrinsics.to(intr_convention="standard")
    through_pixels = through_pixels.scale_intrinsics(resolution=target_resolution)
    through_pixels = through_pixels.to(intr_convention="pytorch3d")
    for key in ("fx", "fy", "cx", "cy"):
        assert aspect.params[key] == pytest.approx(through_pixels.params[key]), (
            "Expected an aspect change to restate the pytorch3d param as the same resize carried through pixels does. "
            f"{key=} {aspect.params[key]=} {through_pixels.params[key]=}"
        )
    assert aspect.params["cx"] != pytest.approx(intrinsics.params["cx"]), (
        "Expected an aspect change to move the pytorch3d principal point. "
        f"{aspect.params=} {intrinsics.params=}"
    )
    return


def test_intrinsics_tensor_state_stays_differentiable_through_project() -> None:
    """Tensor intrinsics state stays on the autograd path through projection.

    Args:
        None.

    Returns:
        None.
    """
    for model, (numeric_params, projection_keys) in {
        "simple_pinhole": (
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            ("f", "cx", "cy"),
        ),
        "pinhole": (
            {
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            },
            ("fx", "fy", "cx", "cy"),
        ),
        "ortho": (
            {
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            },
            ("fx", "fy", "cx", "cy"),
        ),
    }.items():
        params = _tensor_params(params=numeric_params, requires_grad=True)
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        image_points = intrinsics.project(
            points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
        )
        loss = image_points.sum()
        loss.backward()
        for key in projection_keys:
            assert params[key].grad is not None, (
                "Expected every source tensor param to receive a gradient. "
                f"{model=} {key=} {params[key].grad=}"
            )
    return


def test_scale_intrinsics_keeps_tensor_state_differentiable() -> None:
    """CameraIntrinsics.scale_intrinsics keeps tensor state and tensor scale factors on the autograd path.

    Args:
        None.

    Returns:
        None.
    """
    for model, (numeric_params, projection_keys) in {
        "simple_pinhole": (
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            ("f", "cx", "cy"),
        ),
        "pinhole": (
            {
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            },
            ("fx", "fy", "cx", "cy"),
        ),
        "ortho": (
            {
                "fx": 400.0,
                "fy": 410.0,
                "cx": 160.0,
                "cy": 120.0,
                "h": 240,
                "w": 320,
            },
            ("fx", "fy", "cx", "cy"),
        ),
    }.items():
        params = _tensor_params(params=numeric_params, requires_grad=True)
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )
        scale = (
            torch.tensor(2.0, dtype=torch.float32, requires_grad=True),
            torch.tensor(2.0, dtype=torch.float32, requires_grad=True),
        )
        scaled_intrinsics = intrinsics.scale_intrinsics(scale=scale)
        image_points = scaled_intrinsics.project(
            points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
        )
        loss = image_points.sum()
        loss.backward()
        for key in projection_keys:
            assert params[key].grad is not None, (
                "Expected every source tensor param to receive a gradient. "
                f"{model=} {key=} {params[key].grad=}"
            )
        assert scale[0].grad is not None and scale[1].grad is not None, (
            "Expected both tensor scale factors to receive a gradient. "
            f"{model=} {scale[0].grad=} {scale[1].grad=}"
        )
    return


def _tensor_params(
    params: Dict[str, Union[int, float, List[Union[int, float]]]],
    requires_grad: bool = False,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """A test states its params as plain numbers, and this is what makes them the tensor state a camera actually carries.

    Args:
        params: Numeric intrinsics params keyed by model field name, each stated as one number shared by every camera or as one number per camera.
        requires_grad: Whether floating projection params should require gradients.
        batch_size: Number of cameras the params state, or None for a single unbatched camera.

    Returns:
        A dict with every param represented as a float32 tensor, of shape ``[]`` when batch_size is None and ``[batch_size]`` otherwise.
    """
    tensor_params: Dict[str, torch.Tensor] = {}
    if batch_size is not None:
        for key, value in params.items():
            values: List[float] = []
            if isinstance(value, list):
                for item in value:
                    values.append(float(item))
                tensor_params[key] = torch.tensor(
                    values, dtype=torch.float32, requires_grad=requires_grad
                )
                continue
            if isinstance(value, (int, float)):
                for _ in range(batch_size):
                    values.append(float(value))
                tensor_params[key] = torch.tensor(
                    values, dtype=torch.float32, requires_grad=requires_grad
                )
                continue
            assert 0, "Should not reach here."
        return tensor_params
    for key, value in params.items():
        tensor_params[key] = torch.tensor(
            float(value), dtype=torch.float32, requires_grad=requires_grad
        )
    return tensor_params
