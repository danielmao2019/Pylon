import ast
import math
import warnings
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import pytest
import torch

from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsics,
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
    simple_params = {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320}
    pinhole_params = {
        "fx": 400.0,
        "fy": 410.0,
        "cx": 160.0,
        "cy": 120.0,
        "h": 240,
        "w": 320,
    }
    assert (
        validate_camera_intrinsics_params(
            model="simple_pinhole", intr_convention="standard", params=simple_params
        )
        == simple_params
    ), f"{simple_params=}"
    assert (
        validate_camera_intrinsics_params(
            model="pinhole", intr_convention="standard", params=pinhole_params
        )
        == pinhole_params
    ), f"{pinhole_params=}"
    assert (
        validate_camera_intrinsics_params(
            model="ortho", intr_convention="standard", params=pinhole_params
        )
        == pinhole_params
    ), f"{pinhole_params=}"

    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(
            model="simple_pinhole", intr_convention="standard", params=pinhole_params
        )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(
            model="pinhole", intr_convention="standard", params=simple_params
        )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_params(
            model="ortho", intr_convention="standard", params=simple_params
        )


def test_validate_intrinsics_params_rejects_a_params_dict_missing_the_resolution() -> (
    None
):
    """h and w are two of every model's own params, so a projection-only dict is rejected.

    Args:
        None.

    Returns:
        None.
    """
    projection_only: Dict[str, Dict[str, Union[int, float]]] = {
        "simple_pinhole": {"f": 400.0, "cx": 160.0, "cy": 120.0},
        "pinhole": {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
        "ortho": {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0},
    }
    for model, params in projection_only.items():
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_params(
                model=model,
                intr_convention="standard",
                params=params,
            )


def test_the_principal_point_must_lie_on_the_image_in_its_own_frames_extent() -> None:
    """The principal point's bound is the frame's to state, so the check reads the two together.

    Args:
        None.

    Returns:
        None.
    """

    def pinhole(cx: float, cy: float) -> Dict[str, Union[int, float]]:
        return {"fx": 400.0, "fy": 410.0, "cx": cx, "cy": cy, "h": 240, "w": 320}

    # standard: the pixel frame running corner to corner.
    validate_camera_intrinsics_invariants(
        model="pinhole", intr_convention="standard", params=pinhole(320.0, 240.0)
    )
    for outside in (pinhole(320.5, 240.0), pinhole(320.0, 240.5)):
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_invariants(
                model="pinhole", intr_convention="standard", params=outside
            )

    # opengl / vulkan: each axis normalized by its own side, so both bounds are the same.
    for frame in ("opengl", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="pinhole", intr_convention=frame, params=pinhole(1.0, -1.0)
        )
        for outside in (pinhole(1.5, 0.0), pinhole(0.0, -1.5)):
            with pytest.raises(AssertionError):
                validate_camera_intrinsics_invariants(
                    model="pinhole", intr_convention=frame, params=outside
                )

    # pytorch3d: the shorter side alone reaches 1, so the longer axis's bound is the larger.
    validate_camera_intrinsics_invariants(
        model="pinhole", intr_convention="pytorch3d", params=pinhole(1.25, 1.0)
    )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_invariants(
            model="pinhole", intr_convention="pytorch3d", params=pinhole(0.0, 1.25)
        )

    # ortho: cx / cy name where the world origin lands, so any finite pair is valid.
    for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="ortho",
            intr_convention=frame,
            params={"fx": 1.0, "fy": 1.0, "cx": -37.5, "cy": 512.0, "h": 240, "w": 240},
        )


def test_a_centred_principal_point_survives_its_models_own_key_dispatch() -> None:
    """A negative principal point is in range on a centred frame, and the key dispatch does not bound it.

    Args:
        None.

    Returns:
        None.
    """
    centred: Dict[str, Dict[str, Union[int, float]]] = {
        "simple_pinhole": {"f": 2.0, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240},
        "pinhole": {"fx": 2.0, "fy": 2.5, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240},
        "ortho": {"fx": 2.0, "fy": 2.5, "cx": -0.5, "cy": -0.5, "h": 240, "w": 240},
    }
    for model, params in centred.items():
        assert (
            validate_camera_intrinsics_params(
                model=model, intr_convention="opengl", params=params
            )
            == params
        ), f"{model=} {params=}"


def test_a_frame_that_scales_the_axes_apart_cannot_hold_a_shared_focal() -> None:
    """opengl and vulkan normalize each axis by its own side, so they hold a simple_pinhole only on a square image.

    Args:
        None.

    Returns:
        None.
    """

    def params_for(model: str, height: int, width: int) -> Dict[str, Union[int, float]]:
        if model == "simple_pinhole":
            return {"f": 2.0, "cx": 0.0, "cy": 0.0, "h": height, "w": width}
        return {"fx": 2.0, "fy": 2.5, "cx": 0.0, "cy": 0.0, "h": height, "w": width}

    for frame in ("opengl", "vulkan"):
        validate_camera_intrinsics_invariants(
            model="simple_pinhole",
            intr_convention=frame,
            params=params_for("simple_pinhole", 240, 240),
        )
        with pytest.raises(AssertionError):
            validate_camera_intrinsics_invariants(
                model="simple_pinhole",
                intr_convention=frame,
                params=params_for("simple_pinhole", 240, 320),
            )
    for height, width in ((240, 240), (240, 320)):
        validate_camera_intrinsics_invariants(
            model="simple_pinhole",
            intr_convention="pytorch3d",
            params=params_for("simple_pinhole", height, width),
        )
        for model in ("pinhole", "ortho"):
            for frame in ("standard", "opengl", "pytorch3d", "vulkan"):
                validate_camera_intrinsics_invariants(
                    model=model,
                    intr_convention=frame,
                    params=params_for(model, height, width),
                )


def test_validate_intrinsics_attributes_checks_model_intr_convention_params_device() -> (
    None
):
    """validate_camera_intrinsics_attributes validates model, image-plane frame, params, and device.

    Args:
        None.

    Returns:
        None.
    """
    params = {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320}
    validate_camera_intrinsics_attributes(
        model="pinhole", params=params, intr_convention="standard", device="cpu"
    )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(
            model="fisheye",
            intr_convention="standard",
            params=params,
            device="cpu",
        )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(
            model="pinhole",
            params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            intr_convention="standard",
            device="cpu",
        )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(
            model="pinhole",
            intr_convention="ndc",
            params=params,
            device="cpu",
        )
    with pytest.raises(AssertionError):
        validate_camera_intrinsics_attributes(
            model="pinhole",
            intr_convention="standard",
            params=params,
            device=0,
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
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    pinhole = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    ortho = build_camera_intrinsics(
        model="ortho",
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    assert isinstance(simple, CameraIntrinsicsSimplePinhole), f"{type(simple)=}"
    assert isinstance(pinhole, CameraIntrinsicsPinhole), f"{type(pinhole)=}"
    assert isinstance(ortho, CameraIntrinsicsOrtho), f"{type(ortho)=}"
    assert simple.model == "simple_pinhole", f"{simple.model=}"
    assert pinhole.model == "pinhole", f"{pinhole.model=}"
    assert ortho.model == "ortho", f"{ortho.model=}"


def test_simple_pinhole_project_applies_perspective_divide() -> None:
    """CameraIntrinsicsSimplePinhole.project applies the perspective divide.

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
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
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
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
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


_REPO_ROOT = Path(__file__).resolve().parents[5]
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


def _is_a_camera_depth(node: ast.expr) -> bool:
    """Decide whether an expression names camera-space depth.

    Args:
        node: The denominator expression of a division as an ast node.

    Returns:
        True when the expression names depth by name or reads the third
        component off a points-like tensor, else False.
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


def _classify_camera_module(tree: ast.Module) -> Tuple[bool, bool]:
    """Classify one module by how it reaches image coordinates.

    Args:
        tree: The parsed module.

    Returns:
        A ``(projects_through_the_camera, divides_by_a_camera_depth_itself)``
        pair, the second flag set only when the module also reads a camera's
        focal length and principal point.
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


def test_every_camera_consumer_projects_through_the_camera() -> None:
    """Every repo-owned camera consumer reaches image coordinates through CameraIntrinsics.project.

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


@pytest.mark.parametrize(
    "intrinsics",
    [
        CameraIntrinsicsSimplePinhole(
            params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            intr_convention="standard",
            device="cpu",
        ),
        CameraIntrinsicsPinhole(
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
        ),
        CameraIntrinsicsOrtho(
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
        ),
    ],
)
def test_project_inplace_overwrites_input_and_matches_not_inplace(
    intrinsics: CameraIntrinsics,
) -> None:
    """project(inplace=True) overwrites the input and matches the not-inplace result.

    Args:
        intrinsics: A concrete CameraIntrinsics instance to project with.

    Returns:
        None.
    """
    points = torch.tensor([[1.0, 2.0, 4.0], [3.0, -1.0, 8.0]], dtype=torch.float32)
    reference = points.clone()
    expected = intrinsics.project(points_camera=points.clone(), inplace=False)
    result = intrinsics.project(points_camera=points, inplace=True)
    assert result.data_ptr() == points.data_ptr(), (
        "Expected the inplace result to alias the input tensor. "
        f"{result.data_ptr()=} {points.data_ptr()=}"
    )
    assert torch.allclose(result, expected), (
        "Expected the inplace result to match the not-inplace result. "
        f"{result=} {expected=}"
    )
    assert torch.allclose(points[:, :2], expected), (
        "Expected the first two input columns to be overwritten in place. "
        f"{points=} {expected=}"
    )
    assert torch.allclose(points[:, 2], reference[:, 2]), (
        "Expected the input depth column to be preserved. " f"{points=} {reference=}"
    )


@pytest.mark.parametrize(
    "intrinsics",
    [
        CameraIntrinsicsSimplePinhole(
            params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            intr_convention="standard",
            device="cpu",
        ),
        CameraIntrinsicsPinhole(
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
        ),
        CameraIntrinsicsOrtho(
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
        ),
    ],
)
def test_project_not_inplace_preserves_input_and_returns_new_tensor(
    intrinsics: CameraIntrinsics,
) -> None:
    """project(inplace=False) leaves the input untouched and returns a new tensor.

    Args:
        intrinsics: A concrete CameraIntrinsics instance to project with.

    Returns:
        None.
    """
    points = torch.tensor([[1.0, 2.0, 4.0], [3.0, -1.0, 8.0]], dtype=torch.float32)
    reference = points.clone()
    result = intrinsics.project(points_camera=points, inplace=False)
    assert result.data_ptr() != points.data_ptr(), (
        "Expected the not-inplace result to be a freshly allocated tensor. "
        f"{result.data_ptr()=} {points.data_ptr()=}"
    )
    assert result.shape == (2, 2), (
        "Expected the not-inplace result to be a [..., 2] tensor. " f"{result.shape=}"
    )
    assert torch.allclose(points, reference), (
        "Expected the input tensor to be left unchanged. " f"{points=} {reference=}"
    )


@pytest.mark.parametrize(
    "intrinsics",
    [
        CameraIntrinsicsSimplePinhole(
            params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            intr_convention="standard",
            device="cpu",
        ),
        CameraIntrinsicsPinhole(
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
        ),
        CameraIntrinsicsOrtho(
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
        ),
    ],
)
def test_project_supports_batched_leading_dims(
    intrinsics: CameraIntrinsics,
) -> None:
    """project handles [..., 3] leading dims the same as the flattened [N, 3] path.

    Args:
        intrinsics: A concrete CameraIntrinsics instance to project with.

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
    expected = intrinsics.project(
        points_camera=batched.reshape(-1, 3).clone(),
        inplace=False,
    ).reshape(2, 3, 2)

    result = intrinsics.project(points_camera=batched.clone(), inplace=False)
    assert result.shape == (2, 3, 2), (
        "Expected the not-inplace batched result to keep the leading dims. "
        f"{result.shape=}"
    )
    assert torch.allclose(result, expected), (
        "Expected the not-inplace batched result to match the flattened projection. "
        f"{result=} {expected=}"
    )

    points = batched.clone()
    reference = points.clone()
    result_ip = intrinsics.project(points_camera=points, inplace=True)
    assert torch.allclose(result_ip, expected), (
        "Expected the inplace batched result to match the flattened projection. "
        f"{result_ip=} {expected=}"
    )
    assert torch.allclose(points[..., :2], expected), (
        "Expected the first two input columns to be overwritten in place. "
        f"{points=} {expected=}"
    )
    assert torch.allclose(points[..., 2], reference[..., 2]), (
        "Expected the input depth column to be preserved. " f"{points=} {reference=}"
    )


@pytest.mark.parametrize(
    "intrinsics",
    [
        CameraIntrinsicsSimplePinhole(
            params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            intr_convention="standard",
            device="cpu",
        ),
        CameraIntrinsicsPinhole(
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
        ),
        CameraIntrinsicsOrtho(
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
        ),
    ],
)
def test_project_rejects_invalid_inputs(
    intrinsics: CameraIntrinsics,
) -> None:
    """project rejects non-tensor inputs, a wrong last dim, and a non-bool inplace.

    Args:
        intrinsics: A concrete CameraIntrinsics instance to project with.

    Returns:
        None.
    """
    with pytest.raises(AssertionError):
        intrinsics.project(points_camera=[[1.0, 2.0, 4.0]])
    with pytest.raises(AssertionError):
        intrinsics.project(points_camera=torch.zeros(4, 2, dtype=torch.float32))
    with pytest.raises(AssertionError):
        intrinsics.project(
            points_camera=torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32),
            inplace=1,
        )


def test_fx_fy_cx_cy_derived_from_params() -> None:
    """The fx / fy accessors and the cx / cy accessors are derived from params.

    Args:
        None.

    Returns:
        None.
    """
    simple = CameraIntrinsicsSimplePinhole(
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    assert simple.fx == 400.0 and simple.fy == 400.0, f"{simple.fx=} {simple.fy=}"
    assert simple.cx == 160.0 and simple.cy == 120.0, f"{simple.cx=} {simple.cy=}"

    pinhole = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    assert pinhole.fx == 400.0 and pinhole.fy == 410.0, f"{pinhole.fx=} {pinhole.fy=}"
    assert pinhole.cx == 160.0 and pinhole.cy == 120.0, f"{pinhole.cx=} {pinhole.cy=}"

    ortho = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
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
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    pinhole = CameraIntrinsicsPinhole(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    ortho = CameraIntrinsicsOrtho(
        params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    assert isinstance(simple.fov, tuple) and len(simple.fov) == 2, f"{simple.fov=}"
    assert isinstance(pinhole.fov, tuple) and len(pinhole.fov) == 2, f"{pinhole.fov=}"
    assert all(isinstance(value, float) for value in simple.fov), f"{simple.fov=}"
    assert all(isinstance(value, float) for value in pinhole.fov), f"{pinhole.fov=}"
    for intrinsics in (simple, pinhole):
        expected_horizontal = (
            2.0 * math.atan(intrinsics.cx / intrinsics.fx) * 180.0 / math.pi
        )
        expected_vertical = (
            2.0 * math.atan(intrinsics.cy / intrinsics.fy) * 180.0 / math.pi
        )
        assert math.isclose(
            intrinsics.fov[0], expected_horizontal, rel_tol=1.0e-09
        ), f"{intrinsics.fov=} {expected_horizontal=}"
        assert math.isclose(
            intrinsics.fov[1], expected_vertical, rel_tol=1.0e-09
        ), f"{intrinsics.fov=} {expected_vertical=}"
    assert hasattr(ortho, "fov") is False, "Ortho intrinsics must not expose fov."


def test_transform_intrinsics_restates_the_camera_onto_the_named_raster() -> None:
    """The affine says how coordinates move, and the raster named beside it becomes the h / w the result carries.

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
    transform = torch.tensor(
        [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    transformed = intrinsics.transform_intrinsics(
        transform=transform,
        resolution=(480, 640),
    )
    assert transformed.params["h"] == 480, f"{transformed.params=}"
    assert transformed.params["w"] == 640, f"{transformed.params=}"
    assert transformed.params["cx"] == pytest.approx(160.0), f"{transformed.params=}"
    assert transformed.params["cy"] == pytest.approx(130.0), f"{transformed.params=}"


def test_transform_intrinsics_returns_the_frame_it_was_given() -> None:
    """A transform says nothing about the image-plane frame, so the result comes back on the one it went in on.

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
        ).to(intr_convention=frame)
        identity = torch.eye(3, dtype=torch.float32)
        transformed = intrinsics.transform_intrinsics(
            transform=identity,
            resolution=intrinsics.resolution,
        )
        assert (
            transformed.intr_convention == frame
        ), f"{frame=} {transformed.intr_convention=}"
        for key, value in intrinsics.params.items():
            assert transformed.params[key] == pytest.approx(
                value
            ), f"{frame=} {key=} {transformed.params[key]=} {value=}"


def test_a_shared_focal_refuses_a_transform_that_scales_the_axes_apart() -> None:
    """simple_pinhole states one f for both axes, so an affine whose diagonal entries differ has nowhere to put the second.

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
    transform = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    with pytest.raises(AssertionError):
        intrinsics.transform_intrinsics(transform=transform, resolution=(120, 640))


def test_a_resize_is_the_diagonal_case_of_a_transform() -> None:
    """A resize scales both axes about the pixel frame's own origin, which is a diagonal affine.

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
    target_resolution = (120, 640)
    scale_x = float(target_resolution[1]) / 320.0
    scale_y = float(target_resolution[0]) / 240.0
    transform = torch.tensor(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    by_resize = intrinsics.scale_intrinsics(resolution=target_resolution)
    by_transform = intrinsics.transform_intrinsics(
        transform=transform,
        resolution=target_resolution,
    )
    assert (
        by_resize.params == by_transform.params
    ), f"{by_resize.params=} {by_transform.params=}"


def test_scale_intrinsics_scales_focal_and_cx_cy_params() -> None:
    """CameraIntrinsics.scale_intrinsics scales focal and cx / cy params.

    Args:
        None.

    Returns:
        None.
    """
    model_cases: List[
        Tuple[str, Dict[str, Union[int, float]], Dict[str, Union[int, float]]]
    ] = [
        (
            "simple_pinhole",
            {"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"f": 800.0, "cx": 320.0, "cy": 240.0, "h": 480, "w": 640},
        ),
        (
            "pinhole",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"fx": 800.0, "fy": 205.0, "cx": 320.0, "cy": 60.0, "h": 120, "w": 640},
        ),
        (
            "ortho",
            {"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
            {"fx": 800.0, "fy": 205.0, "cx": 320.0, "cy": 60.0, "h": 120, "w": 640},
        ),
    ]
    for model, params, expected_params in model_cases:
        intrinsics = build_camera_intrinsics(
            model=model,
            params=params,
            intr_convention="standard",
            device="cpu",
        )

        # The params carry their own resolution (h, w) = (240, 320); a shared focal
        # takes one factor on both axes, the two-focal models take one per axis.
        target_resolution = (480, 640) if model == "simple_pinhole" else (120, 640)
        target_scale = (2.0, 2.0) if model == "simple_pinhole" else (2.0, 0.5)
        by_resolution = intrinsics.scale_intrinsics(resolution=target_resolution)
        assert (
            by_resolution.params == expected_params
        ), f"{model=} {by_resolution.params=} {expected_params=}"
        assert type(by_resolution) is type(intrinsics), f"{type(by_resolution)=}"

        by_axes = intrinsics.scale_intrinsics(scale=target_scale)
        assert (
            by_axes.params == expected_params
        ), f"{model=} {by_axes.params=} {expected_params=}"


def test_scale_intrinsics_takes_exactly_one_of_a_target_resolution_and_a_factor() -> (
    None
):
    """A target resolution and a factor name the same resize, so both together and neither are equally unstated.

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


def test_only_a_model_carrying_two_focal_params_can_be_scaled_apart() -> None:
    """A resize whose two ratios differ is stated axis by axis on pinhole and ortho, and has nowhere to go on one shared f.

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
        assert scaled.params["fx"] == pytest.approx(800.0), f"{model=} {scaled.params=}"
        assert scaled.params["cx"] == pytest.approx(320.0), f"{model=} {scaled.params=}"
        assert scaled.params["fy"] == pytest.approx(205.0), f"{model=} {scaled.params=}"
        assert scaled.params["cy"] == pytest.approx(60.0), f"{model=} {scaled.params=}"

    simple = build_camera_intrinsics(
        model="simple_pinhole",
        params={"f": 400.0, "cx": 160.0, "cy": 120.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    with pytest.raises(AssertionError):
        simple.scale_intrinsics(scale=(2.0, 0.5))


def test_a_per_axis_normalized_frames_params_do_not_move_with_the_resolution() -> None:
    """opengl and vulkan measure each axis by its own side, so only the size a restatement reports changes.

    Args:
        None.

    Returns:
        None.
    """
    for frame in ("opengl", "vulkan"):
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
        ).to(intr_convention=frame)
        scaled = intrinsics.scale_intrinsics(resolution=(120, 640))
        for key in ("fx", "fy", "cx", "cy"):
            assert scaled.params[key] == pytest.approx(
                intrinsics.params[key]
            ), f"{frame=} {key=} {scaled.params[key]=} {intrinsics.params[key]=}"
        assert scaled.params["h"] == 120, f"{frame=} {scaled.params=}"
        assert scaled.params["w"] == 640, f"{frame=} {scaled.params=}"


def test_the_pytorch3d_frames_params_move_when_the_aspect_ratio_does() -> None:
    """pytorch3d normalizes both axes by the shorter side, so its params hold under a uniform resize and are restated by an aspect change.

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
    ).to(intr_convention="pytorch3d")
    uniform = intrinsics.scale_intrinsics(scale=2.0)
    for key in ("fx", "fy", "cx", "cy"):
        assert uniform.params[key] == pytest.approx(
            intrinsics.params[key]
        ), f"{key=} {uniform.params[key]=} {intrinsics.params[key]=}"

    target_resolution = (120, 640)
    aspect = intrinsics.scale_intrinsics(resolution=target_resolution)
    through_pixels = (
        intrinsics.to(intr_convention="standard")
        .scale_intrinsics(resolution=target_resolution)
        .to(intr_convention="pytorch3d")
    )
    for key in ("fx", "fy", "cx", "cy"):
        assert aspect.params[key] == pytest.approx(
            through_pixels.params[key]
        ), f"{key=} {aspect.params[key]=} {through_pixels.params[key]=}"
    assert aspect.params["cx"] != pytest.approx(
        intrinsics.params["cx"]
    ), f"{aspect.params=} {intrinsics.params=}"
