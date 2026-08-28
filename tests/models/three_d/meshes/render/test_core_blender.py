"""Tests for Blender mesh-render helpers."""

import importlib
import sys
import types
from typing import List, Tuple

import pytest
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.extrinsics.camera_extrinsics import (
    CameraExtrinsics,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)


@pytest.mark.parametrize("intr_convention", ("standard", "pytorch3d"))
def test_named_resolution_scales_blender_camera_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
    intr_convention: str,
) -> None:
    """Blender camera parameters are built from resolution-scaled intrinsics.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to stub Blender modules.
        intr_convention: Input camera image-plane convention.

    Returns:
        None.
    """
    camera_data, linked_objects = _install_blender_stubs(monkeypatch=monkeypatch)
    core_blender = importlib.import_module("models.three_d.meshes.render.core_blender")
    camera = _build_camera(intr_convention=intr_convention)

    camera_obj = core_blender._create_camera_from_parameters_blender(
        camera=camera,
        resolution=(40, 50),
    )

    assert linked_objects == [camera_obj], (
        "Expected the created Blender camera object to be linked into the scene "
        "collection. "
        f"{linked_objects=} {camera_obj=}"
    )
    assert camera_data.lens == pytest.approx(14.4), (
        "Expected the Blender focal length to use resolution-scaled intrinsics. "
        f"{camera_data.lens=}"
    )
    assert camera_data.sensor_fit == "VERTICAL", (
        "Expected the Blender camera sensor fit to follow the scaled focal aspect. "
        f"{camera_data.sensor_fit=}"
    )
    assert camera_data.shift_x == pytest.approx(-0.3), (
        "Expected the Blender horizontal principal-point shift to be scaled. "
        f"{camera_data.shift_x=}"
    )
    assert camera_data.shift_y == pytest.approx(0.2), (
        "Expected the Blender vertical principal-point shift to be scaled. "
        f"{camera_data.shift_y=}"
    )


def _install_blender_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[types.SimpleNamespace, List[types.SimpleNamespace]]:
    """Install minimal bpy and mathutils modules for camera-helper tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to modify `sys.modules`.

    Returns:
        The fake camera data block and the linked fake objects.
    """
    camera_data = types.SimpleNamespace(
        name="mesh_camera_blender",
        sensor_width=36.0,
        sensor_height=24.0,
        lens=None,
        sensor_fit=None,
        shift_x=None,
        shift_y=None,
    )
    linked_objects: List[types.SimpleNamespace] = []

    def new_camera(name: str) -> types.SimpleNamespace:
        """Return the one fake Blender camera data block.

        Args:
            name: Blender data-block name.

        Returns:
            The fake camera data block.
        """
        assert name == "mesh_camera_blender", (
            "Expected Blender camera data to use the render camera name. " f"{name=}"
        )
        return camera_data

    def new_object(name: str, data: types.SimpleNamespace) -> types.SimpleNamespace:
        """Return a fake Blender object wrapping the given data block.

        Args:
            name: Blender object name.
            data: Camera data block assigned to the object.

        Returns:
            A fake Blender object.
        """
        return types.SimpleNamespace(name=name, data=data, matrix_world=None)

    def link(obj: types.SimpleNamespace) -> None:
        """Record the object linked into the fake scene.

        Args:
            obj: Fake Blender object linked to the scene collection.

        Returns:
            None.
        """
        linked_objects.append(obj)

    bpy_module = types.ModuleType("bpy")
    bpy_module.data = types.SimpleNamespace(
        cameras=types.SimpleNamespace(new=new_camera),
        objects=types.SimpleNamespace(new=new_object),
    )
    bpy_module.context = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            collection=types.SimpleNamespace(
                objects=types.SimpleNamespace(link=link),
            )
        )
    )

    def matrix(values: List[List[float]]) -> List[List[float]]:
        """Return a fake mathutils Matrix value.

        Args:
            values: Matrix rows.

        Returns:
            The same matrix rows.
        """
        return values

    mathutils_module = types.ModuleType("mathutils")
    mathutils_module.Matrix = matrix

    monkeypatch.setitem(sys.modules, "bpy", bpy_module)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils_module)
    monkeypatch.delitem(
        sys.modules,
        "models.three_d.meshes.render.core_blender",
        raising=False,
    )
    return camera_data, linked_objects


def _build_camera(intr_convention: str = "standard") -> Camera:
    """Build a pinhole camera with known resolution scaling.

    Args:
        intr_convention: Image-plane convention the returned camera should use.

    Returns:
        A camera whose named render resolution scales its intrinsics by two.
    """
    camera = Camera(
        intrinsics=build_camera_intrinsics(
            model="pinhole",
            params={
                "fx": 10.0,
                "fy": 20.0,
                "cx": 5.0,
                "cy": 6.0,
                "h": 20,
                "w": 25,
            },
            intr_convention="standard",
            device=torch.device("cpu"),
        ),
        extrinsics=CameraExtrinsics(
            extrinsics=torch.eye(4, dtype=torch.float32),
            extr_convention="standard",
            device=torch.device("cpu"),
        ),
        device=torch.device("cpu"),
    )
    return camera.to(intr_convention=intr_convention)
