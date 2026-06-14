import json
import sys
import types
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def _install_namespace_package(package_name: str, package_path: Path) -> None:
    if package_name in sys.modules:
        return

    module = types.ModuleType(package_name)
    module.__file__ = str(package_path / "__init__.py")
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module


REPO_ROOT = Path(__file__).resolve().parents[5]
_install_namespace_package(package_name="utils", package_path=REPO_ROOT / "utils")
_install_namespace_package(
    package_name="utils.ops", package_path=REPO_ROOT / "utils" / "ops"
)
_install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
_install_namespace_package(
    package_name="data.structures", package_path=REPO_ROOT / "data" / "structures"
)
_install_namespace_package(
    package_name="data.structures.three_d",
    package_path=REPO_ROOT / "data" / "structures" / "three_d",
)
_install_namespace_package(
    package_name="data.structures.three_d.camera",
    package_path=REPO_ROOT / "data" / "structures" / "three_d" / "camera",
)

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.io import (
    deserialize_cameras,
    load_cameras,
    save_cameras,
    serialize_cameras,
)


def _make_single_camera() -> Camera:
    """Build a single Camera fixture.

    Args:
        None.

    Returns:
        A Camera on the CPU with non-trivial intrinsics, extrinsics, name, and id.
    """
    intrinsics = torch.tensor(
        [
            [400.0, 0.0, 160.0],
            [0.0, 410.0, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    extrinsics = torch.eye(4, dtype=torch.float32)
    extrinsics[:3, 3] = torch.tensor([0.3, -0.2, 1.1], dtype=torch.float32)
    return Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        convention="opengl",
        name="frame_0",
        id=7,
        device=torch.device("cpu"),
    )


def _make_multi_cameras() -> Cameras:
    """Build a multi-camera Cameras fixture.

    Args:
        None.

    Returns:
        A Cameras of three CPU cameras with mixed conventions, names (one absent),
        and ids (one absent), exercising the has_name / has_id / sentinel paths.
    """
    intrinsics: List[torch.Tensor] = []
    extrinsics: List[torch.Tensor] = []
    conventions = ["opengl", "opencv", "standard"]
    names: List[Optional[str]] = ["frame_0", None, "frame_2"]
    ids: List[Optional[int]] = [7, 8, None]
    for index in range(3):
        intr = torch.tensor(
            [
                [400.0 + index, 0.0, 160.0 + index],
                [0.0, 410.0 + index, 120.0 + index],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        extr = torch.eye(4, dtype=torch.float32)
        extr[:3, 3] = torch.tensor(
            [0.3 + index, -0.2 + index, 1.1 + index],
            dtype=torch.float32,
        )
        intrinsics.append(intr)
        extrinsics.append(extr)
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        conventions=conventions,
        names=names,
        ids=ids,
        device=torch.device("cpu"),
    )


def _assert_camera_fields_equal(loaded: Camera, original: Camera) -> None:
    """Assert two Camera objects carry the same core serialized fields.

    Args:
        loaded: Camera object recovered from serialization.
        original: Original Camera object.

    Returns:
        None.
    """
    assert isinstance(loaded, Camera), f"{type(loaded)=}"
    assert torch.equal(
        loaded.intrinsics, original.intrinsics
    ), f"{loaded.intrinsics=} {original.intrinsics=}"
    assert torch.equal(
        loaded.extrinsics, original.extrinsics
    ), f"{loaded.extrinsics=} {original.extrinsics=}"
    assert (
        loaded.convention == original.convention
    ), f"{loaded.convention=} {original.convention=}"
    assert loaded.name == original.name, f"{loaded.name=} {original.name=}"
    assert loaded.id == original.id, f"{loaded.id=} {original.id=}"


def _assert_cameras_fields_equal(loaded: Cameras, original: Cameras) -> None:
    """Assert two Cameras collections carry the same core serialized fields.

    Args:
        loaded: Cameras collection recovered from serialization.
        original: Original Cameras collection.

    Returns:
        None.
    """
    assert isinstance(loaded, Cameras), f"{type(loaded)=}"
    assert len(loaded) == len(original), f"{len(loaded)=} {len(original)=}"
    assert torch.equal(
        loaded.intrinsics, original.intrinsics
    ), f"{loaded.intrinsics=} {original.intrinsics=}"
    assert torch.equal(
        loaded.extrinsics, original.extrinsics
    ), f"{loaded.extrinsics=} {original.extrinsics=}"
    assert list(loaded.conventions) == list(
        original.conventions
    ), f"{loaded.conventions=} {original.conventions=}"
    assert list(loaded.names) == list(
        original.names
    ), f"{loaded.names=} {original.names=}"
    assert list(loaded.ids) == list(original.ids), f"{loaded.ids=} {original.ids=}"


def test_single_camera_json_round_trip(tmp_path: Path) -> None:
    """A single Camera round-trips through json via the dict and file paths.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="json")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert serialized == camera.serialize(format="json"), f"{serialized=}"
    assert set(serialized.keys()) == {
        "intrinsics",
        "extrinsics",
        "convention",
        "name",
        "id",
    }, f"{set(serialized.keys())=}"

    deserialized = deserialize_cameras(
        payload=serialized,
        device=torch.device("cpu"),
        format="json",
    )
    method_deserialized = Camera.deserialize(
        payload=serialized,
        device=torch.device("cpu"),
        format="json",
    )
    _assert_camera_fields_equal(loaded=deserialized, original=camera)
    _assert_camera_fields_equal(loaded=method_deserialized, original=camera)

    json_path = tmp_path / "camera.json"
    save_cameras(cameras=camera, cameras_path=json_path)
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == serialized, f"{on_disk=} {serialized=}"
    loaded = load_cameras(cameras_path=json_path, device=torch.device("cpu"))
    method_loaded = Camera.load(camera_path=json_path, device=torch.device("cpu"))
    _assert_camera_fields_equal(loaded=loaded, original=camera)
    _assert_camera_fields_equal(loaded=method_loaded, original=camera)


def test_single_camera_npz_round_trip(tmp_path: Path) -> None:
    """A single Camera round-trips through npz via the dict and file paths.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="npz")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert set(serialized.keys()) == {
        "intrinsics",
        "extrinsics",
        "convention",
        "name",
        "has_name",
        "id",
        "has_id",
        "is_single",
    }, f"{set(serialized.keys())=}"
    assert (
        camera.serialize(format=".npz").keys() == serialized.keys()
    ), f"{camera.serialize(format='.npz').keys()=} {serialized.keys()=}"

    deserialized = deserialize_cameras(
        payload=serialized,
        device=torch.device("cpu"),
        format="npz",
    )
    method_deserialized = Camera.deserialize(
        payload=serialized,
        device=torch.device("cpu"),
        format=".npz",
    )
    _assert_camera_fields_equal(loaded=deserialized, original=camera)
    _assert_camera_fields_equal(loaded=method_deserialized, original=camera)

    npz_path = tmp_path / "camera.npz"
    save_cameras(cameras=camera, cameras_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == {
            "intrinsics",
            "extrinsics",
            "convention",
            "name",
            "has_name",
            "id",
            "has_id",
            "is_single",
        }, f"{set(on_disk.files)=}"
    loaded = load_cameras(cameras_path=npz_path, device=torch.device("cpu"))
    method_loaded = Camera.load(camera_path=npz_path, device=torch.device("cpu"))
    _assert_camera_fields_equal(loaded=loaded, original=camera)
    _assert_camera_fields_equal(loaded=method_loaded, original=camera)


def test_multi_cameras_json_round_trip(tmp_path: Path) -> None:
    """A multi-camera Cameras round-trips through json via dict and file paths.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    cameras = _make_multi_cameras()

    serialized = serialize_cameras(cameras=cameras, format="json")
    assert isinstance(serialized, list), f"{type(serialized)=}"
    assert len(serialized) == len(cameras), f"{len(serialized)=} {len(cameras)=}"
    for per_camera_dict in serialized:
        assert set(per_camera_dict.keys()) == {
            "intrinsics",
            "extrinsics",
            "convention",
            "name",
            "id",
        }, f"{set(per_camera_dict.keys())=}"

    deserialized = deserialize_cameras(
        payload=serialized,
        device=torch.device("cpu"),
        format="json",
    )
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    json_path = tmp_path / "cameras.json"
    save_cameras(cameras=cameras, cameras_path=json_path)
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == serialized, f"{on_disk=} {serialized=}"
    loaded = load_cameras(cameras_path=json_path, device=torch.device("cpu"))
    _assert_cameras_fields_equal(loaded=loaded, original=cameras)


def test_multi_cameras_npz_round_trip(tmp_path: Path) -> None:
    """A multi-camera Cameras round-trips through npz via dict and file paths.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    cameras = _make_multi_cameras()

    serialized = serialize_cameras(cameras=cameras, format="npz")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert set(serialized.keys()) == {
        "intrinsics",
        "extrinsics",
        "convention",
        "name",
        "has_name",
        "id",
        "has_id",
    }, f"{set(serialized.keys())=}"
    assert "is_single" not in serialized, f"{set(serialized.keys())=}"
    assert serialized["intrinsics"].shape == (
        len(cameras),
        3,
        3,
    ), f"{serialized['intrinsics'].shape=}"
    assert serialized["extrinsics"].shape == (
        len(cameras),
        4,
        4,
    ), f"{serialized['extrinsics'].shape=}"

    deserialized = deserialize_cameras(
        payload=serialized,
        device=torch.device("cpu"),
        format="npz",
    )
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    npz_path = tmp_path / "cameras.npz"
    save_cameras(cameras=cameras, cameras_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == {
            "intrinsics",
            "extrinsics",
            "convention",
            "name",
            "has_name",
            "id",
            "has_id",
        }, f"{set(on_disk.files)=}"
    loaded = load_cameras(cameras_path=npz_path, device=torch.device("cpu"))
    _assert_cameras_fields_equal(loaded=loaded, original=cameras)
