import json
import sys
import types
from pathlib import Path

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
from data.structures.three_d.camera.io import deserialize_camera, serialize_camera


def _validate_loaded_camera(loaded_camera: Camera, camera: Camera) -> None:
    """Validate that two Camera objects carry the same serialized fields.

    Args:
        loaded_camera: Camera object loaded from serialized data.
        camera: Original Camera object.

    Returns:
        None.
    """
    assert torch.equal(loaded_camera.intrinsics, camera.intrinsics)
    assert torch.equal(loaded_camera.extrinsics, camera.extrinsics)
    assert loaded_camera.convention == camera.convention
    assert loaded_camera.name == camera.name
    assert loaded_camera.id == camera.id


def test_camera_serialization_round_trip_preserves_core_fields(tmp_path: Path) -> None:
    """Camera dict, `.npz`, and `.json` I/O should preserve core fields.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
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
    camera = Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        convention="opengl",
        name="frame_0",
        id=7,
        device=torch.device("cpu"),
    )

    serialized = serialize_camera(camera=camera, format="json")
    assert serialized == camera.serialize(format="json")
    deserialized_camera = deserialize_camera(
        payload=serialized,
        device=torch.device("cpu"),
        format="json",
    )
    method_deserialized_camera = Camera.deserialize(
        payload=serialized,
        device=torch.device("cpu"),
        format="json",
    )
    _validate_loaded_camera(loaded_camera=deserialized_camera, camera=camera)
    _validate_loaded_camera(loaded_camera=method_deserialized_camera, camera=camera)

    serialized_npz = serialize_camera(camera=camera, format="npz")
    assert serialized_npz.keys() == {
        "intrinsics",
        "extrinsics",
        "convention",
        "name",
        "has_name",
        "id",
        "has_id",
    }
    assert camera.serialize(format=".npz").keys() == serialized_npz.keys()
    deserialized_npz_camera = deserialize_camera(
        payload=serialized_npz,
        device=torch.device("cpu"),
        format="npz",
    )
    method_deserialized_npz_camera = Camera.deserialize(
        payload=serialized_npz,
        device=torch.device("cpu"),
        format=".npz",
    )
    _validate_loaded_camera(loaded_camera=deserialized_npz_camera, camera=camera)
    _validate_loaded_camera(loaded_camera=method_deserialized_npz_camera, camera=camera)

    npz_camera_path = tmp_path / "camera.npz"
    camera.save(camera_path=npz_camera_path)
    loaded_npz_camera = Camera.load(
        camera_path=npz_camera_path,
        device=torch.device("cpu"),
    )

    npz_payload = np.load(npz_camera_path, allow_pickle=False)
    assert set(npz_payload.files) == {
        "intrinsics",
        "extrinsics",
        "convention",
        "name",
        "has_name",
        "id",
        "has_id",
    }
    _validate_loaded_camera(loaded_camera=loaded_npz_camera, camera=camera)

    json_camera_path = tmp_path / "camera.json"
    camera.save(camera_path=json_camera_path)
    loaded_json_camera = Camera.load(
        camera_path=json_camera_path,
        device=torch.device("cpu"),
    )
    json_payload = json.loads(json_camera_path.read_text(encoding="utf-8"))
    assert json_payload == serialized
    _validate_loaded_camera(loaded_camera=loaded_json_camera, camera=camera)
