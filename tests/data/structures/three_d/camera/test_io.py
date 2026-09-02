import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)
from data.structures.three_d.camera.io import (
    deserialize_cameras,
    load_cameras,
    save_cameras,
    serialize_cameras,
)

_JSON_KEYS = {
    "model",
    "params",
    "intr_convention",
    "extrinsics",
    "extr_convention",
    "name",
    "id",
}
_NPZ_KEYS = {
    "model",
    "params",
    "extrinsics",
    "intr_convention",
    "extr_convention",
    "name",
    "has_name",
    "id",
    "has_id",
}


def _make_extrinsics(
    translation: List[float], extr_convention: str
) -> CameraExtrinsics:
    """Build a CameraExtrinsics fixture with an identity rotation.

    Args:
        translation: Length-3 camera-center translation as a list of floats.
        extr_convention: Pose-frame convention string.

    Returns:
        A CameraExtrinsics on the CPU with the given translation and pose frame.
    """
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, 3] = torch.tensor(translation, dtype=torch.float32)
    return CameraExtrinsics(
        extrinsics=matrix, extr_convention=extr_convention, device="cpu"
    )


def _make_single_camera() -> Camera:
    """Build a single Camera fixture.

    Args:
        None.

    Returns:
        A Camera on the CPU with pinhole intrinsics, extrinsics, name, and id.
    """
    intrinsics = build_camera_intrinsics(
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
    extrinsics = _make_extrinsics(
        translation=[0.3, -0.2, 1.1], extr_convention="opengl"
    )
    return Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        name="frame_0",
        id=7,
        device="cpu",
    )


def _make_multi_cameras() -> Cameras:
    """Build a multi-camera Cameras fixture spanning all three models.

    Args:
        None.

    Returns:
        A Cameras of three CPU cameras with mixed models, pose frames, names (one
        absent), and ids (one absent), exercising the has_name / has_id / sentinel
        paths.
    """
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
        ),
        build_camera_intrinsics(
            model="simple_pinhole",
            params={
                "f": 405.0,
                "cx": 161.0,
                "cy": 121.0,
                "h": 242,
                "w": 322,
            },
            intr_convention="standard",
            device="cpu",
        ),
        build_camera_intrinsics(
            model="ortho",
            params={
                "fx": 402.0,
                "fy": 412.0,
                "cx": 162.0,
                "cy": 122.0,
                "h": 244,
                "w": 324,
            },
            intr_convention="standard",
            device="cpu",
        ),
    ]
    extrinsics = [
        _make_extrinsics(translation=[0.3, -0.2, 1.1], extr_convention="opengl"),
        _make_extrinsics(translation=[1.3, 0.8, 2.1], extr_convention="opencv"),
        _make_extrinsics(translation=[2.3, 1.8, 3.1], extr_convention="standard"),
    ]
    names: List[Optional[str]] = ["frame_0", None, "frame_2"]
    ids: List[Optional[int]] = [7, 8, None]
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        names=names,
        ids=ids,
        device="cpu",
    )


def _assert_camera_fields_equal(loaded: Camera, original: Camera) -> None:
    """Assert two Camera objects carry the same core serialized fields.

    Args:
        loaded: Camera recovered from serialization.
        original: Original Camera.

    Returns:
        None.
    """
    assert isinstance(loaded, Camera), f"{type(loaded)=}"
    assert (
        loaded.intrinsics.model == original.intrinsics.model
    ), f"{loaded.intrinsics.model=} {original.intrinsics.model=}"
    assert (
        loaded.intrinsics.params == original.intrinsics.params
    ), f"{loaded.intrinsics.params=} {original.intrinsics.params=}"
    assert torch.equal(
        loaded.extrinsics.extrinsics, original.extrinsics.extrinsics
    ), f"{loaded.extrinsics.extrinsics=} {original.extrinsics.extrinsics=}"
    assert (
        loaded.extrinsics.extr_convention == original.extrinsics.extr_convention
    ), f"{loaded.extrinsics.extr_convention=} {original.extrinsics.extr_convention=}"
    assert loaded.name == original.name, f"{loaded.name=} {original.name=}"
    assert loaded.id == original.id, f"{loaded.id=} {original.id=}"


def _assert_cameras_fields_equal(loaded: Cameras, original: Cameras) -> None:
    """Assert two Cameras collections carry the same core serialized fields.

    Args:
        loaded: Cameras recovered from serialization.
        original: Original Cameras.

    Returns:
        None.
    """
    assert isinstance(loaded, Cameras), f"{type(loaded)=}"
    assert len(loaded) == len(original), f"{len(loaded)=} {len(original)=}"
    for index in range(len(original)):
        _assert_camera_fields_equal(loaded=loaded[index], original=original[index])


def test_single_camera_json_round_trip(tmp_path: Path) -> None:
    """A single Camera survives a save then load round trip through json.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="json")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert set(serialized.keys()) == _JSON_KEYS, f"{set(serialized.keys())=}"
    assert serialized == camera.serialize(format="json"), f"{serialized=}"

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="json")
    method_deserialized = Camera.deserialize(
        payload=serialized, device="cpu", format="json"
    )
    _assert_camera_fields_equal(loaded=deserialized, original=camera)
    _assert_camera_fields_equal(loaded=method_deserialized, original=camera)

    json_path = tmp_path / "camera.json"
    camera.save(camera_path=json_path)
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == serialized, f"{on_disk=} {serialized=}"
    _assert_camera_fields_equal(
        loaded=load_cameras(cameras_path=json_path, device="cpu"), original=camera
    )
    _assert_camera_fields_equal(
        loaded=Camera.load(camera_path=json_path, device="cpu"), original=camera
    )


def test_single_camera_npz_round_trip(tmp_path: Path) -> None:
    """A single Camera survives a save then load round trip through npz.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="npz")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert set(serialized.keys()) == _NPZ_KEYS, f"{set(serialized.keys())=}"
    assert serialized["extrinsics"].shape == (
        4,
        4,
    ), f"{serialized['extrinsics'].shape=}"

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="npz")
    _assert_camera_fields_equal(loaded=deserialized, original=camera)

    npz_path = tmp_path / "camera.npz"
    camera.save(camera_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == _NPZ_KEYS, f"{set(on_disk.files)=}"
    _assert_camera_fields_equal(
        loaded=load_cameras(cameras_path=npz_path, device="cpu"), original=camera
    )
    _assert_camera_fields_equal(
        loaded=Camera.load(camera_path=npz_path, device="cpu"), original=camera
    )


def test_multi_cameras_json_round_trip(tmp_path: Path) -> None:
    """A Cameras collection survives a save then load round trip through json.

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
        assert set(per_camera_dict.keys()) == _JSON_KEYS, f"{per_camera_dict=}"

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="json")
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    json_path = tmp_path / "cameras.json"
    save_cameras(cameras=cameras, cameras_path=json_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == serialized
    _assert_cameras_fields_equal(
        loaded=load_cameras(cameras_path=json_path, device="cpu"), original=cameras
    )


def test_multi_cameras_npz_round_trip(tmp_path: Path) -> None:
    """A Cameras collection survives a save then load round trip through npz.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    cameras = _make_multi_cameras()

    serialized = serialize_cameras(cameras=cameras, format="npz")
    assert isinstance(serialized, dict), f"{type(serialized)=}"
    assert set(serialized.keys()) == _NPZ_KEYS, f"{set(serialized.keys())=}"
    assert serialized["extrinsics"].shape == (
        len(cameras),
        4,
        4,
    ), f"{serialized['extrinsics'].shape=}"

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="npz")
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    npz_path = tmp_path / "cameras.npz"
    save_cameras(cameras=cameras, cameras_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == _NPZ_KEYS, f"{set(on_disk.files)=}"
    _assert_cameras_fields_equal(
        loaded=load_cameras(cameras_path=npz_path, device="cpu"), original=cameras
    )


def test_the_intr_convention_and_resolution_survive_round_trip() -> None:
    """The intrinsics' own frame and resolution ride through serialization with the params they name.

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
    ).to(intr_convention="opengl")
    camera = Camera(
        intrinsics=intrinsics,
        extrinsics=_make_extrinsics(
            translation=[0.1, 0.2, 0.3], extr_convention="opencv"
        ),
        name="two-frames",
        id=7,
        device="cpu",
    )
    for format in ("json", "npz"):
        payload = serialize_cameras(cameras=camera, format=format)
        loaded = deserialize_cameras(payload=payload, format=format, device="cpu")
        assert (
            loaded.intrinsics.intr_convention == intrinsics.intr_convention
        ), f"{format=} {loaded.intrinsics.intr_convention=}"
        assert (
            loaded.intrinsics.params["h"] == intrinsics.params["h"]
        ), f"{format=} {loaded.intrinsics.params=}"
        assert (
            loaded.intrinsics.params["w"] == intrinsics.params["w"]
        ), f"{format=} {loaded.intrinsics.params=}"
        assert (
            loaded.extrinsics.extr_convention == "opencv"
        ), f"{format=} {loaded.extrinsics.extr_convention=}"
        assert (
            loaded.intrinsics.intr_convention != loaded.extrinsics.extr_convention
        ), f"{format=} {loaded.intrinsics.intr_convention=}"


def test_model_and_params_survive_round_trip(tmp_path: Path) -> None:
    """A Camera's intrinsics model and params survive json and npz round trips.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    model_params = {
        "simple_pinhole": {
            "f": 405.0,
            "cx": 161.0,
            "cy": 121.0,
            "h": 242,
            "w": 322,
        },
        "pinhole": {
            "fx": 400.0,
            "fy": 410.0,
            "cx": 160.0,
            "cy": 120.0,
            "h": 240,
            "w": 320,
        },
        "ortho": {
            "fx": 402.0,
            "fy": 412.0,
            "cx": 162.0,
            "cy": 122.0,
            "h": 244,
            "w": 324,
        },
    }
    for format in ("json", "npz"):
        for model, params in model_params.items():
            intrinsics = build_camera_intrinsics(
                model=model,
                params=params,
                intr_convention="standard",
                device="cpu",
            )
            extrinsics = _make_extrinsics(
                translation=[0.3, -0.2, 1.1], extr_convention="opengl"
            )
            camera = Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                name="frame_0",
                id=7,
                device="cpu",
            )
            camera_path = tmp_path / f"camera_{model}.{format}"
            camera.save(camera_path=camera_path)
            loaded = Camera.load(camera_path=camera_path, device="cpu")
            assert (
                loaded.intrinsics.model == model
            ), f"{format=} {loaded.intrinsics.model=} {model=}"
            assert (
                loaded.intrinsics.params == params
            ), f"{format=} {loaded.intrinsics.params=} {params=}"


def test_tensor_intrinsics_params_round_trip_as_serialized_values(
    tmp_path: Path,
) -> None:
    """Tensor intrinsics params round-trip through camera I/O as numeric values.

    Args:
        tmp_path: Pytest-provided temporary directory for the round-trip file.

    Returns:
        None.
    """
    numeric_params = {
        "fx": 400.0,
        "fy": 410.0,
        "cx": 160.0,
        "cy": 120.0,
        "h": 240,
        "w": 320,
    }
    for format in ("json", "npz"):
        intrinsics = build_camera_intrinsics(
            model="ortho",
            params={
                key: torch.tensor(float(value), dtype=torch.float32)
                for key, value in numeric_params.items()
            },
            intr_convention="standard",
        )
        extrinsics = _make_extrinsics(
            translation=[0.3, -0.2, 1.1], extr_convention="standard"
        )
        camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
        camera_path = tmp_path / f"camera.{format}"
        camera.save(camera_path=camera_path)
        loaded = Camera.load(camera_path=camera_path)
        for key, value in numeric_params.items():
            assert torch.allclose(
                loaded.intrinsics.params[key],
                torch.tensor(float(value), dtype=loaded.intrinsics.params[key].dtype),
            ), (
                "Expected the loaded params to equal the source tensor values. "
                f"{format=} {key=} {loaded.intrinsics.params[key]=} {value=}"
            )


def test_extrinsics_and_extr_convention_survive_round_trip(tmp_path: Path) -> None:
    """A Camera's extrinsics matrix and extr_convention survive json and npz round trips.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    extr_conventions = ["standard", "opengl", "opencv", "pytorch3d", "arkit"]
    for format in ("json", "npz"):
        for extr_convention in extr_conventions:
            intrinsics = build_camera_intrinsics(
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
            extrinsics = _make_extrinsics(
                translation=[0.3, -0.2, 1.1], extr_convention=extr_convention
            )
            camera = Camera(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                name="frame_0",
                id=7,
                device="cpu",
            )
            camera_path = tmp_path / f"camera_{extr_convention}.{format}"
            camera.save(camera_path=camera_path)
            loaded = Camera.load(camera_path=camera_path, device="cpu")
            assert torch.equal(
                loaded.extrinsics.extrinsics, camera.extrinsics.extrinsics
            ), f"{format=} {extr_convention=} {loaded.extrinsics.extrinsics=}"
            assert (
                loaded.extrinsics.extr_convention == extr_convention
            ), f"{format=} {loaded.extrinsics.extr_convention=}"
