import json
from pathlib import Path
from typing import List

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
    "dtype",
    "name",
    "id",
}
_NPZ_KEYS = {
    "model",
    "params",
    "extrinsics",
    "intr_convention",
    "extr_convention",
    "dtype",
    "name",
    "has_name",
    "id",
    "has_id",
}


def test_single_camera_json_round_trip(tmp_path: Path) -> None:
    """A single Camera survives a save then load round trip through the json format.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="json")
    assert isinstance(
        serialized, dict
    ), f"Expected a single Camera to serialize to one json dict. {type(serialized)=}"
    assert set(serialized.keys()) == _JSON_KEYS, (
        "Expected the json payload to carry exactly the json key set. "
        f"{set(serialized.keys())=} {_JSON_KEYS=}"
    )
    method_serialized = camera.serialize(format="json")
    assert serialized == method_serialized, (
        "Expected serialize_cameras to produce the payload Camera.serialize produces. "
        f"{serialized=} {method_serialized=}"
    )

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="json")
    method_deserialized = Camera.deserialize(
        payload=serialized, device="cpu", format="json"
    )
    _assert_camera_fields_equal(loaded=deserialized, original=camera)
    _assert_camera_fields_equal(loaded=method_deserialized, original=camera)

    json_path = tmp_path / "camera.json"
    camera.save(camera_path=json_path)
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == serialized, (
        "Expected the saved json file to parse back to the serialized payload. "
        f"{on_disk=} {serialized=}"
    )
    loaded = load_cameras(cameras_path=json_path, device="cpu")
    method_loaded = Camera.load(camera_path=json_path, device="cpu")
    _assert_camera_fields_equal(loaded=loaded, original=camera)
    _assert_camera_fields_equal(loaded=method_loaded, original=camera)


def test_single_camera_npz_round_trip(tmp_path: Path) -> None:
    """A single Camera survives a save then load round trip through the npz format.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    camera = _make_single_camera()

    serialized = serialize_cameras(cameras=camera, format="npz")
    assert isinstance(
        serialized, dict
    ), f"Expected a single Camera to serialize to one npz dict. {type(serialized)=}"
    assert set(serialized.keys()) == _NPZ_KEYS, (
        "Expected the npz payload to carry exactly the npz key set. "
        f"{set(serialized.keys())=} {_NPZ_KEYS=}"
    )
    assert serialized["extrinsics"].shape == (4, 4), (
        "Expected a single Camera's npz extrinsics entry to be one 4x4 matrix. "
        f"{serialized['extrinsics'].shape=}"
    )

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="npz")
    _assert_camera_fields_equal(loaded=deserialized, original=camera)

    npz_path = tmp_path / "camera.npz"
    camera.save(camera_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == _NPZ_KEYS, (
            "Expected the saved npz archive to carry exactly the npz key set. "
            f"{set(on_disk.files)=} {_NPZ_KEYS=}"
        )
    loaded = load_cameras(cameras_path=npz_path, device="cpu")
    method_loaded = Camera.load(camera_path=npz_path, device="cpu")
    _assert_camera_fields_equal(loaded=loaded, original=camera)
    _assert_camera_fields_equal(loaded=method_loaded, original=camera)


def _make_single_camera() -> Camera:
    """Builds the one-camera Camera fixture both single round trips run on, carrying a name and an id so the round trip has both to preserve.

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


def test_multi_cameras_json_round_trip(tmp_path: Path) -> None:
    """A Cameras collection survives a save then load round trip through the json format.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    cameras = _make_multi_cameras()

    serialized = serialize_cameras(cameras=cameras, format="json")
    assert isinstance(
        serialized, list
    ), f"Expected a Cameras collection to serialize to a json list. {type(serialized)=}"
    assert len(serialized) == len(
        cameras
    ), f"Expected one json dict per camera. {len(serialized)=} {len(cameras)=}"
    for per_camera_dict in serialized:
        assert set(per_camera_dict.keys()) == _JSON_KEYS, (
            "Expected every per-camera json dict to carry exactly the json key set. "
            f"{set(per_camera_dict.keys())=} {_JSON_KEYS=}"
        )

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="json")
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    json_path = tmp_path / "cameras.json"
    save_cameras(cameras=cameras, cameras_path=json_path)
    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk == serialized, (
        "Expected the saved json file to parse back to the serialized payload. "
        f"{on_disk=} {serialized=}"
    )
    loaded = load_cameras(cameras_path=json_path, device="cpu")
    _assert_cameras_fields_equal(loaded=loaded, original=cameras)


def test_multi_cameras_npz_round_trip(tmp_path: Path) -> None:
    """A Cameras collection survives a save then load round trip through the npz format.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    cameras = _make_multi_cameras()

    serialized = serialize_cameras(cameras=cameras, format="npz")
    assert isinstance(
        serialized, dict
    ), f"Expected a Cameras collection to serialize to one npz dict. {type(serialized)=}"
    assert set(serialized.keys()) == _NPZ_KEYS, (
        "Expected the npz payload to carry exactly the npz key set. "
        f"{set(serialized.keys())=} {_NPZ_KEYS=}"
    )
    assert serialized["extrinsics"].shape == (len(cameras), 4, 4), (
        "Expected the npz extrinsics entry to hold one 4x4 block per camera. "
        f"{serialized['extrinsics'].shape=} {len(cameras)=}"
    )

    deserialized = deserialize_cameras(payload=serialized, device="cpu", format="npz")
    _assert_cameras_fields_equal(loaded=deserialized, original=cameras)

    npz_path = tmp_path / "cameras.npz"
    save_cameras(cameras=cameras, cameras_path=npz_path)
    with np.load(npz_path, allow_pickle=False) as on_disk:
        assert set(on_disk.files) == _NPZ_KEYS, (
            "Expected the saved npz archive to carry exactly the npz key set. "
            f"{set(on_disk.files)=} {_NPZ_KEYS=}"
        )
    loaded = load_cameras(cameras_path=npz_path, device="cpu")
    _assert_cameras_fields_equal(loaded=loaded, original=cameras)


def test_round_trip_keeps_the_batch_dtype(tmp_path: Path) -> None:
    """Both formats record the batch's dtype and rebuild both components in it, so a batch loads back in the dtype it was saved in rather than one the format imposes.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    for format in ("json", "npz"):
        for dtype in (torch.float32, torch.float64):
            # A third has no exact float32 spelling, so a float32 detour on the way back would change it.
            intrinsics = build_camera_intrinsics(
                model="pinhole",
                params={
                    "fx": torch.tensor([1000.0, 1003.0, 1006.0], dtype=torch.float64)
                    / 3.0,
                    "fy": torch.tensor([1001.0, 1004.0, 1007.0], dtype=torch.float64)
                    / 3.0,
                    "cx": torch.tensor([481.0, 484.0, 487.0], dtype=torch.float64)
                    / 3.0,
                    "cy": torch.tensor([361.0, 364.0, 367.0], dtype=torch.float64)
                    / 3.0,
                    "h": torch.full((3,), 240.0, dtype=torch.float64),
                    "w": torch.full((3,), 320.0, dtype=torch.float64),
                },
                intr_convention="standard",
                device="cpu",
                dtype=dtype,
            )
            matrices = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
            matrices[:, :3, 3] = (
                torch.tensor(
                    [[1.0, 2.0, 4.0], [5.0, 7.0, 8.0], [10.0, 11.0, 13.0]],
                    dtype=torch.float64,
                )
                / 3.0
            )
            extrinsics = CameraExtrinsics(
                extrinsics=matrices,
                extr_convention="opengl",
                device="cpu",
                dtype=dtype,
            )
            cameras = Cameras(
                intrinsics=intrinsics, extrinsics=extrinsics, device="cpu"
            )
            cameras_path = tmp_path / f"{dtype}.{format}"
            save_cameras(cameras=cameras, cameras_path=cameras_path)
            loaded = load_cameras(cameras_path=cameras_path, device="cpu")
            param_dtypes = {}
            for key, value in loaded.intrinsics.params.items():
                param_dtypes[key] = value.dtype
            assert (
                loaded.dtype == dtype
                and loaded.extrinsics.dtype == dtype
                and loaded.extrinsics.extrinsics.dtype == dtype
                and loaded.intrinsics.dtype == dtype
            ), (
                "Expected the loaded batch and both its components to carry the dtype "
                "it was saved in. "
                f"{format=} {dtype=} {loaded.dtype=} "
                f"{loaded.extrinsics.extrinsics.dtype=} {param_dtypes=}"
            )
            for param_dtype in param_dtypes.values():
                assert param_dtype == dtype, (
                    "Expected every intrinsics param to carry the dtype it was saved "
                    "in. "
                    f"{format=} {dtype=} {param_dtypes=}"
                )
            assert torch.equal(
                loaded.extrinsics.extrinsics, cameras.extrinsics.extrinsics
            ), (
                "Expected the loaded extrinsics stack to equal the saved one exactly. "
                f"{format=} {dtype=} {loaded.extrinsics.extrinsics=} "
                f"{cameras.extrinsics.extrinsics=}"
            )
            for key, value in cameras.intrinsics.params.items():
                assert torch.equal(loaded.intrinsics.params[key], value), (
                    "Expected every loaded intrinsics param to equal the saved one "
                    "exactly. "
                    f"{format=} {dtype=} {key=} {loaded.intrinsics.params[key]=} "
                    f"{value=}"
                )


def _make_multi_cameras() -> Cameras:
    """Builds the three-camera Cameras fixture both collection round trips run on, its cameras differing in param values, centre, name and id so the payload spans every per-camera path the format has to carry.

    Args:
        None.

    Returns:
        A Cameras of three CPU cameras carrying one batched CameraIntrinsics and one batched CameraExtrinsics, so the batch names one model and one pose frame while every projection param, pose, name (one absent) and id (one absent) still varies per camera, exercising the has_name / has_id / sentinel paths.
    """
    intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={
            "fx": torch.tensor([400.0, 405.0, 402.0]),
            "fy": torch.tensor([410.0, 415.0, 412.0]),
            "cx": torch.tensor([160.0, 161.0, 162.0]),
            "cy": torch.tensor([120.0, 121.0, 122.0]),
            "h": torch.tensor([240.0, 242.0, 244.0]),
            "w": torch.tensor([320.0, 322.0, 324.0]),
        },
        intr_convention="standard",
        device="cpu",
    )
    matrices = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    matrices[:, :3, 3] = torch.tensor(
        [[0.3, -0.2, 1.1], [1.3, 0.8, 2.1], [2.3, 1.8, 3.1]],
        dtype=torch.float32,
    )
    extrinsics = CameraExtrinsics(
        extrinsics=matrices, extr_convention="opengl", device="cpu"
    )
    names = ["frame_0", None, "frame_2"]
    ids = [7, 8, None]
    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        names=names,
        ids=ids,
        device="cpu",
    )


def test_the_intr_convention_and_resolution_survive_round_trip() -> None:
    """An intrinsics' params name nothing without the frame they are stated in, so a payload that dropped it would deserialize into a different camera; the resolution needs no key of its own, riding inside those params.

    Args:
        None.

    Returns:
        None.
    """
    standard_intrinsics = build_camera_intrinsics(
        model="pinhole",
        params={"fx": 400.0, "fy": 410.0, "cx": 150.0, "cy": 110.0, "h": 240, "w": 320},
        intr_convention="standard",
        device="cpu",
    )
    intrinsics = standard_intrinsics.to(intr_convention="opengl")
    extrinsics = _make_extrinsics(translation=[0.1, 0.2, 0.3], extr_convention="opencv")
    camera = Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        name="two-frames",
        id=7,
        device="cpu",
    )
    for format in ("json", "npz"):
        payload = serialize_cameras(cameras=camera, format=format)
        loaded = deserialize_cameras(payload=payload, device="cpu", format=format)
        assert loaded.intrinsics.intr_convention == intrinsics.intr_convention, (
            "Expected the intr_convention to survive the round trip. "
            f"{format=} {loaded.intrinsics.intr_convention=} {intrinsics.intr_convention=}"
        )
        assert loaded.intrinsics.params["h"] == intrinsics.params["h"], (
            "Expected the h param to survive the round trip. "
            f"{format=} {loaded.intrinsics.params=} {intrinsics.params=}"
        )
        assert loaded.intrinsics.params["w"] == intrinsics.params["w"], (
            "Expected the w param to survive the round trip. "
            f"{format=} {loaded.intrinsics.params=} {intrinsics.params=}"
        )
        assert loaded.extrinsics.extr_convention == "opencv", (
            "Expected the extr_convention to survive the round trip. "
            f"{format=} {loaded.extrinsics.extr_convention=}"
        )
        assert loaded.intrinsics.intr_convention != loaded.extrinsics.extr_convention, (
            "Expected the intr_convention and the extr_convention to come back "
            "independently. "
            f"{format=} {loaded.intrinsics.intr_convention=} "
            f"{loaded.extrinsics.extr_convention=}"
        )


def test_model_and_params_survive_round_trip(tmp_path: Path) -> None:
    """A Camera's intrinsics model and params survive a save then load round trip through both the json and npz formats.

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
            assert loaded.intrinsics.model == model, (
                "Expected the intrinsics model to survive the round trip. "
                f"{format=} {loaded.intrinsics.model=} {model=}"
            )
            assert loaded.intrinsics.params == params, (
                "Expected the intrinsics params to survive the round trip. "
                f"{format=} {loaded.intrinsics.params=} {params=}"
            )


def test_tensor_intrinsics_params_round_trip_as_serialized_values(
    tmp_path: Path,
) -> None:
    """Tensor-valued intrinsics params round-trip through camera I/O as serialized numeric values.

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
    tensor_params = {}
    for key, value in numeric_params.items():
        tensor_params[key] = torch.tensor(float(value), dtype=torch.float32)
    for format in ("json", "npz"):
        intrinsics = build_camera_intrinsics(
            model="ortho", params=tensor_params, intr_convention="standard"
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
    """A Camera's extrinsics matrix and extr_convention survive a save then load round trip through both the json and npz formats.

    Args:
        tmp_path: Temporary output directory.

    Returns:
        None.
    """
    for format in ("json", "npz"):
        for extr_convention in ("standard", "opengl", "opencv", "pytorch3d", "arkit"):
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
            ), (
                "Expected the extrinsics matrix to survive the round trip exactly. "
                f"{format=} {extr_convention=} {loaded.extrinsics.extrinsics=} "
                f"{camera.extrinsics.extrinsics=}"
            )
            assert loaded.extrinsics.extr_convention == extr_convention, (
                "Expected the extr_convention to survive the round trip. "
                f"{format=} {loaded.extrinsics.extr_convention=} {extr_convention=}"
            )


def _make_extrinsics(
    translation: List[float], extr_convention: str
) -> CameraExtrinsics:
    """Builds one CameraExtrinsics whose rotation is identity, so a round trip is measured on the centre and the pose frame alone.

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


def _assert_cameras_fields_equal(loaded: Cameras, original: Cameras) -> None:
    """Checks a loaded Cameras against the original by running the single-camera check at every index.

    Args:
        loaded: Cameras recovered from serialization.
        original: Original Cameras.

    Returns:
        None.
    """
    assert isinstance(
        loaded, Cameras
    ), f"Expected the loaded object to be a Cameras. {type(loaded)=}"
    assert len(loaded) == len(original), (
        "Expected the loaded Cameras to hold as many cameras as the original. "
        f"{len(loaded)=} {len(original)=}"
    )
    for index in range(len(original)):
        _assert_camera_fields_equal(loaded=loaded[index], original=original[index])


def _assert_camera_fields_equal(loaded: Camera, original: Camera) -> None:
    """Checks a loaded Camera against the original on the fields serialization has to carry.

    Args:
        loaded: Camera recovered from serialization.
        original: Original Camera.

    Returns:
        None.
    """
    assert isinstance(
        loaded, Camera
    ), f"Expected the loaded object to be a Camera. {type(loaded)=}"
    assert loaded.intrinsics.model == original.intrinsics.model, (
        "Expected the loaded intrinsics model to equal the original's. "
        f"{loaded.intrinsics.model=} {original.intrinsics.model=}"
    )
    assert loaded.intrinsics.params == original.intrinsics.params, (
        "Expected the loaded intrinsics params to equal the original's. "
        f"{loaded.intrinsics.params=} {original.intrinsics.params=}"
    )
    assert torch.equal(loaded.extrinsics.extrinsics, original.extrinsics.extrinsics), (
        "Expected the loaded extrinsics matrix to equal the original's exactly. "
        f"{loaded.extrinsics.extrinsics=} {original.extrinsics.extrinsics=}"
    )
    assert loaded.extrinsics.extr_convention == original.extrinsics.extr_convention, (
        "Expected the loaded extr_convention to equal the original's. "
        f"{loaded.extrinsics.extr_convention=} {original.extrinsics.extr_convention=}"
    )
    assert (
        loaded.name == original.name
    ), f"Expected the loaded name to equal the original's. {loaded.name=} {original.name=}"
    assert (
        loaded.id == original.id
    ), f"Expected the loaded id to equal the original's. {loaded.id=} {original.id=}"
