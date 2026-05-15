"""Generic Camera serialization and I/O helpers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch

from data.structures.three_d.camera.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_intrinsics,
)

if TYPE_CHECKING:
    from data.structures.three_d.camera.camera import Camera

_CAMERA_SERIALIZATION_FORMATS = {
    "json",
    "npz",
}
_CAMERA_JSON_KEYS = {
    "intrinsics",
    "extrinsics",
    "convention",
    "name",
    "id",
}
_CAMERA_NPZ_KEYS = {
    "intrinsics",
    "extrinsics",
    "convention",
    "name",
    "has_name",
    "id",
    "has_id",
}


def serialize_camera(
    camera: "Camera",
    format: str = "json",
) -> Dict[str, Any]:
    """Serialize one Camera into a JSON-compatible or NPZ-field payload.

    Args:
        camera: Camera object to serialize.
        format: Serialization format, either `json` or `npz`.

    Returns:
        Camera payload for the requested format.
    """
    # Input validations
    Camera = _get_camera_class()
    assert isinstance(camera, Camera), (
        "Expected object to serialize to be a Camera. " f"{type(camera)=}"
    )

    # Input normalizations
    format = _normalize_format(format=format)

    json_payload = {
        "intrinsics": camera.intrinsics.detach().cpu().tolist(),
        "extrinsics": camera.extrinsics.detach().cpu().tolist(),
        "convention": camera.convention,
        "name": camera.name,
        "id": camera.id,
    }
    if format == "json":
        return json_payload

    if format == "npz":
        return {
            "intrinsics": np.asarray(json_payload["intrinsics"], dtype=np.float32),
            "extrinsics": np.asarray(json_payload["extrinsics"], dtype=np.float32),
            "convention": np.array(json_payload["convention"]),
            "name": np.array(
                "" if json_payload["name"] is None else json_payload["name"]
            ),
            "has_name": np.array(json_payload["name"] is not None),
            "id": np.array(
                -1 if json_payload["id"] is None else json_payload["id"],
                dtype=np.int64,
            ),
            "has_id": np.array(json_payload["id"] is not None),
        }

    assert False, "Expected Camera serialization format to be handled. " f"{format=}"


def deserialize_camera(
    payload: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    format: str = "json",
) -> "Camera":
    """Deserialize one Camera from a JSON-compatible or NPZ-field payload.

    Args:
        payload: Camera payload for the specified format.
        device: Target device for the deserialized Camera.
        format: Serialization format, either `json` or `npz`.

    Returns:
        Camera object represented by the payload.
    """
    # Input validations
    assert isinstance(payload, dict), (
        "Expected Camera payload to be a dictionary. " f"{type(payload)=}"
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Camera device to be None, a string, or a torch device. " f"{device=}"
    )

    # Input normalizations
    format = _normalize_format(format=format)
    if format == "npz":
        payload = _deserialize_npz_camera_payload(payload=payload)

    _validate_json_camera_payload(payload=payload)

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    intrinsics = torch.as_tensor(
        payload["intrinsics"],
        dtype=torch.float32,
        device=target_device,
    )
    extrinsics = torch.as_tensor(
        payload["extrinsics"],
        dtype=torch.float32,
        device=target_device,
    )

    Camera = _get_camera_class()
    return Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        convention=payload["convention"],
        name=payload["name"],
        id=payload["id"],
        device=target_device,
    )


def save_camera(camera: "Camera", camera_path: Path) -> None:
    """Save one Camera to a `.npz` or `.json` file.

    Args:
        camera: Camera object to save.
        camera_path: Output `.npz` or `.json` filepath.

    Returns:
        None.
    """
    # Input validations
    Camera = _get_camera_class()
    assert isinstance(camera, Camera), (
        "Expected object to save to be a Camera. " f"{type(camera)=}"
    )
    assert isinstance(camera_path, Path), (
        "Expected Camera output path to be a pathlib Path. " f"{type(camera_path)=}"
    )

    # Input normalizations
    format = _resolve_format_from_path(camera_path=camera_path)

    camera_payload = serialize_camera(
        camera=camera,
        format=format,
    )
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        camera_path.write_text(
            json.dumps(camera_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    if format == "npz":
        np.savez(camera_path, **camera_payload)
        return

    assert False, "Expected Camera save format to be handled. " f"{format=}"


def load_camera(
    camera_path: Path,
    device: Optional[Union[str, torch.device]] = None,
) -> "Camera":
    """Load one Camera from a `.npz` or `.json` file.

    Args:
        camera_path: Input `.npz` or `.json` filepath.
        device: Target device for the loaded Camera.

    Returns:
        Camera object loaded from disk.
    """
    # Input validations
    assert isinstance(camera_path, Path), (
        "Expected Camera input path to be a pathlib Path. " f"{type(camera_path)=}"
    )
    assert camera_path.exists(), (
        "Expected Camera input path to exist. " f"{camera_path=}"
    )
    assert camera_path.is_file(), (
        "Expected Camera input path to be a file. " f"{camera_path=}"
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Camera device to be None, a string, or a torch device. " f"{device=}"
    )

    # Input normalizations
    format = _resolve_format_from_path(camera_path=camera_path)

    if format == "json":
        camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
        return deserialize_camera(
            payload=camera_payload,
            device=device,
            format=format,
        )

    if format == "npz":
        with np.load(camera_path, allow_pickle=False) as camera_payload_file:
            camera_payload = {
                key: camera_payload_file[key] for key in camera_payload_file.files
            }
        return deserialize_camera(
            payload=camera_payload,
            device=device,
            format=format,
        )

    assert False, "Expected Camera load format to be handled. " f"{format=}"


def _get_camera_class() -> Any:
    """Load the Camera class without creating an import cycle.

    Args:
        None.

    Returns:
        Camera class object.
    """
    # This local import is intentional: camera.py imports this module so Camera
    # methods can be direct API delegates, while this module still needs the
    # runtime class for validation and object construction.
    from data.structures.three_d.camera.camera import Camera

    return Camera


def _resolve_format_from_path(camera_path: Path) -> str:
    """Resolve a Camera serialization format from a file path.

    Args:
        camera_path: Camera file path.

    Returns:
        Normalized serialization format name.
    """
    # Input validations
    assert isinstance(camera_path, Path), (
        "Expected Camera file path to be a pathlib Path. " f"{type(camera_path)=}"
    )
    assert camera_path.suffix != "", (
        "Expected Camera file path to include a suffix. " f"{camera_path=}"
    )

    return _normalize_format(format=camera_path.suffix)


def _normalize_format(format: str) -> str:
    """Normalize a Camera serialization format name.

    Args:
        format: Serialization format name or file suffix.

    Returns:
        Normalized serialization format name.
    """
    # Input validations
    assert isinstance(format, str), (
        "Expected Camera serialization format to be a string. " f"{type(format)=}"
    )
    assert format != "", (
        "Expected Camera serialization format to be non-empty. " f"{format=}"
    )

    # Input normalizations
    format = format.strip()
    assert format != "", (
        "Expected Camera serialization format to be non-empty after stripping. "
        f"{format=}"
    )
    if format.startswith("."):
        format = format[1:]

    assert format in _CAMERA_SERIALIZATION_FORMATS, (
        "Expected Camera serialization format to be supported. "
        f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
    )
    return format


def _deserialize_npz_camera_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize an NPZ-field Camera payload into the generic payload.

    Args:
        payload: NPZ-field Camera payload.

    Returns:
        Generic Camera payload with intrinsics, extrinsics, convention, name, and id.
    """
    # Input validations
    assert isinstance(payload, dict), (
        "Expected Camera NPZ payload to be a dictionary. " f"{type(payload)=}"
    )
    payload_keys = set(payload.keys())
    assert payload_keys == _CAMERA_NPZ_KEYS, (
        "Expected Camera NPZ payload to match a supported schema. "
        f"{payload_keys=} {_CAMERA_NPZ_KEYS=}"
    )

    intrinsics = payload["intrinsics"]
    assert isinstance(intrinsics, np.ndarray), (
        "Expected Camera NPZ intrinsics to be a numpy array. " f"{type(intrinsics)=}"
    )
    assert intrinsics.dtype == np.float32, (
        "Expected Camera NPZ intrinsics to use float32. " f"{intrinsics.dtype=}"
    )
    validate_camera_intrinsics(intrinsics)

    extrinsics = payload["extrinsics"]
    assert isinstance(extrinsics, np.ndarray), (
        "Expected Camera NPZ extrinsics to be a numpy array. " f"{type(extrinsics)=}"
    )
    assert extrinsics.dtype == np.float32, (
        "Expected Camera NPZ extrinsics to use float32. " f"{extrinsics.dtype=}"
    )
    validate_camera_extrinsics(extrinsics)

    convention_array = payload["convention"]
    assert isinstance(convention_array, np.ndarray), (
        "Expected Camera NPZ convention to be a numpy scalar array. "
        f"{type(convention_array)=}"
    )
    assert convention_array.shape == (), (
        "Expected Camera NPZ convention to be scalar. " f"{convention_array.shape=}"
    )
    convention = convention_array.item()
    assert isinstance(convention, str), (
        "Expected Camera NPZ convention scalar to deserialize to a string. "
        f"{type(convention)=}"
    )
    validate_camera_convention(convention)

    has_name_array = payload["has_name"]
    assert isinstance(has_name_array, np.ndarray), (
        "Expected Camera NPZ has-name flag to be a numpy scalar array. "
        f"{type(has_name_array)=}"
    )
    assert has_name_array.shape == (), (
        "Expected Camera NPZ has-name flag to be scalar. " f"{has_name_array.shape=}"
    )
    has_name = bool(has_name_array.item())
    name_array = payload["name"]
    assert isinstance(name_array, np.ndarray), (
        "Expected Camera NPZ name to be a numpy scalar array. " f"{type(name_array)=}"
    )
    assert name_array.shape == (), (
        "Expected Camera NPZ name to be scalar. " f"{name_array.shape=}"
    )
    name = name_array.item() if has_name else None
    assert name is None or isinstance(name, str), (
        "Expected Camera NPZ name scalar to deserialize to None or a string. "
        f"{type(name)=}"
    )

    has_id_array = payload["has_id"]
    assert isinstance(has_id_array, np.ndarray), (
        "Expected Camera NPZ has-id flag to be a numpy scalar array. "
        f"{type(has_id_array)=}"
    )
    assert has_id_array.shape == (), (
        "Expected Camera NPZ has-id flag to be scalar. " f"{has_id_array.shape=}"
    )
    has_id = bool(has_id_array.item())
    id_array = payload["id"]
    assert isinstance(id_array, np.ndarray), (
        "Expected Camera NPZ id to be a numpy scalar array. " f"{type(id_array)=}"
    )
    assert id_array.shape == (), (
        "Expected Camera NPZ id to be scalar. " f"{id_array.shape=}"
    )
    camera_id = int(id_array.item()) if has_id else None

    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "convention": convention,
        "name": name,
        "id": camera_id,
    }


def _validate_json_camera_payload(payload: Dict[str, Any]) -> None:
    """Validate a generic JSON-compatible Camera payload.

    Args:
        payload: Camera payload with intrinsics, extrinsics, convention, name, and id.

    Returns:
        None.
    """
    # Input validations
    assert isinstance(payload, dict), (
        "Expected serialized Camera payload to be a dictionary. " f"{type(payload)=}"
    )
    assert set(payload.keys()) == _CAMERA_JSON_KEYS, (
        "Expected serialized Camera payload to contain exactly the Camera JSON fields. "
        f"{set(payload.keys())=} {_CAMERA_JSON_KEYS=}"
    )
    assert isinstance(payload["convention"], str), (
        "Expected serialized Camera convention to be a string. "
        f"{type(payload['convention'])=}"
    )
    validate_camera_convention(payload["convention"])
    assert payload["name"] is None or isinstance(payload["name"], str), (
        "Expected serialized Camera name to be None or a string. "
        f"{type(payload['name'])=}"
    )
    assert payload["id"] is None or isinstance(payload["id"], int), (
        "Expected serialized Camera id to be None or an integer. "
        f"{type(payload['id'])=}"
    )
