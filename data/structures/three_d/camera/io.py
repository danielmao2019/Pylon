"""Generic Cameras serialization and I/O helpers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import torch

from data.structures.three_d.camera.validation import (
    validate_camera_convention,
    validate_camera_extrinsics,
    validate_camera_intrinsics,
)

if TYPE_CHECKING:
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

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


def serialize_cameras(
    cameras: Union["Camera", "Cameras"],
    format: str = "json",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Serialize cameras to the canonical payload for the requested format.

    The four format helpers are plural-only. This dispatcher owns all single
    versus plural normalization: a single `Camera` is wrapped into a one-element
    `Cameras` on the way in, and the plural payload is reduced to its single form
    on the way out.

    Args:
        cameras: Either a single `Camera` or a `Cameras` collection to serialize.
        format: Serialization format, either `json` or `npz`.

    Returns:
        For `json`, a list of per-camera dicts for a `Cameras` or a bare dict for
        a single `Camera`. For `npz`, the batched-array payload for a `Cameras` or
        the same batched-array payload tagged with an `is_single` flag for a single
        `Camera`.
    """
    # Inline runtime imports; camera.py and cameras.py import this module, so a
    # module-top import would cycle.
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

    # Input validations
    assert isinstance(cameras, (Camera, Cameras)), (
        "Expected object to serialize to be a Camera or a Cameras. " f"{type(cameras)=}"
    )

    # Input normalizations
    format = _normalize_format(format=format)
    was_single = isinstance(cameras, Camera)
    if was_single:
        cameras = Cameras(
            intrinsics=[cameras.intrinsics],
            extrinsics=[cameras.extrinsics],
            conventions=[cameras.convention],
            names=[cameras.name],
            ids=[cameras.id],
            device=cameras.device,
        )

    if format == "json":
        payload = _serialize_cameras_json(cameras=cameras)
        if was_single:
            return payload[0]
        return payload

    if format == "npz":
        payload = _serialize_cameras_npz(cameras=cameras)
        if was_single:
            payload["is_single"] = np.array(True)
        return payload

    assert False, "Expected Cameras serialization format to be handled. " f"{format=}"


def deserialize_cameras(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    device: Optional[Union[str, torch.device]] = None,
    format: str = "json",
) -> Union["Camera", "Cameras"]:
    """Deserialize the canonical payload back into cameras.

    Inverse of `serialize_cameras`. The four format helpers are plural-only; this
    dispatcher owns all single versus plural normalization: it detects the single
    form, expands it to the plural form for the helper, and reduces the result back
    to a single `Camera` when the input was single.

    Args:
        payload: For `json`, a list of per-camera dicts (plural) or a bare dict
            (single). For `npz`, the batched-array payload, optionally tagged with
            an `is_single` flag.
        device: Target device for the deserialized cameras.
        format: Serialization format, either `json` or `npz`.

    Returns:
        A single `Camera` when the payload was in single form, otherwise a
        `Cameras` collection.
    """
    # Input validations
    assert isinstance(payload, (dict, list)), (
        "Expected Cameras payload to be a dictionary or a list. " f"{type(payload)=}"
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Cameras device to be None, a string, or a torch device. " f"{device=}"
    )

    # Input normalizations
    format = _normalize_format(format=format)
    target_device = torch.device(device) if device is not None else torch.device("cpu")

    if format == "json":
        was_single = isinstance(payload, dict)
        per_camera_dicts = [payload] if was_single else payload
        cameras = _deserialize_cameras_json(
            per_camera_dicts=per_camera_dicts,
            device=target_device,
        )
    elif format == "npz":
        assert isinstance(payload, dict), (
            "Expected Cameras NPZ payload to be a dictionary. " f"{type(payload)=}"
        )
        was_single = "is_single" in payload
        if was_single:
            payload = {
                key: value for key, value in payload.items() if key != "is_single"
            }
        cameras = _deserialize_cameras_npz(
            payload=payload,
            device=target_device,
        )
    else:
        assert False, (
            "Expected Cameras deserialization format to be handled. " f"{format=}"
        )

    if was_single:
        return cameras[0]
    return cameras


def save_cameras(cameras: Union["Camera", "Cameras"], cameras_path: Path) -> None:
    """Save cameras (a Cameras collection or a single Camera) to a file.

    Args:
        cameras: Either a single `Camera` or a `Cameras` collection to save.
        cameras_path: Output `.npz` or `.json` filepath.

    Returns:
        None.
    """
    # Input validations
    assert isinstance(cameras_path, Path), (
        "Expected Cameras output path to be a pathlib Path. " f"{type(cameras_path)=}"
    )

    # Input normalizations
    format = _resolve_format_from_path(cameras_path=cameras_path)

    payload = serialize_cameras(cameras=cameras, format=format)
    cameras_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        cameras_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    if format == "npz":
        np.savez(cameras_path, **payload)
        return

    assert False, "Expected Cameras save format to be handled. " f"{format=}"


def load_cameras(
    cameras_path: Path,
    device: Optional[Union[str, torch.device]] = None,
) -> Union["Camera", "Cameras"]:
    """Load cameras (a Cameras collection or a single Camera) from a file.

    Args:
        cameras_path: Input `.npz` or `.json` filepath.
        device: Target device for the loaded cameras.

    Returns:
        A single `Camera` when the file holds a single form, otherwise a `Cameras`
        collection.
    """
    # Input validations
    assert isinstance(cameras_path, Path), (
        "Expected Cameras input path to be a pathlib Path. " f"{type(cameras_path)=}"
    )
    assert cameras_path.exists(), (
        "Expected Cameras input path to exist. " f"{cameras_path=}"
    )
    assert cameras_path.is_file(), (
        "Expected Cameras input path to be a file. " f"{cameras_path=}"
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected Cameras device to be None, a string, or a torch device. " f"{device=}"
    )

    # Input normalizations
    format = _resolve_format_from_path(cameras_path=cameras_path)

    if format == "json":
        payload = json.loads(cameras_path.read_text(encoding="utf-8"))
        return deserialize_cameras(payload=payload, device=device, format=format)

    if format == "npz":
        with np.load(cameras_path, allow_pickle=False) as payload_file:
            payload = {key: payload_file[key] for key in payload_file.files}
        return deserialize_cameras(payload=payload, device=device, format=format)

    assert False, "Expected Cameras load format to be handled. " f"{format=}"


def _serialize_cameras_json(cameras: "Cameras") -> List[Dict[str, Any]]:
    """Map a Cameras to the plural json payload: one dict per camera.

    Args:
        cameras: A `Cameras` collection to serialize.

    Returns:
        A list with one per-camera json dict (keyed by `_CAMERA_JSON_KEYS`, with
        intrinsics and extrinsics as nested lists) for each camera in `cameras`.
    """
    per_camera_dicts: List[Dict[str, Any]] = []
    for camera in cameras:
        per_camera_dicts.append(
            {
                "intrinsics": camera.intrinsics.detach().cpu().tolist(),
                "extrinsics": camera.extrinsics.detach().cpu().tolist(),
                "convention": camera.convention,
                "name": camera.name,
                "id": camera.id,
            }
        )
    return per_camera_dicts


def _deserialize_cameras_json(
    per_camera_dicts: List[Dict[str, Any]],
    device: torch.device,
) -> "Cameras":
    """Map the plural json per-camera dicts to a Cameras.

    Args:
        per_camera_dicts: A list of per-camera json dicts (keyed by
            `_CAMERA_JSON_KEYS`, with intrinsics and extrinsics as nested lists).
        device: Target device for the decoded intrinsics and extrinsics tensors.

    Returns:
        A `Cameras` collection built from the per-camera dicts.
    """
    # Inline runtime import; cameras.py imports this module, so a module-top
    # import would cycle.
    from data.structures.three_d.camera.cameras import Cameras

    # Input validations
    assert isinstance(per_camera_dicts, list), (
        "Expected json per-camera payload to be a list. " f"{type(per_camera_dicts)=}"
    )
    assert len(per_camera_dicts) > 0, (
        "Expected json per-camera payload to be non-empty. " f"{len(per_camera_dicts)=}"
    )

    intrinsics_list: List[torch.Tensor] = []
    extrinsics_list: List[torch.Tensor] = []
    conventions: List[str] = []
    names: List[Optional[str]] = []
    ids: List[Optional[int]] = []
    for per_camera_dict in per_camera_dicts:
        assert isinstance(per_camera_dict, dict), (
            "Expected each json camera payload to be a dictionary. "
            f"{type(per_camera_dict)=}"
        )
        assert set(per_camera_dict.keys()) == _CAMERA_JSON_KEYS, (
            "Expected each json camera payload to contain exactly the Camera JSON "
            f"fields. {set(per_camera_dict.keys())=} {_CAMERA_JSON_KEYS=}"
        )
        assert isinstance(per_camera_dict["convention"], str), (
            "Expected json camera convention to be a string. "
            f"{type(per_camera_dict['convention'])=}"
        )
        validate_camera_convention(per_camera_dict["convention"])
        assert per_camera_dict["name"] is None or isinstance(
            per_camera_dict["name"], str
        ), (
            "Expected json camera name to be None or a string. "
            f"{type(per_camera_dict['name'])=}"
        )
        assert per_camera_dict["id"] is None or isinstance(
            per_camera_dict["id"], int
        ), (
            "Expected json camera id to be None or an integer. "
            f"{type(per_camera_dict['id'])=}"
        )

        intrinsics_list.append(
            torch.as_tensor(
                per_camera_dict["intrinsics"],
                dtype=torch.float32,
                device=device,
            )
        )
        extrinsics_list.append(
            torch.as_tensor(
                per_camera_dict["extrinsics"],
                dtype=torch.float32,
                device=device,
            )
        )
        conventions.append(per_camera_dict["convention"])
        names.append(per_camera_dict["name"])
        ids.append(per_camera_dict["id"])

    return Cameras(
        intrinsics=intrinsics_list,
        extrinsics=extrinsics_list,
        conventions=conventions,
        names=names,
        ids=ids,
        device=device,
    )


def _serialize_cameras_npz(cameras: "Cameras") -> Dict[str, Any]:
    """Map a Cameras to the plural batched-array npz payload.

    Args:
        cameras: A `Cameras` collection to serialize.

    Returns:
        The batched-array npz payload keyed by `_CAMERA_NPZ_KEYS`: stacked
        intrinsics `[N, 3, 3]` and extrinsics `[N, 4, 4]`, per-camera `convention`
        / `name` / `id` arrays of length N with `has_name` / `has_id` flag arrays
        and a `-1` id sentinel for absent ids.
    """
    intrinsics_list: List[np.ndarray] = []
    extrinsics_list: List[np.ndarray] = []
    conventions: List[str] = []
    names: List[str] = []
    has_names: List[bool] = []
    ids: List[int] = []
    has_ids: List[bool] = []
    for camera in cameras:
        intrinsics_list.append(
            np.asarray(camera.intrinsics.detach().cpu().tolist(), dtype=np.float32)
        )
        extrinsics_list.append(
            np.asarray(camera.extrinsics.detach().cpu().tolist(), dtype=np.float32)
        )
        conventions.append(camera.convention)
        names.append("" if camera.name is None else camera.name)
        has_names.append(camera.name is not None)
        ids.append(-1 if camera.id is None else camera.id)
        has_ids.append(camera.id is not None)

    return {
        "intrinsics": np.stack(intrinsics_list, axis=0),
        "extrinsics": np.stack(extrinsics_list, axis=0),
        "convention": np.array(conventions),
        "name": np.array(names),
        "has_name": np.array(has_names),
        "id": np.array(ids, dtype=np.int64),
        "has_id": np.array(has_ids),
    }


def _deserialize_cameras_npz(
    payload: Dict[str, Any], device: torch.device
) -> "Cameras":
    """Map the plural batched-array npz payload to a Cameras.

    Args:
        payload: The batched-array npz payload keyed by `_CAMERA_NPZ_KEYS`: stacked
            intrinsics `[N, 3, 3]` and extrinsics `[N, 4, 4]`, per-camera
            `convention` / `name` / `id` arrays of length N with `has_name` /
            `has_id` flag arrays and a `-1` id sentinel.
        device: Target device for the decoded intrinsics and extrinsics tensors.

    Returns:
        A `Cameras` collection built from the batched-array payload.
    """
    # Inline runtime import; cameras.py imports this module, so a module-top
    # import would cycle.
    from data.structures.three_d.camera.cameras import Cameras

    # Input validations
    assert isinstance(payload, dict), (
        "Expected Cameras NPZ payload to be a dictionary. " f"{type(payload)=}"
    )
    payload_keys = set(payload.keys())
    assert payload_keys == _CAMERA_NPZ_KEYS, (
        "Expected Cameras NPZ payload to match a supported schema. "
        f"{payload_keys=} {_CAMERA_NPZ_KEYS=}"
    )

    intrinsics = payload["intrinsics"]
    assert isinstance(intrinsics, np.ndarray), (
        "Expected Cameras NPZ intrinsics to be a numpy array. " f"{type(intrinsics)=}"
    )
    assert intrinsics.dtype == np.float32, (
        "Expected Cameras NPZ intrinsics to use float32. " f"{intrinsics.dtype=}"
    )
    assert intrinsics.ndim == 3, (
        "Expected Cameras NPZ intrinsics to be batched as [N, 3, 3]. "
        f"{intrinsics.shape=}"
    )
    validate_camera_intrinsics(intrinsics)

    extrinsics = payload["extrinsics"]
    assert isinstance(extrinsics, np.ndarray), (
        "Expected Cameras NPZ extrinsics to be a numpy array. " f"{type(extrinsics)=}"
    )
    assert extrinsics.dtype == np.float32, (
        "Expected Cameras NPZ extrinsics to use float32. " f"{extrinsics.dtype=}"
    )
    assert extrinsics.ndim == 3, (
        "Expected Cameras NPZ extrinsics to be batched as [N, 4, 4]. "
        f"{extrinsics.shape=}"
    )
    validate_camera_extrinsics(extrinsics)

    batch_size = intrinsics.shape[0]
    convention_array = payload["convention"]
    assert isinstance(convention_array, np.ndarray), (
        "Expected Cameras NPZ convention to be a numpy array. "
        f"{type(convention_array)=}"
    )
    assert convention_array.shape == (batch_size,), (
        "Expected Cameras NPZ convention array length to match the batch size. "
        f"{convention_array.shape=} {batch_size=}"
    )

    name_array = payload["name"]
    assert isinstance(name_array, np.ndarray), (
        "Expected Cameras NPZ name to be a numpy array. " f"{type(name_array)=}"
    )
    assert name_array.shape == (batch_size,), (
        "Expected Cameras NPZ name array length to match the batch size. "
        f"{name_array.shape=} {batch_size=}"
    )
    has_name_array = payload["has_name"]
    assert isinstance(has_name_array, np.ndarray), (
        "Expected Cameras NPZ has-name flags to be a numpy array. "
        f"{type(has_name_array)=}"
    )
    assert has_name_array.shape == (batch_size,), (
        "Expected Cameras NPZ has-name flag array length to match the batch size. "
        f"{has_name_array.shape=} {batch_size=}"
    )

    id_array = payload["id"]
    assert isinstance(id_array, np.ndarray), (
        "Expected Cameras NPZ id to be a numpy array. " f"{type(id_array)=}"
    )
    assert id_array.shape == (batch_size,), (
        "Expected Cameras NPZ id array length to match the batch size. "
        f"{id_array.shape=} {batch_size=}"
    )
    has_id_array = payload["has_id"]
    assert isinstance(has_id_array, np.ndarray), (
        "Expected Cameras NPZ has-id flags to be a numpy array. "
        f"{type(has_id_array)=}"
    )
    assert has_id_array.shape == (batch_size,), (
        "Expected Cameras NPZ has-id flag array length to match the batch size. "
        f"{has_id_array.shape=} {batch_size=}"
    )

    intrinsics_list: List[torch.Tensor] = []
    extrinsics_list: List[torch.Tensor] = []
    conventions: List[str] = []
    names: List[Optional[str]] = []
    ids: List[Optional[int]] = []
    for index in range(batch_size):
        intrinsics_list.append(
            torch.as_tensor(
                intrinsics[index],
                dtype=torch.float32,
                device=device,
            )
        )
        extrinsics_list.append(
            torch.as_tensor(
                extrinsics[index],
                dtype=torch.float32,
                device=device,
            )
        )

        convention = convention_array[index].item()
        assert isinstance(convention, str), (
            "Expected Cameras NPZ convention element to deserialize to a string. "
            f"{type(convention)=}"
        )
        validate_camera_convention(convention)
        conventions.append(convention)

        has_name = bool(has_name_array[index].item())
        name = name_array[index].item() if has_name else None
        assert name is None or isinstance(name, str), (
            "Expected Cameras NPZ name element to deserialize to None or a string. "
            f"{type(name)=}"
        )
        names.append(name)

        has_id = bool(has_id_array[index].item())
        camera_id = int(id_array[index].item()) if has_id else None
        ids.append(camera_id)

    return Cameras(
        intrinsics=intrinsics_list,
        extrinsics=extrinsics_list,
        conventions=conventions,
        names=names,
        ids=ids,
        device=device,
    )


def _resolve_format_from_path(cameras_path: Path) -> str:
    """Resolve a Cameras serialization format from a file path.

    Args:
        cameras_path: Cameras file path.

    Returns:
        Normalized serialization format name.
    """
    # Input validations
    assert isinstance(cameras_path, Path), (
        "Expected Cameras file path to be a pathlib Path. " f"{type(cameras_path)=}"
    )
    assert cameras_path.suffix != "", (
        "Expected Cameras file path to include a suffix. " f"{cameras_path=}"
    )

    return _normalize_format(format=cameras_path.suffix)


def _normalize_format(format: str) -> str:
    """Normalize a path suffix or format name to a supported serialization format.

    Args:
        format: Serialization format name or file suffix.

    Returns:
        Normalized serialization format name.
    """
    # Input validations
    assert isinstance(format, str), (
        "Expected Cameras serialization format to be a string. " f"{type(format)=}"
    )
    assert format != "", (
        "Expected Cameras serialization format to be non-empty. " f"{format=}"
    )

    # Input normalizations
    format = format.strip()
    assert format != "", (
        "Expected Cameras serialization format to be non-empty after stripping. "
        f"{format=}"
    )
    if format.startswith("."):
        format = format[1:]

    assert format in _CAMERA_SERIALIZATION_FORMATS, (
        "Expected Cameras serialization format to be supported. "
        f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
    )
    return format
