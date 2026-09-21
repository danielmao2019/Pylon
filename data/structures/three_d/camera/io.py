"""Generic Camera / Cameras serialization and I/O helpers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    CameraIntrinsics,
    build_camera_intrinsics,
)

if TYPE_CHECKING:
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

_CAMERA_SERIALIZATION_FORMATS = {
    "json",
    "npz",
}
_CAMERA_JSON_KEYS, _CAMERA_NPZ_KEYS = {
    "model",
    "params",
    "intr_convention",
    "extrinsics",
    "extr_convention",
    "dtype",
    "name",
    "id",
}, {
    "model",
    "params",
    "intr_convention",
    "extrinsics",
    "extr_convention",
    "dtype",
    "name",
    "id",
}


def save_cameras(cameras: Union["Camera", "Cameras"], cameras_path: Path) -> None:
    """Save cameras (a Cameras collection or a single Camera) to a .npz or .json file.

    Args:
        cameras: Either a single `Camera` or a `Cameras` collection to save.
        cameras_path: Output `.npz` or `.json` filepath.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras_path, Path), (
            "Expected Cameras output path to be a pathlib Path. "
            f"{type(cameras_path)=}"
        )

    _validate_inputs()

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

    assert 0, "Should not reach here. " f"{format=}"


def load_cameras(
    cameras_path: Path,
    device: Optional[Union[str, torch.device]] = None,
) -> Union["Camera", "Cameras"]:
    """Load cameras (a Cameras collection or a single Camera) from a .npz or .json file.

    Args:
        cameras_path: Input `.npz` or `.json` filepath.
        device: Target device for the loaded cameras; None loads them on cpu.

    Returns:
        A single `Camera` when the file holds one unbatched `[4, 4]` pose, otherwise a `Cameras` collection for a batched `[N, 4, 4]` one.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras_path, Path), (
            "Expected Cameras input path to be a pathlib Path. "
            f"{type(cameras_path)=}"
        )
        assert cameras_path.exists(), (
            "Expected Cameras input path to exist. " f"{cameras_path=}"
        )
        assert cameras_path.is_file(), (
            "Expected Cameras input path to be a file. " f"{cameras_path=}"
        )
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected Cameras device to be None, a string, or a torch device. "
            f"{device=}"
        )

    _validate_inputs()

    format = _resolve_format_from_path(cameras_path=cameras_path)

    def _read_payload() -> Dict[str, Any]:
        """Read the file in the form its own format spells it.

        Args:
            None; reads the resolved `format` and the `cameras_path` of the call.

        Returns:
            The payload the file holds: the parsed json dict, or every npz array keyed by its name.
        """
        if format == "json":
            payload = json.loads(cameras_path.read_text(encoding="utf-8"))
            return payload
        if format == "npz":
            with np.load(cameras_path, allow_pickle=False) as payload_file:
                payload = {key: payload_file[key] for key in payload_file.files}
            return payload
        assert 0, "Should not reach here. " f"{format=}"

    payload = _read_payload()

    return deserialize_cameras(payload=payload, device=device, format=format)


def serialize_cameras(
    cameras: Union["Camera", "Cameras"],
    format: str = "json",
) -> Dict[str, Any]:
    """Serialize cameras to the canonical payload for the requested format, a single Camera's pose, name and id spelled as it holds them.

    Args:
        cameras: Either a single `Camera` or a `Cameras` collection to serialize.
        format: Serialization format, either `json` or `npz`.

    Returns:
        The payload keyed by the format's key set: for `json` a dict of json-native values, for `npz` a dict of numpy arrays; a `Camera`'s pose is one `[4, 4]` and its name and id single values, a `Cameras`' pose an `[N, 4, 4]` stack and its names and ids per-camera lists.
    """
    # Inline runtime imports; camera.py and cameras.py import this module, so a module-top import would cycle.
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

    def _validate_inputs() -> None:
        assert isinstance(cameras, (Camera, Cameras)), (
            "Expected object to serialize to be a Camera or a Cameras. "
            f"{type(cameras)=}"
        )
        assert format in _CAMERA_SERIALIZATION_FORMATS, (
            "Expected Cameras serialization format to be supported. "
            f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
        )

    _validate_inputs()

    def _serialize() -> Dict[str, Any]:
        """Map the cameras to the payload the requested format spells them in.

        Args:
            None; reads the `cameras` and `format` of the enclosing call.

        Returns:
            The payload for the requested format.
        """
        if format == "json":
            return _serialize_cameras_json(cameras=cameras)
        if format == "npz":
            return _serialize_cameras_npz(cameras=cameras)
        assert 0, "Should not reach here. " f"{format=}"

    return _serialize()


def deserialize_cameras(
    payload: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    format: str = "json",
) -> Union["Camera", "Cameras"]:
    """Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.

    Args:
        payload: The payload keyed by the format's key set: for `json` a dict of json-native values, for `npz` a dict of numpy arrays; an unbatched `[4, 4]` pose names one `Camera`, a batched `[N, 4, 4]` one a `Cameras`.
        device: Target device for the deserialized cameras; None deserializes them on cpu.
        format: Serialization format, either `json` or `npz`.

    Returns:
        A single `Camera` when the payload holds one unbatched pose, otherwise a `Cameras` collection.
    """

    def _validate_inputs() -> None:
        assert isinstance(payload, dict), (
            "Expected Cameras payload to be a dictionary. " f"{type(payload)=}"
        )
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected Cameras device to be None, a string, or a torch device. "
            f"{device=}"
        )
        assert format in _CAMERA_SERIALIZATION_FORMATS, (
            "Expected Cameras serialization format to be supported. "
            f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
        )

    _validate_inputs()

    def _normalize_inputs(
        device: Optional[Union[str, torch.device]],
    ) -> torch.device:
        device = torch.device(device) if device is not None else torch.device("cpu")
        return device

    device = _normalize_inputs(device=device)

    def _deserialize() -> Union["Camera", "Cameras"]:
        """Map the payload the requested format spells back to the cameras it carries.

        Args:
            None; reads the `payload`, `format` and normalized `device` of the enclosing call.

        Returns:
            The `Camera` or `Cameras` the payload decodes to.
        """
        if format == "json":
            return _deserialize_cameras_json(payload=payload, device=device)
        if format == "npz":
            return _deserialize_cameras_npz(payload=payload, device=device)
        assert 0, "Should not reach here. " f"{format=}"

    return _deserialize()


def _serialize_cameras_json(cameras: Union["Camera", "Cameras"]) -> Dict[str, Any]:
    """Map a Camera or a Cameras to the json payload: its intrinsics and extrinsics fields as the components hold them, beside its dtype and its name and id.

    Args:
        cameras: A single `Camera`, whose extrinsics are unbatched, or a `Cameras` collection, whose extrinsics are batched.

    Returns:
        The json payload keyed by `_CAMERA_JSON_KEYS`: `model`; `params`, a number per key for an unbatched intrinsics or a list of numbers per key for a batched one, `h` / `w` as ints; `intr_convention`; `extrinsics`, the cam2world matrix as a nested `[4, 4]` or `[N, 4, 4]` list; `extr_convention`; `dtype`, the batch's torch dtype name (e.g. `float64`); `name` and `id`, single values for a `Camera` and per-camera lists for a `Cameras`.
    """
    model, params, intr_convention = _serialize_camera_intrinsics(
        intrinsics=cameras.intrinsics
    )
    matrix, extr_convention = _serialize_camera_extrinsics(
        extrinsics=cameras.extrinsics
    )
    # One pose is one Camera, a batch of poses a Cameras.
    name, id = (
        (cameras.names, cameras.ids)
        if cameras.extrinsics.is_batched
        else (cameras.name, cameras.id)
    )
    # The resolution rides inside params.
    payload = {
        "model": model,
        "params": params,
        "intr_convention": intr_convention,
        "extrinsics": matrix.tolist(),
        "extr_convention": extr_convention,
        "dtype": str(cameras.dtype).removeprefix("torch."),
        "name": name,
        "id": id,
    }
    return payload


def _deserialize_cameras_json(
    payload: Dict[str, Any],
    device: torch.device,
) -> Union["Camera", "Cameras"]:
    """Map the json payload back to the Camera or Cameras it carries.

    Args:
        payload: The json payload keyed by `_CAMERA_JSON_KEYS`, as `_serialize_cameras_json` spells it.
        device: Target device for the rebuilt cameras.

    Returns:
        A single `Camera` when the payload's extrinsics are one `[4, 4]` matrix, otherwise a `Cameras` collection.
    """

    def _validate_inputs() -> None:
        assert set(payload.keys()) == _CAMERA_JSON_KEYS, (
            "Expected the json camera payload to contain exactly the Camera JSON "
            f"fields. {set(payload.keys())=} {_CAMERA_JSON_KEYS=}"
        )
        assert (
            isinstance(payload["dtype"], str)
            and hasattr(torch, payload["dtype"])
            and isinstance(getattr(torch, payload["dtype"]), torch.dtype)
        ), (
            "Expected the json camera dtype to be a string spelling a torch dtype. "
            f"{payload['dtype']=}"
        )

    _validate_inputs()

    dtype = getattr(torch, payload["dtype"])
    camera_intrinsics = _deserialize_camera_intrinsics(
        model=payload["model"],
        params=payload["params"],
        intr_convention=payload["intr_convention"],
        device=device,
        dtype=dtype,
    )
    camera_extrinsics = _deserialize_camera_extrinsics(
        extrinsics=payload["extrinsics"],
        extr_convention=payload["extr_convention"],
        device=device,
        dtype=dtype,
    )
    return _build_cameras(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        name=payload["name"],
        id=payload["id"],
        device=device,
    )


def _serialize_cameras_npz(
    cameras: Union["Camera", "Cameras"],
) -> Dict[str, np.ndarray]:
    """Map a Camera or a Cameras to the npz payload: the json payload's fields, each held as an array.

    Args:
        cameras: A single `Camera`, whose extrinsics are unbatched, or a `Cameras` collection, whose extrinsics are batched.

    Returns:
        The npz payload keyed by `_CAMERA_NPZ_KEYS`, every value a numpy array: `extrinsics` the cam2world `[4, 4]` or `[N, 4, 4]` matrix in the batch's own dtype; every other key a 0-d string array, `model` / `intr_convention` / `extr_convention` / `dtype` (the torch dtype name, e.g. `float64`) spelled as is and `params` / `name` / `id` json-encoded.
    """
    model, params, intr_convention = _serialize_camera_intrinsics(
        intrinsics=cameras.intrinsics
    )
    matrix, extr_convention = _serialize_camera_extrinsics(
        extrinsics=cameras.extrinsics
    )
    # One pose is one Camera, a batch of poses a Cameras.
    name, id = (
        (cameras.names, cameras.ids)
        if cameras.extrinsics.is_batched
        else (cameras.name, cameras.id)
    )
    # A dict, a list or a null has no typed-array form, so those three ride json-encoded.
    payload: Dict[str, Union[str, np.ndarray]] = {
        "model": model,
        "params": json.dumps(params),
        "intr_convention": intr_convention,
        "extrinsics": matrix,
        "extr_convention": extr_convention,
        "dtype": str(cameras.dtype).removeprefix("torch."),
        "name": json.dumps(name),
        "id": json.dumps(id),
    }
    for key, value in payload.items():
        # 0-d for every key but extrinsics, whose [4, 4] or [N, 4, 4] keeps the batch's own dtype.
        payload[key] = np.asarray(value)
    return payload


def _deserialize_cameras_npz(
    payload: Dict[str, Any], device: torch.device
) -> Union["Camera", "Cameras"]:
    """Map the npz payload back to the Camera or Cameras it carries.

    Args:
        payload: The npz payload keyed by `_CAMERA_NPZ_KEYS`, as `_serialize_cameras_npz` spells it: `extrinsics` a `[4, 4]` or `[N, 4, 4]` array, every other key a 0-d string array.
        device: Target device for the rebuilt cameras.

    Returns:
        A single `Camera` when the payload's extrinsics are one `[4, 4]` matrix, otherwise a `Cameras` collection.
    """

    def _validate_inputs() -> None:
        assert set(payload.keys()) == _CAMERA_NPZ_KEYS, (
            "Expected the npz camera payload to contain exactly the Camera NPZ "
            f"fields. {set(payload.keys())=} {_CAMERA_NPZ_KEYS=}"
        )
        for key in _CAMERA_NPZ_KEYS:
            assert isinstance(payload[key], np.ndarray), (
                f"Expected the npz camera {key} to be a numpy array. "
                f"{type(payload[key])=}"
            )
        # Every field but the pose is one string.
        for key in (
            "model",
            "params",
            "intr_convention",
            "extr_convention",
            "dtype",
            "name",
            "id",
        ):
            assert payload[key].ndim == 0, (
                f"Expected the npz camera {key} to be a 0-d array. "
                f"{payload[key].shape=}"
            )
        assert (
            isinstance(payload["dtype"].item(), str)
            and hasattr(torch, payload["dtype"].item())
            and isinstance(getattr(torch, payload["dtype"].item()), torch.dtype)
        ), (
            "Expected the npz camera dtype to be a string spelling a torch dtype. "
            f"{payload['dtype']=}"
        )

    _validate_inputs()

    model, params, intr_convention, extr_convention = (
        payload["model"].item(),
        json.loads(payload["params"].item()),
        payload["intr_convention"].item(),
        payload["extr_convention"].item(),
    )
    dtype = getattr(torch, payload["dtype"].item())
    # A list for a batch; a str, an int or null for one camera.
    name, id = json.loads(payload["name"].item()), json.loads(payload["id"].item())
    camera_intrinsics = _deserialize_camera_intrinsics(
        model=model,
        params=params,
        intr_convention=intr_convention,
        device=device,
        dtype=dtype,
    )
    # Rebuilt in the dtype the archive records.
    camera_extrinsics = _deserialize_camera_extrinsics(
        extrinsics=payload["extrinsics"],
        extr_convention=extr_convention,
        device=device,
        dtype=dtype,
    )
    return _build_cameras(
        intrinsics=camera_intrinsics,
        extrinsics=camera_extrinsics,
        name=name,
        id=id,
        device=device,
    )


def _serialize_camera_intrinsics(
    intrinsics: CameraIntrinsics,
) -> Tuple[str, Dict[str, Union[int, float, List[int], List[float]]], str]:
    """Map a CameraIntrinsics, batched or unbatched, to its model, its params as the numbers the camera I/O boundary spells them in, and its image-plane frame.

    Args:
        intrinsics: The CameraIntrinsics to serialize, its params 0-d floating tensors when unbatched or `[B]` floating tensors when batched.

    Returns:
        The model string; the params, each a Python number for an unbatched intrinsics or a list of `B` numbers for a batched one, ints for the resolution keys `h` / `w` and floats for every other key; and the `intr_convention` string.
    """
    serialized_params: Dict[str, Union[int, float, List[int], List[float]]] = {}
    for key, value in intrinsics.params.items():
        # An unbatched intrinsics' params are scalars, a batched one's [B] columns.
        serialized_value = value.detach().cpu().tolist()
        # The resolution keys serialize as ints.
        if key in {"h", "w"}:
            assert np.array_equal(serialized_value, np.trunc(serialized_value)), (
                "Expected serialized resolution params to be integer-valued. "
                f"{key=} {serialized_value=}"
            )
            serialized_value = np.asarray(serialized_value, dtype=np.int64).tolist()
        serialized_params[key] = serialized_value
    return intrinsics.model, serialized_params, intrinsics.intr_convention


def _serialize_camera_extrinsics(
    extrinsics: CameraExtrinsics,
) -> Tuple[np.ndarray, str]:
    """Map a CameraExtrinsics, batched or unbatched, to its cam2world matrix and the pose frame it is expressed in.

    Args:
        extrinsics: The CameraExtrinsics to serialize.

    Returns:
        The cam2world matrix as a numpy array in the extrinsics' own dtype, `[4, 4]` for an unbatched extrinsics or `[B, 4, 4]` for a batched one, and the `extr_convention` string.
    """
    # [4, 4] for an unbatched extrinsics, [B, 4, 4] for a batched one.
    matrix = extrinsics.extrinsics.detach().cpu().numpy()
    return matrix, extrinsics.extr_convention


def _deserialize_camera_intrinsics(
    model: str,
    params: Dict[str, Union[int, float, List[int], List[float]]],
    intr_convention: str,
    device: torch.device,
    dtype: torch.dtype,
) -> CameraIntrinsics:
    """Map a model, its params as numbers or [N] columns, and an image-plane frame back to the unbatched or batched CameraIntrinsics they spell.

    Args:
        model: Camera-model identifier string.
        params: The serialized params, each a Python number (unbatched) or a list of `N` numbers (batched).
        intr_convention: Image-plane frame the params are stated in.
        device: Target device for the param tensors.
        dtype: Target floating dtype for the param tensors.

    Returns:
        The CameraIntrinsics the params spell, its params 0-d tensors when they were numbers and `[N]` tensors when they were lists, on `device` in `dtype`.
    """
    tensor_params: Dict[str, torch.Tensor] = {}
    for key, value in params.items():
        # A number to a 0-d tensor and a column to an [N] one, so the intrinsics comes back batched or unbatched as it was saved.
        tensor_params[key] = torch.as_tensor(value, device=device, dtype=dtype)
    # Validates the model, its params and the image-plane frame those params name.
    return build_camera_intrinsics(
        model=model,
        params=tensor_params,
        intr_convention=intr_convention,
        device=device,
    )


def _deserialize_camera_extrinsics(
    extrinsics: Union[np.ndarray, List[List[float]], List[List[List[float]]]],
    extr_convention: str,
    device: torch.device,
    dtype: torch.dtype,
) -> CameraExtrinsics:
    """Map a [4, 4] or [N, 4, 4] cam2world matrix and its pose frame back to the unbatched or batched CameraExtrinsics they spell.

    Args:
        extrinsics: The cam2world matrix, `[4, 4]` or `[N, 4, 4]`, as a numpy array (npz) or a nested list (json).
        extr_convention: Pose-frame convention string.
        device: Target device for the extrinsics tensor.
        dtype: Target floating dtype for the extrinsics tensor.

    Returns:
        The CameraExtrinsics, unbatched for a `[4, 4]` matrix and batched for an `[N, 4, 4]` one, on `device` in `dtype`.
    """
    # json hands nested lists, npz the array itself.
    extrinsics = np.asarray(extrinsics)
    return CameraExtrinsics(
        extrinsics=extrinsics,
        extr_convention=extr_convention,
        device=device,
        dtype=dtype,
    )


def _build_cameras(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    name: Union[Optional[str], List[Optional[str]]],
    id: Union[Optional[int], List[Optional[int]]],
    device: torch.device,
) -> Union["Camera", "Cameras"]:
    """Rebuild the one Camera an unbatched pose names, or the Cameras a batch of poses names.

    Args:
        intrinsics: The rebuilt CameraIntrinsics, unbatched or batched.
        extrinsics: The rebuilt CameraExtrinsics; unbatched for one camera, batched for a collection.
        name: The camera's optional name for one camera, the per-camera list of optional names for a collection.
        id: The camera's optional id for one camera, the per-camera list of optional ids for a collection.
        device: Target device for the rebuilt cameras.

    Returns:
        A `Camera` when `extrinsics` is unbatched, a `Cameras` when it is batched.
    """
    # Inline runtime imports; camera.py and cameras.py import this module, so a module-top import would cycle.
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

    # One pose is one camera.
    if not extrinsics.is_batched:
        camera = Camera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            name=name,
            id=id,
            device=device,
        )
        return camera

    # A batch of poses is a batch of cameras.
    if extrinsics.is_batched:
        # Field-validates the batch.
        cameras = Cameras(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            names=name,
            ids=id,
            device=device,
        )
        return cameras

    assert 0, "Should not reach here."


def _resolve_format_from_path(cameras_path: Path) -> str:
    """Resolve a Cameras serialization format from a file path.

    Args:
        cameras_path: Cameras file path.

    Returns:
        Normalized serialization format name.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras_path, Path), (
            "Expected Cameras file path to be a pathlib Path. " f"{type(cameras_path)=}"
        )
        assert cameras_path.suffix != "", (
            "Expected Cameras file path to include a suffix. " f"{cameras_path=}"
        )

    _validate_inputs()

    return _normalize_format(format=cameras_path.suffix)


def _normalize_format(format: str) -> str:
    """Normalize a path suffix or format name to a supported serialization format.

    Args:
        format: Serialization format name or file suffix.

    Returns:
        Normalized serialization format name.
    """
    format = format.strip()
    assert format != "", (
        "Expected the stripped Cameras serialization format to be non-empty. "
        f"{format=}"
    )
    if format.startswith("."):
        format = format[1:]

    assert format in _CAMERA_SERIALIZATION_FORMATS, (
        "Expected Cameras serialization format to be supported. "
        f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
    )
    return format
