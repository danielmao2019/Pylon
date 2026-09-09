"""Generic Camera / Cameras serialization and I/O helpers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
from data.structures.three_d.camera.extrinsics.validation import (
    validate_camera_extrinsics,
)
from data.structures.three_d.camera.intrinsics.camera_intrinsics import (
    build_camera_intrinsics,
)

if TYPE_CHECKING:
    from data.structures.three_d.camera.camera import Camera
    from data.structures.three_d.camera.cameras import Cameras

_CAMERA_SERIALIZATION_FORMATS = {
    "json",
    "npz",
}
_CAMERA_JSON_KEYS = {
    "model",
    "params",
    "intr_convention",
    "extrinsics",
    "extr_convention",
    "name",
    "id",
}
_CAMERA_NPZ_KEYS = {
    "model",
    "params",
    "intr_convention",
    "extrinsics",
    "extr_convention",
    "name",
    "has_name",
    "id",
    "has_id",
}


def save_cameras(cameras: Union["Camera", "Cameras"], cameras_path: Path) -> None:
    """Save cameras (a Cameras collection or a single Camera) to a file.

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

    def _read_payload() -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Read the file in the form its own format spells it.

        Args:
            None; reads the resolved `format` and the `cameras_path` of the call.

        Returns:
            The payload the file holds, plural or single.
        """
        if format == "json":
            return json.loads(cameras_path.read_text(encoding="utf-8"))
        if format == "npz":
            with np.load(cameras_path, allow_pickle=False) as payload_file:
                return {key: payload_file[key] for key in payload_file.files}
        assert False, "Expected Cameras load format to be handled. " f"{format=}"

    payload = _read_payload()

    return deserialize_cameras(payload=payload, device=device, format=format)


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
        the same batched-array payload with the leading batch axis dropped from
        every array for a single `Camera`.
    """
    # Inline runtime imports; camera.py and cameras.py import this module, so a
    # module-top import would cycle.
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

    def _normalize_inputs(
        cameras: Union["Camera", "Cameras"],
    ) -> Tuple["Cameras", bool]:
        was_single = isinstance(cameras, Camera)
        if was_single:
            # A leading axis of one, not a one-element list.
            cameras = Cameras(
                intrinsics=cameras.intrinsics[None],
                extrinsics=cameras.extrinsics[None],
                names=[cameras.name],
                ids=[cameras.id],
                device=cameras.device,
            )
        return cameras, was_single

    cameras, was_single = _normalize_inputs(cameras=cameras)

    def _serialize() -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Map the plural Cameras to the plural payload the format spells it in.

        Args:
            None; reads the normalized `cameras` and `format` of the enclosing call.

        Returns:
            The plural payload for the requested format.
        """
        if format == "json":
            return _serialize_cameras_json(cameras=cameras)
        if format == "npz":
            return _serialize_cameras_npz(cameras=cameras)
        assert False, (
            "Expected Cameras serialization format to be handled. " f"{format=}"
        )

    payload = _serialize()

    def _normalize_outputs() -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Hand back the single form one Camera asked for, else the plural payload.

        Args:
            None; reads the `payload`, `was_single` and `format` of the enclosing call.

        Returns:
            The payload in the form the caller's own input carried.
        """
        if was_single:
            return _normalize_payload_to_single(payload=payload, format=format)
        return payload

    return _normalize_outputs()


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
            (single). For `npz`, the batched-array payload, whose arrays carry no
            leading batch axis when it holds a single camera.
        device: Target device for the deserialized cameras.
        format: Serialization format, either `json` or `npz`.

    Returns:
        A single `Camera` when the payload was in single form, otherwise a
        `Cameras` collection.
    """

    def _validate_inputs() -> None:
        assert isinstance(payload, (dict, list)), (
            "Expected Cameras payload to be a dictionary or a list. "
            f"{type(payload)=}"
        )
        assert device is None or isinstance(device, (str, torch.device)), (
            "Expected Cameras device to be None, a string, or a torch device. "
            f"{device=}"
        )
        assert format in _CAMERA_SERIALIZATION_FORMATS, (
            "Expected Cameras serialization format to be supported. "
            f"{format=} {_CAMERA_SERIALIZATION_FORMATS=}"
        )
        if format == "npz":
            assert isinstance(payload, dict), (
                "Expected Cameras NPZ payload to be a dictionary. " f"{type(payload)=}"
            )

    _validate_inputs()

    def _normalize_inputs(
        payload: Union[Dict[str, Any], List[Dict[str, Any]]],
        device: Optional[Union[str, torch.device]],
    ) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], torch.device, bool]:
        payload, was_single = _normalize_payload_to_plural(
            payload=payload, format=format
        )
        target_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        return payload, target_device, was_single

    payload, target_device, was_single = _normalize_inputs(
        payload=payload, device=device
    )

    def _deserialize() -> "Cameras":
        """Map the plural payload the format spells back to the plural Cameras.

        Args:
            None; reads the normalized `payload`, `format` and `target_device`.

        Returns:
            The `Cameras` the plural payload decodes to.
        """
        if format == "json":
            return _deserialize_cameras_json(
                per_camera_dicts=payload, device=target_device
            )
        if format == "npz":
            return _deserialize_cameras_npz(payload=payload, device=target_device)
        assert False, (
            "Expected Cameras deserialization format to be handled. " f"{format=}"
        )

    cameras = _deserialize()

    def _normalize_outputs() -> Union["Camera", "Cameras"]:
        """Hand back the one Camera the payload carried, else the Cameras whole.

        Args:
            None; reads the decoded `cameras` and `was_single` of the enclosing call.

        Returns:
            A single `Camera` when the payload was in single form, else the `Cameras`.
        """
        if was_single:
            return cameras[0]
        return cameras

    return _normalize_outputs()


def _serialize_cameras_json(cameras: "Cameras") -> List[Dict[str, Any]]:
    """Map a Cameras to the plural json payload: one dict per camera.

    Args:
        cameras: A `Cameras` collection to serialize.

    Returns:
        A list with one per-camera json dict (keyed by `_CAMERA_JSON_KEYS`, with
        params as a nested dict and extrinsics as a nested list) for each camera.
    """
    per_camera_dicts: List[Dict[str, Any]] = []
    for camera in cameras:
        per_camera_dicts.append(
            {
                "model": camera.intrinsics.model,
                "params": _serialize_intrinsics_params(params=camera.intrinsics.params),
                "intr_convention": camera.intrinsics.intr_convention,
                "extrinsics": camera.extrinsics.extrinsics.detach().cpu().tolist(),
                "extr_convention": camera.extrinsics.extr_convention,
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
            `_CAMERA_JSON_KEYS`, with params as a nested dict and extrinsics as a
            nested list).
        device: Target device for the decoded extrinsics tensors.

    Returns:
        A `Cameras` collection built from the per-camera dicts.
    """
    # Inline runtime import; cameras.py imports this module, so a module-top
    # import would cycle.
    from data.structures.three_d.camera.cameras import Cameras

    def _validate_inputs() -> None:
        assert isinstance(per_camera_dicts, list), (
            "Expected json per-camera payload to be a list. "
            f"{type(per_camera_dicts)=}"
        )
        assert len(per_camera_dicts) > 0, (
            "Expected json per-camera payload to be non-empty. "
            f"{len(per_camera_dicts)=}"
        )
        for per_camera_dict in per_camera_dicts:
            assert isinstance(per_camera_dict, dict), (
                "Expected each json camera payload to be a dictionary. "
                f"{type(per_camera_dict)=}"
            )
            assert set(per_camera_dict.keys()) == _CAMERA_JSON_KEYS, (
                "Expected each json camera payload to contain exactly the Camera "
                f"JSON fields. {set(per_camera_dict.keys())=} {_CAMERA_JSON_KEYS=}"
            )
            assert isinstance(per_camera_dict["model"], str), (
                "Expected json camera model to be a string. "
                f"{type(per_camera_dict['model'])=}"
            )
            assert isinstance(per_camera_dict["params"], dict), (
                "Expected json camera params to be a dictionary. "
                f"{type(per_camera_dict['params'])=}"
            )
            assert isinstance(per_camera_dict["intr_convention"], str), (
                "Expected json camera intr_convention to be a string. "
                f"{type(per_camera_dict['intr_convention'])=}"
            )
            assert isinstance(per_camera_dict["extr_convention"], str), (
                "Expected json camera extr_convention to be a string. "
                f"{type(per_camera_dict['extr_convention'])=}"
            )
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

    _validate_inputs()

    # The batch shares one projection expression, so it shares one model and one
    # frame per half.
    model = per_camera_dicts[0]["model"]
    intr_convention = per_camera_dicts[0]["intr_convention"]
    extr_convention = per_camera_dicts[0]["extr_convention"]
    assert all(
        per_camera_dict["model"] == model for per_camera_dict in per_camera_dicts
    ), (
        "Expected every json camera payload to name one shared camera model. "
        f"{model=} {[per_camera_dict['model'] for per_camera_dict in per_camera_dicts]=}"
    )
    assert all(
        per_camera_dict["intr_convention"] == intr_convention
        for per_camera_dict in per_camera_dicts
    ), (
        "Expected every json camera payload to name one shared image-plane frame. "
        f"{intr_convention=} "
        f"{[per_camera_dict['intr_convention'] for per_camera_dict in per_camera_dicts]=}"
    )
    assert all(
        per_camera_dict["extr_convention"] == extr_convention
        for per_camera_dict in per_camera_dicts
    ), (
        "Expected every json camera payload to name one shared pose frame. "
        f"{extr_convention=} "
        f"{[per_camera_dict['extr_convention'] for per_camera_dict in per_camera_dicts]=}"
    )

    # json stores a row per camera where npz stores a column per field.
    params_columns: Dict[str, List[Union[int, float]]] = {
        key: [per_camera_dict["params"][key] for per_camera_dict in per_camera_dicts]
        for key in per_camera_dicts[0]["params"]
    }
    names: List[Optional[str]] = [
        per_camera_dict["name"] for per_camera_dict in per_camera_dicts
    ]
    ids: List[Optional[int]] = [
        per_camera_dict["id"] for per_camera_dict in per_camera_dicts
    ]

    tensor_params = _deserialize_intrinsics_params(params=params_columns, device=device)
    intrinsics = build_camera_intrinsics(
        model=model,
        params=tensor_params,
        intr_convention=intr_convention,
        device=device,
    )
    extrinsics_batched = CameraExtrinsics(
        extrinsics=torch.as_tensor(
            [per_camera_dict["extrinsics"] for per_camera_dict in per_camera_dicts],
            dtype=torch.float32,
            device=device,
        ),
        extr_convention=extr_convention,
        device=device,
    )

    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics_batched,
        names=names,
        ids=ids,
        device=device,
    )


def _serialize_cameras_npz(cameras: "Cameras") -> Dict[str, Any]:
    """Map a Cameras to the plural batched-array npz payload.

    Args:
        cameras: A `Cameras` collection to serialize.

    Returns:
        The batched-array npz payload keyed by `_CAMERA_NPZ_KEYS`: per-camera
        `model` strings, json-encoded `params` strings, stacked extrinsics
        `[N, 4, 4]`, per-camera `intr_convention` / `extr_convention` / `name` / `id` arrays of length N with
        `has_name` / `has_id` flag arrays and a `-1` id sentinel for absent ids.
    """
    serialized_params = _serialize_intrinsics_params(params=cameras.intrinsics.params)
    batch_size = len(cameras)

    # The format keeps a column per camera where the batch keeps one value.
    models = np.array([cameras.intrinsics.model] * batch_size)
    intr_conventions = np.array([cameras.intrinsics.intr_convention] * batch_size)
    extr_conventions = np.array([cameras.extrinsics.extr_convention] * batch_size)

    extrinsics = cameras.extrinsics.extrinsics.detach().cpu()
    assert extrinsics.is_floating_point(), (
        "Expected the batch's cam2world matrices to be floating point before the npz "
        f"float32 cast. {extrinsics.dtype=}"
    )
    extrinsics = extrinsics.to(torch.float32).numpy()

    names = np.array(["" if name is None else name for name in cameras.names])
    has_names = np.array([name is not None for name in cameras.names])
    ids = np.array(
        [-1 if id is None else id for id in cameras.ids],
        dtype=np.int64,
    )
    has_ids = np.array([id is not None for id in cameras.ids])

    return {
        "model": models,
        # npz stores arrays, not a dict, so the columns are re-laid as one json-encoded payload per camera.
        "params": np.array(
            [
                json.dumps(
                    {key: column[index] for key, column in serialized_params.items()}
                )
                for index in range(batch_size)
            ]
        ),
        "intr_convention": intr_conventions,
        "extrinsics": extrinsics,
        "extr_convention": extr_conventions,
        "name": names,
        "has_name": has_names,
        "id": ids,
        "has_id": has_ids,
    }


def _deserialize_cameras_npz(
    payload: Dict[str, Any], device: torch.device
) -> "Cameras":
    """Map the plural batched-array npz payload to a Cameras.

    Args:
        payload: The batched-array npz payload keyed by `_CAMERA_NPZ_KEYS`:
            per-camera `model` strings, json-encoded `params` strings, stacked
            extrinsics `[N, 4, 4]`, per-camera `intr_convention` / `extr_convention` / `name` / `id` arrays of
            length N with `has_name` / `has_id` flag arrays and a `-1` id sentinel.
        device: Target device for the decoded extrinsics tensors.

    Returns:
        A `Cameras` collection built from the batched-array payload.
    """
    # Inline runtime import; cameras.py imports this module, so a module-top
    # import would cycle.
    from data.structures.three_d.camera.cameras import Cameras

    def _validate_inputs() -> None:
        assert isinstance(payload, dict), (
            "Expected Cameras NPZ payload to be a dictionary. " f"{type(payload)=}"
        )
        payload_keys = set(payload.keys())
        assert payload_keys == _CAMERA_NPZ_KEYS, (
            "Expected Cameras NPZ payload to match a supported schema. "
            f"{payload_keys=} {_CAMERA_NPZ_KEYS=}"
        )
        extrinsics = payload["extrinsics"]
        assert isinstance(extrinsics, np.ndarray), (
            "Expected Cameras NPZ extrinsics to be a numpy array. "
            f"{type(extrinsics)=}"
        )
        assert extrinsics.dtype == np.float32, (
            "Expected Cameras NPZ extrinsics to use float32. " f"{extrinsics.dtype=}"
        )
        assert extrinsics.ndim == 3, (
            "Expected Cameras NPZ extrinsics to be batched as [N, 4, 4]. "
            f"{extrinsics.shape=}"
        )
        validate_camera_extrinsics(extrinsics)
        batch_size = extrinsics.shape[0]
        for key in (
            "model",
            "params",
            "intr_convention",
            "extr_convention",
            "name",
            "has_name",
            "id",
            "has_id",
        ):
            array = payload[key]
            assert isinstance(array, np.ndarray), (
                f"Expected Cameras NPZ {key} to be a numpy array. " f"{type(array)=}"
            )
            assert array.shape == (batch_size,), (
                f"Expected Cameras NPZ {key} array length to match the batch size. "
                f"{array.shape=} {batch_size=}"
            )

    _validate_inputs()

    extrinsics = payload["extrinsics"]
    batch_size = extrinsics.shape[0]
    model_array = payload["model"]
    params_array = payload["params"]
    intr_convention_array = payload["intr_convention"]
    extr_convention_array = payload["extr_convention"]
    name_array = payload["name"]
    has_name_array = payload["has_name"]
    id_array = payload["id"]
    has_id_array = payload["has_id"]

    # One model and one frame pair is what lets the batch share a single projection
    # expression.
    model = str(model_array[0].item())
    intr_convention = str(intr_convention_array[0].item())
    extr_convention = str(extr_convention_array[0].item())
    assert bool(np.all(model_array == model_array[0])), (
        "Expected the Cameras NPZ model column to be constant over the batch. "
        f"{model_array=}"
    )
    assert bool(np.all(intr_convention_array == intr_convention_array[0])), (
        "Expected the Cameras NPZ intr_convention column to be constant over the "
        f"batch. {intr_convention_array=}"
    )
    assert bool(np.all(extr_convention_array == extr_convention_array[0])), (
        "Expected the Cameras NPZ extr_convention column to be constant over the "
        f"batch. {extr_convention_array=}"
    )

    names: List[Optional[str]] = [
        str(name_array[index].item()) if bool(has_name_array[index].item()) else None
        for index in range(batch_size)
    ]
    ids: List[Optional[int]] = [
        int(id_array[index].item()) if bool(has_id_array[index].item()) else None
        for index in range(batch_size)
    ]

    # The npz params column spells one camera's params per entry, gathered into the
    # columns the batch is built from.
    params_rows = [json.loads(str(entry.item())) for entry in params_array]
    params_columns: Dict[str, List[Union[int, float]]] = {
        key: [params_row[key] for params_row in params_rows] for key in params_rows[0]
    }

    tensor_params = _deserialize_intrinsics_params(params=params_columns, device=device)
    intrinsics = build_camera_intrinsics(
        model=model,
        params=tensor_params,
        intr_convention=intr_convention,
        device=device,
    )
    extrinsics_batched = CameraExtrinsics(
        extrinsics=torch.as_tensor(extrinsics, dtype=torch.float32, device=device),
        extr_convention=extr_convention,
        device=device,
    )

    return Cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics_batched,
        names=names,
        ids=ids,
        device=device,
    )


def _serialize_intrinsics_params(
    params: Dict[str, torch.Tensor],
) -> Dict[str, Union[int, float, List[int], List[float]]]:
    """Map tensor intrinsics params to numeric values at the camera I/O boundary.

    Args:
        params: Named tensor intrinsics params, every value sharing one leading batch
            shape: `[]` for a single camera, `[N]` for a batch of them.

    Returns:
        A numeric params dict suitable for JSON and NPZ payloads: a scalar per key for
        a scalar param, a length-N list per key for a batched one.
    """
    serialized_params: Dict[str, Union[int, float, List[int], List[float]]] = {}
    for key, value in params.items():
        assert isinstance(value, torch.Tensor), (
            "Expected serialized intrinsics params to be tensor-valued. "
            f"{key=} {type(value)=}"
        )
        assert value.ndim <= 1, (
            "Expected serialized intrinsics params to be scalar or one-axis batched "
            f"tensors. {key=} {value.shape=}"
        )
        column = value.detach().cpu().numpy()
        assert np.issubdtype(column.dtype, np.number), (
            "Expected serialized intrinsics params to be numeric before the payload "
            f"cast. {key=} {column.dtype=}"
        )
        if key in {"h", "w"}:
            assert bool(np.all(column == np.round(column))), (
                "Expected serialized resolution params to be integer-valued. "
                f"{key=} {column=}"
            )
            serialized_params[key] = column.astype(np.int64).tolist()
        else:
            serialized_params[key] = column.astype(np.float64).tolist()
    return serialized_params


def _deserialize_intrinsics_params(
    params: Dict[str, Union[int, float, List[int], List[float]]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Map serialized numeric intrinsics params back to tensors at the I/O boundary.

    Args:
        params: Serialized numeric intrinsics params, every value either a scalar for a
            single camera or a length-N column for a batch of them.
        device: Target device for the param tensors.
        dtype: Target floating dtype for the param tensors.

    Returns:
        A tensor-valued params dict on the requested device and dtype, every value
        shaped `[]` for a scalar param and `[N]` for a column.
    """
    tensor_params: Dict[str, torch.Tensor] = {}
    for key, value in params.items():
        tensor_params[key] = torch.as_tensor(value, device=device, dtype=dtype)
    return tensor_params


def _normalize_payload_to_plural(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    format: str,
) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], bool]:
    """Restore a payload to its format's plural form.

    Args:
        payload: The payload as the caller supplied it, in either form. For `json`,
            a list of per-camera dicts (plural) or a bare dict (single). For `npz`,
            the batched-array payload keyed by `_CAMERA_NPZ_KEYS`, whose arrays carry
            a leading batch axis in plural form and none in single form.
        format: Normalized serialization format, either `json` or `npz`.

    Returns:
        The plural payload, and whether the input carried exactly one camera.
    """
    if format == "json":
        was_single = isinstance(payload, dict)
        if was_single:
            payload = [payload]
        return payload, was_single

    if format == "npz":
        was_single = np.asarray(payload["extrinsics"]).ndim == 2
        if was_single:
            payload = {
                key: np.asarray(value)[None, ...] for key, value in payload.items()
            }
        return payload, was_single

    assert False, "Expected Cameras deserialization format to be handled. " f"{format=}"


def _normalize_payload_to_single(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    format: str,
) -> Dict[str, Any]:
    """Reduce a plural payload to the single form its own format spells.

    Args:
        payload: The plural payload the format helpers produced. For `json`, the list
            of per-camera dicts; for `npz`, the batched-array payload.
        format: Normalized serialization format, either `json` or `npz`.

    Returns:
        For `json`, the sole per-camera dict. For `npz`, the same keys with the
        leading batch axis dropped from every array.
    """
    if format == "json":
        return payload[0]

    if format == "npz":
        return {key: value[0] for key, value in payload.items()}

    assert False, "Expected Cameras serialization format to be handled. " f"{format=}"


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
