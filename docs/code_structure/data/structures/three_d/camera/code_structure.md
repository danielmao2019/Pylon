# Camera Data Structure Code Structure

## 1. Code structure trees

`./data/structures/three_d/camera/camera_vis.py`

```text
camera_vis.py
├── from typing import Any, Dict, List, Optional
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── def cameras_vis(cameras: Cameras, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]
│   ├── # Builds a camera-trajectory visualization payload from a Cameras collection.
│   ├── for each camera
│   │   └── calls camera_vis
│   └── return
└── def camera_vis(camera: Camera, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> Dict[str, Any]
    ├── # Builds one camera visualization primitive from a Camera whose intrinsics may be absent.
    ├── impls computes center, center_color, and axes from camera center, right, forward, and up
    ├── impls computes frustum lines from camera intrinsics and frustum_scale
    └── return
```

`./data/structures/three_d/camera/io.py`

```text
io.py
├── import json
├── from pathlib import Path
├── from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.validation import validate_camera_convention, validate_camera_extrinsics, validate_camera_intrinsics
├── if TYPE_CHECKING                              # annotation-only imports; runtime imports of Camera / Cameras are inline in the functions that need them (camera.py and cameras.py import io.py, so a top-level import would cycle)
│   ├── from data.structures.three_d.camera.camera import Camera
│   └── from data.structures.three_d.camera.cameras import Cameras
├── _CAMERA_SERIALIZATION_FORMATS                # supported formats: {"json", "npz"}
├── _CAMERA_JSON_KEYS, _CAMERA_NPZ_KEYS          # one camera's payload key schema (convention / name / id, plus has_name / has_id for npz); a collection is just many of these
├── def serialize_cameras(cameras: Union["Camera", "Cameras"], format: str = "json") -> Union[Dict[str, Any], List[Dict[str, Any]]]
│   ├── # Serialize cameras to the canonical payload for the requested format.
│   ├── from data.structures.three_d.camera.camera import Camera        # inline runtime import; camera.py imports io.py, so this would cycle at module top
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── calls _normalize_format
│   ├── impls input-normalize: was_single = isinstance(cameras, Camera); if was_single -> cameras = a one-element Cameras wrapping it
│   ├── if format == "json"
│   │   └── calls _serialize_cameras_json           # Cameras -> the list of per-camera dicts
│   ├── if format == "npz"
│   │   └── calls _serialize_cameras_npz            # Cameras -> the batched-array npz payload
│   ├── impls output-normalize: if was_single -> reduce the plural payload to its single form (json: the sole dict; npz: tag the batched payload with an is_single flag)
│   └── return
├── def deserialize_cameras(payload: Union[Dict[str, Any], List[Dict[str, Any]]], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> Union["Camera", "Cameras"]
│   ├── # Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.
│   ├── calls _normalize_format
│   ├── impls input-normalize: was_single = the payload is in single form (json: a bare dict; npz: carries an is_single flag); if was_single -> expand it to the plural form (json: wrap in a list; npz: drop the flag)
│   ├── if format == "json"
│   │   └── calls _deserialize_cameras_json         # the list of per-camera dicts -> Cameras
│   ├── if format == "npz"
│   │   └── calls _deserialize_cameras_npz          # the batched-array npz payload -> Cameras
│   ├── impls output-normalize: if was_single -> return cameras[0]
│   └── return
├── def save_cameras(cameras: Union["Camera", "Cameras"], cameras_path: Path) -> None
│   ├── # Save cameras (a Cameras collection or a single Camera) to a .npz or .json file.
│   ├── calls _resolve_format_from_path
│   ├── calls serialize_cameras
│   ├── impls writes json text or an npz archive per the resolved format
│   └── return
├── def load_cameras(cameras_path: Path, device: Optional[Union[str, torch.device]] = None) -> Union["Camera", "Cameras"]
│   ├── # Load cameras (a Cameras collection or a single Camera) from a .npz or .json file.
│   ├── calls _resolve_format_from_path
│   ├── impls reads json text or an npz archive per the resolved format
│   ├── calls deserialize_cameras
│   └── return
├── def _serialize_cameras_json(cameras: "Cameras") -> List[Dict[str, Any]]
│   ├── # Map a Cameras to the plural json payload: one dict per camera.
│   ├── for each camera in cameras
│   │   └── impls builds that camera's json dict from intrinsics, extrinsics, convention, name, and id
│   └── return
├── def _deserialize_cameras_json(per_camera_dicts: List[Dict[str, Any]], device: torch.device) -> "Cameras"
│   ├── # Map the plural json per-camera dicts to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── for each per-camera dict
│   │   ├── impls asserts the keys match _CAMERA_JSON_KEYS and the convention / name / id field types
│   │   └── impls decodes intrinsics and extrinsics to tensors on device
│   ├── calls Cameras                               # constructs and field-validates the batch
│   └── return
├── def _serialize_cameras_npz(cameras: "Cameras") -> Dict[str, Any]
│   ├── # Map a Cameras to the plural batched-array npz payload.
│   ├── for each camera in cameras
│   │   └── impls appends that camera's intrinsics, extrinsics, convention, name, and id to the batch
│   ├── impls stacks the batch into npz arrays with has_name / has_id flag arrays and a -1 id sentinel
│   └── return
├── def _deserialize_cameras_npz(payload: Dict[str, Any], device: torch.device) -> "Cameras"
│   ├── # Map the plural batched-array npz payload to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── impls asserts the keys match _CAMERA_NPZ_KEYS
│   ├── for each batch index
│   │   └── impls decodes that index's intrinsics, extrinsics, convention, name, and id (resolving has_name / has_id flags and the -1 id sentinel) to tensors on device
│   ├── calls Cameras                               # constructs and field-validates the batch
│   └── return
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   └── calls _normalize_format
└── def _normalize_format(format: str) -> str
    └── # Normalize a path suffix or format name to a supported serialization format.
```
