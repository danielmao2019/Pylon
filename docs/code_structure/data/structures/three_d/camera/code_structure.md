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
│   ├── # Serialize cameras to the canonical payload: a list with one dict per camera.
│   ├── from data.structures.three_d.camera.camera import Camera        # inline runtime import; camera.py imports io.py, so this would cycle at module top
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── calls _normalize_format
│   ├── impls was_single = isinstance(cameras, Camera); if was_single -> cameras = a one-element Cameras wrapping it
│   ├── for each camera in cameras
│   │   └── calls _serialize_one_camera           # the per-camera dict the list is built from
│   ├── impls json -> the list of per-camera dicts; npz -> the per-camera fields stacked into batched arrays
│   ├── if was_single
│   │   └── impls unwrap the one-element result (json: the bare dict; npz: the scalar-field form)
│   └── return
├── def deserialize_cameras(payload: Union[Dict[str, Any], List[Dict[str, Any]]], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> Union["Camera", "Cameras"]
│   ├── # Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── calls _normalize_format
│   ├── impls was_single = payload carries one camera (json: a bare dict; npz: scalar fields); if was_single -> normalize to a one-element collection
│   ├── for each per-camera payload
│   │   └── calls _deserialize_one_camera          # validated per-camera fields: intrinsics, extrinsics, convention, name, id
│   ├── calls Cameras                              # assemble the per-camera fields into one Cameras
│   ├── if was_single
│   │   └── impls return the single element cameras[0]
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
├── def _serialize_one_camera(camera: "Camera", format: str) -> Dict[str, Any]
│   ├── # Build one camera's payload dict (the unit the canonical list is made of).
│   ├── impls builds the json dict from intrinsics, extrinsics, convention, name, and id
│   ├── if format == "npz"
│   │   └── impls converts to npz fields with has_name / has_id flags and a -1 id sentinel
│   └── return
├── def _deserialize_one_camera(payload: Dict[str, Any], device: torch.device, format: str) -> Dict[str, Any]
│   ├── # Validate and decode one camera's payload into generic per-camera fields (intrinsics, extrinsics, convention, name, id).
│   ├── if format == "npz"
│   │   └── calls _deserialize_npz_camera_payload
│   ├── calls _validate_json_camera_payload
│   └── return
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   └── calls _normalize_format
├── def _normalize_format(format: str) -> str
│   └── # Normalize a path suffix or format name to a supported serialization format.
├── def _deserialize_npz_camera_payload(payload: Dict[str, Any]) -> Dict[str, Any]
│   └── # Decode one camera's npz-field payload into its generic per-camera dict.
└── def _validate_json_camera_payload(payload: Dict[str, Any]) -> None
    └── # Validate one camera's generic per-camera dict schema.
```
