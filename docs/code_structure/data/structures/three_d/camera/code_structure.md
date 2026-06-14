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
│   │   └── calls _serialize_one_camera           # the format-agnostic generic dict for one camera
│   ├── if format == "json"
│   │   └── impls payload = the list of per-camera dicts; if was_single -> unwrap to the bare dict
│   ├── if format == "npz"
│   │   └── calls _serialize_npz_cameras_payload  # encode the per-camera dicts as one batched-array npz payload
│   └── return
├── def deserialize_cameras(payload: Union[Dict[str, Any], List[Dict[str, Any]]], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> Union["Camera", "Cameras"]
│   ├── # Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras      # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── calls _normalize_format
│   ├── if format == "json"
│   │   └── impls was_single = payload is a bare dict; per_camera_dicts = [payload] if was_single else payload
│   ├── if format == "npz"
│   │   └── calls _deserialize_npz_cameras_payload   # decode the batched-array npz payload into the list of per-camera dicts
│   ├── for each per-camera dict
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
├── def _serialize_one_camera(camera: "Camera") -> Dict[str, Any]
│   └── # Convert one Camera into its format-agnostic generic dict (intrinsics, extrinsics, convention, name, id).
├── def _deserialize_one_camera(payload: Dict[str, Any], device: torch.device) -> Dict[str, Any]
│   ├── # Convert one generic dict into its per-camera tensor fields (intrinsics, extrinsics, convention, name, id).
│   └── calls _validate_json_camera_payload         # read side validates untrusted input; serialize trusts the already-valid Camera, so it has no matching call
├── def _serialize_npz_cameras_payload(per_camera_dicts: List[Dict[str, Any]]) -> Dict[str, Any]
│   └── # Encode the per-camera generic dicts as one batched-array npz payload (stacked intrinsics / extrinsics, per-camera convention / name / id arrays with has_name / has_id flags and a -1 id sentinel).
├── def _deserialize_npz_cameras_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]
│   └── # Decode one batched-array npz payload back into the list of per-camera generic dicts.
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   └── calls _normalize_format
├── def _normalize_format(format: str) -> str
│   └── # Normalize a path suffix or format name to a supported serialization format.
└── def _validate_json_camera_payload(payload: Dict[str, Any]) -> None
    └── # Validate one generic per-camera dict schema.
```
