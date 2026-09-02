# Camera Data Structure Code Structure

## 1. Code structure trees

`data/structures/three_d/camera/validation.py`

```text
validation.py
├── from typing import TYPE_CHECKING, List, Optional, Union
├── import torch
├── if TYPE_CHECKING  # annotation-only imports; the runtime type checks import the two classes inline
│   ├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
│   └── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── def validate_cameras_attributes(intrinsics: List["CameraIntrinsics"], extrinsics: List["CameraExtrinsics"], names: List[Optional[str]], ids: List[Optional[int]], device: Optional[Union[str, torch.device]], dtype: Optional[torch.dtype]) -> None
│   ├── # Single-entry validation for Cameras.__init__: validate the parallel per-camera lists, metadata, and optional tensor placement request.
│   ├── impls asserts len(intrinsics) == len(extrinsics) == len(names) == len(ids)
│   ├── for each index-aligned (intrinsic, extrinsic, name, id)
│   │   └── calls validate_camera_attributes
│   ├── impls asserts device is None or a valid torch device spec
│   ├── impls asserts dtype is None or a floating torch dtype
│   └── return
└── def validate_camera_attributes(intrinsics: "CameraIntrinsics", extrinsics: "CameraExtrinsics", name: Optional[str], id: Optional[int], device: Optional[Union[str, torch.device]], dtype: Optional[torch.dtype]) -> None
    ├── # Single-entry validation for Camera.__init__: validate component objects, metadata, and optional tensor placement request.
    ├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics  # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics  # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── impls asserts isinstance(intrinsics, CameraIntrinsics)
    ├── impls asserts isinstance(extrinsics, CameraExtrinsics)
    ├── impls asserts name is None or a str
    ├── impls asserts id is None or an int
    ├── impls asserts device is None or a valid torch device spec
    ├── impls asserts dtype is None or a floating torch dtype
    └── return
```

`data/structures/three_d/camera/camera.py`

```text
camera.py
├── from pathlib import Path
├── from typing import Any, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_extr_convention
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_intr_convention
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── from data.structures.three_d.camera.validation import validate_camera_attributes
└── class Camera
    ├── # One camera: a CameraIntrinsics paired with a CameraExtrinsics, plus metadata and optional tensor placement.
    ├── def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, name: Optional[str] = None, id: Optional[int] = None, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> None
    │   ├── # Construct a Camera from tensor-backed CameraIntrinsics and CameraExtrinsics components.
    │   ├── calls validate_camera_attributes(intrinsics=intrinsics, extrinsics=extrinsics, name=name, id=id, device=device, dtype=dtype)
    │   ├── if device is not None or dtype is not None
    │   │   ├── calls intrinsics.to(device=device, dtype=dtype)  # -> intrinsics
    │   │   └── calls extrinsics.to(device=device, dtype=dtype)  # -> extrinsics
    │   ├── impls self._intrinsics = intrinsics
    │   ├── impls self._extrinsics = extrinsics
    │   ├── impls self._name = name
    │   ├── impls self._id = id
    │   ├── impls self._device = the common component device
    │   └── impls self._dtype = the common component dtype
    ├── def intrinsics(self) -> CameraIntrinsics  # @property
    │   ├── # The camera's CameraIntrinsics ("what the camera is").
    │   └── return self._intrinsics
    ├── def extrinsics(self) -> CameraExtrinsics  # @property
    │   ├── # The camera's CameraExtrinsics ("where the camera is").
    │   └── return self._extrinsics
    ├── def name(self) -> Optional[str]  # @property
    │   ├── # The optional human-readable label a Cameras collection can index this camera by.
    │   └── return self._name
    ├── def id(self) -> Optional[int]  # @property
    │   ├── # The optional integer identity that survives a serialize / deserialize round trip.
    │   └── return self._id
    ├── def device(self) -> torch.device  # @property
    │   ├── # The device the camera tensors live on.
    │   └── return self._device
    ├── def dtype(self) -> torch.dtype  # @property
    │   ├── # The dtype shared by the camera tensors.
    │   └── return self._dtype
    ├── def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False, intr_convention: Optional[str] = None, extr_convention: Optional[str] = None) -> "Camera"
    │   ├── # Return this Camera with Tensor.to-style placement / copy semantics plus optional image-plane and pose-frame conversions.
    │   ├── def _validate_inputs [local]
    │   │   ├── impls assert device is None, str, or torch.device
    │   │   ├── impls assert dtype is None or a floating torch dtype
    │   │   ├── impls assert non_blocking is a bool
    │   │   ├── impls assert copy is a bool
    │   │   ├── impls assert intr_convention is None or a str
    │   │   ├── if intr_convention is not None
    │   │   │   └── calls validate_intr_convention(intr_convention)
    │   │   ├── impls assert extr_convention is None or a str
    │   │   └── if extr_convention is not None
    │   │       └── calls validate_extr_convention(extr_convention)
    │   ├── calls _validate_inputs
    │   ├── def _normalize_inputs [local]
    │   │   ├── impls device = torch.device(device) if device is not None else self._device
    │   │   ├── impls dtype = dtype if dtype is not None else self._dtype
    │   │   ├── impls intr_convention = intr_convention if intr_convention is not None else self._intrinsics.intr_convention
    │   │   ├── impls extr_convention = extr_convention if extr_convention is not None else self._extrinsics.extr_convention
    │   │   └── return device, dtype, intr_convention, extr_convention
    │   ├── calls _normalize_inputs(device=device, dtype=dtype, intr_convention=intr_convention, extr_convention=extr_convention)
    │   ├── impls device, dtype, intr_convention, extr_convention = the returned values from _normalize_inputs
    │   ├── if device == self._device and dtype == self._dtype and intr_convention == self._intrinsics.intr_convention and extr_convention == self._extrinsics.extr_convention and copy is False
    │   │   └── return self
    │   ├── calls self._intrinsics.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, intr_convention=intr_convention)
    │   ├── calls self._extrinsics.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, extr_convention=extr_convention)
    │   ├── impls camera = Camera(...)
    │   └── return camera
    ├── def transform_intrinsics(self, transform: torch.Tensor, resolution: Tuple[int, int]) -> "Camera"
    │   ├── # Return this Camera with its CameraIntrinsics restated onto another image by a pixel-frame affine and that image's own raster.
    │   ├── calls self._intrinsics.transform_intrinsics(transform=transform, resolution=resolution)
    │   ├── impls camera = Camera(...)
    │   └── return camera
    ├── def scale_intrinsics(self, resolution: Optional[Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]] = None, scale: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], np.ndarray, torch.Tensor]] = None) -> "Camera"
    │   ├── # Return this Camera with its CameraIntrinsics scaled to an integer or array-like resolution, or by a factor.
    │   ├── calls self._intrinsics.scale_intrinsics(resolution=resolution, scale=scale)
    │   ├── impls camera = Camera(...)
    │   └── return camera
    ├── def transform_extrinsics(self, scale: Union[int, float, np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], translation: Union[np.ndarray, torch.Tensor, Tuple[Union[int, float], Union[int, float], Union[int, float]], List[Union[int, float]]]) -> "Camera"
    │   ├── # Return this Camera under array-like scale, rotation, and translation inputs of its CameraExtrinsics pose.
    │   ├── calls self._extrinsics.transform_extrinsics(scale=scale, rotation=rotation, translation=translation)
    │   ├── impls camera = Camera(...)
    │   └── return camera
    ├── def serialize(self, format: str = "json") -> Dict[str, Any]
    │   ├── # Serialize this Camera into a single-form payload.
    │   └── calls serialize_cameras
    ├── def deserialize(cls, payload: Dict[str, Any], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> "Camera"  # @classmethod
    │   ├── # Deserialize one Camera from a single-form payload.
    │   └── calls deserialize_cameras
    ├── def save(self, camera_path: Path) -> None
    │   ├── # Save this Camera to a .npz or .json file.
    │   └── calls save_cameras
    └── def load(cls, camera_path: Path, device: Optional[Union[str, torch.device]] = None) -> "Camera"  # @classmethod
        ├── # Load one Camera from a .npz or .json file.
        └── calls load_cameras
```

`data/structures/three_d/camera/cameras.py`

```text
cameras.py
├── from typing import Iterator, List, Optional, Sequence, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_extr_convention
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_intr_convention
├── from data.structures.three_d.camera.validation import validate_cameras_attributes
└── class Cameras
    ├── # A batch of cameras held as parallel per-camera intrinsics / extrinsics lists, addressable by position or by name.
    ├── def __init__(self, intrinsics: List[CameraIntrinsics], extrinsics: List[CameraExtrinsics], names: Optional[List[Optional[str]]] = None, ids: Optional[List[Optional[int]]] = None, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> None
    │   ├── # Construct a Cameras from parallel tensor-backed CameraIntrinsics and CameraExtrinsics lists.
    │   ├── calls validate_cameras_attributes(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device=device, dtype=dtype)
    │   ├── if device is not None or dtype is not None
    │   │   ├── impls replace intrinsics with the list of CameraIntrinsics.to results
    │   │   └── impls replace extrinsics with the list of CameraExtrinsics.to results
    │   ├── impls self._intrinsics = intrinsics
    │   ├── impls self._extrinsics = extrinsics
    │   ├── impls self._names = names
    │   ├── impls self._ids = ids
    │   ├── impls self._device = the common component device
    │   ├── impls self._dtype = the common component dtype
    │   └── impls self._name_to_index = the name → index map
    ├── def __len__(self) -> int
    │   ├── # The number of cameras in the collection.
    │   ├── impls length = len(self._intrinsics)
    │   └── return length
    ├── def __getitem__(self, index: Union[int, slice, List[int], str]) -> Union["Camera", "Cameras"]
    │   ├── # Index the collection.
    │   ├── if isinstance(index, str)
    │   │   ├── impls camera_index = self._name_to_index[index]
    │   │   ├── impls camera = Camera(...)
    │   │   └── return camera
    │   ├── if isinstance(index, (slice, list))
    │   │   ├── impls cameras = Cameras(...)
    │   │   └── return cameras
    │   ├── if isinstance(index, int)
    │   │   ├── impls camera = Camera(...)
    │   │   └── return camera
    │   └── assert 0, "Should not reach here."
    ├── def __iter__(self) -> Iterator["Camera"]
    │   ├── # Iterate the collection one Camera at a time.
    │   └── for each index in range(len(self))
    │       └── yield  # self[index]
    ├── def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False, intr_convention: Optional[str] = None, extr_convention: Optional[str] = None) -> "Cameras"
    │   ├── # Return this Cameras with Tensor.to-style placement / copy semantics plus optional image-plane and pose-frame conversions.
    │   ├── def _validate_inputs [local]
    │   │   ├── impls assert device is None, str, or torch.device
    │   │   ├── impls assert dtype is None or a floating torch dtype
    │   │   ├── impls assert non_blocking is a bool
    │   │   ├── impls assert copy is a bool
    │   │   ├── impls assert intr_convention is None or a str
    │   │   ├── if intr_convention is not None
    │   │   │   └── calls validate_intr_convention(intr_convention)
    │   │   ├── impls assert extr_convention is None or a str
    │   │   └── if extr_convention is not None
    │   │       └── calls validate_extr_convention(extr_convention)
    │   ├── calls _validate_inputs
    │   ├── def _normalize_inputs [local]
    │   │   ├── impls device = torch.device(device) if device is not None else self._device
    │   │   ├── impls dtype = dtype if dtype is not None else self._dtype
    │   │   └── return device, dtype
    │   ├── calls _normalize_inputs(device=device, dtype=dtype)
    │   ├── impls device, dtype = the returned values from _normalize_inputs
    │   ├── if device == self._device and dtype == self._dtype and (intr_convention is None or all(intrinsic.intr_convention == intr_convention for intrinsic in self._intrinsics)) and (extr_convention is None or all(extrinsic.extr_convention == extr_convention for extrinsic in self._extrinsics)) and copy is False
    │   │   └── return self
    │   ├── for each camera in self
    │   │   └── calls camera.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, intr_convention=intr_convention, extr_convention=extr_convention)
    │   ├── impls cameras = Cameras(...)
    │   └── return cameras
    ├── def transform_extrinsics(self, scale: Union[int, float, np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], translation: Union[np.ndarray, torch.Tensor, Tuple[Union[int, float], Union[int, float], Union[int, float]], List[Union[int, float]]]) -> "Cameras"
    │   ├── # Return this Cameras under array-like scale, rotation, and translation inputs applied to each CameraExtrinsics pose.
    │   ├── for each camera in self
    │   │   └── calls camera.transform_extrinsics(scale=scale, rotation=rotation, translation=translation)
    │   ├── impls cameras = Cameras(...)
    │   └── return cameras
    ├── def intrinsics(self) -> Sequence[CameraIntrinsics]  # @property
    │   ├── # The per-camera intrinsics, positionally parallel to extrinsics so index i names the same camera in both.
    │   └── return self._intrinsics
    ├── def extrinsics(self) -> Sequence[CameraExtrinsics]  # @property
    │   ├── # The per-camera extrinsics, positionally parallel to intrinsics so index i names the same camera in both.
    │   └── return self._extrinsics
    ├── def names(self) -> Sequence[Optional[str]]  # @property
    │   ├── # The per-camera names, the keys __getitem__ resolves a string index through.
    │   └── return self._names
    ├── def ids(self) -> Sequence[Optional[int]]  # @property
    │   ├── # The per-camera ids, carried through serialization alongside a flag marking which cameras have one.
    │   └── return self._ids
    ├── def device(self) -> torch.device  # @property
    │   ├── # The single device the whole collection is held on, fixed at construction for every camera in it.
    │   └── return self._device
    ├── def dtype(self) -> torch.dtype  # @property
    │   ├── # The dtype shared by the cameras' tensor state.
    │   └── return self._dtype
    ├── def center(self) -> torch.Tensor  # @property
    │   ├── # The [N, 3] stack of per-camera centers.
    │   └── impls stacks each CameraExtrinsics center into [N, 3]
    ├── def right(self) -> torch.Tensor  # @property
    │   ├── # The [N, 3] stack of per-camera physical right axes.
    │   └── impls stacks each CameraExtrinsics right axis into [N, 3]
    ├── def forward(self) -> torch.Tensor  # @property
    │   ├── # The [N, 3] stack of per-camera physical forward axes.
    │   └── impls stacks each CameraExtrinsics forward axis into [N, 3]
    └── def up(self) -> torch.Tensor  # @property
        ├── # The [N, 3] stack of per-camera physical up axes.
        └── impls stacks each CameraExtrinsics up axis into [N, 3]
```

`data/structures/three_d/camera/camera_vis.py`

```text
camera_vis.py
├── from typing import Any, Dict, List, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── DEFAULT_FRUSTUM_SIZE = 0.25            # world-unit frustum/axis size, resolved when frustum_size is None
├── DEFAULT_FRUSTUM_COLOR = (255, 214, 0)  # RGB line color, resolved when frustum_color is None
├── DEFAULT_POINT_SIZE = 0.01              # world-unit size of the camera-center point marker, resolved when point_size is None
├── DEFAULT_POINT_COLOR = (255, 214, 0)    # RGB center-point color, resolved when point_color is None
├── def cameras_vis(cameras: Cameras, frustum_size: Optional[float] = None, frustum_color: Optional[Tuple[int, int, int]] = None, point_size: Optional[float] = None, point_color: Optional[Tuple[int, int, int]] = None) -> List[Dict[str, Any]]
│   ├── # The cameras atomic-display data-layer mapping.
│   ├── for each camera
│   │   └── calls camera_vis(camera, frustum_size, frustum_color, point_size, point_color)
│   └── return
└── def camera_vis(camera: Camera, frustum_size: Optional[float] = None, frustum_color: Optional[Tuple[int, int, int]] = None, point_size: Optional[float] = None, point_color: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]
    ├── # The per-camera atomic-display data-layer mapping.
    ├── impls resolves frustum_size / frustum_color / point_size / point_color from None to DEFAULT_FRUSTUM_SIZE / DEFAULT_FRUSTUM_COLOR / DEFAULT_POINT_SIZE / DEFAULT_POINT_COLOR
    ├── impls computes the center marker from the camera's extrinsics center, its color from point_color, and its size from point_size                               # impls-node-one-step:skip
    ├── impls computes axes and frustum lines colored by frustum_color from the camera's extrinsics right / forward / up, the camera's intrinsics, and frustum_size  # impls-node-one-step:skip
    └── return
```

`data/structures/three_d/camera/io.py`

```text
io.py
├── import json
├── from pathlib import Path
├── from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── if TYPE_CHECKING  # annotation-only imports; runtime imports of Camera / Cameras are inline in the functions that need them (camera.py and cameras.py import io.py, so a top-level import would cycle)
│   ├── from data.structures.three_d.camera.camera import Camera
│   └── from data.structures.three_d.camera.cameras import Cameras
├── _CAMERA_SERIALIZATION_FORMATS        # supported formats: {"json", "npz"}
├── _CAMERA_JSON_KEYS, _CAMERA_NPZ_KEYS  # one camera's payload key schema (model / params / intr_convention / extrinsics / extr_convention / name / id, plus has_name / has_id for npz); a collection is just many of these
├── def save_cameras(cameras: Union["Camera", "Cameras"], cameras_path: Path) -> None
│   ├── # Save cameras (a Cameras collection or a single Camera) to a .npz or .json file.
│   ├── def _validate_inputs [local]
│   │   └── impls assert cameras_path is a Path  # the cameras themselves are serialize_cameras' to check
│   ├── calls _validate_inputs
│   ├── calls _resolve_format_from_path(cameras_path=cameras_path)
│   ├── calls serialize_cameras(cameras=cameras, format=format)
│   ├── impls cameras_path.parent.mkdir(parents=True, exist_ok=True)
│   ├── if format == "json"
│   │   ├── impls write the payload as indented json text, utf-8 encoded
│   │   └── return
│   ├── if format == "npz"
│   │   ├── impls np.savez(cameras_path, **payload)
│   │   └── return
│   └── assert 0, "Should not reach here."
├── def load_cameras(cameras_path: Path, device: Optional[Union[str, torch.device]] = None) -> Union["Camera", "Cameras"]
│   ├── # Load cameras (a Cameras collection or a single Camera) from a .npz or .json file.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras_path is a Path
│   │   ├── impls assert cameras_path exists
│   │   ├── impls assert cameras_path is a file
│   │   └── impls assert device is None, a str, or a torch.device
│   ├── calls _validate_inputs
│   ├── calls _resolve_format_from_path(cameras_path=cameras_path)
│   ├── def _read_payload [local]
│   │   ├── # Read the file in the form its own format spells it.
│   │   ├── if format == "json"
│   │   │   ├── impls payload = the file's utf-8 text parsed as json
│   │   │   └── return payload
│   │   ├── if format == "npz"
│   │   │   ├── impls payload = every array in the npz archive, keyed by its name
│   │   │   └── return payload
│   │   └── assert 0, "Should not reach here."
│   ├── calls _read_payload
│   ├── calls deserialize_cameras(payload=payload, device=device, format=format)
│   └── return
├── def serialize_cameras(cameras: Union["Camera", "Cameras"], format: str = "json") -> Union[Dict[str, Any], List[Dict[str, Any]]]
│   ├── # Serialize cameras to the canonical payload for the requested format, in the single or plural form the caller's own input carried.
│   ├── from data.structures.three_d.camera.camera import Camera    # inline runtime import; camera.py imports io.py, so this would cycle at module top
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras is a Camera or a Cameras
│   │   └── impls assert format is in _CAMERA_SERIALIZATION_FORMATS  # drawn because this is format's only owner on this path
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── impls was_single = isinstance(cameras, Camera)
│   │   ├── if was_single
│   │   │   └── calls Cameras(intrinsics=[cameras.intrinsics], extrinsics=[cameras.extrinsics], names=[cameras.name], ids=[cameras.id], device=cameras.device)
│   │   └── return cameras, was_single
│   ├── calls _normalize_inputs(cameras=cameras)
│   ├── def _serialize [local]
│   │   ├── # Map the plural Cameras to the plural payload the requested format spells it in.
│   │   ├── if format == "json"
│   │   │   ├── calls _serialize_cameras_json(cameras=cameras)
│   │   │   └── return
│   │   ├── if format == "npz"
│   │   │   ├── calls _serialize_cameras_npz(cameras=cameras)
│   │   │   └── return
│   │   └── assert 0, "Should not reach here."
│   ├── calls _serialize
│   ├── def _normalize_outputs [local]
│   │   ├── # Hand back the single form the caller's own one Camera asked for, else the plural payload whole.
│   │   ├── if was_single
│   │   │   ├── calls _normalize_payload_to_single(payload=payload, format=format)
│   │   │   └── return
│   │   └── return payload
│   ├── calls _normalize_outputs
│   └── return
├── def deserialize_cameras(payload: Union[Dict[str, Any], List[Dict[str, Any]]], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> Union["Camera", "Cameras"]
│   ├── # Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert payload is a dict or a list
│   │   ├── impls assert device is None, a str, or a torch.device
│   │   ├── impls assert format is in _CAMERA_SERIALIZATION_FORMATS  # drawn because this is format's only owner on this path
│   │   └── if format == "npz"
│   │       └── impls assert payload is a dict  # cross-field: an npz payload is keyed, never a list
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── calls _normalize_payload_to_plural(payload=payload, format=format)
│   │   ├── impls target_device = torch.device(device) if device is not None else torch.device("cpu")
│   │   └── return payload, target_device, was_single
│   ├── calls _normalize_inputs(payload=payload, device=device)
│   ├── def _deserialize [local]
│   │   ├── # Map the plural payload the requested format spells back to the plural Cameras.
│   │   ├── if format == "json"
│   │   │   ├── calls _deserialize_cameras_json(per_camera_dicts=payload, device=target_device)
│   │   │   └── return
│   │   ├── if format == "npz"
│   │   │   ├── calls _deserialize_cameras_npz(payload=payload, device=target_device)
│   │   │   └── return
│   │   └── assert 0, "Should not reach here."
│   ├── calls _deserialize
│   ├── def _normalize_outputs [local]
│   │   ├── # Hand back the one Camera the payload carried, else the Cameras whole.
│   │   ├── if was_single
│   │   │   └── return cameras[0]
│   │   └── return cameras
│   ├── calls _normalize_outputs
│   └── return
├── def _serialize_cameras_json(cameras: "Cameras") -> List[Dict[str, Any]]
│   ├── # Map a Cameras to the plural json payload: one dict per camera.
│   ├── impls per_camera_dicts — an empty accumulator the loop appends to
│   ├── for each camera in cameras
│   │   ├── calls _serialize_intrinsics_params(params=camera.intrinsics.params)
│   │   └── impls builds that camera's json dict from intrinsics.model, serialized_params, intrinsics.intr_convention, extrinsics.extrinsics, extrinsics.extr_convention, name, and id  # impls-node-one-step:skip; each frame is keyed for the half it came off, and the resolution rides inside serialized_params
│   └── return
├── def _deserialize_cameras_json(per_camera_dicts: List[Dict[str, Any]], device: torch.device) -> "Cameras"
│   ├── # Map the plural json per-camera dicts to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── impls assert per_camera_dicts is a non-empty list
│   │   └── for each per-camera dict
│   │       ├── impls assert it is a dict whose keys are exactly _CAMERA_JSON_KEYS
│   │       ├── impls assert its model is a str
│   │       ├── impls assert its params is a dict
│   │       ├── impls assert its intr_convention is a str
│   │       ├── impls assert its extr_convention is a str
│   │       ├── impls assert its name is None or a str
│   │       └── impls assert its id is None or an int
│   ├── calls _validate_inputs
│   ├── impls intrinsics_list, extrinsics_list, names, ids — four empty accumulators the loop appends to
│   ├── for each per-camera dict
│   │   ├── impls decodes serialized params to scalar tensors on device
│   │   ├── impls decodes extrinsics to a tensor on device
│   │   ├── calls _deserialize_intrinsics_params(params=per_camera_dict["params"], device=device)
│   │   ├── calls build_camera_intrinsics(model=per_camera_dict["model"], params=tensor_params, intr_convention=per_camera_dict["intr_convention"], device=device)  # validates the model, its params and the image-plane frame those params name; the resolution rides inside tensor_params
│   │   ├── calls CameraExtrinsics(extrinsics=extrinsics, extr_convention=per_camera_dict["extr_convention"], device=device)                                        # validates extrinsics + extr_convention
│   │   └── impls appends per_camera_dict["name"] and per_camera_dict["id"] unchanged  # impls-node-one-step:skip; json stores both directly, where npz needs has_name / has_id flags
│   ├── calls Cameras(intrinsics=intrinsics_list, extrinsics=extrinsics_list, names=names, ids=ids, device=device)  # field-validates the batch
│   └── return
├── def _serialize_cameras_npz(cameras: "Cameras") -> Dict[str, Any]
│   ├── # Map a Cameras to the plural batched-array npz payload.
│   ├── impls models, params, intr_conventions, extrinsics_list, extr_conventions, names, has_names, ids, has_ids — nine empty accumulators the loop appends to
│   ├── for each camera in cameras
│   │   ├── calls _serialize_intrinsics_params(params=camera.intrinsics.params)
│   │   └── impls appends that camera's model, serialized_params (json-encoded, its h and w among them), intr_convention, extrinsics, extr_convention, name ("" when absent), and id (-1 when absent) to the batch, each with its has_name / has_id flag  # impls-node-one-step:skip
│   ├── impls stacks each accumulator into its npz array, extrinsics along a new leading axis  # impls-node-one-step:skip
│   └── return
├── def _deserialize_cameras_npz(payload: Dict[str, Any], device: torch.device) -> "Cameras"
│   ├── # Map the plural batched-array npz payload to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── impls assert payload is a dict whose keys are exactly _CAMERA_NPZ_KEYS
│   │   ├── impls assert payload["extrinsics"] is a float32 ndarray batched as [N, 4, 4]
│   │   ├── calls validate_camera_extrinsics(extrinsics)  # batched validation of all views' 4x4 cam2world
│   │   └── for each of the eight per-camera keys
│   │       └── impls assert its array is an ndarray of shape (batch_size,)
│   ├── calls _validate_inputs
│   ├── impls extrinsics = payload["extrinsics"], the batched [N, 4, 4] cam2world array
│   ├── impls batch_size = extrinsics.shape[0]
│   ├── impls model_array, params_array, intr_convention_array, extr_convention_array, name_array, has_name_array, id_array, has_id_array — the eight per-camera arrays read from payload
│   ├── impls intrinsics_list, extrinsics_list, names, ids — four empty accumulators the loop appends to
│   ├── for each batch index
│   │   ├── impls decodes that index's model, serialized params, extrinsics, two frames, name, and id on device, each name and id taken only when its has_name / has_id flag is set  # impls-node-one-step:skip
│   │   ├── calls _deserialize_intrinsics_params(params=serialized_params, device=device)
│   │   ├── calls build_camera_intrinsics(model=model, params=tensor_params, intr_convention=str(intr_convention_array[index].item()), device=device)  # validates the model, its params and the image-plane frame those params name; the resolution rides inside tensor_params
│   │   └── calls CameraExtrinsics(extrinsics=torch.as_tensor(extrinsics[index], dtype=torch.float32, device=device), extr_convention=str(extr_convention_array[index].item()), device=device)  # validates extr_convention
│   ├── calls Cameras(intrinsics=intrinsics_list, extrinsics=extrinsics_list, names=names, ids=ids, device=device)  # field-validates the batch
│   └── return
├── def _serialize_intrinsics_params(params: Dict[str, torch.Tensor]) -> Dict[str, Union[int, float]]
│   ├── # Map scalar tensor intrinsics params to numeric scalar values at the camera I/O boundary.
│   ├── impls serialized_params = an empty dict
│   ├── for each param key/value
│   │   └── impls materialize the scalar tensor value as its Python numeric scalar
│   └── return serialized_params
├── def _deserialize_intrinsics_params(params: Dict[str, Union[int, float]], device: torch.device, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]
│   ├── # Map serialized numeric scalar intrinsics params back to scalar tensors at the camera I/O boundary.
│   ├── impls tensor_params = an empty dict
│   ├── for each param key/value
│   │   └── impls convert the numeric scalar to a torch scalar tensor with the requested device and dtype  # impls-node-one-step:skip
│   └── return tensor_params
├── def _normalize_payload_to_plural(payload: Union[Dict[str, Any], List[Dict[str, Any]]], format: str) -> Tuple[Union[Dict[str, Any], List[Dict[str, Any]]], bool]
│   ├── # Restore a payload to its format's plural form, reporting whether it arrived carrying one camera.
│   ├── if format == "json"
│   │   ├── impls was_single = isinstance(payload, dict)
│   │   ├── if was_single
│   │   │   └── impls payload = that bare per-camera dict wrapped in a list
│   │   └── return payload, was_single
│   ├── if format == "npz"
│   │   ├── impls was_single = payload["extrinsics"] carries no leading batch axis
│   │   ├── if was_single
│   │   │   └── impls payload = each array given its leading batch axis back
│   │   └── return payload, was_single
│   └── assert 0, "Should not reach here."
├── def _normalize_payload_to_single(payload: Union[Dict[str, Any], List[Dict[str, Any]]], format: str) -> Dict[str, Any]
│   ├── # Reduce a plural payload to the single form its own format spells.
│   ├── if format == "json"
│   │   └── return payload[0]  # json spells one camera as the bare per-camera dict standing where the list would
│   ├── if format == "npz"
│   │   ├── impls payload = each array in payload indexed at 0  # npz spells it as the same keys, one axis shorter
│   │   └── return payload
│   └── assert 0, "Should not reach here."
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   ├── # Resolve a Cameras serialization format from a file path.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras_path is a Path
│   │   └── impls assert cameras_path has a non-empty suffix
│   ├── calls _validate_inputs
│   ├── calls _normalize_format(format=cameras_path.suffix)
│   └── return
└── def _normalize_format(format: str) -> str
    ├── # Normalize a path suffix or format name to a supported serialization format.
    ├── impls format = format.strip()
    ├── impls asserts the stripped format is non-empty
    ├── if format.startswith(".")
    │   └── impls format = format[1:]
    ├── impls asserts format is in _CAMERA_SERIALIZATION_FORMATS
    └── return format
```
