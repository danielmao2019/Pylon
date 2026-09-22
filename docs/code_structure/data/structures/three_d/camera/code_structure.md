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
├── def validate_cameras_attributes(intrinsics: "CameraIntrinsics", extrinsics: "CameraExtrinsics", names: Optional[List[Optional[str]]], ids: Optional[List[Optional[int]]], device: Optional[Union[str, torch.device]], dtype: Optional[torch.dtype]) -> None
│   ├── # Single-entry validation for Cameras.__init__: validate the batched component pair, the metadata parallel to its batch axis, and the optional tensor placement request.
│   ├── calls validate_camera_attributes(intrinsics=intrinsics, extrinsics=extrinsics, name=None, id=None, device=device, dtype=dtype)  # the component checks are shape-agnostic, so the batched pair takes the same ones a single camera does
│   ├── assert extrinsics.is_batched  # the poses count the cameras
│   ├── assert not intrinsics.is_batched or len(intrinsics) == 1 or len(intrinsics) == len(extrinsics)
│   ├── assert names is None or len(names) == len(extrinsics)
│   ├── assert ids is None or len(ids) == len(extrinsics)
│   └── return
└── def validate_camera_attributes(intrinsics: "CameraIntrinsics", extrinsics: "CameraExtrinsics", name: Optional[str], id: Optional[int], device: Optional[Union[str, torch.device]], dtype: Optional[torch.dtype]) -> None
    ├── # Single-entry validation for Camera.__init__: validate component objects, metadata, and optional tensor placement request.
    ├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics  # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics  # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── assert isinstance(intrinsics, CameraIntrinsics)
    ├── assert isinstance(extrinsics, CameraExtrinsics)
    ├── assert intrinsics.device == extrinsics.device  # one camera's two halves live on one device, whatever device it is then brought to
    ├── assert intrinsics.dtype == extrinsics.dtype  # one camera's two halves hold one dtype, whatever dtype it is then cast to
    ├── assert name is None or isinstance(name, str)
    ├── assert id is None or isinstance(id, int)
    ├── assert device is None or isinstance(device, (str, torch.device))
    ├── assert dtype is None or isinstance(dtype, torch.dtype)
    ├── if dtype is not None
    │   └── assert dtype is a floating dtype
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
    │   ├── def _validate_inputs [local]
    │   │   └── calls validate_camera_attributes(intrinsics=intrinsics, extrinsics=extrinsics, name=name, id=id, device=device, dtype=dtype)
    │   ├── calls _validate_inputs
    │   ├── def _normalize_inputs [local]
    │   │   ├── if device is None
    │   │   │   ├── impls component_devices = {intrinsics.device, extrinsics.device}  # a set of both, so neither component is the one read
    │   │   │   └── impls device = the single device in component_devices  # single, since validate_camera_attributes asserts intrinsics.device == extrinsics.device
    │   │   ├── impls device = device as a torch.device
    │   │   ├── if device.type == "cuda" and device.index is None  # one physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal
    │   │   │   └── impls device = the cuda device at the index of torch's current cuda device  # where a tensor sent to a bare cuda lands, and so the device it reports
    │   │   ├── if dtype is None
    │   │   │   ├── impls component_dtypes = {intrinsics.dtype, extrinsics.dtype}  # a set of both, so neither component is the one read
    │   │   │   └── impls dtype = the single dtype in component_dtypes  # single, since validate_camera_attributes asserts intrinsics.dtype == extrinsics.dtype
    │   │   ├── calls intrinsics.to(device=device, dtype=dtype)  # -> intrinsics, brought to the resolved device and dtype
    │   │   ├── calls extrinsics.to(device=device, dtype=dtype)  # -> extrinsics, brought to the resolved device and dtype, never the other way around
    │   │   └── return intrinsics, extrinsics, device, dtype
    │   ├── calls _normalize_inputs(intrinsics=intrinsics, extrinsics=extrinsics, device=device, dtype=dtype)
    │   ├── impls intrinsics, extrinsics, device, dtype = the returned values from _normalize_inputs
    │   ├── impls self._intrinsics = intrinsics
    │   ├── impls self._extrinsics = extrinsics
    │   ├── impls self._name = name
    │   ├── impls self._id = id
    │   ├── impls self._device = device  # the resolved device the components were brought to, not read back off them
    │   └── impls self._dtype = dtype  # the resolved dtype the components were cast to, not read back off them
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
├── from typing import Iterator, List, Optional, Sequence, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.camera.validation import validate_cameras_attributes
└── class Cameras
    ├── # A batch of cameras: one CameraExtrinsics carrying the leading batch axis and one CameraIntrinsics either carrying it too or shared by every camera, so every method they already have operates on the whole batch.
    ├── def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, names: Optional[List[Optional[str]]] = None, ids: Optional[List[Optional[int]]] = None, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> None
    │   ├── # Construct a Cameras from a batched CameraIntrinsics whose params are [B] and a batched CameraExtrinsics whose matrix is [B, 4, 4].
    │   ├── def _validate_inputs [local]
    │   │   └── calls validate_cameras_attributes(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device=device, dtype=dtype)
    │   ├── calls _validate_inputs
    │   ├── def _normalize_inputs [local]
    │   │   ├── if device is None
    │   │   │   ├── impls component_devices = {intrinsics.device, extrinsics.device}  # a set of both, so neither component is the one read
    │   │   │   └── impls device = the single device in component_devices  # single, since validate_camera_attributes asserts intrinsics.device == extrinsics.device
    │   │   ├── impls device = device as a torch.device
    │   │   ├── if device.type == "cuda" and device.index is None  # one physical device has one spelling here, so a cuda and a cuda:0 naming it never compare unequal
    │   │   │   └── impls device = the cuda device at the index of torch's current cuda device  # where a tensor sent to a bare cuda lands, and so the device it reports
    │   │   ├── if dtype is None
    │   │   │   ├── impls component_dtypes = {intrinsics.dtype, extrinsics.dtype}  # a set of both, so neither component is the one read
    │   │   │   └── impls dtype = the single dtype in component_dtypes  # single, since validate_camera_attributes asserts intrinsics.dtype == extrinsics.dtype
    │   │   ├── calls intrinsics.to(device=device, dtype=dtype)  # -> intrinsics, brought to the resolved device and dtype
    │   │   ├── calls extrinsics.to(device=device, dtype=dtype)  # -> extrinsics, brought to the resolved device and dtype, never the other way around
    │   │   ├── impls batch_size = len(extrinsics)  # the poses count the cameras
    │   │   ├── if names is None  # the batch named by omission
    │   │   │   └── impls names = [None] * batch_size
    │   │   ├── if ids is None  # the batch identified by omission
    │   │   │   └── impls ids = [None] * batch_size
    │   │   └── return intrinsics, extrinsics, names, ids, device, dtype, batch_size
    │   ├── calls _normalize_inputs(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device=device, dtype=dtype)
    │   ├── impls intrinsics, extrinsics, names, ids, device, dtype, batch_size = the returned values from _normalize_inputs
    │   ├── impls name_to_index = an empty dict
    │   ├── for each index, name of names, numbered from zero
    │   │   ├── if name is None  # an unnamed camera contributes no entry
    │   │   │   └── continue
    │   │   ├── assert name not in name_to_index  # refused rather than silently resolving to one of them
    │   │   └── impls name_to_index[name] = index
    │   ├── impls self._intrinsics = intrinsics  # params each [B], or scalars and [1] columns where the intrinsics broadcasts over the batch
    │   ├── impls self._extrinsics = extrinsics  # matrix [B, 4, 4]
    │   ├── impls self._names = names
    │   ├── impls self._ids = ids
    │   ├── impls self._name_to_index = name_to_index
    │   ├── impls self._device = device  # the resolved device the components were brought to, not read back off them
    │   ├── impls self._dtype = dtype  # the resolved dtype the components were cast to, not read back off them
    │   └── impls self._batch_size = batch_size  # the one batch its components state between them, resolved here and not read again
    ├── @property def intrinsics(self) -> CameraIntrinsics
    │   ├── # The batch's intrinsics, whose params carry the batch axis, or broadcast over it, so its own project / scale_intrinsics cover every camera at once.
    │   └── return self._intrinsics
    ├── @property def extrinsics(self) -> CameraExtrinsics
    │   ├── # The batch's extrinsics, whose [B, 4, 4] matrix every pose op runs over.
    │   └── return self._extrinsics
    ├── def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False, intr_convention: Optional[str] = None, extr_convention: Optional[str] = None) -> "Cameras"
    │   ├── # Return this batch with Tensor.to-style placement / copy semantics plus optional frame conversions, each delegated to the component that owns it.
    │   ├── calls self._intrinsics.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, intr_convention=intr_convention)
    │   ├── calls self._extrinsics.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, extr_convention=extr_convention)
    │   ├── impls cameras = Cameras(intrinsics=the intrinsics it moved, extrinsics=the extrinsics it moved, names=self._names, ids=self._ids, device=device, dtype=dtype)  # the requested placement is handed on, so the new batch's device and dtype follow it
    │   └── return cameras
    ├── def scale_intrinsics(self, resolution: Optional[Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]] = None, scale: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], np.ndarray, torch.Tensor]] = None) -> "Cameras"
    │   ├── # Return this batch restated against a different resolution, the rescale being elementwise on the [B] params its intrinsics already holds.
    │   ├── calls self._intrinsics.scale_intrinsics(resolution=resolution, scale=scale)  # -> intrinsics
    │   ├── impls cameras = Cameras(intrinsics=intrinsics, extrinsics=self._extrinsics, names=self._names, ids=self._ids)  # a method constructing its own enclosing class, drawn as impls because no order puts this method above its class
    │   └── return cameras
    ├── def transform_intrinsics(self, transform: torch.Tensor, resolution: Tuple[int, int]) -> "Cameras"
    │   ├── # Return this batch with its intrinsics restated by a pixel-frame affine, broadcast over the batch axis.
    │   ├── calls self._intrinsics.transform_intrinsics(transform=transform, resolution=resolution)  # -> intrinsics
    │   ├── impls cameras = Cameras(intrinsics=intrinsics, extrinsics=self._extrinsics, names=self._names, ids=self._ids)  # a method constructing its own enclosing class, drawn as impls because no order puts this method above its class
    │   └── return cameras
    ├── def transform_extrinsics(self, scale: Union[int, float, np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], translation: Union[np.ndarray, torch.Tensor, Tuple[Union[int, float], Union[int, float], Union[int, float]], List[Union[int, float]]]) -> "Cameras"
    │   ├── # Return this batch under array-like scale, rotation and translation applied to every pose at once.
    │   ├── calls self._extrinsics.transform_extrinsics(scale=scale, rotation=rotation, translation=translation)  # -> extrinsics
    │   ├── impls cameras = Cameras(intrinsics=self._intrinsics, extrinsics=extrinsics, names=self._names, ids=self._ids)  # a method constructing its own enclosing class, drawn as impls because no order puts this method above its class
    │   └── return cameras
    ├── def __len__(self) -> int
    │   ├── # The number of cameras in the batch, resolved once at construction.
    │   └── return self._batch_size
    ├── def __getitem__(self, index: Union[int, slice, List[int], str]) -> Union["Camera", "Cameras"]
    │   ├── # Index the batch by slicing the leading axis of both components, never by selecting from stored per-camera objects.
    │   ├── def _validate_inputs [local]
    │   │   ├── assert isinstance(index, (int, slice, list, str))
    │   │   ├── if isinstance(index, int)
    │   │   │   ├── assert -len(self) <= index < len(self)  # a component that broadcasts is never indexed, so the batch bounds the position itself
    │   │   │   └── return
    │   │   ├── if isinstance(index, slice)
    │   │   │   └── return
    │   │   ├── if isinstance(index, list)
    │   │   │   ├── for each item of index
    │   │   │   │   ├── assert isinstance(item, int)
    │   │   │   │   └── assert -len(self) <= item < len(self)
    │   │   │   └── return
    │   │   ├── if isinstance(index, str)
    │   │   │   ├── assert index in self._name_to_index  # only a named camera can be looked up by its name
    │   │   │   └── return
    │   │   └── assert 0, "Should not reach here."
    │   ├── calls _validate_inputs
    │   ├── def _normalize_inputs [local]
    │   │   ├── if isinstance(index, str)  # a camera's name stands for the position it holds in the batch
    │   │   │   ├── impls index = self._name_to_index[index]
    │   │   │   └── return index
    │   │   ├── if isinstance(index, (int, slice, list))
    │   │   │   └── return index
    │   │   └── assert 0, "Should not reach here."
    │   ├── calls _normalize_inputs(index=index)
    │   ├── impls index = the returned value from _normalize_inputs
    │   ├── if not self._intrinsics.is_batched or (len(self._intrinsics) == 1 and len(self) > 1)  # an intrinsics that broadcasts over the batch broadcasts over any slice of it, so it is carried whole; a length-1 intrinsics of a length-1 batch is that batch, so it is indexed
    │   │   └── impls intrinsics = self._intrinsics
    │   ├── else
    │   │   └── impls intrinsics = self._intrinsics[index]
    │   ├── impls extrinsics = self._extrinsics[index]  # the poses count the cameras, so the extrinsics are always indexed
    │   ├── if isinstance(index, int)
    │   │   ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, name=self._names[index], id=self._ids[index])  # -> camera
    │   │   └── return camera
    │   ├── if isinstance(index, slice)
    │   │   ├── impls names = self._names[index]
    │   │   ├── impls ids = self._ids[index]
    │   │   ├── impls cameras = Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids)  # a method constructing its own enclosing class, drawn as impls because no order puts this method above its class
    │   │   └── return cameras
    │   ├── if isinstance(index, list)
    │   │   ├── impls names = an empty list
    │   │   ├── impls ids = an empty list
    │   │   ├── for each item of index
    │   │   │   ├── impls append self._names[item] to names
    │   │   │   └── impls append self._ids[item] to ids
    │   │   ├── impls cameras = Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids)  # a method constructing its own enclosing class, drawn as impls because no order puts this method above its class
    │   │   └── return cameras
    │   └── assert 0, "Should not reach here."
    ├── def __iter__(self) -> Iterator["Camera"]
    │   ├── # Iterate one Camera at a time, for callers that genuinely need a single camera rather than the batch.
    │   └── for each index in range(len(self))
    │       └── yield  # self[index]
    ├── @property def names(self) -> Sequence[Optional[str]]
    │   ├── # The per-camera labels, metadata that never enters a tensor op.
    │   └── return self._names
    ├── @property def ids(self) -> Sequence[Optional[int]]
    │   ├── # The per-camera integer identities that survive a serialize / deserialize round trip.
    │   └── return self._ids
    ├── @property def device(self) -> torch.device
    │   ├── # The device this batch was constructed on, the one its component tensors were brought to.
    │   └── return self._device
    ├── @property def dtype(self) -> torch.dtype
    │   ├── # The dtype this batch was constructed with, the one its component tensors were cast to.
    │   └── return self._dtype
    ├── @property def center(self) -> torch.Tensor
    │   ├── # The [B, 3] camera centers, its extrinsics' own center under the batch axis.
    │   └── return self._extrinsics.center
    ├── @property def right(self) -> torch.Tensor
    │   ├── # The [B, 3] physical right axes, its extrinsics' own right under the batch axis.
    │   └── return self._extrinsics.right
    ├── @property def forward(self) -> torch.Tensor
    │   ├── # The [B, 3] physical forward axes, its extrinsics' own forward under the batch axis.
    │   └── return self._extrinsics.forward
    └── @property def up(self) -> torch.Tensor
        ├── # The [B, 3] physical up axes, its extrinsics' own up under the batch axis.
        └── return self._extrinsics.up
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
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics, build_camera_intrinsics
├── if TYPE_CHECKING  # annotation-only imports; runtime imports of Camera / Cameras are inline in the functions that need them (camera.py and cameras.py import io.py, so a top-level import would cycle)
│   ├── from data.structures.three_d.camera.camera import Camera
│   └── from data.structures.three_d.camera.cameras import Cameras
├── _CAMERA_SERIALIZATION_FORMATS        # supported formats: {"json", "npz"}
├── _CAMERA_JSON_KEYS, _CAMERA_NPZ_KEYS  # the payload key schema (model / params / intr_convention / extrinsics / extr_convention / dtype / name / id); the intrinsics and extrinsics fields spell each component as it holds itself, so a single Camera's pose is one [4, 4] and a Cameras' one per camera
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
├── def serialize_cameras(cameras: Union["Camera", "Cameras"], format: str = "json") -> Dict[str, Any]
│   ├── # Serialize cameras to the canonical payload for the requested format, a single Camera's pose, name and id spelled as it holds them.
│   ├── from data.structures.three_d.camera.camera import Camera    # inline runtime import; camera.py imports io.py, so this would cycle at module top
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── assert isinstance(cameras, (Camera, Cameras))
│   │   └── assert format in _CAMERA_SERIALIZATION_FORMATS  # drawn because this is format's only owner on this path
│   ├── calls _validate_inputs
│   ├── def _serialize [local]
│   │   ├── # Map the cameras to the payload the requested format spells them in.
│   │   ├── if format == "json"
│   │   │   ├── calls _serialize_cameras_json(cameras=cameras)
│   │   │   └── return
│   │   ├── if format == "npz"
│   │   │   ├── calls _serialize_cameras_npz(cameras=cameras)
│   │   │   └── return
│   │   └── assert 0, "Should not reach here."
│   ├── calls _serialize
│   └── return
├── def deserialize_cameras(payload: Dict[str, Any], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> Union["Camera", "Cameras"]
│   ├── # Deserialize the canonical payload back into cameras, the inverse of serialize_cameras.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert payload is a dict  # both formats key the payload by field
│   │   ├── impls assert device is None, a str, or a torch.device
│   │   └── impls assert format is in _CAMERA_SERIALIZATION_FORMATS  # drawn because this is format's only owner on this path
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── impls device = torch.device(device) if device is not None else torch.device("cpu")
│   │   └── return device
│   ├── calls _normalize_inputs(device=device)
│   ├── impls device = the returned value from _normalize_inputs
│   ├── def _deserialize [local]
│   │   ├── # Map the payload the requested format spells back to the cameras it carries.
│   │   ├── if format == "json"
│   │   │   ├── calls _deserialize_cameras_json(payload=payload, device=device)
│   │   │   └── return
│   │   ├── if format == "npz"
│   │   │   ├── calls _deserialize_cameras_npz(payload=payload, device=device)
│   │   │   └── return
│   │   └── assert 0, "Should not reach here."
│   ├── calls _deserialize
│   └── return
├── def _serialize_cameras_json(cameras: Union["Camera", "Cameras"]) -> Dict[str, Any]
│   ├── # Map a Camera or a Cameras to the json payload: its intrinsics and extrinsics fields as the components hold them, beside its dtype and its name and id.
│   ├── calls _serialize_camera_intrinsics(intrinsics=cameras.intrinsics)  # -> model, params, intr_convention
│   ├── calls _serialize_camera_extrinsics(extrinsics=cameras.extrinsics)  # -> matrix, extr_convention
│   ├── impls name, id = cameras.name and cameras.id for a Camera, cameras.names and cameras.ids for a Cameras
│   ├── impls payload = {"model": model, "params": params, "intr_convention": intr_convention, "extrinsics": matrix as a nested list, "extr_convention": extr_convention, "dtype": cameras.dtype spelled by its torch name (e.g. "float64"), "name": name, "id": id}  # the resolution rides inside params
│   └── return payload
├── def _deserialize_cameras_json(payload: Dict[str, Any], device: torch.device) -> Union["Camera", "Cameras"]
│   ├── # Map the json payload back to the Camera or Cameras it carries.
│   ├── def _validate_inputs [local]
│   │   ├── assert set(payload.keys()) == _CAMERA_JSON_KEYS  # the payload schema this function reads; what each entry holds is for the component rebuilt from it to check
│   │   └── assert payload["dtype"] is a str naming a torch.dtype attribute of torch  # this function is what maps that name to the dtype the batch is rebuilt in
│   ├── calls _validate_inputs
│   ├── impls dtype = the torch dtype payload["dtype"] spells
│   ├── calls _deserialize_camera_intrinsics(model=payload["model"], params=payload["params"], intr_convention=payload["intr_convention"], device=device, dtype=dtype)  # -> camera_intrinsics
│   ├── calls _deserialize_camera_extrinsics(extrinsics=payload["extrinsics"], extr_convention=payload["extr_convention"], device=device, dtype=dtype)  # -> camera_extrinsics
│   ├── calls _build_cameras(intrinsics=camera_intrinsics, extrinsics=camera_extrinsics, name=payload["name"], id=payload["id"], device=device)
│   └── return
├── def _serialize_cameras_npz(cameras: Union["Camera", "Cameras"]) -> Dict[str, np.ndarray]
│   ├── # Map a Camera or a Cameras to the npz payload: the json payload's fields, each held as an array.
│   ├── calls _serialize_camera_intrinsics(intrinsics=cameras.intrinsics)  # -> model, params, intr_convention
│   ├── calls _serialize_camera_extrinsics(extrinsics=cameras.extrinsics)  # -> matrix, extr_convention
│   ├── impls name, id = cameras.name and cameras.id for a Camera, cameras.names and cameras.ids for a Cameras
│   ├── impls payload = {"model": model, "params": params json-encoded, "intr_convention": intr_convention, "extrinsics": matrix, "extr_convention": extr_convention, "dtype": cameras.dtype spelled by its torch name (e.g. "float64"), "name": name json-encoded, "id": id json-encoded}  # a dict, a list or a null has no typed-array form, so those three ride json-encoded
│   ├── for each key, value of payload
│   │   └── impls payload[key] = value as an np.ndarray  # 0-d for every key but extrinsics, whose [4, 4] or [N, 4, 4] keeps the batch's own dtype
│   └── return payload
├── def _deserialize_cameras_npz(payload: Dict[str, Any], device: torch.device) -> Union["Camera", "Cameras"]
│   ├── # Map the npz payload back to the Camera or Cameras it carries.
│   ├── def _validate_inputs [local]
│   │   ├── assert set(payload.keys()) == _CAMERA_NPZ_KEYS  # the payload schema this function reads; what each entry holds is for the component rebuilt from it to check
│   │   ├── for each key of _CAMERA_NPZ_KEYS
│   │   │   └── assert isinstance(payload[key], np.ndarray)
│   │   ├── for key in ("model", "params", "intr_convention", "extr_convention", "dtype", "name", "id")  # every field but the pose is one string
│   │   │   └── assert payload[key].ndim == 0
│   │   └── assert the str payload["dtype"] holds names a torch.dtype attribute of torch  # this function is what maps that name to the dtype the batch is rebuilt in
│   ├── calls _validate_inputs
│   ├── impls model, params, intr_convention, extr_convention = the str payload["model"], payload["params"], payload["intr_convention"] and payload["extr_convention"] each hold, params decoded from json
│   ├── impls dtype = the torch dtype payload["dtype"] spells
│   ├── impls name, id = the str payload["name"] and payload["id"] each hold, decoded from json  # a list for a batch; a str, an int or null for one camera
│   ├── calls _deserialize_camera_intrinsics(model=model, params=params, intr_convention=intr_convention, device=device, dtype=dtype)  # -> camera_intrinsics
│   ├── calls _deserialize_camera_extrinsics(extrinsics=payload["extrinsics"], extr_convention=extr_convention, device=device, dtype=dtype)  # -> camera_extrinsics, rebuilt in the dtype the archive records
│   ├── calls _build_cameras(intrinsics=camera_intrinsics, extrinsics=camera_extrinsics, name=name, id=id, device=device)
│   └── return
├── def _serialize_camera_intrinsics(intrinsics: CameraIntrinsics) -> Tuple[str, Dict[str, Union[int, float, List[int], List[float]]], str]
│   ├── # Map a CameraIntrinsics, batched or unbatched, to its model, its params as the numbers the camera I/O boundary spells them in, and its image-plane frame.
│   ├── impls serialized_params = an empty dict
│   ├── for each key, value of intrinsics.params
│   │   ├── impls serialized_value = value detached and materialized on cpu as the Python number it holds, or as the list of numbers where it is a column  # an unbatched intrinsics' params are scalars, a batched one's [B] columns
│   │   ├── if key in {"h", "w"}  # the resolution keys serialize as ints
│   │   │   ├── assert n == int(n) for every number n serialized_value holds
│   │   │   └── impls serialized_value = serialized_value with every number it holds as an int
│   │   └── impls serialized_params[key] = serialized_value
│   └── return intrinsics.model, serialized_params, intrinsics.intr_convention
├── def _serialize_camera_extrinsics(extrinsics: CameraExtrinsics) -> Tuple[np.ndarray, str]
│   ├── # Map a CameraExtrinsics, batched or unbatched, to its cam2world matrix and the pose frame it is expressed in.
│   ├── impls matrix = extrinsics.extrinsics detached and materialized on cpu as an ndarray in its own dtype  # [4, 4] for an unbatched extrinsics, [B, 4, 4] for a batched one
│   └── return matrix, extrinsics.extr_convention
├── def _deserialize_camera_intrinsics(model: str, params: Dict[str, Union[int, float, List[int], List[float]]], intr_convention: str, device: torch.device, dtype: torch.dtype) -> CameraIntrinsics
│   ├── # Map a model, its params as numbers or [N] columns, and an image-plane frame back to the unbatched or batched CameraIntrinsics they spell.
│   ├── impls tensor_params = an empty dict
│   ├── for each key, value of params
│   │   └── impls tensor_params[key] = value converted to a torch tensor with the requested device and dtype  # a number to a 0-d tensor and a column to an [N] one, so the intrinsics comes back batched or unbatched as it was saved
│   ├── calls build_camera_intrinsics(model=model, params=tensor_params, intr_convention=intr_convention, device=device)  # validates the model, its params and the image-plane frame those params name
│   └── return
├── def _deserialize_camera_extrinsics(extrinsics: Union[np.ndarray, List[List[float]], List[List[List[float]]]], extr_convention: str, device: torch.device, dtype: torch.dtype) -> CameraExtrinsics
│   ├── # Map a [4, 4] or [N, 4, 4] cam2world matrix and its pose frame back to the unbatched or batched CameraExtrinsics they spell.
│   ├── impls extrinsics = extrinsics as an np.ndarray  # json hands nested lists, npz the array itself
│   ├── calls CameraExtrinsics(extrinsics=extrinsics, extr_convention=extr_convention, device=device, dtype=dtype)
│   └── return
├── def _build_cameras(intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, name: Union[Optional[str], List[Optional[str]]], id: Union[Optional[int], List[Optional[int]]], device: torch.device) -> Union["Camera", "Cameras"]
│   ├── # Rebuild the one Camera an unbatched pose names, or the Cameras a batch of poses names.
│   ├── from data.structures.three_d.camera.camera import Camera    # inline runtime import; camera.py imports io.py, so this would cycle at module top
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── if not extrinsics.is_batched  # one pose is one camera
│   │   ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, name=name, id=id, device=device)  # -> camera
│   │   └── return camera
│   ├── if extrinsics.is_batched  # a batch of poses is a batch of cameras
│   │   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=name, ids=id, device=device)  # -> cameras; field-validates the batch
│   │   └── return cameras
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
