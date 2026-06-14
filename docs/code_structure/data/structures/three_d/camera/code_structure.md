# Camera Data Structure Code Structure

## 1. Code structure trees

`./data/structures/three_d/camera/validation.py`

```text
validation.py
├── from typing import Any, Union
├── import numpy as np
├── import torch
├── from utils.ops.materialize_tensor import materialize_tensor
├── _ROTATION_MATRIX_ATOL_SAFETY_FACTOR                                   # dimensionless backend-neutral margin over a dtype's machine epsilon; each backend validator derives its own per-dtype atol from its own finfo (concrete magnitude is a phase-3 empirical value)
├── def validate_camera_convention(convention: Any) -> str
│   ├── # Validate a camera convention string against the supported set.
│   ├── impls asserts convention is a str in {standard, opengl, opencv, pytorch3d, arkit}
│   └── return convention
├── def validate_camera_intrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]
│   ├── # Dispatch camera-intrinsics validation on the array backend.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_camera_intrinsics_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_camera_intrinsics_torch
│   └── raise TypeError                                                   # obj is neither a numpy array nor a torch tensor
├── def _validate_camera_intrinsics_numpy(obj: Any) -> np.ndarray
│   ├── # Validate a (..., 3, 3) float32 numpy camera-intrinsics matrix.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (3, 3), dtype float32
│   ├── impls asserts fx > 0, fy > 0, cx >= 0, cy >= 0, zero skew, last row [0, 0, 1]
│   └── return obj
├── def _validate_camera_intrinsics_torch(obj: Any) -> torch.Tensor
│   ├── # Validate a (..., 3, 3) float32 torch camera-intrinsics matrix.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (3, 3), dtype float32
│   ├── impls asserts fx > 0, fy > 0, cx >= 0, cy >= 0, zero skew, last row [0, 0, 1]
│   └── return obj
├── def validate_camera_extrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]
│   ├── # Dispatch camera-extrinsics validation on the array backend.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_camera_extrinsics_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_camera_extrinsics_torch
│   └── raise TypeError                                                   # obj is neither a numpy array nor a torch tensor
├── def _validate_camera_extrinsics_numpy(obj: Any) -> np.ndarray
│   ├── # Validate a (..., 4, 4) numpy camera-extrinsics (cam2world) matrix.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (4, 4), dtype in {np.float32, np.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── calls _validate_rotation_matrix_numpy                            # validates obj[..., :3, :3]; tolerance dispatched on its dtype
│   └── return obj
├── def _validate_camera_extrinsics_torch(obj: Any) -> torch.Tensor
│   ├── # Validate a (..., 4, 4) torch camera-extrinsics (cam2world) matrix.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (4, 4), dtype in {torch.float32, torch.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── calls _validate_rotation_matrix_torch                            # validates obj[..., :3, :3]; tolerance dispatched on its dtype
│   └── return obj
├── def validate_rotation_matrix(obj: Any) -> Union[np.ndarray, torch.Tensor]
│   ├── # Dispatch rotation-matrix validation on the array backend.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_rotation_matrix_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_rotation_matrix_torch
│   └── raise TypeError                                                   # obj is neither a numpy array nor a torch tensor
├── def _validate_rotation_matrix_numpy(obj: Any) -> np.ndarray
│   ├── # Validate a (..., 3, 3) numpy rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (3, 3), dtype in {np.float32, np.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_ATOL_SAFETY_FACTOR * float(np.finfo(np.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_ATOL_SAFETY_FACTOR * float(np.finfo(np.float64).eps)
│   ├── if obj.dtype == np.float32
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float32)
│   ├── if obj.dtype == np.float64
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float64)
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_torch(obj: Any) -> torch.Tensor
│   ├── # Validate a (..., 3, 3) torch rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (3, 3), dtype in {torch.float32, torch.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_ATOL_SAFETY_FACTOR * float(torch.finfo(torch.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_ATOL_SAFETY_FACTOR * float(torch.finfo(torch.float64).eps)
│   ├── if obj.dtype == torch.float32
│   │   └── return _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float32)
│   ├── if obj.dtype == torch.float64
│   │   └── return _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float64)
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_numpy_against_threshold(obj: np.ndarray, threshold: float) -> np.ndarray
│   ├── # Core numpy rotation check: orthogonality and determinant within the given atol.
│   ├── impls asserts RR^T close to I at atol=threshold, rtol=0
│   ├── impls asserts det(R) close to 1 at atol=threshold, rtol=0
│   └── return obj
└── def _validate_rotation_matrix_torch_against_threshold(obj: torch.Tensor, threshold: float) -> torch.Tensor
    ├── # Core torch rotation check: orthogonality and determinant within the given atol.
    ├── impls materialize_tensor(obj)
    ├── impls asserts RR^T close to I at atol=threshold, rtol=0
    ├── impls asserts det(R) close to 1 at atol=threshold, rtol=0
    └── return obj
```

`./data/structures/three_d/camera/camera.py`

```text
camera.py
├── import math
├── from pathlib import Path
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.conventions import transform_convention
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── from data.structures.three_d.camera.scaling import scale_intrinsics
├── from data.structures.three_d.camera.validation import validate_camera_convention, validate_camera_extrinsics, validate_camera_intrinsics, validate_rotation_matrix
├── _ORTHOGONALITY_REPAIR_ATOL                                           # dtype-independent input-quality guard: max RR^T-vs-I / determinant residual a raw rotation may carry and still be trusted as SVD-repairable
├── class Camera
│   ├── def __init__(self, intrinsics: Optional[torch.Tensor], extrinsics: torch.Tensor, convention: str, name: Optional[str] = None, id: Optional[int] = None, device: Union[str, torch.device] = torch.device("cuda")) -> None
│   │   ├── # Construct a Camera from intrinsics/extrinsics/convention, validating them and placing the tensors on device.
│   │   ├── def _validate_inputs()
│   │   │   ├── # Validate the constructor arguments.
│   │   │   ├── calls validate_camera_intrinsics                         # when intrinsics is not None
│   │   │   ├── calls validate_camera_extrinsics
│   │   │   └── calls validate_camera_convention
│   │   ├── calls _validate_inputs
│   │   ├── def _normalize_inputs(intrinsics, extrinsics, device)
│   │   │   ├── # Resolve device, default missing intrinsics, and move the tensors onto device.
│   │   │   ├── impls resolves device, defaults None intrinsics to eye(3) float32, moves tensors to device
│   │   │   ├── calls validate_camera_intrinsics
│   │   │   └── calls validate_camera_extrinsics
│   │   ├── calls _normalize_inputs
│   │   └── impls stores _intrinsics, _extrinsics, _convention, _name, _id, _device
│   ├── def intrinsics(self) -> torch.Tensor                            # @property
│   │   └── # The camera intrinsics matrix.
│   ├── def extrinsics(self) -> torch.Tensor                            # @property
│   │   └── # The camera extrinsics (cam2world) matrix.
│   ├── def convention(self) -> str                                     # @property
│   │   └── # The camera convention name.
│   ├── def name(self) -> Optional[str]                                 # @property
│   │   └── # The camera name.
│   ├── def id(self) -> Optional[int]                                   # @property
│   │   └── # The camera id.
│   ├── def device(self) -> torch.device                                # @property
│   │   └── # The device the camera tensors live on.
│   ├── def w2c(self) -> torch.Tensor                                   # @property
│   │   └── # The world-to-camera matrix (inverse of extrinsics).
│   ├── def fx(self) -> float                                           # @property
│   │   └── # The horizontal focal length.
│   ├── def fy(self) -> float                                           # @property
│   │   └── # The vertical focal length.
│   ├── def cx(self) -> float                                           # @property
│   │   └── # The horizontal principal-point coordinate.
│   ├── def cy(self) -> float                                           # @property
│   │   └── # The vertical principal-point coordinate.
│   ├── def fov(self) -> Tuple[float, float]                            # @property
│   │   └── # The horizontal/vertical field of view in degrees.
│   ├── def center(self) -> torch.Tensor                                # @property
│   │   └── # The camera center extrinsics[:3, 3].
│   ├── def right(self) -> torch.Tensor                                 # @property
│   │   ├── # The convention-dispatched right axis.
│   │   └── impls selects the right axis per convention and asserts unit norm
│   ├── def forward(self) -> torch.Tensor                               # @property
│   │   ├── # The convention-dispatched forward axis.
│   │   └── impls selects the forward axis per convention and asserts unit norm
│   ├── def up(self) -> torch.Tensor                                    # @property
│   │   ├── # The convention-dispatched up axis.
│   │   └── impls selects the up axis per convention and asserts unit norm
│   ├── def to(self, device: Optional[Union[str, torch.device]] = None, convention: Optional[str] = None) -> "Camera"
│   │   ├── # Return this Camera on a target device / convention (self when unchanged).
│   │   ├── calls validate_camera_convention                            # when convention is not None
│   │   ├── calls transform_convention                                  # when convention differs
│   │   └── return Camera(...)
│   ├── def scale_intrinsics(self, resolution: Optional[Tuple[int, int]] = None, scale: Optional[Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]] = None) -> "Camera"
│   │   ├── # Return this Camera with intrinsics scaled to a resolution or by a factor.
│   │   ├── calls scale_intrinsics
│   │   └── return Camera(...)
│   ├── def transform(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "Camera"
│   │   ├── # Return this Camera under a similarity transform (scale, rotation, translation) of its cam2world pose.
│   │   ├── def _validate_inputs()
│   │   │   ├── # Validate the transform scale/rotation/translation arguments.
│   │   │   └── calls validate_rotation_matrix                          # rotation is (3, 3) np.float32
│   │   ├── calls _validate_inputs
│   │   ├── def _normalize_inputs(rotation, translation)
│   │   │   └── # Move rotation/translation onto self device and extrinsics dtype as tensors.
│   │   ├── calls _normalize_inputs
│   │   ├── impls composes the new cam2world rotation/translation from scale, rotation, translation
│   │   ├── calls _stabilize_rotation_matrix(extrinsics_new[:3, :3])
│   │   └── return Camera(...)                                          # re-validates via validate_camera_extrinsics
│   ├── def serialize(self, format: str = "json") -> Dict[str, Any]
│   │   ├── # Serialize this Camera into a single-form payload.
│   │   └── calls serialize_cameras
│   ├── def deserialize(cls, payload: Dict[str, Any], device: Optional[Union[str, torch.device]] = None, format: str = "json") -> "Camera"     # @classmethod
│   │   ├── # Deserialize one Camera from a single-form payload.
│   │   └── calls deserialize_cameras
│   ├── def save(self, camera_path: Path) -> None
│   │   ├── # Save this Camera to a .npz or .json file.
│   │   └── calls save_cameras
│   └── def load(cls, camera_path: Path, device: Optional[Union[str, torch.device]] = None) -> "Camera"                                        # @classmethod
│       ├── # Load one Camera from a .npz or .json file.
│       └── calls load_cameras
└── def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor
    ├── # Project a near-orthogonal (3, 3) rotation onto the nearest proper rotation, in the received dtype.
    ├── def _validate_inputs()
    │   ├── # Validate the rotation argument's type, shape, and dtype.
    │   └── impls asserts torch.Tensor, shape (3, 3), dtype in {torch.float32, torch.float64}   # acknowledges the received dtype; rejects every other dtype
    ├── calls _validate_inputs
    ├── impls computes the RR^T-vs-I residual and the |det(R) - 1| residual in rotation.dtype
    ├── impls asserts max(orthogonality residual, determinant residual) <= _ORTHOGONALITY_REPAIR_ATOL
    ├── impls u, _, v_h = svd(rotation) in rotation.dtype; rotation_fixed = u @ v_h; if det(rotation_fixed) < 0 -> flip u[:, -1] and recompute rotation_fixed
    ├── calls validate_rotation_matrix(rotation_fixed)                   # re-validates at rotation.dtype's atol
    └── return rotation_fixed
```

`./data/structures/three_d/camera/camera_vis.py`

```text
camera_vis.py
├── from typing import Any, Dict, List, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── DEFAULT_FRUSTUM_SIZE = 0.25             # world-unit frustum/axis size, resolved when frustum_size is None
├── DEFAULT_FRUSTUM_COLOR = (255, 214, 0)   # RGB line color, resolved when frustum_color is None
├── DEFAULT_POINT_SIZE = 0.01               # world-unit size of the camera-center point marker, resolved when point_size is None
├── DEFAULT_POINT_COLOR = (255, 214, 0)     # RGB center-point color, resolved when point_color is None
├── def cameras_vis(cameras: Cameras, frustum_size: Optional[float] = None, frustum_color: Optional[Tuple[int, int, int]] = None, point_size: Optional[float] = None, point_color: Optional[Tuple[int, int, int]] = None) -> List[Dict[str, Any]]
│   ├── # The cameras atomic-display data-layer mapping.
│   ├── for each camera
│   │   └── calls camera_vis(camera, frustum_size, frustum_color, point_size, point_color)
│   └── return
└── def camera_vis(camera: Camera, frustum_size: Optional[float] = None, frustum_color: Optional[Tuple[int, int, int]] = None, point_size: Optional[float] = None, point_color: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]
    ├── # The per-camera atomic-display data-layer mapping.
    ├── impls resolves frustum_size / frustum_color / point_size / point_color from None to DEFAULT_FRUSTUM_SIZE / DEFAULT_FRUSTUM_COLOR / DEFAULT_POINT_SIZE / DEFAULT_POINT_COLOR
    ├── impls computes center, center_color from point_color, and center_size from point_size
    ├── impls computes axes and frustum lines colored by frustum_color from camera center, right, forward, up, intrinsics, and frustum_size
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
│   │   ├── calls validate_camera_convention
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
│   ├── calls validate_camera_intrinsics
│   ├── calls validate_camera_extrinsics
│   ├── for each batch index
│   │   ├── impls decodes that index's intrinsics, extrinsics, convention, name, and id (resolving has_name / has_id flags and the -1 id sentinel) to tensors on device
│   │   └── calls validate_camera_convention
│   ├── calls Cameras                               # constructs and field-validates the batch
│   └── return
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   ├── # Resolve a Cameras serialization format from a file path.
│   └── calls _normalize_format
└── def _normalize_format(format: str) -> str
    └── # Normalize a path suffix or format name to a supported serialization format.
```
