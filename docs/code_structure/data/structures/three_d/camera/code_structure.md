# Camera Data Structure Code Structure

## 1. Inheritance / type trees

```text
class ABC
└── class CameraIntrinsics  # from here down, its complete set of direct subclasses
    ├── class CameraIntrinsicsSimplePinhole
    ├── class CameraIntrinsicsPinhole
    └── class CameraIntrinsicsOrtho
```

## 2. Code structure trees

`./data/structures/three_d/camera/intrinsics/validation.py`

```text
validation.py
├── from typing import Any, Dict, Union
├── def validate_camera_intrinsics_attributes(model: str, params: Any, device: Any) -> None
│   ├── # Single-entry validation for CameraIntrinsics.__init__: validate the camera model, its named params, and the device.
│   ├── calls validate_camera_model
│   ├── calls validate_camera_intrinsics_params
│   ├── impls asserts device is a valid torch device spec (str or torch.device)
│   └── return
├── def validate_camera_model(model: Any) -> str
│   ├── # Validate a camera-model string against the supported set.
│   ├── impls asserts model is a str in {simple_pinhole, pinhole, ortho}
│   └── return model
├── def validate_camera_intrinsics_params(model: str, params: Any) -> Dict[str, Union[int, float]]
│   ├── # Validate the named intrinsics params for a camera model by dispatching on the model string (all models are structurally equivalent siblings).
│   ├── if model == "simple_pinhole"
│   │   └── return _validate_camera_intrinsics_params_simple_pinhole(params)
│   ├── if model == "pinhole"
│   │   └── return _validate_camera_intrinsics_params_pinhole(params)
│   ├── if model == "ortho"
│   │   └── return _validate_camera_intrinsics_params_ortho(params)
│   └── assert 0, "Should not reach here."
├── def _validate_camera_intrinsics_params_simple_pinhole(params: Any) -> Dict[str, Union[int, float]]
│   ├── # Validate simple_pinhole params: a single shared focal length f plus the principal point cx / cy.
│   ├── impls asserts params is a Dict[str, Union[int, float]] with exactly keys {f, cx, cy}
│   ├── impls asserts f > 0 and cx >= 0 and cy >= 0  # impls-node-one-step:skip
│   └── return params
├── def _validate_camera_intrinsics_params_pinhole(params: Any) -> Dict[str, Union[int, float]]
│   ├── # Validate pinhole params: independent focal lengths fx / fy plus the principal point cx / cy.
│   ├── impls asserts params is a Dict[str, Union[int, float]] with exactly keys {fx, fy, cx, cy}
│   ├── impls asserts fx > 0 and fy > 0 and cx >= 0 and cy >= 0  # impls-node-one-step:skip
│   └── return params
└── def _validate_camera_intrinsics_params_ortho(params: Any) -> Dict[str, Union[int, float]]
    ├── # Validate ortho (weak-perspective) params: focal scales fx / fy plus the principal-point offset cx / cy.
    ├── impls asserts params is a Dict[str, Union[int, float]] with exactly keys {fx, fy, cx, cy}
    ├── impls asserts fx > 0 and fy > 0 and cx and cy are finite  # impls-node-one-step:skip
    └── return params
```

`./data/structures/three_d/camera/extrinsics/validation.py`

```text
validation.py
├── from typing import Any, Union
├── import numpy as np
├── import torch
├── from utils.ops.materialize_tensor import materialize_tensor
├── _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS = 32  # orthogonality/determinant residual floor of the float SVD-projection, in machine-epsilon units; the eps-scaling is derived, the O(1) prefactor is the empirical LAPACK SVD/det floor (measured worst <= 11 over the reference poses + 53k synthetic rotations; set to 32 for margin, still orders of magnitude below any genuinely non-orthogonal rotation)
├── def validate_camera_extrinsics_attributes(extrinsics: Any, convention: Any, device: Any) -> None
│   ├── # Single-entry validation for CameraExtrinsics.__init__: validate the 4x4 cam2world matrix, the coordinate-frame convention, and the device.
│   ├── calls validate_camera_convention
│   ├── calls validate_camera_extrinsics
│   ├── impls asserts device is a valid torch device spec (str or torch.device)
│   └── return
├── def validate_camera_convention(convention: Any) -> str
│   ├── # Validate a camera convention string against the supported set.
│   ├── impls asserts convention is a str in {standard, opengl, opencv, pytorch3d, arkit}
│   └── return convention
├── def validate_camera_extrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]
│   ├── # Dispatch camera-extrinsics validation on the array backend.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_camera_extrinsics_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_camera_extrinsics_torch
│   └── raise TypeError  # obj is neither a numpy array nor a torch tensor
├── def _validate_camera_extrinsics_numpy(obj: Any) -> np.ndarray
│   ├── # Validate a (..., 4, 4) numpy camera-extrinsics (cam2world) matrix.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (4, 4), dtype in {np.float32, np.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── calls _validate_rotation_matrix_numpy(obj[..., :3, :3])  # the tolerance is dispatched on that block's dtype
│   └── return obj
├── def _validate_camera_extrinsics_torch(obj: Any) -> torch.Tensor
│   ├── # Validate a (..., 4, 4) torch camera-extrinsics (cam2world) matrix.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (4, 4), dtype in {torch.float32, torch.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── calls _validate_rotation_matrix_torch(obj[..., :3, :3])  # the tolerance is dispatched on that block's dtype
│   └── return obj
├── def validate_rotation_matrix(obj: Any) -> Union[np.ndarray, torch.Tensor]
│   ├── # Dispatch rotation-matrix validation on the array backend.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_rotation_matrix_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_rotation_matrix_torch
│   └── raise TypeError  # obj is neither a numpy array nor a torch tensor
├── def _validate_rotation_matrix_numpy(obj: Any) -> np.ndarray
│   ├── # Validate a (..., 3, 3) numpy rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (3, 3), dtype in {np.float32, np.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float64).eps)
│   ├── if obj.dtype == np.float32
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float32)
│   ├── if obj.dtype == np.float64
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float64)
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_torch(obj: Any) -> torch.Tensor
│   ├── # Validate a (..., 3, 3) torch rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (3, 3), dtype in {torch.float32, torch.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float64).eps)
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

`./data/structures/three_d/camera/validation.py`

```text
validation.py
├── from typing import TYPE_CHECKING, List, Optional, Union
├── import torch
├── if TYPE_CHECKING  # annotation-only imports; the runtime type checks import the two classes inline (no cycle, but the top-level refs stay annotation-only)
│   ├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
│   └── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── def validate_cameras_attributes(intrinsics: List["CameraIntrinsics"], extrinsics: List["CameraExtrinsics"], names: List[Optional[str]], ids: List[Optional[int]], device: Union[str, torch.device]) -> None
│   ├── # Single-entry validation for Cameras.__init__: validate the parallel per-camera lists, names / ids, and device, plus the inter-relationship that all four per-camera lists are equal length.
│   ├── impls asserts len(intrinsics) == len(extrinsics) == len(names) == len(ids)
│   ├── for each index-aligned (intrinsic, extrinsic, name, id)
│   │   └── calls validate_camera_attributes
│   ├── impls asserts device is a valid torch device spec
│   └── return
└── def validate_camera_attributes(intrinsics: "CameraIntrinsics", extrinsics: "CameraExtrinsics", name: Optional[str], id: Optional[int], device: Union[str, torch.device]) -> None
    ├── # Single-entry validation for Camera.__init__: assert the parts are a CameraIntrinsics / CameraExtrinsics and validate the name / id / device attributes, relying on each part's own validation for its internals.
    ├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics             # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics             # inline runtime import; the top-level import is TYPE_CHECKING-only
    ├── impls asserts isinstance(intrinsics, CameraIntrinsics) and isinstance(extrinsics, CameraExtrinsics)  # impls-node-one-step:skip
    ├── impls asserts name is None or a str and id is None or an int                                         # impls-node-one-step:skip
    ├── impls asserts device is a valid torch device spec
    └── return
```

`./data/structures/three_d/camera/intrinsics/camera_intrinsics.py`

```text
camera_intrinsics.py
├── from abc import ABC
├── from typing import ClassVar, Dict, Optional, Tuple, Union
├── import torch
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_attributes
├── class CameraIntrinsics(ABC)   [abstract]
│   ├── # Abstract base for a camera's intrinsics: owns the named params + device and the projection contract, with each concrete subclass being exactly one camera model.
│   ├── MODEL: ClassVar[str]  # each concrete subclass sets its camera-model identifier (simple_pinhole / pinhole / ortho)
│   ├── def __init__(self, params: Dict[str, Union[int, float]], device: Union[str, torch.device] = torch.device("cuda")) -> None
│   │   ├── # Construct a CameraIntrinsics from its model's named params and a device, validating every attribute.
│   │   ├── calls validate_camera_intrinsics_attributes(model=type(self).MODEL, params=params, device=device)
│   │   ├── impls self._params = params
│   │   └── impls self._device = device
│   ├── def model(self) -> str  # @property
│   │   ├── # The camera-model identifier type(self).MODEL.
│   │   └── return type(self).MODEL
│   ├── def params(self) -> Dict[str, Union[int, float]]  # @property
│   │   ├── # The model's named intrinsics parameters.
│   │   └── return self._params
│   ├── def device(self) -> torch.device  # @property
│   │   ├── # The device the intrinsics live on.
│   │   └── return self._device
│   ├── def cx(self) -> float  # @property
│   │   ├── # The horizontal principal-point coordinate params["cx"].
│   │   └── return self._params["cx"]
│   ├── def cy(self) -> float  # @property
│   │   ├── # The vertical principal-point coordinate params["cy"].
│   │   └── return self._params["cy"]
│   ├── def fx(self) -> float  # @property [abstract]
│   │   └── # Abstract: the horizontal focal length / scale, whose params key differs per model.
│   ├── def fy(self) -> float  # @property [abstract]
│   │   └── # Abstract: the vertical focal length / scale, whose params key differs per model.
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [abstract]
│   │   └── # Abstract: map camera-space 3D points [..., 3] to 2D image points [..., 2] under this model.
│   ├── def scale_intrinsics(self, resolution: Optional[Tuple[int, int]] = None, scale: Optional[Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]] = None) -> "CameraIntrinsics"
│   │   ├── # Return this CameraIntrinsics scaled to a resolution or by a factor, scaling the focal and principal-point params.
│   │   ├── impls resolve the per-axis (sx, sy) scale from resolution (target over current) or from scale
│   │   ├── impls scale the focal params (f / fx / fy) and the principal-point params (cx, cy) in a copy of self.params by (sx, sy)  # impls-node-one-step:skip
│   │   └── return  # type(self)(scaled_params, self._device)
│   └── def to(self, device: Optional[Union[str, torch.device]] = None) -> "CameraIntrinsics"
│       ├── # Return this CameraIntrinsics on a target device (self when the device is unchanged).
│       ├── if device is None or torch.device(device) == self._device
│       │   └── return self
│       └── return type(self)(self._params, device)
├── class CameraIntrinsicsSimplePinhole(CameraIntrinsics)
│   ├── # Simple-pinhole intrinsics: a single shared focal length f under a perspective projection.
│   ├── MODEL: ClassVar[str] = "simple_pinhole"
│   ├── def fx(self) -> float  # @property [override]
│   │   ├── # The shared focal length params["f"].
│   │   └── return self._params["f"]
│   ├── def fy(self) -> float  # @property [override]
│   │   ├── # The shared focal length params["f"].
│   │   └── return self._params["f"]
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│   │   ├── # Perspective projection with a single shared focal length.
│   │   ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│   │   ├── impls z = points_camera[..., 2]
│   │   ├── impls in place: out[..., 0] = f * out[..., 0] / z + cx  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   ├── impls in place: out[..., 1] = f * out[..., 1] / z + cy  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
│   └── def fov(self) -> Tuple[float, float]  # @property
│       ├── # The horizontal / vertical field of view in degrees (perspective model only).
│       └── impls computes (horizontal, vertical) fov in degrees from f, cx, cy
├── class CameraIntrinsicsPinhole(CameraIntrinsics)
│   ├── # Pinhole intrinsics: independent focal lengths fx / fy under a perspective projection.
│   ├── MODEL: ClassVar[str] = "pinhole"
│   ├── def fx(self) -> float  # @property [override]
│   │   ├── # The horizontal focal length params["fx"].
│   │   └── return self._params["fx"]
│   ├── def fy(self) -> float  # @property [override]
│   │   ├── # The vertical focal length params["fy"].
│   │   └── return self._params["fy"]
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│   │   ├── # Perspective projection with independent fx / fy.
│   │   ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│   │   ├── impls z = points_camera[..., 2]
│   │   ├── impls in place: out[..., 0] = fx * out[..., 0] / z + cx  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   ├── impls in place: out[..., 1] = fy * out[..., 1] / z + cy  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
│   └── def fov(self) -> Tuple[float, float]  # @property
│       ├── # The horizontal / vertical field of view in degrees (perspective model only).
│       └── impls computes (horizontal, vertical) fov in degrees from fx, fy, cx, cy
├── class CameraIntrinsicsOrtho(CameraIntrinsics)
│   ├── # Ortho (weak-perspective) intrinsics: independent focal scales fx / fy with no perspective divide.
│   ├── MODEL: ClassVar[str] = "ortho"
│   ├── def fx(self) -> float  # @property [override]
│   │   ├── # The horizontal focal scale params["fx"].
│   │   └── return self._params["fx"]
│   ├── def fy(self) -> float  # @property [override]
│   │   ├── # The vertical focal scale params["fy"].
│   │   └── return self._params["fy"]
│   └── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│       ├── # Orthographic projection with independent fx / fy scales (no perspective divide).
│       ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│       ├── impls in place: out[..., 0] = fx * out[..., 0] + cx  (mul_ / add_)                                      # impls-node-one-step:skip
│       ├── impls in place: out[..., 1] = fy * out[..., 1] + cy  (mul_ / add_)                                      # impls-node-one-step:skip
│       └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
└── def build_camera_intrinsics(model: str, params: Dict[str, Union[int, float]], device: Union[str, torch.device] = torch.device("cuda")) -> CameraIntrinsics
    ├── # Build the CameraIntrinsics subclass for a camera-model string (the serialization-boundary factory) by dispatching on the model.
    ├── if model == "simple_pinhole"
    │   └── return CameraIntrinsicsSimplePinhole(params, device)
    ├── if model == "pinhole"
    │   └── return CameraIntrinsicsPinhole(params, device)
    ├── if model == "ortho"
    │   └── return CameraIntrinsicsOrtho(params, device)
    └── assert 0, "Should not reach here."
```

`./data/structures/three_d/camera/extrinsics/camera_extrinsics.py`

```text
camera_extrinsics.py
├── from typing import Optional, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.extrinsics.conventions import transform_convention
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_convention, validate_camera_extrinsics_attributes, validate_rotation_matrix
├── _ORTHOGONALITY_REPAIR_ATOL = 1.0e-05  # dtype-independent input-quality guard: max RR^T-vs-I / determinant residual a raw rotation may carry and still be trusted as SVD-repairable
├── class CameraExtrinsics
│   ├── def __init__(self, extrinsics: torch.Tensor, convention: str, device: Union[str, torch.device] = torch.device("cuda")) -> None
│   │   ├── # Construct a CameraExtrinsics from a 4x4 cam2world matrix and its coordinate-frame convention, validating both.
│   │   ├── calls validate_camera_extrinsics_attributes(extrinsics=extrinsics, convention=convention, device=device)
│   │   ├── impls move the extrinsics to device
│   │   ├── impls self._extrinsics = extrinsics
│   │   ├── impls self._convention = convention
│   │   └── impls self._device = device
│   ├── def extrinsics(self) -> torch.Tensor  # @property
│   │   ├── # The 4x4 camera-to-world extrinsics matrix.
│   │   └── return self._extrinsics
│   ├── def convention(self) -> str  # @property
│   │   ├── # The coordinate-frame convention (standard / opengl / opencv / pytorch3d / arkit).
│   │   └── return self._convention
│   ├── def device(self) -> torch.device  # @property
│   │   ├── # The device the extrinsics live on.
│   │   └── return self._device
│   ├── def w2c(self) -> torch.Tensor  # @property
│   │   ├── # The world-to-camera matrix (inverse of extrinsics).
│   │   └── return the matrix inverse of self._extrinsics
│   ├── def center(self) -> torch.Tensor  # @property
│   │   ├── # The camera center extrinsics[:3, 3].
│   │   └── return self._extrinsics[:3, 3]
│   ├── def right(self) -> torch.Tensor  # @property
│   │   ├── # The convention-dispatched physical right axis.
│   │   ├── impls select the right axis per convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def forward(self) -> torch.Tensor  # @property
│   │   ├── # The convention-dispatched physical forward axis.
│   │   ├── impls select the forward axis per convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def up(self) -> torch.Tensor  # @property
│   │   ├── # The convention-dispatched physical up axis.
│   │   ├── impls select the up axis per convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def to(self, device: Optional[Union[str, torch.device]] = None, convention: Optional[str] = None) -> "CameraExtrinsics"
│   │   ├── # Return this CameraExtrinsics on a target device / convention (self when unchanged).
│   │   ├── if convention is not None
│   │   │   └── calls validate_camera_convention
│   │   ├── if convention is not None and convention != self._convention
│   │   │   └── calls transform_convention
│   │   └── return CameraExtrinsics(...)
│   └── def transform(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "CameraExtrinsics"
│       ├── # Return this CameraExtrinsics under a similarity transform (scale, rotation, translation) of its cam2world pose.
│       ├── impls composes the new cam2world rotation/translation from scale, rotation, translation
│       ├── calls _stabilize_rotation_matrix
│       └── return CameraExtrinsics(...)  # re-validates via validate_camera_extrinsics_attributes
└── def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor
    ├── # Project a near-orthogonal (3, 3) rotation onto the nearest proper rotation, in the received dtype (float32 or float64).
    ├── impls computes the RR^T-vs-I residual in rotation.dtype
    ├── impls computes the |det(R) - 1| residual in rotation.dtype
    ├── impls asserts max(orthogonality residual, determinant residual) <= _ORTHOGONALITY_REPAIR_ATOL
    ├── impls u, _, v_h = svd(rotation) in rotation.dtype
    ├── impls rotation_fixed = u @ v_h
    ├── if det(rotation_fixed) < 0
    │   ├── impls flip u[:, -1]
    │   └── impls recompute rotation_fixed = u @ v_h
    ├── calls validate_rotation_matrix
    └── return rotation_fixed
```

`./data/structures/three_d/camera/camera.py`

```text
camera.py
├── from pathlib import Path
├── from typing import Any, Dict, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── from data.structures.three_d.camera.validation import validate_camera_attributes
└── class Camera
    ├── def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics, name: Optional[str] = None, id: Optional[int] = None, device: Union[str, torch.device] = torch.device("cuda")) -> None
    │   ├── # Construct a Camera from a CameraIntrinsics and a CameraExtrinsics, keeping name / id / device.
    │   ├── calls validate_camera_attributes(intrinsics=intrinsics, extrinsics=extrinsics, name=name, id=id, device=device)
    │   ├── impls move the intrinsics / extrinsics to device
    │   ├── impls self._intrinsics = intrinsics
    │   ├── impls self._extrinsics = extrinsics
    │   ├── impls self._name = name
    │   ├── impls self._id = id
    │   └── impls self._device = device
    ├── def intrinsics(self) -> CameraIntrinsics  # @property
    │   ├── # The camera's CameraIntrinsics ("what the camera is").
    │   └── return self._intrinsics
    ├── def extrinsics(self) -> CameraExtrinsics  # @property
    │   ├── # The camera's CameraExtrinsics ("where the camera is").
    │   └── return self._extrinsics
    ├── def name(self) -> Optional[str]  # @property
    │   ├── # The camera name.
    │   └── return self._name
    ├── def id(self) -> Optional[int]  # @property
    │   ├── # The camera id.
    │   └── return self._id
    ├── def device(self) -> torch.device  # @property
    │   ├── # The device the camera tensors live on.
    │   └── return self._device
    ├── def to(self, device: Optional[Union[str, torch.device]] = None, convention: Optional[str] = None) -> "Camera"
    │   ├── # Return this Camera on a target device / extrinsics convention (self when unchanged).
    │   ├── calls self._intrinsics.to(device=device)
    │   ├── calls self._extrinsics.to(device=device, convention=convention)
    │   └── return Camera(...)
    ├── def scale_intrinsics(self, resolution: Optional[Tuple[int, int]] = None, scale: Optional[Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]] = None) -> "Camera"
    │   ├── # Return this Camera with its CameraIntrinsics scaled to a resolution or by a factor.
    │   ├── calls self._intrinsics.scale_intrinsics(resolution=resolution, scale=scale)
    │   └── return Camera(...)
    ├── def transform(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "Camera"
    │   ├── # Return this Camera under a similarity transform of its CameraExtrinsics pose.
    │   ├── calls self._extrinsics.transform(scale=scale, rotation=rotation, translation=translation)
    │   └── return Camera(...)
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

`./data/structures/three_d/camera/cameras.py`

```text
cameras.py
├── from typing import Iterator, List, Optional, Sequence, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.validation import validate_cameras_attributes
└── class Cameras
    ├── def __init__(self, intrinsics: List[CameraIntrinsics], extrinsics: List[CameraExtrinsics], names: Optional[List[Optional[str]]] = None, ids: Optional[List[Optional[int]]] = None, device: Union[str, torch.device] = torch.device("cuda")) -> None
    │   ├── # Construct a Cameras from parallel lists of CameraIntrinsics and CameraExtrinsics, keeping per-camera names / ids.
    │   ├── calls validate_cameras_attributes(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device=device)
    │   ├── impls move each CameraIntrinsics / CameraExtrinsics to device
    │   ├── impls self._intrinsics = intrinsics
    │   ├── impls self._extrinsics = extrinsics
    │   ├── impls self._names = names
    │   ├── impls self._ids = ids
    │   ├── impls self._device = device
    │   └── impls self._name_to_index = the name → index map
    ├── def __len__(self) -> int
    │   ├── # The number of cameras in the collection.
    │   └── return len(self._intrinsics)
    ├── def __getitem__(self, index: Union[int, slice, List[int], str]) -> Union["Camera", "Cameras"]
    │   ├── # Index the collection: a name / int yields one Camera, a slice / int-list yields a sub-Cameras.
    │   ├── if isinstance(index, str)
    │   │   └── return  # the Camera at the name's index
    │   ├── if isinstance(index, (slice, list))
    │   │   └── return Cameras(...)  # the selected sub-collection
    │   └── return Camera(...)  # the single indexed Camera
    ├── def __iter__(self) -> Iterator["Camera"]
    │   ├── # Iterate the collection one Camera at a time.
    │   └── for each index in range(len(self))
    │       └── yield  # self[index]
    ├── def to(self, device: Optional[Union[str, torch.device]] = None, convention: Optional[str] = None) -> "Cameras"
    │   ├── # Return this Cameras on a target device / convention (self when unchanged).
    │   ├── for each camera in self
    │   │   └── calls camera.to(device=device, convention=convention)
    │   └── return Cameras(...)
    ├── def transform(self, scale: float, rotation: np.ndarray, translation: np.ndarray) -> "Cameras"
    │   ├── # Return this Cameras under a similarity transform applied to each camera's CameraExtrinsics pose.
    │   ├── for each camera in self
    │   │   └── calls camera.transform(scale=scale, rotation=rotation, translation=translation)
    │   └── return Cameras(...)
    ├── def intrinsics(self) -> Sequence[CameraIntrinsics]  # @property
    │   ├── # The per-camera CameraIntrinsics.
    │   └── return self._intrinsics
    ├── def extrinsics(self) -> Sequence[CameraExtrinsics]  # @property
    │   ├── # The per-camera CameraExtrinsics.
    │   └── return self._extrinsics
    ├── def conventions(self) -> Sequence[str]  # @property
    │   ├── # The per-camera coordinate-frame conventions, one per CameraExtrinsics.
    │   └── return [extrinsics.convention for extrinsics in self._extrinsics]
    ├── def names(self) -> Sequence[Optional[str]]  # @property
    │   ├── # The per-camera names.
    │   └── return self._names
    ├── def ids(self) -> Sequence[Optional[int]]  # @property
    │   ├── # The per-camera ids.
    │   └── return self._ids
    ├── def device(self) -> torch.device  # @property
    │   ├── # The device the cameras live on.
    │   └── return self._device
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

`./data/structures/three_d/camera/camera_vis.py`

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

`./data/structures/three_d/camera/io.py`

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
├── _CAMERA_JSON_KEYS, _CAMERA_NPZ_KEYS  # one camera's payload key schema (model / params / extrinsics / convention / name / id, plus has_name / has_id for npz); a collection is just many of these
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
│   │   │   └── return the file's utf-8 text parsed as json
│   │   ├── if format == "npz"
│   │   │   └── return every array in the npz archive, keyed by its name
│   │   └── assert 0, "Should not reach here."
│   ├── calls _read_payload
│   └── return deserialize_cameras(payload=payload, device=device, format=format)
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
│   │   │   └── return _serialize_cameras_json(cameras=cameras)
│   │   ├── if format == "npz"
│   │   │   └── return _serialize_cameras_npz(cameras=cameras)
│   │   └── assert 0, "Should not reach here."
│   ├── calls _serialize
│   ├── def _normalize_outputs [local]
│   │   ├── # Hand back the single form the caller's own one Camera asked for, else the plural payload whole.
│   │   ├── if was_single
│   │   │   └── return _normalize_payload_to_single(payload=payload, format=format)
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
│   │   │   └── return _deserialize_cameras_json(per_camera_dicts=payload, device=target_device)
│   │   ├── if format == "npz"
│   │   │   └── return _deserialize_cameras_npz(payload=payload, device=target_device)
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
│   │   └── impls builds that camera's json dict from intrinsics.model, intrinsics.params, extrinsics.extrinsics, extrinsics.convention, name, and id  # impls-node-one-step:skip
│   └── return
├── def _deserialize_cameras_json(per_camera_dicts: List[Dict[str, Any]], device: torch.device) -> "Cameras"
│   ├── # Map the plural json per-camera dicts to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── impls assert per_camera_dicts is a non-empty list
│   │   └── for each per-camera dict
│   │       ├── impls assert it is a dict whose keys are exactly _CAMERA_JSON_KEYS
│   │       ├── impls assert its model and convention are str, and its params a dict
│   │       └── impls assert its name is None or a str, and its id None or an int
│   ├── calls _validate_inputs
│   ├── impls intrinsics_list, extrinsics_list, names, ids — four empty accumulators the loop appends to
│   ├── for each per-camera dict
│   │   ├── impls decodes extrinsics to a tensor on device
│   │   ├── calls build_camera_intrinsics(model=per_camera_dict["model"], params=per_camera_dict["params"], device=device)  # validates model + params
│   │   ├── calls CameraExtrinsics(extrinsics=extrinsics, convention=per_camera_dict["convention"], device=device)          # validates extrinsics + convention
│   │   └── impls appends per_camera_dict["name"] and per_camera_dict["id"] unchanged  # json stores both directly, where npz needs has_name / has_id flags
│   ├── calls Cameras(intrinsics=intrinsics_list, extrinsics=extrinsics_list, names=names, ids=ids, device=device)  # field-validates the batch
│   └── return
├── def _serialize_cameras_npz(cameras: "Cameras") -> Dict[str, Any]
│   ├── # Map a Cameras to the plural batched-array npz payload.
│   ├── impls models, params, extrinsics_list, conventions, names, has_names, ids, has_ids — eight empty accumulators the loop appends to
│   ├── for each camera in cameras
│   │   └── impls appends that camera's model, params (json-encoded), extrinsics, convention, name ("" when absent), and id (-1 when absent) to the batch, each with its has_name / has_id flag  # impls-node-one-step:skip
│   ├── impls stacks each accumulator into its npz array, extrinsics along a new leading axis  # impls-node-one-step:skip
│   └── return
├── def _deserialize_cameras_npz(payload: Dict[str, Any], device: torch.device) -> "Cameras"
│   ├── # Map the plural batched-array npz payload to a Cameras.
│   ├── from data.structures.three_d.camera.cameras import Cameras  # inline runtime import; cameras.py imports io.py, so this would cycle at module top
│   ├── def _validate_inputs [local]
│   │   ├── impls assert payload is a dict whose keys are exactly _CAMERA_NPZ_KEYS
│   │   ├── impls assert payload["extrinsics"] is a float32 ndarray batched as [N, 4, 4]
│   │   ├── calls validate_camera_extrinsics(extrinsics)  # batched validation of all views' 4x4 cam2world
│   │   └── for each of the seven per-camera keys
│   │       └── impls assert its array is an ndarray of shape (batch_size,)
│   ├── calls _validate_inputs
│   ├── impls extrinsics = payload["extrinsics"], the batched [N, 4, 4] cam2world array
│   ├── impls batch_size = extrinsics.shape[0]
│   ├── impls model_array, params_array, convention_array, name_array, has_name_array, id_array, has_id_array — the seven per-camera arrays read from payload
│   ├── impls intrinsics_list, extrinsics_list, names, ids — four empty accumulators the loop appends to
│   ├── for each batch index
│   │   ├── impls decodes that index's model, params, extrinsics, convention, name, and id on device, each name and id taken only when its has_name / has_id flag is set  # impls-node-one-step:skip
│   │   ├── calls build_camera_intrinsics(model=model, params=params, device=device)  # validates model + params
│   │   └── calls CameraExtrinsics(extrinsics=torch.as_tensor(extrinsics[index], dtype=torch.float32, device=device), convention=str(convention_array[index].item()), device=device)  # validates convention
│   ├── calls Cameras(intrinsics=intrinsics_list, extrinsics=extrinsics_list, names=names, ids=ids, device=device)  # field-validates the batch
│   └── return
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
│   │   └── return each array in payload indexed at 0  # npz spells it as the same keys, one axis shorter
│   └── assert 0, "Should not reach here."
├── def _resolve_format_from_path(cameras_path: Path) -> str
│   ├── # Resolve a Cameras serialization format from a file path.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras_path is a Path
│   │   └── impls assert cameras_path has a non-empty suffix
│   ├── calls _validate_inputs
│   └── return _normalize_format(format=cameras_path.suffix)
└── def _normalize_format(format: str) -> str
    ├── # Normalize a path suffix or format name to a supported serialization format.
    ├── impls format = format.strip()
    ├── impls asserts the stripped format is non-empty
    ├── if format.startswith(".")
    │   └── impls format = format[1:]
    ├── impls asserts format is in _CAMERA_SERIALIZATION_FORMATS
    └── return format
```
