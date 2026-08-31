# Camera Extrinsics Code Structure

## 1. Code structure trees

`data/structures/three_d/camera/extrinsics/validation.py`

```text
validation.py
├── from typing import Any, List, Union
├── import numpy as np
├── import torch
├── from utils.ops.materialize_tensor import materialize_tensor
├── _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS = 32  # orthogonality/determinant residual floor of the float SVD-projection, in machine-epsilon units; the eps-scaling is derived, the O(1) prefactor is the empirical LAPACK SVD/det floor (measured worst <= 11 over the reference poses + 53k synthetic rotations; set to 32 for margin, still orders of magnitude below any genuinely non-orthogonal rotation)
├── def validate_camera_extrinsics_attributes(extrinsics: Any, extr_convention: Any, device: Any, dtype: Any) -> None
│   ├── # Single-entry validation for CameraExtrinsics.__init__: validate the cam2world input, pose frame, device target, and dtype target.
│   ├── calls validate_camera_extrinsics
│   ├── calls validate_extr_convention
│   ├── impls asserts device is a str or torch.device
│   ├── impls asserts dtype is a floating torch dtype
│   └── return
├── def validate_camera_extrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]]
│   ├── # Dispatch camera-extrinsics validation on the input representation.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_camera_extrinsics_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_camera_extrinsics_torch
│   ├── if isinstance(obj, list)
│   │   └── calls _validate_camera_extrinsics_list
│   └── raise TypeError  # obj is neither a numpy array, torch tensor, nor nested numeric list
├── def validate_extr_convention(extr_convention: Any) -> str
│   ├── # Validate a camera-pose convention string against the supported set.
│   ├── impls asserts extr_convention is a str in {standard, opengl, opencv, pytorch3d, arkit}
│   └── return extr_convention
├── def _validate_camera_extrinsics_numpy(obj: np.ndarray) -> np.ndarray
│   ├── # Validate a (..., 4, 4) numpy camera-extrinsics (cam2world) matrix.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (4, 4), dtype in {np.float32, np.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── impls rotation = obj[..., :3, :3]
│   ├── calls _validate_rotation_matrix_numpy(rotation)  # tolerance is selected from rotation.dtype
│   └── return obj
├── def _validate_camera_extrinsics_torch(obj: torch.Tensor) -> torch.Tensor
│   ├── # Validate a (..., 4, 4) torch camera-extrinsics (cam2world) matrix.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (4, 4), dtype in {torch.float32, torch.float64}
│   ├── impls asserts last row exactly [0, 0, 0, 1] (atol=0, rtol=0)
│   ├── impls rotation = obj[..., :3, :3]
│   ├── calls _validate_rotation_matrix_torch(rotation)  # tolerance is selected from rotation.dtype
│   └── return obj
├── def _validate_camera_extrinsics_list(obj: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]
│   ├── # Validate a (4, 4) nested-list camera-extrinsics (cam2world) matrix.
│   ├── impls asserts obj is a length-4 list of length-4 numeric rows
│   ├── impls asserts last row exactly [0, 0, 0, 1]
│   ├── impls rotation = [row[:3] for row in obj[:3]]
│   ├── calls _validate_rotation_matrix_list(rotation)
│   └── return obj
├── def validate_rotation_matrix(obj: Any) -> Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]]
│   ├── # Dispatch rotation-matrix validation on the input representation.
│   ├── if isinstance(obj, np.ndarray)
│   │   └── calls _validate_rotation_matrix_numpy
│   ├── if isinstance(obj, torch.Tensor)
│   │   └── calls _validate_rotation_matrix_torch
│   ├── if isinstance(obj, list)
│   │   └── calls _validate_rotation_matrix_list
│   └── raise TypeError  # obj is neither a numpy array, torch tensor, nor nested numeric list
├── def _validate_rotation_matrix_numpy(obj: np.ndarray) -> np.ndarray
│   ├── # Validate a (..., 3, 3) numpy rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts ndarray, ndim >= 2, last two dims (3, 3), dtype in {np.float32, np.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float64).eps)
│   ├── if obj.dtype == np.float32
│   │   ├── calls _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float32)
│   │   └── return
│   ├── if obj.dtype == np.float64
│   │   ├── calls _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float64)
│   │   └── return
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_torch(obj: torch.Tensor) -> torch.Tensor
│   ├── # Validate a (..., 3, 3) torch rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (3, 3), dtype in {torch.float32, torch.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float64).eps)
│   ├── if obj.dtype == torch.float32
│   │   ├── calls _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float32)
│   │   └── return
│   ├── if obj.dtype == torch.float64
│   │   ├── calls _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float64)
│   │   └── return
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_list(obj: List[List[Union[int, float]]]) -> List[List[Union[int, float]]]
│   ├── # Validate a (3, 3) nested-list rotation matrix using the float64 residual threshold.
│   ├── impls asserts obj is a length-3 list of length-3 numeric rows
│   ├── impls computes RR^T residual from the numeric entries
│   ├── impls computes |det(R) - 1| residual from the numeric entries
│   ├── impls asserts both residuals are within _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float64).eps)
│   └── return obj
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

`data/structures/three_d/camera/extrinsics/camera_extrinsics.py`

```text
camera_extrinsics.py
├── from typing import List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.extrinsics.conventions import transform_extr_convention
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics_attributes, validate_extr_convention, validate_rotation_matrix
├── _ORTHOGONALITY_REPAIR_ATOL = 1.0e-05  # dtype-independent input-quality guard: max RR^T-vs-I / determinant residual a raw rotation may carry and still be trusted as SVD-repairable
├── class CameraExtrinsics
│   ├── # A camera's pose: the 4x4 camera-to-world matrix together with the pose frame it is expressed in, so a pose is never read without its frame.
│   ├── def __init__(self, extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], extr_convention: str, device: Union[str, torch.device] = "cpu", dtype: torch.dtype = torch.float32) -> None
│   │   ├── # Construct a CameraExtrinsics from an array-like 4x4 cam2world matrix and the pose frame it is expressed in.
│   │   ├── calls validate_camera_extrinsics_attributes(extrinsics=extrinsics, extr_convention=extr_convention, device=device, dtype=dtype)
│   │   ├── def _normalize_inputs(extrinsics: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], device: Union[str, torch.device], dtype: torch.dtype) -> Tuple[torch.Tensor, torch.device] [local]
│   │   │   ├── impls extrinsics = torch.as_tensor(extrinsics, device=device, dtype=dtype)
│   │   │   ├── impls device = the placed tensor's own device, so an index-free target resolves to the index the tensor actually landed on
│   │   │   └── return extrinsics, device
│   │   ├── calls _normalize_inputs(extrinsics=extrinsics, device=device, dtype=dtype)
│   │   ├── impls extrinsics, device = the returned values from _normalize_inputs
│   │   ├── impls self._extrinsics = extrinsics
│   │   ├── impls self._extr_convention = extr_convention
│   │   ├── impls self._device = device
│   │   └── impls self._dtype = dtype
│   ├── def extrinsics(self) -> torch.Tensor  # @property
│   │   ├── # The 4x4 camera-to-world extrinsics tensor.
│   │   └── return self._extrinsics
│   ├── def extr_convention(self) -> str  # @property
│   │   ├── # The pose frame this cam2world matrix is expressed in (standard / opengl / opencv / pytorch3d / arkit).
│   │   └── return self._extr_convention
│   ├── def device(self) -> torch.device  # @property
│   │   ├── # The device the 4x4 matrix sits on, which to() compares its argument against before deciding a move is needed.
│   │   └── return self._device
│   ├── def dtype(self) -> torch.dtype  # @property
│   │   ├── # The dtype of the extrinsics tensor.
│   │   └── return self._dtype
│   ├── def w2c(self) -> torch.Tensor  # @property
│   │   ├── # The world-to-camera matrix (inverse of extrinsics).
│   │   ├── impls w2c = the matrix inverse of self._extrinsics
│   │   └── return w2c
│   ├── def center(self) -> torch.Tensor  # @property
│   │   ├── # The camera center extrinsics[:3, 3].
│   │   └── return self._extrinsics[:3, 3]
│   ├── def right(self) -> torch.Tensor  # @property
│   │   ├── # The extr_convention-dispatched physical right axis.
│   │   ├── impls select the right axis per extr_convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def forward(self) -> torch.Tensor  # @property
│   │   ├── # The extr_convention-dispatched physical forward axis.
│   │   ├── impls select the forward axis per extr_convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def up(self) -> torch.Tensor  # @property
│   │   ├── # The extr_convention-dispatched physical up axis.
│   │   ├── impls select the up axis per extr_convention
│   │   └── impls assert the selected axis has unit norm
│   ├── def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False, extr_convention: Optional[str] = None) -> "CameraExtrinsics"
│   │   ├── # Return this CameraExtrinsics with Tensor.to-style placement / copy semantics plus optional pose-frame conversion.
│   │   ├── def _validate_inputs [local]
│   │   │   ├── impls assert device is None, str, or torch.device
│   │   │   ├── impls assert dtype is None or a floating torch dtype
│   │   │   ├── impls assert non_blocking is a bool
│   │   │   ├── impls assert copy is a bool
│   │   │   ├── impls assert extr_convention is None or a str
│   │   │   └── if extr_convention is not None
│   │   │       └── calls validate_extr_convention(extr_convention)
│   │   ├── calls _validate_inputs
│   │   ├── def _normalize_inputs [local]
│   │   │   ├── impls device = torch.device(device) if device is not None else self._device
│   │   │   ├── impls dtype = dtype if dtype is not None else self._dtype
│   │   │   ├── impls extr_convention = extr_convention if extr_convention is not None else self._extr_convention
│   │   │   └── return device, dtype, extr_convention
│   │   ├── calls _normalize_inputs(device=device, dtype=dtype, extr_convention=extr_convention)
│   │   ├── impls device, dtype, extr_convention = the returned values from _normalize_inputs
│   │   ├── if device == self._device and dtype == self._dtype and extr_convention == self._extr_convention and copy is False
│   │   │   └── return self
│   │   ├── if extr_convention != self._extr_convention
│   │   │   └── calls transform_extr_convention(camera_extrinsics=self, target_extr_convention=extr_convention)  # -> extrinsics
│   │   ├── else
│   │   │   └── impls extrinsics = self._extrinsics
│   │   ├── impls extrinsics = extrinsics.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
│   │   ├── impls extrinsics = CameraExtrinsics(...)
│   │   └── return extrinsics
│   └── def transform_extrinsics(self, scale: Union[int, float, np.ndarray, torch.Tensor], rotation: Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]], translation: Union[np.ndarray, torch.Tensor, Tuple[Union[int, float], Union[int, float], Union[int, float]], List[Union[int, float]]]) -> "CameraExtrinsics"
│       ├── # Return this CameraExtrinsics under array-like scale, rotation, and translation inputs of its cam2world pose.
│       ├── def _validate_inputs [local]
│       │   ├── calls validate_rotation_matrix(rotation)
│       │   └── impls assert translation is a length-3 numeric array-like or a torch Tensor with shape (3,)
│       ├── calls _validate_inputs
│       ├── def _normalize_inputs [local]
│       │   ├── impls scale = torch.as_tensor(scale, device=self.device, dtype=self.dtype)
│       │   ├── impls asserts scale.shape == ()
│       │   ├── impls asserts scale.device == self.device
│       │   ├── impls asserts scale.dtype == self.dtype
│       │   ├── impls rotation = torch.as_tensor(rotation, device=self.device, dtype=self.dtype)
│       │   ├── impls asserts rotation.shape == (3, 3)
│       │   ├── impls asserts rotation.device == self.device
│       │   ├── impls asserts rotation.dtype == self.dtype
│       │   ├── impls translation = torch.as_tensor(translation, device=self.device, dtype=self.dtype)
│       │   ├── impls asserts translation.shape == (3,)
│       │   ├── impls asserts translation.device == self.device
│       │   ├── impls asserts translation.dtype == self.dtype
│       │   └── return scale, rotation, translation
│       ├── calls _normalize_inputs
│       ├── impls scale, rotation, translation = the returned values from _normalize_inputs
│       ├── impls composes the new cam2world rotation/translation from scale, rotation, translation
│       ├── calls _stabilize_rotation_matrix
│       ├── impls extrinsics = CameraExtrinsics(...)  # re-validates via validate_camera_extrinsics_attributes
│       └── return extrinsics
└── def _stabilize_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor
    ├── # Project a near-orthogonal (3, 3) rotation onto the nearest proper rotation, in the dtype it received.
    ├── impls asserts rotation.dtype is torch.float32 or torch.float64
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
