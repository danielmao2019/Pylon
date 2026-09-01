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

`data/structures/three_d/camera/intrinsics/validation.py`

```text
validation.py
├── from typing import Any, Dict
├── import torch
├── def validate_camera_intrinsics_attributes(model: str, intr_convention: Any, params: Any, device: Any, dtype: Any) -> None
│   ├── # Single-entry validation for CameraIntrinsics.__init__: validate the camera model, image-plane convention, tensor named params, and optional placement request.
│   ├── calls validate_camera_model(model=model)
│   ├── calls validate_intr_convention(intr_convention=intr_convention)
│   ├── calls validate_camera_intrinsics_params(model=model, intr_convention=intr_convention, params=params)  # the frame goes in ahead of the params, what they mean together depending on it
│   ├── impls asserts device is None or a valid torch device spec (str or torch.device)
│   ├── impls asserts dtype is None or a floating torch dtype
│   └── return
├── def validate_camera_model(model: Any) -> str
│   ├── # Validate a camera-model string against the supported set.
│   ├── impls asserts model is a str in {simple_pinhole, pinhole, ortho}
│   └── return model
├── def validate_intr_convention(intr_convention: Any) -> str
│   ├── # Validate an image-plane convention string against the supported set, standard being the pixel raster frame the other three convert through.
│   ├── impls asserts intr_convention is a str in {standard, opengl, pytorch3d, vulkan}
│   └── return intr_convention
├── def validate_camera_intrinsics_params(model: str, intr_convention: str, params: Any) -> Dict[str, torch.Tensor]
│   ├── # Validate the named tensor intrinsics params: the resolution keys every model carries, the projection keys that model's own dispatch owns, and the invariants holding only across those keys together.
│   ├── impls assert params carries h and w, both positive scalar tensors  # impls-node-one-step:skip; the resolution, named the way every resolution in this repo is ordered: h first
│   ├── def _validate_projection_params() -> Dict[str, torch.Tensor] [local]
│   │   ├── # Dispatches the projection keys onto the model that owns them, every model being a structurally equivalent sibling here.
│   │   ├── if model == "simple_pinhole"
│   │   │   ├── calls _validate_camera_intrinsics_params_simple_pinhole(params=params)
│   │   │   └── return params
│   │   ├── if model == "pinhole"
│   │   │   ├── calls _validate_camera_intrinsics_params_pinhole(params=params)
│   │   │   └── return params
│   │   ├── if model == "ortho"
│   │   │   ├── calls _validate_camera_intrinsics_params_ortho(params=params)
│   │   │   └── return params
│   │   └── assert 0, "Should not reach here."
│   ├── calls _validate_projection_params
│   ├── calls validate_camera_intrinsics_invariants(model=model, intr_convention=intr_convention, params=params)
│   └── return params
├── def _validate_camera_intrinsics_params_simple_pinhole(params: Any) -> Dict[str, torch.Tensor]
│   ├── # Validate simple_pinhole params: a single shared focal length f plus the principal point cx / cy.
│   ├── impls asserts every param is a scalar torch.Tensor
│   ├── impls asserts params is a Dict[str, torch.Tensor] with exactly keys {f, cx, cy, h, w}
│   ├── impls asserts f > 0 and cx and cy are finite  # impls-node-one-step:skip; where on the image the principal point may fall is the frame's to say
│   └── return params
├── def _validate_camera_intrinsics_params_pinhole(params: Any) -> Dict[str, torch.Tensor]
│   ├── # Validate pinhole params: independent focal lengths fx / fy plus the principal point cx / cy.
│   ├── impls asserts every param is a scalar torch.Tensor
│   ├── impls asserts params is a Dict[str, torch.Tensor] with exactly keys {fx, fy, cx, cy, h, w}
│   ├── impls asserts fx > 0 and fy > 0 and cx and cy are finite  # impls-node-one-step:skip
│   └── return params
├── def _validate_camera_intrinsics_params_ortho(params: Any) -> Dict[str, torch.Tensor]
│   ├── # Validate ortho (weak-perspective) params: focal scales fx / fy plus the principal-point offset cx / cy.
│   ├── impls asserts every param is a scalar torch.Tensor
│   ├── impls asserts params is a Dict[str, torch.Tensor] with exactly keys {fx, fy, cx, cy, h, w}
│   ├── impls asserts fx > 0 and fy > 0 and cx and cy are finite  # impls-node-one-step:skip
│   └── return params
├── def validate_camera_intrinsics_invariants(model: str, intr_convention: str, params: Dict[str, torch.Tensor]) -> None
│   ├── # Validate what the params state only together, the resolution having joined the dict the principal point and the focal already live in and formed a pair with each.
│   ├── calls _validate_principal_point_within_image(model=model, intr_convention=intr_convention, params=params)
│   ├── calls _validate_model_is_representable_in_frame(model=model, intr_convention=intr_convention, params=params)
│   └── return
├── def _validate_principal_point_within_image(model: str, intr_convention: str, params: Dict[str, torch.Tensor]) -> None
│   ├── # Bounds a perspective camera's principal point the way the frame it is stated in measures it.
│   ├── if model == "ortho"
│   │   └── return  # a weak-perspective cx / cy is where the world origin lands rather than where an axis pierces, and a fit drives that off the frame while the camera stays valid
│   ├── if intr_convention == "standard"
│   │   ├── impls assert cx in [0, w] and cy in [0, h]  # impls-node-one-step:skip; the pixel frame running corner to corner
│   │   └── return
│   ├── if intr_convention in {"opengl", "vulkan"}
│   │   ├── impls assert cx in [-1, +1] and cy in [-1, +1]  # impls-node-one-step:skip; each axis normalized by its own side, so both bounds are the same
│   │   └── return
│   ├── if intr_convention == "pytorch3d"
│   │   ├── impls assert abs(cx) <= w / min(h, w) and abs(cy) <= h / min(h, w)  # impls-node-one-step:skip; the shorter side alone reaches 1, so the longer axis's bound is the larger
│   │   └── return
│   └── assert 0, "Should not reach here."
└── def _validate_model_is_representable_in_frame(model: str, intr_convention: str, params: Dict[str, torch.Tensor]) -> None
    ├── # A model states as many focal params as it has axes to scale independently, so a frame that scales the two axes differently can hold only the models carrying two of them.
    ├── if model == "simple_pinhole" and intr_convention in {"opengl", "vulkan"}
    │   └── impls assert h == w  # these frames normalize each axis by its own side, and one shared f cannot carry two different units, so a non-square image has no simple_pinhole in them
    └── return
```

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
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float32)
│   ├── if obj.dtype == np.float64
│   │   └── return _validate_rotation_matrix_numpy_against_threshold(obj, threshold=atol_float64)
│   └── assert 0, "should not reach here."
├── def _validate_rotation_matrix_torch(obj: torch.Tensor) -> torch.Tensor
│   ├── # Validate a (..., 3, 3) torch rotation matrix; dispatch the tolerance on dtype.
│   ├── impls asserts Tensor, ndim >= 2, last two dims (3, 3), dtype in {torch.float32, torch.float64}
│   ├── impls atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float32).eps)
│   ├── impls atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(torch.finfo(torch.float64).eps)
│   ├── if obj.dtype == torch.float32
│   │   └── return _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float32)
│   ├── if obj.dtype == torch.float64
│   │   └── return _validate_rotation_matrix_torch_against_threshold(obj, threshold=atol_float64)
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

`data/structures/three_d/camera/validation.py`

```text
validation.py
├── from typing import TYPE_CHECKING, List, Optional, Union
├── import torch
├── if TYPE_CHECKING  # annotation-only imports; the runtime type checks import the two classes inline (no cycle, but the top-level refs stay annotation-only)
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

`data/structures/three_d/camera/intrinsics/scaling.py`

```text
scaling.py
├── from typing import Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── def rescale_intr_params(params: Dict[str, Union[int, float]], model: str, unit_x: float, unit_y: float) -> Dict[str, Union[int, float]]
│   ├── # Restates params measured in the image-plane unit; cx / cy are coordinates for perspective models and weak-perspective offsets for ortho.
│   ├── impls params = a copy of params
│   ├── impls cx = unit_x * cx
│   ├── impls cy = unit_y * cy
│   ├── def _rescale_focal(params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]] [local]
│   │   ├── # Scales whichever focal params the model carries, the one place the camera models differ under a rescale.
│   │   ├── if model == "simple_pinhole"
│   │   │   ├── impls assert unit_x == unit_y  # one shared f cannot carry two different axis scales, and a pair that disagrees is a pinhole rather than this model
│   │   │   ├── impls f = unit_x * f in a copy of params
│   │   │   └── return params
│   │   ├── if model in {"pinhole", "ortho"}
│   │   │   ├── impls fx, fy = unit_x * fx, unit_y * fy in a copy of params  # the two models carry the same focal params and take the same rule, a focal being a pixels-per-camera-unit ratio either way
│   │   │   └── return params
│   │   └── raise NotImplementedError  # a camera model whose focal params no rescale here has a rule for yet
│   ├── calls _rescale_focal
│   └── return  # params, in the target unit, h and w as they came in
└── def resolve_target_resolution(params: Dict[str, Union[int, float]], resolution: Optional[Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]] = None, scale: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], np.ndarray, torch.Tensor]] = None) -> Tuple[int, int]
    ├── # Resolves the two ways a caller names a target resolution — the size itself, or a factor on the size the params already carry — into the single form a rescale reads.
    ├── def _validate_inputs [local]
    │   ├── impls assert exactly one of resolution and scale is given  # impls-node-one-step:skip; a target resolution and a factor are two ways to name the same thing, and giving both leaves unstated which one wins
    │   ├── if resolution is not None
    │   │   └── impls assert resolution is a positive int or a length-2 array-like of positive integer-valued entries
    │   └── if scale is not None
    │       └── impls assert scale is a positive number, or a length-2 array-like pair of positive numbers
    ├── calls _validate_inputs
    ├── def _normalize_inputs [local]
    │   ├── if resolution is not None
    │   │   ├── if resolution is a single int
    │   │   │   └── impls resolution = (resolution, resolution)
    │   │   └── if resolution is a length-2 array-like
    │   │       └── impls resolution = (int(resolution[0]), int(resolution[1]))
    │   ├── if scale is not None
    │   │   ├── if scale is a single number
    │   │   │   └── impls scale = (scale, scale)  # one factor names the same one on both axes, in the (sx, sy) form the pair case already arrives in
    │   │   └── if scale is a length-2 array-like pair
    │   │       └── impls scale = (scale[0], scale[1])
    │   └── return resolution, scale
    ├── calls _normalize_inputs
    ├── impls resolution, scale = the returned values from _normalize_inputs
    ├── if resolution is not None
    │   └── return resolution
    ├── if scale is not None
    │   ├── impls h, w = round(the params' own h * scale[1]), round(the params' own w * scale[0])
    │   ├── impls assert both sides came out positive  # a factor small enough to round a side to zero names no image
    │   └── return h, w
    └── assert 0, "Should not reach here."
```

`data/structures/three_d/camera/intrinsics/conventions.py`

```text
conventions.py
├── from typing import Dict, Tuple, Union
├── from data.structures.three_d.camera.intrinsics.scaling import rescale_intr_params
├── def transform_intr_convention(params: Dict[str, Union[int, float]], model: str, source_intr_convention: str, target_intr_convention: str) -> Dict[str, Union[int, float]]
│   ├── # Restates one camera model's named params from the image-plane frame they were stated in into another, routed through the standard frame so each frame brings its own two helpers rather than one against every frame already here.
│   ├── if source_intr_convention == target_intr_convention
│   │   └── return params
│   ├── def _to_standard(params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]] [local]
│   │   ├── # Dispatches the source frame onto its own inbound spoke, the standard frame being already there.
│   │   ├── if source_intr_convention == "standard"
│   │   │   └── return params
│   │   ├── if source_intr_convention == "opengl"
│   │   │   ├── calls _opengl_to_standard(params=params, model=model)  # -> params, on the standard frame
│   │   │   └── return params
│   │   ├── if source_intr_convention == "pytorch3d"
│   │   │   ├── calls _pytorch3d_to_standard(params=params, model=model)  # -> params, on the standard frame
│   │   │   └── return params
│   │   ├── if source_intr_convention == "vulkan"
│   │   │   ├── calls _vulkan_to_standard(params=params, model=model)  # -> params, on the standard frame
│   │   │   └── return params
│   │   └── assert 0, "Should not reach here."
│   ├── calls _to_standard
│   ├── def _from_standard(params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]] [local]
│   │   ├── # Dispatches the target frame onto its own outbound spoke, the standard frame needing none.
│   │   ├── if target_intr_convention == "standard"
│   │   │   └── return params
│   │   ├── if target_intr_convention == "opengl"
│   │   │   ├── calls _standard_to_opengl(params=params, model=model)  # -> params, on the opengl frame
│   │   │   └── return params
│   │   ├── if target_intr_convention == "pytorch3d"
│   │   │   ├── calls _standard_to_pytorch3d(params=params, model=model)  # -> params, on the pytorch3d frame
│   │   │   └── return params
│   │   ├── if target_intr_convention == "vulkan"
│   │   │   ├── calls _standard_to_vulkan(params=params, model=model)  # -> params, on the vulkan frame
│   │   │   └── return params
│   │   └── assert 0, "Should not reach here."
│   ├── calls _from_standard
│   └── return  # params, restated on target_intr_convention
├── def _opengl_to_standard(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # The inbound half of the same frame, the three steps run in reverse so a round trip returns what it started as.
│   ├── impls unit_x, unit_y = w / 2, h / 2
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit_x, unit_y=unit_y)
│   ├── calls _reverse_axes(params=params, axes=("y",))
│   ├── calls _uncentre_principal_point(params=params)
│   └── return  # params, on the standard frame
├── def _pytorch3d_to_standard(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # The inbound half of the same frame, the three steps run in reverse.
│   ├── impls unit = min(h, w) / 2
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit, unit_y=unit)
│   ├── calls _reverse_axes(params=params, axes=("x", "y"))
│   ├── calls _uncentre_principal_point(params=params)
│   └── return  # params, on the standard frame
├── def _vulkan_to_standard(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # The inbound half of the same frame, the two steps run in reverse.
│   ├── impls unit_x, unit_y = w / 2, h / 2
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit_x, unit_y=unit_y)
│   ├── calls _uncentre_principal_point(params=params)
│   └── return  # params, on the standard frame
├── def _standard_to_opengl(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # Restates pixel params on OpenGL's device frame, whose origin is the image's centre, whose x runs with standard's toward the right edge and whose y runs against it toward the top, each axis spanning its own side.
│   ├── impls unit_x, unit_y = 2 / w, 2 / h              # each axis spans [-1, 1] across its own side
│   ├── calls _centre_principal_point(params=params)     # -> params, off the top-left corner onto the image's centre
│   ├── calls _reverse_axes(params=params, axes=("y",))  # standard's y runs toward the bottom edge and OpenGL's toward the top
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit_x, unit_y=unit_y)
│   └── return  # params, on the opengl frame
├── def _standard_to_pytorch3d(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # Restates pixel params on PyTorch3D's device frame, whose origin is the image's centre, whose x runs toward the left edge and y toward the top, and whose shorter side alone spans [-1, 1].
│   ├── impls unit = 2 / min(h, w)  # the one frame here normalizing both axes by a single side, letting the longer one reach past $1$
│   ├── calls _centre_principal_point(params=params)
│   ├── calls _reverse_axes(params=params, axes=("x", "y"))  # standard runs x toward the right edge and y toward the bottom, PyTorch3D x toward the left and y toward the top
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit, unit_y=unit)
│   └── return  # params, on the pytorch3d frame
├── def _standard_to_vulkan(params: Dict[str, Union[int, float]], model: str) -> Dict[str, Union[int, float]]
│   ├── # Restates pixel params on Vulkan's device frame, which agrees with standard on both axis directions and differs from OpenGL's in exactly that.
│   ├── impls unit_x, unit_y = 2 / w, 2 / h
│   ├── calls _centre_principal_point(params=params)
│   ├── calls rescale_intr_params(params=params, model=model, unit_x=unit_x, unit_y=unit_y)
│   └── return  # params, on the vulkan frame
├── def _centre_principal_point(params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]
│   ├── # Moves the principal point off the image's top-left corner onto its centre, the separation no axis reversal can carry and the largest of the three.
│   ├── impls cx, cy = cx - w / 2, cy - h / 2 in a copy of params  # every model states its principal point and its size as the same four params
│   └── return  # params, on a centred origin
├── def _reverse_axes(params: Dict[str, Union[int, float]], axes: Tuple[str, ...]) -> Dict[str, Union[int, float]]
│   ├── # Reverses the named image axes, which reaches the principal point alone.
│   ├── for each named axis
│   │   └── impls negate that axis's principal-point param in a copy of params  # the offset is stated on the output side alone, so a reversal reaches it unopposed
│   └── return  # params, on the reversed axes
└── def _uncentre_principal_point(params: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]
    ├── # Moves the principal point back off the image's centre onto its top-left corner.
    ├── impls cx, cy = cx + w / 2, cy + h / 2 in a copy of params
    └── return  # params, on a corner origin
```

`data/structures/three_d/camera/intrinsics/camera_intrinsics.py`

```text
camera_intrinsics.py
├── from abc import ABC
├── from typing import ClassVar, Dict, List, Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.intrinsics.conventions import transform_intr_convention
├── from data.structures.three_d.camera.intrinsics.scaling import resolve_target_resolution
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_attributes, validate_intr_convention
├── class CameraIntrinsics(ABC)   [abstract]
│   ├── # Abstract base for a camera's intrinsics: owns tensor named params, image-plane frame, device, and dtype, with each subclass being one camera model.
│   ├── MODEL: ClassVar[str]  # each concrete subclass sets its camera-model identifier (simple_pinhole / pinhole / ortho)
│   ├── def __init__(self, params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]], intr_convention: str, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> None
│   │   ├── # Construct a CameraIntrinsics from tensor-compatible named scalar params and the image-plane frame they are stated in.
│   │   ├── calls validate_camera_intrinsics_attributes(model=type(self).MODEL, intr_convention=intr_convention, params=params, device=device, dtype=dtype)
│   │   ├── def _normalize_inputs [local]
│   │   │   ├── impls params = each value materialized as a scalar torch.Tensor without applying the placement request
│   │   │   ├── impls asserts every normalized param is a scalar torch.Tensor
│   │   │   ├── impls asserts every normalized param shares one device
│   │   │   ├── impls asserts every normalized param shares one dtype
│   │   │   └── return params
│   │   ├── calls _normalize_inputs(params=params)
│   │   ├── impls self._params = params
│   │   ├── impls self._intr_convention = intr_convention
│   │   ├── impls self._device = the common device of self._params values
│   │   ├── impls self._dtype = the common dtype of self._params values
│   │   └── if device is not None or dtype is not None
│   │       ├── calls self.to(device=device, dtype=dtype)
│   │       └── impls replace this object's params / intr_convention / device / dtype with the returned object's state
│   ├── def model(self) -> str  # @property
│   │   ├── # The camera-model identifier type(self).MODEL.
│   │   ├── impls model = type(self).MODEL
│   │   └── return model
│   ├── def params(self) -> Dict[str, torch.Tensor]  # @property
│   │   ├── # The model's named scalar tensor parameters.
│   │   └── return self._params
│   ├── def intr_convention(self) -> str  # @property
│   │   ├── # The image-plane frame these params are stated in (standard / opengl / pytorch3d / vulkan), without which a principal point names no location.
│   │   └── return self._intr_convention
│   ├── def device(self) -> torch.device  # @property
│   │   ├── # The device the params tensors sit on, which to() compares its argument against before deciding a move is needed.
│   │   └── return self._device
│   ├── def dtype(self) -> torch.dtype  # @property
│   │   ├── # The dtype shared by the intrinsics params.
│   │   └── return self._dtype
│   ├── def cx(self) -> torch.Tensor  # @property
│   │   ├── # The horizontal principal-point coordinate params["cx"].
│   │   └── return self._params["cx"]
│   ├── def cy(self) -> torch.Tensor  # @property
│   │   ├── # The vertical principal-point coordinate params["cy"].
│   │   └── return self._params["cy"]
│   ├── def resolution(self) -> Tuple[torch.Tensor, torch.Tensor]  # @property
│   │   ├── # The resolution tensor pair these params are stated against, read off h and w because a principal point in pixels names a location only against them.
│   │   └── return self._params["h"], self._params["w"]
│   ├── def fx(self) -> torch.Tensor  # @property [abstract]
│   │   └── # Abstract: the horizontal focal length / scale, whose params key differs per model.
│   ├── def fy(self) -> torch.Tensor  # @property [abstract]
│   │   └── # Abstract: the vertical focal length / scale, whose params key differs per model.
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [abstract]
│   │   └── # Abstract: map camera-space 3D points [..., 3] to 2D image points [..., 2] under this model.
│   ├── def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, non_blocking: bool = False, copy: bool = False, intr_convention: Optional[str] = None) -> "CameraIntrinsics"
│   │   ├── # Return this CameraIntrinsics with Tensor.to-style placement / copy semantics plus optional image-plane frame conversion.
│   │   ├── def _validate_inputs [local]
│   │   │   └── if intr_convention is not None
│   │   │       └── calls validate_intr_convention
│   │   ├── calls _validate_inputs
│   │   ├── impls params = self._params
│   │   ├── if intr_convention is not None and intr_convention != self._intr_convention
│   │   │   └── calls transform_intr_convention(params=params, model=type(self).MODEL, source_intr_convention=self._intr_convention, target_intr_convention=intr_convention)  # -> params, restated on the target frame; the size that change is measured against is two of those params
│   │   ├── impls params = each param moved with torch.Tensor.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)  # impls-node-one-step:skip
│   │   ├── if device and dtype match self, intr_convention is unchanged, and copy is False
│   │   │   └── return self
│   │   ├── impls intrinsics = type(self)(params=params, intr_convention=intr_convention or self._intr_convention)
│   │   └── return intrinsics
│   ├── def transform_intrinsics(self, transform: torch.Tensor, resolution: Tuple[int, int]) -> "CameraIntrinsics"
│   │   ├── # Return this CameraIntrinsics restated onto another image by a pixel-frame affine, the raster that image is named alongside it because a 3x3 carries no size of its own.
│   │   ├── def _validate_inputs [local]
│   │   │   ├── impls assert transform is a (3, 3) float32 whose last row is [0, 0, 1]
│   │   │   └── impls assert resolution is an (h, w) pair of positive ints
│   │   ├── calls _validate_inputs
│   │   ├── calls transform_intr_convention(params=self._params, model=type(self).MODEL, source_intr_convention=self._intr_convention, target_intr_convention="standard")  # -> params, in pixels; an affine between two rasters composes only with a K stated in them
│   │   ├── impls K = transform @ the [3, 3] assembled from self.fx, self.fy and params' cx, cy  # impls-node-one-step:skip; the per-model accessors, since simple_pinhole states its two focals as one f
│   │   ├── if type(self).MODEL == "simple_pinhole"
│   │   │   └── impls assert K[0][0] == K[1][1]  # one shared f holds one ratio, so an affine scaling the axes apart leaves this model nothing to state the second in
│   │   ├── impls params = this model's own focal and cx / cy params read back off K, with h, w = resolution  # impls-node-one-step:skip
│   │   ├── calls transform_intr_convention(params=params, model=type(self).MODEL, source_intr_convention="standard", target_intr_convention=self._intr_convention)  # -> params, back on the frame this intrinsics states them in
│   │   ├── impls intrinsics = type(self)(params=params, intr_convention=self._intr_convention)
│   │   └── return intrinsics
│   └── def scale_intrinsics(self, resolution: Optional[Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]] = None, scale: Optional[Union[int, float, Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], np.ndarray, torch.Tensor]] = None) -> "CameraIntrinsics"
│       ├── # Return this CameraIntrinsics restated against a different resolution — the diagonal case of an intrinsics transform, so this builds that transform and the one owner applies it.
│       ├── def _validate_inputs [local]
│       │   └── impls assert exactly one of resolution and scale is given  # impls-node-one-step:skip; a target resolution and a factor are two ways to name the same thing, and giving both leaves unstated which one wins
│       ├── calls _validate_inputs
│       ├── def _normalize_inputs [local]
│       │   └── calls resolve_target_resolution(params=self._params, resolution=resolution, scale=scale)  # -> resolution; the two forms a caller names a target resolution in, reduced to the one a transform is built from
│       ├── calls _normalize_inputs
│       ├── impls sx, sy = resolution[1] / self._params["w"], resolution[0] / self._params["h"]  # the size the params are already stated against is two of those params, the one place every model states it
│       ├── impls transform = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]                                # a resize scales both axes about the pixel frame's own origin, its top-left corner, which is what makes it diagonal
│       ├── impls intrinsics = self.transform_intrinsics(transform=transform, resolution=resolution)
│       └── return intrinsics
├── class CameraIntrinsicsSimplePinhole(CameraIntrinsics)
│   ├── # Simple-pinhole intrinsics: a single shared focal length f under a perspective projection.
│   ├── MODEL: ClassVar[str] = "simple_pinhole"
│   ├── def fx(self) -> torch.Tensor  # @property [override]
│   │   ├── # The shared focal length params["f"].
│   │   └── return self._params["f"]
│   ├── def fy(self) -> torch.Tensor  # @property [override]
│   │   ├── # The shared focal length params["f"].
│   │   └── return self._params["f"]
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│   │   ├── # Perspective projection with a single shared focal length.
│   │   ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│   │   ├── impls z = points_camera[..., 2]
│   │   ├── impls in place: out[..., 0] = f * out[..., 0] / z + cx  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   ├── impls in place: out[..., 1] = f * out[..., 1] / z + cy  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
│   └── def fov(self) -> Tuple[torch.Tensor, torch.Tensor]  # @property
│       ├── # The horizontal / vertical field of view in degrees (perspective model only).
│       └── impls computes horizontal and vertical fov tensors in degrees from f, cx, and cy
├── class CameraIntrinsicsPinhole(CameraIntrinsics)
│   ├── # Pinhole intrinsics: independent focal lengths fx / fy under a perspective projection.
│   ├── MODEL: ClassVar[str] = "pinhole"
│   ├── def fx(self) -> torch.Tensor  # @property [override]
│   │   ├── # The horizontal focal length params["fx"].
│   │   └── return self._params["fx"]
│   ├── def fy(self) -> torch.Tensor  # @property [override]
│   │   ├── # The vertical focal length params["fy"].
│   │   └── return self._params["fy"]
│   ├── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│   │   ├── # Perspective projection with independent fx / fy.
│   │   ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│   │   ├── impls z = points_camera[..., 2]
│   │   ├── impls in place: out[..., 0] = fx * out[..., 0] / z + cx  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   ├── impls in place: out[..., 1] = fy * out[..., 1] / z + cy  (div_ / mul_ / add_)  # impls-node-one-step:skip
│   │   └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
│   └── def fov(self) -> Tuple[torch.Tensor, torch.Tensor]  # @property
│       ├── # The horizontal / vertical field of view in degrees (perspective model only).
│       └── impls computes horizontal and vertical fov tensors in degrees from fx, fy, cx, and cy
├── class CameraIntrinsicsOrtho(CameraIntrinsics)
│   ├── # Ortho (weak-perspective) intrinsics: independent focal scales fx / fy with no perspective divide.
│   ├── MODEL: ClassVar[str] = "ortho"
│   ├── def fx(self) -> torch.Tensor  # @property [override]
│   │   ├── # The horizontal focal scale params["fx"].
│   │   └── return self._params["fx"]
│   ├── def fy(self) -> torch.Tensor  # @property [override]
│   │   ├── # The vertical focal scale params["fy"].
│   │   └── return self._params["fy"]
│   └── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│       ├── # Orthographic projection with independent fx / fy scales (no perspective divide).
│       ├── impls out = points_camera[..., :2] when inplace, else a fresh [..., 2] clone of points_camera[..., :2]  # impls-node-one-step:skip
│       ├── impls in place: out[..., 0] = fx * out[..., 0] + cx  (mul_ / add_)                                      # impls-node-one-step:skip
│       ├── impls in place: out[..., 1] = fy * out[..., 1] + cy  (mul_ / add_)                                      # impls-node-one-step:skip
│       └── return  # out, the [..., 2] image points (a view into points_camera when inplace)
└── def build_camera_intrinsics(model: str, params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]], intr_convention: str, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> CameraIntrinsics
    ├── # Build the CameraIntrinsics subclass for a camera-model string (the serialization-boundary factory) by dispatching on the model.
    ├── if model == "simple_pinhole"
    │   ├── impls intrinsics = CameraIntrinsicsSimplePinhole(params=params, intr_convention=intr_convention, device=device, dtype=dtype)
    │   └── return intrinsics
    ├── if model == "pinhole"
    │   ├── impls intrinsics = CameraIntrinsicsPinhole(params=params, intr_convention=intr_convention, device=device, dtype=dtype)
    │   └── return intrinsics
    ├── if model == "ortho"
    │   ├── impls intrinsics = CameraIntrinsicsOrtho(params=params, intr_convention=intr_convention, device=device, dtype=dtype)
    │   └── return intrinsics
    └── assert 0, "Should not reach here."
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
│   │   │   ├── impls device = torch.device(device)
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
│   │   └── return the matrix inverse of self._extrinsics
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
│   │   └── impls convert the numeric scalar to a torch scalar tensor with the requested device and dtype
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
