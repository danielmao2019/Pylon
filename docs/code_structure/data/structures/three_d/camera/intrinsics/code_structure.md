# Camera Intrinsics Code Structure

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
│       ├── impls computes the horizontal fov tensor in degrees from f, cx
│       └── impls computes the vertical fov tensor in degrees from f, cy
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
│       ├── impls computes the horizontal fov tensor in degrees from fx, cx
│       └── impls computes the vertical fov tensor in degrees from fy, cy
├── class CameraIntrinsicsOrtho(CameraIntrinsics)
│   ├── # Ortho (weak-perspective) intrinsics: independent focal scales fx / fy under an orthographic projection.
│   ├── MODEL: ClassVar[str] = "ortho"
│   ├── def fx(self) -> torch.Tensor  # @property [override]
│   │   ├── # The horizontal focal scale params["fx"].
│   │   └── return self._params["fx"]
│   ├── def fy(self) -> torch.Tensor  # @property [override]
│   │   ├── # The vertical focal scale params["fy"].
│   │   └── return self._params["fy"]
│   └── def project(self, points_camera: torch.Tensor, inplace: bool = False) -> torch.Tensor   [override]
│       ├── # Orthographic projection with independent fx / fy scales.
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
