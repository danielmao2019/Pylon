# Camera Intrinsics Tests Structure

## 1. Tests implementation structure

`tests/data/structures/three_d/camera/intrinsics/test_intrinsics.py`

```text
test_intrinsics.py
├── import ast
├── import pytest
├── import torch
├── import warnings
├── from pathlib import Path
├── from typing import Dict, Set, Tuple
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsicsOrtho, CameraIntrinsicsPinhole, CameraIntrinsicsSimplePinhole, build_camera_intrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_attributes, validate_camera_intrinsics_invariants, validate_camera_intrinsics_params, validate_camera_model
├── def test_validate_camera_model_accepts_all_supported
│   ├── # validate_camera_model accepts simple_pinhole, pinhole, and ortho.
│   ├── for each model in {simple_pinhole, pinhole, ortho}
│   │   ├── calls validate_camera_model
│   │   └── impls assert the returned string is the model that was passed in
│   └── return
├── def test_validate_camera_model_rejects_unsupported
│   ├── # validate_camera_model raises on a camera-model string outside the supported set.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_camera_model(model=a string outside the supported set)
│   └── return
├── def test_validate_intrinsics_params_dispatches_per_model_tensor_keys
│   ├── # validate_camera_intrinsics_params enforces each model's named scalar tensor keys beside the h and w every model carries.
│   ├── for each (model, its named parameter keys)
│   │   ├── calls validate_camera_intrinsics_params(model=this model, intr_convention="standard", params=its own key set)
│   │   ├── impls assert the returned params dict equals the accepted one
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_params(model=this model, intr_convention="standard", params=another model's key set)
│   └── return
├── def test_validate_intrinsics_params_rejects_a_params_dict_missing_the_resolution
│   ├── # h and w are two of every model's own params rather than a resolution supplied beside them, so a dict carrying the projection keys alone is rejected ahead of the model's own dispatch.
│   ├── for each model in {simple_pinhole, pinhole, ortho}
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_params(model=this model, intr_convention="standard", params=its projection keys without h and w)
│   └── return
├── def test_the_principal_point_must_lie_on_the_image_in_its_own_frames_extent
│   ├── # A principal point is where the optical axis meets the image, so it lies on the image — and what that bound is depends on the frame, which is why the check reads the two together rather than either alone.
│   ├── calls validate_camera_intrinsics_invariants(model=a supported model, intr_convention=each frame in turn, params=a principal point placed against that frame's own extent)
│   ├── impls assert a standard cx of w and cy of h pass, and either one past its own side fails                            # impls-node-one-step:skip
│   ├── impls assert an opengl or vulkan principal point passes within plus or minus one on both axes and fails outside it  # impls-node-one-step:skip
│   ├── impls assert a pytorch3d principal point may reach past one on the longer side and not on the shorter               # impls-node-one-step:skip
│   ├── impls assert an ortho principal point passes anywhere finite under every frame, its cx and cy naming where the world origin lands rather than where an optical axis pierces  # impls-node-one-step:skip
│   └── return
├── def test_a_centred_principal_point_survives_its_models_own_key_dispatch
│   ├── # Every frame but standard puts the origin at the image's centre, so half of it carries a negative principal point — which the per-model key dispatch must not read as out of range, that bound belonging to the frame alone.
│   ├── for each model in {simple_pinhole, pinhole, ortho}
│   │   ├── calls validate_camera_intrinsics_params(model=this model, intr_convention="opengl", params=its own key set at a negative cx and cy)
│   │   └── impls assert the returned params dict equals the accepted one
│   └── return
├── def test_a_frame_that_scales_the_axes_apart_cannot_hold_a_shared_focal
│   ├── # A model states as many focal params as it has axes to scale independently, so opengl and vulkan, which normalize each axis by its own side, hold a simple_pinhole only on a square image.
│   ├── calls validate_camera_intrinsics_invariants(model=each model in turn, intr_convention=each frame in turn, params=that model's key set at a square and a non-square resolution)
│   ├── impls assert a simple_pinhole with h equal to w passes under opengl and vulkan  # impls-node-one-step:skip
│   ├── impls assert one with h different from w fails under both
│   ├── impls assert it passes under pytorch3d at any h and w, that frame normalizing both axes by the shorter side   # impls-node-one-step:skip
│   ├── impls assert pinhole and ortho pass under every frame at any h and w, each carrying its own two focal params  # impls-node-one-step:skip
│   └── return
├── def test_validate_intrinsics_attributes_checks_model_intr_convention_params_device_dtype
│   ├── # validate_camera_intrinsics_attributes validates the camera model, image-plane frame, tensor params, device, and dtype together.
│   ├── calls validate_camera_intrinsics_attributes(model=a supported model, intr_convention="standard", params=its matching tensor params, device=a valid device, dtype=a floating torch dtype)
│   ├── for each attribute broken in turn (the model, the intr_convention, the params, the device, the dtype)
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_attributes
│   └── return
├── def test_validate_intrinsics_params_rejects_python_scalars
│   ├── # validate_camera_intrinsics_params rejects Python numeric params because live camera state is tensor-only.
│   ├── for each supported model
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_params(model=model, intr_convention="standard", params=matching Python scalar params)
│   └── return
├── def test_build_camera_intrinsics_dispatches_to_model_subclass
│   ├── # build_camera_intrinsics returns the CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole / CameraIntrinsicsOrtho instance for its model string.
│   ├── for each (model, its expected CameraIntrinsics subclass)
│   │   ├── calls build_camera_intrinsics
│   │   ├── impls assert the built instance's type is that subclass
│   │   └── impls assert the built instance's model property equals the model string
│   └── return
├── def test_intrinsics_constructor_applies_requested_device_dtype_through_to
│   ├── # CameraIntrinsics.__init__ delegates requested device / dtype movement to the object's to method.
│   ├── calls build_camera_intrinsics(model=a supported model, params=tensor scalar params, device=a valid device, dtype=a floating torch dtype)
│   ├── impls assert every param tensor has the requested device
│   ├── impls assert every param tensor has the requested dtype
│   ├── impls assert intrinsics.device matches the returned param tensors
│   ├── impls assert intrinsics.dtype matches the returned param tensors
│   └── return
├── def test_intrinsics_to_follows_tensor_to_semantics
│   ├── # CameraIntrinsics.to applies Tensor.to-style device / dtype / copy semantics to every scalar param tensor.
│   ├── calls build_camera_intrinsics(model=a supported model, params=tensor scalar params)
│   ├── calls intrinsics.to(device=a valid device, dtype=a floating torch dtype, copy=True)
│   ├── impls assert every returned param tensor has the requested device
│   ├── impls assert every returned param tensor has the requested dtype
│   ├── impls assert copy=True returns tensors with distinct storage from the source params
│   └── return
├── def test_simple_pinhole_project_applies_perspective_divide
│   ├── # CameraIntrinsicsSimplePinhole.project applies the perspective divide with a single shared focal length.
│   ├── calls CameraIntrinsicsSimplePinhole
│   ├── calls intrinsics.project
│   ├── impls assert the image points equal f * x / z + cx and f * y / z + cy under torch.allclose  # impls-node-one-step:skip
│   └── return
├── def test_pinhole_project_applies_perspective_divide
│   ├── # CameraIntrinsicsPinhole.project applies the perspective divide with independent fx / fy.
│   ├── calls CameraIntrinsicsPinhole
│   ├── calls intrinsics.project
│   ├── impls assert the image points equal fx * x / z + cx and fy * y / z + cy under torch.allclose  # impls-node-one-step:skip
│   └── return
├── def test_ortho_project_skips_perspective_divide
│   ├── # CameraIntrinsicsOrtho.project maps points without the perspective divide.
│   ├── calls CameraIntrinsicsOrtho
│   ├── calls intrinsics.project
│   ├── impls assert the image points equal fx * x + cx and fy * y + cy under torch.allclose  # impls-node-one-step:skip
│   ├── impls assert rescaling the input depth column leaves the image points where they were
│   └── return
├── _REPO_ROOT = Path(__file__).resolve().parents[6]
├── _REPO_SOURCE_ROOTS = tuple of Pylon source-code roots
├── _CAMERA_DEPTH_NAMES = set of camera-space depth names
├── def test_every_camera_consumer_projects_through_the_camera
│   ├── # A camera's image coordinates come from its own project, so a consumer forming them another way carries a second copy of one model's formula.
│   ├── impls owner = path of the intrinsics implementation being tested
│   ├── impls consumers: Dict[str, Tuple[bool, bool]] = {}
│   ├── for each repo-owned source module
│   │   ├── calls warnings.catch_warnings
│   │   └── calls _classify_camera_module
│   ├── impls assert the scan finds at least one camera consumer
│   ├── impls hand_rolled: Set[str] = consumers that still divide by camera depth
│   ├── impls assert hand_rolled is empty
│   └── return
├── def _classify_camera_module(tree: ast.Module) -> Tuple[bool, bool]
│   ├── # Classifies one module by how it reaches image coordinates.
│   ├── for each ast node
│   │   ├── impls track whether the module reads focal or principal-point attributes
│   │   ├── impls track whether it calls an intrinsics project method
│   │   └── impls track whether it divides by camera depth
│   └── return  # whether it projects through the camera, and whether it hand-rolls projection
├── def _is_a_camera_depth(node: ast.expr) -> bool
│   ├── # Decides whether an expression names camera-space depth.
│   ├── if node is a name or attribute
│   │   └── return  # whether the name is one of _CAMERA_DEPTH_NAMES
│   ├── if node is a call
│   │   └── calls _is_a_camera_depth
│   ├── if node is not a subscript
│   │   └── return False  # return-nodes-no-construction:skip
│   ├── impls read the final subscript index
│   └── return  # whether index 2 is read from a points-like expression
├── def test_project_inplace_overwrites_input_and_matches_not_inplace
│   ├── # project(inplace=True) overwrites points_camera cols 0,1 with the image points (matching inplace=False), preserves the depth col 2, and returns a tensor aliasing the input, across all three models.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   ├── calls intrinsics.project(points_camera=a pristine copy of points_camera, inplace=False)
│   │   ├── calls intrinsics.project(points_camera=points_camera itself, inplace=True)
│   │   ├── impls assert points_camera cols 0,1 equal the not-inplace image points
│   │   ├── impls assert points_camera col 2 still holds the pristine copy's depth
│   │   └── impls assert the returned tensor shares storage with points_camera
│   └── return
├── def test_project_not_inplace_preserves_input_and_returns_new_tensor
│   ├── # project(inplace=False) returns a fresh [..., 2] and leaves points_camera unchanged, across all three models.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   ├── calls intrinsics.project(points_camera=points_camera, inplace=False)
│   │   ├── impls assert the result's last dim is 2 and its leading dims match points_camera's  # impls-node-one-step:skip
│   │   ├── impls assert points_camera equals its pristine copy
│   │   └── impls assert the result shares no storage with points_camera
│   └── return
├── def test_project_supports_batched_leading_dims
│   ├── # project handles [..., 3] leading dims: a batched input (inplace and not-inplace) matches projecting its flattened [N, 3] view, across all three models.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   └── for each inplace setting
│   │       ├── calls intrinsics.project(points_camera=a [B, M, 3] batch, inplace=this setting)
│   │       ├── calls intrinsics.project(points_camera=its flattened [B * M, 3] view, inplace=this setting)
│   │       └── impls assert the batched image points reshaped to [B * M, 2] equal the flat ones
│   └── return
├── def test_project_rejects_invalid_inputs
│   ├── # project raises AssertionError on a non-tensor points_camera, a wrong last dim, and a non-bool inplace, across all three models.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   └── for each rejected input (a non-tensor points_camera, a [..., 2] points_camera, a non-bool inplace)
│   │       └── with pytest.raises(AssertionError)
│   │           └── calls intrinsics.project
│   └── return
├── def test_fx_fy_cx_cy_derived_from_params
│   ├── # The per-subclass fx / fy accessors and the base cx / cy accessors are derived from the model params.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   ├── impls assert fx and fy read that model's focal keys (simple_pinhole: both params["f"]; pinhole / ortho: params["fx"] and params["fy"])  # impls-node-one-step:skip
│   │   └── impls assert cx and cy equal params["cx"] and params["cy"]  # impls-node-one-step:skip
│   └── return
├── def test_fov_defined_for_perspective_subclasses_only
│   ├── # CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole expose fov in degrees while CameraIntrinsicsOrtho has no fov method.
│   ├── for each of the simple_pinhole and pinhole models
│   │   ├── calls build_camera_intrinsics
│   │   ├── impls assert fov is a (horizontal, vertical) pair in degrees
│   │   └── impls assert each angle matches the one implied by that model's focal length and principal point  # impls-node-one-step:skip
│   ├── calls CameraIntrinsicsOrtho
│   ├── impls assert hasattr(ortho_intrinsics, "fov") is False
│   └── return
├── def test_transform_intrinsics_restates_the_camera_onto_the_named_raster
│   ├── # An affine between two rasters says how coordinates move but not what image they land on, so the raster is named beside it and becomes the h and w the result carries.
│   ├── calls build_camera_intrinsics
│   ├── calls intrinsics.transform_intrinsics(transform=a hand-built (3, 3) affine, resolution=a raster differing from the intrinsics' own)
│   ├── impls assert the returned h and w params are the named raster's  # impls-node-one-step:skip
│   ├── impls assert its principal point is where that affine sends the original's, both read in pixels
│   └── return
├── def test_transform_intrinsics_returns_the_frame_it_was_given
│   ├── # Applying a transform says nothing about which image-plane frame a caller states its camera in, so the result comes back on the frame it went in on rather than on the pixel frame the composition happens in.
│   ├── for each of the four image-plane frames
│   │   ├── calls build_camera_intrinsics
│   │   ├── calls intrinsics.transform_intrinsics(transform=the identity, resolution=the intrinsics' own)
│   │   ├── impls assert the returned intr_convention is the one it was given
│   │   └── impls assert every param comes back unchanged
│   └── return
├── def test_a_shared_focal_refuses_a_transform_that_scales_the_axes_apart
│   ├── # simple_pinhole states one f for both axes, so an affine whose two diagonal entries differ has nowhere to put the second and aborts rather than picking one.
│   ├── calls build_camera_intrinsics
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.transform_intrinsics(transform=an affine whose two diagonal entries differ, resolution=a raster)
│   └── return
├── def test_a_resize_is_the_diagonal_case_of_a_transform
│   ├── # A resize scales both axes about the pixel frame's own origin, which is a diagonal affine, so the two entries agree rather than each carrying its own rule.
│   ├── calls build_camera_intrinsics
│   ├── calls intrinsics.scale_intrinsics(resolution=a target raster)
│   ├── calls intrinsics.transform_intrinsics(transform=the diagonal built from that target over the intrinsics' own, resolution=that target raster)
│   ├── impls assert the two results are equal param for param
│   └── return
├── def test_scale_intrinsics_scales_focal_and_cx_cy_params
│   ├── # CameraIntrinsics.scale_intrinsics restates focal length(s) and cx / cy params against a different resolution, the size it is currently stated against being two of its own params rather than a second thing the caller supplies.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics
│   │   ├── calls intrinsics.scale_intrinsics(resolution=a target resolution at this intrinsics' own aspect ratio)
│   │   ├── impls assert the focal and cx / cy params scaled by the target over the intrinsics' own h and w params, per axis  # impls-node-one-step:skip
│   │   ├── impls assert the returned intrinsics' h and w params are the target ones                                          # impls-node-one-step:skip
│   │   ├── calls intrinsics.scale_intrinsics(scale=a single factor)
│   │   └── impls assert the focal and cx / cy params scaled by that factor  # impls-node-one-step:skip
│   └── return
├── def test_scale_intrinsics_takes_exactly_one_of_a_target_resolution_and_a_factor
│   ├── # A target resolution and a factor are two ways to name the same resize, so naming both leaves which one wins unstated and naming neither names no resize at all.
│   ├── calls build_camera_intrinsics
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.scale_intrinsics(resolution=a target resolution, scale=a factor)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.scale_intrinsics
│   └── return
├── def test_only_a_model_carrying_two_focal_params_can_be_scaled_apart
│   ├── # A model states as many focal params as it has axes to scale independently, so a resize whose two ratios differ is stated axis by axis on pinhole and ortho and has nowhere to go on simple_pinhole's one shared f.
│   ├── for each of the pinhole and ortho models
│   │   ├── calls build_camera_intrinsics
│   │   ├── calls intrinsics.scale_intrinsics(scale=an (sx, sy) pair whose two factors differ)
│   │   └── impls assert fx and cx scaled by sx, and fy and cy by sy  # impls-node-one-step:skip
│   ├── calls build_camera_intrinsics(model="simple_pinhole", params=its own key set, intr_convention="standard")
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.scale_intrinsics(scale=that same unequal (sx, sy) pair)
│   └── return
├── def test_a_per_axis_normalized_frames_params_do_not_move_with_the_resolution
│   ├── # opengl and vulkan each measure an axis by its own side, so restating one against a different size — of a different aspect ratio included — moves no param and only the size it reports changes.
│   ├── for each of the opengl and vulkan frames
│   │   ├── calls build_camera_intrinsics(model="pinhole", params=its own key set, intr_convention=this frame)  # a shared focal is what a change of aspect ratio has nowhere to put, so the model carrying two is the one this reads
│   │   ├── calls intrinsics.scale_intrinsics(resolution=a target resolution of a different aspect ratio)
│   │   ├── impls assert every param but h and w comes back equal to the one it was given  # impls-node-one-step:skip
│   │   └── impls assert its h and w params are the target ones                            # impls-node-one-step:skip
│   └── return
├── def test_the_pytorch3d_frames_params_move_when_the_aspect_ratio_does
│   ├── # pytorch3d normalizes both axes by the shorter side alone, so its params hold under a resize that keeps the aspect ratio and are restated by one that does not.
│   ├── calls build_camera_intrinsics(model="pinhole", params=its own tensor key set, intr_convention="pytorch3d")
│   ├── calls intrinsics.scale_intrinsics(scale=a single tensor factor)
│   ├── impls assert every param but h and w comes back equal to the one it was given  # impls-node-one-step:skip
│   ├── calls intrinsics.scale_intrinsics(resolution=a target tensor resolution of a different aspect ratio)
│   ├── impls assert its principal point comes back restated, equal to the same params carried into pixels, resized there, and carried back  # impls-node-one-step:skip
│   └── return
├── def test_intrinsics_tensor_state_stays_differentiable_through_project
│   ├── # Tensor intrinsics state stays on the autograd path through projection.
│   ├── for each of the three camera models
│   │   ├── calls build_camera_intrinsics(model=model, params=tensor scalar params with requires_grad, intr_convention="standard")
│   │   ├── calls intrinsics.project(points_camera=valid camera-space points)
│   │   ├── impls loss = image_points.sum()
│   │   ├── calls loss.backward
│   │   └── impls assert every source tensor param receives a gradient
│   └── return
└── def test_scale_intrinsics_keeps_tensor_state_differentiable
    ├── # CameraIntrinsics.scale_intrinsics keeps tensor state and tensor scale factors on the autograd path.
    ├── for each of the three camera models
    │   ├── calls build_camera_intrinsics(model=model, params=tensor scalar params with requires_grad, intr_convention="standard")
    │   ├── calls intrinsics.scale_intrinsics(scale=tensor scalar scale factors with requires_grad)
    │   ├── calls scaled_intrinsics.project(points_camera=valid camera-space points)
    │   ├── impls loss = image_points.sum()
    │   ├── calls loss.backward
    │   ├── impls assert source tensor params receive gradients
    │   └── impls assert scale factors receive gradients
    └── return
```
