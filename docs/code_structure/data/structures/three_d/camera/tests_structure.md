# Camera Data Structure Tests Structure

## 1. Tests implementation structure

`tests/data/structures/three_d/camera/test_intrinsics.py`

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
├── def test_validate_intrinsics_params_dispatches_per_model_keys
│   ├── # validate_camera_intrinsics_params enforces each model's named parameter keys (simple_pinhole: f / cx / cy; pinhole / ortho: fx / fy / cx / cy) beside the h and w every model carries, and rejects a mismatched params dict.
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
├── def test_validate_intrinsics_attributes_checks_model_intr_convention_params_device
│   ├── # validate_camera_intrinsics_attributes validates the camera model, the image-plane frame, the params stated in that frame, and the device together as the single CameraIntrinsics.__init__ entry.
│   ├── calls validate_camera_intrinsics_attributes(model=a supported model, intr_convention="standard", params=its matching params, device=a valid device)
│   ├── for each attribute broken in turn (the model, the intr_convention, the params, the device)
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_attributes
│   └── return
├── def test_build_camera_intrinsics_dispatches_to_model_subclass
│   ├── # build_camera_intrinsics returns the CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole / CameraIntrinsicsOrtho instance for its model string.
│   ├── for each (model, its expected CameraIntrinsics subclass)
│   │   ├── calls build_camera_intrinsics
│   │   ├── impls assert the built instance's type is that subclass
│   │   └── impls assert the built instance's model property equals the model string
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
├── _REPO_ROOT = Path(__file__).resolve().parents[5]
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
│   │   ├── impls assert the returned intrinsics' h and w params are the target ones  # impls-node-one-step:skip
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
└── def test_the_pytorch3d_frames_params_move_when_the_aspect_ratio_does
    ├── # pytorch3d normalizes both axes by the shorter side alone, so its params hold under a resize that keeps the aspect ratio and are restated by one that does not — the case a frame-blind resize would silently leave wrong.
    ├── calls build_camera_intrinsics(model="pinhole", params=its own key set, intr_convention="pytorch3d")
    ├── calls intrinsics.scale_intrinsics(scale=a single factor)
    ├── impls assert every param but h and w comes back equal to the one it was given  # impls-node-one-step:skip
    ├── calls intrinsics.scale_intrinsics(resolution=a target resolution of a different aspect ratio)
    ├── impls assert its principal point comes back restated, equal to the same params carried into pixels, resized there, and carried back  # impls-node-one-step:skip
    └── return
```

`tests/data/structures/three_d/camera/test_conventions.py`

```text
test_conventions.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics import conventions
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics, validate_extr_convention
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.intrinsics.conventions import transform_intr_convention
├── from data.structures.three_d.camera.intrinsics.scaling import rescale_intr_params
├── from data.structures.three_d.camera.intrinsics.validation import validate_intr_convention
├── def test_validate_extr_convention_accepts_all_supported
│   ├── # validate_extr_convention accepts every supported convention string.
│   ├── for each extr_convention in {standard, opengl, opencv, pytorch3d, arkit}
│   │   ├── calls validate_extr_convention
│   │   └── impls assert the returned string is the extr_convention that was passed in
│   └── return
├── def test_extr_convention_module_has_one_main_api_and_eight_helpers
│   ├── # The relocated extrinsics/conventions module exposes exactly one main API plus eight helpers.
│   ├── impls collect the function names the conventions module defines
│   ├── impls assert transform_extr_convention is its only public function
│   ├── impls assert the eight helpers are the to-standard and from-standard converters for opengl, opencv, pytorch3d, and arkit  # impls-node-one-step:skip
│   └── return
├── def test_extrinsics_conversion_preserves_physical_axes_and_center
│   ├── # Converting a CameraExtrinsics between extr_conventions preserves its physical right / forward / up axes and center.
│   ├── calls CameraExtrinsics
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to
│   │   ├── impls assert the converted right / forward / up axes equal the source ones under torch.allclose
│   │   └── impls assert the converted center equals the source center under torch.allclose
│   └── return
├── def test_extrinsics_direct_and_via_standard_conversion_match
│   ├── # Converting a CameraExtrinsics directly between two extr_conventions matches converting via the standard one.
│   ├── for each (source, target) extr_convention pair
│   │   ├── calls CameraExtrinsics(extrinsics=the same pose, extr_convention=the source extr_convention)
│   │   ├── calls extrinsics.to(extr_convention=the target extr_convention)
│   │   ├── calls extrinsics.to(extr_convention="standard")
│   │   ├── calls standardized.to(extr_convention=the target extr_convention)
│   │   └── impls assert the two target-frame 4x4 matrices agree under torch.allclose
│   └── return
├── def test_extrinsics_round_trip_returns_original_matrix
│   ├── # Converting a CameraExtrinsics to another extr_convention and back returns the original 4x4 matrix.
│   ├── calls CameraExtrinsics
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to(extr_convention=the target extr_convention)
│   │   ├── calls converted.to(extr_convention=the original extr_convention)
│   │   └── impls assert the round-tripped 4x4 matrix equals the original under torch.allclose
│   └── return
├── def test_extrinsics_w2c_is_inverse_of_extrinsics
│   ├── # CameraExtrinsics.w2c is the inverse of the 4x4 camera-to-world extrinsics matrix.
│   ├── calls CameraExtrinsics
│   ├── impls assert w2c @ extrinsics equals the 4x4 identity under torch.allclose
│   └── return
├── def test_transform_extrinsics_applies_the_similarity_and_restabilizes
│   ├── # A similarity carries a pose the way it carries the world that pose sits in, so the composed cam2world is the one the scale, rotation and translation name and comes back revalidated.
│   ├── calls CameraExtrinsics
│   ├── calls extrinsics.transform_extrinsics(scale=a known factor, rotation=a known (3, 3) rotation, translation=a known (3,) offset)
│   ├── impls assert the returned rotation block equals the known rotation composed onto the source's
│   ├── impls assert the returned centre equals the source centre scaled, rotated and translated by those three  # impls-node-one-step:skip
│   ├── calls validate_camera_extrinsics(obj=the returned 4x4 cam2world)
│   ├── calls Camera
│   ├── calls camera.transform_extrinsics(scale=that same factor, rotation=that same rotation, translation=that same offset)
│   ├── impls assert its extrinsics matrix equals the standalone CameraExtrinsics result
│   ├── calls Cameras
│   ├── calls cameras.transform_extrinsics(scale=that same factor, rotation=that same rotation, translation=that same offset)
│   ├── impls assert every camera in the batch carries that same result
│   └── return
├── def test_cameras_conversion_preserves_physical_axes_and_center
│   ├── # Converting a Cameras collection between extr_conventions preserves each camera's physical axes and center.
│   ├── for each pose in the collection
│   │   ├── calls build_camera_intrinsics
│   │   └── calls CameraExtrinsics
│   ├── calls Cameras
│   ├── for each target extr_convention
│   │   ├── calls cameras.to
│   │   ├── impls assert the converted [N, 3] right / forward / up stacks equal the source ones under torch.allclose
│   │   └── impls assert the converted [N, 3] center stack equals the source one under torch.allclose
│   └── return
├── def test_every_supported_extr_convention_is_right_handed
│   ├── # A camera carries no change of handedness: each supported pose frame's (right, forward, up) triple is positively oriented, so converting between two of them keeps the rotation determinant at +1.
│   ├── calls CameraExtrinsics
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to
│   │   ├── impls assert the converted right / forward / up triple has a positive scalar triple product
│   │   └── impls assert the converted rotation block's determinant is +1
│   └── return
├── def test_validate_intr_convention_accepts_all_supported
│   ├── # The intrinsics name their own frame from a closed set, so every supported image-plane frame validates and anything else is rejected.
│   ├── calls validate_intr_convention
│   ├── impls assert standard, opengl, pytorch3d and vulkan each come back unchanged  # impls-node-one-step:skip
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_intr_convention
│   └── return
├── def test_intr_convention_module_has_one_main_api_and_six_spoke_helpers
│   ├── # Each frame brings its own inbound and outbound helper against the standard one rather than a helper against every other frame, so the oblique conversions are compositions and a frame added later edits none of them.
│   ├── impls assert the module exposes transform_intr_convention as its one public entry, reaching into scaling for the per-axis step each spoke ends in rather than owning one of its own
│   ├── impls assert it defines a to-standard and a from-standard helper for opengl, pytorch3d and vulkan, and none between two non-standard frames  # impls-node-one-step:skip
│   └── return
├── def test_a_frame_change_comes_down_to_the_same_per_axis_rescale
│   ├── # A frame change's only length step is the per-axis rescale scaling owns, which is why a shared focal is refused identically whether a caller goes through the frame change or reaches that rescale directly.
│   ├── calls rescale_intr_params(params=a pinhole's key set, model="pinhole", unit_x=a factor, unit_y=a different factor)
│   ├── impls assert the focal and cx / cy params come back scaled per axis and h and w come back untouched  # impls-node-one-step:skip
│   ├── with pytest.raises(AssertionError)
│   │   └── calls transform_intr_convention(params=a simple_pinhole's key set at a non-square size, model="simple_pinhole", source_intr_convention="standard", target_intr_convention="opengl")
│   ├── with pytest.raises(AssertionError)
│   │   └── calls rescale_intr_params(params=that same simple_pinhole key set, model="simple_pinhole", unit_x=a factor, unit_y=a different factor)
│   └── return
├── def test_three_separations_stand_between_standard_and_a_device_frame
│   ├── # Where the origin sits, which way each axis runs and what one unit is worth are independent, so a principal point at the image's own centre lands on the device origin, which no axis reversal alone could put it at.
│   ├── calls transform_intr_convention
│   ├── impls assert a standard principal point at half the resolution comes back at the origin under every device frame
│   ├── impls assert a point one pixel below centre comes back positive under vulkan and negative under opengl, those two frames disagreeing on y alone  # impls-node-one-step:skip
│   ├── impls assert a point one pixel right of centre comes back negative under pytorch3d and positive under opengl, those two disagreeing on x alone   # impls-node-one-step:skip
│   └── return
├── def test_each_frame_normalizes_by_the_side_its_own_definition_names
│   ├── # PyTorch3D spans its shorter side alone while opengl and vulkan span each axis by its own, so a non-square resolution tells the two normalizations apart.
│   ├── calls transform_intr_convention
│   ├── impls assert under a non-square resolution pytorch3d scales both axes by two over the shorter side
│   ├── impls assert opengl and vulkan scale x by two over w and y by two over h  # impls-node-one-step:skip
│   └── return
├── def test_only_the_unit_reaches_the_focal_params
│   ├── # An axis reversal reaches the linear term at both ends and cancels there, and moving the origin never touches a coefficient, so the unit is the whole of what a focal length sees.
│   ├── calls transform_intr_convention
│   ├── impls assert every focal param comes back scaled by its own axis's unit, its sign unchanged under every frame
│   └── return
├── def test_the_perspective_and_weak_perspective_models_take_the_same_focal_rule
│   ├── # A focal is a pixels-per-camera-unit ratio whether or not the projection divides by depth, so pinhole and ortho params come back scaled identically under every frame.
│   ├── calls transform_intr_convention
│   ├── impls assert a pinhole and an ortho carrying equal fx, fy, cx and cy come back equal under every frame  # impls-node-one-step:skip
│   └── return
├── def test_one_shared_focal_cannot_carry_two_different_axis_scales
│   ├── # simple_pinhole states a single f for both axes, so a frame that normalizes the axes by different sides cannot be expressed in that model and aborts rather than picking one of the two scales.
│   ├── calls transform_intr_convention
│   ├── impls assert a simple_pinhole converts to pytorch3d under a non-square resolution, that frame scaling both axes by the shorter side
│   ├── with pytest.raises(AssertionError)
│   │   └── calls transform_intr_convention
│   └── return
├── def test_a_camera_model_with_no_focal_rule_is_refused
│   ├── # The models this frame change knows are a closed set, so one it has no focal rule for is refused outright rather than silently carrying its principal point and leaving its focal in the old unit.
│   ├── with pytest.raises(NotImplementedError)
│   │   └── calls transform_intr_convention
│   └── return
├── def test_a_direct_conversion_matches_the_one_through_standard
│   ├── # Every oblique pair is served by composing the two spokes, so converting between two device frames agrees with converting out to standard and back in.
│   ├── calls transform_intr_convention
│   ├── impls assert opengl to pytorch3d equals opengl to standard followed by standard to pytorch3d  # impls-node-one-step:skip
│   ├── impls assert the same holds for every ordered pair of the supported frames
│   └── return
├── def test_an_intr_convention_round_trip_returns_the_original_params
│   ├── # A frame change is a restatement rather than a loss, so carrying an intrinsics out to another frame and back returns exactly what it started as.
│   ├── calls build_camera_intrinsics
│   ├── calls intrinsics.to
│   ├── impls assert the round-tripped params equal the original's under every frame
│   └── return
├── def test_a_frame_change_is_measured_against_the_intrinsics_own_resolution
│   ├── # The resolution is what fixes where a centred origin sits and what a normalized unit is worth, so the conversion reads the h and w the params already carry rather than a resolution the caller supplies and could get wrong.
│   ├── calls build_camera_intrinsics
│   ├── calls intrinsics.to
│   ├── impls assert two intrinsics whose projection params match but whose h and w differ convert to different results                   # impls-node-one-step:skip
│   ├── impls assert the converted intrinsics carries the same h and w it was built with, the image being the same image in either frame  # impls-node-one-step:skip
│   └── return
├── def test_an_intrinsics_without_a_resolution_is_refused
│   ├── # A principal point in the standard frame names a location only against a resolution, so params missing h or w are refused for every camera model.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls build_camera_intrinsics
│   └── return
└── def test_a_camera_names_the_frame_of_each_half_separately
    ├── # A pose frame and an image-plane frame are different kinds of thing, so a camera carries one name for each and converting one leaves the other where it was.
    ├── calls Camera
    ├── calls camera.to
    ├── impls assert the returned camera's extr_convention is the camera-space frame that was named
    ├── impls assert its intr_convention is the image-plane frame that was named
    ├── impls assert naming only one of the two leaves the other half's frame unchanged
    └── return
```

`tests/data/structures/three_d/camera/test_io.py`

```text
test_io.py
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── def test_single_camera_json_round_trip
│   ├── # A single Camera survives a save then load round trip through the json format.
│   ├── calls build_camera_intrinsics
│   ├── calls CameraExtrinsics
│   ├── calls Camera
│   ├── calls camera.save(camera_path=a .json path under tmp_path)
│   ├── calls Camera.load
│   ├── impls assert the loaded object is a Camera instance
│   ├── impls assert its intrinsics model / params, extrinsics matrix / extr_convention, name, and id match the saved camera's  # impls-node-one-step:skip
│   └── return
├── def test_single_camera_npz_round_trip
│   ├── # A single Camera survives a save then load round trip through the npz format.
│   ├── calls build_camera_intrinsics
│   ├── calls CameraExtrinsics
│   ├── calls Camera
│   ├── calls camera.save(camera_path=a .npz path under tmp_path)
│   ├── calls Camera.load
│   ├── impls assert the loaded object is a Camera instance
│   ├── impls assert its intrinsics model / params, extrinsics matrix / extr_convention, name, and id match the saved camera's  # impls-node-one-step:skip
│   └── return
├── def test_multi_cameras_json_round_trip
│   ├── # A Cameras collection survives a save then load round trip through the json format.
│   ├── for each camera in the collection
│   │   ├── calls build_camera_intrinsics
│   │   └── calls CameraExtrinsics
│   ├── calls Cameras
│   ├── calls save_cameras(cameras=the Cameras, cameras_path=a .json path under tmp_path)
│   ├── calls load_cameras
│   ├── impls assert the loaded object is a Cameras of the same length
│   ├── impls assert each loaded camera's intrinsics, extrinsics, name, and id match the saved one's at that index  # impls-node-one-step:skip
│   └── return
├── def test_multi_cameras_npz_round_trip
│   ├── # A Cameras collection survives a save then load round trip through the npz format.
│   ├── for each camera in the collection
│   │   ├── calls build_camera_intrinsics
│   │   └── calls CameraExtrinsics
│   ├── calls Cameras
│   ├── calls save_cameras(cameras=the Cameras, cameras_path=a .npz path under tmp_path)
│   ├── calls load_cameras
│   ├── impls assert the loaded object is a Cameras of the same length
│   ├── impls assert each loaded camera's intrinsics, extrinsics, name, and id match the saved one's at that index  # impls-node-one-step:skip
│   └── return
├── def test_the_intr_convention_and_resolution_survive_round_trip
│   ├── # An intrinsics' params name nothing without the frame they are stated in, so a payload that dropped it would deserialize into a different camera; the resolution needs no key of its own, riding inside those params.
│   ├── calls serialize_cameras
│   ├── calls deserialize_cameras
│   ├── impls assert the round-tripped intr_convention equals the original's, under json and npz alike                      # impls-node-one-step:skip
│   ├── impls assert its h and w params come back with the rest of them, under json and npz alike                           # impls-node-one-step:skip
│   ├── impls assert the intr_convention and the extr_convention come back independently, neither taking the other's value  # impls-node-one-step:skip
│   └── return
├── def test_model_and_params_survive_round_trip
│   ├── # A Camera's intrinsics model and params survive a save then load round trip through both the json and npz formats.
│   ├── for each format in {json, npz}
│   │   └── for each of the three camera models
│   │       ├── calls build_camera_intrinsics
│   │       ├── calls CameraExtrinsics
│   │       ├── calls Camera
│   │       ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
│   │       ├── calls Camera.load
│   │       ├── impls assert the loaded intrinsics model equals the saved model string
│   │       └── impls assert the loaded params dict equals the saved params dict
│   └── return
└── def test_extrinsics_and_extr_convention_survive_round_trip
    ├── # A Camera's extrinsics matrix and extr_convention survive a save then load round trip through both the json and npz formats.
    ├── for each format in {json, npz}
    │   └── for each supported extr_convention
    │       ├── calls build_camera_intrinsics
    │       ├── calls CameraExtrinsics
    │       ├── calls Camera
    │       ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
    │       ├── calls Camera.load
    │       ├── impls assert the loaded 4x4 extrinsics matrix equals the saved one exactly
    │       └── impls assert the loaded extr_convention equals the saved extr_convention
    └── return
```

`tests/data/structures/three_d/camera/test_rotation_stabilize_validate_compat.py`

```text
test_rotation_stabilize_validate_compat.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import _stabilize_rotation_matrix
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics, validate_rotation_matrix
├── def test_stabilize_accepts_float32_and_float64
│   ├── # _stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.
│   ├── for each dtype in {torch.float32, torch.float64}
│   │   ├── calls _stabilize_rotation_matrix(rotation=a near-orthogonal (3, 3) rotation in that dtype)
│   │   ├── impls assert the returned rotation keeps that dtype
│   │   └── calls validate_rotation_matrix(obj=the returned rotation)
│   └── return
├── def test_stabilize_rejects_unsupported_dtype
│   ├── # _stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _stabilize_rotation_matrix(rotation=a float16 near-orthogonal rotation)
│   └── return
├── def test_stabilized_batch_passes_validator
│   ├── # A batch of stabilized cam2world extrinsics passes the batched validate_camera_extrinsics for both float32 and float64.
│   ├── for each dtype in {torch.float32, torch.float64}
│   │   ├── for each pose in the batch
│   │   │   └── calls _stabilize_rotation_matrix
│   │   ├── impls stack the stabilized rotations into a (B, 4, 4) cam2world batch in that dtype
│   │   └── calls validate_camera_extrinsics(obj=the (B, 4, 4) cam2world batch)
│   └── return
├── def test_validator_threshold_is_dtype_aware
│   ├── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.
│   ├── impls build a (3, 3) rotation whose orthogonality residual sits between the float64 and float32 tolerances  # impls-node-one-step:skip
│   ├── calls validate_rotation_matrix(obj=that rotation cast to float32)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_rotation_matrix(obj=that rotation cast to float64)
│   └── return
└── def test_validator_requires_determinant_plus_one
    ├── # A camera's rotation is validated to have determinant +1, so a reflection is inexpressible as camera extrinsics and any change of handedness has to be carried by the geometry instead.
    ├── calls validate_rotation_matrix(obj=a proper rotation of determinant +1)
    ├── impls build an orthonormal (3, 3) matrix whose determinant is -1 by negating one of its columns
    ├── with pytest.raises(AssertionError)
    │   └── calls validate_rotation_matrix(obj=that determinant -1 matrix)
    ├── with pytest.raises(AssertionError)
    │   └── calls validate_camera_extrinsics(obj=a cam2world batch carrying that reflection)
    └── return
```
