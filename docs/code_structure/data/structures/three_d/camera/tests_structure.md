# Camera Data Structure Tests Structure

## 1. Code structure trees

`tests/data/structures/three_d/camera/test_intrinsics.py`

```text
test_intrinsics.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsicsOrtho, CameraIntrinsicsPinhole, CameraIntrinsicsSimplePinhole, build_camera_intrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_attributes, validate_camera_intrinsics_params, validate_camera_model
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
│   ├── # validate_camera_intrinsics_params enforces each model's named parameter keys (simple_pinhole: f / cx / cy; pinhole / ortho: fx / fy / cx / cy) and rejects a mismatched params dict.
│   ├── for each (model, its named parameter keys)
│   │   ├── calls validate_camera_intrinsics_params
│   │   ├── impls assert the returned params dict equals the accepted one
│   │   └── with pytest.raises(AssertionError)
│   │       └── calls validate_camera_intrinsics_params(model=this model, params=another model's key set)
│   └── return
├── def test_validate_intrinsics_attributes_checks_model_params_device
│   ├── # validate_camera_intrinsics_attributes validates the camera model, its params, and the device together as the single CameraIntrinsics.__init__ entry.
│   ├── calls validate_camera_intrinsics_attributes(model=a supported model, params=its matching params, device=a valid device)
│   ├── for each attribute broken in turn (the model, the params, the device)
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
└── def test_scale_intrinsics_scales_focal_and_principal_point
    ├── # CameraIntrinsics.scale_intrinsics scales the focal length(s) and principal point to a resolution or by a factor.
    ├── for each of the three camera models
    │   ├── calls build_camera_intrinsics
    │   ├── calls intrinsics.scale_intrinsics(resolution=a target resolution)
    │   ├── impls assert the focal and principal-point params scaled by the per-axis target-over-current ratio  # impls-node-one-step:skip
    │   ├── calls intrinsics.scale_intrinsics(scale=an explicit (sx, sy) pair)
    │   └── impls assert the focal and principal-point params scaled by that factor  # impls-node-one-step:skip
    └── return
```

`tests/data/structures/three_d/camera/test_conventions.py`

```text
test_conventions.py
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics import conventions
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_convention
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── def test_validate_camera_convention_accepts_all_supported
│   ├── # validate_camera_convention accepts every supported convention string.
│   ├── for each convention in {standard, opengl, opencv, pytorch3d, arkit}
│   │   ├── calls validate_camera_convention
│   │   └── impls assert the returned string is the convention that was passed in
│   └── return
├── def test_conventions_module_has_one_main_api_and_eight_helpers
│   ├── # The relocated extrinsics/conventions module exposes exactly one main API plus eight helpers.
│   ├── impls collect the function names the conventions module defines
│   ├── impls assert transform_convention is its only public function
│   ├── impls assert the eight helpers are the to-standard and from-standard converters for opengl, opencv, pytorch3d, and arkit  # impls-node-one-step:skip
│   └── return
├── def test_extrinsics_conversion_preserves_physical_axes_and_center
│   ├── # Converting a CameraExtrinsics between conventions preserves its physical right / forward / up axes and center.
│   ├── calls CameraExtrinsics
│   ├── for each target convention
│   │   ├── calls extrinsics.to
│   │   ├── impls assert the converted right / forward / up axes equal the source ones under torch.allclose
│   │   └── impls assert the converted center equals the source center under torch.allclose
│   └── return
├── def test_extrinsics_direct_and_via_standard_conversion_match
│   ├── # Converting a CameraExtrinsics directly between two conventions matches converting via the standard convention.
│   ├── for each (source, target) convention pair
│   │   ├── calls CameraExtrinsics(extrinsics=the same pose, convention=the source convention)
│   │   ├── calls extrinsics.to(convention=the target convention)
│   │   ├── calls extrinsics.to(convention="standard")
│   │   ├── calls standardized.to(convention=the target convention)
│   │   └── impls assert the two target-convention 4x4 matrices agree under torch.allclose
│   └── return
├── def test_extrinsics_round_trip_returns_original_matrix
│   ├── # Converting a CameraExtrinsics to another convention and back returns the original 4x4 matrix.
│   ├── calls CameraExtrinsics
│   ├── for each target convention
│   │   ├── calls extrinsics.to(convention=the target convention)
│   │   ├── calls converted.to(convention=the original convention)
│   │   └── impls assert the round-tripped 4x4 matrix equals the original under torch.allclose
│   └── return
├── def test_extrinsics_w2c_is_inverse_of_extrinsics
│   ├── # CameraExtrinsics.w2c is the inverse of the 4x4 camera-to-world extrinsics matrix.
│   ├── calls CameraExtrinsics
│   ├── impls assert w2c @ extrinsics equals the 4x4 identity under torch.allclose
│   └── return
├── def test_cameras_conversion_preserves_physical_axes_and_center
│   ├── # Converting a Cameras collection between conventions preserves each camera's physical axes and center.
│   ├── for each pose in the collection
│   │   ├── calls build_camera_intrinsics
│   │   └── calls CameraExtrinsics
│   ├── calls Cameras
│   ├── for each target convention
│   │   ├── calls cameras.to
│   │   ├── impls assert the converted [N, 3] right / forward / up stacks equal the source ones under torch.allclose
│   │   └── impls assert the converted [N, 3] center stack equals the source one under torch.allclose
│   └── return
└── def test_every_supported_convention_is_right_handed
    ├── # A camera carries no change of handedness: each supported convention's (right, forward, up) triple is positively oriented, so converting between two of them keeps the rotation determinant at +1.
    ├── calls CameraExtrinsics
    ├── for each target convention
    │   ├── calls extrinsics.to
    │   ├── impls assert the converted right / forward / up triple has a positive scalar triple product
    │   └── impls assert the converted rotation block's determinant is +1
    └── return
```

`tests/data/structures/three_d/camera/test_io.py`

```text
test_io.py
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.io import load_cameras, save_cameras
├── def test_single_camera_json_round_trip
│   ├── # A single Camera survives a save then load round trip through the json format.
│   ├── calls build_camera_intrinsics
│   ├── calls CameraExtrinsics
│   ├── calls Camera
│   ├── calls camera.save(camera_path=a .json path under tmp_path)
│   ├── calls Camera.load
│   ├── impls assert the loaded object is a Camera instance
│   ├── impls assert its intrinsics model / params, extrinsics matrix / convention, name, and id match the saved camera's  # impls-node-one-step:skip
│   └── return
├── def test_single_camera_npz_round_trip
│   ├── # A single Camera survives a save then load round trip through the npz format.
│   ├── calls build_camera_intrinsics
│   ├── calls CameraExtrinsics
│   ├── calls Camera
│   ├── calls camera.save(camera_path=a .npz path under tmp_path)
│   ├── calls Camera.load
│   ├── impls assert the loaded object is a Camera instance
│   ├── impls assert its intrinsics model / params, extrinsics matrix / convention, name, and id match the saved camera's  # impls-node-one-step:skip
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
└── def test_extrinsics_and_convention_survive_round_trip
    ├── # A Camera's extrinsics matrix and convention survive a save then load round trip through both the json and npz formats.
    ├── for each format in {json, npz}
    │   └── for each supported convention
    │       ├── calls build_camera_intrinsics
    │       ├── calls CameraExtrinsics
    │       ├── calls Camera
    │       ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
    │       ├── calls Camera.load
    │       ├── impls assert the loaded 4x4 extrinsics matrix equals the saved one exactly
    │       └── impls assert the loaded convention equals the saved convention
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
