# Camera Data Structure Tests Structure

## 1. Code structure trees

`tests/data/structures/three_d/camera/test_intrinsics.py`

```text
test_intrinsics.py
├── def test_validate_camera_model_accepts_all_supported
│   └── # validate_camera_model accepts simple_pinhole, pinhole, and ortho.
├── def test_validate_camera_model_rejects_unsupported
│   └── # validate_camera_model raises on a camera-model string outside the supported set.
├── def test_validate_intrinsics_params_dispatches_per_model_keys
│   └── # validate_camera_intrinsics_params enforces each model's named parameter keys (simple_pinhole: f / cx / cy; pinhole / ortho: fx / fy / cx / cy) and rejects a mismatched params dict.
├── def test_validate_intrinsics_attributes_checks_model_params_device
│   └── # validate_camera_intrinsics_attributes validates the camera model, its params, and the device together as the single CameraIntrinsics.__init__ entry.
├── def test_build_camera_intrinsics_dispatches_to_model_subclass
│   └── # build_camera_intrinsics returns the CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole / CameraIntrinsicsOrtho instance for its model string.
├── def test_simple_pinhole_project_applies_perspective_divide
│   └── # CameraIntrinsicsSimplePinhole.project applies the perspective divide with a single shared focal length.
├── def test_pinhole_project_applies_perspective_divide
│   └── # CameraIntrinsicsPinhole.project applies the perspective divide with independent fx / fy.
├── def test_ortho_project_skips_perspective_divide
│   └── # CameraIntrinsicsOrtho.project maps points without the perspective divide.
├── def test_fx_fy_cx_cy_derived_from_params
│   └── # The per-subclass fx / fy accessors and the base cx / cy accessors are derived from the model params.
├── def test_fov_defined_for_perspective_subclasses_only
│   └── # CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole expose fov in degrees while CameraIntrinsicsOrtho has no fov method.
└── def test_scale_intrinsics_scales_focal_and_principal_point
    └── # CameraIntrinsics.scale_intrinsics scales the focal length(s) and principal point to a resolution or by a factor.
```

`tests/data/structures/three_d/camera/test_conventions.py`

```text
test_conventions.py
├── def test_validate_camera_convention_accepts_all_supported
│   └── # validate_camera_convention accepts every supported convention string.
├── def test_conventions_module_has_one_main_api_and_eight_helpers
│   └── # The relocated extrinsics/conventions module exposes exactly one main API plus eight helpers.
├── def test_extrinsics_conversion_preserves_physical_axes_and_center
│   └── # Converting a CameraExtrinsics between conventions preserves its physical right / forward / up axes and center.
├── def test_extrinsics_direct_and_via_standard_conversion_match
│   └── # Converting a CameraExtrinsics directly between two conventions matches converting via the standard convention.
├── def test_extrinsics_round_trip_returns_original_matrix
│   └── # Converting a CameraExtrinsics to another convention and back returns the original 4x4 matrix.
├── def test_extrinsics_w2c_is_inverse_of_extrinsics
│   └── # CameraExtrinsics.w2c is the inverse of the 4x4 camera-to-world extrinsics matrix.
└── def test_cameras_conversion_preserves_physical_axes_and_center
    └── # Converting a Cameras collection between conventions preserves each camera's physical axes and center.
```

`tests/data/structures/three_d/camera/test_io.py`

```text
test_io.py
├── def test_single_camera_json_round_trip
│   └── # A single Camera survives a save then load round trip through the json format.
├── def test_single_camera_npz_round_trip
│   └── # A single Camera survives a save then load round trip through the npz format.
├── def test_multi_cameras_json_round_trip
│   └── # A Cameras collection survives a save then load round trip through the json format.
├── def test_multi_cameras_npz_round_trip
│   └── # A Cameras collection survives a save then load round trip through the npz format.
├── def test_model_and_params_survive_round_trip
│   └── # A Camera's intrinsics model and params survive a save then load round trip through both the json and npz formats.
└── def test_extrinsics_and_convention_survive_round_trip
    └── # A Camera's extrinsics matrix and convention survive a save then load round trip through both the json and npz formats.
```

`tests/data/structures/three_d/camera/test_rotation_stabilize_validate_compat.py`

```text
test_rotation_stabilize_validate_compat.py
├── def test_stabilize_accepts_float32_and_float64
│   └── # _stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.
├── def test_stabilize_rejects_unsupported_dtype
│   └── # _stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).
├── def test_stabilized_batch_passes_validator
│   └── # A batch of stabilized cam2world extrinsics passes the batched validate_camera_extrinsics for both float32 and float64.
└── def test_validator_threshold_is_dtype_aware
    └── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.
```
