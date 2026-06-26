# Camera Data Structure Tests Structure

Test skeleton for `tests/data/structures/three_d/camera/`. Branches mirror the system-under-test structure declared in `code_structure.md`; leaves are individual pytest test functions with a one-line purpose. The skeleton fixes what each test pins and the phase-3 implementation follows it exactly. Grow leaves as we find regressions, contracts, or spec items worth pinning.

`data/structures/three_d/camera/` is a pure data-structure library, so there are unit tests only — no end-to-end / running-app tests. For module code structure see `code_structure.md`; for the module folder layout see `folder_structure.md`.

## 1. Code structure trees

`tests/data/structures/three_d/camera/test_conventions.py`

```text
test_conventions.py
├── def test_validate_camera_convention_accepts_all_supported
│   └── # validate_camera_convention accepts every supported convention string.
├── def test_conventions_module_has_one_main_api_and_eight_helpers
│   └── # The conventions module exposes exactly one main API plus eight helpers.
├── def test_camera_conversion_preserves_physical_axes_and_center
│   └── # Converting a Camera between conventions preserves its physical right / forward / up axes and center.
├── def test_camera_direct_and_via_standard_conversion_match
│   └── # Converting a Camera directly between two conventions matches converting via the standard convention.
├── def test_camera_round_trip_returns_original_extrinsics
│   └── # Converting a Camera to another convention and back returns the original extrinsics.
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
└── def test_multi_cameras_npz_round_trip
    └── # A Cameras collection survives a save then load round trip through the npz format.
```

`tests/data/structures/three_d/camera/test_rotation_stabilize_validate_compat.py`

```text
test_rotation_stabilize_validate_compat.py
├── def test_stabilize_accepts_float32_and_float64
│   └── # _stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.
├── def test_stabilize_rejects_unsupported_dtype
│   └── # _stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).
├── def test_stabilized_batch_passes_validator
│   └── # A batch of stabilized cam2world extrinsics passes the batched validate_camera_extrinsics for both float32 and float64 — the stabilizer-validator compatibility the fix restores.
└── def test_validator_threshold_is_dtype_aware
    └── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64, confirming the per-dtype tolerance dispatch.
```
