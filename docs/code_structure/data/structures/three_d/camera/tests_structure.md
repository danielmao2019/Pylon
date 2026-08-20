# Camera Data Structure Tests Structure

## 1. Code structure trees

`tests/data/structures/three_d/camera/test_intrinsics.py`

```text
test_intrinsics.py
├── import pytest
├── import torch
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics, CameraIntrinsicsOrtho, CameraIntrinsicsPinhole, CameraIntrinsicsSimplePinhole, build_camera_intrinsics
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_attributes, validate_camera_intrinsics_params, validate_camera_model
├── def test_validate_camera_model_accepts_all_supported() -> None
│   ├── # validate_camera_model accepts simple_pinhole, pinhole, and ortho.
│   └── for model in ("simple_pinhole", "pinhole", "ortho")
│       ├── calls validate_camera_model(model=that model string)
│       └── assert it returns that same model string  # reporting the offending model
├── def test_validate_camera_model_rejects_unsupported() -> None
│   ├── # validate_camera_model raises on a camera-model string outside the supported set.
│   └── with pytest.raises(AssertionError)
│       └── calls validate_camera_model(model="fisheye")
├── def test_validate_intrinsics_params_dispatches_per_model_keys() -> None
│   ├── # validate_camera_intrinsics_params enforces each model's named parameter keys (simple_pinhole: f / cx / cy; pinhole / ortho: fx / fy / cx / cy) and rejects a mismatched params dict.
│   ├── impls the simple_pinhole params dict — f 400, cx 160, cy 120
│   ├── impls the pinhole params dict — fx 400, fy 410, cx 160, cy 120
│   ├── calls validate_camera_intrinsics_params(model="simple_pinhole", params=the simple_pinhole dict)
│   ├── assert it returns that same simple_pinhole dict
│   ├── calls validate_camera_intrinsics_params(model="pinhole", params=the pinhole dict)
│   ├── assert it returns that same pinhole dict
│   ├── calls validate_camera_intrinsics_params(model="ortho", params=the pinhole dict)
│   ├── assert it returns that same pinhole dict  # ortho shares the pinhole key set
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_camera_intrinsics_params(model="simple_pinhole", params=the pinhole dict)
│   └── with pytest.raises(AssertionError)
│       └── calls validate_camera_intrinsics_params(model="pinhole", params=the simple_pinhole dict)
├── def test_validate_intrinsics_attributes_checks_model_params_device() -> None
│   ├── # validate_camera_intrinsics_attributes validates the camera model, its params, and the device together as the single CameraIntrinsics.__init__ entry.
│   ├── impls the pinhole params dict — fx 400, fy 410, cx 160, cy 120
│   ├── calls validate_camera_intrinsics_attributes(model="pinhole", params=that dict, device="cpu")
│   ├── with pytest.raises(AssertionError)
│   │   └── calls validate_camera_intrinsics_attributes(model="pinhole", params=that same dict, device=0)
│   └── with pytest.raises(AssertionError)
│       └── calls validate_camera_intrinsics_attributes(model="pinhole", params=the simple_pinhole f / cx / cy dict, device="cpu")
├── def test_build_camera_intrinsics_dispatches_to_model_subclass() -> None
│   ├── # build_camera_intrinsics returns the CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole / CameraIntrinsicsOrtho instance for its model string.
│   ├── calls build_camera_intrinsics(model="simple_pinhole", params={"f": 400.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls build_camera_intrinsics(model="pinhole", params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls build_camera_intrinsics(model="ortho", params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── assert the simple_pinhole build is a CameraIntrinsicsSimplePinhole
│   ├── assert the pinhole build is a CameraIntrinsicsPinhole
│   └── assert the ortho build is a CameraIntrinsicsOrtho
├── def test_simple_pinhole_project_applies_perspective_divide() -> None
│   ├── # CameraIntrinsicsSimplePinhole.project applies the perspective divide with a single shared focal length.
│   ├── calls CameraIntrinsicsSimplePinhole(params={"f": 400.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── impls one [1, 3] float32 camera-space point
│   ├── calls intrinsics.project(points_camera=that point)
│   ├── impls the expected image point — f * x / z + cx and f * y / z + cy
│   └── assert the projection matches it  # torch.allclose, atol 1e-05
├── def test_pinhole_project_applies_perspective_divide() -> None
│   ├── # CameraIntrinsicsPinhole.project applies the perspective divide with independent fx / fy.
│   ├── calls CameraIntrinsicsPinhole(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── impls one [1, 3] float32 camera-space point
│   ├── calls intrinsics.project(points_camera=that point)
│   ├── impls the expected image point — fx * x / z + cx and fy * y / z + cy
│   └── assert the projection matches it  # torch.allclose, atol 1e-05
├── def test_ortho_project_skips_perspective_divide() -> None
│   ├── # CameraIntrinsicsOrtho.project maps points without the perspective divide.
│   ├── calls CameraIntrinsicsOrtho(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── impls a near [1, 3] point and a far [1, 3] point sharing the same x and y
│   ├── calls intrinsics.project(points_camera=the near point)
│   ├── calls intrinsics.project(points_camera=the far point)
│   ├── impls the expected image point — fx * x + cx and fy * y + cy
│   ├── assert the near projection matches it                                  # torch.allclose, atol 1e-05
│   └── assert the far projection is allclose to the near one at atol 1.0e-05  # "Ortho projection must ignore depth (no perspective divide)."
├── @pytest.mark.parametrize def test_project_inplace_overwrites_input_and_matches_not_inplace(intrinsics: CameraIntrinsics) -> None  # over the simple_pinhole, pinhole, and ortho intrinsics instances
│   ├── # project(inplace=True) overwrites points_camera cols 0,1 with the image points (matching inplace=False), preserves the depth col 2, and returns a tensor aliasing the input, across all three models.
│   ├── impls a [2, 3] float32 points_camera
│   ├── impls a pristine reference clone of it
│   ├── calls intrinsics.project(points_camera=a clone, inplace=False)              # -> the expected image points
│   ├── calls intrinsics.project(points_camera=points_camera itself, inplace=True)  # -> the result
│   ├── assert the result's data_ptr equals points_camera's                         # "Expected the inplace result to alias the input tensor."
│   ├── assert the result matches the not-inplace image points                      # "Expected the inplace result to match the not-inplace result."
│   ├── assert points_camera cols 0 and 1 now hold those image points               # "Expected the first two input columns to be overwritten in place."
│   └── assert points_camera col 2 still holds the reference depth                  # "Expected the input depth column to be preserved."
├── @pytest.mark.parametrize def test_project_not_inplace_preserves_input_and_returns_new_tensor(intrinsics: CameraIntrinsics) -> None  # over the simple_pinhole, pinhole, and ortho intrinsics instances
│   ├── # project(inplace=False) returns a fresh [..., 2] and leaves points_camera unchanged, across all three models.
│   ├── impls a [2, 3] float32 points_camera
│   ├── impls a pristine reference clone of it
│   ├── calls intrinsics.project(points_camera=points, inplace=False)  # -> the result
│   ├── assert the result's data_ptr differs from points_camera's      # "Expected the not-inplace result to be a freshly allocated tensor."
│   ├── assert the result's shape is (2, 2)                            # "Expected the not-inplace result to be a [..., 2] tensor."
│   └── assert points_camera still equals the reference clone          # "Expected the input tensor to be left unchanged."
├── @pytest.mark.parametrize def test_project_supports_batched_leading_dims(intrinsics: CameraIntrinsics) -> None  # over the simple_pinhole, pinhole, and ortho intrinsics instances
│   ├── # project handles [..., 3] leading dims: a batched input (inplace and not-inplace) matches projecting its flattened [N, 3] view, across all three models.
│   ├── impls a [2, 3, 3] float32 batch of camera-space points
│   ├── calls intrinsics.project(points_camera=a clone of the batch flattened to [6, 3], inplace=False)  # -> the expected image points reshaped back to [2, 3, 2]
│   ├── calls intrinsics.project(points_camera=a clone of the batch, inplace=False)                      # -> the not-inplace result
│   ├── assert that result's shape is (2, 3, 2)     # "Expected the not-inplace batched result to keep the leading dims."
│   ├── assert it matches the flattened projection  # "Expected the not-inplace batched result to match the flattened projection."
│   ├── impls a fresh clone of the batch as points_camera
│   ├── impls a pristine reference clone of that
│   ├── calls intrinsics.project(points_camera=points, inplace=True)  # -> the inplace result
│   ├── assert it matches the flattened projection                    # "Expected the inplace batched result to match the flattened projection."
│   ├── assert points_camera[..., :2] now holds those image points    # "Expected the first two input columns to be overwritten in place."
│   └── assert points_camera[..., 2] still holds the reference depth  # "Expected the input depth column to be preserved."
├── @pytest.mark.parametrize def test_project_rejects_invalid_inputs(intrinsics: CameraIntrinsics) -> None  # over the simple_pinhole, pinhole, and ortho intrinsics instances
│   ├── # project raises AssertionError on a non-tensor points_camera, a wrong last dim, and a non-bool inplace, across all three models.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.project(points_camera=a nested list rather than a tensor)
│   ├── with pytest.raises(AssertionError)
│   │   └── calls intrinsics.project(points_camera=a [4, 2] float32 tensor)  # whose last dim is 2
│   └── with pytest.raises(AssertionError)
│       └── calls intrinsics.project(points_camera=a valid [1, 3] tensor, inplace=1 rather than a bool)
├── def test_fx_fy_cx_cy_derived_from_params() -> None
│   ├── # The per-subclass fx / fy accessors and the base cx / cy accessors are derived from the model params.
│   ├── calls CameraIntrinsicsSimplePinhole(params={"f": 400.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── assert its fx and fy both read the single f param
│   ├── assert its cx and cy read the cx / cy params
│   ├── calls CameraIntrinsicsPinhole(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── assert its fx and fy read the independent fx / fy params
│   ├── assert its cx and cy read the cx / cy params
│   ├── calls CameraIntrinsicsOrtho(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── assert its fx and fy read the independent fx / fy params
│   └── assert its cx and cy read the cx / cy params
├── def test_fov_defined_for_perspective_subclasses_only() -> None
│   ├── # CameraIntrinsicsSimplePinhole / CameraIntrinsicsPinhole expose fov in degrees while CameraIntrinsicsOrtho has no fov method.
│   ├── calls CameraIntrinsicsSimplePinhole(params={"f": 400.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls CameraIntrinsicsPinhole(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls CameraIntrinsicsOrtho(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── assert the simple_pinhole fov is a length-2 tuple
│   ├── assert the pinhole fov is a length-2 tuple
│   ├── assert every simple_pinhole fov entry is a float
│   ├── assert every pinhole fov entry is a float
│   └── assert the ortho intrinsics carries no fov attribute  # "Ortho intrinsics must not expose fov."
└── def test_scale_intrinsics_scales_focal_and_principal_point() -> None
    ├── # CameraIntrinsics.scale_intrinsics scales the focal length(s) and principal point to a resolution or by a factor.
    ├── calls CameraIntrinsicsPinhole(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
    ├── calls intrinsics.scale_intrinsics(scale=2.0)
    ├── assert its params are fx 800 / fy 820 / cx 320 / cy 240
    ├── assert it is a CameraIntrinsicsPinhole
    ├── calls intrinsics.scale_intrinsics(scale=(2.0, 0.5))
    ├── assert its params are fx 800 / fy 205 / cx 320 / cy 60
    ├── # Current resolution inferred from the principal point is (W, H) = (320, 240).
    ├── calls intrinsics.scale_intrinsics(resolution=(480, 640))
    └── assert its params are fx 800 / fy 820 / cx 320 / cy 240
```

`tests/data/structures/three_d/camera/test_conventions.py`

```text
test_conventions.py
├── import inspect
├── from itertools import product
├── from typing import List
├── import pytest
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics import conventions as conventions_module
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_convention
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsicsPinhole
├── CONVENTIONS  # List[str] — the five supported conventions: standard, opengl, opencv, pytorch3d, arkit
├── @pytest.mark.parametrize def test_validate_camera_convention_accepts_all_supported(convention: str) -> None  # over each convention in CONVENTIONS
│   ├── # validate_camera_convention accepts every supported convention string.
│   ├── calls validate_camera_convention(convention)
│   └── assert it returns that same convention string  # reporting the offending convention
├── def test_conventions_module_has_one_main_api_and_eight_helpers() -> None
│   ├── # The relocated extrinsics/conventions module exposes exactly one main API plus eight helpers.
│   ├── impls the expected helper names — the to-standard and standard-to converter for each of opengl, opencv, pytorch3d, and arkit
│   ├── calls inspect.getmembers(conventions_module, inspect.isfunction)  # -> its (name, function) pairs
│   ├── impls the helper names among them — underscore-prefixed and naming a to-standard or standard-to conversion
│   ├── assert that helper-name set equals the expected eight
│   ├── assert conventions_module exposes transform_convention
│   ├── assert conventions_module exposes no _opengl_to_opencv
│   └── assert conventions_module exposes no _opencv_to_pytorch3d
├── @pytest.mark.parametrize def test_extrinsics_conversion_preserves_physical_axes_and_center(source_convention: str, target_convention: str) -> None  # over every (source, target) pair from product(CONVENTIONS, CONVENTIONS)
│   ├── # Converting a CameraExtrinsics between conventions preserves its physical right / forward / up axes and center.
│   ├── calls _build_extrinsics(convention=source_convention)
│   ├── calls extrinsics.to(convention=target_convention)                  # -> the converted extrinsics
│   ├── assert the converted center matches the source center              # torch.allclose, atol 1e-06, rtol 0
│   ├── assert the converted right axis matches the source right axis      # torch.allclose, atol 1e-06, rtol 0
│   ├── assert the converted forward axis matches the source forward axis  # torch.allclose, atol 1e-06, rtol 0
│   └── assert the converted up axis matches the source up axis            # torch.allclose, atol 1e-06, rtol 0
├── @pytest.mark.parametrize def test_extrinsics_direct_and_via_standard_conversion_match(source_convention: str, target_convention: str) -> None  # over every (source, target) pair from product(CONVENTIONS, CONVENTIONS)
│   ├── # Converting a CameraExtrinsics directly between two conventions matches converting via the standard convention.
│   ├── calls _build_extrinsics(convention=source_convention)
│   ├── calls extrinsics.to(convention=target_convention)                            # -> the direct conversion
│   ├── calls extrinsics.to(convention="standard").to(convention=target_convention)  # -> the routed conversion
│   └── assert the two 4x4 matrices agree  # torch.allclose, atol 1e-06, rtol 0
├── @pytest.mark.parametrize def test_extrinsics_round_trip_returns_original_matrix(source_convention: str, target_convention: str) -> None  # over every (source, target) pair from product(CONVENTIONS, CONVENTIONS)
│   ├── # Converting a CameraExtrinsics to another convention and back returns the original 4x4 matrix.
│   ├── calls _build_extrinsics(convention=source_convention)
│   ├── calls extrinsics.to(convention=target_convention).to(convention=source_convention)
│   └── assert the round-tripped 4x4 matrix equals the original  # torch.allclose, atol 1e-06, rtol 0
├── @pytest.mark.parametrize def test_extrinsics_w2c_is_inverse_of_extrinsics(convention: str) -> None  # over each convention in CONVENTIONS
│   ├── # CameraExtrinsics.w2c is the inverse of the 4x4 camera-to-world extrinsics matrix.
│   ├── calls _build_extrinsics(convention=convention)
│   ├── impls the matrix product of w2c and the cam2world extrinsics
│   ├── calls torch.eye(4, dtype=extrinsics.extrinsics.dtype)  # -> the identity to compare against
│   └── assert that product equals the identity                # torch.allclose, atol 1e-05, rtol 0
├── @pytest.mark.parametrize def test_cameras_conversion_preserves_physical_axes_and_center(source_convention: str, target_convention: str) -> None  # over every (source, target) pair from product(CONVENTIONS, CONVENTIONS)
│   ├── # Converting a Cameras collection between conventions preserves each camera's physical axes and center.
│   ├── calls _build_cameras(convention=source_convention)
│   ├── calls cameras.to(convention=target_convention)                 # -> the converted Cameras
│   ├── assert the converted center[0] matches the source center[0]    # torch.allclose, atol 1e-06, rtol 0
│   ├── assert the converted right[0] matches the source right[0]      # torch.allclose, atol 1e-06, rtol 0
│   ├── assert the converted forward[0] matches the source forward[0]  # torch.allclose, atol 1e-06, rtol 0
│   └── assert the converted up[0] matches the source up[0]            # torch.allclose, atol 1e-06, rtol 0
├── def _build_cameras(convention: str) -> Cameras
│   ├── # Lifts the same pose into a one-camera Cameras, so the collection path is exercised on identical input.
│   ├── calls _build_intrinsics
│   ├── calls _build_extrinsics(convention=convention)
│   ├── calls Cameras(intrinsics=[that intrinsics], extrinsics=[that extrinsics], device="cpu")
│   └── return  # that one-camera Cameras
├── def _build_intrinsics() -> CameraIntrinsicsPinhole
│   ├── # Supplies the intrinsics a Cameras needs, so a convention test states nothing about intrinsics.
│   ├── calls CameraIntrinsicsPinhole(params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   └── return  # that CameraIntrinsicsPinhole
├── def _build_extrinsics(convention: str) -> CameraExtrinsics
│   ├── # Tags the shared pose with whichever convention a test parametrizes over.
│   ├── calls _build_extrinsics_matrix
│   ├── calls CameraExtrinsics(extrinsics=that matrix, convention=convention, device="cpu")
│   └── return  # that CameraExtrinsics
└── def _build_extrinsics_matrix() -> torch.Tensor
    ├── # Pins one fixed pose every fixture in this file shares, so a conversion result differs only by convention.
    ├── impls rows = the quarter-turn about z, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], with translation [0.3, -0.2, 1.1]
    └── return that 4x4 float32 tensor  # a cam2world matrix whose 3x3 block is a proper rotation
```

`tests/data/structures/three_d/camera/test_io.py`

```text
test_io.py
├── import json
├── from pathlib import Path
├── from typing import List, Optional
├── import numpy as np
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── _JSON_KEYS  # the six keys one json-serialized camera carries: model, params, extrinsics, convention, name, id
├── _NPZ_KEYS   # the eight arrays an npz payload carries: the json keys plus the has_name / has_id sentinel flags
├── def test_single_camera_json_round_trip(tmp_path: Path) -> None
│   ├── # A single Camera survives a save then load round trip through the json format.
│   ├── calls _make_single_camera
│   ├── calls serialize_cameras(cameras=that camera, format="json")  # -> the payload
│   ├── assert the payload is a dict
│   ├── assert its keys are exactly _JSON_KEYS
│   ├── calls camera.serialize(format="json")
│   ├── assert the free function's payload equals the method's
│   ├── calls deserialize_cameras(payload=the payload, device="cpu", format="json")
│   ├── calls Camera.deserialize(payload=the payload, device="cpu", format="json")
│   ├── calls _assert_camera_fields_equal(loaded=the free-function result, original=the camera)
│   ├── calls _assert_camera_fields_equal(loaded=the classmethod result, original=the camera)
│   ├── impls a camera.json path under tmp_path
│   ├── calls save_cameras(cameras=that camera, cameras_path=that json path)
│   ├── calls json.loads(json_path.read_text(encoding="utf-8"))  # -> what landed on disk
│   ├── assert the on-disk payload equals the serialized payload
│   ├── calls load_cameras(cameras_path=that json path, device="cpu")
│   ├── calls _assert_camera_fields_equal(loaded=the load_cameras result, original=the camera)
│   ├── calls Camera.load(camera_path=that json path, device="cpu")
│   └── calls _assert_camera_fields_equal(loaded=the Camera.load result, original=the camera)
├── def test_single_camera_npz_round_trip(tmp_path: Path) -> None
│   ├── # A single Camera survives a save then load round trip through the npz format.
│   ├── calls _make_single_camera
│   ├── calls serialize_cameras(cameras=that camera, format="npz")  # -> the payload
│   ├── assert the payload is a dict
│   ├── assert its keys are exactly _NPZ_KEYS plus is_single
│   ├── calls deserialize_cameras(payload=the payload, device="cpu", format="npz")
│   ├── calls _assert_camera_fields_equal(loaded=the deserialized result, original=the camera)
│   ├── impls a camera.npz path under tmp_path
│   ├── calls save_cameras(cameras=that camera, cameras_path=that npz path)
│   ├── with np.load(npz_path, allow_pickle=False) as on_disk
│   │   └── assert the archive's files are exactly _NPZ_KEYS plus is_single
│   ├── calls load_cameras(cameras_path=that npz path, device="cpu")
│   ├── calls _assert_camera_fields_equal(loaded=the load_cameras result, original=the camera)
│   ├── calls Camera.load(camera_path=that npz path, device="cpu")
│   └── calls _assert_camera_fields_equal(loaded=the Camera.load result, original=the camera)
├── def test_multi_cameras_json_round_trip(tmp_path: Path) -> None
│   ├── # A Cameras collection survives a save then load round trip through the json format.
│   ├── calls _make_multi_cameras
│   ├── calls serialize_cameras(cameras=that collection, format="json")  # -> the payload
│   ├── assert the payload is a list
│   ├── assert it holds one entry per camera
│   ├── for per_camera_dict in serialized
│   │   └── assert its keys are exactly _JSON_KEYS
│   ├── calls deserialize_cameras(payload=the payload, device="cpu", format="json")
│   ├── calls _assert_cameras_fields_equal(loaded=the deserialized result, original=the collection)
│   ├── impls a cameras.json path under tmp_path
│   ├── calls save_cameras(cameras=that collection, cameras_path=that json path)
│   ├── calls json.loads(json_path.read_text(encoding="utf-8"))  # -> what landed on disk
│   ├── assert the on-disk payload equals the serialized payload
│   ├── calls load_cameras(cameras_path=that json path, device="cpu")
│   └── calls _assert_cameras_fields_equal(loaded=the load_cameras result, original=the collection)
├── def test_multi_cameras_npz_round_trip(tmp_path: Path) -> None
│   ├── # A Cameras collection survives a save then load round trip through the npz format.
│   ├── calls _make_multi_cameras
│   ├── calls serialize_cameras(cameras=that collection, format="npz")  # -> the payload
│   ├── assert the payload is a dict
│   ├── assert its keys are exactly _NPZ_KEYS
│   ├── assert it carries no is_single flag  # that marker belongs to the single-camera path
│   ├── assert its stacked extrinsics array has shape (len(cameras), 4, 4)
│   ├── calls deserialize_cameras(payload=the payload, device="cpu", format="npz")
│   ├── calls _assert_cameras_fields_equal(loaded=the deserialized result, original=the collection)
│   ├── impls a cameras.npz path under tmp_path
│   ├── calls save_cameras(cameras=that collection, cameras_path=that npz path)
│   ├── with np.load(npz_path, allow_pickle=False) as on_disk
│   │   └── assert the archive's files are exactly _NPZ_KEYS
│   ├── calls load_cameras(cameras_path=that npz path, device="cpu")
│   └── calls _assert_cameras_fields_equal(loaded=the load_cameras result, original=the collection)
├── def test_model_and_params_survive_round_trip() -> None
│   ├── # A Camera's intrinsics model and params survive a save then load round trip through both the json and npz formats.
│   ├── calls _make_single_camera
│   └── for format in ("json", "npz")
│       ├── calls serialize_cameras(cameras=that camera, format=that format)
│       ├── calls deserialize_cameras(payload=that payload, device="cpu", format=that format)
│       ├── assert the loaded intrinsics model equals the camera's  # reporting the offending format
│       └── assert the loaded intrinsics params equal the camera's  # reporting the offending format
├── def test_extrinsics_and_convention_survive_round_trip() -> None
│   ├── # A Camera's extrinsics matrix and convention survive a save then load round trip through both the json and npz formats.
│   ├── calls _make_single_camera
│   └── for format in ("json", "npz")
│       ├── calls serialize_cameras(cameras=that camera, format=that format)
│       ├── calls deserialize_cameras(payload=that payload, device="cpu", format=that format)
│       ├── assert the loaded 4x4 extrinsics matrix equals the camera's exactly  # torch.equal, reporting the offending format
│       └── assert the loaded convention equals the camera's                     # reporting the offending format
├── def _make_single_camera() -> Camera
│   ├── # Gives the single-camera round trips one Camera carrying every optional field, so none goes silently untested.
│   ├── calls build_camera_intrinsics(model="pinhole", params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls _make_extrinsics(translation=[0.3, -0.2, 1.1], convention="opengl")
│   ├── calls Camera(intrinsics=those intrinsics, extrinsics=those extrinsics, name="frame_0", id=7, device="cpu")
│   └── return  # that Camera
├── def _make_multi_cameras() -> Cameras
│   ├── # Mixes all three models and conventions with one absent name and one absent id, so the has_name / has_id sentinel paths are exercised.
│   ├── calls build_camera_intrinsics(model="pinhole", params={"fx": 400.0, "fy": 410.0, "cx": 160.0, "cy": 120.0}, device="cpu")
│   ├── calls build_camera_intrinsics(model="simple_pinhole", params={"f": 405.0, "cx": 161.0, "cy": 121.0}, device="cpu")
│   ├── calls build_camera_intrinsics(model="ortho", params={"fx": 402.0, "fy": 412.0, "cx": 162.0, "cy": 122.0}, device="cpu")
│   ├── calls _make_extrinsics(translation=[0.3, -0.2, 1.1], convention="opengl")
│   ├── calls _make_extrinsics(translation=[1.3, 0.8, 2.1], convention="opencv")
│   ├── calls _make_extrinsics(translation=[2.3, 1.8, 3.1], convention="standard")
│   ├── impls the per-camera names as a List[Optional[str]] — "frame_0", None, "frame_2"
│   ├── impls the per-camera ids as a List[Optional[int]] — 7, 8, None
│   ├── calls Cameras(intrinsics=those intrinsics, extrinsics=those extrinsics, names=those names, ids=those ids, device="cpu")
│   └── return  # that three-camera Cameras
├── def _assert_cameras_fields_equal(loaded: Cameras, original: Cameras) -> None
│   ├── # Extends the per-camera comparison over a collection, so a Cameras round trip is checked camera by camera.
│   ├── assert the loaded object is a Cameras
│   ├── assert it holds as many cameras as the original
│   └── for index in range(len(original))
│       └── calls _assert_camera_fields_equal(loaded=loaded[index], original=original[index])
├── def _make_extrinsics(translation: List[float], convention: str) -> CameraExtrinsics
│   ├── # Holds the rotation at identity so a round trip is judged on the translation and convention alone.
│   ├── impls a 4x4 float32 identity
│   ├── impls write the given translation into its top-right column
│   ├── calls CameraExtrinsics(extrinsics=that matrix, convention=convention, device="cpu")
│   └── return  # that CameraExtrinsics
└── def _assert_camera_fields_equal(loaded: Camera, original: Camera) -> None
    ├── # Collects the per-camera field comparison every round trip repeats, so each test states only what it round-tripped.
    ├── assert the loaded object is a Camera
    ├── assert its intrinsics model equals the original's
    ├── assert its intrinsics params equal the original's
    ├── assert its 4x4 extrinsics matrix equals the original's exactly  # torch.equal
    ├── assert its extrinsics convention equals the original's
    ├── assert its name equals the original's
    └── assert its id equals the original's
```

`tests/data/structures/three_d/camera/test_rotation_stabilize_validate_compat.py`

```text
test_rotation_stabilize_validate_compat.py
├── import numpy as np
├── import pytest
├── import torch
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import _stabilize_rotation_matrix
├── from data.structures.three_d.camera.extrinsics.validation import validate_camera_extrinsics, validate_rotation_matrix
├── @pytest.mark.parametrize def test_stabilize_accepts_float32_and_float64(dtype: torch.dtype) -> None  # over the torch.float32 and torch.float64 dtypes
│   ├── # _stabilize_rotation_matrix accepts a float32 or float64 near-orthogonal rotation, returns the same dtype, and its output passes validate_rotation_matrix.
│   ├── calls _random_rotation(dtype, 1)
│   ├── calls _random_rotation(dtype, 2)
│   ├── impls their matrix product, whose orthogonality has drifted with rounding
│   ├── calls _stabilize_rotation_matrix(r)  # -> the stabilized rotation
│   ├── assert it keeps the input dtype
│   └── calls validate_rotation_matrix(out)
├── def test_stabilize_rejects_unsupported_dtype() -> None
│   ├── # _stabilize_rotation_matrix raises on a dtype outside {float32, float64} (e.g. float16).
│   ├── impls a 3x3 float16 identity
│   └── with pytest.raises(AssertionError)
│       └── calls _stabilize_rotation_matrix(r)
├── @pytest.mark.parametrize def test_stabilized_batch_passes_validator(dtype: torch.dtype) -> None  # over the torch.float32 and torch.float64 dtypes
│   ├── # A batch of stabilized cam2world extrinsics passes the batched validate_camera_extrinsics for both float32 and float64.
│   ├── for index in range(200)
│   │   ├── impls a 4x4 identity in that dtype
│   │   ├── calls _random_rotation(dtype, index)
│   │   ├── calls _random_rotation(dtype, index + 5000)
│   │   ├── calls _stabilize_rotation_matrix(_random_rotation(dtype, index) @ _random_rotation(dtype, index + 5000))  # -> this pose's rotation
│   │   ├── impls write that rotation into the 4x4's 3x3 block
│   │   └── impls collect the 4x4
│   ├── calls torch.stack(extrinsics_list)  # -> the [200, 4, 4] cam2world batch
│   ├── assert the batch's shape is (200, 4, 4)
│   └── calls validate_camera_extrinsics(batch)
├── def test_validator_threshold_is_dtype_aware() -> None
│   ├── # A fixed near-orthogonality deviation between the float64 and float32 tolerances passes validate_rotation_matrix as float32 but is rejected as float64.
│   ├── impls the float64 and float32 machine epsilons from np.finfo
│   ├── impls the fixed deviation 5e-7
│   ├── assert it exceeds 32 float64 epsilons
│   ├── assert it stays under 32 float32 epsilons
│   ├── impls a 3x3 float64 identity whose [0, 0] entry carries that deviation
│   ├── calls validate_rotation_matrix(m.to(torch.float32))
│   └── with pytest.raises(AssertionError)
│       └── calls validate_rotation_matrix(m)
└── def _random_rotation(dtype: torch.dtype, seed: int) -> torch.Tensor
    ├── # Draws a seeded proper rotation, so every case runs on a reproducible non-trivial orientation rather than the identity.
    ├── impls a torch generator seeded with seed
    ├── impls a random 3x3 float64 matrix drawn from that generator
    ├── calls torch.linalg.qr(a)  # -> its orthonormal factor q and upper-triangular factor r
    ├── impls fix q's column signs from the signs of r's diagonal
    ├── if q's determinant is negative
    │   └── impls negate q's first column, making the rotation proper
    └── return  # q cast to the requested dtype
```
