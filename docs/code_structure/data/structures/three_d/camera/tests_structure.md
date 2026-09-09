# Camera Data Structure Tests Structure

## 1. Tests implementation structure

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
├── from data.structures.three_d.camera.intrinsics.validation import validate_camera_intrinsics_invariants, validate_intr_convention
├── def test_validate_extr_convention_accepts_all_supported
│   ├── # validate_extr_convention accepts every supported convention string.
│   ├── for each extr_convention in {standard, opengl, opencv, pytorch3d, arkit}
│   │   ├── calls validate_extr_convention
│   │   └── impls assert the returned string is the extr_convention that was passed in
│   └── return
├── def test_extr_convention_module_has_one_main_api_and_eight_helpers
│   ├── # The extrinsics conventions module exposes exactly one main API plus eight helpers.
│   ├── impls collect the function names the conventions module defines
│   ├── impls assert transform_extr_convention is its only public function
│   ├── impls assert the eight helpers are the to-standard and from-standard converters for opengl, opencv, pytorch3d, and arkit  # impls-node-one-step:skip
│   └── return
├── def test_extrinsics_conversion_preserves_physical_axes_and_center
│   ├── # Converting a CameraExtrinsics between extr_conventions preserves its physical right / forward / up axes and center.
│   ├── calls _build_extrinsics(extr_convention=the source extr_convention)
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to
│   │   ├── impls assert the converted right / forward / up axes equal the source ones under torch.allclose
│   │   └── impls assert the converted center equals the source center under torch.allclose
│   └── return
├── def test_extrinsics_direct_and_via_standard_conversion_match
│   ├── # Converting a CameraExtrinsics directly between two extr_conventions matches converting via the standard one.
│   ├── for each (source, target) extr_convention pair
│   │   ├── calls _build_extrinsics(extr_convention=the source extr_convention)
│   │   ├── calls extrinsics.to(extr_convention=the target extr_convention)
│   │   ├── impls converted_direct = the extrinsics it returned
│   │   ├── calls extrinsics.to(extr_convention="standard")
│   │   ├── impls converted_via_standard = that result carried on to the target extr_convention by a second to
│   │   └── impls assert the two target-frame 4x4 matrices agree under torch.allclose
│   └── return
├── def test_extrinsics_round_trip_returns_original_matrix
│   ├── # Converting a CameraExtrinsics to another extr_convention and back returns the original 4x4 matrix.
│   ├── calls _build_extrinsics(extr_convention=the source extr_convention)
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to(extr_convention=the target extr_convention)
│   │   ├── impls round_trip = that result carried back to the source extr_convention by a second to
│   │   └── impls assert the round-tripped 4x4 matrix equals the original under torch.allclose
│   └── return
├── def test_extrinsics_w2c_is_inverse_of_extrinsics
│   ├── # CameraExtrinsics.w2c is the inverse of the 4x4 camera-to-world extrinsics matrix.
│   ├── calls _build_extrinsics(extr_convention=the pose frame the case names)
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
│   ├── calls _batch_one_camera(camera=that camera)
│   ├── calls cameras.transform_extrinsics(scale=that same factor, rotation=that same rotation, translation=that same offset)
│   ├── impls assert every camera in the batch carries that same result
│   └── return
├── def test_camera_and_cameras_to_keep_tensor_state_on_the_autograd_path
│   ├── # Camera.to and Cameras.to each keep the tensor state handed to them on the autograd path, the moved camera and the moved collection backpropagating separately to the same source params and cam2world.
│   ├── calls _build_extrinsics_matrix
│   ├── impls matrix = the cam2world matrix it built, marked requires_grad in place
│   ├── calls build_camera_intrinsics(model="pinhole", params=tensor scalar params whose four projection entries require grad, intr_convention="standard", device="cpu")
│   ├── calls CameraExtrinsics(extrinsics=matrix, extr_convention="standard", device="cpu")
│   ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, device="cpu")
│   ├── calls camera.to(dtype=a floating torch dtype, extr_convention="pytorch3d")
│   ├── impls camera_loss = moved_camera.intrinsics.fx + moved_camera.extrinsics.center.sum()
│   ├── calls camera_loss.backward(retain_graph=True)
│   ├── impls assert the source fx param receives a gradient
│   ├── impls assert the source cam2world tensor receives a gradient
│   ├── calls _batch_one_camera(camera=camera)
│   ├── calls cameras.to(dtype=a floating torch dtype, extr_convention="pytorch3d")
│   ├── impls cameras_loss = moved_cameras[0].intrinsics.fx + moved_cameras[0].extrinsics.center.sum()  # the batch is indexed to one camera, whose own components then answer
│   ├── calls cameras_loss.backward
│   ├── impls assert the source fx param receives a gradient
│   ├── impls assert the source cam2world tensor receives a gradient
│   └── return
├── def test_transform_extrinsics_normalizes_rotation_input
│   ├── # CameraExtrinsics.transform_extrinsics accepts each validated rotation representation and normalizes it to the pose tensor's placement.
│   ├── for each rotation in {a (3, 3) numpy array, a (3, 3) torch tensor, a length-3 nested numeric list}
│   │   ├── calls extrinsics.transform_extrinsics(scale=a known factor, rotation=rotation, translation=a known tensor offset)
│   │   └── impls assert the returned extrinsics match the tensor-rotation result
│   └── return
├── def test_transform_extrinsics_normalizes_translation_input
│   ├── # CameraExtrinsics.transform_extrinsics accepts each valid translation representation and normalizes it to the pose tensor's placement.
│   ├── for each translation in {a length-3 numpy array, a length-3 torch tensor, a length-3 numeric tuple, a length-3 numeric list}
│   │   ├── calls extrinsics.transform_extrinsics(scale=a known factor, rotation=a known tensor rotation, translation=translation)
│   │   └── impls assert the returned extrinsics match the tensor-translation result
│   └── return
├── def test_cameras_conversion_preserves_physical_axes_and_center
│   ├── # Converting a Cameras collection between extr_conventions preserves each camera's physical axes and center.
│   ├── calls _build_cameras(extr_convention=the source pose frame of the pair)
│   ├── for each target extr_convention
│   │   ├── calls cameras.to
│   │   ├── impls assert the converted [N, 3] right / forward / up stacks equal the source ones under torch.allclose
│   │   └── impls assert the converted [N, 3] center stack equals the source one under torch.allclose
│   └── return
├── def _build_cameras
│   ├── # Builds the three-camera Cameras fixture the collection conversions run on, its poses distinct so a per-camera axis or centre error cannot hide behind one shared pose.
│   ├── calls _build_extrinsics_matrices
│   ├── impls pose_matrices = the three distinct cam2world matrices it built
│   ├── calls build_camera_intrinsics(model="pinhole", params=one fixed pinhole param set broadcast to a [3] column per key, intr_convention="standard", device="cpu")
│   ├── impls intrinsics = that one batched pinhole, its params carrying the pose count
│   ├── calls CameraExtrinsics(extrinsics=the [3, 4, 4] stack of pose_matrices, extr_convention=the pose frame asked for, device="cpu")
│   ├── impls extrinsics = the one batched extrinsics it built
│   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, device="cpu")
│   └── return  # that three-camera collection
├── def test_every_supported_extr_convention_is_right_handed
│   ├── # A camera carries no change of handedness: each supported pose frame's (right, forward, up) triple is positively oriented, so converting between two of them keeps the rotation determinant at +1.
│   ├── calls _build_extrinsics(extr_convention="standard")
│   ├── for each target extr_convention
│   │   ├── calls extrinsics.to
│   │   ├── impls assert the converted right / forward / up triple has a positive scalar triple product
│   │   └── impls assert the converted rotation block's determinant is +1
│   └── return
├── def _build_extrinsics
│   ├── # Builds the one-camera CameraExtrinsics fixture the pose-frame conversions run on, in whichever frame the case asks for.
│   ├── calls _build_extrinsics_matrix
│   ├── calls CameraExtrinsics(extrinsics=the matrix it built, extr_convention=the pose frame asked for, device="cpu")
│   └── return  # that extrinsics
├── def _build_extrinsics_matrices
│   ├── # Builds the three distinct cam2world matrices the multi-camera fixture poses its cameras with.
│   ├── calls _build_extrinsics_matrix
│   ├── impls rotation_about_z = the matrix it built
│   ├── impls identity_rotation = a 4x4 float32 identity whose translation column carries its own centre
│   ├── impls rotation_about_x = a 4x4 float32 cam2world turning about x, with its own centre
│   └── return  # rotation_about_z, identity_rotation and rotation_about_x, in that order
├── def _build_extrinsics_matrix
│   ├── # Builds the one cam2world matrix the fixtures start from, its rotation block a proper rotation.
│   ├── impls build a 4x4 float32 literal whose rotation block is a quarter turn about z
│   ├── impls set its translation column to a fixed camera centre
│   └── return  # that matrix
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
├── def test_a_converted_intrinsics_still_satisfies_its_own_invariants
│   ├── # A frame change restates a camera, so converted params must keep satisfying their own model and image-frame invariants.
│   ├── for each model in {pinhole, ortho}
│   │   └── for each target intr_convention
│   │       ├── calls transform_intr_convention
│   │       └── calls validate_camera_intrinsics_invariants
│   └── return
├── def test_a_frame_change_is_measured_against_the_intrinsics_own_resolution
│   ├── # The resolution is what fixes where a centred origin sits and what a normalized unit is worth, so the conversion reads the h and w the params already carry rather than a resolution the caller supplies and could get wrong.
│   ├── calls _build_pinhole_params(height=a fixed height, width=each of two differing widths)
│   ├── calls build_camera_intrinsics
│   ├── impls narrow, wide = each build's result carried to "opengl" by a chained to                                     # impls-node-one-step:skip
│   ├── impls assert two intrinsics whose projection params match but whose h and w differ convert to different results  # impls-node-one-step:skip
│   ├── impls assert the converted intrinsics carries the same h and w it was built with, the image being the same image in either frame  # impls-node-one-step:skip
│   └── return
├── def _build_pinhole_params
│   ├── # Builds the pinhole params dict the intrinsics cases are stated with, its h and w the only thing a case varies.
│   ├── impls build a params dict of fixed fx, fy, cx and cy under the height and width asked for  # impls-node-one-step:skip — one step; the "and" names what it is made of
│   └── return  # that params dict
├── def test_an_intrinsics_without_a_resolution_is_refused
│   ├── # A principal point in the standard frame names a location only against a resolution, so params missing h or w are refused for every camera model.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls build_camera_intrinsics
│   └── return
├── def test_a_camera_names_the_frame_of_each_half_separately
│   ├── # A pose frame and an image-plane frame are different kinds of thing, so a camera carries one name for each and converting one leaves the other where it was.
│   ├── calls Camera
│   ├── calls camera.to
│   ├── impls assert the returned camera's extr_convention is the camera-space frame that was named
│   ├── impls assert its intr_convention is the image-plane frame that was named
│   ├── impls assert naming only one of the two leaves the other half's frame unchanged
│   └── return
├── def test_extrinsics_tensor_matrix_stays_differentiable_through_pose_accessors
│   ├── # Tensor-valued extrinsics stay on the autograd path through pose accessors.
│   ├── calls CameraExtrinsics(extrinsics=a valid cam2world tensor with tensor-valued translation, extr_convention="standard")
│   ├── impls loss = extrinsics.w2c.sum() + extrinsics.center.sum()
│   ├── calls loss.backward
│   └── impls assert the source extrinsics tensor receives a gradient
├── def test_extrinsics_constructor_applies_requested_device_dtype_through_to
│   ├── # CameraExtrinsics.__init__ delegates requested device / dtype movement to the object's to method.
│   ├── calls CameraExtrinsics(extrinsics=a valid cam2world tensor, extr_convention="standard", device=a valid device, dtype=a floating torch dtype)
│   ├── impls assert extrinsics.extrinsics has the requested device
│   ├── impls assert extrinsics.extrinsics has the requested dtype
│   ├── impls assert extrinsics.device matches its tensor state
│   ├── impls assert extrinsics.dtype matches its tensor state
│   └── return
├── def test_extrinsics_to_follows_tensor_to_semantics
│   ├── # CameraExtrinsics.to applies Tensor.to-style device / dtype / copy semantics to the cam2world tensor.
│   ├── calls CameraExtrinsics(extrinsics=a valid cam2world tensor, extr_convention="standard")
│   ├── calls extrinsics.to(device=a valid device, dtype=a floating torch dtype, copy=True)
│   ├── impls assert the returned cam2world tensor has the requested device
│   ├── impls assert the returned cam2world tensor has the requested dtype
│   ├── impls assert copy=True returns a tensor with distinct storage from the source extrinsics
│   └── return
├── def test_camera_and_cameras_to_preserve_tensor_parameter_graphs
│   ├── # Camera.to and Cameras.to keep tensor-valued intrinsics and extrinsics on their autograd paths.
│   ├── calls build_camera_intrinsics(model="ortho", params=tensor scalar params with requires_grad, intr_convention="standard")
│   ├── calls CameraExtrinsics(extrinsics=a valid cam2world tensor with tensor-valued translation, extr_convention="standard")
│   ├── calls Camera
│   ├── calls camera.to(device=the current device, dtype=a floating torch dtype, extr_convention="pytorch3d")
│   ├── calls _batch_one_camera(camera=camera)
│   ├── calls cameras.to(device=the current device, dtype=a floating torch dtype, extr_convention="pytorch3d")
│   ├── impls loss = moved_camera.intrinsics.project(points).sum() + moved_cameras.center.sum()
│   ├── calls loss.backward
│   ├── impls assert source intrinsics params receive gradients
│   ├── impls assert source extrinsics receive gradients
│   └── return
└── def _batch_one_camera(camera: Camera) -> Cameras
    ├── # Makes one Camera into a length-one Cameras, which the components cannot do themselves since neither is subscriptable.
    ├── impls batched_params = each of the camera's intrinsics params under a leading axis of one
    ├── calls build_camera_intrinsics(model=camera.intrinsics.model, params=batched_params, intr_convention=camera.intrinsics.intr_convention)
    ├── calls CameraExtrinsics(extrinsics=the cam2world matrix under a leading axis of one, extr_convention=camera.extrinsics.extr_convention)
    ├── calls Cameras(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, names=[camera.name], ids=[camera.id])
    └── return  # that length-one batch
```

`tests/data/structures/three_d/camera/test_io.py`

```text
test_io.py
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.camera.io import deserialize_cameras, load_cameras, save_cameras, serialize_cameras
├── def test_single_camera_json_round_trip
│   ├── # A single Camera survives a save then load round trip through the json format.
│   ├── calls _make_single_camera
│   ├── calls serialize_cameras(cameras=camera, format="json")
│   ├── impls serialized = the payload it produced
│   ├── impls assert serialized is a dict whose keys are the json key set
│   ├── impls assert serialized equals what camera.serialize(format="json") returns
│   ├── calls deserialize_cameras(payload=serialized, device="cpu", format="json")
│   ├── calls Camera.deserialize(payload=serialized, device="cpu", format="json")
│   ├── calls _assert_camera_fields_equal(loaded=each of those two, original=camera)
│   ├── calls camera.save(camera_path=a .json path under tmp_path)
│   ├── impls assert the file on disk parses back to serialized
│   ├── calls load_cameras(cameras_path=that path, device="cpu")
│   ├── calls Camera.load(camera_path=that path, device="cpu")
│   ├── calls _assert_camera_fields_equal(loaded=each of those two, original=camera)
│   └── return
├── def test_single_camera_npz_round_trip
│   ├── # A single Camera survives a save then load round trip through the npz format.
│   ├── calls _make_single_camera
│   ├── calls serialize_cameras(cameras=camera, format="npz")
│   ├── impls serialized = the payload it produced
│   ├── impls assert serialized is a dict whose keys are the npz key set
│   ├── impls assert its extrinsics entry has shape (4, 4)
│   ├── calls deserialize_cameras(payload=serialized, device="cpu", format="npz")
│   ├── calls _assert_camera_fields_equal(loaded=what it returned, original=camera)
│   ├── calls camera.save(camera_path=a .npz path under tmp_path)
│   ├── impls assert the archive on disk carries exactly the npz key set
│   ├── calls load_cameras(cameras_path=that path, device="cpu")
│   ├── calls Camera.load(camera_path=that path, device="cpu")
│   ├── calls _assert_camera_fields_equal(loaded=each of those two, original=camera)
│   └── return
├── def _make_single_camera
│   ├── # Builds the one-camera Camera fixture both single round trips run on, carrying a name and an id so the round trip has both to preserve.
│   ├── calls build_camera_intrinsics(model="pinhole", params=that model's param set, intr_convention="standard", device="cpu")
│   ├── impls intrinsics = the intrinsics it built
│   ├── calls _make_extrinsics(translation=a camera centre, extr_convention="opengl")
│   ├── impls extrinsics = the extrinsics it built
│   ├── calls Camera(intrinsics=intrinsics, extrinsics=extrinsics, name=a label, id=an identifier, device="cpu")
│   └── return  # that camera
├── def test_multi_cameras_json_round_trip
│   ├── # A Cameras collection survives a save then load round trip through the json format.
│   ├── calls _make_multi_cameras
│   ├── calls serialize_cameras(cameras=cameras, format="json")
│   ├── impls serialized = the payload it produced
│   ├── impls assert serialized is one dict per camera, each keyed by the json key set
│   ├── calls deserialize_cameras(payload=serialized, device="cpu", format="json")
│   ├── calls _assert_cameras_fields_equal(loaded=what it returned, original=cameras)
│   ├── calls save_cameras(cameras=cameras, cameras_path=a .json path under tmp_path)
│   ├── impls assert the file on disk parses back to serialized
│   ├── calls load_cameras(cameras_path=that path, device="cpu")
│   ├── calls _assert_cameras_fields_equal(loaded=what it loaded, original=cameras)
│   └── return
├── def test_multi_cameras_npz_round_trip
│   ├── # A Cameras collection survives a save then load round trip through the npz format.
│   ├── calls _make_multi_cameras
│   ├── calls serialize_cameras(cameras=cameras, format="npz")
│   ├── impls serialized = the payload it produced
│   ├── impls assert serialized is a dict whose keys are the npz key set
│   ├── impls assert its extrinsics entry has one 4x4 block per camera
│   ├── calls deserialize_cameras(payload=serialized, device="cpu", format="npz")
│   ├── calls _assert_cameras_fields_equal(loaded=what it returned, original=cameras)
│   ├── calls save_cameras(cameras=cameras, cameras_path=a .npz path under tmp_path)
│   ├── impls assert the archive on disk carries exactly the npz key set
│   ├── calls load_cameras(cameras_path=that path, device="cpu")
│   ├── calls _assert_cameras_fields_equal(loaded=what it loaded, original=cameras)
│   └── return
├── def _make_multi_cameras
│   ├── # Builds the three-camera Cameras fixture both collection round trips run on, its cameras differing in param values, centre, name and id so the payload spans every per-camera path the format has to carry.
│   ├── calls build_camera_intrinsics(model="pinhole", params=that model's param set with a distinct [3] column per key, intr_convention="standard", device="cpu")
│   ├── impls intrinsics = that one batched pinhole  # a batch names one model and one image-plane frame, so those two are what the fixture holds fixed
│   ├── impls matrices = three 4x4 float32 identities whose translation columns are three distinct camera centres
│   ├── calls CameraExtrinsics(extrinsics=the [3, 4, 4] stack of matrices, extr_convention="opengl", device="cpu")
│   ├── impls extrinsics = that one batched extrinsics  # a batch names one pose frame, so the frame is what the fixture holds fixed
│   ├── impls names = a label for every camera but the second
│   ├── impls ids = an id for every camera but the third
│   ├── calls Cameras(intrinsics=intrinsics, extrinsics=extrinsics, names=names, ids=ids, device="cpu")
│   └── return  # that three-camera collection
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
│   │       ├── calls _make_extrinsics
│   │       ├── calls Camera
│   │       ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
│   │       ├── calls Camera.load
│   │       ├── impls assert the loaded intrinsics model equals the saved model string
│   │       └── impls assert the loaded params dict equals the saved params dict
│   └── return
├── def test_tensor_intrinsics_params_round_trip_as_serialized_values
│   ├── # Tensor-valued intrinsics params round-trip through camera I/O as serialized numeric values.
│   ├── for each format in {json, npz}
│   │   ├── calls build_camera_intrinsics(model="ortho", params=tensor scalar params, intr_convention="standard")
│   │   ├── calls _make_extrinsics
│   │   ├── calls Camera
│   │   ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
│   │   ├── calls Camera.load
│   │   └── impls assert the loaded params equal the source tensor values materialized as numeric scalars
│   └── return
├── def test_extrinsics_and_extr_convention_survive_round_trip
│   ├── # A Camera's extrinsics matrix and extr_convention survive a save then load round trip through both the json and npz formats.
│   ├── for each format in {json, npz}
│   │   └── for each supported extr_convention
│   │       ├── calls build_camera_intrinsics
│   │       ├── calls _make_extrinsics
│   │       ├── calls Camera
│   │       ├── calls camera.save(camera_path=a tmp_path file with that format's suffix)
│   │       ├── calls Camera.load
│   │       ├── impls assert the loaded 4x4 extrinsics matrix equals the saved one exactly
│   │       └── impls assert the loaded extr_convention equals the saved extr_convention
│   └── return
├── def _make_extrinsics
│   ├── # Builds one CameraExtrinsics whose rotation is identity, so a round trip is measured on the centre and the pose frame alone.
│   ├── impls matrix = a 4x4 float32 identity whose translation column is set to the translation asked for
│   ├── calls CameraExtrinsics(extrinsics=matrix, extr_convention=the pose frame asked for, device="cpu")
│   └── return  # that extrinsics
├── def _assert_cameras_fields_equal
│   ├── # Checks a loaded Cameras against the original by running the single-camera check at every index.
│   ├── impls assert the loaded object is a Cameras of the same length
│   ├── for each index of the original
│   │   └── calls _assert_camera_fields_equal(loaded=the loaded camera at that index, original=the original at that index)
│   └── return
└── def _assert_camera_fields_equal
    ├── # Checks a loaded Camera against the original on the fields serialization has to carry.
    ├── impls assert the loaded object is a Camera
    ├── impls assert its intrinsics model equals the original's
    ├── impls assert its intrinsics params equal the original's
    ├── impls assert its extrinsics matrix equals the original's
    ├── impls assert its extr_convention equals the original's
    ├── impls assert its name equals the original's
    ├── impls assert its id equals the original's
    └── return
```
