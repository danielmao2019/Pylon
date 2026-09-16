# `models/three_d/meshes/render/` tests skeleton

## Tests implementation structure

`tests/models/three_d/meshes/render/test_core.py`

```text
test_core.py
├── from math import log
├── from typing import Any, List, Tuple
├── import pytest
├── import torch
├── from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras
├── import models.three_d.meshes.render.core as render_core
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from models.three_d.meshes.render.core import _prepare_cameras, render_rgb_from_mesh, render_soft_mask_from_mesh
├── def test_the_mask_is_a_continuous_coverage_the_silhouette_carries_a_gradient_through
│   ├── # The mask is what the silhouette loss is scored against, so it holds a coverage that falls off gradually across the silhouette.
│   ├── calls _build_camera(model="ortho")
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert every returned value lies in [0, 1]
│   ├── impls assert the returned shape is the requested resolution
│   ├── impls assert values strictly between 0 and 1 appear along the mesh's outline  # impls-node-one-step:skip — "0 and 1" names the open interval
│   └── return
├── def test_the_mask_is_differentiable_in_the_mesh_and_in_the_camera
│   ├── # The fit optimizes geometry and camera through this one render, so a gradient reaches both from the mask alone.
│   ├── impls device = the torch device cuda:0
│   ├── calls _build_camera(model="ortho", resolution=(32, 32), device=device, focal_requires_grad=True)
│   ├── calls _build_mesh(view_depth=2.0, device=device, requires_grad=True)
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert the summed mask backpropagates a non-zero gradient to the mesh's vertices
│   ├── impls assert it backpropagates a non-zero gradient to the camera's own parameters
│   └── return
├── def test_the_blur_ends_exactly_at_the_configured_coverage_threshold
│   ├── # The blur radius is derived from the sigma and the threshold, so a face's influence reaches that threshold exactly at the blur's edge.
│   ├── calls _build_camera(model="ortho")
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert the coverage at the blur's own edge equals coverage_threshold
│   ├── impls assert a larger coverage_threshold pulls that edge in and a smaller one pushes it out  # impls-node-one-step:skip — "in and out" names the two ends of one comparison
│   └── return
├── def test_the_render_reproduces_itself_between_runs
│   ├── # Every render is held to the naive kernel, so a mask a fit is scored against reproduces itself run to run.
│   ├── calls _build_camera(model="ortho")
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert two renders of the same mesh and camera are bitwise equal  # impls-node-one-step:skip — "mesh and camera" names the shared inputs
│   └── return
├── def test_what_the_mesh_occupies_is_what_the_render_covers
│   ├── # A mesh's back faces reach the mask exactly as its front ones do, coverage being the geometry's own.
│   ├── calls _build_camera(model="ortho")
│   ├── calls _build_mesh(view_depth=2.0, reverse_winding=False)
│   ├── calls _build_mesh(view_depth=2.0, reverse_winding=True)
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert a mesh and its winding-reversed copy render the same coverage  # impls-node-one-step:skip — "a mesh and its copy" names the pair compared
│   └── return
├── def test_the_rgb_renders_mask_is_read_off_the_rasterization_rather_than_the_shaded_image
│   ├── # The background is a colour a face can carry, so the coverage comes off the rasterization's own face indices.
│   ├── calls _build_camera(model="pinhole")
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_rgb_from_mesh(mesh=a mesh whose albedo is the background colour, camera=a camera, background=that colour, return_mask=True)
│   ├── impls assert every pixel the mesh covers is marked covered in the returned mask
│   ├── impls assert the pixels no face reaches are marked uncovered
│   └── return
├── def test_both_camera_models_are_rendered_rather_than_one_falling_through
│   ├── # The weak-perspective projection is a camera model this module renders, so an ortho camera builds its own PyTorch3D camera and an unknown model aborts.
│   ├── calls _build_camera(model="pinhole")
│   ├── calls _build_camera(model="ortho")
│   ├── calls _prepare_cameras
│   ├── impls assert a pinhole camera builds a PerspectiveCameras
│   ├── impls assert an ortho camera builds an OrthographicCameras
│   ├── calls _build_camera(model="pinhole")  # the camera the unsupported-model check starts from
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _prepare_cameras
│   └── return
├── def test_the_camera_reaches_pytorch3d_in_pytorch3ds_own_frames
│   ├── # Both halves of the frame change are the camera's own work, so the vertices arrive on the axes PyTorch3D names and the intrinsics on the plane it reads them in.
│   ├── calls _build_camera(model="pinhole", resolution=(100, 150))
│   ├── calls _prepare_cameras
│   ├── impls assert the converted camera's extrinsics convention and its intrinsics convention are both pytorch3d  # impls-node-one-step:skip — the two conventions are one pair, asserted together
│   ├── impls assert both branches name in_ndc explicitly
│   ├── impls assert the principal point reaching PyTorch3D is the converted camera's own cx and cy  # impls-node-one-step:skip — "cx and cy" names the one point
│   ├── impls assert the rotation handed to PyTorch3D is the transpose of the camera's own world-to-camera block
│   └── return
├── def test_a_render_that_names_no_resolution_renders_the_cameras_own
│   ├── # The resolution is two of the camera's own params, so a render is fully determined by the camera it is given, and a named resolution is an override.
│   ├── calls _build_camera(model="ortho", resolution=(24, 28))
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_soft_mask_from_mesh(mesh=a mesh, camera=a camera whose intrinsics carry a known h and w, blend_sigma=..., blend_gamma=..., faces_per_pixel=..., coverage_threshold=...)
│   ├── impls assert the returned mask's shape is the camera's own h and w  # impls-node-one-step:skip — "h and w" names the one resolution
│   ├── calls render_rgb_from_mesh(mesh=that mesh, camera=that camera)
│   ├── impls assert the returned image's trailing two dimensions are the camera's own h and w  # impls-node-one-step:skip — "h and w" names the one resolution
│   └── return
├── def test_a_named_resolution_overrides_the_cameras_own
│   ├── # A caller that does name a raster gets it, the camera's own resolution being only the default.
│   ├── calls _build_camera(model="ortho", resolution=(24, 28))
│   ├── calls _build_mesh(view_depth=2.0)
│   ├── calls render_soft_mask_from_mesh(mesh=a mesh, camera=that camera, blend_sigma=..., blend_gamma=..., faces_per_pixel=..., coverage_threshold=..., resolution=a size differing from the camera's own)
│   ├── impls assert the returned mask's shape is the named size
│   └── return
├── def test_the_camera_a_render_is_given_is_the_camera_it_renders_through
│   ├── # Where a camera stands is its extrinsics' own statement under either model, so this module places no camera of its own and a mesh a camera is turned away from renders as absent rather than dragged into view.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert the translation reaching PyTorch3D is the camera's own world-to-camera block under either camera model
│   ├── impls assert a mesh sitting behind the camera it is rendered through covers no pixel
│   └── return
├── def _build_camera(model: str = "pinhole", resolution: Tuple[int, int] = (32, 32), device: torch.device = torch.device("cpu"), focal_requires_grad: bool = False) -> Tuple[Camera, torch.Tensor]
│   ├── # Builds the one standard-frame camera every test in this module renders through or prepares for PyTorch3D, handing back its focal tensor so a test can score a gradient against it.
│   ├── impls height, width = resolution
│   ├── impls focal = the smaller image side scaled by 0.8, as a float32 scalar on device tracking gradients when focal_requires_grad
│   ├── impls params = focal as fx, a fresh gradient-free tensor of the same value as fy, half the width as cx, half the height as cy, those three float32 on device, height as h, width as w
│   ├── calls build_camera_intrinsics(model=model, params=params, intr_convention="standard", device=device)
│   ├── calls CameraExtrinsics(extrinsics=the [4, 4] float32 identity on device, extr_convention="standard", device=device)
│   ├── calls Camera(intrinsics=the intrinsics it built, extrinsics=the extrinsics it built, device=device)
│   └── return  # that camera and focal, the focal being the one tensor a gradient test reaches for
└── def _build_mesh(view_depth: float = 2.0, device: torch.device = torch.device("cpu"), reverse_winding: bool = False, requires_grad: bool = False) -> Mesh
    ├── # Builds the one small black triangle the render tests in this module draw, placed at a named forward depth in the standard camera frame.
    ├── impls verts = the three float32 triangle corners on device, each sharing view_depth as its forward coordinate
    ├── if requires_grad
    │   └── impls make verts track gradients
    ├── impls face = the corner order reversed if reverse_winding, else the natural one  # a test covers that coverage is the geometry's own whichever way a face winds, so it needs both
    │   ├── if reverse_winding
    │   │   └── impls the corners 0, 2, 1
    │   └── else
    │       └── impls the corners 0, 1, 2
    ├── impls faces = that one face as an int64 [1, 3] tensor on device
    ├── calls MeshTextureVertexColor(vertex_color=a [3, 3] float32 zero tensor on device)  # -> texture
    ├── calls Mesh(verts=verts, faces=faces, texture=texture)
    └── return  # the one-face mesh, black so a test can set the background to the albedo
```

`tests/models/three_d/meshes/render/test_core_blender.py`

```text
test_core_blender.py
├── import importlib
├── import sys
├── import types
├── from typing import List, Tuple
├── import pytest
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
└── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
```

`tests/models/three_d/meshes/render/test_shading.py`

```text
test_shading.py
├── import pytest
├── import torch
├── from models.three_d.meshes.render.shading import compute_sh_shading
├── def test_band_count_selects_the_spherical_harmonic_order
│   ├── # Coefficients of any perfect-square band count evaluate at the order that count implies, so a caller's band count is never assumed.
│   ├── for each perfect-square band count
│   │   ├── calls compute_sh_shading
│   │   └── impls assert the shading has one RGB triple per input normal
│   └── return
├── def test_non_square_band_count_is_rejected
│   ├── # A coefficient set whose band count is not a perfect square names no spherical-harmonic order, so it fails the assertion rather than evaluating.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls compute_sh_shading
│   └── return
├── def test_higher_order_coefficients_affect_shading() -> None
│   ├── # Bands above degree 2 reach the result, so the evaluation runs to whatever order the band count names.
│   ├── impls normals = one unit-length [1, 3] float32 normal, along (1, 2, 3)
│   ├── impls sh_coefficients = a float32 zero vector of 16 bands for each of 3 channels
│   ├── impls set its entries 9 to 15, the red channel's degree-3 bands, to one
│   ├── calls compute_sh_shading(normals=normals, sh_coefficients=sh_coefficients)  # -> shading
│   └── assert shading is not all close to zero                                     # f"Expected degree-3 coefficients to contribute to the shading. {shading=}, {normals=}, {sh_coefficients=}"
└── def test_shading_varies_with_the_normal_direction
    ├── # Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.
    ├── calls compute_sh_shading
    ├── impls assert the two normals' shading differs
    └── return
```
