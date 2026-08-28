# `models/three_d/meshes/render/` tests skeleton

## Tests implementation structure

### tests/models/three_d/meshes/render/test_core.py

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
│   ├── # The mask is what the silhouette loss is scored against, so it holds a coverage rather than a decision and that coverage falls off across the silhouette rather than stepping at it.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert every returned value lies in [0, 1] and the returned shape is the requested resolution
│   ├── impls assert values strictly between 0 and 1 appear along the mesh's outline
│   └── return
├── def test_the_mask_is_differentiable_in_the_mesh_and_in_the_camera
│   ├── # The fit optimizes geometry and camera through this one render, so a gradient reaches both from the mask alone.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert the summed mask backpropagates a non-zero gradient to the mesh's vertices
│   ├── impls assert it backpropagates a non-zero gradient to the camera's own parameters
│   └── return
├── def test_the_blur_ends_exactly_at_the_configured_coverage_threshold
│   ├── # The blur radius is derived from the sigma and the threshold rather than configured beside them, so a face's influence reaches that threshold at the blur's edge and no further.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert the coverage at the blur's own edge equals coverage_threshold
│   ├── impls assert a larger coverage_threshold pulls that edge in and a smaller one pushes it out
│   └── return
├── def test_the_render_reproduces_itself_between_runs
│   ├── # A binned rasterization appends faces to a tile's list atomically and so does not reproduce itself, which a fit scored against this mask cannot tolerate, so every render is held to the naive kernel.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert two renders of the same mesh and camera are bitwise equal
│   └── return
├── def test_what_the_mesh_occupies_is_what_the_render_covers
│   ├── # No face is culled by its winding, so a mesh's back faces reach the mask exactly as its front ones do.
│   ├── calls render_soft_mask_from_mesh
│   ├── impls assert a mesh and its winding-reversed copy render the same coverage
│   └── return
├── def test_the_rgb_renders_mask_is_read_off_the_rasterization_rather_than_the_shaded_image
│   ├── # The background is a colour a face can carry, so the coverage comes off the rasterization's own face indices rather than off the shaded image, where such a face and the background read alike.
│   ├── calls render_rgb_from_mesh(mesh=a mesh whose albedo is the background colour, camera=a camera, background=that colour, return_mask=True)
│   ├── impls assert every pixel the mesh covers is marked covered in the returned mask
│   ├── impls assert the pixels no face reaches are marked uncovered
│   └── return
├── def test_both_camera_models_are_rendered_rather_than_one_falling_through
│   ├── # The weak-perspective projection is a camera model this module renders, so an ortho camera builds its own PyTorch3D camera and an unknown model aborts rather than being served a perspective one.
│   ├── calls Camera
│   ├── calls _prepare_cameras
│   ├── impls assert a pinhole camera builds a perspective PyTorch3D camera and an ortho one an orthographic PyTorch3D camera
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _prepare_cameras
│   └── return
├── def test_the_camera_reaches_pytorch3d_in_pytorch3ds_own_frames
│   ├── # Both halves of the frame change are the camera's own work, so the vertices arrive on the axes PyTorch3D names and the intrinsics arrive on the plane it reads them in, with nothing restated at this call site.
│   ├── calls _prepare_cameras
│   ├── impls assert the converted camera's extrinsics convention and its intrinsics convention are both pytorch3d
│   ├── impls assert this module negates no principal-point component of its own, the frame change being camera.to's
│   ├── impls assert both branches name in_ndc rather than leaving PyTorch3D to default it, the other reading taking those params for pixels
│   ├── impls assert the rotation handed to PyTorch3D is the transpose of the camera's own world-to-camera block
│   └── return
├── def test_a_render_that_names_no_resolution_renders_the_cameras_own
│   ├── # The resolution is two of the camera's own params, so a render is fully determined by the camera it is given and a resolution is an override rather than a second thing every caller must supply.
│   ├── calls render_soft_mask_from_mesh(mesh=a mesh, camera=a camera whose intrinsics carry a known h and w, blend_sigma=..., blend_gamma=..., faces_per_pixel=..., coverage_threshold=...)
│   ├── impls assert the returned mask's shape is the camera's own h and w
│   ├── calls render_rgb_from_mesh(mesh=that mesh, camera=that camera)
│   ├── impls assert the returned image's trailing two dimensions are the camera's own h and w
│   └── return
├── def test_a_named_resolution_overrides_the_cameras_own
│   ├── # A caller that does name a raster gets it, the camera's own resolution being the default rather than a constraint.
│   ├── calls render_soft_mask_from_mesh(mesh=a mesh, camera=that camera, blend_sigma=..., blend_gamma=..., faces_per_pixel=..., coverage_threshold=..., resolution=a size differing from the camera's own)
│   ├── impls assert the returned mask's shape is the named size rather than the camera's
│   └── return
└── def test_the_camera_a_render_is_given_is_the_camera_it_renders_through
    ├── # Where a camera stands is its extrinsics' own statement under either model, so this module places no camera of its own and a mesh a camera is turned away from renders as absent rather than dragged into view.
    ├── calls render_soft_mask_from_mesh
    ├── impls assert the translation reaching PyTorch3D is the camera's own world-to-camera block under either camera model
    ├── impls assert a mesh sitting behind the camera it is rendered through covers no pixel
    └── return
```

### tests/models/three_d/meshes/render/test_core_blender.py

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

### tests/models/three_d/meshes/render/test_shading.py

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
└── def test_shading_varies_with_the_normal_direction
    ├── # Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.
    ├── calls compute_sh_shading
    ├── impls assert the two normals' shading differs
    └── return
```

### tests/models/three_d/meshes/render/test_shading.py

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
└── def test_shading_varies_with_the_normal_direction
    ├── # Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.
    ├── calls compute_sh_shading
    ├── impls assert the two normals' shading differs
    └── return
```
